import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from uaml_ab.data import load_data_from_csv, resolve_wt_fitness
from uaml_ab.config import load_config
from uaml_ab.metrics import regression_metrics, uq_metrics

from .pool import ProteinPool
from .round import fit_round
# _mean_min_hamming_distance is a module-level helper in bo_loop; importing
# bo_loop is side-effect-free (it only defines a class + functions), so we reuse
# it directly rather than duplicating the novelty metric.
from .loop import _mean_min_hamming_distance
# Both search algorithms use the same acquisition scores.
from ..acquisition import propose_candidates, ACQUISITION_SCORE_COL


def _antigen_name(dataset, idx, override=None):
    """Short antigen name, mirroring genetic_algorithm.run_ga._antigen_name."""
    if override:
        return override
    m = re.search(r"(SI06|M22)", dataset)
    return m.group(1) if m else f"ag{idx}"


def _pareto_front_mask(points):
    """
    Boolean mask of the strictly non-dominated points (maximizing BOTH columns of
    `points`, shape (n, 2)). Sort by objective-0 desc then objective-1 desc; a
    point is on the front iff its objective-1 value strictly exceeds that of every
    point with a strictly larger objective-0. Ties/duplicates keep only the first.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    on_front_sorted = np.zeros(n, dtype=bool)
    max_y = -np.inf
    for i, idx in enumerate(order):
        y = points[idx, 1]
        if y > max_y:
            on_front_sorted[i] = True
            max_y = y
    mask = np.zeros(n, dtype=bool)
    mask[order[on_front_sorted]] = True
    return mask


def _hypervolume_2d(points, ref):
    """
    Dominated 2-D hypervolume of `points` (maximizing both objectives) relative to
    the lower-left reference point `ref = (rx, ry)`. Only points strictly above ref
    in both objectives contribute. Computed as the area of the union of the
    rectangles [rx, x_i] x [ry, y_i] via an x-descending sweep with a running max
    over y. Passing all labeled points is equivalent to passing only the front.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])  # x descending
    xs = pts[order, 0]
    ys = pts[order, 1]
    x_bounds = np.append(xs, ref[0])  # each x-segment runs down to the next lower x (or ref)
    hv = 0.0
    run_max_y = ref[1]
    for i in range(len(xs)):
        if ys[i] > run_max_y:
            run_max_y = ys[i]
        hv += (x_bounds[i] - x_bounds[i + 1]) * (run_max_y - ref[1])
    return float(hv)


def _unc_decomposition(epistds, alestds, totalstds):
    """
    Mean epistemic / aleatoric / total sigma of a set of predictions plus the
    epistemic share of the predictive VARIANCE. The share is pooled
    (mean(epistd^2) / mean(totalstd^2)) rather than the mean of the per-sequence
    ratios, so it is the fraction of the set's total predictive variance that is
    reducible by acquiring more labels — sigma means alone cannot be added or
    compared across the two terms.
    """
    epi = np.asarray(epistds, dtype=float)
    ale = np.asarray(alestds, dtype=float)
    tot = np.asarray(totalstds, dtype=float)
    if epi.size == 0:
        return {'mean_epistd': float('nan'), 'mean_alestd': float('nan'),
                'mean_totalstd': float('nan'), 'epi_var_frac': float('nan')}
    tot_var = float(np.mean(tot ** 2))
    return {
        'mean_epistd':   float(np.mean(epi)),
        'mean_alestd':   float(np.mean(ale)),
        'mean_totalstd': float(np.mean(tot)),
        'epi_var_frac':  float(np.mean(epi ** 2) / tot_var) if tot_var > 0 else float('nan'),
    }


class MultiObjectiveBOLoop:
    """
    Retrospective multi-objective (two-antigen) Bayesian optimization loop.

    Extends the single-objective BOLoop to jointly optimize antibody binding
    against TWO antigens (e.g. C05 vs M22 and SI06). Sequences assayed against
    both antigens are joined on the sequence string; each round trains ONE
    ModelArcEnsemble surrogate per antigen on the shared labeled pool, scores the
    candidate pool per antigen with the same gate-margin acquisition the one-shot
    proposal / GA use, and then SCALARIZES the per-antigen acquisition scores
    across antigens (combine='min' -> cross-reactive, or 'mean') to pick the
    batch. This makes the BO objective definitionally identical to the GA's
    make_combined_score_fn.
    """

    def __init__(self, config_paths, antigen_names=None, combine='min',
                 acquisition='ei', unc_weight=1.0, batch_size=20, n_rounds=10,
                 n_init=200, held_out_frac=0.2, epochs=None, seed=42, out_dir=None,
                 acq_sigma_col='totalstd', acq_normalize='gate_margin',
                 adaptive_weight=False, init_max_mut=None, out_group='_mo_bo_runs', device=None):
        if len(config_paths) != 2:
            raise ValueError(f"MultiObjectiveBOLoop expects exactly 2 antigen configs, got {len(config_paths)}.")
        self.combine = combine
        self.acquisition = acquisition
        self.unc_weight = unc_weight
        self.adaptive_weight = adaptive_weight
        self.acq_sigma_col = acq_sigma_col
        self.acq_normalize = acq_normalize
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.n_init = n_init
        self.held_out_frac = held_out_frac
        self.init_max_mut = init_max_mut
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # --- Load both antigens and join on the shared sequence -----------------
        antigen_names = antigen_names or [None, None]
        self.antigens = []
        per_ag_frames = []
        wt_seen = None
        for idx, (cfg_path, name_override) in enumerate(zip(config_paths, antigen_names)):
            full = load_config(cfg_path, encoding="onehot")
            config = full['global']
            config['cnn_encoder'] = 'onehot_cnn'
            if config['encoder'] != 'onehot':
                raise ValueError(
                    f"MultiObjectiveBOLoop currently only supports the 'onehot' encoder, "
                    f"got '{config['encoder']}' for {config['dataset']}."
                )
            wt = config['wt']
            if wt_seen is None:
                wt_seen = wt
            elif wt != wt_seen:
                raise ValueError(
                    "The two antigen configs have different WT sequences "
                    f"('{wt_seen}' vs '{wt}'). A multi-objective run requires the same "
                    "antibody (identical WT / n_mut) assayed against both antigens."
                )
            name = _antigen_name(config['dataset'], idx, name_override)
            model_config = full.get('methods', {}).get('ModelArcEnsemble', {})
            data = load_data_from_csv(config['data_path'], config['seq_col'], config['label_col'], wt)
            self.antigens.append({
                'name': name, 'config': config, 'model_config': model_config,
                'label_col': f'label_{name}',
            })
            per_ag_frames.append(data[['sequence', 'n_mut', 'label']].rename(
                columns={'label': f'label_{name}'}))

        names = [a['name'] for a in self.antigens]
        if names[0] == names[1]:
            raise ValueError(f"The two antigens resolved to the same name '{names[0]}'; "
                             f"pass distinct --antigen_names.")

        # Inner join keeps only sequences assayed against BOTH antigens.
        joined = per_ag_frames[0].merge(
            per_ag_frames[1][['sequence', self.antigens[1]['label_col']]],
            on='sequence', how='inner',
        )
        logging.info(
            f"MO join: {names[0]}={len(per_ag_frames[0])}, {names[1]}={len(per_ag_frames[1])}, "
            f"intersection={len(joined)} sequences assayed against both."
        )
        if joined[[a['label_col'] for a in self.antigens]].isna().any().any():
            raise ValueError("Joined multi-objective frame contains NaN labels — check the inputs.")
        self.data = joined.reset_index(drop=True)
        self.epochs = epochs if epochs is not None else self.antigens[0]['model_config'].get('epochs', 100)

        # Per-antigen true references.
        for ag in self.antigens:
            ag['wt_label'] = resolve_wt_fitness(self.data, label_col=ag['label_col'])
            ag['global_best'] = float(self.data[ag['label_col']].max())
        self.wt_row = self.data[self.data['n_mut'] == 0].iloc[[0]].reset_index(drop=True)

        self.pool = ProteinPool.from_dataframe(
            self.data, n_init=n_init, seed=seed, held_out_frac=held_out_frac,
            init_max_mut=init_max_mut,
        )

        # Cross-reactive reference counts on the ground-truth pool (labeled + candidate).
        ground_truth = pd.concat([self.pool.labeled_pool, self.pool.candidate_pool], ignore_index=True)
        self.n_above_both_total = int(self._above_both_mask(ground_truth).sum())

        # Combined dataset tag, e.g. Chen-2026-C05_M22+SI06_ED5_BO
        ds0 = self.antigens[0]['config']['dataset']
        combined_dataset = ds0.replace(names[0], f'{names[0]}+{names[1]}')
        self.dataset_tag = combined_dataset

        if out_dir is None:
            weight_tag = 'adaptive' if adaptive_weight else unc_weight
            init_tag = f"_initmut{init_max_mut}" if init_max_mut is not None else ""
            run_id = (f"MOEnsemble_{combine}_{acquisition}_w{weight_tag}"
                      f"_{acq_sigma_col}_{acq_normalize}{init_tag}_seed{seed}")
            out_dir = os.path.join('results', combined_dataset, out_group or '_mo_bo_runs', run_id)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)


        self.round_metrics = []

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _view(df, label_col):
        """Per-antigen single-objective view: sequence, n_mut, label=<that antigen>."""
        out = df[['sequence', 'n_mut']].copy()
        out['label'] = df[label_col].to_numpy()
        return out

    def _above_both_mask(self, df):
        m = np.ones(len(df), dtype=bool)
        for ag in self.antigens:
            m &= (df[ag['label_col']].to_numpy() > ag['wt_label'])
        return m

    def _train_score_antigen(self, ag, cand_df, round_idx):
        """
        Train antigen `ag`'s surrogate on the current labeled pool and return its
        per-candidate acquisition score (aligned to cand_df order) plus the
        diagnostics needed for metrics. Mirrors one antigen's slice of BOLoop.run.
        """
        config, model_config = ag['config'], ag['model_config']
        label_col = ag['label_col']

        fitted = fit_round(
            self._view(self.pool.labeled_pool, label_col), config, model_config,
            self.device, epochs=self.epochs, seed=self.seed,
            checkpoint_dir=os.path.join(self.out_dir, 'checkpoints', f"{ag['name']}_round_{round_idx}"),
        )

        wt_pred = fitted.predict(self._view(self.wt_row, label_col))["pred"]
        pred_threshold = float(wt_pred[0])

        ei_threshold = None
        if self.acquisition == 'ei':
            labeled_preds = fitted.predict(self._view(self.pool.labeled_pool, label_col))["pred"]
            ei_threshold = float(np.max(labeled_preds))

        raw = fitted.predict(self._view(cand_df, label_col))
        preds, epistds, alestds, totalstds = (raw[k] for k in ("pred", "epistd", "alestd", "totalstd"))

        held_view = self._view(self.pool.held_out_eval, label_col)
        held = fitted.predict(held_view)
        held_preds, held_epistds, held_alestds, held_totalstds = (held[k] for k in ("pred", "epistd", "alestd", "totalstd"))
        held_labels = held_view["label"].to_numpy()
        norm_ref = {
            'pred':     (float(np.mean(held_preds)),     float(np.std(held_preds))),
            'epistd':   (float(np.mean(held_epistds)),   float(np.std(held_epistds))),
            'alestd':   (float(np.mean(held_alestds)),   float(np.std(held_alestds))),
            'totalstd': (float(np.mean(held_totalstds)), float(np.std(held_totalstds))),
        }

        round_unc_weight = self.unc_weight
        if self.adaptive_weight:
            held_sigma = {'epistd': held_epistds, 'alestd': held_alestds,
                          'totalstd': held_totalstds}[self.acq_sigma_col]
            held_abs_err = np.abs(held_preds - held_labels)
            rho, _ = spearmanr(held_sigma, held_abs_err)
            round_unc_weight = max(0.0, float(rho)) if rho == rho else 0.0

        # Per-antigen acquisition score with gate-margin normalization so the two
        # antigens are comparable before the cross-antigen scalarization.
        score_df = pd.DataFrame({
            'sequence': cand_df['sequence'].to_numpy(),
            'pred': preds, 'epistd': epistds, 'alestd': alestds, 'totalstd': totalstds,
            'split_label': 'test',
        })
        scored = propose_candidates(
            score_df, pred_threshold, unc_weight=round_unc_weight,
            pred_col='pred', sigma_col=self.acq_sigma_col, cal_sigma_col=None,
            det_method=(self.acquisition == 'greedy'),
            normalize=self.acq_normalize, ei_threshold=ei_threshold, norm_ref=norm_ref,
        )
        # propose_candidates preserves row order and returns all 'test' rows.
        assert len(scored) == len(cand_df), "propose_candidates changed row count/order"
        score = scored[ACQUISITION_SCORE_COL[self.acquisition]].to_numpy()

        held_metrics = regression_metrics(held_preds, held_labels)
        held_metrics.update(uq_metrics(held_preds, held_totalstds, held_labels))
        # Uncertainty decomposition of the held-out set (totalstd^2 = epistd^2 +
        # alestd^2, see ModelArcEnsembleTrainer.evaluate). Epistemic should shrink as
        # rounds add labels; aleatoric is assay/label noise and should not, so their
        # ratio says whether more acquisition can still help.
        held_metrics.update(_unc_decomposition(held_epistds, held_alestds, held_totalstds))

        return {
            'score': score, 'pred': preds, 'totalstd': totalstds,
            'epistd': epistds, 'alestd': alestds,
            'pred_threshold': pred_threshold, 'unc_weight_used': round_unc_weight,
            'held_metrics': held_metrics,
        }

    def run(self):
        for round_idx in range(self.n_rounds):
            if self.pool.candidate_pool.empty:
                break
            logging.info(
                f"=== MO BO round {round_idx} | labeled={len(self.pool.labeled_pool)} "
                f"candidates={len(self.pool.candidate_pool)} ==="
            )
            cand_df = self.pool.candidate_pool.copy()

            if self.acquisition == 'random':
                selected = cand_df.sample(
                    n=min(self.batch_size, len(cand_df)),
                    random_state=self.seed * 1000 + round_idx,
                )
                per_ag = {}
                combined_score = np.full(len(cand_df), np.nan)
            else:
                per_ag = {ag['name']: self._train_score_antigen(ag, cand_df, round_idx)
                          for ag in self.antigens}
                score_mat = np.column_stack([per_ag[ag['name']]['score'] for ag in self.antigens])
                combined_score = score_mat.mean(axis=1) if self.combine == 'mean' else score_mat.min(axis=1)
                cand_df = cand_df.assign(combined_score=combined_score)
                selected = cand_df.nlargest(min(self.batch_size, len(cand_df)), 'combined_score')

            # Batch signature (before acquire moves rows into labeled_pool).
            sel_mask = self.pool.candidate_pool.index.isin(selected.index)
            batch_mean_n_mut = float(selected['n_mut'].mean()) if 'n_mut' in selected.columns else float('nan')
            batch_mean_min_hamming = _mean_min_hamming_distance(
                selected['sequence'].tolist(), self.pool.labeled_pool['sequence'].tolist(), seed=self.seed)
            batch_mean_combined = (float(np.nanmean(combined_score[sel_mask]))
                                   if self.acquisition != 'random' else float('nan'))

            acquired = self.pool.acquire(selected.index, round_idx)

            metrics = {
                'round': round_idx,
                'n_labeled': len(self.pool.labeled_pool),
                'n_candidates': len(self.pool.candidate_pool),
                'combine': self.combine,
                'batch_size_actual': len(acquired),
                'batch_mean_n_mut': batch_mean_n_mut,
                'batch_mean_min_hamming_to_labeled': batch_mean_min_hamming,
                'batch_mean_combined_score': batch_mean_combined,
            }

            # Per-antigen metrics.
            for ag in self.antigens:
                name, label_col = ag['name'], ag['label_col']
                best = float(self.pool.labeled_pool[label_col].max())
                metrics[f'best_label_{name}_so_far'] = best
                metrics[f'regret_{name}'] = float(ag['global_best'] - best)
                metrics[f'wt_label_{name}'] = ag['wt_label']
                res = per_ag.get(name)
                if res is not None:
                    metrics[f'unc_weight_used_{name}'] = res['unc_weight_used']
                    metrics[f'pred_threshold_wt_{name}'] = res['pred_threshold']
                    metrics[f'batch_mean_pred_{name}'] = float(np.mean(res['pred'][sel_mask]))
                    # Same decomposition for the batch actually acquired (what the
                    # acquisition chased) and for the whole candidate pool (what the
                    # surrogate still does not know about the search space).
                    for k, v in _unc_decomposition(res['epistd'][sel_mask],
                                                   res['alestd'][sel_mask],
                                                   res['totalstd'][sel_mask]).items():
                        metrics[f'batch_{k}_{name}'] = v
                    for k, v in _unc_decomposition(res['epistd'], res['alestd'],
                                                   res['totalstd']).items():
                        metrics[f'cand_{k}_{name}'] = v
                    metrics.update({f'held_{name}_{k}': v for k, v in res['held_metrics'].items()})

            # Cross-reactive (above WT for BOTH antigens).
            acq_above_both = int(self._above_both_mask(acquired).sum())
            labeled_above_both = int(self._above_both_mask(self.pool.labeled_pool).sum())
            metrics['batch_n_above_both'] = acq_above_both
            metrics['batch_cross_reactive_hit_rate'] = (
                acq_above_both / len(acquired) if len(acquired) > 0 else None)
            metrics['cum_n_above_both'] = labeled_above_both
            metrics['cum_recall_above_both'] = (
                labeled_above_both / self.n_above_both_total if self.n_above_both_total > 0 else None)

            # Pareto front + dominated hypervolume of the labeled pool over the two
            # true objectives, referenced to the (WT, WT) corner.
            obj = np.column_stack([self.pool.labeled_pool[ag['label_col']].to_numpy()
                                   for ag in self.antigens])
            ref = np.array([ag['wt_label'] for ag in self.antigens])
            metrics['n_pareto_front'] = int(_pareto_front_mask(obj).sum())
            metrics['hypervolume'] = _hypervolume_2d(obj, ref)

            self.round_metrics.append(metrics)
            logging.info(" | ".join(f"{k}={v}" for k, v in metrics.items()))
            self._save_state()

        return pd.DataFrame(self.round_metrics)

    def _save_state(self):
        self.pool.save_state(self.out_dir)
        pd.DataFrame(self.round_metrics).to_csv(
            os.path.join(self.out_dir, 'round_metrics.csv'), index=False)
