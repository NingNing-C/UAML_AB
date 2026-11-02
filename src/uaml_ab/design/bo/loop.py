import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from uaml_ab.data import load_data_from_csv, resolve_wt_fitness
from uaml_ab.config import load_config
from uaml_ab.metrics import regression_metrics, uq_metrics

from .pool import ProteinPool
from .round import fit_round
from ..acquisition import select_batch


def _mean_min_hamming_distance(query_seqs, reference_seqs, max_reference=2000, seed=0):
    """
    Mean over query_seqs of the minimum Hamming distance to any sequence in
    reference_seqs. Used as a per-round "novelty" signal for the acquired batch
    relative to the labeled pool. reference_seqs is subsampled to max_reference
    to bound cost as the labeled pool grows.
    """
    if len(query_seqs) == 0 or len(reference_seqs) == 0:
        return float('nan')

    if len(reference_seqs) > max_reference:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(reference_seqs), size=max_reference, replace=False)
        reference_seqs = [reference_seqs[i] for i in idx]

    q = np.array([list(s) for s in query_seqs])
    r = np.array([list(s) for s in reference_seqs])
    dists = (q[:, None, :] != r[None, :, :]).sum(axis=2)
    return float(dists.min(axis=1).mean())


class BOLoop:
    """
    Retrospective Bayesian optimization loop for a single dataset, using
    ModelArcEnsemble as the surrogate.

    Each round:
      1. Retrain ModelArcEnsemble from scratch on the current labeled_pool.
      2. Score the candidate_pool with the chosen acquisition function.
      3. Reveal the true labels of the top `batch_size` candidates (oracle
         lookup, since this is a retrospective simulation) and move them into
         labeled_pool.
      4. Evaluate the surrogate on the fixed held_out_eval slice to track
         sample-efficiency / calibration trajectories.
    """

    def __init__(self, config_path, acquisition='ei', unc_weight=1.0,
                 batch_size=20, n_rounds=10, n_init=200, held_out_frac=0.2,
                 epochs=None, seed=42, out_dir=None,
                 acq_sigma_col='totalstd', acq_normalize='gate_margin',
                 adaptive_weight=False, init_max_mut=None, device=None):
        self.acquisition = acquisition
        self.unc_weight = unc_weight
        # When True, unc_weight is (re)set each round to the surrogate's held-out
        # Spearman(|error|, sigma) — the same rank-based, calibration-free trust
        # signal the one-shot candidate proposal and the GA use. It anneals
        # naturally: as the surrogate improves the correlation shrinks, so the
        # explore/exploit weight shrinks with it.
        self.adaptive_weight = adaptive_weight
        self.acq_sigma_col = acq_sigma_col
        self.acq_normalize = acq_normalize
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.n_init = n_init
        # Optional cap on the number of mutations a sequence may have to be
        # eligible for the *initial* labeled seed. None -> random init over the
        # whole pool. Higher-mutation sequences remain acquirable in later rounds.
        self.init_max_mut = init_max_mut
        self.held_out_frac = held_out_frac
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        full_config = load_config(config_path, encoding="onehot")
        self.model_config = full_config.get('methods', {}).get('ModelArcEnsemble', {})
        self.config = full_config['global']
        self.config['cnn_encoder'] = 'onehot_cnn'
        if self.config['encoder'] != 'onehot':
            raise ValueError(
                f"BOLoop currently only supports the 'onehot' encoder, got '{self.config['encoder']}'."
            )

        self.epochs = epochs if epochs is not None else self.model_config.get('epochs', 100)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        data = load_data_from_csv(
            self.config['data_path'], self.config['seq_col'], self.config['label_col'], self.config['wt']
        )


        self.wt_label = resolve_wt_fitness(data)
        self.wt_row = data[data['n_mut'] == 0].iloc[[0]].reset_index(drop=True)

        self.pool = ProteinPool.from_dataframe(
            data, n_init=n_init, seed=seed, held_out_frac=held_out_frac,
            init_max_mut=init_max_mut,
        )
        self.n_above_wt_total = int((self.pool.candidate_pool['label'] > self.wt_label).sum())
        self.global_best_label = float(data['label'].max())

        if out_dir is None:
            weight_tag = 'adaptive' if adaptive_weight else unc_weight
            init_tag = f"_initmut{init_max_mut}" if init_max_mut is not None else ""
            run_id = (f"ModelArcEnsemble_{acquisition}_w{weight_tag}"
                      f"_{acq_sigma_col}_{acq_normalize}{init_tag}_seed{seed}")
            out_dir = os.path.join('results', self.config['dataset'], '_bo_runs', run_id)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)


        self.round_metrics = []

    def run(self):
        for round_idx in range(self.n_rounds):
            if self.pool.candidate_pool.empty:
                break
            logging.info(
                f"=== BO round {round_idx} | labeled={len(self.pool.labeled_pool)} "
                f"candidates={len(self.pool.candidate_pool)} ==="
            )

            fitted = fit_round(
                self.pool.labeled_pool, self.config, self.model_config,
                self.device, epochs=self.epochs, seed=self.seed,
                checkpoint_dir=os.path.join(self.out_dir, 'checkpoints', f'round_{round_idx}'),
            )

            # Round-specific acquisition threshold: the surrogate's own WT prediction.
            # Used as the feasibility cutoff for the *_candidate booleans/metrics and
            # as the improvement reference for EI by default.
            wt_pred = fitted.predict(self.wt_row)["pred"]
            pred_threshold = float(wt_pred[0])

            # Standard EI improves over the *incumbent* (best observed so far), not
            # over WT. WT sits far below the frontier, so EI-over-WT saturates into
            # pure exploitation (~greedy). We therefore reference EI to the best
            # predicted mean over the labeled pool — in the surrogate's own pred
            # space, so it is comparable to candidate preds (labels are on a
            # different offset). Computed only for EI runs to avoid an extra
            # labeled-pool forward pass for the other acquisitions.
            ei_threshold = None
            if self.acquisition == 'ei':
                labeled_preds = fitted.predict(self.pool.labeled_pool)["pred"]
                ei_threshold = float(np.max(labeled_preds))

            # Score the remaining candidate pool
            cand_df = self.pool.candidate_pool.copy()
            raw = fitted.predict(cand_df)
            preds, epistds, alestds, totalstds = (raw[k] for k in ("pred", "epistd", "alestd", "totalstd"))
            cand_df['pred'] = preds
            cand_df['epistd'] = epistds
            cand_df['alestd'] = alestds
            cand_df['totalstd'] = totalstds
            cand_df['split_label'] = 'test'

            # Held-out evaluation set, predicted once here (the surrogate is fixed
            # for this round; acquiring does not touch held_out_eval). Used both as
            # the fixed normalization reference for acquisition and for the held-out
            # metrics below. Anchoring the z-score to this fixed representative
            # sample — rather than the candidate pool, which depletes as the best
            # sequences are acquired — keeps unc_weight comparable across rounds and
            # makes the reference the same kind GA/one-shot use (a held-out split).
            held = fitted.predict(self.pool.held_out_eval)
            held_preds, held_epistds, held_alestds, held_totalstds = (held[k] for k in ("pred", "epistd", "alestd", "totalstd"))
            held_labels = self.pool.held_out_eval["label"].to_numpy()
            norm_ref = {
                'pred':     (float(np.mean(held_preds)),     float(np.std(held_preds))),
                'epistd':   (float(np.mean(held_epistds)),   float(np.std(held_epistds))),
                'alestd':   (float(np.mean(held_alestds)),   float(np.std(held_alestds))),
                'totalstd': (float(np.mean(held_totalstds)), float(np.std(held_totalstds))),
            }

            # Effective explore/exploit weight for this round. In adaptive mode it
            # is the surrogate's held-out Spearman(|error|, sigma) — computed on
            # the same held-out split and sigma used for acquisition, before any
            # batch is selected, so it never peeks at the acquisition outcome.
            # Clamped to >= 0 (a negative correlation would flip LCB/UCB).
            round_unc_weight = self.unc_weight
            if self.adaptive_weight:
                held_sigma = {'epistd': held_epistds, 'alestd': held_alestds,
                              'totalstd': held_totalstds}[self.acq_sigma_col]
                held_abs_err = np.abs(held_preds - held_labels)
                rho, _ = spearmanr(held_sigma, held_abs_err)
                round_unc_weight = max(0.0, float(rho)) if rho == rho else 0.0
                logging.info(f"  adaptive unc_weight (held-out spearman) = {round_unc_weight:.3f}")

            # Optional diagnostic: persist the full per-candidate score table so
            # acquisition variants (epistemic vs total sigma, ranking, clipping,
            # diversity, ...) can be back-tested offline from a single trained
            # round without re-running the GPU loop. Off by default — the table
            # is one row per candidate, so it is only dumped when explicitly
            # requested (e.g. an n_rounds=1 diagnostic run).
            if getattr(self, 'dump_candidate_scores', False):
                dump_cols = [c for c in ['sequence', 'pred', 'epistd', 'alestd',
                                         'totalstd', 'label', 'n_mut']
                             if c in cand_df.columns]
                cand_df[dump_cols].to_csv(
                    os.path.join(self.out_dir, f'candidate_scores_round{round_idx}.csv'),
                    index=False,
                )

            selected = select_batch(
                cand_df, pred_threshold, self.acquisition, round_unc_weight, self.batch_size,
                random_state=self.seed * 1000 + round_idx,
                sigma_col=getattr(self, 'acq_sigma_col', 'totalstd'),
                normalize=getattr(self, 'acq_normalize', 'gate_margin'),
                ei_threshold=ei_threshold,
                norm_ref=norm_ref,
            )

            # Exploration/exploitation signature of the acquired batch, computed
            # before pool.acquire() moves it into labeled_pool.
            batch_mean_pred = float(selected['pred'].mean())
            batch_mean_totalstd = float(selected['totalstd'].mean())
            batch_mean_epistd = float(selected['epistd'].mean()) if 'epistd' in selected.columns else float('nan')
            batch_mean_alestd = float(selected['alestd'].mean()) if 'alestd' in selected.columns else float('nan')
            batch_mean_n_mut = float(selected['n_mut'].mean()) if 'n_mut' in selected.columns else float('nan')
            batch_mean_min_hamming_to_labeled = _mean_min_hamming_distance(
                selected['sequence'].tolist(), self.pool.labeled_pool['sequence'].tolist(), seed=self.seed,
            )

            acquired = self.pool.acquire(selected.index, round_idx)

            # Held-out evaluation with the same (just-trained) surrogate. The
            # predictions were already computed above (for the normalization
            # reference); reuse them rather than predict held_out_eval twice.
            held_metrics = regression_metrics(held_preds, held_labels)
            held_metrics.update(uq_metrics(held_preds, held_totalstds, held_labels))

            # The held-out set is a random sample, so its overall metrics can
            # drift over rounds as the surrogate specializes towards
            # high-fitness sequences. Track the same metrics restricted to the
            # held-out sequences with the highest true labels (top 1% and top
            # 20), where this drift is most relevant.
            order = np.argsort(held_labels)[::-1]
            n_top1pct = max(1, int(np.ceil(len(held_labels) * 0.01)))
            top1pct_idx = order[:n_top1pct]
            top20_idx = order[:20]

            top1pct_metrics = regression_metrics(held_preds[top1pct_idx], held_labels[top1pct_idx])
            top1pct_metrics.update(uq_metrics(held_preds[top1pct_idx], held_totalstds[top1pct_idx], held_labels[top1pct_idx]))
            top1pct_metrics['mean_totalstd'] = float(np.mean(held_totalstds[top1pct_idx]))

            top20_metrics = regression_metrics(held_preds[top20_idx], held_labels[top20_idx])
            top20_metrics.update(uq_metrics(held_preds[top20_idx], held_totalstds[top20_idx], held_labels[top20_idx]))
            top20_metrics['mean_totalstd'] = float(np.mean(held_totalstds[top20_idx]))

            n_acquired_above_wt = int((acquired['label'] > self.wt_label).sum())
            n_labeled_above_wt = int((self.pool.labeled_pool['label'] > self.wt_label).sum())
            cum_recall_above_wt = (
                n_labeled_above_wt / self.n_above_wt_total if self.n_above_wt_total > 0 else None
            )

            metrics = {
                'round': round_idx,
                'n_labeled': len(self.pool.labeled_pool),
                'n_candidates': len(self.pool.candidate_pool),
                'unc_weight_used': round_unc_weight,
                'pred_threshold_wt': pred_threshold,
                'wt_label': self.wt_label,
                'best_label_so_far': float(self.pool.labeled_pool['label'].max()),
                'top20_median_label_so_far': float(self.pool.labeled_pool['label'].nlargest(20).median()),
                'regret': float(self.global_best_label - self.pool.labeled_pool['label'].max()),
                'batch_n_above_wt': n_acquired_above_wt,
                'batch_size_actual': len(acquired),
                'batch_hit_rate': n_acquired_above_wt / len(acquired) if len(acquired) > 0 else None,
                'cum_n_above_wt': n_labeled_above_wt,
                'cum_recall_above_wt': cum_recall_above_wt,
                'batch_mean_pred': batch_mean_pred,
                'batch_mean_totalstd': batch_mean_totalstd,
                'batch_mean_epistd': batch_mean_epistd,
                'batch_mean_alestd': batch_mean_alestd,
                'batch_mean_n_mut': batch_mean_n_mut,
                'batch_mean_min_hamming_to_labeled': batch_mean_min_hamming_to_labeled,
                'held_mean_totalstd': float(np.mean(held_totalstds)),
                'held_mean_epistd': float(np.mean(held_epistds)),
                'held_mean_alestd': float(np.mean(held_alestds)),
            }
            metrics.update({f'held_{k}': v for k, v in held_metrics.items()})
            metrics.update({f'held_top1pct_{k}': v for k, v in top1pct_metrics.items()})
            metrics.update({f'held_top20_{k}': v for k, v in top20_metrics.items()})

            self.round_metrics.append(metrics)
            logging.info(" | ".join(f"{k}={v}" for k, v in metrics.items()))


            self._save_state()


        return pd.DataFrame(self.round_metrics)

    def _save_state(self):
        self.pool.save_state(self.out_dir)
        pd.DataFrame(self.round_metrics).to_csv(os.path.join(self.out_dir, 'round_metrics.csv'), index=False)
