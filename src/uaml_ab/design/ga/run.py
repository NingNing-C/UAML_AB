"""
Driver for the GA candidate-proposal pipeline, scoped to ONE pre-trained
surrogate (one antigen x ED dataset). Runs every acquisition / sigma / weight /
seeding / ga-seed arm in a single process so they share one per-surrogate
PredictionCache (acquisition variety is then nearly free on the model-eval
budget; see prediction_cache.py).

Outputs (under results/<dataset>/_ga_runs/):
  prediction_cache.csv   shared eval ledger = retrospective universe
  weights.json               seed-42 calibration weights + Spearman diagnostics
  <arm_id>/                   shortlist.csv, genealogy.csv, trajectories.csv,
                             insilico_metrics.json, resolved_config.yaml

Run select_candidates.py to pool the per-search shortlists for final candidate selection.
"""
import argparse
import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml


from uaml_ab.data import load_data_from_csv, split_data_random, resolve_wt_fitness  # noqa: E402
from uaml_ab.config import load_config
from uaml_ab.records import source_fingerprint
from scipy.stats import spearmanr  # noqa: E402

from .alphabet import build_position_alphabet, build_seed_pool  # noqa: E402
from .cache import PredictionCache  # noqa: E402
from .fitness import load_surrogate, make_predict_fn, make_combined_score_fn, SIGMA_COL  # noqa: E402
from .engine import GAEngine  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="uaml-ab design", description="GA candidate proposal (frozen ModelArcEnsemble surrogate)")
    # Single-objective OR multi-objective via the *_s/plural args.
    p.add_argument("--config", default=None, help="Single-antigen dataset config.yaml")
    p.add_argument("--model_path", "--checkpoint", default=None, help="Single-antigen pre-trained .pth")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Multi-objective: one config per antigen (aligned with --model_paths)")
    p.add_argument("--model_paths", "--checkpoints", nargs="+", default=None,
                   help="Multi-objective: one .pth per antigen (aligned with --configs)")
    p.add_argument("--antigen_names", nargs="+", default=None,
                   help="Optional short names per antigen (else derived from each dataset)")
    p.add_argument("--combine", choices=["min", "mean"], default="min",
                   help="How to scalarise across antigens: min=cross-reactive (good for ALL), mean=average")
    p.add_argument("--method", choices=["ModelArcEnsemble"], default="ModelArcEnsemble")
    p.add_argument("--acquisitions", nargs="+", choices=["greedy", "ei", "lcb"], default=["greedy", "ei", "lcb"])
    p.add_argument("--sigma_types", nargs="+", choices=["total", "epistemic"], default=["total"],
                   help="Uncertainty source(s) for ei/lcb arms (default total only — the "
                        "posterior-predictive variance of the observed label; pass 'epistemic' too "
                        "to also run epistemic-only arms)")
    p.add_argument("--acq_normalize", type=str, default="gate_margin",
                   choices=["gate_margin", "zscore", "none"],
                   help="How pred/sigma are put on a scale-free footing (fixed reference stats "
                        "from the checkpoint reference split, so GA fitness is stationary). "
                        "'gate_margin' (default): every objective is scored as pred's "
                        "standard-deviation margin above ITS OWN gate — margin=(pred-gate)/pred_sd, "
                        "sigma/pred_sd — so greedy/EI/LCB and the cross-reactive min are all "
                        "comparable across antigens (no antigen dominates the min by raw scale). "
                        "'zscore': mean-centered z-scores of pred and sigma (greedy/EI unaffected). "
                        "'none': raw-scale scoring.")
    p.add_argument("--weight_modes", nargs="+", choices=["calibrated", "unweighted"], default=["calibrated", "unweighted"],
                   help="calibrated=Spearman(sigma,|err|) on checkpoint reference split; unweighted=1.0")
    p.add_argument("--seedings", nargs="+", choices=["top_binders", "random_wt"], default=["top_binders", "random_wt"])
    p.add_argument("--ga_seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--generations", type=int, default=300,
                   help="Hard ceiling on generations (backstop; the run stops earlier when the qualifier shortlist stalls)")
    p.add_argument("--trust_radius", type=int, default=11, help="MAX mutations from WT")
    p.add_argument("--min_mut", type=int, default=0,
                   help="MIN mutations from WT for every proposal (0 = off). E.g. 6 forces "
                        "proposals strictly beyond an ED2 model's 1-5 training range (fully OOD) "
                        "to probe generalisation into deeper mutational space.")
    p.add_argument("--n_select", type=int, default=15)
    p.add_argument("--crossover_rate", type=float, default=0.7)
    p.add_argument("--mutation_rate", type=float, default=0.06)
    p.add_argument("--tournament_k", type=int, default=3)
    p.add_argument("--elitism_frac", type=float, default=0.02,
                   help="Fraction of the population carried over UNCHANGED to the next "
                        "generation (the fittest are copied verbatim). n_elite = "
                        "max(1, round(elitism_frac * pop)); e.g. 0.02 * pop 3000 = 60 elites. "
                        "Higher = more exploitation / faster convergence but less turnover; "
                        "0 keeps only the forced floor of 1 elite.")
    p.add_argument("--early_stop_patience", type=int, default=25,
                   help="Stall patience: stop after this many generations with no improvement in the qualifier shortlist (ignored when --max_evals is set)")
    p.add_argument("--max_evals", type=int, default=None,
                   help="Fixed evaluation budget PER ARM (sequences scored). When set, it "
                        "supersedes early stopping: every arm runs until it has scored this many "
                        "sequences, even without improvement, so all arms do equal search work. "
                        "gen 0 scores `pop`, each later generation scores `pop - n_elite`, so "
                        "generations ≈ (max_evals - pop)/(pop - round(0.02*pop)); keep --generations "
                        "above that so the budget (not the ceiling) is what stops the arm.")
    p.add_argument("--alphabet_mode", choices=["library", "all"], default="library",
                   help="Residues allowed at the (library-variable) mutable positions: "
                        "library=only observed residues; all=all 20 AAs (restricted by position, not residue)")
    p.add_argument("--seed_pool_size", type=int, default=256)
    p.add_argument("--sigma_cap_mult", type=float, default=3.0,
                   help="Clamp predictive sigma at this x the in-distribution 99.9th percentile")
    p.add_argument("--sigma_cap_sample", type=int, default=4000,
                   help="Training-data sample size used to estimate the sigma cap")
    p.add_argument("--gate_percentile", type=float, default=99.0,
                   help="Gate threshold = this percentile of the model's predictions over the "
                        "antigen's train/val/test set (a novel proposal must beat it). 100 = the "
                        "absolute training max, which is essentially unreachable for novel "
                        "sequences (the surrogate rarely extrapolates above its training peak); "
                        "99 = 'better than ~all assayed sequences' and is reachable. Applied "
                        "identically to every antigen.")
    p.add_argument("--gate_max_sample", type=int, default=None,
                   help="Estimate the gate threshold from this many sampled sequences "
                        "(default: full dataset; sampling is an approximation for quick tests only)")
    p.add_argument("--budget", type=int, default=None, help="Max distinct surrogate evaluations (optional)")
    p.add_argument("--out_dir", "--output_dir", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    if min(args.pop, args.generations, args.n_select, args.seed_pool_size, args.sigma_cap_sample) < 1:
        p.error("Population, generation, selection and reference sample counts must be positive.")
    if args.min_mut < 0 or args.trust_radius < args.min_mut:
        p.error("Mutation bounds must satisfy 0 <= min_mut <= trust_radius.")
    if not 0 <= args.gate_percentile <= 100 or args.sigma_cap_mult <= 0:
        p.error("gate_percentile must be in [0,100] and sigma_cap_mult must be positive.")
    if not 0 <= args.mutation_rate <= 1 or not 0 <= args.crossover_rate <= 1 or not 0 <= args.elitism_frac < 1:
        p.error("Mutation/crossover rates must be in [0,1], and elitism_frac in [0,1).")
    if args.tournament_k < 1:
        p.error("tournament_k must be positive.")
    return args


def _antigen_name(dataset, idx, override=None):
    if override:
        return override
    m = re.search(r"(SI06|M22)", dataset)
    return m.group(1) if m else f"ag{idx}"


def load_antigen(cfg_path, model_path, name_override, idx, args, device):
    """Load one antigen's config/data/frozen surrogate, build its shared cache and
    derive its calibration weights, EI threshold (WT pred) and the gate threshold
    (max model prediction over its full train/val/test set)."""
    full = load_config(cfg_path, encoding="onehot")
    config = full["global"]
    config.setdefault("cnn_encoder", "onehot_cnn")
    if config["encoder"] != "onehot":
        raise ValueError(f"GA requires the 'onehot' encoder, got '{config['encoder']}'.")
    model_config = full.get("methods", {}).get(args.method, {})
    wt = config["wt"]
    dataset = config["dataset"]
    name = _antigen_name(dataset, idx, name_override)

    data = load_data_from_csv(config["data_path"], config["seq_col"], config["label_col"], wt)
    fitted = load_surrogate(config, model_config, model_path, device, data)
    metadata = fitted.metadata
    if metadata.get("no_saved_test_partition"):
        raise ValueError("This design checkpoint has no saved held-out calibration partition; use an ordinary training checkpoint with recorded test rows.")
    calibration_seed = int(metadata.get("split_seed", fitted.seed))
    stored_indices = metadata.get("splits", {}).get("test")
    if metadata.get("source_fingerprint") is not None:
        if source_fingerprint(full, data, "onehot") != metadata["source_fingerprint"]:
            raise ValueError(f"[{name}] GA configuration data or column mapping differs from the checkpoint's training source.")
    if stored_indices is not None:
        if not metadata.get("source_fingerprint"):
            raise ValueError("Saved calibration partitions require a training source fingerprint.")
        if not stored_indices or len(stored_indices) != len(set(stored_indices)) or not set(stored_indices).issubset(data.index):
            raise ValueError("Checkpoint calibration row identities are empty, duplicated, or absent from the supplied data.")
        test_data = data.loc[stored_indices]
        calibration_source = "checkpoint_held_out"
    else:
        # Legacy checkpoints predate recorded partitions; reconstruct their
        # seeded reference split explicitly rather than labeling it verified.
        _, _, test_data = split_data_random(data, test_size=0.2, val_frac=0.1, random_state=calibration_seed)
        calibration_source = "legacy_reconstructed_split"
        logging.warning("[%s] checkpoint has no saved partitions; reconstructing calibration with seed %s", name, calibration_seed)
    raw_predict = make_predict_fn(fitted)
    sample = data["sequence"].astype(str).sample(
        min(args.sigma_cap_sample, len(data)), random_state=0).tolist()
    s = raw_predict(sample)
    sigma_caps = {f: float(np.nanpercentile(s[f], 99.9) * args.sigma_cap_mult)
                  for f in ("epistd", "alestd", "totalstd")}
    predict_fn = make_predict_fn(fitted, sigma_caps=sigma_caps)
    cache = PredictionCache(predict_fn, budget=args.budget)

    threshold = float(cache.get([wt])["pred"][0])           # WT pred = EI threshold

    raw = cache.get(test_data["sequence"].astype(str).tolist())
    test_scored = pd.DataFrame({"pred": raw["pred"], "epistd": raw["epistd"],
                                "totalstd": raw["totalstd"],
                                "label": test_data["label"].to_numpy()})
    cal = compute_calibration_weights(test_scored, args.sigma_types)

    # gate threshold = the `gate_percentile`-th percentile of model PREDICTIONS over
    # the antigen's data (the bar a novel proposal must beat). The absolute max (pct
    # 100) is near-unreachable for novel sequences, so we default to the 99th pct
    # ("better than ~all assayed sequences"). Full by default; sample for quick tests.
    gate_seqs = data["sequence"].astype(str)
    if args.gate_max_sample and len(gate_seqs) > args.gate_max_sample:
        gate_seqs = gate_seqs.sample(args.gate_max_sample, random_state=0)
    train_preds = cache.get(gate_seqs.tolist())["pred"]
    gate_pred = float(np.percentile(train_preds, args.gate_percentile))

    try:
        wt_label = resolve_wt_fitness(data)
    except Exception as e:
        logging.warning(f"[{name}] could not resolve WT label: {e}")
        wt_label = None

    logging.info(f"[{name}] dataset={dataset} WT_pred={threshold:.3f} "
                 f"gate(p{args.gate_percentile:g})={gate_pred:.3f} "
                 f"train_max={float(np.max(train_preds)):.3f} cal={cal}")
    # Fixed-reference normalization stats (mean, std) per column from the test
    # split, scored by this frozen surrogate. Applied to every generated
    # sequence so the z-scored fitness is stationary across GA generations
    # (in-pool per-population stats would not be).
    norm_ref = {c: (float(test_scored[c].mean()), float(test_scored[c].std()))
                for c in ("pred", "epistd", "totalstd")}

    return dict(name=name, config=config, dataset=dataset, data=data, wt=wt,
                cache=cache, cal=cal, threshold=threshold, gate_pred=gate_pred,
                sigma_caps=sigma_caps, wt_label=wt_label, model_path=model_path,
                test_scored=test_scored, norm_ref=norm_ref,
                calibration_seed=calibration_seed, calibration_source=calibration_source)


def compute_calibration_weights(score_test_df, sigma_types):
    """unc_weight = max(0, Spearman(sigma, |pred-label|)) on held-out reference rows,
    per sigma-type. Returns {sigma_type: {'weight':, 'spearman':}}."""
    abs_err = np.abs(score_test_df["pred"] - score_test_df["label"])
    out = {}
    for st in sigma_types:
        col = SIGMA_COL[st]
        rho, _ = spearmanr(score_test_df[col], abs_err)
        rho = float(rho) if rho == rho else 0.0
        out[st] = {"weight": max(0.0, rho), "spearman": rho}
    return out


def arm_iter(args):
    """Yield (arm_id, acquisition, sigma_type, weight_mode, seeding, ga_seed).
    The per-antigen unc_weight is resolved later (it differs per antigen).
    greedy ignores sigma_type/weight_mode (single arm per seeding/seed)."""
    for seeding in args.seedings:
        for ga_seed in args.ga_seeds:
            for acq in args.acquisitions:
                if acq == "greedy":
                    yield (f"greedy_{seeding}_seed{ga_seed}", acq, "none", "none", seeding, ga_seed)
                    continue
                for st in args.sigma_types:
                    for wm in args.weight_modes:
                        yield (f"{acq}_{st}_{wm}_{seeding}_seed{ga_seed}", acq, st, wm, seeding, ga_seed)


def _arm_weight(weight_mode, sigma_type, antigen):
    """Per-antigen σ-weight for an arm: greedy/none → 0, unweighted → 1,
    calibrated → that antigen's Spearman weight for this sigma_type."""
    if weight_mode in ("none", "greedy0"):
        return 0.0
    if weight_mode == "unweighted":
        return 1.0
    return antigen["cal"][sigma_type]["weight"]


def combined_top_binders(antigens, wt, n):
    """Multi-objective 'top_binders': real sequences assayed against ALL antigens,
    ranked by the MEAN of their measured labels — so seeds are the best *cross-
    reactive* binders rather than the best binders for one antigen (which would
    always favour that antigen). Restricted to the intersection of the antigens'
    assayed sequences (a mean of measured labels needs a measurement for each), and
    each antigen's labels are z-scored over that intersection first so the two
    assays' different scales (e.g. M22 WT≈5.2 vs SI06 WT≈0) can't let one dominate.

    In ADDITION, the single best binder from EACH antigen's own dataset (by that
    antigen's `label`, over its full data — need not be in the intersection) is
    always appended, so the per-antigen extremes seed the search even if they are
    not strong cross-reactively. Deterministic; no RNG."""
    L = len(wt)
    maps = []                                           # per-antigen {sequence: mean label}
    for a in antigens:
        d = a["data"].dropna(subset=["label"])
        s = d.groupby(d["sequence"].astype(str))["label"].mean()
        maps.append(s[[len(k) == L for k in s.index]])
    # per-antigen single top binder (highest label in that antigen's own dataset)
    champions = [m.idxmax() for m in maps if len(m)]
    # cross-reactive ranking over the intersection (mean of z-scored labels)
    common = set(maps[0].index) if maps else set()
    for m in maps[1:]:
        common &= set(m.index)
    common = sorted(common)                             # assayed for ALL antigens
    ranked = []
    if common:
        zs = []
        for m in maps:
            v = m.loc[common].to_numpy(dtype=float)
            zs.append((v - v.mean()) / (v.std() + 1e-9))
        combined = np.mean(zs, axis=0)                  # mean of z-scored labels
        ranked = [common[i] for i in np.argsort(-combined)]
    # top n cross-reactive seeds, then guarantee each antigen's champion is present
    pool = ranked[:n]
    for c in champions:
        if c not in pool:
            pool.append(c)
    return pool


def run_ga(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    # --- resolve single- vs multi-objective antigen list ----------------------
    configs = args.configs or ([args.config] if args.config else None)
    model_paths = args.model_paths or ([args.model_path] if args.model_path else None)
    if not configs or not model_paths:
        raise SystemExit("Provide --config/--model_path (single) or --configs/--model_paths (multi).")
    if len(configs) != len(model_paths):
        raise SystemExit("--configs and --model_paths must have the same length.")
    names_override = args.antigen_names or [None] * len(configs)
    if len(names_override) != len(configs):
        raise SystemExit("--antigen_names must have one name per config.")
    if len(set(n for n in names_override if n is not None)) != len([n for n in names_override if n is not None]):
        raise SystemExit("--antigen_names must be distinct.")

    antigens = [load_antigen(c, m, n, i, args, device)
                for i, (c, m, n) in enumerate(zip(configs, model_paths, names_override))]
    multi = len(antigens) > 1
    primary = antigens[0]                                   # defines WT/alphabet/seeds/splits
    wt = primary["wt"]
    if any(a["wt"] != wt for a in antigens):
        raise SystemExit("All antigens must share the same WT (same antibody / mut_loci).")

    out_dir = args.out_dir or os.path.join("results", primary["dataset"], "_ga_runs")
    os.makedirs(out_dir, exist_ok=True)

    # novelty = not assayed for ANY antigen
    train_seqs = set().union(*[set(a["data"]["sequence"].astype(str)) for a in antigens])

    # gate columns: per-antigen "pred_<name> > gate" (single → plain "pred")
    if multi:
        qualify_cols = [(f"pred_{a['name']}", a["gate_pred"]) for a in antigens]
    else:
        qualify_cols = [("pred", primary["gate_pred"])]
    logging.info(f"{'MULTI' if multi else 'SINGLE'}-objective "
                 f"({[a['name'] for a in antigens]}, combine={args.combine}); gate={qualify_cols}")

    # --- weights.json (per-antigen; single also mirrored at top level) --------
    antigens_meta = {a["name"]: {"dataset": a["dataset"], "calibration": a["cal"],
                                 "wt_pred_threshold": a["threshold"],
                                 "gate_pred": a["gate_pred"], "gate_percentile": args.gate_percentile,
                                 "wt_label": a["wt_label"], "sigma_caps": a["sigma_caps"],
                                 "model_path": a["model_path"], "calibration_source": a["calibration_source"],
                                 "calibration_seed": a["calibration_seed"]} for a in antigens}
    weights_json = {"objective": "multi" if multi else "single", "combine": args.combine,
                    "antigens": antigens_meta, "antigen_order": [a["name"] for a in antigens]}
    if not multi:                                          # single: mirror the primary antigen at the top level
        weights_json.update({"calibration": primary["cal"], "wt_pred_threshold": primary["threshold"],
                             "wt_label": primary["wt_label"], "sigma_caps": primary["sigma_caps"],
                             "gate_pred": primary["gate_pred"], "gate_percentile": args.gate_percentile,
                             "model_path": primary["model_path"]})
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights_json, f, indent=2)
    # Reference predictions used to derive the uncertainty weights
    if multi:
        for a in antigens:
            a["test_scored"].to_csv(os.path.join(out_dir, f"calibration_{a['name']}.csv"), index=False)
    else:
        primary["test_scored"].to_csv(os.path.join(out_dir, "calibration.csv"), index=False)

    # --- alphabet + seed pools (from the primary antigen's library) -----------
    position_alphabet = build_position_alphabet(primary["data"], wt, aa_mode=args.alphabet_mode)
    n_mutable = sum(1 for a in position_alphabet if len(a) > 1)
    if args.min_mut > n_mutable:
        raise ValueError("min_mut exceeds the number of library-variable positions available for design.")
    logging.info(f"Alphabet mode '{args.alphabet_mode}': {n_mutable} mutable positions "
                 f"(mean {np.mean([len(a) for a in position_alphabet if len(a) > 1]):.1f} residues each)")

    # --- full run record: every parameter actually used, for later inspection --
    n_arms = sum(1 for _ in arm_iter(args))
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "objective": "multi" if multi else "single",
        "combine": args.combine,
        "out_dir": out_dir,
        "wt": wt,
        "n_mutable_positions": int(n_mutable),
        "n_arms": n_arms,
        "args": {k: v for k, v in vars(args).items()},
        "gate_percentile": args.gate_percentile,
        "antigens": [{"name": a["name"], "dataset": a["dataset"], "model_path": a["model_path"],
                      "wt_pred_threshold": a["threshold"], "gate_pred": a["gate_pred"],
                      "gate_percentile": args.gate_percentile,
                      "calibration": a["cal"], "sigma_caps": a["sigma_caps"]} for a in antigens],
    }
    with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False, default_flow_style=False)
    logging.info(f"Wrote run_config.yaml ({n_arms} arms planned)")
    seed_pools = {}
    for seeding in args.seedings:
        if seeding == "top_binders" and multi:
            # cross-reactive seeds: sequences assayed for ALL antigens, ranked by the
            # mean of their z-scored measured labels (not the primary antigen's label
            # alone, which would always favour that antigen)
            seed_pools[seeding] = combined_top_binders(antigens, wt, args.seed_pool_size)
            logging.info(f"Seed pool 'top_binders' (multi: {args.seed_pool_size} by mean "
                         f"z-scored label over {[a['name'] for a in antigens]} intersection "
                         f"+ each antigen's single top binder): "
                         f"{len(seed_pools[seeding])} sequences")
        else:
            seed_pools[seeding] = build_seed_pool(
                primary["data"], wt, position_alphabet, seeding, n=args.seed_pool_size,
                rng=np.random.default_rng(12345), trust_radius=args.trust_radius,
                min_mut=args.min_mut)
            logging.info(f"Seed pool '{seeding}': {len(seed_pools[seeding])} sequences")

    # persist the initial seed pools (the sequences each seeding starts from), so
    # the GA's starting point is inspectable / reproducible after the fact
    seed_rows = [{"seeding": seeding, "index": i, "sequence": s,
                  "num_mut": sum(c != w for c, w in zip(s, wt))}
                 for seeding, pool in seed_pools.items() for i, s in enumerate(pool)]
    pd.DataFrame(seed_rows,
                 columns=["seeding", "index", "sequence", "num_mut"]).to_csv(
        os.path.join(out_dir, "seed_pools.csv"), index=False)
    logging.info(f"Wrote seed_pools.csv ({len(seed_rows)} seeds across "
                 f"{len(seed_pools)} seeding(s))")

    # --- run every arm (each antigen's cache shared across arms) ---------------
    summary = []
    for arm_id, acq, st, wm, seeding, ga_seed in arm_iter(args):
        specs = [dict(name=a["name"], cache=a["cache"], acquisition=acq,
                      sigma_type=st, unc_weight=_arm_weight(wm, st, a),
                      threshold=a["threshold"],
                      normalize=args.acq_normalize, norm_ref=a["norm_ref"],
                      ei_threshold=a["gate_pred"]) for a in antigens]
        score_fn, _ = make_combined_score_fn(specs, combine=args.combine)
        logging.info(f"=== ARM {arm_id} | acq={acq} sigma={st} weight={wm} "
                     f"seeding={seeding} ga_seed={ga_seed} ===")
        engine = GAEngine(
            wt=wt, position_alphabet=position_alphabet, seed_pool=seed_pools[seeding],
            score_fn=score_fn, rng=np.random.default_rng(ga_seed),
            pop_size=args.pop, max_generations=args.generations, trust_radius=args.trust_radius,
            min_mut=args.min_mut,
            crossover_rate=args.crossover_rate, mutation_rate=args.mutation_rate,
            tournament_k=args.tournament_k, n_select=args.n_select,
            elitism_frac=args.elitism_frac,
            stall_patience=args.early_stop_patience, qualify_cols=qualify_cols,
            forbidden_seqs=train_seqs, max_evals=args.max_evals)
        result = engine.run()
        m = result["metrics"]

        arm_dir = os.path.join(out_dir, arm_id)
        os.makedirs(arm_dir, exist_ok=True)
        result["shortlist"].to_csv(os.path.join(arm_dir, "shortlist.csv"), index=False)
        result["genealogy"].to_csv(os.path.join(arm_dir, "genealogy.csv"), index=False)
        result["trajectories"].to_csv(os.path.join(arm_dir, "trajectories.csv"), index=False)
        result["operator_stats"].to_csv(os.path.join(arm_dir, "operator_stats.csv"), index=False)
        with open(os.path.join(arm_dir, "insilico_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)
        with open(os.path.join(arm_dir, "resolved_config.yaml"), "w") as f:
            yaml.safe_dump({"arm_id": arm_id, "acquisition": acq, "sigma_type": st,
                            "weight_mode": wm, "seeding": seeding, "ga_seed": ga_seed,
                            "objective": "multi" if multi else "single",
                            "antigens": [a["name"] for a in antigens], "combine": args.combine},
                           f)

        summary.append({"arm_id": arm_id, "acquisition": acq, "sigma_type": st,
                        "weight_mode": wm, "seeding": seeding, "ga_seed": ga_seed,
                        "n_shortlist": m["n_shortlist"], "n_qualifiers": m["n_qualifiers"],
                        "stop_reason": m["stop_reason"], "n_generations_run": m["n_generations_run"],
                        "n_scored": m["n_scored"],
                        "shortlist_mean_num_mut": m["shortlist_mean_num_mut"],
                        "explored_num_mut_mean": m["explored_num_mut_mean"],
                        "n_unique_seqs": m["n_unique_seqs"],
                        "realised_crossover_rate_mean": m["realised_crossover_rate_mean"],
                        "realised_mut_rate_mean": m["realised_mut_rate_mean"],
                        "repair_frac_mean": m["repair_frac_mean"],
                        "recomb_effectiveness_mean": m["recomb_effectiveness_mean"],
                        "cum_cache_evals": sum(a["cache"].n_evals for a in antigens)})
        logging.info(f"  qualifiers={m['n_qualifiers']} shortlist={m['n_shortlist']} "
                     f"gens={m['n_generations_run']} stop={m['stop_reason']}")

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "arm_summary.csv"), index=False)
    # per-antigen prediction caches (retrospective universe). Single → unsuffixed name.
    if multi:
        for a in antigens:
            a["cache"].save(os.path.join(out_dir, f"prediction_cache_{a['name']}.csv"))
    else:
        primary["cache"].save(os.path.join(out_dir, "prediction_cache.csv"))
    total = sum(a["cache"].n_evals for a in antigens)
    logging.info(f"GA finished. {len(summary)} arms | distinct evals (summed over antigens) = {total}")
    logging.info(f"Results in {out_dir}")
    return pd.DataFrame(summary)


def main(argv=None):
    run_ga(parse_args(argv))
    return 0


if __name__ == "__main__":
    main()
