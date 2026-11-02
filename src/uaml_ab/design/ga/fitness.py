"""Frozen shared-runtime predictions and acquisition scores for genetic search."""
import numpy as np
import pandas as pd
from uaml_ab.runtime import load
from ..acquisition import propose_candidates, ACQUISITION_SCORE_COL

SIGMA_COL = {"epistemic": "epistd", "total": "totalstd"}


def load_surrogate(config, model_config, model_path, device, reference_data):
    fitted = load(model_path, device=device,
                  config={"global": config, "methods": {"ModelArcEnsemble": model_config}},
                  reference_data=reference_data)
    if fitted.method != "ModelArcEnsemble" or fitted.encoding != "onehot":
        raise ValueError("Genetic design requires a onehot ModelArcEnsemble checkpoint.")
    return fitted


def make_predict_fn(fitted, predict_batch_size=4096, sigma_caps=None):
    """Predict unlabeled sequences directly; cache keys remain sequence strings."""
    def predict_fn(seqs):
        frame = pd.DataFrame({"sequence": list(seqs)})
        raw = fitted.predict(frame, df_cnn=frame, batch_size=predict_batch_size)
        out = {}
        for name in ("pred", "epistd", "alestd", "totalstd"):
            if raw.get(name) is None:
                raise ValueError(f"Design checkpoint has no {name} predictions.")
            values = np.asarray(raw[name])
            if values.shape != (len(frame),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid design predictions for {name}.")
            if name != "pred" and (values < 0).any():
                raise ValueError(f"Negative design standard deviations for {name}.")
            cap = sigma_caps.get(name) if sigma_caps else None
            out[name] = np.minimum(values, cap) if cap is not None else values
        return out
    return predict_fn


def make_score_fn(cache, acquisition, sigma_type, unc_weight, threshold,
                  normalize=None, norm_ref=None, ei_threshold=None):
    """
    Build score_fn(seqs) -> DataFrame[sequence, score, pred, epistd, alestd,
    totalstd], in `seqs` order. `score` is the GA fitness (to be maximized) for
    this arm. With the default `normalize='gate_margin'` every acquisition is built
    from the gate-margin `margin = (pred - gate)/pred_sd` and `scaled_sigma =
    sigma/pred_sd` (gate = `ei_threshold`; `pred_sd` from the fixed test-split
    `norm_ref`):

      - greedy:       score = margin                       (sigma / weight ignored)
      - lcb: score = margin - unc_weight * scaled_sigma
      - ei:           score = ExpectedImprovement(margin, unc_weight * scaled_sigma) over 0

    Expressing each antigen in "SDs above its own gate" makes the per-antigen
    scores comparable, so the cross-reactive `min` over antigens is balanced (no
    antigen dominates the min purely because its assay has larger raw numbers).
    (`normalize='zscore'` instead uses mean-centered z-scores — greedy = raw pred,
    EI on raw units; `'none'` = raw-scale.) `sigma` is the epistemic or total
    predictive std per `sigma_type`.

    Because the GA scores an evolving population, `norm_ref` MUST be fixed-reference
    stats (mean/std per column from the held-out/test split, computed once); in-pool
    normalization would make fitness non-stationary across generations. `ei_threshold`
    is the gate/frontier reference (the gate-percentile of predictions in the GA);
    falls back to `threshold` (WT) when None.
    """
    score_col = ACQUISITION_SCORE_COL[acquisition]
    sigma_col = SIGMA_COL.get(sigma_type, "totalstd")

    def score_fn(seqs):
        seqs = list(seqs)
        raw = cache.get(seqs)
        df = pd.DataFrame({
            "sequence": seqs,
            "pred": raw["pred"],
            "epistd": raw["epistd"],
            "alestd": raw["alestd"],
            "totalstd": raw["totalstd"],
            "split_label": "test",            # propose_candidates filters to 'test'
        })
        scored = propose_candidates(
            df, threshold,
            pred_col="pred", sigma_col=sigma_col, cal_sigma_col=None,
            unc_weight=unc_weight,
            det_method=(acquisition == "greedy"),
            normalize=normalize, norm_ref=norm_ref, ei_threshold=ei_threshold,
        )
        # propose_candidates preserves row order and returns all rows (all 'test').
        out = scored[["sequence", "pred", "epistd", "alestd", "totalstd"]].copy()
        out["score"] = scored[score_col].to_numpy()
        return out.reset_index(drop=True)

    return score_fn

def make_combined_score_fn(antigen_specs, combine="min"):
    """Multi-objective score_fn over one OR more antigens.

    `antigen_specs` is a list of dicts, one per antigen, each with:
        name, cache, acquisition, sigma_type, unc_weight, threshold
    (the per-antigen acquisition is the SAME acquisition for the arm; unc_weight /
    threshold are that antigen's own calibration weight / WT-prediction).

    The arm's acquisition is computed per antigen and then **scalarised across
    antigens**:
      - combine="min"  → a sequence is only as good as its weakest antigen
        (cross-reactive: must be good for BOTH). `score`/`pred` are the per-antigen
        min; the reported epistd/alestd/totalstd are taken from the
        binding-limiting antigen (the one achieving the min pred).
      - combine="mean" → average across antigens (a strong antigen can compensate).

    Returns (score_fn, antigen_names). With a single antigen this reduces to the
    plain single-objective scorer and emits NO per-antigen suffixed columns, so the
    output schema is identical to make_score_fn. With ≥2 antigens
    it additionally emits pred_<name>/epistd_<name>/alestd_<name>/totalstd_<name>
    for per-antigen gates and run records."""
    fns = [(s["name"], make_score_fn(s["cache"], s["acquisition"], s["sigma_type"],
                                     s["unc_weight"], s["threshold"],
                                     normalize=s.get("normalize"), norm_ref=s.get("norm_ref"),
                                     ei_threshold=s.get("ei_threshold")))
           for s in antigen_specs]
    names = [n for n, _ in fns]
    multi = len(names) > 1

    def score_fn(seqs):
        seqs = list(seqs)
        per = {n: f(seqs) for n, f in fns}
        if not multi:                                       # single-objective passthrough
            return per[names[0]]

        score_mat = np.column_stack([per[n]["score"].to_numpy() for n in names])
        pred_mat = np.column_stack([per[n]["pred"].to_numpy() for n in names])
        if combine == "mean":
            comb_score = score_mat.mean(axis=1)
            comb_pred = pred_mat.mean(axis=1)
            limiting = np.zeros(len(seqs), dtype=int)       # sigma from antigen 0
        else:                                               # "min" (cross-reactive)
            comb_score = score_mat.min(axis=1)
            comb_pred = pred_mat.min(axis=1)
            limiting = pred_mat.argmin(axis=1)              # binding-limiting antigen
        out = pd.DataFrame({"sequence": seqs, "score": comb_score, "pred": comb_pred})
        # single-number sigma columns report the limiting antigen's, so
        # single-objective output records have one value to read.
        rows = np.arange(len(seqs))
        for f in ("epistd", "alestd", "totalstd"):
            stacked = np.column_stack([per[n][f].to_numpy() for n in names])
            out[f] = stacked[rows, limiting]
        # per-antigen detail (used by the qualification gate + multi-objective plots)
        for n in names:
            for f in ("pred", "epistd", "alestd", "totalstd"):
                out[f"{f}_{n}"] = per[n][f].to_numpy()
        return out

    return score_fn, names
