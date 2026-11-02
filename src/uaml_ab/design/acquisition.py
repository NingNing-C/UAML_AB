"""Acquisition scores shared by frozen genetic search and retrospective BO."""
import numpy as np
from scipy.stats import norm

ACQUISITION_SCORE_COL = {"greedy": "greedy_score", "lcb": "lcb_score", "ucb": "ucb_score", "ei": "ei_score"}

def _compute_ei(pred, sigma, threshold, unc_weight=1.0):
    """
    Expected Improvement over threshold assuming Gaussian predictive distribution.

    `unc_weight` scales sigma before computing EI, giving the same
    explore/exploit knob as the lcb score (pred - unc_weight*sigma): larger
    unc_weight -> more weight on the uncertainty term (more exploration),
    smaller -> EI collapses toward ranking by pred (more exploitation).
    """
    sigma_scaled = sigma * unc_weight
    z = (pred - threshold) / (sigma_scaled + 1e-8)
    return (pred - threshold) * norm.cdf(z) + sigma_scaled * norm.pdf(z)

def _normalize_series(s, method=None, ref=None):
    """
    Put a score component on a scale-free footing for the *linear* acquisition
    scores (lcb, ucb). Returns `s` unchanged for method in (None,
    'none'), i.e. raw-scale scoring.

    'zscore' -> (s - mean) / std (equal-variance footing; makes unc_weight=1 a
                parameter-free explore/exploit balance).

    `ref` is an optional (mean, std) tuple to standardize against a *fixed
    reference set* instead of `s`'s own stats. In-pool (ref=None) is fine for
    one-shot / BO (a single scoring pass over a stationary set), but GA scores
    an evolving population — per-call stats would make fitness non-stationary —
    so it passes ref stats computed once from the held-out/test split.
    """
    if method in (None, 'none'):
        return s
    if method == 'zscore':
        mu, sd = ref if ref is not None else (s.mean(), s.std())
        return (s - mu) / sd if sd > 0 else s - mu
    raise ValueError(f"Unknown normalize method: {method!r} (expected None/'none'/'zscore')")

def propose_candidates(df, label_threshold, split_col="split_label",
                       pred_col="pred", sigma_col="totalstd", cal_sigma_col=None,
                       unc_weight=1.0, ei_sigma_clip_percentile=99, ei_min_improvement=1.0, det_method=False,
                       normalize=None, ei_threshold=None, norm_ref=None):

    # A negative weight would silently swap LCB/UCB semantics — floor at 0.
    if unc_weight is not None and unc_weight < 0:
        unc_weight = 0.0

    # EI's improvement reference. Defaults to label_threshold (WT), the one-shot
    # "beat WT" framing; the BO loop passes the incumbent (best-so-far pred) so EI
    # references the frontier instead of WT and does not saturate into pure
    # exploitation. Only EI uses this.
    if ei_threshold is None:
        ei_threshold = label_threshold

    test_data = df[df[split_col] == "test"].copy()

    # Greedy: just better than WT prediction
    test_data['greedy_candidate'] = test_data[pred_col] > label_threshold

    # --- gate-margin normalization (normalize='gate_margin') --------------------
    # Express the prediction as a standard-deviation MARGIN above the per-objective
    # gate/frontier, and put sigma in the SAME pred-SD units:
    #     margin       = (pred - gate) / pred_sd
    #     scaled_sigma = sigma / pred_sd
    # `gate` = ei_threshold (the 99th-pct gate in the GA, WT in one-shot, the
    # incumbent in BO); `pred_sd` = the prediction SD from a FIXED calibration set
    # (norm_ref), never the current pool. Because every objective is expressed in
    # its own "SDs above its own gate", the per-objective scores are comparable, so
    # a cross-reactive min over objectives is balanced (no objective dominates just
    # because its assay has larger raw numbers). greedy/lcb/ei are all built from
    # (margin, scaled_sigma) below. `greedy_score` is greedy's ranking column
    # (= margin here; = raw pred otherwise, so nothing changes off this path).
    gate_margin = normalize == "gate_margin"
    if gate_margin:
        pred_sd = norm_ref.get(pred_col, (None, None))[1] if norm_ref else None
        if not pred_sd or pred_sd <= 0:                  # fall back to in-pool SD (one-shot/BO)
            pred_sd = test_data[pred_col].std() or 1.0
        gate_center = ei_threshold                       # gate (GA) / WT (one-shot) / incumbent (BO)
        margin = (test_data[pred_col] - gate_center) / pred_sd
        test_data['greedy_score'] = margin
    else:
        test_data['greedy_score'] = test_data[pred_col]  # ranking == ranking by pred

    if det_method:
        return test_data

    # lcb, ucb and EI clip sigma at ei_sigma_clip_percentile (default
    # 99th) before use, trimming the most extreme uncertainty outliers. In
    # cross-dataset OOD extrapolation it is the *aleatoric* head that spikes
    # (sigma ~25-40x above the label scale); the epistemic component stays well
    # behaved. Clipping only removes that top tail, though: after a p99 clip
    # totalstd still spans ~25x its median, enough for the sigma term to swamp
    # pred, so on totalstd lcb still degenerates toward "pick
    # low-sigma / low-pred" and ucb toward "pick high-sigma". The real scale fix
    # is `normalize` below (put pred and sigma on a common footing) and/or
    # scoring on the well-behaved epistemic sigma rather than totalstd.
    sigma_cap     = np.percentile(test_data[sigma_col], ei_sigma_clip_percentile)
    sigma_clipped = test_data[sigma_col].clip(upper=sigma_cap)

    # Scale-free normalization of pred / sigma for the *linear* scores
    # (lcb, ucb). normalize=None -> identities (raw-scale scoring).
    # 'zscore' puts pred and sigma on a common
    # equal-variance footing so unc_weight=1 is a parameter-free explore/exploit
    # balance and neither term dominates by raw scale alone. The *_candidate
    # booleans stay on the raw scale so their ">threshold" semantics remain
    # meaningful; ranking uses the (possibly normalized) *_score. Greedy is
    # unaffected (monotone in pred); EI is left on raw values below — it
    # self-normalizes via z=(pred-T)/sigma and needs real label units.
    #
    # When normalization is active, normalize the *unclipped* sigma: the p99
    # clip is a raw-scale outlier guard for EI/raw-scale scores, but z-score
    # already handles scale, and clipping would flatten the genuine high tail —
    # for epistemic sigma that tail is exactly the region exploration should
    # target (clipping it collapses ucb back toward greedy). normalize=None
    # keeps the clipped sigma for raw-scale scoring.
    # norm_ref (optional) is {column_name: (mean, std)} computed on a fixed
    # reference set, used instead of in-pool stats (see _normalize_series).
    sigma_for_linear = sigma_clipped if normalize in (None, 'none') else test_data[sigma_col]
    if gate_margin:
        # margin (from above) and sigma scaled by the SAME pred-SD; sigma is NOT
        # mean-centered, so zero uncertainty stays zero.
        pred_norm  = margin
        sigma_norm = test_data[sigma_col] / pred_sd
    else:
        pred_ref  = norm_ref.get(pred_col) if norm_ref else None
        sigma_ref = norm_ref.get(sigma_col) if norm_ref else None
        pred_norm  = _normalize_series(test_data[pred_col], normalize, ref=pred_ref)
        sigma_norm = _normalize_series(sigma_for_linear, normalize, ref=sigma_ref)

    # lcb (risk-averse): lower confidence bound must beat threshold
    test_data['lcb_score']     = pred_norm - sigma_norm * unc_weight
    test_data['lcb_candidate'] = (test_data[pred_col] - sigma_clipped * unc_weight) > label_threshold

    # ucb (optimistic): upper confidence bound beats threshold. Same explore
    # knob as lcb but with the opposite sign on sigma, so it surfaces
    # high-prediction / high-uncertainty sequences instead of suppressing them.
    test_data['ucb_score']     = pred_norm + sigma_norm * unc_weight
    test_data['ucb_candidate'] = (test_data[pred_col] + sigma_clipped * unc_weight) > label_threshold

    # EI is scored on raw, real-unit pred/sigma (never normalized — it needs
    # label units for its Gaussian) and references `ei_threshold` (the incumbent
    # when provided, else WT). It uses the same clipped/unclipped sigma as the
    # linear scores (sigma_for_linear): clipped under normalize=None, but
    # unclipped when normalization is active — clipping the well-behaved
    # epistemic tail would flatten EI's uncertainty bonus and collapse it to
    # greedy, the same failure it causes for ucb.
    if gate_margin:
        # EI on gate-margin units: improvement over the gate (threshold 0) with sigma
        # already in pred-SD units. `unc_weight` still applies on top of that scaling —
        # normalization and the explore/exploit knob are separate things: gate_margin
        # only makes sigma commensurate with pred, while unc_weight says how far to
        # TRUST that sigma (the adaptive weight = max(0, Spearman(sigma, |err|))).
        # Dropping it here would silently pin EI to weight 1 for every caller
        # (one-shot, BO, MO-BO, GA all route through this function), unlike lcb/ucb
        # which do apply it under the same normalization. unc_weight→0 collapses
        # scaled_sigma to 0, so EI → max(margin, 0), i.e. greedy's ranking.
        test_data['ei_score'] = _compute_ei(margin, test_data[sigma_col] / pred_sd, 0.0,
                                            unc_weight=unc_weight)
    else:
        test_data['ei_score'] = _compute_ei(test_data[pred_col], sigma_for_linear, ei_threshold, unc_weight)
    test_data['ei_candidate'] = test_data['ei_score'] > ei_min_improvement

    # lcb / ucb / EI with calibrated uncertainty
    if cal_sigma_col is not None:
        cal_sigma_cap     = np.percentile(test_data[cal_sigma_col], ei_sigma_clip_percentile)
        cal_sigma_clipped = test_data[cal_sigma_col].clip(upper=cal_sigma_cap)

        test_data['cal_lcb_score']     = test_data[pred_col] - cal_sigma_clipped * unc_weight
        test_data['cal_lcb_candidate'] = test_data['cal_lcb_score'] > label_threshold
        test_data['cal_ucb_score']     = test_data[pred_col] + cal_sigma_clipped * unc_weight
        test_data['cal_ucb_candidate'] = test_data['cal_ucb_score'] > label_threshold

        test_data['cal_ei_score']     = _compute_ei(test_data[pred_col], cal_sigma_clipped, label_threshold, unc_weight)
        test_data['cal_ei_candidate'] = test_data['cal_ei_score'] > ei_min_improvement

    return test_data

def select_batch(df, pred_threshold, acquisition, unc_weight, batch_size, random_state=None,
                 sigma_col='totalstd', normalize='gate_margin', ei_threshold=None, norm_ref=None):
    """
    Score `df` (must contain 'pred', the chosen `sigma_col` and
    'split_label'=='test' rows) with the given acquisition function and return
    the top `batch_size` rows.

    `acquisition='random'` is a baseline that ignores 'pred'/sigma entirely and
    samples `batch_size` candidates uniformly at random, to test whether the
    surrogate-guided acquisitions outperform chance.

    `sigma_col` selects the uncertainty driving acquisition. Default 'totalstd'
    (epistemic+aleatoric = the posterior-predictive variance of the *observed*
    label, which is what the acquisitions bound/optimize); 'epistd' (epistemic
    only) is selectable for information-seeking exploration of the latent
    function. `normalize` (below) tames the cross-dataset aleatoric blow-up on
    raw totalstd, so totalstd is safe to default.

    `normalize` puts pred and sigma on a common, scale-free footing. Default
    'gate_margin': every objective is scored as pred's standard-deviation margin
    above its own reference — margin=(pred-ei_threshold)/pred_sd, sigma/pred_sd —
    so greedy/EI/LCB (and a cross-reactive min over objectives) are comparable
    across objectives regardless of raw scale. 'zscore' uses mean-centered z-scores
    of pred and sigma (greedy unaffected, EI on raw units); 'none' is raw-scale.

    `unc_weight` controls the explore/exploit balance for 'lcb'
    (z(pred) - unc_weight*z(sigma)), 'ucb' (z(pred) + unc_weight*z(sigma)) and
    'ei' (sigma scaled by unc_weight before computing Expected Improvement); it
    has no effect on 'greedy' or 'random'.

    `ei_threshold`, if given, is the improvement reference for EI (the
    incumbent / best-so-far). Falls back to `pred_threshold` (WT) when None.
    Only affects EI; greedy/ucb/lcb rank independently of the threshold.

    `norm_ref`, if given ({column: (mean, std)}), standardizes pred/sigma against
    a fixed reference set (e.g. the held-out split) instead of the in-pool stats
    — keeps the z-score anchor stable as the candidate pool depletes over rounds.
    """
    if acquisition == 'random':
        return df.sample(n=min(batch_size, len(df)), random_state=random_state)

    scored = propose_candidates(
        df,
        pred_threshold,
        unc_weight=unc_weight,
        pred_col='pred',
        sigma_col=sigma_col,
        cal_sigma_col=None,
        det_method=False,
        normalize=normalize,
        ei_threshold=ei_threshold,
        norm_ref=norm_ref,
    )

    score_col = ACQUISITION_SCORE_COL[acquisition]
    return scored.nlargest(min(batch_size, len(scored)), score_col)
