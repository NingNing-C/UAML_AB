"""Numerical prediction and Gaussian uncertainty metrics; no plotting dependencies."""
import warnings
import numpy as np
from scipy import stats


def spearman(y_preds, y_true):
    a, b = np.asarray(y_preds), np.asarray(y_true)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.ptp(a[mask]) == 0 or np.ptp(b[mask]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[mask], b[mask]).statistic)


def pearson(y_preds, y_true):
    a, b = np.asarray(y_preds), np.asarray(y_true)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.ptp(a[mask]) == 0 or np.ptp(b[mask]) == 0:
        return float("nan")
    return float(stats.pearsonr(a[mask], b[mask]).statistic)


def regression_metrics(y_preds, y_true):
    pred, true = np.asarray(y_preds), np.asarray(y_true)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if not len(true):
        return {name: float("nan") for name in Metric_fns}
    residual = pred - true
    denom = np.sum((true - true.mean()) ** 2)
    return {"pearson": pearson(pred, true), "spearman": spearman(pred, true),
            "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1 - np.sum(residual**2)/denom) if len(true)>1 and denom>0 else float("nan")}


def uq_metrics(y_preds, y_stds, y_true):
    pred, std, true = map(np.asarray, (y_preds, y_stds, y_true))
    mask = np.isfinite(pred) & np.isfinite(std) & np.isfinite(true) & (std > 0)
    pred, std, true = pred[mask], std[mask], true[mask]
    if not len(true):
        return {name: float("nan") for name in UQ_Metrics_fns}
    z = np.abs(pred - true)/std
    levels = np.linspace(0, 1, 100)
    observed = (z[:, None] <= stats.norm.ppf((1 + levels)/2)).mean(axis=0)
    # Numerical area between nominal and empirical central Gaussian coverage.
    miscal = float(np.trapz(np.abs(observed - levels), levels))
    scale = np.std(true)
    return {"unc_spearman": spearman(np.abs(pred-true), std),
            "nll": float(np.mean(0.5*np.log(2*np.pi*std**2) + 0.5*z**2)),
            "coverage_95": float(np.mean(z <= stats.norm.ppf(0.975))),
            "sharpness": float(np.sqrt(np.mean(std**2))/scale) if scale>0 else float("nan"),
            "miscal_area": miscal}


# Names retained for modules that assemble output columns.
Metric_fns = dict.fromkeys(["pearson", "spearman", "mae", "rmse", "r2"])
UQ_Metrics_fns = dict.fromkeys(["unc_spearman", "nll", "coverage_95", "sharpness", "miscal_area"])
