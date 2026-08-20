"""Pure metric helpers for the cover-shadow RQ1 + pass-risk validation cycle.

No corpus, no I/O. Consumed by ``validate_cover_shadow_rq1`` / ``validate_pass_risk_calibration``.
The calibration primitives (``ece``, ``reliability_slope``) are wrapped over the library's extra-free
``silly_kicks._calibration_metrics`` so the drivers import them from one place.

**Every score-consuming metric is NaN-safe on the score axis.** Real GS data carries a small number of
non-finite lane / pitch-control values (degenerate geometry, unlinked actions -- measured ~0.8% of
`control`), and the library ``ece``/``reliability_slope`` (``np.polyfit``) blow up on NaN. Each helper
drops non-finite scores first; `p_blocked` is an unbounded blocking INTENSITY (can exceed 1), so the
binning clips into the last bin rather than assuming a [0, 1] probability.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from silly_kicks._calibration_metrics import ece as _lib_ece
from silly_kicks._calibration_metrics import reliability_slope as _lib_reliability_slope

__all__ = [
    "auc",
    "confusion",
    "ece",
    "false_alarm_rate",
    "false_positive_rate",
    "low_control_completion_band",
    "reliability_curve",
    "reliability_slope",
]


def _finite(y, score) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows whose SCORE is non-finite (NaN/inf), keeping ``y`` paired with the survivors."""
    y = np.asarray(y, float)
    score = np.asarray(score, float)
    mask = np.isfinite(score)
    return y[mask], score[mask]


def false_positive_rate(is_blocked, is_completed) -> float:
    """``P(predicted-blocked | completed)`` -- the leakage-free over-prediction rate (Driver A headline)."""
    is_blocked = np.asarray(is_blocked, bool)
    is_completed = np.asarray(is_completed, bool)
    denom = int(is_completed.sum())
    return float(is_blocked[is_completed].mean()) if denom else float("nan")


def false_alarm_rate(control, is_completed, tau: float) -> float:
    """``P(control < tau | completed, control finite)`` -- the leakage-free false-alarm rate (Driver B)."""
    control = np.asarray(control, float)
    is_completed = np.asarray(is_completed, bool)
    mask = is_completed & np.isfinite(control)  # completed passes with a computed control
    denom = int(mask.sum())
    return float((control[mask] < tau).mean()) if denom else float("nan")


def auc(y, score) -> float:
    """NaN-safe ROC-AUC: NaN when the finite subset has a single class."""
    y, score = _finite(y, score)
    if len(score) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over the finite-score subset (the recalibration baseline)."""
    y, p = _finite(y, p)
    return float(_lib_ece(y, p, n_bins)) if len(p) else float("nan")


def reliability_slope(y, p, n_bins: int = 10) -> float:
    """Reliability-diagram slope over the finite-score subset; NaN if < 2 finite points."""
    y, p = _finite(y, p)
    return float(_lib_reliability_slope(y, p, n_bins)) if len(p) >= 2 else float("nan")


def reliability_curve(y, score, n_bins: int = 10) -> dict:
    """Binned mean-predicted vs empirical rate over the finite-score subset (the curve figure)."""
    y, score = _finite(y, score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, edges) - 1, 0, n_bins - 1)  # scores > 1 land in the last bin
    out: dict[str, list] = {"bin_mid": [], "mean_pred": [], "emp_rate": [], "n": []}
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out["bin_mid"].append(float((edges[b] + edges[b + 1]) / 2))
            out["mean_pred"].append(float(score[mask].mean()))
            out["emp_rate"].append(float(y[mask].mean()))
            out["n"].append(int(mask.sum()))
    return out


def confusion(pred, actual_pos) -> dict:
    """Binary confusion + precision/recall/specificity/balanced-accuracy (paper-comparable secondary)."""
    pred = np.asarray(pred, bool)
    actual_pos = np.asarray(actual_pos, bool)
    tp = int((pred & actual_pos).sum())
    fp = int((pred & ~actual_pos).sum())
    tn = int((~pred & ~actual_pos).sum())
    fn = int((~pred & actual_pos).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = float(np.nanmean([rec, spec]))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "balanced_accuracy": ba,
    }


def low_control_completion_band(control, is_success, taus=(0.1, 0.2, 0.3)) -> dict:
    """``P(success | control < tau)`` over ALL finite-control passes -- the CONTAMINATED low-control read."""
    control = np.asarray(control, float)
    is_success = np.asarray(is_success, bool)
    band = {}
    for tau in taus:
        mask = np.isfinite(control) & (control < tau)
        band[tau] = float(is_success[mask].mean()) if mask.any() else float("nan")
    return band
