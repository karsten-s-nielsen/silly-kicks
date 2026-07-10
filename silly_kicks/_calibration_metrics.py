"""Pure-numpy calibration metrics (ECE + reliability slope). No optuna, no ``[calibration]`` extra
-- importable by the xtgk retention model + the v2 validation suite + the GK-completion trainer.

Bodies single-sourced here (formerly duplicated in scripts/train_gk_completion.py). See ADR-036
§Part 3 (the rho calibration gate) and ADR-024 (the completion trainer's gate).
"""

from __future__ import annotations

import numpy as np


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error (binned |mean_pred - mean_obs|, weighted by bin mass)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * (m.mean())
    return float(e)


def reliability_slope(y, p, n_bins: int = 10) -> float:
    """Reliability-diagram slope: weighted linear fit of binned mean-observed on binned
    mean-predicted. ~1 = calibrated; <1 over-confident; >1 under-confident. NaN if predictions
    don't span >1 occupied bin (slope undefined). Complements ECE (magnitude) with a shape check."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mp, mo, w = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean())
            mo.append(y[m].mean())
            w.append(m.sum())
    if len(mp) < 2 or np.ptp(mp) < 1e-9:
        return float("nan")
    coef = np.polyfit(np.asarray(mp), np.asarray(mo), 1, w=np.sqrt(np.asarray(w, float)))
    return float(coef[0])
