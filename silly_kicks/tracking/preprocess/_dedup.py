"""Collapse consecutive-duplicate trajectory samples.

The ADR-120 gap-handling sibling: that reindex/segment machinery fixes frame *gaps*
corrupting velocity; this fixes frame *duplicates*. A duplicate ball frame yields a
per-frame velocity of 0, so a run of duplicates makes a 0<->real velocity sawtooth
that masks a real break. Collapsing consecutive duplicates before any consecutive-frame
differencing removes the sawtooth. Pure (numpy in, numpy out; no I/O, no state).
"""

from __future__ import annotations

import numpy as np


def trajectory_dedup_mask(x: np.ndarray, y: np.ndarray, *, tol_m: float = 1e-6) -> np.ndarray:
    """Keep-mask collapsing runs of consecutive-duplicate positions.

    Keep index 0; keep ``i > 0`` iff the point moved more than ``tol_m`` from the
    IMMEDIATELY preceding point (``hypot(x[i]-x[i-1], y[i]-y[i-1]) > tol_m``), so a genuine
    ``A -> B -> A`` oscillation is preserved while a run ``A -> A -> A -> B`` collapses to
    ``A -> B``. ``tol_m`` defaults to ``1e-6`` m: coordinates are float32 storage (ADR-106),
    so true consecutive duplicates differ by exactly 0; a sub-micrometre tolerance removes
    only non-moving repeats, never real motion. It is NOT a smoothing threshold.

    Returns a boolean ndarray of length ``len(x)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    # Drop ONLY a finite consecutive-duplicate (both endpoints finite AND displacement <= tol).
    # A NaN (non-detection) frame is NOT a duplicate -- keep it, so detection-quality / gap handling
    # downstream is unchanged.
    both_finite = np.isfinite(x[1:]) & np.isfinite(y[1:]) & np.isfinite(x[:-1]) & np.isfinite(y[:-1])
    is_dup = both_finite & (np.hypot(np.diff(x), np.diff(y)) <= tol_m)
    keep[1:] = ~is_dup
    return keep
