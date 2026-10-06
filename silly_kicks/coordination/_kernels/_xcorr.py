"""Lagged cross-correlation and Fisher-z pooling for coordination (windowed relative timing).

Pure numpy/scipy. Exact per-lag Pearson over each lag's overlap, from an FFT cross-sum plus prefix sums --
so the whole lag profile costs one FFT, not a Python loop over lags. Signals are centred first for FFT
numerical stability (Pearson is shift-invariant, so this changes no value; raw ~50 m positions would lose
precision in the cross-sum).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.signal


def lagged_pearson(a: npt.ArrayLike, b: npt.ArrayLike, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-lag Pearson r and overlap size for lags ``-L..L``.

    Returns ``(r, n)``: ``r[l + L]`` is ``corr(a[t], b[t + l])`` over the valid overlap (float64, NaN where a
    side is constant on the overlap); ``n[l + L]`` is that overlap's size (int64). A leading ``a`` shows a
    POSITIVE lag.
    """
    xa = np.asarray(a, dtype=np.float64)
    xb = np.asarray(b, dtype=np.float64)
    n_samples = xa.shape[0]
    if xb.shape[0] != n_samples:
        raise ValueError(f"a and b must be the same length; got {n_samples} and {xb.shape[0]}")
    lag = int(max_lag)
    if lag < 0:
        raise ValueError(f"max_lag must be >= 0; got {lag}")
    if lag >= n_samples:
        raise ValueError(f"max_lag ({lag}) must be < signal length ({n_samples})")

    xa = xa - xa.mean()  # centre for FFT numerical stability (Pearson is shift-invariant -> no value change)
    xb = xb - xb.mean()
    lags = np.arange(-lag, lag + 1)
    ns = (n_samples - np.abs(lags)).astype(np.int64)

    # prefix sums: p*[k] == sum of the first k samples, so any contiguous overlap sum is one subtraction.
    p_a = np.concatenate(([0.0], np.cumsum(xa)))
    p_a2 = np.concatenate(([0.0], np.cumsum(xa * xa)))
    p_b = np.concatenate(([0.0], np.cumsum(xb)))
    p_b2 = np.concatenate(([0.0], np.cumsum(xb * xb)))

    pos = lags >= 0
    l_pos = np.clip(lags, 0, None)  # l for l >= 0
    l_neg = np.clip(-lags, 0, None)  # m = -l for l < 0
    n_full = n_samples
    # l >= 0: a[0:N-l], b[l:N];  l < 0 (m=-l): a[m:N], b[0:N-m]
    s_a = np.where(pos, p_a[n_full - l_pos], p_a[n_full] - p_a[l_neg])
    s_aa = np.where(pos, p_a2[n_full - l_pos], p_a2[n_full] - p_a2[l_neg])
    s_b = np.where(pos, p_b[n_full] - p_b[l_pos], p_b[n_full - l_neg])
    s_bb = np.where(pos, p_b2[n_full] - p_b2[l_pos], p_b2[n_full - l_neg])

    full = cast("np.ndarray", scipy.signal.correlate(xb, xa, mode="full", method="fft"))
    s_ab = full[lags + n_full - 1]  # == sum_t a[t] * b[t + l] over the overlap

    nf = ns.astype(np.float64)
    num = nf * s_ab - s_a * s_b
    var_a = nf * s_aa - s_a * s_a
    var_b = nf * s_bb - s_b * s_b
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(var_a * var_b)
        r = np.where(denom > 0, num / denom, np.nan)
    return r.astype(np.float64), ns


def lagged_pearson_batch(a: npt.ArrayLike, b: npt.ArrayLike, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """:func:`lagged_pearson` of ``a`` against every row of ``b`` (the direct surrogate null, one row per draw).

    ``b`` is ``(rows, N)``; returns ``r`` ``(rows, 2L+1)`` and the shared overlap sizes ``n`` ``(2L+1,)``. Each row is
    the 1-D :func:`lagged_pearson` call itself, so it is bit-identical per row on every scipy version. (A vectorised
    ``fftconvolve(..., axes=1)`` is NOT: scipy 1.18's batched FFT rounds rows differently from the 1-D transform the
    observed value and the per-draw oracle take -- measured, ADR-111.)
    """
    xb = np.asarray(b, dtype=np.float64)
    if xb.ndim != 2:
        raise ValueError(f"b must be 2-D (rows, N); got shape {xb.shape}")
    xa = np.asarray(a, dtype=np.float64)
    if xb.shape[1] != xa.shape[0]:
        raise ValueError(f"a and b must be the same length; got {xa.shape[0]} and {xb.shape[1]}")
    lag = int(max_lag)
    if lag < 0:
        raise ValueError(f"max_lag must be >= 0; got {lag}")
    r = np.empty((xb.shape[0], 2 * lag + 1), dtype=np.float64)
    ns = (xa.shape[0] - np.abs(np.arange(-lag, lag + 1))).astype(np.int64)
    for i in range(xb.shape[0]):
        r[i], ns = lagged_pearson(xa, xb[i], lag)
    return r, ns


def fisher_pool(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Fisher-z pool per-lag correlations across slices (axis ``-2``), weighting each by ``n - 3``.

    ``r`` is ``(..., S, 2L+1)`` -- any leading axes (e.g. surrogate draws) are pooled independently -- and ``n`` is
    ``(S, 2L+1)``; returns ``(..., 2L+1)``. Slices with ``n <= 3`` or non-finite ``r`` get zero weight; a lag with no
    weight anywhere is NaN. The transcendentals run on C-contiguous arrays, so a batched call takes the same ufunc
    loops as the 2-D one and pools each leading row bit-identically.
    """
    w = np.ascontiguousarray(np.where(np.isfinite(r) & (n > 3), n - 3.0, 0.0))
    z = np.arctanh(np.ascontiguousarray(np.clip(np.where(w > 0, r, 0.0), -1.0 + 1e-15, 1.0 - 1e-15)))
    sw = np.ascontiguousarray(w.sum(axis=-2))
    pooled = np.ascontiguousarray(np.ascontiguousarray((w * z).sum(axis=-2)) / np.where(sw > 0, sw, 1.0))
    return np.where(sw > 0, np.tanh(pooled), np.nan)


def xcorr_summary(r: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    """Summarise a lag profile: ``(max_abs_r, lag_s, r_at_max, r_lag0)``.

    The peak is the lag of greatest ``|r|``; ties break to the smallest ``|lag|``, then to the negative lag.
    ``lag_s`` is that lag in seconds. All-NaN input returns four NaNs.
    """
    r = np.asarray(r, dtype=np.float64)
    lag = (len(r) - 1) // 2
    lags = np.arange(-lag, lag + 1)
    finite = np.isfinite(r)
    if not finite.any():
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    abs_r = np.where(finite, np.abs(r), -1.0)
    peak = float(abs_r.max())
    cand = np.flatnonzero(abs_r == peak)
    idx = min(cand, key=lambda i: (abs(int(lags[i])), int(lags[i])))  # smallest |lag|, then the negative lag
    return (peak, float(lags[idx] / fs), float(r[idx]), float(r[lag]))


def min_slice_samples(max_lag: int) -> int:
    """Minimum slice length for a stable lag profile: four times ``max_lag`` (spec 7.8.2)."""
    return 4 * int(max_lag)
