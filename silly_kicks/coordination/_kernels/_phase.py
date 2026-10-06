"""Analytic-phase extraction for coordination (Hilbert transform with reflect padding).

Pure numpy/scipy. The signal is centred before the transform (the Hilbert phase is defined on the
oscillation, not its DC offset), reflect-padded to tame the transform's edge transient, then the pad is
stripped so the returned phase aligns sample-for-sample with the input.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.fft


def _hilbert_1d(x: np.ndarray) -> np.ndarray:
    """``scipy.signal.hilbert(x)`` for a real 1-D ``x``, step for step: ``scipy.fft.fft``, the exact x2 of the positive
    half and zero of the negative half, ``scipy.fft.ifft`` -- the same values without the generic array-namespace /
    ``moveaxis`` per-call overhead (ADR-111 ruling C). The fence test holds it to the real ``hilbert`` bit for bit."""
    n = x.size
    xf = cast("np.ndarray", scipy.fft.fft(x))  # scipy stubs type the fft functions as Dispatchable
    if n % 2 == 0:
        xf[1 : n // 2] *= 2.0
        xf[n // 2 + 1 :] = 0.0
    else:
        xf[1 : (n + 1) // 2] *= 2.0
        xf[(n + 1) // 2 :] = 0.0
    return cast("np.ndarray", scipy.fft.ifft(xf))


def pad_length(n: int, fs: float, band_low_cpm: float) -> int:
    """Reflect-pad length: one period of the slowest band edge, capped at the signal length.

    ``band_low_cpm`` is the low band edge in cycles-per-minute, so one period is ``60 * fs / band_low_cpm``
    samples; a run shorter than that is padded by its whole length.
    """
    return min(int(n), round(60.0 * fs / band_low_cpm))


def analytic_phase(x: npt.ArrayLike, pad: int) -> np.ndarray:
    """Instantaneous phase (radians, ``(-pi, pi]``) of ``x`` via the Hilbert transform.

    ``x`` is centred, reflect-padded by ``min(pad, len(x) - 1)`` (numpy ``reflect`` needs >= 2 samples),
    transformed, and the pad stripped so the result is the same length as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    p = min(int(pad), len(xc) - 1)
    # numpy ``reflect`` padding by slicing (p <= len - 1, so one reflection per side): the same values as np.pad
    xp = np.concatenate((xc[p:0:-1], xc, xc[-2 : -p - 2 : -1])) if p > 0 else xc
    analytic = _hilbert_1d(xp)
    if p > 0:
        analytic = analytic[p:-p]
    return np.angle(analytic)


def phasor(theta: npt.ArrayLike) -> np.ndarray:
    """Unit complex phasor ``exp(i * theta)`` for a phase (radians)."""
    return np.exp(1j * np.asarray(theta, dtype=np.float64))


def phase_advance_indicator(theta: npt.ArrayLike) -> np.ndarray:
    """Per-sample monotone-advance indicator of a phase: ``1.0`` where the unwrapped phase advances from the previous
    sample, ``0.0`` where it does not, ``NaN`` for the FIRST sample (no predecessor, so its advance is undefined).

    The leading ``NaN`` is the fix for review A-45: a run's first sample is not a positive-or-not observation, so
    counting it as non-advancing biases a multi-run window's valid fraction down by ``n_runs / n``. Callers slice this
    per run and take :func:`numpy.nanmean`, so run-starts drop out of BOTH numerator and denominator. ``len < 2`` ->
    an all-NaN array of the input's shape.
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = np.full(theta.shape, np.nan)
    if len(theta) >= 2:
        out[1:] = (np.diff(np.unwrap(theta)) > 0).astype(np.float64)
    return out


def phase_valid_fraction(theta: npt.ArrayLike) -> float:
    """Fraction of samples whose unwrapped phase advances (monotone-increasing check); NaN if ``len < 2``.

    The nan-mean of :func:`phase_advance_indicator` -- ONE implementation of the advance test. The first sample has no
    predecessor and is excluded, so this is ``(# advancing) / (n - 1)`` (unchanged from the bit-for-bit prior value).
    """
    ind = phase_advance_indicator(theta)
    if not np.isfinite(ind).any():
        return float("nan")
    return float(np.nanmean(ind))
