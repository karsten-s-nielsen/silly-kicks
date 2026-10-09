"""Spectral median frequency and pooled Welch coherence for coordination.

Pure numpy/scipy. Median frequency drives per-provider band derivation; coherence is pooled by pooling the
SPECTRA (weighted by Welch segment count) and only THEN forming the coherence -- averaging per-slice
coherences would bias the estimate (a low-power slice's noisy ratio would count as much as a clean one).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.signal


@dataclass(frozen=True)
class WelchSpectra:
    """One slice's Welch auto/cross spectra plus the number of averaged segments ``k``."""

    f: np.ndarray  # Hz
    pxx: np.ndarray
    pyy: np.ndarray
    pxy: np.ndarray  # complex cross-spectrum
    k: int


def median_frequency_cpm(x: npt.ArrayLike, fs: float) -> float:
    """Spectral median frequency (cycles·min⁻¹) via the periodogram; NaN for a constant series.

    The mean is removed (``detrend="constant"``) and DC excluded, so a constant offset does not move it. One
    interpolation rule (spec 7.8.4): each bin's power is spread uniformly over its ``df``-wide band centred on the bin,
    so the cumulative power is linear between bin EDGES and the median is where it reaches half the total. A pure
    tone on bin k then returns exactly k·df (spec 9.1); placing a bin's cumulative power at the bin centre instead
    reads half a bin low (review A-06).
    """
    f_hz, p = cast(
        "tuple[np.ndarray, np.ndarray]",
        scipy.signal.periodogram(np.asarray(x, dtype=np.float64), fs=fs, detrend="constant", window="boxcar"),
    )
    f_hz, p = f_hz[1:], p[1:]  # exclude DC
    c = np.cumsum(p)
    if c[-1] <= 0.0:
        return float("nan")  # constant / zero-power -> degenerate_constant
    half = 0.5 * c[-1]
    k = int(np.searchsorted(c, half))  # the bin whose band holds the half-power point (p[k] > 0 there)
    below = float(c[k - 1]) if k > 0 else 0.0
    df = float(f_hz[0])  # the bins sit at j·df, j = 1, 2, ... once DC is dropped
    frac = min(max((half - below) / float(p[k]), 0.0), 1.0)
    return float(60.0 * (f_hz[k] - 0.5 * df + frac * df))


def pooled_median_frequency(values: npt.ArrayLike, durations_s: npt.ArrayLike) -> float:
    """Duration-weighted mean of per-run median frequencies (non-finite / zero-duration runs dropped)."""
    v = np.asarray(values, dtype=np.float64)
    d = np.asarray(durations_s, dtype=np.float64)
    m = np.isfinite(v) & (d > 0)
    if not m.any():
        return float("nan")
    return float(np.sum(v[m] * d[m]) / np.sum(d[m]))


def min_spectral_samples(fs: float, band_low_cpm: float) -> int:
    """Minimum samples for a stable spectrum: two periods of the band's lower edge."""
    return int(np.ceil(2.0 * 60.0 * fs / band_low_cpm))


def welch_segment_count(n: int, nperseg: int) -> int:
    """Number of 50%-overlap Welch segments in ``n`` samples (0 if ``n < nperseg``).

    The step is ``nperseg - noverlap`` with ``noverlap = nperseg // 2`` (scipy's convention), i.e.
    ``nperseg - nperseg // 2`` -- the CEIL half, not the floor; they differ for an odd ``nperseg`` (review A-46)."""
    if n < nperseg:
        return 0
    return 1 + (n - nperseg) // (nperseg - nperseg // 2)


def welch_spectra(a: npt.ArrayLike, b: npt.ArrayLike, fs: float, nperseg: int) -> WelchSpectra:
    """Welch auto/cross spectra for a dyad (Hann window, 50% overlap, per-segment mean removed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    f_hz, pxx = cast(
        "tuple[np.ndarray, np.ndarray]",
        scipy.signal.welch(a, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant"),
    )
    _, pyy = cast(
        "tuple[np.ndarray, np.ndarray]",
        scipy.signal.welch(b, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant"),
    )
    _, pxy = cast(
        "tuple[np.ndarray, np.ndarray]",
        scipy.signal.csd(a, b, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant"),
    )
    return WelchSpectra(f=f_hz, pxx=pxx, pyy=pyy, pxy=pxy, k=welch_segment_count(len(a), nperseg))


@dataclass(frozen=True)
class CrossSpectraBatch:
    """:func:`welch_spectra` of one ``a`` against many ``b`` rows: ``pxx`` once, ``pyy``/``pxy`` one row per ``b``."""

    f: np.ndarray  # Hz
    pxx: np.ndarray  # (F,)
    pyy: np.ndarray  # (rows, F)
    pxy: np.ndarray  # (rows, F) complex
    k: int

    def row(self, i: int) -> WelchSpectra:
        """Row ``i`` as the :class:`WelchSpectra` :func:`welch_spectra` returns for ``b[i]``."""
        return WelchSpectra(f=self.f, pxx=self.pxx, pyy=self.pyy[i], pxy=self.pxy[i], k=self.k)


def cross_spectra_batch(a: npt.ArrayLike, b: npt.ArrayLike, fs: float, nperseg: int) -> CrossSpectraBatch:
    """:func:`welch_spectra` of ``a`` against every row of ``b`` (the coherence null, one row per draw): ``a``'s
    auto-spectrum ONCE (the per-draw loop recomputed the same value every draw), then each row's auto- and cross-spectra
    by the same 1-D scipy calls :func:`welch_spectra` makes -- so every row is bit-identical on every scipy version.
    (One call along the last axis is NOT: scipy 1.18's batched FFT rounds rows differently -- measured, ADR-111.)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"b must be 2-D (rows, N); got shape {b.shape}")
    overlap = nperseg // 2
    f_hz, pxx = cast(
        "tuple[np.ndarray, np.ndarray]",
        scipy.signal.welch(a, fs=fs, window="hann", nperseg=nperseg, noverlap=overlap, detrend="constant"),
    )
    pyy_rows: list[np.ndarray] = []
    pxy_rows: list[np.ndarray] = []
    for row in b:
        _, pyy = cast(
            "tuple[np.ndarray, np.ndarray]",
            scipy.signal.welch(row, fs=fs, window="hann", nperseg=nperseg, noverlap=overlap, detrend="constant"),
        )
        _, pxy = cast(
            "tuple[np.ndarray, np.ndarray]",
            scipy.signal.csd(a, row, fs=fs, window="hann", nperseg=nperseg, noverlap=overlap, detrend="constant"),
        )
        pyy_rows.append(pyy)
        pxy_rows.append(pxy)
    n_f = f_hz.shape[0]
    return CrossSpectraBatch(
        f=f_hz,
        pxx=pxx,
        pyy=np.array(pyy_rows).reshape(b.shape[0], n_f),
        pxy=np.array(pxy_rows, dtype=np.complex128).reshape(b.shape[0], n_f),
        k=welch_segment_count(len(a), nperseg),
    )


def pooled_coherence(
    spectra: Sequence[WelchSpectra], band_low_cpm: float, band_high_cpm: float
) -> tuple[float, float, int]:
    """Pool spectra (segment-weighted), THEN form coherence; return ``(band_mean, peak_cpm, K)``.

    Never averages per-slice coherences. An empty band or zero total segments returns NaN summaries.
    """
    k = sum(s.k for s in spectra)
    if k == 0:
        return (float("nan"), float("nan"), 0)
    pxx = sum((s.k * s.pxx for s in spectra), start=np.zeros_like(spectra[0].pxx)) / k
    pyy = sum((s.k * s.pyy for s in spectra), start=np.zeros_like(spectra[0].pyy)) / k
    pxy = sum((s.k * s.pxy for s in spectra), start=np.zeros_like(spectra[0].pxy)) / k
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(pxy) ** 2 / (pxx * pyy)
    f_cpm = 60.0 * spectra[0].f
    band = (f_cpm >= band_low_cpm) & (f_cpm <= band_high_cpm)
    if not band.any():
        return (float("nan"), float("nan"), k)
    coh_band = coh[band]
    return (float(np.nanmean(coh_band)), float(f_cpm[band][np.nanargmax(coh_band)]), k)
