"""Surrogate-data machinery for coordination significance (time shifts + IAAFT).

Pure numpy (plus the numba wrap-correction/near-count kernels). The accelerated statistics compute all K
draws from ONE FFT pair (whole-segment) plus non-FFT corrections, so the FFT call count is constant in K
(the structural gate). Each accelerated statistic feeds the SAME exact functions the observed value uses, so
the surrogate and the observation are directly comparable (estimator identity, spec 7.9).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy import fft as _fft

from silly_kicks.coordination._kernels._numba import HAVE_NUMBA, use_numba

if HAVE_NUMBA:
    from silly_kicks.coordination._kernels._numba import near_in_phase_counts_at_nb, xcorr_edge_correction_nb

#: Positions x draws per chunk in the numpy near-in-phase count (bounds its ``(draws, positions)`` gather arrays).
_NEAR_CHUNK_ELEMENTS = 1 << 20


# --------------------------------------------------------------------------- seeding (order-independent)
def key_words(key: tuple[object, ...]) -> tuple[int, int, int, int]:
    """Deterministic 4x uint32 spawn key from a coordination key.

    The caller passes CANONICAL id values (``str(canonical_id(...))`` / ``int``), so id dtype cannot matter;
    the kernel stays free of any silly-kicks import (spec 7.6 -- ``_kernels/*`` is numpy/scipy/stdlib/numba only).
    """
    payload = json.dumps([str(k) for k in key])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    words = tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
    return (words[0], words[1], words[2], words[3])


def surrogate_rng(seed: int, key: tuple[object, ...]) -> np.random.Generator:
    """A generator seeded by ``seed`` and the key's spawn words -- identical regardless of processing order."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=key_words(key)))


# --------------------------------------------------------------------------- time-shift draws
def shift_bounds(n: int, tau: int) -> tuple[int, int] | None:
    """Valid shift range ``(tau, n - tau)`` when ``n >= 2*tau + 1``; else ``None`` (segment too short, R1)."""
    if n >= 2 * tau + 1:
        return (tau, n - tau)
    return None


def draw_shifts(rng: np.random.Generator, n: int, tau: int, k: int) -> np.ndarray | None:
    """``k`` uniform integer shifts in ``[tau, n - tau)``, or ``None`` if the segment is too short."""
    bounds = shift_bounds(n, tau)
    if bounds is None:
        return None
    lo, hi = bounds
    return rng.integers(lo, hi, size=k)


def shifted_window(
    x: np.ndarray, segments: Sequence[tuple[int, int]], shifts: Sequence[np.ndarray], start: int, end: int, n_draws: int
) -> np.ndarray:
    """Rows ``[start, end)`` of ``x`` under every time-shift draw, shape ``(n_draws, end - start)``.

    Draw ``d`` circularly shifts each segment ``(lo, hi)`` of ``x`` by ``shifts[g][d]`` (``np.roll``
    semantics: ``rolled[i] = x[lo + (i - shift) mod (hi - lo)]``); rows outside every segment keep their value.
    A pure gather, so each row equals ``np.roll``-ing a full copy of ``x`` and slicing ``[start, end)``, bit for
    bit -- without the per-draw full-length copy.
    """
    x = np.asarray(x)
    width = int(end) - int(start)
    if len(segments) != len(shifts):
        raise ValueError(f"segments ({len(segments)}) and shifts ({len(shifts)}) must pair up")
    rows = np.empty((n_draws, max(width, 0)), dtype=np.int64)
    rows[:] = np.arange(start, end, dtype=np.int64)
    for (lo, hi), sh in zip(segments, shifts, strict=True):
        n = int(hi) - int(lo)
        sh = np.asarray(sh, dtype=np.int64)
        if sh.shape != (n_draws,):
            raise ValueError(f"shifts for segment ({lo}, {hi}) have shape {sh.shape}; expected {(n_draws,)}")
        if sh.size and (int(sh.min()) < 0 or int(sh.max()) >= n):
            raise ValueError(f"shifts for segment ({lo}, {hi}) must lie in [0, {n})")
        a, b = max(int(lo), int(start)), min(int(hi), int(end))
        if b <= a:
            continue
        offset = np.arange(a - lo, b - lo, dtype=np.int64)[None, :] - sh[:, None]
        offset += n * (offset < 0)  # == offset mod n, since offset lies in (-n, n)
        rows[:, a - start : b - start] = offset + lo
    return x[rows]


# --------------------------------------------------------------------------- IAAFT
def iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int) -> tuple[np.ndarray, bool]:
    """IAAFT surrogate (Schreiber & Schmitz 2000): preserve the amplitude distribution AND the spectrum."""
    x = np.asarray(x, dtype=np.float64)
    sorted_x = np.sort(x)
    target_amp = np.abs(_fft.rfft(x))
    s = rng.permutation(x)
    ranks = np.argsort(np.argsort(s))
    for _ in range(max_iter):
        spec = _fft.rfft(s)
        s = _fft.irfft(target_amp * np.exp(1j * np.angle(spec)), n=x.size)
        new_ranks = np.argsort(np.argsort(s))
        s = sorted_x[new_ranks]
        if np.array_equal(new_ranks, ranks):
            return s, True
        ranks = new_ranks
    return s, False  # non-convergence is reported, never hidden


# --------------------------------------------------------------------------- percentile summaries
def percentile_rank(obs: float, surr: np.ndarray) -> float:
    """``(#{s < obs} + 0.5 * #{s == obs}) / K`` over finite surrogates; NaN if ``obs`` is NaN or ``K == 0``."""
    if np.isnan(obs):
        return float("nan")
    s = np.asarray(surr, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float((np.sum(s < obs) + 0.5 * np.sum(s == obs)) / s.size)


def surrogate_triple(obs: float, surr: np.ndarray) -> tuple[float, float, float]:
    """``(surrogate_mean, percentile, excess = obs - mean)``; NaNs where undefined."""
    s = np.asarray(surr, dtype=np.float64)
    s = s[np.isfinite(s)]
    mean = float(np.mean(s)) if s.size else float("nan")
    excess = float(obs - mean) if (s.size and not np.isnan(obs)) else float("nan")
    return (mean, percentile_rank(obs, surr), excess)


# --------------------------------------------------------------------------- accelerated statistics
def phasor_spectrum(zb: np.ndarray) -> np.ndarray:
    """``conj(fft(zb))``: the B side of :func:`shifted_phasor_sums`' whole-segment identity. A caller scoring many
    windows against one B segment computes it once and passes it as ``zb_spectrum=`` (the same values)."""
    return np.conj(_fft.fft(np.asarray(zb, dtype=np.complex128)))


def shifted_phasor_sums(
    za: np.ndarray,
    zb: np.ndarray,
    shifts: np.ndarray,
    start: int | None = None,
    end: int | None = None,
    *,
    zb_spectrum: np.ndarray | None = None,
) -> np.ndarray:
    """``S[s] = sum_{t in [start, end)} za[t] * conj(zb[(t - s) mod N])`` for every shift.

    Whole-segment windows use one FFT pair (constant in K); sub-segment windows gather in chunks of 32.
    ``zb_spectrum`` is :func:`phasor_spectrum` of ``zb`` when the caller already holds it.
    """
    za = np.asarray(za, dtype=np.complex128)
    zb = np.asarray(zb, dtype=np.complex128)
    n = za.size
    lo = 0 if start is None else int(start)
    hi = n if end is None else int(end)
    if lo == 0 and hi == n:  # C11: one FFT pair gives all N circular shifts
        fb = phasor_spectrum(zb) if zb_spectrum is None else zb_spectrum
        if fb.shape != (n,):
            raise ValueError(f"zb_spectrum has shape {fb.shape}; the segment has {n} samples")
        cross = _fft.ifft(_fft.fft(za) * fb)
        return cross[np.mod(shifts, n)]
    idx_t = np.arange(lo, hi)
    za_w = za[lo:hi]
    out = np.empty(shifts.size, dtype=np.complex128)
    for c0 in range(0, shifts.size, 32):
        chunk = shifts[c0 : c0 + 32]
        gathered = zb[(idx_t[None, :] - chunk[:, None]) % n]
        out[c0 : c0 + chunk.size] = (za_w[None, :] * np.conj(gathered)).sum(axis=1)
    return out


def _near_in_phase_counts_at_np(
    a_re: np.ndarray,
    a_im: np.ndarray,
    t_loc: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
    shifts: np.ndarray,
    cos_thr: float,
) -> np.ndarray:
    """The numpy reference of :func:`near_in_phase_counts_at`: per real part, the numba kernel's two rounded products
    and one rounded sum (separate ufuncs never fuse into an FMA), so the integer counts agree exactly."""
    n = b_re.size
    out = np.empty(shifts.size, dtype=np.int64)
    per = max(1, _NEAR_CHUNK_ELEMENTS // max(1, t_loc.size))
    for d0 in range(0, shifts.size, per):
        sh = shifts[d0 : d0 + per]
        j = t_loc[None, :] - sh[:, None]
        j += n * (j < 0)  # t_loc and the shifts lie in [0, n): one wrap at most
        re = a_re[None, :] * b_re[j] + a_im[None, :] * b_im[j]
        out[d0 : d0 + sh.size] = np.count_nonzero(re >= cos_thr, axis=1)
    return out


def near_in_phase_counts_at(
    a: np.ndarray, t_loc: np.ndarray, b: np.ndarray, shifts: np.ndarray, cos_thr: float
) -> np.ndarray:
    """Per shift ``s``, how many positions ``t_loc[i]`` have ``Re(a[i] * conj(b[(t_loc[i] - s) mod n])) >= cos_thr``.

    The % near-in-phase surrogate count (spec 7.9: direct, O(N K), numba optional). ``a`` holds A's phasors at the
    positions ``t_loc`` of the B segment ``b`` (length ``n``); draw ``s`` rolls the segment by ``s`` (``np.roll``
    semantics). The real part is ``a.re * b.re + a.im * b.im`` in both backends -- two rounded products, one rounded
    sum -- so the counts never depend on whether numba is installed (spec 7.15). A non-finite product never counts.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    t_loc = np.ascontiguousarray(t_loc, dtype=np.int64)
    shifts = np.ascontiguousarray(shifts, dtype=np.int64)
    n = b.size
    if a.shape != t_loc.shape:
        raise ValueError(f"a has shape {a.shape} but t_loc has shape {t_loc.shape}")
    if t_loc.size and (int(t_loc.min()) < 0 or int(t_loc.max()) >= n):
        raise ValueError(f"t_loc positions must lie in [0, {n})")
    if shifts.size and (int(shifts.min()) < 0 or int(shifts.max()) >= n):
        raise ValueError(f"shifts must lie in [0, {n})")
    a_re, a_im = np.ascontiguousarray(np.real(a)), np.ascontiguousarray(np.imag(a))
    b_re, b_im = np.ascontiguousarray(np.real(b)), np.ascontiguousarray(np.imag(b))
    if use_numba():
        return near_in_phase_counts_at_nb(a_re, a_im, t_loc, b_re, b_im, shifts, float(cos_thr))
    return _near_in_phase_counts_at_np(a_re, a_im, t_loc, b_re, b_im, shifts, float(cos_thr))


def shifted_near_in_phase_counts(
    za: np.ndarray, zb: np.ndarray, shifts: np.ndarray, start: int, end: int, cos_thr: float
) -> np.ndarray:
    """Per shift, count samples in ``[start, end)`` whose relative phasor ``za * conj(roll(zb, s))`` is within
    ``acos(cos_thr)`` of 0: :func:`near_in_phase_counts_at` over the contiguous positions of one circular series."""
    za = np.asarray(za, dtype=np.complex128)
    zb = np.asarray(zb, dtype=np.complex128)
    t_loc = np.arange(int(start), int(end), dtype=np.int64)
    shifts = np.mod(np.asarray(shifts, dtype=np.int64), zb.size)
    return near_in_phase_counts_at(za[t_loc], t_loc, zb, shifts, cos_thr)


def _xcorr_edge_correction_np(
    a: np.ndarray, b: np.ndarray, shifts: np.ndarray, offset: int, max_lag: int
) -> np.ndarray:
    """The numpy reference of the exact edge correction: per draw and lag, the ``|lag|`` slice terms ``a[q] * b[j]``
    whose partner falls outside the slice ``[offset, offset + len(a))`` of the rolled segment ``b``, summed in
    ascending ``q`` (cumsum) -- the numba kernel's products and order, so the two agree bit for bit."""
    m = a.size
    n = b.size
    w = np.zeros((shifts.size, 2 * max_lag + 1))
    for li in range(2 * max_lag + 1):
        lag = li - max_lag
        if lag == 0:
            continue
        q = np.arange(m - lag, m) if lag > 0 else np.arange(0, -lag)
        j = np.mod(offset + q[None, :] + lag - shifts[:, None], n)
        w[:, li] = np.cumsum(a[q][None, :] * b[j], axis=1)[:, -1]
    return w


def _circ_sum_vec(prefix: np.ndarray, starts: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    ends = starts + lengths
    wrap = ends > n
    safe_end = np.where(wrap, ends - n, ends)
    nowrap_val = prefix[np.minimum(ends, n)] - prefix[starts]
    wrap_val = (prefix[n] - prefix[starts]) + prefix[safe_end]
    return np.where(wrap, wrap_val, nowrap_val)


@dataclass(frozen=True)
class LaggedPearsonBSide:
    """The B-segment quantities of :func:`shifted_slice_lagged_pearson`, shared by every slice rolled within it."""

    xb: np.ndarray  # b - b.mean()
    spectrum: np.ndarray  # rfft(xb)
    p_b: np.ndarray  # [0, cumsum(xb)]
    p_b2: np.ndarray  # [0, cumsum(xb * xb)]


def lagged_pearson_b_side(b: np.ndarray) -> LaggedPearsonBSide:
    """:func:`shifted_slice_lagged_pearson`'s B side of the finite segment ``b``. A caller scoring many windows against
    one B segment prepares it once and passes it as ``b_side=`` (the same values)."""
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(b).all():
        raise ValueError("lagged_pearson_b_side needs a finite b")
    xb = b - b.mean()  # centred: Pearson is shift-invariant (as lagged_pearson); FFT numerical stability
    return LaggedPearsonBSide(
        xb=xb,
        spectrum=_fft.rfft(xb),
        p_b=np.concatenate(([0.0], np.cumsum(xb))),
        p_b2=np.concatenate(([0.0], np.cumsum(xb * xb))),
    )


def shifted_slice_lagged_pearson(
    a: np.ndarray,
    b: np.ndarray,
    offset: int,
    shifts: np.ndarray,
    max_lag: int,
    *,
    b_side: LaggedPearsonBSide | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``lagged_pearson(a, roll(b, s)[offset : offset + len(a)], max_lag)`` for every shift ``s``, one FFT pair for all.

    The cross-correlation surrogate identity (spec 7.9) for a slice ``a`` lying at ``offset`` inside the B segment
    ``b`` that each draw rolls (``np.roll`` semantics). The circular FFT cross-correlation of ``a`` (zero outside the
    slice) with ``b`` gives every shift's lag sum over the whole slice; the exact correction removes, per lag, the
    ``|lag|`` terms whose partner falls outside the slice (ascending sums, numba optional, bit-identical backends);
    B's overlap sums come from circular prefix sums. ``r`` is ``(K, 2L+1)`` (NaN where a side is constant on the
    overlap), ``n`` the ``(2L+1,)`` overlap sizes. Inputs must be finite: the FFT would carry a non-finite value to
    every shift, where the direct computation NaNs only the draws that roll it into the slice. ``b_side`` is
    :func:`lagged_pearson_b_side` of ``b`` when the caller already holds it (it checked ``b`` finite).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    shifts = np.ascontiguousarray(shifts, dtype=np.int64)
    m, n, off, lag = a.size, b.size, int(offset), int(max_lag)
    if off < 0 or off + m > n:
        raise ValueError(f"slice [{off}, {off + m}) must lie inside the segment [0, {n})")
    if not 0 <= lag < m:
        raise ValueError(f"max_lag ({lag}) must lie in [0, {m}) (the slice length)")
    if b_side is None:
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("shifted_slice_lagged_pearson needs finite a and b")
        b_side = lagged_pearson_b_side(b)
    elif b_side.xb.shape != (n,):
        raise ValueError(f"b_side covers {b_side.xb.size} samples; the segment has {n}")
    elif not np.isfinite(a).all():
        raise ValueError("shifted_slice_lagged_pearson needs finite a and b")
    if shifts.size and (int(shifts.min()) < 0 or int(shifts.max()) >= n):
        raise ValueError(f"shifts must lie in [0, {n})")
    xa = a - a.mean()  # centred: Pearson is shift-invariant (as lagged_pearson); FFT numerical stability
    xb = b_side.xb
    lags = np.arange(-lag, lag + 1)
    ns = (m - np.abs(lags)).astype(np.int64)
    pos = lags >= 0
    l_pos = np.clip(lags, 0, None)
    l_neg = np.clip(-lags, 0, None)
    p_a = np.concatenate(([0.0], np.cumsum(xa)))
    p_a2 = np.concatenate(([0.0], np.cumsum(xa * xa)))
    s_a = np.where(pos, p_a[m - l_pos], p_a[m] - p_a[l_neg])
    s_aa = np.where(pos, p_a2[m - l_pos], p_a2[m] - p_a2[l_neg])

    a_seg = np.zeros(n)
    a_seg[off : off + m] = xa
    c_circ = _fft.irfft(np.conj(_fft.rfft(a_seg)) * b_side.spectrum, n=n)  # C(d) = sum_t a_seg[t] * xb[(t + d) mod n]
    if use_numba():
        w = xcorr_edge_correction_nb(np.ascontiguousarray(xa), np.ascontiguousarray(xb), shifts, off, lag)
    else:
        w = _xcorr_edge_correction_np(xa, xb, shifts, off, lag)
    s_ab = c_circ[np.mod(lags[None, :] - shifts[:, None], n)] - w

    start_b = np.mod(np.where(pos, off + lags, off)[None, :] - shifts[:, None], n)  # B's overlap, rolled
    lengths = np.broadcast_to(ns, start_b.shape)
    s_b = _circ_sum_vec(b_side.p_b, start_b, lengths, n)
    s_bb = _circ_sum_vec(b_side.p_b2, start_b, lengths, n)

    nf = ns.astype(np.float64)
    num = nf * s_ab - s_a * s_b
    var_a = nf * s_aa - s_a * s_a
    var_b = nf * s_bb - s_b * s_b
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(var_a * var_b)
        r = np.where(denom > 0, num / denom, np.nan)
    return r, ns


def shifted_lagged_pearson(
    a: np.ndarray, b: np.ndarray, shifts: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Lagged Pearson of ``a`` against each circularly shifted ``b`` (whole-segment), one FFT pair for all K.

    ``r`` is ``(K, 2L+1)``; ``n`` is the ``(2L+1,)`` overlap sizes. Equals ``lagged_pearson(a, roll(b, s))``: the
    whole-segment case (``offset = 0``, ``len(a) == len(b)``) of :func:`shifted_slice_lagged_pearson`.
    """
    b = np.asarray(b, dtype=np.float64)
    return shifted_slice_lagged_pearson(a, b, 0, np.mod(np.asarray(shifts, dtype=np.int64), b.size), max_lag)
