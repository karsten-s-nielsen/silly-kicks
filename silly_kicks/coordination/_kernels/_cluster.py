"""Cluster-phase synchrony for a team (Richardson et al. 2012).

Pure numpy (plus the numba null kernel in ``_numba``). The group phasor ``q(t)`` is the normalised sum of the valid
player phasors; each player's relative phase is ``z * conj(q)``. Window synchrony (``rho_group``) is measured on the
relative phase AFTER removing each player's mean relative phase, so a set of constant per-player lags reads as perfect
synchrony (Frank & Richardson): the players hold station relative to the group even though their offsets differ.

**The reference arithmetic (ADR-111 D4).** Every complex product is written out in real arithmetic (two rounded
products and one rounded sum per part); every sum runs in a fixed order -- over players in column order, over time in
ascending row order, each starting from ``+0.0`` and adding ``0.0`` for a masked entry; a unit phasor is
``s / sqrt(re^2 + im^2)`` with ``1 + 0j`` where ``|s| == 0`` (the value ``exp(1j * angle(0))`` gave) -- no
transcendental in the hot path. The numba kernel follows the same operations in the same order, so the surrogate null
never depends on whether numba is installed (spec 7.15), and on any CPU (numpy's complex multiply takes a
CPU-dispatched FMA path; the explicit form does not). The as-built complex arithmetic is kept in
``_cluster_reference`` -- the parity oracle (within 1e-12) and the corpus no-flip gate's reference leg.

:func:`shifted_rho_group_means` is the surrogate null (spec 7.9, per-player shifted phasors, plan R1's shift unit =
each player's own phase run): the window's ``rho_group_mean`` for every time-shift draw at once, bit-identical to
re-running :func:`cluster_phase` + :func:`window_cluster_stats` per draw (the observed estimator).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from silly_kicks.coordination._kernels._numba import HAVE_NUMBA, use_numba

if HAVE_NUMBA:
    from silly_kicks.coordination._kernels._numba import pearson_rows_nb, shifted_rho_group_means_nb

#: Complex elements per draw-chunk intermediate in the numpy :func:`shifted_rho_group_means` (~4 MiB each): bounds
#: peak memory for a whole-period window while keeping short (possession) windows to a single chunk of all draws.
SURROGATE_CHUNK_ELEMENTS = 1 << 18


class ShiftRun(NamedTuple):
    """One player's continuous phase run ``[lo, hi)`` and its circular shift for every surrogate draw."""

    player: int  # column of ``z``
    lo: int
    hi: int
    shifts: np.ndarray  # (n_draws,) ints in [0, hi - lo)


@dataclass(frozen=True)
class ClusterWindowStats:
    """Per-window cluster-phase statistics (Richardson et al. 2012)."""

    phi_bar: np.ndarray  # (K,) radians; NaN for a player with no usable sample
    rho_k: np.ndarray  # (K,) per-player locking to the group over the window
    rho_group_i: np.ndarray  # (n_usable,) instantaneous group synchrony over usable samples
    rho_group_mean: float
    rho_group_sd: float  # np.std(ddof=0): descriptive dispersion of the window's series
    n_players_mean: float


# --------------------------------------------------------------------------- the reference arithmetic (D4)
def _sum_players(x: np.ndarray) -> np.ndarray:
    """Sum over the LAST axis (players) in column order, starting from ``+0.0``."""
    acc = np.zeros(x.shape[:-1])
    for p in range(x.shape[-1]):
        acc = acc + x[..., p]
    return acc


def _sum_rows(x: np.ndarray, axis: int) -> np.ndarray:
    """Sum along ``axis`` (time) in ascending order, starting from ``+0.0`` (a sequential ``cumsum``)."""
    shape = list(x.shape)
    shape[axis] = 1
    padded = np.concatenate((np.zeros(shape), x), axis=axis)
    return np.take(np.cumsum(padded, axis=axis), -1, axis=axis)


def _rel_phasors(zr, zi, valid, usable):
    """``cluster_phase`` on ``(..., T, K)`` real/imag parts: the row-local group phasor and each valid player's relative
    phasor ``z * conj(q)`` (0 where not valid or the row is not usable; NaN where the group sum is exactly zero)."""
    sr = _sum_players(np.where(valid, zr, 0.0))
    si = _sum_players(np.where(valid, zi, 0.0))
    mag = np.sqrt(sr * sr + si * si)
    ok = usable & (mag > 0)
    safe = np.where(mag > 0, mag, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        qr = np.where(ok, sr / safe, np.nan)
        qi = np.where(ok, si / safe, np.nan)
    m = valid & usable[..., None]
    rel_re = np.where(m, zr * qr[..., None] + zi * qi[..., None], 0.0)
    rel_im = np.where(m, zi * qr[..., None] - zr * qi[..., None], 0.0)
    return qr, qi, rel_re, rel_im


def _unit_phasor(pr: np.ndarray, pi: np.ndarray, cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``s / |s|`` of each player's summed relative phasor; ``1 + 0j`` where ``|s| == 0``; NaN with no sample."""
    mp = np.sqrt(pr * pr + pi * pi)
    safe = np.where(mp > 0, mp, 1.0)
    er = np.where(mp > 0, pr / safe, 1.0)
    ei = np.where(mp > 0, pi / safe, 0.0)
    undefined = (cnt == 0) | np.isnan(mp)
    return np.where(undefined, np.nan, er), np.where(undefined, np.nan, ei)


def _group_synchrony(rel_re, rel_im, valid_u, er, ei):
    """``rho_group_i`` on usable rows ``(..., U, K)``: |sum over valid players of rel * conj(e)| / n_valid."""
    cr = rel_re * er[..., None, :] + rel_im * ei[..., None, :]
    ci = rel_im * er[..., None, :] - rel_re * ei[..., None, :]
    gr = _sum_players(np.where(valid_u, cr, 0.0))
    gi = _sum_players(np.where(valid_u, ci, 0.0))
    n_valid = valid_u.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_valid > 0, np.sqrt(gr * gr + gi * gi) / np.where(n_valid > 0, n_valid, 1), np.nan)


def _mean_finite(x: np.ndarray) -> np.ndarray:
    """Mean over the last axis of the non-NaN entries, summed in ascending order; NaN when there are none."""
    finite = ~np.isnan(x)
    total = _sum_rows(np.where(finite, x, 0.0), axis=-1)
    count = finite.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)


# --------------------------------------------------------------------------- the observed estimator
def cluster_phase(z: np.ndarray, valid: np.ndarray, min_players: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group phasor, per-player relative phase, and per-sample usability.

    ``z`` (T, K) unit phasors and ``valid`` (T, K) bool -> ``q`` (T,) group phasor (NaN+NaNj where a sample
    has fewer than ``min_players`` valid, or its players' phasors cancel exactly), ``rel`` (T, K) = ``z * conj(q)``
    (0 where not valid or not usable), and ``usable`` (T,) = ``valid.sum(axis=1) >= min_players``.
    """
    z = np.asarray(z, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    usable = valid.sum(axis=1) >= min_players
    qr, qi, rel_re, rel_im = _rel_phasors(np.real(z), np.imag(z), valid, usable)
    q = np.empty(qr.shape, dtype=np.complex128)
    q.real, q.imag = qr, qi
    rel = np.empty(rel_re.shape, dtype=np.complex128)
    rel.real, rel.imag = rel_re, rel_im
    return q, rel, usable


def window_cluster_stats(
    rel: np.ndarray, valid: np.ndarray, usable: np.ndarray, start: int, end: int
) -> ClusterWindowStats:
    """Cluster-phase statistics over the half-open window ``[start, end)``."""
    rel = np.asarray(rel, dtype=np.complex128)[start:end]
    valid = np.asarray(valid, dtype=bool)[start:end]
    usable = np.asarray(usable, dtype=bool)[start:end]
    m = valid & usable[:, None]  # per-player usable samples
    rel_re = np.ascontiguousarray(np.real(rel))
    rel_im = np.ascontiguousarray(np.imag(rel))
    pr = _sum_rows(np.where(m, rel_re, 0.0), axis=0)
    pi = _sum_rows(np.where(m, rel_im, 0.0), axis=0)
    cnt = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_bar = np.where(cnt > 0, np.arctan2(pi, pr), np.nan)
        rho_k = np.where(cnt > 0, np.sqrt(pr * pr + pi * pi) / np.where(cnt > 0, cnt, 1), np.nan)
    usable_idx = np.flatnonzero(usable)
    if usable_idx.size == 0:
        return ClusterWindowStats(
            phi_bar=phi_bar,
            rho_k=rho_k,
            rho_group_i=np.empty(0, dtype=np.float64),
            rho_group_mean=float("nan"),
            rho_group_sd=float("nan"),
            n_players_mean=float("nan"),
        )
    er, ei = _unit_phasor(pr, pi, cnt)
    valid_u = valid[usable_idx]
    rho_group_i = _group_synchrony(rel_re[usable_idx], rel_im[usable_idx], valid_u, er, ei)
    return ClusterWindowStats(
        phi_bar=phi_bar,
        rho_k=rho_k,
        rho_group_i=rho_group_i,
        rho_group_mean=float(_mean_finite(rho_group_i)),
        rho_group_sd=float(np.nanstd(rho_group_i, ddof=0)),
        n_players_mean=float(valid_u.sum(axis=1).mean()),
    )


# --------------------------------------------------------------------------- the surrogate null
def _validated_runs(runs: Sequence[ShiftRun], n_players: int, n_draws: int) -> list[ShiftRun]:
    """``runs`` with int64 shifts, after checking each is a non-empty in-range run with in-range shifts and that no
    player's runs overlap (the shift owning a shared row would be ambiguous)."""
    out: list[ShiftRun] = []
    last_hi: dict[int, int] = {}
    for run in sorted(runs, key=lambda r: (int(r.player), int(r.lo))):
        player, lo, hi = int(run.player), int(run.lo), int(run.hi)
        if not 0 <= player < n_players:
            raise ValueError(f"run ({lo}, {hi}) names player {player}; z has {n_players} player columns")
        if hi <= lo:
            raise ValueError(f"player {player} run ({lo}, {hi}) is empty")
        if lo < last_hi.get(player, lo):
            raise ValueError(f"player {player} runs overlap at ({lo}, {hi})")
        last_hi[player] = hi
        shifts = np.asarray(run.shifts, dtype=np.int64)
        if shifts.shape != (n_draws,):
            raise ValueError(
                f"player {player} run ({lo}, {hi}) shifts have shape {shifts.shape}; expected ({n_draws},)"
            )
        if shifts.size and (int(shifts.min()) < 0 or int(shifts.max()) >= hi - lo):
            raise ValueError(f"player {player} run ({lo}, {hi}) shifts must lie in [0, {hi - lo})")
        out.append(ShiftRun(player, lo, hi, shifts))
    return out


def _shifted_rho_group_means_np(z, valid, usable, start, end, runs, n_draws) -> np.ndarray:
    """The numpy reference of the null: the observed estimator's operations on a leading draw axis, in draw chunks."""
    k = z.shape[1]
    out = np.full(n_draws, np.nan)
    w = end - start
    vw = valid[start:end]
    uw = usable[start:end]
    usable_idx = np.flatnonzero(uw)
    m = vw & uw[:, None]
    cnt = m.sum(axis=0)
    valid_u = vw[usable_idx]
    cols = np.arange(k, dtype=np.int64)
    base_flat = np.arange(start, end, dtype=np.int64)[:, None] * k + cols[None, :]
    z_flat = z.reshape(-1)
    per_chunk = max(1, SURROGATE_CHUNK_ELEMENTS // (w * max(k, 1)))
    for d0 in range(0, n_draws, per_chunk):
        d1 = min(n_draws, d0 + per_chunk)
        flat = np.empty((d1 - d0, w, k), dtype=np.int64)
        flat[:] = base_flat
        for player, lo, hi, sh in runs:
            a, b = max(lo, start), min(hi, end)
            if b <= a:
                continue
            n = hi - lo
            offset = np.arange(a - lo, b - lo, dtype=np.int64)[None, :] - sh[d0:d1, None]
            offset += n * (offset < 0)  # == offset mod n, since offset lies in (-n, n)
            flat[:, a - start : b - start, player] = (offset + lo) * k + player
        zk = np.take(z_flat, flat)
        _qr, _qi, rel_re, rel_im = _rel_phasors(np.real(zk), np.imag(zk), vw, uw)
        pr = _sum_rows(np.where(m, rel_re, 0.0), axis=1)
        pi = _sum_rows(np.where(m, rel_im, 0.0), axis=1)
        er, ei = _unit_phasor(pr, pi, cnt)
        rho = _group_synchrony(rel_re[:, usable_idx], rel_im[:, usable_idx], valid_u, er, ei)
        out[d0:d1] = _mean_finite(rho)
    return out


def phasor_parts(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C-contiguous ``(real, imag)`` of ``z``, the operands the numba null reads. A caller scoring many windows of one
    ``z`` splits it once and passes the pair as ``shifted_rho_group_means(..., parts=)`` (identical values)."""
    return np.ascontiguousarray(np.real(z)), np.ascontiguousarray(np.imag(z))


def shifted_rho_group_means(
    z: np.ndarray,
    valid: np.ndarray,
    usable: np.ndarray,
    start: int,
    end: int,
    runs: Sequence[ShiftRun],
    n_draws: int,
    *,
    parts: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Window ``rho_group_mean`` of every per-player time-shift surrogate draw, shape ``(n_draws,)``.

    Draw ``d`` circularly shifts player ``run.player``'s phasor within each of its runs ``[lo, hi)`` by
    ``run.shifts[d]`` (``np.roll`` semantics: ``rolled[i] = z[lo + (i - shift) mod (hi - lo), player]``); ``valid``
    is NOT shifted. Every draw is exactly :func:`cluster_phase` on the whole shifted period followed by
    :func:`window_cluster_stats` ``(...).rho_group_mean`` over ``[start, end)`` -- the observed estimator, bit for bit
    -- because ``cluster_phase`` is row-local (only the window rows are gathered; ``usable``, a function of the
    unshifted ``valid``, is the caller's base ``cluster_phase(z, valid, min_players)[2]``) and both backends take the
    module's reference arithmetic in its fixed order. A player's samples outside its runs are left as they are (in real
    data: invalid NaN phasors, masked out). ``parts`` is :func:`phasor_parts` of ``z`` when the caller already holds
    it (the numba path then skips the per-call split; the values are the same).
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    valid = np.ascontiguousarray(valid, dtype=bool)
    usable = np.ascontiguousarray(usable, dtype=bool)
    k = z.shape[1]
    checked = _validated_runs(runs, k, n_draws)
    start, end = int(start), int(end)
    if n_draws == 0 or end <= start or not usable[start:end].any():
        return np.full(n_draws, np.nan)  # no usable sample: NaN for every draw (the observed estimator's branch)
    if use_numba():
        n_runs = len(checked)
        z_re, z_im = phasor_parts(z) if parts is None else parts
        if z_re.shape != z.shape or z_im.shape != z.shape:
            raise ValueError(f"parts have shapes {z_re.shape} / {z_im.shape}; z has {z.shape}")
        return shifted_rho_group_means_nb(
            z_re,
            z_im,
            valid,
            usable,
            start,
            end,
            np.array([r.player for r in checked], dtype=np.int64),
            np.array([r.lo for r in checked], dtype=np.int64),
            np.array([r.hi for r in checked], dtype=np.int64),
            np.ascontiguousarray(np.stack([r.shifts for r in checked]) if n_runs else np.zeros((0, n_draws), np.int64)),
            n_draws,
        )
    return _shifted_rho_group_means_np(z, valid, usable, start, end, checked, n_draws)


# --------------------------------------------------------------------------- team-sync Pearson (D4)
def _pearson_rows_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    """The numpy reference of :func:`pearson_rows`: the numba kernel's two-pass sums, in the same ascending order."""
    m = mask[None, :] & np.isfinite(b)
    n = m.sum(axis=1)
    a_rows = np.broadcast_to(a, b.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = _sum_rows(np.where(m, a_rows, 0.0), axis=1) / n
        mb = _sum_rows(np.where(m, b, 0.0), axis=1) / n
        da = np.where(m, a_rows - ma[:, None], 0.0)
        db = np.where(m, b - mb[:, None], 0.0)
        sab = _sum_rows(da * db, axis=1)
        saa = _sum_rows(da * da, axis=1)
        sbb = _sum_rows(db * db, axis=1)
        r = np.clip(sab / np.sqrt(saa * sbb), -1.0, 1.0)
    return np.where((n >= min_n) & (saa != 0.0) & (sbb != 0.0), r, np.nan)


def pearson_rows(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    """Pearson r of ``a`` against every row of ``b`` over ``mask & isfinite(row)`` -- the team-sync estimator (D4).

    Two-pass (means, then centred sums), every sum in ascending order from ``+0.0``; clipped to ``[-1, 1]``. NaN for a
    row with fewer than ``min_n`` rows in play or a zero-variance side. Both backends agree bit for bit.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(np.atleast_2d(b), dtype=np.float64)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if b.shape[1] != a.size or mask.shape != a.shape:
        raise ValueError(f"a {a.shape}, mask {mask.shape} and the rows of b {b.shape} must share one length")
    if use_numba():
        return pearson_rows_nb(a, b, mask, int(min_n))
    return _pearson_rows_np(a, b, mask, int(min_n))
