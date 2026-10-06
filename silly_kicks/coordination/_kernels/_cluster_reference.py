"""The AS-BUILT cluster-phase and team-sync arithmetic (TF-58 before ADR-111 D4), kept verbatim as the reference.

Selected by ``SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS=1`` in ``_compute``: the parity oracle of the D4 tests (the
production arithmetic in ``_cluster`` lies within 1e-12 of it) and the reference leg of the corpus no-flip gate
(``scripts/validate_coordination_numerics.py``). Production never scores through it.

These functions keep numpy's complex arithmetic and reductions: ``z * conj(q)`` (a CPU-dispatched FMA path), pairwise
player sums, ``exp(1j * angle(.))`` unit phasors, ``np.nanmean`` and ``np.corrcoef`` -- which is exactly why the null
could not be ported to numba bit for bit and why D4 redefined it. Their bits are therefore CPU-dependent: the no-flip
gate compares the two numerics on one machine.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from silly_kicks.coordination._kernels._cluster import ClusterWindowStats, ShiftRun, _validated_runs

#: Complex elements per draw-chunk intermediate in :func:`shifted_rho_group_means` (~4 MiB each).
SURROGATE_CHUNK_ELEMENTS = 1 << 18


def cluster_phase(z: np.ndarray, valid: np.ndarray, min_players: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As-built group phasor, per-player relative phase and per-sample usability (see ``_cluster.cluster_phase``)."""
    z = np.asarray(z, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    zv = np.where(valid, z, 0.0)
    ssum = zv.sum(axis=1)
    n_valid = valid.sum(axis=1)
    usable = n_valid >= min_players
    mag = np.abs(ssum)
    ok = usable & (mag > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(ok, ssum / np.where(mag > 0, mag, 1.0), complex(np.nan, np.nan))
    rel_mask = valid & usable[:, None]
    rel = np.where(rel_mask, z * np.conj(q)[:, None], 0.0)
    return q, rel, usable


def window_cluster_stats(
    rel: np.ndarray, valid: np.ndarray, usable: np.ndarray, start: int, end: int
) -> ClusterWindowStats:
    """As-built cluster-phase statistics over the half-open window ``[start, end)``."""
    rel = np.asarray(rel, dtype=np.complex128)[start:end]
    valid = np.asarray(valid, dtype=bool)[start:end]
    usable = np.asarray(usable, dtype=bool)[start:end]

    m = valid & usable[:, None]  # per-player usable samples
    ssum = np.where(m, rel, 0.0).sum(axis=0)
    cnt = m.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_bar = np.where(cnt > 0, np.angle(ssum), np.nan)
        rho_k = np.where(cnt > 0, np.abs(ssum) / np.where(cnt > 0, cnt, 1), np.nan)

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
    # Group synchrony on the MEAN-CENTRED relative phase: e^{i(rel_angle - phi_bar)}.
    centred = rel[usable_idx] * np.conj(np.exp(1j * phi_bar))[None, :]
    valid_u = valid[usable_idx]
    n_valid_u = valid_u.sum(axis=1)
    gsum = np.where(valid_u, centred, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_group_i = np.where(n_valid_u > 0, np.abs(gsum) / np.where(n_valid_u > 0, n_valid_u, 1), np.nan)
    return ClusterWindowStats(
        phi_bar=phi_bar,
        rho_k=rho_k,
        rho_group_i=rho_group_i,
        rho_group_mean=float(np.nanmean(rho_group_i)),
        rho_group_sd=float(np.nanstd(rho_group_i, ddof=0)),
        n_players_mean=float(n_valid_u.mean()),
    )


def shifted_rho_group_means(
    z: np.ndarray,
    valid: np.ndarray,
    usable: np.ndarray,
    start: int,
    end: int,
    runs: Sequence[ShiftRun],
    n_draws: int,
) -> np.ndarray:
    """As-built batched null: byte-identical to, per draw, the as-built ``cluster_phase`` + ``window_cluster_stats``.

    Every reduction runs on a C-contiguous array along the axis the reference reduces (per-row player sums and the
    final 1-D ``nanmean`` pairwise on the fast axis, per-player window sums sequential in time).
    """
    z = np.ascontiguousarray(z, dtype=np.complex128)
    valid = np.asarray(valid, dtype=bool)
    usable = np.asarray(usable, dtype=bool)
    k = z.shape[1]
    runs = _validated_runs(runs, k, n_draws)
    out = np.full(n_draws, np.nan)
    w = int(end) - int(start)
    vw = valid[start:end]
    uw = usable[start:end]
    usable_idx = np.flatnonzero(uw)
    if n_draws == 0 or w <= 0 or usable_idx.size == 0:
        return out  # the reference's no-usable-sample branch: NaN for every draw
    m = vw & uw[:, None]  # == cluster_phase's rel_mask and window_cluster_stats' m, on the window rows
    cnt = m.sum(axis=0)
    valid_u = np.ascontiguousarray(vw[usable_idx])
    n_valid_u = valid_u.sum(axis=1)
    denom_u = np.where(n_valid_u > 0, n_valid_u, 1)
    cols = np.arange(k, dtype=np.int64)
    base_flat = np.arange(start, end, dtype=np.int64)[:, None] * k + cols[None, :]  # unshifted (row, player)
    z_flat = z.reshape(-1)
    contig = np.ascontiguousarray
    per_chunk = max(1, SURROGATE_CHUNK_ELEMENTS // (w * max(k, 1)))
    for d0 in range(0, n_draws, per_chunk):
        d1 = min(n_draws, d0 + per_chunk)
        flat = np.empty((d1 - d0, w, k), dtype=np.int64)
        flat[:] = base_flat
        for player, lo, hi, sh in runs:
            a, b = max(lo, int(start)), min(hi, int(end))
            if b <= a:
                continue
            n = hi - lo
            offset = np.arange(a - lo, b - lo, dtype=np.int64)[None, :] - sh[d0:d1, None]
            offset += n * (offset < 0)  # == offset mod n, since offset lies in (-n, n)
            flat[:, a - start : b - start, player] = (offset + lo) * k + player
        zk = np.take(z_flat, flat)  # (dc, w, k), C-contiguous
        # cluster_phase on the window rows (row-local).
        zv = contig(np.where(vw, zk, 0.0))
        ssum = contig(zv.sum(axis=2))
        mag = np.abs(ssum)
        ok = uw & (mag > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = contig(np.where(ok, ssum / np.where(mag > 0, mag, 1.0), complex(np.nan, np.nan)))
        rel = contig(np.where(m, zk * np.conj(q)[..., None], 0.0))
        # window_cluster_stats(...).rho_group_mean (same errstate scopes as the reference).
        psum = contig(contig(np.where(m, rel, 0.0)).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            phi_bar = contig(np.where(cnt > 0, np.angle(psum), np.nan))
        rel_u = rel if usable_idx.size == w else contig(np.take(rel, usable_idx, axis=1))
        centred = contig(rel_u * np.conj(np.exp(1j * phi_bar))[:, None, :])
        gsum = contig(contig(np.where(valid_u, centred, 0.0)).sum(axis=2))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho_i = contig(np.where(n_valid_u > 0, np.abs(gsum) / denom_u, np.nan))
        out[d0:d1] = np.nanmean(rho_i, axis=1)
    return out


def pearson_rows(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int) -> np.ndarray:
    """As-built team-sync estimator per row of ``b``: over ``mask & isfinite(row)``, NaN below ``min_n`` rows or when
    either side's ``np.std`` is 0, else ``np.corrcoef`` (a's std is reused while the mask repeats -- value-neutral)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    mask = np.asarray(mask, dtype=bool)
    out = np.empty(b.shape[0])
    a_std_mask: np.ndarray | None = None
    a_std = float("nan")
    for i, row in enumerate(b):
        m = mask & np.isfinite(row)
        if int(m.sum()) < min_n:
            out[i] = np.nan
            continue
        if a_std_mask is None or not np.array_equal(m, a_std_mask):
            a_std_mask, a_std = m, float(np.std(a[m]))
        if a_std == 0 or np.std(row[m]) == 0:
            out[i] = np.nan
        else:
            out[i] = float(np.corrcoef(a[m], row[m])[0, 1])
    return out
