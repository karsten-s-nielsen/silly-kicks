"""Numba-accelerated exact pair counters for coordination sample entropy (with numpy fallbacks), plus the two
surrogate-identity kernels (spec 7.9): the % near-in-phase count and the cross-correlation edge correction, each with a
bit-identical numpy reference in ``_surrogates`` (same arithmetic, same ascending order).

Pure counting. The distance predicate is always the naive Chebyshev one, ``|p - q| <= r`` per coordinate.
The numba path is an O(N log N) Fenwick sweep; the numpy fallback is a ``cKDTree`` neighbour count. Both
return identical INTEGER counts on data with no pair sitting exactly at ``r`` (the numba path additionally
matches the naive predicate at exact-``r`` ties, via predicate bisection rather than ``searchsorted(s + r)``).
Selection honours ``SILLY_KICKS_COORDINATION_FORCE_NUMPY`` so the fallback is testable with numba installed.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAVE_NUMBA = False

#: numba's on-disk cache is OPT-IN (the repo rule, ``tests/tracking/test_numba_cache_gating.py``): an on-disk cache
#: resolves a writable cache locator AT DECORATION -- these signatures are eager, so at ``import
#: silly_kicks.coordination`` -- and raises ``RuntimeError`` on a read-only install. Off by default;
#: ``SILLY_KICKS_NUMBA_CACHE=1`` or numba's own ``NUMBA_CACHE_DIR`` opts in. Off costs one per-process compile,
#: never speed.
_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))


def use_numba() -> bool:
    """True iff numba is importable AND ``SILLY_KICKS_COORDINATION_FORCE_NUMPY`` is not ``"1"``."""
    return HAVE_NUMBA and os.environ.get("SILLY_KICKS_COORDINATION_FORCE_NUMPY") != "1"


if HAVE_NUMBA:
    from numba.core import types

    # Read-only, C-contiguous INPUT array types for the surrogate kernels: callers hand them read-only cached arrays
    # (``_ClusterCtx``), and a read-only signature accepts a writeable array too -- a mutable one would reject them.
    _F1 = types.Array(types.float64, 1, "C", readonly=True)
    _F2 = types.Array(types.float64, 2, "C", readonly=True)
    _I1 = types.Array(types.int64, 1, "C", readonly=True)
    _I2 = types.Array(types.int64, 2, "C", readonly=True)
    _B1 = types.Array(types.boolean, 1, "C", readonly=True)
    _B2 = types.Array(types.boolean, 2, "C", readonly=True)

    @njit("int64(float64[:], float64)", cache=_NUMBA_CACHE)
    def _pairs_within_sorted_nb(s, r):  # pragma: no cover - compiled
        # #{p < q : s[q] - s[p] <= r} on ascending s, by two pointers. For a fixed p the naive predicate holds on a
        # prefix of q > p, and that prefix only grows with p (IEEE subtraction is monotone in each operand), so the
        # sweep evaluates the bisection's predicate on the same pairs and returns the same integer.
        n = s.size
        total = 0
        hi = 0
        for p in range(n):
            if hi < p + 1:
                hi = p + 1
            while hi < n and (s[hi] - s[p]) <= r:
                hi += 1
            total += hi - (p + 1)
        return total

    @njit("int64(float64[:], float64[:], float64)", cache=_NUMBA_CACHE)
    def _dominance_count_self_nb(a, b, r):  # pragma: no cover - compiled
        n = a.size
        if n < 2:
            return 0
        order = np.argsort(a)
        a_s = a[order]
        b_s = b[order]
        b_ord = np.argsort(b_s)
        sb = b_s[b_ord]
        rank = np.empty(n, np.int64)
        for k in range(n):
            rank[b_ord[k]] = k
        bit = np.zeros(n + 1, np.int64)
        total = 0
        lo_ptr = 0
        for ii in range(n):
            ap = a_s[ii]
            while lo_ptr < ii and (ap - a_s[lo_ptr]) > r:
                idx = rank[lo_ptr] + 1
                while idx <= n:
                    bit[idx] -= 1
                    idx += idx & (-idx)
                lo_ptr += 1
            bp = b_s[ii]
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if (bp - sb[mid]) <= r:
                    hi = mid
                else:
                    lo = mid + 1
            lo_rank = lo
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if (sb[mid] - bp) > r:
                    hi = mid
                else:
                    lo = mid + 1
            hi_rank = lo
            s = 0
            i2 = hi_rank
            while i2 > 0:
                s += bit[i2]
                i2 -= i2 & (-i2)
            i2 = lo_rank
            while i2 > 0:
                s -= bit[i2]
                i2 -= i2 & (-i2)
            total += s
            idx = rank[ii] + 1
            while idx <= n:
                bit[idx] += 1
                idx += idx & (-idx)
        return total

    @njit("int64(float64[:], float64[:], float64[:], float64[:], float64)", cache=_NUMBA_CACHE)
    def _dominance_count_cross_nb(a1, b1, a2, b2, r):  # pragma: no cover - compiled
        n1 = a1.size
        n2 = a2.size
        if n1 == 0 or n2 == 0:
            return 0
        o1 = np.argsort(a1)
        a1s = a1[o1]
        b1s = b1[o1]
        b_ord = np.argsort(b1s)
        sb = b1s[b_ord]
        rank = np.empty(n1, np.int64)
        for k in range(n1):
            rank[b_ord[k]] = k
        # per-query a-index window [alo, ahi) via predicate bisection on a1s
        alo = np.empty(n2, np.int64)
        ahi = np.empty(n2, np.int64)
        for j in range(n2):
            lo = 0
            hi = n1
            while lo < hi:
                mid = (lo + hi) // 2
                if (a2[j] - a1s[mid]) <= r:
                    hi = mid
                else:
                    lo = mid + 1
            alo[j] = lo
            lo = 0
            hi = n1
            while lo < hi:
                mid = (lo + hi) // 2
                if (a1s[mid] - a2[j]) > r:
                    hi = mid
                else:
                    lo = mid + 1
            ahi[j] = lo
        # events: +1 at threshold ahi[j], -1 at threshold alo[j]; sweep inserting a1-sorted points
        ev_m = np.empty(2 * n2, np.int64)
        ev_j = np.empty(2 * n2, np.int64)
        ev_s = np.empty(2 * n2, np.int64)
        for j in range(n2):
            ev_m[2 * j] = ahi[j]
            ev_j[2 * j] = j
            ev_s[2 * j] = 1
            ev_m[2 * j + 1] = alo[j]
            ev_j[2 * j + 1] = j
            ev_s[2 * j + 1] = -1
        ev_order = np.argsort(ev_m)
        bit = np.zeros(n1 + 1, np.int64)
        inserted = 0
        ans = np.zeros(n2, np.int64)
        for e in range(2 * n2):
            ei = ev_order[e]
            m = ev_m[ei]
            j = ev_j[ei]
            sgn = ev_s[ei]
            while inserted < m:
                idx = rank[inserted] + 1
                while idx <= n1:
                    bit[idx] += 1
                    idx += idx & (-idx)
                inserted += 1
            lo = 0
            hi = n1
            while lo < hi:
                mid = (lo + hi) // 2
                if (b2[j] - sb[mid]) <= r:
                    hi = mid
                else:
                    lo = mid + 1
            blo = lo
            lo = 0
            hi = n1
            while lo < hi:
                mid = (lo + hi) // 2
                if (sb[mid] - b2[j]) > r:
                    hi = mid
                else:
                    lo = mid + 1
            bhi = lo
            s = 0
            i2 = bhi
            while i2 > 0:
                s += bit[i2]
                i2 -= i2 & (-i2)
            i2 = blo
            while i2 > 0:
                s -= bit[i2]
                i2 -= i2 & (-i2)
            ans[j] += sgn * s
        total = 0
        for j in range(n2):
            total += ans[j]
        return total

    @njit(types.int64[:](_F1, _F1, _I1, _F1, _F1, _I1, types.float64), cache=_NUMBA_CACHE)
    def near_in_phase_counts_at_nb(a_re, a_im, t_loc, b_re, b_im, shifts, cos_thr):  # pragma: no cover - compiled
        n = b_re.size
        m = t_loc.size
        k = shifts.size
        out = np.zeros(k, np.int64)
        # Maximal runs of consecutive positions, [blk_lo, blk_hi) in index space: within one, a shift maps the
        # positions onto a contiguous stretch of b (split once where it passes the segment end), so the inner loops
        # read both operands contiguously and vectorise. Integer counts: the loop order cannot change the result.
        blk_lo = np.empty(m, np.int64)
        blk_hi = np.empty(m, np.int64)
        n_blk = 0
        if m > 0:
            blk_lo[0] = 0
            for i in range(1, m):
                if t_loc[i] != t_loc[i - 1] + 1:
                    blk_hi[n_blk] = i
                    n_blk += 1
                    blk_lo[n_blk] = i
            blk_hi[n_blk] = m
            n_blk += 1
        for ki in range(k):
            s = shifts[ki]
            c = 0
            for blk in range(n_blk):
                i0 = blk_lo[blk]
                i1 = blk_hi[blk]
                j0 = t_loc[i0] - s  # t_loc and s lie in [0, n): one wrap at most
                if j0 < 0:
                    j0 += n
                run = min(i1 - i0, n - j0)
                for q in range(run):  # Re(a * conj(b_j)): two rounded products, one rounded sum
                    re = a_re[i0 + q] * b_re[j0 + q] + a_im[i0 + q] * b_im[j0 + q]
                    c += 1 if re >= cos_thr else 0
                for q in range(run, i1 - i0):  # past the segment end, b continues from its start
                    jj = q - run
                    re = a_re[i0 + q] * b_re[jj] + a_im[i0 + q] * b_im[jj]
                    c += 1 if re >= cos_thr else 0
            out[ki] = c
        return out

    @njit(types.float64[:, :](_F1, _F1, _I1, types.int64, types.int64), cache=_NUMBA_CACHE)
    def xcorr_edge_correction_nb(a, b, shifts, offset, max_lag):  # pragma: no cover - compiled
        m = a.size
        n = b.size
        k = shifts.size
        w = np.zeros((k, 2 * max_lag + 1))
        for ki in range(k):
            s = shifts[ki]
            for li in range(2 * max_lag + 1):
                lag = li - max_lag
                if lag == 0:
                    continue
                if lag > 0:  # slice terms whose partner lies past the slice end
                    q0 = m - lag
                    q1 = m
                else:  # slice terms whose partner lies before the slice start
                    q0 = 0
                    q1 = -lag
                j = ((offset + q0 + lag - s) % n + n) % n
                tot = 0.0
                for q in range(q0, q1):  # ascending q: the numpy reference's cumsum order
                    tot += a[q] * b[j]
                    j += 1
                    if j == n:
                        j = 0
                w[ki, li] = tot
        return w

    @njit(
        types.float64[:](_F2, _F2, _B2, _B1, types.int64, types.int64, _I1, _I1, _I1, _I2, types.int64),
        cache=_NUMBA_CACHE,
    )
    def shifted_rho_group_means_nb(
        z_re, z_im, valid, usable, start, end, run_player, run_lo, run_hi, run_shifts, n_draws
    ):  # pragma: no cover - compiled
        # ADR-111 D4: ``_cluster``'s reference arithmetic, operation for operation and in its order (players in column
        # order, rows ascending, every sum from +0.0 adding 0.0 for a masked entry) -- bit-identical to the numpy null.
        k = z_re.shape[1]
        w = end - start
        out = np.empty(n_draws)
        gr = np.empty((w, k))
        gi = np.empty((w, k))
        rel_re = np.empty((w, k))
        rel_im = np.empty((w, k))
        pr = np.empty(k)
        pi = np.empty(k)
        er = np.empty(k)
        ei = np.empty(k)
        cnt = np.zeros(k, np.int64)
        for t in range(w):
            if usable[start + t]:
                for p in range(k):
                    if valid[start + t, p]:
                        cnt[p] += 1
        for d in range(n_draws):
            for t in range(w):  # the draw's window rows: unshifted, then each run rolled by its shift
                for p in range(k):
                    gr[t, p] = z_re[start + t, p]
                    gi[t, p] = z_im[start + t, p]
            for r in range(run_player.size):
                p = run_player[r]
                lo = run_lo[r]
                n = run_hi[r] - lo
                s = run_shifts[r, d]
                for t in range(max(lo, start), min(run_hi[r], end)):
                    off = t - lo - s
                    if off < 0:
                        off += n
                    gr[t - start, p] = z_re[lo + off, p]
                    gi[t - start, p] = z_im[lo + off, p]
            for p in range(k):
                pr[p] = 0.0
                pi[p] = 0.0
            for t in range(w):  # cluster_phase on the row, then each player's window sum of rel
                row = start + t
                sr = 0.0
                si = 0.0
                for p in range(k):
                    if valid[row, p]:
                        sr += gr[t, p]
                        si += gi[t, p]
                    else:
                        sr += 0.0
                        si += 0.0
                mag = np.sqrt(sr * sr + si * si)
                u = usable[row]
                if u and mag > 0.0:
                    qr = sr / mag
                    qi = si / mag
                else:
                    qr = np.nan
                    qi = np.nan
                for p in range(k):
                    if valid[row, p] and u:
                        xr = gr[t, p] * qr + gi[t, p] * qi
                        xi = gi[t, p] * qr - gr[t, p] * qi
                    else:
                        xr = 0.0
                        xi = 0.0
                    rel_re[t, p] = xr
                    rel_im[t, p] = xi
                    pr[p] += xr
                    pi[p] += xi
            for p in range(k):  # unit phasor of each player's sum: 1 + 0j where |s| == 0, NaN with no sample
                mp = np.sqrt(pr[p] * pr[p] + pi[p] * pi[p])
                if cnt[p] == 0 or np.isnan(mp):
                    er[p] = np.nan
                    ei[p] = np.nan
                elif mp > 0.0:
                    er[p] = pr[p] / mp
                    ei[p] = pi[p] / mp
                else:
                    er[p] = 1.0
                    ei[p] = 0.0
            total = 0.0
            count = 0
            for t in range(w):  # rho_group over the usable rows; the mean of the finite ones
                row = start + t
                if not usable[row]:
                    continue
                gsr = 0.0
                gsi = 0.0
                nv = 0
                for p in range(k):
                    if valid[row, p]:
                        gsr += rel_re[t, p] * er[p] + rel_im[t, p] * ei[p]
                        gsi += rel_im[t, p] * er[p] - rel_re[t, p] * ei[p]
                        nv += 1
                    else:
                        gsr += 0.0
                        gsi += 0.0
                if nv > 0:
                    rho = np.sqrt(gsr * gsr + gsi * gsi) / nv
                    if not np.isnan(rho):
                        total += rho
                        count += 1
            out[d] = total / count if count > 0 else np.nan
        return out

    @njit(types.float64[:](_F1, _F2, _B1, types.int64), cache=_NUMBA_CACHE)
    def pearson_rows_nb(a, b, mask, min_n):  # pragma: no cover - compiled
        # ADR-111 D4 team-sync Pearson: two passes, every sum ascending from +0.0 (0.0 added for a masked entry) --
        # bit-identical to ``_cluster._pearson_rows_np``.
        rows = b.shape[0]
        w = b.shape[1]
        out = np.empty(rows)
        for r in range(rows):
            n = 0
            sa = 0.0
            sb = 0.0
            for t in range(w):
                if mask[t] and np.isfinite(b[r, t]):
                    n += 1
                    sa += a[t]
                    sb += b[r, t]
                else:
                    sa += 0.0
                    sb += 0.0
            if n < min_n:
                out[r] = np.nan
                continue
            ma = sa / n
            mb = sb / n
            sab = 0.0
            saa = 0.0
            sbb = 0.0
            for t in range(w):
                if mask[t] and np.isfinite(b[r, t]):
                    da = a[t] - ma
                    db = b[r, t] - mb
                else:
                    da = 0.0
                    db = 0.0
                sab += da * db
                saa += da * da
                sbb += db * db
            if saa == 0.0 or sbb == 0.0:
                out[r] = np.nan
                continue
            v = sab / np.sqrt(saa * sbb)
            out[r] = min(max(v, -1.0), 1.0)
        return out


def _pairs_within_sorted_np(s: np.ndarray, r: float) -> int:
    """#{p < q : s[q] - s[p] <= r} on ascending ``s``, via predicate bisection per ``p`` (tie-exact)."""
    n = s.size
    p = np.arange(n)
    lo, hi = p + 1, np.full(n, n)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        mid = (lo + hi) // 2
        ok = (mid < n) & (s[np.minimum(mid, n - 1)] - s <= r)
        lo = np.where(ok & (lo < hi), mid + 1, lo)
        hi = np.where(~ok & (lo < hi), mid, hi)
    return int((lo - (p + 1)).sum())


def pairs_within_sorted(s: np.ndarray, r: float) -> int:
    """Count pairs ``p < q`` of the ascending ``s`` with ``s[q] - s[p] <= r`` (the naive predicate, ties at ``r``
    included). Sample entropy's 1-D template count; both backends return the same integer."""
    s = np.ascontiguousarray(s, dtype=np.float64)
    if use_numba():
        return int(_pairs_within_sorted_nb(s, float(r)))
    return _pairs_within_sorted_np(s, r)


def _dominance_count_self_np(a: np.ndarray, b: np.ndarray, r: float) -> int:
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    pts = np.column_stack([a, b])
    n = len(pts)
    if n < 2:
        return 0
    tree = cKDTree(pts)
    return (int(tree.count_neighbors(tree, r, p=np.inf)) - n) // 2


def _dominance_count_cross_np(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray, r: float) -> int:
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    if len(a1) == 0 or len(a2) == 0:
        return 0
    u = cKDTree(np.column_stack([a1, b1]))
    v = cKDTree(np.column_stack([a2, b2]))
    return int(u.count_neighbors(v, r, p=np.inf))


def dominance_count_self(a: np.ndarray, b: np.ndarray, r: float) -> int:
    """Count unordered pairs ``i < j`` of 2-D points ``(a, b)`` within Chebyshev distance ``r``."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if use_numba():
        return int(_dominance_count_self_nb(a, b, float(r)))
    return _dominance_count_self_np(a, b, r)


def dominance_count_cross(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray, r: float) -> int:
    """Count ordered pairs ``(i in set 1, j in set 2)`` of 2-D points within Chebyshev distance ``r``."""
    a1 = np.ascontiguousarray(a1, dtype=np.float64)
    b1 = np.ascontiguousarray(b1, dtype=np.float64)
    a2 = np.ascontiguousarray(a2, dtype=np.float64)
    b2 = np.ascontiguousarray(b2, dtype=np.float64)
    if use_numba():
        return int(_dominance_count_cross_nb(a1, b1, a2, b2, float(r)))
    return _dominance_count_cross_np(a1, b1, a2, b2, r)
