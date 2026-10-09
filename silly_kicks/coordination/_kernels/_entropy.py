"""Sample entropy and cross sample entropy for coordination (Richman & Moorman 2000).

Pure numpy (plus the numba/cKDTree pair counters in ``_numba``). Counts are integers, so the numba and
numpy paths give bit-identical entropy. The ``m = 1`` path uses exact predicate-bisection counters (tie-safe
at exact-``r`` boundaries); ``m >= 2`` uses the ``cKDTree`` neighbour count.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from silly_kicks.coordination._kernels._numba import dominance_count_cross, dominance_count_self, pairs_within_sorted


def _count_1d_pairs_within(v: np.ndarray, r: float) -> int:
    """#{p < q : |v_p - v_q| <= r} on the sorted values (tie-exact; numba two-pointer or numpy bisection)."""
    return pairs_within_sorted(np.sort(v), r)


def _count_within_sorted(q: np.ndarray, s: np.ndarray, r: float) -> np.ndarray:
    """Per query in ``q``, the count of sorted-reference values ``s`` within ``[q - r, q + r]`` (bisection)."""
    n = s.size
    m = q.size
    if n == 0 or m == 0:
        return np.zeros(m, dtype=np.int64)
    steps = int(np.ceil(np.log2(max(n, 2)))) + 1
    lo = np.zeros(m, dtype=np.int64)
    hi = np.full(m, n, dtype=np.int64)
    for _ in range(steps):  # lower bound: first k with (q - s[k]) <= r
        mid = (lo + hi) // 2
        ok = (mid < n) & ((q - s[np.minimum(mid, n - 1)]) <= r)
        hi = np.where(ok & (lo < hi), mid, hi)
        lo = np.where((~ok) & (lo < hi), mid + 1, lo)
    lo_k = lo
    lo = np.zeros(m, dtype=np.int64)
    hi = np.full(m, n, dtype=np.int64)
    for _ in range(steps):  # upper bound: first k with (s[k] - q) > r
        mid = (lo + hi) // 2
        ok = (mid < n) & ((s[np.minimum(mid, n - 1)] - q) > r)
        hi = np.where(ok & (lo < hi), mid, hi)
        lo = np.where((~ok) & (lo < hi), mid + 1, lo)
    return (lo - lo_k).astype(np.int64)


def _templates(x: np.ndarray, m: int, nt: int) -> np.ndarray:
    return np.column_stack([x[i : i + nt] for i in range(m)])


def _pair_count_md(pts: np.ndarray, r: float) -> int:
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    n = len(pts)
    if n < 2:
        return 0
    tree = cKDTree(pts)
    return (int(tree.count_neighbors(tree, r, p=np.inf)) - n) // 2


def _cross_count_md(u_pts: np.ndarray, v_pts: np.ndarray, r: float) -> int:
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    if len(u_pts) == 0 or len(v_pts) == 0:
        return 0
    return int(cKDTree(u_pts).count_neighbors(cKDTree(v_pts), r, p=np.inf))


def sampen_over_runs(runs: list[np.ndarray], m: int = 1, r_sd: float = 0.2) -> tuple[float, int, int]:
    """Sample entropy over a series split into contiguous RUNS -- every template lies within one run, never spanning
    a gap (review A-25), while pairs are counted over the union of all runs' templates. ``r = r_sd * std`` of the
    concatenated runs (one global scale). ``A == 0`` or ``B == 0`` -> NaN. A single run reproduces :func:`sampen`.
    """
    arrs = [np.asarray(x, dtype=np.float64) for x in runs]
    concat = np.concatenate(arrs) if arrs else np.empty(0)
    if concat.size == 0:
        return (float("nan"), 0, 0)
    r = float(r_sd * np.std(concat, ddof=0))
    if m == 1:
        firsts = [x[:-1] for x in arrs if x.size >= 2]
        seconds = [x[1:] for x in arrs if x.size >= 2]
        if not firsts:
            return (float("nan"), 0, 0)
        first, second = np.concatenate(firsts), np.concatenate(seconds)
        b_count = _count_1d_pairs_within(first, r)
        a_count = dominance_count_self(first, second, r)
    else:
        mt = [_templates(x, m, x.size - m) for x in arrs if x.size - m >= 1]
        m1t = [_templates(x, m + 1, x.size - m) for x in arrs if x.size - m >= 1]
        if not mt:
            return (float("nan"), 0, 0)
        b_count = _pair_count_md(np.vstack(mt), r)
        a_count = _pair_count_md(np.vstack(m1t), r)
    if a_count == 0 or b_count == 0:
        return (float("nan"), a_count, b_count)
    return (float(-np.log(a_count / b_count)), a_count, b_count)


def sampen(x: npt.ArrayLike, m: int = 1, r_sd: float = 0.2) -> tuple[float, int, int]:
    """Sample entropy ``(SampEn, A, B)``; ``r = r_sd * std(x, ddof=0)``. ``A == 0`` or ``B == 0`` -> NaN.

    One contiguous run; :func:`sampen_over_runs` is the gap-aware form (byte-identical here)."""
    return sampen_over_runs([np.asarray(x, dtype=np.float64)], m, r_sd)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = np.std(x, ddof=0)
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def _split_like(flat: np.ndarray, arrs: list[np.ndarray]) -> list[np.ndarray]:
    """Split ``flat`` back into pieces of the lengths of ``arrs`` (used to re-run a globally z-scored series)."""
    return list(np.split(flat, np.cumsum([x.size for x in arrs])[:-1])) if arrs else []


def cross_sampen_over_runs(
    u_runs: list[np.ndarray], v_runs: list[np.ndarray], m: int = 1, r: float = 0.2
) -> tuple[float, int, int]:
    """Cross sample entropy over two series split into contiguous RUNS -- templates lie within one run, never
    spanning a gap (review A-25); cross pairs are counted over the union. Both series are z-scored GLOBALLY (over
    their concatenation, ddof=0) before splitting back into runs. A single run each reproduces :func:`cross_sampen`.
    """
    u_arrs = [np.asarray(x, dtype=np.float64) for x in u_runs]
    v_arrs = [np.asarray(x, dtype=np.float64) for x in v_runs]
    u_all = np.concatenate(u_arrs) if u_arrs else np.empty(0)
    v_all = np.concatenate(v_arrs) if v_arrs else np.empty(0)
    if u_all.size == 0 or v_all.size == 0:
        return (float("nan"), 0, 0)
    uz, vz = _split_like(_zscore(u_all), u_arrs), _split_like(_zscore(v_all), v_arrs)
    if m == 1:
        uf = [x[:-1] for x in uz if x.size >= 2]
        us = [x[1:] for x in uz if x.size >= 2]
        vf = [x[:-1] for x in vz if x.size >= 2]
        vs = [x[1:] for x in vz if x.size >= 2]
        if not uf or not vf:
            return (float("nan"), 0, 0)
        u1, u2, v1, v2 = (np.concatenate(a) for a in (uf, us, vf, vs))
        b_count = int(_count_within_sorted(u1, np.sort(v1), r).sum())
        a_count = dominance_count_cross(u1, u2, v1, v2, r)
    else:
        u_mt = [_templates(x, m, x.size - m) for x in uz if x.size - m >= 1]
        v_mt = [_templates(x, m, x.size - m) for x in vz if x.size - m >= 1]
        u_m1 = [_templates(x, m + 1, x.size - m) for x in uz if x.size - m >= 1]
        v_m1 = [_templates(x, m + 1, x.size - m) for x in vz if x.size - m >= 1]
        if not u_mt or not v_mt:
            return (float("nan"), 0, 0)
        b_count = _cross_count_md(np.vstack(u_mt), np.vstack(v_mt), r)
        a_count = _cross_count_md(np.vstack(u_m1), np.vstack(v_m1), r)
    if a_count == 0 or b_count == 0:
        return (float("nan"), a_count, b_count)
    return (float(-np.log(a_count / b_count)), a_count, b_count)


def cross_sampen(u: npt.ArrayLike, v: npt.ArrayLike, m: int = 1, r: float = 0.2) -> tuple[float, int, int]:
    """Cross sample entropy ``(value, A, B)``; both series z-scored (ddof=0). Direction-independent.

    One contiguous run each; :func:`cross_sampen_over_runs` is the gap-aware form (byte-identical here)."""
    return cross_sampen_over_runs([np.asarray(u, dtype=np.float64)], [np.asarray(v, dtype=np.float64)], m, r)
