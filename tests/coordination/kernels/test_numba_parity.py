"""TF-58 Task 12: numba pair-counter parity with the naive predicate and the cKDTree reference."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from silly_kicks.coordination._kernels._numba import (
    dominance_count_cross,
    dominance_count_self,
    near_in_phase_counts_at_nb,
    pairs_within_sorted,
    use_numba,
    xcorr_edge_correction_nb,
)
from silly_kicks.coordination._kernels._surrogates import (
    _near_in_phase_counts_at_np,
    _xcorr_edge_correction_np,
)


def _naive_self(a: np.ndarray, b: np.ndarray, r: float) -> int:
    n = len(a)
    return sum(1 for i in range(n) for j in range(i + 1, n) if abs(a[i] - a[j]) <= r and abs(b[i] - b[j]) <= r)


def _naive_cross(a1, b1, a2, b2, r: float) -> int:
    return sum(1 for i in range(len(a1)) for j in range(len(a2)) if abs(a1[i] - a2[j]) <= r and abs(b1[i] - b2[j]) <= r)


def _kdtree_self(a: np.ndarray, b: np.ndarray, r: float) -> int:
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    pts = np.column_stack([a, b])
    t = cKDTree(pts)
    return (int(t.count_neighbors(t, r, p=np.inf)) - len(pts)) // 2


def _continuous():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, 300), rng.normal(0, 1, 300)


def _tie_heavy():
    rng = np.random.default_rng(1)
    return rng.integers(0, 8, 300).astype(np.float64), rng.integers(0, 8, 300).astype(np.float64)


def test_dominance_self_matches_naive_continuous_and_ties():
    a, b = _continuous()
    assert dominance_count_self(a, b, 0.3) == _naive_self(a, b, 0.3)
    a, b = _tie_heavy()
    assert dominance_count_self(a, b, 2.0) == _naive_self(a, b, 2.0)


def test_dominance_self_matches_kdtree_continuous():
    a, b = _continuous()
    assert dominance_count_self(a, b, 0.3) == _kdtree_self(a, b, 0.3)


def test_dominance_cross_matches_naive_continuous_and_ties():
    rng = np.random.default_rng(5)
    a1, b1 = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    a2, b2 = rng.normal(0, 1, 150), rng.normal(0, 1, 150)
    assert dominance_count_cross(a1, b1, a2, b2, 0.3) == _naive_cross(a1, b1, a2, b2, 0.3)
    a1 = rng.integers(0, 8, 200).astype(np.float64)
    b1 = rng.integers(0, 8, 200).astype(np.float64)
    a2 = rng.integers(0, 8, 150).astype(np.float64)
    b2 = rng.integers(0, 8, 150).astype(np.float64)
    assert dominance_count_cross(a1, b1, a2, b2, 2.0) == _naive_cross(a1, b1, a2, b2, 2.0)


def test_dominance_cross_matches_kdtree_continuous():
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]  # runtime-valid; absent from stubs

    rng = np.random.default_rng(6)
    a1, b1 = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    a2, b2 = rng.normal(0, 1, 150), rng.normal(0, 1, 150)
    ref = int(cKDTree(np.column_stack([a1, b1])).count_neighbors(cKDTree(np.column_stack([a2, b2])), 0.3, p=np.inf))
    assert dominance_count_cross(a1, b1, a2, b2, 0.3) == ref


def test_force_numpy_env_disables_numba(monkeypatch):
    monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    assert use_numba() is False
    a, b = _continuous()
    # the cKDTree fallback path still matches the naive predicate on continuous data
    assert dominance_count_self(a, b, 0.3) == _naive_self(a, b, 0.3)


def _naive_pairs_1d(v: np.ndarray, r: float) -> int:
    s = np.sort(v)
    return sum(1 for p in range(len(s)) for q in range(p + 1, len(s)) if s[q] - s[p] <= r)


@pytest.mark.parametrize("force_numpy", [False, True])
def test_pairs_within_sorted_matches_the_naive_predicate_at_ties(monkeypatch, force_numpy):
    # Sample entropy's 1-D template count (m = 1): the numba two-pointer sweep and the numpy predicate bisection both
    # count #{p < q : s[q] - s[p] <= r} with the naive predicate -- identical integers, ties at exactly r included.
    if force_numpy:
        monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    assert use_numba() is not force_numpy
    rng = np.random.default_rng(12)
    cases = [
        (rng.normal(0, 1, 300), 0.3),
        (rng.integers(0, 8, 300).astype(np.float64), 2.0),  # tie-heavy: many pairs exactly at r
        (rng.integers(0, 8, 300).astype(np.float64), 0.0),  # r = 0: equal values only
        (np.array([0.1, 0.2, 0.30000000000000004, 0.4]), 0.1),  # float differences an ulp either side of r
        (np.array([1.0, np.nan, 2.0, 3.0]), 1.0),  # a NaN never counts
        (np.array([]), 1.0),
        (np.array([5.0]), 1.0),
    ]
    for v, r in cases:
        assert pairs_within_sorted(np.sort(v), r) == _naive_pairs_1d(v, r), (v[:5], r)


def _parts(z):
    return np.ascontiguousarray(np.real(z)), np.ascontiguousarray(np.imag(z))


def test_near_in_phase_counts_at_nb_matches_numpy_at_exact_ties():
    # Thresholds sit EXACTLY on computed real parts AND one ulp above them. Both backends must form
    # ``a.re * b.re + a.im * b.im`` with two rounded products and one rounded sum; a backend that rounds a real part
    # even one ulp differently (an FMA, the complex multiply, a reordered sum) flips ``>=`` at one of the two
    # thresholds -- a value rounded low drops out at the exact tie, one rounded high sneaks in one ulp above it.
    rng = np.random.default_rng(8)
    n = 400
    za = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    zb = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    t_loc = np.sort(rng.choice(n, size=300, replace=False)).astype(np.int64)
    a = za[t_loc]
    shifts = rng.integers(0, n, 12).astype(np.int64)
    a_re, a_im = _parts(a)
    b_re, b_im = _parts(zb)
    j = (t_loc - shifts[0]) % n
    ties = a_re * b_re[j] + a_im * b_im[j]
    for tie in ties[:150]:
        for thr in (float(tie), float(np.nextafter(tie, np.inf))):
            got = near_in_phase_counts_at_nb(a_re, a_im, t_loc, b_re, b_im, shifts, thr)
            ref = _near_in_phase_counts_at_np(a_re, a_im, t_loc, b_re, b_im, shifts, thr)
            np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(("n", "offset", "m"), [(256, 0, 256), (500, 120, 300), (500, 0, 330), (500, 170, 330)])
def test_xcorr_edge_correction_nb_matches_numpy_exactly(n, offset, m):
    rng = np.random.default_rng(9 + offset)
    lag = 20
    a = rng.normal(0, 1, m)
    b = rng.normal(0, 1, n)
    shifts = rng.integers(0, n, 10).astype(np.int64)
    got = xcorr_edge_correction_nb(np.ascontiguousarray(a), np.ascontiguousarray(b), shifts, offset, lag)
    ref = _xcorr_edge_correction_np(a, b, shifts, offset, lag)
    assert got.view(np.int64).tolist() == ref.view(np.int64).tolist()  # same ascending add order -> bit-identical
    assert np.abs(got).max() > 0.0  # non-vacuous
