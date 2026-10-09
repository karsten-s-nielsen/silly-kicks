"""TF-58 Task 9: lagged cross-correlation and Fisher-z pooling kernels."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.coordination._kernels._xcorr import (
    fisher_pool,
    lagged_pearson,
    lagged_pearson_batch,
    min_slice_samples,
    xcorr_summary,
)


def _same_bits(got: np.ndarray, want: np.ndarray) -> bool:
    got = np.ascontiguousarray(got, dtype=np.float64)
    want = np.ascontiguousarray(want, dtype=np.float64)
    return got.shape == want.shape and bool(
        ((got.view(np.int64) == want.view(np.int64)) | (np.isnan(got) & np.isnan(want))).all()
    )


@pytest.mark.parametrize(("n", "lag"), [(600, 150), (601, 150), (1000, 50), (257, 60), (4000, 150)])
def test_lagged_pearson_batch_is_bit_identical_per_row(n, lag):
    # The D2(a) surrogate batching: one call scores every shifted draw of series B against A.
    rng = np.random.default_rng(n + lag)
    a = rng.normal(50.0, 10.0, n)
    b = rng.normal(30.0, 5.0, (9, n)) + np.linspace(0.0, 3.0, n)
    b[3] = 7.0  # a constant row: the zero-variance (NaN) branch
    r_batch, ns_batch = lagged_pearson_batch(a, b, lag)
    assert r_batch.shape == (9, 2 * lag + 1)
    for d in range(b.shape[0]):
        r_row, ns_row = lagged_pearson(a, b[d], lag)
        assert _same_bits(r_batch[d], r_row), d
        np.testing.assert_array_equal(ns_batch, ns_row)
    assert np.isnan(r_batch[3]).all()


def test_lagged_pearson_batch_validates_like_the_row_kernel():
    with pytest.raises(ValueError, match="same length"):
        lagged_pearson_batch(np.zeros(10), np.zeros((2, 9)), 2)
    with pytest.raises(ValueError, match="max_lag"):
        lagged_pearson_batch(np.zeros(10), np.zeros((2, 10)), 10)


def test_fisher_pool_pools_each_leading_row_like_the_2d_call():
    rng = np.random.default_rng(5)
    r = rng.uniform(-0.95, 0.95, (7, 3, 11))
    r[2, 1, 4] = np.nan
    n = rng.integers(2, 60, (3, 11))
    pooled = fisher_pool(r, n)
    assert pooled.shape == (7, 11)
    for d in range(7):
        assert _same_bits(pooled[d], fisher_pool(r[d], n)), d


def test_matches_numpy_corrcoef_per_lag():
    rng = np.random.default_rng(0)
    n, lag = 900, 150
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    r, ns = lagged_pearson(a, b, lag)
    for i, ell in enumerate(range(-lag, lag + 1)):
        if ell >= 0:
            a_ov, b_ov = a[0 : n - ell], b[ell:n]
        else:
            a_ov, b_ov = a[-ell:n], b[0 : n + ell]
        assert ns[i] == len(a_ov)
        expected = np.corrcoef(a_ov, b_ov)[0, 1]
        assert abs(r[i] - expected) < 1e-9, (ell, r[i], expected)


def test_shifted_copy_positive_lag_when_a_leads():
    rng = np.random.default_rng(1)
    n, k, fs = 600, 12, 10.0
    base = rng.normal(0, 1, n + k)
    a = base[k : k + n]  # a[t] = base[t + k]
    b = base[0:n]  # b[t] = base[t]  -> a leads b by k
    r, _ = lagged_pearson(a, b, 50)
    max_abs_r, lag_s, r_at_max, _ = xcorr_summary(r, fs)
    assert lag_s == k / fs
    assert abs(r_at_max - 1.0) < 1e-12
    assert abs(max_abs_r - 1.0) < 1e-12


def test_inverted_b_negative_r():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 400)
    b = -a
    r, _ = lagged_pearson(a, b, 40)
    max_abs_r, lag_s, r_at_max, r_lag0 = xcorr_summary(r, 10.0)
    assert lag_s == 0.0
    assert abs(r_at_max + 1.0) < 1e-12
    assert abs(r_lag0 + 1.0) < 1e-12
    assert abs(max_abs_r - 1.0) < 1e-12


def test_fisher_single_slice_identity():
    rng = np.random.default_rng(3)
    r = rng.uniform(-0.9, 0.9, 21)
    n = np.full(21, 50, dtype=np.int64)
    pooled = fisher_pool(r[None, :], n[None, :])
    np.testing.assert_allclose(pooled, r, atol=1e-12)


def test_fisher_weights_by_n_minus_3():
    r = np.array([[0.5], [0.8]])
    n = np.array([[10], [28]])
    # hand-computed: weights n-3 = [7, 25], Fisher-z average, tanh back
    z = np.arctanh([0.5, 0.8])
    expected = np.tanh((7 * z[0] + 25 * z[1]) / 32)
    pooled = fisher_pool(r, n)
    assert pooled.shape == (1,)
    assert abs(float(pooled[0]) - float(expected)) < 1e-12


def test_fisher_low_n_slice_dropped():
    # a slice with n <= 3 contributes nothing; result equals the single surviving slice.
    r = np.array([[0.4], [0.9]])
    n = np.array([[100], [3]])  # second slice dropped (n not > 3)
    pooled = fisher_pool(r, n)
    assert abs(float(pooled[0]) - 0.4) < 1e-12


def test_tie_break_smallest_abs_lag_then_negative():
    # equal |r| at lags -3, +3, +5 -> pick -3 (smallest |lag|, then negative)
    lag = 5
    r = np.full(2 * lag + 1, 0.2)
    lags = np.arange(-lag, lag + 1)
    for ell, val in ((-3, 0.9), (3, 0.9), (5, 0.9)):
        r[np.where(lags == ell)[0][0]] = val
    _, lag_s, r_at_max, _ = xcorr_summary(r, 10.0)
    assert lag_s == -3 / 10.0
    assert r_at_max == 0.9


def test_min_slice_samples_is_four_L():
    assert min_slice_samples(150) == 600
    assert min_slice_samples(0) == 0


def test_constant_side_is_nan():
    a = np.linspace(0, 1, 200)
    b = np.full(200, 7.0)  # zero variance
    r, ns = lagged_pearson(a, b, 20)
    assert np.isnan(r).all()
    assert (ns == 200 - np.abs(np.arange(-20, 21))).all()
    # summary of an all-NaN profile -> four NaNs
    assert all(np.isnan(x) for x in xcorr_summary(r, 10.0))


def test_max_lag_too_large_raises():
    with pytest.raises(ValueError, match="must be <"):
        lagged_pearson(np.zeros(10), np.zeros(10), 10)
