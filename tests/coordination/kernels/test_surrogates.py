"""TF-58 Task 13: surrogate machinery (time shifts, IAAFT, accelerated statistics)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.coordination._kernels import _surrogates
from silly_kicks.coordination._kernels._phase import analytic_phase, pad_length, phasor
from silly_kicks.coordination._kernels._surrogates import (
    draw_shifts,
    iaaft,
    key_words,
    near_in_phase_counts_at,
    percentile_rank,
    shift_bounds,
    shifted_lagged_pearson,
    shifted_near_in_phase_counts,
    shifted_phasor_sums,
    shifted_slice_lagged_pearson,
    shifted_window,
    surrogate_rng,
)
from silly_kicks.coordination._kernels._xcorr import lagged_pearson


def test_shifted_window_equals_roll_then_slice_bit_for_bit():
    # The gather must reproduce "copy the full series, np.roll every segment, slice the window" exactly --
    # including NaNs and rows outside every segment (which keep their value) -- for any window placement.
    rng = np.random.default_rng(5)
    for _ in range(200):
        n_rows = int(rng.integers(20, 300))
        x = rng.normal(size=n_rows)
        x[rng.random(n_rows) < 0.1] = np.nan
        cuts = np.sort(rng.choice(np.arange(1, n_rows), size=4, replace=False))
        segments = [(int(cuts[0]), int(cuts[1])), (int(cuts[2]), int(cuts[3]))]
        n_draws = int(rng.integers(1, 9))
        shifts = [rng.integers(0, hi - lo, size=n_draws) for lo, hi in segments]
        start = int(rng.integers(0, n_rows))
        end = int(rng.integers(start, n_rows + 1))
        got = shifted_window(x, segments, shifts, start, end, n_draws)
        assert got.shape == (n_draws, end - start)
        for d in range(n_draws):
            full = x.copy()
            for (lo, hi), sh in zip(segments, shifts, strict=True):
                full[lo:hi] = np.roll(x[lo:hi], int(sh[d]))
            want = full[start:end]
            same = (got[d].view(np.int64) == want.view(np.int64)) | (np.isnan(got[d]) & np.isnan(want))
            assert same.all()


def test_shifted_window_without_segments_repeats_the_window():
    x = np.arange(10.0)
    got = shifted_window(x, [], [], 2, 6, 3)
    np.testing.assert_array_equal(got, np.tile(x[2:6], (3, 1)))


@pytest.mark.parametrize(
    ("shifts", "match"),
    [
        ([np.zeros(4, dtype=np.int64)], "shape"),
        ([np.full(3, -1, dtype=np.int64)], r"\[0, 8\)"),
        ([np.full(3, 8, dtype=np.int64)], r"\[0, 8\)"),
        ([], "pair up"),
    ],
)
def test_shifted_window_rejects_malformed_shifts(shifts, match):
    with pytest.raises(ValueError, match=match):
        shifted_window(np.arange(10.0), [(1, 9)], shifts, 0, 10, 3)


def test_time_shift_preserves_autocorrelation_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 512)

    def circ_autocorr(v):
        f = np.fft.rfft(v - v.mean())
        return np.fft.irfft(np.abs(f) ** 2, n=v.size)

    np.testing.assert_allclose(circ_autocorr(np.roll(x, 37)), circ_autocorr(x), atol=1e-12)


def test_shift_bounds_both_sides():
    tau = 25
    assert shift_bounds(2 * tau + 1, tau) == (tau, tau + 1)
    assert shift_bounds(2 * tau, tau) is None


def test_draws_within_bounds():
    rng = np.random.default_rng(1)
    n, tau = 1000, 100
    shifts = draw_shifts(rng, n, tau, 500)
    assert shifts is not None
    assert shifts.min() >= tau
    assert shifts.max() < n - tau
    assert draw_shifts(rng, 10, 100, 5) is None


def test_seed_independent_of_processing_order():
    k1 = (7, "centroid_x")
    k2 = (9, "centroid_y")
    forward = [draw_shifts(surrogate_rng(2026, k), 1000, 50, 8) for k in (k1, k2)]
    backward = {k: draw_shifts(surrogate_rng(2026, k), 1000, 50, 8) for k in (k2, k1)}
    np.testing.assert_array_equal(forward[0], backward[k1])
    np.testing.assert_array_equal(forward[1], backward[k2])


def test_key_words_stable_across_id_dtypes():
    assert key_words((1, 2, "centroid_x")) == key_words(("1", 2, "centroid_x"))


def _phasor_R(za, zb, shifts=None):
    if shifts is None:
        rel = za * np.conj(zb)
        return abs(rel.sum()) / za.size
    return np.abs(shifted_phasor_sums(za, zb, shifts)) / za.size


def test_coupled_pair_percentile_separates_from_uncoupled():
    fs, f = 10.0, 0.02
    t = np.arange(0.0, 600.0, 1.0 / fs)
    pad = pad_length(len(t), fs, 0.22)
    za = phasor(analytic_phase(np.sin(2 * np.pi * f * t), pad))
    zb = phasor(analytic_phase(np.sin(2 * np.pi * f * t - np.radians(40.0)), pad))
    rng = surrogate_rng(1, ("coupled",))
    shifts = draw_shifts(rng, za.size, int(0.1 * za.size), 200)
    assert shifts is not None
    obs = float(_phasor_R(za, zb))
    surr = _phasor_R(za, zb, shifts)
    assert percentile_rank(obs, surr) >= 0.99

    # independent AR(1) processes -> the observed R is unremarkable against its own shift surrogates.
    rng2 = np.random.default_rng(3)

    def ar1(n):
        e = rng2.normal(0, 1, n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.9 * y[i - 1] + e[i]
        return y

    za2 = phasor(analytic_phase(ar1(t.size), pad))
    zb2 = phasor(analytic_phase(ar1(t.size), pad))
    obs2 = float(_phasor_R(za2, zb2))
    surr2 = _phasor_R(za2, zb2, shifts)
    p2 = percentile_rank(obs2, surr2)
    assert 0.05 <= p2 <= 0.95


def test_accelerated_R_equals_direct():
    rng = np.random.default_rng(4)
    n = 512
    za = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    zb = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    shifts = rng.integers(1, n - 1, 20)
    whole = shifted_phasor_sums(za, zb, shifts)
    direct = np.array([(za * np.conj(np.roll(zb, s))).sum() for s in shifts])
    np.testing.assert_allclose(whole, direct, atol=1e-9)
    sub = shifted_phasor_sums(za, zb, shifts, start=100, end=400)
    direct_sub = np.array([(za[100:400] * np.conj(np.roll(zb, s)[100:400])).sum() for s in shifts])
    np.testing.assert_allclose(sub, direct_sub, atol=1e-9)


def test_accelerated_near_in_phase_equals_direct():
    rng = np.random.default_rng(5)
    n = 400
    za = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    zb = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    shifts = rng.integers(1, n - 1, 15)
    cos_thr = float(np.cos(np.radians(30.0)))
    got = shifted_near_in_phase_counts(za, zb, shifts, 0, n, cos_thr)
    direct = np.array([int((np.real(za * np.conj(np.roll(zb, s))) >= cos_thr).sum()) for s in shifts])
    np.testing.assert_array_equal(got, direct)


def test_accelerated_xcorr_equals_direct():
    rng = np.random.default_rng(6)
    n, lag = 400, 20
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    shifts = rng.integers(1, n - 1, 12)
    r, ns = shifted_lagged_pearson(a, b, shifts, lag)
    for ki, s in enumerate(shifts):
        rr, nn = lagged_pearson(a, np.roll(b, s), lag)
        np.testing.assert_allclose(r[ki], rr, atol=1e-9, equal_nan=True)
        np.testing.assert_array_equal(ns, nn)


def test_percentile_formula_with_ties():
    surr = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0])  # 4 below, 3 equal to obs=5
    assert percentile_rank(5.0, surr) == (4 + 1.5) / 10


def test_iaaft_preserves_amplitude_distribution_and_spectrum():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 512)
    s, converged = iaaft(x, rng, 200)
    np.testing.assert_array_equal(np.sort(s), np.sort(x))  # amplitude distribution preserved exactly
    amp_x = np.abs(np.fft.rfft(x))
    amp_s = np.abs(np.fft.rfft(s))
    rel = np.mean(np.abs(amp_s - amp_x) / (amp_x + 1e-9))
    assert rel < 0.05
    assert isinstance(converged, bool)


def test_fft_and_phasor_calls_constant_in_k(monkeypatch):
    from tests._perf_structural import call_counter

    def count_for(k: int) -> int:
        counters = [call_counter(monkeypatch, _surrogates._fft, name) for name in ("fft", "ifft", "rfft", "irfft")]
        rng = np.random.default_rng(0)
        za = np.exp(1j * rng.uniform(-np.pi, np.pi, 512))
        zb = np.exp(1j * rng.uniform(-np.pi, np.pi, 512))
        shifts = rng.integers(1, 500, k)
        shifted_phasor_sums(za, zb, shifts)  # whole-segment -> one FFT pair regardless of k
        return sum(c["n"] for c in counters)

    assert count_for(19) == count_for(199)


# --------------------------------------------------------------------------- D2(b): the spec 7.9 identities, wired
# The relative-phase and cross-correlation nulls score every time-shift draw through the spec's algebraic identities.
# Parity is asserted against the DIRECT computation AND the measured max deviation is reported (owner's rule: the
# number, not only the bound, goes on record).
def _direct_near_counts_at(a, t_loc, b, shifts, cos_thr):
    """The count, per shift, by the reference arithmetic ``Re(a * conj(b')) = a.re * b'.re + a.im * b'.im``."""
    out = []
    for s in shifts:
        bj = b[(t_loc - s) % b.size]
        re = a.real * bj.real + a.imag * bj.imag
        out.append(int(np.count_nonzero(re >= cos_thr)))
    return np.array(out, dtype=np.int64)


@pytest.mark.parametrize("force_numpy", [False, True])
def test_near_in_phase_counts_at_equals_the_direct_count(monkeypatch, force_numpy):
    if force_numpy:
        monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    rng = np.random.default_rng(12)
    n = 700
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    t_loc = np.sort(rng.choice(n, size=431, replace=False)).astype(np.int64)  # idx rows with gaps
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, t_loc.size))
    shifts = rng.integers(0, n, 37)
    cos_thr = float(np.cos(np.radians(30.0)))
    got = near_in_phase_counts_at(a, t_loc, b, shifts, cos_thr)
    np.testing.assert_array_equal(got, _direct_near_counts_at(a, t_loc, b, shifts, cos_thr))
    assert got.max() > 0 and got.min() < t_loc.size  # non-vacuous: counts are neither all zero nor all full


def test_near_in_phase_counts_at_ignores_non_finite_products():
    # A NaN phasor never counts (``NaN >= thr`` is False), exactly as the direct ``np.real(z) >= thr`` mask.
    b = np.exp(1j * np.linspace(0.0, 1.0, 10))
    a = np.full(3, complex(np.nan, np.nan))
    got = near_in_phase_counts_at(a, np.array([0, 4, 9]), b, np.array([0, 3]), 0.5)
    np.testing.assert_array_equal(got, [0, 0])


@pytest.mark.parametrize(
    ("t_loc", "shifts", "match"),
    [
        (np.array([0, 10]), np.array([0]), r"t_loc"),  # position == n
        (np.array([-1, 3]), np.array([0]), r"t_loc"),
        (np.array([0, 3]), np.array([10]), r"shifts"),  # shift == n
        (np.array([0, 3]), np.array([-1]), r"shifts"),
    ],
)
def test_near_in_phase_counts_at_rejects_out_of_range_indices(t_loc, shifts, match):
    b = np.exp(1j * np.zeros(10))
    with pytest.raises(ValueError, match=match):
        near_in_phase_counts_at(np.ones(t_loc.size, dtype=complex), t_loc, b, shifts, 0.5)


def test_masked_phasor_sums_equal_the_direct_sums_over_idx_rows():
    # R's identity on a segment whose idx rows have gaps: zero-padding A outside idx turns the one-FFT-pair circular
    # cross-correlation into the sum over idx rows only.
    rng = np.random.default_rng(14)
    n = 900
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    t_loc = np.sort(rng.choice(n, size=600, replace=False))
    a = np.exp(1j * rng.uniform(-np.pi, np.pi, t_loc.size))
    shifts = rng.integers(0, n, 50)
    a_full = np.zeros(n, dtype=np.complex128)
    a_full[t_loc] = a
    got = np.abs(shifted_phasor_sums(a_full, b, shifts)) / t_loc.size
    want = np.array([abs((a * np.conj(b[(t_loc - s) % n])).sum()) / t_loc.size for s in shifts])
    dev = float(np.max(np.abs(got - want)))
    print(f"masked-FFT R vs direct: max |dR| = {dev:.3e}")
    assert dev <= 1e-12


def _roll_reference(a, b, offset, shifts, lag):
    """``lagged_pearson`` of the slice against each rolled segment -- the direct computation the identity replaces."""
    m = a.size
    return [lagged_pearson(a, np.roll(b, int(s))[offset : offset + m], lag) for s in shifts]


@pytest.mark.parametrize(("n", "offset", "m", "lag"), [(400, 0, 400, 20), (900, 130, 520, 60), (1300, 0, 700, 150)])
@pytest.mark.parametrize("force_numpy", [False, True])
def test_shifted_slice_lagged_pearson_equals_the_direct_computation(monkeypatch, n, offset, m, lag, force_numpy):
    # Whole-segment (offset 0, m == n) and sub-segment slices: the circular FFT cross-correlation over the segment,
    # the exact correction of the |l| terms that fall outside the slice, and circular prefix sums for B's overlap.
    if force_numpy:
        monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    rng = np.random.default_rng(n + offset + m + lag)
    t = np.arange(n) / 10.0
    b = 30.0 + 8.0 * np.sin(2 * np.pi * 0.03 * t) + np.cumsum(rng.normal(0.0, 0.2, n))  # pitch-like, drifting mean
    a = 50.0 + 6.0 * np.sin(2 * np.pi * 0.03 * t[offset : offset + m] + 0.4) + rng.normal(0.0, 0.5, m)
    shifts = rng.integers(0, n, 23)
    r, ns = shifted_slice_lagged_pearson(a, b, offset, shifts, lag)
    ref = _roll_reference(a, b, offset, shifts, lag)
    dev = max(float(np.nanmax(np.abs(r[k] - rr))) for k, (rr, _nn) in enumerate(ref))
    print(f"slice identity vs direct lagged_pearson (n={n}, offset={offset}, m={m}, L={lag}): max |dr| = {dev:.3e}")
    for k, (rr, nn) in enumerate(ref):
        np.testing.assert_array_equal(np.isnan(r[k]), np.isnan(rr))
        np.testing.assert_array_equal(ns, nn)
    assert dev <= 1e-9
    assert np.nanmax(np.abs(r)) > 0.2  # non-vacuous: real correlation structure, not noise around zero


def test_shifted_slice_lagged_pearson_is_numba_independent(monkeypatch):
    # Spec 7.15: results never depend on whether numba is installed -- the edge correction sums in the same ascending
    # order in both paths, so the two backends agree bit for bit.
    pytest.importorskip("numba")
    rng = np.random.default_rng(21)
    b = rng.normal(0.0, 1.0, 800)
    a = rng.normal(0.0, 1.0, 500)
    shifts = rng.integers(0, 800, 17)
    r_nb, _ = shifted_slice_lagged_pearson(a, b, 150, shifts, 90)
    monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    r_np, _ = shifted_slice_lagged_pearson(a, b, 150, shifts, 90)
    assert r_nb.view(np.int64).tolist() == r_np.view(np.int64).tolist()


@pytest.mark.parametrize(
    ("a_len", "offset", "lag", "match"),
    [
        (500, 350, 10, r"slice"),  # slice runs past the segment end
        (500, -1, 10, r"slice"),
        (40, 0, 40, r"max_lag"),  # lag must be < slice length
        (float("nan"), 0, 10, r"finite"),  # sentinel: a non-finite A value
    ],
)
def test_shifted_slice_lagged_pearson_validates_inputs(a_len, offset, lag, match):
    b = np.arange(800, dtype=np.float64)
    a = np.array([1.0, np.nan, 2.0] * 20) if isinstance(a_len, float) else np.linspace(0.0, 1.0, int(a_len))
    with pytest.raises(ValueError, match=match):
        shifted_slice_lagged_pearson(a, b, offset, np.array([0, 5]), lag)


# ADR-111 ruling C: a caller scoring many windows against one B segment prepares its B side once -- R's
# `conj(fft(B))`, cross-correlation's centred B, its rfft and prefix sums -- and passes it in. Same values, bit for bit.
def test_shifted_phasor_sums_with_a_prepared_b_spectrum_is_byte_identical():
    rng = np.random.default_rng(31)
    n = 700
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    a = np.zeros(n, dtype=np.complex128)
    rows = np.sort(rng.choice(n, size=400, replace=False))
    a[rows] = np.exp(1j * rng.uniform(-np.pi, np.pi, rows.size))
    shifts = rng.integers(0, n, 29)
    want = shifted_phasor_sums(a, b, shifts)
    got = shifted_phasor_sums(a, b, shifts, zb_spectrum=_surrogates.phasor_spectrum(b))
    assert got.view(np.int64).tolist() == want.view(np.int64).tolist()
    with pytest.raises(ValueError, match="zb_spectrum"):
        shifted_phasor_sums(a, b, shifts, zb_spectrum=_surrogates.phasor_spectrum(b[:-1]))


@pytest.mark.parametrize("force_numpy", [False, True])
def test_shifted_slice_lagged_pearson_with_a_prepared_b_side_is_byte_identical(monkeypatch, force_numpy):
    if force_numpy:
        monkeypatch.setenv("SILLY_KICKS_COORDINATION_FORCE_NUMPY", "1")
    rng = np.random.default_rng(32)
    b = 30.0 + np.cumsum(rng.normal(0.0, 0.3, 900))
    a = rng.normal(0.0, 1.0, 480)
    shifts = rng.integers(0, 900, 19)
    r_want, n_want = shifted_slice_lagged_pearson(a, b, 210, shifts, 60)
    r_got, n_got = shifted_slice_lagged_pearson(a, b, 210, shifts, 60, b_side=_surrogates.lagged_pearson_b_side(b))
    assert r_got.view(np.int64).tolist() == r_want.view(np.int64).tolist()
    np.testing.assert_array_equal(n_got, n_want)
    with pytest.raises(ValueError, match="b_side"):
        shifted_slice_lagged_pearson(a, b, 210, shifts, 60, b_side=_surrogates.lagged_pearson_b_side(b[:-1]))
    with pytest.raises(ValueError, match="finite"):
        _surrogates.lagged_pearson_b_side(np.array([1.0, np.nan, 2.0]))


def test_shifted_lagged_pearson_is_the_whole_segment_case():
    rng = np.random.default_rng(22)
    a, b = rng.normal(0.0, 1.0, 300), rng.normal(0.0, 1.0, 300)
    shifts = rng.integers(0, 300, 9)
    whole, n_whole = shifted_lagged_pearson(a, b, shifts, 25)
    sliced, n_sliced = shifted_slice_lagged_pearson(a, b, 0, shifts, 25)
    assert whole.view(np.int64).tolist() == sliced.view(np.int64).tolist()
    np.testing.assert_array_equal(n_whole, n_sliced)
