"""TF-58 Task 11: spectral median frequency and pooled Welch coherence."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.coordination._kernels._spectral import (
    cross_spectra_batch,
    median_frequency_cpm,
    min_spectral_samples,
    pooled_coherence,
    pooled_median_frequency,
    welch_segment_count,
    welch_spectra,
)


def _same_bits(got, want) -> bool:
    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want)
    if np.iscomplexobj(got) or np.iscomplexobj(want):
        return _same_bits(got.real, want.real) and _same_bits(got.imag, want.imag)
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    return got.shape == want.shape and bool(
        ((got.view(np.int64) == want.view(np.int64)) | (np.isnan(got) & np.isnan(want))).all()
    )


@pytest.mark.parametrize(("n", "nperseg"), [(1200, 1200), (1799, 1200), (4400, 1200), (3001, 200)])
def test_cross_spectra_batch_is_bit_identical_per_row(n, nperseg):
    # The D2(a) coherence batching: A's auto-spectrum once, B's auto- and the cross-spectrum for every draw at once.
    rng = np.random.default_rng(n + nperseg)
    fs = 10.0
    a = 50.0 + 15.0 * np.sin(2 * np.pi * (0.5 / 60.0) * np.arange(n) / fs) + rng.normal(0.0, 0.3, n)
    b = 40.0 + rng.normal(0.0, 1.0, (6, n)).cumsum(axis=1) * 0.1
    batch = cross_spectra_batch(a, b, fs, nperseg)
    for d in range(b.shape[0]):
        row = welch_spectra(a, b[d], fs, nperseg)
        assert _same_bits(batch.f, row.f)
        assert _same_bits(batch.pxx, row.pxx)
        assert _same_bits(batch.pyy[d], row.pyy), d
        assert _same_bits(batch.pxy[d], row.pxy), d
        assert batch.k == row.k


@pytest.mark.parametrize("duration_s", [545.5, 3600.0])  # the default minimum slice, and a full half
@pytest.mark.parametrize("k", [1, 2, 30])
def test_pure_tone_median_frequency_is_the_tone(duration_s, k):
    # spec 9.1: a pure tone at f0 -> median frequency f0. The tone sits exactly on periodogram bin k, so ALL its power
    # is in that one bin; review A-06: placing a bin's cumulative power at the bin CENTRE returned f0 - half a bin
    # (0.33 cpm -> 0.275 on the 545.5-s slice) for every bin but the first, which had its own rule
    fs = 10.0
    n = round(duration_s * fs)
    t = np.arange(n) / fs
    f0_hz = k * fs / n
    bin_cpm = 60.0 * fs / n
    got = median_frequency_cpm(np.sin(2 * np.pi * f0_hz * t), fs)
    assert abs(got - 60.0 * f0_hz) < 1e-6 * bin_cpm


def test_median_frequency_spreads_each_bins_power_over_its_width():
    # The one interpolation rule (spec 7.8.4): bin k's power is spread uniformly over [f_k - df/2, f_k + df/2], so the
    # cumulative power is linear between bin EDGES. Powers 1 : 3 on bins 8 and 9: half the total (2) is reached a third
    # of the way into bin 9 -> f_8 + df/2 + df/3 (interpolating mid-bin cumulatives between bin centres gives
    # f_8 + 3df/4)
    fs, n = 10.0, 6000
    t = np.arange(n) / fs
    df = fs / n
    x = np.sin(2 * np.pi * 8 * df * t) + np.sqrt(3.0) * np.sin(2 * np.pi * 9 * df * t)
    assert median_frequency_cpm(x, fs) == pytest.approx(60.0 * (8 * df + df / 2 + df / 3), rel=1e-9)


def test_possession_square_wave_median_is_the_fundamental():
    # A-42: a possession series is a 0/1 square wave. Its fundamental carries 8/pi^2 ~= 81% of the AC power -- more than
    # half -- so the cumulative power reaches one half INSIDE the fundamental bin and the median frequency is the
    # fundamental f0 (the DC offset is removed, like test_positive_offset_does_not_move_median). An analytic oracle for
    # the possession spectral path (spec 9.1 "reference values on example series").
    fs, n = 10.0, 6000
    for period in (40, 50, 30):  # f0 = fs/period -> 15, 12, 20 cpm
        square = (np.arange(n) % period < period / 2).astype(float)
        assert median_frequency_cpm(square, fs) == pytest.approx(60.0 * fs / period, abs=0.05)  # < one 0.1-cpm bin


def test_positive_offset_does_not_move_median():
    fs = 10.0
    t = np.arange(0.0, 1200.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 0.02 * t)
    assert median_frequency_cpm(x + 30.0, fs) == median_frequency_cpm(x, fs)


def test_min_spectral_samples_is_two_periods():
    assert min_spectral_samples(10.0, 0.22) == int(np.ceil(2 * 60 * 10 / 0.22))


def test_pooled_median_is_duration_weighted():
    got = pooled_median_frequency([0.4, 0.8], [10.0, 30.0])
    assert got == (0.4 * 10 + 0.8 * 30) / 40


def test_pooled_median_drops_nan_and_zero_duration():
    got = pooled_median_frequency([0.4, np.nan, 0.9], [10.0, 100.0, 0.0])
    assert got == 0.4  # only the first run survives


def test_constant_series_is_nan():
    assert np.isnan(median_frequency_cpm(np.full(1000, 3.14), 10.0))


def test_segment_count_rule():
    assert welch_segment_count(2048, 256) == 15
    assert welch_segment_count(256, 256) == 1
    assert welch_segment_count(100, 256) == 0


@pytest.mark.parametrize(("n", "nperseg"), [(11, 5), (23, 7), (30, 9), (2049, 257)])
def test_segment_count_matches_scipy_welch_for_odd_nperseg(n, nperseg):
    # A-46: the step is nperseg - nperseg//2 (scipy's nperseg - noverlap), not nperseg//2 -- they differ for an odd
    # nperseg. Pin against scipy's actual segment count (the length of its per-segment axis).
    from scipy.signal import spectrogram

    rng = np.random.default_rng(n + nperseg)
    _f, _t, sxx = spectrogram(
        rng.normal(0, 1, n), fs=10.0, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant"
    )
    assert welch_segment_count(n, nperseg) == sxx.shape[-1]


def test_coherence_near_one_for_linearly_filtered_pair():
    rng = np.random.default_rng(0)
    fs, nperseg = 10.0, 256
    a = rng.normal(0, 1, 2048)
    b = np.convolve(a, np.ones(3) / 3.0, mode="same")  # 3-tap moving average: a linear filter of a
    spec = welch_spectra(a, b, fs, nperseg)
    # band 6-18 cpm (0.1-0.3 Hz) stays below the MA's spectral null at 1/3 Hz.
    band_mean, _peak, k = pooled_coherence([spec], 6.0, 18.0)
    assert k >= 8
    assert band_mean > 0.95


def test_coherence_near_inverse_k_for_independent_noise():
    fs, nperseg = 10.0, 256
    k = welch_segment_count(2048, nperseg)
    means = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        spec = welch_spectra(rng.normal(0, 1, 2048), rng.normal(0, 1, 2048), fs, nperseg)
        band_mean, _, _ = pooled_coherence([spec], 6.0, 180.0)
        means.append(band_mean)
    mean_coh = float(np.mean(means))
    assert 0.5 / k <= mean_coh <= 2.0 / k, (mean_coh, 1 / k)


def test_pools_spectra_not_coherences():
    """Non-vacuity: pooling SPECTRA differs from a segment-weighted mean of per-slice COHERENCES."""
    fs, nperseg = 10.0, 256
    rng = np.random.default_rng(7)
    # slice 1: perfectly coherent, unit power. slice 2: independent, 25x power.
    a1 = rng.normal(0, 1, 2048)
    s1 = welch_spectra(a1, a1.copy(), fs, nperseg)
    s2 = welch_spectra(rng.normal(0, 5, 2048), rng.normal(0, 5, 2048), fs, nperseg)

    pooled, _, _ = pooled_coherence([s1, s2], 6.0, 180.0)
    per_slice = [pooled_coherence([s], 6.0, 180.0)[0] for s in (s1, s2)]
    k_weighted = (s1.k * per_slice[0] + s2.k * per_slice[1]) / (s1.k + s2.k)
    assert abs(pooled - k_weighted) > 0.05, (pooled, k_weighted)
