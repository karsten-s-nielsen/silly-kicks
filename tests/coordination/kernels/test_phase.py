"""TF-58 Task 8: analytic-phase kernels (and phase x circular integration)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import scipy.signal

from silly_kicks.coordination._kernels._circular import circular_summary, wrap_deg
from silly_kicks.coordination._kernels._phase import (
    analytic_phase,
    pad_length,
    phase_advance_indicator,
    phase_valid_fraction,
    phasor,
)

_FS = 10.0
_F = 0.02  # Hz, in-band for the 0.22 cpm low edge


def _relative_phase(a: np.ndarray, b: np.ndarray, pad: int):
    """Circular summary of the sample-wise relative phase arg(z_a . conj z_b)."""
    z = phasor(analytic_phase(a, pad)) * np.conj(phasor(analytic_phase(b, pad)))
    return circular_summary(z.sum(), float(len(z)))


def test_pad_length_rule():
    assert pad_length(10_000, 10.0, 0.22) == round(60 * 10 / 0.22)
    assert pad_length(100, 10.0, 0.22) == 100


def test_centring_makes_offset_irrelevant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300).cumsum()
    pad = pad_length(len(x), _FS, 0.22)
    np.testing.assert_allclose(analytic_phase(x + 50.0, pad), analytic_phase(x, pad), atol=1e-9)


def test_known_offset_sinusoids():
    # 3000 s (not the spec example's 1200 s): the reflect pad slightly perturbs an otherwise FFT-exact clean
    # sinusoid, so R clears 0.999 only with more samples (owner-approved, TF-58 Task 8). Offset recovery holds
    # at any length -- this asserts both the recovered offset AND high concentration.
    t = np.arange(0.0, 3000.0, 1.0 / _FS)
    a = np.sin(2 * np.pi * _F * t)
    b = np.sin(2 * np.pi * _F * t - np.radians(40.0))
    pad = pad_length(len(t), _FS, 0.22)
    mean_deg, r, _ = _relative_phase(a, b, pad)
    assert abs(float(wrap_deg(mean_deg - 40.0))) < 0.5
    assert float(r) > 0.999


def test_anti_phase_is_180():
    t = np.arange(0.0, 1200.0, 1.0 / _FS)
    a = np.sin(2 * np.pi * _F * t)
    b = np.sin(2 * np.pi * _F * t - np.pi)
    pad = pad_length(len(t), _FS, 0.22)
    mean_deg, r, _ = _relative_phase(a, b, pad)
    assert abs(abs(float(mean_deg)) - 180.0) < 0.5
    assert float(r) > 0.999


def test_noise_lowers_R_monotonically():
    t = np.arange(0.0, 600.0, 1.0 / _FS)
    pad = pad_length(len(t), _FS, 0.22)
    base = np.sin(2 * np.pi * _F * t)
    offset = np.sin(2 * np.pi * _F * t - np.radians(40.0))
    means_r = []
    for sigma in (0.0, 0.2, 0.5, 1.0):
        rng = np.random.default_rng(1000 + int(sigma * 100))
        rs = []
        for _ in range(20):
            b = offset + rng.normal(0.0, sigma, len(t))
            _, r, _ = _relative_phase(base, b, pad)
            rs.append(float(r))
        means_r.append(float(np.mean(rs)))
    assert all(means_r[i] > means_r[i + 1] for i in range(len(means_r) - 1)), means_r


def test_mean_centring_is_load_bearing():
    """The load-bearing pre-Hilbert step: mean-centring (Lamb & Stockl 2014).

    A DC offset must not bias the phase -- the phasor has to rotate about the origin. The kernel centres
    internally, so a large offset leaves the phase correct; a Hilbert WITHOUT the mean subtraction is badly
    biased (the phasor orbits the offset, barely rotating). This is the robust, non-vacuous benefit of the
    pre-processing -- more so than the reflect padding, whose edge benefit is regime-dependent.
    """
    fs, f = 10.0, 0.02
    t = np.arange(0.0, 300.0, 1.0 / fs)
    clean = np.sin(2 * np.pi * f * t)
    true = 2 * np.pi * f * t - np.pi / 2
    pad = pad_length(len(t), fs, 0.22)

    def rms_central(ph: np.ndarray) -> float:
        # away from the edges, where the offset bias (global) is isolated from the pad transient (edges).
        k = max(1, len(ph) // 10)
        s = slice(k, len(ph) - k)
        return float(np.sqrt(np.mean(np.angle(np.exp(1j * (ph[s] - true[s]))) ** 2)))

    ph_centred = analytic_phase(clean + 5.0, pad)  # kernel subtracts the mean -> offset-invariant
    xp = np.pad(clean + 5.0, pad, mode="reflect")
    ph_naive = np.angle(cast("np.ndarray", scipy.signal.hilbert(xp))[pad:-pad])  # same pad, NO centring
    assert rms_central(ph_centred) < 0.1  # centred phase tracks the true linear sweep
    assert rms_central(ph_naive) > 10.0 * rms_central(ph_centred)  # un-centred is globally biased


def test_reflect_padding_reduces_edge_error():
    """Non-vacuity of the reflect-padding branch, on a REPRESENTATIVE slow window.

    A ~1-minute coordination rhythm (0.015 Hz) over a possession-length 90 s window is 1.35 cycles -- the
    realistic non-integer-cycle regime where the Hilbert FFT-wrap transient bites and reflect padding tames
    it. (Integer-cycle sinusoids are artificially FFT-periodic, so pad=0 is exact there -- not
    representative.) The >=2x edge reduction is regime-specific, not universal (TF-58 Task 8, owner-approved);
    this asserts the branch runs and helps materially here.
    """
    fs, f = 10.0, 0.015
    t = np.arange(0.0, 90.0, 1.0 / fs)
    x = np.sin(2 * np.pi * f * t)
    true = 2 * np.pi * f * t - np.pi / 2  # analytic phase of sin(wt) is wt - 90 deg
    pad = pad_length(len(t), fs, 0.22)
    ph_pad = analytic_phase(x, pad)
    ph_none = analytic_phase(x, 0)

    def edge_rms(ph: np.ndarray) -> float:
        k = max(1, len(ph) // 10)
        idx = np.r_[0:k, len(ph) - k : len(ph)]
        d = np.angle(np.exp(1j * (ph[idx] - true[idx])))
        return float(np.sqrt(np.mean(d**2)))

    assert not np.allclose(ph_pad, ph_none)  # the padding branch actually changed the edge phase
    assert edge_rms(ph_none) > 1.5 * edge_rms(ph_pad), (edge_rms(ph_none), edge_rms(ph_pad))


def test_phase_valid_fraction():
    fs, f = 10.0, 0.05
    t = np.arange(0.0, 120.0, 1.0 / fs)
    pad = pad_length(len(t), fs, 0.22)
    ph = analytic_phase(np.sin(2 * np.pi * f * t), pad)
    assert phase_valid_fraction(ph) > 0.99

    rng = np.random.default_rng(3)
    walk = rng.normal(0.0, 1.0, 1200).cumsum()
    ph_walk = analytic_phase(walk, pad)
    assert phase_valid_fraction(ph_walk) < 0.9

    assert np.isnan(phase_valid_fraction(np.array([0.1])))


def test_phase_advance_indicator_excludes_the_first_sample():
    # A-45: the first sample has NO predecessor, so its advance is undefined (NaN) -- counting it as non-advancing
    # biased coord_rp_phase_valid_fraction down by n_runs/n. A monotone phase advances at every later sample.
    fs, f = 10.0, 0.05
    t = np.arange(0.0, 120.0, 1.0 / fs)
    ph = analytic_phase(np.sin(2 * np.pi * f * t), pad_length(len(t), fs, 0.22))
    ind = phase_advance_indicator(ph)
    assert ind.shape == ph.shape
    assert np.isnan(ind[0]) and np.isfinite(ind[1:]).all()
    assert set(np.unique(ind[1:]).tolist()) <= {0.0, 1.0}
    # the scalar is the nan-mean of the per-sample indicator (ONE implementation, no run-first bias)
    assert np.nanmean(ind) == pytest.approx(phase_valid_fraction(ph))


def test_phase_advance_indicator_short_input_is_all_nan():
    assert np.isnan(phase_advance_indicator(np.array([0.1]))).all()
    assert np.isnan(phase_advance_indicator(np.array([]))).all()


# --------------------------------------------------------------------------- scipy hilbert replica (ADR-111 ruling C)
# `analytic_phase` replicates `scipy.signal.hilbert` for 1-D input step for step (fft, the exact x2 / zero of the two
# spectrum halves, ifft) and numpy's `reflect` pad by slicing: the generic per-call overhead (array namespace,
# moveaxis, np.pad) dominated ~12k short SkillCorner runs per match. FENCE, so a scipy change to `hilbert` fails here
# loudly. scipy >= 1.17 scales the two halves in place (verified 1.17.1, 1.18.1) -- the form the replica follows, so
# there it is bit for bit. scipy <= 1.16 builds `Xf * h` with a COMPLEX h (verified 1.15.3, 1.16.0; CI's py3.10 leg):
# a complex multiply by `1 + 0j` can flip the sign of an exact zero, so there the analytic signals are equal in VALUE
# and may differ only in the sign of a zero (measured on 1.15.3: 9 of 33 battery cases, every difference at an exact
# zero). The replica thus makes the phase independent of which form the installed scipy uses.
_SCIPY_HILBERT_IN_PLACE = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (1, 17)


def _padded(x, pad):
    """The centred, reflect-padded series `analytic_phase` transforms, and its pad."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x)
    p = min(int(pad), len(xc) - 1)
    return (np.pad(xc, p, mode="reflect") if p > 0 else xc), p


def _scipy_analytic_phase(x, pad):
    xp, p = _padded(x, pad)
    analytic = cast("np.ndarray", scipy.signal.hilbert(xp))
    if p > 0:
        analytic = analytic[p:-p]
    return np.angle(analytic)


def test_analytic_phase_is_the_scipy_hilbert_path_bit_for_bit():
    from silly_kicks.coordination._kernels._phase import _hilbert_1d

    rng = np.random.default_rng(58)
    checked = 0
    for n in (1, 2, 3, 4, 13, 14, 101, 102, 997, 2048, 2049):
        x = 30.0 + 5.0 * np.sin(2 * np.pi * _F * np.arange(n) / _FS) + rng.normal(0.0, 0.3, n)
        for pad in (0, 1, n // 3, n - 1, n, 5 * n, 2727):
            if _SCIPY_HILBERT_IN_PLACE:
                got = analytic_phase(x, pad)
                want = _scipy_analytic_phase(x, pad)
                assert got.view(np.int64).tolist() == want.view(np.int64).tolist(), (n, pad)
            else:
                xp, _p = _padded(x, pad)
                assert np.array_equal(_hilbert_1d(xp), np.asarray(scipy.signal.hilbert(xp))), (n, pad)
            checked += 1
    assert checked == 77


def test_analytic_phase_skips_the_generic_scipy_wrappers(monkeypatch):
    # R8 structural guard: the replica is the hot path -- no scipy.signal.hilbert (nor np.pad) call per run.
    from tests._perf_structural import call_counter

    hilberts = call_counter(monkeypatch, scipy.signal, "hilbert")
    pads = call_counter(monkeypatch, np, "pad")
    analytic_phase(np.sin(np.arange(500) / 7.0), 300)
    assert hilberts["n"] == 0 and pads["n"] == 0
