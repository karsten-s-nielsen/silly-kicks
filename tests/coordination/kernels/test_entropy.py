"""TF-58 Task 12: sample entropy and cross sample entropy (Richman & Moorman 2000)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from silly_kicks.coordination._kernels._entropy import (
    _count_1d_pairs_within,
    cross_sampen,
    cross_sampen_over_runs,
    sampen,
    sampen_over_runs,
)


def _naive_1d(v: np.ndarray, r: float) -> int:
    n = len(v)
    return sum(1 for i in range(n) for j in range(i + 1, n) if abs(v[i] - v[j]) <= r)


def test_sampen_gaussian_white_noise_matches_published_analytic_value():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 20_000)
    value, _a, _b = sampen(x, m=1, r_sd=0.2)
    analytic = -np.log(erf(0.1))  # Richman & Moorman random-number case, r = 0.2 sigma -> 2.185
    assert abs(value - analytic) < 0.03


def test_sampen_over_runs_single_run_is_byte_identical_to_sampen():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 2000)
    assert sampen_over_runs([x], m=1, r_sd=0.2) == sampen(x, m=1, r_sd=0.2)
    u, v = rng.normal(0, 1, 1500), rng.normal(0, 1, 1500)
    assert cross_sampen_over_runs([u], [v], m=1, r=0.2) == cross_sampen(u, v, m=1, r=0.2)


def test_sampen_over_runs_templates_never_span_a_gap():
    # A-25: two runs chosen so collapsing them builds a (m+1)-template STRADDLING the gap that matches another
    # template -- a spurious A-count that within-run templates never create. The run form and the collapsed form
    # must therefore give different counts (non-vacuity: the gap-awareness actually bites here).
    runs = [np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])]  # collapsed gap-straddling 2-template (1,1)
    collapsed = np.concatenate(runs)
    _rv, run_a, run_b = sampen_over_runs(runs, m=1, r_sd=0.5)
    _cv, col_a, col_b = sampen(collapsed, m=1, r_sd=0.5)
    assert (run_a, run_b) != (col_a, col_b)  # the gap-spanning template changed the collapsed counts
    # cross form too: a straddling template must not inflate the cross count
    assert cross_sampen_over_runs(runs, runs, m=1, r=0.5)[1:] != cross_sampen(collapsed, collapsed, m=1, r=0.5)[1:]


def test_sampen_hand_counted_small_case_pins_the_N_minus_m_convention():
    # A-42: a 4-sample series counted BY HAND, m=1. The length-1 templates are the first N-1=3 samples [1,1,1]; the
    # length-2 templates pair each with its successor -> (1,1),(1,1),(1,5). r = 0.2*std = 0.2*sqrt(3) = 0.3464.
    # B = unordered length-1 matches among [1,1,1] = C(3,2) = 3; A = those that also match at length 2 = the two equal
    # (1,1) templates = 1. SampEn = -ln(1/3) = ln 3. This pins the N-m template count (both lengths use N-1 = 3).
    value, a, b = sampen(np.array([1.0, 1.0, 1.0, 5.0]), m=1, r_sd=0.2)
    assert (a, b) == (1, 3)
    assert value == pytest.approx(np.log(3.0))


def test_cross_sampen_hand_counted_small_case():
    # A-42: two 4-sample series already at mean 0 / std 1 (ddof=0), so the global z-score is the identity; r = 0.2.
    # Cross counts are ORDERED (every u-template vs every v-template). u1=[-1,1,-1], v1=[1,1,-1]: length-1 matches
    # (equal within r) number B=4. Extending to length 2 (u2=[1,-1,1], v2=[1,-1,-1]) only (i=1,j=1) survives: A=1.
    # value = -ln(1/4) = ln 4.
    u = np.array([-1.0, 1.0, -1.0, 1.0])
    v = np.array([1.0, 1.0, -1.0, -1.0])
    value, a, b = cross_sampen(u, v, m=1, r=0.2)
    assert (a, b) == (1, 4)
    assert value == pytest.approx(np.log(4.0))


def test_sampen_periodic_series_is_low():
    t = np.linspace(0.0, 40 * np.pi, 4000)
    value, _a, _b = sampen(np.sin(t), m=1, r_sd=0.2)
    assert value < 0.5


def test_entropy_undefined_reachable():
    # two matching values (B > 0) but no matching length-2 template (A == 0) -> undefined.
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    value, a, b = sampen(x, m=1, r_sd=0.2)
    assert np.isnan(value)
    assert a == 0
    assert b > 0


def test_cross_sampen_direction_independent():
    rng = np.random.default_rng(1)
    u = rng.normal(0.0, 1.0, 300)
    v = rng.normal(0.0, 2.0, 300) + 0.5
    val_uv, a_uv, b_uv = cross_sampen(u, v)
    val_vu, a_vu, b_vu = cross_sampen(v, u)
    assert a_uv == a_vu
    assert b_uv == b_vu
    assert val_uv == pytest.approx(val_vu)


def test_counters_exact_against_naive():
    rng = np.random.default_rng(2)
    cont = rng.normal(0.0, 1.0, 400)
    assert _count_1d_pairs_within(cont, 0.3) == _naive_1d(cont, 0.3)
    ints = rng.integers(0, 20, 400).astype(np.float64)  # tie-heavy: exact-r pairs occur
    assert _count_1d_pairs_within(ints, 2.0) == _naive_1d(ints, 2.0)


def test_sampen_general_m_runs():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 2000)
    value, a, b = sampen(x, m=2, r_sd=0.2)
    assert np.isfinite(value)
    assert a > 0 and b > 0
