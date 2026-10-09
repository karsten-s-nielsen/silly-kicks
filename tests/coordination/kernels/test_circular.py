"""TF-58 Task 8: circular statistics kernels."""

from __future__ import annotations

import numpy as np
import scipy.stats

from silly_kicks.coordination._kernels._circular import (
    HIST_BIN_CENTRES_DEG,
    circular_reliability,
    circular_summary,
    hist_bin_index,
    near_in_phase,
    wrap_deg,
)


def _ref_circular_reliability(values_deg, groups):
    """Independent reference: ``1 - V_within / V_total`` via mean resultant lengths."""
    v = np.asarray(values_deg, dtype="float64")
    g = np.asarray(groups, dtype=object)
    m = np.isfinite(v)
    v, g = v[m], g[m]
    n = v.size
    c, s = np.cos(np.radians(v)), np.sin(np.radians(v))
    s_all = float(np.hypot(c.sum(), s.sum()))
    rbar_total = s_all / n if n else float("nan")
    uniq = list(dict.fromkeys(g.tolist()))
    k = len(uniq)
    if n == 0 or k < 2 or n <= k:
        return float("nan"), rbar_total
    sg = 0.0
    for key in uniq:
        sel = g == key
        sg += float(np.hypot(c[sel].sum(), s[sel].sum()))
    v_within = 1.0 - sg / n
    v_total = 1.0 - rbar_total
    if v_total <= 0.0:
        return float("nan"), rbar_total
    return 1.0 - v_within / v_total, rbar_total


def test_circular_summary_parity_with_scipy():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, 500)
    z = np.exp(1j * theta)
    mean_deg, r, circ_sd_deg = circular_summary(z.sum(), float(len(z)))

    sp_mean = np.degrees(scipy.stats.circmean(theta))  # default [0, 2pi); wrap_deg makes the diff range-safe
    sp_std = np.degrees(scipy.stats.circstd(theta))
    assert abs(float(wrap_deg(mean_deg - sp_mean))) < 1e-10
    assert abs(float(circ_sd_deg) - sp_std) < 1e-10
    assert 0.0 <= float(r) <= 1.0


def test_circular_summary_edge_cases():
    # n == 0 -> NaN triple
    m, r, s = circular_summary(0.0 + 0.0j, 0.0)
    assert np.isnan(m) and np.isnan(r) and np.isnan(s)
    # R == 0 (zero resultant, n > 0) -> circ SD is +inf
    _m0, r0, s0 = circular_summary(0.0 + 0.0j, 4.0)
    assert float(r0) == 0.0
    assert np.isposinf(float(s0))
    # perfectly concentrated -> R == 1, SD ~ 0 (the circular SD sqrt-amplifies R's float error near 1)
    _m1, r1, s1 = circular_summary(np.exp(1j * 0.3) * 5, 5.0)
    assert abs(float(r1) - 1.0) < 1e-9
    assert abs(float(s1)) < 1e-3


def test_histogram_edges_both_sides():
    for c in HIST_BIN_CENTRES_DEG:
        want = HIST_BIN_CENTRES_DEG.index(c)
        prev = (want - 1) % 12
        lo = c - 15.0
        assert int(hist_bin_index(np.exp(1j * np.radians(lo + 1e-9)))) == want
        assert int(hist_bin_index(np.exp(1j * np.radians(lo - 1e-9)))) == prev
    # the +/-180 wrap
    assert int(hist_bin_index(np.exp(1j * np.radians(165.0 + 1e-9)))) == 0
    assert int(hist_bin_index(np.exp(1j * np.radians(165.0 - 1e-9)))) == 11
    assert int(hist_bin_index(np.exp(1j * np.radians(-180.0 + 1e-9)))) == 0
    assert int(hist_bin_index(np.exp(1j * np.radians(180.0)))) == 0


def test_hist_bin_index_is_int8_and_vectorised():
    z = np.exp(1j * np.radians(np.array([0.0, 30.0, -30.0, 179.0])))
    out = hist_bin_index(z)
    assert out.dtype == np.int8
    assert list(out) == [6, 7, 5, 0]


def test_near_in_phase_threshold_both_sides():
    for ang, want in [(29.999, True), (30.001, False), (-29.999, True), (-30.001, False)]:
        z = np.exp(1j * np.radians(ang))
        assert bool(near_in_phase(z, 30.0)) is want


def test_wrap_deg_range():
    assert float(wrap_deg(180.0)) == 180.0
    assert float(wrap_deg(-180.0)) == 180.0
    assert float(wrap_deg(190.0)) == -170.0
    assert float(wrap_deg(-190.0)) == 170.0
    assert float(wrap_deg(0.0)) == 0.0


# --- A-09: rotation-invariant circular reliability -------------------------------------------------


def test_circular_reliability_matches_worked_case():
    # Two units (groups), tight within, well separated between -> high reliability.
    values = [10.0, 20.0, 100.0, 110.0]
    groups = ["A", "A", "B", "B"]
    rel, rbar = circular_reliability(values, groups)
    ref_rel, ref_rbar = _ref_circular_reliability(values, groups)
    assert abs(rel - ref_rel) < 1e-12
    assert abs(rbar - ref_rbar) < 1e-12
    assert abs(rel - 0.98712616) < 1e-6  # hand-computed
    assert 0.0 < rbar < 1.0


def test_circular_reliability_is_rotation_invariant():
    # The whole point: adding a constant phase to every value leaves reliability unchanged,
    # where a cos/sin-component ICC would move with the origin.
    base = np.array([10.0, 20.0, 100.0, 110.0, 205.0, 215.0])
    groups = ["A", "A", "B", "B", "C", "C"]
    rel0, _ = circular_reliability(base, groups)
    for offset in (37.0, 180.0, -123.4, 359.0):
        rel, _ = circular_reliability(base + offset, groups)
        assert abs(rel - rel0) < 1e-9
    # non-vacuity: a naive component ICC(1) on cos IS origin-dependent (that is why it is wrong here).
    from scripts._reliability import icc1

    icc_cos_0 = icc1(np.cos(np.radians(base)), np.asarray(groups))
    icc_cos_90 = icc1(np.cos(np.radians(base + 90.0)), np.asarray(groups))
    assert abs(icc_cos_0 - icc_cos_90) > 1e-3


def test_circular_reliability_perfect_within_concentration_is_one():
    values = [10.0, 10.0, 100.0, 100.0, 200.0, 200.0]
    groups = ["A", "A", "B", "B", "C", "C"]
    rel, rbar = circular_reliability(values, groups)
    assert abs(rel - 1.0) < 1e-12
    assert 0.0 < rbar < 1.0


def test_circular_reliability_no_discrimination_is_zero():
    # Every group holds the SAME pair of angles -> within spread == total spread -> reliability 0.
    values = [0.0, 90.0, 0.0, 90.0]
    groups = ["A", "A", "B", "B"]
    rel, _ = circular_reliability(values, groups)
    assert abs(rel - 0.0) < 1e-12


def test_circular_reliability_reports_rbar_for_the_floor():
    # Uniformly spread -> total resultant ~ 0; the kernel reports rbar so the caller can floor it.
    values = [0.0, 180.0, 90.0, 270.0]
    groups = ["A", "A", "B", "B"]
    _rel, rbar = circular_reliability(values, groups)
    assert abs(rbar) < 1e-12


def test_circular_reliability_undefined_when_total_variance_zero():
    # All observations in one direction -> no total variance -> reliability undefined (NaN), rbar == 1.
    values = [45.0, 45.0, 45.0, 45.0]
    groups = ["A", "A", "B", "B"]
    rel, rbar = circular_reliability(values, groups)
    assert np.isnan(rel)
    assert abs(rbar - 1.0) < 1e-12


def test_circular_reliability_guards_and_nan_handling():
    # fewer than 2 groups -> NaN reliability (rbar still reported)
    rel, rbar = circular_reliability([10.0, 20.0], ["A", "A"])
    assert np.isnan(rel)
    assert np.isfinite(rbar)
    # n <= k (one obs per group) -> NaN
    rel2, _ = circular_reliability([10.0, 100.0], ["A", "B"])
    assert np.isnan(rel2)
    # NaN values are dropped, not propagated
    with_nan = circular_reliability([10.0, 20.0, np.nan, 100.0, 110.0], ["A", "A", "A", "B", "B"])
    without = circular_reliability([10.0, 20.0, 100.0, 110.0], ["A", "A", "B", "B"])
    assert abs(with_nan[0] - without[0]) < 1e-12
    # empty -> NaN pair
    rele, rbare = circular_reliability([], [])
    assert np.isnan(rele) and np.isnan(rbare)
