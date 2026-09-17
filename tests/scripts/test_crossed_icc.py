"""Known-truth tests for the crossed variance-components ICC estimator (Task 10).

Every scenario draws a synthetic crossed design with KNOWN variance components at a FIXED seed, so the
assertions are deterministic. The estimator is Henderson Method III (a moment estimator): between-
source components carry sampling error that shrinks with the number of LEVELS, not observations, so
the per-component tolerances below are set to what a design with this many defenders/teams achieves in
one draw (measured), with margin -- they are not arbitrarily tight, and the report records the exact
achieved relative errors.
"""

from __future__ import annotations

import numpy as np
from _crossed_icc import bootstrap_icc_ci, crossed_variance_components


def _balanced_crossed_design(seed, n_def, n_team, reps, *, sd_def, sd_team, sd_resid, mu=5.0):
    """A fully-crossed BALANCED design: every (defender, team) cell has ``reps`` observations.

    Balanced so the Henderson-III moment estimator is well-conditioned (the two-way random EMS is
    exact and orthogonal), which is the regime where recovery is testable against known truth.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd_def, n_def)  # defender effects ~ N(0, sd_def)
    b = rng.normal(0.0, sd_team, n_team)  # team effects   ~ N(0, sd_team)
    d_grid, t_grid = np.meshgrid(np.arange(n_def), np.arange(n_team), indexing="ij")
    defender_codes = np.repeat(d_grid.ravel(), reps)
    team_codes = np.repeat(t_grid.ravel(), reps)
    y = mu + a[defender_codes] + b[team_codes] + rng.normal(0.0, sd_resid, size=defender_codes.size)
    return y, defender_codes, team_codes


def test_recovers_known_variance_components_and_icc():
    """A balanced design with known s2_def=4, s2_team=2.25, s2_resid=9 -> ICC = 4/15.25 ~ 0.2623.

    Seed 4 at 80 defenders x 60 teams x 5 reps (24,000 obs) recovers, MEASURED:
    var_defender rel-err 0.050, var_team rel-err 0.064, var_resid rel-err 0.004, ICC err +0.0076.
    Tolerances carry margin above those achieved values for cross-platform BLAS float robustness.
    """
    sd_def, sd_team, sd_resid = 2.0, 1.5, 3.0
    true_def, true_team, true_resid = sd_def**2, sd_team**2, sd_resid**2
    true_icc = true_def / (true_def + true_team + true_resid)

    y, dc, tc = _balanced_crossed_design(4, 80, 60, 5, sd_def=sd_def, sd_team=sd_team, sd_resid=sd_resid)
    out = crossed_variance_components(y, dc, tc)

    assert out["had_negative_estimate"] is False
    # each component within a stated RELATIVE tolerance (measured errs 0.050 / 0.064 / 0.004)
    assert abs(out["var_defender"] - true_def) / true_def < 0.15
    assert abs(out["var_team"] - true_team) / true_team < 0.15
    assert abs(out["var_resid"] - true_resid) / true_resid < 0.05
    # ICC within +/- 0.03 of truth (measured err +0.0076)
    assert abs(out["icc_defender"] - true_icc) < 0.03


def test_defender_null_design_gives_near_zero_icc_with_bootstrap_lo_at_zero():
    """s2_defender = 0: the ICC estimate is ~0 and the bootstrap lower bound does not exceed ~0.

    A defender-null design is the "not identifiable" case the census must not mistake for a ranking.
    The small design (30 x 20 x 4) keeps the 80-replicate bootstrap fast; the directional claim
    (icc ~ 0, lo <= epsilon) is robust to the seed.
    """
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=0.0, sd_team=1.5, sd_resid=3.0)

    point = crossed_variance_components(y, dc, tc)
    assert point["icc_defender"] < 0.03  # near zero

    ci = bootstrap_icc_ci(y, dc, tc, n_boot=80, alpha=0.10, rng_seed=1)
    assert abs(ci["icc"] - point["icc_defender"]) < 1e-12  # point matches the direct fit
    assert ci["lo"] <= 0.05  # lower bound includes / does not exceed ~0
    assert ci["lo"] <= ci["hi"]


def test_strong_defender_design_gives_bootstrap_lo_above_zero():
    """A large s2_defender: the whole bootstrap interval sits above 0, i.e. the ICC is detectable.

    Same small fast design as the null case (30 x 20 x 4, 80 replicates); the point ICC is ~0.8 and
    the lower bound clears 0 comfortably -- the "defender ranking is identifiable" verdict.
    """
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=4.0, sd_team=1.0, sd_resid=2.0)

    point = crossed_variance_components(y, dc, tc)
    assert point["icc_defender"] > 0.5

    ci = bootstrap_icc_ci(y, dc, tc, n_boot=80, alpha=0.10, rng_seed=1)
    assert ci["lo"] > 0.0
    assert ci["lo"] <= ci["icc"] <= ci["hi"]


def test_tiny_degenerate_design_truncates_negative_estimate_to_zero():
    """A tiny defender-null design forces a negative RAW variance estimate -> truncated to 0.0 + flag.

    4 defenders x 3 teams x 2 reps (24 obs) with s2_defender = 0: sampling noise pushes the raw
    Henderson-III solve for var_defender negative, so the estimator truncates it to exactly 0.0 and
    sets had_negative_estimate=True. Seed 1 is the committed choice.
    """
    y, dc, tc = _balanced_crossed_design(1, 4, 3, 2, sd_def=0.0, sd_team=3.0, sd_resid=3.0)
    out = crossed_variance_components(y, dc, tc)

    assert out["had_negative_estimate"] is True
    assert out["var_defender"] == 0.0
    # the truncated components stay non-negative and the ICC reads as a clean non-detection
    assert out["var_team"] >= 0.0
    assert out["var_resid"] >= 0.0
    assert out["icc_defender"] == 0.0


def test_icc_denominator_zero_returns_zero_not_nan():
    """A perfectly flat outcome (all components truncate to 0) returns icc_defender = 0.0, not NaN.

    Documents the guarded-denominator choice: a downstream ``icc > threshold`` gate reads 0.0 as
    "measured, no defender separation" rather than tripping on a NaN comparison.
    """
    # constant y across a valid 3-level design: MS_d = MS_t = MSE = 0 -> all components 0.
    y, dc, tc = _balanced_crossed_design(0, 5, 4, 3, sd_def=0.0, sd_team=0.0, sd_resid=0.0)
    out = crossed_variance_components(y, dc, tc)
    assert out["var_defender"] == 0.0
    assert out["icc_defender"] == 0.0
    assert not np.isnan(out["icc_defender"])
