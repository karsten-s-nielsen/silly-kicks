"""TF-56 validation-battery machinery (spec section 8, AMENDED 2026-09-22; commit-2 GO/NO-GO).

The dose/responsiveness instrument was RETIRED (spike: gap non-monotonic in any dose). The battery is
now optimizer STABILITY + DISCRIMINATION (fixture instrument validity) and PREDICTIVE
corr(gap_t, conceded_threat_{t+dt}) (corpus construct validity). Every verdict is tested from BOTH
sides; the stability test is non-vacuous (it FLIPS to seed_unstable at a large init_sigma_m).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.positioning import (
    ReachabilityConstraint,
    ReachabilityParams,
    SAParams,
    ThreatObjective,
    compute_positioning_gap,
)
from silly_kicks.positioning._probe import (
    averaging_artifact_demo,
    discrimination_verdict,
    horizon_sensitivity,
    optimizer_stability_verdict,
    predictive_verdict,
)
from silly_kicks.tracking import resolve_defended_goals


def _threat_setup(one_frame, fitted_xt):
    goal_map = resolve_defended_goals(one_frame)
    objective = ThreatObjective(xt=fitted_xt, goal_map=goal_map, attacking_team_id=2)
    constraints = [ReachabilityConstraint(ReachabilityParams.default())]
    return objective, constraints


@pytest.mark.slow  # ~22s: 8-seed + doubled-iteration SA solves; deterministic/invariant (ADR-023)
def test_optimizer_stability_is_seed_and_iteration_invariant(one_frame, fitted_xt, movable_ids):
    """At the shipped init_sigma_m=2.0 the SA optimum is seed-invariant (the amended-battery gate)."""
    objective, constraints = _threat_setup(one_frame, fitted_xt)
    v = optimizer_stability_verdict(
        one_frame, movable=movable_ids, objective=objective, constraints=constraints, seeds=(0, 1, 2, 3)
    )
    assert v["verdict"] == "stable"
    assert v["gap_std"] <= 1e-6
    assert v["iter_delta"] <= 1e-6


@pytest.mark.slow  # ~45s: 8-seed + doubled-iteration SA solves; deterministic/invariant (ADR-023)
def test_optimizer_stability_flips_to_seed_unstable_at_the_old_sigma(one_frame, fitted_xt, movable_ids):
    """NON-VACUITY: the verdict is not always 'stable' -- the RETIRED init_sigma_m=5.0 default flips it.

    This is the spike finding that moved the default: at the old 5.0 the optimum is seed-NOISY
    (per-frame gap std ~0.32), so the column was a partial seed artifact; at the shipped 2.0 it is
    seed-invariant (the test above). (A too-large sigma instead collapses feasibility to a degenerate
    no-move optimum, so 5.0 -- the reachability-mismatched-but-still-exploring value -- is the honest
    demonstration.)
    """
    objective, constraints = _threat_setup(one_frame, fitted_xt)
    old_sigma = dataclasses.replace(SAParams.default(), init_sigma_m=5.0)
    v = optimizer_stability_verdict(
        one_frame,
        movable=movable_ids,
        objective=objective,
        constraints=constraints,
        seeds=(0, 1, 2, 3, 4, 5, 6, 7),
        params=old_sigma,
    )
    assert v["verdict"] == "seed_unstable"
    assert v["gap_std"] > 1e-6


def test_discrimination_verdict_both_sides():
    assert discrimination_verdict([0.5, 1.5, 3.0, 2.0]) == "discriminating"
    assert discrimination_verdict([2.0, 2.0, 2.0]) == "degenerate"
    assert discrimination_verdict([1.0]) == "degenerate"  # too few finite values


def test_predictive_verdict_both_sides():
    rng = np.random.default_rng(0)
    g = np.arange(300.0)
    # planted positive correlation -> predictive
    c_pos = 0.5 * g + rng.normal(0.0, 5.0, size=g.size)
    pos = predictive_verdict(g, c_pos, n_min=100)
    assert pos["verdict"] == "predictive"
    assert pos["r"] > 0.0 and pos["p"] < 0.05
    # independent draw -> no correlation -> not_predictive (deterministic under the fixed seed)
    c_none = rng.normal(0.0, 1.0, size=g.size)
    none = predictive_verdict(g, c_none, n_min=100)
    assert none["verdict"] == "not_predictive"


def test_predictive_verdict_arm_unscoreable_on_thin_or_degenerate():
    # below n_min -> arm_unscoreable (distinct token, not not_predictive)
    thin = predictive_verdict(np.arange(10.0), np.arange(10.0), n_min=200)
    assert thin["verdict"] == "arm_unscoreable"
    # a constant leg (zero variance) -> arm_unscoreable, never a fabricated correlation
    const = predictive_verdict(np.arange(300.0), np.full(300, 2.0), n_min=100)
    assert const["verdict"] == "arm_unscoreable"


def test_horizon_sensitivity_returns_a_distribution_per_horizon(one_frame, fitted_xt):
    dists = horizon_sensitivity(one_frame, xt=fitted_xt, horizons=(0.5, 0.7, 1.0))
    assert set(dists.keys()) == {0.5, 0.7, 1.0}
    for arr in dists.values():
        assert isinstance(arr, np.ndarray)
        assert np.isfinite(arr).all()
        assert (arr >= 0.0).all()  # gap >= 0 at every horizon


@pytest.mark.slow  # ~75s: uncapped + capped full SA solves + leave-one-out scoring; invariant (ADR-023)
def test_averaging_artifact_demo_cap_is_load_bearing(one_frame, fitted_xt, movable_ids):
    demo = averaging_artifact_demo(one_frame, xt=fitted_xt, attacking_team_id=2, movable=movable_ids, cap=0.2, seed=0)
    # both optima computed + the worst-attacker exposure measured on each side
    assert np.isfinite(demo["uncapped_worst_attacker_exposure"])
    assert np.isfinite(demo["capped_worst_attacker_exposure"])
    # NON-VACUITY: the cap measurably changes the optimizer's result (it is not decorative).
    assert demo["optima_differ"] is True


def test_scored_gap_is_seed_invariant_end_to_end(one_frame, fitted_xt):
    """The metric column is a pure function of inputs at the shipped sigma (compute uses a frame seed)."""
    a, _ = compute_positioning_gap(one_frame, xt=fitted_xt)
    b, _ = compute_positioning_gap(one_frame, xt=fitted_xt)
    ga = float(a.loc[a["positioning_gap_source"] == "scored", "positioning_gap"].iloc[0])
    gb = float(b.loc[b["positioning_gap_source"] == "scored", "positioning_gap"].iloc[0])
    assert ga == pytest.approx(gb)
