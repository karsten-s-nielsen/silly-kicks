"""TF-56 commit-1 perf benchmark: MEASURE per-frame cost + structural guards (spec section 8b).

Structural, not wall-clock (ADR-073): the SA's dominant expensive primitive is ``compute_threat_pc``
(one full pitch-control surface per scored proposal). The guards assert its call-count is exactly
``1 + n_feasible_proposals`` (no hidden quadratic re-scoring) and grows sub-quadratically in the
iteration budget. The per-frame WALL-CLOCK is MEASURED (printed) to feed the commit-2 corpus
feasibility decision -- it is never asserted as a ceiling (that is what the structural guards do).
"""

from __future__ import annotations

import time

import silly_kicks.positioning._constraints as _con_mod
import silly_kicks.positioning._objectives as _obj_mod
from silly_kicks.positioning import (
    ReachabilityConstraint,
    ReachabilityParams,
    SAParams,
    SimulatedAnnealing,
    ThreatObjective,
    compute_positioning_gap,
    optimise_positions,
)
from silly_kicks.tracking import resolve_defended_goals
from tests._perf_structural import assert_subquadratic_growth, call_counter


def _threat_obj(one_frame, fitted_xt):
    return ThreatObjective(xt=fitted_xt, goal_map=resolve_defended_goals(one_frame), attacking_team_id=2)


def test_threat_pc_called_once_per_feasible_proposal_plus_incumbent(one_frame, fitted_xt, movable_ids, monkeypatch):
    """No hidden quadratic re-scoring: compute_threat_pc runs exactly once for the incumbent-0
    (actual_score) plus once per FEASIBLE proposal (infeasible proposals never score)."""
    calls = call_counter(monkeypatch, _obj_mod, "compute_threat_pc")
    res = optimise_positions(
        one_frame,
        movable=movable_ids,
        objective=_threat_obj(one_frame, fitted_xt),
        constraints=[ReachabilityConstraint(ReachabilityParams.default())],
        optimizer=SimulatedAnnealing(SAParams(num_iterations=300, patience=300)),
        seed=0,
    )
    assert calls["n"] == 1 + res.n_feasible_proposals


def test_score_calls_never_exceed_the_iteration_budget(one_frame, fitted_xt, movable_ids, monkeypatch):
    """Anti-quadratic invariant: at most ONE compute_threat_pc per iteration (plus the incumbent-0),
    so total score calls <= num_iterations + 1 -- no O(n) re-scoring inside an iteration."""
    n = 300
    calls = call_counter(monkeypatch, _obj_mod, "compute_threat_pc")
    optimise_positions(
        one_frame,
        movable=movable_ids,
        objective=_threat_obj(one_frame, fitted_xt),
        constraints=[ReachabilityConstraint(ReachabilityParams.default())],
        optimizer=SimulatedAnnealing(SAParams(num_iterations=n, patience=n)),
        seed=0,
    )
    assert calls["n"] <= n + 1


def test_loop_is_linear_in_iterations(one_frame, fitted_xt, movable_ids, monkeypatch):
    """The SA loop is linear in the iteration budget: exactly ONE feasibility check (compute_tti)
    per iteration, no nested per-iteration scan. (compute_threat_pc call growth is deliberately NOT
    the proxy -- its count is confounded by the cooling-driven feasibility ramp, spec section 8b.)"""
    obj = _threat_obj(one_frame, fitted_xt)
    reach = [ReachabilityConstraint(ReachabilityParams.default())]

    def measure(n: int) -> int:
        calls = call_counter(monkeypatch, _con_mod, "compute_tti")
        optimise_positions(
            one_frame,
            movable=movable_ids,
            objective=obj,
            constraints=reach,
            optimizer=SimulatedAnnealing(SAParams(num_iterations=n, patience=n)),
            seed=0,
        )
        return calls["n"]

    assert_subquadratic_growth(
        measure, sizes=(100, 200, 400), max_exponent=1.5, label="positioning SA feasibility checks"
    )


def test_measured_per_frame_cost_is_reported_not_gated(one_frame, fitted_xt):
    """MEASURE (never gate) the per-scored-frame cost at the frozen defaults -- the number that
    feeds the commit-2 bounded-corpus feasibility decision (spec section 8b)."""
    t0 = time.perf_counter()
    samples, _ = compute_positioning_gap(one_frame, xt=fitted_xt)
    dt = time.perf_counter() - t0
    n_iter = SAParams.default().num_iterations
    print(f"\n[TF-56 PERF] compute_positioning_gap 1 scored frame @ {n_iter} iters: {dt * 1000:.0f} ms")
    assert (samples["positioning_gap_source"] == "scored").sum() == 1
    assert dt >= 0.0  # MEASURE only -- never a wall-clock ceiling
