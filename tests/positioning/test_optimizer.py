"""TF-56 SimulatedAnnealing optimizer (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.positioning import OptimizeResult, SAParams, SimulatedAnnealing


def test_result_is_an_optimize_result_and_actual_is_incumbent0(
    one_frame, threat_objective, reachability_constraint, movable_ids
):
    sa = SimulatedAnnealing(SAParams(num_iterations=300, patience=150))
    res = sa.optimize(
        one_frame,
        movable=movable_ids,
        objective=threat_objective,
        constraints=[reachability_constraint],
        rng=np.random.default_rng(0),
    )
    assert isinstance(res, OptimizeResult)
    # actual_score is the factual shape via the SAME path (incumbent-0), and best <= actual always.
    assert res.actual_score == threat_objective.score(one_frame)
    assert res.best_score <= res.actual_score


def test_finds_a_reachable_improving_move(one_frame, threat_objective, reachability_constraint, movable_ids):
    sa = SimulatedAnnealing(SAParams(num_iterations=1500, patience=1500))
    res = sa.optimize(
        one_frame,
        movable=movable_ids,
        objective=threat_objective,
        constraints=[reachability_constraint],
        rng=np.random.default_rng(1),
    )
    assert res.n_feasible_proposals > 0, "SA never found a feasible proposal -- the search is dead"
    assert res.best_score < res.actual_score, "no reachable reposition lowered the conceded threat"


def test_determinism_same_seed_same_result(one_frame, threat_objective, reachability_constraint, movable_ids):
    def run():
        return SimulatedAnnealing(SAParams(num_iterations=400, patience=400)).optimize(
            one_frame,
            movable=movable_ids,
            objective=threat_objective,
            constraints=[reachability_constraint],
            rng=np.random.default_rng(42),
        )

    a, b = run(), run()
    assert a.best_score == b.best_score
    assert a.actual_score == b.actual_score
    assert a.n_feasible_proposals == b.n_feasible_proposals
    assert a.best_frame.equals(b.best_frame)


def test_optimize_does_not_mutate_the_input_frame(one_frame, threat_objective, reachability_constraint, movable_ids):
    before = one_frame.copy(deep=True)
    SimulatedAnnealing(SAParams(num_iterations=200, patience=200)).optimize(
        one_frame,
        movable=movable_ids,
        objective=threat_objective,
        constraints=[reachability_constraint],
        rng=np.random.default_rng(7),
    )
    pd.testing.assert_frame_equal(one_frame, before)


def test_empty_movable_returns_zero_gap(one_frame, threat_objective, reachability_constraint):
    res = SimulatedAnnealing().optimize(
        one_frame,
        movable=[],
        objective=threat_objective,
        constraints=[reachability_constraint],
        rng=np.random.default_rng(0),
    )
    assert res.n_feasible_proposals == 0
    assert res.best_score == res.actual_score  # nothing to move -> no gap
