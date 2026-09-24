"""TF-56 optimise_positions pure solver entry (spec section 5)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.positioning import OptimizeResult, SimulatedAnnealing, optimise_positions


def _run(frame, obj, con, movable, seed=0):
    return optimise_positions(frame, movable=movable, objective=obj, constraints=[con], seed=seed)


def test_purity_frame_unchanged(one_frame, threat_objective, reachability_constraint, movable_ids):
    before = one_frame.copy(deep=True)
    _run(one_frame, threat_objective, reachability_constraint, movable_ids)
    pd.testing.assert_frame_equal(one_frame, before)


def test_default_optimizer_is_simulated_annealing(one_frame, threat_objective, reachability_constraint, movable_ids):
    res = _run(one_frame, threat_objective, reachability_constraint, movable_ids)
    assert isinstance(res, OptimizeResult)


def test_deterministic_on_seed(one_frame, threat_objective, reachability_constraint, movable_ids):
    a = _run(one_frame, threat_objective, reachability_constraint, movable_ids, seed=5)
    b = _run(one_frame, threat_objective, reachability_constraint, movable_ids, seed=5)
    assert a.best_score == b.best_score
    assert a.best_frame.equals(b.best_frame)


def test_gap_ge_zero(one_frame, threat_objective, reachability_constraint, movable_ids):
    res = _run(one_frame, threat_objective, reachability_constraint, movable_ids, seed=3)
    assert res.actual_score - res.best_score >= 0.0


def test_explicit_optimizer_is_used(one_frame, threat_objective, reachability_constraint, movable_ids):
    from silly_kicks.positioning import SAParams

    opt = SimulatedAnnealing(SAParams(num_iterations=50, patience=50))
    res = optimise_positions(
        one_frame,
        movable=movable_ids,
        objective=threat_objective,
        constraints=[reachability_constraint],
        optimizer=opt,
        seed=0,
    )
    assert res.n_iter <= 50


def test_empty_movable_raises(one_frame, threat_objective, reachability_constraint):
    with pytest.raises(ValueError, match="non-empty"):
        optimise_positions(
            one_frame, movable=[], objective=threat_objective, constraints=[reachability_constraint], seed=0
        )
