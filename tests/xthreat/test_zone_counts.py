"""`XtZoneCounts` + `ExpectedThreat.zone_counts` (spec §4.4, amends ADR-102).

`zone_counts` shares `fit()`'s counting extractors, so `fit_from_counts(**zone_counts(a).as_fit_kwargs())`
is byte-identical to `fit(a)`. The oracle here is INDEPENDENT (mirrors the ADR-102 test's `_aggregate`,
never the function under test) -- including the valid-start / NaN-end move that separates the
`_action_prob` move population from the Singh denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import ExpectedThreat, XtZoneCounts
from silly_kicks.xthreat._grid import _count, _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]
_PASS = spadlconfig.actiontype_id["pass"]
_DRIBBLE = spadlconfig.actiontype_id["dribble"]
_CROSS = spadlconfig.actiontype_id["cross"]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]
_MOVE_TYPES = (_PASS, _DRIBBLE, _CROSS)
_L, _W = 16, 12


def _row(t, r, sx, sy, ex, ey):
    return {"type_id": t, "result_id": r, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(_SHOT, _SUCCESS, 20.0, 34.0, 105.0, 34.0),  # goal
            _row(_SHOT, _FAIL, 22.0, 36.0, 104.0, 30.0),
            _row(_SHOT, _FAIL, 95.0, 34.0, 105.0, 34.0),  # shots, no goal (case b)
            _row(_PASS, _SUCCESS, 10.0, 10.0, 40.0, 20.0),
            _row(_PASS, _SUCCESS, 40.0, 20.0, 70.0, 40.0),
            _row(_DRIBBLE, _SUCCESS, 70.0, 40.0, 78.0, 44.0),
            _row(_CROSS, _SUCCESS, 88.0, 8.0, 96.0, 34.0),
            _row(_PASS, _FAIL, 50.0, 30.0, 65.0, 25.0),  # failed move, valid end (Singh denom, not numerator)
            _row(_PASS, _SUCCESS, 45.0, 45.0, np.nan, np.nan),  # (a) valid start, NaN end
            _row(_PASS, _SUCCESS, 60.0, 30.0, 110.0, 34.0),  # (c) off-pitch end clamps
            _row(_PASS, _SUCCESS, 30.0, 60.0, 60.0, 62.0),
        ]
    )


def _oracle(actions: pd.DataFrame) -> dict:
    """Independent 5-count oracle (mirrors the ADR-102 test's `_aggregate`), never `zone_counts`."""
    n = _L * _W
    shots = actions[actions.type_id == _SHOT]
    goals = shots[shots.result_id == _SUCCESS]
    moves = actions[actions.type_id.isin(_MOVE_TYPES)]
    moves_se = moves.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    succ = moves_se[moves_se.result_id == _SUCCESS]
    start_flat = _get_flat_indexes(succ.start_x, succ.start_y, _L, _W).to_numpy()
    end_flat = _get_flat_indexes(succ.end_x, succ.end_y, _L, _W).to_numpy()
    tc = np.zeros((n, n), dtype=np.int64)
    np.add.at(tc, (start_flat, end_flat), 1)
    return {
        "shot_counts": _count(shots.start_x, shots.start_y, _L, _W),
        "goal_counts": _count(goals.start_x, goals.start_y, _L, _W),
        "move_counts": _count(moves.start_x, moves.start_y, _L, _W),  # valid start only
        "transition_start_counts": _count(moves_se.start_x, moves_se.start_y, _L, _W),  # valid start+end
        "transition_counts": tc,
    }


# ---- XtZoneCounts value object -------------------------------------------------------------------


def test_zeros_shapes_and_int64():
    z = XtZoneCounts.zeros(_L, _W)
    assert z.shot_counts.shape == (_W, _L) and z.transition_counts.shape == (_W * _L, _W * _L)
    assert all(a.dtype == np.int64 for a in (z.shot_counts, z.move_counts, z.transition_counts))


def test_add_sums_elementwise_and_raises_on_grid_mismatch():
    a = XtZoneCounts.zeros(_L, _W)
    b = ExpectedThreat(l=_L, w=_W).zone_counts(_actions())
    summed = a + b
    assert np.array_equal(summed.shot_counts, b.shot_counts)
    with pytest.raises(ValueError, match="grid mismatch"):
        _ = b + XtZoneCounts.zeros(8, 6)


def test_as_fit_kwargs_round_trips_the_five_arrays():
    z = ExpectedThreat(l=_L, w=_W).zone_counts(_actions())
    kw = z.as_fit_kwargs()
    assert set(kw) == {"shot_counts", "goal_counts", "move_counts", "transition_start_counts", "transition_counts"}


# ---- zone_counts == independent oracle -----------------------------------------------------------


def test_zone_counts_equals_the_independent_oracle():
    z = ExpectedThreat(l=_L, w=_W).zone_counts(_actions())
    oracle = _oracle(_actions())
    for name, expected in oracle.items():
        assert np.array_equal(getattr(z, name), expected), name


# ---- round-trip: fit_from_counts(zone_counts) == fit --------------------------------------------


def test_fit_from_zone_counts_is_byte_identical_to_fit():
    a = _actions()
    from_counts = ExpectedThreat(l=_L, w=_W).fit_from_counts(
        **ExpectedThreat(l=_L, w=_W).zone_counts(a).as_fit_kwargs()
    )
    from_actions = ExpectedThreat(l=_L, w=_W).fit(a)
    for m in ("scoring_prob_matrix", "shot_prob_matrix", "move_prob_matrix", "transition_matrix"):
        assert np.array_equal(getattr(from_counts, m), getattr(from_actions, m), equal_nan=True), m
    assert np.array_equal(from_counts.xT, from_actions.xT, equal_nan=True)


# ---- additivity across pseudo-matches ------------------------------------------------------------


def test_additivity_across_pseudo_matches():
    a = _actions()
    parts = [a.iloc[idx] for idx in np.array_split(np.arange(len(a)), 3) if len(idx)]
    summed = parts[0].pipe(lambda df: ExpectedThreat(l=_L, w=_W).zone_counts(df))
    for p in parts[1:]:
        summed = summed + ExpectedThreat(l=_L, w=_W).zone_counts(p)
    pooled = ExpectedThreat(l=_L, w=_W).zone_counts(a)
    for name in ("shot_counts", "goal_counts", "move_counts", "transition_start_counts", "transition_counts"):
        assert np.array_equal(getattr(summed, name), getattr(pooled, name)), name
    xt_summed = ExpectedThreat(l=_L, w=_W).fit_from_counts(**summed.as_fit_kwargs())
    xt_pooled = ExpectedThreat(l=_L, w=_W).fit(a)
    assert np.array_equal(xt_summed.xT, xt_pooled.xT, equal_nan=True)
