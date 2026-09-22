"""SK-XT-COUNTS -- ExpectedThreat.fit_from_counts (distributed counts-based fit) + zone-binning.

Red-green TDD (spec ``docs/superpowers/specs/2026-09-22-expectedthreat-fit-from-counts-design.md``).
``fit_from_counts`` must be byte-identical to ``fit(actions)`` on the exact aggregates -- the crux is
that the ``_action_prob`` move population (valid START only) DIFFERS from the Singh denominator (valid
START *and* END), so the contract carries THREE move aggregates and a valid-start/NaN-end fixture move
(case a) is the non-vacuous trigger.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import ExpectedThreat, KDEParams
from silly_kicks.xthreat._grid import _count, _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]
_PASS = spadlconfig.actiontype_id["pass"]
_DRIBBLE = spadlconfig.actiontype_id["dribble"]
_CROSS = spadlconfig.actiontype_id["cross"]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]
_MOVE_TYPES = (_PASS, _DRIBBLE, _CROSS)

_L, _W = 16, 12


def _row(type_id, result_id, sx, sy, ex, ey):
    return {"type_id": type_id, "result_id": result_id, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}


def _fixture_a() -> pd.DataFrame:
    """A SPADL action set carrying the three named boundary cases (spec §6.1)."""
    rows = [
        # shots -- zone (20,34) has 2 shots + 1 goal; zone (95,34) has shots but ZERO goals (case b)
        _row(_SHOT, _SUCCESS, 20.0, 34.0, 105.0, 34.0),
        _row(_SHOT, _FAIL, 22.0, 36.0, 104.0, 30.0),
        _row(_SHOT, _FAIL, 95.0, 34.0, 105.0, 34.0),
        _row(_SHOT, _FAIL, 96.0, 30.0, 105.0, 38.0),
        # moves -- successful, spread across zones (drive xT non-degenerate)
        _row(_PASS, _SUCCESS, 10.0, 10.0, 40.0, 20.0),
        _row(_PASS, _SUCCESS, 40.0, 20.0, 70.0, 40.0),
        _row(_DRIBBLE, _SUCCESS, 70.0, 40.0, 78.0, 44.0),
        _row(_CROSS, _SUCCESS, 88.0, 8.0, 96.0, 34.0),
        _row(_PASS, _SUCCESS, 55.0, 50.0, 80.0, 55.0),
        _row(_PASS, _SUCCESS, 30.0, 60.0, 60.0, 62.0),
        # a FAILED move with a valid end (counts in the Singh denominator, not the numerator)
        _row(_PASS, _FAIL, 50.0, 30.0, 65.0, 25.0),
        # (a) valid start, NaN end -- counts in _action_prob move-count, DROPPED from the Singh denominator
        _row(_PASS, _SUCCESS, 45.0, 45.0, np.nan, np.nan),
        # (c) move whose end is off-pitch (end_x > 105) -- clamps to l-1 in the end cell
        _row(_PASS, _SUCCESS, 60.0, 30.0, 110.0, 34.0),
    ]
    return pd.DataFrame(rows)


def _fixture_b() -> pd.DataFrame:
    rows = [
        _row(_SHOT, _SUCCESS, 85.0, 40.0, 105.0, 34.0),
        _row(_PASS, _SUCCESS, 15.0, 55.0, 45.0, 50.0),
        _row(_DRIBBLE, _SUCCESS, 45.0, 50.0, 52.0, 48.0),
        _row(_CROSS, _FAIL, 90.0, 12.0, 98.0, 40.0),
        _row(_PASS, _SUCCESS, 25.0, 20.0, 55.0, 30.0),
    ]
    return pd.DataFrame(rows)


class _Counts(TypedDict):
    shot_counts: np.ndarray
    goal_counts: np.ndarray
    move_counts: np.ndarray
    transition_start_counts: np.ndarray
    transition_counts: np.ndarray


def _aggregate(actions: pd.DataFrame) -> _Counts:
    """The 5-count contract computed from raw actions with sk's exact filters (spec sec 4).

    Mirrors what the lakehouse computes in one Spark ``groupBy`` pass.
    """
    n = _L * _W
    shots = actions[actions.type_id == _SHOT]
    goals = shots[shots.result_id == _SUCCESS]
    moves = actions[actions.type_id.isin(_MOVE_TYPES)]
    moves_se = moves.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    succ = moves_se[moves_se.result_id == _SUCCESS]

    start_flat = _get_flat_indexes(succ.start_x, succ.start_y, _L, _W).to_numpy()
    end_flat = _get_flat_indexes(succ.end_x, succ.end_y, _L, _W).to_numpy()
    transition_counts = np.zeros((n, n), dtype=int)
    np.add.at(transition_counts, (start_flat, end_flat), 1)

    return _Counts(
        shot_counts=_count(shots.start_x, shots.start_y, _L, _W),
        goal_counts=_count(goals.start_x, goals.start_y, _L, _W),
        move_counts=_count(moves.start_x, moves.start_y, _L, _W),  # valid start only
        transition_start_counts=_count(moves_se.start_x, moves_se.start_y, _L, _W),  # valid start+end
        transition_counts=transition_counts,
    )


_MATRICES = ("scoring_prob_matrix", "shot_prob_matrix", "move_prob_matrix", "transition_matrix")


def _fit_actions(actions: pd.DataFrame) -> ExpectedThreat:
    return ExpectedThreat(l=_L, w=_W).fit(actions)


def _fit_counts(actions: pd.DataFrame) -> ExpectedThreat:
    return ExpectedThreat(l=_L, w=_W).fit_from_counts(**_aggregate(actions))


# --------------------------------------------------------------------------- #
# 1. Functional equivalence -- byte-identical matrices, with named boundaries
# --------------------------------------------------------------------------- #
def test_functional_equivalence_matrices_byte_identical() -> None:
    a = _fixture_a()
    xt_c, xt_a = _fit_counts(a), _fit_actions(a)
    for m in _MATRICES:
        assert np.array_equal(getattr(xt_c, m), getattr(xt_a, m)), f"{m} not byte-identical"
    assert np.allclose(xt_c.xT, xt_a.xT, equal_nan=True)


def test_case_a_valid_start_nan_end_move_is_the_divergence_trigger() -> None:
    """The valid-start/NaN-end pass counts in move_prob but NOT the Singh denominator; a single
    move_counts would break BOTH. Non-vacuity: assert the move actually differs the two populations."""
    a = _fixture_a()
    counts = _aggregate(a)
    # the NaN-end move makes move_counts strictly exceed transition_start_counts SOMEWHERE
    assert counts["move_counts"].sum() > counts["transition_start_counts"].sum()
    xt_c, xt_a = _fit_counts(a), _fit_actions(a)
    for m in ("move_prob_matrix", "transition_matrix"):
        assert np.array_equal(getattr(xt_c, m), getattr(xt_a, m)), m


# --------------------------------------------------------------------------- #
# 2. Additivity -- global == sum of per-competition counts
# --------------------------------------------------------------------------- #
def test_additivity() -> None:
    a, b = _fixture_a(), _fixture_b()
    ca, cb = _aggregate(a), _aggregate(b)
    summed = _Counts(
        shot_counts=ca["shot_counts"] + cb["shot_counts"],
        goal_counts=ca["goal_counts"] + cb["goal_counts"],
        move_counts=ca["move_counts"] + cb["move_counts"],
        transition_start_counts=ca["transition_start_counts"] + cb["transition_start_counts"],
        transition_counts=ca["transition_counts"] + cb["transition_counts"],
    )
    xt_sum = ExpectedThreat(l=_L, w=_W).fit_from_counts(**summed)
    xt_cat = _fit_actions(pd.concat([a, b], ignore_index=True))
    for m in _MATRICES:
        assert np.array_equal(getattr(xt_sum, m), getattr(xt_cat, m)), f"{m} not additive"
    assert np.allclose(xt_sum.xT, xt_cat.xT, equal_nan=True)


# --------------------------------------------------------------------------- #
# 3. Round-trip (ADR-100)
# --------------------------------------------------------------------------- #
def test_round_trip() -> None:
    xt = _fit_counts(_fixture_a())
    xt2 = ExpectedThreat.from_dict(xt.to_dict())
    for m in ("xT", "transition_matrix"):
        assert np.array_equal(getattr(xt, m), getattr(xt2, m)), m


# --------------------------------------------------------------------------- #
# 4. Zone-binning parity
# --------------------------------------------------------------------------- #
def test_zone_binning_parity() -> None:
    from silly_kicks.xthreat._grid import _get_cell_indexes

    xt = ExpectedThreat(l=_L, w=_W)
    xs = np.array([0.0, 52.5, 105.0, 120.0, 60.0])
    ys = np.array([0.0, 34.0, 68.0, -5.0, 10.0])
    xi_ref, yj_ref = _get_cell_indexes(pd.Series(xs), pd.Series(ys), _L, _W)
    flat_ref = _get_flat_indexes(pd.Series(xs), pd.Series(ys), _L, _W)
    xi, yj = xt.zones_of(xs, ys)
    assert np.array_equal(xi, xi_ref.to_numpy())
    assert np.array_equal(yj, yj_ref.to_numpy())
    assert np.array_equal(xt.flat_indexes_of(xs, ys), flat_ref.to_numpy())
    # clamp: x=120 -> l-1=15 ; y=-5 -> 0
    assert xi[3] == _L - 1
    assert yj[3] == 0


# --------------------------------------------------------------------------- #
# 5/6. Fail-closed
# --------------------------------------------------------------------------- #
def test_kde_params_raises() -> None:
    counts = _aggregate(_fixture_a())
    with pytest.raises(ValueError, match="KDE"):
        ExpectedThreat(l=_L, w=_W).fit_from_counts(**counts, params=KDEParams())


def test_shape_guard_raises() -> None:
    counts = _aggregate(_fixture_a())
    counts["shot_counts"] = np.zeros((_W, _L + 1), dtype=int)  # wrong shape
    with pytest.raises(ValueError):
        ExpectedThreat(l=_L, w=_W).fit_from_counts(**counts)


# --------------------------------------------------------------------------- #
# 7. fit(actions) parity oracle unaffected by the core extraction
# --------------------------------------------------------------------------- #
def test_membership_constants_match_spadlconfig() -> None:
    assert set(ExpectedThreat.MOVE_TYPE_NAMES) == {"pass", "dribble", "cross"}
    assert ExpectedThreat.SHOT_TYPE_NAME == "shot"
    assert all(t in spadlconfig.actiontype_id for t in ExpectedThreat.MOVE_TYPE_NAMES)
    assert ExpectedThreat.SHOT_TYPE_NAME in spadlconfig.actiontype_id
