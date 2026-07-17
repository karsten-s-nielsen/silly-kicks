"""resolve_next_touch_receiver (TF-49 / PR-S117) -- packing-agnostic next-touch resolution."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import resolve_next_touch_receiver

_PASS = spadlconfig.actiontype_id["pass"]
_NON_ACTION = spadlconfig.actiontype_id["non_action"]
_DRIBBLE = spadlconfig.actiontype_id["dribble"]
_FOUL = spadlconfig.actiontype_id["foul"]


def _actions(rows, id_dtype="int64"):
    df = pd.DataFrame(rows)
    for col, default in [("game_id", 1), ("period_id", 1)]:
        if col not in df.columns:
            df[col] = default
    df["player_id"] = df["player_id"].astype(pd.api.types.pandas_dtype(id_dtype))
    return df


def test_same_team_next_touch_resolves():
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": 11, "type_id": _DRIBBLE},
        ]
    )
    out = resolve_next_touch_receiver(a)
    assert out.iloc[0] == 11


def test_opponent_next_is_na():
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 2, "player_id": 20, "type_id": _PASS},
        ]
    )
    assert pd.isna(resolve_next_touch_receiver(a).iloc[0])


def test_non_action_rows_are_skipped():
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": pd.NA, "type_id": _NON_ACTION},
            {"action_id": 2, "team_id": 1, "player_id": 12, "type_id": _PASS},
        ],
        id_dtype="Int64",
    )
    assert resolve_next_touch_receiver(a).iloc[0] == 12


def test_period_end_is_na():
    a = _actions([{"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS}])
    assert pd.isna(resolve_next_touch_receiver(a).iloc[0])


def test_same_team_foul_row_is_not_the_receiver():
    """Execution-review D1: an off-ball same-team foul between pass and reception --
    the fouler never touched the ball and must not resolve as the receiver."""
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": 9, "type_id": _FOUL},
            {"action_id": 2, "team_id": 1, "player_id": 7, "type_id": _PASS},
        ]
    )
    assert resolve_next_touch_receiver(a).iloc[0] == 7


def test_opponent_advantage_foul_does_not_block_resolution():
    """Execution-review D1: an advantage-played OPPONENT foul between pass and the
    genuine same-team reception must not degrade the receiver to <NA>."""
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 2, "player_id": 90, "type_id": _FOUL},
            {"action_id": 2, "team_id": 1, "player_id": 7, "type_id": _PASS},
        ]
    )
    assert resolve_next_touch_receiver(a).iloc[0] == 7


def test_two_touches_after_a_skip_stay_aligned():
    """Review blocker 1 (adversarial): rows AFTER a skipped non_action must not shift."""
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": pd.NA, "type_id": _NON_ACTION},
            {"action_id": 2, "team_id": 1, "player_id": 12, "type_id": _PASS},
            {"action_id": 3, "team_id": 1, "player_id": 13, "type_id": _PASS},
        ],
        id_dtype="Int64",
    )
    out = resolve_next_touch_receiver(a)
    assert out.iloc[0] == 12
    assert out.iloc[2] == 13


def test_non_rangeindex_input_resolves():
    """Review blocker 1 (adversarial): a pre-filtered/sliced caller must still resolve."""
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": 11, "type_id": _PASS},
        ]
    )
    a.index = pd.Index([10, 11])
    out = resolve_next_touch_receiver(a)
    assert list(out.index) == [10, 11]
    assert out.loc[10] == 11


def test_period_boundary_not_crossed():
    a = _actions(
        [
            {"action_id": 0, "period_id": 1, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "period_id": 2, "team_id": 1, "player_id": 11, "type_id": _PASS},
        ]
    )
    assert pd.isna(resolve_next_touch_receiver(a).iloc[0])


@pytest.mark.parametrize("dtype", ["int64", "Int64", "object", "float64"])
def test_dtype_trio_no_float_upcast(dtype):
    """F5: plain int64 AND NaN-coded float64 pre-convert to Int64; never float64
    (the '366.0' class). float64 sources are the h5-fixture / ADR-028-fixture shape."""
    a = _actions(
        [
            {"action_id": 0, "team_id": 1, "player_id": 10, "type_id": _PASS},
            {"action_id": 1, "team_id": 1, "player_id": 11, "type_id": _PASS},
            {"action_id": 2, "team_id": 2, "player_id": 20, "type_id": _PASS},
        ],
        id_dtype=dtype,
    )
    out = resolve_next_touch_receiver(a)
    assert out.dtype == ("object" if dtype == "object" else "Int64")
    assert out.iloc[0] == (11 if dtype != "object" else a["player_id"].iloc[1])
    assert pd.isna(out.iloc[1])  # opponent next
