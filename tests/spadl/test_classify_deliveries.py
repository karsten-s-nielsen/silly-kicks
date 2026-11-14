"""The shared delivery rule (spec 2026-11-13-cross-definition-design §6.1 / §6.1b, ADR-133).

Every band is tested from both sides. Fixtures are canonical SPADL LTR rows in final chronological order with no
synthetic dribbles (the function's precondition).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import CROSS_START_X, classify_deliveries

_ID = spadlconfig.actiontype_id
_NAME = dict(enumerate(spadlconfig.actiontypes))
W = spadlconfig.field_width
PA_X = spadlconfig.field_length - spadlconfig.penalty_area_depth
WIDE_LO = W / 2 - spadlconfig.penalty_area_half_width  # 13.84
WIDE_HI = W / 2 + spadlconfig.penalty_area_half_width  # 54.16


def _actions(rows: list[tuple]) -> pd.DataFrame:
    """rows: (type_name, start_x, start_y[, team_id[, period_id]]); team defaults to "A", period to 1."""
    recs = []
    for i, r in enumerate(rows):
        name, x, y = r[0], r[1], r[2]
        team = r[3] if len(r) > 3 else "A"
        period = r[4] if len(r) > 4 else 1
        recs.append(
            {
                "game_id": 1,
                "period_id": period,
                "action_id": i,
                "time_seconds": float(i),
                "team_id": team,
                "player_id": 100 + i,
                "type_id": _ID[name],
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
            }
        )
    return pd.DataFrame(recs)


def _types(actions: pd.DataFrame) -> list[str]:
    return [_NAME[t] for t in classify_deliveries(actions)["type_id"]]


def _first(rows: list[tuple]) -> str:
    return _types(_actions(rows))[0]


def test_wide_advanced_delivery_into_box_is_cross():
    assert _first([("pass", 80.0, 5.0), ("shot", 95.0, 34.0)]) == "cross"


def test_start_line_both_sides():
    assert _first([("pass", CROSS_START_X - 0.01, 5.0), ("shot", 95.0, 34.0)]) == "pass"
    assert _first([("pass", CROSS_START_X, 5.0), ("shot", 95.0, 34.0)]) == "cross"


@pytest.mark.parametrize(("inside", "outside"), [(WIDE_LO - 0.01, WIDE_LO), (WIDE_HI + 0.01, WIDE_HI)])
def test_wide_edge_both_sides(inside, outside):
    assert _first([("pass", 80.0, inside), ("shot", 95.0, 34.0)]) == "cross"
    assert _first([("pass", 80.0, outside), ("shot", 95.0, 34.0)]) == "pass"


def test_box_edges_both_sides():
    assert _first([("pass", 80.0, 5.0), ("shot", PA_X, 34.0)]) == "cross"
    assert _first([("pass", 80.0, 5.0), ("shot", PA_X - 0.01, 34.0)]) == "pass"
    assert _first([("pass", 80.0, 5.0), ("shot", 95.0, WIDE_LO)]) == "cross"
    assert _first([("pass", 80.0, 5.0), ("shot", 95.0, WIDE_LO - 0.01)]) == "pass"


def test_opponent_next_action_is_mirrored():
    # An opponent clearance at (10, 34) in ITS frame is in the passer's box.
    assert _first([("pass", 80.0, 5.0, "A"), ("clearance", 10.0, 34.0, "B")]) == "cross"
    assert _first([("pass", 80.0, 5.0, "A"), ("clearance", 95.0, 34.0, "B")]) == "pass"


def test_skips_foul_and_non_action():
    rows = [("pass", 80.0, 5.0), ("foul", 40.0, 40.0, "B"), ("non_action", 0.0, 0.0), ("shot", 95.0, 34.0)]
    assert _first(rows) == "cross"


def test_native_dribble_counts_as_next_action():
    assert _first([("pass", 80.0, 5.0), ("dribble", 80.0, 20.0), ("shot", 95.0, 34.0)]) == "pass"


@pytest.mark.parametrize("restart", ["goalkick", "corner_crossed", "throw_in"])
def test_restart_next_action_is_not_a_delivery(restart):
    # A restart's start can sit in the passer's box (a goal kick mirrors into it); it never delivers.
    assert _first([("pass", 80.0, 5.0, "A"), (restart, 100.0, 34.0, "B")]) == "pass"
    assert _first([("pass", 80.0, 5.0, "A"), (restart, 5.0, 34.0, "B")]) == "pass"


def test_foul_then_penalty_is_not_a_delivery():
    rows = [("pass", 80.0, 5.0, "A"), ("foul", 10.0, 34.0, "B"), ("shot_penalty", 94.0, 34.0, "A")]
    assert _first(rows) == "pass"


def test_native_cross_with_short_delivery_becomes_pass():
    assert _first([("cross", 80.0, 5.0), ("pass", 82.0, 8.0)]) == "pass"


@pytest.mark.parametrize(("base", "crossed", "short"), [("corner", "corner_crossed", "corner_short")])
def test_corner_delivery_both_sides(base, crossed, short):
    for native in (crossed, short):
        assert _first([(native, 105.0, 0.0), ("shot", PA_X, 34.0)]) == crossed
        assert _first([(native, 105.0, 0.0), ("pass", PA_X - 0.01, 34.0)]) == short
        assert _first([(native, 105.0, 0.0, "A"), ("goalkick", 100.0, 34.0, "B")]) == short


def test_freekick_delivery_both_sides():
    for native in ("freekick_crossed", "freekick_short"):
        assert _first([(native, 70.0, 34.0), ("shot", PA_X, 34.0)]) == "freekick_crossed"  # central, no wide rule
        assert _first([(native, 70.0, 34.0), ("pass", PA_X - 0.01, 34.0)]) == "freekick_short"
        assert _first([(native, 70.0, 34.0, "A"), ("throw_in", 60.0, 0.0, "B")]) == "freekick_short"


@pytest.mark.parametrize(
    "name", ["throw_in", "goalkick", "shot_freekick", "shot_penalty", "shot", "dribble", "clearance"]
)
def test_unchanged_types(name):
    assert _first([(name, 80.0, 5.0), ("shot", 95.0, 34.0)]) == name


def test_last_row_in_period_is_not_a_delivery():
    assert _types(_actions([("shot", 95.0, 34.0), ("cross", 80.0, 5.0)])) == ["shot", "pass"]
    assert _types(_actions([("corner_crossed", 105.0, 0.0)])) == ["corner_short"]


def test_nan_coordinates_never_deliver():
    assert _first([("pass", np.nan, 5.0), ("shot", 95.0, 34.0)]) == "pass"
    assert _first([("pass", 80.0, np.nan), ("shot", 95.0, 34.0)]) == "pass"
    assert _first([("pass", 80.0, 5.0), ("shot", np.nan, 34.0)]) == "pass"
    assert _first([("corner_crossed", 105.0, 0.0), ("shot", 95.0, np.nan)]) == "corner_short"


@pytest.mark.parametrize("cast", ["int64", "object", "category"])
def test_team_id_dtype_invariance(cast):
    a = _actions(
        [("pass", 80.0, 5.0, 7), ("clearance", 10.0, 34.0, 9), ("pass", 80.0, 5.0, 9), ("shot", 95.0, 34.0, 9)]
    )
    a["team_id"] = a["team_id"].astype(cast)
    assert _types(a) == ["cross", "clearance", "cross", "shot"]


def test_extra_time_and_shootout_periods():
    rows = [("pass", 80.0, 5.0, "A", 3), ("shot", 95.0, 34.0, "A", 3), ("pass", 80.0, 5.0, "A", 4)]
    rows += [("shot", 95.0, 34.0, "A", 4), ("shot_penalty", 94.0, 34.0, "A", 5), ("shot_penalty", 94.0, 34.0, "B", 5)]
    assert _types(_actions(rows)) == ["cross", "shot", "cross", "shot", "shot_penalty", "shot_penalty"]


def test_period_boundary():
    assert _first([("pass", 80.0, 5.0, "A", 1), ("shot", 95.0, 34.0, "A", 2)]) == "pass"


def test_pure():
    a = _actions([("pass", 80.0, 5.0), ("shot", 95.0, 34.0)])
    before = a.copy(deep=True)
    out = classify_deliveries(a)
    pd.testing.assert_frame_equal(a, before)
    assert out is not a


def test_only_type_id_changes():
    a = _actions([("cross", 80.0, 5.0), ("pass", 82.0, 8.0), ("pass", 80.0, 5.0), ("shot", 95.0, 34.0)])
    out = classify_deliveries(a)
    pd.testing.assert_frame_equal(out.drop(columns="type_id"), a.drop(columns="type_id"))
    pd.testing.assert_index_equal(out.index, a.index)


def test_cross_start_x_override():
    a = _actions([("pass", 60.0, 5.0), ("shot", 95.0, 34.0)])
    assert _NAME[classify_deliveries(a, cross_start_x=60.0)["type_id"].iloc[0]] == "cross"
    assert _NAME[classify_deliveries(a, cross_start_x=60.01)["type_id"].iloc[0]] == "pass"


def test_empty_frame():
    a = _actions([("pass", 80.0, 5.0)]).iloc[0:0]
    assert classify_deliveries(a).empty
