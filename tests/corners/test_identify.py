"""TF-46 identify_corners: provider-aware corner anchoring."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import CORNER_IDENTIFY_COLUMNS, identify_corners

_CORNER_CROSSED = 5
_CORNER_SHORT = 6
_PASS = 0


def _actions(rows: list[dict]) -> pd.DataFrame:
    base = {
        "game_id": "g1",
        "original_event_id": "e",
        "action_id": 0,
        "period_id": 1,
        "time_seconds": 0.0,
        "team_id": "t1",
        "player_id": "p1",
        "start_x": 1.0,
        "start_y": 0.5,
        "end_x": 20.0,
        "end_y": 34.0,
        "type_id": _PASS,
        "result_id": 1,
        "bodypart_id": 0,
    }
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i
        out.append(d)
    return pd.DataFrame(out)


def test_spadl_corner_crossed_is_direct():
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2, "time_seconds": 100.0}])
    out = identify_corners(acts, provider="sportec")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["corner_kind"] == "direct"
    assert row["corner_source"] == "spadl_corner"
    assert row["t_kick"] == 100.0
    assert row["attacking_team_id"] == "t1"
    assert list(out.columns) == list(CORNER_IDENTIFY_COLUMNS)


def test_spadl_corner_short_is_short():
    acts = _actions([{"type_id": _CORNER_SHORT, "start_x": 104.5, "start_y": 67.8}])
    out = identify_corners(acts, provider="sportec")
    assert out.iloc[0]["corner_kind"] == "short"


def test_corner_side_from_kick_y():
    acts = _actions(
        [
            {"type_id": _CORNER_CROSSED, "start_y": 1.0, "time_seconds": 10.0},
            {"type_id": _CORNER_CROSSED, "start_y": 67.0, "time_seconds": 20.0},
        ]
    )
    out = identify_corners(acts, provider="sportec").sort_values("t_kick")
    assert set(out["corner_side"]) == {"left", "right"}


def test_corner_id_is_per_game_chronological():
    acts = _actions(
        [
            {"type_id": _CORNER_CROSSED, "time_seconds": 50.0},
            {"type_id": _PASS, "time_seconds": 55.0},
            {"type_id": _CORNER_CROSSED, "time_seconds": 60.0},
        ]
    )
    out = identify_corners(acts, provider="sportec").sort_values("t_kick")
    assert list(out["corner_id"]) == [0, 1]


def test_skillcorner_source_token():
    acts = _actions([{"type_id": _CORNER_CROSSED}])
    out = identify_corners(acts, provider="skillcorner")
    assert out.iloc[0]["corner_source"] == "skillcorner_gi"


def test_gi_refinement_marks_unclear():
    acts = _actions([{"type_id": _CORNER_CROSSED, "period_id": 1, "time_seconds": 100.0}])
    gi = pd.DataFrame(
        {
            "game_id": ["g1"],
            "period_id": [1],
            "time_seconds": [100.0],
            "game_interruption_before": ["corner_for"],
            "start_type": ["pass_reception"],  # not corner_reception/interception -> unclear
        }
    )
    out = identify_corners(acts, provider="skillcorner", gi=gi)
    assert out.iloc[0]["corner_kind"] == "unclear"


def test_gi_reception_stays_direct():
    acts = _actions([{"type_id": _CORNER_CROSSED, "period_id": 1, "time_seconds": 100.0}])
    gi = pd.DataFrame(
        {
            "game_id": ["g1"],
            "period_id": [1],
            "time_seconds": [100.0],
            "game_interruption_before": ["corner_for"],
            "start_type": ["corner_reception"],
        }
    )
    out = identify_corners(acts, provider="skillcorner", gi=gi)
    assert out.iloc[0]["corner_kind"] == "direct"


def test_no_corners_returns_empty_schema():
    out = identify_corners(_actions([{"type_id": _PASS}]), provider="sportec")
    assert out.empty
    assert list(out.columns) == list(CORNER_IDENTIFY_COLUMNS)


def test_missing_required_column_raises():
    acts = _actions([{"type_id": _CORNER_CROSSED}]).drop(columns=["start_y"])
    try:
        identify_corners(acts, provider="sportec")
    except KeyError as e:
        assert "start_y" in str(e)
    else:
        raise AssertionError("expected KeyError for missing start_y")


def test_pure_does_not_mutate_input():
    acts = _actions([{"type_id": _CORNER_CROSSED}])
    before = acts.copy(deep=True)
    identify_corners(acts, provider="sportec")
    pd.testing.assert_frame_equal(acts, before)
