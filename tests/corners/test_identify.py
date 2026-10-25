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
    assert out.iloc[0]["corner_source"] == "skillcorner_marker"


def test_gi_param_deprecated_and_ignored():
    import warnings as _w

    acts = _actions([{"type_id": _CORNER_CROSSED, "period_id": 1, "time_seconds": 100.0}])
    gi = pd.DataFrame(
        {
            "game_id": ["g1"],
            "period_id": [1],
            "time_seconds": [100.0],
            "game_interruption_before": ["corner_for"],
            "start_type": ["pass_reception"],  # once marked "unclear"; the refinement is retired (B1)
        }
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = identify_corners(acts, provider="skillcorner", gi=gi)
    assert any(issubclass(x.category, DeprecationWarning) for x in caught)
    assert out.iloc[0]["corner_kind"] != "unclear"  # gi ignored, never refines to unclear


def test_nan_start_y_yields_nan_side():
    # Non-vacuous: a NaN start_y was silently "right" before the section 9.4 fix.
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": float("nan")}])
    out = identify_corners(acts, provider="sportec")
    assert pd.isna(out.iloc[0]["corner_side"])


def test_derived_corner_has_nan_kick_fields():
    # A derived (events-only) corner carries a NaN start -> NaN t_kick/kick_x/kick_y/corner_side.
    acts = _actions(
        [{"type_id": _CORNER_CROSSED, "start_x": float("nan"), "start_y": float("nan"), "time_seconds": 100.0}]
    )
    row = identify_corners(acts, provider="skillcorner").iloc[0]
    assert pd.isna(row["t_kick"]) and pd.isna(row["kick_x"]) and pd.isna(row["kick_y"])
    assert pd.isna(row["corner_side"])


def test_type_source_travels_from_actions():
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2}])
    acts["type_source"] = "default"
    out = identify_corners(acts, provider="skillcorner")
    assert out.iloc[0]["type_source"] == "default"


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


def test_event_coded_first_touch_on_gradientsports_only():
    # Spec 9.3: GS events code the first touch -> the next action by a DIFFERENT player within the 6 s window.
    rows = [
        {"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2, "time_seconds": 100.0, "player_id": "pT"},
        {"type_id": _PASS, "time_seconds": 100.4, "player_id": "pT", "team_id": "t1"},  # the taker again: skipped
        {"type_id": _PASS, "time_seconds": 101.3, "player_id": "pD", "team_id": "t2"},
        {"type_id": _PASS, "time_seconds": 101.9, "player_id": "pE", "team_id": "t1"},
    ]
    gs = identify_corners(_actions(rows), provider="gradientsports").iloc[0]
    assert (gs["event_first_touch_time"], gs["event_first_touch_player_id"], gs["event_first_touch_team_id"]) == (
        101.3,
        "pD",
        "t2",
    )
    sportec = identify_corners(_actions(rows), provider="sportec").iloc[0]
    assert pd.isna(sportec["event_first_touch_time"]) and pd.isna(sportec["event_first_touch_player_id"])


def test_event_coded_first_touch_outside_the_window_is_nan():
    rows = [
        {"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2, "time_seconds": 100.0, "player_id": "pT"},
        {"type_id": _PASS, "time_seconds": 106.5, "player_id": "pD", "team_id": "t2"},  # > t_kick + 6 s
    ]
    assert pd.isna(identify_corners(_actions(rows), provider="gradientsports").iloc[0]["event_first_touch_time"])
