"""The idsse penalty row (R57, action-time repair spec §4.16).

The DFL ``<Penalty>`` event holds a nested ``<ShotAtGoal>``; the bronze parser flattens the nested shot into
``shot_*`` columns on the Penalty row. The converter types that row ``shot_penalty``; a Penalty with no nested shot
is excluded by name.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec as sportec_mod

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _events(*penalty_rows: dict) -> pd.DataFrame:
    base = {
        "match_id": "J03WMX",
        "event_id": "e0",
        "event_type": "Play",
        "period": 1,
        "timestamp_seconds": 10.0,
        "player_id": "DFL-OBJ-0001",
        "team": "DFL-CLU-A",
        "x": 50.0,
        "y": 34.0,
    }
    rows = [base]
    for i, extra in enumerate(penalty_rows, start=1):
        rows.append({**base, "event_id": f"p{i}", "timestamp_seconds": 60.0 + i, "x": 94.0, "y": 34.0, **extra})
    return pd.DataFrame(rows)


def _convert(events: pd.DataFrame):
    return sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A", home_team_start_left=True)


def test_scored_penalty_is_shot_penalty_success():
    actions, report = _convert(
        _events({"event_type": "Penalty", "penalty_team": "DFL-CLU-A", "shot_outcome_type": "successful"})
    )
    pen = actions[actions["original_event_id"] == "p1"]
    assert pen["type_id"].tolist() == [_T["shot_penalty"]]
    assert pen["result_id"].tolist() == [_R["success"]]
    assert pen["time_source"].tolist() == ["event"]
    assert "Penalty" not in report.unrecognized_counts


def test_missed_penalty_is_shot_penalty_fail():
    actions, _ = _convert(_events({"event_type": "Penalty", "penalty_team": "DFL-CLU-A", "shot_outcome_type": "saved"}))
    pen = actions[actions["original_event_id"] == "p1"]
    assert pen["type_id"].tolist() == [_T["shot_penalty"]]
    assert pen["result_id"].tolist() == [_R["fail"]]


def test_penalty_without_nested_shot_is_excluded_by_name():
    actions, report = _convert(_events({"event_type": "Penalty", "penalty_team": "DFL-CLU-A"}))
    assert "p1" not in set(actions["original_event_id"])
    assert report.excluded_counts.get("Penalty") == 1
    assert "Penalty" not in report.unrecognized_counts


def test_penalty_player_is_the_row_player():
    actions, _ = _convert(
        _events(
            {
                "event_type": "Penalty",
                "penalty_team": "DFL-CLU-A",
                "shot_outcome_type": "successful",
                "player_id": "DFL-OBJ-SHOOTER",
            }
        )
    )
    pen = actions[actions["original_event_id"] == "p1"]
    assert pen["player_id"].tolist() == ["DFL-OBJ-SHOOTER"]


def test_penalty_player_is_nested_shooter():
    # the parsed Penalty row has no player of its own; the nested ShotAtGoal's Player is `shot_player` (spec 4.16)
    actions, _ = _convert(
        _events(
            {
                "event_type": "Penalty",
                "penalty_team": "DFL-CLU-A",
                "shot_outcome_type": "successful",
                "player_id": "",
                "shot_player": "DFL-OBJ-SHOOTER",
            }
        )
    )
    pen = actions[actions["original_event_id"] == "p1"]
    assert pen["player_id"].tolist() == ["DFL-OBJ-SHOOTER"]


def test_penalty_with_a_nested_shot_but_no_outcome_warns_and_names_the_row():
    # bar A23 point 7: bronze from a parser before R63 (or a provider file with no outcome tag) has the nested shot
    # (shot_player / shot_team) but no shot_outcome_type. The row cannot be typed, and the converter says so.
    with pytest.warns(UserWarning, match=r"p1.*nested shot.*no outcome.*parse_dfl_events"):
        actions, report = _convert(
            _events(
                {
                    "event_type": "Penalty",
                    "penalty_team": "DFL-CLU-A",
                    "shot_player": "DFL-OBJ-SHOOTER",
                    "shot_team": "DFL-CLU-A",
                }
            )
        )
    assert "p1" not in set(actions["original_event_id"])
    assert report.excluded_counts.get("Penalty") == 1


def test_penalty_without_nested_shot_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _convert(_events({"event_type": "Penalty", "penalty_team": "DFL-CLU-A"}))


def test_committed_fixture_penalty_through_native_shaping_is_the_home_teams_goal():
    # R64 / bar A26 point 6: the fixture is the parser's bronze of J03WMX (scripts/build_idsse_per_period_fixture.py).
    # Through shape_events_to_native the scored penalty (event 18226500001426) is a shot_penalty of the home team by
    # the nested shooter, at the attacked end.
    from tests._idsse_fixtures import idsse_fixture_native

    native = idsse_fixture_native("per_period_match")
    actions, _ = sportec_mod.convert_to_actions(native, home_team_id="home", home_team_start_left=False)
    pen = actions[actions["original_event_id"].astype(str) == "18226500001426"]
    assert pen["type_id"].tolist() == [_T["shot_penalty"]]
    assert pen["result_id"].tolist() == [_R["success"]]
    assert pen["team_id"].tolist() == ["home"]
    assert pen["player_id"].tolist() == ["DFL-OBJ-J01BGM"]
    assert pen["start_x"].iloc[0] > 52.5 + 30.0  # the attacked end (SPADL action-LTR); the mark is 11 m out
