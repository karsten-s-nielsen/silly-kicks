"""Convention-pinning lock tests: SPADL time_seconds is PERIOD-RELATIVE.

silly_kicks' canonical convention is that ``time_seconds`` resets to 0 at the
start of each period (NOT absolute match-clock). These tests turn that prose
contract (spec 2026-06-04 4.1 / ADR-017) into enforced behavior for the
converters whose time arithmetic the library OWNS: a future refactor that
emits absolute, continuous-across-periods time makes them fail.

Scope: Opta, StatsBomb and Gradient Sports. Since ADR-133 r8 (spec 2026-11-13
6.6 B) the Gradient Sports converter computes time_seconds itself, as the
possession event's ``event_time`` minus its period's kickoff ``start_time``; it no
longer passes the input ``startGameClock`` (a match clock) through. See ADR-017.
"""

import pandas as pd
import pytest

from silly_kicks.spadl import gradientsports, opta, statsbomb
from tests.spadl.test_gradientsports_event_time import _frame, _row


def _opta_event(event_id, period_id, minute, second):
    # Mirrors the event dict in tests/spadl/test_opta.py (proven-accepted shape).
    return {
        "game_id": 318175,
        "event_id": event_id,
        "type_id": 1,
        "period_id": period_id,
        "minute": minute,
        "second": second,
        "timestamp": "2010-01-27 19:47:14",
        "player_id": 8786,
        "team_id": 157,
        "outcome": True,
        "start_x": 50.0,
        "start_y": 50.0,
        "end_x": 60.0,
        "end_y": 50.0,
        "assist": False,
        "keypass": False,
        "qualifiers": {1: True},
        "type_name": "pass",
    }


def test_opta_time_seconds_is_period_relative():
    # P1 at 02:14 (134s). P2 at 47:00 absolute = 02:00 into the 2nd half (120s relative).
    events = pd.DataFrame([_opta_event(1, 1, 2, 14), _opta_event(2, 2, 47, 0)])
    actions, _ = opta.convert_to_actions(events, home_team_id=157)
    p1 = actions.loc[actions["period_id"] == 1, "time_seconds"].iloc[0]
    p2 = actions.loc[actions["period_id"] == 2, "time_seconds"].iloc[0]
    assert p1 == 134.0
    assert p2 == 120.0  # period-relative: 47min - 45min = 2min
    assert p2 < p1  # absolute would give 2820s >> 134s; period-relative resets


def _sb_event(event_id, period_id, timestamp):
    # Mirrors _make_statsbomb_events() in tests/spadl/test_statsbomb.py (minimal accepted shape).
    return {
        "game_id": 1,
        "event_id": event_id,
        "period_id": period_id,
        "timestamp": timestamp,
        "team_id": 100,
        "player_id": 200,
        "type_name": "Pass",
        "location": [60.0, 40.0],
        "extra": {
            "pass": {"end_location": [70.0, 40.0], "outcome": {"name": "Complete"}, "height": {"name": "Ground Pass"}}
        },
    }


def test_statsbomb_time_seconds_is_period_relative():
    events = pd.DataFrame(
        [
            _sb_event("abc-1", 1, "00:02:14.000"),
            _sb_event("abc-2", 2, "00:01:00.000"),  # 1 min into the 2nd half
        ]
    )
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    p1 = actions.loc[actions["period_id"] == 1, "time_seconds"].iloc[0]
    p2 = actions.loc[actions["period_id"] == 2, "time_seconds"].iloc[0]
    assert p1 == 134.0
    assert p2 == 60.0  # period-relative; an absolute clock would be ~2760s
    assert p2 < p1


def test_gradientsports_time_seconds_is_period_relative():
    # Video times as the raw feed has them: P1 kickoff at 186.12 s, a pass 134 s later; P2 kickoff at 3042.68 s, a
    # pass 120 s later. The input startGameClock (`clock`) is the match clock (2820 s in P2) and must not reach the
    # output.
    rows = [
        _row(-1, 1, 186.12, ge="FIRSTKICKOFF", pe=None),
        _row(1, 1, 186.12 + 134.0, clock=134.0),
        _row(-2, 2, 3042.68, ge="SECONDKICKOFF", pe=None),
        _row(2, 2, 3042.68 + 120.0, clock=2820.0),
    ]
    actions, _ = gradientsports.convert_to_actions(
        _frame(rows), home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True
    )
    p1 = actions.loc[actions["period_id"] == 1, "time_seconds"].iloc[0]
    p2 = actions.loc[actions["period_id"] == 2, "time_seconds"].iloc[0]
    assert p1 == pytest.approx(134.0)
    assert p2 == pytest.approx(120.0)  # period-relative; the input match clock would give 2820 s
    assert p2 < p1
