"""Convention-pinning lock tests: SPADL time_seconds is PERIOD-RELATIVE.

silly_kicks' canonical convention is that ``time_seconds`` resets to 0 at the
start of each period (NOT absolute match-clock). These tests turn that prose
contract (spec 2026-06-04 4.1 / ADR-017) into enforced behavior for the
converters whose time arithmetic the library OWNS: a future refactor that
emits absolute, continuous-across-periods time makes them fail.

Scope: Opta + StatsBomb only. GradientSports time_seconds is a verbatim
pass-through (gradientsports.py:416) originating upstream in the lakehouse, so
it is guarded lakehouse-side (validate_time_base), not here. See ADR-017.
"""

import pandas as pd

from silly_kicks.spadl import opta, statsbomb


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
