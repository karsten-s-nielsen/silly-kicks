"""sportec.py pass/set-piece completion from native DFL play_evaluation (BUG 2 fix, 2026-06-09).

Previously every Play/set-piece was hard-wired result=success, zeroing failed-pass / failed-
goalkick labels (IDSSE goalkicks showed 100% success vs the real ~71%). DFL Play AND set-piece
events (GoalKick/FreeKick/CornerKick/ThrowIn, via the nested Play) carry an Evaluation
(`successfullyCompleted` / `successful` -> success; `unsuccessful` -> fail; NULL -> success,
conservative). Confirmed on 7 real DFL matches (lakehouse, 10,497 events).
"""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import sportec as sportec_mod

_FAIL = spadlconfig.result_id["fail"]
_SUCCESS = spadlconfig.result_id["success"]
_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_PASS = spadlconfig.actiontype_id["pass"]


def _ev(rows: list[dict]) -> pd.DataFrame:
    base = dict(match_id="M1", event_type="Play", period=1, player_id="P1", team="DFL-CLU-A", x=50.0, y=34.0)
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("event_id", f"e{i}")
        d.setdefault("timestamp_seconds", 10.0 + i)
        out.append(d)
    return pd.DataFrame(out)


def _convert(events):
    actions, _ = sportec_mod.convert_to_actions(events, home_team_id="DFL-CLU-A", home_team_start_left=True)
    return actions


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        ("unsuccessful", _FAIL),
        ("successfullyCompleted", _SUCCESS),
        ("successful", _SUCCESS),  # rare second success synonym (lakehouse: 23 occurrences)
        ("", _SUCCESS),  # NULL/missing -> conservative success
    ],
)
def test_pass_result_from_play_evaluation(evaluation, expected):
    actions = _convert(_ev([dict(event_type="Play", play_evaluation=evaluation)]))
    p = actions[actions["type_id"] == _PASS].iloc[0]
    assert p["result_id"] == expected


def test_goalkick_event_unsuccessful_is_fail():
    # formal DFL GoalKick event carries play_evaluation via its nested Play
    actions = _convert(_ev([dict(event_type="GoalKick", play_evaluation="unsuccessful")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _FAIL


def test_goalkick_event_completed_is_success():
    actions = _convert(_ev([dict(event_type="GoalKick", play_evaluation="successfullyCompleted")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _SUCCESS


def test_punt_synth_goalkick_inherits_parent_evaluation():
    # Play + punt qualifier -> keeper_pick_up + synthesized goalkick; the synth goalkick must
    # inherit the parent Play's play_evaluation (not a hard-wired success).
    actions = _convert(_ev([dict(event_type="Play", play_goal_keeper_action="punt", play_evaluation="unsuccessful")]))
    gk = actions[actions["type_id"] == _GOALKICK].iloc[0]
    assert gk["result_id"] == _FAIL
