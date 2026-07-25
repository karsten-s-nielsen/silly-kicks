import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking.defensive_credit._chaining import (
    recovery_after_pass,
    resulting_shot_in_possession,
    with_possessions,
)


def _stream(rows):
    base: dict[str, object] = dict(
        game_id="g1",
        period_id=1,
        bodypart_id=spadlconfig.bodypart_id["foot"],
        start_x=60.0,
        start_y=34.0,
        end_x=70.0,
        end_y=34.0,
    )
    out = []
    for i, r in enumerate(rows):
        d: dict[str, object] = dict(base)
        d.update(action_id=i, time_seconds=float(i), **r)
        d["type_id"] = spadlconfig.actiontype_id[str(d.pop("type_name"))]
        d["result_id"] = spadlconfig.result_id[str(d.pop("result_name"))]
        out.append(d)
    return with_possessions(pd.DataFrame(out))


def test_resulting_shot_found_in_same_possession():
    actions = _stream(
        [
            {"type_name": "pass", "result_name": "success", "team_id": 10, "player_id": 1},
            {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
            {"type_name": "shot", "result_name": "fail", "team_id": 10, "player_id": 3},
        ]
    )
    shot = resulting_shot_in_possession(actions, 0, attacking_team_id=10, max_actions=10)
    assert shot is not None
    assert shot["action_id"] == 2


def test_resulting_shot_none_when_no_shot():
    actions = _stream(
        [
            {"type_name": "pass", "result_name": "success", "team_id": 10, "player_id": 1},
            {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 2},
        ]
    )
    assert resulting_shot_in_possession(actions, 0, attacking_team_id=10, max_actions=10) is None


def test_recovery_after_failed_pass():
    actions = _stream(
        [
            {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 1},
            {"type_name": "interception", "result_name": "success", "team_id": 20, "player_id": 99},
        ]
    )
    rec = recovery_after_pass(actions, 0, max_actions=3)
    assert rec is not None
    assert rec["player_id"] == 99


def test_recovery_none_beyond_cap():
    actions = _stream(
        [
            {"type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 1},
            {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
            {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
            {"type_name": "dribble", "result_name": "success", "team_id": 10, "player_id": 2},
            {"type_name": "interception", "result_name": "success", "team_id": 20, "player_id": 99},
        ]
    )
    assert recovery_after_pass(actions, 0, max_actions=3) is None


def _multi_game_stream(rows):
    """Like _stream but each row supplies its own game_id/period_id (multi-game batch)."""
    out = []
    for i, r in enumerate(rows):
        d: dict[str, object] = dict(
            time_seconds=float(i),
            bodypart_id=spadlconfig.bodypart_id["foot"],
            start_x=60.0,
            start_y=34.0,
            end_x=70.0,
            end_y=34.0,
            action_id=i,
        )
        d.update(r)
        d["type_id"] = spadlconfig.actiontype_id[str(d.pop("type_name"))]
        d["result_id"] = spadlconfig.result_id[str(d.pop("result_name"))]
        out.append(d)
    return with_possessions(pd.DataFrame(out))


def test_recovery_does_not_cross_game_boundary():
    # Game 1: a failed pass at its LAST action. Game 2 opens (within max_actions rows) with an
    # opponent action -- a DIFFERENT match, so it must NOT count as a recovery (B2 / spec section 7).
    actions = _multi_game_stream(
        [
            {
                "game_id": 1,
                "period_id": 1,
                "type_name": "pass",
                "result_name": "success",
                "team_id": 10,
                "player_id": 1,
            },
            {"game_id": 1, "period_id": 1, "type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 2},
            {
                "game_id": 2,
                "period_id": 1,
                "type_name": "interception",
                "result_name": "success",
                "team_id": 20,
                "player_id": 99,
            },
            {
                "game_id": 2,
                "period_id": 1,
                "type_name": "dribble",
                "result_name": "success",
                "team_id": 20,
                "player_id": 98,
            },
        ]
    )
    # pass_idx = 1 (game 1's failed pass); positions 2,3 are game 2.
    assert recovery_after_pass(actions, pass_idx=1, max_actions=3) is None


def test_recovery_still_fires_within_same_game():
    # NON-VACUITY (N1): the boundary fix must NOT kill a legitimate same-game/period recovery.
    actions = _multi_game_stream(
        [
            {"game_id": 1, "period_id": 1, "type_name": "pass", "result_name": "fail", "team_id": 10, "player_id": 1},
            {
                "game_id": 1,
                "period_id": 1,
                "type_name": "interception",
                "result_name": "success",
                "team_id": 20,
                "player_id": 99,
            },
        ]
    )
    rec = recovery_after_pass(actions, pass_idx=0, max_actions=3)
    assert rec is not None
    assert rec["player_id"] == 99
