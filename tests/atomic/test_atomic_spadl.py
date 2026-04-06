import pandas as pd

import silly_kicks.atomic.spadl.config as atomicconfig
import silly_kicks.spadl.config as spadlcfg
from silly_kicks.atomic.spadl.base import convert_to_atomic


def test_blocked_shot_produces_out() -> None:
    """Bug #831: Blocked/saved shots must produce an atomic 'out' action."""
    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "original_event_id": [100, 101],
            "period_id": [1, 1],
            "action_id": [0, 1],
            "time_seconds": [10.0, 11.0],
            "team_id": [1, 2],
            "player_id": [101, 201],
            "start_x": [90.0, 10.0],
            "start_y": [34.0, 34.0],
            "end_x": [100.0, 15.0],
            "end_y": [34.0, 34.0],
            "type_id": [spadlcfg.actiontype_id["shot"], spadlcfg.actiontype_id["keeper_save"]],
            "result_id": [spadlcfg.result_id["fail"], spadlcfg.result_id["success"]],
            "bodypart_id": [spadlcfg.bodypart_id["foot"], spadlcfg.bodypart_id["foot"]],
        }
    )
    atomic = convert_to_atomic(actions)
    out_id = atomicconfig.actiontype_id["out"]
    assert out_id in atomic["type_id"].values, "Blocked shots must produce an 'out' atomic action"
