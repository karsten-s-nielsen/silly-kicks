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


def test_interception_is_not_duplicated() -> None:
    """TF-51 Item 4 prereq: 'interception' is inherited from std (idx 10), never re-appended.

    The duplicate append made the reverse dict resolve interception->24, silently shadowing the
    std interception events (idx 10) that convert_to_atomic keeps unremapped.
    """
    assert atomicconfig.actiontypes.count("interception") == 1
    assert atomicconfig.actiontype_id["interception"] == 10
    assert len(atomicconfig.actiontypes) == 32


def test_tail_ids_after_dedup() -> None:
    """The atomic-only tail renumbers down by one once the duplicate interception is removed."""
    expected = {
        "receival": 23,
        "out": 24,
        "offside": 25,
        "goal": 26,
        "owngoal": 27,
        "yellow_card": 28,
        "red_card": 29,
        "corner": 30,
        "freekick": 31,
    }
    for name, idx in expected.items():
        assert atomicconfig.actiontype_id[name] == idx
