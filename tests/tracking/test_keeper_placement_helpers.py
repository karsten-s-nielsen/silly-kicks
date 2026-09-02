from __future__ import annotations

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.keeper_identity import (
    add_defending_gk_player_id,
    apply_keeper_identities_to_frames,
    resolve_keeper_identities,
)
from silly_kicks.spadl import config as spadlconfig


def _sb360_fixture():
    """A shot by team 10 with team 20's keeper in the freeze-frame; synthetic numbered frame ids."""
    # ``type_id`` is REQUIRED, not decoration: the ``_pre_shot_gk_position`` kernel gates on
    # ``actions["type_id"]`` (a real SPADL shot carries it), so WITHOUT it BOTH the control and the
    # treatment legs of the R1 test are all-NaN and the treatment assertion (``notna().any()``)
    # would pass VACUOUSLY. The brief's fixture text omitted it; it is added here so the treatment
    # genuinely produces a real GK position (the load-bearing non-vacuity the R1 test demands).
    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [5.0],
            "team_id": [10],
            "player_id": [101],
            "type_name": ["shot"],
            "type_id": [spadlconfig.actiontype_id["shot"]],
            "start_x": [90.0],
            "start_y": [34.0],
        }
    )
    snapshots = pd.DataFrame(
        {
            "action_id": [0, 0, 0],
            "team_id": [10, 10, 20],
            "x": [90.0, 80.0, 104.0],
            "y": [34.0, 40.0, 34.0],
            "is_goalkeeper": [False, False, True],
        }
    )
    frames, _ = T.snapshot_to_tracking_frames(snapshots, actions)
    return actions, frames


def test_add_defending_gk_player_id_stamps_opponent_keeper_and_is_pure():
    actions, frames = _sb360_fixture()
    m, _ = resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    snap = actions.copy(deep=True)
    out = add_defending_gk_player_id(actions, m)
    pd.testing.assert_frame_equal(actions, snap)  # pure
    # the shot is by team 10 -> defending keeper is team 20's (902)
    assert out["defending_gk_player_id"].iloc[0] == 902


def test_frame_bridge_stamps_real_id_onto_the_synthetic_keeper_row_and_is_pure():
    actions, frames = _sb360_fixture()
    m, _ = resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    snap = frames.copy(deep=True)
    bridged = apply_keeper_identities_to_frames(frames, m)
    pd.testing.assert_frame_equal(frames, snap)  # pure -- caller's frames untouched
    krow = bridged[(bridged["team_id"] == 20) & bridged["is_goalkeeper"].astype("boolean").fillna(False)]
    assert (krow["player_id"] == 902).all(), "the synthetic keeper-row id must be bridged to the roster id"


def test_bridge_unlocks_pre_shot_gk_position_the_R1_deliverable():
    """The whole point: without the bridge, add_pre_shot_gk_position is NaN on SB360 (frame ids are
    synthetic). With the bridge (real keeper id on the frame row + on the action), it produces a real
    position."""
    actions, frames = _sb360_fixture()
    m, _ = resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    stamped_actions = add_defending_gk_player_id(actions, m)

    # WITHOUT the bridge: the synthetic keeper id (a small int) != 902 -> NaN.
    unbridged = T.add_pre_shot_gk_position(stamped_actions, frames)
    assert unbridged["pre_shot_gk_x"].isna().all(), "control: unbridged SB360 frames yield NaN GK position"

    # WITH the bridge: the keeper row now carries 902, matching the action stamp -> real position.
    bridged = T.add_pre_shot_gk_position(stamped_actions, apply_keeper_identities_to_frames(frames, m))
    assert bridged["pre_shot_gk_x"].notna().any(), "bridged frames must yield a REAL GK position (R1)"


def test_bridge_tolerates_a_dtype_incompatible_roster_id():
    """A roster gk_id the frames' player_id column cannot hold (a str id into the Int64 snapshot
    column) is bridged by promoting to object -- it must NOT raise (defensive; live SB360 ids are ints)."""
    actions, frames = _sb360_fixture()
    m, _ = resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: "GK-902"})
    bridged = apply_keeper_identities_to_frames(frames, m)  # must NOT raise despite a str id into Int64
    krow = bridged[(bridged["team_id"] == 20) & bridged["is_goalkeeper"].astype("boolean").fillna(False)]
    assert (krow["player_id"] == "GK-902").all()
