"""TF-54b: ``snapshot_to_tracking_frames`` carries the SB360 ``is_actor`` flag through to frames.

``is_actor`` is a SNAPSHOT-ONLY extension column -- it is NOT added to the base
``TRACKING_FRAMES_COLUMNS`` (continuous tracking has no per-frame "actor"), so a caller whose
snapshots lack it gets byte-identical output.
"""

import pandas as pd

from silly_kicks.tracking import snapshot_to_tracking_frames
from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS


def _snaps():
    return pd.DataFrame(
        {
            "action_id": [1, 1, 1, 1],
            "team_id": [100, 100, 100, 200],
            "is_goalkeeper": [False, False, False, True],
            "is_actor": [False, True, False, False],
            "x": [60.0, 55.0, 50.0, 8.0],
            "y": [40.0, 34.0, 30.0, 34.0],
            "player_id": [0, 1, 2, 3],
        }
    )


def _actions():
    return pd.DataFrame(
        {
            "action_id": [1],
            "game_id": [9],
            "period_id": [1],
            "time_seconds": [12.0],
            "start_x": [55.0],
            "start_y": [34.0],
        }
    )


def test_frames_carry_is_actor_ball_false_one_true():
    frames, _links = snapshot_to_tracking_frames(_snaps(), _actions())
    assert "is_actor" in frames.columns
    is_ball = frames["is_ball"].astype("boolean").fillna(False)
    is_actor = frames["is_actor"].astype("boolean").fillna(False)
    # the ball row is never the actor
    assert not is_actor[is_ball].any()
    # exactly one actor row (the marked teammate)
    assert int(is_actor.sum()) == 1
    # base schema stays 20-wide: is_actor is an EXTENSION beyond the declared schema
    assert "is_actor" not in TRACKING_FRAMES_COLUMNS


def test_snapshots_without_is_actor_produce_no_is_actor_column():
    # a caller whose snapshots lack is_actor -> output frames unchanged (no is_actor column, additive)
    snaps = _snaps().drop(columns=["is_actor"])
    frames, _links = snapshot_to_tracking_frames(snaps, _actions())
    assert "is_actor" not in frames.columns
