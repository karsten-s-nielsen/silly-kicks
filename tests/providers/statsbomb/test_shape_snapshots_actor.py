"""TF-54b: the actor identity bridge starts here.

``shape_snapshots`` currently reads the SB360 ``teammate``/``keeper`` freeze-frame flags but
discards ``actor``. The actor is the ONLY reliable outfield identity SB360 provides (exactly one
per freeze-frame, and the acting player's real ``player_id`` is on the SPADL action), so it must be
re-plumbed into an ``is_actor`` column for the downstream actor bridge.
"""

import pandas as pd

from silly_kicks.providers.statsbomb.parse import shape_snapshots


def _raw_frame(event_uuid, actor_idx):
    # 3 teammates + 1 opponent keeper; `actor_idx` marks the acting player among the teammates.
    players = [
        {"location": [60.0, 40.0 + i], "teammate": True, "keeper": False, "actor": i == actor_idx} for i in range(3)
    ]
    players.append({"location": [8.0, 34.0], "teammate": False, "keeper": True, "actor": False})
    return {"event_uuid": event_uuid, "freeze_frame": players, "visible_area": []}


def _actions():
    # Two distinct teams so shape_snapshots resolves the REAL team ids (not the synthetic {0,1}).
    return pd.DataFrame(
        {
            "action_id": [7, 8],
            "original_event_id": ["e7", "e8"],
            "team_id": [100, 200],
        }
    )


def test_shape_snapshots_carries_is_actor_exactly_one_true_on_acting_team():
    snaps, _va, _rep = shape_snapshots([_raw_frame("e7", actor_idx=1)], _actions())
    assert "is_actor" in snaps.columns
    assert snaps["is_actor"].dtype == bool
    actor_rows = snaps[snaps["is_actor"]]
    assert len(actor_rows) == 1
    # the actor is a teammate -> the acting team (100), not the opponent keeper's team (200)
    assert actor_rows.iloc[0]["team_id"] == 100
    assert not bool(actor_rows.iloc[0]["is_goalkeeper"])


def test_missing_actor_key_is_false_never_raises():
    # a freeze-frame whose player rows omit `actor` entirely -> is_actor all False, no crash
    players = [{"location": [50.0, 30.0], "teammate": True, "keeper": False}]
    frame = {"event_uuid": "e7", "freeze_frame": players, "visible_area": []}
    snaps, _va, _rep = shape_snapshots([frame], _actions())
    assert "is_actor" in snaps.columns
    assert not snaps["is_actor"].any()


def test_empty_freeze_frame_keeps_is_actor_column_with_zero_rows():
    frame = {"event_uuid": "e7", "freeze_frame": [], "visible_area": []}
    snaps, _va, _rep = shape_snapshots([frame], _actions())
    assert list(snaps.columns) == [
        "action_id",
        "team_id",
        "is_goalkeeper",
        "is_actor",
        "x",
        "y",
        "player_id",
    ]
    assert len(snaps) == 0
