"""TF-54b: the outfield actor identity bridge (shared home in keeper_identity, ADR-078/084).

``apply_actor_identities_to_frames`` stamps each action's real ``player_id`` onto its single
``is_actor`` SB360 frame row -- the outfield analogue of ``apply_keeper_identities_to_frames``.
"""

import pandas as pd

from silly_kicks.keeper_identity import apply_actor_identities_to_frames


def _frames(player_id, is_actor):
    return pd.DataFrame(
        {
            "game_id": [9] * len(player_id),
            "period_id": [1] * len(player_id),
            "frame_id": [1] * len(player_id),
            "team_id": [100] * len(player_id),
            "player_id": player_id,
            "is_ball": [False] * len(player_id),
            "is_actor": is_actor,
            "x": [float(i) for i in range(len(player_id))],
            "y": [float(i) for i in range(len(player_id))],
        }
    )


def test_actor_row_gets_real_player_id_others_unchanged():
    frames = _frames([0, 1, 2], [False, True, False])
    actions = pd.DataFrame({"action_id": [1], "player_id": [5551], "team_id": [100]})
    out = apply_actor_identities_to_frames(frames, actions)
    assert out is not frames  # pure
    assert frames["player_id"].tolist() == [0, 1, 2]  # input not mutated
    actor_mask = out["is_actor"].astype("boolean").fillna(False)
    assert out.loc[actor_mask, "player_id"].iloc[0] == 5551
    assert out.loc[~actor_mask, "player_id"].tolist() == [0, 2]


def test_mixed_dtype_stamps_via_object_fallback():
    # int64 frame ids, a STRING real player_id -> object fallback (ADR-019), never a crash
    frames = _frames([0, 1], [False, True])
    actions = pd.DataFrame({"action_id": [1], "player_id": ["GK-77"], "team_id": [100]})
    out = apply_actor_identities_to_frames(frames, actions)
    actor_mask = out["is_actor"].astype("boolean").fillna(False)
    assert out.loc[actor_mask, "player_id"].iloc[0] == "GK-77"


def test_no_actor_rows_returns_pure_copy_unchanged():
    frames = _frames([0, 1], [False, False])
    actions = pd.DataFrame({"action_id": [1], "player_id": [5551], "team_id": [100]})
    out = apply_actor_identities_to_frames(frames, actions)
    assert out is not frames
    assert out["player_id"].tolist() == [0, 1]
