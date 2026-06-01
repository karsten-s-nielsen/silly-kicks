"""Behaviour-preserving check for the de-iloc'd distance lookup (Phase 0b)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._elastic_sync import _build_player_ball_distance_lookup


def _frames() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for fid in range(40):
        rows.append(
            {"game_id": 1, "period_id": 1, "frame_id": fid, "player_id": None,
             "team_id": None, "is_ball": True, "x": 50.0 + rng.normal(), "y": 34.0 + rng.normal()}
        )
        for pid in range(6):
            rows.append(
                {"game_id": 1, "period_id": 1, "frame_id": fid, "player_id": f"p{pid}",
                 "team_id": 1 + pid % 2, "is_ball": False,
                 "x": rng.uniform(0, 105), "y": rng.uniform(0, 68)}
            )
    return pd.DataFrame(rows)


def _expected_lookup(frames: pd.DataFrame) -> dict:
    """Independent oracle: the intended key/value contract, built without .iloc."""
    ball = frames[frames["is_ball"]].drop_duplicates(["game_id", "period_id", "frame_id"])
    bpos = {(r.game_id, r.period_id, r.frame_id): (r.x, r.y) for r in ball.itertuples(index=False)}
    out: dict = {}
    for r in frames[~frames["is_ball"]].itertuples(index=False):
        bx, by = bpos.get((r.game_id, r.period_id, r.frame_id), (np.nan, np.nan))
        d = float("inf") if (np.isnan(bx) or np.isnan(by)) else float(np.hypot(r.x - bx, r.y - by))
        out[(r.game_id, r.period_id, int(r.frame_id), str(r.player_id))] = d
    return out


def test_lookup_matches_oracle_and_key_dtypes():
    frames = _frames()
    lookup = _build_player_ball_distance_lookup(frames)
    assert len(lookup) == 240  # 40 frames * 6 players
    # Behaviour-preservation: dict equality (keys compare by hash/eq, values by ==),
    # robust to np.int64-vs-int key-type subtleties.
    assert lookup == _expected_lookup(frames)
    # Complete key-dtype contract (all four elements):
    k = next(iter(lookup))
    assert isinstance(k[0], (int, np.integer))  # game_id
    assert isinstance(k[1], (int, np.integer))  # period_id
    assert isinstance(k[2], int)                # frame_id -> Python int (int(...))
    assert isinstance(k[3], str)                # player_id -> Python str (str(...))
    assert all(isinstance(v, float) for v in lookup.values())
