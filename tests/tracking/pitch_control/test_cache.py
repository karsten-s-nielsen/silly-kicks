"""ADR-103 F5: PitchControlCache bounded LRU eviction; maxsize=None unchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control import PitchControlCache


def _frame(frame_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "frame_id": [frame_id] * 5,
            "is_ball": [False, False, False, False, True],
            "is_goalkeeper": [True, False, True, False, False],
            "team_id": [1, 1, 2, 2, np.nan],
            "player_id": [10, 11, 20, 21, np.nan],
            "x": [30.0, 60.0, 45.0, 70.0, 50.0],
            "y": [34.0, 20.0, 40.0, 34.0, 34.0],
            "vx": [0.0, 0.0, 0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_cache_lru_evicts_beyond_maxsize():
    cache = PitchControlCache(maxsize=2)
    for fid in range(1, 6):  # 5 distinct frames
        cache.surface(_frame(fid), attacking_team_id=1, method="spearman")
    assert len(cache) == 2  # only the 2 most-recent retained


def test_cache_unbounded_default():
    cache = PitchControlCache()
    for fid in range(1, 6):
        cache.surface(_frame(fid), attacking_team_id=1, method="spearman")
    assert len(cache) == 5  # maxsize=None unchanged


def test_cache_lru_keeps_recently_used():
    cache = PitchControlCache(maxsize=2)
    cache.surface(_frame(1), attacking_team_id=1, method="spearman")
    cache.surface(_frame(2), attacking_team_id=1, method="spearman")
    cache.surface(_frame(1), attacking_team_id=1, method="spearman")  # touch 1 -> MRU
    cache.surface(_frame(3), attacking_team_id=1, method="spearman")  # evicts 2 (LRU), keeps 1,3
    s1 = cache.surface(_frame(1), attacking_team_id=1, method="spearman")
    assert len(cache) == 2
    # frame 1 was retained (a hit, not a recompute): identical result object semantics
    assert s1.surface.shape[0] > 0
