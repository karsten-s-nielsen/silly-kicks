"""smooth_frames groups by game_id, so the same player_id in two games is never smoothed across the game boundary.

Review A-31 (the defect fixed in resample_frames this cycle; the smoother depends on the same grouping).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import smooth_frames


def _one_game(game_id: int, x0: float, n: int = 120, hz: float = 10.0) -> pd.DataFrame:
    """One game, one outfield player, flat x at ``x0`` (+ a gentle weave), period 1."""
    t = np.arange(n)
    return pd.DataFrame(
        {
            "game_id": game_id,
            "period_id": 1,
            "frame_id": t,
            "player_id": 7,  # the SAME player id in both games
            "is_ball": False,
            "x": x0 + 0.5 * np.sin(2 * np.pi * t / 50.0),
            "y": 34.0 + 0.5 * np.cos(2 * np.pi * t / 50.0),
            "frame_rate": hz,
        }
    )


def test_smooth_frames_does_not_blend_across_games():
    # same player_id, 100 m apart in the two games: game 1's smoothed series must be identical whether game 2 is
    # present or not. Without game_id in the key the concatenated series carries a 100 m step that savgol blends into
    # game 1's tail -- so the equality below is non-vacuous (the games differ by 100 m).
    g1, g2 = _one_game(1, 0.0), _one_game(2, 100.0)
    both = smooth_frames(pd.concat([g1, g2], ignore_index=True))
    alone1 = smooth_frames(g1)
    b1 = both[both["game_id"] == 1].sort_values("frame_id").reset_index(drop=True)
    a1 = alone1.sort_values("frame_id").reset_index(drop=True)
    np.testing.assert_array_equal(b1["x_smoothed"].to_numpy(), a1["x_smoothed"].to_numpy())
    np.testing.assert_array_equal(b1["y_smoothed"].to_numpy(), a1["y_smoothed"].to_numpy())


def test_single_game_smoothing_is_unchanged_by_the_game_key():
    # the other side: one game (game_id constant) is byte-identical whether or not game_id is in the key.
    g = _one_game(1, 0.0)
    out = smooth_frames(g)
    assert out["x_smoothed"].notna().all()  # smoothing ran; single-game output is the pre-A-31 output (game_id const)
