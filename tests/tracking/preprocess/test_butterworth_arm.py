"""TF-65 §5: Butterworth velocity/acceleration arm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames


def test_butterworth_arm_velocity_bounded_across_gap():
    hz = 29.97
    pre = np.arange(0, 20)
    post = np.arange(1000, 1020)
    fid = np.concatenate([pre, post])
    f = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": fid,
            "time_seconds": fid / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": True,
            "x": np.r_[np.full(20, 113.0), np.full(20, 11.0)],
            "y": 34.0,
        }
    )
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg, method="butterworth"), config=cfg)
    assert (v["speed"].dropna() < 40).all()  # no through-gap fabrication
