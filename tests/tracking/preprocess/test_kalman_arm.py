"""TF-65 §5: Kalman point smoother (smoothing_method="kalman") -- coherent kinematics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames


def test_kalman_arm_produces_coherent_kinematics():
    hz = 25.0
    n = 80
    f = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": np.arange(n) / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": 2.0 * np.arange(n) / hz,
            "y": 30.0,
        }
    )
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg, method="kalman"), config=cfg)
    assert np.allclose(v["vx"].iloc[20:60], 2.0, atol=0.1)
    assert {"pos_var", "accel"} <= set(v.columns)  # coherent: point + uncertainty from one estimator
