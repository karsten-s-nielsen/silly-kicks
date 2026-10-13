"""TF-65 §7: acceleration columns (ax/ay/accel)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames


def _const_accel_frames(a=2.0, hz=25.0, n=50):
    t = np.arange(n) / hz
    return pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": t,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": 0.5 * a * t**2,
            "y": 30.0,
        }
    )


def test_accel_columns_present_and_correct_sign_and_dtype():
    f = _const_accel_frames(a=2.0)
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    assert {"accel_x", "accel_y", "accel"} <= set(v.columns)
    assert v["accel_x"].dtype == np.float32 and v["accel"].dtype == np.float32
    mid = v.iloc[20:30]
    assert np.allclose(mid["accel_x"], 2.0, atol=0.2)  # recovers the constant acceleration
    assert (mid["accel"] >= 0).all()  # magnitude non-negative
