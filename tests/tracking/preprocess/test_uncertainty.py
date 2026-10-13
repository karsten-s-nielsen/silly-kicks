"""TF-65 §7: always-on per-frame uncertainty, honest-gap property (not mere liveness)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames


def _frames_with_gap(hz=25.0):
    pre = np.arange(0, 40)
    post = np.arange(60, 100)  # a 20-frame (0.8 s > max_gap) gap
    fid = np.concatenate([pre, post])
    return pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": fid,
            "time_seconds": fid / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": 50.0 + 0.05 * fid,
            "y": 30.0,
        }
    )


def test_variance_strictly_rises_toward_a_gap():
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(_frames_with_gap(), config=cfg), config=cfg)
    assert {"pos_var", "vel_var", "accel_var"} <= set(v.columns)
    for col in ("pos_var", "accel_var"):  # spec §11 names BOTH
        gap_adjacent = v[v["frame_id"] == 39][col].iloc[0]
        dense_interior = v[v["frame_id"] == 20][col].iloc[0]
        assert gap_adjacent > dense_interior  # honest-gap property, not merely non-constant


def test_uncertainty_present_for_every_smoother():
    cfg = PreprocessConfig.default()
    for method in ("savgol", "butterworth"):
        v = derive_velocities(smooth_frames(_frames_with_gap(), config=cfg, method=method), config=cfg)
        assert v["accel_var"].notna().any()  # decoupled from the point smoother
