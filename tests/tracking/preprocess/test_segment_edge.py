"""TF-65 §4.2: gap-adjacent segment-edge NaN (no pre-gap edge spike)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames


def test_no_edge_spike_on_short_pre_gap_segment():
    # a short (near-window-length) segment immediately before a big gap must not emit an edge-spike velocity
    hz = 29.97
    seg = np.arange(0, 6)  # 6-frame segment (< an 11-frame window)
    post = np.arange(1000, 1020)
    fid = np.concatenate([seg, post])
    f = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": fid,
            "time_seconds": fid / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": True,
            "x": np.r_[np.full(6, 113.0), np.full(20, 11.0)],
            "y": 34.0,
        }
    )
    cfg = PreprocessConfig.default()
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    pre_gap = v[v["frame_id"] <= 5]
    assert (pre_gap["speed"].isna() | (pre_gap["speed"] < 40)).all()  # edge NaN'd, not a 300+ m/s spike
