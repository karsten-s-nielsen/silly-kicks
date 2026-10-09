"""TF-65 §4.1: dense-grid reindex helpers + the missing-row-gap velocity fix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames
from silly_kicks.tracking.preprocess._densify import bridge_small, densify_group, segments


def test_densify_inserts_missing_rows_as_nan():
    g = pd.DataFrame({"frame_id": [0, 1, 2, 10, 11], "x": [1.0, 1.1, 1.2, 5.0, 5.1], "y": [0.0] * 5})
    full, x, _y, real = densify_group(g)
    assert list(full) == list(range(0, 12))
    assert np.isnan(x[3:10]).all()  # frames 3..9 are missing -> NaN
    assert real.sum() == 5 and not real[3]


def test_densify_dedups_duplicate_frames():  # GS duplicate-frame guard
    g = pd.DataFrame({"frame_id": [0, 0, 1], "x": [1.0, 1.0, 1.1], "y": [0.0] * 3})
    full, _x, _y, _real = densify_group(g)
    assert list(full) == [0, 1]


def test_segments_split_at_big_gap():
    valid = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)  # a 4-frame gap
    assert segments(valid, max_gap_frames=2) == [(0, 3), (7, 9)]  # split (gap 4 > 2)
    assert segments(valid, max_gap_frames=5) == [(0, 9)]  # not split (gap 4 <= 5)


def test_bridge_small_fills_short_leaves_long():
    v = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0])
    out = bridge_small(v, max_gap_frames=1)
    assert out[1] == 1.0  # 1-frame gap bridged
    assert np.isnan(out[3:6]).all()  # 3-frame gap left NaN


def test_missing_row_gap_yields_nan_velocity_not_spike():
    hz = 25.0
    pre = np.arange(0, 20)  # contiguous run A
    post = np.arange(200, 220)  # run B after a 180-frame (7.2 s) gap
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
            "x": np.r_[np.full(20, 10.0), np.full(20, 100.0)],  # 90 m jump across the gap
            "y": 34.0,
        }
    )
    cfg = PreprocessConfig.default()  # max_gap_seconds=0.5 -> ~12 frames; 180 >> 12
    v = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    boundary = v[(v["frame_id"] == 19) | (v["frame_id"] == 200)]
    assert boundary["speed"].isna().all()  # no fabricated through-gap velocity


def test_idempotent_and_single_run_byte_identical():
    # a contiguous single-detection-run group: dense-grid reindex is a no-op (sportec control analogue)
    hz = 25.0
    n = 60
    f = pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": np.arange(n) / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": 50.0 + 0.1 * np.arange(n),
            "y": 30.0,
        }
    )
    cfg = PreprocessConfig.default()
    once = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    twice = derive_velocities(smooth_frames(once, config=cfg), config=cfg)  # idempotent
    pd.testing.assert_series_equal(once["speed"], twice["speed"])
