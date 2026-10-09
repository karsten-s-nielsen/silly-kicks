"""TF-65 §4.1a: game_id in the group key for derive_velocities + interpolate_frames.

Single-game output must be byte-identical to the old 3-key grouping (game_id constant ->
identical partition); a two-game frame must NOT bridge velocity across the game boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import (
    PreprocessConfig,
    derive_velocities,
    interpolate_frames,
    smooth_frames,
)


def _one_game(game_id, hz=25.0, n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "game_id": game_id,
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": np.arange(n) / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": np.cumsum(rng.normal(0, 0.1, n)) + 50.0,
            "y": np.cumsum(rng.normal(0, 0.1, n)) + 30.0,
        }
    )


def _set_three_key(monkeypatch):
    # force the OLD 3-key grouping (no game_id) in every preprocess module, to compare against
    import silly_kicks.tracking.preprocess._interpolation as interp
    import silly_kicks.tracking.preprocess._smoothing as smo
    import silly_kicks.tracking.preprocess._velocity as vel

    three = ["period_id", "is_ball", "player_id"]
    monkeypatch.setattr(vel, "_GROUP_KEYS", three)
    monkeypatch.setattr(interp, "_GROUP_KEYS", three)
    monkeypatch.setattr(smo, "_GROUP_KEYS", three)  # single-game: identical partition either way


def test_single_game_byte_identical_vs_three_key(monkeypatch):
    # single-game: adding game_id to the key must be a NO-OP (game_id constant -> identical partition)
    f = _one_game("g1")
    cfg = PreprocessConfig.default()
    new = derive_velocities(smooth_frames(f, config=cfg), config=cfg)  # 4-key (game_id)
    i_new = interpolate_frames(f.assign(x=f["x"].mask(f["frame_id"].eq(30))), config=cfg)
    _set_three_key(monkeypatch)
    old = derive_velocities(smooth_frames(f, config=cfg), config=cfg)  # 3-key
    i_old = interpolate_frames(f.assign(x=f["x"].mask(f["frame_id"].eq(30))), config=cfg)
    pd.testing.assert_frame_equal(new, old)  # byte-identical
    pd.testing.assert_frame_equal(i_new, i_old)


def _game(game_id, x, hz=25.0, n=40):
    return pd.DataFrame(
        {
            "game_id": game_id,
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": np.arange(n) / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": x,
            "y": 30.0,
        }
    )


def test_two_game_no_cross_bridge_discriminating(monkeypatch):
    # Two games share (period, is_ball, player) and the same frame_ids (0..39). g1 is stationary, g2
    # moves at ~5 m/s. 4-key: g2 is its own group and keeps its motion. 3-key: the dense-grid dedup on
    # frame_id drops g2's duplicate rows, so g2 wrongly inherits g1's stationary kinematics. Asserting
    # BOTH sides gives the test discriminating power (it detects the fix's absence).
    n = 40
    g1 = _game("g1", np.full(n, 50.0))  # stationary
    g2 = _game("g2", 30.0 + 0.2 * np.arange(n))  # +5 m/s (0.2 m/frame @ 25 Hz)
    f = pd.concat([g1, g2], ignore_index=True)
    cfg = PreprocessConfig.default()
    v4 = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    s4 = v4[(v4["game_id"] == "g2") & (v4["frame_id"] == 20)]["speed"].iloc[0]
    assert s4 > 2.0  # 4-key: g2 keeps its ~5 m/s motion
    _set_three_key(monkeypatch)
    v3 = derive_velocities(smooth_frames(f, config=cfg), config=cfg)
    s3 = v3[(v3["game_id"] == "g2") & (v3["frame_id"] == 20)]["speed"].iloc[0]
    assert s3 < 2.0  # discriminating: 3-key drops g2, inheriting g1's stationary kinematics
