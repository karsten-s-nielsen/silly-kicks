"""W7: ``smooth_frames`` / ``derive_velocities`` speed-up (spec 2026-11-08 tracking-batch section 4.7; plan Task 13).

The bit identity to the pre-change code was proven against a frozen oracle (``preprocess_v4139``) and on the whole
gradientsports 10517 in the C1 record (``docs/research/tracking-compute-perf/``); the oracle ended with ruling C5
(spec section 7). The other tests in this folder pin the behaviour of each method; none pins the output bits. The
covariance reuse guard is the structural proof of W7.1: the Kalman covariance recursion depends only on (dt, the
detection mask, the Kalman parameters), so it runs once per distinct key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._bitwise import assert_bits_equal
from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames
from tests._perf_structural import call_counter
from tests._prod_fixtures import make_two_game_inputs

_KINEMATICS = ["vx", "vy", "speed", "accel_x", "accel_y", "accel", "pos_var", "vel_var", "accel_var"]


def _shared_mask_frames(n_players: int = 12, n_frames: int = 200) -> pd.DataFrame:
    """Every player detected on every frame: they all share one detection mask."""
    rng = np.random.default_rng(5)
    rows = []
    for p in range(n_players):
        xy = np.cumsum(rng.normal(0.0, 0.2, size=(n_frames, 2)), axis=0) + np.array([30.0 + 3 * p, 34.0])
        rows.append(
            pd.DataFrame(
                {
                    "game_id": 7,
                    "period_id": 1,
                    "frame_id": np.arange(n_frames),
                    "time_seconds": np.arange(n_frames) / 25.0,
                    "frame_rate": 25.0,
                    "player_id": p,
                    "team_id": 5 if p % 2 else 6,
                    "is_ball": False,
                    "x": xy[:, 0].astype(np.float32),
                    "y": xy[:, 1].astype(np.float32),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_kalman_cov_reuse_guard(monkeypatch):
    from silly_kicks.tracking.preprocess import _kalman, _velocity

    frames = _shared_mask_frames()
    smoothed = smooth_frames(frames, method="kalman")
    calls = call_counter(monkeypatch, _kalman, "_kalman_covariance")
    filters = call_counter(monkeypatch, _velocity, "kalman_ca")  # as _velocity resolves it
    derive_velocities(smoothed)
    assert filters["n"] >= 2 * 12  # x and y of every player
    assert calls["n"] == 1  # one (dt, mask) key: one recursion


def test_two_games_per_game_results():
    # TF-65 section 4.1a: a game is its own series; two games in one call equal two one-game calls
    _actions, frames = make_two_game_inputs()
    games = [g for _gid, g in frames.groupby("game_id", sort=True)]
    assert len(games) == 2
    cfg = PreprocessConfig.default()
    whole = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    parts = pd.concat([derive_velocities(smooth_frames(g, config=cfg), config=cfg) for g in games])
    key = ["game_id", "period_id", "frame_id", "is_ball", "player_id"]
    cols = ["x_smoothed", "y_smoothed", *_KINEMATICS]
    a = whole.sort_values(key, kind="mergesort").reset_index(drop=True)[cols]
    b = parts.sort_values(key, kind="mergesort").reset_index(drop=True)[cols]
    assert_bits_equal(a, b)
