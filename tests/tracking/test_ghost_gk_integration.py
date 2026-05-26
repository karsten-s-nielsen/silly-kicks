"""Integration tests for Ghost-GK positioning model (TF-18)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GRID_NX, GRID_NY
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


class TestAtomicMirror:
    def test_atomic_mirror(self):
        from silly_kicks.atomic.tracking.features import add_ghost_gk as atomic_add

        model, _, _ = _fitted_model()
        actions = pd.DataFrame(
            {
                "game_id": ["100", "100"],
                "action_id": [1, 2],
                "period_id": [1, 1],
                "time_seconds": [1.0, 2.0],
                "team_id": [2, 2],
                "player_id": ["a10", "a11"],
                "start_x": [50.0, 55.0],
                "start_y": [34.0, 34.0],
                "end_x": [55.0, 60.0],
                "end_y": [34.0, 34.0],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "bodypart_id": [0, 0],
            }
        )
        frames = pd.concat(
            [
                _make_ghost_gk_frames(frame_id=1, timestamp=1.0),
                _make_ghost_gk_frames(frame_id=2, timestamp=2.0),
            ],
            ignore_index=True,
        )

        atomic = atomic_add(actions, frames, model=model, home_team_id=1)

        for col in ["ghost_gk_x", "ghost_gk_y", "ghost_gk_spread"]:
            assert col in atomic.columns


class TestDtypeMismatch:
    def test_add_ghost_gk_dtype_mismatch(self):
        """int64 actions + str frames -> no crash."""
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()

        actions = pd.DataFrame(
            {
                "game_id": [100, 100],
                "action_id": [1, 2],
                "period_id": [1, 1],
                "time_seconds": [1.0, 2.0],
                "team_id": [2, 2],
                "player_id": ["a10", "a11"],
                "start_x": [50.0, 55.0],
                "start_y": [34.0, 34.0],
                "end_x": [55.0, 60.0],
                "end_y": [34.0, 34.0],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "bodypart_id": [0, 0],
            }
        )
        frames = pd.concat(
            [
                _make_ghost_gk_frames(game_id="100", frame_id=1, timestamp=1.0),
                _make_ghost_gk_frames(game_id="100", frame_id=2, timestamp=2.0),
            ],
            ignore_index=True,
        )
        result = add_ghost_gk(actions, frames, model=model, home_team_id=1)
        assert len(result) == 2


class TestTF19Interface:
    def test_ghost_gk_with_gk_deterrent_interface(self):
        """Density grid compatible with TF-19 consumption."""
        model, X, _ = _fitted_model()
        densities = model.predict_density(X[:3])
        for d in densities:
            assert d.grid_x.shape == (GRID_NX,)
            assert d.grid_y.shape == (GRID_NY,)
            assert d.probabilities.shape == (GRID_NX, GRID_NY)
            # Element-wise multiplication with shot region
            shot_region = np.random.default_rng(42).random((GRID_NX, GRID_NY))
            threat = float((d.probabilities * shot_region).sum())
            assert 0.0 <= threat <= 1.0
