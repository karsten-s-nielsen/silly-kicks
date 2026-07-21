"""Integration tests for Ghost-GK positioning model (TF-18)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GRID_NX, GRID_NY, GhostGkModel
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames

# ---------------------------------------------------------------------------
# Pre-existing integration tests (atomic mirror, dtype mismatch, TF-19)
# ---------------------------------------------------------------------------


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

        for col in ["ghost_gk_x", "ghost_gk_y"]:
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


# ---------------------------------------------------------------------------
# New integration tests: training pipeline round-trip + script smoke
# ---------------------------------------------------------------------------


def _build_synthetic_parquets(
    tmpdir: Path,
    n_games: int = 3,
) -> tuple[Path, Path, Path]:
    """Build synthetic tracking + actions parquets + home_teams.json."""
    tracking_dir = tmpdir / "tracking"
    actions_dir = tmpdir / "actions"
    tracking_dir.mkdir()
    actions_dir.mkdir()

    for g in range(n_games):
        game_id = str(100 + g)
        rows = []
        for fid in range(1, 6):  # 5 frames per game
            ts = float(fid) * 0.04  # 25fps
            base = dict(
                game_id=game_id,
                period_id=1,
                frame_id=fid,
                time_seconds=ts,
                frame_rate=25.0,
                ball_state="alive",
                source_provider="gradientsports",  # classified fully-observed provider (PR-S115 detected-only filter)
                team_attacking_direction=None,
                confidence=None,
                visibility=None,
                is_goalkeeper_source="native",
                z=0.0,
            )
            rows.append(
                {
                    **base,
                    "player_id": "ball",
                    "team_id": None,
                    "x": 50.0 + fid,
                    "y": 34.0,
                    "vx": 2.0,
                    "vy": 0.0,
                    "speed": 2.0,
                    "is_ball": True,
                    "is_goalkeeper": False,
                }
            )
            rows.append(
                {
                    **base,
                    "player_id": "p1",
                    "team_id": "1",
                    "x": 5.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": True,
                }
            )
            for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
                rows.append(
                    {
                        **base,
                        "player_id": f"p{10 + i}",
                        "team_id": "1",
                        "x": float(px),
                        "y": float(py),
                        "vx": 0.5,
                        "vy": 0.0,
                        "speed": 0.5,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            for i, (px, py) in enumerate([(40, 30), (45, 34), (38, 40), (50, 34)]):
                rows.append(
                    {
                        **base,
                        "player_id": f"a{10 + i}",
                        "team_id": "2",
                        "x": float(px),
                        "y": float(py),
                        "vx": -1.0,
                        "vy": 0.0,
                        "speed": 1.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            rows.append(
                {
                    **base,
                    "player_id": "a1",
                    "team_id": "2",
                    "x": 100.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": True,
                }
            )

        pd.DataFrame(rows).to_parquet(
            tracking_dir / f"game_{game_id}.parquet",
            index=False,
        )

    # Home teams JSON
    home_teams = {str(100 + g): "1" for g in range(n_games)}
    ht_path = tmpdir / "home_teams.json"
    with open(ht_path, "w") as f:
        json.dump(home_teams, f)

    return tracking_dir, actions_dir, ht_path


class TestRoundTripTrainPredict:
    """prepare_training_data -> fit -> predict round-trip."""

    def test_round_trip(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        # Build multi-frame fixture
        rows = []
        for fid in range(1, 11):
            ts = float(fid) * 0.04
            base = dict(
                game_id="100",
                period_id=1,
                frame_id=fid,
                time_seconds=ts,
                frame_rate=25.0,
                ball_state="alive",
                source_provider="gradientsports",  # classified fully-observed provider (PR-S115 detected-only filter)
                team_attacking_direction=None,
                confidence=None,
                visibility=None,
                is_goalkeeper_source="native",
                z=0.0,
            )
            rows.append(
                {
                    **base,
                    "player_id": "ball",
                    "team_id": None,
                    "x": 50.0 + fid,
                    "y": 34.0,
                    "vx": 2.0,
                    "vy": 0.0,
                    "speed": 2.0,
                    "is_ball": True,
                    "is_goalkeeper": False,
                }
            )
            rows.append(
                {
                    **base,
                    "player_id": "p1",
                    "team_id": 1,
                    "x": 5.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": True,
                }
            )
            for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
                rows.append(
                    {
                        **base,
                        "player_id": f"p{10 + i}",
                        "team_id": 1,
                        "x": float(px),
                        "y": float(py),
                        "vx": 0.5,
                        "vy": 0.0,
                        "speed": 0.5,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            for i, (px, py) in enumerate([(40, 30), (45, 34), (38, 40), (50, 34)]):
                rows.append(
                    {
                        **base,
                        "player_id": f"a{10 + i}",
                        "team_id": 2,
                        "x": float(px),
                        "y": float(py),
                        "vx": -1.0,
                        "vy": 0.0,
                        "speed": 1.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            rows.append(
                {
                    **base,
                    "player_id": "a1",
                    "team_id": 2,
                    "x": 100.0,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": True,
                }
            )

        frames = pd.DataFrame(rows)
        features, labels = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            subsample_fps=None,
        )
        assert len(features) > 0

        model = GhostGkModel(n_estimators=10)
        model.fit(features, labels)
        preds = model.predict(features)
        assert preds.shape == (len(features), 2)

        # Predictions should be in plausible range
        assert np.all(preds[:, 0] >= -5)
        assert np.all(preds[:, 0] <= 35)


class TestTrainScriptSmoke:
    """Train script runs on synthetic data and produces artifacts."""

    @pytest.mark.slow
    def test_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            tracking_dir, _, ht_path = _build_synthetic_parquets(
                tmpdir_p,
                n_games=3,
            )

            output_dir = tmpdir_p / "output"
            result = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "scripts/train_ghost_gk.py",
                    "--data-dir",
                    str(tracking_dir),
                    "--home-teams",
                    str(ht_path),
                    "--output-dir",
                    str(output_dir),
                    "--n-estimators",
                    "10",
                    "--max-depth",
                    "3",
                    "--cv-folds",
                    "3",
                    "--subsample-fps",
                    "25.0",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"

            # Check artifacts exist
            artifact_dir = output_dir / "ghost_gk_v1"
            assert artifact_dir.exists()
            assert (artifact_dir / "metrics.json").exists()

            # Verify metrics.json schema
            with open(artifact_dir / "metrics.json") as f:
                metrics = json.load(f)
            assert "n_games" in metrics
            assert "cv_mae_euclidean_mean" in metrics
            assert "acceptance" in metrics
            assert "artifact_size_bytes" in metrics
