"""Unit tests for Ghost-GK positioning model (TF-18)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GRID_X_MAX, GRID_X_MIN, GRID_Y_MAX, GRID_Y_MIN

# ---------------------------------------------------------------------------
# Shared fixture (used across all test classes — avoids duplication M1)
# READ-ONLY: tests must NOT mutate model, X, or labels. If mutation is needed,
# copy first (e.g., X_copy = X.copy()).
# ---------------------------------------------------------------------------

_FITTED_MODEL_CACHE: dict[str, tuple] = {}


def _fitted_model(*, n_estimators: int = 10, n_samples: int = 100):
    """Module-level model fixture. Cached to avoid re-training per test.

    Returns a shared (model, X, labels) tuple. Treat as READ-ONLY —
    the same objects are returned to all callers within the module.
    """
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    cache_key = f"{n_estimators}_{n_samples}"
    if cache_key in _FITTED_MODEL_CACHE:
        return _FITTED_MODEL_CACHE[cache_key]

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n_samples, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n_samples).astype(float)
    X["team_in_possession"] = rng.integers(0, 2, n_samples).astype(float)
    X["ball_in_own_half"] = rng.integers(0, 2, n_samples).astype(float)
    labels = pd.DataFrame(
        {
            "gk_x": rng.uniform(2, 20, n_samples),
            "gk_y": rng.uniform(25, 45, n_samples),
        }
    )
    model = GhostGkModel(n_estimators=n_estimators)
    model.fit(X, labels)
    _FITTED_MODEL_CACHE[cache_key] = (model, X, labels)
    return model, X, labels


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_ghost_gk_frames(
    *,
    home_team_id: int = 1,
    away_team_id: int = 2,
    period_id: int = 1,
    frame_id: int = 1,
    game_id: str = "100",
    timestamp: float = 1.0,
) -> pd.DataFrame:
    """Synthetic frames with ball, GK, defenders, attackers."""
    rows = []
    base = {
        "game_id": game_id,
        "period_id": period_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "ball_state": "alive",
        "time_seconds": timestamp,
        "source_provider": "test",
    }
    # Ball
    rows.append(
        {
            **base,
            "player_id": "ball",
            "team_id": None,
            "x": 50.0,
            "y": 34.0,
            "vx": 2.0,
            "vy": 0.0,
            "speed": 2.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    # Home GK (defending goal at x=0)
    rows.append(
        {
            **base,
            "player_id": "p1",
            "team_id": home_team_id,
            "x": 5.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": False,
            "is_goalkeeper": True,
        }
    )
    # Home defenders (4)
    for i, (px, py) in enumerate([(20, 25), (22, 30), (21, 38), (23, 45)]):
        rows.append(
            {
                **base,
                "player_id": f"p{10 + i}",
                "team_id": home_team_id,
                "x": float(px),
                "y": float(py),
                "vx": 0.5,
                "vy": 0.0,
                "speed": 0.5,
                "is_ball": False,
                "is_goalkeeper": False,
            }
        )
    # Away attackers (4)
    for i, (px, py) in enumerate([(40, 30), (45, 34), (38, 40), (50, 34)]):
        rows.append(
            {
                **base,
                "player_id": f"a{10 + i}",
                "team_id": away_team_id,
                "x": float(px),
                "y": float(py),
                "vx": -1.0,
                "vy": 0.0,
                "speed": 1.0,
                "is_ball": False,
                "is_goalkeeper": False,
            }
        )
    # Away GK
    rows.append(
        {
            **base,
            "player_id": "a1",
            "team_id": away_team_id,
            "x": 100.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "is_ball": False,
            "is_goalkeeper": True,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 2: Dataclass + Grid
# ---------------------------------------------------------------------------


class TestGhostGkDensity:
    """GhostGkDensity frozen dataclass and grid specification."""

    def test_density_grid_bounds(self):
        """grid_x covers [0,30], grid_y covers [18,50], probabilities shape (60,64)."""
        from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GhostGkDensity

        probs = np.ones((60, 64)) / (60 * 64)
        density = GhostGkDensity(
            mode_x=5.0,
            mode_y=34.0,
            mean_x=5.5,
            mean_y=34.2,
            spread=12.0,
            grid_x=_GRID_X.copy(),
            grid_y=_GRID_Y.copy(),
            probabilities=probs,
        )
        assert density.grid_x.shape == (60,)
        assert density.grid_y.shape == (64,)
        assert density.probabilities.shape == (60, 64)
        assert density.grid_x[0] >= 0.0
        assert density.grid_x[-1] <= 30.0
        assert density.grid_y[0] >= 18.0
        assert density.grid_y[-1] <= 50.0
        np.testing.assert_allclose(density.probabilities.sum(), 1.0, atol=1e-6)

    def test_density_is_frozen(self):
        """GhostGkDensity is immutable (attributes + array write protection)."""
        from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GhostGkDensity

        probs = np.ones((60, 64)) / (60 * 64)
        density = GhostGkDensity(
            mode_x=5.0,
            mode_y=34.0,
            mean_x=5.5,
            mean_y=34.2,
            spread=12.0,
            grid_x=_GRID_X.copy(),
            grid_y=_GRID_Y.copy(),
            probabilities=probs,
        )
        with pytest.raises(AttributeError):
            density.mode_x = 10.0  # type: ignore[misc]
        # Array mutation prevented
        with pytest.raises(ValueError):
            density.probabilities[0, 0] = 999.0


# ---------------------------------------------------------------------------
# Task 3: Feature Extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """extract_ghost_gk_features: 26 columns, goal-relative normalization."""

    def test_extract_ghost_gk_features(self):
        """26 feature columns, correct dtypes, team_in_possession present."""
        from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, extract_ghost_gk_features

        frames = _make_ghost_gk_frames(home_team_id=1, away_team_id=2)
        features = extract_ghost_gk_features(
            frames,
            gk_team_id=1,
            goal_x=0.0,
        )
        assert features.shape[1] == 26
        assert list(features.columns) == GHOST_GK_FEATURE_NAMES
        assert "team_in_possession" in features.columns
        assert "ball_x" in features.columns

    def test_goal_relative_transform_symmetry(self):
        """LTR vs RTL frames produce identical goal-relative coords."""
        from silly_kicks.tracking._ghost_gk import extract_ghost_gk_features

        frames_ltr = _make_ghost_gk_frames(home_team_id=1, away_team_id=2)
        # Mirror: flip x and vx for RTL equivalent
        frames_rtl = frames_ltr.copy()
        frames_rtl["x"] = 105.0 - frames_rtl["x"]
        frames_rtl["vx"] = -frames_rtl["vx"]

        feat_ltr = extract_ghost_gk_features(frames_ltr, gk_team_id=1, goal_x=0.0)
        feat_rtl = extract_ghost_gk_features(frames_rtl, gk_team_id=1, goal_x=105.0)

        np.testing.assert_allclose(
            feat_ltr["ball_x"].iloc[0],
            feat_rtl["ball_x"].iloc[0],
            atol=0.01,
        )
        np.testing.assert_allclose(
            feat_ltr["ball_distance_to_goal"].iloc[0],
            feat_rtl["ball_distance_to_goal"].iloc[0],
            atol=0.01,
        )

    def test_defending_team_compactness_degenerate(self):
        """Collinear defenders -> NaN compactness (QhullError caught)."""
        from silly_kicks.tracking._ghost_gk import extract_ghost_gk_features

        frames = _make_ghost_gk_frames()
        # Make all defenders collinear
        mask = (~frames["is_ball"]) & (~frames["is_goalkeeper"]) & (frames["team_id"] == 1)
        frames.loc[mask, "y"] = 34.0

        features = extract_ghost_gk_features(frames, gk_team_id=1, goal_x=0.0)
        assert np.isnan(features["defending_team_compactness"].iloc[0])


# ---------------------------------------------------------------------------
# Task 4: GhostGkModel
# ---------------------------------------------------------------------------


class TestGhostGkModel:
    """GhostGkModel fit/predict/predict_density."""

    def test_ghost_gk_model_fit_predict(self):
        """fit() + predict() returns (n, 2) joint mode."""
        model, X, _ = _fitted_model()
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 2)
        assert np.all(predictions[:, 0] >= GRID_X_MIN)
        assert np.all(predictions[:, 0] <= GRID_X_MAX)
        assert np.all(predictions[:, 1] >= GRID_Y_MIN)
        assert np.all(predictions[:, 1] <= GRID_Y_MAX)

    def test_ghost_gk_density(self):
        """predict_density() grid sums to ~1.0, mode in bounds."""
        from silly_kicks.tracking._ghost_gk import GhostGkDensity

        model, X, _ = _fitted_model()
        densities = model.predict_density(X[:5])
        assert len(densities) == 5
        for d in densities:
            assert isinstance(d, GhostGkDensity)
            np.testing.assert_allclose(d.probabilities.sum(), 1.0, atol=0.01)
            assert GRID_X_MIN <= d.mode_x <= GRID_X_MAX
            assert GRID_Y_MIN <= d.mode_y <= GRID_Y_MAX

    def test_ghost_gk_density_joint_mode(self):
        """Mode is argmax of 2D grid (not two independent 1D argmaxes)."""
        model, X, _ = _fitted_model()
        densities = model.predict_density(X[:3])
        for d in densities:
            flat_idx = np.argmax(d.probabilities)
            ix, iy = np.unravel_index(flat_idx, d.probabilities.shape)
            np.testing.assert_allclose(d.mode_x, d.grid_x[ix])
            np.testing.assert_allclose(d.mode_y, d.grid_y[iy])

    def test_ghost_gk_predict_with_nan_features(self):
        """NaN velocity features -> finite predictions in bounds."""
        model, X, _ = _fitted_model()
        X_nan = X[:5].copy()
        for col in ["ball_vx", "ball_vy", "ball_speed", "defensive_line_speed", "defending_centroid_vx"]:
            X_nan[col] = np.nan
        predictions = model.predict(X_nan)
        assert predictions.shape == (5, 2)
        assert np.all(np.isfinite(predictions))


# ---------------------------------------------------------------------------
# Task 5: Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """GhostGkModel save/load — npz + SHA-256, no pickle."""

    def test_ghost_gk_model_save_load(self):
        """Round-trip: save -> load -> predict matches."""
        model, X, _ = _fitted_model()
        preds_before = model.predict(X[:5])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ghost_gk_test"
            model.save(save_path)

            # Verify artifact structure
            assert (save_path / "rfcde_weights.npz").exists()
            assert (save_path / "metadata.json").exists()
            assert (save_path / "SHA256SUMS").exists()

            # No pickle/joblib files
            import glob as glob_mod

            bad = glob_mod.glob(str(save_path / "*.pkl")) + glob_mod.glob(str(save_path / "*.joblib"))
            assert len(bad) == 0

            loaded = model.__class__.load(save_path)
            preds_after = loaded.predict(X[:5])

        np.testing.assert_allclose(preds_before, preds_after, atol=0.01)

    def test_ghost_gk_model_sha256_verification(self):
        """Tampered artifact raises IntegrityError on load."""
        from silly_kicks.tracking._ghost_gk import IntegrityError

        model, _, _ = _fitted_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ghost_gk_tampered"
            model.save(save_path)

            # Tamper with npz
            npz_path = save_path / "rfcde_weights.npz"
            with open(npz_path, "ab") as f:
                f.write(b"TAMPERED")

            with pytest.raises(IntegrityError):
                model.__class__.load(save_path)


# ---------------------------------------------------------------------------
# Task 6: Fail-Fast + _resolve_model
# ---------------------------------------------------------------------------


class TestFailFast:
    """Fail-fast when [ghost-gk] extra not installed or model unavailable."""

    def test_ghost_gk_missing_huggingface_hub(self):
        """Mock huggingface_hub unavailable -> ImportError with install instructions."""
        from silly_kicks.tracking._ghost_gk import _resolve_model

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match=r"pip install silly-kicks\[ghost-gk\]"):
                _resolve_model(None)

    def test_resolve_model_passthrough(self):
        """Pre-loaded model passed through without download."""
        from silly_kicks.tracking._ghost_gk import _resolve_model

        model, _, _ = _fitted_model()
        assert _resolve_model(model) is model


# ---------------------------------------------------------------------------
# Task 7: compute_ghost_gk
# ---------------------------------------------------------------------------


class TestComputeGhostGk:
    """compute_ghost_gk batched per-frame primitive."""

    def test_compute_ghost_gk_adds_columns(self):
        """Adds ghost_gk_x, ghost_gk_y, ghost_gk_spread per frame."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model, _, _ = _fitted_model()

        frames = pd.concat(
            [
                _make_ghost_gk_frames(frame_id=1, timestamp=1.0),
                _make_ghost_gk_frames(frame_id=2, timestamp=1.5),
            ],
            ignore_index=True,
        )

        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        assert "ghost_gk_x" in result.columns
        assert "ghost_gk_y" in result.columns
        assert "ghost_gk_spread" in result.columns
        # Results on GK rows
        gk_mask = result["is_goalkeeper"].astype(bool) & ~result["is_ball"].astype(bool)
        assert result.loc[gk_mask, "ghost_gk_x"].notna().any()

    def test_compute_ghost_gk_requires_ltr_normalized(self):
        """Documents: input must be LTR-normalized."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model, _, _ = _fitted_model()
        frames = _make_ghost_gk_frames()
        # Should work without error (frames are already LTR-like)
        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        assert len(result) == len(frames)


# ---------------------------------------------------------------------------
# Task 8: add_ghost_gk + ghost_gk_xfns
# ---------------------------------------------------------------------------


class TestAggregatorAndXfns:
    """add_ghost_gk aggregator + ghost_gk_xfns VAEP factory."""

    def _make_actions(self):
        return pd.DataFrame(
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

    def test_add_ghost_gk_aggregator(self):
        """Expected columns, no provenance leak."""
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()
        actions = self._make_actions()
        frames = pd.concat(
            [
                _make_ghost_gk_frames(frame_id=1, timestamp=1.0),
                _make_ghost_gk_frames(frame_id=2, timestamp=2.0),
            ],
            ignore_index=True,
        )

        result = add_ghost_gk(actions, frames, model=model, home_team_id=1)
        assert "ghost_gk_x" in result.columns
        assert "ghost_gk_y" in result.columns
        assert "ghost_gk_spread" in result.columns
        assert len(result) == len(actions)
        # No provenance leak
        assert "time_offset_seconds" not in result.columns
        assert "link_quality_score" not in result.columns

    def test_ghost_gk_xfns_factory(self):
        """Correct column names, silent NaN on dummy gamestate."""
        from silly_kicks.tracking.features import ghost_gk_xfns

        model, _, _ = _fitted_model()
        xfns = ghost_gk_xfns(model=model, home_team_id=1)
        assert len(xfns) == 1

        dummy = pd.DataFrame(
            {
                "game_id": ["100"] * 10,
                "action_id": range(10),
                "period_id": [1] * 10,
                "time_seconds": [1.0] * 10,
                "team_id": [1] * 10,
                "player_id": [1] * 10,
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [55.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [1] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]

        # With frames=None -> silent NaN (VAEP introspection path)
        result = xfns[0](states, None)
        assert all("ghost_gk" in col for col in result.columns)
        assert result.isna().all().all()
        # 3 columns x 3 states = 9
        assert result.shape[1] == 9
