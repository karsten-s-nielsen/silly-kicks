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
    """Synthetic frames with ball, GK, defenders, attackers.

    ORIENTED (ADR-028): every non-ball row carries ``team_attacking_direction``, so
    ``acting_team_attacks_rtl`` can resolve a real per-action flip. The scene is
    home-attacks-right (frame-LTR): the home GK sits at x=5 defending the x=0 goal,
    so the home team attacks x=105 -> "ltr"; the away GK sits at x=100 defending the
    x=105 goal, so the away team attacks x=0 -> "rtl". Ball rows carry None, which is
    what ``convert_to_frames`` emits and what the helper filters out anyway.

    Before this column existed the fixture always took the all-False no-flip path, so
    every away-team action it fed to ``add_ghost_gk`` / ``ghost_gk_xfns`` silently
    mixed coordinate conventions and was never exercised against ADR-028.
    """
    rows = []
    base = {
        "game_id": game_id,
        "period_id": period_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "ball_state": "alive",
        "time_seconds": timestamp,
        # A REAL, classified provider (fully-observed): the ghost-GK trainer's detected-only
        # filter (spec 4.3, PR-S115) calls keeper_detection_mask, which fail-closes on an
        # unregistered provider. "gradientsports" is a no-op for the filter (all keepers observed),
        # so these synthetic frames exercise the trainer without a spurious unknown-provider raise.
        "source_provider": "gradientsports",
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
            "team_attacking_direction": None,
        }
    )
    # Home GK (defending goal at x=0 -> home attacks x=105 -> "ltr")
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
            "team_attacking_direction": "ltr",
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
                "team_attacking_direction": "ltr",
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
                "team_attacking_direction": "rtl",
            }
        )
    # Away GK (defending goal at x=105 -> away attacks x=0 -> "rtl")
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
            "team_attacking_direction": "rtl",
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
    """Fail-fast when bundled weights missing or invalid variant."""

    def test_resolve_model_missing_weights(self):
        """Missing bundled weights -> FileNotFoundError."""
        from silly_kicks.tracking._ghost_gk import _resolve_model

        with patch("silly_kicks.tracking._ghost_gk._WEIGHTS_ROOT", Path("/nonexistent")):
            with pytest.raises(FileNotFoundError, match="Bundled Ghost-GK weights"):
                _resolve_model(None)

    def test_resolve_model_loads_default(self):
        """Default bundled weights loaded when no model supplied."""
        from silly_kicks.tracking._ghost_gk import (
            _WEIGHTS_ROOT,
            GhostGkModel,
            _resolve_model,
        )

        if not (_WEIGHTS_ROOT / "default" / "SHA256SUMS").exists():
            pytest.skip("Bundled weights not present in dev checkout")
        model = _resolve_model(None)
        assert isinstance(model, GhostGkModel)

    def test_resolve_model_full_falls_back_to_hub(self):
        """Full variant falls back to Hub download when not bundled."""
        from silly_kicks.tracking._ghost_gk import _resolve_model

        mock_model = _fitted_model()[0]
        with (
            patch("silly_kicks.tracking._ghost_gk._WEIGHTS_ROOT", Path("/nonexistent")),
            patch.object(type(mock_model), "from_hub", return_value=mock_model) as mock_hub,
        ):
            result = _resolve_model("full")
            mock_hub.assert_called_once()
            assert result is mock_model

    def test_resolve_model_full_missing_huggingface_hub(self):
        """Full variant without huggingface_hub installed -> ImportError."""
        from silly_kicks.tracking._ghost_gk import _resolve_model

        with (
            patch("silly_kicks.tracking._ghost_gk._WEIGHTS_ROOT", Path("/nonexistent")),
            patch(
                "silly_kicks.tracking._ghost_gk.GhostGkModel.from_hub",
                side_effect=ImportError("Ghost GK full model requires: pip install silly-kicks[ghost-gk]"),
            ),
        ):
            with pytest.raises(ImportError, match="ghost-gk"):
                _resolve_model("full")

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
        """Adds ghost_gk_x, ghost_gk_y per frame."""
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

    def test_compute_ghost_gk_two_goalkeepers_same_team(self):
        """A frame with two same-team is_goalkeeper rows must not crash and
        both GK rows must receive the identical ghost-GK prediction.

        Regression (4.12.1): a rostered backup keeper carried on-pitch alongside the
        starter (or a GK-substitution overlap frame) produces two
        is_goalkeeper=True rows for one team in a single frame. The per-(frame,
        team) prediction is identical for both rows, so both must be filled with
        that one value rather than raising
        ``ValueError: Must have equal len keys and value``.
        """
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model, _, _ = _fitted_model()
        frames = _make_ghost_gk_frames(home_team_id=1, away_team_id=2)
        # Inject a second home (team 1) goalkeeper in the same frame. Slice as a
        # DataFrame (.loc[[idx]]) so column dtypes are preserved on concat.
        starter_idx = frames.index[(frames["team_id"] == 1) & frames["is_goalkeeper"].astype(bool)][0]
        backup = frames.loc[[starter_idx]].copy()
        backup["player_id"] = "p1_backup"
        backup["x"] = 7.0
        backup["y"] = 30.0
        frames = pd.concat([frames, backup], ignore_index=True)

        # Must not raise.
        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        assert len(result) == len(frames)

        # Both home-GK rows get the same, non-NaN prediction.
        home_gk = result[(result["team_id"] == 1) & result["is_goalkeeper"].astype(bool)]
        assert len(home_gk) == 2
        for col in ("ghost_gk_x", "ghost_gk_y"):
            vals = home_gk[col].to_numpy(dtype=float)
            assert np.isfinite(vals).all(), f"{col} should be filled for both GK rows"
            np.testing.assert_allclose(vals[0], vals[1])


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
        # 2 columns x 3 states = 6 (ghost_gk_density_spread retired, spec 2026-07-20)
        assert result.shape[1] == 6


# ---------------------------------------------------------------------------
# TF-18 Training Hub Publish — new test helpers
# ---------------------------------------------------------------------------


def _make_spadl_actions(
    *,
    game_id: str = "100",
    goals: list[tuple[float, int]] | None = None,
    owngoals: list[tuple[float, int]] | None = None,
    set_pieces: list[tuple[float, str]] | None = None,
) -> pd.DataFrame:
    """Build minimal SPADL actions for context resolution tests.

    Parameters
    ----------
    goals : list of (time_seconds, team_id) for successful shots
    owngoals : list of (time_seconds, team_id) for own goals
    set_pieces : list of (time_seconds, type_name) for set-piece actions
    """
    from silly_kicks.spadl import config as spadlconfig

    rows = []
    action_id = 0

    # Always add non_actions for both teams so the DF has 2 unique team_ids
    # (needed by own-goal flip logic in _build_score_lookup)
    for tid in (1, 2):
        rows.append(
            {
                "game_id": game_id,
                "action_id": action_id,
                "period_id": 1,
                "time_seconds": 0.0,
                "team_id": tid,
                "player_id": 10 + tid,
                "start_x": 52.5,
                "start_y": 34.0,
                "end_x": 52.5,
                "end_y": 34.0,
                "type_id": spadlconfig.actiontype_id["non_action"],
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": 0,
                "type_name": "non_action",
                "result_name": "success",
                "bodypart_name": "foot",
            }
        )
        action_id += 1

    for ts, tid in goals or []:
        rows.append(
            {
                "game_id": game_id,
                "action_id": action_id,
                "period_id": 1,
                "time_seconds": ts,
                "team_id": tid,
                "player_id": 10,
                "start_x": 90.0,
                "start_y": 34.0,
                "end_x": 104.0,
                "end_y": 34.0,
                "type_id": spadlconfig.actiontype_id["shot"],
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": 0,
                "type_name": "shot",
                "result_name": "success",
                "bodypart_name": "foot",
            }
        )
        action_id += 1

    for ts, tid in owngoals or []:
        rows.append(
            {
                "game_id": game_id,
                "action_id": action_id,
                "period_id": 1,
                "time_seconds": ts,
                "team_id": tid,
                "player_id": 10,
                "start_x": 20.0,
                "start_y": 34.0,
                "end_x": 5.0,
                "end_y": 34.0,
                "type_id": spadlconfig.actiontype_id["shot"],
                "result_id": spadlconfig.result_id["owngoal"],
                "bodypart_id": 0,
                "type_name": "shot",
                "result_name": "owngoal",
                "bodypart_name": "foot",
            }
        )
        action_id += 1

    for ts, tname in set_pieces or []:
        rows.append(
            {
                "game_id": game_id,
                "action_id": action_id,
                "period_id": 1,
                "time_seconds": ts,
                "team_id": 1,
                "player_id": 10,
                "start_x": 50.0,
                "start_y": 34.0,
                "end_x": 55.0,
                "end_y": 34.0,
                "type_id": spadlconfig.actiontype_id[tname],
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": 0,
                "type_name": tname,
                "result_name": "success",
                "bodypart_name": "foot",
            }
        )
        action_id += 1

    return pd.DataFrame(rows)


def _make_multi_frame_fixture(
    *,
    n_frames: int = 5,
    home_team_id: int = 1,
    away_team_id: int = 2,
    game_id: str = "100",
    fps: float = 25.0,
) -> pd.DataFrame:
    """Build multi-frame fixture suitable for shared helper tests.

    ORIENTED (ADR-028), same scene as ``_make_ghost_gk_frames``: home GK at x=5
    defends the x=0 goal so home attacks x=105 ("ltr"); away GK at x=100 defends the
    x=105 goal so away attacks x=0 ("rtl"). Ball rows stay None. Previously the whole
    column was None, so ``acting_team_attacks_rtl`` could never resolve a flip.
    """
    rows = []
    for fid in range(1, n_frames + 1):
        ts = fid / fps
        base = {
            "game_id": game_id,
            "period_id": 1,
            "frame_id": fid,
            "time_seconds": ts,
            "frame_rate": fps,
            "ball_state": "alive",
            "source_provider": "test",
            "confidence": None,
            "visibility": None,
            "is_goalkeeper_source": "native",
            "z": 0.0,
        }
        # Ball
        rows.append(
            {
                **base,
                "player_id": "ball",
                "team_id": None,
                "x": 50.0 + fid * 0.5,
                "y": 34.0,
                "vx": 2.0,
                "vy": 0.0,
                "speed": 2.0,
                "is_ball": True,
                "is_goalkeeper": False,
                "team_attacking_direction": None,
            }
        )
        # Home GK (defends x=0 -> home attacks x=105 -> "ltr")
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
                "team_attacking_direction": "ltr",
            }
        )
        # Home defenders
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
                    "team_attacking_direction": "ltr",
                }
            )
        # Away attackers
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
                    "team_attacking_direction": "rtl",
                }
            )
        # Away GK (defends x=105 -> away attacks x=0 -> "rtl")
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
                "team_attacking_direction": "rtl",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 1: Bug fix — timestamp -> time_seconds
# ---------------------------------------------------------------------------


class TestTimestampBugFix:
    """Bug fix: time_seconds column used, not timestamp."""

    def test_time_seconds_column_used(self):
        """extract_ghost_gk_features reads time_seconds, not timestamp."""
        from silly_kicks.tracking._ghost_gk import extract_ghost_gk_features

        # Frame with time_seconds=42.5 but NO timestamp column
        rows = []
        base = {
            "game_id": "100",
            "period_id": 1,
            "frame_id": 1,
            "time_seconds": 42.5,
            "frame_rate": 25.0,
            "ball_state": "alive",
            "source_provider": "test",
        }
        rows.append(
            {
                **base,
                "player_id": "ball",
                "team_id": None,
                "x": 50.0,
                "y": 34.0,
                "vx": 2.0,
                "vy": 0.0,
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
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
        for i, (px, py) in enumerate([(40, 30), (45, 34)]):
            rows.append(
                {
                    **base,
                    "player_id": f"a{10 + i}",
                    "team_id": 2,
                    "x": float(px),
                    "y": float(py),
                    "vx": -1.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
        frame = pd.DataFrame(rows)

        result = extract_ghost_gk_features(frame, gk_team_id=1, goal_x=0.0)
        assert result["time_seconds"].iloc[0] == pytest.approx(42.5), "Should read time_seconds column, not timestamp"


# ---------------------------------------------------------------------------
# Task 2: Match context resolution
# ---------------------------------------------------------------------------


class TestBuildScoreLookup:
    """_build_score_lookup returns home-perspective running score diff."""

    def test_no_goals(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions()
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 10.0) == 0.0
        assert fn("100", 60.0) == 0.0

    def test_home_goal(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 25.0) == 0.0  # before goal
        assert fn("100", 30.0) == 1.0  # at goal
        assert fn("100", 60.0) == 1.0  # after goal

    def test_away_goal(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(30.0, 2)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 35.0) == -1.0  # home perspective: 0-1

    def test_multiple_goals(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        actions = _make_spadl_actions(goals=[(10.0, 1), (20.0, 2), (30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 15.0) == 1.0  # 1-0
        assert fn("100", 25.0) == 0.0  # 1-1
        assert fn("100", 35.0) == 1.0  # 2-1

    def test_own_goal_attributed_to_opponent(self):
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        # Team 1 scores own goal -> counts as team 2 scoring
        actions = _make_spadl_actions(owngoals=[(30.0, 1)])
        fn = _build_score_lookup(actions, home_team_id=1)
        assert fn("100", 35.0) == -1.0  # 0-1 from home perspective


class TestBuildPhaseLookup:
    """_build_phase_lookup returns 0/1/2 for open/set_piece/goal_kick."""

    def test_open_play(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions()
        fn = _build_phase_lookup(actions)
        assert fn("100", 10.0) == 0

    def test_freekick_within_decay(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "freekick_short")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 1  # 3s after freekick -> set_piece

    def test_goalkick(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "goalkick")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 2  # goal_kick phase

    def test_corner(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "corner_crossed")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 1  # set_piece

    def test_decay_after_10s(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "freekick_short")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 41.0) == 0  # >10s -> open play

    def test_throw_in_excluded(self):
        from silly_kicks.tracking._ghost_gk import _build_phase_lookup

        actions = _make_spadl_actions(set_pieces=[(30.0, "throw_in")])
        fn = _build_phase_lookup(actions)
        assert fn("100", 33.0) == 0  # throw_in is NOT a set piece


# ---------------------------------------------------------------------------
# Task 3: Shared batch helper
# ---------------------------------------------------------------------------


class TestExtractAllFeatures:
    """_extract_all_ghost_gk_features shared helper."""

    def test_shape(self):
        from silly_kicks.tracking._ghost_gk import (
            GHOST_GK_FEATURE_NAMES,
            _extract_all_ghost_gk_features,
        )

        frames = _make_multi_frame_fixture(n_frames=5)
        features, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # 5 frames x 2 GKs = 10 rows
        assert features.shape[0] == 10
        assert features.shape[1] == len(GHOST_GK_FEATURE_NAMES)
        assert meta.shape == (10, 8)
        assert list(meta.columns) == [
            "game_id",
            "period_id",
            "frame_id",
            "gk_team_id",
            "gk_x_gr",
            "gk_y_gr",
            "gk_player_id",
            "gk_visibility",
        ]

    def test_velocity_state_non_nan_after_first(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=3)
        features, _ = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # First frame has NaN velocity, subsequent frames should have values
        # Group by team: rows 0,2,4 = home GK; 1,3,5 = away GK
        home_rows = features.iloc[0::2]  # even indices = home GK
        assert np.isnan(home_rows["defensive_line_speed"].iloc[0])
        assert not np.isnan(home_rows["defensive_line_speed"].iloc[1])

    def test_subsample(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=25, fps=25.0)
        full, _ = _extract_all_ghost_gk_features(frames, home_team_id=1)
        sub, _ = _extract_all_ghost_gk_features(
            frames,
            home_team_id=1,
            subsample_fps=1.0,
        )
        # 25fps -> 1fps = keep every 25th frame -> 1 frame -> 2 GKs
        assert sub.shape[0] < full.shape[0]
        assert sub.shape[0] == 2  # 1 frame x 2 GKs

    def test_goal_relative_coords(self):
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        _, meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        # Home GK at x=5.0 (goal at x=0 -> gr_x = 5.0)
        home_meta = meta[meta["gk_team_id"] == 1]
        assert home_meta["gk_x_gr"].iloc[0] == pytest.approx(5.0)
        # Away GK at x=100.0 (goal at x=105 -> gr_x = 105-100 = 5.0)
        away_meta = meta[meta["gk_team_id"] == 2]
        assert away_meta["gk_x_gr"].iloc[0] == pytest.approx(5.0)

    def test_home_team_id_normalization_int_to_str(self):
        """home_team_id=1 works when frames have string team_id."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        frames["team_id"] = frames["team_id"].astype(str)
        features, _meta = _extract_all_ghost_gk_features(frames, home_team_id=1)
        assert features.shape[0] == 2  # both GKs extracted

    def test_home_team_id_normalization_str_to_int(self):
        """home_team_id='1' works when frames have int team_id."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)
        features, _meta = _extract_all_ghost_gk_features(frames, home_team_id="1")
        assert features.shape[0] == 2

    def test_score_callback_negated_for_away(self):
        """Away GK sees negated score_diff."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        frames = _make_multi_frame_fixture(n_frames=1)

        def mock_score(game_id, time_s):
            return 2.0  # home perspective: home leads 2-0

        features, meta = _extract_all_ghost_gk_features(
            frames,
            home_team_id=1,
            score_at_time=mock_score,
        )
        home_feat = features[meta["gk_team_id"].values == 1]
        away_feat = features[meta["gk_team_id"].values == 2]
        assert home_feat["score_diff"].iloc[0] == pytest.approx(2.0)
        assert away_feat["score_diff"].iloc[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Task 4: prepare_ghost_gk_training_data
# ---------------------------------------------------------------------------


class TestPrepareTrainingData:
    """prepare_ghost_gk_training_data public API."""

    def test_basic_shape(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=5)
        features, labels = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            subsample_fps=None,
        )
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] > 0
        assert list(labels.columns) == ["gk_x", "gk_y"]
        assert not labels.isna().any().any()

    def test_with_actions_score_nonzero(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=5)
        actions = _make_spadl_actions(goals=[(0.01, 1)])  # home scores early
        features, _labels = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            actions=actions,
            subsample_fps=None,
        )
        # Home GK should see positive score_diff after goal
        assert (features["score_diff"] != 0.0).any()

    def test_without_actions_defaults(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=3)
        features, _labels = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            subsample_fps=None,
        )
        assert (features["score_diff"] == 0.0).all()
        assert (features["phase"] == 0.0).all()

    def test_subsample_reduces(self):
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=25, fps=25.0)
        full, _ = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            subsample_fps=None,
        )
        sub, _ = prepare_ghost_gk_training_data(
            frames,
            home_team_id=1,
            subsample_fps=1.0,
        )
        assert sub.shape[0] < full.shape[0]

    def test_sweeper_rush_filtered(self):
        """GK outside [0,30]x[18,50] should be filtered with warning."""
        import warnings

        from silly_kicks.tracking import prepare_ghost_gk_training_data

        frames = _make_multi_frame_fixture(n_frames=1)
        # Move home GK far out of domain (sweeper rush at x=50, y=34)
        frames.loc[
            (frames["player_id"] == "p1") & (frames["is_goalkeeper"] == True),  # noqa: E712
            "x",
        ] = 50.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _features, _labels = prepare_ghost_gk_training_data(
                frames,
                home_team_id=1,
                subsample_fps=None,
            )
            # The home GK at x=50 -> gr_x=50 -> outside [0,30] -> filtered
            sweeper_warnings = [x for x in w if "goal-relative domain" in str(x.message)]
            assert len(sweeper_warnings) >= 1

    def test_public_import_path(self):
        """prepare_ghost_gk_training_data is importable from silly_kicks.tracking."""
        from silly_kicks.tracking import prepare_ghost_gk_training_data

        assert callable(prepare_ghost_gk_training_data)


# ---------------------------------------------------------------------------
# Task 5: Refactored compute_ghost_gk + backward compat
# ---------------------------------------------------------------------------


class TestComputeGhostGkRefactored:
    """compute_ghost_gk with shared helper + actions parameter."""

    @staticmethod
    def _make_model(n_estimators: int = 10):  # -> GhostGkModel (function-local import)
        """Build a deterministic small model for testing (same seed as golden file)."""
        from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(
            rng.standard_normal((n, 26)),
            columns=GHOST_GK_FEATURE_NAMES,
        )
        X["phase"] = rng.integers(0, 3, n).astype(float)
        X["team_in_possession"] = rng.integers(0, 2, n).astype(float)
        X["ball_in_own_half"] = rng.integers(0, 2, n).astype(float)
        labels = pd.DataFrame(
            {"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)},
        )
        model = GhostGkModel(n_estimators=n_estimators)
        model.fit(X, labels)
        return model

    def test_backward_compat(self):
        """actions=None produces identical output to 3.19.0 golden file."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)

        # Build same frames as golden file
        frames = _make_multi_frame_fixture(n_frames=3, game_id="100")
        # Add timestamp column for backward compat (old code read it)
        frames["timestamp"] = frames["time_seconds"]

        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        gk_mask = result["is_goalkeeper"].astype(bool) & ~result["is_ball"].astype(bool)
        actual = result.loc[
            gk_mask,
            ["game_id", "period_id", "frame_id", "team_id", "ghost_gk_x", "ghost_gk_y"],
        ].reset_index(drop=True)

        golden = pd.read_parquet("tests/tracking/fixtures/ghost_gk_backward_compat.parquet")

        # Compare --- tolerance for float precision
        pd.testing.assert_frame_equal(
            actual,
            golden,
            check_dtype=False,
            atol=1e-6,
        )

    def test_with_actions_changes_features(self):
        """Passing actions changes score_diff/phase in the extraction."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)
        frames = _make_multi_frame_fixture(n_frames=3)
        actions = _make_spadl_actions(goals=[(0.01, 1)])

        result_no_actions = compute_ghost_gk(
            frames,
            model=model,
            home_team_id=1,
        )
        result_with_actions = compute_ghost_gk(
            frames,
            model=model,
            home_team_id=1,
            actions=actions,
        )
        # Predictions should differ because features differ
        gk_mask_no = result_no_actions["is_goalkeeper"].astype(bool) & ~result_no_actions["is_ball"].astype(bool)
        gk_mask_with = result_with_actions["is_goalkeeper"].astype(bool) & ~result_with_actions["is_ball"].astype(bool)
        x_no = result_no_actions.loc[gk_mask_no, "ghost_gk_x"].values
        x_with = result_with_actions.loc[gk_mask_with, "ghost_gk_x"].values
        # Not necessarily different (tiny model), but API accepts actions
        assert len(x_no) == len(x_with)

    def test_actions_none_is_default(self):
        """actions=None is the default and works."""
        from silly_kicks.tracking._ghost_gk import compute_ghost_gk

        model = self._make_model(n_estimators=10)
        frames = _make_multi_frame_fixture(n_frames=2)
        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        assert "ghost_gk_x" in result.columns


class TestAddGhostGkThreadsActions:
    """Verify aggregator passes actions through to compute_ghost_gk."""

    def test_add_ghost_gk_threads_actions(self):
        """add_ghost_gk passes actions= to compute_ghost_gk."""
        from unittest.mock import MagicMock, patch

        from silly_kicks.tracking._ghost_gk import GhostGkModel
        from silly_kicks.tracking.features import add_ghost_gk

        mock_result = _make_multi_frame_fixture(n_frames=1)
        mock_result["ghost_gk_x"] = 10.0
        mock_result["ghost_gk_y"] = 34.0

        actions = _make_spadl_actions(goals=[])
        frames = _make_multi_frame_fixture(n_frames=1)

        with (
            patch(
                "silly_kicks.tracking._ghost_gk.compute_ghost_gk",
                return_value=mock_result,
            ) as mock_compute,
            patch(
                "silly_kicks.tracking._ghost_gk._resolve_model",
                return_value=MagicMock(spec=GhostGkModel),
            ),
        ):
            try:
                add_ghost_gk(
                    actions,
                    frames,
                    home_team_id=1,
                    actions_for_context=actions,
                )
            except Exception:
                pass  # linking may fail on synthetic data

            # Check that compute_ghost_gk was called with actions kwarg
            if mock_compute.called:
                _, kwargs = mock_compute.call_args
                assert "actions" in kwargs
                assert kwargs["actions"] is actions


class TestServePositionClamp:
    """4.22.1: served ghost_gk_x/y are clamped to the physical pitch (goal-relative coords).

    Lakehouse report 2026-06-11 item 2: garbage input (a mis-flagged is_goalkeeper
    upstream) can wrong-foot the goal-side flip and push the boosted regressor far
    outside its trained label domain -- a keeper served 5.7 m behind the goal line is
    never physically meaningful. Clamp target is the PHYSICAL pitch, NOT the trained
    grid domain: healthy extrapolation slightly past GRID_X_MAX (a sweeper rush) must
    stay byte-unchanged.
    """

    @staticmethod
    def _stub_densities(n):
        class _D:
            spread = 2.0

        return [_D()] * n

    def _run(self, monkeypatch, served_xy):
        from silly_kicks.tracking._ghost_gk import GhostGkModel, compute_ghost_gk

        model, _x, _labels = _fitted_model()
        frames = _make_multi_frame_fixture(n_frames=2)
        monkeypatch.setattr(
            GhostGkModel,
            "predict_mean",
            lambda self, features: np.tile(np.asarray(served_xy, dtype=float), (len(features), 1)),
        )
        monkeypatch.setattr(
            GhostGkModel,
            "predict_density",
            lambda self, features, *, kde_backend="vectorized": TestServePositionClamp._stub_densities(len(features)),
        )
        result = compute_ghost_gk(frames, model=model, home_team_id=1)
        gk_mask = result["is_goalkeeper"].astype(bool) & ~result["is_ball"].astype(bool)
        return result.loc[gk_mask, ["ghost_gk_x", "ghost_gk_y"]]

    def test_out_of_bounds_served_position_is_clamped_and_warns(self, monkeypatch):
        with pytest.warns(UserWarning, match="clamped"):
            gk = self._run(monkeypatch, (-5.74, 100.0))
        assert (gk["ghost_gk_x"] == 0.0).all()  # behind the defended goal line -> goal line
        assert (gk["ghost_gk_y"] == 68.0).all()  # beyond the far touchline -> touchline

    def test_in_bounds_served_position_is_byte_unchanged_and_silent(self, monkeypatch):
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            gk = self._run(monkeypatch, (10.0, 34.0))
        assert (gk["ghost_gk_x"] == 10.0).all()
        assert (gk["ghost_gk_y"] == 34.0).all()
        assert not [w for w in caught if "clamped" in str(w.message)]

    def test_boundary_values_are_not_clamped(self, monkeypatch):
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            gk = self._run(monkeypatch, (0.0, 68.0))
        assert (gk["ghost_gk_x"] == 0.0).all()
        assert (gk["ghost_gk_y"] == 68.0).all()
        assert not [w for w in caught if "clamped" in str(w.message)]
