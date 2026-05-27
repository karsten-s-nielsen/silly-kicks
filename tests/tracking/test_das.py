"""Unit tests for DAS adapter."""

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das import (
    _X_OFFSET,
    _Y_OFFSET,
)


class TestTodasCoords:
    def test_origin_shifts(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame(
            {
                "x": [0.0, 105.0, 52.5],
                "y": [0.0, 68.0, 34.0],
                "vx": [1.0, 2.0, 0.0],
                "vy": [0.5, -0.5, 0.0],
            }
        )
        result = _to_das_coords(df)
        np.testing.assert_allclose(result["x"].values, [-_X_OFFSET, _X_OFFSET, 0.0])
        np.testing.assert_allclose(result["y"].values, [-_Y_OFFSET, _Y_OFFSET, 0.0])
        np.testing.assert_allclose(result["vx"].values, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(result["vy"].values, [0.5, -0.5, 0.0])

    def test_does_not_mutate_input(self) -> None:
        from silly_kicks.tracking._das import _to_das_coords

        df = pd.DataFrame({"x": [50.0], "y": [30.0], "vx": [1.0], "vy": [0.0]})
        _ = _to_das_coords(df)
        assert df["x"].iloc[0] == 50.0
        assert df["y"].iloc[0] == 30.0


class TestInputValidation:
    def _minimal_frames(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [1],
                "period_id": [1],
                "frame_id": [0],
                "player_id": ["P1"],
                "team_id": ["A"],
                "x": [50.0],
                "y": [34.0],
                "vx": [0.0],
                "vy": [0.0],
                "is_ball": [False],
                "team_in_possession": ["A"],
            }
        )

    def test_missing_vx_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_vy_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vy"])
        with pytest.raises(ValueError, match="velocity columns"):
            _validate_das_inputs(df)

    def test_missing_team_in_possession_raises(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["team_in_possession"])
        with pytest.raises(ValueError, match="team_in_possession"):
            _validate_das_inputs(df)

    def test_valid_frames_pass(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames()
        _validate_das_inputs(df)


class TestPrepareFramesNumericPlayerId:
    """Regression: numeric player_id must not trigger FutureWarning on ball assignment."""

    def test_int64_player_id_no_warning(self) -> None:
        from silly_kicks.tracking._das import _prepare_frames

        df = pd.DataFrame(
            {
                "game_id": [1, 1],
                "period_id": [1, 1],
                "frame_id": [0, 0],
                "player_id": [12345, 0],  # int64
                "team_id": ["Home", "Home"],
                "x": [50.0, 52.5],
                "y": [34.0, 34.0],
                "vx": [1.0, 0.0],
                "vy": [0.0, 0.0],
                "is_ball": [False, True],
                "team_in_possession": ["Home", "Home"],
            }
        )
        assert df["player_id"].dtype == np.dtype("int64")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = _prepare_frames(df)
        assert result.loc[result["is_ball"] == True, "player_id"].iloc[0] == "ball"  # noqa: E712

    def test_float64_player_id_no_warning(self) -> None:
        from silly_kicks.tracking._das import _prepare_frames

        df = pd.DataFrame(
            {
                "game_id": [1, 1],
                "period_id": [1, 1],
                "frame_id": [0, 0],
                "player_id": [12345.0, np.nan],  # float64 (NaN for ball)
                "team_id": ["Home", "Home"],
                "x": [50.0, 52.5],
                "y": [34.0, 34.0],
                "vx": [1.0, 0.0],
                "vy": [0.0, 0.0],
                "is_ball": [False, True],
                "team_in_possession": ["Home", "Home"],
            }
        )
        assert df["player_id"].dtype == np.dtype("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = _prepare_frames(df)
        assert result.loc[result["is_ball"] == True, "player_id"].iloc[0] == "ball"  # noqa: E712


class TestImportGuard:
    def test_missing_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        monkeypatch.setattr(
            das_mod,
            "_import_accessible_space",
            lambda: (_ for _ in ()).throw(
                ImportError(
                    "accessible-space is required for DAS features. Install with: pip install 'silly-kicks[das]'"
                )
            ),
        )
        with pytest.raises(ImportError, match="silly-kicks\\[das\\]"):
            das_mod._import_accessible_space()


@pytest.mark.e2e
class TestGetDasShapeAlignment:
    def test_output_length_matches_input(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        rng = np.random.default_rng(42)
        n_frames = 3
        rows = []
        for fid in range(n_frames):
            for pid, tid in [("P1", "Home"), ("P2", "Away")]:
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "player_id": pid,
                        "team_id": tid,
                        "x": rng.uniform(10, 95),
                        "y": rng.uniform(5, 63),
                        "vx": rng.normal(0, 2),
                        "vy": rng.normal(0, 2),
                        "is_ball": False,
                        "team_in_possession": "Home",
                    }
                )
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": "ball",
                    "team_id": None,
                    "x": rng.uniform(20, 80),
                    "y": rng.uniform(10, 58),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": True,
                    "team_in_possession": "Home",
                }
            )
        frames = pd.DataFrame(rows)
        result = get_das(frames, use_progress_bar=False)
        assert len(result) == len(frames), f"get_das output length {len(result)} != input length {len(frames)}"


@pytest.mark.e2e
class TestDasTeamAsymmetry:
    """DAS must differ between attacking and defending teams.

    Bug: _precompute_das_lookup used get_das() which returns a single per-frame
    scalar, so both teams get identical DAS values and das_diff is always 0.
    Fix: use get_individual_das() aggregated per-team.
    """

    def _make_asymmetric_frame(self) -> pd.DataFrame:
        """11v11 frame with clear spatial asymmetry for DAS differentiation."""
        rng = np.random.default_rng(42)
        rows = []
        # Home team: attacking right, players in opponent half (high DAS expected)
        for i in range(11):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": 0,
                    "player_id": f"H{i}",
                    "team_id": "Home",
                    "x": rng.uniform(50, 95),
                    "y": rng.uniform(10, 58),
                    "vx": rng.normal(2, 1),
                    "vy": rng.normal(0, 1),
                    "is_ball": False,
                    "team_in_possession": "Home",
                }
            )
        # Away team: defending, clustered near own goal (low DAS expected)
        for i in range(11):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": 0,
                    "player_id": f"A{i}",
                    "team_id": "Away",
                    "x": rng.uniform(10, 40),
                    "y": rng.uniform(10, 58),
                    "vx": rng.normal(-1, 1),
                    "vy": rng.normal(0, 1),
                    "is_ball": False,
                    "team_in_possession": "Home",
                }
            )
        # Ball near midfield
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 0,
                "player_id": "ball",
                "team_id": None,
                "x": 60.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "team_in_possession": "Home",
            }
        )
        return pd.DataFrame(rows)

    def test_team_das_differs_between_teams(self) -> None:
        """get_individual_das aggregated per-team must produce different values."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_individual_das

        frames = self._make_asymmetric_frame()
        result = get_individual_das(frames, use_progress_bar=False)

        player_rows = result[result["is_ball"] != True]  # noqa: E712
        home_das = player_rows[player_rows["team_id"] == "Home"]["DAS"].sum()
        away_das = player_rows[player_rows["team_id"] == "Away"]["DAS"].sum()
        assert not np.isclose(home_das, away_das), (
            f"Individual DAS should differ between asymmetric teams: Home={home_das:.4f}, Away={away_das:.4f}"
        )

    def test_precompute_das_lookup_asymmetric(self) -> None:
        """_precompute_das_lookup must produce different DAS for each team."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = self._make_asymmetric_frame()
        lookup = _precompute_das_lookup(frames)

        # Frame (1, 0) should have both teams with different DAS
        frame_key = (1, 0)
        assert frame_key in lookup, f"Frame {frame_key} not in lookup"
        team_das = lookup[frame_key]
        assert len(team_das) == 2, f"Expected 2 teams, got {len(team_das)}"

        home_das = team_das.get("Home")
        away_das = team_das.get("Away")
        assert home_das is not None and away_das is not None
        assert not np.isclose(home_das, away_das), (
            f"DAS must differ between asymmetric teams: "
            f"Home={home_das:.4f}, Away={away_das:.4f}. "
            "If equal, _precompute_das_lookup is using per-frame scalar instead of per-team."
        )


class TestChunkSizePassthrough:
    """chunk_size kwarg must thread through add_das/das_at_action to get_individual_das."""

    def test_precompute_passes_chunk_size_to_get_individual_das(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_precompute_das_lookup(chunk_size=N) must forward to get_individual_das."""
        import silly_kicks.tracking._das as das_mod

        captured_kwargs: dict = {}

        def fake_get_individual_das(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
            captured_kwargs.update(kwargs)
            out = frames.copy()
            out["AS"] = 0.0
            out["DAS"] = 0.0
            return out

        monkeypatch.setattr(das_mod, "get_individual_das", fake_get_individual_das)

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = pd.DataFrame(
            {
                "period_id": [1],
                "frame_id": [0],
                "team_id": ["A"],
                "is_ball": [False],
                "DAS": [0.0],
            }
        )
        _precompute_das_lookup(frames, chunk_size=500)
        assert captured_kwargs.get("chunk_size") == 500

    def test_precompute_omits_chunk_size_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When chunk_size is None, chunk_size must NOT appear in kwargs."""
        import silly_kicks.tracking._das as das_mod

        captured_kwargs: dict = {}

        def fake_get_individual_das(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
            captured_kwargs.update(kwargs)
            out = frames.copy()
            out["AS"] = 0.0
            out["DAS"] = 0.0
            return out

        monkeypatch.setattr(das_mod, "get_individual_das", fake_get_individual_das)

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = pd.DataFrame(
            {
                "period_id": [1],
                "frame_id": [0],
                "team_id": ["A"],
                "is_ball": [False],
                "DAS": [0.0],
            }
        )
        _precompute_das_lookup(frames)
        assert "chunk_size" not in captured_kwargs

    def test_add_das_threads_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """add_das(chunk_size=N) must reach _precompute_das_lookup."""
        import silly_kicks.tracking.features as feat_mod

        captured_cs: list = []

        def spy_precompute(frames, *, chunk_size=None):
            captured_cs.append(chunk_size)
            raise ImportError("short-circuit")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)

        actions = pd.DataFrame(
            {
                "action_id": [1],
                "game_id": [1],
                "period_id": [1],
                "team_id": ["A"],
            }
        )
        frames = pd.DataFrame({"x": [1.0]})

        # add_das catches ImportError and returns NaN columns
        result = feat_mod.add_das(actions, frames, chunk_size=250)
        assert captured_cs == [250]
        assert "das_team" in result.columns

    def test_das_at_action_threads_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """das_at_action(chunk_size=N) must reach _precompute_das_lookup."""
        import silly_kicks.tracking.features as feat_mod

        captured_cs: list = []

        def spy_precompute(frames, *, chunk_size=None):
            captured_cs.append(chunk_size)
            raise ImportError("short-circuit")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)

        actions = pd.DataFrame({"action_id": [1], "team_id": ["A"]})
        frames = pd.DataFrame({"x": [1.0]})

        result = feat_mod.das_at_action(actions, frames, chunk_size=100)
        assert captured_cs == [100]
        assert result.isna().all()


class TestDasXfns:
    def test_das_xfns_are_frame_aware(self) -> None:
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        for xfn in das_xfns:
            assert is_frame_aware(xfn), f"{xfn.__name__} is not frame_aware"

    def test_das_xfns_feature_column_names(self) -> None:
        """feature_column_names introspection with empty frames must not crash."""
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.features import feature_column_names

        cols = feature_column_names(das_xfns, nb_prev_actions=3)
        expected_cols = {
            "das_team_a0",
            "das_team_a1",
            "das_team_a2",
            "das_opponent_a0",
            "das_opponent_a1",
            "das_opponent_a2",
            "das_diff_a0",
            "das_diff_a1",
            "das_diff_a2",
        }
        assert expected_cols == set(cols)

    def test_das_xfns_length(self) -> None:
        from silly_kicks.tracking.features import das_xfns

        # Single custom transformer produces all 9 columns
        assert len(das_xfns) == 1

    def test_das_at_action_introspection(self) -> None:
        """frames=None introspection must return NaN Series with correct name."""
        from silly_kicks.tracking.features import das_at_action

        dummy = pd.DataFrame({"action_id": [1, 2], "team_id": [1, 1]})
        result = das_at_action(dummy, None)
        assert result.name == "das_team"
        assert result.isna().all()
        assert len(result) == 2
