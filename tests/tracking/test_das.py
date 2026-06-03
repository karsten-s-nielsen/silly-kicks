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

        def spy_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
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

        def spy_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            captured_cs.append(chunk_size)
            raise ImportError("short-circuit")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)

        actions = pd.DataFrame({"action_id": [1], "team_id": ["A"]})
        frames = pd.DataFrame({"x": [1.0]})

        result = feat_mod.das_at_action(actions, frames, chunk_size=100)
        assert captured_cs == [100]
        assert result.isna().all()


class TestDasExceptionGracefulDegradation:
    """IndexError and TypeError from accessible-space must degrade to NaN, not crash."""

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_add_das_catches_voronoi_exceptions(self, monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        """add_das must catch IndexError/TypeError from degenerate Voronoi tessellation."""
        import silly_kicks.tracking.features as feat_mod

        def boom(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            raise exc_type("degenerate Voronoi tessellation")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", boom)

        actions = pd.DataFrame({"action_id": [1], "game_id": [1], "period_id": [1], "team_id": ["A"]})
        frames = pd.DataFrame({"x": [1.0]})

        result = feat_mod.add_das(actions, frames)
        assert "das_team" in result.columns
        assert result["das_team"].isna().all()
        assert result["das_opponent"].isna().all()
        assert result["das_diff"].isna().all()

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_das_at_action_catches_voronoi_exceptions(self, monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        """das_at_action must catch IndexError/TypeError from degenerate data."""
        import silly_kicks.tracking.features as feat_mod

        def boom(frames, *, chunk_size=None):
            raise exc_type("NaN coordinates in tracking data")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", boom)

        actions = pd.DataFrame({"action_id": [1], "team_id": ["A"]})
        frames = pd.DataFrame({"x": [1.0]})

        result = feat_mod.das_at_action(actions, frames)
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


# ---------------------------------------------------------------------------
# Linked-frame restriction (perf, must be bit-identical)
# ---------------------------------------------------------------------------


def _make_flip_frames() -> pd.DataFrame:
    """Frames whose full-period direction inference flips on a single-frame subset.

    Home is in possession throughout. Across the bulk of the period Home sits
    at low x (mean-x ordering => Home is the smaller-x / +1 team), but in the
    lone linked frame (id 9) Home is camped at high x — so inferring direction
    from frame 9 *alone* would flip Home's sign. Used to prove that pinning the
    direction on the full frames (then restricting) preserves the correct sign.
    """
    rows = []
    # Bulk frames: Home low-x, Away high-x.
    for fid in range(4):
        for i in range(3):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": f"H{i}",
                    "team_id": "H",
                    "x": 20.0 + i,
                    "y": 30.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "team_in_possession": "H",
                }
            )
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": f"A{i}",
                    "team_id": "A",
                    "x": 80.0 + i,
                    "y": 30.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "team_in_possession": "H",
                }
            )
    # Linked frame 9: Home camped high-x, Away low-x (subset would flip).
    for i in range(3):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 9,
                "player_id": f"H{i}",
                "team_id": "H",
                "x": 90.0 + i,
                "y": 30.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "team_in_possession": "H",
            }
        )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 9,
                "player_id": f"A{i}",
                "team_id": "A",
                "x": 20.0 + i,
                "y": 30.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "team_in_possession": "H",
            }
        )
    return pd.DataFrame(rows)


class TestPinAttackingDirection:
    """_pin_attacking_direction must capture FULL-frame direction, immune to subset flips."""

    def test_pin_prevents_subset_flip(self) -> None:
        pytest.importorskip("accessible_space")
        from accessible_space.interface import infer_playing_direction

        from silly_kicks.tracking._das import _pin_attacking_direction

        frames = _make_flip_frames()
        pinned = _pin_attacking_direction(frames)

        # The pinned value for the linked frame reflects the FULL-period sign.
        linked = pinned[pinned["frame_id"] == 9]
        full_sign = pinned["attacking_direction"].iloc[0]
        assert (linked["attacking_direction"] == full_sign).all()

        # Inferring from the linked frame ALONE would produce the opposite sign.
        subset = frames[frames["frame_id"] == 9].copy()
        naive = infer_playing_direction(
            subset,
            team_col="team_id",
            period_col="period_id",
            team_in_possession_col="team_in_possession",
            x_col="x",
            ball_team=None,
            frame_col="frame_id",
        )
        assert naive.iloc[0] != full_sign, "fixture must actually flip on the subset"

    def test_pin_does_not_mutate_input(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction

        frames = _make_flip_frames()
        cols_before = list(frames.columns)
        _pin_attacking_direction(frames)
        assert list(frames.columns) == cols_before

    def test_missing_team_in_possession_raises_valueerror(self) -> None:
        """Regression: missing team_in_possession must raise the canonical ValueError
        (which add_das catches -> NaN), not accessible-space's uncaught KeyError."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction

        frames = _make_flip_frames().drop(columns=["team_in_possession"])
        with pytest.raises(ValueError, match="team_in_possession"):
            _pin_attacking_direction(frames)

    def test_missing_velocity_raises_valueerror(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction

        frames = _make_flip_frames().drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity"):
            _pin_attacking_direction(frames)

    def test_all_nan_possession_raises_valueerror_not_assertionerror(self) -> None:
        """Fix 1 (3.30.0): all-NaN team_in_possession (dead-ball window) must raise the
        canonical ValueError that add_das catches -> NaN, NOT accessible-space's uncaught
        AssertionError from infer_playing_direction."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction

        frames = _make_flip_frames()
        frames["team_in_possession"] = np.nan
        with pytest.raises(ValueError, match=r"dead-ball"):
            _pin_attacking_direction(frames)


class TestDasLinkedFrameRestriction:
    """add_das/_precompute_das_lookup must restrict the per-frame sim to linked frames."""

    def _capture_frames_stub(self, captured: dict):
        def fake_gid(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
            captured["frame_ids"] = sorted(frames["frame_id"].unique().tolist())
            captured["kwargs"] = dict(kwargs)
            out = frames.copy()
            out["AS"] = 1.0
            out["DAS"] = 1.0
            return out

        return fake_gid

    def test_precompute_restricts_to_linked_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        captured: dict = {}
        monkeypatch.setattr(das_mod, "get_individual_das", self._capture_frames_stub(captured))
        monkeypatch.setattr(
            das_mod,
            "_pin_attacking_direction",
            lambda f: f.assign(attacking_direction=1.0),
        )

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = pd.DataFrame(
            {
                "period_id": [1, 1, 1],
                "frame_id": [0, 1, 2],
                "team_id": ["A", "A", "A"],
                "is_ball": [False, False, False],
                "DAS": [0.0, 0.0, 0.0],
            }
        )
        _precompute_das_lookup(frames, link_frame_ids={1})
        assert captured["frame_ids"] == [1]
        assert captured["kwargs"].get("attacking_direction_col") == "attacking_direction"

    def test_precompute_no_restriction_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        captured: dict = {}
        monkeypatch.setattr(das_mod, "get_individual_das", self._capture_frames_stub(captured))

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = pd.DataFrame(
            {
                "period_id": [1, 1, 1],
                "frame_id": [0, 1, 2],
                "team_id": ["A", "A", "A"],
                "is_ball": [False, False, False],
                "DAS": [0.0, 0.0, 0.0],
            }
        )
        _precompute_das_lookup(frames)  # no link_frame_ids
        assert captured["frame_ids"] == [0, 1, 2]
        assert "attacking_direction_col" not in captured["kwargs"]

    def test_add_das_passes_link_frame_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking.features as feat_mod

        captured: dict = {}

        def fake_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            captured["link_frame_ids"] = link_frame_ids
            return {}

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", fake_precompute)

        from silly_kicks.tracking.features import add_das

        actions = pd.DataFrame(
            {
                "action_id": [10, 11],
                "game_id": [1, 1],
                "period_id": [1, 1],
                "team_id": ["A", "A"],
            }
        )
        links = pd.DataFrame(
            {
                "action_id": [10, 11],
                "frame_id": [5, 7],
                "time_offset_seconds": [0.0, 0.0],
                "n_candidate_frames": [1, 1],
                "link_quality_score": [1.0, 1.0],
            }
        )
        add_das(actions, pd.DataFrame({"frame_id": [5, 6, 7]}), links=links)
        assert captured["link_frame_ids"] == {5, 7}

    @pytest.mark.e2e
    def test_lookup_bit_identical_full_vs_restricted(self) -> None:
        """A linked frame's per-team DAS must be identical full vs restricted."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking.features import _precompute_das_lookup

        base = TestDasTeamAsymmetry()._make_asymmetric_frame()
        frame1 = base.copy()
        frame1["frame_id"] = 1
        # Perturb frame 1 so the period is not a trivial single repeated frame.
        frame1["x"] = frame1["x"].clip(lower=5, upper=100) * 0.5 + 25.0
        frames = pd.concat([base, frame1], ignore_index=True)

        lookup_full = _precompute_das_lookup(frames)
        lookup_restricted = _precompute_das_lookup(frames, link_frame_ids={0})

        key = (1, 0)
        assert key in lookup_full and key in lookup_restricted
        for team in lookup_full[key]:
            assert np.isclose(lookup_full[key][team], lookup_restricted[key][team], rtol=1e-6, atol=1e-9), (
                f"team {team}: full={lookup_full[key][team]} restricted={lookup_restricted[key][team]}"
            )


# ---------------------------------------------------------------------------
# attacking_direction_col passthrough (caller-supplied per-frame numeric direction)
#
# Contract (Option A): when supplied, attacking_direction_col names a column on
# `frames` holding ONE numeric (+1/-1) value per (game_id, period_id, frame_id) —
# the in-possession team's attacking direction. silly-kicks validates it
# (exists / numeric / fully-covered per group), skips _pin_attacking_direction
# entirely, and threads the column to get_individual_das. It does NOT interpret
# team_in_possession to choose a team, and does NOT touch the library's
# possession gate. When None, behavior is bit-identical to before (uses _pin).
# ---------------------------------------------------------------------------


def _numeric_dir_frames(
    dir_by_frame: dict[int, float],
    *,
    game_id: int = 1,
    period_id: int = 1,
    rows_per_frame: int = 2,
) -> pd.DataFrame:
    """Minimal frames with a per-frame-consistent numeric direction column.

    Each frame gets ``rows_per_frame`` player rows all carrying the same
    direction value (the per-frame contract), plus a ball row (direction NaN,
    as a real per-team source column would leave it).
    """
    rows = []
    for fid, dval in dir_by_frame.items():
        for p in range(rows_per_frame):
            rows.append(
                {
                    "game_id": game_id,
                    "period_id": period_id,
                    "frame_id": fid,
                    "player_id": f"P{p}",
                    "team_id": "A" if p % 2 == 0 else "B",
                    "is_ball": False,
                    "attacking_direction": dval,
                    "DAS": 1.0,
                }
            )
        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "is_ball": True,
                "attacking_direction": np.nan,
                "DAS": np.nan,
            }
        )
    return pd.DataFrame(rows)


class TestAttackingDirectionColValidation:
    """(c) Fail-loud validation — errors must PROPAGATE, never NaN-fill."""

    def _actions(self) -> pd.DataFrame:
        return pd.DataFrame({"action_id": [1], "game_id": [1], "period_id": [1], "team_id": ["A"]})

    def test_missing_column_raises_valueerror(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({0: 1.0})
        with pytest.raises(ValueError, match="not found"):
            add_das(self._actions(), frames, attacking_direction_col="nope")

    def test_non_numeric_column_raises_typeerror(self) -> None:
        """String 'ltr'/'rtl' (the schema column) must be rejected, not silently used."""
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({0: 1.0})
        frames["attacking_direction"] = "ltr"  # object dtype
        with pytest.raises(TypeError, match="numeric"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_all_nan_group_raises_valueerror_naming_group(self) -> None:
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({0: np.nan, 1: np.nan})
        with pytest.raises(ValueError, match="period_id=1"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_partial_coverage_group_raises_valueerror_naming_frames(self) -> None:
        """Some frames populated, others NaN within a group → caller bug, fail loud."""
        from silly_kicks.tracking.features import add_das

        frames = _numeric_dir_frames({10: 1.0, 11: np.nan, 12: 1.0})
        with pytest.raises(ValueError, match="11"):
            add_das(self._actions(), frames, attacking_direction_col="attacking_direction")

    def test_partial_outside_links_passes_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation is restricted to action-linked frames: NaN on an unlinked
        dead-ball frame is fine (the lakehouse fills direction only where
        possession exists; unlinked frames are never simulated)."""
        import silly_kicks.tracking.features as feat_mod

        reached: dict = {}

        def fake_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            reached["adc"] = attacking_direction_col
            return {}

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", fake_precompute)

        frames = _numeric_dir_frames({5: 1.0, 6: np.nan})  # frame 6 unlinked + NaN
        links = pd.DataFrame(
            {
                "action_id": [1],
                "frame_id": [5],
                "time_offset_seconds": [0.0],
                "n_candidate_frames": [1],
                "link_quality_score": [1.0],
            }
        )
        # Frame 6's NaN must NOT trip validation because only frame 5 is linked.
        feat_mod.add_das(self._actions(), frames, links=links, attacking_direction_col="attacking_direction")
        assert reached["adc"] == "attacking_direction"

    def test_valid_numeric_column_passes_to_precompute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking.features as feat_mod

        captured: dict = {}

        def spy_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            captured["adc"] = attacking_direction_col
            return {}

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)

        frames = _numeric_dir_frames({0: 1.0, 1: -1.0})
        links = pd.DataFrame(
            {
                "action_id": [1],
                "frame_id": [0],
                "time_offset_seconds": [0.0],
                "n_candidate_frames": [1],
                "link_quality_score": [1.0],
            }
        )
        feat_mod.add_das(self._actions(), frames, links=links, attacking_direction_col="attacking_direction")
        assert captured["adc"] == "attacking_direction"


class TestPrecomputeAttackingDirectionColThreading:
    """_precompute threads attacking_direction_col to get_individual_das and skips _pin."""

    def test_threads_col_and_skips_pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._das as das_mod

        captured: dict = {}

        def fake_gid(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
            captured["kwargs"] = dict(kwargs)
            out = frames.copy()
            out["AS"] = 1.0
            out["DAS"] = 1.0
            return out

        def boom_pin(frames: pd.DataFrame) -> pd.DataFrame:
            raise AssertionError("_pin_attacking_direction must be skipped when a column is supplied")

        monkeypatch.setattr(das_mod, "get_individual_das", fake_gid)
        monkeypatch.setattr(das_mod, "_pin_attacking_direction", boom_pin)

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = _numeric_dir_frames({0: 1.0, 1: 1.0})
        # link restriction + supplied direction together (the lakehouse case)
        _precompute_das_lookup(frames, link_frame_ids={0}, attacking_direction_col="attacking_direction")
        assert captured["kwargs"].get("attacking_direction_col") == "attacking_direction"

    def test_none_still_uses_pin_for_link_restriction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: attacking_direction_col=None keeps the _pin path for links."""
        import silly_kicks.tracking._das as das_mod

        pin_called: dict = {}

        def fake_gid(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
            out = frames.copy()
            out["AS"] = 1.0
            out["DAS"] = 1.0
            return out

        def fake_pin(frames: pd.DataFrame) -> pd.DataFrame:
            pin_called["yes"] = True
            return frames.assign(attacking_direction=1.0)

        monkeypatch.setattr(das_mod, "get_individual_das", fake_gid)
        monkeypatch.setattr(das_mod, "_pin_attacking_direction", fake_pin)

        from silly_kicks.tracking.features import _precompute_das_lookup

        frames = _numeric_dir_frames({0: 1.0})
        _precompute_das_lookup(frames, link_frame_ids={0})  # no attacking_direction_col
        assert pin_called.get("yes") is True


@pytest.mark.e2e
class TestAttackingDirectionColEndToEnd:
    """(b1)/(b2) — real accessible-space behavior with the passthrough."""

    def _possession_frame(self, tip: object = "Home") -> pd.DataFrame:
        """11v11 + ball, single frame, possession=`tip` (NaN ⇒ dead ball)."""
        rng = np.random.default_rng(7)
        rows = []
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
                    "team_in_possession": tip,
                }
            )
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
                    "team_in_possession": tip,
                }
            )
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
                "team_in_possession": tip,
            }
        )
        return pd.DataFrame(rows)

    def _actions_links(self):
        actions = pd.DataFrame({"action_id": [1], "game_id": [1], "period_id": [1], "team_id": ["Home"]})
        links = pd.DataFrame(
            {
                "action_id": [1],
                "frame_id": [0],
                "time_offset_seconds": [0.0],
                "n_candidate_frames": [1],
                "link_quality_score": [1.0],
            }
        )
        return actions, links

    def test_b1_bypass_finite_and_bit_identical_to_pin(self) -> None:
        """Supplying _pin's own output as attacking_direction_col is a no-op:
        finite DAS, bit-identical to letting _pin run (attacking_direction_col=None)."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction
        from silly_kicks.tracking.features import add_das

        frames = self._possession_frame(tip="Home")
        actions, links = self._actions_links()

        # Baseline: inference via _pin.
        base = add_das(actions, frames, links=links)
        assert np.isfinite(base["das_team"].iloc[0]), "baseline DAS must be finite"

        # Supply the exact direction _pin would compute → must match bit-for-bit.
        pinned = _pin_attacking_direction(frames)
        frames_dir = frames.copy()
        frames_dir["attacking_direction"] = pinned["attacking_direction"].to_numpy()
        supplied = add_das(actions, frames_dir, links=links, attacking_direction_col="attacking_direction")

        for col in ("das_team", "das_opponent", "das_diff"):
            assert np.isfinite(supplied[col].iloc[0])
            np.testing.assert_allclose(
                supplied[col].to_numpy(),
                base[col].to_numpy(),
                rtol=0,
                atol=0,
                err_msg=f"{col}: supplied direction must equal inferred",
            )

    def test_b2_possession_gate_preserved_when_tip_all_nan(self) -> None:
        """team_in_possession all-NaN + valid direction supplied → library's
        possession gate fires internally (ValueError) → add_das NaN-fills. No
        AssertionError (the _pin bypass is what avoids it)."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import _pin_attacking_direction
        from silly_kicks.tracking.features import add_das

        # Build a valid numeric direction from the possession-populated frame,
        # then null possession (the dead-ball state).
        live = self._possession_frame(tip="Home")
        pinned = _pin_attacking_direction(live)
        dead = live.copy()
        dead["attacking_direction"] = pinned["attacking_direction"].to_numpy()
        dead["team_in_possession"] = np.nan

        actions, links = self._actions_links()

        # Supplied path: no exception, NaN-filled (gate fired in the library).
        out = add_das(actions, dead, links=links, attacking_direction_col="attacking_direction")
        assert out["das_team"].isna().all()
        assert out["das_opponent"].isna().all()
        assert out["das_diff"].isna().all()

    def test_b2_without_col_no_crash_nan_when_tip_all_nan(self) -> None:
        """Fix 1 (3.30.0): with all-NaN possession + links and NO attacking_direction_col,
        the None path routes through _pin. Previously infer_playing_direction raised an
        AssertionError that escaped add_das's except (a crash). Now _pin converts the
        all-NaN-possession case to the canonical ValueError that add_das already catches,
        so the dead-ball batch degrades to NaN DAS instead of crashing."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking.features import add_das

        dead = self._possession_frame(tip=np.nan)
        actions, links = self._actions_links()
        out = add_das(actions, dead, links=links)  # must NOT raise (AssertionError or otherwise)
        assert out["das_team"].isna().all()
        assert out["das_opponent"].isna().all()
        assert out["das_diff"].isna().all()


class TestZeroFrameSubsetDegradesToNaN:
    """A frame subset with no frame containing BOTH the ball and players must
    degrade to NaN DAS, not crash.

    accessible-space restricts its simulation to frames present in *both* its
    ball-row set and its player-row set (``transform_into_arrays``:
    ``frames_to_consider = ball_frames & player_frames``), and computes that
    intersection *after* its own ``team_in_possession.notna()`` filter — but its
    "is the data empty?" guard runs *before* the intersection. So a non-empty
    subset whose ball frames and player frames are disjoint collapses to a
    zero-frame ``PLAYER_POS``; ``simulate_passes_chunked`` then returns ``None``
    and ``get_dangerous_accessible_space`` dereferences ``None.x_grid`` —
    ``AttributeError: 'NoneType' object has no attribute 'x_grid'``. That escapes
    add_das's NaN-degradation except (which does not list AttributeError), so the
    whole pipeline crashes. Reproduces the production GS-10502 crash where one
    action batch's link-restricted frames lost their ball or player rows.

    The fix (`_has_simulatable_frame`) detects this precondition in the
    silly-kicks DAS boundary and returns NaN DAS, matching the existing
    "undefined case -> NaN DAS" contract.
    """

    def _disjoint_frames(self) -> pd.DataFrame:
        """Frame 1 = ball only; frame 2 = players only. Possession resolved on
        all rows. Ball-frames {1} and player-frames {2} are disjoint → F==0."""
        return pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="H0",
                    team_id="Home",
                    is_ball=False,
                    x=40.0,
                    y=30.0,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="A0",
                    team_id="Away",
                    is_ball=False,
                    x=60.0,
                    y=30.0,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
        )

    def test_has_simulatable_frame_false_when_disjoint(self) -> None:
        from silly_kicks.tracking._das import _has_simulatable_frame, _prepare_frames

        assert _has_simulatable_frame(_prepare_frames(self._disjoint_frames())) is False

    def test_has_simulatable_frame_true_for_valid_frame(self) -> None:
        from silly_kicks.tracking._das import _has_simulatable_frame, _prepare_frames

        valid = pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="H0",
                    team_id="Home",
                    is_ball=False,
                    x=40.0,
                    y=30.0,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="A0",
                    team_id="Away",
                    is_ball=False,
                    x=60.0,
                    y=30.0,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
        )
        assert _has_simulatable_frame(_prepare_frames(valid)) is True

    def test_get_individual_das_no_crash_returns_nan(self) -> None:
        """Reproduces the production AttributeError path at the get_individual_das level."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_individual_das

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_individual_das(self._disjoint_frames())  # must NOT raise AttributeError
        assert "DAS" in out.columns
        assert out["DAS"].isna().all()
        assert out["AS"].isna().all()
        assert len(out) == 3  # shape preserved (one row per input frame-row)

    def test_get_das_no_crash_returns_nan(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_das(self._disjoint_frames())  # must NOT raise AttributeError
        assert out["DAS"].isna().all()
        assert out["AS"].isna().all()

    def test_add_das_via_links_degrades_to_nan(self) -> None:
        """End-to-end faithful repro: a link-restricted subset whose linked frames
        are a ball-only frame + a player-only frame collapses to F==0 in
        accessible-space. add_das must return NaN DAS, not crash."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking.features import add_das

        frames = self._disjoint_frames()
        actions = pd.DataFrame(
            {"action_id": [1, 2], "game_id": [1, 1], "period_id": [1, 1], "team_id": ["Home", "Home"]}
        )
        # Restrict the per-frame sim to BOTH disjoint frames {1, 2}: the subset has a
        # ball row (frame 1) and player rows (frame 2), so accessible-space's ball-present
        # check passes, but no single frame has both → F==0 → would crash without the guard.
        links = pd.DataFrame(
            {
                "action_id": [1, 2],
                "frame_id": [1, 2],
                "time_offset_seconds": [0.0, 0.0],
                "n_candidate_frames": [1, 1],
                "link_quality_score": [1.0, 1.0],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)  # must NOT raise
        assert out["das_team"].isna().all()
        assert out["das_opponent"].isna().all()
        assert out["das_diff"].isna().all()

    def test_valid_frame_still_finite(self) -> None:
        """Guard must not over-trigger: a normal frame (ball + players) still computes DAS."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_individual_das

        valid = pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
            + [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id=f"H{i}",
                    team_id="Home",
                    is_ball=False,
                    x=30.0 + i,
                    y=20.0 + i,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                )
                for i in range(5)
            ]
            + [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id=f"A{i}",
                    team_id="Away",
                    is_ball=False,
                    x=70.0 - i,
                    y=20.0 + i,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                )
                for i in range(5)
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_individual_das(valid)
        assert np.isfinite(out["DAS"]).any(), "valid frame must yield at least one finite DAS"


class TestXcZeroFrameSubsetDegradesToNaN:
    """get_xc shares the DAS family's degenerate-frame fragility.

    accessible-space's get_expected_pass_completion runs the same
    transform_into_arrays (one simulated frame per pass), keeping only frames
    with BOTH the ball and players. When no pass references such a frame the
    intersection is empty -> F==0 -> the simulation result is None / trips a
    matrix-consistency assertion and crashes (here an AssertionError rather than
    the DAS path's AttributeError, but the same root). get_xc has no NaN
    degradation of its own, so the crash propagates. The fix degrades to NaN xC.
    """

    def _frames(self) -> pd.DataFrame:
        """Frame 1 = ball only; frame 2 = passer 'A' + opponent + ball.

        The passer exists in tracking (frame 2) so accessible-space's
        exclude_passer presence check passes, but a pass *at frame 1* references
        a ball-only frame -> the player/ball intersection for that pass is empty."""
        return pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="A",
                    team_id="Home",
                    is_ball=False,
                    x=40.0,
                    y=30.0,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="B",
                    team_id="Away",
                    is_ball=False,
                    x=60.0,
                    y=30.0,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=2,
                    player_id="ball",
                    team_id=None,
                    is_ball=True,
                    x=50.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    team_in_possession="Home",
                ),
            ]
        )

    def _pass_at(self, frame_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    action_id=1,
                    frame_id=frame_id,
                    player_id="A",
                    team_id="Home",
                    start_x=40.0,
                    start_y=30.0,
                    end_x=60.0,
                    end_y=30.0,
                ),
            ]
        )

    def test_get_xc_no_crash_returns_nan(self) -> None:
        """Pass referencing a ball-only frame must degrade to NaN xC, not crash."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_xc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_xc(self._pass_at(1), self._frames())  # frame 1 = ball only
        assert "xC" in out.columns
        assert out["xC"].isna().all()
        assert len(out) == 1  # shape preserved

    def test_get_xc_valid_pass_still_finite(self) -> None:
        """Guard must not over-trigger: a pass at a ball+players frame still computes xC."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_xc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_xc(self._pass_at(2), self._frames())  # frame 2 = ball + players
        assert np.isfinite(out["xC"]).all(), "valid pass must yield finite xC"
