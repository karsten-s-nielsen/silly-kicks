"""Unit tests for DAS adapter."""

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das import (
    _X_OFFSET,
    _Y_OFFSET,
    DasUnscoreableError,
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
        np.testing.assert_allclose(result["x"].values, [-_X_OFFSET, _X_OFFSET, 0.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(result["y"].values, [-_Y_OFFSET, _Y_OFFSET, 0.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(result["vx"].values, [1.0, 2.0, 0.0])  # type: ignore[arg-type]
        np.testing.assert_allclose(result["vy"].values, [0.5, -0.5, 0.0])  # type: ignore[arg-type]

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

    # -- ADR-043: the velocity-structurally-unavailable marker -------------------
    #
    # Both directions matter. The marker exists ONLY to separate "this source can
    # never have velocity" from "the caller forgot derive_velocities()"; a test that
    # only proved the degrade would let the marker silently re-absorb the caller bug.

    def test_marked_frames_missing_velocity_raise_degradable_error(self) -> None:
        from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE
        from silly_kicks.tracking._das import DAS_SOURCE_UNSCOREABLE_FRAME, _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vx", "vy"])
        df["speed_source"] = SPEED_SOURCE_UNAVAILABLE
        with pytest.raises(DasUnscoreableError) as excinfo:
            _validate_das_inputs(df)
        assert excinfo.value.das_source == DAS_SOURCE_UNSCOREABLE_FRAME

    def test_unmarked_frames_missing_velocity_raise_plain_value_error(self) -> None:
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["vx", "vy"])
        df["speed_source"] = None
        with pytest.raises(ValueError, match="velocity columns") as excinfo:
            _validate_das_inputs(df)
        assert not isinstance(excinfo.value, DasUnscoreableError)

    def test_partially_marked_frames_raise_plain_value_error(self) -> None:
        """ALL rows must be marked: an unmarked row is a genuine velocity-bearing source."""
        from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE
        from silly_kicks.tracking._das import _validate_das_inputs

        df = pd.concat([self._minimal_frames(), self._minimal_frames()], ignore_index=True)
        df = df.drop(columns=["vx", "vy"])
        df["speed_source"] = [SPEED_SOURCE_UNAVAILABLE, None]
        with pytest.raises(ValueError, match="velocity columns") as excinfo:
            _validate_das_inputs(df)
        assert not isinstance(excinfo.value, DasUnscoreableError)

    def test_marker_does_not_excuse_missing_team_in_possession(self) -> None:
        """The marker speaks for VELOCITY only; every other contract column still fails loud."""
        from silly_kicks.tracking import SPEED_SOURCE_UNAVAILABLE
        from silly_kicks.tracking._das import _validate_das_inputs

        df = self._minimal_frames().drop(columns=["team_in_possession"])
        df["speed_source"] = SPEED_SOURCE_UNAVAILABLE
        with pytest.raises(ValueError, match="team_in_possession") as excinfo:
            _validate_das_inputs(df)
        assert not isinstance(excinfo.value, DasUnscoreableError)

    def test_das_source_token_is_validated(self) -> None:
        """The carried provenance must stay inside the closed vocabulary."""
        with pytest.raises(ValueError, match="das_source must be one of"):
            DasUnscoreableError("boom", das_source="not_a_token")


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
            # ADR-043: DasUnscoreableError is the ONLY exception add_das degrades on, so it
            # is the only short-circuit that still exercises the NaN-column path.
            raise DasUnscoreableError("short-circuit")

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

        result = feat_mod.add_das(actions, frames, chunk_size=250)
        assert captured_cs == [250]
        assert "das_team" in result.columns

    def test_das_at_action_threads_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """das_at_action(chunk_size=N) must reach _precompute_das_lookup."""
        import silly_kicks.tracking.features as feat_mod

        captured_cs: list = []

        def spy_precompute(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            captured_cs.append(chunk_size)
            raise DasUnscoreableError("short-circuit")  # ADR-043: the only degrading exception

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", spy_precompute)

        actions = pd.DataFrame({"action_id": [1], "team_id": ["A"]})
        frames = pd.DataFrame({"x": [1.0]})

        result = feat_mod.das_at_action(actions, frames, chunk_size=100)
        assert captured_cs == [100]
        assert result.isna().all()


class TestDasExceptionGracefulDegradation:
    """PR-S60's contract, restated under ADR-043's narrowed catch.

    IndexError (degenerate Voronoi: collinear players / <3 per team) and TypeError (NaN
    tracking coordinates) raised *from inside accessible-space* must still degrade to NaN
    rather than crash the pipeline. What CHANGED in ADR-043 is where the decision is made:
    the conversion to ``DasUnscoreableError`` now happens at the ``_das`` library seam, so
    the callers catch that one type instead of a broad tuple. The same exception types
    raised from silly-kicks' OWN code are real defects and now PROPAGATE -- pinned by
    ``TestDasUnscoreableTaxonomy::test_non_unscoreable_exceptions_propagate``, which is the
    counterpart these tests used to (wrongly) cover.
    """

    @staticmethod
    def _patch_library(monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        import silly_kicks.tracking._das as das_mod

        def behaviour(frames):
            raise exc_type("degenerate Voronoi tessellation / NaN coordinates")

        monkeypatch.setattr(das_mod, "_import_accessible_space", lambda: _FakeAsModule(behaviour))

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_add_das_catches_voronoi_exceptions(self, monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        """add_das must degrade on IndexError/TypeError from a degenerate tessellation."""
        from silly_kicks.tracking.features import add_das

        self._patch_library(monkeypatch, exc_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = add_das(_das_actions([(1, "Home")]), _live_frames((1,)))
        assert result["das_team"].isna().all()
        assert result["das_opponent"].isna().all()
        assert result["das_diff"].isna().all()
        # ...and the degrade is now NAMED, not just an all-NaN column (ADR-043).
        assert (result["das_source"] == "unscoreable_call").all()

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_das_at_action_catches_voronoi_exceptions(self, monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        """das_at_action must degrade on IndexError/TypeError from degenerate data."""
        from silly_kicks.tracking.features import das_at_action

        self._patch_library(monkeypatch, exc_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = das_at_action(_das_actions([(1, "Home")]), _live_frames((1,)))
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

        cols = feature_column_names(das_xfns, nb_prev_actions=3)  # type: ignore[arg-type]
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

    def test_get_xc_pyarrow_string_team_columns(self) -> None:
        """Regression: get_xc must coerce pyarrow-backed StringDtype team columns to
        numpy object before calling accessible-space.

        accessible-space's offside path 2-D-indexes the team arrays
        (``passer_teams[:, np.newaxis]``); a pyarrow StringDtype column rejects 2-D
        indexing with ``IndexError: too many indices for array``. This is the default
        string dtype on newer pandas / py3.11+, so it bit only the CI 3.11/3.12 legs
        (3.10 infers object strings). Constructing the dtype explicitly makes the lock
        deterministic regardless of the ambient pandas string-inference default."""
        pytest.importorskip("accessible_space")
        pytest.importorskip("pyarrow")
        from silly_kicks.tracking._das import get_xc

        frames = self._frames().astype({"team_id": "string[pyarrow]", "team_in_possession": "string[pyarrow]"})
        passes = self._pass_at(2).astype({"team_id": "string[pyarrow]"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_xc(passes, frames)  # must NOT raise IndexError
        assert np.isfinite(out["xC"]).all(), "valid pass must yield finite xC even with pyarrow team dtype"


class TestDasOutputAlignmentGuard:
    """The output guard must fail loud on MISALIGNMENT, not on a legitimate shrink.

    ``result["AS"] = ret.acc_space`` aligns by index LABEL, and accessible-space
    restores the caller's own labels before returning, so a shorter return is
    correct-by-construction: rows whose ``team_in_possession`` is NaN are dropped
    (most rows on real provider data) and honestly become NaN. A length comparison
    therefore rejects every real match. The real hazard is a return carrying labels
    the input never had -- a positional reset, a shifted index, a foreign frame set --
    which aligns values onto the wrong rows or into nothing. These tests pin the
    subset contract in BOTH directions.
    """

    def _frames(self, n: int = 4) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [7] * n,
                "period_id": [1] * n,
                "frame_id": list(range(n)),
                "player_id": ["P1"] * n,
                "team_id": ["A"] * n,
                "source_provider": ["gradientsports"] * n,
                "x": [50.0] * n,
                "y": [34.0] * n,
                "vx": [0.0] * n,
                "vy": [0.0] * n,
                "is_ball": [False] * n,
                "team_in_possession": ["A"] * n,
            }
        )

    def test_full_length_matching_index_is_silent(self) -> None:
        """Guard must not over-trigger: the exact input index passes through untouched."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        values = pd.Series([1.0, 2.0, 3.0, 4.0], index=prepared.index)
        # Must neither raise nor warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_legitimate_shrink_is_accepted(self) -> None:
        """THE regression this guard exists to permit.

        accessible-space drops NaN-``team_in_possession`` rows and restores the
        caller's labels for the rest. That is a strict index SUBSET and must pass;
        a length comparison here rejected every real provider match.
        """
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        kept = prepared.index[[0, 2]]
        values = pd.Series([1.0, 3.0], index=kept)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_shrink_on_a_non_default_index_is_accepted(self) -> None:
        """The subset rule must key on LABELS, not on positions."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4).set_axis([100, 101, 102, 103])
        values = pd.Series([1.0, 3.0], index=pd.Index([101, 103]))
        _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_foreign_index_raises(self) -> None:
        """Labels the input never had would align onto nothing (or the wrong row)."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        values = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.Index([900, 901, 902, 903]))
        with pytest.raises(ValueError, match="absent from the input index"):
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_positional_reset_on_non_default_index_raises(self) -> None:
        """A positional ``reset_index`` is the concrete misalignment mode we fear."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4).set_axis([100, 101, 102, 103])
        values = pd.Series([1.0, 2.0, 3.0, 4.0])  # RangeIndex(0..3) -- reset, not restored
        with pytest.raises(ValueError, match="absent from the input index"):
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_shifted_index_raises(self) -> None:
        """A partially-overlapping (shifted) index still carries foreign labels."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        values = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.Index([2, 3, 4, 5]))
        with pytest.raises(ValueError, match="absent from the input index"):
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_indexless_values_are_held_to_exact_length(self) -> None:
        """No index -> pandas assigns POSITIONALLY, so only an exact length is safe."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        _check_das_output_alignment(np.arange(4.0), prepared, field="AS", entry_point="get_das")
        with pytest.raises(ValueError, match="no index"):
            _check_das_output_alignment(np.arange(2.0), prepared, field="AS", entry_point="get_das")

    def test_message_is_actionable(self) -> None:
        """Counts, the offending labels, the frame context, and a remedy must appear."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4)
        values = pd.Series([1.0, 2.0], index=pd.Index([900, 901]))
        with pytest.raises(ValueError) as exc:
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_individual_das")
        msg = str(exc.value)
        assert "2" in msg and "4" in msg, f"both counts must appear: {msg}"
        assert "900" in msg, f"an offending label must appear: {msg}"
        assert "get_individual_das" in msg, f"entry point must appear: {msg}"
        assert "gradientsports" in msg, f"provider context must appear: {msg}"
        assert "game_id=7" in msg, f"game context must appear: {msg}"
        assert "team_in_possession" in msg, f"message must explain the legitimate shrink: {msg}"
        assert "accessible-space version" in msg, f"remedy must appear: {msg}"

    def test_context_degrades_without_optional_columns(self) -> None:
        """_frame_context must never raise when the optional context columns are absent."""
        from silly_kicks.tracking._das import _check_das_output_alignment

        prepared = self._frames(4).drop(columns=["source_provider", "game_id", "period_id"])
        values = pd.Series([1.0], index=pd.Index([900]))
        with pytest.raises(ValueError, match="no source_provider/game_id/period_id"):
            _check_das_output_alignment(values, prepared, field="AS", entry_point="get_das")

    def test_get_das_tolerates_dropped_nan_possession_rows(self) -> None:
        """Integration: a NaN-possession shrink must NOT raise, and must land as NaN.

        This is the real-provider shape (gradientsports is ~67% dead-ball). The
        surviving frames must still carry finite DAS -- proving the values landed on
        the right rows rather than the whole column degrading.
        """
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        rows = []
        for fid in range(4):
            poss = None if fid in (1, 3) else "Home"
            for pid, tid, x in [(10, "Home", 30.0), (20, "Away", 70.0)]:
                rows.append(
                    dict(
                        game_id=7,
                        period_id=1,
                        frame_id=fid,
                        player_id=pid,
                        team_id=tid,
                        source_provider="gradientsports",
                        x=x,
                        y=34.0,
                        vx=0.0,
                        vy=0.0,
                        is_ball=False,
                        team_in_possession=poss,
                        ball_carrier_player_id=10,
                    )
                )
            rows.append(
                dict(
                    game_id=7,
                    period_id=1,
                    frame_id=fid,
                    player_id="ball",
                    team_id=None,
                    source_provider="gradientsports",
                    x=30.0,
                    y=34.0,
                    vx=0.0,
                    vy=0.0,
                    is_ball=True,
                    team_in_possession=poss,
                    ball_carrier_player_id=10,
                )
            )
        frames = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_das(frames, player_in_possession_col="ball_carrier_player_id")  # must NOT raise

        assert len(out) == len(frames), "every input row must survive"
        scored = out["frame_id"].isin([0, 2])
        assert out.loc[scored, "DAS"].notna().all(), "possession frames must carry a finite DAS"
        assert out.loc[~scored, "DAS"].isna().all(), (
            "NaN-possession frames must land as NaN, not as another row's value"
        )


# ---------------------------------------------------------------------------
# ADR-043 -- DAS unscoreable taxonomy (narrowed catch) + das_source provenance
# ---------------------------------------------------------------------------


def _live_frames(frame_ids: tuple[int, ...] = (1,)) -> pd.DataFrame:
    """Ball + 5v5 per frame -- a frame set on which DAS genuinely computes."""
    rows: list[dict] = []
    for fid in frame_ids:
        t = float(fid)
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                player_id="ball",
                team_id=None,
                is_ball=True,
                x=50.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                team_in_possession="Home",
            )
        )
        for i in range(5):
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    time_seconds=t,
                    player_id=f"H{i}",
                    team_id="Home",
                    is_ball=False,
                    x=30.0 + i,
                    y=20.0 + i,
                    vx=1.0,
                    vy=0.0,
                    team_in_possession="Home",
                )
            )
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    time_seconds=t,
                    player_id=f"A{i}",
                    team_id="Away",
                    is_ball=False,
                    x=70.0 - i,
                    y=20.0 + i,
                    vx=-1.0,
                    vy=0.0,
                    team_in_possession="Home",
                )
            )
    return pd.DataFrame(rows)


def _links(pairs: list[tuple[int, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [a for a, _ in pairs],
            "frame_id": [f for _, f in pairs],
            "time_offset_seconds": [0.0] * len(pairs),
            "n_candidate_frames": [1] * len(pairs),
            "link_quality_score": [1.0] * len(pairs),
        }
    )


def _das_actions(rows: list[tuple[int, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [a for a, _ in rows],
            "game_id": [1] * len(rows),
            "period_id": [1] * len(rows),
            "time_seconds": [1.0] * len(rows),
            "team_id": [t for _, t in rows],
        }
    )


class _FakeAsRet:
    def __init__(self, acc, das) -> None:
        self.player_acc_space = acc
        self.player_das = das
        self.acc_space = acc
        self.das = das


class _FakeAsModule:
    """Stand-in for accessible-space whose simulation entry point is controllable."""

    def __init__(self, behaviour) -> None:
        self._behaviour = behaviour

    def get_individual_dangerous_accessible_space(self, frames, **kwargs):
        return self._behaviour(frames)

    def get_dangerous_accessible_space(self, frames, **kwargs):
        return self._behaviour(frames)


class _NarrowSignatureAsModule:
    """accessible-space whose API has drifted: no **kwargs, so call binding fails."""

    def get_individual_dangerous_accessible_space(self, frames):
        raise AssertionError("must never be reached -- binding fails first")

    def get_dangerous_accessible_space(self, frames):
        raise AssertionError("must never be reached -- binding fails first")


class TestDasSourceProvenance:
    """das_source makes "DAS could not be computed" distinguishable from "NaN here"."""

    def test_vocabulary_is_closed_and_add_das_only_emits_it(self) -> None:
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking import DAS_SOURCE_VALUES, add_das

        frames = _live_frames((1,))
        # action 1: links to the live frame, real team  -> computed
        # action 2: no pointer at all                   -> unlinked
        # action 3: pointer to a frame that has no DAS  -> unscoreable_frame
        # action 4: links to the live frame, alien team -> team_unresolved
        actions = _das_actions([(1, "Home"), (2, "Home"), (3, "Home"), (4, "Nowhere")])
        links = _links([(1, 1), (3, 99), (4, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)

        assert set(out["das_source"]) <= set(DAS_SOURCE_VALUES)
        by_action = dict(zip(out["action_id"], out["das_source"], strict=True))
        # Non-vacuity: every branch of the classifier is exercised by this one call.
        assert by_action == {1: "computed", 2: "unlinked", 3: "unscoreable_frame", 4: "team_unresolved"}
        # ...and the provenance agrees with the values it describes.
        vals = dict(zip(out["action_id"], out["das_team"], strict=True))
        assert np.isfinite(vals[1]), "the 'computed' row must actually carry a finite DAS"
        assert all(pd.isna(vals[a]) for a in (2, 3, 4))

    def test_unscoreable_call_on_a_genuine_dead_ball_window(self) -> None:
        """A real dead-ball batch (all-NaN possession) degrades with the CALL-scoped token."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking import add_das

        frames = _live_frames((1,))
        frames["team_in_possession"] = np.nan
        actions = _das_actions([(1, "Home")])
        links = _links([(1, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)

        assert out["das_team"].isna().all(), "non-vacuity: the degrade path must really have run"
        assert (out["das_source"] == "unscoreable_call").all()

    def test_provenance_survives_the_no_simulatable_frame_guard(self) -> None:
        """The ball/player-disjoint subset yields NaN without raising -- it must NOT
        masquerade as "unlinked" (the action DID link; the frame is unscoreable)."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking import add_das

        def _row(fid, pid, team, is_ball, x, y, vx, vy):
            return dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=float(fid),
                player_id=pid,
                team_id=team,
                is_ball=is_ball,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                team_in_possession="Home",
            )

        frames = pd.DataFrame(
            [
                _row(1, "ball", None, True, 50.0, 34.0, 0.0, 0.0),
                _row(2, "H0", "Home", False, 40.0, 30.0, 1.0, 0.0),
                _row(2, "A0", "Away", False, 60.0, 30.0, -1.0, 0.0),
            ]
        )
        actions = _das_actions([(1, "Home"), (2, "Home")])
        links = _links([(1, 1), (2, 2)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)
        assert out["das_team"].isna().all()
        assert (out["das_source"] == "unscoreable_frame").all()

    def test_das_xfns_do_not_leak_the_string_provenance_column(self) -> None:
        """VAEP feature matrices stay numeric: das_source is an aggregator column only."""
        from silly_kicks.tracking.features import das_xfns
        from silly_kicks.vaep.features import feature_column_names

        cols = feature_column_names(das_xfns, nb_prev_actions=3)  # type: ignore[arg-type]
        assert not any("das_source" in c for c in cols)


class TestForwardedCarrierColumnSurvivesTwoDIndexing:
    """The forwarded ball-carrier column must be cast for accessible-space's 2-D indexing.

    With ``respect_offside`` on (the DAS default) accessible-space receives the carrier
    column as PASSERS and indexes it two-dimensionally (``passers[:, None]``) to build the
    offside mask -- exactly like the team arrays that ``_prepare_frames`` already casts. A
    pandas ``StringDtype`` / pyarrow-backed column rejects that with "IndexError: too many
    indices for array"; ``_call_simulation`` converts it to :class:`DasUnscoreableError`,
    and DAS degrades to **silently all-NaN**.

    That is the failure mode these tests exist for: it is SILENT. Before the cast, every
    DAS call on pandas 3 (which infers ``StringDtype`` for such a column by default) scored
    ZERO frames, and nothing in the suite noticed -- the other DAS fixtures carry no carrier
    column at all, so their ``player_in_possession_col`` resolves to ``None`` and the
    offside path never runs.

    The dtype is pinned EXPLICITLY rather than left to pandas inference so these guards
    fail deterministically on every interpreter: on pandas 2 the inferred dtype is already
    ``object`` and the defect is invisible, which is precisely how it shipped.
    """

    CARRIER = "ball_carrier_player_id"

    @classmethod
    def _frames_with_string_carrier(cls, col: str = "") -> pd.DataFrame:
        """``_live_frames`` plus a StringDtype carrier column naming a real Home player."""
        frames = _live_frames((1,))
        frames[col or cls.CARRIER] = pd.Series(["H0"] * len(frames), dtype="string", index=frames.index)
        return frames

    def test_string_dtype_carrier_still_computes_a_finite_das(self) -> None:
        """The regression guard: a StringDtype carrier must not collapse DAS to NaN."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking import add_das

        frames = self._frames_with_string_carrier()
        # Non-vacuity: the fixture must really present the dtype that triggers the bug.
        # Were this to silently arrive as object, the guard would pass without testing.
        assert isinstance(frames[self.CARRIER].dtype, pd.StringDtype), (
            "fixture must carry a StringDtype carrier column, else the 2-D indexing path is untested"
        )

        actions = _das_actions([(1, "Home")])
        links = _links([(1, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(actions, frames, links=links)

        assert (out["das_source"] == "computed").all(), (
            f"DAS must genuinely compute, not degrade: das_source={sorted(set(out['das_source']))}"
        )
        assert np.isfinite(out["das_team"]).all(), "das_team must be finite, not a silent NaN degrade"
        assert (out["das_team"] > 0).all(), "a real accessible-space area is strictly positive"

    def test_the_cast_follows_the_caller_supplied_column_name(self) -> None:
        """The cast must name the column the CALLER passed, not the default literal.

        ``player_in_possession_col`` is public and caller-configurable, so hard-coding
        ``ball_carrier_player_id`` in the cast set would leave every caller who renames it
        on the original silently-all-NaN path.
        """
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking._das import get_das

        custom = "my_carrier"
        frames = self._frames_with_string_carrier(custom)
        assert self.CARRIER not in frames.columns, "the default name must be absent for this to discriminate"
        assert isinstance(frames[custom].dtype, pd.StringDtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = get_das(frames, player_in_possession_col=custom)

        assert out["DAS"].notna().any(), "a caller-renamed carrier column must be cast like the default one"

    def test_opting_out_of_the_carrier_leaves_the_column_uncast(self) -> None:
        """``player_in_possession_col=None`` forwards no carrier, so no cast is owed.

        Pins the cast to the columns accessible-space actually 2-D-indexes rather than
        letting it widen into "cast anything that looks like a carrier".
        """
        from silly_kicks.tracking._das import _prepare_frames

        frames = self._frames_with_string_carrier()
        prepared = _prepare_frames(frames, player_in_possession_col=None)
        assert isinstance(prepared[self.CARRIER].dtype, pd.StringDtype), (
            "an unforwarded carrier column must be left exactly as the caller supplied it"
        )
        # ...while the always-forwarded columns are cast regardless.
        assert prepared["team_id"].dtype == object


class TestDasUnscoreableTaxonomy:
    """Only the three documented unscoreable conditions degrade; everything else propagates."""

    def test_dasunscoreableerror_subclasses_valueerror(self) -> None:
        from silly_kicks.tracking import DasUnscoreableError

        assert issubclass(DasUnscoreableError, ValueError)

    def test_unexpected_valueerror_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plain ValueError is NOT the dead-ball degrade -- it must reach the caller."""
        import silly_kicks.tracking.features as feat_mod

        def boom(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            raise ValueError("a real defect, not an unscoreable window")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", boom)
        with pytest.raises(ValueError, match="a real defect"):
            feat_mod.add_das(_das_actions([(1, "Home")]), _live_frames((1,)))

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError, RuntimeError, ImportError])
    def test_non_unscoreable_exceptions_propagate(self, monkeypatch: pytest.MonkeyPatch, exc_type: type) -> None:
        """Raised from silly-kicks OWN code (not the accessible-space call), these are bugs."""
        import silly_kicks.tracking.features as feat_mod

        def boom(frames, *, chunk_size=None, link_frame_ids=None, attacking_direction_col=None):
            raise exc_type("from silly-kicks own code")

        monkeypatch.setattr(feat_mod, "_precompute_das_lookup", boom)
        with pytest.raises(exc_type):
            feat_mod.add_das(_das_actions([(1, "Home")]), _live_frames((1,)))

    def test_missing_velocity_columns_propagate(self) -> None:
        """A caller-contract violation must fail loud, not become an all-NaN column."""
        pytest.importorskip("accessible_space")
        from silly_kicks.tracking import add_das

        frames = _live_frames((1,)).drop(columns=["vx"])
        with pytest.raises(ValueError, match="velocity columns"):
            add_das(_das_actions([(1, "Home")]), frames)

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_library_degenerate_geometry_becomes_unscoreable(
        self, monkeypatch: pytest.MonkeyPatch, exc_type: type
    ) -> None:
        """Degenerate Voronoi (IndexError) / NaN coordinates (TypeError) raised INSIDE
        accessible-space are the widened-for conditions: converted, degraded, tagged."""
        import silly_kicks.tracking._das as das_mod
        from silly_kicks.tracking import add_das

        def behaviour(frames):
            raise exc_type("degenerate tessellation / NaN coordinates")

        monkeypatch.setattr(das_mod, "_import_accessible_space", lambda: _FakeAsModule(behaviour))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = add_das(_das_actions([(1, "Home")]), _live_frames((1,)))
        assert out["das_team"].isna().all()
        assert (out["das_source"] == "unscoreable_call").all()

    @pytest.mark.parametrize("exc_type", [IndexError, TypeError])
    def test_library_degenerate_geometry_raises_the_taxonomy_error(
        self, monkeypatch: pytest.MonkeyPatch, exc_type: type
    ) -> None:
        """At the _das seam the conversion is explicit and chains the original cause."""
        import silly_kicks.tracking._das as das_mod

        def behaviour(frames):
            raise exc_type("degenerate tessellation / NaN coordinates")

        monkeypatch.setattr(das_mod, "_import_accessible_space", lambda: _FakeAsModule(behaviour))
        with pytest.raises(das_mod.DasUnscoreableError) as exc_info:
            das_mod.get_individual_das(_live_frames((1,)))
        assert isinstance(exc_info.value.__cause__, exc_type)

    def test_accessible_space_signature_drift_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An API change raises TypeError at CALL BINDING; that must never be mistaken for
        the NaN-coordinate TypeError raised from inside the function body."""
        import silly_kicks.tracking._das as das_mod

        monkeypatch.setattr(das_mod, "_import_accessible_space", lambda: _NarrowSignatureAsModule())
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            das_mod.get_individual_das(_live_frames((1,)))

    def test_output_alignment_breach_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The integrity guard rewritten this release must not be swallowed by add_das."""
        import silly_kicks.tracking._das as das_mod
        from silly_kicks.tracking import add_das

        def behaviour(frames):
            foreign = pd.Series(np.zeros(len(frames)), index=range(10_000, 10_000 + len(frames)))
            return _FakeAsRet(foreign, foreign)

        monkeypatch.setattr(das_mod, "_import_accessible_space", lambda: _FakeAsModule(behaviour))
        with pytest.raises(ValueError, match="absent from the input index"):
            add_das(_das_actions([(1, "Home")]), _live_frames((1,)))
