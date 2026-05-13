"""Unit tests for GK identification algorithm (PR-S26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.schema import (
    GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_FRAMES_COLUMNS,
    TrackingConversionReport,
)


class TestTrackingConversionReportGkFields:
    """Tests for new GK-related fields on TrackingConversionReport."""

    def test_report_has_n_teams_gk_derived_field(self):
        report = TrackingConversionReport(
            provider="metrica",
            total_input_frames=100,
            total_output_rows=2200,
            n_periods=2,
            frame_coverage_per_period={1: 1.0, 2: 1.0},
            ball_out_seconds_per_period={1: 0.0, 2: 0.0},
            nan_rate_per_column={},
            derived_speed_rows=0,
            unrecognized_player_ids=set(),
            n_teams_gk_derived=2,
            derived_gk_picks={("game1", "teamA"): ["player1"]},
        )
        assert report.n_teams_gk_derived == 2

    def test_report_has_derived_gk_picks_field(self):
        picks = {("game1", "teamA"): ["player1"], ("game1", "teamB"): ["player2", "player3"]}
        report = TrackingConversionReport(
            provider="skillcorner",
            total_input_frames=100,
            total_output_rows=2200,
            n_periods=2,
            frame_coverage_per_period={1: 1.0, 2: 1.0},
            ball_out_seconds_per_period={1: 0.0, 2: 0.0},
            nan_rate_per_column={},
            derived_speed_rows=0,
            unrecognized_player_ids=set(),
            n_teams_gk_derived=2,
            derived_gk_picks=picks,
        )
        assert report.derived_gk_picks == picks
        assert len(report.derived_gk_picks[("game1", "teamB")]) == 2


class TestSchemaGkSourceColumn:
    """Tests for is_goalkeeper_source column in schema."""

    def test_tracking_frames_columns_has_is_goalkeeper_source(self):
        assert "is_goalkeeper_source" in TRACKING_FRAMES_COLUMNS
        assert TRACKING_FRAMES_COLUMNS["is_goalkeeper_source"] == "object"

    def test_kloppy_tracking_frames_columns_inherits_is_goalkeeper_source(self):
        assert "is_goalkeeper_source" in KLOPPY_TRACKING_FRAMES_COLUMNS
        assert KLOPPY_TRACKING_FRAMES_COLUMNS["is_goalkeeper_source"] == "object"

    def test_sportec_tracking_frames_columns_inherits_is_goalkeeper_source(self):
        assert "is_goalkeeper_source" in SPORTEC_TRACKING_FRAMES_COLUMNS

    def test_gradientsports_tracking_frames_columns_inherits_is_goalkeeper_source(self):
        assert "is_goalkeeper_source" in GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS

    def test_categorical_domains_has_is_goalkeeper_source(self):
        assert "is_goalkeeper_source" in TRACKING_CATEGORICAL_DOMAINS
        assert TRACKING_CATEGORICAL_DOMAINS["is_goalkeeper_source"] == frozenset({"native", "derived"})


class TestDeriveGoalkeepersInputValidation:
    """Tests for derive_goalkeepers input validation."""

    def test_required_columns_missing_raises(self):
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        df = pd.DataFrame({"x": [1.0], "y": [1.0]})  # missing required columns
        with pytest.raises(ValueError, match="frames missing columns"):
            derive_goalkeepers(df)

    def test_nan_game_id_raises(self):
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        df = pd.DataFrame(
            {
                "game_id": [np.nan],
                "team_id": ["team1"],
                "player_id": ["player1"],
                "x": [10.0],
                "y": [34.0],
                "is_ball": [False],
                "is_goalkeeper": [False],
            }
        )
        with pytest.raises(ValueError, match="NaN game_id/team_id"):
            derive_goalkeepers(df)

    def test_coord_range_outside_spadl_raises(self):
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # Centered coords (not SPADL 0-105)
        df = pd.DataFrame(
            {
                "game_id": ["m1"],
                "team_id": ["t1"],
                "player_id": ["p1"],
                "x": [-52.5],  # centered, not SPADL
                "y": [0.0],
                "is_ball": [False],
                "is_goalkeeper": [False],
            }
        )
        with pytest.raises(ValueError, match="coords must be SPADL"):
            derive_goalkeepers(df)

    def test_empty_frames_no_exception(self):
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        df = pd.DataFrame(
            {
                "game_id": pd.Series([], dtype="object"),
                "team_id": pd.Series([], dtype="object"),
                "player_id": pd.Series([], dtype="object"),
                "x": pd.Series([], dtype="float64"),
                "y": pd.Series([], dtype="float64"),
                "is_ball": pd.Series([], dtype="bool"),
                "is_goalkeeper": pd.Series([], dtype="bool"),
            }
        )
        frames_out, picks = derive_goalkeepers(df)
        assert len(frames_out) == 0
        assert picks == {}


class TestBPlusAlgorithm:
    """Tests for the B+ filtered algorithm core logic."""

    def _make_team_frames(
        self,
        players: list[dict],
        n_frames: int = 100,
        game_id: str = "m1",
        team_id: str = "t1",
    ) -> pd.DataFrame:
        """Helper to build synthetic frames for one team."""
        rows = []
        for frame_id in range(n_frames):
            for p in players:
                # Skip frames if player has 'skip_first_n_frames'
                if frame_id < p.get("skip_first_n_frames", 0):
                    continue
                rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team_id,
                        "player_id": p["player_id"],
                        "x": p["x"],
                        "y": p["y"],
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
        return pd.DataFrame(rows)

    def test_strict_criteria_one_gk(self):
        """Standard match: only actual GK has pa_dwell>=0.4 AND dist<20."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # GK at x=5 (dist=5, in PA), outfielders at x=50
        players = [
            {"player_id": "gk1", "x": 5.0, "y": 34.0},  # dist=5, pa=1.0
            {"player_id": "p2", "x": 50.0, "y": 34.0},  # dist=50, pa=0.0
            {"player_id": "p3", "x": 55.0, "y": 34.0},  # dist=50, pa=0.0
        ]
        frames = self._make_team_frames(players)
        frames_out, picks = derive_goalkeepers(frames)

        assert ("m1", "t1") in picks
        assert picks[("m1", "t1")] == ["gk1"]
        gk_rows = frames_out[(frames_out["player_id"] == "gk1")]
        assert gk_rows["is_goalkeeper"].all()

    def test_strict_criteria_two_gks_substitution(self):
        """Substitution: starter + sub both pass strict criteria."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # Starter GK for first 50 frames, sub GK for last 50 frames
        rows = []
        for frame_id in range(100):
            if frame_id < 50:
                # Starter GK + 10 outfielders
                rows.append(
                    {
                        "game_id": "m1",
                        "team_id": "t1",
                        "player_id": "gk_starter",
                        "x": 5.0,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            else:
                # Sub GK + 10 outfielders
                rows.append(
                    {
                        "game_id": "m1",
                        "team_id": "t1",
                        "player_id": "gk_sub",
                        "x": 5.0,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            # Outfielders present all 100 frames
            for i in range(10):
                rows.append(
                    {
                        "game_id": "m1",
                        "team_id": "t1",
                        "player_id": f"p{i}",
                        "x": 50.0 + i,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
        frames = pd.DataFrame(rows)
        _frames_out, picks = derive_goalkeepers(frames)

        assert ("m1", "t1") in picks
        # Both GKs should be flagged (multi-GK output)
        gk_picks = set(picks[("m1", "t1")])
        assert gk_picks == {"gk_starter", "gk_sub"}

    def test_sweeper_keeper_fallback(self):
        """GK plays high but with some PA time, pa_dwell<0.4 (below threshold). Fallback fires."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # Sweeper-keeper: 70% at x=20 (outside PA), 30% at x=10 (in PA)
        # This gives pa_dwell ~0.30, below the 0.40 threshold but higher than outfielders
        rows = []
        for frame_id in range(100):
            if frame_id < 30:
                # In PA (x=10)
                rows.append(
                    {
                        "game_id": "m1",
                        "team_id": "t1",
                        "player_id": "sweeper_gk",
                        "x": 10.0,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            else:
                # Outside PA (x=20)
                rows.append(
                    {
                        "game_id": "m1",
                        "team_id": "t1",
                        "player_id": "sweeper_gk",
                        "x": 20.0,
                        "y": 34.0,
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
            # Outfielders always at midfield
            rows.append(
                {
                    "game_id": "m1",
                    "team_id": "t1",
                    "player_id": "p2",
                    "x": 50.0,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
            rows.append(
                {
                    "game_id": "m1",
                    "team_id": "t1",
                    "player_id": "p3",
                    "x": 60.0,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
        frames = pd.DataFrame(rows)
        _frames_out, picks = derive_goalkeepers(frames)

        # Sweeper-keeper should be picked via fallback (lowest rank-sum)
        assert ("m1", "t1") in picks
        assert picks[("m1", "t1")] == ["sweeper_gk"]

    def test_candidate_filter_excludes_brief_substitute(self):
        """Brief sub (<30% frames) excluded from candidates."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # Real GK for all 100 frames, brief sub appears in only 20 frames near goal
        players = [
            {"player_id": "real_gk", "x": 5.0, "y": 34.0},
            {"player_id": "brief_sub", "x": 5.0, "y": 34.0, "skip_first_n_frames": 80},  # only 20 frames
            {"player_id": "outfielder", "x": 50.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players, n_frames=100)
        _frames_out, picks = derive_goalkeepers(frames)

        # Brief sub should be excluded by n_frames filter
        assert picks[("m1", "t1")] == ["real_gk"]

    def test_ball_rows_excluded_from_aggregation(self):
        """Ball rows (is_ball=True) should not affect algorithm."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        players = [
            {"player_id": "gk1", "x": 5.0, "y": 34.0},
            {"player_id": "p2", "x": 50.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players, n_frames=50)
        # Add ball rows with NaN team_id/player_id
        ball_rows = pd.DataFrame(
            {
                "game_id": ["m1"] * 50,
                "team_id": [None] * 50,
                "player_id": [None] * 50,
                "x": [52.5] * 50,
                "y": [34.0] * 50,
                "is_ball": [True] * 50,
                "is_goalkeeper": [False] * 50,
            }
        )
        frames = pd.concat([frames, ball_rows], ignore_index=True)
        _frames_out, picks = derive_goalkeepers(frames)

        assert picks[("m1", "t1")] == ["gk1"]

    def test_pa_dwell_coordinate_symmetric(self):
        """Players in x < 16.5 OR x > 88.5 both count as in-PA."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # GK at x=100 (opponent's PA), outfielders at midfield
        players = [
            {"player_id": "gk_far", "x": 100.0, "y": 34.0},  # dist=5 (from 105), in opponent PA
            {"player_id": "p2", "x": 50.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players)
        _frames_out, picks = derive_goalkeepers(frames)

        assert picks[("m1", "t1")] == ["gk_far"]

    def test_single_player_team_degenerate(self):
        """Single player on team should be picked by default."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        frames = pd.DataFrame(
            {
                "game_id": ["m1"] * 50,
                "team_id": ["t1"] * 50,
                "player_id": ["solo"] * 50,
                "x": [50.0] * 50,
                "y": [34.0] * 50,
                "is_ball": [False] * 50,
                "is_goalkeeper": [False] * 50,
            }
        )
        _frames_out, picks = derive_goalkeepers(frames)

        assert picks[("m1", "t1")] == ["solo"]


class TestBPlusScoreFunction:
    """Tests for B+ rank-sum scoring mechanics."""

    def test_b_plus_score_function(self):
        """Hand-crafted feature vectors: only GK passes strict criteria."""
        from silly_kicks.tracking._gk_identification import derive_goalkeepers

        # Player A: in PA (x=5), dist=5 < 20, pa=1.0 >= 0.4 -> passes strict
        # Player B: outside PA (x=20), dist=20, pa=0.0 -> fails strict
        # Player C: midfield (x=40), dist=40, pa=0.0 -> fails strict
        rows = []
        for _i in range(100):
            # Player A: in PA (x=5)
            rows.append(
                {
                    "game_id": "m1",
                    "team_id": "t1",
                    "player_id": "A",
                    "x": 5.0,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
            # Player B: outside PA (x=20, which is >16.5)
            rows.append(
                {
                    "game_id": "m1",
                    "team_id": "t1",
                    "player_id": "B",
                    "x": 20.0,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
            # Player C: midfield (x=40)
            rows.append(
                {
                    "game_id": "m1",
                    "team_id": "t1",
                    "player_id": "C",
                    "x": 40.0,
                    "y": 34.0,
                    "is_ball": False,
                    "is_goalkeeper": False,
                }
            )
        frames = pd.DataFrame(rows)
        _frames_out, picks = derive_goalkeepers(frames)

        # Player A should be picked (via strict criteria, pa>=0.4 and dist<20)
        assert picks[("m1", "t1")] == ["A"]


class TestNativePathSource:
    """Tests for is_goalkeeper_source on native paths (Sportec/Gradient Sports)."""

    def test_gradientsports_schema_includes_is_goalkeeper_source(self):
        """Gradient Sports schema includes is_goalkeeper_source column."""
        assert "is_goalkeeper_source" in GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS

    def test_sportec_schema_includes_is_goalkeeper_source(self):
        """Sportec schema includes is_goalkeeper_source column."""
        assert "is_goalkeeper_source" in SPORTEC_TRACKING_FRAMES_COLUMNS
