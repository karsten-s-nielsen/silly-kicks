"""Tests for silly_kicks.tracking._team_shape.compute_team_shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_team_frames(
    *,
    team_id=1,
    outfield_positions: list[tuple[float, float]],
    gk_pos: tuple[float, float] = (3.0, 34.0),
    frame_id: int = 1,
    period_id: int = 1,
    game_id: int = 1,
    time_seconds: float = 1.0,
) -> pd.DataFrame:
    """Build a single-frame fixture for one team with known positions."""
    rows = []
    pid = 100
    # Ball
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # GK
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=pid,
            team_id=team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=gk_pos[0],
            y=gk_pos[1],
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    pid += 1
    # Outfield
    for x, y in outfield_positions:
        rows.append(
            dict(
                game_id=game_id,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=time_seconds,
                frame_rate=25.0,
                player_id=pid,
                team_id=team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=x,
                y=y,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
        pid += 1
    return pd.DataFrame(rows)


class TestComputeTeamShape:
    def test_known_square_geometry(self):
        """4 players at (0,0), (10,0), (10,10), (0,10) -> known metrics."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(
            outfield_positions=[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ]
        )
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_outfield_players"] == 4
        assert row["centroid_x"] == pytest.approx(5.0)
        assert row["centroid_y"] == pytest.approx(5.0)
        assert row["convex_hull_area"] == pytest.approx(100.0)
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(10.0)
        # stretch_index = mean distance from (5,5) to each corner = sqrt(50) = 7.071...
        assert row["stretch_index"] == pytest.approx(np.sqrt(50.0))

    def test_triangle_geometry(self):
        """3 players in right triangle -> hull area = 0.5 * base * height."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(
            outfield_positions=[
                (0.0, 0.0),
                (10.0, 0.0),
                (0.0, 6.0),
            ]
        )
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert row["convex_hull_area"] == pytest.approx(30.0)  # 0.5 * 10 * 6
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(6.0)

    def test_zero_players_all_nan(self):
        """No outfield players for the team -> all metrics NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        # Only build a GK, no outfield
        frames = _make_team_frames(outfield_positions=[])
        result = compute_team_shape(frames, team_id=1)

        # Should return 0 rows (no outfield players -> no entry for this frame)
        assert len(result) == 0

    def test_one_player_degenerate(self):
        """1 player -> centroid=position, length/width/stretch=0, hull=NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[(20.0, 30.0)])
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_outfield_players"] == 1
        assert row["centroid_x"] == pytest.approx(20.0)
        assert row["centroid_y"] == pytest.approx(30.0)
        assert row["team_length"] == pytest.approx(0.0)
        assert row["team_width"] == pytest.approx(0.0)
        assert row["stretch_index"] == pytest.approx(0.0)
        assert pd.isna(row["convex_hull_area"])

    def test_two_players_hull_nan(self):
        """2 players -> hull NaN, rest valid."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(outfield_positions=[(10.0, 34.0), (20.0, 34.0)])
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 2
        assert pd.isna(row["convex_hull_area"])
        assert row["team_length"] == pytest.approx(10.0)
        assert row["team_width"] == pytest.approx(0.0)

    def test_collinear_players_hull_zero(self):
        """3+ collinear players -> QhullError caught, hull=0.0."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = _make_team_frames(
            outfield_positions=[
                (10.0, 34.0),
                (20.0, 34.0),
                (30.0, 34.0),
            ]
        )
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert row["convex_hull_area"] == pytest.approx(0.0)  # degenerate
        assert row["team_length"] == pytest.approx(20.0)
        assert row["team_width"] == pytest.approx(0.0)

    def test_filters_goalkeeper(self):
        """GK is NOT included in outfield metrics."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        # GK at (3, 34) should not affect centroid of outfield at (50, 34)
        frames = _make_team_frames(
            outfield_positions=[(50.0, 34.0), (60.0, 34.0), (70.0, 34.0)],
            gk_pos=(3.0, 34.0),
        )
        result = compute_team_shape(frames, team_id=1)

        row = result.iloc[0]
        assert row["n_outfield_players"] == 3
        assert row["centroid_x"] == pytest.approx(60.0)  # mean(50, 60, 70)

    def test_empty_frames(self):
        """Empty frames -> empty result."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "team_id",
                "player_id",
                "is_ball",
                "is_goalkeeper",
                "x",
                "y",
            ]
        )
        result = compute_team_shape(frames, team_id=1)
        assert len(result) == 0

    def test_multi_frame_batch(self):
        """Multiple frames produce one row per frame."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        f1 = _make_team_frames(
            outfield_positions=[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)],
            frame_id=1,
            time_seconds=0.0,
        )
        f2 = _make_team_frames(
            outfield_positions=[(30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0)],
            frame_id=2,
            time_seconds=0.04,
        )
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_team_shape(frames, team_id=1)

        assert len(result) == 2
        assert result["frame_id"].tolist() == [1, 2]
        # Frame 1 centroid at (15, 15); Frame 2 at (35, 35)
        assert result.iloc[0]["centroid_x"] == pytest.approx(15.0)
        assert result.iloc[1]["centroid_x"] == pytest.approx(35.0)


class TestWardInterLineGaps:
    """TF-44: Ward clustering defensive_line_height + inter-line gaps."""

    def test_known_3_cluster_geometry(self):
        """Three clear groups at x=15, x=40, x=65 -> known centroids + gaps."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        positions = [
            (14.0, 20.0),
            (15.0, 30.0),
            (16.0, 40.0),
            (39.0, 15.0),
            (40.0, 35.0),
            (41.0, 50.0),
            (64.0, 25.0),
            (65.0, 34.0),
            (66.0, 45.0),
        ]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["defensive_line_height"] == pytest.approx(15.0, abs=1.0)
        assert row["inter_line_gap_1"] == pytest.approx(25.0, abs=2.0)
        assert row["inter_line_gap_2"] == pytest.approx(25.0, abs=2.0)

    def test_fewer_than_3_players_gaps_nan(self):
        """2 players: 1 gap only (gap_2 = NaN)."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        positions = [(20.0, 30.0), (60.0, 40.0)]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert not pd.isna(row["inter_line_gap_1"])
        assert pd.isna(row["inter_line_gap_2"])
        assert row["defensive_line_height"] == pytest.approx(20.0, abs=1.0)

    def test_single_player(self):
        """1 player: line height = player x, both gaps NaN."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        positions = [(30.0, 34.0)]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert row["defensive_line_height"] == pytest.approx(30.0)
        assert pd.isna(row["inter_line_gap_1"])
        assert pd.isna(row["inter_line_gap_2"])

    def test_tight_cluster_small_gap(self):
        """Three tight groups near x=20,21,22 -> small but nonzero gaps."""
        from silly_kicks.tracking._team_shape import compute_team_shape

        positions = [
            (20.0, 20.0),
            (20.0, 40.0),
            (20.0, 55.0),
            (21.0, 15.0),
            (21.0, 35.0),
            (21.0, 50.0),
            (22.0, 25.0),
            (22.0, 45.0),
            (22.0, 60.0),
        ]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert row["inter_line_gap_1"] == pytest.approx(1.0, abs=0.5)
        assert row["inter_line_gap_2"] == pytest.approx(1.0, abs=0.5)


class TestAddTeamShape:
    def test_enriches_actions_with_20_columns(self):
        """add_team_shape adds 20 team-shape columns (10 metrics x 2 teams)."""
        from silly_kicks.tracking.features import add_team_shape

        # Build frames with two teams
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0],
        )
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [1.0],
                "team_id": [1],
                "player_id": [3],  # a home outfield player
                "start_x": [30.0],
                "start_y": [30.0],
                "end_x": [40.0],
                "end_y": [35.0],
                "type_id": [0],
            }
        )

        result = add_team_shape(actions, frames, home_team_id=1)

        expected_cols = [
            "team_shape_n_outfield_players_attacking",
            "team_shape_centroid_x_attacking",
            "team_shape_centroid_y_attacking",
            "team_shape_convex_hull_area_attacking",
            "team_shape_team_length_attacking",
            "team_shape_team_width_attacking",
            "team_shape_stretch_index_attacking",
            "team_shape_defensive_line_height_attacking",
            "team_shape_inter_line_gap_1_attacking",
            "team_shape_inter_line_gap_2_attacking",
            "team_shape_n_outfield_players_defending",
            "team_shape_centroid_x_defending",
            "team_shape_centroid_y_defending",
            "team_shape_convex_hull_area_defending",
            "team_shape_team_length_defending",
            "team_shape_team_width_defending",
            "team_shape_stretch_index_defending",
            "team_shape_defensive_line_height_defending",
            "team_shape_inter_line_gap_1_defending",
            "team_shape_inter_line_gap_2_defending",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == 1

    def test_attacking_is_action_team(self):
        """team_shape_centroid_x_attacking reflects the acting team's centroid."""
        from silly_kicks.tracking.features import add_team_shape
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[34.0] * 5,
        )
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [1],
                "period_id": [1],
                "time_seconds": [1.0],
                "team_id": [1],  # home team is attacking
                "player_id": [3],
                "start_x": [30.0],
                "start_y": [34.0],
                "end_x": [40.0],
                "end_y": [34.0],
                "type_id": [0],
            }
        )

        result = add_team_shape(actions, frames, home_team_id=1)
        # Home outfield centroid_x = mean(10,20,30,40,50) = 30
        assert result.iloc[0]["team_shape_centroid_x_attacking"] == pytest.approx(30.0)
        # Away outfield centroid_x = mean(60,70,80,90,95) = 79
        assert result.iloc[0]["team_shape_centroid_x_defending"] == pytest.approx(79.0)


class TestTeamShapeXfns:
    def test_xfn_column_count(self):
        """team_shape_xfns produces 54 columns (18 features x 3 states)."""
        from silly_kicks.tracking.features import team_shape_xfns

        xfns = team_shape_xfns(home_team_id=1)
        assert len(xfns) == 1

        xfn = xfns[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_xfn_introspection_nan(self):
        """frames=None -> NaN DataFrame with 54 correct column names."""
        from silly_kicks.tracking.features import team_shape_xfns

        xfns = team_shape_xfns(home_team_id=1)
        xfn = xfns[0]

        # Simulate VAEP introspection: 3 gamestates of 10 rows, no frames
        dummy = pd.DataFrame(
            {
                "game_id": [1] * 10,
                "action_id": range(10),
                "period_id": [1] * 10,
                "time_seconds": [float(i) for i in range(10)],
                "team_id": [1] * 10,
                "player_id": [1] * 10,
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [60.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [0] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]
        result = xfn(states, None)

        assert len(result.columns) == 54
        assert result.isna().all().all()
        # Verify naming pattern
        assert "team_shape_centroid_x_attacking_a0" in result.columns
        assert "team_shape_stretch_index_defending_a2" in result.columns
        assert "team_shape_defensive_line_height_attacking_a0" in result.columns
        assert "team_shape_inter_line_gap_2_defending_a2" in result.columns
