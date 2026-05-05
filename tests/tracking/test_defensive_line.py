"""Tests for silly_kicks.tracking._defensive_line.compute_defensive_line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_frame_rows(
    *,
    home_team_id=1,
    away_team_id=2,
    home_outfield_xs: list[float],
    home_outfield_ys: list[float],
    away_outfield_xs: list[float],
    away_outfield_ys: list[float],
    home_gk_pos=(3.0, 34.0),
    away_gk_pos=(102.0, 34.0),
    frame_id=1,
    period_id=1,
    time_seconds=1.0,
):
    """Build a single-frame fixture with specified outfield positions."""
    rows = []
    pid = 1
    # Ball
    rows.append(
        dict(
            game_id=1,
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
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
    )
    # Home GK
    rows.append(
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=pid,
            team_id=home_team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=home_gk_pos[0],
            y=home_gk_pos[1],
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
    )
    pid += 1
    # Home outfield
    for x, y in zip(home_outfield_xs, home_outfield_ys, strict=True):
        rows.append(
            dict(
                game_id=1,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=time_seconds,
                frame_rate=25.0,
                player_id=pid,
                team_id=home_team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=x,
                y=y,
                source_provider="sportec",
                team_attacking_direction="ltr",
            )
        )
        pid += 1
    # Away GK
    rows.append(
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=pid,
            team_id=away_team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=away_gk_pos[0],
            y=away_gk_pos[1],
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
    )
    pid += 1
    # Away outfield
    for x, y in zip(away_outfield_xs, away_outfield_ys, strict=True):
        rows.append(
            dict(
                game_id=1,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=time_seconds,
                frame_rate=25.0,
                player_id=pid,
                team_id=away_team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=x,
                y=y,
                source_provider="sportec",
                team_attacking_direction="ltr",
            )
        )
        pid += 1
    return pd.DataFrame(rows)


class TestFixedN4:
    def test_basic_4_defenders(self):
        """4 home defenders at known positions -> exact values."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        # Home defends x=0; back line at x=10,12,11,13 sorted -> 10,11,12,13
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 40.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 60.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        # Home team row
        home = result[result["team_id"] == 1].iloc[0]
        # Back 4 (lowest x): 10, 11, 12, 13
        assert home["defensive_line_x"] == pytest.approx(11.5)  # mean(10,11,12,13)
        assert home["back_line_high_x"] == pytest.approx(13.0)  # max
        assert home["compactness_x"] == pytest.approx(3.0)  # 13-10
        # y values for back 4: 20, 40, 25, 48 -> sorted: 20, 25, 40, 48
        assert home["lateral_width"] == pytest.approx(28.0)  # 48-20
        # gaps: 5, 15, 8 -> max = 15
        assert home["max_lateral_gap"] == pytest.approx(15.0)
        assert home["back_n_count"] == 4

    def test_both_teams_computed(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        teams = result["team_id"].unique()
        assert set(teams) == {1, 2}

    def test_away_team_defends_x105(self):
        """Away team's back line = highest-x outfield players."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 11.0, 13.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 94.0, 92.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        away = result[result["team_id"] == 2].iloc[0]
        # Away defends x=105; back 4 (highest x): 95, 94, 93, 92
        assert away["defensive_line_x"] == pytest.approx(93.5)
        # back_line_high_x = min(x) for away (furthest from x=105)
        assert away["back_line_high_x"] == pytest.approx(92.0)
        assert away["compactness_x"] == pytest.approx(3.0)
        assert away["back_n_count"] == 4


class TestFixedN3N5:
    def test_n3(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 40.0, 50.0],
            home_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 60.0, 50.0],
            away_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=3)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3
        assert home["defensive_line_x"] == pytest.approx(12.0)  # mean(10,12,14)

    def test_n5(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 18.0, 50.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 87.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=5)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 5
        assert home["defensive_line_x"] == pytest.approx(14.0)  # mean(10,12,14,16,18)


class TestEdgeCases:
    def test_fewer_than_3_outfield_nan(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0],  # only 2
            home_outfield_ys=[20.0, 40.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        assert pd.isna(home["defensive_line_x"])
        assert pd.isna(home["back_n_count"])
        # Away should still work
        away = result[result["team_id"] == 2].iloc[0]
        assert not pd.isna(away["defensive_line_x"])

    def test_fixed_n_clamped_to_available(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0],  # 3 available, n=4 requested
            home_outfield_ys=[20.0, 34.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3  # clamped

    def test_gk_excluded(self):
        """Even if GK is at x=2 (lower than outfield), it shouldn't be in back line."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_gk_pos=(2.0, 34.0),  # GK very close to goal
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        # Back 4 should be 10,12,14,16 -- NOT include GK at x=2
        assert home["defensive_line_x"] == pytest.approx(13.0)

    def test_nan_coordinates_excluded(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, float("nan"), 14.0, 50.0],
            home_outfield_ys=[20.0, 25.0, float("nan"), 48.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        home = result[result["team_id"] == 1].iloc[0]
        # Only 4 valid outfield (NaN excluded); n=4 takes all 4: 10,12,14,50
        assert home["back_n_count"] == 4

    def test_empty_frames(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "frame_rate",
                "player_id",
                "team_id",
                "is_ball",
                "is_goalkeeper",
                "x",
                "y",
                "source_provider",
                "team_attacking_direction",
            ]
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert len(result) == 0

    def test_invalid_n_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        with pytest.raises(ValueError, match="n must be"):
            compute_defensive_line(frames, home_team_id=1, n=2)
        with pytest.raises(ValueError, match="n must be"):
            compute_defensive_line(frames, home_team_id=1, n=6)

    def test_invalid_adaptive_max_n_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        with pytest.raises(ValueError, match="adaptive_max_n"):
            compute_defensive_line(frames, home_team_id=1, n="adaptive", adaptive_max_n=10)

    def test_ltr_guard_raises(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        frames["team_attacking_direction"] = "rtl"
        with pytest.raises(ValueError, match="LTR-normalized"):
            compute_defensive_line(frames, home_team_id=1, n=4)

    def test_ltr_guard_allows_nan(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 25.0, 40.0, 48.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 25.0, 40.0, 48.0],
        )
        frames["team_attacking_direction"] = None  # all NaN
        # Should not raise
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert len(result) > 0


class TestAdaptive:
    def test_detects_4_back(self):
        """4 defenders clustered at x=10-13, big gap, then midfielders at x=40+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 13.0, 40.0, 42.0, 55.0, 60.0, 65.0, 70.0],
            home_outfield_ys=[15.0, 25.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0, 48.0, 40.0, 35.0, 30.0, 25.0],
            away_outfield_ys=[15.0, 25.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_detects_5_back(self):
        """5 clustered at x=10-14, big gap at [4]->[5], then midfield at x=45+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 13.0, 14.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 87.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 5

    def test_detects_3_back(self):
        """3 clustered at x=10-12, big gap to midfield at x=40+."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        # gaps: 1, 1, 28, 2, 2, 2, 2, 2, 2
        # cut N=3 -> gaps[2]=28; cut N=4 -> gaps[3]=2; cut N=5 -> gaps[4]=2
        # 28 vs 2 -> 28 >= 1.5*2 -> dominant at N=3
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 11.0, 12.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0],
            home_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 70.0, 65.0, 50.0, 45.0, 40.0, 35.0, 30.0],
            away_outfield_ys=[20.0, 34.0, 48.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3

    def test_no_dominant_gap_defaults_to_4(self):
        """Evenly spaced players -> no dominant gap -> N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        # Evenly spaced at 10, 15, 20, 25, 30, 35, ... (gap=5 everywhere)
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            home_outfield_ys=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
            away_outfield_xs=[95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0],
            away_outfield_ys=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_all_same_x_defaults_to_4(self):
        """Degenerate: all at same x -> all gaps 0 -> N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[20.0] * 10,
            home_outfield_ys=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            away_outfield_xs=[80.0] * 10,
            away_outfield_ys=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4

    def test_exactly_3_outfield(self):
        """Only 3 outfield -> N=3 (no cuts to examine)."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0],
            home_outfield_ys=[20.0, 34.0, 48.0],
            away_outfield_xs=[95.0, 85.0, 75.0],
            away_outfield_ys=[20.0, 34.0, 48.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 3

    def test_exactly_4_outfield_defaults_to_4(self):
        """P=4, single cut [2]->[3] -> defaults to N=4."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n="adaptive")
        home = result[result["team_id"] == 1].iloc[0]
        assert home["back_n_count"] == 4


class TestMultiGame:
    def test_game_id_in_output_columns(self):
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        assert "game_id" in result.columns

    def test_multi_game_no_collision(self):
        """Two games with same (period_id, frame_id) produce separate rows."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        f1 = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        f2 = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[85.0, 83.0, 81.0, 79.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        f2["game_id"] = 2  # different game, same period_id=1, frame_id=1
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        # Should have 4 rows: 2 games x 2 teams
        assert len(result) == 4
        g1_home = result[(result["game_id"] == 1) & (result["team_id"] == 1)].iloc[0]
        g2_home = result[(result["game_id"] == 2) & (result["team_id"] == 1)].iloc[0]
        assert g1_home["defensive_line_x"] == pytest.approx(13.0)
        assert g2_home["defensive_line_x"] == pytest.approx(23.0)


class TestMultiPeriod:
    def test_period_isolation(self):
        """Two periods don't bleed into each other."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        f1 = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[95.0, 93.0, 91.0, 89.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            period_id=1,
            frame_id=1,
        )
        f2 = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[85.0, 83.0, 81.0, 79.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            period_id=2,
            frame_id=1,
        )
        frames = pd.concat([f1, f2], ignore_index=True)
        result = compute_defensive_line(frames, home_team_id=1, n=4)
        # Period 1 home line
        p1_home = result[(result["period_id"] == 1) & (result["team_id"] == 1)].iloc[0]
        assert p1_home["defensive_line_x"] == pytest.approx(13.0)
        # Period 2 home line (different positions)
        p2_home = result[(result["period_id"] == 2) & (result["team_id"] == 1)].iloc[0]
        assert p2_home["defensive_line_x"] == pytest.approx(23.0)
