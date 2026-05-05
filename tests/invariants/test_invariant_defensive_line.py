"""Physical invariants for defensive-line geometry (TF-14)."""

from __future__ import annotations

import pytest

from tests.tracking.conftest import _make_frame_rows


@pytest.fixture
def defensive_line_both_teams():
    """Multi-frame fixture with both teams having valid back lines."""
    from silly_kicks.tracking._defensive_line import compute_defensive_line

    frames = _make_frame_rows(
        home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
        home_outfield_ys=[10.0, 20.0, 40.0, 55.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        away_outfield_xs=[95.0, 93.0, 91.0, 89.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0],
        away_outfield_ys=[10.0, 20.0, 40.0, 55.0, 34.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    return compute_defensive_line(frames, home_team_id=1, n=4)


class TestRangeInvariants:
    def test_defensive_line_x_in_pitch(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["defensive_line_x"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_back_line_high_x_in_pitch(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["back_line_high_x"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_compactness_non_negative(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["compactness_x"].dropna()
        assert (valid >= 0).all()

    def test_lateral_width_in_range(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["lateral_width"].dropna()
        assert (valid >= 0).all() and (valid <= 68).all()

    def test_max_gap_bounded_by_width(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl[dl["max_lateral_gap"].notna()]
        assert (valid["max_lateral_gap"] <= valid["lateral_width"] + 1e-9).all()

    def test_back_n_count_domain(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        valid = dl["back_n_count"].dropna()
        assert set(valid.unique()).issubset({3, 4, 5})


class TestTriangleInequality:
    def test_home_back_line_high_minus_mean_le_compactness(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        home = dl[dl["team_id"] == 1].dropna(subset=["defensive_line_x"])
        # back_line_high_x - defensive_line_x <= compactness_x
        diff = home["back_line_high_x"] - home["defensive_line_x"]
        assert (diff <= home["compactness_x"] + 1e-9).all()

    def test_away_mean_minus_back_line_high_le_compactness(self, defensive_line_both_teams):
        dl = defensive_line_both_teams
        away = dl[dl["team_id"] == 2].dropna(subset=["defensive_line_x"])
        # For away: defensive_line_x - back_line_high_x <= compactness_x
        diff = away["defensive_line_x"] - away["back_line_high_x"]
        assert (diff <= away["compactness_x"] + 1e-9).all()


class TestCrossTeamSanity:
    def test_lines_not_both_near_same_goal(self, defensive_line_both_teams):
        """Both teams' lines shouldn't cluster near the same goal."""
        dl = defensive_line_both_teams
        home = dl[dl["team_id"] == 1]["defensive_line_x"].iloc[0]
        away = dl[dl["team_id"] == 2]["defensive_line_x"].iloc[0]
        # Home line near x=0, away near x=105
        # Invariant: they shouldn't BOTH be < 20 or BOTH be > 85
        assert not (home < 20 and away < 20)
        assert not (home > 85 and away > 85)
