"""Physical invariants for team shape envelope (TF-31)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.tracking.test_team_shape import _make_team_frames


@pytest.fixture
def team_shape_result():
    """Known 10-outfield-player fixture."""
    from silly_kicks.tracking._team_shape import compute_team_shape

    frames = _make_team_frames(
        outfield_positions=[
            (10.0, 10.0),
            (20.0, 10.0),
            (30.0, 20.0),
            (40.0, 30.0),
            (50.0, 34.0),
            (60.0, 40.0),
            (70.0, 50.0),
            (80.0, 50.0),
            (90.0, 60.0),
            (95.0, 60.0),
        ],
    )
    return compute_team_shape(frames, team_id=1)


class TestRangeInvariants:
    def test_convex_hull_area_non_negative(self, team_shape_result):
        valid = team_shape_result["convex_hull_area"].dropna()
        assert (valid >= 0).all()

    def test_stretch_index_non_negative(self, team_shape_result):
        valid = team_shape_result["stretch_index"].dropna()
        assert (valid >= 0).all()

    def test_team_length_in_pitch(self, team_shape_result):
        valid = team_shape_result["team_length"].dropna()
        assert (valid >= 0).all() and (valid <= 105).all()

    def test_team_width_in_pitch(self, team_shape_result):
        valid = team_shape_result["team_width"].dropna()
        assert (valid >= 0).all() and (valid <= 68).all()

    def test_n_outfield_players_in_range(self, team_shape_result):
        valid = team_shape_result["n_outfield_players"].dropna()
        assert (valid >= 1).all() and (valid <= 11).all()

    def test_stretch_index_bounded_by_max_extent(self, team_shape_result):
        """stretch_index <= max(team_length, team_width)."""
        df = team_shape_result.dropna(subset=["stretch_index"])
        max_extent = np.maximum(df["team_length"], df["team_width"])
        assert (df["stretch_index"] <= max_extent + 1e-9).all()
