"""Tests for Voronoi pitch control model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control._params import VoronoiParams
from silly_kicks.tracking.pitch_control._voronoi import compute_voronoi


def _make_frame(att_positions, def_positions, att_team_id=1, def_team_id=2):
    """Build a minimal tracking frame for testing."""
    rows = []
    for i, (x, y) in enumerate(att_positions):
        rows.append(
            {
                "player_id": 100 + i,
                "team_id": att_team_id,
                "x": x,
                "y": y,
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    for i, (x, y) in enumerate(def_positions):
        rows.append(
            {
                "player_id": 200 + i,
                "team_id": def_team_id,
                "x": x,
                "y": y,
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    # Ball row
    rows.append(
        {
            "player_id": np.nan,
            "team_id": np.nan,
            "x": 52.5,
            "y": 34.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


class TestVoronoiBasic:
    def test_single_attacker_controls_all(self):
        frame = _make_frame([(52.5, 34.0)], [])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 1.0).all()

    def test_single_defender_controls_none(self):
        frame = _make_frame([], [(52.5, 34.0)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 0.0).all()

    def test_symmetric_equal_split(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        # Left half should be attacker (1.0), right half defender (0.0)
        mid_x_idx = len(s.grid_x) // 2
        assert s.surface[:, :mid_x_idx].mean() > 0.8
        assert s.surface[:, mid_x_idx:].mean() < 0.2

    def test_binary_output(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        unique_vals = np.unique(s.surface)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert s.method == "voronoi"

    def test_grid_bounds(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert s.grid_x[0] >= 0 and s.grid_x[-1] <= 105
        assert s.grid_y[0] >= 0 and s.grid_y[-1] <= 68


class TestVoronoiDecomposition:
    def test_decompose_binary(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams(), decompose=True)
        assert s.per_player_influence is not None
        assert s.player_ids is not None
        # Each cell assigned to exactly one player
        assert (s.per_player_influence.sum(axis=0) == 1.0).all()

    def test_player_share_is_team_fraction(self):
        frame = _make_frame([(30, 34), (50, 50)], [(70, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams(), decompose=True)
        # Two attackers: shares within team 1 sum to 1.0
        share_att_0 = s.player_share(100)
        share_att_1 = s.player_share(101)
        assert abs(share_att_0 + share_att_1 - 1.0) < 1e-10
        # Solo defender: 100% of their team
        share_def = s.player_share(200)
        assert abs(share_def - 1.0) < 1e-10


class TestVoronoiEdgeCases:
    def test_empty_frame(self):
        frame = pd.DataFrame(columns=["player_id", "team_id", "x", "y", "is_ball", "is_goalkeeper"])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 0.5).all()

    def test_nan_positions_filtered(self):
        frame = _make_frame([(50, 34), (np.nan, np.nan)], [(80, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        # Should not crash; NaN player ignored
        assert not np.isnan(s.surface).any()

    def test_ball_position_ignored(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s1 = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        s2 = compute_voronoi(
            frame,
            attacking_team_id=1,
            params=VoronoiParams(),
            ball_position=(10, 10),
        )
        np.testing.assert_array_equal(s1.surface, s2.surface)
