"""Physical invariant tests for all pitch control methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    compute_pitch_control,
)

METHODS = ["spearman", "fernandez_bornn", "voronoi"]


def _make_frame(att_pos, def_pos, att_vel=None, def_vel=None):
    rows = []
    for i, (x, y) in enumerate(att_pos):
        vx = att_vel[i][0] if att_vel else 0.0
        vy = att_vel[i][1] if att_vel else 0.0
        rows.append(
            {
                "player_id": 100 + i,
                "team_id": 1,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    for i, (x, y) in enumerate(def_pos):
        vx = def_vel[i][0] if def_vel else 0.0
        vy = def_vel[i][1] if def_vel else 0.0
        rows.append(
            {
                "player_id": 200 + i,
                "team_id": 2,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    rows.append(
        {
            "player_id": np.nan,
            "team_id": np.nan,
            "x": 52.5,
            "y": 34,
            "vx": 0,
            "vy": 0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", METHODS)
class TestBounds:
    def test_surface_in_unit_interval(self, method):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_pitch_control(frame, 1, method=method)
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()


@pytest.mark.parametrize("method", METHODS)
class TestGridBounds:
    def test_grid_within_pitch(self, method):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_pitch_control(frame, 1, method=method)
        assert s.grid_x[0] >= 0 and s.grid_x[-1] <= 105
        assert s.grid_y[0] >= 0 and s.grid_y[-1] <= 68


class TestSelfDominance:
    """Player on a cell with distant opponents -> high control."""

    @pytest.mark.parametrize(
        "method,threshold",
        [
            ("spearman", 0.95),
            ("voronoi", 0.95),
            ("fernandez_bornn", 0.70),
        ],
    )
    def test_player_on_cell_distant_opponents(self, method, threshold):
        # Attacker at (50, 34), defenders > 40m away
        # Off-pitch ball_position disables ball-travel-time filter
        frame = _make_frame([(50, 34)], [(95, 60)])
        s = compute_pitch_control(frame, 1, method=method, ball_position=(-10, -10))
        assert s.at_point(50, 34) > threshold


@pytest.mark.parametrize("method", METHODS)
class TestSymmetry:
    def test_mirrored_teams_near_half(self, method):
        frame = _make_frame([(26.25, 34)], [(78.75, 34)])
        s = compute_pitch_control(frame, 1, method=method)
        center = s.at_point(52.5, 34.0)
        assert 0.35 < center < 0.65


@pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
class TestMonotonicity:
    def test_closer_player_higher_control(self, method):
        # Attacker 5m from target, defender 45m from target
        # Off-pitch ball_position disables ball-travel-time filter
        frame = _make_frame([(45, 34)], [(95, 34)])
        s = compute_pitch_control(frame, 1, method=method, ball_position=(-10, -10))
        assert s.at_point(50, 34) > 0.5


@pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
class TestVelocityEffect:
    def test_running_toward_increases_control(self, method):
        # Off-pitch ball_position disables ball-travel-time filter
        frame_static = _make_frame([(30, 34)], [(70, 34)], att_vel=[(0, 0)], def_vel=[(0, 0)])
        frame_running = _make_frame([(30, 34)], [(70, 34)], att_vel=[(6, 0)], def_vel=[(0, 0)])
        s_static = compute_pitch_control(frame_static, 1, method=method, ball_position=(-10, -10))
        s_running = compute_pitch_control(frame_running, 1, method=method, ball_position=(-10, -10))
        # Control at a point ahead of the attacker should increase
        assert s_running.at_point(50, 34) > s_static.at_point(50, 34)


class TestDecompositionConsistency:
    def test_spearman_sum_reconstructs(self):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_pitch_control(frame, 1, method="spearman", decompose=True)
        att_mask = np.isin(s.player_ids, [100, 101])
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        all_sum = s.per_player_influence.sum(axis=0)
        safe_all = np.maximum(all_sum, 1e-10)
        reconstructed = np.where(all_sum > 1e-10, att_sum / safe_all, 0.5)
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-8)

    def test_voronoi_binary_sums_to_one(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_pitch_control(frame, 1, method="voronoi", decompose=True)
        assert (s.per_player_influence.sum(axis=0) == 1.0).all()


class TestNoNaN:
    def test_fernandez_bornn_near_max_speed(self):
        frame = _make_frame([(50, 34)], [(70, 34)], att_vel=[(12.99, 0)], def_vel=[(0, 0)])
        s = compute_pitch_control(frame, 1, method="fernandez_bornn")
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()
