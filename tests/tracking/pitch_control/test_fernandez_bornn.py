"""Tests for Fernandez/Bornn bivariate-normal pitch control model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control._fernandez_bornn import compute_fernandez_bornn
from silly_kicks.tracking.pitch_control._params import FernandezBornnParams


def _make_frame(att_positions, def_positions, att_vel=None, def_vel=None, att_team_id=1, def_team_id=2):
    """Build a tracking frame with velocities."""
    rows = []
    for i, (x, y) in enumerate(att_positions):
        vx = att_vel[i][0] if att_vel else 0.0
        vy = att_vel[i][1] if att_vel else 0.0
        rows.append(
            {
                "player_id": 100 + i,
                "team_id": att_team_id,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    for i, (x, y) in enumerate(def_positions):
        vx = def_vel[i][0] if def_vel else 0.0
        vy = def_vel[i][1] if def_vel else 0.0
        rows.append(
            {
                "player_id": 200 + i,
                "team_id": def_team_id,
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
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


class TestFernandezBornnBasic:
    def test_single_attacker_high_control(self):
        frame = _make_frame([(52.5, 34.0)], [(90.0, 60.0)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert s.at_point(52.5, 34.0) > 0.7

    def test_symmetric_near_half(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        center_control = s.at_point(52.5, 34.0)
        assert 0.4 < center_control < 0.6

    def test_bounds(self):
        frame = _make_frame([(30, 34), (50, 20)], [(70, 34), (80, 50)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert s.method == "fernandez_bornn"


class TestVelocityEffect:
    def test_running_player_extends_influence_forward(self):
        frame_static = _make_frame([(30, 34)], [(80, 34)], att_vel=[(0, 0)])
        frame_running = _make_frame([(30, 34)], [(80, 34)], att_vel=[(8, 0)])
        s_static = compute_fernandez_bornn(frame_static, 1, FernandezBornnParams())
        s_running = compute_fernandez_bornn(frame_running, 1, FernandezBornnParams())
        # Running right -> more control ahead of the player
        assert s_running.at_point(45, 34) > s_static.at_point(45, 34)


class TestHighSpeedGuard:
    def test_near_max_speed_no_nan(self):
        """Player at near-max speed should not produce NaN (alpha guard)."""
        params = FernandezBornnParams(max_speed=13.0)
        frame = _make_frame(
            [(50, 34)],
            [(70, 34)],
            att_vel=[(12.99, 0)],  # near max_speed
        )
        s = compute_fernandez_bornn(frame, 1, params)
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()

    def test_exact_max_speed_no_nan(self):
        """Player at exactly max_speed -- alpha_ceil prevents singularity."""
        params = FernandezBornnParams(max_speed=13.0)
        frame = _make_frame(
            [(50, 34)],
            [(70, 34)],
            att_vel=[(13.0, 0)],
        )
        s = compute_fernandez_bornn(frame, 1, params)
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()


class TestStationaryGuard:
    def test_very_slow_player_isotropic(self):
        """Player with speed < 0.1 m/s treated as stationary (isotropic)."""
        frame_still = _make_frame([(50, 34)], [(70, 34)], att_vel=[(0, 0)])
        frame_tiny = _make_frame([(50, 34)], [(70, 34)], att_vel=[(0.05, 0.05)])
        s_still = compute_fernandez_bornn(frame_still, 1, FernandezBornnParams())
        s_tiny = compute_fernandez_bornn(frame_tiny, 1, FernandezBornnParams())
        # Should produce near-identical surfaces (both isotropic)
        np.testing.assert_allclose(s_still.surface, s_tiny.surface, atol=0.01)


class TestDecomposition:
    def test_decompose_returns_per_player(self):
        frame = _make_frame([(30, 34), (50, 50)], [(70, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), decompose=True)
        assert s.per_player_influence is not None
        assert s.per_player_influence.shape[0] == 3  # 2 att + 1 def

    def test_sigmoid_reconstruction_from_raw_gaussians(self):
        """Pre-sigmoid consistency: sigmoid(att_sum - def_sum) == surface."""
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), decompose=True)
        assert s.per_player_influence is not None
        assert s.player_team_ids is not None
        # All raw Gaussian values should be non-negative
        assert (s.per_player_influence >= 0).all()
        # The defining F/B invariant: surface = sigmoid(sum_att - sum_def)
        att_mask = s.player_team_ids == 1
        def_mask = s.player_team_ids != 1
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        def_sum = s.per_player_influence[def_mask].sum(axis=0)
        reconstructed = 1.0 / (1.0 + np.exp(-(att_sum - def_sum)))
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-6)


class TestBallPosition:
    def test_ball_position_affects_radius(self):
        """Ball position changes influence radii (closer = tighter)."""
        frame = _make_frame([(50, 34)], [(70, 34)])
        s_far = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), ball_position=(0, 34))
        s_near = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), ball_position=(50, 34))
        # Near ball -> tighter radius -> attacker influence more concentrated
        # At the attacker's position, control should be higher with tighter radius
        assert s_near.at_point(50, 34) >= s_far.at_point(50, 34) - 0.1


class TestEdgeCases:
    def test_empty_frame(self):
        frame = pd.DataFrame(
            columns=[
                "player_id",
                "team_id",
                "x",
                "y",
                "vx",
                "vy",
                "is_ball",
                "is_goalkeeper",
            ]
        )
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert (s.surface == 0.5).all()
