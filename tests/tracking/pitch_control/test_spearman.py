"""Tests for Spearman kinematic pitch control model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control._params import SpearmanParams
from silly_kicks.tracking.pitch_control._spearman import (
    _compute_influence,
    compute_spearman,
    compute_tti,
)


def _make_frame(
    att_positions, def_positions, att_vel=None, def_vel=None, att_team_id=1, def_team_id=2, att_gk_idx=0, def_gk_idx=0
):
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
                "is_goalkeeper": (i == att_gk_idx),
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
                "is_goalkeeper": (i == def_gk_idx),
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


class TestTTI:
    def test_stationary_player(self):
        """Stationary player TTI = reaction_time + sqrt(2*a*d) / a."""
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        target = np.array([[10.0, 0.0]])
        tti = compute_tti(pos, vel, target, reaction_time=0.7, max_acceleration=7.0)
        # d=10, v_proj=0: TTI = 0.7 + sqrt(2*7*10)/7 = 0.7 + sqrt(140)/7
        expected = 0.7 + np.sqrt(140.0) / 7.0
        np.testing.assert_allclose(tti[0, 0], expected, rtol=1e-10)

    def test_player_moving_toward_target(self):
        """Player moving toward target arrives sooner."""
        pos = np.array([[0.0, 0.0]])
        target = np.array([[10.0, 0.0]])
        vel_toward = np.array([[5.0, 0.0]])
        vel_away = np.array([[-5.0, 0.0]])
        tti_toward = compute_tti(pos, vel_toward, target, 0.7, 7.0)[0, 0]
        tti_away = compute_tti(pos, vel_away, target, 0.7, 7.0)[0, 0]
        assert tti_toward < tti_away

    def test_player_at_target(self):
        """Player already at target -> TTI = reaction_time."""
        pos = np.array([[5.0, 5.0]])
        vel = np.array([[3.0, 0.0]])
        target = np.array([[5.0, 5.0]])
        tti = compute_tti(pos, vel, target, 0.7, 7.0)[0, 0]
        assert abs(tti - 0.7) < 1e-10

    def test_broadcast_shape(self):
        """Multiple players x multiple targets."""
        pos = np.array([[0, 0], [10, 10], [20, 20]], dtype="float64")
        vel = np.zeros((3, 2))
        targets = np.array([[5, 5], [15, 15]], dtype="float64")
        tti = compute_tti(pos, vel, targets, 0.7, 7.0)
        assert tti.shape == (3, 2)


class TestInfluence:
    def test_earlier_arrival_higher_influence(self):
        """Player arriving much earlier than opponent -> influence near 1."""
        team_tti = np.array([[1.0]])  # arrives at t=1
        opponent_min = np.array([5.0])  # opponent arrives at t=5
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert influence[0, 0] > 0.95

    def test_later_arrival_lower_influence(self):
        """Player arriving much later -> influence near 0."""
        team_tti = np.array([[5.0]])
        opponent_min = np.array([1.0])
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert influence[0, 0] < 0.05

    def test_equal_arrival_half_influence(self):
        """Same arrival time -> influence = 0.5."""
        team_tti = np.array([[3.0]])
        opponent_min = np.array([3.0])
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert abs(influence[0, 0] - 0.5) < 1e-10


class TestComputeSpearman:
    def test_single_attacker_dominates(self):
        frame = _make_frame([(52.5, 34.0)], [(90.0, 34.0)])
        s = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
        # Attacker near center, defender far -> high control at center
        assert s.at_point(52.5, 34.0) > 0.7

    def test_symmetric_equals_half(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
        assert abs(s.at_point(52.5, 34.0) - 0.5) < 0.05

    def test_velocity_effect(self):
        """Player running toward a cell gets higher control there."""
        frame_static = _make_frame(
            [(30, 34)],
            [(70, 34)],
            att_vel=[(0, 0)],
            def_vel=[(0, 0)],
        )
        frame_running = _make_frame(
            [(30, 34)],
            [(70, 34)],
            att_vel=[(5, 0)],
            def_vel=[(0, 0)],
        )
        s_static = compute_spearman(frame_static, 1, SpearmanParams())
        s_running = compute_spearman(frame_running, 1, SpearmanParams())
        # Attacker running right -> more control on right side
        assert s_running.at_point(50, 34) > s_static.at_point(50, 34)

    def test_gk_weighting(self):
        """GK with lambda_gk > 1 contributes more influence."""
        frame = _make_frame([(20, 34)], [(80, 34)])
        params_no_gk = SpearmanParams(lambda_gk=1.0)
        params_gk = SpearmanParams(lambda_gk=3.0)
        s_no = compute_spearman(frame, 1, params_no_gk)
        s_gk = compute_spearman(frame, 1, params_gk)
        # GK is player 100 (att_gk_idx=0); with higher lambda, attacker controls more
        assert s_gk.at_point(20, 34) >= s_no.at_point(20, 34)

    def test_bounds(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_spearman(frame, 1, SpearmanParams())
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()

    def test_decomposition_sums_to_surface(self):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_spearman(frame, 1, SpearmanParams(), decompose=True)
        assert s.per_player_influence is not None
        # Sum attacking influence / (sum att + sum def) ~ surface
        att_mask = np.isin(s.player_ids, [100, 101])
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        all_sum = s.per_player_influence.sum(axis=0)
        reconstructed = np.where(all_sum > 1e-10, att_sum / all_sum, 0.5)
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-10)

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_spearman(frame, 1, SpearmanParams())
        assert s.method == "spearman"

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
        s = compute_spearman(frame, 1, SpearmanParams())
        assert (s.surface == 0.5).all()
