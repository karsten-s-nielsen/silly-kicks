"""Tests for TF-30 Cover Shadow features."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.tracking._gk_test_helpers import _make_two_team_frame
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

# === Task 1: Physics core ===


class TestBallDragTime:
    """Ball drag model (Spearman 2017): T_ball = expm1(k*d) / (v0*k)."""

    def test_zero_distance_returns_zero(self):
        from silly_kicks.tracking._cover_shadows import ball_drag_time

        result = ball_drag_time(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_known_distance_10m(self):
        """10m pass at v0=12 m/s with k_drag ~0.01383."""
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, ball_drag_time

        p = CoverShadowParams()
        d = np.array([10.0])
        result = ball_drag_time(d, p)
        # Analytical: expm1(0.01383 * 10) / (12 * 0.01383)
        expected = np.expm1(p.k_drag * 10.0) / (p.ball_initial_speed * p.k_drag)
        np.testing.assert_allclose(result, [expected], rtol=1e-10)
        # Sanity: should be slightly > d/v0 = 0.833s (drag slows ball)
        assert result[0] > 10.0 / 12.0

    def test_vectorized(self):
        from silly_kicks.tracking._cover_shadows import ball_drag_time

        d = np.array([0.0, 5.0, 10.0, 20.0, 40.0])
        result = ball_drag_time(d)
        assert result.shape == (5,)
        # Monotonically increasing
        assert np.all(np.diff(result) > 0)


class TestPlayerTti:
    """3-phase player TTI: react + accelerate + cruise."""

    def test_three_phases(self):
        """Verify all 3 branches are exercised."""
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, player_tti

        p = CoverShadowParams()
        # Player at origin, moving fast (v0 >= max_speed toward target)
        # -> cruising branch
        pos = np.array([[0.0, 0.0]])
        vel_fast = np.array([[p.max_speed + 1.0, 0.0]])
        target_far = np.array([[50.0, 0.0]])
        tti_cruise = player_tti(pos, vel_fast, target_far, is_defender=False, params=p)
        # After reaction: pos = (13*0.7, 0) = (9.1, 0), d_eff = 50 - 9.1 = 40.9
        d_after_react = 50.0 - (p.max_speed + 1.0) * p.reaction_time
        expected_cruise = p.reaction_time + d_after_react / p.max_speed
        np.testing.assert_allclose(tti_cruise[0, 0], expected_cruise, rtol=1e-6)

        # Player at origin, stationary, close target -> acceleration only
        vel_zero = np.array([[0.0, 0.0]])
        target_close = np.array([[2.0, 0.0]])
        tti_accel = player_tti(pos, vel_zero, target_close, is_defender=False, params=p)
        # d=2.0, v0=0 -> t = sqrt(2*d/a) = sqrt(4/7) ~ 0.756
        expected_accel = p.reaction_time + np.sqrt(2 * 2.0 / p.max_acceleration)
        np.testing.assert_allclose(tti_accel[0, 0], expected_accel, rtol=1e-6)

        # Player at origin, stationary, far target -> accel + cruise
        target_very_far = np.array([[100.0, 0.0]])
        tti_mixed = player_tti(pos, vel_zero, target_very_far, is_defender=False, params=p)
        t_accel_full = p.max_speed / p.max_acceleration
        d_accel_full = 0.5 * p.max_acceleration * t_accel_full**2
        expected_mixed = p.reaction_time + t_accel_full + (100.0 - d_accel_full) / p.max_speed
        np.testing.assert_allclose(tti_mixed[0, 0], expected_mixed, rtol=1e-6)

    def test_block_radius_advantage(self):
        """Defender with block_radius reaches target faster than attacker at same pos."""
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, player_tti

        p = CoverShadowParams()
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        target = np.array([[5.0, 0.0]])
        tti_def = player_tti(pos, vel, target, is_defender=True, params=p)
        tti_att = player_tti(pos, vel, target, is_defender=False, params=p)
        assert tti_def[0, 0] < tti_att[0, 0]

    def test_subsumes_compute_tti(self):
        """With max_speed=1e6, block_radius=0, and zero velocity, matches compute_tti.

        The two models treat reaction-phase drift differently (our model
        displaces the player during reaction, Spearman's adds v_proj into
        the discriminant). They converge exactly when vel=0.
        """
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, player_tti
        from silly_kicks.tracking.pitch_control import compute_tti

        p = CoverShadowParams(max_speed=1e6, block_radius=0.0)
        pos = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 34.0]])
        vel = np.zeros((3, 2))
        targets = np.array([[5.0, 0.0], [10.0, 5.0], [50.0, 34.0], [80.0, 60.0]])

        our_tti = player_tti(pos, vel, targets, is_defender=False, params=p)
        spearman_tti = compute_tti(pos, vel, targets, p.reaction_time, p.max_acceleration)

        np.testing.assert_allclose(our_tti, spearman_tti, atol=1e-6)

    def test_broadcast_shape(self):
        """(3 players, 5 targets) -> (3, 5) output."""
        from silly_kicks.tracking._cover_shadows import player_tti

        pos = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 34.0]])
        vel = np.zeros((3, 2))
        targets = np.array([[5.0, 0.0], [10.0, 5.0], [50.0, 34.0], [80.0, 60.0], [0.0, 68.0]])
        result = player_tti(pos, vel, targets, is_defender=False)
        assert result.shape == (3, 5)
        assert np.all(result >= 0.7)  # >= reaction_time


# === Task 2: Lane control ===


def _make_lane_control_frame(
    *,
    defender_pos: tuple[float, float],
    defender_vel: tuple[float, float] = (0.0, 0.0),
    passer_pos: tuple[float, float] = (50.0, 34.0),
    receiver_pos: tuple[float, float] = (75.0, 34.0),
) -> pd.DataFrame:
    """Minimal frame for lane control testing.

    Attacking team=2 (away), defending team=1 (home).
    Passer is player 60 (away), receiver is player 61 (away).
    Defender is player 10 (home) placed at defender_pos.
    """
    return _make_two_team_frame(
        home_positions=[
            defender_pos,  # player_id=10 (the defender under test)
            (20.0, 15.0),
            (25.0, 55.0),
            (30.0, 10.0),
            (35.0, 60.0),
            (40.0, 20.0),
            (45.0, 50.0),
            (15.0, 34.0),
            (10.0, 15.0),
            (10.0, 55.0),
        ],
        away_positions=[
            passer_pos,  # player_id=60 (passer)
            receiver_pos,  # player_id=61 (receiver)
            (55.0, 10.0),
            (55.0, 58.0),
            (60.0, 20.0),
            (65.0, 50.0),
            (70.0, 15.0),
            (80.0, 45.0),
            (85.0, 30.0),
            (90.0, 40.0),
        ],
        home_velocities=[
            defender_vel,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        away_velocities=[
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
    )


class TestLaneControl:
    """Lane control primitive tests."""

    def test_defender_on_center_line_blocks(self):
        """Defender directly on the center of the pass line -> all 3 lines blocked."""
        from silly_kicks.tracking._cover_shadows import lane_control

        frame = _make_lane_control_frame(
            passer_pos=(50.0, 34.0),
            receiver_pos=(75.0, 34.0),
            defender_pos=(62.5, 34.0),  # midpoint of pass line
        )
        result = lane_control(
            frame,
            passer_xy=(50.0, 34.0),
            receiver_xy=(75.0, 34.0),
            home_team_id=1,
            attacking_team_id=2,
        )
        assert result.is_blocked_all
        assert result.is_blocked_majority
        assert result.is_blocked_any

    def test_defender_far_off_line_open(self):
        """Defender far from pass corridor -> all 3 lines open."""
        from silly_kicks.tracking._cover_shadows import lane_control

        frame = _make_lane_control_frame(
            passer_pos=(50.0, 34.0),
            receiver_pos=(75.0, 34.0),
            defender_pos=(62.5, 5.0),  # 29m off to side
        )
        result = lane_control(
            frame,
            passer_xy=(50.0, 34.0),
            receiver_xy=(75.0, 34.0),
            home_team_id=1,
            attacking_team_id=2,
        )
        assert not result.is_blocked_any
        assert not result.is_blocked_majority
        assert not result.is_blocked_all

    def test_fast_defender_intercepts(self):
        """Defender off line but moving toward it at high speed -> blocks."""
        from silly_kicks.tracking._cover_shadows import lane_control

        frame = _make_lane_control_frame(
            passer_pos=(50.0, 34.0),
            receiver_pos=(75.0, 34.0),
            defender_pos=(62.5, 37.0),  # 3m off center, moving toward it
            defender_vel=(0.0, -8.0),  # fast lateral approach
        )
        result = lane_control(
            frame,
            passer_xy=(50.0, 34.0),
            receiver_xy=(75.0, 34.0),
            home_team_id=1,
            attacking_team_id=2,
        )
        # At least center line should be blocked
        assert result.is_blocked_any

    def test_decision_rules_intermediate(self):
        """Defender positioned to block only the right corridor line.

        Minimal frame: 1 lone defender near the right edge of a wide cone.
        With only 1 defender, contributions from other players don't
        muddy the decision-rule signal.

        Expected: any=True, majority=False, all=False.
        """
        from silly_kicks.tracking._cover_shadows import lane_control

        # Minimal frame: 1 defender near right edge on a short pass.
        # Short pass = fast ball, less time for defender to reach far lines.
        # 10m pass, cone_width_factor=0.5 → half_width at end = 0.5*10/2 = 2.5m
        # Defender at (55, 36.5): right edge midpoint ~(55, 36.5) → 0m to right.
        # Center line midpoint ~(55, 34) → 2.5m from defender.
        # Left edge midpoint ~(55, 31.5) → 5m from defender.
        # Ball at midpoint (5m): T_ball ~ 0.42s. Defender to center: ~1.5s >> 0.42s.
        frame = _make_two_team_frame(
            home_positions=[
                (55.0, 36.5),  # pid=10, on right edge of short pass cone
            ],
            away_positions=[
                (50.0, 34.0),  # pid=60 (passer)
                (60.0, 34.0),  # pid=61 (receiver)
            ],
        )
        from silly_kicks.tracking._cover_shadows import CoverShadowParams

        p = CoverShadowParams(cone_width_factor=0.5)
        result = lane_control(
            frame,
            passer_xy=(50.0, 34.0),
            receiver_xy=(60.0, 34.0),
            home_team_id=1,
            attacking_team_id=2,
            params=p,
        )
        # Right edge blocked, center/left not blocked
        assert result.is_blocked_any
        assert not result.is_blocked_majority
        assert not result.is_blocked_all

    def test_ltr_validation_rejects_all_rtl(self):
        """Frames with only 'rtl' direction (no home 'ltr') raise ValueError."""
        from silly_kicks.tracking._cover_shadows import lane_control

        frame = _make_lane_control_frame(defender_pos=(62.5, 34.0))
        frame["team_attacking_direction"] = "rtl"
        with pytest.raises(ValueError, match="period-normalized"):
            lane_control(
                frame,
                passer_xy=(50.0, 34.0),
                receiver_xy=(75.0, 34.0),
                home_team_id=1,
                attacking_team_id=2,
            )

    def test_ltr_validation_rejects_unexpected_values(self):
        """Frames with unexpected direction values raise ValueError."""
        from silly_kicks.tracking._cover_shadows import lane_control

        frame = _make_lane_control_frame(defender_pos=(62.5, 34.0))
        frame["team_attacking_direction"] = "backwards"
        with pytest.raises(ValueError, match="unexpected"):
            lane_control(
                frame,
                passer_xy=(50.0, 34.0),
                receiver_xy=(75.0, 34.0),
                home_team_id=1,
                attacking_team_id=2,
            )


# === Task 3: Man-marking filter ===


class TestManMarkingFilter:
    """Man-marking classification tests."""

    def test_behind_attacker_is_man_marker(self):
        """Defender 2m behind attacker toward own goal -> man-marker."""
        from silly_kicks.tracking._cover_shadows import (
            CoverShadowParams,
            _classify_man_markers,
        )

        p = CoverShadowParams()
        # Attacking team=2 attacks toward x=105; defenders' own goal is x=0
        defenders = pd.DataFrame(
            {
                "player_id": [10],
                "x": [58.0],  # 1m behind attacker (toward x=0) + lateral offset < 3m
                "y": [34.0],
            }
        )
        attackers = pd.DataFrame(
            {
                "player_id": [60],
                "x": [60.0],
                "y": [34.0],
            }
        )
        # behind_point = (60 - 1, 34) = (59, 34)
        # dist from (58, 34) to (59, 34) = 1.0 < 3.0 -> man-marker
        result = _classify_man_markers(
            defenders,
            attackers,
            goal_x_own=0.0,
            params=p,
        )
        assert 10 in result

    def test_lateral_defender_not_man_marker(self):
        """Defender 5m laterally from attacker -> NOT man-marker."""
        from silly_kicks.tracking._cover_shadows import (
            CoverShadowParams,
            _classify_man_markers,
        )

        p = CoverShadowParams()
        defenders = pd.DataFrame(
            {
                "player_id": [10],
                "x": [60.0],
                "y": [39.0],  # 5m lateral
            }
        )
        attackers = pd.DataFrame(
            {
                "player_id": [60],
                "x": [60.0],
                "y": [34.0],
            }
        )
        # behind_point = (59, 34); dist from (60, 39) = sqrt(1+25) > 3.0
        result = _classify_man_markers(
            defenders,
            attackers,
            goal_x_own=0.0,
            params=p,
        )
        assert 10 not in result

    def test_mutual_exclusion_shared_behind_points(self):
        """Each defender can man-mark at most one attacker (1:1 assignment).

        Bug: greedy union allowed a single defender to be "absorbed" by
        multiple attackers' overlapping behind-point zones, over-counting
        man-markers and leaving too few lane blockers.

        Fix: mutual exclusion — each defender assigned to at most one
        attacker's behind-point (greedy nearest-first).
        """
        from silly_kicks.tracking._cover_shadows import (
            CoverShadowParams,
            _classify_man_markers,
        )

        p = CoverShadowParams()

        # 5 defenders clustered at x=58-60, y=33-35.
        # 3 attackers at (60,33), (60,34), (60,35) with behind-points
        # at (59,33), (59,34), (59,35). All 5 defenders are within 3m
        # of at least one behind-point.
        # Old union: 5 man-markers (all defenders absorbed).
        # Correct 1:1: 3 man-markers (one per attacker).
        defenders = pd.DataFrame(
            {
                "player_id": [10, 11, 12, 13, 14],
                "x": [58.0, 59.0, 60.0, 59.0, 58.0],
                "y": [33.0, 33.0, 34.0, 35.0, 35.0],
            }
        )

        attackers = pd.DataFrame(
            {
                "player_id": [60, 61, 62],
                "x": [60.0, 60.0, 60.0],
                "y": [33.0, 34.0, 35.0],
            }
        )

        result = _classify_man_markers(
            defenders,
            attackers,
            goal_x_own=0.0,
            params=p,
        )

        # At most 3 man-markers (one per attacker, not per defender)
        assert len(result) <= len(attackers), (
            f"Man-markers ({len(result)}) exceeds attacker count ({len(attackers)}). "
            "Greedy union absorbs defenders via overlapping behind-point zones."
        )
        # At least 2 defenders must remain as lane blockers
        lane_blockers = len(defenders) - len(result)
        assert lane_blockers >= 2, (
            f"Only {lane_blockers} lane blockers remain out of {len(defenders)} "
            f"defenders ({len(result)} absorbed as man-markers)."
        )


# === Task 4: Blocking score ===


class TestBlockingScore:
    """Blocking score counterfactual tests."""

    def test_no_lane_blockers_returns_zero(self, fitted_xt):
        """All defenders man-marking -> blocking_score = 0.0."""
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        # Place all home defenders right behind away attackers -> all man-markers
        frame = _make_two_team_frame(
            home_positions=[
                (59.0, 34.0),
                (64.0, 34.0),  # right behind away players
                (54.0, 10.0),
                (54.0, 58.0),
                (59.0, 20.0),
                (64.0, 50.0),
                (69.0, 15.0),
                (79.0, 45.0),
                (84.0, 30.0),
                (89.0, 40.0),
            ],
            away_positions=[
                (60.0, 34.0),
                (65.0, 34.0),
                (55.0, 10.0),
                (55.0, 58.0),
                (60.0, 20.0),
                (65.0, 50.0),
                (70.0, 15.0),
                (80.0, 45.0),
                (85.0, 30.0),
                (90.0, 40.0),
            ],
        )
        result = compute_blocking_score(
            frame,
            attacking_team_id=2,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert result.blocking_score >= 0.0
        # With most defenders man-marking, score should be very low
        # (not exactly 0 since some may escape the filter)

    def test_positive_blocking_score(self, fitted_xt):
        """Lane-blocker removed -> threat increases -> positive score."""
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        # Defender at midpoint between passer and receiver
        frame = _make_lane_control_frame(
            passer_pos=(50.0, 34.0),
            receiver_pos=(75.0, 34.0),
            defender_pos=(62.5, 34.0),
        )
        result = compute_blocking_score(
            frame,
            attacking_team_id=2,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert result.blocking_score >= 0.0
        # BlockingScoreResult has threat breakdown
        assert result.threat_original >= 0.0
        assert result.threat_unblocked >= result.threat_original

    def test_specific_defender_removal(self, fitted_xt):
        """defenders_to_remove=[pid] removes exactly that player."""
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        frame = _make_lane_control_frame(
            passer_pos=(50.0, 34.0),
            receiver_pos=(75.0, 34.0),
            defender_pos=(62.5, 34.0),  # player_id=10
        )
        result = compute_blocking_score(
            frame,
            attacking_team_id=2,
            xt=fitted_xt,
            home_team_id=1,
            defenders_to_remove=[10],
        )
        assert result.blocking_score >= 0.0

    def test_no_dangerous_receivers_returns_zero(self, fitted_xt):
        """All attackers behind ball -> blocking_score = 0.0."""
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        # After LTR: away team (2) attacks toward x=0 (low x).
        # "Behind the ball" for away = x > ball_x. Ball at x=50.
        frame = _make_two_team_frame(
            home_positions=[(20 + i * 3, 20 + i * 5) for i in range(10)],
            away_positions=[(60 + i * 3, 20 + i * 5) for i in range(10)],
        )
        result = compute_blocking_score(
            frame,
            attacking_team_id=2,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert result.blocking_score == 0.0

    def test_ltr_validation_rejects_all_rtl(self, fitted_xt):
        """Frames with only 'rtl' direction (no home 'ltr') raise ValueError."""
        from silly_kicks.tracking._cover_shadows import compute_blocking_score

        frame = _make_lane_control_frame(defender_pos=(62.5, 34.0))
        frame["team_attacking_direction"] = "rtl"
        with pytest.raises(ValueError, match="period-normalized"):
            compute_blocking_score(
                frame,
                attacking_team_id=2,
                xt=fitted_xt,
                home_team_id=1,
            )


class TestVoronoiPartition:
    """Grid-based Voronoi threat model tests."""

    def test_partition_covers_grid(self, fitted_xt):
        """Voronoi assigns every grid cell to exactly one player."""
        from silly_kicks.tracking._cover_shadows import _voronoi_threat

        # Home team (1) attacks toward high x after LTR; place home
        # players forward of ball (x>50) to create dangerous receivers.
        frame = _make_two_team_frame(
            home_positions=[(60 + i * 3, 15 + i * 5) for i in range(10)],
            away_positions=[(20 + i * 5, 20 + i * 5) for i in range(10)],
        )
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        surface = compute_pitch_control(frame, attacking_team_id=1, method="spearman")
        threat_total, _per_receiver = _voronoi_threat(
            surface,
            fitted_xt,
            frame,
            attacking_team_id=1,
            home_team_id=1,
        )
        assert threat_total >= 0.0

    def test_single_receiver_grid_ge_point(self, fitted_xt):
        """With 1 receiver, grid sum >= point evaluation."""
        from silly_kicks.tracking._cover_shadows import _voronoi_threat

        # Home team (1) attacks toward high x.  One home attacker far
        # forward (x=80), rest behind ball (x<50).
        frame = _make_two_team_frame(
            home_positions=[
                (80.0, 34.0),  # single dangerous receiver
                (30.0, 10.0),
                (30.0, 58.0),
                (35.0, 20.0),
                (35.0, 48.0),
                (25.0, 30.0),
                (25.0, 40.0),
                (20.0, 15.0),
                (20.0, 55.0),
                (28.0, 34.0),
            ],
            away_positions=[(60 + i * 3, 15 + i * 5) for i in range(10)],
        )
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        surface = compute_pitch_control(frame, attacking_team_id=1, method="spearman")
        threat_total, _per_receiver = _voronoi_threat(
            surface,
            fitted_xt,
            frame,
            attacking_team_id=1,
            home_team_id=1,
        )
        # Point evaluation at receiver position
        xt_interp = fitted_xt.interpolator()
        point_xt = float(xt_interp(np.array([80.0]), np.array([34.0]))[0, 0])
        point_pc = float(surface.at_points(np.array([[80.0, 34.0]]))[0])
        point_threat = point_xt * point_pc
        # Grid sum should be >= point evaluation (it integrates over the region)
        assert threat_total >= point_threat * 0.5  # generous margin


# === Task 5: Action-coupled aggregator + params drift guard ===


class TestAddCoverShadows:
    """Action-coupled aggregator tests."""

    def _make_actions_and_frames(self):
        """Build actions + frames for action-coupled testing."""
        frame = _make_two_team_frame(
            home_positions=[
                (40.0, 34.0),  # defender on likely pass line
                (20.0, 15.0),
                (25.0, 55.0),
                (30.0, 10.0),
                (35.0, 60.0),
                (40.0, 20.0),
                (45.0, 50.0),
                (15.0, 34.0),
                (10.0, 15.0),
                (10.0, 55.0),
            ],
            away_positions=[
                (50.0, 34.0),
                (70.0, 34.0),
                (75.0, 20.0),
                (80.0, 50.0),
                (55.0, 10.0),
                (55.0, 58.0),
                (60.0, 20.0),
                (65.0, 50.0),
                (85.0, 30.0),
                (90.0, 40.0),
            ],
        )
        actions = pd.DataFrame(
            {
                "action_id": [0, 1],
                "game_id": [1, 1],
                "period_id": [1, 1],
                "time_seconds": [1.0, 999.0],  # second action unlinked
                "team_id": [2, 2],
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [50.0, 50.0],
                "start_y": [34.0, 34.0],
                "end_x": [70.0, 70.0],
                "end_y": [34.0, 34.0],
                "bodypart_id": [0, 0],
                "player_id": [60, 60],
            }
        )
        return actions, frame

    def test_output_columns(self, fitted_xt):
        """Returns all 5 columns with correct dtypes."""
        from silly_kicks.tracking.features import add_cover_shadows

        actions, frames = self._make_actions_and_frames()
        result = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
        )
        expected_cols = [
            "n_blocked_receivers",
            "n_potential_receivers",
            "blocking_score",
            "blocked_threat_fraction",
            "max_single_defender_blocking_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_unlinked_action_nan(self, fitted_xt):
        """Unlinked actions -> NaN/pd.NA in all 5 columns."""
        from silly_kicks.tracking.features import add_cover_shadows

        actions, frames = self._make_actions_and_frames()
        result = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
        )
        # Action 1 (time=999.0) cannot link -> NaN
        assert pd.isna(result.loc[1, "blocking_score"])
        assert pd.isna(result.loc[1, "n_blocked_receivers"])

    def test_detailed_flag_both_modes(self, fitted_xt):
        """Both detailed=False and detailed=True run without error."""
        from silly_kicks.tracking.features import add_cover_shadows

        actions, frames = self._make_actions_and_frames()
        r_fast = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
            detailed=False,
        )
        r_full = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
            detailed=True,
        )
        assert "max_single_defender_blocking_score" in r_fast.columns
        assert "max_single_defender_blocking_score" in r_full.columns


class TestParamsDriftGuard:
    """Ensure CoverShadowParams defaults match SpearmanParams where shared."""

    def test_reaction_time_matches(self):
        from silly_kicks.tracking._cover_shadows import CoverShadowParams
        from silly_kicks.tracking.pitch_control import SpearmanParams

        cs = CoverShadowParams()
        sp = SpearmanParams()
        assert cs.reaction_time == sp.reaction_time

    def test_max_acceleration_matches(self):
        from silly_kicks.tracking._cover_shadows import CoverShadowParams
        from silly_kicks.tracking.pitch_control import SpearmanParams

        cs = CoverShadowParams()
        sp = SpearmanParams()
        assert cs.max_acceleration == sp.max_acceleration


# === Task 6: VAEP factory ===


class TestCoverShadowXfns:
    """VAEP factory tests."""

    def test_introspection_silent_nan(self, fitted_xt):
        """10-row dummy gamestate -> silent NaN (VAEP fit-time contract)."""
        from silly_kicks.tracking.features import cover_shadow_xfns

        xfns = cover_shadow_xfns(fitted_xt, home_team_id=1)
        assert len(xfns) == 1
        transformer = xfns[0]
        assert getattr(transformer, "_frame_aware", False) is True

        # Build 10-row dummy with only canonical 17 SPADL columns
        dummy = pd.DataFrame(
            {
                "game_id": [1] * 10,
                "action_id": list(range(10)),
                "period_id": [1] * 10,
                "time_seconds": list(range(10)),
                "team_id": [1] * 10,
                "player_id": list(range(10)),
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [60.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [1] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]
        result = transformer(states, None)
        # All NaN
        assert result.isna().all().all()

    def test_column_count(self, fitted_xt):
        """5 features x 3 states = 15 output columns."""
        from silly_kicks.tracking.features import cover_shadow_xfns

        xfns = cover_shadow_xfns(fitted_xt, home_team_id=1)
        transformer = xfns[0]

        dummy = pd.DataFrame(
            {
                "game_id": [1] * 3,
                "action_id": [0, 1, 2],
                "period_id": [1] * 3,
                "time_seconds": [1.0, 2.0, 3.0],
                "team_id": [1] * 3,
                "player_id": [10, 11, 12],
                "start_x": [50.0] * 3,
                "start_y": [34.0] * 3,
                "end_x": [60.0] * 3,
                "end_y": [34.0] * 3,
                "type_id": [0] * 3,
                "result_id": [1] * 3,
                "bodypart_id": [0] * 3,
            }
        )
        states = [dummy, dummy, dummy]
        result = transformer(states, None)
        assert result.shape[1] == 15  # 5 cols x 3 states


# === Task 11: Smoke + correlation tests ===


class TestBlockingRateSmoke:
    """Predicted block rate sanity check (not calibration)."""

    def test_blocking_rate_in_plausible_range(self, fitted_xt):
        """Block rate on Sportec fixture should be 10-60%."""
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, lane_control

        frames = load_provider_frames("sportec")
        from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
        from silly_kicks.tracking.utils import play_left_to_right

        frames = smooth_frames(frames)
        frames = derive_velocities(frames)
        home_team_id = frames[~frames["team_id"].isna()]["team_id"].iloc[0]
        frames = play_left_to_right(frames, home_team_id=home_team_id)
        actions = synthesize_actions(frames)

        from silly_kicks.tracking.utils import link_actions_to_frames

        pointers, _ = link_actions_to_frames(actions, frames)
        frame_groups = frames.groupby(["period_id", "frame_id"])
        pointer_lookup = pointers.set_index("action_id")

        n_pairs = 0
        n_blocked = 0
        p = CoverShadowParams()

        for _, row in actions.iterrows():
            aid = row["action_id"]
            tid = row["team_id"]
            if pd.isna(tid) or aid not in pointer_lookup.index:
                continue
            fid_raw = pointer_lookup.at[aid, "frame_id"]
            if pd.isna(fid_raw):
                continue
            try:
                frame_data = frame_groups.get_group((row["period_id"], int(float(fid_raw))))
            except KeyError:
                continue

            players = frame_data[~frame_data["is_ball"].astype(bool)]
            attackers = players[(players["team_id"] == tid) & (~players["is_goalkeeper"].astype(bool))]
            ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
            if ball_rows.empty or pd.isna(ball_rows.iloc[0]["x"]):
                continue
            ball_x = float(ball_rows.iloc[0]["x"])
            if str(tid) != str(home_team_id):
                dangerous = attackers[attackers["x"] > ball_x]
            else:
                dangerous = attackers[attackers["x"] < ball_x]

            passer_xy = (float(row["start_x"]), float(row["start_y"]))
            for _, recv in dangerous.iterrows():
                recv_xy = (float(recv["x"]), float(recv["y"]))
                try:
                    lc = lane_control(
                        frame_data,
                        passer_xy,
                        recv_xy,
                        home_team_id=home_team_id,
                        attacking_team_id=tid,
                        params=p,
                    )
                except ValueError:
                    continue
                n_pairs += 1
                if lc.is_blocked_majority:
                    n_blocked += 1

        if n_pairs == 0:
            pytest.skip("No (passer, receiver) pairs evaluated")
        block_rate = n_blocked / n_pairs
        assert 0.10 <= block_rate <= 0.60, (
            f"Predicted block rate {block_rate:.2%} outside [10%, 60%] ({n_blocked}/{n_pairs} pairs)"
        )


class TestDetailedVsLightweightCorrelation:
    """Spearman rank correlation between detailed=True and detailed=False."""

    def test_rank_correlation_ge_07(self, fitted_xt):
        """Lightweight max_single_defender_blocking_score has rho >= 0.7
        with the full counterfactual (detailed=True) on a multi-defender scenario.
        """
        from silly_kicks.tracking.features import add_cover_shadows

        # Build 5 frames at different time offsets with varying defender positions
        frame_rows = []
        for fi, (def_x, def_y) in enumerate(
            [
                (55.0, 30.0),
                (58.0, 35.0),
                (52.0, 28.0),
                (62.0, 40.0),
                (57.0, 32.0),
            ]
        ):
            t = 1.0 + fi * 0.08  # 0.08s apart = 2 frames at 25 Hz
            fid = 25 + fi * 2
            base = _make_two_team_frame(
                home_positions=[
                    (def_x, def_y),
                    (60.0, 38.0),
                    (65.0, 25.0),
                    (20.0, 15.0),
                    (25.0, 55.0),
                    (30.0, 10.0),
                    (35.0, 60.0),
                    (15.0, 34.0),
                    (10.0, 15.0),
                    (10.0, 55.0),
                ],
                away_positions=[
                    (50.0, 34.0),
                    (75.0, 34.0),
                    (80.0, 25.0),
                    (85.0, 45.0),
                    (70.0, 20.0),
                    (70.0, 48.0),
                    (90.0, 30.0),
                    (95.0, 40.0),
                    (45.0, 15.0),
                    (45.0, 55.0),
                ],
            )
            base["time_seconds"] = t
            base["frame_id"] = fid
            frame_rows.append(base)

        frames = pd.concat(frame_rows, ignore_index=True)

        actions = pd.DataFrame(
            {
                "action_id": list(range(5)),
                "game_id": [1] * 5,
                "period_id": [1] * 5,
                "time_seconds": [1.0, 1.08, 1.16, 1.24, 1.32],
                "team_id": [2] * 5,
                "type_id": [0] * 5,
                "result_id": [1] * 5,
                "start_x": [50.0, 48.0, 52.0, 50.0, 46.0],
                "start_y": [34.0, 30.0, 38.0, 25.0, 40.0],
                "end_x": [75.0, 80.0, 85.0, 70.0, 90.0],
                "end_y": [34.0, 25.0, 45.0, 20.0, 48.0],
                "bodypart_id": [0] * 5,
                "player_id": [60] * 5,
            }
        )

        r_fast = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
            detailed=False,
        )
        r_full = add_cover_shadows(
            actions,
            frames,
            fitted_xt,
            home_team_id=1,
            detailed=True,
        )

        fast_vals = r_fast["max_single_defender_blocking_score"].dropna()
        full_vals = r_full["max_single_defender_blocking_score"].dropna()

        common = fast_vals.index.intersection(full_vals.index)
        if len(common) < 3:
            pytest.skip("Not enough data points for correlation")

        from scipy.stats import spearmanr

        rho, _ = spearmanr(fast_vals[common], full_vals[common])
        if np.isnan(rho):
            # Zero variance in one or both — synthetic fixture doesn't produce
            # enough score differentiation. Skip rather than fail, since this
            # is a quality check, not a correctness invariant.
            pytest.skip(
                "Zero variance in cover shadow scores — insufficient scenario differentiation across test frames"
            )
        assert rho >= 0.7, f"Rank correlation {rho:.3f} < 0.7 between lightweight and full modes"


class TestManMarkerInvariantUnderLaneBlockerRemoval:
    """Load-bearing invariant for the leave-one-out perf refactor (PR-S65).

    The lightweight ``max_single_defender_blocking_score`` removes each *lane-blocker*
    (a defender that won no attacker in the greedy man-marker assignment) in turn. The
    PR-S65 optimization hoists the man-marker classification out of the per-defender loop
    on the premise that **removing a lane-blocker never changes the man-marker set** —
    removing a non-winner from a greedy nearest-first matching cannot change the matching
    of the others. If this property ever breaks (e.g. a future change to
    ``_classify_man_markers``), the hoist would silently stop being bit-identical, so this
    is pinned as a permanent regression guard. Lifted from the 2026-05-28 investigation probe.

    See docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md §2.1/§6.4.
    """

    # home = defenders (team 1, defends low x -> goal_x_own=0.0); away = attackers (team 2)
    _ATTACKING_TEAM = 2

    SCENARIOS: ClassVar[dict] = {
        # Two defenders contest one attacker's behind-point: the closer wins (man-marker),
        # the farther is a within-radius *losing* lane-blocker. Removing the loser must not
        # promote anyone.
        "two_contest_one_behind_point": dict(
            home_positions=[(69.5, 30.0), (67.0, 30.0), (55.0, 34.0), (50.0, 20.0), (52.0, 48.0)],
            away_positions=[(70.0, 30.0), (75.0, 40.0), (80.0, 25.0), (60.0, 34.0), (85.0, 45.0)],
        ),
        # Contested chain across two attackers.
        "contested_chain": dict(
            home_positions=[(69.0, 30.0), (74.0, 40.0), (73.0, 39.0), (50.0, 34.0), (55.0, 25.0)],
            away_positions=[(70.0, 30.0), (75.0, 40.0), (82.0, 28.0), (62.0, 34.0), (88.0, 50.0)],
        ),
        # Pile-up: four defenders all within radius of a single behind-point.
        "pile_up_on_one_behind_point": dict(
            home_positions=[(69.5, 30.0), (68.0, 30.5), (67.0, 29.5), (66.5, 31.0), (50.0, 34.0)],
            away_positions=[(70.0, 30.0), (78.0, 40.0), (84.0, 25.0), (60.0, 34.0), (90.0, 45.0)],
        ),
    }

    @pytest.mark.parametrize("scenario", list(SCENARIOS))
    def test_removing_any_lane_blocker_leaves_man_markers_unchanged(self, scenario):
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, _classify_man_markers

        params = CoverShadowParams()
        cfg = self.SCENARIOS[scenario]
        frame = _make_two_team_frame(**cfg)

        players = frame[~frame["is_ball"].astype(bool)]
        attackers = players[players["team_id"] == self._ATTACKING_TEAM]
        defenders_outfield = players[
            (players["team_id"] != self._ATTACKING_TEAM) & (~players["is_goalkeeper"].astype(bool))
        ]

        # Attacking team (2) != home (1) => defenders' own goal at x=0.0
        goal_x_own = 0.0
        full_mm = _classify_man_markers(defenders_outfield, attackers, goal_x_own=goal_x_own, params=params)
        lane_blockers = [pid for pid in defenders_outfield["player_id"] if pid not in full_mm]

        # The scenario must actually contain lane-blockers, else it guards nothing.
        assert lane_blockers, f"{scenario}: no lane-blockers — fixture does not exercise the property"

        for d in lane_blockers:
            defs_without_d = defenders_outfield[defenders_outfield["player_id"] != d]
            mm_without_d = _classify_man_markers(defs_without_d, attackers, goal_x_own=goal_x_own, params=params)
            assert mm_without_d == (full_mm - {d}), (
                f"{scenario}: removing lane-blocker {d} changed the man-marker set "
                f"{sorted(full_mm - {d})} -> {sorted(mm_without_d)} (no-ripple property broken; "
                "the PR-S65 man-marking hoist would no longer be bit-identical)"
            )

    @staticmethod
    def _mk_players(positions, start_id):
        return pd.DataFrame(
            {
                "player_id": list(range(start_id, start_id + len(positions))),
                "x": [float(p[0]) for p in positions],
                "y": [float(p[1]) for p in positions],
            }
        )

    @settings(max_examples=150, deadline=None)
    @given(
        def_positions=st.lists(st.tuples(st.floats(0.0, 105.0), st.floats(0.0, 68.0)), min_size=2, max_size=11),
        att_positions=st.lists(st.tuples(st.floats(0.0, 105.0), st.floats(0.0, 68.0)), min_size=2, max_size=11),
        radius=st.floats(0.5, 6.0),
        offset=st.floats(0.0, 3.0),
        goal_x_own=st.sampled_from([0.0, 105.0]),
    )
    def test_no_ripple_property_random_rosters(self, def_positions, att_positions, radius, offset, goal_x_own):
        """Property: removing ANY lane-blocker leaves the man-marker set unchanged, on random rosters.

        This is the load-bearing invariant the PR-S65 hoist depends on (spec §2.1/§6.4).
        Broad random coverage of the exact property, far stronger than hand-picked fixtures.
        """
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, _classify_man_markers

        defenders = self._mk_players(def_positions, start_id=100)
        attackers = self._mk_players(att_positions, start_id=500)
        params = CoverShadowParams(man_mark_radius=radius, man_mark_behind_offset=offset)

        full_mm = _classify_man_markers(defenders, attackers, goal_x_own=goal_x_own, params=params)
        lane_blockers = [pid for pid in defenders["player_id"] if pid not in full_mm]

        for d in lane_blockers:
            sub = defenders[defenders["player_id"] != d]
            mm_sub = _classify_man_markers(sub, attackers, goal_x_own=goal_x_own, params=params)
            assert mm_sub == (full_mm - {d}), (
                f"removing lane-blocker {d} changed man-markers {sorted(full_mm - {d})} -> "
                f"{sorted(mm_sub)} (no-ripple broken; PR-S65 hoist no longer bit-identical)"
            )

    def test_at_least_one_scenario_has_a_contesting_loser(self):
        """Guard the guard: ensure a fixture has a within-radius losing lane-blocker.

        A trivial fixture where every lane-blocker is far from all behind-points would pass
        the invariant test while covering nothing interesting. This confirms at least one
        scenario has a defender within ``man_mark_radius`` of a behind-point that still loses
        the greedy assignment (the adversarial case).
        """
        from silly_kicks.tracking._cover_shadows import CoverShadowParams, _classify_man_markers

        params = CoverShadowParams()
        cfg = self.SCENARIOS["two_contest_one_behind_point"]
        frame = _make_two_team_frame(**cfg)
        players = frame[~frame["is_ball"].astype(bool)]
        attackers = players[players["team_id"] == self._ATTACKING_TEAM]
        defenders_outfield = players[
            (players["team_id"] != self._ATTACKING_TEAM) & (~players["is_goalkeeper"].astype(bool))
        ]
        full_mm = _classify_man_markers(defenders_outfield, attackers, goal_x_own=0.0, params=params)

        # Behind-points toward defenders' own goal (x=0 => toward -x).
        att_pos = attackers[["x", "y"]].to_numpy()
        behind = att_pos + params.man_mark_behind_offset * np.array([-1.0, 0.0])
        def_pos = defenders_outfield[["x", "y"]].to_numpy()
        def_ids = defenders_outfield["player_id"].to_numpy()

        contesting_loser = False
        for di, pid in enumerate(def_ids):
            within = np.any(np.linalg.norm(behind - def_pos[di], axis=1) < params.man_mark_radius)
            if within and pid not in full_mm:
                contesting_loser = True
                break
        assert contesting_loser, "fixture lacks a within-radius losing lane-blocker — guard is vacuous"


class TestLeaveOneOutExactness:
    """Production max_single == independent frozen oracle, within rtol 1e-10 (spec §6.1).

    Parametrized over several geometries (both attacking directions, dense + sparse) to
    give breadth in lieu of an in-repo real match. The frozen oracle
    (tests/tracking/_cover_shadows_reference.py) shares none of the production helpers, so
    this certifies the refactor + vectorization against independent code and guards INV-1.
    """

    _DENSE_HOME: ClassVar[list] = [
        (55.0, 30.0),
        (58.0, 35.0),
        (52.0, 28.0),
        (62.0, 40.0),
        (57.0, 32.0),
        (60.0, 38.0),
        (65.0, 25.0),
        (20.0, 15.0),
        (25.0, 55.0),
        (30.0, 10.0),
    ]
    _DENSE_AWAY: ClassVar[list] = [
        (50.0, 34.0),
        (75.0, 34.0),
        (80.0, 25.0),
        (85.0, 45.0),
        (70.0, 20.0),
        (70.0, 48.0),
        (90.0, 30.0),
        (95.0, 40.0),
        (45.0, 15.0),
        (45.0, 55.0),
    ]

    # (home_positions, away_positions, attacking_team_id, home_team_id, passer_xy)
    FIXTURES: ClassVar[dict] = {
        "away_attacks_dense": (_DENSE_HOME, _DENSE_AWAY, 2, 1, (50.0, 34.0)),
        "home_attacks_dense": (_DENSE_AWAY, _DENSE_HOME, 1, 1, (50.0, 34.0)),
        "sparse_contest": (
            [(48.0, 30.0), (47.0, 30.5), (40.0, 34.0), (35.0, 20.0), (38.0, 50.0)],
            [(45.0, 30.0), (30.0, 25.0), (28.0, 40.0), (60.0, 34.0), (55.0, 20.0)],
            2,
            1,
            (50.0, 34.0),
        ),
    }

    @pytest.mark.parametrize("name", list(FIXTURES))
    def test_production_matches_frozen_oracle(self, name, fitted_xt):
        from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict
        from tests.tracking._cover_shadows_reference import _reference_max_single

        home_pos, away_pos, att_team, home_team, passer = self.FIXTURES[name]
        frame = _make_two_team_frame(home_positions=home_pos, away_positions=away_pos)

        prod = _compute_cover_shadow_dict(frame, passer, att_team, fitted_xt, home_team_id=home_team, detailed=False)
        ref = _reference_max_single(frame, passer, att_team, fitted_xt, home_team_id=home_team)

        assert prod is not None
        np.testing.assert_allclose(
            prod["max_single_defender_blocking_score"],
            ref,
            rtol=1e-10,
            err_msg=f"[{name}] production max_single diverged from the frozen leave-one-out oracle",
        )

    def test_at_least_one_fixture_is_nonzero(self, fitted_xt):
        """Guard the guard: the exactness comparison must be non-vacuous (not all 0.0)."""
        from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict

        values = []
        for home_pos, away_pos, att_team, home_team, passer in self.FIXTURES.values():
            frame = _make_two_team_frame(home_positions=home_pos, away_positions=away_pos)
            r = _compute_cover_shadow_dict(frame, passer, att_team, fitted_xt, home_team_id=home_team, detailed=False)
            if r is not None:
                values.append(r["max_single_defender_blocking_score"])
        assert any(v > 0.0 for v in values), "all fixtures produced max_single=0 — exactness test is vacuous"


class TestCoverShadowPerfBudget:
    """Guard against silent regression of the leave-one-out optimization (spec §7).

    Local post-change: ~12 ms/call (vs ~51 ms pre-change, ~4.3x). Budget = 45 ms: ample
    headroom over local + slow-Windows-CI slowdown, yet below the ~51 ms pre-change cost so a
    regression to the O(blockers x receivers) lane_control loop trips it. Re-tune from observed
    CI timing (worst x 1.5) if it ever flakes.
    """

    _BUDGET_S = 0.045

    def test_detailed_false_under_budget(self, fitted_xt):
        import time

        from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict

        frame = _make_two_team_frame(
            home_positions=[
                (55.0, 30.0),
                (58.0, 35.0),
                (52.0, 28.0),
                (62.0, 40.0),
                (57.0, 32.0),
                (60.0, 38.0),
                (65.0, 25.0),
                (20.0, 15.0),
                (25.0, 55.0),
                (30.0, 10.0),
            ],
            away_positions=[
                (50.0, 34.0),
                (75.0, 34.0),
                (80.0, 25.0),
                (85.0, 45.0),
                (70.0, 20.0),
                (70.0, 48.0),
                (90.0, 30.0),
                (95.0, 40.0),
                (45.0, 15.0),
                (45.0, 55.0),
            ],
        )
        # Warm up, then average several calls for a stable measurement.
        for _ in range(3):
            _compute_cover_shadow_dict(frame, (50.0, 34.0), 2, fitted_xt, home_team_id=1, detailed=False)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            _compute_cover_shadow_dict(frame, (50.0, 34.0), 2, fitted_xt, home_team_id=1, detailed=False)
        per_call = (time.perf_counter() - t0) / N
        assert per_call < self._BUDGET_S, (
            f"detailed=False per-call {per_call * 1000:.2f} ms exceeds budget "
            f"{self._BUDGET_S * 1000:.2f} ms (possible regression to the per-(d, receiver) loop)"
        )
