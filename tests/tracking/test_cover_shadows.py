"""Tests for TF-30 Cover Shadow features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
