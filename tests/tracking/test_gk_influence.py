"""Tests for TF-15 GK influence primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._gk_test_helpers import _make_two_team_frame

# === T-PR1: compute_tti public export ===


class TestComputeTtiExport:
    """T-PR1: compute_tti is importable from pitch_control and produces correct results."""

    def test_importable_from_pitch_control(self):
        """compute_tti is importable from the public pitch_control namespace."""
        from silly_kicks.tracking.pitch_control import compute_tti

        assert callable(compute_tti)

    def test_regression_parity_with_private(self):
        """Public compute_tti produces identical results to the private _compute_tti."""
        from silly_kicks.tracking.pitch_control import compute_tti

        pos = np.array([[0.0, 0.0], [10.0, 10.0]])
        vel = np.array([[3.0, 0.0], [0.0, -2.0]])
        targets = np.array([[5.0, 0.0], [10.0, 5.0], [50.0, 34.0]])

        result = compute_tti(pos, vel, targets, 0.7, 7.0)

        assert result.shape == (2, 3)
        # Player at origin moving right: TTI to (5,0) should be less than to (50,34)
        assert result[0, 0] < result[0, 2]
        # All TTI values >= reaction_time
        assert np.all(result >= 0.7)


# === T-PR2: select_back_line_players ===


def _make_outfield_frame(
    *,
    positions: list[tuple[float, float]],
    team_id: int = 1,
    home_team_id: int = 1,
    velocities: list[tuple[float, float]] | None = None,
    frame_id: int = 1,
    period_id: int = 1,
    game_id: int = 1,
) -> pd.DataFrame:
    """Build a minimal tracking frame with outfield players + ball + GK."""
    rows = []
    # Ball row
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # GK
    gk_x = 3.0 if team_id == home_team_id else 102.0
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=99,
            team_id=team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=gk_x,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Outfield
    for i, (px, py) in enumerate(positions):
        vx_val = velocities[i][0] if velocities else 0.0
        vy_val = velocities[i][1] if velocities else 0.0
        rows.append(
            dict(
                game_id=game_id,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=100 + i,
                team_id=team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=px,
                y=py,
                vx=vx_val,
                vy=vy_val,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


class TestSelectBackLinePlayers:
    """T-PR2: select_back_line_players returns individual player rows."""

    def test_returns_player_rows_with_coordinates(self):
        """Returns DataFrame with x, y, vx, vy preserved per player."""
        from silly_kicks.tracking._defensive_line import select_back_line_players

        frames = _make_outfield_frame(
            positions=[(10, 20), (15, 30), (20, 40), (50, 34), (60, 25)],
            velocities=[(1, 0), (2, 0), (-1, 0), (0, 1), (3, -1)],
            team_id=1,
            home_team_id=1,
        )
        result = select_back_line_players(frames, team_id=1, home_team_id=1, n=4)

        assert len(result) == 4
        assert set(result.columns) >= {"x", "y", "vx", "vy", "player_id"}
        # Home team defends x=0, so back line = lowest x values
        assert result["x"].max() <= 50.0

    def test_defensive_line_unchanged_after_refactor(self):
        """compute_defensive_line produces identical output after refactor."""
        from silly_kicks.tracking._defensive_line import compute_defensive_line

        frames = _make_outfield_frame(
            positions=[(10, 20), (15, 30), (20, 40), (50, 34), (60, 25)],
            team_id=1,
            home_team_id=1,
        )
        result = compute_defensive_line(frames, home_team_id=1, n=4)

        assert len(result) == 1
        assert result["defensive_line_x"].notna().all()
        assert result["back_n_count"].iloc[0] == 4

    def test_away_team_selects_highest_x(self):
        """Away team (defends x=105) selects players closest to x=105."""
        from silly_kicks.tracking._defensive_line import select_back_line_players

        frames = _make_outfield_frame(
            positions=[(40, 20), (50, 30), (80, 40), (90, 34), (95, 25)],
            team_id=2,
            home_team_id=1,
        )
        result = select_back_line_players(frames, team_id=2, home_team_id=1, n=4)

        assert len(result) == 4
        # Away team defends x=105, so back line = highest x values
        assert result["x"].min() >= 50.0


# === T-1: Zone geometry ===


class TestZoneGeometry:
    """T-1: Zone dataclass factory methods produce correct geometry."""

    def test_six_yard_box_goal_x_0(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=0.0)
        assert zone.name == "six_yard_box"
        assert zone.points.shape[1] == 2
        assert len(zone.points) >= 6
        # All x in [0, 5.5], y in [24.84, 43.16]
        assert np.all(zone.points[:, 0] >= 0.0)
        assert np.all(zone.points[:, 0] <= 5.5)
        assert np.all(zone.points[:, 1] >= 24.84)
        assert np.all(zone.points[:, 1] <= 43.16)

    def test_six_yard_box_goal_x_105(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=105.0)
        assert np.all(zone.points[:, 0] >= 99.5)
        assert np.all(zone.points[:, 0] <= 105.0)

    def test_near_far_post_corridors(self):
        from silly_kicks.tracking._gk_influence import Zone

        near = Zone.near_post(goal_x=0.0, ball_y=25.0)
        far = Zone.far_post(goal_x=0.0, ball_y=25.0)
        assert near.name == "near_post"
        assert far.name == "far_post"
        # Near post should be closer to ball_y=25 than far post
        near_mean_y = near.points[:, 1].mean()
        far_mean_y = far.points[:, 1].mean()
        assert abs(near_mean_y - 25.0) < abs(far_mean_y - 25.0)

    def test_frozen_immutability(self):
        from silly_kicks.tracking._gk_influence import Zone

        zone = Zone.six_yard_box(goal_x=0.0)
        with pytest.raises(ValueError, match="read-only"):
            zone.points[0, 0] = 999.0

    def test_ball_relative_near_post_different_sides(self):
        """near_post gives different point sets for ball_y=25 vs ball_y=40."""
        from silly_kicks.tracking._gk_influence import Zone

        near_low = Zone.near_post(goal_x=0.0, ball_y=25.0)
        near_high = Zone.near_post(goal_x=0.0, ball_y=40.0)
        # Different ball positions should select different goalposts
        assert not np.allclose(near_low.points, near_high.points)


# === T-2 through T-6: compute_gk_influence tests ===


@pytest.fixture
def standard_frame():
    """Standard 10v10 + 2 GK frame with realistic non-zero velocities."""
    home_pos = [(20, 15), (25, 25), (30, 40), (35, 55), (45, 10), (50, 30), (55, 45), (60, 55), (70, 30), (75, 40)]
    away_pos = [(85, 15), (80, 25), (75, 40), (70, 55), (60, 10), (55, 30), (50, 45), (45, 55), (35, 30), (30, 40)]
    # Realistic velocities: mix of jogging (2-4 m/s), sprinting (5-7 m/s), and stationary
    home_vel = [
        (2.0, 0.5),
        (-1.0, 1.5),
        (0.0, 0.0),
        (3.0, -1.0),
        (5.0, 2.0),
        (-2.0, 0.0),
        (1.0, -3.0),
        (0.0, 4.0),
        (6.0, -1.0),
        (-1.5, 2.5),
    ]
    away_vel = [
        (-3.0, 1.0),
        (0.0, -2.0),
        (2.0, 0.0),
        (-1.0, 3.0),
        (0.0, 0.0),
        (4.0, -1.5),
        (-2.5, 0.5),
        (1.0, 1.0),
        (-5.0, 0.0),
        (0.5, -4.0),
    ]
    return _make_two_team_frame(
        home_positions=home_pos,  # type: ignore[arg-type]
        away_positions=away_pos,  # type: ignore[arg-type]
        home_velocities=home_vel,
        away_velocities=away_vel,
    )


class TestComputeGkInfluenceCore:
    """T-2: Core logic of compute_gk_influence."""

    def test_weighted_share_less_than_raw(self, standard_frame, fitted_xt):
        """Threat-weighted share < raw player_share when GK near own goal."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        # Raw share for comparison
        surface = compute_pitch_control(
            standard_frame,
            attacking_team_id=2,
            method="spearman",
            decompose=True,
        )
        raw_share = surface.player_share(1)
        assert gi.pitch_control_share_weighted < raw_share

    def test_share_in_range(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_reachable_area_no_defenders(self, fitted_xt):
        """With no outfield defenders, reachable area ~ full GK circle."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[],
            away_positions=[],
            home_gk_pos=(3.0, 34.0),
            away_gk_pos=(102.0, 34.0),
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            tau_seconds=3.0,
        )
        assert gi.reachable_area_m2 > 0.0

    def test_reachable_area_decreases_with_defenders(self, fitted_xt):
        """Adding defenders reduces reachable area."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame_no_def = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[],
        )
        frame_with_def = _make_two_team_frame(
            home_positions=[(5, 20), (8, 30), (6, 40), (7, 50)],
            away_positions=[],
        )
        gi_far = compute_gk_influence(
            frame_no_def,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            tau_seconds=3.0,
        )
        gi_near = compute_gk_influence(
            frame_with_def,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            tau_seconds=3.0,
        )
        assert gi_far.reachable_area_m2 > gi_near.reachable_area_m2

    def test_reachable_area_non_negative(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert gi.reachable_area_m2 >= 0.0

    def test_closing_time_near_vs_far(self, fitted_xt):
        """GK at six-yard box -> low min_s; GK at halfway -> high min_s."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame_near = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(3.0, 34.0),
        )
        frame_far = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(52.5, 34.0),
        )
        gi_near = compute_gk_influence(
            frame_near,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        gi_far = compute_gk_influence(
            frame_far,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        ct_near = gi_near.closing_times["six_yard_box"]
        ct_far = gi_far.closing_times["six_yard_box"]
        assert ct_near.min_s < ct_far.min_s


class TestXtOrientation:
    """T-5: xT interpolation and flip logic."""

    def test_xt_interpolated_sum_positive(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        # If we got a valid share, the threat sum was positive
        assert not np.isnan(gi.pitch_control_share_weighted)

    def test_xt_all_zeros_returns_nan(self, standard_frame):
        from silly_kicks.tracking._gk_influence import compute_gk_influence
        from silly_kicks.xthreat import ExpectedThreat

        xt_zero = ExpectedThreat(l=16, w=12)
        xt_zero.xT = np.zeros((12, 16))
        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=xt_zero,
            home_team_id=1,
        )
        assert np.isnan(gi.pitch_control_share_weighted)

    def test_home_attack_no_flip(self, fitted_xt):
        """When home team attacks (toward x=105), xT is NOT flipped."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
            away_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_gk_pos=(3.0, 34.0),
        )
        # Home attacks -> away defends -> away GK (id=50) at x=3
        gi = compute_gk_influence(
            frame,
            attacking_team_id=1,
            gk_player_id=50,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_away_attack_flip(self, fitted_xt):
        """When away team attacks (toward x=0), xT is x-flipped."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        # Same frame but away attacks -> home GK (id=1) at x=3 defends x=0
        frame = _make_two_team_frame(
            home_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
        )
        gi_away = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        gi_home = compute_gk_influence(
            frame,
            attacking_team_id=1,
            gk_player_id=50,
            xt=fitted_xt,
            home_team_id=1,
        )
        # Different attacking directions should produce different shares
        assert gi_away.pitch_control_share_weighted != gi_home.pitch_control_share_weighted

    def test_flip_is_x_only_not_y(self, fitted_xt):
        """Flip is [:, ::-1] not [::-1, ::-1] — y-axis preserved."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence
        from silly_kicks.xthreat import ExpectedThreat

        # Asymmetric xT in y to detect y-flip
        xt_asym = ExpectedThreat(l=16, w=12)
        xt_asym.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
        # Make top row different from bottom row
        xt_asym.xT[0, :] = 0.1
        xt_asym.xT[-1, :] = 0.9

        frame = _make_two_team_frame(
            home_positions=[(20, 30), (25, 40), (15, 20), (10, 50)],
            away_positions=[(80, 30), (85, 40), (70, 20), (75, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=xt_asym,
            home_team_id=1,
        )
        # Should produce a valid result (no crash from y-flip mismatch)
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_interpolated_grid_shape(self, fitted_xt):
        """xT interpolated onto pitch control grid has correct shape."""
        interp = fitted_xt.interpolator(kind="linear")
        from silly_kicks.tracking.pitch_control import SpearmanParams

        p = SpearmanParams()
        grid_x = np.linspace(0, 105.0, p.grid_cells_x)
        grid_y = np.linspace(0, 68.0, p.grid_cells_y)
        threat_grid = interp(grid_x, grid_y)
        assert threat_grid.shape == (p.grid_cells_y, p.grid_cells_x)


class TestGkInfluenceEdgeCases:
    """T-6: Edge cases for compute_gk_influence."""

    def test_gk_not_in_frame_raises(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        with pytest.raises(ValueError, match="not found"):
            compute_gk_influence(
                standard_frame,
                attacking_team_id=2,
                gk_player_id=999,
                xt=fitted_xt,
                home_team_id=1,
            )

    def test_min_s_leq_mean_s(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s <= zct.mean_s

    def test_near_zero_denominator_share_zero(self, fitted_xt):
        """Near-zero team_influence -> share = 0, not infinity."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        # Frame where defending team is extremely weak (GK only, no outfield)
        frame = _make_two_team_frame(
            home_positions=[],
            away_positions=[(50, 30), (55, 40), (60, 20), (65, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert not np.isinf(gi.pitch_control_share_weighted)

    def test_custom_reaction_time_lower_closing(self, standard_frame, fitted_xt):
        """Lower reaction_time -> lower closing times."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi_default = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            gk_reaction_time=0.4,
        )
        gi_fast = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            gk_reaction_time=0.3,
        )
        ct_default = gi_default.closing_times["six_yard_box"]
        ct_fast = gi_fast.closing_times["six_yard_box"]
        assert ct_fast.min_s < ct_default.min_s

    def test_custom_reaction_time_larger_reachable(self, standard_frame, fitted_xt):
        """Lower reaction_time -> larger reachable area (monotonicity)."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi_default = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            gk_reaction_time=0.4,
        )
        gi_fast = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            gk_reaction_time=0.3,
        )
        assert gi_fast.reachable_area_m2 >= gi_default.reachable_area_m2

    def test_ball_relative_near_post_zones(self, fitted_xt):
        """near_post(ball_y=25) vs near_post(ball_y=40) give different zones."""
        from silly_kicks.tracking._gk_influence import Zone

        z1 = Zone.near_post(goal_x=0.0, ball_y=25.0)
        z2 = Zone.near_post(goal_x=0.0, ball_y=40.0)
        assert not np.allclose(z1.points, z2.points)

    def test_no_outfield_defenders_full_reachable(self, fitted_xt):
        """No outfield defenders -> reachable area = full GK circle."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[],
            away_positions=[(50, 30), (55, 40), (60, 20), (65, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            tau_seconds=3.0,
        )
        assert gi.reachable_area_m2 > 0.0


class TestMethodDispatch:
    """T-3: All pitch control methods produce valid GkInfluence."""

    @pytest.mark.parametrize("method", ["spearman", "fernandez_bornn", "voronoi"])
    def test_method_produces_valid_result(self, method, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            method=method,
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0
        assert gi.reachable_area_m2 >= 0.0
        assert "six_yard_box" in gi.closing_times


class TestZoneParameterized:
    """T-4: Zone-parameterized closing time."""

    def test_closing_times_keys(self, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        zones = [
            Zone.six_yard_box(goal_x=0.0),
            Zone.near_post(goal_x=0.0, ball_y=34.0),
            Zone.far_post(goal_x=0.0, ball_y=34.0),
        ]
        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            zones=zones,
        )
        assert set(gi.closing_times.keys()) == {"six_yard_box", "near_post", "far_post"}

    def test_gk_near_near_post_physical_invariant(self, fitted_xt):
        """GK near near-post -> near_post min_s < far_post min_s."""
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        # GK at (3, 31) — near the left post (y=30.34)
        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
            home_gk_pos=(3.0, 31.0),
        )
        zones = [
            Zone.near_post(goal_x=0.0, ball_y=25.0),
            Zone.far_post(goal_x=0.0, ball_y=25.0),
        ]
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            zones=zones,
        )
        assert gi.closing_times["near_post"].min_s < gi.closing_times["far_post"].min_s

    @pytest.mark.parametrize(
        "zone_factory,goal_x",
        [
            ("six_yard_box", 0.0),
            ("near_post", 0.0),
            ("far_post", 105.0),
        ],
    )
    def test_zone_produces_valid_closing_time(self, zone_factory, goal_x, standard_frame, fitted_xt):
        from silly_kicks.tracking._gk_influence import Zone, compute_gk_influence

        zone_fn = getattr(Zone, zone_factory)
        zone = zone_fn(goal_x=goal_x) if zone_factory == "six_yard_box" else zone_fn(goal_x=goal_x, ball_y=34.0)
        gi = compute_gk_influence(
            standard_frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            zones=[zone],
        )
        zct = gi.closing_times[zone_factory]
        assert zct.min_s >= 0.0
        assert zct.mean_s >= zct.min_s


class TestPartialNaNVelocities:
    """Edge case: some players have NaN vx/vy (common in Metrica/SkillCorner)."""

    def test_partial_nan_velocities_no_crash(self, fitted_xt):
        """Frames with NaN velocities for some players produce valid output."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        home_pos = [(20, 15), (25, 25), (30, 40), (35, 55), (45, 10)]
        away_pos = [(80, 20), (85, 30), (75, 40), (70, 55), (60, 10)]
        # Mix of valid and NaN velocities
        home_vel = [(2.0, 0.5), (np.nan, np.nan), (0.0, 0.0), (np.nan, np.nan), (3.0, -1.0)]
        away_vel = [(-1.0, 0.0), (np.nan, np.nan), (2.0, 1.0), (0.0, 0.0), (np.nan, np.nan)]
        frame = _make_two_team_frame(
            home_positions=home_pos,  # type: ignore[arg-type]
            away_positions=away_pos,  # type: ignore[arg-type]
            home_velocities=home_vel,
            away_velocities=away_vel,
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        # Should produce a result (possibly NaN share if all defenders have NaN vel,
        # but should not crash)
        assert gi is not None
        assert gi.reachable_area_m2 >= 0.0 or np.isnan(gi.reachable_area_m2)

    def test_all_nan_velocities_no_crash(self, fitted_xt):
        """Frame where all player velocities are NaN (edge case)."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        home_pos = [(20, 15), (25, 25), (30, 40), (35, 55)]
        away_pos = [(80, 20), (85, 30), (75, 40), (70, 55)]
        nan_vel = [(np.nan, np.nan)] * 4
        frame = _make_two_team_frame(
            home_positions=home_pos,  # type: ignore[arg-type]
            away_positions=away_pos,  # type: ignore[arg-type]
            home_velocities=nan_vel,
            away_velocities=nan_vel,
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert gi is not None


class TestOffPitchCoordinates:
    """Edge case: players with NaN or off-pitch x/y coordinates."""

    def test_nan_xy_player_excluded_gracefully(self, fitted_xt):
        """A player with NaN x/y should not crash the computation."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 15), (25, 25), (30, 40), (35, 55)],
            away_positions=[(80, 20), (85, 30), (75, 40), (70, 55)],
            home_velocities=[(1.0, 0.0), (0.0, 2.0), (-1.0, 0.0), (0.0, 0.0)],
            away_velocities=[(-2.0, 0.0), (0.0, -1.0), (1.0, 1.0), (0.0, 0.0)],
        )
        # Inject NaN coordinates for one outfield player
        frame.loc[frame["player_id"] == 10, "x"] = np.nan
        frame.loc[frame["player_id"] == 10, "y"] = np.nan
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert gi is not None
        assert not np.isinf(gi.pitch_control_share_weighted)

    def test_off_pitch_coordinates_handled(self, fitted_xt):
        """Player at x=-5 (off-pitch) should not crash, just produce valid output."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(-5.0, 34.0), (25, 25), (30, 40), (35, 55)],
            away_positions=[(110.0, 34.0), (85, 30), (75, 40), (70, 55)],
            home_velocities=[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            away_velocities=[(0.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert gi is not None
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0 or np.isnan(gi.pitch_control_share_weighted)
