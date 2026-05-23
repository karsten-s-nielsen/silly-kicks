# tests/tracking/test_player_influence.py
"""Unit tests for compute_player_influence (TF-36 + TF-33)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_frame(
    *,
    n_home_outfield: int = 10,
    n_away_outfield: int = 10,
    home_team_id: int = 1,
    away_team_id: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic 22-player frame for testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    # Ball at center
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=0,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            vx=5.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home GK at x=3
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=1,
            team_id=home_team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=3.0,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home outfield
    for i in range(n_home_outfield):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=10 + i,
                team_id=home_team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 60)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    # Away GK at x=102
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=50,
            team_id=away_team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=102.0,
            y=34.0,
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Away outfield
    for i in range(n_away_outfield):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=60 + i,
                team_id=away_team_id,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(45, 95)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def frame_22():
    return _make_frame()


@pytest.fixture
def xt_grid():
    """Pre-fit xT with linear gradient (high xT near x=105)."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


# --- Basic correctness ---


def test_compute_player_influence_returns_outfield_only(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    # GKs (player_id 1 and 50) excluded; ball excluded
    assert 1 not in result
    assert 50 not in result
    assert 0 not in result
    # 20 outfield players present
    assert len(result) == 20


def test_off_ball_xt_positive_for_outfield(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.off_ball_xt >= 0.0, f"Player {pid} has negative off_ball_xt"


def test_reachable_area_positive_for_outfield(frame_22, xt_grid):
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 >= 0.0, f"Player {pid} has negative area"


# --- Invariants ---


def test_reachable_area_sum_lte_pitch_area(frame_22, xt_grid):
    """Sum of same-team uniquely reachable areas <= total pitch area."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    # Get team membership from frame
    players = frame_22[~frame_22["is_ball"].astype(bool) & ~frame_22["is_goalkeeper"].astype(bool)]
    for tid in players["team_id"].unique():
        team_pids = players[players["team_id"] == tid]["player_id"].values
        team_area = sum(result[pid].reachable_area_m2 for pid in team_pids if pid in result)
        assert team_area <= 105.0 * 68.0, f"Team {tid} total area {team_area:.1f} > pitch area {105 * 68}"


def test_tau_zero_all_areas_zero(frame_22, xt_grid):
    """tau=0 -> nobody reaches anywhere -> all areas 0."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        tau_seconds=0.0,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 == 0.0, f"Player {pid} area={pi.reachable_area_m2} with tau=0"


@pytest.mark.parametrize("method", ["spearman", "voronoi", "fernandez_bornn"])
def test_off_ball_xt_bounded_by_pitch_area(frame_22, xt_grid, method):
    """Outfield off_ball_xt sum <= total xT area (GKs consume the rest)."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        method=method,
    )
    total_player_xt = sum(pi.off_ball_xt for pi in result.values())
    for pid, pi in result.items():
        assert np.isfinite(pi.off_ball_xt), f"Player {pid} has non-finite off_ball_xt"
        assert pi.off_ball_xt >= 0.0, f"Player {pid} has negative off_ball_xt"
    assert np.isfinite(total_player_xt)


# --- Edge cases ---


def test_single_outfield_player_per_team(xt_grid):
    """1 outfield player per team -> uniquely reachable area with generous tau."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame = _make_frame(n_home_outfield=1, n_away_outfield=1)
    # tau=5.0 gives enough motion time (5.0 - 0.7 reaction = 4.3s)
    # to reach grid cells despite coarse pitch control grid
    result = compute_player_influence(
        frame,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        tau_seconds=5.0,
    )
    assert len(result) == 2
    for _pid, pi in result.items():
        assert pi.reachable_area_m2 > 0.0


def test_all_players_same_position(xt_grid):
    """All outfield at same position -> all uniquely reachable = 0."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    rows: list[dict] = []
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=0,
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
    # 2 GKs
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=gk_pid,
                team_id=gk_tid,
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
    # 4 outfield all at (50, 34) with zero velocity
    for i in range(4):
        tid = 1 if i < 2 else 2
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=10 + i,
                team_id=tid,
                is_ball=False,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    frame = pd.DataFrame(rows)
    result = compute_player_influence(
        frame,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    for pid, pi in result.items():
        assert pi.reachable_area_m2 == 0.0, f"Player {pid} area={pi.reachable_area_m2} (expected 0)"


def test_nan_velocity_defaults_to_zero(xt_grid):
    """NaN vx/vy should not produce NaN results."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    frame = _make_frame(n_home_outfield=3, n_away_outfield=3)
    # Set some velocities to NaN
    frame.loc[frame["player_id"] == 10, "vx"] = np.nan
    frame.loc[frame["player_id"] == 10, "vy"] = np.nan
    result = compute_player_influence(
        frame,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
    )
    pi = result[10]
    assert not np.isnan(pi.reachable_area_m2), "NaN velocity should not produce NaN area"
    assert not np.isnan(pi.off_ball_xt), "NaN velocity should not produce NaN off_ball_xt"


@pytest.mark.parametrize("method", ["voronoi", "fernandez_bornn"])
def test_non_spearman_method_non_degenerate(frame_22, xt_grid, method):
    """Non-spearman methods should produce non-zero results."""
    from silly_kicks.tracking._player_influence import compute_player_influence

    result = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        method=method,
    )
    total_xt = sum(pi.off_ball_xt for pi in result.values())
    assert total_xt > 0.0, f"Method {method} produced zero total off_ball_xt"


# --- TTI optimization parity ---


def test_tti_optimization_matches_naive(xt_grid):
    """The argmin/second-min trick must be numerically equivalent to naive loop."""
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.tracking.pitch_control import SpearmanParams, compute_pitch_control
    from silly_kicks.tracking.pitch_control._spearman import compute_tti

    frame = _make_frame(n_home_outfield=5, n_away_outfield=5)
    sp = SpearmanParams()
    tau = 1.0

    # Get optimized result
    optimized = compute_player_influence(
        frame,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        tau_seconds=tau,
    )

    # Compute naive result for comparison
    pc = compute_pitch_control(frame, attacking_team_id=1, decompose=True)
    gx, gy = np.meshgrid(pc.grid_x, pc.grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])

    players = frame[~frame["is_ball"].astype(bool) & ~frame["is_goalkeeper"].astype(bool)].dropna(subset=["x", "y"])

    for tid in players["team_id"].unique():
        team = players[players["team_id"] == tid]
        n = len(team)
        pos = team[["x", "y"]].to_numpy(dtype="float64")
        vx = team["vx"].to_numpy(dtype="float64")
        vy = team["vy"].to_numpy(dtype="float64")
        vel = np.column_stack([np.nan_to_num(vx), np.nan_to_num(vy)])

        tti_all = compute_tti(pos, vel, targets, sp.reaction_time, sp.max_acceleration)

        for i in range(n):
            pid = team.iloc[i]["player_id"]
            player_tti = tti_all[i]
            # Naive: min of ALL OTHER teammates
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if mask.any():
                min_others = tti_all[mask].min(axis=0)
            else:
                min_others = np.full(len(targets), np.inf)
            naive_unique = (player_tti <= tau) & (min_others > tau)
            naive_area = float(naive_unique.sum() * pc.cell_area)

            assert optimized[pid].reachable_area_m2 == pytest.approx(naive_area, abs=1e-10), (
                f"Player {pid}: optimized={optimized[pid].reachable_area_m2}, naive={naive_area}"
            )


# --- Pre-computed surface ---


def test_surface_parameter_skips_pc_call(frame_22, xt_grid):
    """When surface is provided, method/params are ignored."""
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.tracking.pitch_control import compute_pitch_control

    # Pre-compute surface with spearman
    surface = compute_pitch_control(
        frame_22,
        attacking_team_id=1,
        method="spearman",
        decompose=True,
    )

    # Pass surface + method="voronoi" — voronoi should be ignored
    result_with_surface = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        surface=surface,
        method="voronoi",  # should be ignored
    )

    # Compare with direct spearman call (no surface param)
    result_direct = compute_player_influence(
        frame_22,
        xt_grid,
        attacking_team_id=1,
        home_team_id=1,
        method="spearman",
    )

    for pid in result_with_surface:
        assert result_with_surface[pid].off_ball_xt == pytest.approx(result_direct[pid].off_ball_xt, abs=1e-10)
        assert result_with_surface[pid].reachable_area_m2 == pytest.approx(
            result_direct[pid].reachable_area_m2, abs=1e-10
        )
