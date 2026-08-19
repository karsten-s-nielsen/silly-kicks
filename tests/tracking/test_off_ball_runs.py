"""Tests for off-ball runs + line-break features (TF-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import resolve_defended_goals


def _make_multi_frame_fixture(
    *,
    home_team_id=1,
    away_team_id=2,
    n_frames=5,
    frame_rate=25.0,
    period_id=1,
    game_id=1,
    players: list[dict] | None = None,
):
    """Build a multi-frame tracking fixture with controlled player movement.

    Each player dict: {player_id, team_id, is_goalkeeper, positions: [(x, y), ...]}
    where positions[i] is the (x, y) at frame i. len(positions) must == n_frames.
    """
    if players is None:
        players = []
    rows = []
    for fi in range(n_frames):
        time_s = fi * (1.0 / frame_rate)
        # Ball row
        rows.append(
            dict(
                game_id=game_id,
                period_id=period_id,
                frame_id=fi + 1,
                time_seconds=time_s,
                frame_rate=frame_rate,
                player_id=np.nan,
                team_id=np.nan,
                is_ball=True,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                ball_state="alive",
                team_attacking_direction="ltr",
                source_provider="synthetic",
            )
        )
        for p in players:
            pos = p["positions"][fi]
            rows.append(
                dict(
                    game_id=game_id,
                    period_id=period_id,
                    frame_id=fi + 1,
                    time_seconds=time_s,
                    frame_rate=frame_rate,
                    player_id=p["player_id"],
                    team_id=p["team_id"],
                    is_ball=False,
                    is_goalkeeper=p.get("is_goalkeeper", False),
                    x=pos[0],
                    y=pos[1],
                    ball_state="alive",
                    # ADR-041: per-team direction, NOT a blanket "ltr". Labelling both
                    # teams "ltr" would silently make acting_team_attacks_rtl return no-flip
                    # for the away team and mis-orient its geometry -- which is exactly why
                    # toward_goal could not be re-keyed onto the frames' own direction until
                    # per-team labels were used. (validate_period_directions accepts uniform
                    # "ltr"; it rejects only a SINGLE team self-contradicting.)
                    team_attacking_direction="ltr" if p["team_id"] == home_team_id else "rtl",
                    source_provider="synthetic",
                )
            )
    return pd.DataFrame(rows)


def _make_action_at(
    *,
    time_seconds: float,
    player_id: int,
    team_id: int,
    start_x: float = 50.0,
    start_y: float = 34.0,
    end_x: float = 60.0,
    end_y: float = 34.0,
    period_id: int = 1,
    game_id: int = 1,
    action_id: int = 1,
    type_id: int = 0,
):
    """Create a single-row actions DataFrame."""
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "action_id": [action_id],
            "period_id": [period_id],
            "time_seconds": [time_seconds],
            "team_id": [team_id],
            "player_id": [player_id],
            "start_x": [start_x],
            "start_y": [start_y],
            "end_x": [end_x],
            "end_y": [end_y],
            "type_id": [type_id],
        }
    )


class TestOffBallRunsKernel:
    def test_basic_two_qualifying_runners(self):
        """Two teammates move >=3m, one doesn't -> count=2."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(50 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {
                "player_id": 12,
                "team_id": 1,
                "positions": [(30, 34 + i * 4.0 / (n_frames - 1)) for i in range(n_frames)],
            },
            {
                "player_id": 13,
                "team_id": 1,
                "positions": [(40, 34 + i * 1.0 / (n_frames - 1)) for i in range(n_frames)],
            },
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(
            players=players,
            n_frames=n_frames,
            frame_rate=n_frames / 1.5,
        )
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)

        assert result["n_off_ball_runners_pre_window"].iloc[0] == 2
        assert result["max_off_ball_run_displacement_pre_window"].iloc[0] == pytest.approx(5.0, abs=0.01)
        # mean speed: (5.0/1.5 + 4.0/1.5) / 2 = 3.0
        assert result["mean_off_ball_run_speed_pre_window"].iloc[0] == pytest.approx(3.0, abs=0.1)
        # Only player 11 moves toward goal (positive dx for home team)
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1

    def test_actor_excluded(self):
        """Actor's own movement is not counted."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {
                "player_id": 10,
                "team_id": 1,
                "positions": [(50 + i * 10.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_opponent_excluded(self):
        """Opponents' movement is not counted."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 20,
                "team_id": 2,
                "positions": [(80 + i * 10.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_below_threshold_all_nan(self):
        """All teammates move < min_displacement_m -> 0 runners, NaN max/mean."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(30 + i * 1.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0
        assert pd.isna(result["max_off_ball_run_displacement_pre_window"].iloc[0])
        assert pd.isna(result["mean_off_ball_run_speed_pre_window"].iloc[0])

    def test_toward_goal_home_team(self):
        """Home-team runners: positive dx = toward goal."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            # Moves LEFT (negative dx) - NOT toward goal for home team
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(60 - i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1  # qualifies by displacement
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 0  # but NOT toward goal

    def test_toward_goal_away_team(self):
        """Away-team runners: negative dx = toward goal (x=0 is their attacking direction)."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 20, "team_id": 2, "positions": [(60, 34)] * n_frames},
            # Away teammate moves LEFT (negative dx) -> toward x=0 = toward goal for away
            {
                "player_id": 22,
                "team_id": 2,
                "positions": [(40 - i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=20, team_id=2)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1
        assert result["n_off_ball_runners_toward_goal_pre_window"].iloc[0] == 1

    def test_dead_ball_at_action_time_nan(self):
        """Dead ball at action timestamp -> entire action NaN."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        # Mark the last frame (action time) as dead ball
        last_frame_mask = frames["frame_id"] == n_frames
        frames.loc[last_frame_mask, "ball_state"] = "dead"

        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert pd.isna(result["n_off_ball_runners_pre_window"].iloc[0])

    def test_no_teammates_zero(self):
        """Actor is only outfield player on team -> 0 runners."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

    def test_custom_params(self):
        """Non-default pre_seconds and min_displacement_m are respected."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        # Player moves 2m total
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(30 + i * 2.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        # default threshold=3.0 -> 0 runners
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0

        # lower threshold=1.5 -> 1 runner
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1, min_displacement_m=1.5)
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1

    def test_multi_game_no_cross_contamination(self):
        """Two games with same period_id=1 don't cross-contaminate."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 5
        # Game 1: teammate moves 5m
        players_g1 = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        # Game 2: NO teammates for actor (different player set)
        players_g2 = [
            {"player_id": 30, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 2, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 40, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames_g1 = _make_multi_frame_fixture(
            players=players_g1, n_frames=n_frames, frame_rate=n_frames / 1.5, game_id=1
        )
        frames_g2 = _make_multi_frame_fixture(
            players=players_g2, n_frames=n_frames, frame_rate=n_frames / 1.5, game_id=2
        )
        frames = pd.concat([frames_g1, frames_g2], ignore_index=True)

        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions_g1 = _make_action_at(time_seconds=action_time, player_id=10, team_id=1, game_id=1, action_id=1)
        actions_g2 = _make_action_at(time_seconds=action_time, player_id=30, team_id=1, game_id=2, action_id=2)
        actions = pd.concat([actions_g1, actions_g2], ignore_index=True)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        # Game 1 should see 1 runner (player 11); game 2 should see 0 runners
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 1
        assert result["n_off_ball_runners_pre_window"].iloc[1] == 0

    def test_ltr_guard_raises(self):
        """Non-LTR frames raise ValueError."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        n_frames = 3
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=25.0)
        frames["team_attacking_direction"] = "rtl"
        actions = _make_action_at(time_seconds=0.08, player_id=10, team_id=1)

        with pytest.raises(ValueError, match="period-normalized"):
            _off_ball_runs_kernel(actions, frames, home_team_id=1)

    def test_empty_frames_returns_columns(self):
        """Empty frames -> result with correct columns, all NaN."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        actions = _make_action_at(time_seconds=1.0, player_id=10, team_id=1)
        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "frame_rate",
                "player_id",
                "team_id",
                "is_ball",
                "is_goalkeeper",
                "x",
                "y",
                "ball_state",
                "team_attacking_direction",
                "source_provider",
            ]
        )
        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        expected_cols = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) == 1

    def test_single_frame_per_player_nan(self):
        """Only 1 frame per player (< 2 needed for displacement) -> 0 runners."""
        from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

        # 1 frame => no displacement measurable
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)]},
            {"player_id": 11, "team_id": 1, "positions": [(30, 34)]},
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)]},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)]},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=1, frame_rate=25.0)
        actions = _make_action_at(time_seconds=0.0, player_id=10, team_id=1)

        result = _off_ball_runs_kernel(actions, frames, home_team_id=1)
        # With only 1 frame, linked but no displacement possible -> 0 runners
        assert result["n_off_ball_runners_pre_window"].iloc[0] == 0


class TestLineBreakKernel:
    def test_crosses_line_home_team(self):
        """Home-team action end_x past away team's defensive line -> True."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Away back 4 at x=90,92,94,96 -> mean defensive_line_x = 93
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert result["line_break"].iloc[0] == True  # noqa: E712

    def test_does_not_cross_line(self):
        """Home-team action end_x short of line -> False."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=80.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert result["line_break"].iloc[0] == False  # noqa: E712

    def test_crosses_line_away_team(self):
        """Away-team action: coordinate flip applied correctly."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Home back 4 at x=10,12,14,16 -> mean defensive_line_x = 13
        # For away team action: spadl_def_line_x = 105 - 13 = 92
        # Away-team action with end_x=95 > 92 -> line_break=True
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        away_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 2) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(away_outfield["player_id"].iloc[0]),
            team_id=2,
            end_x=95.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert result["line_break"].iloc[0] == True  # noqa: E712

    def test_no_defensive_line_returns_na(self):
        """< 3 outfield opponents -> pd.NA."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Only 2 away outfield players (need 3 for defensive line)
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0],
            away_outfield_ys=[30.0, 40.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert pd.isna(result["line_break"].iloc[0])

    def test_n_attackers_behind_line_home(self):
        """Home-team action: count attackers with tracking x > defensive_line_x."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Away back 4 at x=70,72,74,76 -> defensive_line_x = 73
        # Home outfield at x=10,12,14,16,75 -> player at x=75 is behind away line
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 75.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[70.0, 72.0, 74.0, 76.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=80.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        # One home player (x=75) is behind away defensive line (73)
        assert result["n_attackers_behind_line"].iloc[0] == 1

    def test_n_attackers_behind_line_away(self):
        """Away-team action: count attackers with tracking x < defensive_line_x."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Home back 4 at x=30,32,34,36 -> defensive_line_x = 33
        # Away outfield at x=90,92,94,96,25 -> player at x=25 is behind home line
        frames = _make_frame_rows(
            home_outfield_xs=[30.0, 32.0, 34.0, 36.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 25.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        away_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 2) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(away_outfield["player_id"].iloc[0]),
            team_id=2,
            end_x=95.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        # One away player (x=25) is behind home defensive line (33) -- x < 33
        assert result["n_attackers_behind_line"].iloc[0] == 1

    def test_line_break_dtype_is_boolean(self):
        """line_break column is nullable boolean, not object."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
            end_y=34.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert str(result["line_break"].dtype) == "boolean"

    def test_no_linked_frame_returns_na(self):
        """Action can't link to any frame -> pd.NA for both columns."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        # Action at time_seconds=999 — far from any frame -> no link
        actions = _make_action_at(
            time_seconds=999.0,
            player_id=50,
            team_id=1,
            end_x=95.0,
        )

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        assert pd.isna(result["line_break"].iloc[0])
        assert pd.isna(result["n_attackers_behind_line"].iloc[0])


class TestAggregators:
    def test_add_off_ball_runs_columns(self):
        from silly_kicks.tracking.features import add_off_ball_runs

        n_frames = 5
        players = [
            {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
            {
                "player_id": 11,
                "team_id": 1,
                "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)],
            },
            {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
            {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
            {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        ]
        frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
        action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
        actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

        result = add_off_ball_runs(actions, frames, home_team_id=1)
        new_cols = set(result.columns) - set(actions.columns)
        assert "n_off_ball_runners_pre_window" in new_cols
        assert "max_off_ball_run_displacement_pre_window" in new_cols
        assert "mean_off_ball_run_speed_pre_window" in new_cols
        assert "n_off_ball_runners_toward_goal_pre_window" in new_cols
        assert len(new_cols) == 4

    def test_add_line_break_columns(self):
        from silly_kicks.tracking.features import add_line_break
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
        )

        result = add_line_break(actions, frames)
        new_cols = set(result.columns) - set(actions.columns)
        assert "line_break" in new_cols
        assert "n_attackers_behind_line" in new_cols
        assert len(new_cols) == 2
        assert str(result["line_break"].dtype) == "boolean"
        assert str(result["n_attackers_behind_line"].dtype) == "Int64"

    def test_add_off_ball_context_all_six(self):
        from silly_kicks.tracking.features import add_off_ball_context
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
        )

        result = add_off_ball_context(actions, frames)
        expected = {
            "n_off_ball_runners_pre_window",
            "max_off_ball_run_displacement_pre_window",
            "mean_off_ball_run_speed_pre_window",
            "n_off_ball_runners_toward_goal_pre_window",
            "line_break",
            "n_attackers_behind_line",
        }
        new_cols = set(result.columns) - set(actions.columns)
        assert expected.issubset(new_cols)


class TestXfnFactory:
    def test_factory_returns_frame_aware(self):
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.feature_framework import is_frame_aware

        xfns = off_ball_context_xfns()
        assert len(xfns) == 1
        assert is_frame_aware(xfns[0])

    def test_factory_column_count(self):
        """Factory transformer emits 6 x 3 = 18 columns."""
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.feature_framework import gamestates
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
        )
        # gamestates needs enough rows; duplicate for 4 actions
        actions = pd.concat([actions] * 4, ignore_index=True)
        actions["action_id"] = list(range(1, 5))
        states = gamestates(actions, nb_prev_actions=3)

        xfn = off_ball_context_xfns()[0]
        result = xfn(states, frames)
        assert result.shape[1] == 18

    def test_vaep_introspection_no_crash(self):
        """feature_column_names with off_ball_context_xfns -> no crash."""
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.features.core import feature_column_names

        xfns = off_ball_context_xfns()
        cols = feature_column_names(xfns)
        assert len(cols) == 18


class TestLineBreakKernelGameIdTypeMismatch:
    """game_id type mismatch between actions (str) and frames (int).

    Same vulnerability as _line_breaking.py Bug 3: dict-based frame lookup
    silently fails when game_id types don't match across actions and frames.
    """

    def test_mismatched_game_id_types_still_works(self):
        """Str game_id in actions + int game_id in frames -> line_break still computed."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        # Frames have int game_id=1
        home_outfield = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_outfield["player_id"].iloc[0]),
            team_id=1,
            end_x=95.0,
            end_y=34.0,
        )
        # Actions have str game_id — type mismatch
        actions["game_id"] = "1"

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        # Before fix: n_attackers_behind_line would be 0 (lookup miss)
        assert result["line_break"].iloc[0] == True  # noqa: E712
        assert result["n_attackers_behind_line"].iloc[0] >= 0


class TestLineBreakPathNaTeamFramesDegrade:
    """A NaN-team_id 'team' in the FRAMES (>=3 unassigned/false-positive tracking detections)
    makes `compute_defensive_line` try to resolve the NA team's end -> `goal_map.get` returns
    None (NA key) -> `GoalEndUnresolvedError`. `add_defensive_line` catches it and NaN-degrades;
    the three `_line_break_kernel` callers (`add_line_break` threshold, `add_off_ball_context`,
    the `off_ball_context_xfns` transformer) must do the SAME, not raise uncaught.
    """

    @staticmethod
    def _na_team_frames():
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
            home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
            away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
            away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
        )
        frames["team_id"] = frames["team_id"].astype("Int64")
        frames["player_id"] = frames["player_id"].astype("Int64")
        base = frames[frames["team_id"].notna() & (~frames["is_ball"])].iloc[0].to_dict()
        na_rows = []
        for x, y in [(55.0, 20.0), (55.0, 34.0), (55.0, 48.0)]:
            row = dict(base)
            row.update(player_id=pd.NA, team_id=pd.NA, is_goalkeeper=False, is_ball=False, x=x, y=y)
            na_rows.append(row)
        na_df = pd.DataFrame(na_rows).astype({"team_id": "Int64", "player_id": "Int64"})
        frames = pd.concat([frames, na_df], ignore_index=True)
        frames["team_id"] = frames["team_id"].astype("Int64")
        frames["player_id"] = frames["player_id"].astype("Int64")
        return frames

    @classmethod
    def _action(cls, frames):
        home = frames[(frames["team_id"] == 1) & (~frames["is_ball"]) & (~frames["is_goalkeeper"])]
        return _make_action_at(
            time_seconds=1.0,
            player_id=int(home["player_id"].iloc[0]),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

    def test_kernel_degrades_not_raises(self):
        from silly_kicks.tracking._off_ball_runs import _LINE_BREAK_COLS, _line_break_kernel

        frames = self._na_team_frames()
        out = _line_break_kernel(self._action(frames), frames, goal_map=resolve_defended_goals(frames))
        assert list(out.columns) == _LINE_BREAK_COLS
        assert out["line_break"].isna().all()
        assert out["n_attackers_behind_line"].isna().all()

    def test_add_line_break_threshold_degrades(self):
        from silly_kicks.tracking.features import add_line_break

        frames = self._na_team_frames()
        out = add_line_break(self._action(frames), frames)  # method="threshold" default
        assert out["line_break"].isna().all()

    def test_add_off_ball_context_degrades(self):
        from silly_kicks.tracking.features import add_off_ball_context

        frames = self._na_team_frames()
        out = add_off_ball_context(self._action(frames), frames)
        assert out["line_break"].isna().all()

    def test_off_ball_context_xfns_transformer_degrades(self):
        from silly_kicks.tracking.features import off_ball_context_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frames = self._na_team_frames()
        actions = pd.concat([self._action(frames)] * 4, ignore_index=True)
        actions["action_id"] = list(range(1, 5))
        states = gamestates(actions, nb_prev_actions=3)
        result = off_ball_context_xfns()[0](states, frames)  # must not raise
        assert result.shape[1] == 18

    def test_add_defensive_line_reference_degrades(self):
        # The sibling that ALREADY catches GoalEndUnresolvedError -- the pattern being mirrored.
        from silly_kicks.tracking.features import add_defensive_line

        frames = self._na_team_frames()
        out = add_defensive_line(self._action(frames), frames)
        assert out["defensive_line_x"].isna().all()

    def test_resolvable_control_does_not_degrade(self):
        # Without the NA rows the same entry point resolves normally (no raise; real values).
        from silly_kicks.tracking.features import add_line_break
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
            home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
            away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
            away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
        )
        home = frames[(frames["team_id"] == 1) & (~frames["is_ball"]) & (~frames["is_goalkeeper"])]
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home["player_id"].iloc[0]),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )
        out = add_line_break(actions, frames)
        assert out["line_break"].notna().any()
