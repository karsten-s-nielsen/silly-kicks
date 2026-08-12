"""Physical invariants for off-ball runs + line-break (TF-4)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking import resolve_defended_goals
from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture


@pytest.fixture
def off_ball_fixture():
    """Multi-player fixture with known off-ball movement."""
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    n_frames = 10
    players = [
        {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * n_frames},
        {"player_id": 11, "team_id": 1, "positions": [(30 + i * 5.0 / (n_frames - 1), 34) for i in range(n_frames)]},
        {"player_id": 12, "team_id": 1, "positions": [(40, 20 + i * 4.0 / (n_frames - 1)) for i in range(n_frames)]},
        {"player_id": 13, "team_id": 1, "positions": [(60 + i * 6.0 / (n_frames - 1), 34) for i in range(n_frames)]},
        {"player_id": 14, "team_id": 1, "positions": [(45, 34 + i * 1.0 / (n_frames - 1)) for i in range(n_frames)]},
        {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * n_frames},
        {"player_id": 21, "team_id": 2, "positions": [(85, 34)] * n_frames},
        {"player_id": 22, "team_id": 2, "positions": [(90, 34)] * n_frames},
        {"player_id": 23, "team_id": 2, "positions": [(95, 34)] * n_frames},
        {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * n_frames},
        {"player_id": 24, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * n_frames},
    ]
    frames = _make_multi_frame_fixture(players=players, n_frames=n_frames, frame_rate=n_frames / 1.5)
    action_time = (n_frames - 1) * (1.5 / (n_frames - 1))
    actions = _make_action_at(time_seconds=action_time, player_id=10, team_id=1)

    return _off_ball_runs_kernel(actions, frames, home_team_id=1)


class TestOffBallRunsInvariants:
    def test_n_runners_non_negative(self, off_ball_fixture):
        valid = off_ball_fixture["n_off_ball_runners_pre_window"].dropna()
        assert (valid >= 0).all()

    def test_toward_goal_subset_of_runners(self, off_ball_fixture):
        df = off_ball_fixture.dropna(subset=["n_off_ball_runners_pre_window"])
        assert (df["n_off_ball_runners_toward_goal_pre_window"] <= df["n_off_ball_runners_pre_window"]).all()

    def test_max_displacement_exceeds_threshold(self, off_ball_fixture):
        has_runners = off_ball_fixture[off_ball_fixture["n_off_ball_runners_pre_window"] > 0]
        if not has_runners.empty:
            assert (has_runners["max_off_ball_run_displacement_pre_window"] >= 3.0 - 1e-9).all()

    def test_mean_speed_non_negative(self, off_ball_fixture):
        valid = off_ball_fixture["mean_off_ball_run_speed_pre_window"].dropna()
        assert (valid >= 0).all()


class TestLineBreakInvariants:
    def test_n_attackers_non_negative(self):
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
            home_outfield_ys=[10.0, 20.0, 30.0, 40.0, 50.0, 34.0, 34.0, 34.0, 34.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        actions = _make_action_at(time_seconds=1.0, player_id=50, team_id=1, end_x=95.0)
        # Pick a real player_id from frames
        actions["player_id"] = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))
        valid = result["n_attackers_behind_line"].dropna()
        assert (valid >= 0).all()

    def test_line_break_true_implies_end_x_past_line(self):
        """line_break=True -> end_x past the opposing defensive line (coordinate-resolved)."""
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel
        from tests.tracking.test_defensive_line import _make_frame_rows

        # Away back 4 at x=90,92,94,96 -> defensive_line_x ~93
        # For home action: spadl_def_line_x = 93 (no flip)
        frames = _make_frame_rows(
            home_outfield_xs=[10.0, 12.0, 14.0, 16.0, 50.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
            away_outfield_xs=[90.0, 92.0, 94.0, 96.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0, 34.0],
        )
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        # Action PAST line -> True
        a_past = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            end_x=95.0,
            action_id=1,
        )
        # Action SHORT of line -> False
        a_short = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            end_x=80.0,
            action_id=2,
        )

        import pandas as pd

        actions = pd.concat([a_past, a_short], ignore_index=True)
        result = _line_break_kernel(actions, frames, goal_map=resolve_defended_goals(frames))

        # Where line_break is True, end_x must exceed the spadl defensive line
        lb_true = result[result["line_break"] == True]  # noqa: E712
        if not lb_true.empty:
            # For home team, spadl_def_line_x = defensive_line_x (no flip)
            # We know away back 4 mean is 93, so end_x=95 > 93 should be True
            true_indices = lb_true.index
            true_actions = actions.loc[true_indices]
            assert (true_actions["end_x"] > 90).all()  # comfortably past the line area
