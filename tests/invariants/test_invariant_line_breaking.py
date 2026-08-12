"""Physical invariants for Ward line-breaking (TF-32)."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_off_ball_runs import _make_action_at


@pytest.fixture
def line_breaking_result():
    """Pass across 3 defensive lines -> known line-breaking result."""
    from silly_kicks.tracking._line_breaking import detect_line_breaking

    frames = _make_frame_rows(
        home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
        home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        away_outfield_xs=[
            50.0,
            50.0,
            50.0,
            70.0,
            70.0,
            70.0,
            90.0,
            90.0,
            90.0,
            90.0,
        ],
        away_outfield_ys=[
            15.0,
            34.0,
            53.0,
            15.0,
            34.0,
            53.0,
            10.0,
            24.0,
            44.0,
            58.0,
        ],
    )
    home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
        "player_id"
    ].iloc[0]

    # Multiple actions: through all, through 1, none
    actions = pd.concat(
        [
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=10.0,
                start_y=34.0,
                end_x=100.0,
                end_y=34.0,
                action_id=1,
            ),
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=55.0,
                start_y=34.0,
                end_x=75.0,
                end_y=34.0,
                action_id=2,
            ),
            _make_action_at(
                time_seconds=1.0,
                player_id=int(home_player),
                team_id=1,
                start_x=30.0,
                start_y=34.0,
                end_x=20.0,
                end_y=34.0,
                action_id=3,
            ),
        ],
        ignore_index=True,
    )

    return detect_line_breaking(actions, frames)


class TestLineBreakingInvariants:
    def test_lines_broken_domain(self, line_breaking_result):
        valid = line_breaking_result["lines_broken__ward"].dropna()
        assert set(valid.unique()).issubset({0, 1, 2, 3})

    def test_is_line_breaking_consistent(self, line_breaking_result):
        """is_line_breaking == (lines_broken > 0)."""
        df = line_breaking_result.dropna(subset=["lines_broken__ward"])
        expected = df["lines_broken__ward"] > 0
        actual = df["line_break__ward"]
        assert (actual == expected).all()

    def test_type_domain(self, line_breaking_result):
        valid = line_breaking_result["line_breaking_type__ward"].dropna()
        assert set(valid.unique()).issubset({"between_lines", "around_line"})

    def test_type_none_when_no_break(self, line_breaking_result):
        """line_breaking_type is None when lines_broken == 0."""
        no_break = line_breaking_result[line_breaking_result["lines_broken__ward"] == 0]
        if not no_break.empty:
            assert no_break["line_breaking_type__ward"].isna().all()
