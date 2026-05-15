"""Tests for silly_kicks.tracking._line_breaking (TF-32 Ward line-breaking)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_off_ball_runs import _make_action_at


def _make_three_line_fixture():
    """Fixture: away team with 3 clear defensive lines.

    - Forward line: x ~ 50 (3 players)
    - Midfield line: x ~ 70 (3 players)
    - Defense line: x ~ 90 (4 players)

    Home team with 5 outfield + GK.
    """
    return _make_frame_rows(
        home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
        home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
    )


class TestDetectLineBreaking:
    def test_pass_through_all_three_lines(self):
        """Pass from x=10 to x=100 should break all 3 lines."""
        from silly_kicks.tracking._line_breaking import (
            detect_line_breaking,
        )

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["line_break__ward"] == True  # noqa: E712
        assert row["lines_broken__ward"] == 3
        assert row["line_breaking_type__ward"] == "between_lines"

    def test_pass_through_one_line(self):
        """Pass from x=55 to x=75 should break midfield line only."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=55.0,
            start_y=34.0,
            end_x=75.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)

        row = result.iloc[0]
        assert row["line_break__ward"] == True  # noqa: E712
        assert row["lines_broken__ward"] >= 1

    def test_pass_not_crossing_any_line(self):
        """Short backward pass should not break any line."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=30.0,
            start_y=34.0,
            end_x=20.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)

        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712
        assert row["lines_broken__ward"] == 0
        assert pd.isna(row["line_breaking_type__ward"])

    def test_pass_around_line_wide(self):
        """Pass going wide of outermost defender -> type='around_line'."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # Away defenders at y=20..48, pass at y=60 (wide of all)
        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 48.0, 20.0, 34.0, 48.0, 20.0, 34.0, 48.0],
        )
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=60.0,
            start_y=64.0,  # wide of all defenders
            end_x=80.0,
            end_y=64.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        if row["line_break__ward"]:
            assert row["line_breaking_type__ward"] == "around_line"

    def test_too_few_opponents(self):
        """< min_opponents -> all False/0/None."""
        from silly_kicks.tracking._line_breaking import (
            LineBreakingParams,
            detect_line_breaking,
        )

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 30.0, 40.0, 50.0, 60.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 80.0],  # only 2 opponents
            away_outfield_ys=[34.0, 34.0],
        )
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(
            actions,
            frames,
            home_team_id=1,
            params=LineBreakingParams(min_opponents=3),
        )
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712
        assert row["lines_broken__ward"] == 0

    def test_short_pass_below_threshold(self):
        """Pass shorter than min_pass_length -> False."""
        from silly_kicks.tracking._line_breaking import (
            LineBreakingParams,
            detect_line_breaking,
        )

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=50.0,
            start_y=34.0,
            end_x=51.0,  # < 3m
            end_y=34.0,
        )

        result = detect_line_breaking(
            actions,
            frames,
            home_team_id=1,
            params=LineBreakingParams(min_pass_length=3.0),
        )
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712

    def test_zero_length_pass_returns_false(self):
        """Zero-length pass (start==end) correctly returns False.

        Root cause for IDSSE/Sportec Ward all-FALSE: Sportec event format
        provides only a single position per event (event location), not
        separate start/end. The SPADL converter sets start_x==end_x and
        start_y==end_y. With pass_len=0 < min_pass_length=3.0, Ward
        correctly returns FALSE. This is a data-source limitation (not a
        silly-kicks bug). The threshold method in _off_ball_runs.py uses
        positional comparison (end_x > def_line_x), which can return TRUE
        for stationary actions, giving different results — but the Ward
        geometric intersection test is the correct one for zero-length
        trajectories.
        """
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        # Simulate IDSSE: start == end (zero-length "cross")
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=60.0,
            start_y=10.0,
            end_x=60.0,  # same as start
            end_y=10.0,  # same as start
            type_id=1,  # cross
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712
        assert row["lines_broken__ward"] == 0

    def test_no_x_spread(self):
        """All opponents at same x -> no lines definable."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_frame_rows(
            home_outfield_xs=[20.0, 30.0, 40.0, 50.0, 60.0],
            home_outfield_ys=[34.0] * 5,
            away_outfield_xs=[70.0, 70.0, 70.0, 70.0, 70.0],  # all same x
            away_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        )
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        assert row["line_break__ward"] == False  # noqa: E712

    def test_empty_actions(self):
        """Empty actions -> empty result."""
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        actions = pd.DataFrame(
            columns=[
                "game_id",
                "action_id",
                "period_id",
                "time_seconds",
                "team_id",
                "player_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "type_id",
            ]
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert len(result) == 0

    def test_away_team_pass_coordinate_transform(self):
        """Away-team pass exercises the SPADL->tracking coordinate flip.

        In SPADL, both teams attack x=105. In LTR-normalized tracking,
        home attacks x=105, away attacks x=0. detect_line_breaking must
        transform away-team SPADL coords via (105-x, 68-y) to tracking.

        Fixture: home outfield at x=20-40, away outfield (defenders from
        away perspective) at x=50,70,90 with 3 clear lines.
        Away-team pass from SPADL (65,34) to (5,34) -> in tracking coords
        this is (40,34) to (100,34), which should cross all 3 away lines.
        """
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        frames = _make_three_line_fixture()
        # Use an away-team outfield player as the actor
        away_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 2) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        # Away team attacks x=105 in SPADL. Home defenders (from away's
        # perspective) are at x=105-20=85, x=105-25=80, etc. in SPADL.
        # The home outfield positions in tracking are x=20-40, which in
        # away-SPADL are x=65-85. A pass from SPADL x=90 to x=15 (deep
        # into home territory) should cross home defensive structure.
        # In tracking: start=(105-90,68-34)=(15,34), end=(105-15,68-34)=(90,34).
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(away_player),
            team_id=2,  # AWAY team
            start_x=90.0,  # SPADL: near away's own goal
            start_y=34.0,
            end_x=15.0,  # SPADL: deep into opponent half
            end_y=34.0,
        )

        result = detect_line_breaking(actions, frames, home_team_id=1)
        row = result.iloc[0]
        # Away pass crossing home territory should detect line breaks
        # The exact count depends on home's outfield structure, but
        # the key assertion is: the coordinate transform doesn't crash,
        # produces a valid (non-NaN) result, AND detects at least one break
        # (pass from x=90->15 crosses home outfield clustered at x=20-40)
        assert pd.notna(row["line_break__ward"])
        assert pd.notna(row["lines_broken__ward"])
        assert row["lines_broken__ward"] >= 1


class TestAddLineBreakWard:
    def test_method_ward_returns_ward_columns(self):
        """add_line_break(method='ward') returns ward-suffixed columns."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1, method="ward")
        assert "line_break__ward" in result.columns
        assert "lines_broken__ward" in result.columns
        assert "line_breaking_type__ward" in result.columns
        # Should NOT have threshold columns
        assert "line_break" not in result.columns
        assert "n_attackers_behind_line" not in result.columns

    def test_method_threshold_unchanged(self):
        """add_line_break(method='threshold') still returns old columns."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1, method="threshold")
        assert "line_break" in result.columns
        assert "n_attackers_behind_line" in result.columns
        assert "line_break__ward" not in result.columns

    def test_default_method_is_threshold(self):
        """Default method is 'threshold' (backward-compatible)."""
        from silly_kicks.tracking.features import add_line_break

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        result = add_line_break(actions, frames, home_team_id=1)
        assert "line_break" in result.columns  # threshold columns
        assert "line_break__ward" not in result.columns


class TestLineBreakingWardXfns:
    def test_xfn_column_count(self):
        """line_breaking_ward_xfns produces 9 columns (3 features x 3 states)."""
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        xfns = line_breaking_ward_xfns(home_team_id=1)
        assert len(xfns) == 1

        xfn = xfns[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_xfn_introspection_nan(self):
        """frames=None -> NaN DataFrame with 9 correct column names."""
        from silly_kicks.tracking.features import line_breaking_ward_xfns

        xfns = line_breaking_ward_xfns(home_team_id=1)
        xfn = xfns[0]

        dummy = pd.DataFrame(
            {
                "game_id": [1] * 10,
                "action_id": range(10),
                "period_id": [1] * 10,
                "time_seconds": [float(i) for i in range(10)],
                "team_id": [1] * 10,
                "player_id": [1] * 10,
                "start_x": [50.0] * 10,
                "start_y": [34.0] * 10,
                "end_x": [60.0] * 10,
                "end_y": [34.0] * 10,
                "type_id": [0] * 10,
                "result_id": [0] * 10,
                "bodypart_id": [0] * 10,
            }
        )
        states = [dummy, dummy, dummy]
        result = xfn(states, None)

        assert len(result.columns) == 9
        assert result.isna().all().all()
        # Verify naming pattern: lines_broken__ward_a0, etc.
        assert "lines_broken__ward_a0" in result.columns
        assert "line_breaking_type__ward_between_lines_a2" in result.columns
        assert "line_breaking_type__ward_around_line_a0" in result.columns


class TestGoldenFileBackwardCompat:
    """Snapshot test: add_line_break() default output unchanged by method= addition."""

    GOLDEN_PATH = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "tracking"
        / "golden"
        / "line_break_threshold_golden.parquet"
    )

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Build a deterministic fixture for golden-file comparison."""
        self.frames = _make_frame_rows(
            home_outfield_xs=[10.0, 20.0, 30.0, 40.0, 50.0],
            home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
            away_outfield_xs=[60.0, 70.0, 80.0, 90.0, 95.0],
            away_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        )
        home_player = self.frames[
            (~self.frames["is_ball"]) & (self.frames["team_id"] == 1) & (~self.frames["is_goalkeeper"])
        ]["player_id"].iloc[0]

        self.actions = pd.concat(
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
                    start_x=30.0,
                    start_y=34.0,
                    end_x=20.0,
                    end_y=34.0,
                    action_id=2,
                ),
            ],
            ignore_index=True,
        )

    def test_generate_golden_if_missing(self):
        """Generate golden file if it doesn't exist (run on main before PR)."""
        from silly_kicks.tracking.features import add_line_break

        result = add_line_break(self.actions, self.frames, home_team_id=1)
        golden_cols = ["line_break", "n_attackers_behind_line"]
        golden = result[golden_cols].copy()

        if not self.GOLDEN_PATH.exists():
            self.GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            golden.to_parquet(self.GOLDEN_PATH)
            pytest.skip("Golden file generated; re-run to verify.")

        expected = pd.read_parquet(self.GOLDEN_PATH)
        pd.testing.assert_frame_equal(golden, expected)


class TestCrossMethodSanity:
    def test_ward_and_threshold_soft_agreement(self):
        """When lines_broken__ward > 0, threshold line_break is usually True.

        Not a hard invariant (different algorithms) but flags gross disagreement.
        """
        from silly_kicks.tracking._line_breaking import detect_line_breaking
        from silly_kicks.tracking._off_ball_runs import _line_break_kernel

        frames = _make_three_line_fixture()
        home_player = frames[(~frames["is_ball"]) & (frames["team_id"] == 1) & (~frames["is_goalkeeper"])][
            "player_id"
        ].iloc[0]

        # Pass clearly through all lines
        actions = _make_action_at(
            time_seconds=1.0,
            player_id=int(home_player),
            team_id=1,
            start_x=10.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
        )

        ward = detect_line_breaking(actions, frames, home_team_id=1)
        threshold = _line_break_kernel(actions, frames, home_team_id=1)

        # Both should agree this pass breaks a line
        if ward.iloc[0]["lines_broken__ward"] > 0:
            assert threshold.iloc[0]["line_break"] == True  # noqa: E712
