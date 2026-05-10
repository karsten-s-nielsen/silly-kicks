"""Bundled fixes for TF-31/TF-32 (National Park Principle)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_line_breaking_fixture(
    *,
    action_team_id: int = 1,
    opp_positions: list[tuple[float, float]],
    start_xy: tuple[float, float] = (40.0, 34.0),
    end_xy: tuple[float, float] = (70.0, 34.0),
    home_team_id: int = 1,
    action_type_id: int = 0,  # 0 = pass in SPADL
):
    """Build minimal action + frame for line-breaking testing."""
    rows = []
    # Ball
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
            x=start_xy[0],
            y=start_xy[1],
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Opponents
    opp_team = 2 if action_team_id == 1 else 1
    for i, (ox, oy) in enumerate(opp_positions):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=50 + i,
                team_id=opp_team,
                is_ball=False,
                is_goalkeeper=False,
                x=ox,
                y=oy,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    # Action-team player (passer)
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=10,
            team_id=action_team_id,
            is_ball=False,
            is_goalkeeper=False,
            x=start_xy[0],
            y=start_xy[1],
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [1.0],
            "team_id": [action_team_id],
            "type_id": [action_type_id],
            "result_id": [1],
            "start_x": [start_xy[0]],
            "start_y": [start_xy[1]],
            "end_x": [end_xy[0]],
            "end_y": [end_xy[1]],
            "bodypart_id": [0],
            "player_id": [10],
        }
    )
    return actions, frames


class TestH1DropnaMisalignment:
    """H1: Joint dropna for opponent x/y prevents misalignment."""

    def test_partial_nan_no_crash(self):
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # One opponent has valid x but NaN y
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50.0, 20.0), (55.0, 30.0), (60.0, 40.0)],
        )
        # Inject NaN y on one opponent
        frames.loc[frames["player_id"] == 52, "y"] = np.nan
        result = detect_line_breaking(actions, frames, home_team_id=1)
        # Should not crash
        assert len(result) == 1


class TestH2ExtensionPoisoning:
    """H2: between_lines dominates when both extension + through intersect."""

    def test_between_lines_dominates(self):
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        # Setup: opponents arranged so pass intersects BOTH an extension
        # segment AND a between-players segment of the same cluster.
        # Cluster 1 at x~50: players at (50,10), (50,30), (50,50)
        # Cluster 2 at x~70: players at (70,10), (70,30), (70,50)
        # Pass from (40,34) to (80,34) goes through both clusters.
        # The pass trajectory y=34 goes between y=30 and y=50 (between_lines)
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[
                (50, 10),
                (50, 30),
                (50, 50),
                (70, 10),
                (70, 30),
                (70, 50),
            ],
            start_xy=(40.0, 34.0),
            end_xy=(80.0, 34.0),
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        # When both extension AND between-players segments intersect,
        # type should be "between_lines" (not "around_line")
        if result["line_break__ward"].iloc[0]:
            assert result["line_breaking_type__ward"].iloc[0] == "between_lines"


class TestM4NonPassFiltering:
    """M4: Non-pass actions produce pd.NA."""

    def test_shot_produces_na(self):
        from silly_kicks.spadl import config as spadlconfig
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        shot_type_id = spadlconfig.actiontype_id["shot"]
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50, 20), (55, 30), (60, 40)],
            action_type_id=shot_type_id,
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert pd.isna(result["line_break__ward"].iloc[0])

    def test_dribble_produces_na(self):
        from silly_kicks.spadl import config as spadlconfig
        from silly_kicks.tracking._line_breaking import detect_line_breaking

        dribble_type_id = spadlconfig.actiontype_id["dribble"]
        actions, frames = _make_line_breaking_fixture(
            opp_positions=[(50, 20), (55, 30), (60, 40)],
            action_type_id=dribble_type_id,
        )
        result = detect_line_breaking(actions, frames, home_team_id=1)
        assert pd.isna(result["line_break__ward"].iloc[0])
