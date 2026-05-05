"""Tests for silly_kicks.tracking._gk_resolve.defending_gk_from_frames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_frames(
    *,
    home_team_id=1,
    away_team_id=2,
    gk_player_id_home=100,
    gk_player_id_away=200,
    frame_id=10,
    period_id=1,
    time_seconds=5.0,
    include_away_gk=True,
):
    """Minimal 1-frame tracking fixture with both teams + GKs."""
    rows = [
        # Ball
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            source_provider="sportec",
            team_attacking_direction="ltr",
        ),
        # Home GK
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=gk_player_id_home,
            team_id=home_team_id,
            is_ball=False,
            is_goalkeeper=True,
            x=5.0,
            y=34.0,
            source_provider="sportec",
            team_attacking_direction="ltr",
        ),
        # Home outfield
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=101,
            team_id=home_team_id,
            is_ball=False,
            is_goalkeeper=False,
            x=40.0,
            y=30.0,
            source_provider="sportec",
            team_attacking_direction="ltr",
        ),
        # Away outfield
        dict(
            game_id=1,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=time_seconds,
            frame_rate=25.0,
            player_id=201,
            team_id=away_team_id,
            is_ball=False,
            is_goalkeeper=False,
            x=60.0,
            y=34.0,
            source_provider="sportec",
            team_attacking_direction="ltr",
        ),
    ]
    if include_away_gk:
        rows.append(
            dict(
                game_id=1,
                period_id=period_id,
                frame_id=frame_id,
                time_seconds=time_seconds,
                frame_rate=25.0,
                player_id=gk_player_id_away,
                team_id=away_team_id,
                is_ball=False,
                is_goalkeeper=True,
                x=100.0,
                y=34.0,
                source_provider="sportec",
                team_attacking_direction="ltr",
            ),
        )
    return pd.DataFrame(rows)


def _make_actions(team_id=1, time_seconds=5.0, period_id=1):
    """Single-action DataFrame."""
    return pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [period_id],
            "time_seconds": [time_seconds],
            "team_id": [team_id],
            "player_id": [101],
            "start_x": [40.0],
            "start_y": [30.0],
            "type_id": [0],  # pass
        }
    )


class TestDefendingGkFromFrames:
    def test_resolves_opposing_gk(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions(team_id=1)  # home team acts -> should get away GK
        result = defending_gk_from_frames(actions, frames)
        assert result.iloc[0] == 200

    def test_all_actions_not_just_shots(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        # 3 actions: pass, dribble, tackle -- all should resolve
        actions = pd.DataFrame(
            {
                "action_id": [1, 2, 3],
                "period_id": [1, 1, 1],
                "time_seconds": [5.0, 5.0, 5.0],
                "team_id": [1, 1, 2],
                "player_id": [101, 101, 201],
                "start_x": [40.0, 40.0, 60.0],
                "start_y": [30.0, 30.0, 34.0],
                "type_id": [0, 5, 8],
            }
        )
        result = defending_gk_from_frames(actions, frames)
        assert result.iloc[0] == 200  # home acts -> away GK
        assert result.iloc[1] == 200
        assert result.iloc[2] == 100  # away acts -> home GK

    def test_nan_when_no_gk_in_frame(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(include_away_gk=False)
        actions = _make_actions(team_id=1)  # wants away GK, but none exists
        result = defending_gk_from_frames(actions, frames)
        assert pd.isna(result.iloc[0])

    def test_nan_when_unlinked(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(time_seconds=100.0)  # far from action time
        actions = _make_actions(time_seconds=5.0)
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.2)
        assert pd.isna(result.iloc[0])

    def test_nan_when_team_id_nan(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames()
        actions = _make_actions()
        actions["team_id"] = pd.array([pd.NA], dtype="Int64")
        result = defending_gk_from_frames(actions, frames)
        assert pd.isna(result.iloc[0])

    def test_dtype_matches_frames_object(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(
            gk_player_id_away="DFL-OBJ-200",
            away_team_id="team_b",
            home_team_id="team_a",
            gk_player_id_home="DFL-OBJ-100",
        )
        # Fix outfield player IDs to strings too
        frames["player_id"] = frames["player_id"].astype(object)
        frames["team_id"] = frames["team_id"].astype(object)
        actions = _make_actions(team_id="team_a")
        actions["player_id"] = actions["player_id"].astype(object)
        actions["team_id"] = actions["team_id"].astype(object)
        result = defending_gk_from_frames(actions, frames)
        assert result.dtype == object
        assert result.iloc[0] == "DFL-OBJ-200"

    def test_multi_gk_deterministic(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(gk_player_id_away=200)
        # Add second GK on away team (substitution overlap)
        extra = pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=10,
                    time_seconds=5.0,
                    frame_rate=25.0,
                    player_id=199,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=True,
                    x=100.0,
                    y=34.0,
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                )
            ]
        )
        frames = pd.concat([frames, extra], ignore_index=True)
        actions = _make_actions(team_id=1)
        result = defending_gk_from_frames(actions, frames)
        # Lowest player_id wins
        assert result.iloc[0] == 199

    def test_tolerance_respected(self):
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        frames = _make_frames(time_seconds=5.3)  # 0.3s offset
        actions = _make_actions(time_seconds=5.0)
        # tolerance=0.2 should miss
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.2)
        assert pd.isna(result.iloc[0])
        # tolerance=0.5 should hit
        result = defending_gk_from_frames(actions, frames, tolerance_seconds=0.5)
        assert result.iloc[0] == 200
