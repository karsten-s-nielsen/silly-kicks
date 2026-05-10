"""Shared test helpers for TF-15 GK influence tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_two_team_frame(
    *,
    home_positions: list[tuple[float, float]],
    away_positions: list[tuple[float, float]],
    home_gk_pos: tuple[float, float] = (3.0, 34.0),
    away_gk_pos: tuple[float, float] = (102.0, 34.0),
    home_team_id: int = 1,
    away_team_id: int = 2,
    home_velocities: list[tuple[float, float]] | None = None,
    away_velocities: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build a two-team frame with GKs, outfield players, and ball."""
    rows = []
    # Ball
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
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
    # Home GK
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
            x=home_gk_pos[0],
            y=home_gk_pos[1],
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Away GK
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
            x=away_gk_pos[0],
            y=away_gk_pos[1],
            vx=0.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    # Home outfield
    for i, (px, py) in enumerate(home_positions):
        vx_v = home_velocities[i][0] if home_velocities else 0.0
        vy_v = home_velocities[i][1] if home_velocities else 0.0
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
                x=px,
                y=py,
                vx=vx_v,
                vy=vy_v,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    # Away outfield
    for i, (px, py) in enumerate(away_positions):
        vx_v = away_velocities[i][0] if away_velocities else 0.0
        vy_v = away_velocities[i][1] if away_velocities else 0.0
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
                x=px,
                y=py,
                vx=vx_v,
                vy=vy_v,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)
