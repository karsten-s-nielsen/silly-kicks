"""Voronoi tessellation pitch control — nearest-player assignment.

Binary control surface: 1.0 (attacking) or 0.0 (defending) per cell.
No physics, no probabilities — baseline for validation and fast spatial queries.

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 6.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

from ._params import VoronoiParams
from ._surface import PitchControlSurface


def compute_voronoi(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    params: VoronoiParams,
    *,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Voronoi tessellation pitch control.

    For each grid cell, assigns control to the team of the nearest player.
    ball_position accepted for API consistency but ignored.

    Examples
    --------
    >>> from silly_kicks.tracking.pitch_control._voronoi import compute_voronoi
    >>> from silly_kicks.tracking.pitch_control._params import VoronoiParams
    >>> surface = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
    >>> surface.at_point(50, 34)
    1.0
    """
    grid_x = np.linspace(0, 105.0, params.grid_cells_x)
    grid_y = np.linspace(0, 68.0, params.grid_cells_y)
    n_cells = params.grid_cells_x * params.grid_cells_y

    # Filter players (no ball rows, no NaN positions)
    players = frame[~frame["is_ball"].astype(bool)].copy()
    players = players.dropna(subset=["x", "y"])

    if players.empty:
        surface = np.full((params.grid_cells_y, params.grid_cells_x), 0.5)
        return PitchControlSurface(
            grid_x=grid_x,
            grid_y=grid_y,
            surface=surface,
            method="voronoi",
            attacking_team_id=attacking_team_id,
        )

    # Build target grid
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])  # (n_cells, 2)

    # Player positions
    player_pos = players[["x", "y"]].to_numpy(dtype="float64")  # (n_players, 2)
    player_ids_arr = players["player_id"].to_numpy()
    is_attacking = ids_match(players["team_id"], attacking_team_id).to_numpy()

    # Broadcast distance: (n_cells, n_players)
    diff = targets[:, np.newaxis, :] - player_pos[np.newaxis, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))

    # Nearest player per cell
    nearest_idx = distances.argmin(axis=1)  # (n_cells,)

    # Assign control based on nearest player's team
    control_flat = np.where(is_attacking[nearest_idx], 1.0, 0.0)
    surface = control_flat.reshape(params.grid_cells_y, params.grid_cells_x)

    # Decomposition
    per_player = None
    p_ids = None
    p_team_ids = None
    if decompose:
        n_players = len(players)
        per_player_flat = np.zeros((n_players, n_cells))
        per_player_flat[nearest_idx, np.arange(n_cells)] = 1.0
        per_player = per_player_flat.reshape(n_players, params.grid_cells_y, params.grid_cells_x)
        p_ids = player_ids_arr
        p_team_ids = players["team_id"].to_numpy()

    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="voronoi",
        attacking_team_id=attacking_team_id,
        per_player_influence=per_player,
        player_ids=p_ids,
        player_team_ids=p_team_ids,
    )
