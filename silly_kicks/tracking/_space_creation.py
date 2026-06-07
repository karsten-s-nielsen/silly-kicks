"""Space Creation quantification (Fernandez & Bornn 2018).

Measures each player's contribution to the team's off-ball scoring
opportunity by computing differential OBSO: how much the team's
OBSO surface changes when that player is removed.

See NOTICE for full bibliographic citations.

References
----------
Fernandez, J. & Bornn, L. (2018). "Wide Open Spaces: A statistical
technique for measuring space creation in professional soccer."
MIT Sloan Sports Analytics Conference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._id_compat import ids_match

if TYPE_CHECKING:
    from .pitch_control import PitchControlCache


@dataclass(frozen=True)
class SpaceCreationParams:
    """Parameters for space creation computation.

    All spatial parameters are in meters on a [0, pitch_length] x [0, pitch_width]
    coordinate system.

    Examples
    --------
    >>> params = SpaceCreationParams()
    >>> params.pitch_length
    105.0
    """

    pitch_length: float = 105.0
    pitch_width: float = 68.0


def compute_space_created(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    ball_position: tuple[float, float] | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    params: SpaceCreationParams | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    obso_sigma_x: float = 26.25,
    obso_sigma_y: float = 17.0,
    pitch_control_cache: PitchControlCache | None = None,
) -> pd.DataFrame:
    """Per-player space creation via leave-one-out differential OBSO.

    For each attacking-team outfield player, removes that player from the
    frame, re-computes pitch control, and measures the resulting change in
    OBSO surface.  Positive delta = space the player creates; negative =
    space they destroy.

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data with columns per TRACKING_FRAMES_COLUMNS.
    attacking_team_id : int or str
        Team in possession.
    ball_position : tuple[float, float] or None
        (x, y) in meters.  If None, inferred from ball row.
    transition_grid : np.ndarray or None
        Pre-computed ball transition probability grid.
    epv_grid : np.ndarray or None
        Pre-computed EPV grid.
    params : SpaceCreationParams or None
        Grid/pitch parameters.  None uses defaults.
    pitch_control_method : str
        Pitch control model (default ``"spearman"``).
    obso_sigma_x, obso_sigma_y : float
        Gaussian decay sigmas for OBSO distance weighting (meters).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id``, ``team_id``, ``space_created_m2``,
        ``space_destroyed_m2``, ``net_space_m2``.

    Examples
    --------
    >>> result = compute_space_created(frame, attacking_team_id=1)
    >>> result.columns.tolist()
    ['player_id', 'team_id', 'space_created_m2', 'space_destroyed_m2', 'net_space_m2']
    """
    from ._obso import _get_default_grids, _interpolate_grid
    from .pitch_control import PitchControlCache

    if params is None:
        params = SpaceCreationParams()

    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)

    # Ensure velocity columns
    if "vx" not in frame.columns or "vy" not in frame.columns:
        frame = frame.copy()
        if "vx" not in frame.columns:
            frame["vx"] = 0.0
        if "vy" not in frame.columns:
            frame["vy"] = 0.0

    # Resolve ball position
    if ball_position is None:
        ball_rows = frame[frame["is_ball"] == True]  # noqa: E712
        if len(ball_rows) > 0:
            ball_position = (float(ball_rows.iloc[0]["x"]), float(ball_rows.iloc[0]["y"]))
        else:
            ball_position = (params.pitch_length / 2, params.pitch_width / 2)

    # 1. Compute baseline pitch control with per-player decomposition
    #    (decompose=True is free for Spearman/F&B — same influence computation,
    #    just retains the per-player arrays instead of discarding them)
    use_analytical = pitch_control_method in ("spearman", "fernandez_bornn")
    # Canonical-frame baseline via the shared cache (TF-7 shared surface). The
    # leave-one-out counterfactuals below operate on modified frames and stay
    # uncached.
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()
    baseline_surface = cache.surface(
        frame,
        attacking_team_id,
        method=pitch_control_method,
        decompose=use_analytical,
        ball_position=ball_position,
    )

    grid_x = np.asarray(baseline_surface.grid_x)
    grid_y = np.asarray(baseline_surface.grid_y)
    ny, nx = baseline_surface.surface.shape

    # 2. Hoist loop-invariant: OBSO multiplier
    transition_interp = _interpolate_grid(transition_grid, (ny, nx))
    epv_interp = _interpolate_grid(epv_grid, (ny, nx))

    ball_x, ball_y = ball_position
    xx, yy = np.meshgrid(grid_x, grid_y)
    distance_weight = np.exp(
        -((xx - ball_x) ** 2) / (2.0 * obso_sigma_x**2) - (yy - ball_y) ** 2 / (2.0 * obso_sigma_y**2)
    )
    effective_transition = transition_interp * distance_weight
    max_trans = np.max(effective_transition)
    if max_trans > 1e-10:
        effective_transition = effective_transition / max_trans
    obso_multiplier = effective_transition * epv_interp  # (ny, nx) -- constant

    # Baseline OBSO
    baseline_obso = np.clip(np.asarray(baseline_surface.surface) * obso_multiplier, 0.0, 1.0)

    # 3. Identify attacking-team players (including GK)
    atk_mask = ids_match(frame["team_id"], attacking_team_id) & (frame["is_ball"] != True)  # noqa: E712
    atk_players = frame.loc[atk_mask]

    if atk_players.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "team_id",
                "space_created_m2",
                "space_destroyed_m2",
                "net_space_m2",
            ]
        )

    # Cell area
    dx = float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0
    dy = float(grid_y[1] - grid_y[0]) if len(grid_y) > 1 else 1.0
    cell_area = dx * dy

    # 4. Leave-one-out: analytical delta (Spearman/F&B) or naive (Voronoi)
    if use_analytical:
        results = _analytical_leave_one_out(
            baseline_surface,
            baseline_obso,
            obso_multiplier,
            attacking_team_id,
            atk_players,
            cell_area,
            pitch_control_method,
        )
    else:
        results = _naive_leave_one_out(
            frame,
            baseline_obso,
            obso_multiplier,
            attacking_team_id,
            atk_players,
            cell_area,
            pitch_control_method,
            ball_position,
        )

    return pd.DataFrame(results)


def _analytical_leave_one_out(
    baseline_surface: object,
    baseline_obso: np.ndarray,
    obso_multiplier: np.ndarray,
    attacking_team_id: int | str,
    atk_players: pd.DataFrame,
    cell_area: float,
    method: Literal["spearman", "fernandez_bornn"],
) -> list[dict]:
    """Analytical per-player delta — 1 PC computation instead of N+1.

    Exploits additive decomposition of Spearman (ratio) and Fernandez-Bornn
    (sigmoid) pitch control models. Voronoi is NOT decomposable and must
    use the naive fallback.
    """
    # per_player_influence: (n_players, ny, nx) — post-GK-weighting influence
    ppi = np.asarray(baseline_surface.per_player_influence)  # type: ignore[union-attr]
    p_ids = np.asarray(baseline_surface.player_ids)  # type: ignore[union-attr]
    p_teams = np.asarray(baseline_surface.player_team_ids)  # type: ignore[union-attr]

    # Dtype-safe id match (ADR-019): canonical collapses 1.0/1/"1" consistently, so it is correct
    # across numeric/string callers (the old raw == broke when p_teams was string + team numeric).
    is_atk = ids_match(p_teams, attacking_team_id).to_numpy()
    att_total = ppi[is_atk].sum(axis=0)  # (ny, nx)
    def_total = ppi[~is_atk].sum(axis=0)  # (ny, nx)

    results: list[dict] = []
    for player_row in atk_players.itertuples():
        pid = player_row.player_id

        # Find this player in the decomposed arrays
        pid_matches = np.flatnonzero(p_ids == pid)
        if len(pid_matches) == 0:
            # Player not in PC (dropped by NaN filter) → zero contribution
            results.append(
                {
                    "player_id": pid,
                    "team_id": attacking_team_id,
                    "space_created_m2": 0.0,
                    "space_destroyed_m2": 0.0,
                    "net_space_m2": 0.0,
                }
            )
            continue

        player_inf = ppi[pid_matches[0]]  # (ny, nx)
        removed_att = att_total - player_inf

        # Reconstruct PC surface without this player
        if method == "spearman":
            total = removed_att + def_total
            safe_total = np.maximum(total, 1e-10)
            removed_pc = np.where(total > 1e-10, removed_att / safe_total, 0.5)
        else:  # fernandez_bornn
            removed_pc = 1.0 / (1.0 + np.exp(-(removed_att - def_total)))

        removed_obso = np.clip(removed_pc * obso_multiplier, 0.0, 1.0)
        delta = baseline_obso - removed_obso
        space_created = float(np.sum(np.maximum(delta, 0.0)) * cell_area)
        space_destroyed = float(np.sum(np.abs(np.minimum(delta, 0.0))) * cell_area)

        results.append(
            {
                "player_id": pid,
                "team_id": attacking_team_id,
                "space_created_m2": space_created,
                "space_destroyed_m2": space_destroyed,
                "net_space_m2": space_created - space_destroyed,
            }
        )

    return results


def _naive_leave_one_out(
    frame: pd.DataFrame,
    baseline_obso: np.ndarray,
    obso_multiplier: np.ndarray,
    attacking_team_id: int | str,
    atk_players: pd.DataFrame,
    cell_area: float,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"],
    ball_position: tuple[float, float],
) -> list[dict]:
    """Naive N-recompute fallback for non-decomposable models (Voronoi)."""
    from .pitch_control import compute_pitch_control

    results: list[dict] = []
    for player_row in atk_players.itertuples():
        pid = player_row.player_id

        removed_frame = frame[
            ~((frame["player_id"] == pid) & (frame["is_ball"] != True))  # noqa: E712
        ]

        removed_surface = compute_pitch_control(
            removed_frame,
            attacking_team_id,
            method=pitch_control_method,
            ball_position=ball_position,
        )

        removed_obso = np.clip(np.asarray(removed_surface.surface) * obso_multiplier, 0.0, 1.0)
        delta = baseline_obso - removed_obso
        space_created = float(np.sum(np.maximum(delta, 0.0)) * cell_area)
        space_destroyed = float(np.sum(np.abs(np.minimum(delta, 0.0))) * cell_area)

        results.append(
            {
                "player_id": pid,
                "team_id": attacking_team_id,
                "space_created_m2": space_created,
                "space_destroyed_m2": space_destroyed,
                "net_space_m2": space_created - space_destroyed,
            }
        )

    return results
