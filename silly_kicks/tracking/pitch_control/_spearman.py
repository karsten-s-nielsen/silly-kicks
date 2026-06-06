"""Spearman 2017 kinematic pitch control — ratio approximation.

Three-stage pipeline:
  1. compute_tti() — acceleration-based time-to-intercept
  2. _compute_influence() — per-player logistic sigmoid
  3. compute_spearman() — GK weighting + ratio aggregation

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 6.1.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._id_compat import ids_match
from ._params import SpearmanParams
from ._surface import PitchControlSurface

try:
    from ._numba_kernels import influence_numba, tti_numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def compute_tti(
    pos: np.ndarray,
    vel: np.ndarray,
    targets: np.ndarray,
    reaction_time: float,
    max_acceleration: float,
) -> np.ndarray:
    """Broadcast-vectorized kinematic TTI.

    Parameters
    ----------
    pos : (n_players, 2)
    vel : (n_players, 2)
    targets : (n_targets, 2)
    reaction_time : seconds before movement begins
    max_acceleration : m/s^2

    Returns
    -------
    (n_players, n_targets) time-to-intercept in seconds.

    Examples
    --------
    >>> pos = np.array([[0.0, 0.0]])
    >>> vel = np.array([[3.0, 0.0]])
    >>> target = np.array([[10.0, 0.0]])
    >>> compute_tti(pos, vel, target, 0.7, 7.0)
    array([[...]])
    """
    if _HAS_NUMBA:
        return tti_numba(pos, vel, targets, reaction_time, max_acceleration)

    # displacement: (n_players, n_targets, 2)
    disp = targets[np.newaxis, :, :] - pos[:, np.newaxis, :]

    # distance: (n_players, n_targets)
    distance = np.sqrt((disp**2).sum(axis=2))

    # Unit vector toward target (avoid div-by-zero)
    safe_dist = np.maximum(distance, 1e-10)
    unit = disp / safe_dist[:, :, np.newaxis]

    # Velocity projection toward target: (n_players, n_targets)
    v_proj = (vel[:, np.newaxis, :] * unit).sum(axis=2)

    # Kinematic TTI: reaction_time + (-v_proj + sqrt(v_proj^2 + 2*a*d)) / a
    discriminant = v_proj**2 + 2.0 * max_acceleration * distance
    # discriminant is always >= 0 since distance >= 0
    tti = reaction_time + (-v_proj + np.sqrt(discriminant)) / max_acceleration

    # At target (d=0): TTI = reaction_time
    tti = np.where(distance < 1e-10, reaction_time, tti)

    return tti


def _compute_influence(
    team_tti: np.ndarray,
    opponent_min_tti: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Per-player logistic influence.

    Parameters
    ----------
    team_tti : (n_team_players, n_targets)
    opponent_min_tti : (n_targets,) — minimum TTI across opposing team
    sigma : logistic steepness parameter

    Returns
    -------
    (n_team_players, n_targets) influence values in [0, 1].

    Examples
    --------
    >>> team_tti = np.array([[1.0]])
    >>> opp_min = np.array([5.0])
    >>> _compute_influence(team_tti, opp_min, 0.45)
    array([[0.99...]])
    """
    if _HAS_NUMBA:
        return influence_numba(team_tti, opponent_min_tti, sigma)

    k = np.pi / (np.sqrt(3.0) * sigma)
    # Logistic: 1 / (1 + exp(-k * (opp_min - team_tti)))
    # Positive (opp_min > team_tti) -> player arrives first -> high influence
    exponent = -k * (opponent_min_tti[np.newaxis, :] - team_tti)
    influence = 1.0 / (1.0 + np.exp(exponent))
    return influence


def compute_spearman(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    params: SpearmanParams,
    *,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Spearman kinematic pitch control (ratio approximation).

    Computes per-cell control as the ratio of attacking influence to total
    influence, where influence is derived from acceleration-based TTI through
    a logistic sigmoid.

    Examples
    --------
    >>> from silly_kicks.tracking.pitch_control._spearman import compute_spearman
    >>> from silly_kicks.tracking.pitch_control._params import SpearmanParams
    >>> surface = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
    >>> surface.at_point(52.5, 34.0)
    0.55
    """
    grid_x = np.linspace(0, 105.0, params.grid_cells_x)
    grid_y = np.linspace(0, 68.0, params.grid_cells_y)

    # Filter players (no ball rows, no NaN positions)
    players = frame[~frame["is_ball"].astype(bool)].copy()
    players = players.dropna(subset=["x", "y"])

    if players.empty:
        surface = np.full((params.grid_cells_y, params.grid_cells_x), 0.5)
        return PitchControlSurface(
            grid_x=grid_x,
            grid_y=grid_y,
            surface=surface,
            method="spearman",
            attacking_team_id=attacking_team_id,
        )

    # Build target grid: (n_targets, 2)
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    n_targets = targets.shape[0]

    # Extract player data
    pos = players[["x", "y"]].to_numpy(dtype="float64")
    vel_cols = ["vx", "vy"] if "vx" in players.columns else []
    if vel_cols:
        vel = players[vel_cols].to_numpy(dtype="float64")
    else:
        vel = np.zeros_like(pos)
    # Fill NaN velocities with zero
    vel = np.nan_to_num(vel, nan=0.0)

    is_attacking = ids_match(players["team_id"], attacking_team_id).to_numpy()
    is_gk = players["is_goalkeeper"].astype(bool).to_numpy()
    player_ids_arr = players["player_id"].to_numpy()

    # Stage 1: Compute TTI for all players to all targets
    tti_all = compute_tti(pos, vel, targets, params.reaction_time, params.max_acceleration)

    # Ball-travel-time filter (optional)
    if ball_position is not None:
        ball_pos = np.array(ball_position, dtype="float64")
        ball_dist = np.sqrt(((targets - ball_pos[np.newaxis, :]) ** 2).sum(axis=1))
        ball_travel_time = ball_dist / params.average_ball_speed
        # Zero influence for players whose TTI > ball_travel_time
        too_slow = tti_all > ball_travel_time[np.newaxis, :]
    else:
        too_slow = None

    # Stage 2: Per-player influence via logistic
    # Compute minimum TTI per target for each team
    att_mask = is_attacking
    def_mask = ~is_attacking

    att_tti = tti_all[att_mask]  # (n_att, n_targets)
    def_tti = tti_all[def_mask]  # (n_def, n_targets)

    # Minimum opponent TTI at each target
    if def_tti.shape[0] > 0:
        def_min_tti = def_tti.min(axis=0)  # (n_targets,)
    else:
        def_min_tti = np.full(n_targets, np.inf)

    if att_tti.shape[0] > 0:
        att_min_tti = att_tti.min(axis=0)  # (n_targets,)
    else:
        att_min_tti = np.full(n_targets, np.inf)

    # Compute influence for attackers (opponent = defenders)
    if att_tti.shape[0] > 0:
        att_influence = _compute_influence(att_tti, def_min_tti, params.sigma)
    else:
        att_influence = np.zeros((0, n_targets))

    # Compute influence for defenders (opponent = attackers)
    if def_tti.shape[0] > 0:
        def_influence = _compute_influence(def_tti, att_min_tti, params.sigma)
    else:
        def_influence = np.zeros((0, n_targets))

    # Apply ball-travel-time filter
    if too_slow is not None:
        if att_influence.shape[0] > 0:
            att_influence[too_slow[att_mask]] = 0.0
        if def_influence.shape[0] > 0:
            def_influence[too_slow[def_mask]] = 0.0

    # GK weighting: scale GK rows by lambda_gk
    att_gk_mask = is_gk[att_mask]
    def_gk_mask = is_gk[def_mask]
    if att_gk_mask.any():
        att_influence[att_gk_mask] *= params.lambda_gk
    if def_gk_mask.any():
        def_influence[def_gk_mask] *= params.lambda_gk

    # Stage 3: Ratio aggregation
    att_sum = att_influence.sum(axis=0)  # (n_targets,)
    def_sum = def_influence.sum(axis=0)
    total = att_sum + def_sum
    safe_total = np.maximum(total, 1e-10)
    surface_flat = np.where(total > 1e-10, att_sum / safe_total, 0.5)
    surface = surface_flat.reshape(params.grid_cells_y, params.grid_cells_x)

    # Decomposition: store per-player influence in original player order
    per_player = None
    p_ids = None
    p_team_ids = None
    if decompose:
        n_players = len(players)
        per_player_flat = np.zeros((n_players, n_targets))
        # Place attacking influences at their original indices
        att_indices = np.flatnonzero(att_mask)
        for local_i, global_i in enumerate(att_indices):
            per_player_flat[global_i] = att_influence[local_i]
        # Place defending influences
        def_indices = np.flatnonzero(def_mask)
        for local_i, global_i in enumerate(def_indices):
            per_player_flat[global_i] = def_influence[local_i]
        per_player = per_player_flat.reshape(n_players, params.grid_cells_y, params.grid_cells_x)
        p_ids = player_ids_arr
        p_team_ids = players["team_id"].to_numpy()

    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="spearman",
        attacking_team_id=attacking_team_id,
        per_player_influence=per_player,
        player_ids=p_ids,
        player_team_ids=p_team_ids,
    )
