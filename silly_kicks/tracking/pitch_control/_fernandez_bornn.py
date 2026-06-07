"""Fernandez & Bornn 2018 bivariate-normal pitch control.

Each player projects a directional Gaussian influence field. Team aggregation
via sigmoid: control = sigmoid(sum_att - sum_def).

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 6.2.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._id_compat import ids_match
from ._params import FernandezBornnParams
from ._surface import PitchControlSurface

try:
    from ._numba_kernels import gaussian_influence_numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Velocity guards
SPEED_FLOOR = 0.1  # m/s — below meaningful movement -> isotropic
ALPHA_CEIL = 0.99  # prevents singular covariance (minor eigenvalue >= 0.01*R^2)


def _compute_gaussian_influence(
    targets: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    det_cov: np.ndarray,
) -> np.ndarray:
    """Evaluate bivariate Gaussian for each player at each target.

    Parameters
    ----------
    targets : (n_targets, 2)
    mu : (n_players, 2) — shifted mean positions
    inv_cov : (n_players, 2, 2) — inverse covariance per player
    det_cov : (n_players,) — determinant of covariance per player

    Returns
    -------
    (n_players, n_targets) normalized influence values in [0, 1].
    """
    if _HAS_NUMBA:
        return gaussian_influence_numba(targets, mu, inv_cov, det_cov)

    # diff: (n_players, n_targets, 2)
    diff = targets[np.newaxis, :, :] - mu[:, np.newaxis, :]

    # Mahalanobis distance squared via einsum: diff @ inv_cov @ diff^T
    # (n_players, n_targets, 2) @ (n_players, 1, 2, 2) -> contract
    tmp = np.einsum("ptj,pjk->ptk", diff, inv_cov)  # (n_players, n_targets, 2)
    mahal_sq = np.einsum("ptk,ptk->pt", tmp, diff)  # (n_players, n_targets)

    # Gaussian (unnormalized — we normalize per player to [0,1])
    raw = np.exp(-0.5 * mahal_sq)

    # Normalize each player's max to 1.0 (at mu, mahal_sq=0 -> raw=1)
    # But handle edge case where player might have all-zero influence
    player_max = raw.max(axis=1, keepdims=True)
    safe_max = np.maximum(player_max, 1e-20)
    influence = raw / safe_max

    return influence


def compute_fernandez_bornn(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    params: FernandezBornnParams,
    *,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Fernandez/Bornn bivariate-normal pitch control.

    Each player projects an anisotropic Gaussian influence field shaped by
    velocity direction. Team control is sigmoid(att_sum - def_sum).

    Examples
    --------
    >>> from silly_kicks.tracking.pitch_control._fernandez_bornn import compute_fernandez_bornn
    >>> from silly_kicks.tracking.pitch_control._params import FernandezBornnParams
    >>> surface = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
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
            method="fernandez_bornn",
            attacking_team_id=attacking_team_id,
        )

    # Build target grid
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])  # (n_targets, 2)
    targets.shape[0]
    n_players = len(players)

    # Extract player data
    pos = players[["x", "y"]].to_numpy(dtype="float64")
    vel_cols = ["vx", "vy"] if "vx" in players.columns else []
    if vel_cols:
        vel = players[vel_cols].to_numpy(dtype="float64")
    else:
        vel = np.zeros_like(pos)
    vel = np.nan_to_num(vel, nan=0.0)

    is_attacking = ids_match(players["team_id"], attacking_team_id).to_numpy()
    player_ids_arr = players["player_id"].to_numpy()

    # --- Per-player Gaussian parameters ---

    # 1. Speed and velocity direction
    speed = np.sqrt((vel**2).sum(axis=1))  # (n_players,)
    theta = np.arctan2(vel[:, 1], vel[:, 0])  # (n_players,)

    # 2. Alpha (velocity scaling) with guards
    alpha = np.clip((speed / params.max_speed) ** 2, 0.0, ALPHA_CEIL)
    alpha[speed < SPEED_FLOOR] = 0.0  # isotropic for stationary

    # 3. Influence radius (ball-aware)
    if ball_position is not None:
        ball_pos = np.array(ball_position, dtype="float64")
        dist_to_ball = np.sqrt(((pos - ball_pos[np.newaxis, :]) ** 2).sum(axis=1))
    else:
        # Default: use ball row position if available, else center
        ball_rows = frame[frame["is_ball"].astype(bool)]
        if not ball_rows.empty:
            bx = float(ball_rows["x"].iloc[0])
            by = float(ball_rows["y"].iloc[0])
            dist_to_ball = np.sqrt((pos[:, 0] - bx) ** 2 + (pos[:, 1] - by) ** 2)
        else:
            dist_to_ball = np.full(n_players, 52.5)  # default mid-field

    radius = np.clip(
        params.min_radius + dist_to_ball**3 / 972.0,
        params.min_radius,
        params.max_radius,
    )

    # 4. Mean position (anticipation shift)
    mu = pos + 0.5 * vel  # (n_players, 2)

    # 5. Build covariance matrices
    # Scaling matrix S: [[radius*(1+alpha), 0], [0, radius*(1-alpha)]]
    sx = radius * (1.0 + alpha)  # (n_players,)
    sy = radius * (1.0 - alpha)  # (n_players,)

    # Rotation matrix R_theta
    cos_t = np.cos(theta)  # (n_players,)
    sin_t = np.sin(theta)

    # Covariance = R_theta @ S @ S^T @ R_theta^T
    # Since S is diagonal: Sigma = R_theta @ diag(sx^2, sy^2) @ R_theta^T
    sx2 = sx**2
    sy2 = sy**2

    # Sigma[0,0] = cos^2 * sx^2 + sin^2 * sy^2
    # Sigma[0,1] = Sigma[1,0] = cos*sin*(sx^2 - sy^2)
    # Sigma[1,1] = sin^2 * sx^2 + cos^2 * sy^2
    cov_00 = cos_t**2 * sx2 + sin_t**2 * sy2
    cov_01 = cos_t * sin_t * (sx2 - sy2)
    cov_11 = sin_t**2 * sx2 + cos_t**2 * sy2

    # Determinant and inverse
    det_cov = cov_00 * cov_11 - cov_01**2  # (n_players,)
    safe_det = np.maximum(det_cov, 1e-20)

    # inv_cov: (n_players, 2, 2)
    inv_cov = np.zeros((n_players, 2, 2))
    inv_cov[:, 0, 0] = cov_11 / safe_det
    inv_cov[:, 0, 1] = -cov_01 / safe_det
    inv_cov[:, 1, 0] = -cov_01 / safe_det
    inv_cov[:, 1, 1] = cov_00 / safe_det

    # 6. Evaluate Gaussian influence
    influence = _compute_gaussian_influence(targets, mu, inv_cov, det_cov)
    # influence shape: (n_players, n_targets)

    # --- Team aggregation: sigmoid(att_sum - def_sum) ---
    att_sum = influence[is_attacking].sum(axis=0)  # (n_targets,)
    def_sum = influence[~is_attacking].sum(axis=0)
    surface_flat = 1.0 / (1.0 + np.exp(-(att_sum - def_sum)))
    surface = surface_flat.reshape(params.grid_cells_y, params.grid_cells_x)

    # Decomposition: store raw Gaussian influence per player
    per_player = None
    p_ids = None
    p_team_ids = None
    if decompose:
        per_player = influence.reshape(n_players, params.grid_cells_y, params.grid_cells_x)
        p_ids = player_ids_arr
        p_team_ids = players["team_id"].to_numpy()

    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="fernandez_bornn",
        attacking_team_id=attacking_team_id,
        per_player_influence=per_player,
        player_ids=p_ids,
        player_team_ids=p_team_ids,
    )
