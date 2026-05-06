"""Optional numba-accelerated kernels for pitch control computation.

These mirror the NumPy implementations in _spearman.py and _fernandez_bornn.py
but use @numba.njit for ~5-10x speedup on large grids.

Import pattern:
    try:
        from ._numba_kernels import tti_numba, influence_numba, gaussian_influence_numba
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 9.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("numba is required for _numba_kernels. Install with: pip install silly-kicks[numba]") from e


@njit(cache=True)
def tti_numba(
    pos: np.ndarray,
    vel: np.ndarray,
    targets: np.ndarray,
    reaction_time: float,
    max_acceleration: float,
) -> np.ndarray:
    """Numba-accelerated TTI computation.

    Parameters
    ----------
    pos : (n_players, 2)
    vel : (n_players, 2)
    targets : (n_targets, 2)
    reaction_time : float
    max_acceleration : float

    Returns
    -------
    (n_players, n_targets) TTI values.
    """
    n_players = pos.shape[0]
    n_targets = targets.shape[0]
    result = np.empty((n_players, n_targets))

    for i in range(n_players):
        px, py = pos[i, 0], pos[i, 1]
        vx, vy = vel[i, 0], vel[i, 1]
        for j in range(n_targets):
            dx = targets[j, 0] - px
            dy = targets[j, 1] - py
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-10:
                result[i, j] = reaction_time
            else:
                ux = dx / d
                uy = dy / d
                v_proj = vx * ux + vy * uy
                disc = v_proj * v_proj + 2.0 * max_acceleration * d
                result[i, j] = reaction_time + (-v_proj + np.sqrt(disc)) / max_acceleration
    return result


@njit(cache=True)
def influence_numba(
    team_tti: np.ndarray,
    opponent_min_tti: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Numba-accelerated logistic influence computation.

    Parameters
    ----------
    team_tti : (n_players, n_targets)
    opponent_min_tti : (n_targets,)
    sigma : float

    Returns
    -------
    (n_players, n_targets) influence values.
    """
    k = np.pi / (np.sqrt(3.0) * sigma)
    n_players = team_tti.shape[0]
    n_targets = team_tti.shape[1]
    result = np.empty((n_players, n_targets))

    for i in range(n_players):
        for j in range(n_targets):
            exponent = -k * (opponent_min_tti[j] - team_tti[i, j])
            result[i, j] = 1.0 / (1.0 + np.exp(exponent))
    return result


@njit(cache=True)
def gaussian_influence_numba(
    targets: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    det_cov: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated bivariate Gaussian influence.

    Parameters
    ----------
    targets : (n_targets, 2)
    mu : (n_players, 2)
    inv_cov : (n_players, 2, 2)
    det_cov : (n_players,)

    Returns
    -------
    (n_players, n_targets) normalized influence values in [0, 1].
    """
    n_players = mu.shape[0]
    n_targets = targets.shape[0]
    result = np.empty((n_players, n_targets))

    for i in range(n_players):
        max_val = 0.0
        for j in range(n_targets):
            dx = targets[j, 0] - mu[i, 0]
            dy = targets[j, 1] - mu[i, 1]
            # Mahalanobis: [dx, dy] @ inv_cov @ [dx, dy]^T
            m00 = inv_cov[i, 0, 0]
            m01 = inv_cov[i, 0, 1]
            m10 = inv_cov[i, 1, 0]
            m11 = inv_cov[i, 1, 1]
            mahal_sq = dx * (m00 * dx + m01 * dy) + dy * (m10 * dx + m11 * dy)
            val = np.exp(-0.5 * mahal_sq)
            result[i, j] = val
            if val > max_val:
                max_val = val

        # Normalize per player to [0, 1]
        if max_val > 1e-20:
            for j in range(n_targets):
                result[i, j] /= max_val

    return result
