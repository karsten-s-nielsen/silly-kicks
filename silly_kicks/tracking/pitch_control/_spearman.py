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

from ._params import SpearmanParams
from ._surface import PitchControlSurface


def _kernels():
    """The numba kernels when the PITCH_CONTROL family selects numba, else None (HFU spec 13.2: numba is core; only
    ``SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY=1`` selects the numpy reference, and a missing numba raises)."""
    from silly_kicks import _numba_dispatch

    if not _numba_dispatch.use_numba("PITCH_CONTROL"):
        return None
    return _numba_dispatch.import_kernels("silly_kicks.tracking.pitch_control._numba_kernels")


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
    kernels = _kernels()
    if kernels is not None:
        return kernels.tti_numba(pos, vel, targets, reaction_time, max_acceleration)

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
    kernels = _kernels()
    if kernels is not None:
        return kernels.influence_numba(team_tti, opponent_min_tti, sigma)

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
    Compute a Spearman kinematic pitch-control surface and query a cell::

        from silly_kicks.tracking.pitch_control._spearman import compute_spearman
        from silly_kicks.tracking.pitch_control._params import SpearmanParams
        surface = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
        surface.at_point(52.5, 34.0)  # -> 0.55
    """
    # One frame is a one-frame index through the batched core: one implementation for both paths (spec
    # 2026-11-08 tracking-batch section 4.2 W2.6), so the per-frame and the batch surfaces are bit-identical.
    from ._index import _SpearmanFrameIndex
    from ._spearman_batch import _spearman_batch_core

    index = _SpearmanFrameIndex.from_slices([frame])
    return _spearman_batch_core(index, [(0, attacking_team_id, decompose, ball_position)], params)[0]


def _empty_spearman_surface(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    params: SpearmanParams,
    attacking_team_id: int | str,
) -> PitchControlSurface:
    """The players-empty degenerate surface: a uniform 0.5 field (no attacker, no defender)."""
    surface = np.full((params.grid_cells_y, params.grid_cells_x), 0.5)
    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="spearman",
        attacking_team_id=attacking_team_id,
    )
