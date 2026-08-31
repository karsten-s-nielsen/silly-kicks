"""OBPV field-value weighting for the deep-zone threat (TF-60 PR2, ADR-081).

Ogawa/Fujii et al. (2025) ("Space evaluation at the starting point of soccer transitions", OBPV)
weight space value as a LONGITUDINAL SIGMOID (in distance from the attacked goal) times a LATERAL
GAUSSIAN (in y), because a pure goal-proximity weighting misbehaves in the transition zone. This
module ships that FORM as an opt-in re-weighting of ``rd_danger_behind_line`` (gated by
``RestDefenseParams.danger_field_weight``). Defaults are un-tuned spec-time values (ADR-009); a
per-provider tune is a separate gated apply PR. See NOTICE (Ogawa 2025).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WFieldParams:
    """OBPV field-value weighting parameters (longitudinal sigmoid x lateral Gaussian).

    Examples
    --------
    >>> from silly_kicks.restdefense import WFieldParams
    >>> p = WFieldParams()
    >>> p.x_midpoint_m, p.y_sigma_m
    (30.0, 20.0)
    """

    x_midpoint_m: float = 30.0
    """Sigmoid midpoint: distance from the defended goal G_A (m)."""
    x_steepness_m: float = 8.0
    """Sigmoid width (m)."""
    y_center_m: float = 34.0
    """Lateral Gaussian centre (pitch middle)."""
    y_sigma_m: float = 20.0
    """Lateral Gaussian width (m)."""


def build_w_field(own_goal_x: float, params: WFieldParams) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return ``w(grid_x, grid_y) -> (ny, nx)`` OBPV weights, oriented toward the defended goal G_A.

    Absolute pitch coords: row ``i`` <-> ``grid_y[i]``, col ``j`` <-> ``grid_x[j]``. Highest near
    ``own_goal_x`` (deep zone), central-channel-weighted in y. Values in ``(0, 1]``.

    Examples
    --------
    Build the weight surface for a team defending x=0 and check its shape and orientation::

        import numpy as np
        from silly_kicks.restdefense import WFieldParams
        from silly_kicks.restdefense._wfield import build_w_field

        w = build_w_field(own_goal_x=0.0, params=WFieldParams())
        grid = w(np.linspace(0, 105, 50), np.linspace(0, 68, 32))
        grid.shape                       # -> (32, 50)
        grid[16, 0] > grid[16, -1]       # -> True (weight peaks near the defended goal x=0)
    """

    def w(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        d = np.abs(np.asarray(grid_x, dtype=float)[None, :] - own_goal_x)  # (1, nx) distance from G_A
        longitudinal = 1.0 / (1.0 + np.exp((d - params.x_midpoint_m) / params.x_steepness_m))  # (1, nx)
        y = np.asarray(grid_y, dtype=float)[:, None]  # (ny, 1)
        lateral = np.exp(-((y - params.y_center_m) ** 2) / (2.0 * params.y_sigma_m**2))  # (ny, 1)
        return lateral * longitudinal  # (ny, nx), broadcast

    return w
