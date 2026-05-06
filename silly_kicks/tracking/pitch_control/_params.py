"""Pitch control model parameters — frozen dataclasses per flavor.

Three published methodologies:
  - spearman         -- Spearman 2017 kinematic TTI (ratio approximation)
  - fernandez_bornn  -- Fernandez & Bornn 2018 bivariate normal influence
  - voronoi          -- Nearest-player tessellation (baseline)

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md sections 5, 6.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Method = Literal["spearman", "fernandez_bornn", "voronoi"]


@dataclass(frozen=True)
class SpearmanParams:
    """Parameters for Spearman 2017 kinematic pitch control (ratio approximation).

    Uses acceleration-based TTI (not Shaw's constant-velocity max_speed model).
    See spec section 6.1 Note on model lineage for full provenance.

    Examples
    --------
    >>> p = SpearmanParams(sigma=0.5)
    >>> p.sigma
    0.5
    """

    reaction_time: float = 0.7
    """Seconds before player begins moving toward target."""
    max_acceleration: float = 7.0
    """Peak acceleration in m/s^2 (lakehouse-calibrated)."""
    sigma: float = 0.45
    """Logistic curve steepness in seconds (Shaw 2020)."""
    lambda_gk: float = 3.0
    """GK control-rate multiplier (Shaw: lambda_gk = 3 * lambda_outfield)."""
    average_ball_speed: float = 15.0
    """Ball speed in m/s for travel-time filter (Shaw 2020)."""
    grid_cells_x: int = 50
    """Grid resolution along pitch length (105 m)."""
    grid_cells_y: int = 32
    """Grid resolution along pitch width (68 m)."""


@dataclass(frozen=True)
class FernandezBornnParams:
    """Parameters for Fernandez & Bornn 2018 bivariate-normal influence model.

    Radius formula from DataBallPy (visual inspection of paper appendix figure).
    See spec section 5.2 provenance note.

    Examples
    --------
    >>> p = FernandezBornnParams(max_speed=12.0)
    >>> p.max_speed
    12.0
    """

    max_speed: float = 13.0
    """Elite sprint ceiling in m/s — normalizes velocity scaling alpha."""
    min_radius: float = 4.0
    """Minimum influence radius in meters (near ball)."""
    max_radius: float = 10.0
    """Maximum influence radius in meters (far from ball)."""
    grid_cells_x: int = 50
    """Grid resolution along pitch length (105 m)."""
    grid_cells_y: int = 32
    """Grid resolution along pitch width (68 m)."""


@dataclass(frozen=True)
class VoronoiParams:
    """Parameters for Voronoi tessellation baseline.

    Nearest-player assignment — no physics parameters. Grid resolution
    controls rasterization only.

    Examples
    --------
    >>> p = VoronoiParams(grid_cells_x=100, grid_cells_y=64)
    >>> p.grid_cells_x
    100
    """

    grid_cells_x: int = 50
    """Grid resolution along pitch length (105 m)."""
    grid_cells_y: int = 32
    """Grid resolution along pitch width (68 m)."""


PitchControlParams = SpearmanParams | FernandezBornnParams | VoronoiParams

_METHOD_TO_PARAMS_TYPE: dict[Method, type] = {
    "spearman": SpearmanParams,
    "fernandez_bornn": FernandezBornnParams,
    "voronoi": VoronoiParams,
}


def validate_params_for_method(method: Method, params: PitchControlParams | None) -> None:
    """Raise if method/params combination is invalid.

    None means use defaults for the chosen method.

    Examples
    --------
    >>> validate_params_for_method("spearman", SpearmanParams())
    >>> validate_params_for_method("voronoi", None)
    """
    if method not in _METHOD_TO_PARAMS_TYPE:
        raise ValueError(f"Unknown method '{method}'. Valid: {sorted(_METHOD_TO_PARAMS_TYPE)}")
    if params is None:
        return
    expected = _METHOD_TO_PARAMS_TYPE[method]
    if not isinstance(params, expected):
        raise TypeError(
            f"method='{method}' expects {expected.__name__}, "
            f"got {type(params).__name__}. "
            f"Use {expected.__name__}() (or omit params=) for defaults."
        )
