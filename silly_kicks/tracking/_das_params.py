"""Native DAS simulation parameters, profiles and quadrature grids (TF-28; ADR-107/108).

``PassSimParams`` freezes the pass-simulation constants; ``DAS_PARAMS`` / ``XC_PARAMS`` are the two
profiles copied verbatim from ``accessible-space`` 2.0.15 (``core._DEFAULT_*`` and
``interface._DEFAULT_*_FOR_DAS`` / ``_FOR_XC``; provenance in the golden ``metadata.json``, asserted by
``tests/tracking/test_das_params.py``). ``simulation_grids`` builds the data-INDEPENDENT angle / speed /
radial grids and the polar area weights, cached on the frozen params.

The ``quadrature`` field is the ADR-108 decision: ``"periodic"`` (the shipped default) gives every
angular ray a full ``2*pi/n`` wedge; ``"reference"`` reproduces accessible-space's non-periodic
bug (rays 0 and n-1 get a half wedge) and is reachable ONLY from the parity gate, never the public
surface.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import numpy as np

Quadrature = Literal["periodic", "reference"]

# Pitch is fixed for the grid (centred coordinate frame); DAS/xC pitch bounds are constant.
_X_PITCH_MIN, _X_PITCH_MAX = -52.5, 52.5
_Y_PITCH_MIN, _Y_PITCH_MAX = -34.0, 34.0


@dataclass(frozen=True)
class PassSimParams:
    """Frozen pass-simulation constants for one profile (DAS or xC)."""

    n_angles: int
    phi_offset: float
    n_v0: int
    v0_min: float
    v0_max: float
    radial_gridsize: float
    pass_start_location_offset: float
    time_offset_ball: float
    b0: float
    b1: float
    player_velocity: float
    inertial_seconds: float
    tol_distance: float
    use_max: bool
    v_max: float
    a_max: float
    factor: float
    factor2: float
    normalize: bool
    respect_offside: bool
    exclude_passer: bool
    danger_weight: float
    quadrature: Quadrature = "periodic"

    def __post_init__(self) -> None:
        if self.n_angles < 1:
            raise ValueError(f"n_angles must be >= 1, got {self.n_angles}")
        if self.n_v0 < 1:
            raise ValueError(f"n_v0 must be >= 1, got {self.n_v0}")
        if not (self.v0_min > 0 and self.v0_max > self.v0_min):
            raise ValueError(f"require 0 < v0_min < v0_max, got {self.v0_min}, {self.v0_max}")
        if self.radial_gridsize <= 0:
            raise ValueError(f"radial_gridsize must be > 0, got {self.radial_gridsize}")
        if self.player_velocity <= 0:
            raise ValueError(f"player_velocity must be > 0, got {self.player_velocity}")
        if self.danger_weight == 0:
            raise ValueError("danger_weight must be non-zero (it is used as 1/danger_weight)")
        if self.quadrature not in ("periodic", "reference"):
            raise ValueError(f"quadrature must be 'periodic' or 'reference', got {self.quadrature!r}")


# accessible-space 2.0.15 DAS profile (interface._DEFAULT_*_FOR_DAS, core._DEFAULT_*).
DAS_PARAMS = PassSimParams(
    n_angles=30,
    phi_offset=0.0,
    n_v0=15,
    v0_min=3.0,
    v0_max=30.0,
    radial_gridsize=3.0,
    pass_start_location_offset=0.0,
    time_offset_ball=0.0,
    b0=-4.565680899844368,
    b1=-2000.0,
    player_velocity=9.0,
    inertial_seconds=0.17,
    tol_distance=5.0,
    use_max=False,
    v_max=19.85563874348074,
    a_max=10.659091365334193,
    factor=5.077423030272923,
    factor2=1.0063028450754512,
    normalize=True,
    respect_offside=True,
    exclude_passer=False,
    danger_weight=1.0,
    quadrature="periodic",
)

# accessible-space 2.0.15 xC profile (interface get_expected_pass_completion defaults + _FOR_XC).
# n_angles is unused (xC simulates one angle per pass); n_v0 = round(13.751097117532021) = 14.
XC_PARAMS = PassSimParams(
    n_angles=1,
    phi_offset=0.0,
    n_v0=14,
    v0_min=8.886015553615485,
    v0_max=42.18118275402132,
    radial_gridsize=5.034759576558597,
    pass_start_location_offset=-1.5245340256423476,
    time_offset_ball=-0.4384754490159207,
    b0=-4.565680899844368,
    b1=-188.74468208593532,
    player_velocity=34.6836072667285,
    inertial_seconds=1.1043767821571149,
    tol_distance=9.986761680941445,
    use_max=True,
    v_max=19.85563874348074,
    a_max=10.659091365334193,
    factor=5.077423030272923,
    factor2=1.0063028450754512,
    normalize=False,
    respect_offside=False,
    exclude_passer=True,
    danger_weight=1.0,
    quadrature="periodic",
)


@dataclass(frozen=True)
class SimGrids:
    """Data-independent simulation grids for one profile (all shapes are (Phi,) / (V,) / (T,) / (V,T))."""

    phi: np.ndarray
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    v0: np.ndarray
    d: np.ndarray
    t_ball: np.ndarray
    dt: np.ndarray
    dt0: np.ndarray
    rate_divisor: np.ndarray
    dr: np.ndarray
    d_area: np.ndarray  # (Phi, T) polar area weights (quadrature-dependent at rays 0 and n-1)


@functools.lru_cache(maxsize=16)
def simulation_grids(params: PassSimParams) -> SimGrids:
    """Build (and cache) the angle/speed/radial grids and polar area weights for ``params``.

    Reproduces accessible-space's grid construction verbatim (``core.simulate_passes`` /
    ``integrate_surfaces``); see plan Appendix A.1. Data-independent, so caching on the frozen params
    is correct.
    """
    n = params.n_angles
    phi = np.linspace(params.phi_offset, 2.0 * np.pi + params.phi_offset, n, endpoint=False)
    v0 = np.linspace(params.v0_min, params.v0_max, params.n_v0)

    max_pass_length = (
        np.sqrt((_X_PITCH_MAX - _X_PITCH_MIN) ** 2 + (_Y_PITCH_MAX - _Y_PITCH_MIN) ** 2) + params.radial_gridsize * 3
    )
    d = np.arange(
        params.pass_start_location_offset,
        max_pass_length + params.pass_start_location_offset + params.radial_gridsize,
        params.radial_gridsize,
    )

    t_ball = (d[np.newaxis, :] - d[0]) / v0[:, np.newaxis]
    t_ball = t_ball + params.time_offset_ball
    dt = np.diff(t_ball, axis=-1)
    dt0 = t_ball[:, 1] - t_ball[:, 0]
    rate_divisor = params.factor * (v0 ** (-params.factor2))

    # Radial area bounds (identical in both quadratures; end cells are half-width by construction).
    r_lo = np.zeros_like(d)
    r_lo[1:] = (d[:-1] + d[1:]) / 2
    r_lo[0] = d[0]
    r_hi = np.zeros_like(d)
    r_hi[:-1] = (d[:-1] + d[1:]) / 2
    r_hi[-1] = d[-1]
    dr = r_hi - r_lo

    # Angular wedge widths. INTERIOR bounds are accessible-space's midpoints in BOTH modes, so interior
    # d_area is bitwise identical across quadratures (the exact-relation gate, ADR-108). Only the two
    # end rays differ: reference copies the radial end-cell rule (phi_lo[0]=phi[0], phi_hi[-1]=phi[-1])
    # -> a HALF wedge at rays 0 and n-1 (the defect); periodic uses the WRAP-AROUND midpoints -> a full
    # wedge, correct for a cyclic axis.
    phi_lo = np.zeros_like(phi)
    phi_lo[1:] = (phi[:-1] + phi[1:]) / 2
    phi_hi = np.zeros_like(phi)
    phi_hi[:-1] = (phi[:-1] + phi[1:]) / 2
    if params.quadrature == "periodic":
        phi_lo[0] = (phi[-1] - 2.0 * np.pi + phi[0]) / 2
        phi_hi[-1] = (phi[-1] + phi[0] + 2.0 * np.pi) / 2
    else:
        phi_lo[0] = phi[0]
        phi_hi[-1] = phi[-1]
    dphi = phi_hi - phi_lo

    d_area = dphi[:, np.newaxis] / (2 * np.pi) * (np.pi * r_hi[np.newaxis, :] ** 2) - dphi[:, np.newaxis] / (
        2 * np.pi
    ) * (np.pi * r_lo[np.newaxis, :] ** 2)

    return SimGrids(
        phi=phi,
        cos_phi=np.cos(phi),
        sin_phi=np.sin(phi),
        v0=v0,
        d=d,
        t_ball=t_ball,
        dt=dt,
        dt0=dt0,
        rate_divisor=rate_divisor,
        dr=dr,
        d_area=d_area,
    )
