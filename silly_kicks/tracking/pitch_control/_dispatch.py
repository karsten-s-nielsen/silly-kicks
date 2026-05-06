"""Pitch control dispatch — routes method kwarg to the correct model.

Public API:
- compute_pitch_control(frame, attacking_team_id, ...) -> PitchControlSurface
- compute_pitch_control_at_points(frame, targets, ...) -> np.ndarray

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._fernandez_bornn import compute_fernandez_bornn
from ._params import (
    FernandezBornnParams,
    Method,
    PitchControlParams,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)
from ._spearman import compute_spearman
from ._surface import PitchControlSurface
from ._voronoi import compute_voronoi

_VELOCITY_REQUIRED_METHODS: set[Method] = {"spearman", "fernandez_bornn"}


def compute_pitch_control(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Compute pitch control surface for a single tracking frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data with columns: player_id, team_id, x, y,
        vx, vy (for Spearman/F&B), is_ball, is_goalkeeper.
    attacking_team_id : int | str
        Team whose control maps to 1.0 in the output surface.
    method : {"spearman", "fernandez_bornn", "voronoi"}
        Pitch control model to use. Default: "spearman".
    params : PitchControlParams | None
        Model-specific parameters. None uses defaults.
    decompose : bool
        If True, compute per-player influence decomposition.
    ball_position : tuple[float, float] | None
        Explicit ball position (x, y) in meters. If None, inferred from
        the ball row in the frame.

    Returns
    -------
    PitchControlSurface

    Examples
    --------
    >>> from silly_kicks.tracking.pitch_control import compute_pitch_control
    >>> surface = compute_pitch_control(frame, attacking_team_id=1)
    >>> surface.at_point(52.5, 34.0)
    0.55
    """
    # Validate params for method
    validate_params_for_method(method, params)

    # Check velocity columns for methods that require them
    if method in _VELOCITY_REQUIRED_METHODS:
        if "vx" not in frame.columns or "vy" not in frame.columns:
            raise ValueError(
                f"method='{method}' requires velocity columns ('vx', 'vy') "
                f"in the tracking frame. Use derive_velocities() or "
                f"smooth_frames() to add them, or use method='voronoi' "
                f"for position-only pitch control."
            )

    # Resolve ball position
    bp = _resolve_ball_position(frame, ball_position)

    # Dispatch to model
    if method == "spearman":
        sp = params if isinstance(params, SpearmanParams) else SpearmanParams()
        return compute_spearman(
            frame,
            attacking_team_id,
            sp,
            decompose=decompose,
            ball_position=bp,
        )
    elif method == "fernandez_bornn":
        fp = params if isinstance(params, FernandezBornnParams) else FernandezBornnParams()
        return compute_fernandez_bornn(
            frame,
            attacking_team_id,
            fp,
            decompose=decompose,
            ball_position=bp,
        )
    else:  # voronoi
        vp = params if isinstance(params, VoronoiParams) else VoronoiParams()
        return compute_voronoi(
            frame,
            attacking_team_id,
            vp,
            decompose=decompose,
            ball_position=bp,
        )


def compute_pitch_control_at_points(
    frame: pd.DataFrame,
    targets: np.ndarray,
    attacking_team_id: int | str,
    *,
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    ball_position: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute pitch control at specific (x, y) points.

    Computes the full surface then interpolates at the given points.
    For repeated queries on the same frame, prefer computing the surface
    once and calling surface.at_points().

    Parameters
    ----------
    frame : pd.DataFrame
        Single-frame tracking data.
    targets : np.ndarray
        (N, 2) array of (x, y) positions to query.
    attacking_team_id : int | str
        Team whose control maps to 1.0.
    method : Method
        Pitch control model to use.
    params : PitchControlParams | None
        Model-specific parameters.
    ball_position : tuple[float, float] | None
        Explicit ball position override.

    Returns
    -------
    np.ndarray of shape (N,) with control values in [0, 1].

    Examples
    --------
    >>> targets = np.array([[50, 34], [80, 20]])
    >>> compute_pitch_control_at_points(frame, targets, 1)
    array([0.52, 0.35])
    """
    if targets.shape[0] == 0:
        return np.empty(0, dtype="float64")

    surface = compute_pitch_control(
        frame,
        attacking_team_id,
        method=method,
        params=params,
        decompose=False,
        ball_position=ball_position,
    )
    return surface.at_points(targets)


def _resolve_ball_position(
    frame: pd.DataFrame,
    explicit: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Resolve ball position: explicit > frame ball row > None."""
    if explicit is not None:
        # Validate on-pitch (loose bounds)
        x, y = explicit
        if x < -5 or x > 110 or y < -5 or y > 73:
            return None  # off-pitch, treat as no conditioning
        return explicit

    # Try to extract from ball row
    ball_rows = frame[frame["is_ball"].astype(bool)]
    if ball_rows.empty:
        return None
    bx = float(ball_rows["x"].iloc[0])
    by = float(ball_rows["y"].iloc[0])
    if np.isnan(bx) or np.isnan(by):
        return None
    return (bx, by)
