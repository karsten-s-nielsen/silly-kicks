"""Off-Ball Scoring Opportunity (OBSO) value surface computation.

OBSO = PPCF x Transition(ball -> cell) x EPV(cell)

Computes a continuous value surface indicating the scoring opportunity at
each point on the pitch for the team in possession, accounting for pitch
control, ball transition probabilities, and expected possession value.

See NOTICE for full bibliographic citations.

References
----------
Spearman (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
Conference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

if TYPE_CHECKING:
    from silly_kicks.xthreat import ExpectedThreat

    from .pitch_control import PitchControlCache

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsoParams:
    """Configuration for OBSO surface computation.

    All spatial parameters are in meters on a [0, pitch_length] x [0, pitch_width]
    coordinate system.  Default ``sigma_x / sigma_y`` are scaled from the original
    StatsBomb 120x80 constants (30.0 / 20.0) to the silly-kicks meter coordinate
    system: ``30 / 120 * 105 = 26.25``, ``20 / 80 * 68 = 17.0``.

    Examples
    --------
    >>> params = ObsoParams()
    >>> params.sigma_x
    26.25
    """

    grid_nx: int = 104
    grid_ny: int = 68
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    sigma_x: float = 26.25
    sigma_y: float = 17.0


@dataclass(frozen=True)
class ObsoSurface:
    """OBSO value surface for a single frame.

    Values in [0, 1] representing the off-ball scoring opportunity at each
    grid cell.

    Examples
    --------
    Build an OBSO surface from a pitch-control surface and ball position::

        surface = compute_obso_surface(pc_surface, (52.5, 34.0))
        surface.values.shape  # (68, 104)
    """

    values: np.ndarray
    """(grid_ny, grid_nx) OBSO values in [0, 1]."""
    grid_x: np.ndarray
    """(grid_nx,) x-coordinates of grid cells in meters."""
    grid_y: np.ndarray
    """(grid_ny,) y-coordinates of grid cells in meters."""


# ---------------------------------------------------------------------------
# Synthetic grid fallbacks
# ---------------------------------------------------------------------------


def _make_synthetic_reachability_grid(ny: int = 100, nx: int = 64) -> np.ndarray:
    """Gaussian distance decay proxy for ball reachability.

    Used as fallback when trained grids are not available.
    Shape: (ny, nx) -- OBSO convention.  Dimensionless (0-1 normalized).
    """
    y = np.linspace(0, 1, ny)
    x = np.linspace(0, 1, nx)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    center_y, center_x = 0.5, 0.5
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    return np.exp(-(dist**2) / (2 * 0.3**2))


def _make_synthetic_epv_grid(ny: int = 50, nx: int = 32) -> np.ndarray:
    """Linear ramp proxy for EPV.  Shape: (ny, nx).  Dimensionless."""
    x = np.linspace(0.01, 0.3, nx)
    return np.tile(x, (ny, 1))


def _get_default_grids(
    reachability: np.ndarray | None = None,
    epv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reachability and EPV grids, falling back to synthetic defaults.

    Both grids use (ny, nx) shape convention matching ``compute_obso_surface``.
    When pre-loaded arrays are provided, they are used directly.  Otherwise,
    synthetic proxy grids are generated (pure computation, no I/O).

    Examples
    --------
    >>> reach, epv = _get_default_grids()
    >>> reach.shape
    (100, 64)
    """
    if reachability is None:
        reachability = _make_synthetic_reachability_grid()
    if epv is None:
        epv = _make_synthetic_epv_grid()
    return reachability, epv


# ---------------------------------------------------------------------------
# Grid interpolation
# ---------------------------------------------------------------------------


def _interpolate_grid(grid: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a grid to match PPCF grid dimensions via bilinear interpolation.

    Uses numpy-only bilinear interpolation (no scipy dependency at runtime).

    Parameters
    ----------
    grid : np.ndarray
        Source 2D array of shape (src_rows, src_cols).
    target_shape : tuple[int, int]
        Desired output shape (target_rows, target_cols).

    Returns
    -------
    np.ndarray
        Interpolated 2D array of shape ``target_shape``.

    Examples
    --------
    >>> grid = np.ones((10, 10))
    >>> _interpolate_grid(grid, (20, 20)).shape
    (20, 20)
    """
    src_rows, src_cols = grid.shape
    tgt_rows, tgt_cols = target_shape

    if (src_rows, src_cols) == target_shape:
        return grid.copy()

    # Build target coordinate grids mapping to source indices
    row_coords = np.linspace(0, src_rows - 1, tgt_rows)
    col_coords = np.linspace(0, src_cols - 1, tgt_cols)
    col_grid, row_grid = np.meshgrid(col_coords, row_coords)

    # Floor/ceil indices
    r0 = np.clip(np.floor(row_grid).astype(int), 0, src_rows - 2)
    r1 = r0 + 1
    c0 = np.clip(np.floor(col_grid).astype(int), 0, src_cols - 2)
    c1 = c0 + 1

    # Fractional parts
    dr = row_grid - r0
    dc = col_grid - c0

    # Bilinear interpolation
    result = (
        grid[r0, c0] * (1 - dr) * (1 - dc)
        + grid[r1, c0] * dr * (1 - dc)
        + grid[r0, c1] * (1 - dr) * dc
        + grid[r1, c1] * dr * dc
    )
    return result


# ---------------------------------------------------------------------------
# OBSO surface computation
# ---------------------------------------------------------------------------


def compute_obso_surface(
    pitch_control: object,
    ball_position: tuple[float, float],
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    params: ObsoParams | None = None,
) -> ObsoSurface:
    """Compute OBSO surface: PPCF x Transition(ball -> cell) x EPV(cell).

    The transition grid gives P(ball reaches cell | ball at ball_position).
    This requires interpolating the pre-computed transition grid from the
    ball position to the target grid shape.

    The EPV grid gives the expected possession value at each cell if the
    team gains control there.

    Parameters
    ----------
    pitch_control : PitchControlSurface
        Pre-computed pitch control surface from
        ``silly_kicks.tracking.pitch_control.compute_pitch_control``.
    ball_position : tuple[float, float]
        (x, y) ball coordinates in meters on [0, 105] x [0, 68].
    transition_grid : np.ndarray or None
        Pre-computed ball transition probability grid.  None uses a
        synthetic Gaussian decay proxy.
    epv_grid : np.ndarray or None
        Pre-computed expected possession value grid.  None uses a
        synthetic linear ramp proxy.
    params : ObsoParams or None
        OBSO configuration.  None uses defaults.

    Returns
    -------
    ObsoSurface
        Surface with ``values`` in [0, 1].

    Examples
    --------
    Compute an OBSO surface from a single frame's pitch control::

        from silly_kicks.tracking.pitch_control import compute_pitch_control
        surface = compute_pitch_control(frame, attacking_team_id=1)
        obso = compute_obso_surface(surface, (52.5, 34.0))
        0 <= obso.values.max() <= 1  # True
    """
    if params is None:
        params = ObsoParams()

    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)

    # Extract PPCF grid from PitchControlSurface
    ppcf_grid = pitch_control.surface  # type: ignore[attr-defined]
    grid_x = np.asarray(pitch_control.grid_x)  # type: ignore[attr-defined]
    grid_y = np.asarray(pitch_control.grid_y)  # type: ignore[attr-defined]

    ny, nx = ppcf_grid.shape

    # Interpolate static grids to match PPCF dimensions
    transition_interp = _interpolate_grid(transition_grid, (ny, nx))
    epv_interp = _interpolate_grid(epv_grid, (ny, nx))

    # Shift transition grid based on ball position:
    # Gaussian decay from ball position to approximate transition likelihood
    ball_x, ball_y = ball_position
    xx, yy = np.meshgrid(grid_x, grid_y)
    distance_weight = np.exp(
        -((xx - ball_x) ** 2) / (2.0 * params.sigma_x**2) - (yy - ball_y) ** 2 / (2.0 * params.sigma_y**2)
    )

    # Combine: transition probability conditioned on ball position
    effective_transition = transition_interp * distance_weight
    # Normalize so max transition = 1 (probabilities relative to best target)
    max_trans = np.max(effective_transition)
    if max_trans > 1e-10:
        effective_transition = effective_transition / max_trans

    # OBSO = PPCF x Transition x EPV
    obso = np.asarray(ppcf_grid) * effective_transition * epv_interp

    return ObsoSurface(
        values=np.clip(obso, 0.0, 1.0),
        grid_x=grid_x,
        grid_y=grid_y,
    )


# ---------------------------------------------------------------------------
# Pass-level OBSO metrics (PAUSA inputs)
# ---------------------------------------------------------------------------


def compute_pass_obso(
    pass_window_frames: list[pd.DataFrame],
    event_frame_idx: int,
    target_position: tuple[float, float],
    attacking_team_id: int | str,
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    xt: ExpectedThreat | None = None,
    params: ObsoParams | None = None,
    pitch_control_method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    pitch_control_cache: PitchControlCache | None = None,
) -> dict[str, float]:
    """Compute PAUSA-relevant OBSO metrics for one pass.

    Evaluates OBSO at the target position across all frames in a
    pre-windowed pass window to determine temporal judgment (when to pass)
    and spatial selection (where to pass).

    .. note::
        **Coordinate contract (ADR-041):** ``target_position`` and the ``transition_grid``
        / ``epv_grid`` are all interpreted in the FRAMES' coordinate convention (for
        canonical converted frames: home-attacks-right). This engine is deliberately
        orientation-blind. Per-action re-projection of an acting-team-LTR SPADL target,
        and the matching x-flip of an attack-LTR EPV grid, are the AGGREGATOR's
        responsibility -- see ``features.add_obso``. Passing a raw action-LTR target here
        alongside home-attacks-right frames silently samples the reflected point for
        away-team actions (this was DEFECT A).

    Parameters
    ----------
    pass_window_frames : list[pd.DataFrame]
        List of single-frame DataFrames (one per timestep around the pass).
        Callers typically produce these from ``slice_around_event``.
    event_frame_idx : int
        Index into ``pass_window_frames`` for the actual event frame.
    target_position : tuple[float, float]
        (x, y) of the actual pass target in meters.
    attacking_team_id : int | str
        Team in possession (whose control maps to 1.0 in pitch control).
    transition_grid : np.ndarray or None
        Pre-computed ball transition probability grid.
    epv_grid : np.ndarray or None
        Pre-computed expected possession value grid.
    params : ObsoParams or None
        OBSO configuration.  None uses defaults.
    pitch_control_method : str
        Pitch control model to use (default: ``"spearman"``).

    Returns
    -------
    dict[str, float]
        ``actual_obso`` -- OBSO at target position at event frame.
        ``peak_obso`` -- Maximum OBSO at target position across all frames.
        ``optimal_obso`` -- Maximum OBSO across all teammate positions at
        event frame.

        ``peak_obso`` and ``optimal_obso`` maximize over DIFFERENT axes (time
        at the fixed target vs teammate positions at the fixed event frame), so
        they are NOT mutually ordered: ``peak_obso > optimal_obso`` is
        legitimate (the target spot got better later in the window than any
        teammate's spot was at the event frame). Both are seeded from
        ``actual_obso``, so ``actual_obso <= peak_obso`` and
        ``actual_obso <= optimal_obso`` always hold.

    Examples
    --------
    Compute pass OBSO over a window of frames around an event::

        result = compute_pass_obso(window_frames, 5, (80.0, 30.0), 1)
        result["peak_obso"] >= result["actual_obso"]  # True
    """
    from .pitch_control import PitchControlCache

    if params is None:
        params = ObsoParams()

    if xt is not None:
        if epv_grid is not None:
            raise ValueError("compute_pass_obso: pass either xt= or epv_grid=, not both")
        from silly_kicks.xthreat import physical_grid, require_fitted_xt

        require_fitted_xt(xt, caller="compute_pass_obso")
        # Built at the EFFECTIVE params geometry, never a hardcoded default: this is the
        # one OBSO entry point that accepts a caller-supplied ObsoParams (ADR-041).
        #
        # NODE registration, identical to _resolve_epv_grid (ADR-042 review finding 3).
        # This site originally built CELL CENTRES ((i + 0.5) * L / n) -- the exact bug
        # _resolve_epv_grid's comment describes, and uncorrected here for the same reason:
        # building at exactly (grid_ny, grid_nx) makes _interpolate_grid's identity
        # shortcut return the grid unresampled, while the index map below reads it as
        # node-registered. The two entry points disagreed by up to ~0.9% of grid max,
        # worst at the byline where crosses live.
        _gx = np.linspace(0.0, params.pitch_length, params.grid_nx)
        _gy = np.linspace(0.0, params.pitch_width, params.grid_ny)
        epv_grid = physical_grid(xt, _gx, _gy)
    # No synthetic-surface warning here: engines stay silent, the aggregator edge owns
    # that policy (policy-at-edge).

    # Canonical-frame surfaces routed through the shared cache so overlapping
    # pass windows (and the event/teammate queries on the same frame) reuse
    # each surface (TF-7 shared surface). compute_pitch_control_at_points is
    # equivalent to surface(...).at_points(...).
    cache = pitch_control_cache if pitch_control_cache is not None else PitchControlCache()

    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)

    if not pass_window_frames or event_frame_idx >= len(pass_window_frames):
        return {"actual_obso": np.nan, "peak_obso": np.nan, "optimal_obso": np.nan}

    target_arr = np.array([list(target_position)])

    # --- actual_obso: OBSO at target at the event frame ---
    event_df = pass_window_frames[event_frame_idx]

    # Ensure velocity columns exist
    event_df = _ensure_velocity_columns(event_df, method=pitch_control_method)

    event_surface = cache.surface(event_df, attacking_team_id, method=pitch_control_method)
    event_ppcf_at_target = event_surface.at_points(target_arr)
    actual_ppcf = float(event_ppcf_at_target[0])

    # Interpolate grids to ObsoParams dimensions for point lookup
    transition_interp = _interpolate_grid(transition_grid, (params.grid_ny, params.grid_nx))
    epv_interp = _interpolate_grid(epv_grid, (params.grid_ny, params.grid_nx))

    # Map target to grid indices. NEAREST node, not floor (ADR-041): the index space
    # `x / pitch_length * (grid_nx - 1)` is NODE-registered (bit-identical to
    # np.linspace(0, pitch_length, grid_nx)), and for node registration the correct
    # nearest-neighbour rule is round. int() truncates, which is a systematic half-node
    # bias toward the origin on every OBSO lookup, and is also why the target->cell map
    # was not mirror-symmetric (x=15 -> floor 14, mirror x=90 -> floor 88, but the mirror
    # of column 14 is 89; with round both give 15/88 and 103-15 == 88).
    tx_idx = int(
        np.clip(
            round(target_position[0] / params.pitch_length * (params.grid_nx - 1)),
            0,
            params.grid_nx - 1,
        )
    )
    ty_idx = int(
        np.clip(
            round(target_position[1] / params.pitch_width * (params.grid_ny - 1)),
            0,
            params.grid_ny - 1,
        )
    )
    trans_at_target = float(transition_interp[ty_idx, tx_idx])
    epv_at_target = float(epv_interp[ty_idx, tx_idx])
    actual_obso = float(np.clip(actual_ppcf * trans_at_target * epv_at_target, 0.0, 1.0))

    # --- peak_obso: max OBSO at target across all frames ---
    peak_obso = actual_obso
    for i, frame_df in enumerate(pass_window_frames):
        if i == event_frame_idx:
            continue
        frame_df = _ensure_velocity_columns(frame_df, method=pitch_control_method)
        ppcf_val = cache.surface(frame_df, attacking_team_id, method=pitch_control_method).at_points(target_arr)
        frame_obso = float(np.clip(float(ppcf_val[0]) * trans_at_target * epv_at_target, 0.0, 1.0))
        if frame_obso > peak_obso:
            peak_obso = frame_obso

    # --- optimal_obso: max OBSO across teammate positions at event frame ---
    optimal_obso = actual_obso
    teammate_positions = _extract_teammate_positions(event_df, attacking_team_id)
    if len(teammate_positions) > 0:
        # Reuse the event-frame surface already computed above.
        tm_ppcf = event_surface.at_points(teammate_positions)
        for j in range(len(teammate_positions)):
            # `round`, matching the target map above: a bare int() TRUNCATES, biasing every
            # teammate lookup toward the low-index cell by up to a full cell. The two maps
            # index the SAME node-registered grid, so they must round the same way (ADR-041).
            tm_x_idx = int(
                np.clip(
                    round(teammate_positions[j, 0] / params.pitch_length * (params.grid_nx - 1)),
                    0,
                    params.grid_nx - 1,
                )
            )
            tm_y_idx = int(
                np.clip(
                    round(teammate_positions[j, 1] / params.pitch_width * (params.grid_ny - 1)),
                    0,
                    params.grid_ny - 1,
                )
            )
            tm_trans = float(transition_interp[tm_y_idx, tm_x_idx])
            tm_epv = float(epv_interp[tm_y_idx, tm_x_idx])
            tm_obso = float(np.clip(float(tm_ppcf[j]) * tm_trans * tm_epv, 0.0, 1.0))
            if tm_obso > optimal_obso:
                optimal_obso = tm_obso

    return {
        "actual_obso": actual_obso,
        "peak_obso": peak_obso,
        "optimal_obso": optimal_obso,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_velocity_columns(frame: pd.DataFrame, *, method: str = "spearman") -> pd.DataFrame:
    """Prepare a frame for a velocity-requiring pitch-control call (ADR-063).

    A declared-velocity-less frame (``speed_source == "unavailable"``) gets the zero-velocity
    positional model; a frame merely MISSING ``vx``/``vy`` (a forgotten ``derive_velocities()``)
    RAISES -- so the public engine ``compute_pass_obso`` fails fast on the caller bug rather than
    silently zero-filling it, matching ``compute_pitch_control``. Single-sourced through the shared
    ``zero_velocity_if_unavailable`` edge seam; on the ``add_obso``/``add_pausa`` aggregator path the
    frames are already prepared by ``_precompute_obso_lookup``, so this is a no-op there.
    """
    from ._velocity_availability import zero_velocity_if_unavailable

    return zero_velocity_if_unavailable(frame, method=method)


def _extract_teammate_positions(frame: pd.DataFrame, attacking_team_id: int | str) -> np.ndarray:
    """Extract (n, 2) array of non-ball attacking-team positions from a frame."""
    mask = (
        ids_match(frame["team_id"], attacking_team_id) & (frame["is_ball"] != True)  # noqa: E712
    )
    teammates = frame.loc[mask, ["x", "y"]].dropna()
    if teammates.empty:
        return np.empty((0, 2), dtype="float64")
    return np.asarray(teammates.values)
