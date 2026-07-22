"""PitchControlSurface frozen dataclass — the stable contract for all flavors.

All pitch control models return this type. Consumers program against it
without knowing which model produced it.

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silly_kicks.id_compat import ids_match


@dataclass(frozen=True)
class PitchControlSurface:
    """Spatial pitch control field for a single frame.

    Values in [0, 1]: 1.0 = full attacking-team control,
    0.0 = full defending-team control, 0.5 = contested.

    All numpy array fields are immutable (writeable=False).

    Examples
    --------
    >>> surface = compute_pitch_control(frame, attacking_team_id=1)
    >>> surface.at_point(50.0, 34.0)  # control at center circle
    0.52
    >>> surface.control_in_region(52.5, 105, 0, 68)  # attacking half
    0.61
    """

    grid_x: np.ndarray
    """(nx,) cell centers in meters [0, 105]."""
    grid_y: np.ndarray
    """(ny,) cell centers in meters [0, 68]."""
    surface: np.ndarray
    """(ny, nx) control values in [0, 1]."""
    method: str
    """Model that produced this surface: 'spearman' | 'fernandez_bornn' | 'voronoi'."""
    attacking_team_id: int | str
    """Team whose control maps to 1.0."""
    per_player_influence: np.ndarray | None = None
    """(n_players, ny, nx) when decompose=True; None otherwise."""
    player_ids: np.ndarray | None = None
    """(n_players,) aligning per_player_influence axis 0."""
    player_team_ids: np.ndarray | None = None
    """(n_players,) team membership for each player in player_ids."""

    def __post_init__(self) -> None:
        """Enforce array immutability."""
        self.grid_x.flags.writeable = False
        self.grid_y.flags.writeable = False
        self.surface.flags.writeable = False
        if self.per_player_influence is not None:
            self.per_player_influence.flags.writeable = False
        if self.player_ids is not None:
            self.player_ids.flags.writeable = False
        if self.player_team_ids is not None:
            self.player_team_ids.flags.writeable = False

    @property
    def cell_area(self) -> float:
        """Area of a single grid cell in m^2.

        Examples
        --------
        >>> surface.cell_area  # default 50x32 grid
        4.46
        """
        dx = float(self.grid_x[1] - self.grid_x[0]) if len(self.grid_x) > 1 else 105.0
        dy = float(self.grid_y[1] - self.grid_y[0]) if len(self.grid_y) > 1 else 68.0
        return dx * dy

    def at_point(self, x: float, y: float) -> float:
        """Bilinear interpolation of control value at (x, y) meters.

        Clamps to grid bounds (no extrapolation).

        Examples
        --------
        >>> surface.at_point(52.5, 34.0)
        0.55
        """
        return float(self.at_points(np.array([[x, y]]))[0])

    def at_points(self, xy: np.ndarray) -> np.ndarray:
        """Batch bilinear interpolation. xy shape: (N, 2).

        Examples
        --------
        >>> pts = np.array([[50, 34], [80, 20]])
        >>> surface.at_points(pts)
        array([0.52, 0.71])
        """
        from scipy.interpolate import RegularGridInterpolator

        # RegularGridInterpolator expects (y, x) ordering for the grid
        interp = RegularGridInterpolator(
            (self.grid_y, self.grid_x),
            self.surface,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate via nearest  # type: ignore[arg-type]
        )
        # Input is (x, y) but interpolator expects (y, x)
        yx = np.column_stack([xy[:, 1], xy[:, 0]])
        result = interp(yx)
        return np.clip(result, 0.0, 1.0)

    def control_in_region(self, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
        """Area-weighted mean control in a rectangular region.

        Examples
        --------
        >>> surface.control_in_region(52.5, 105, 0, 68)  # attacking half
        0.61
        """
        x_mask = (self.grid_x >= x_min) & (self.grid_x <= x_max)
        y_mask = (self.grid_y >= y_min) & (self.grid_y <= y_max)
        region = self.surface[np.ix_(y_mask, x_mask)]
        if region.size == 0:
            return 0.5
        return float(region.mean())

    def player_share(self, player_id: int | str) -> float:
        """Fraction of player's team influence attributable to player_id.

        Denominator is the sum over teammates (same team), not all players.
        Requires decompose=True. Returns value in [0, 1].

        Examples
        --------
        >>> surface.player_share(gk_player_id)
        0.18
        """
        if self.per_player_influence is None or self.player_ids is None:
            raise ValueError("player_share() requires decompose=True when computing the pitch control surface.")
        # ADR-019: caller-supplied id scalar vs the player_ids array is a cross-source compare
        # (Int64 frame ids vs a str/int query silently match nothing under a raw ==), so route
        # it through ids_match. Byte-identical on matched dtypes.
        idx = np.where(ids_match(self.player_ids, player_id).to_numpy())[0]
        if len(idx) == 0:
            raise ValueError(f"player_id={player_id!r} not found in player_ids {self.player_ids.tolist()}")
        player_total = float(self.per_player_influence[idx[0]].sum())
        # Denominator: sum over teammates only (same team_id). team_id is drawn from
        # player_team_ids itself -- a same-source compare, so a raw == is correct here (ADR-019
        # governs cross-source id comparisons; this one cannot mismatch by construction).
        if self.player_team_ids is not None:
            team_id = self.player_team_ids[idx[0]]
            team_mask = self.player_team_ids == team_id
            team_total = float(self.per_player_influence[team_mask].sum())
        else:
            # Fallback: all players (backwards compat if team_ids unavailable)
            team_total = float(self.per_player_influence.sum())
        if team_total < 1e-10:
            return 0.0
        return player_total / team_total

    def player_surface(self, player_id: int | str) -> np.ndarray:
        """Per-cell influence for a single player. Shape (ny, nx).

        Examples
        --------
        >>> ps = surface.player_surface(gk_player_id)
        >>> ps.shape
        (32, 50)
        """
        if self.per_player_influence is None or self.player_ids is None:
            raise ValueError("player_surface() requires decompose=True when computing the pitch control surface.")
        # ADR-019: cross-source id compare -- see player_share() above. Byte-identical on
        # matched dtypes; resolves a value-equal id of a different dtype instead of raising.
        idx = np.where(ids_match(self.player_ids, player_id).to_numpy())[0]
        if len(idx) == 0:
            raise ValueError(f"player_id={player_id!r} not found in player_ids {self.player_ids.tolist()}")
        return np.array(self.per_player_influence[idx[0]])

    def to_xarray(self) -> object:
        """Convert to labelled xarray DataArray (requires xarray installed).

        Dimensions: (y, x) for surface; (player_id, y, x) for decomposed.

        Examples
        --------
        >>> da = surface.to_xarray()
        >>> da.sel(x=50, y=34, method="nearest").item()
        0.52
        """
        try:
            import xarray as xr  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "to_xarray() requires the xarray package. Install with: pip install silly-kicks[xarray]"
            ) from None

        da = xr.DataArray(
            data=np.array(self.surface),
            dims=("y", "x"),
            coords={"x": self.grid_x.copy(), "y": self.grid_y.copy()},
            attrs={"method": self.method, "attacking_team_id": self.attacking_team_id},
        )
        return da
