# TF-7: Pitch Control Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three-flavor pitch control (Spearman, Fernandez/Bornn, Voronoi) as a first-class subpackage with per-player decomposition, optional numba acceleration, and VAEP integration.

**Architecture:** Subpackage `silly_kicks/tracking/pitch_control/` with per-flavor modules sharing a common `PitchControlSurface` return type. Multi-flavor dispatch via `method=` kwarg (matching `tracking/pressure.py` pattern). NumPy vectorized baseline with optional numba `@njit` kernels.

**Tech Stack:** numpy (vectorized broadcast), scipy.spatial (transitive dep, for fallback), numba (optional), xarray (optional), pytest + pytest-benchmark

**Spec:** `docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md`

**Commit policy:** Single commit per branch. All tasks are logical steps; one commit at the end after `/final-review`.

---

## File Map

### New files (create)

| File | Responsibility |
|------|---------------|
| `silly_kicks/tracking/pitch_control/__init__.py` | Public API re-exports |
| `silly_kicks/tracking/pitch_control/_params.py` | SpearmanParams, FernandezBornnParams, VoronoiParams, validation |
| `silly_kicks/tracking/pitch_control/_surface.py` | PitchControlSurface frozen dataclass + convenience methods |
| `silly_kicks/tracking/pitch_control/_spearman.py` | Kinematic TTI + logistic influence + GK weighting |
| `silly_kicks/tracking/pitch_control/_fernandez_bornn.py` | Bivariate normal influence + sigmoid aggregation |
| `silly_kicks/tracking/pitch_control/_voronoi.py` | Nearest-player broadcast argmin |
| `silly_kicks/tracking/pitch_control/_dispatch.py` | `compute_pitch_control()` + `compute_pitch_control_at_points()` |
| `silly_kicks/tracking/pitch_control/_numba_kernels.py` | Optional `@njit` TTI + influence + Gaussian kernels |
| `tests/tracking/pitch_control/__init__.py` | Test package marker |
| `tests/tracking/pitch_control/test_params.py` | Params validation tests |
| `tests/tracking/pitch_control/test_surface.py` | PitchControlSurface dataclass tests |
| `tests/tracking/pitch_control/test_spearman.py` | Spearman model tests |
| `tests/tracking/pitch_control/test_fernandez_bornn.py` | Fernandez/Bornn model tests |
| `tests/tracking/pitch_control/test_voronoi.py` | Voronoi model tests |
| `tests/tracking/pitch_control/test_dispatch.py` | Dispatch routing + validation tests |
| `tests/tracking/pitch_control/test_numba_parity.py` | NumPy == numba golden-master |
| `tests/tracking/pitch_control/test_action_coupled.py` | VAEP xfn integration |
| `tests/tracking/pitch_control/test_perf_budget.py` | pytest-benchmark performance gates |
| `tests/tracking/pitch_control/test_lakehouse_parity.py` | Cross-reference against lakehouse |
| `tests/invariants/test_pitch_control_invariants.py` | Physical invariant tests |
| `docs/superpowers/adrs/ADR-008-pitch-control.md` | Architectural decision record |

### Modified files

| File | Change |
|------|--------|
| `silly_kicks/tracking/__init__.py` | Add pitch_control re-exports |
| `silly_kicks/tracking/features.py` | Add `pitch_control_at_action`, `add_pitch_control`, `pitch_control_xfns`, `pitch_control_default_xfns` |
| `silly_kicks/atomic/tracking/features.py` | Mirror atomic-SPADL pitch control wrappers |
| `NOTICE` | Add Spearman 2017, Fernandez/Bornn 2018 entries |
| `pyproject.toml` | Add numba + xarray optional deps |
| `TODO.md` | Move TF-7 from On Deck to shipped |

---

## Task 1: Foundation types (`_params.py` + `_surface.py`)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/__init__.py`
- Create: `silly_kicks/tracking/pitch_control/_params.py`
- Create: `silly_kicks/tracking/pitch_control/_surface.py`
- Create: `tests/tracking/pitch_control/__init__.py`
- Create: `tests/tracking/pitch_control/test_params.py`
- Create: `tests/tracking/pitch_control/test_surface.py`

- [ ] **Step 1: Write params tests**

```python
# tests/tracking/pitch_control/test_params.py
"""Tests for pitch control parameter dataclasses and validation."""
from __future__ import annotations

import pytest

from silly_kicks.tracking.pitch_control._params import (
    FernandezBornnParams,
    Method,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)


class TestSpearmanParams:
    def test_defaults(self):
        p = SpearmanParams()
        assert p.reaction_time == 0.7
        assert p.max_acceleration == 7.0
        assert p.sigma == 0.45
        assert p.lambda_gk == 3.0
        assert p.average_ball_speed == 15.0
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32

    def test_frozen(self):
        p = SpearmanParams()
        with pytest.raises(Exception):
            p.sigma = 1.0  # type: ignore[misc]


class TestFernandezBornnParams:
    def test_defaults(self):
        p = FernandezBornnParams()
        assert p.max_speed == 13.0
        assert p.min_radius == 4.0
        assert p.max_radius == 10.0
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32

    def test_frozen(self):
        p = FernandezBornnParams()
        with pytest.raises(Exception):
            p.max_speed = 20.0  # type: ignore[misc]


class TestVoronoiParams:
    def test_defaults(self):
        p = VoronoiParams()
        assert p.grid_cells_x == 50
        assert p.grid_cells_y == 32


class TestValidateParamsForMethod:
    def test_none_params_accepted(self):
        validate_params_for_method("spearman", None)
        validate_params_for_method("fernandez_bornn", None)
        validate_params_for_method("voronoi", None)

    def test_correct_type_accepted(self):
        validate_params_for_method("spearman", SpearmanParams())
        validate_params_for_method("fernandez_bornn", FernandezBornnParams())
        validate_params_for_method("voronoi", VoronoiParams())

    def test_wrong_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="spearman.*expects SpearmanParams"):
            validate_params_for_method("spearman", FernandezBornnParams())

    def test_unknown_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown method"):
            validate_params_for_method("bogus", None)  # type: ignore[arg-type]
```

- [ ] **Step 2: Write surface tests**

```python
# tests/tracking/pitch_control/test_surface.py
"""Tests for PitchControlSurface dataclass."""
from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking.pitch_control._surface import PitchControlSurface


def _make_surface(nx=10, ny=7, value=0.6, decompose=False):
    """Helper to build a test surface."""
    grid_x = np.linspace(0, 105, nx)
    grid_y = np.linspace(0, 68, ny)
    surface = np.full((ny, nx), value)
    per_player = None
    player_ids = None
    player_team_ids = None
    if decompose:
        per_player = np.full((3, ny, nx), value / 3)
        player_ids = np.array([1, 2, 3])
        player_team_ids = np.array([1, 1, 2])  # players 1,2 on team 1; player 3 on team 2
    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="spearman",
        attacking_team_id=1,
        per_player_influence=per_player,
        player_ids=player_ids,
        player_team_ids=player_team_ids,
    )


class TestImmutability:
    def test_frozen_attribute(self):
        s = _make_surface()
        with pytest.raises(Exception):
            s.method = "voronoi"  # type: ignore[misc]

    def test_array_not_writeable(self):
        s = _make_surface()
        with pytest.raises(ValueError):
            s.surface[0, 0] = 999.0

    def test_grid_not_writeable(self):
        s = _make_surface()
        with pytest.raises(ValueError):
            s.grid_x[0] = -1.0


class TestCellArea:
    def test_cell_area_correct(self):
        s = _make_surface(nx=10, ny=7)
        dx = 105.0 / 9  # linspace(0, 105, 10) has 9 gaps
        dy = 68.0 / 6
        assert abs(s.cell_area - dx * dy) < 1e-10


class TestAtPoint:
    def test_uniform_surface(self):
        s = _make_surface(value=0.7)
        assert abs(s.at_point(50.0, 34.0) - 0.7) < 1e-10

    def test_at_points_batch(self):
        s = _make_surface(value=0.7)
        pts = np.array([[50.0, 34.0], [10.0, 10.0]])
        result = s.at_points(pts)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, 0.7, atol=1e-10)

    def test_edge_clamp(self):
        s = _make_surface(value=0.5)
        # Point at grid boundary should not crash
        val = s.at_point(0.0, 0.0)
        assert 0.0 <= val <= 1.0


class TestControlInRegion:
    def test_uniform_surface(self):
        s = _make_surface(value=0.8)
        val = s.control_in_region(0, 105, 0, 68)
        assert abs(val - 0.8) < 1e-10

    def test_half_pitch(self):
        s = _make_surface(value=0.8)
        val = s.control_in_region(52.5, 105, 0, 68)
        assert abs(val - 0.8) < 1e-10


class TestPlayerShare:
    def test_raises_without_decomposition(self):
        s = _make_surface(decompose=False)
        with pytest.raises(ValueError, match="decompose"):
            s.player_share(1)

    def test_equal_shares_within_team(self):
        s = _make_surface(decompose=True)
        # Players 1 & 2 on team 1 with equal influence → 50% each of team total
        assert abs(s.player_share(1) - 0.5) < 1e-10
        assert abs(s.player_share(2) - 0.5) < 1e-10
        # Player 3 alone on team 2 → 100% of team total
        assert abs(s.player_share(3) - 1.0) < 1e-10

    def test_unknown_player_raises(self):
        s = _make_surface(decompose=True)
        with pytest.raises(ValueError, match="not found"):
            s.player_share(999)


class TestPlayerSurface:
    def test_returns_correct_shape(self):
        s = _make_surface(nx=10, ny=7, decompose=True)
        ps = s.player_surface(1)
        assert ps.shape == (7, 10)

    def test_raises_without_decomposition(self):
        s = _make_surface(decompose=False)
        with pytest.raises(ValueError, match="decompose"):
            s.player_surface(1)


class TestToXarray:
    def test_raises_without_xarray(self):
        """If xarray not installed, should raise ImportError with message."""
        # This test may pass or skip depending on env
        s = _make_surface()
        try:
            import xarray  # noqa: F401
            # xarray is installed — test it works
            da = s.to_xarray()
            assert da.dims == ("y", "x")
        except ImportError:
            with pytest.raises(ImportError, match="xarray"):
                s.to_xarray()
```

- [ ] **Step 3: Implement `_params.py`**

```python
# silly_kicks/tracking/pitch_control/_params.py
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
```

- [ ] **Step 4: Implement `_surface.py`**

```python
# silly_kicks/tracking/pitch_control/_surface.py
"""PitchControlSurface frozen dataclass — the stable contract for all flavors.

All pitch control models return this type. Consumers program against it
without knowing which model produced it.

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        """Area of a single grid cell in m^2."""
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
            fill_value=None,  # extrapolate via nearest
        )
        # Input is (x, y) but interpolator expects (y, x)
        yx = np.column_stack([xy[:, 1], xy[:, 0]])
        result = interp(yx)
        return np.clip(result, 0.0, 1.0)

    def control_in_region(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> float:
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
            raise ValueError(
                "player_share() requires decompose=True when computing "
                "the pitch control surface."
            )
        idx = np.where(self.player_ids == player_id)[0]
        if len(idx) == 0:
            raise ValueError(
                f"player_id={player_id!r} not found in player_ids "
                f"{self.player_ids.tolist()}"
            )
        player_total = float(self.per_player_influence[idx[0]].sum())
        # Denominator: sum over teammates only (same team_id)
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
            raise ValueError(
                "player_surface() requires decompose=True when computing "
                "the pitch control surface."
            )
        idx = np.where(self.player_ids == player_id)[0]
        if len(idx) == 0:
            raise ValueError(
                f"player_id={player_id!r} not found in player_ids "
                f"{self.player_ids.tolist()}"
            )
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
            import xarray as xr
        except ImportError:
            raise ImportError(
                "to_xarray() requires the xarray package. "
                "Install with: pip install silly-kicks[xarray]"
            ) from None

        da = xr.DataArray(
            data=np.array(self.surface),
            dims=("y", "x"),
            coords={"x": self.grid_x.copy(), "y": self.grid_y.copy()},
            attrs={"method": self.method, "attacking_team_id": self.attacking_team_id},
        )
        return da
```

- [ ] **Step 5: Create `__init__.py` (minimal, expanded in Task 5)**

```python
# silly_kicks/tracking/pitch_control/__init__.py
"""Pitch control models — three-flavor spatial control computation.

Public API:
- compute_pitch_control(frame, attacking_team_id, ...) -> PitchControlSurface
- compute_pitch_control_at_points(frame, targets, ...) -> np.ndarray
- PitchControlSurface — rich frozen dataclass return type
- SpearmanParams / FernandezBornnParams / VoronoiParams
- Method type alias

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md
and ADR-008 for architectural decisions.
"""
from __future__ import annotations

from ._params import (
    FernandezBornnParams,
    Method,
    PitchControlParams,
    SpearmanParams,
    VoronoiParams,
    validate_params_for_method,
)
from ._surface import PitchControlSurface

__all__ = [
    "FernandezBornnParams",
    "Method",
    "PitchControlParams",
    "PitchControlSurface",
    "SpearmanParams",
    "VoronoiParams",
    "validate_params_for_method",
]
```

- [ ] **Step 6: Create test `__init__.py`**

```python
# tests/tracking/pitch_control/__init__.py
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_params.py tests/tracking/pitch_control/test_surface.py -v --tb=short`

Expected: All PASSED.

---

## Task 2: Voronoi model (simplest, validates pipeline)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_voronoi.py`
- Create: `tests/tracking/pitch_control/test_voronoi.py`

- [ ] **Step 1: Write Voronoi tests**

```python
# tests/tracking/pitch_control/test_voronoi.py
"""Tests for Voronoi pitch control model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control._params import VoronoiParams
from silly_kicks.tracking.pitch_control._voronoi import compute_voronoi


def _make_frame(att_positions, def_positions, att_team_id=1, def_team_id=2):
    """Build a minimal tracking frame for testing."""
    rows = []
    for i, (x, y) in enumerate(att_positions):
        rows.append({
            "player_id": 100 + i,
            "team_id": att_team_id,
            "x": x, "y": y,
            "is_ball": False,
            "is_goalkeeper": i == 0,
        })
    for i, (x, y) in enumerate(def_positions):
        rows.append({
            "player_id": 200 + i,
            "team_id": def_team_id,
            "x": x, "y": y,
            "is_ball": False,
            "is_goalkeeper": i == 0,
        })
    # Ball row
    rows.append({
        "player_id": np.nan,
        "team_id": np.nan,
        "x": 52.5, "y": 34.0,
        "is_ball": True,
        "is_goalkeeper": False,
    })
    return pd.DataFrame(rows)


class TestVoronoiBasic:
    def test_single_attacker_controls_all(self):
        frame = _make_frame([(52.5, 34.0)], [])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 1.0).all()

    def test_single_defender_controls_none(self):
        frame = _make_frame([], [(52.5, 34.0)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 0.0).all()

    def test_symmetric_equal_split(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        # Left half should be attacker (1.0), right half defender (0.0)
        mid_x_idx = len(s.grid_x) // 2
        assert s.surface[:, :mid_x_idx].mean() > 0.8
        assert s.surface[:, mid_x_idx:].mean() < 0.2

    def test_binary_output(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        unique_vals = np.unique(s.surface)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert s.method == "voronoi"

    def test_grid_bounds(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert s.grid_x[0] >= 0 and s.grid_x[-1] <= 105
        assert s.grid_y[0] >= 0 and s.grid_y[-1] <= 68


class TestVoronoiDecomposition:
    def test_decompose_binary(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_voronoi(
            frame, attacking_team_id=1, params=VoronoiParams(), decompose=True
        )
        assert s.per_player_influence is not None
        assert s.player_ids is not None
        # Each cell assigned to exactly one player
        assert (s.per_player_influence.sum(axis=0) == 1.0).all()

    def test_player_share_is_team_fraction(self):
        frame = _make_frame([(30, 34), (50, 50)], [(70, 34)])
        s = compute_voronoi(
            frame, attacking_team_id=1, params=VoronoiParams(), decompose=True
        )
        # Two attackers: shares within team 1 sum to 1.0
        share_att_0 = s.player_share(100)
        share_att_1 = s.player_share(101)
        assert abs(share_att_0 + share_att_1 - 1.0) < 1e-10
        # Solo defender: 100% of their team
        share_def = s.player_share(200)
        assert abs(share_def - 1.0) < 1e-10


class TestVoronoiEdgeCases:
    def test_empty_frame(self):
        frame = pd.DataFrame(columns=["player_id", "team_id", "x", "y", "is_ball", "is_goalkeeper"])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        assert (s.surface == 0.5).all()

    def test_nan_positions_filtered(self):
        frame = _make_frame([(50, 34), (np.nan, np.nan)], [(80, 34)])
        s = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        # Should not crash; NaN player ignored
        assert not np.isnan(s.surface).any()

    def test_ball_position_ignored(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s1 = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
        s2 = compute_voronoi(
            frame, attacking_team_id=1, params=VoronoiParams(),
            ball_position=(10, 10),
        )
        np.testing.assert_array_equal(s1.surface, s2.surface)
```

- [ ] **Step 2: Implement `_voronoi.py`**

```python
# silly_kicks/tracking/pitch_control/_voronoi.py
"""Voronoi tessellation pitch control — nearest-player assignment.

Binary control surface: 1.0 (attacking) or 0.0 (defending) per cell.
No physics, no probabilities — baseline for validation and fast spatial queries.

See docs/superpowers/specs/2026-05-05-tf7-pitch-control-design.md section 6.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._params import VoronoiParams
from ._surface import PitchControlSurface


def compute_voronoi(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    params: VoronoiParams,
    *,
    decompose: bool = False,
    ball_position: tuple[float, float] | None = None,
) -> PitchControlSurface:
    """Voronoi tessellation pitch control.

    For each grid cell, assigns control to the team of the nearest player.
    ball_position accepted for API consistency but ignored.

    Examples
    --------
    >>> from silly_kicks.tracking.pitch_control._voronoi import compute_voronoi
    >>> from silly_kicks.tracking.pitch_control._params import VoronoiParams
    >>> surface = compute_voronoi(frame, attacking_team_id=1, params=VoronoiParams())
    >>> surface.at_point(50, 34)
    1.0
    """
    grid_x = np.linspace(0, 105.0, params.grid_cells_x)
    grid_y = np.linspace(0, 68.0, params.grid_cells_y)
    n_cells = params.grid_cells_x * params.grid_cells_y

    # Filter players (no ball rows, no NaN positions)
    players = frame[~frame["is_ball"].astype(bool)].copy()
    players = players.dropna(subset=["x", "y"])

    if players.empty:
        surface = np.full((params.grid_cells_y, params.grid_cells_x), 0.5)
        return PitchControlSurface(
            grid_x=grid_x,
            grid_y=grid_y,
            surface=surface,
            method="voronoi",
            attacking_team_id=attacking_team_id,
        )

    # Build target grid
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])  # (n_cells, 2)

    # Player positions
    player_pos = players[["x", "y"]].to_numpy(dtype="float64")  # (n_players, 2)
    player_ids_arr = players["player_id"].to_numpy()
    is_attacking = (players["team_id"] == attacking_team_id).to_numpy()

    # Broadcast distance: (n_cells, n_players)
    diff = targets[:, np.newaxis, :] - player_pos[np.newaxis, :, :]
    distances = np.sqrt((diff**2).sum(axis=2))

    # Nearest player per cell
    nearest_idx = distances.argmin(axis=1)  # (n_cells,)

    # Assign control based on nearest player's team
    control_flat = np.where(is_attacking[nearest_idx], 1.0, 0.0)
    surface = control_flat.reshape(params.grid_cells_y, params.grid_cells_x)

    # Decomposition
    per_player = None
    p_ids = None
    p_team_ids = None
    if decompose:
        n_players = len(players)
        per_player_flat = np.zeros((n_players, n_cells))
        per_player_flat[nearest_idx, np.arange(n_cells)] = 1.0
        per_player = per_player_flat.reshape(n_players, params.grid_cells_y, params.grid_cells_x)
        p_ids = player_ids_arr
        p_team_ids = players["team_id"].to_numpy()

    return PitchControlSurface(
        grid_x=grid_x,
        grid_y=grid_y,
        surface=surface,
        method="voronoi",
        attacking_team_id=attacking_team_id,
        per_player_influence=per_player,
        player_ids=p_ids,
        player_team_ids=p_team_ids,
    )
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_voronoi.py -v --tb=short`

Expected: All PASSED.

---

## Task 3: Spearman model (primary, most complex)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_spearman.py`
- Create: `tests/tracking/pitch_control/test_spearman.py`

- [ ] **Step 1: Write Spearman tests**

```python
# tests/tracking/pitch_control/test_spearman.py
"""Tests for Spearman kinematic pitch control model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control._params import SpearmanParams
from silly_kicks.tracking.pitch_control._spearman import (
    _compute_influence,
    _compute_tti,
    compute_spearman,
)


def _make_frame(att_positions, def_positions, att_vel=None, def_vel=None,
                att_team_id=1, def_team_id=2, att_gk_idx=0, def_gk_idx=0):
    """Build a tracking frame with velocities."""
    rows = []
    for i, (x, y) in enumerate(att_positions):
        vx = att_vel[i][0] if att_vel else 0.0
        vy = att_vel[i][1] if att_vel else 0.0
        rows.append({
            "player_id": 100 + i,
            "team_id": att_team_id,
            "x": x, "y": y, "vx": vx, "vy": vy,
            "is_ball": False,
            "is_goalkeeper": (i == att_gk_idx),
        })
    for i, (x, y) in enumerate(def_positions):
        vx = def_vel[i][0] if def_vel else 0.0
        vy = def_vel[i][1] if def_vel else 0.0
        rows.append({
            "player_id": 200 + i,
            "team_id": def_team_id,
            "x": x, "y": y, "vx": vx, "vy": vy,
            "is_ball": False,
            "is_goalkeeper": (i == def_gk_idx),
        })
    rows.append({
        "player_id": np.nan, "team_id": np.nan,
        "x": 52.5, "y": 34.0, "vx": 0.0, "vy": 0.0,
        "is_ball": True, "is_goalkeeper": False,
    })
    return pd.DataFrame(rows)


class TestTTI:
    def test_stationary_player(self):
        """Stationary player TTI = reaction_time + sqrt(2*a*d) / a."""
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[0.0, 0.0]])
        target = np.array([[10.0, 0.0]])
        tti = _compute_tti(pos, vel, target, reaction_time=0.7, max_acceleration=7.0)
        # d=10, v_proj=0: TTI = 0.7 + sqrt(2*7*10)/7 = 0.7 + sqrt(140)/7
        expected = 0.7 + np.sqrt(140.0) / 7.0
        np.testing.assert_allclose(tti[0, 0], expected, rtol=1e-10)

    def test_player_moving_toward_target(self):
        """Player moving toward target arrives sooner."""
        pos = np.array([[0.0, 0.0]])
        target = np.array([[10.0, 0.0]])
        vel_toward = np.array([[5.0, 0.0]])
        vel_away = np.array([[-5.0, 0.0]])
        tti_toward = _compute_tti(pos, vel_toward, target, 0.7, 7.0)[0, 0]
        tti_away = _compute_tti(pos, vel_away, target, 0.7, 7.0)[0, 0]
        assert tti_toward < tti_away

    def test_player_at_target(self):
        """Player already at target → TTI = reaction_time."""
        pos = np.array([[5.0, 5.0]])
        vel = np.array([[3.0, 0.0]])
        target = np.array([[5.0, 5.0]])
        tti = _compute_tti(pos, vel, target, 0.7, 7.0)[0, 0]
        assert abs(tti - 0.7) < 1e-10

    def test_broadcast_shape(self):
        """Multiple players × multiple targets."""
        pos = np.array([[0, 0], [10, 10], [20, 20]], dtype="float64")
        vel = np.zeros((3, 2))
        targets = np.array([[5, 5], [15, 15]], dtype="float64")
        tti = _compute_tti(pos, vel, targets, 0.7, 7.0)
        assert tti.shape == (3, 2)


class TestInfluence:
    def test_earlier_arrival_higher_influence(self):
        """Player arriving much earlier than opponent → influence near 1."""
        team_tti = np.array([[1.0]])  # arrives at t=1
        opponent_min = np.array([5.0])  # opponent arrives at t=5
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert influence[0, 0] > 0.95

    def test_later_arrival_lower_influence(self):
        """Player arriving much later → influence near 0."""
        team_tti = np.array([[5.0]])
        opponent_min = np.array([1.0])
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert influence[0, 0] < 0.05

    def test_equal_arrival_half_influence(self):
        """Same arrival time → influence = 0.5."""
        team_tti = np.array([[3.0]])
        opponent_min = np.array([3.0])
        influence = _compute_influence(team_tti, opponent_min, sigma=0.45)
        assert abs(influence[0, 0] - 0.5) < 1e-10


class TestComputeSpearman:
    def test_single_attacker_dominates(self):
        frame = _make_frame([(52.5, 34.0)], [(90.0, 34.0)])
        s = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
        # Attacker near center, defender far → high control at center
        assert s.at_point(52.5, 34.0) > 0.7

    def test_symmetric_equals_half(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_spearman(frame, attacking_team_id=1, params=SpearmanParams())
        assert abs(s.at_point(52.5, 34.0) - 0.5) < 0.05

    def test_velocity_effect(self):
        """Player running toward a cell gets higher control there."""
        frame_static = _make_frame(
            [(30, 34)], [(70, 34)],
            att_vel=[(0, 0)], def_vel=[(0, 0)],
        )
        frame_running = _make_frame(
            [(30, 34)], [(70, 34)],
            att_vel=[(5, 0)], def_vel=[(0, 0)],
        )
        s_static = compute_spearman(frame_static, 1, SpearmanParams())
        s_running = compute_spearman(frame_running, 1, SpearmanParams())
        # Attacker running right → more control on right side
        assert s_running.at_point(50, 34) > s_static.at_point(50, 34)

    def test_gk_weighting(self):
        """GK with lambda_gk > 1 contributes more influence."""
        frame = _make_frame([(20, 34)], [(80, 34)])
        params_no_gk = SpearmanParams(lambda_gk=1.0)
        params_gk = SpearmanParams(lambda_gk=3.0)
        s_no = compute_spearman(frame, 1, params_no_gk)
        s_gk = compute_spearman(frame, 1, params_gk)
        # GK is player 100 (att_gk_idx=0); with higher lambda, attacker controls more
        assert s_gk.at_point(20, 34) >= s_no.at_point(20, 34)

    def test_bounds(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_spearman(frame, 1, SpearmanParams())
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()

    def test_decomposition_sums_to_surface(self):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_spearman(frame, 1, SpearmanParams(), decompose=True)
        assert s.per_player_influence is not None
        # Sum attacking influence / (sum att + sum def) ≈ surface
        att_mask = np.isin(s.player_ids, [100, 101])
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        all_sum = s.per_player_influence.sum(axis=0)
        reconstructed = np.where(all_sum > 1e-10, att_sum / all_sum, 0.5)
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-10)

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_spearman(frame, 1, SpearmanParams())
        assert s.method == "spearman"

    def test_empty_frame(self):
        frame = pd.DataFrame(columns=[
            "player_id", "team_id", "x", "y", "vx", "vy",
            "is_ball", "is_goalkeeper",
        ])
        s = compute_spearman(frame, 1, SpearmanParams())
        assert (s.surface == 0.5).all()
```

- [ ] **Step 2: Implement `_spearman.py`**

Implementation follows the three-stage pipeline from spec section 6.1:
1. `_compute_tti()` — broadcast-vectorized kinematic equation
2. `_compute_influence()` — per-player logistic sigmoid (returns per-player, not summed)
3. `compute_spearman()` — orchestration including GK weighting and aggregation

The full implementation should be ~180-220 lines. Key details:
- GK rows identified via `is_goalkeeper.astype(bool)`
- `lambda_gk` multiplies GK influence AFTER sigmoid, BEFORE ratio aggregation
- Ball-travel-time filter zeros influence for players with TTI > ball_travel_time
- `decompose=True` retains the full `(n_players, n_targets)` influence matrix
- When `decompose=True`, also pass `player_team_ids=players["team_id"].to_numpy()` to `PitchControlSurface`

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_spearman.py -v --tb=short`

Expected: All PASSED.

---

## Task 4: Fernandez/Bornn model

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_fernandez_bornn.py`
- Create: `tests/tracking/pitch_control/test_fernandez_bornn.py`

- [ ] **Step 1: Write Fernandez/Bornn tests**

```python
# tests/tracking/pitch_control/test_fernandez_bornn.py
"""Tests for Fernandez/Bornn bivariate-normal pitch control model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control._params import FernandezBornnParams
from silly_kicks.tracking.pitch_control._fernandez_bornn import compute_fernandez_bornn


def _make_frame(att_positions, def_positions, att_vel=None, def_vel=None,
                att_team_id=1, def_team_id=2):
    """Build a tracking frame with velocities."""
    rows = []
    for i, (x, y) in enumerate(att_positions):
        vx = att_vel[i][0] if att_vel else 0.0
        vy = att_vel[i][1] if att_vel else 0.0
        rows.append({
            "player_id": 100 + i,
            "team_id": att_team_id,
            "x": x, "y": y, "vx": vx, "vy": vy,
            "is_ball": False,
            "is_goalkeeper": i == 0,
        })
    for i, (x, y) in enumerate(def_positions):
        vx = def_vel[i][0] if def_vel else 0.0
        vy = def_vel[i][1] if def_vel else 0.0
        rows.append({
            "player_id": 200 + i,
            "team_id": def_team_id,
            "x": x, "y": y, "vx": vx, "vy": vy,
            "is_ball": False,
            "is_goalkeeper": i == 0,
        })
    rows.append({
        "player_id": np.nan, "team_id": np.nan,
        "x": 52.5, "y": 34.0, "vx": 0.0, "vy": 0.0,
        "is_ball": True, "is_goalkeeper": False,
    })
    return pd.DataFrame(rows)


class TestFernandezBornnBasic:
    def test_single_attacker_high_control(self):
        frame = _make_frame([(52.5, 34.0)], [(90.0, 60.0)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert s.at_point(52.5, 34.0) > 0.7

    def test_symmetric_near_half(self):
        frame = _make_frame([(26.25, 34.0)], [(78.75, 34.0)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        center_control = s.at_point(52.5, 34.0)
        assert 0.4 < center_control < 0.6

    def test_bounds(self):
        frame = _make_frame([(30, 34), (50, 20)], [(70, 34), (80, 50)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()

    def test_method_field(self):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert s.method == "fernandez_bornn"


class TestVelocityEffect:
    def test_running_player_extends_influence_forward(self):
        frame_static = _make_frame([(30, 34)], [(80, 34)], att_vel=[(0, 0)])
        frame_running = _make_frame([(30, 34)], [(80, 34)], att_vel=[(8, 0)])
        s_static = compute_fernandez_bornn(frame_static, 1, FernandezBornnParams())
        s_running = compute_fernandez_bornn(frame_running, 1, FernandezBornnParams())
        # Running right → more control ahead of the player
        assert s_running.at_point(45, 34) > s_static.at_point(45, 34)


class TestHighSpeedGuard:
    def test_near_max_speed_no_nan(self):
        """Player at near-max speed should not produce NaN (alpha guard)."""
        params = FernandezBornnParams(max_speed=13.0)
        frame = _make_frame(
            [(50, 34)], [(70, 34)],
            att_vel=[(12.99, 0)],  # near max_speed
        )
        s = compute_fernandez_bornn(frame, 1, params)
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()

    def test_exact_max_speed_no_nan(self):
        """Player at exactly max_speed — alpha_ceil prevents singularity."""
        params = FernandezBornnParams(max_speed=13.0)
        frame = _make_frame(
            [(50, 34)], [(70, 34)],
            att_vel=[(13.0, 0)],
        )
        s = compute_fernandez_bornn(frame, 1, params)
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()


class TestStationaryGuard:
    def test_very_slow_player_isotropic(self):
        """Player with speed < 0.1 m/s treated as stationary (isotropic)."""
        frame_still = _make_frame([(50, 34)], [(70, 34)], att_vel=[(0, 0)])
        frame_tiny = _make_frame([(50, 34)], [(70, 34)], att_vel=[(0.05, 0.05)])
        s_still = compute_fernandez_bornn(frame_still, 1, FernandezBornnParams())
        s_tiny = compute_fernandez_bornn(frame_tiny, 1, FernandezBornnParams())
        # Should produce near-identical surfaces (both isotropic)
        np.testing.assert_allclose(s_still.surface, s_tiny.surface, atol=0.01)


class TestDecomposition:
    def test_decompose_returns_per_player(self):
        frame = _make_frame([(30, 34), (50, 50)], [(70, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), decompose=True)
        assert s.per_player_influence is not None
        assert s.per_player_influence.shape[0] == 3  # 2 att + 1 def

    def test_sigmoid_reconstruction_from_raw_gaussians(self):
        """Pre-sigmoid consistency: sigmoid(att_sum - def_sum) == surface."""
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams(), decompose=True)
        assert s.per_player_influence is not None
        assert s.player_team_ids is not None
        # All raw Gaussian values should be non-negative
        assert (s.per_player_influence >= 0).all()
        # The defining F/B invariant: surface = sigmoid(sum_att - sum_def)
        att_mask = s.player_team_ids == 1
        def_mask = s.player_team_ids != 1
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        def_sum = s.per_player_influence[def_mask].sum(axis=0)
        reconstructed = 1.0 / (1.0 + np.exp(-(att_sum - def_sum)))
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-6)


class TestBallPosition:
    def test_ball_position_affects_radius(self):
        """Ball position changes influence radii (closer = tighter)."""
        frame = _make_frame([(50, 34)], [(70, 34)])
        s_far = compute_fernandez_bornn(
            frame, 1, FernandezBornnParams(), ball_position=(0, 34)
        )
        s_near = compute_fernandez_bornn(
            frame, 1, FernandezBornnParams(), ball_position=(50, 34)
        )
        # Near ball → tighter radius → attacker influence more concentrated
        # At the attacker's position, control should be higher with tighter radius
        assert s_near.at_point(50, 34) >= s_far.at_point(50, 34) - 0.1


class TestEdgeCases:
    def test_empty_frame(self):
        frame = pd.DataFrame(columns=[
            "player_id", "team_id", "x", "y", "vx", "vy",
            "is_ball", "is_goalkeeper",
        ])
        s = compute_fernandez_bornn(frame, 1, FernandezBornnParams())
        assert (s.surface == 0.5).all()
```

- [ ] **Step 2: Implement `_fernandez_bornn.py`**

Implementation follows spec section 6.2:
1. Compute velocity direction (`theta`) and scaling factor (`alpha`) with guards
2. Build per-player covariance matrices via rotation + scaling
3. Evaluate bivariate Gaussian at all grid targets via `einsum`
4. Normalize per player to [0, 1]
5. Sigmoid team aggregation

Key guards:
- `SPEED_FLOOR = 0.1` → isotropic
- `ALPHA_CEIL = 0.99` → prevents singular covariance
- When `decompose=True`, also pass `player_team_ids=players["team_id"].to_numpy()` to `PitchControlSurface`

~150-180 lines.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_fernandez_bornn.py -v --tb=short`

Expected: All PASSED.

---

## Task 5: Dispatch layer + `__init__.py`

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_dispatch.py`
- Modify: `silly_kicks/tracking/pitch_control/__init__.py`
- Create: `tests/tracking/pitch_control/test_dispatch.py`

- [ ] **Step 1: Write dispatch tests**

```python
# tests/tracking/pitch_control/test_dispatch.py
"""Tests for pitch control dispatch layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    FernandezBornnParams,
    SpearmanParams,
    VoronoiParams,
    compute_pitch_control,
    compute_pitch_control_at_points,
)


def _make_frame(with_velocity=True):
    """Standard 2v2 test frame."""
    rows = [
        {"player_id": 1, "team_id": 10, "x": 30, "y": 34,
         "is_ball": False, "is_goalkeeper": True},
        {"player_id": 2, "team_id": 10, "x": 50, "y": 50,
         "is_ball": False, "is_goalkeeper": False},
        {"player_id": 3, "team_id": 20, "x": 70, "y": 34,
         "is_ball": False, "is_goalkeeper": True},
        {"player_id": 4, "team_id": 20, "x": 80, "y": 20,
         "is_ball": False, "is_goalkeeper": False},
        {"player_id": np.nan, "team_id": np.nan, "x": 52.5, "y": 34,
         "is_ball": True, "is_goalkeeper": False},
    ]
    if with_velocity:
        for r in rows:
            r["vx"] = 0.0
            r["vy"] = 0.0
    return pd.DataFrame(rows)


class TestDispatchRouting:
    def test_spearman_default(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, attacking_team_id=10)
        assert s.method == "spearman"

    def test_fernandez_bornn(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, 10, method="fernandez_bornn")
        assert s.method == "fernandez_bornn"

    def test_voronoi(self):
        frame = _make_frame()
        s = compute_pitch_control(frame, 10, method="voronoi")
        assert s.method == "voronoi"

    def test_wrong_params_type_raises(self):
        frame = _make_frame()
        with pytest.raises(TypeError):
            compute_pitch_control(frame, 10, method="spearman",
                                  params=VoronoiParams())


class TestVelocityRequirement:
    def test_spearman_without_velocity_raises(self):
        frame = _make_frame(with_velocity=False)
        with pytest.raises(ValueError, match="requires velocity"):
            compute_pitch_control(frame, 10, method="spearman")

    def test_fernandez_bornn_without_velocity_raises(self):
        frame = _make_frame(with_velocity=False)
        with pytest.raises(ValueError, match="requires velocity"):
            compute_pitch_control(frame, 10, method="fernandez_bornn")

    def test_voronoi_without_velocity_ok(self):
        frame = _make_frame(with_velocity=False)
        s = compute_pitch_control(frame, 10, method="voronoi")
        assert s.method == "voronoi"


class TestBallPositionInference:
    def test_infers_from_ball_row(self):
        frame = _make_frame()
        # Ball at (52.5, 34) in the frame
        s = compute_pitch_control(frame, 10, method="spearman")
        # Should not crash — ball position auto-inferred
        assert s.surface.shape == (32, 50)

    def test_explicit_overrides_frame(self):
        frame = _make_frame()
        s = compute_pitch_control(
            frame, 10, method="spearman", ball_position=(10, 10)
        )
        assert s.surface.shape == (32, 50)


class TestComputeAtPoints:
    def test_batch_point_query(self):
        frame = _make_frame()
        targets = np.array([[30, 34], [70, 34], [52.5, 34]], dtype="float64")
        result = compute_pitch_control_at_points(frame, targets, 10)
        assert result.shape == (3,)
        assert (result >= 0).all() and (result <= 1).all()
        # Attacker near (30, 34), defender near (70, 34)
        assert result[0] > result[1]

    def test_empty_targets(self):
        frame = _make_frame()
        targets = np.empty((0, 2))
        result = compute_pitch_control_at_points(frame, targets, 10)
        assert result.shape == (0,)


class TestOffPitchBall:
    def test_ball_outside_bounds_treated_as_none(self):
        frame = _make_frame()
        # Ball at x=200 (off-pitch per TRACKING_CONSTRAINTS)
        s = compute_pitch_control(frame, 10, ball_position=(200, 34))
        # Should not crash; treated as no ball conditioning
        assert s.surface.shape == (32, 50)
```

- [ ] **Step 2: Implement `_dispatch.py`**

~120 lines implementing:
- `_extract_frame_data()` — shared input extraction with `.astype(bool)` pattern
- `_infer_ball_position()` — explicit > frame ball row > None
- `compute_pitch_control()` — main router
- `compute_pitch_control_at_points()` — batch point queries

- [ ] **Step 3: Update `__init__.py` with full re-exports**

Add `compute_pitch_control` and `compute_pitch_control_at_points` to `__all__` and imports.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/ -v --tb=short`

Expected: All PASSED.

---

## Task 6: Numba acceleration (optional)

**Files:**
- Create: `silly_kicks/tracking/pitch_control/_numba_kernels.py`
- Create: `tests/tracking/pitch_control/test_numba_parity.py`

- [ ] **Step 1: Write numba parity tests**

```python
# tests/tracking/pitch_control/test_numba_parity.py
"""Golden-master tests: numba kernels produce identical output to NumPy."""
from __future__ import annotations

import numpy as np
import pytest

try:
    from silly_kicks.tracking.pitch_control._numba_kernels import (
        gaussian_influence_numba,
        influence_numba,
        tti_numba,
    )
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from silly_kicks.tracking.pitch_control._spearman import _compute_influence, _compute_tti

pytestmark = pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")


class TestTTIParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 105, (22, 2))
        vel = rng.uniform(-5, 5, (22, 2))
        targets = np.column_stack([
            np.linspace(0, 105, 50).repeat(32),
            np.tile(np.linspace(0, 68, 32), 50),
        ])
        numpy_out = _compute_tti(pos, vel, targets, 0.7, 7.0)
        numba_out = tti_numba(pos, vel, targets, 0.7, 7.0)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestInfluenceParity:
    def test_fixed_seed_parity(self):
        rng = np.random.default_rng(123)
        team_tti = rng.uniform(0.5, 5.0, (11, 1600))
        opp_min = rng.uniform(0.5, 5.0, (1600,))
        numpy_out = _compute_influence(team_tti, opp_min, 0.45)
        numba_out = influence_numba(team_tti, opp_min, 0.45)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-12)


class TestGaussianInfluenceParity:
    def test_fixed_seed_parity(self):
        from silly_kicks.tracking.pitch_control._fernandez_bornn import _compute_gaussian_influence

        rng = np.random.default_rng(456)
        targets = rng.uniform(0, 105, (1600, 2))
        mu = rng.uniform(20, 80, (11, 2))
        # Build per-player inverse covariance and determinants from random PD matrices
        inv_cov = np.zeros((11, 2, 2))
        det_cov = np.zeros(11)
        for i in range(11):
            A = rng.standard_normal((2, 2))
            cov = A @ A.T + 0.1 * np.eye(2)
            inv_cov[i] = np.linalg.inv(cov)
            det_cov[i] = np.linalg.det(cov)
        numpy_out = _compute_gaussian_influence(targets, mu, inv_cov, det_cov)
        numba_out = gaussian_influence_numba(targets, mu, inv_cov, det_cov)
        np.testing.assert_allclose(numpy_out, numba_out, rtol=1e-10)
```

- [ ] **Step 2: Implement `_numba_kernels.py`**

~150 lines with `@numba.njit(cache=True)` kernels mirroring the NumPy implementations.

- [ ] **Step 3: Wire numba dispatch into `_spearman.py` and `_fernandez_bornn.py`**

Add try/except import at module level; dispatch in `_compute_tti` and `_compute_influence`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_numba_parity.py -v --tb=short`

Expected: All PASSED (or all SKIPPED if numba not installed).

---

## Task 7: Action-coupled VAEP integration

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `silly_kicks/atomic/tracking/features.py`
- Create: `tests/tracking/pitch_control/test_action_coupled.py`

- [ ] **Step 1: Write action-coupled tests**

```python
# tests/tracking/pitch_control/test_action_coupled.py
"""Tests for pitch control VAEP integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import (
    add_pitch_control,
    pitch_control_at_action,
    pitch_control_default_xfns,
    pitch_control_xfns,
)


def _make_actions():
    """Minimal actions DataFrame."""
    return pd.DataFrame({
        "action_id": [1, 2],
        "game_id": [1, 1],
        "period_id": [1, 1],
        "time_seconds": [10.0, 20.0],
        "team_id": [10, 10],
        "player_id": [1, 2],
        "start_x": [30.0, 50.0],
        "start_y": [34.0, 34.0],
        "end_x": [50.0, 70.0],
        "end_y": [34.0, 34.0],
        "type_id": [0, 0],
    })


def _make_frames():
    """Minimal frames DataFrame matching action timestamps."""
    rows = []
    for t in [10.0, 20.0]:
        for pid, tid, x, y in [(1, 10, 30, 34), (2, 10, 50, 50),
                                (3, 20, 70, 34), (4, 20, 80, 20)]:
            rows.append({
                "game_id": 1, "period_id": 1, "frame_id": int(t * 25),
                "time_seconds": t, "frame_rate": 25.0,
                "player_id": pid, "team_id": tid,
                "x": x, "y": y, "vx": 0.0, "vy": 0.0,
                "is_ball": False, "is_goalkeeper": pid in (1, 3),
                "speed": 0.0, "speed_source": "derived",
                "z": np.nan, "ball_state": "alive",
                "team_attacking_direction": "ltr",
                "confidence": np.nan, "visibility": np.nan,
                "source_provider": "sportec",
                "is_goalkeeper_source": "native",
            })
        # Ball row
        rows.append({
            "game_id": 1, "period_id": 1, "frame_id": int(t * 25),
            "time_seconds": t, "frame_rate": 25.0,
            "player_id": np.nan, "team_id": np.nan,
            "x": 52.5, "y": 34.0, "vx": 0.0, "vy": 0.0,
            "is_ball": True, "is_goalkeeper": False,
            "speed": 0.0, "speed_source": "derived",
            "z": np.nan, "ball_state": "alive",
            "team_attacking_direction": np.nan,
            "confidence": np.nan, "visibility": np.nan,
            "source_provider": "sportec",
            "is_goalkeeper_source": np.nan,
        })
    return pd.DataFrame(rows)


class TestPitchControlAtAction:
    def test_returns_series(self):
        actions = _make_actions()
        frames = _make_frames()
        result = pitch_control_at_action(actions, frames)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_values_in_bounds(self):
        actions = _make_actions()
        frames = _make_frames()
        result = pitch_control_at_action(actions, frames)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


class TestAddPitchControl:
    def test_adds_column(self):
        actions = _make_actions()
        frames = _make_frames()
        result = add_pitch_control(actions, frames)
        assert "pitch_control_at_ball__spearman" in result.columns


class TestXfnFactory:
    def test_default_xfns_is_list(self):
        assert isinstance(pitch_control_default_xfns, list)
        assert len(pitch_control_default_xfns) == 1

    def test_xfn_has_frame_aware_marker(self):
        xfn = pitch_control_xfns("spearman")[0]
        assert getattr(xfn, "_frame_aware", False) is True

    def test_introspection_mode_no_crash(self):
        """VAEP fit-time introspection: 10-row dummy, frames=None."""
        xfn = pitch_control_xfns("spearman")[0]
        dummy = pd.DataFrame({
            "game_id": range(10), "period_id": 1,
            "time_seconds": range(10), "team_id": 1,
            "player_id": 1, "start_x": 50, "start_y": 34,
            "end_x": 60, "end_y": 34, "type_id": 0,
            "result_id": 0, "bodypart_id": 0,
            "action_id": range(10), "original_event_id": range(10),
            "score_home": 0, "score_away": 0,
        })
        states = [dummy] * 3
        result = xfn(states, None)
        assert result.shape[0] == 10
        assert result.isna().all().all()
```

- [ ] **Step 2: Implement action-coupled surface in `features.py`**

Add to `silly_kicks/tracking/features.py`:
- `pitch_control_at_action()` — per-Series helper
- `add_pitch_control()` — aggregator
- `pitch_control_xfns()` — factory
- `pitch_control_default_xfns` — module-level list

~80 lines following the existing pattern from `add_action_context`.

- [ ] **Step 3: Mirror in `silly_kicks/atomic/tracking/features.py`**

Atomic variant uses `x, y` instead of `start_x, start_y`. ~20 lines.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/tracking/pitch_control/test_action_coupled.py -v --tb=short`

Expected: All PASSED.

---

## Task 8: Physical invariants

**Files:**
- Create: `tests/invariants/test_pitch_control_invariants.py`

- [ ] **Step 1: Write invariant tests**

```python
# tests/invariants/test_pitch_control_invariants.py
"""Physical invariant tests for all pitch control methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    FernandezBornnParams,
    SpearmanParams,
    VoronoiParams,
    compute_pitch_control,
)

METHODS = ["spearman", "fernandez_bornn", "voronoi"]


def _make_frame(att_pos, def_pos, att_vel=None, def_vel=None):
    rows = []
    for i, (x, y) in enumerate(att_pos):
        vx = att_vel[i][0] if att_vel else 0.0
        vy = att_vel[i][1] if att_vel else 0.0
        rows.append({"player_id": 100+i, "team_id": 1, "x": x, "y": y,
                     "vx": vx, "vy": vy, "is_ball": False, "is_goalkeeper": i==0})
    for i, (x, y) in enumerate(def_pos):
        vx = def_vel[i][0] if def_vel else 0.0
        vy = def_vel[i][1] if def_vel else 0.0
        rows.append({"player_id": 200+i, "team_id": 2, "x": x, "y": y,
                     "vx": vx, "vy": vy, "is_ball": False, "is_goalkeeper": i==0})
    rows.append({"player_id": np.nan, "team_id": np.nan, "x": 52.5, "y": 34,
                 "vx": 0, "vy": 0, "is_ball": True, "is_goalkeeper": False})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", METHODS)
class TestBounds:
    def test_surface_in_unit_interval(self, method):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_pitch_control(frame, 1, method=method)
        assert (s.surface >= 0.0).all()
        assert (s.surface <= 1.0).all()


@pytest.mark.parametrize("method", METHODS)
class TestGridBounds:
    def test_grid_within_pitch(self, method):
        frame = _make_frame([(50, 34)], [(60, 34)])
        s = compute_pitch_control(frame, 1, method=method)
        assert s.grid_x[0] >= 0 and s.grid_x[-1] <= 105
        assert s.grid_y[0] >= 0 and s.grid_y[-1] <= 68


class TestSelfDominance:
    """Player on a cell with distant opponents → high control."""

    @pytest.mark.parametrize("method,threshold", [
        ("spearman", 0.95),
        ("voronoi", 0.95),
        ("fernandez_bornn", 0.80),
    ])
    def test_player_on_cell_distant_opponents(self, method, threshold):
        # Attacker at (50, 34), defenders > 40m away
        frame = _make_frame([(50, 34)], [(95, 60)])
        s = compute_pitch_control(frame, 1, method=method)
        assert s.at_point(50, 34) > threshold


@pytest.mark.parametrize("method", METHODS)
class TestSymmetry:
    def test_mirrored_teams_near_half(self, method):
        frame = _make_frame([(26.25, 34)], [(78.75, 34)])
        s = compute_pitch_control(frame, 1, method=method)
        center = s.at_point(52.5, 34.0)
        assert 0.35 < center < 0.65


@pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
class TestMonotonicity:
    def test_closer_player_higher_control(self, method):
        # Attacker 10m from target, defender 40m from target
        frame = _make_frame([(45, 34)], [(95, 34)])
        s = compute_pitch_control(frame, 1, method=method)
        assert s.at_point(50, 34) > 0.5


@pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
class TestVelocityEffect:
    def test_running_toward_increases_control(self, method):
        frame_static = _make_frame([(30, 34)], [(70, 34)],
                                   att_vel=[(0, 0)], def_vel=[(0, 0)])
        frame_running = _make_frame([(30, 34)], [(70, 34)],
                                    att_vel=[(6, 0)], def_vel=[(0, 0)])
        s_static = compute_pitch_control(frame_static, 1, method=method)
        s_running = compute_pitch_control(frame_running, 1, method=method)
        # Control at a point ahead of the attacker should increase
        assert s_running.at_point(50, 34) > s_static.at_point(50, 34)


class TestDecompositionConsistency:
    def test_spearman_sum_reconstructs(self):
        frame = _make_frame([(30, 34), (40, 50)], [(70, 34), (80, 20)])
        s = compute_pitch_control(frame, 1, method="spearman", decompose=True)
        att_mask = np.isin(s.player_ids, [100, 101])
        att_sum = s.per_player_influence[att_mask].sum(axis=0)
        all_sum = s.per_player_influence.sum(axis=0)
        reconstructed = np.where(all_sum > 1e-10, att_sum / all_sum, 0.5)
        np.testing.assert_allclose(s.surface, reconstructed, atol=1e-8)

    def test_voronoi_binary_sums_to_one(self):
        frame = _make_frame([(30, 34)], [(70, 34)])
        s = compute_pitch_control(frame, 1, method="voronoi", decompose=True)
        assert (s.per_player_influence.sum(axis=0) == 1.0).all()


class TestNoNaN:
    def test_fernandez_bornn_near_max_speed(self):
        frame = _make_frame([(50, 34)], [(70, 34)],
                            att_vel=[(12.99, 0)], def_vel=[(0, 0)])
        s = compute_pitch_control(frame, 1, method="fernandez_bornn")
        assert not np.isnan(s.surface).any()
        assert not np.isinf(s.surface).any()
```

- [ ] **Step 2: Run invariant tests**

Run: `python -m pytest tests/invariants/test_pitch_control_invariants.py -v --tb=short`

Expected: All PASSED.

---

## Task 9: Performance benchmark

**Files:**
- Create: `tests/tracking/pitch_control/test_perf_budget.py`

- [ ] **Step 1: Write benchmark tests**

```python
# tests/tracking/pitch_control/test_perf_budget.py
"""pytest-benchmark gates per spec §10.6 performance budget.

Single-frame 22-player pitch control must complete within 50ms (Linux) / 75ms (Windows).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import (
    FernandezBornnParams,
    SpearmanParams,
    VoronoiParams,
    compute_pitch_control,
)

_BUDGET = 0.05 if sys.platform != "win32" else 0.075  # seconds


@pytest.fixture(scope="module")
def full_frame_22():
    """Realistic 22-player frame (11v11 + ball)."""
    rng = np.random.default_rng(777)
    rows = []
    for i in range(11):
        rows.append({
            "player_id": 100 + i, "team_id": 1,
            "x": rng.uniform(5, 100), "y": rng.uniform(5, 63),
            "vx": rng.uniform(-3, 3), "vy": rng.uniform(-3, 3),
            "is_ball": False, "is_goalkeeper": i == 0,
        })
    for i in range(11):
        rows.append({
            "player_id": 200 + i, "team_id": 2,
            "x": rng.uniform(5, 100), "y": rng.uniform(5, 63),
            "vx": rng.uniform(-3, 3), "vy": rng.uniform(-3, 3),
            "is_ball": False, "is_goalkeeper": i == 0,
        })
    rows.append({
        "player_id": np.nan, "team_id": np.nan,
        "x": 52.5, "y": 34.0, "vx": 0, "vy": 0,
        "is_ball": True, "is_goalkeeper": False,
    })
    return pd.DataFrame(rows)


def test_spearman_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="spearman")
    assert result.surface.shape[0] > 0
    assert benchmark.stats.stats.mean < _BUDGET


def test_fernandez_bornn_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="fernandez_bornn")
    assert result.surface.shape[0] > 0
    assert benchmark.stats.stats.mean < _BUDGET


def test_voronoi_single_frame_budget(benchmark, full_frame_22) -> None:
    result = benchmark(compute_pitch_control, full_frame_22, 1, method="voronoi")
    assert result.surface.shape[0] > 0
    assert benchmark.stats.stats.mean < _BUDGET * 0.5  # Voronoi is trivially fast
```

- [ ] **Step 2: Run benchmark tests**

Run: `python -m pytest tests/tracking/pitch_control/test_perf_budget.py -v --benchmark-disable`

Expected: All PASSED (benchmark-disable just verifies correctness; full benchmarks run in CI with `--benchmark-enable`).

---

## Task 10: Provider coverage + lakehouse parity

**Files:**
- Create: `tests/tracking/pitch_control/test_lakehouse_parity.py`
- Possibly regenerate: `tests/datasets/tracking/` fixtures

- [ ] **Step 1: Write lakehouse parity test**

Port a known input/output pair from lakehouse `pitch_control.py` (converting
from 120x80 → 105x68 coordinates) and assert numerical agreement within
tolerance.

```python
# tests/tracking/pitch_control/test_lakehouse_parity.py
"""Cross-reference Spearman output against lakehouse implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control import SpearmanParams, compute_pitch_control


def test_spearman_matches_lakehouse_tti_formula():
    """Verify TTI formula produces same values as lakehouse _tti_numpy.

    Uses a known 3-player setup with hand-computed expected values.
    Lakehouse formula (identical math, different coordinate system):
        TTI = reaction_time + (-v_proj + sqrt(v_proj^2 + 2*a*d)) / a
    """
    from silly_kicks.tracking.pitch_control._spearman import _compute_tti

    # Player at (10, 10), velocity (3, 0), target at (20, 10)
    # d = 10, v_proj = 3 (full projection along x)
    # TTI = 0.7 + (-3 + sqrt(9 + 140)) / 7 = 0.7 + (-3 + 12.2066) / 7
    pos = np.array([[10.0, 10.0]])
    vel = np.array([[3.0, 0.0]])
    target = np.array([[20.0, 10.0]])
    tti = _compute_tti(pos, vel, target, 0.7, 7.0)
    expected = 0.7 + (-3.0 + np.sqrt(9.0 + 140.0)) / 7.0
    np.testing.assert_allclose(tti[0, 0], expected, rtol=1e-12)


def test_spearman_surface_structure_matches_lakehouse():
    """Verify surface spatial structure: attacker side > 0.5, defender side < 0.5."""
    frame = pd.DataFrame([
        {"player_id": 1, "team_id": 10, "x": 25, "y": 34, "vx": 2, "vy": 0,
         "is_ball": False, "is_goalkeeper": True},
        {"player_id": 2, "team_id": 10, "x": 45, "y": 34, "vx": 1, "vy": 0,
         "is_ball": False, "is_goalkeeper": False},
        {"player_id": 3, "team_id": 20, "x": 60, "y": 34, "vx": -1, "vy": 0,
         "is_ball": False, "is_goalkeeper": False},
        {"player_id": 4, "team_id": 20, "x": 80, "y": 34, "vx": 0, "vy": 0,
         "is_ball": False, "is_goalkeeper": True},
        {"player_id": np.nan, "team_id": np.nan, "x": 52.5, "y": 34,
         "vx": 0, "vy": 0, "is_ball": True, "is_goalkeeper": False},
    ])
    s = compute_pitch_control(frame, 10, method="spearman",
                              params=SpearmanParams(grid_cells_x=20, grid_cells_y=13))
    # Attacker half (x < 52.5) should have higher control
    mid_idx = len(s.grid_x) // 2
    att_mean = s.surface[:, :mid_idx].mean()
    def_mean = s.surface[:, mid_idx:].mean()
    assert att_mean > def_mean
```

- [ ] **Step 2: Verify existing synthetic fixtures have velocity data**

Check that `tests/datasets/tracking/{provider}/` parquet fixtures contain `vx`,
`vy` columns. If missing (pre-PR-S24 fixtures may not have derived velocities),
regenerate using `derive_velocities()` on the raw fixture or add velocity
columns to the fixture generators.

- [ ] **Step 3: Write provider coverage smoke test**

```python
# Add to tests/tracking/pitch_control/test_dispatch.py

@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
def test_provider_fixture_computes_without_error(provider):
    """Smoke test: each provider's fixture produces a valid surface."""
    # Load fixture, derive velocities if needed, compute PC
    ...
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/tracking/pitch_control/ tests/invariants/test_pitch_control_invariants.py -v --tb=short`

Expected: All PASSED.

---

## Task 11: Documentation + integration

**Files:**
- Create: `docs/superpowers/adrs/ADR-008-pitch-control.md`
- Modify: `NOTICE`
- Modify: `pyproject.toml`
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `TODO.md`

- [ ] **Step 1: Write ADR-008**

Document the four architectural decisions:
1. Subpackage pattern for complex spatial computations
2. `PitchControlSurface` rich-return-type precedent
3. Optional numba acceleration contract
4. Optional xarray bridge contract

- [ ] **Step 2: Add NOTICE entries**

Add Spearman 2017, Fernandez/Bornn 2018, and Shaw 2020 to
Mathematical/Methodological References section.

- [ ] **Step 3: Update `pyproject.toml`**

Add `numba` and `xarray` to `[project.optional-dependencies]`.

- [ ] **Step 4: Update `silly_kicks/tracking/__init__.py`**

Add pitch_control re-exports to `__all__` and imports:
- `PitchControlSurface`
- `SpearmanParams`, `FernandezBornnParams`, `VoronoiParams`
- `compute_pitch_control`, `compute_pitch_control_at_points`

Add `pitch_control_xfns`, `pitch_control_default_xfns`, `add_pitch_control`
to the features re-exports.

- [ ] **Step 5: Update `TODO.md`**

Delete TF-7 row from On Deck. Bump header date.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`

Expected: All PASSED (including existing tests — no regressions).

- [ ] **Step 7: Lint + type check**

Run: `ruff check . && ruff format --check . && uv run pyright`

Expected: Clean.

- [ ] **Step 8: `/final-review`**

Run the final-review skill before committing.

---

## Expected test counts

| Test file | Approx tests |
|-----------|-------------|
| test_params.py | 8 |
| test_surface.py | 14 |
| test_voronoi.py | 12 |
| test_spearman.py | 14 |
| test_fernandez_bornn.py | 13 |
| test_dispatch.py | 12 |
| test_numba_parity.py | 3 (skip if no numba) |
| test_action_coupled.py | 6 |
| test_perf_budget.py | 3 |
| test_lakehouse_parity.py | 2 |
| test_pitch_control_invariants.py | ~20 (parametrized) |
| **Total** | **~110** |
