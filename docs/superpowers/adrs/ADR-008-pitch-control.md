# ADR-008: Pitch Control Subpackage Architecture

**Status:** Accepted
**Date:** 2026-05-05
**Drivers:** TF-7 pitch control implementation; GKDV research program (TF-15..TF-19)

## Context

Pitch control is the first spatial-computation module in silly-kicks that requires:
1. A multi-model dispatch pattern (Spearman/Fernandez-Bornn/Voronoi)
2. A rich return type beyond a scalar or Series
3. Optional compiled acceleration (numba)
4. A grid-based surface that may need interoperability with xarray

The existing `tracking/` namespace handles per-action scalar features. Pitch control
returns a 2D surface per frame and needs its own internal decomposition.

## Decisions

### D1: Subpackage pattern for spatial computations

Pitch control lives in `silly_kicks/tracking/pitch_control/` as a subpackage with
private modules per model (`_spearman.py`, `_fernandez_bornn.py`, `_voronoi.py`),
shared params (`_params.py`), a rich return type (`_surface.py`), and a dispatch
router (`_dispatch.py`).

**Rationale:** Each model is 60-120 lines. Keeping them in separate files enables
focused reading, independent testing, and future additions without growing any
single file beyond comfortable context-window size.

### D2: PitchControlSurface rich-return-type

`compute_pitch_control()` returns a frozen `PitchControlSurface` dataclass rather
than a raw numpy array. The surface carries grid coordinates, method metadata,
optional per-player decomposition, and convenience methods (`at_point`, `at_points`,
`control_in_region`, `player_share`, `to_xarray`).

**Rationale:** Downstream consumers (VAEP xfns, visualization, lakehouse export)
need different projections of the result. A rich type eliminates the need to
separately track grid metadata alongside arrays.

### D3: Optional numba acceleration contract

`_numba_kernels.py` provides `@njit(cache=True)` mirrors of the numpy kernels.
Import is guarded by try/except in each model module; dispatch is a simple
`if _HAS_NUMBA: return numba_kernel(...)` at the top of each function body.

**Rationale:** Numba provides 5-10x speedup for full-match batch processing but
is not a runtime dependency. Both numpy fallback and numba paths are exercised
in CI (numba is a test dependency). Parity is enforced by golden-master tests.

### D4: Optional xarray bridge contract

`PitchControlSurface.to_xarray()` returns an `xr.DataArray` with labeled
dimensions (y, x). Import is deferred and guarded; `ImportError` raises with
install instructions.

**Rationale:** xarray interop is valuable for visualization (hvplot, matplotlib
pcolormesh) and netCDF export but should not be a required dependency for the
core computation path.

## Consequences

- New spatial models (e.g., EPV, pass probability) can follow the same subpackage
  pattern without API-breaking changes.
- The `PitchControlSurface` type becomes a consumer contract; field additions
  require CHANGELOG enumeration.
- Numba cache files (`.nbi`/`.nbc`) are gitignored.
