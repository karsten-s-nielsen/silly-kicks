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

## Amendment (2026-05-28, silly-kicks 3.25.0): shared surface + linked-frame restriction

**Drivers:** lakehouse AC-1 perf handoff — `compute_pitch_control` was invoked
independently by `add_obso`, `add_cover_shadows`, `add_gk_influence`,
`add_player_influence`, `add_space_creation`, and `pitch_control_at_action` on
overlapping frames, recomputing the same per-frame surface many times per match.
The standard for every perf change here is **bit-identical output** — never trade
accuracy for speed.

### D5: Per-pass surface cache threaded via an optional kwarg (not global memoization)

`PitchControlCache` (`pitch_control/_cache.py`) memoizes canonical per-frame
surfaces keyed on `(game_id, period_id, frame_id, attacking_team_id, method,
params, ball_position, decompose)` (with `params=None` normalized to the method
default). It is threaded through the aggregators via an optional
`pitch_control_cache: PitchControlCache | None = None` kwarg, mirroring the
established `links` pre-linking pattern. Each aggregator instantiates a fresh
local cache by default (within-pass reuse, e.g. OBSO's overlapping pass windows);
a pipeline caller pre-builds one cache and passes it to all steps for cross-family
reuse.

**Rationale:** global memoization (an LRU/dict on `compute_pitch_control`) was
rejected — it violates the hexagonal "zero global state mutation" rule, is unsafe
under frame mutation (velocity derivation re-creates rows for the same
`frame_id`), and is fragile under Databricks `applyInPandas`. The explicit threaded
object has none of these problems and matches the existing `links` idiom.

### D6: Only canonical-frame surfaces are cached; counterfactuals stay direct

The cache is valid **only** for surfaces computed on the original tracking frame.
Counterfactual surfaces — `cover_shadows`' defender-removed `surface_reduced` and
`space_creation`'s leave-one-out `removed_surface` — share the canonical frame's
`(game_id, period_id, frame_id)` but have different content, so they are never
routed through the cache (direct `compute_pitch_control` calls). The cache also
bypasses (computes uncached) when a frame does not resolve to a single
`(game_id, period_id, frame_id)` or the method is unknown.

`decompose` is part of the key: `decompose=True` consumers (`gk_influence`,
`player_influence`) and `decompose=False` consumers share within their group
(≈3× reduction). Promoting a decomposed surface to serve a non-decomposed request
is a possible future refinement (deferred — requires verifying the aggregate field
is identical).

### D7: Linked-frame restriction must pin attacking direction (DAS)

When `links` is supplied, `add_das` / `add_shape_graph` restrict the expensive
per-frame computation to the action-linked frames. For `add_shape_graph` this is
trivially bit-identical (pure per-frame snapshot). For DAS it is **not** naively
bit-identical: `accessible-space` infers attacking direction per period from the
mean x-position over the **input** frames, so a restricted ~3-frame subset can
infer a flipped sign and change DAS. Resolution: `_pin_attacking_direction` runs
`accessible-space`'s own `infer_playing_direction` on the **full** frames first
(cheap groupby-mean), attaches the result, then restricts — and `get_individual_das`
gained an `attacking_direction_col` passthrough so the simulation uses the pinned
direction. Provably bit-identical for any data; e2e-verified.

### D8: Ghost-GK linked-frame restriction preserves cross-frame deps, not direction (3.26.0)

`add_ghost_gk` / `ghost_gk_xfns` restrict the expensive per-sample density KDE
(and, since the same change, the heavy per-frame feature extraction) to the
action-linked frames via a `link_frame_ids` kwarg on `compute_ghost_gk`
(`add_ghost_gk` derives it from its link pointers; `ghost_gk_xfns` from the union
of its three gamestate slots). Unlike D7's DAS case, the obstacle here is **not**
direction inference: `_extract_all_ghost_gk_features` carries two cross-frame
dependencies — a per-period defending-goal mean-x (`groupby` over the whole
period) and a cross-period one-step velocity state (each frame's velocity vs its
true predecessor for that `(game_id, gk_team)`). Resolution: the extractor still
**walks every frame** to maintain the velocity state and computes the goal-mean
over the full frames, and only builds a feature row / runs the KDE for linked
frames. The per-sample KDE has zero cross-sample coupling, so the restricted
output is byte-identical; golden tests cover the goal-flip + velocity edge cases
plus a discrimination test proving a naive frame pre-filter would **not** be
bit-identical. Measured (bundled model): ~100× faster per 250-frame fan-out batch;
the residual is the irreducible per-linked-frame KDE (~4.4 s/eval), not extraction
(~4.7%). PR-S66.

### Consequences (amendment)

- `pitch_control_cache` joins `links` as a standard optional optimization kwarg on
  tracking aggregators; new pitch-control consumers should accept and thread it.
- The cover_shadows lightweight nested-loop pruning was **not** adopted: it is not
  bit-identical because `lane_control` depends on a global greedy man-marker
  assignment (removing an out-of-corridor defender can change an in-corridor
  defender's lane-blocker status). Tracked in TODO for a dedicated, golden-master-
  validated effort.
- The VAEP `*_xfns` transformers do not yet share one cache across families in a
  single pass (each keeps its own per-frame precompute); tracked in TODO.
