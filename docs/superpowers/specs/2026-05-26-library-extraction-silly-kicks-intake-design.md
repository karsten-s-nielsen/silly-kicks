# Library Extraction: silly-kicks Intake Design

**Date**: 2026-05-26
**Status**: Revised after lakehouse review (v3)
**Context**: Response to lakehouse spec `2026-05-26-library-extraction-architecture-design.md`
**Scope**: What silly-kicks needs to do to absorb the Phase 1 analytics modules

---

## 1. Critical Review of the Lakehouse Spec

### 1.1 The `silly_kicks.analytics` Namespace is Wrong

The lakehouse spec proposes a new `silly_kicks/analytics/` subpackage. This breaks silly-kicks' established architecture:

- **team_shape** already ships in `tracking/_team_shape.py` (3.11.0, PR-S33)
- **line_breaking** already ships in `tracking/_line_breaking.py` (3.11.0, PR-S33)
- **smoothing** already ships in `tracking/preprocess/` (3.6.0, PR-S24) with more capability than the lakehouse version
- **OBSO** and **space_creation** are per-frame tracking computations that depend on `tracking/pitch_control/`
- **elastic_sync** is event-tracking linkage — `tracking/` is its natural home

Creating a parallel `analytics/` namespace would split related code across two trees, force users to discover features in two places, and break the pattern where every tracking primitive has a corresponding `add_*` aggregator + `*_xfns` VAEP factory in the same namespace.

**Decision: All modules go into `silly_kicks/tracking/`, following existing patterns.** The lakehouse import path becomes `from silly_kicks.tracking import ...`, not `from silly_kicks.analytics import ...`.

### 1.2 Overlap Assessment

| Lakehouse Module | silly-kicks Status | Action |
|---|---|---|
| team_shape | **SHIPPED** (3.11.0) — convex hull, centroid, stretch, length/width | **ENHANCE**: add Ward inter-line gaps + defensive_line_height from lakehouse version |
| line_breaking | **SHIPPED** (3.11.0) — Ward clustering + straddle test, batch kernel | **DONE**: lakehouse version is structurally identical |
| smoothing | **SHIPPED** (3.6.0) — SG/EMA + provider-aware dispatch + velocity derivation | **DONE**: silly-kicks version is strictly more capable |
| coordinates | N/A — lakehouse converts to StatsBomb 120x80 | **SKIP**: silly-kicks uses [0,105]x[0,68]; lakehouse's SB convention is irrelevant |
| array_utils | 12-line `_col_f64` helper | **SKIP**: trivial; inline `np.asarray(df[col], dtype=np.float64)` where needed |
| pitch_control | **SHIPPED** (3.13.0) — 3-flavor subpackage | **DONE**: already the dependency target for OBSO |
| shape_graph | Not in silly-kicks | **NEW**: Sotudeh 2026 Delaunay tactical position inference |
| obso | Not in silly-kicks | **NEW**: Spearman 2018 off-ball scoring opportunity |
| space_creation | Not in silly-kicks | **NEW**: Fernandez 2018 differential OBSO |
| pausa | Not in silly-kicks | **NEW**: Lee 2026 pass timing metric |
| elastic_sync | Not in silly-kicks (TF-34 deferred) | **NEW**: Kim 2025 ELASTIC event-tracking sync |

**Net new work: 5 modules.** The rest is already shipped or irrelevant.

### 1.3 Coordinate System Impact (TF-38)

TF-38 tracks the switch from bottom-left `[0,105]x[0,68]` to CDF center-origin `[-52.5,52.5]x[-34,34]` as a 4.0.0 breaking change. All new code ported from the lakehouse **must be coordinate-system-agnostic**:

- **No hardcoded pitch dimensions.** Use `pitch_length` / `pitch_width` parameters (or pull from a shared config) instead of literals `105.0` / `68.0`.
- **No hardcoded origin assumptions.** The lakehouse `coordinates.py` assumes StatsBomb 120x80 origin at bottom-left. silly-kicks will switch to center-origin. Any ported code that computes "half-pitch" as `x < 52.5` must use `x < 0` in center-origin or parameterize via `pitch_length / 2`.
- **Goal positions must be parameterized.** Currently `goal_x = 0` or `goal_x = 105`; post-TF-38 this becomes `goal_x = -52.5` or `goal_x = 52.5`. Use `pitch_length / 2` arithmetic, not constants.
- **Grid construction must respect origin.** OBSO/space_creation build evaluation grids. Use `np.linspace(x_min, x_max, n)` where min/max come from config, not hardcoded `(0, 105)`.

**Concrete implication for lakehouse review**: the lakehouse's `coordinates.py` does NOT move to silly-kicks. It converts to StatsBomb 120x80 which is a lakehouse-internal convention that silly-kicks will never adopt. Post-TF-38, silly-kicks will provide its own `coordinate_system` config surface for CDF interop.

### 1.4 Lakehouse `smoothing.py` is Superseded

The lakehouse's `smoothing.py` (67 lines, basic SG wrapper) is a strict subset of silly-kicks' `tracking/preprocess/` (8 files, SG + EMA + linear interpolation + velocity derivation + provider-aware dispatch + `PreprocessConfig` frozen dataclass). No import needed. The lakehouse should switch to `from silly_kicks.tracking.preprocess import smooth_frames` directly in Phase 2.

---

## 2. Task Inventory

### TF-39: Shape Graph (Sotudeh 2026 Delaunay Tactical Position Inference)

**New module**: `silly_kicks/tracking/_shape_graph.py`

**What it does**: Builds a stable subgraph from Delaunay triangulation by iteratively removing unstable edges (low angular stability). Then decomposes player coordinates into a 5x5 tactical role grid (B/DM/M/AM/F x L/LC/C/RC/R) via face-center decomposition.

**API surface**:
```python
@dataclass(frozen=True)
class ShapeGraph:
    edges: np.ndarray          # (n_edges, 2)
    faces: list[tuple[int, ...]]
    stabilities: np.ndarray    # per-edge
    points: np.ndarray         # (n_players, 2)

@dataclass(frozen=True)
class PositionLabel:
    vertical: str              # B | DM | M | AM | F
    horizontal: str            # L | LC | C | RC | R
    label: str                 # "DM-C"

def compute_shape_graph(positions: np.ndarray, *, stability_threshold: float = 45.0) -> ShapeGraph
def infer_positions(graph: ShapeGraph, positions: np.ndarray, *, attacking_direction: float) -> list[PositionLabel]
```

**Action coupling**: `add_shape_graph` aggregator attaching per-action team topology metrics (graph density, n_edges, mean stability). `shape_graph_xfns` VAEP factory.

**Dependencies**: numpy, scipy.spatial (Delaunay). No silly-kicks internal deps.

**Coordinate note**: `positions` is an (n, 2) array — coordinate-system-agnostic by construction. `attacking_direction` (radians) handles orientation. No pitch dimension constants needed.

**Source**: lakehouse `shape_graph_construction.py` (396 lines) + `shape_graph_inference.py` (586 lines). Consolidate into single module or keep as `_shape_graph_construction.py` + `_shape_graph_inference.py` per existing subpackage pattern if complexity warrants.

**Size**: ~980 lines lakehouse. Expect ~600-800 after adapting to silly-kicks patterns (remove lakehouse-specific imports, consolidate re-export module).

---

### TF-40: OBSO (Spearman 2018 Off-Ball Scoring Opportunity)

**New module**: `silly_kicks/tracking/_obso.py`

**What it does**: Computes per-cell off-ball scoring value as `PPCF(cell) x Transition(ball -> cell) x EPV(cell)`. Produces a continuous value surface indicating where the best off-ball opportunities are for the team in possession.

**API surface**:
```python
@dataclass(frozen=True)
class ObsoParams:
    grid_nx: int = 104
    grid_ny: int = 68
    pitch_length: float = 105.0    # TF-38: will become constructor param
    pitch_width: float = 68.0
    sigma_x: float = 26.25         # Gaussian distance weight (meters). Lakehouse uses 30.0 in SB 120x80 units.
    sigma_y: float = 17.0          # Gaussian distance weight (meters). Lakehouse uses 20.0 in SB 120x80 units.

@dataclass(frozen=True)
class ObsoSurface:
    values: np.ndarray              # (grid_ny, grid_nx)
    grid_x: np.ndarray              # (grid_nx,)
    grid_y: np.ndarray              # (grid_ny,)

def compute_obso_surface(
    pitch_control: PitchControlSurface,
    ball_position: tuple[float, float],
    *,
    transition_grid: np.ndarray | None = None,  # defaults to Gaussian-decay synthetic fallback
    epv_grid: np.ndarray | None = None,         # defaults to synthetic EPV
    params: ObsoParams | None = None,
) -> ObsoSurface

def compute_pass_obso(
    pass_window_frames: list[pd.DataFrame],     # pre-windowed tracking frames around the pass event
    target_position: tuple[float, float],
    attacking_team_id: str | int,
    *,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    params: ObsoParams | None = None,
    pitch_control_method: str = "spearman",
) -> dict[str, float]               # actual_obso, peak_obso, optimal_obso
```

**Action coupling**: `add_obso` aggregator producing per-action `obso_actual`, `obso_peak`, `obso_optimal` columns. `obso_xfns` VAEP factory.

**Dependencies**: numpy, pandas. Internal: `tracking.pitch_control` (PitchControlSurface, compute_pitch_control), `xthreat` (ExpectedThreat grid as default EPV).

**Coordinate note**: `ObsoParams` includes `pitch_length` / `pitch_width`. Grid construction uses `np.linspace(0, pitch_length, grid_nx)` (post-TF-38: `np.linspace(-pitch_length/2, pitch_length/2, grid_nx)`). **No hardcoded origin.**

**Windowing note**: `compute_pass_obso` accepts pre-windowed frame lists (same pattern as the lakehouse). The windowing logic (extracting a time window around each pass event, computing pitch control at each frame, finding peak/optimal across the window) lives in the lakehouse's ingestion layer (`src/ingestion/compute_obso.py`), not in `src/analytics/obso.py`. The silly-kicks v1 keeps this separation: `compute_pass_obso` is the analytics primitive accepting pre-windowed frames, and a convenience wrapper `obso_pass_window` (or integration via `add_obso`) handles the windowing using silly-kicks' existing `slice_around_event` infrastructure.

**Source**: lakehouse `obso.py` (283 lines) + windowing logic from `src/ingestion/compute_obso.py` (~80 lines). The lakehouse version uses StatsBomb 120x80 grids — must be re-parameterized to silly-kicks coordinate system.

---

### TF-41: Space Creation (Fernandez 2018 Differential OBSO)

**New module**: `silly_kicks/tracking/_space_creation.py`

**What it does**: Quantifies each player's contribution to team off-ball scoring opportunity via leave-one-out differential OBSO. "How much does the OBSO surface change when player X is removed?"

**API surface**:
```python
@dataclass(frozen=True)
class SpaceCreationParams:
    grid_nx: int = 104
    grid_ny: int = 68
    pitch_length: float = 105.0     # Lakehouse uses 120.0 (SB units) — must NOT copy verbatim
    pitch_width: float = 68.0       # Lakehouse uses 80.0 (SB units) — must NOT copy verbatim

def compute_space_created(
    frames: pd.DataFrame,
    attacking_team_id: str | int,
    xt_grid: np.ndarray,
    *,
    ball_position: tuple[float, float] | None = None,
    params: SpaceCreationParams | None = None,
    pitch_control_method: str = "spearman",
) -> pd.DataFrame                   # player_id, space_created
```

**Action coupling**: `add_space_creation` aggregator. `space_creation_xfns` VAEP factory.

**Dependencies**: Internal: `_obso` (compute_obso_surface), `tracking.pitch_control`.

**Source**: lakehouse `space_creation.py` (175 lines). Loop-invariant hoisting optimization (avoid redundant OBSO recomputation) must be preserved.

---

### TF-42: PAUSA (Lee et al. 2026 Pass Timing)

**New module**: `silly_kicks/tracking/_pausa.py`

**What it does**: Pure decomposition of pass quality into temporal judgment (did you pass at the right moment?) and spatial selection (did you pass to the right location?). Requires pre-computed OBSO triplet (actual, peak, optimal) per pass.

**API surface**:
```python
def compute_pausa(
    actual_obso: float,
    peak_obso: float,
    optimal_obso: float,
) -> dict[str, float]                # temporal, spatial, composite

def add_pausa(
    actions: pd.DataFrame,           # must have obso_actual, obso_peak, obso_optimal columns
) -> pd.DataFrame                    # + pausa_temporal, pausa_spatial, pausa_composite
```

**Action coupling**: `add_pausa` is pass-level, not frame-level. Operates on columns produced by `add_obso`. `pausa_xfns` VAEP factory.

**Dependencies**: numpy, pandas. Internal: requires `_obso` outputs as input columns.

**Coordinate note**: Coordinate-system-agnostic — operates on scalar OBSO values, not positions.

**Discoverability note**: PAUSA is purely action-level arithmetic (3 scalar divisions + a multiply), not a frame-level computation. It lives in `tracking/` because it's part of the OBSO chain, but `add_pausa` and `compute_pausa` should also be importable from the top-level `from silly_kicks.tracking import add_pausa` path (which they will be via `__init__.py` re-export, same as all other `add_*` helpers).

**Source**: lakehouse `pausa.py` (75 lines). Trivial port — 75 lines of pure arithmetic. Main risk is ensuring OBSO column naming convention matches.

---

### TF-43: ELASTIC Sync (Kim et al. 2025)

**New module**: `silly_kicks/tracking/_elastic_sync.py`

**What it does**: Aligns event timestamps to tracking frames by matching ball acceleration spikes and player-ball proximity features. Replaces/supplements the existing `link_actions_to_frames` nearest-time approach with a feature-based matching method.

**API surface**:
```python
@dataclass(frozen=True)
class ElasticSyncParams:
    window_seconds: float = 3.0
    frame_rate: float = 25.0        # canonical location for frame rate (used by both extract_ball_features and align)
    accel_weight: float = 0.6
    proximity_weight: float = 0.4
    min_confidence: float = 0.3

def extract_ball_features(
    frames: pd.DataFrame,            # TRACKING_FRAMES_COLUMNS schema
    *,
    params: ElasticSyncParams | None = None,  # frame_rate from params; defaults to 25.0
) -> pd.DataFrame                    # frame_id, ball_speed, ball_accel

def align_events_to_frames(
    events: pd.DataFrame,            # action_id, time_seconds, x, y, player_id
    frames: pd.DataFrame,            # TRACKING_FRAMES_COLUMNS schema
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame                    # action_id, frame_id, confidence
```

`extract_ball_features` is a standalone utility computing ball speed and acceleration from tracking frame deltas — useful independently of ELASTIC sync (e.g., for ball-in-play detection, shot speed analysis).

**Integration**: Can serve as an alternative linkage strategy for `link_actions_to_frames`, or as a refinement pass on top of the existing nearest-time approach. The existing `sync_score` (TF-6) measures link quality — ELASTIC could produce higher-quality links that `sync_score` then validates.

**Dependencies**: numpy, pandas. Internal: uses tracking schema conventions.

**Coordinate note**: Ball acceleration/proximity features are computed from frame deltas — coordinate-system-agnostic (distances and accelerations don't depend on origin).

**Source**: lakehouse `elastic_sync.py` (299 lines). Column naming must adapt from lakehouse conventions (`frame`, `timestamp`, `ball_x`) to silly-kicks conventions (`frame_id`, `time_seconds`, `x` with `is_ball=True`).

**Note**: This was already tracked as TF-34 (deferred). This spec promotes it since the lakehouse has a working implementation to port.

---

### TF-44: team_shape Enhancement (Ward Inter-Line Gaps)

**Existing module**: `silly_kicks/tracking/_team_shape.py`

**What changes**: The lakehouse version adds two metrics not in the current silly-kicks `compute_team_shape`:
1. **defensive_line_height** — deepest Ward cluster centroid along attacking axis
2. **inter_line_gaps** — distances between Ward cluster centroids

These are related to but distinct from the existing `_line_breaking.py` Ward clustering (which clusters opponents for pass intersection, not own-team for shape analysis).

**API change**: Add `defensive_line_height` and `inter_line_gaps` columns to the output of `compute_team_shape`. The `add_team_shape` aggregator already handles column propagation.

**Size**: ~30 lines of additional Ward clustering logic in `compute_team_shape`.

---

## 3. Implementation Sequence

```
TF-44 (team_shape enhance)     — standalone, smallest, validates porting pattern
TF-39 (shape_graph)            — standalone, no internal deps
TF-40 (OBSO)                   — depends on pitch_control (shipped) + xT (shipped)
TF-41 (space_creation)         — depends on TF-40
TF-42 (PAUSA)                  — depends on TF-40 outputs
TF-43 (ELASTIC sync)           — standalone, can parallel with TF-40-42
```

Each TF follows the established per-PR pattern:
- Private module `_<name>.py` in `tracking/`
- Per-frame primitive function
- `add_<name>` action-coupled aggregator (with `links` kwarg for pre-linking optimization)
- `<name>_xfns` VAEP factory
- Atomic mirror in `atomic/tracking/features.py`
- Tests: unit + snapshot + invariant
- NOTICE entry for academic attribution
- CHANGELOG entry

### Dependency Extra

scipy is already a core dependency. No new extras needed for these modules.

---

## 4. What Does NOT Move to silly-kicks

| Lakehouse Module | Reason |
|---|---|
| `coordinates.py` | Converts to StatsBomb 120x80 — a lakehouse-internal convention. silly-kicks will provide its own coordinate config post-TF-38. |
| `array_utils.py` | 12-line `_col_f64`. Trivial — inline where needed, no module. |
| `smoothing.py` | Superseded by `tracking/preprocess/` (shipped 3.6.0, strictly more capable). |
| `line_breaking.py` | Already shipped as `tracking/_line_breaking.py` (3.11.0). Structurally identical. |

---

## 5. Lakehouse Phase 2 Import Map

Once silly-kicks ships TF-39 through TF-43, the lakehouse switches imports:

| Lakehouse Current | silly-kicks Target | Migration |
|---|---|---|
| `from analytics.team_shape import compute_team_shape` | `from silly_kicks.tracking import compute_team_shape` | Drop-in |
| `from analytics.shape_graph import compute_shape_graph, infer_positions` | `from silly_kicks.tracking import compute_shape_graph, infer_positions` | Drop-in |
| `from analytics.line_breaking import detect_line_breaking` | `from silly_kicks.tracking import detect_line_breaking` | Drop-in |
| `from analytics.obso import compute_obso_surface` | `from silly_kicks.tracking import compute_obso_surface` | **Adapter needed** — returns `ObsoSurface` dataclass instead of raw `np.ndarray`; callers access `.values` for the grid |
| `from analytics.space_creation import compute_frame_space_creation` | `from silly_kicks.tracking import compute_space_created` | **Refactor needed** — silly-kicks computes OBSO internally; lakehouse callers currently pass pre-built ghost trajectories and explicit grids |
| `from analytics.pausa import compute_pausa_scores` | `from silly_kicks.tracking import compute_pausa, add_pausa` | **Adapter needed** — function renamed; signature uses OBSO column names not positional args |
| `from analytics.elastic_sync import align_events_to_frames` | `from silly_kicks.tracking import align_events_to_frames` | **Adapter needed** — column naming convention changes (frame→frame_id, timestamp→time_seconds, ball_x→x with is_ball filter) |
| `from analytics.smoothing import smooth_positions` | `from silly_kicks.tracking.preprocess import smooth_frames` | **Adapter needed** — different function name + PreprocessConfig-based parametrization |
| `from analytics.coordinates import *` | **DROP** — lakehouse keeps its own SB conversion; silly-kicks uses [0,105]x[0,68] | N/A |

**Migration complexity summary**: 3 drop-in, 4 adapter-needed (thin wrapper or call-site changes), 1 refactor-needed (space_creation callers must pass raw frames instead of pre-built ghost trajectories), 1 drop.

### 5.1 VAEP Factories — Phase 2+ Adoption Opportunity

Every new TF ships with a `<name>_xfns` VAEP factory (per silly-kicks convention). These are **new capabilities** the lakehouse does not currently have — the lakehouse analytics modules are pure primitives without VAEP integration. Phase 2 can optionally adopt them:

| silly-kicks VAEP Factory | What it provides |
|---|---|
| `shape_graph_xfns` | Per-action team topology features (graph density, mean stability) for VAEP |
| `obso_xfns` | Per-action OBSO triplet (actual, peak, optimal) as VAEP features |
| `space_creation_xfns` | Per-action space created by actor/teammates as VAEP features |
| `pausa_xfns` | Per-action PAUSA decomposition (temporal, spatial, composite) as VAEP features |
| `elastic_sync_xfns` | Per-action sync confidence as VAEP feature |

These are additive — the lakehouse can adopt any subset without affecting existing VAEP pipelines. Each factory follows the established `_frame_aware` xfn marker dispatch pattern (ADR-005).

---

## 6. Findings from Lakehouse Source Code Review

Verified against `D:\Development\karstenskyt__luxury-lakehouse\src\analytics\`.

### 6.1 OBSO: transition_grid is NOT xT

The OBSO formula is `PPCF x Transition x EPV` where `transition_grid` and `epv_grid` are **independent, pre-computed grids** — NOT derived from `ExpectedThreat`. The lakehouse provides a synthetic Gaussian-decay fallback (`_make_synthetic_reachability_grid()`) when no external model is supplied.

**silly-kicks implication**: We need a default transition model. Options:
- (a) Ship the same Gaussian-decay synthetic fallback (simplest, matches lakehouse behavior)
- (b) Derive from `ExpectedThreat` transition matrix (more principled but different semantics)
- (c) Accept external grid, require caller to provide

**Recommendation**: (a) for v1, with API accepting optional override grids.

### 6.2 OBSO: Hardcoded StatsBomb Units

The lakehouse OBSO uses `sigma_x=30.0, sigma_y=20.0` for the Gaussian distance weight. These are in StatsBomb 120x80 units. In silly-kicks [0,105]x[0,68] meters, the equivalents are `sigma_x = 30 * (105/120) = 26.25` and `sigma_y = 20 * (68/80) = 17.0`. Post-TF-38 center-origin, the same physical distances apply but grid construction changes.

**silly-kicks implication**: Parameterize sigma values in physical meters, not coordinate-system-specific units.

### 6.3 Space Creation: Algebraic Loop-Invariant Hoisting

The lakehouse uses algebraic loop-invariant hoisting to batch the N+1 pitch control evaluations (baseline + N player removals). It pre-computes `transition * distance_weight * epv` once (this product is player-independent), then only varies the PPCF per player removal. The module is pure numpy/pandas — no JAX.

**silly-kicks implication**: Port the same algebraic optimization. Pre-compute the invariant `obso_multiplier = transition * distance_weight * epv`, then run N serial `compute_pitch_control` calls (one per player removal) and multiply by the cached multiplier. N=10 outfield players × 104×68 grid is <100ms total — acceptable for action-coupled use.

### 6.4 Shape Graph: stability_threshold is 45.0 Degrees

The lakehouse spec said `0.4` but the actual code uses `_STABILITY_THRESHOLD = 45.0` (degrees). No empirical justification in code or comments — cited from Sotudeh 2026 thesis without discussion of provider sensitivity.

**silly-kicks implication**: Ship with 45.0 default; note in docstring that threshold may need provider-specific tuning.

### 6.5 coordinates.py is Not Used by Analytics Modules

Grep confirms zero imports of `coordinates.py` from `src/analytics/`. It IS imported from `src/ingestion/line_breaking_tracking.py` (3 imports for provider coordinate conversion), but that's a lakehouse-internal ingestion concern. No analytics module depends on it. **Confirms decision to skip** — the module is a lakehouse ingestion utility, not an analytics primitive.

### 6.6 OBSO Grid Dimensions Are Flexible

`compute_obso_surface` interpolates transition and EPV grids to match the PPCF grid dimensions. No hardcoded grid sizes. Good — silly-kicks can use its standard 104x68 default or any caller-specified size.

## 7. Resolved Questions (from Lakehouse Review)

1. **ELASTIC sync integration pattern**: **Separate refinement pass.** In the lakehouse pipeline, `elastic_sync` runs as a standalone post-processing step producing `(action_id, frame_id, confidence)`. It does NOT replace the initial nearest-time linkage — it refines it. For silly-kicks: model as a standalone function, don't integrate into `link_actions_to_frames`. Users apply it as an optional refinement after basic linkage.

2. **PAUSA computation scope**: **Per-pass only.** The lakehouse computes the OBSO triplet only for pass actions, not every frame. Pipeline: aligned frame → pitch control → OBSO surface → extract actual/peak/optimal → PAUSA decomposition. Confirms silly-kicks' action-coupled pattern.

3. **Space creation loop-invariant correctness**: **Confirmed correct and tested.** The distance weight `exp(-d²/(2σ²))` depends on `(ball_position, cell_position)`, neither of which changes per player removal. Only PPCF varies. The lakehouse has a benchmark test verifying algebraic equivalence.

4. **Shape graph special cases**: **All from Sotudeh 2026 thesis (Chapter 4)**, not lakehouse heuristics. Tree-like structures, asymmetric middle groups, diamond patterns, and the equal-frequency binning fallback for degenerate cases (fewer than 5 players) are all documented in the thesis. Academic attribution goes to Sotudeh 2026.

---

## 8. Testing Strategy

### 8a. Golden-Fixture Numerical Correctness

For each TF, create a golden test: run the lakehouse version on a test fixture, serialize the exact output, and ship as a test asset in silly-kicks. The silly-kicks test asserts numerical equality within floating-point tolerance (`np.testing.assert_allclose(atol=1e-10)`).

**Mechanism**: The lakehouse session provides `scripts/generate_golden_fixtures.py` that runs each analytics module on synthetic data and serializes output to `.npz`. The silly-kicks session copies the output files into `tests/tracking/golden/<module>_input.npz` + `<module>_expected.npz`. Each TF's test suite loads the golden fixture and asserts exact numerical parity.

**Responsibility**: Lakehouse generates (it owns the working implementations and test data). silly-kicks consumes.

### 8b. Benchmark Baselines

Port lakehouse `pytest-benchmark` tests as documented baselines (not CI hard-gates):
- team_shape: ≤1ms per frame for 10 outfield players
- team_shape_frame (both teams): ≤2ms per frame for 22 players
- line_breaking: ≤2ms per pass

New modules get similar baselines established during initial implementation.

### 8c. OBSO → PAUSA → Space Creation Chain Integration Test

An E2E test running the full pipeline on a synthetic match:
```
pitch_control → OBSO surface → space_creation (leave-one-out)
                      ↓
              pass_obso triplet → PAUSA decomposition
```
Asserts realistic output ranges, compatible grid shapes, and correct column propagation between modules. Catches integration bugs that per-module unit tests miss.

### 8d. Phase 2 Verification Gate

After all TFs ship, the lakehouse runs its existing test suite with imports switched from `src/analytics/` to `from silly_kicks.tracking import ...`. If numerical results are identical (within tolerance), Phase 2 is green. This is an explicit acceptance gate, not an assumption.

---

## 9. Acceptance Criteria

- All 5 new modules (TF-39 through TF-43) + 1 enhancement (TF-44) follow existing silly-kicks patterns: private `_<name>.py`, `add_<name>` aggregator, `<name>_xfns` factory, atomic mirror, tests, NOTICE
- Zero hardcoded pitch dimensions — all parameterized for TF-38 compatibility
- Golden-fixture tests pass for each module (§8a) — numerical parity with lakehouse within `atol=1e-10`
- OBSO → space_creation → PAUSA chain integration test passes (§8c)
- Benchmark baselines documented per module (§8b)
- No new required dependencies (scipy already core)
- Each TF ships as an independent PR (can merge independently)
- **Phase 2 gate** (§8d): Lakehouse test suite passes with imports switched — verified by running lakehouse CI with `silly_kicks.tracking` imports replacing `src/analytics/` imports
