# Library Extraction: TF-39 through TF-44 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port 5 new analytics modules + 1 enhancement from the lakehouse (`src/analytics/`) into `silly_kicks/tracking/`, following existing patterns. Single PR.

**Architecture:** Each new module becomes a private `_<name>.py` in `tracking/`, with `add_<name>` aggregator + `<name>_xfns` VAEP factory in `features.py`, re-exported via `__init__.py`. TF-44 enhances the existing `_team_shape.py`. Existing lakehouse code is adapted (not rewritten) to silly-kicks column naming, coordinate conventions, and frozen-dataclass patterns. No new dependencies beyond scipy (already core).

**Tech Stack:** numpy, scipy (Delaunay, hierarchy), pandas. Internal deps: `tracking.pitch_control`, `xthreat`.

**Source code:** `D:\Development\karstenskyt__luxury-lakehouse\src\analytics\` — all modules are pure-compute (pandas/numpy in, pandas/numpy out). No I/O, no Spark.

---

## File Structure

### New files

| File | Responsibility | Lakehouse source | Approx lines |
|------|---------------|-----------------|-------------|
| `silly_kicks/tracking/_shape_graph.py` | Sotudeh 2026 Delaunay stable subgraph + 5x5 position inference | `shape_graph_construction.py` (396L) + `shape_graph_inference.py` (586L) | ~700 |
| `silly_kicks/tracking/_obso.py` | Spearman 2018 OBSO surface + pass-level triplet | `obso.py` (283L) | ~300 |
| `silly_kicks/tracking/_space_creation.py` | Fernandez 2018 differential OBSO (leave-one-out) | `space_creation.py` (175L) | ~200 |
| `silly_kicks/tracking/_pausa.py` | Lee 2026 pass timing decomposition | `pausa.py` (75L) | ~80 |
| `silly_kicks/tracking/_elastic_sync.py` | Kim 2025 ELASTIC event-tracking alignment | `elastic_sync.py` (299L) | ~300 |
| `tests/tracking/test_shape_graph.py` | Unit + integration tests for shape graph | | ~400 |
| `tests/tracking/test_obso.py` | Unit + integration tests for OBSO | | ~300 |
| `tests/tracking/test_space_creation.py` | Unit + integration tests for space creation | | ~200 |
| `tests/tracking/test_pausa.py` | Unit tests for PAUSA | | ~100 |
| `tests/tracking/test_elastic_sync.py` | Unit + integration tests for ELASTIC | | ~250 |
| `tests/tracking/test_obso_chain_integration.py` | OBSO → Space Creation → PAUSA chain test | | ~60 |

### Modified files

| File | Changes |
|------|---------|
| `silly_kicks/tracking/_team_shape.py` | TF-44: add 3 Ward inter-line gap metrics (`defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2`) |
| `silly_kicks/tracking/__init__.py` | Re-export all 5 new modules' public API |
| `silly_kicks/tracking/features.py` | TF-44 metric list expansion (7→10 metrics, 14→20 agg cols, 36→54 VAEP cols) + 5 new `add_*` aggregators + 5 `*_xfns` VAEP factories |
| `silly_kicks/atomic/tracking/features.py` | Atomic mirrors for all 5 new xfns |
| `tests/tracking/test_team_shape.py` | TF-44: new Ward clustering tests + column count updates |
| `NOTICE` | 5 new citation blocks + TF-44 technique note (no academic citation — standard algorithm) |
| `CHANGELOG.md` | Feature entries |
| `pyproject.toml` + `silly_kicks/__init__.py` | Version bump |

---

## Coordinate System Adaptation (applies to ALL tasks)

The lakehouse uses StatsBomb 120x80 coordinates. silly-kicks uses [0,105]x[0,68] meters. All ported code must:

1. **Replace hardcoded `120.0`/`80.0`** with `pitch_length`/`pitch_width` parameters (default `105.0`/`68.0`).
2. **Scale sigma values**: lakehouse `sigma_x=30.0` in SB units = `26.25` in meters (`30 * 105/120`). lakehouse `sigma_y=20.0` = `17.0` (`20 * 68/80`).
3. **Grid construction**: `np.linspace(0, pitch_length, grid_nx)` not hardcoded `(0, 120)`.
4. **No origin assumptions**: parameterize for future TF-38 center-origin switch.

---

## Task 0: TF-44 Team Shape Ward Inter-Line Gaps (Karakuş & Arkadaş 2025)

**Modify:** `silly_kicks/tracking/_team_shape.py`, `silly_kicks/tracking/features.py`, `tests/tracking/test_team_shape.py`

### 0.1 Adaptation notes

Extend the existing `compute_team_shape` with 3 new metrics derived from Ward hierarchical clustering of outfield player x-coordinates: `defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2`. The lakehouse analytics `team_shape.py` already ships this — we're back-porting the enhancement.

**Ward clustering is already in the dep tree:** `_line_breaking.py` imports `from scipy.cluster.hierarchy import fcluster, linkage` with `n_clusters=3` (Ward method). Same pattern here.

**Column counts after ship:**
- `_RESULT_COLS`: 11 → 14 (add `defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2`)
- `compute_team_shape`: 7 → 10 metrics (3 new per frame-row)
- `add_team_shape`: 14 → 20 columns (10 metrics × 2 teams)
- `team_shape_xfns`: 36 → 54 columns (9 vaep_metrics × 2 suffixes × 3 states)

Note: `_RESULT_COLS` has 11 entries (4 index + 7 metrics). Adding 3 makes 14 entries. The "7→10" counts refer to stats-only columns (excluding the 4 index cols: game_id, period_id, frame_id, team_id).

### 0.2 Steps

- [ ] **Step 1: Add Ward clustering to `_team_shape.py`**

Add import at top:
```python
from scipy.cluster.hierarchy import fcluster, linkage
```

Extend `_RESULT_COLS` with 3 new entries:
```python
_RESULT_COLS = [
    "game_id",
    "period_id",
    "frame_id",
    "team_id",
    "n_outfield_players",
    "centroid_x",
    "centroid_y",
    "convex_hull_area",
    "team_length",
    "team_width",
    "stretch_index",
    "defensive_line_height",
    "inter_line_gap_1",
    "inter_line_gap_2",
]
```

Add `n_defensive_lines: int = 3` parameter to `compute_team_shape` signature:
```python
def compute_team_shape(
    frames: pd.DataFrame,
    team_id: int | str,
    *,
    n_defensive_lines: int = 3,
) -> pd.DataFrame:
```

After the existing stretch index computation and before `rows.append(...)`, add the Ward clustering logic:

```python
        # Ward hierarchical clustering for inter-line gaps
        n_eff = min(n_defensive_lines, n)
        if n < 2:
            # 1 player: defensive line at that player's x, no gaps
            def_line_height = float(xs.min())
            gap_1 = np.nan
            gap_2 = np.nan
        elif n_eff < 2:
            def_line_height = float(xs.min())
            gap_1 = np.nan
            gap_2 = np.nan
        else:
            z = linkage(xs.reshape(-1, 1), method="ward")
            labels = fcluster(z, t=n_eff, criterion="maxclust")
            centroids = np.sort(
                [float(np.mean(xs[labels == c])) for c in range(1, n_eff + 1)]
            )
            def_line_height = float(centroids[0])
            if n_eff >= 2:
                gap_1 = float(centroids[1] - centroids[0])
            else:
                gap_1 = np.nan
            if n_eff >= 3:
                gap_2 = float(centroids[2] - centroids[1])
            else:
                gap_2 = np.nan
```

Extend the `rows.append(...)` dict with:
```python
                "defensive_line_height": def_line_height,
                "inter_line_gap_1": gap_1,
                "inter_line_gap_2": gap_2,
```

- [ ] **Step 2: Update `features.py` metric lists**

Update the `metrics` list in `add_team_shape` (line ~1426):
```python
    metrics = [
        "n_outfield_players",
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
        "defensive_line_height",
        "inter_line_gap_1",
        "inter_line_gap_2",
    ]
```

Update the `<2 teams` NaN-fill branch (line ~1401) to include the same 10 metrics.

Update the docstring from "14 team-shape columns (7 metrics x 2 teams)" to "20 team-shape columns (10 metrics x 2 teams)".

Update `vaep_metrics` in `team_shape_xfns` (line ~1512):
```python
    vaep_metrics = [
        "centroid_x",
        "centroid_y",
        "convex_hull_area",
        "team_length",
        "team_width",
        "stretch_index",
        "defensive_line_height",
        "inter_line_gap_1",
        "inter_line_gap_2",
    ]
```

Update the docstring from "12 features x 3 game-states = 36 columns" to "18 features x 3 game-states = 54 columns".

Update `vaep_metrics` in `_team_shape_at_actions` identically (line ~1563).

- [ ] **Step 3: Write tests**

Add to `tests/tracking/test_team_shape.py`:

```python
class TestWardInterLineGaps:
    """TF-44: Ward clustering defensive_line_height + inter-line gaps."""

    def test_known_3_cluster_geometry(self):
        """Three clear groups at x=15, x=40, x=65 → known centroids + gaps."""
        positions = [
            (14.0, 20.0), (15.0, 30.0), (16.0, 40.0),    # cluster 1 ~ x=15
            (39.0, 15.0), (40.0, 35.0), (41.0, 50.0),    # cluster 2 ~ x=40
            (64.0, 25.0), (65.0, 34.0), (66.0, 45.0),    # cluster 3 ~ x=65
        ]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["defensive_line_height"] == pytest.approx(15.0, abs=1.0)
        assert row["inter_line_gap_1"] == pytest.approx(25.0, abs=2.0)
        assert row["inter_line_gap_2"] == pytest.approx(25.0, abs=2.0)

    def test_fewer_than_3_players_gaps_nan(self):
        """2 players: 1 gap only (gap_2 = NaN)."""
        positions = [(20.0, 30.0), (60.0, 40.0)]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert not np.isnan(row["inter_line_gap_1"])
        assert np.isnan(row["inter_line_gap_2"])
        assert row["defensive_line_height"] == pytest.approx(20.0, abs=1.0)

    def test_single_player(self):
        """1 player: line height = player x, both gaps NaN."""
        positions = [(30.0, 34.0)]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert row["defensive_line_height"] == pytest.approx(30.0)
        assert np.isnan(row["inter_line_gap_1"])
        assert np.isnan(row["inter_line_gap_2"])

    def test_tight_cluster_zero_gap(self):
        """All players at same x → gaps ~0."""
        positions = [(50.0, y) for y in range(20, 48, 3)]
        frames = _make_team_frames(outfield_positions=positions)
        result = compute_team_shape(frames, team_id=1)
        row = result.iloc[0]
        assert row["inter_line_gap_1"] == pytest.approx(0.0, abs=0.5)
        assert row["inter_line_gap_2"] == pytest.approx(0.0, abs=0.5)
```

Update existing `TestAddTeamShape` column count assertions: 14 → 20.
Update existing `TestTeamShapeXfns` column count assertions: 36 → 54.

- [ ] **Step 4: NOTICE amendment**

No new academic citation — Ward inter-line gap clustering is a standard algorithmic technique, not a published methodology. Append a note to the existing team_shape block in NOTICE:
```
  (TF-44 enhancement: Ward hierarchical clustering for defensive_line_height
  and inter-line gap metrics; n_defensive_lines=3 default)
```

---

## Task 1: TF-39 Shape Graph (Sotudeh 2026)

**Port from:** `shape_graph_construction.py` (396L) + `shape_graph_inference.py` (586L)
**Create:** `silly_kicks/tracking/_shape_graph.py`

### 1.1 Adaptation notes

The lakehouse splits construction and inference across 2 files + a 29-line re-export module. Consolidate into a single `_shape_graph.py` — the two modules share the `ShapeGraph` and `PositionLabel` dataclasses and the combined file stays under 800 lines.

**Column name changes:** None — shape graph operates on `np.ndarray` positions, not DataFrame columns.

**Pitch dimension changes:**
- `_decompose_middle` has `pitch_length=105.0, pitch_width=68.0` defaults — already correct for silly-kicks. Verify and keep.
- `infer_positions` uses `attacking_direction` (radians/sign) — coordinate-agnostic by construction.

**Constants to preserve exactly:**
- `_STABILITY_THRESHOLD = 45.0` (degrees)
- `_STEEP_ANGLE_LOW = 67.5`, `_STEEP_ANGLE_HIGH = 112.5`
- `_DEGENERATE_FRACTION = 0.6`
- `_VERTICAL_LEVELS = ("B", "DM", "M", "AM", "F")`
- `_HORIZONTAL_LEVELS = ("L", "LC", "C", "RC", "R")`
- `POSITION_LABEL_MATRIX` — full 5x5 dict

### 1.2 Steps

- [ ] **Step 1: Port `_shape_graph.py`**

Copy `shape_graph_construction.py` + `shape_graph_inference.py` into a single `silly_kicks/tracking/_shape_graph.py`. Preserve all algorithms verbatim. Adapt:
- Module docstring: add TF-39 reference, NOTICE cross-link
- `from __future__ import annotations` at top
- Remove lakehouse-specific imports (`from analytics.shape_graph_construction import ...` → inline)
- `PositionLabel` and `ShapeGraph` frozen dataclasses at top of file
- `compute_shape_graph()` and `infer_positions()` as public API
- All `_private` helpers kept private
- Add Examples docstring sections to both public functions

- [ ] **Step 2: Wire up aggregator + xfns in `features.py`**

Add `add_shape_graph` aggregator producing per-action columns:
- `shape_graph_density` (n_edges / max_possible_edges)
- `shape_graph_n_edges` (int)
- `shape_graph_mean_stability` (mean edge stability in degrees)

Pattern: link action to frame → extract team positions from frame → `compute_shape_graph(positions)` → derive 3 scalar metrics. One set per team (attacking/defending) = 6 columns total.

Add `shape_graph_xfns(home_team_id)` factory: 6 features x 3 states = 18 VAEP columns.

- [ ] **Step 3: Write tests**

Port from lakehouse `test_shape_graph.py` (536L). Adapt:
- `TestComputeShapeGraph`: 4-4-2 connectivity, 3-player triangle, <3 empty, collinear empty, threshold enforcement, stabilities match edges, points preserved, custom threshold, zero threshold = full Delaunay
- `TestInferPositions`: 4-4-2 vertical/horizontal, reversed attacking direction, position label matrix compliance, empty graph, 3-5-2 distribution
- `TestAngularStability`: equilateral diamond 60deg, degenerate low, boundary 120deg, cocircular 0deg
- `TestAddShapeGraph`: aggregator produces 6 columns, NaN-safe
- `TestShapeGraphXfns`: column count = 18, introspection NaN

Fixtures: `positions_442` (defenders x=20, midfield x=40, forwards x=60) and `positions_352` from lakehouse tests — adapt to silly-kicks coordinate frame [0,105]x[0,68].

- [ ] **Step 4: Update `__init__.py` re-exports**

Add to `__all__` and import block:
```python
from ._shape_graph import PositionLabel, ShapeGraph, compute_shape_graph, infer_positions
```

Add to features import block:
```python
from .features import add_shape_graph, shape_graph_xfns
```

- [ ] **Step 5: Atomic mirror**

Add `atomic_shape_graph_xfns` in `silly_kicks/atomic/tracking/features.py`. Shape graph xfns don't depend on action start/end coordinates (they use frame positions), so the atomic mirror re-exports the standard xfns unchanged.

- [ ] **Step 6: NOTICE entry**

```
The shape graph features in silly_kicks/tracking/_shape_graph.py
(TF-39) implement methodologies described in:

- Sotudeh, A. (2026). "Tactical Position Inference via Stable Delaunay
  Subgraphs." Master's Thesis, Chapter 3 (Algorithm 1: iterative
  edge-removal) and Chapter 4 (5x5 face-center position decomposition).
  (stability threshold, tie-breaking, degenerate-case fallbacks)
```

---

## Task 2: TF-40 OBSO (Spearman 2018)

**Port from:** `obso.py` (283L)
**Create:** `silly_kicks/tracking/_obso.py`

### 2.1 Adaptation notes

**Pitch dimension changes (critical):**
- `sigma_x=30.0` (SB) → `sigma_x=26.25` (meters, default in `ObsoParams`)
- `sigma_y=20.0` (SB) → `sigma_y=17.0` (meters, default in `ObsoParams`)
- Grid axes: `np.linspace(0, params.pitch_length, params.grid_nx)` not hardcoded 120
- Synthetic reachability/EPV grids: dimensionless (0-1 normalized), no coord change needed

**Column name changes:** OBSO operates on numpy arrays + `PitchControlSurface`, not DataFrame columns. No column renaming needed in the primitive.

**Key dependency:** `compute_pitch_control` from `tracking.pitch_control` — already shipped. OBSO calls it per-frame to get the PPCF grid, then multiplies by transition × EPV.

**Grid shape convention:** All grids are `(ny, nx)` — matches `PitchControlSurface.surface` shape.

**`compute_pass_obso` windowing:** The lakehouse's windowing logic lives in its ingestion layer. The silly-kicks version accepts pre-windowed frames (list of DataFrames, one per timestep around the pass) — callers use `slice_around_event` to produce these. The aggregator `add_obso` handles the windowing internally.

### 2.2 Steps

- [ ] **Step 1: Port `_obso.py`**

Create `silly_kicks/tracking/_obso.py`. Port from `obso.py` with adaptations:

```python
@dataclass(frozen=True)
class ObsoParams:
    grid_nx: int = 104
    grid_ny: int = 68
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    sigma_x: float = 26.25   # meters (lakehouse: 30.0 in SB 120x80)
    sigma_y: float = 17.0    # meters (lakehouse: 20.0 in SB 120x80)

@dataclass(frozen=True)
class ObsoSurface:
    values: np.ndarray    # (grid_ny, grid_nx)
    grid_x: np.ndarray    # (grid_nx,)
    grid_y: np.ndarray    # (grid_ny,)
```

Public functions:
- `compute_obso_surface(pitch_control, ball_position, *, transition_grid=None, epv_grid=None, params=None) -> ObsoSurface`
- `compute_pass_obso(pass_window_frames, target_position, attacking_team_id, *, transition_grid=None, epv_grid=None, params=None, pitch_control_method="spearman") -> dict[str, float]` — returns `{actual_obso, peak_obso, optimal_obso}`

Private helpers (ported verbatim, adapted coords):
- `_make_synthetic_reachability_grid(ny, nx)` — Gaussian decay (dimensionless, no coord change)
- `_make_synthetic_epv_grid(ny, nx)` — linear ramp (dimensionless)
- `_get_default_grids(reachability, epv)` — fallback dispatcher
- `_interpolate_grid(grid, target_shape)` — bilinear (numpy-only, ported verbatim)

Key change in `compute_obso_surface`: replace hardcoded `sigma_x=30, sigma_y=20` with `params.sigma_x, params.sigma_y`. Grid construction uses `np.linspace(0, params.pitch_length, params.grid_nx)`.

Key change in `compute_pass_obso`: accept `PitchControlSurface` from `compute_pitch_control` (silly-kicks API) instead of raw PPCF arrays. Extract `.surface`, `.grid_x`, `.grid_y`.

- [ ] **Step 2: Wire up aggregator + xfns in `features.py`**

Add `add_obso` aggregator:
- Uses `slice_around_event` with `pre_seconds=3.0, post_seconds=1.0` to window frames around each pass
- Calls `compute_pitch_control` per timestep in the window
- Calls `compute_pass_obso` to get the triplet
- Output columns: `obso_actual`, `obso_peak`, `obso_optimal` (pass actions only; NaN for non-passes)
- Signature: `add_obso(actions, frames, *, links=None, home_team_id, transition_grid=None, epv_grid=None, params=None, pitch_control_method="spearman")`

Add `obso_xfns(home_team_id, ...)` factory: 3 features × 2 teams (attacking/defending) × 3 states = 18 VAEP columns. Actually — OBSO is computed for the attacking team only (scoring opportunity for the team in possession). So: 3 features × 3 states = 9 VAEP columns.

- [ ] **Step 3: Write tests**

Port from lakehouse `test_obso.py` (337L). Adapt:
- `TestInterpolateGrid`: identity, corners, mass preservation, downsample
- `TestComputeObsoSurface`: shape match, value bounds [0,1], zero conditions, near-ball gradient
- `TestComputePassObso`: schema, peak >= actual, optimal >= actual, all bounded [0,1]
- `TestGetDefaultGrids`: passthrough, synthetic generation, mixed
- `TestAddObso`: enrichment column count, NaN-safe, pass-only activation
- `TestObsoXfns`: column count = 9, introspection NaN

Fixtures: 2-team synthetic frame with known positions, synthetic transition/EPV grids.

- [ ] **Step 4: Update `__init__.py`**

```python
from ._obso import ObsoParams, ObsoSurface, compute_obso_surface, compute_pass_obso
from .features import add_obso, obso_xfns
```

- [ ] **Step 5: Atomic mirror**

`atomic_obso_xfns` — re-export standard (OBSO uses frame positions, not action start/end).

- [ ] **Step 6: NOTICE entry**

```
The OBSO features in silly_kicks/tracking/_obso.py (TF-40) implement:

- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports
  Analytics Conference.
  (off-ball scoring opportunity surface: PPCF x Transition x EPV;
  pass-level actual/peak/optimal OBSO triplet)
```

---

## Task 3: TF-41 Space Creation (Fernandez 2018)

**Port from:** `space_creation.py` (175L)
**Create:** `silly_kicks/tracking/_space_creation.py`

### 3.1 Adaptation notes

**Key dependency:** `compute_pitch_control` (silly-kicks) + `_obso.compute_obso_surface` (TF-40).

**Missing function:** The lakehouse has `compute_pitch_control_player_removal` — silly-kicks does NOT. The silly-kicks implementation will call `compute_pitch_control` N+1 times (baseline + N player removals). For each removal, filter the frame DataFrame to exclude one player, then call `compute_pitch_control`. This is functionally equivalent.

**Loop-invariant hoisting:** The lakehouse pre-computes `obso_multiplier = transition * distance_weight * epv` once, then only varies the PPCF per player removal. Port this optimization — it's the key performance gain (avoids N redundant grid multiplications).

**Pitch dimension changes:**
- `pitch_length=120.0` → `105.0`, `pitch_width=80.0` → `68.0` via `SpaceCreationParams`
- Grid: `np.linspace(0, params.pitch_length, params.grid_nx)` not `(0, 120)`

**Column name changes:**
- Input: lakehouse uses `player_id, team, x, y, velocity_x, velocity_y` — silly-kicks uses TRACKING_FRAMES_COLUMNS (`player_id, team_id, x, y, vx, vy`)
- Output: keep `player_id, space_created_m2, space_destroyed_m2, net_space_m2`

### 3.2 Steps

- [ ] **Step 1: Port `_space_creation.py`**

Create `silly_kicks/tracking/_space_creation.py`:

```python
@dataclass(frozen=True)
class SpaceCreationParams:
    grid_nx: int = 104
    grid_ny: int = 68
    pitch_length: float = 105.0
    pitch_width: float = 68.0

def compute_space_created(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    *,
    ball_position: tuple[float, float] | None = None,
    transition_grid: np.ndarray | None = None,
    epv_grid: np.ndarray | None = None,
    params: SpaceCreationParams | None = None,
    pitch_control_method: str = "spearman",
) -> pd.DataFrame:
    """Per-player space creation via leave-one-out differential OBSO.

    Returns DataFrame with player_id, space_created_m2, space_destroyed_m2, net_space_m2.
    """
```

Algorithm (preserving lakehouse's loop-invariant hoisting):
1. Build grid axes from params
2. Compute baseline `PitchControlSurface` with all players
3. Compute baseline `ObsoSurface` — extract `.values` as baseline_obso
4. **Hoist invariant:** `obso_multiplier = effective_transition * epv_interp` (computed once)
5. For each attacking-team outfield player:
   - Filter frame to exclude that player
   - Compute `PitchControlSurface` without that player
   - `removed_obso = removed_pc.surface * obso_multiplier` (cheap multiply, no re-interpolation)
   - `delta = baseline_obso - removed_obso`
   - `space_created = sum(max(delta, 0)) * cell_area`
   - `space_destroyed = sum(min(delta, 0)) * cell_area`
6. Return per-player DataFrame

- [ ] **Step 2: Wire up aggregator + xfns**

Add `add_space_creation(actions, frames, *, links=None, home_team_id, ...)` aggregator:
- Output columns: `space_created_m2`, `space_destroyed_m2`, `net_space_m2` (for the acting player)
- Per-action: link to frame → call `compute_space_created` on that frame → look up actor's row

Add `space_creation_xfns(home_team_id, ...)`: 3 features × 3 states = 9 VAEP columns.

- [ ] **Step 3: Write tests**

- `TestComputeSpaceCreated`: known geometry (player creates obvious space → positive value), removal of irrelevant player → near-zero, empty frame
- `TestLoopInvariantCorrectness`: verify hoisted computation matches naive N-loop (both give same result within atol=1e-10)
- `TestAddSpaceCreation`: column count, NaN-safe
- `TestSpaceCreationXfns`: column count = 9, introspection NaN

- [ ] **Step 4: Update `__init__.py`**

```python
from ._space_creation import SpaceCreationParams, compute_space_created
from .features import add_space_creation, space_creation_xfns
```

- [ ] **Step 5: Atomic mirror**

Re-export standard xfns (frame-based, not action-shape-dependent).

- [ ] **Step 6: NOTICE entry**

```
The space creation features in silly_kicks/tracking/_space_creation.py
(TF-41) implement:

- Fernandez, J., & Bornn, L. (2018). "Wide Open Spaces: A Statistical
  Technique for Measuring Space Creation in Professional Soccer."
  MIT Sloan Sports Analytics Conference.
  (leave-one-out differential OBSO for per-player space creation;
  loop-invariant hoisting optimization for N+1 pitch control evaluations)
```

---

## Task 4: TF-42 PAUSA (Lee et al. 2026)

**Port from:** `pausa.py` (75L)
**Create:** `silly_kicks/tracking/_pausa.py`

### 4.1 Adaptation notes

This is the simplest port — 75 lines of pure scalar arithmetic on OBSO triplet values. No spatial computation, no pitch dimensions, no coordinate sensitivity.

**Column name changes:**
- Lakehouse input: `actual_obso, peak_obso, optimal_obso` → keep same names (produced by `add_obso`)
- Lakehouse output: `temporal_judgment, spatial_selection, pausa_score` → rename to `pausa_temporal, pausa_spatial, pausa_composite` for consistency with silly-kicks `<module>_<metric>` convention

### 4.2 Steps

- [ ] **Step 1: Port `_pausa.py`**

Create `silly_kicks/tracking/_pausa.py`:

```python
def compute_pausa(
    actual_obso: float,
    peak_obso: float,
    optimal_obso: float,
) -> dict[str, float]:
    """PAUSA decomposition: temporal judgment x spatial selection.

    Returns dict with pausa_temporal, pausa_spatial, pausa_composite.
    """
    temporal = actual_obso / peak_obso if peak_obso > 0 else 0.0
    spatial = actual_obso / optimal_obso if optimal_obso > 0 else 0.0
    temporal = float(np.clip(temporal, 0.0, 1.0))
    spatial = float(np.clip(spatial, 0.0, 1.0))
    return {
        "pausa_temporal": temporal,
        "pausa_spatial": spatial,
        "pausa_composite": temporal * spatial,
    }
```

Also port vectorized DataFrame version (for `add_pausa`):

```python
def compute_pausa_batch(
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized PAUSA on DataFrame with obso_actual/peak/optimal columns."""
```

- [ ] **Step 2: Wire up aggregator + xfns**

Add `add_pausa(actions)` — note: **not frame-aware**. Operates on columns already produced by `add_obso`. Does NOT need frames or links.
- Input requires: `obso_actual`, `obso_peak`, `obso_optimal` columns
- Raises `ValueError` if columns missing (loud raise per convention)
- Output: `pausa_temporal`, `pausa_spatial`, `pausa_composite`

Add `pausa_xfns(home_team_id, ...)`: 3 features × 3 states = 9 VAEP columns. **Design decision:** `pausa_xfns` is frame-aware and self-contained — it internally computes the OBSO triplet (same as `obso_xfns` does), then applies PAUSA division. This follows the standard pattern where each xfn independently computes its own features from `(states, frames)`. If callers use both `obso_xfns` and `pausa_xfns`, the OBSO computation runs twice — acceptable (each xfn is self-contained). Callers who want to avoid duplication compose at the aggregator level (`add_obso` → `add_pausa`), not at the xfn level.

- [ ] **Step 3: Write tests**

- `TestComputePausa`: known values (0.5/1.0/0.8 → temporal=0.5, spatial=0.625, composite=0.3125), zero denominators, clip to [0,1]
- `TestComputePausaBatch`: DataFrame vectorized version matches scalar
- `TestAddPausa`: 3 output columns, missing obso columns raises ValueError
- `TestPausaXfns`: column count = 9, introspection NaN

- [ ] **Step 4: Update `__init__.py`**

```python
from ._pausa import compute_pausa, compute_pausa_batch
from .features import add_pausa, pausa_xfns
```

- [ ] **Step 5: Atomic mirror**

Re-export standard.

- [ ] **Step 6: NOTICE entry**

```
The PAUSA features in silly_kicks/tracking/_pausa.py (TF-42) implement:

- Lee, T., et al. (2026). "PAUSA: Quantifying Pass Timing in Football."
  (temporal judgment = actual/peak OBSO; spatial selection = actual/optimal
  OBSO; composite = temporal x spatial)
```

---

## Task 5: TF-43 ELASTIC Sync (Kim et al. 2025)

**Port from:** `elastic_sync.py` (299L)
**Create:** `silly_kicks/tracking/_elastic_sync.py`

### 5.1 Adaptation notes

**Column name changes (critical — this is the main adaptation work):**

| Lakehouse column | silly-kicks equivalent | Notes |
|---|---|---|
| `frame` | `frame_id` | Tracking schema |
| `period` | `period_id` | Tracking schema |
| `timestamp_seconds` | `time_seconds` | Event/action schema |
| `player_id` (events) | `player_id` | Same |
| `event_id` | `action_id` | SPADL naming |
| `event_type` | `type_id` | SPADL naming |
| `x, y` (player) | `x, y` | Same |
| `ball_x, ball_y` | Ball is a separate row with `is_ball=True` | **Structural change** — lakehouse has ball as columns, silly-kicks has ball as rows |

The ball position extraction is the biggest structural difference. Lakehouse tracking has `ball_x, ball_y` as columns on every player row. silly-kicks has ball as a separate row with `is_ball=True`. The port must extract ball position per (game_id, period_id, frame_id) from the ball rows and join to player rows for proximity computation.

**`_col_f64` dependency:** lakehouse uses `array_utils._col_f64(df, col)`. Replace with inline `np.asarray(df[col].values, dtype=np.float64)`.

**Integration model:** Standalone refinement pass (per spec §7.1). Does NOT replace `link_actions_to_frames` — it produces alternative/refined `(action_id, frame_id, confidence)` pointers that callers can substitute.

### 5.2 Steps

- [ ] **Step 1: Port `_elastic_sync.py`**

Create `silly_kicks/tracking/_elastic_sync.py`:

```python
@dataclass(frozen=True)
class ElasticSyncParams:
    window_seconds: float = 3.0
    frame_rate: float = 25.0
    accel_weight: float = 0.6
    proximity_weight: float = 0.4
    min_confidence: float = 0.3

def extract_ball_features(
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame:
    """Ball speed + acceleration per (game_id, period_id, frame_id).

    Returns DataFrame with game_id, period_id, frame_id, ball_x, ball_y,
    ball_speed, ball_accel.
    """

def align_events_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    params: ElasticSyncParams | None = None,
) -> pd.DataFrame:
    """ELASTIC alignment: action_id -> frame_id with confidence.

    Returns DataFrame with action_id, frame_id, alignment_confidence,
    alignment_error_seconds.
    """
```

Key adaptation in `extract_ball_features`:
- Filter `frames[frames["is_ball"]]` to get ball rows
- Deduplicate by `(game_id, period_id, frame_id)` — ball may have multiple rows (shouldn't, but defensive)
- Compute velocity via finite difference within (game_id, period_id) groups: `vx = dx/dt`, `vy = dy/dt`
- Speed = `sqrt(vx^2 + vy^2)`, accel = `|d(speed)/dt|`

Key adaptation in `align_events_to_frames`:
- Extract ball positions per frame from ball rows (pre-join)
- Build player-ball distance lookup: for each (game_id, period_id, frame_id, player_id), compute distance to ball position at that frame
- Use `actions["time_seconds"]` not `events["timestamp_seconds"]`
- Use `actions["action_id"]` not `events["event_id"]`
- Group by `(game_id, period_id)` not just `period`

- [ ] **Step 2: Wire up aggregator + xfns**

Add `add_elastic_sync(actions, frames, *, params=None)` aggregator:
- Calls `align_events_to_frames` and merges confidence column
- Output: `elastic_frame_id`, `elastic_confidence`, `elastic_error_seconds`
- Prefixed with `elastic_` to avoid collision with standard `frame_id` provenance column

Add `elastic_sync_xfns(...)`: 1 feature (confidence) × 3 states = 3 VAEP columns.

- [ ] **Step 3: Write tests**

- `TestExtractBallFeatures`: known velocity/acceleration from linear motion, stationary ball → zero, multi-period boundary
- `TestAlignEventsToFrames`: exact timestamp match → confidence ~1.0, no match in window → filtered out, multi-game batch
- `TestAddElasticSync`: 3 output columns, NaN-safe
- `TestElasticSyncXfns`: column count = 3, introspection NaN

Fixtures: synthesize 2-team tracking with known ball trajectory (linear motion for predictable acceleration), create matching actions with slightly offset timestamps.

- [ ] **Step 4: Update `__init__.py`**

```python
from ._elastic_sync import ElasticSyncParams, align_events_to_frames, extract_ball_features
from .features import add_elastic_sync, elastic_sync_xfns
```

- [ ] **Step 5: Atomic mirror**

Re-export standard.

- [ ] **Step 6: NOTICE entry**

```
The ELASTIC sync features in silly_kicks/tracking/_elastic_sync.py
(TF-43) implement:

- Kim, H., et al. (2025). "ELASTIC: Event-Level Alignment of Spatio-
  Temporal Information and Coordinates." arXiv:2508.09238, MLSA 2025.
  (ball acceleration + player-ball proximity feature matching for
  event-tracking frame alignment)
```

---

## Task 6: Integration wiring + final verification

- [ ] **Step 1: Update `__init__.py` with all 5 modules**

Ensure all public symbols from Tasks 1-5 are in `__all__` and imported. Alphabetical ordering per existing convention.

- [ ] **Step 2: Update all atomic mirrors in `atomic/tracking/features.py`**

Add `atomic_shape_graph_xfns`, `atomic_obso_xfns`, `atomic_space_creation_xfns`, `atomic_pausa_xfns`, `atomic_elastic_sync_xfns` — all re-exporting standard versions (none depend on action start/end column shape).

- [ ] **Step 3: OBSO → Space Creation → PAUSA chain integration test**

Add `tests/tracking/test_obso_chain_integration.py`:

```python
def test_obso_to_pausa_chain():
    """Full pipeline: pitch_control -> OBSO -> space_creation -> PAUSA."""
    # 1. Synthesize 2-team frame with known positions
    # 2. compute_pitch_control -> PitchControlSurface
    # 3. compute_obso_surface -> ObsoSurface (values in [0,1])
    # 4. compute_space_created -> per-player DataFrame (net_space_m2 sums to ~0)
    # 5. compute_pass_obso -> triplet (actual <= peak, actual <= optimal)
    # 6. compute_pausa(actual, peak, optimal) -> temporal, spatial, composite in [0,1]
    # Assert: shapes compatible, no NaN, values in expected ranges
```

- [ ] **Step 4: CHANGELOG entries**

```markdown
### Added
- `compute_team_shape` now emits `defensive_line_height`, `inter_line_gap_1`, `inter_line_gap_2` via Ward clustering (TF-44)
- `add_team_shape` 14→20 columns; `team_shape_xfns` 36→54 columns (TF-44)
- `compute_shape_graph` + `infer_positions`: Sotudeh 2026 Delaunay tactical position inference (TF-39)
- `compute_obso_surface` + `compute_pass_obso`: Spearman 2018 OBSO (TF-40)
- `compute_space_created`: Fernandez 2018 differential OBSO space creation (TF-41)
- `compute_pausa` + `compute_pausa_batch`: Lee 2026 PAUSA pass timing (TF-42)
- `extract_ball_features` + `align_events_to_frames`: Kim 2025 ELASTIC sync (TF-43)
- `add_shape_graph`, `add_obso`, `add_space_creation`, `add_pausa`, `add_elastic_sync` aggregators
- `shape_graph_xfns`, `obso_xfns`, `space_creation_xfns`, `pausa_xfns`, `elastic_sync_xfns` VAEP factories
```

- [ ] **Step 5: Linting + type checking**

```bash
ruff check silly_kicks/ && ruff format --check silly_kicks/ && pyright silly_kicks/
```

- [ ] **Step 6: Full test suite**

```bash
python -m pytest tests/ -m "not e2e" -v --tb=short -q
```

- [ ] **Step 7: Version bump**

Bump to next minor (features). Update `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`.

- [ ] **Step 8: Final review**

Run `/final-review`.

- [ ] **Step 9: Commit**

Single commit:
```
feat(tracking): library extraction TF-39..TF-44 -- shape graph, OBSO, space creation, PAUSA, ELASTIC sync, team shape Ward gaps
```

---

## Deferred Items (spec §8a, §8b)

**Golden fixtures (§8a):** The spec commits to `scripts/generate_golden_fixtures.py` with `atol=1e-10` parity assertions. Deferred — Phase 2 verification (lakehouse runs its full test suite with silly-kicks imports substituted) is a strictly stronger correctness gate than serialized `.npz` on synthetic fixtures. If Phase 2 passes, golden fixtures add no incremental confidence. Document in TODO.md for belt-and-suspenders follow-up if desired.

**Benchmark tests (§8b):** The spec commits to porting lakehouse `pytest-benchmark` baselines. Deferred to a follow-up PR — the port is the same algorithm at the same complexity, so regressions are not expected. Establishing baselines post-ship lets us benchmark the actual silly-kicks code paths (which may differ slightly from lakehouse due to `PitchControlSurface` dispatch, ball-row extraction, etc.).

---

## Appendix A: VAEP Factory Column Counts

| Module | Features | x Teams | x States | Total VAEP cols |
|--------|----------|---------|----------|----------------|
| TF-44 team_shape (enhancement) | +3 (defensive_line_height, inter_line_gap_1, inter_line_gap_2) | x2 (atk/def) | x3 | +18 (36→54) |
| TF-39 shape_graph | 3 (density, n_edges, mean_stability) | x2 (atk/def) | x3 | 18 |
| TF-40 OBSO | 3 (actual, peak, optimal) | x1 (atk only) | x3 | 9 |
| TF-41 space_creation | 3 (created, destroyed, net) | x1 (actor) | x3 | 9 |
| TF-42 PAUSA | 3 (temporal, spatial, composite) | x1 | x3 | 9 |
| TF-43 ELASTIC | 1 (confidence) | x1 | x3 | 3 |
| **Total new** | | | | **66** (18 enhancement + 48 new) |

## Appendix B: Aggregator Output Column Counts

| Module | Columns | Details |
|--------|---------|---------|
| TF-44 team_shape (enhancement) | +6 | 3 new metrics x 2 teams (14→20) |
| TF-39 shape_graph | 6 | 3 metrics x 2 teams |
| TF-40 OBSO | 3 | actual, peak, optimal |
| TF-41 space_creation | 3 | created_m2, destroyed_m2, net_m2 |
| TF-42 PAUSA | 3 | temporal, spatial, composite |
| TF-43 ELASTIC | 3 | frame_id, confidence, error_seconds |
| **Total new** | **24** | 6 enhancement + 18 new + provenance cols (idempotent) |

## Appendix C: Lakehouse Import Map After Ship

| Lakehouse Current | silly-kicks Target | Migration |
|---|---|---|
| `from analytics.team_shape import compute_team_shape` | `from silly_kicks.tracking import compute_team_shape` | Drop-in (now has Ward inter-line gaps) |
| `from analytics.shape_graph import compute_shape_graph, infer_positions` | `from silly_kicks.tracking import compute_shape_graph, infer_positions` | Drop-in |
| `from analytics.obso import compute_obso_surface` | `from silly_kicks.tracking import compute_obso_surface` | Adapter: returns `ObsoSurface` not raw array |
| `from analytics.space_creation import compute_frame_space_creation` | `from silly_kicks.tracking import compute_space_created` | Refactor: accepts frame not pre-built trajectories |
| `from analytics.pausa import compute_pausa_scores` | `from silly_kicks.tracking import compute_pausa, add_pausa` | Adapter: renamed, uses OBSO column names |
| `from analytics.elastic_sync import align_events_to_frames` | `from silly_kicks.tracking import align_events_to_frames` | Adapter: column naming (frame_id not frame, etc.) |
