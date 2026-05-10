# TF-15: GK Influence Primitives -- Design Spec

**Date:** 2026-05-09
**Status:** Draft (post lakehouse review v2)
**GKDV Layer:** 1 (foundation for TF-16..TF-19)
**Depends on:** TF-7 (pitch control, shipped 3.7.0), TF-13 (frame-based GK ID, shipped 3.4.0), TF-14 (defensive line, shipped 3.4.0)

## Problem

Raw pitch control `player_share()` over-credits the GK for controlling low-threat
space (own half, center circle). The Get Goalside critique: a GK who controls 20%
of their team's total pitch control sounds impressive, but most of that control is
over irrelevant territory. No published GK-evaluation framework today measures
positioning-as-deterrent (see GKDV research program, 2026-05-01 literature survey).

TF-15 ships three per-frame GK influence primitives that form the foundation
(Layer 1) for the GKDV metric (TF-19). Each primitive isolates a distinct aspect
of the GK's spatial contribution.

## Prerequisite API Additions

### PR-1: Export `compute_tti` from pitch control subpackage

`_compute_tti` in `pitch_control/_spearman.py:28-79` is currently private. Primitives
(b) and (c) need the Spearman TTI kinematic model independently of full pitch control
computation. TTI is a reusable kinematic primitive also needed by future GKDV consumers
(TF-16..TF-19) and TF-30 (cover shadow corridor TTI).

**Change:** Rename `_compute_tti` -> `compute_tti` in `_spearman.py`. Export from
`pitch_control/__init__.py` and add to `__all__`.

Public signature (unchanged from current private function):

```python
def compute_tti(
    pos: np.ndarray,       # (n_players, 2)
    vel: np.ndarray,       # (n_players, 2)
    targets: np.ndarray,   # (n_targets, 2)
    reaction_time: float,  # seconds before movement begins
    max_acceleration: float,  # m/s^2
) -> np.ndarray:           # (n_players, n_targets) TTI in seconds
```

Internal callers in `_spearman.py` updated to use the new name. Numba dispatch
path unchanged. No behavioral change.

### PR-2: Extract `select_back_line_players` from `_defensive_line.py`

`compute_defensive_line` returns aggregate metrics (defensive_line_x, etc.) -- not
individual player positions. Primitive (b) needs per-defender (x, y, vx, vy) tuples
to compute TTI from each back-line defender to each grid cell.

**Change:** Extract the player-selection logic at `_defensive_line.py:132-150` into:

```python
def select_back_line_players(
    frames: pd.DataFrame,
    team_id: int | str,
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Select the N outfield players closest to their own goal.

    Returns a DataFrame of player rows (preserving x, y, vx, vy, player_id, etc.)
    grouped by (game_id, period_id, frame_id), sorted by proximity to own goal.
    """
```

Both `compute_defensive_line` and `_gk_influence.compute_gk_influence` consume this
helper. `compute_defensive_line` refactored to call it internally (no behavioral
change). Public because `_gk_influence` needs it across module boundaries.

## Primitives

### (a) `gk_pitch_control_share_weighted`

GK's share of the defending team's pitch control, weighted by threat-at-(x,y).

**Computation:**
1. Compute `PitchControlSurface` with `decompose=True`.
2. Extract GK's per-cell influence via `player_surface(gk_id)` -> (ny, nx).
3. Extract defending team's per-cell influence by summing all defending teammates'
   surfaces (using `player_team_ids` to filter).
4. Per-cell share: `gk_influence[cell] / team_influence[cell]`. **Threshold guard:**
   cells where `team_influence < 1e-8` -> share = 0.0 (avoids astronomical values
   from near-zero floating-point denominators dominating the weighted average).
5. Interpolate xT grid (12x16) onto pitch control grid (50x32) via
   `ExpectedThreat.interpolator(kind="linear")`.
6. **xT orientation and away-team flip.** In LTR-normalized frames, the home team
   attacks toward x=105. The standard xT grid has high values near x=105. When
   `attacking_team_id == home_team_id`, the xT grid is correctly oriented (high
   threat near the goal the GK defends, x=105). When `attacking_team_id` is the
   away team (attacking toward x=0 in LTR frames), the xT grid must be flipped:
   `threat_grid = xT_interpolated[:, ::-1]` (x-axis flip only) so that high threat
   values are near x=0 (the goal the defending home-team GK protects). The y-axis
   is NOT flipped -- lateral position on the pitch is invariant to attack direction.
   Real xT grids trained on data have slight left-right asymmetry (crossing
   frequency, penalty tendencies); an unnecessary y-flip would introduce a subtle
   distortion compounding across frames. **Without the x-flip, the weighted share
   is inverted for away-team attacks** -- high weight on low-threat areas, low
   weight on high-threat areas.

   **Coordinate alignment note:** The `ExpectedThreat.interpolator()` output and the
   pitch control grid must use the same y-ordering (ascending from y=0 to y=68) for
   element-wise multiplication to be correct. The internal xT matrix has an inverted
   y-layout (`yj.rsub(w - 1)` at `xthreat.py:27`) but the `RectBivariateSpline`
   handles this via coordinate-based interpolation. Verify alignment during
   implementation before assuming the y-axis is consistent between the two grids.
7. Weighted average: `sum(share * threat * cell_area) / sum(threat * cell_area)`.

**Output:** Scalar in [0, 1]. Lower than raw `player_share()` when GK control
concentrates in low-threat areas (the expected and desired correction).

**Degenerate cases:** xT grid all-zeros -> return NaN (no threat surface to weight
against). GK not in frame -> raise ValueError. `sum(threat * cell_area) < 1e-8` ->
return NaN.

**`method` parameter scope:** The `method` parameter governs pitch control computation
for this primitive only. Primitives (b) and (c) always use the Spearman kinematic TTI
model regardless of `method`, because Voronoi and Fernandez-Bornn have no TTI model.
Documented in the function docstring.

### (b) `gk_reachable_area_m2`

Area the GK can reach within tau seconds that no back-line defender can also reach.
Measures the GK's unique spatial contribution.

**Computation (grid-based):**
1. For each pitch control grid cell, compute `compute_tti` (newly public, see PR-1)
   from GK position to cell center. Uses GK-specific kinematic parameters
   (`gk_reaction_time`, `gk_max_acceleration`).
2. Identify back-line defenders via `select_back_line_players` (newly extracted, see
   PR-2) on the defending team. Extract their (x, y, vx, vy) tuples.
3. For each grid cell, compute `compute_tti` from each back-line defender to cell
   center using standard `SpearmanParams` kinematic values.
4. Count cells where `TTI_gk <= tau` AND `min(TTI_defenders) > tau`.
5. Multiply by `cell_area` (from `PitchControlSurface`).

**Parameters:** `tau_seconds: float = 1.0` (configurable).

**Output:** Scalar in [0, pitch_area] where pitch_area = 7140 m^2 (105 x 68).

**Edge cases:** No outfield defenders -> reachable area = full GK reachable circle
(capped at pitch bounds). GK not in frame -> raise ValueError.

### (c) `gk_closing_time` (zone-parameterized)

Time-to-intercept from GK position to configurable target zones using the Spearman
TTI kinematic model with GK-specific parameters.

**Zone abstraction:**

```python
@dataclasses.dataclass(frozen=True)
class Zone:
    name: str                  # e.g. "six_yard_box", "near_post", "far_post"
    points: np.ndarray         # shape (N, 2), each row [x, y] in meters (LTR-normalized)
    # Array marked non-writeable in __post_init__

    @staticmethod
    def six_yard_box(goal_x: float) -> Zone     # ~9 evenly-spaced points
    @staticmethod
    def near_post(goal_x: float, ball_y: float | None = None) -> Zone  # ~4 points
    @staticmethod
    def far_post(goal_x: float, ball_y: float | None = None) -> Zone   # ~4 points
```

Factory methods take `goal_x` (0.0 or 105.0) to handle LTR-normalized frames for
either team. Goalpost y-coordinates derived from `spadlconfig.field_width` (68m ->
posts at y=30.34 and y=37.66).

**Near-post vs. far-post definition:** Relative to ball position. `ball_y` determines
which goalpost is "near" (closest to ball's y-coordinate) and which is "far." When
`ball_y` is provided: near-post = goalpost with smaller `|post_y - ball_y|`, far-post
= the other. When `ball_y is None`: falls back to fixed left/right proxy (left
half = y < 34.0, right half = y >= 34.0). The action-coupled layer always passes
`ball_y` from `start_y`; the per-frame entry point may not have ball context.

**Computation:** For each zone, compute `compute_tti` from GK (pos, vel) to each
zone point using GK-specific kinematic parameters. Aggregate to:
- `min_s`: minimum TTI across zone points (best-case interception)
- `mean_s`: mean TTI across zone points (overall zone coverage)

**Output per zone:** Two scalars (min_s, mean_s). Both >= 0.

## GK-Specific Kinematic Parameters

GKs typically have shorter reaction times (~0.3-0.4s) than outfield players (~0.7s)
due to specialized training. Using uniform outfield parameters for GK closing time
computations would systematically overestimate how long it takes the GK to reach a
zone.

`compute_gk_influence` accepts GK-specific kinematic parameters:
- `gk_reaction_time: float = 0.4` (seconds; GK-trained reaction time)
- `gk_max_acceleration: float = 7.0` (m/s^2; same as outfield default)

These are used for the GK's TTI in primitives (b) and (c). Back-line defenders use
the standard `SpearmanParams` values (reaction_time=0.7, max_acceleration=7.0).

The pitch control `lambda_gk=3.0` multiplier (which affects control probability, not
raw TTI) remains independent -- it governs primitive (a) only via
`compute_pitch_control`.

## Return Type

```python
@dataclasses.dataclass(frozen=True)
class ZoneClosingTime:
    min_s: float
    mean_s: float

@dataclasses.dataclass(frozen=True)
class GkInfluence:
    pitch_control_share_weighted: float
    reachable_area_m2: float
    closing_times: dict[str, ZoneClosingTime]   # zone.name -> ZoneClosingTime
```

Frozen, immutable. Consistent with `PitchControlSurface` pattern.

## Per-Frame Entry Point

```python
def compute_gk_influence(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    gk_player_id: int | str,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,           # REQUIRED -- no inference
    method: Method = "spearman",
    params: PitchControlParams | None = None,
    zones: list[Zone] | None = None,   # default: [Zone.six_yard_box(goal_x)]
    tau_seconds: float = 1.0,
    gk_reaction_time: float = 0.4,
    gk_max_acceleration: float = 7.0,
) -> GkInfluence:
```

**`home_team_id` is required (not optional).** The caller always knows it. Inference
from GK team vs. attacking team is ambiguous (both teams always have GKs) and adds
error-prone complexity for no benefit.

**Goal-end resolution:** In LTR-normalized frames, home team attacks toward x=105.
If defending team = home team, their goal is at x=0; if defending team = away team,
their goal is at x=105.

**Default zones:** When `zones=None`, auto-constructed as
`[Zone.six_yard_box(goal_x)]` using the resolved `goal_x`.

## Module Location

`silly_kicks/tracking/_gk_influence.py` -- single private module in `tracking/`,
following the pattern of `_defensive_line.py`, `_ball_carrier.py`,
`_off_ball_runs.py`. The pitch control subpackage (ADR-008) stays scoped to the
model itself; TF-15 is a consumer.

## Action-Coupled Integration (ADR-005)

### Per-Series Helpers (in `tracking/features.py`)

```python
def gk_pitch_control_share_weighted(
    actions, frames, xt, *, home_team_id, method="spearman",
) -> pd.Series
def gk_reachable_area_m2(
    actions, frames, xt, *, home_team_id, method="spearman", tau_seconds=1.0,
) -> pd.Series
def gk_closing_time_min_s(
    actions, frames, *, home_team_id, method="spearman", zone_name="six_yard_box",
) -> pd.Series
def gk_closing_time_mean_s(
    actions, frames, *, home_team_id, method="spearman", zone_name="six_yard_box",
) -> pd.Series
```

`home_team_id` is required on all helpers — matches established pattern
(`add_defensive_line`, `add_line_break`, `add_team_shape`, etc.). No "resolve from
game context" mechanism exists in silly-kicks; the caller always provides it.

Each: links actions to frames, resolves defending GK via `defending_gk_from_frames`,
calls `compute_gk_influence` on linked frame with the provided `home_team_id`,
extracts scalar. Returns NaN for unlinked actions or missing GK. Tolerates NaN
identifiers (ADR-003).

**Introspection mode:** When `frames=None`, returns all-NaN Series with correct name
(VAEP fit-time column discovery per ADR-005).

### Aggregator

```python
@nan_safe_enrichment
def add_gk_influence(
    actions, frames, xt, *,
    home_team_id: int | str,
    method="spearman",
    zones=None,           # default: [Zone.six_yard_box(goal_x)]
    tau_seconds=1.0,
) -> pd.DataFrame
```

**Emitted columns (default zones):**
- `gk_pitch_control_share_weighted` (float64)
- `gk_reachable_area_m2` (float64)
- `gk_closing_time_min_s__six_yard_box` (float64)
- `gk_closing_time_mean_s__six_yard_box` (float64)

With additional zones:
- `gk_closing_time_min_s__near_post`, `gk_closing_time_mean_s__near_post`
- `gk_closing_time_min_s__far_post`, `gk_closing_time_mean_s__far_post`

Column naming follows ADR-005 section 8: `<feature>__<zone>` suffix convention for zone-
parameterized columns.

**Optimization:** Single `compute_gk_influence()` call per linked frame extracts all
3 primitives. No redundant pitch control computation.

**Provenance columns:** Idempotent -- skipped if already present from prior enrichment
(e.g., `add_action_context`).

### xfns Factory

```python
def gk_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Method = "spearman",
    zones: list[Zone] | None = None,
    tau_seconds: float = 1.0,
) -> list
```

Returns a single `_frame_aware`-marked transformer. Default zones (six_yard_box only):
4 columns x 3 game states = 12 VAEP columns. With near_post + far_post: 8 columns x
3 states = 24 columns.

**Frame precomputation optimization:** The xfn factory precomputes
`compute_gk_influence` per unique `(period_id, frame_id)`, not per action-state.
The same frame appears across 3 game states and potentially multiple actions. A
dict cache keyed on `(period_id, frame_id)` avoids redundant pitch control
computation (~3x speedup over naive per-action-state calls).

**No module-level default list.** `ExpectedThreat` must be pre-fit on data, so no
sensible default exists. Consumers construct explicitly:
`xfns = gk_influence_xfns(xt)`.

### Atomic SPADL Mirror

`silly_kicks/atomic/tracking/features.py` -- delegates to standard computation with
atomic anchor columns (`x`/`y` vs `start_x`/`start_y`).

## Performance Budget

**Target: <= 10ms per `compute_gk_influence` call** (single frame, 22 players,
default 50x32 grid). Breakdown:
- `compute_pitch_control(decompose=True)`: ~5ms (existing budget)
- TTI for GK x ~1600 cells: ~0.5ms
- TTI for ~4 defenders x ~1600 cells: ~1ms
- xT interpolation: ~0.5ms
- Overhead: ~3ms

Benchmark test: `tests/tracking/test_gk_influence_perf_budget.py` following
`test_pressure_perf_budget.py` pattern. Windows CI: 1.5x ceiling (15ms).

The xfn factory's frame-precomputation cache reduces per-game cost from
~3000 calls (1000 actions x 3 states) to ~unique_frames calls (typically
~100-500 per game).

## Academic Attribution

NOTICE file entry:

- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan SAC. (pitch control
  foundation, TTI kinematic model)
- Fernandez, J., & Bornn, L. (2018). "Wide Open Spaces." MIT Sloan SAC.
  (alternate pitch control formulation)
- Get Goalside. (critique of raw pitch control GK over-crediting; motivation for
  threat-weighting)
- Karun Singh (2018). xT grid. (threat surface for weighting)

Per-function docstrings cross-link: "See NOTICE for full bibliographic citations."

## Testing Strategy

Full TDD. All tests written before implementation. No skips, no deferrals.
Spatial fixture coverage applies: input-data pathologies (partial NaN, off-pitch) +
combinatorial branch coverage (home vs away xT flip, zone x method cross-product).

### Unit Tests (`tests/tracking/test_gk_influence.py`)

**T-1: Zone geometry (5 tests)**
- `six_yard_box(goal_x=0.0)`: ~9 points, all within [0, 5.5] x [24.84, 43.16]
- `six_yard_box(goal_x=105.0)`: ~9 points, all within [99.5, 105.0] x [24.84, 43.16]
- `near_post(goal_x=0.0)` / `far_post(goal_x=0.0)`: correct y-corridors
- Frozen immutability: array non-writeable
- Repr/equality

**T-2: compute_gk_influence core logic (6 tests)**
- Threat-weighted share < raw player_share (synthetic frame, GK near own goal)
- Share in [0, 1]
- Reachable area with no defenders ~ pi * r^2 (within grid resolution)
- Reachable area decreases when defenders added
- Reachable area >= 0
- Closing times: GK at six-yard box center -> min_s ~ 0; GK at halfway line -> min_s large

**T-3: Method dispatch parity (3 tests)**
- spearman / fernandez_bornn / voronoi all produce valid GkInfluence
- Voronoi works without vx/vy columns

**T-4: Zone-parameterized closing time (3 tests, parametrized)**
- Parametrize across [six_yard_box, near_post, far_post]
- closing_times dict has correct keys
- GK near near-post -> near_post min_s < far_post min_s (physical invariant)

**T-5: xT grid interpolation + orientation (6 tests)**
- 12x16 xT correctly interpolated onto 50x32 pitch control grid
- Sum of threat weights > 0
- xT all-zeros -> share = NaN
- **Home-team attack: xT grid NOT flipped (high threat near x=105)**
- **Away-team attack: xT grid x-FLIPPED only (high threat near x=0). Two identical
  frames, home vs away attacking. Assert share differs and away-attack produces
  correct direction (higher share when GK controls high-threat zone).**
- **Y-axis preserved across flip: compare threat_grid[0, :] (bottom row) between
  home-attack and away-attack — values differ in x-order but NOT in y-position.
  Confirms the flip is `[:, ::-1]` not `[::-1, ::-1]`.**

**T-6: Edge cases (7 tests)**
- GK not in frame -> ValueError
- No outfield defenders -> reachable area = full GK reachable circle
- min_s <= mean_s always
- Near-zero team_influence denominator (1e-15) -> share = 0.0, not infinity
- GK with custom reaction_time=0.3 produces lower closing times than default 0.4
- GK with custom reaction_time=0.3 produces larger reachable_area_m2 than default
  0.4 (lower reaction time -> GK reaches more cells within tau -> monotonicity)
- near_post(goal_x=0.0, ball_y=25.0) vs near_post(goal_x=0.0, ball_y=40.0):
  different point sets (ball-relative definition exercised)

### Prerequisite API Tests

**T-PR1: compute_tti public export (2 tests)**
- Importable from `silly_kicks.tracking.pitch_control`
- Produces identical results to current private `_compute_tti` (regression guard)

**T-PR2: select_back_line_players (3 tests)**
- Returns individual player rows with x, y, vx, vy preserved
- `compute_defensive_line` unchanged behavior after refactor (regression guard)
- Correct player selection for home team (defend x=0) vs away team (defend x=105)

### Action-Coupled Tests (`tests/tracking/test_gk_influence_action_coupled.py`)

**T-7: Per-Series helpers (4 tests)**
- Known GK position -> correct scalar extraction
- Unlinked actions -> NaN
- NaN team_id -> NaN
- frames=None introspection -> all-NaN with correct Series name

**T-8: Aggregator (4 tests)**
- Correct column set (4 default + provenance)
- Idempotent provenance (after add_action_context)
- @nan_safe_enrichment decorator present
- Additional zones -> additional columns

**T-9: xfns factory (6 tests)**
- Returns list with one _frame_aware-marked transformer
- Introspection: frames=None -> correct column names, all NaN
- Full mode: 12 columns (4 x 3 states) with default zones
- Column naming: `<metric>__<zone>_a{i}`
- **Frame precomputation cache: multi-action fixture where 2 actions share the same
  frame_id across 3 game states. Instrument compute_gk_influence call count. Assert
  invocations = unique frames, not total action-states.**
- **Cache key includes all relevant parameters: changing method or tau_seconds
  produces different results (no stale cache hit)**

**T-10: Atomic SPADL mirror (4 tests)**
- All 3 primitives produce identical values through both anchor paths (x/y vs
  start_x/start_y) on same fixture
- Closing time values match between standard and atomic paths
- Aggregator column set matches between standard and atomic

### Provider e2e Tests (`tests/tracking/test_gk_influence_e2e.py`)

**T-11: Per-provider e2e** (marked `@pytest.mark.e2e`, parametrized)
- Sportec (lakehouse fixture): native GK, native velocities -> all 3 primitives non-NaN
- Metrica (kloppy fixture): derived GK, derived velocities -> all 3 primitives non-NaN
- SkillCorner (kloppy fixture): extrapolated GK -> all 3 primitives non-NaN
- PFF (local WC2022 fixture): native GK -> all 3 primitives non-NaN
- Per provider: assert n_valid >= 1 per primitive (no silent all-NaN regression)
- Physical invariants: reachable_area_m2 >= 0, closing_time_min_s >= 0, share in [0, 1]

### Synthesizer Fixture Enhancement

Existing slim-parquet synthesizer (`tests/tracking/_provider_inputs.py`) must provide
per provider:
- >= 1 frame with valid defending GK (is_goalkeeper=True on opposing team)
- Outfield players with valid x/y + vx/vy
- >= 1 linked action within 0.2s tolerance
- **Input-data pathologies:** >= 1 player with partial NaN (valid x, NaN y) to exercise
  any NaN-guarding logic (spatial fixture coverage discipline)

If existing synthesized fixtures lack this coverage, generate additional test frames.
Follows PR-S21 shot+keeper_save pattern.

### Performance Benchmark (`tests/tracking/test_gk_influence_perf_budget.py`)

**T-PB1: Per-frame budget**
- compute_gk_influence on 22-player frame <= 10ms (Linux) / 15ms (Windows)
- Follows test_pressure_perf_budget.py pattern

### Invariant Tests (`tests/invariants/test_gk_influence_invariants.py`)

**T-12: Physical invariants** (parametrized across providers)
- `gk_pitch_control_share_weighted` in [0, 1]
- `gk_reachable_area_m2` in [0, 7140]
- `gk_closing_time_min_s <= gk_closing_time_mean_s`
- `gk_closing_time_min_s >= 0`
- GK closer to zone -> lower closing time (monotonicity)

### Test Count Summary

| Category | Tests |
|----------|-------|
| T-1 Zone geometry | 5 |
| T-2 Core logic | 6 |
| T-3 Method dispatch | 3 |
| T-4 Zone-parameterized | 3 |
| T-5 xT interpolation + orientation | 6 |
| T-6 Edge cases | 7 |
| T-PR1 compute_tti export | 2 |
| T-PR2 select_back_line_players | 3 |
| T-7 Per-Series helpers | 4 |
| T-8 Aggregator | 4 |
| T-9 xfns factory | 6 |
| T-10 Atomic mirror | 4 |
| T-11 Provider e2e | 4 (parametrized) |
| T-PB1 Perf benchmark | 1 |
| T-12 Physical invariants | 5 (parametrized) |
| **Total** | **~63** |

## Bundled Fixes: TF-31/TF-32 (National Park Principle)

Lakehouse review of 3.9.0 surfaced bugs in `_line_breaking.py` and gaps in
`_team_shape.py`. Bundled into this PR rather than a separate hotfix.

### H1 -- BUG: Independent dropna misalignment (`_line_breaking.py:178-179`)

`opp_x = opp_df["x"].dropna()` and `opp_y = opp_df["y"].dropna()` independently.
If any opponent has valid x but NaN y (or vice versa), arrays have different lengths.
Ward clustering on `opp_x` produces labels that index into misaligned `opp_y` --
silent data corruption.

**Fix:** Joint dropna.

```python
valid_mask = opp_df["x"].notna() & opp_df["y"].notna()
valid_opp = opp_df[valid_mask]
opp_x = valid_opp["x"].to_numpy(dtype="float64")
opp_y = valid_opp["y"].to_numpy(dtype="float64")
```

**Test:** Synthetic fixture with one opponent having y=NaN, assert no crash and
correct array alignment.

### H2 -- BUG: Extension-poisoning on line_breaking_type (`_line_breaking.py:241-266`)

`broke_on_extension` is set True if ANY intersecting segment is a sideline extension.
But if the same cluster has BOTH a between-players intersection AND an extension
intersection, `broke_on_extension=True` blocks `any_through` from being set --
producing "around_line" instead of "between_lines". Spec says "between_lines"
should dominate.

**Fix:** Track `cluster_has_through` independently.

```python
cluster_has_through = False
for si in range(n_segments):
    if _segments_intersect(...):
        cluster_broken = True
        if si != 0 and si != n_segments - 1:
            cluster_has_through = True
if cluster_broken:
    lines_broken += 1
    if cluster_has_through:
        any_through = True
```

**Test:** Pass intersecting both an extension segment and a between-players segment
of the same cluster -- assert "between_lines".

### H3 -- Algorithm divergence documentation

Paper (Karakus & Arkadas 2025) uses centroid + vertical-span intersection test.
Implementation uses polyline + cross-product straddle test (arguably better, captures
actual defensive geometry). Add "Deviations from reference" note to `_line_breaking.py`
module docstring documenting the divergence.

### M1 -- Performance benchmark tests

Add `test_team_shape_perf_budget.py` and `test_line_breaking_perf_budget.py` following
`test_pressure_perf_budget.py` pattern. Budgets: team shape <= 1ms/frame (10 outfield),
line-breaking <= 2ms/pass. Windows CI: 1.5x ceiling per `feedback_windows_ci_perf_budget`.

### M2 -- 0-player frame contract clarification

`compute_team_shape` returns 0 rows for 0-player frames. This is correct and consistent
with `compute_defensive_line`. Add docstring clarification: "Frames with zero visible
outfield players are omitted from output (consumers should LEFT JOIN and fill NaN)."

### M3 -- k=3 design choice documentation

Add docstring note: "`n_clusters=3` is a design choice (defense/midfield/attack
partition), not from the reference paper. Configurable via `LineBreakingParams`."

### M4 -- Non-pass action type filtering

Non-pass actions (shots, dribbles, tackles, etc.) should produce NaN/pd.NA, not be
analyzed. A shot cannot "break a line" in the tactical sense. Add action-type filter:
only passes and crosses are analyzed; all others get the empty/NA row.

**Test:** Synthetic actions with shot + dribble action types, assert all produce pd.NA.

### L1 -- Out-of-scope paper metrics

Add note to spec/docstring listing SBR, LBPCh1, LBPCh2 as out-of-scope paper metrics.

### L2 -- Provider sweep value invariants

Add invariant assertions to `test_line_breaking_providers.py` and
`test_team_shape_providers.py`: `lines_broken in {0,1,2,3}`,
`convex_hull_area >= 0`, `stretch_index >= 0`, `n_outfield_players > 0`, etc.

### Bundled Fix Test Count

| Fix | Tests Added |
|-----|-------------|
| H1 dropna misalignment | 1 |
| H2 extension-poisoning | 1 |
| M1 perf benchmarks | 2 (team shape + line-breaking) |
| M4 non-pass filtering | 1 |
| L2 provider invariants | ~8 (parametrized across providers x feature) |
| **Total** | **~13** |

## Scope Boundaries

**In scope:**
- Export `compute_tti` from pitch control subpackage (prerequisite PR-1)
- Extract `select_back_line_players` from `_defensive_line.py` (prerequisite PR-2)
- Per-frame primitives (a), (b), (c) in `_gk_influence.py`
- Zone dataclass with 3 factory constructors
- GkInfluence + ZoneClosingTime return dataclasses
- GK-specific kinematic parameters (gk_reaction_time, gk_max_acceleration)
- xT orientation flip for away-team attacks
- Action-coupled per-Series helpers, aggregator, xfns factory (with frame precomputation cache)
- Atomic SPADL mirror
- Performance benchmark
- NOTICE entry + docstring cross-links
- Full TDD + e2e test suite (~63 TF-15 tests + ~13 bundled fix tests)
- Bundled TF-31/TF-32 bugfixes (H1, H2, M4) + documentation (H3, M2, M3, L1) + tests (M1, L2)

**Out of scope:**
- DAS-based GK primitives (TF-28 complement, future TF-15 extension)
- Cover-shadow GK blocking_score (TF-30, future)
- Ghost-GK regression (TF-18, Layer 2)
- Decision-probability surfaces (TF-16/17, Layer 2)
- GKDV composition (TF-19, Layer 3)
- Cascioli et al. 2025 defensive repositioning optimization (Hungarian algorithm on
  cone-intersection polygons; TF-30 scope)
- Auto-k for Ward clustering (future TF, M3 documents k=3 as design choice)
- Fallback threat surface for per-frame use without pre-fit xT (L3 — YAGNI; callers
  needing quick approximation can fit ExpectedThreat on a small action set)

## Lakehouse Review Disposition

### Review v1

| Item | Severity | Resolution |
|------|----------|------------|
| H1 TTI private | High | PR-1: export compute_tti as public API |
| H2 No individual player data | High | PR-2: extract select_back_line_players |
| H3 xT flip for away-team | High | Explicit flip logic + T-5 tests |
| M1 GK kinematic params | Medium | gk_reaction_time=0.4 / gk_max_acceleration=7.0 params |
| M2 method scope | Medium | Documented in spec + docstring |
| M3 No perf budget | Medium | 10ms budget + benchmark test |
| M4 Epsilon guard | Medium | Threshold 1e-8 instead of exact-zero |
| M5 home_team_id optional | Medium | Made required on compute_gk_influence |
| L1 Cascioli reference | Low | Clarified attribution to TF-30 scope |
| L2 Atomic mirror thin | Low | Expanded to 4 tests |
| L3 xT pre-fitting heavy | Low | Deferred (YAGNI) |

### Review v2

| Item | Severity | Resolution |
|------|----------|------------|
| H1 home_team_id missing from 6 action-coupled signatures | High | Added to all per-Series helpers, aggregator, and xfns factory. Matches established pattern (add_defensive_line, add_line_break, add_team_shape, etc.) |
| M1 xT flip [::-1, ::-1] should be [:, ::-1] | Medium | Changed to x-only flip. Y-axis invariant to attack direction. Added coordinate alignment verification note + T-5 y-preservation test |
| M2 Zone.near_post/far_post factories lack ball_y | Medium | Added ball_y: float \| None = None parameter. Ball-relative when provided, fixed proxy when None. Action-coupled layer passes start_y |
| L1 Missing cache + reaction_time reachable area tests | Low | Added cache invocation count test (T-9) + reaction_time monotonicity on reachable_area (T-6) |
