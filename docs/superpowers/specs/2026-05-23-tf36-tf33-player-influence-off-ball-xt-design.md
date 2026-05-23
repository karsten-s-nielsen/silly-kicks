# TF-36 + TF-33: Per-Player Influence Primitives + Off-Ball xT

**Date:** 2026-05-23
**Status:** Draft
**PR:** TBD (PR-S51)
**Depends on:** TF-7 (pitch control, shipped 3.7.0), TF-15 (GK influence, shipped 3.10.0)

---

## 1. Problem

Two TODO items share the same computational hot path (`compute_pitch_control(decompose=True)` + xT interpolation):

- **TF-36** — per-player uniquely reachable area. Generalizes TF-15's GK-specific primitive (b) to all outfield players. Answers "how much dangerous space does only this player cover?"
- **TF-33** — off-ball xT. Per-player positional value via `player_surface(pid) * xT_grid`. Composes Spearman 2017 pitch control + Karun Singh 2018 xT.

Shipping them separately means two PRs that each independently call `compute_pitch_control(decompose=True)` — the single most expensive tracking operation. Bundling shares the call.

## 2. Design decisions

### 2.1 Single unified module

New `silly_kicks/tracking/_player_influence.py` with one per-frame primitive (`compute_player_influence`) that computes both metrics for all players in one pass. One `compute_pitch_control(decompose=True)` call per frame serves both metrics.

**Rejected alternatives:**
- Composition-only in `features.py` (no per-frame primitive): untestable per-frame logic, can't reuse outside VAEP.
- Two separate modules (`_off_ball_xt.py` + `_reachable_area.py`): two independent PC calls for the same frame, wasted compute.

### 2.2 Uniform outfield kinematic constants

All outfield players use the same kinematic parameters for TTI computation: `reaction_time` and `max_acceleration` default to `SpearmanParams` values (the same constants pitch control itself uses).

**Extension path to per-position-group constants:** The primitive accepts `reaction_time` and `max_acceleration` as parameters. A future caller can pass per-player kinematic tables without any API break. The extension requires: (1) position classifier or provider-supplied roles, (2) kinematic lookup `{position_group: (reaction_time, max_accel)}`, (3) per-player values into the same primitive. No API change, no deprecation.

### 2.3 GK exclusion and comparison-set scope

GKs are excluded from both the output dict and the "teammates" comparison set for uniquely-reachable-area. GK coverage operates under different kinematics (TF-15's `compute_gk_influence` handles GKs). Including GKs in the outfield comparison set would artificially reduce field players' unique area.

**Difference from TF-15:** TF-15's `compute_gk_influence` compares GK TTI against back-line defenders only (via `select_back_line_players`), measuring unique sweeping reach beyond the defensive line. TF-36 compares each outfield player against **all same-team outfield players except self**, measuring unique spatial contribution where any teammate's coverage counts. The broader comparison set is intentional: GK sweeping is constrained to the space behind the back line, while field-player influence spans the entire pitch.

### 2.4 Dual-team aggregation with diffs

Action-coupled output follows the DAS `_team`/`_opponent`/`_diff` convention (PR-S32). Both teams' spatial state matters — attacking off-ball value AND defending coverage quality are independent VAEP-relevant signals.

## 3. Per-frame primitive

### 3.1 Module: `silly_kicks/tracking/_player_influence.py`

```python
@dataclass(frozen=True)
class PlayerInfluence:
    """Per-player per-frame influence measurement."""
    off_ball_xt: float           # xT-weighted spatial contribution (m^2 * xT)
    reachable_area_m2: float     # uniquely reachable area (m^2)


def compute_player_influence(
    frame: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
    surface: PitchControlSurface | None = None,
    tau_seconds: float = 1.0,
    reaction_time: float | None = None,
    max_acceleration: float | None = None,
) -> dict[int | str, PlayerInfluence]:
```

### 3.2 Internal flow

1. **Pitch control surface:** If `surface` is provided, use it directly — `method` and `params` are ignored (PC computation is skipped). The provided surface MUST have been computed with `decompose=True`; no additional runtime validation is added — the existing `ValueError` from `PitchControlSurface.player_surface()` is the error path. Otherwise, `compute_pitch_control(frame, attacking_team_id, method, decompose=True)` — single call. The `surface` parameter allows callers (e.g., a future shared PC cache) to avoid redundant computation.
2. **xT interpolation:** `xt.interpolator("linear")(grid_x, grid_y)` → `threat_grid (ny, nx)`. Flip for away-team attack direction (same logic as `_gk_influence.py` lines 346-349).
3. **Per outfield player** (non-ball, non-GK, valid x/y):
   - **Off-ball xT** = `sum(player_surface(pid) * threat_grid * cell_area)`.
   - **Uniquely reachable area** — uses the team-TTI-matrix optimization (§3.4).
4. Return `dict[player_id -> PlayerInfluence]`.

**NaN velocity handling:** NaN `vx`/`vy` values default to 0.0 (stationary player assumption), consistent with `_gk_influence.py` lines 199-201. This prevents NaN propagation through `compute_tti` which would silently zero out all reachable areas (NaN comparisons are always False).

**Ball-carrier semantics:** The per-frame primitive returns values for ALL outfield players, including the ball carrier. It has no concept of "on-ball" vs "off-ball" — that distinction is enforced at the aggregator level (§4.4) via actor exclusion. The primitive is a pure spatial measurement; the "off-ball" label applies only to the aggregated column name.

### 3.3 Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| `tau_seconds` | 1.0 | Same as TF-15 |
| `reaction_time` | `SpearmanParams().reaction_time` | Outfield default; TF-15 uses 0.4s for GKs |
| `max_acceleration` | `SpearmanParams().max_acceleration` | Outfield default; TF-15 uses 7.0 for GKs |

### 3.4 Implementation optimization: team-TTI-matrix with argmin/second-min

The naive per-player loop for uniquely reachable area is O(n^2): for each of N outfield players per team, compute TTI of all N-1 teammates to all grid cells. With 10 outfield players per team and 1700 grid cells, that's 10 × 9 × 1700 = 153K TTI evaluations per team — not the 20 × 1700 = 34K claimed by the naive estimate.

**Mandatory optimization:** Compute the full-team TTI matrix once per team:

1. For each team, compute `tti_matrix[i, j]` = TTI of player `i` to grid cell `j`. Shape `(N_team, n_cells)`. This is `N_team` `compute_tti` calls, not `N_team × (N_team - 1)`.
2. Compute `global_min[j] = min(tti_matrix[:, j])` and `global_argmin[j] = argmin(tti_matrix[:, j])`.
3. Compute `second_min[j]` = minimum of `tti_matrix[:, j]` excluding the argmin row.
4. For player `i` at cell `j`:
   ```
   min_tti_excluding_i[j] = second_min[j]  if argmin[j] == i
                            global_min[j]   otherwise
   ```
5. Unique cells for player `i` = `tti_matrix[i, :] <= tau` AND `min_tti_excluding_i > tau`.

This reduces TTI calls from `2 × N × (N-1)` to `2 × N` (one per player per team). For 20 outfield players: 20 `compute_tti` calls instead of ~180.

The `second_min` is computed via `np.partition(tti_matrix, kth=1, axis=0)[1, :]` — O(n) per cell via introselect, no full sort needed. When `N_team == 1`, skip the partition step — every cell within tau is uniquely reachable (no teammates to exclude). `np.partition(..., kth=1)` requires axis 0 to have >= 2 elements; the 1-player guard prevents an IndexError.

## 4. Action-coupled layer

### 4.1 Batch kernel

`_player_influence_at_actions(actions, frames, xt, *, home_team_id, links)` in `features.py`.

Cache key: `(period_id, frame_id, attacking_team_id)`. One `compute_player_influence` call per unique key. Returns `(result_df, pointers)`. The cache is scoped to a single `_player_influence_at_actions` invocation — `method`, `tau_seconds`, and kinematic parameters are fixed for the entire call, so they don't need to be part of the cache key.

### 4.2 Aggregator

```python
@nan_safe_enrichment
def add_player_influence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> pd.DataFrame:
```

### 4.3 Output columns (7 + 4 provenance)

| Column | Derivation |
|--------|------------|
| `actor_reachable_area_m2` | Actor's own `reachable_area_m2` from per-frame dict |
| `off_ball_xt_team` | Sum of teammates' `off_ball_xt` (same team as actor, excluding actor) |
| `off_ball_xt_opponent` | Sum of opponents' `off_ball_xt` |
| `off_ball_xt_diff` | `off_ball_xt_team - off_ball_xt_opponent` |
| `reachable_area_team` | Sum of same-team `reachable_area_m2` (including actor) |
| `reachable_area_opponent` | Sum of opponent-team `reachable_area_m2` |
| `reachable_area_diff` | `reachable_area_team - reachable_area_opponent` |

Plus 4 linkage-provenance columns (`frame_id`, `time_offset_seconds`, `n_candidate_frames`, `link_quality_score`) with idempotent skip-guard.

**Actor inclusion/exclusion asymmetry rationale:**
- `off_ball_xt_team` **excludes** actor: "off-ball" means players not performing the action. The actor is on the ball; their spatial xT contribution is captured by the action's own start_x/start_y, not by this feature.
- `reachable_area_team` **includes** actor: this measures total team spatial footprint — how much unique coverage the team provides collectively. The actor's coverage contributes to the team's defensive/offensive shape regardless of possession status.
- `actor_reachable_area_m2` is the actor's personal unique coverage — useful as an independent signal of positional quality at the moment of action.

### 4.4 Team aggregation logic

Per action:
1. Look up actor's `player_id` from action row.
2. From cached `dict[player_id -> PlayerInfluence]`, partition by team membership (from PC surface's `player_team_ids`).
3. Actor's team = `action["team_id"]`. Opponent team = the other team_id in the dict.
4. `off_ball_xt_team` = sum of teammates' `off_ball_xt`, excluding actor.
5. `off_ball_xt_opponent` = sum of opponents' `off_ball_xt`.
6. `actor_reachable_area_m2` = actor's own `reachable_area_m2`.
7. `reachable_area_team` = sum of same-team `reachable_area_m2` (including actor).
8. `reachable_area_opponent` = sum of opponent-team `reachable_area_m2`.
9. `_diff` = `_team - _opponent`.

**xT orientation:** All values (both teams) are computed from the attacking team's threat perspective — the xT grid is oriented so high-xT cells are near the attacking team's target goal. This means `off_ball_xt_opponent` measures "how much threat-relevant space opponents cover from the attacker's viewpoint" — i.e., defensive positioning quality against the current attack direction. This is the correct VAEP signal. The alternative (each team's xT uses their own attack direction) would require two PC calls per frame and produce values that aren't directly comparable for the `_diff` column.

### 4.5 Per-Series helpers

5 standalone functions (each calls batch kernel, extracts one column):

- `actor_reachable_area_m2(actions, frames | None, xt, *, home_team_id) -> pd.Series`
- `off_ball_xt_team(actions, frames | None, xt, *, home_team_id) -> pd.Series`
- `off_ball_xt_opponent(actions, frames | None, xt, *, home_team_id) -> pd.Series`
- `reachable_area_team(actions, frames | None, xt, *, home_team_id) -> pd.Series`
- `reachable_area_opponent(actions, frames | None, xt, *, home_team_id) -> pd.Series`

All return NaN Series when `frames is None` (column-name probing tolerance per `feedback_vaep_feature_column_names_introspection.md`).

`_diff` columns have no per-Series helpers — trivial subtractions only meaningful in aggregator context.

## 5. VAEP integration

### 5.1 Factory

```python
def player_influence_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    tau_seconds: float = 1.0,
) -> list:
```

Returns a single `FrameAwareTransformer` that emits 7 columns x 3 gamestate slots = **21 VAEP columns**.

### 5.2 Precomputation

The transformer precomputes `compute_player_influence` for all unique `(period_id, frame_id, attacking_team_id)` across the full frames DataFrame in one pass. Cached results are reused across the 3 gamestate slots and repeated actions.

### 5.3 Column-name probing

Empty frames (10-row dummy for `feature_column_names` introspection): all 21 columns filled with NaN.

### 5.4 Composition

```python
# Standalone
xfns = tracking_default_xfns + player_influence_xfns(xt, home_team_id=1)

# Combined with GK influence
xfns = (tracking_default_xfns
        + gk_influence_xfns(xt, home_team_id=1)
        + player_influence_xfns(xt, home_team_id=1))
```

**PC surface sharing trade-off:** When both `gk_influence_xfns` and `player_influence_xfns` are composed, each transformer has its own internal cache and calls `compute_pitch_control(decompose=True)` independently for the same frame. This doubles the cost of the most expensive operation. The `surface` parameter on `compute_player_influence` (§3.1) enables callers who control both calls to share the surface, but the xfns transformers are independently encapsulated and do not share today.

**Accepted for this PR.** A shared `_pitch_control_cache` helper that both xfns factories read from is the right long-term fix but requires refactoring `gk_influence_xfns`'s internal cache too — out of scope. The duplication only matters when both xfns are composed (power-user case). The `surface` parameter ensures the primitive itself is ready for sharing when the cache is built.

### 5.5 Atomic mirror

`atomic.tracking.features` gets the same aggregator + xfns factory, following the established pattern.

## 6. Testing strategy

### 6.1 Unit tests (`tests/tracking/test_player_influence.py`)

**Per-frame primitive correctness:** Synthetic frame with known positions + pre-fit xT grid.
- `off_ball_xt` > 0 for players in high-xT zones, ~0 for low-xT corners.
- `reachable_area_m2` > 0 for isolated players, smaller for clustered teammates.
- GKs excluded from output dict. Ball rows excluded.

**Uniquely-reachable-area invariants:**
- Sum of all same-team uniquely reachable areas <= total pitch area (7140 m^2) — no double-counting.
- Player far from teammates -> large area; adjacent to teammate -> small area.
- `tau_seconds=0` -> all areas = 0 (nobody reaches anywhere).

**Off-ball xT invariants:**
- Sum of all players' `off_ball_xt` (both teams) approximates total xT-weighted pitch area (conservation). This invariant holds for Spearman (PC decomposition sums to 1.0 at every cell). For Fernandez-Bornn/Voronoi, the invariant is: sum <= total xT-weighted pitch area (PC may not sum to exactly 1.0 due to normalization differences). Tests are parametrized by method.
- Attacker near opponent goal -> higher `off_ball_xt` than attacker at own corner.

**Edge cases:**
- Frame with 1 outfield player per team: no teammates -> every cell within tau is uniquely reachable (correct behavior — maximum individual coverage).
- Frame with <2 total outfield players on one team: output dict contains the single player's values; `off_ball_xt_team` (excl actor) = 0.
- `method="voronoi"`: binary (0/1) per-player surfaces produce integer cell-count x xT values. Test verifies non-degenerate (non-zero) results.
- `method="fernandez_bornn"`: smooth surface, test verifies values are in sensible range.
- All players at same position: all uniquely reachable areas = 0 (symmetric tie, tau threshold never uniquely met).

**Action-coupled aggregator:**
- `_diff` = `_team - _opponent` identity (exact equality).
- `actor_reachable_area_m2` matches actor's entry from per-frame dict.
- `off_ball_xt_team` excludes actor's own contribution.
- NaN actor `team_id` -> all output columns NaN.
- Provenance skip-guard: pre-existing `frame_id` not duplicated.

**VAEP xfns smoke:**
- `feature_column_names` probing returns 21 column names.
- 3-row gamestate with linked frames -> 21 non-NaN columns.

**TTI optimization correctness:**
- Verify that the argmin/second-min trick produces identical results to the naive per-player loop on a small (5-player) test frame. Exact floating-point equality — the optimization must be numerically equivalent, not approximate.

### 6.2 Snapshot test (`tests/tracking/test_player_influence_snapshot.py`)

Multi-hash set pattern (per `feedback_multi_hash_snapshot_sets.md`) for numpy runner drift.

### 6.3 CI gates

- `tests/tracking/test_provenance_skip_guard.py` — `add_player_influence` added to parametrized guard.
- `tests/test_enrichment_nan_safety.py` — `@nan_safe_enrichment` auto-discovered.
- Atomic mirror test: `atomic.tracking.features.player_influence_xfns` exists + same column count.

### 6.4 Benchmark test (`tests/tracking/test_player_influence_benchmark.py`)

`pytest-benchmark` test for `compute_player_influence` with a realistic synthetic frame (20 outfield players, 50x34 grid). Per-frame budget: TBD — measured during implementation from first passing test, then set with 1.5x headroom per `feedback_windows_ci_perf_budget.md`. Flat ceiling (no platform ternary). The benchmark validates that the §3.4 TTI optimization is effective; regression to O(n^2) would exceed the budget.

### 6.5 E2E provider test (`tests/tracking/test_player_influence_e2e.py`, marked `@pytest.mark.e2e`)

Full pipeline test following the GK influence E2E pattern: load vendored provider fixture -> preprocess -> compute_player_influence -> add_player_influence -> verify 7 output columns are non-NaN on at least 1 action. Run against available vendored tracking fixtures (Sportec, Metrica, Gradient Sports). Verifies the pipeline doesn't crash on real-world data shapes, column dtypes, and missing-data patterns.

## 7. Files touched

### New
- `silly_kicks/tracking/_player_influence.py` (~100-130 LOC)
- `tests/tracking/test_player_influence.py`
- `tests/tracking/test_player_influence_snapshot.py`
- `tests/tracking/test_player_influence_benchmark.py`
- `tests/tracking/test_player_influence_e2e.py`

### Modified
- `silly_kicks/tracking/features.py` — batch kernel, aggregator, per-Series, xfns factory (~120 LOC)
- `silly_kicks/tracking/__init__.py` — re-export `compute_player_influence`, `PlayerInfluence`
- `silly_kicks/atomic/tracking/features.py` — atomic mirror
- `tests/tracking/test_provenance_skip_guard.py` — add `add_player_influence`
- `NOTICE` — academic attribution entry
- `CHANGELOG.md` — new feature
- `TODO.md` — delete TF-33 and TF-36 rows

## 8. Performance

Hot path: `compute_pitch_control(decompose=True)` + per-player TTI.

Per frame (20 outfield players, 2 teams of ~10, 50x34 grid = 1700 cells):
- 1 PC call (same cost as GK influence; shared within this primitive via `surface` parameter)
- TTI with §3.4 optimization: 20 `compute_tti` calls (one per outfield player) x 1700 targets = 34K TTI evaluations (numba-accelerated). Without the optimization, this would be ~180 calls (O(n^2)); the argmin/second-min trick makes it O(n).
- `np.partition` for second-min: O(n_players) per cell, ~1700 × 20 = 34K comparisons — negligible vs TTI.

Per match (~50 unique action-linked frames): ~50 PC calls total, cached by `(period_id, frame_id, attacking_team_id)`.

**Cross-xfn duplication:** When composed with `gk_influence_xfns`, PC is computed twice per frame (see §5.4 trade-off discussion). Accepted for this PR.

## 9. Academic attribution (NOTICE entry)

- Spearman, W. (2017). "Beyond Expected Goals." MIT Sloan SAC. (pitch control + TTI) — already cited.
- Singh, K. (2018). "Introducing Expected Threat (xT)." (xT framework) — already cited.
- Composition (per-player xT x PC share as tracking feature) is novel to silly-kicks.

## 10. Dependencies

None new. All building blocks shipped: pitch control (TF-7, 3.7.0), xT (`xthreat.py`), `compute_tti` (pitch_control subpackage), GK influence pattern (TF-15, 3.10.0).
