# TF-31 + TF-32: Team Shape Envelope & Ward Line-Breaking

**Date:** 2026-05-09
**Target release:** silly-kicks 3.9.0
**Branch:** `pr-s33-team-shape-line-breaking`

---

## 1. TF-31 --- Team Shape Envelope

### 1.1 Motivation

Team shape descriptors (centroid, convex hull area, length, width, stretch
index) are the canonical per-team spatial metrics used across virtually every
tracking-data paper since Clemente et al. (2013). They quantify whole-team
spatial organization: how compact or spread the team is, its center of mass,
and the envelope it occupies.

silly-kicks ships TF-14 (`compute_defensive_line`) which covers the defending
team's back-line geometry (6 columns). TF-31 complements this with the
whole-team spatial envelope for *either* team --- complementary, not
overlapping.

**References:**
- Clemente, F. M., Couceiro, M. S., Martins, F. M. L., & Mendes, R. (2013).
  "Measuring Tactical Behaviour Using Technological Metrics: Case Study of a
  Football Game." International Journal of Sports Science & Coaching, 8(4).
- Zhang, G., Kempe, M., McRobert, A., Folgado, H., & Olthof, S. B. H. (2025).
  "Navigating team tactical analysis in football: An analytical pipeline
  leveraging player tracking technology." International Journal of Sports
  Science & Coaching.

### 1.2 Module Layout

New private module `silly_kicks/tracking/_team_shape.py` with public symbols
re-exported from `silly_kicks/tracking/__init__.py`.

### 1.3 Per-Frame Primitive

```python
def compute_team_shape(
    frames: pd.DataFrame,
    team_id: int | str,
) -> pd.DataFrame:
```

**Input:** Long-form tracking frames (TRACKING_FRAMES_COLUMNS schema).

**Return shape:** One row per `(game_id, period_id, frame_id)` where the team
has at least one visible outfield player. Batch result via groupby, matching
the `compute_defensive_line` pattern.

**Processing per (game_id, period_id, frame_id):**
1. Filter to rows matching `team_id` where `is_goalkeeper == False` and
   `is_ball == False`.
2. Drop rows with NaN x or y.
3. Compute metrics from the remaining outfield player positions.

**Output columns:**

| Column | Type | Semantic |
|--------|------|----------|
| `game_id` | int64 | Game identifier |
| `period_id` | int64 | Period identifier |
| `frame_id` | int64 | Frame identifier |
| `team_id` | int64/object | Team identifier |
| `n_outfield_players` | Int64 | Count of visible outfield players contributing to metrics |
| `centroid_x` | float64 | Mean x of outfield players |
| `centroid_y` | float64 | Mean y of outfield players |
| `convex_hull_area` | float64 | `scipy.spatial.ConvexHull.volume` (2D = area, m²) |
| `team_length` | float64 | x-axis spread: `max(x) - min(x)` |
| `team_width` | float64 | y-axis spread: `max(y) - min(y)` |
| `stretch_index` | float64 | Mean Euclidean distance from centroid |

**Edge cases:**
- 0 visible outfield players: all metrics = NaN for that frame.
- 1 player: centroid = player position; length = width = stretch_index = 0.0;
  convex_hull_area = NaN (undefined for < 3 points).
- 2 players: centroid/length/width/stretch_index computable;
  convex_hull_area = NaN (need ≥ 3 non-collinear points).
- ≥ 3 collinear players: `ConvexHull` raises `QhullError`; catch and emit
  convex_hull_area = NaN (degenerate hull with zero area).

**Implementation notes:**
- Explicit guard: `if n_valid < 3: convex_hull_area = NaN` before calling
  `ConvexHull`. Keep `QhullError` catch as a safety net for the collinear
  case, but don't rely on exceptions for expected conditions.
- Use `np.max(arr) - np.min(arr)` rather than deprecated `np.ptp`.
- `scipy.spatial.ConvexHull` is already in the dependency tree (scipy).
- Stretch index formula: `mean(sqrt((x_i - cx)² + (y_i - cy)²))`.

### 1.4 Action-Coupled Aggregator

```python
@nan_safe_enrichment
def add_team_shape(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
) -> pd.DataFrame:
```

**Processing:**
1. Calls `link_actions_to_frames(actions, frames)` to get per-action frame
   pointers. Note: `link_actions_to_frames` is O(n log n) per call and is
   recomputed internally by each `add_*` aggregator when chained. This is the
   existing pattern across all silly-kicks aggregators; shared-linkage
   optimization is a future concern, not in scope for this PR.
2. Computes `compute_team_shape` for both teams (batch, not per-action).
3. For each action, resolves which team is attacking (action's `team_id`) and
   which is defending (the other team).
4. Joins the per-frame metrics to actions via (game_id, period_id, frame_id).

**Output: 14 columns** (7 metrics × 2 teams):
- `team_shape_n_outfield_players_attacking`
- `team_shape_centroid_x_attacking`
- `team_shape_centroid_y_attacking`
- `team_shape_convex_hull_area_attacking`
- `team_shape_team_length_attacking`
- `team_shape_team_width_attacking`
- `team_shape_stretch_index_attacking`
- `team_shape_n_outfield_players_defending`
- `team_shape_centroid_x_defending`
- `team_shape_centroid_y_defending`
- `team_shape_convex_hull_area_defending`
- `team_shape_team_length_defending`
- `team_shape_team_width_defending`
- `team_shape_stretch_index_defending`

Plus the 4 standard provenance columns (frame_id, time_offset_seconds,
n_candidate_frames, link_quality_score).

### 1.5 VAEP xfn Factory

```python
def team_shape_xfns(home_team_id: int | str) -> list[Callable]:
```

Returns a single batch transformer (closure capturing `home_team_id`) that
emits 12 × 3 = 36 columns (12 features × 3 game state slots, suffixed
`_a0`, `_a1`, `_a2`). Marked `_frame_aware = True`.

**`n_outfield_players` is excluded from the VAEP xfn factory** --- it is a
data-quality indicator, not a tactical feature. Consumers get it from
`add_team_shape` for interpretation but VAEP does not waste parameters on it.

**VAEP introspection mode:** When `frames is None`, returns DataFrame of NaN
columns with correct names (per ADR-005 silent-NaN pattern).

---

## 2. TF-32 --- Ward-Clustering Line-Breaking

### 2.1 Motivation

The current TF-4 `line_break` feature (`_off_ball_runs.py:_line_break_kernel`)
is a simple threshold test: "did action.end_x cross the defending team's
`defensive_line_x`?" — binary yes/no. This is fast and appropriate for VAEP
binary features, but it cannot distinguish:
- How many lines were broken (1 vs 2 vs 3)
- Whether the pass went *through* defenders or *around* them

The Ward-clustering approach identifies actual defensive lines from opponent
positions, constructs line segments, and tests geometric intersection. This is
the standard academic approach (Karakuş & Arkadaş 2025).

**References:**
- Karakuş, O., & Arkadaş, H. (2025). "Through the Gaps: Uncovering Tactical
  Line-Breaking Passes with Clustering." arXiv:2506.06666. ECML/PKDD MLSA 2025.

### 2.2 Module Layout

New private module `silly_kicks/tracking/_line_breaking.py` with public symbols
re-exported from `silly_kicks/tracking/__init__.py`.

The existing `_line_break_kernel` in `_off_ball_runs.py` remains untouched ---
it serves the `method="threshold"` path. The new module provides the
`method="ward"` path.

### 2.3 Algorithm

**Input:** For a given pass action, the opposing team's outfield player
positions at the linked frame.

**Step 1: Filter and validate.**
- Require `min_opponents` (default 3) visible outfield opponents.
- Require `min_x_spread` (default 5.0 m) between outermost opponents on x-axis.
  (If opponents are clustered in a tight band, line-breaking is undefined.)
- Require pass length ≥ `min_pass_length` (default 3.0 m). Short passes cannot
  meaningfully break a line.
- If any check fails → `is_line_breaking=False`, `lines_broken=0`,
  `line_breaking_type=None`.

**Step 2: Ward hierarchical clustering on 1D x-coordinates.**
- `scipy.cluster.hierarchy.linkage(x_positions.reshape(-1, 1), method="ward")`
- `scipy.cluster.hierarchy.fcluster(Z, t=n_clusters, criterion="maxclust")`
  with `n_clusters=3` (attack/midfield/defense lines).
- Sort clusters by ascending mean x-coordinate within the attacking direction.

**Design rationale (1D vs 2D):** The paper (arXiv:2506.06666) explicitly
clusters on x-coordinates only: "opponent player positions are grouped into k
clusters based on their lateral (x-axis) coordinates using agglomerative
clustering." Y-coordinates are computed post-hoc to define each cluster's line
segment span. This is conceptually correct: a "defensive line" is defined by
depth (x-position), not lateral spread. A back-four spread 40m apart laterally
IS one defensive line. With 2D clustering, Ward could split them into separate
clusters based on lateral distance, producing a less accurate tactical model.

**Step 3: Construct line segments per cluster.**
- Within each cluster, sort players by y-coordinate.
- Extend to sidelines: prepend virtual point at
  `(first_player_x, pitch_y_min)`, append virtual point at
  `(last_player_x, pitch_y_max)` --- using the nearest player's x-coordinate
  for each sideline extension, not the cluster mean. This preserves the actual
  line geometry at the edges rather than jumping to a centroid that no defender
  occupies.
- Connect adjacent points (including extensions) to form line segments:
  `segments = [(p_i, p_{i+1}) for i in range(len(points) - 1)]`

**Step 4: Cross-product straddle test.**
- Pass trajectory: segment from `(start_x, start_y)` to `(end_x, end_y)` in
  tracking coordinates (LTR-normalized via `play_left_to_right`).
- For each defensive line segment `(A, B)`, test if pass segment `(C, D)`
  intersects it using the standard cross-product straddle test:
  ```
  d1 = cross(B-A, C-A)
  d2 = cross(B-A, D-A)
  d3 = cross(D-C, A-C)
  d4 = cross(D-C, B-C)
  intersects = (d1 * d2 < 0) and (d3 * d4 < 0)
  ```
- A cluster is "broken" if ANY of its segments is intersected.

**Step 5: Classify type.**
- `lines_broken` = count of distinct clusters whose segments were intersected.
- `is_line_breaking` = `lines_broken > 0`.
- `line_breaking_type` reflects the **dominant type across all broken
  clusters**, not per-cluster:
  - `"between_lines"` if the intersection point is between two actual players
    (not a sideline extension segment).
  - `"around_line"` if the intersection is on a sideline-extension segment (pass
    went wide of the outermost defender).
  - `None` if `is_line_breaking == False`.
  - If multiple clusters are broken with mixed types, classify as `"between_lines"`
    (the more tactically significant event).
- **Known simplification:** This is a pass-level classification. A pass that
  goes "between" one line and "around" another is classified as "between_lines."
  Per-cluster type detail is available internally in `detect_line_breaking`
  and can be surfaced in a future iteration if consumers need it.

### 2.4 Parameters

```python
@dataclass(frozen=True)
class LineBreakingParams:
    min_opponents: int = 3
    n_clusters: int = 3
    min_pass_length: float = 3.0   # metres
    min_x_spread: float = 5.0      # metres
    pitch_y_min: float = 0.0       # SPADL y-coordinate of near sideline
    pitch_y_max: float = 68.0      # SPADL y-coordinate of far sideline
```

### 2.5 Public API

```python
def detect_line_breaking(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params: LineBreakingParams | None = None,
) -> pd.DataFrame:
```

Returns DataFrame aligned with `actions.index`, columns:
- `line_break__ward` (boolean, nullable)
- `lines_broken__ward` (Int64, 0–3)
- `line_breaking_type__ward` (object/str: "between_lines", "around_line", or None)

### 2.6 Integration with Existing `add_line_break`

The existing aggregator gains a `method` parameter:

```python
@nan_safe_enrichment
def add_line_break(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    method: Literal["threshold", "ward"] = "threshold",
    n: int = 4,                              # threshold-specific
    params: LineBreakingParams | None = None, # ward-specific (ignored when method="threshold")
) -> pd.DataFrame:
```

- `method="threshold"` → dispatches to existing `_line_break_kernel` from
  `_off_ball_runs.py`. Returns `line_break`, `n_attackers_behind_line`.
  **Unchanged behavior** (backward-compatible default). `params` is ignored.
- `method="ward"` → dispatches to `detect_line_breaking` from
  `_line_breaking.py`. Returns `line_break__ward`, `lines_broken__ward`,
  `line_breaking_type__ward`. `n` is ignored.

Column sets are **disjoint** between methods (no column name collision). A
consumer can call both methods if they want all 5 columns (note: each call
performs its own `link_actions_to_frames` --- see §1.4 linkage cost note).

### 2.7 VAEP xfn Factory

```python
def line_breaking_ward_xfns(home_team_id: int | str) -> list[Callable]:
```

Returns a single batch transformer. The `line_breaking_type__ward` categorical
is one-hot encoded for VAEP consumption:
- `lines_broken__ward_a{i}` (numeric, 0–3)
- `line_breaking_type__ward_through_a{i}` (bool)
- `line_breaking_type__ward_around_a{i}` (bool)

Total: 3 features × 3 game states = 9 columns.

`line_break__ward` is excluded --- redundant with `lines_broken__ward > 0`;
VAEP should not waste a parameter on a linearly dependent feature.

Marked `_frame_aware = True`. Silent-NaN on introspection.

### 2.8 Why Not Fisher-Jenks

Fisher-Jenks (natural breaks) and Ward linkage on 1D data are functionally
near-equivalent — both minimize within-cluster variance. Ward is chosen because
`scipy.cluster.hierarchy` is already in the dependency tree (no new dep), and
the academic reference (Karakuş & Arkadaş 2025) uses agglomerative clustering.
If empirical evaluation later shows Fisher-Jenks produces meaningfully
different results, it can be added as a third `method` value.

---

## 3. Shared Infrastructure

### 3.1 Public API Registration (`tracking/__init__.py`)

New exports:
- `compute_team_shape` (per-frame primitive)
- `add_team_shape` (action-coupled aggregator)
- `team_shape_xfns` (VAEP factory)
- `detect_line_breaking` (per-action Ward detection)
- `LineBreakingParams` (frozen dataclass)
- `line_breaking_ward_xfns` (VAEP factory)

The existing `add_line_break` export stays (signature gains `method` kwarg,
default unchanged).

### 3.2 NOTICE File

Add entries for:
- Clemente et al. 2013 (stretch index, team shape descriptors)
- Karakuş & Arkadaş 2025 (Ward-clustering line-breaking)

### 3.3 Testing Strategy

| Layer | TF-31 | TF-32 |
|-------|-------|-------|
| **Unit** | Synthetic 11v11 frame with known positions; assert exact metric values (`pytest.approx`) for all 7 metrics including `n_outfield_players` | Synthetic frame with 3 clear defensive lines (back 4, midfield 3, forward 3); pass through all 3 → `lines_broken=3`; pass through 1 → `lines_broken=1`; pass wide of outermost → `line_breaking_type="around_line"` |
| **Invariants** | `convex_hull_area ≥ 0`; `stretch_index ≥ 0`; `team_length ∈ [0, 105]`; `team_width ∈ [0, 68]`; `stretch_index ≤ max(team_length, team_width)`; `n_outfield_players ∈ [1, 11]` | `lines_broken ∈ {0,1,2,3}`; `is_line_breaking == (lines_broken > 0)`; `line_breaking_type ∈ {"between_lines", "around_line", None}` |
| **Edge cases** | (a) 0 players → all NaN; (b) 1 player → centroid = position, length/width/stretch = 0, hull = NaN, n_outfield = 1; (c) 2 players → hull = NaN, rest valid; (d) ≥ 3 collinear → hull = NaN via `QhullError` catch, rest valid | (a) < 3 opponents → all False/0/None; (b) short pass (< 3 m) → False; (c) no x-spread (< 5 m) → False; (d) pass parallel to defensive line (no intersection) → False |
| **Provider sweep** | Sportec, Metrica, PFF, SkillCorner (matching existing widest provider set from `test_action_context_cross_provider.py`) | Same providers, using synthesized pass actions with offset end_x/end_y (see §3.4) |
| **NaN-safety** | `@nan_safe_enrichment` auto-discovered | Same |
| **VAEP introspection** | `frames=None` → NaN DataFrame with 36 correct column names | `frames=None` → NaN DataFrame with 9 correct column names |
| **xfn column count** | `assert len(xfn_output.columns) == 36` | `assert len(xfn_output.columns) == 9` |
| **Backward compat** | N/A (new feature) | Golden-file regression: snapshot `add_line_break()` default output on provider fixtures to `.parquet` golden file pre-PR; assert `pd.testing.assert_frame_equal(actual, expected)` post-PR |
| **Cross-method sanity** | N/A | Soft check on provider fixtures: when `lines_broken__ward > 0`, the threshold `line_break` should also be True in the majority of cases (not a hard invariant — different algorithms — but flags gross disagreement) |

### 3.4 Synthesizer Fixture Amendment

The existing `synthesize_actions()` in `tests/tracking/_provider_inputs.py`
sets `end_x == start_x` and `end_y == start_y` (passes go nowhere). TF-32's
Ward line-breaking needs pass trajectories that actually cross the pitch. For
the TF-32 provider sweep tests, either:

- (a) Amend `synthesize_actions()` so pass actions get `end_x = start_x + Δ`
  (e.g., +20 m forward) and `end_y = start_y + small_offset`, or
- (b) Provide a separate `synthesize_pass_actions_with_trajectory()` helper
  specific to line-breaking tests.

Option (a) is preferred --- it makes the existing synthesizer more realistic
for all consumers, not just TF-32. The offset should be large enough that at
least one synthesized pass crosses the opposing team's defensive line structure.
The existing off-ball-runs / line-break tests should be re-verified after this
change (their assertions are on column presence and non-NaN counts, not exact
values, so they should tolerate the change).

### 3.5 Out of Scope

- Fisher-Jenks as alternative clustering method (add later if empirical
  evidence warrants; see §2.8).

---

## 4. Coordinate System Notes

Both features operate on tracking-frame coordinates (LTR-normalized via
`play_left_to_right`). The SPADL coordinate system is `[0, 105] × [0, 68]`.

- TF-31: purely coordinate-agnostic (just x/y arrays). Works in any frame.
- TF-32: pass trajectory uses `(start_x, start_y) → (end_x, end_y)` from
  SPADL actions (per-action coordinates where the acting team attacks x=105).
  `detect_line_breaking` transforms these to tracking coordinates internally
  for intersection testing against opponent positions from LTR-normalized
  tracking frames.

---

## 5. Dependencies

No new runtime dependencies. Both features use:
- `scipy.spatial.ConvexHull` (TF-31)
- `scipy.cluster.hierarchy.linkage`, `fcluster` (TF-32)
- `numpy`, `pandas` (both)

All already in the dependency tree via existing scipy/numpy/pandas requirements.

---

## Appendix: Lakehouse Review Disposition (2026-05-09)

| # | Severity | Item | Disposition |
|---|----------|------|-------------|
| H1 | High | Cluster on 2D (x,y) not 1D x | **Rejected.** Paper explicitly uses 1D x. Defensive lines are defined by depth, not lateral spread. A back-four spread 40m apart laterally is one line; 2D clustering could incorrectly split it. See §2.3 design rationale. |
| H2 | High | Sideline extension uses nearest-player x | **Accepted.** §2.3 Step 3 updated. |
| H3 | High | Document line_breaking_type is pass-level | **Accepted.** §2.3 Step 5 updated with known-simplification note. |
| M1 | Medium | Clarify compute_team_shape batch return | **Accepted.** §1.3 return shape clarified. |
| M2 | Medium | Document redundant linkage cost | **Accepted.** §1.4 note added. |
| M3 | Medium | Golden-file regression test | **Accepted.** §3.3 backward-compat row updated. |
| M4 | Medium | Rename ward_params → params | **Accepted.** §2.6 signature updated. |
| M5 | Medium | Guard < 3 points before ConvexHull | **Accepted.** §1.3 implementation notes updated. |
| L1 | Low | Add n_outfield_players column | **Accepted** (output only, excluded from VAEP xfns). §1.3 table + §1.4 columns updated. |
| L2 | Low | Prefer option (a) for synthesizer | **Accepted.** Already in §3.4. |
| L3 | Low | Assert exact xfn column counts | **Accepted.** §3.3 xfn-column-count row added. |

### Round 2 (2026-05-09) — PASS

All round 1 findings verified as correctly incorporated. H1 pushback accepted.

| # | Severity | Item | Disposition |
|---|----------|------|-------------|
| L1 | Low | §4: clarify SPADL = raw fixed-orientation, not VAEP per-action flip | **Accepted.** §4 updated. |
| L2 | Low | §2.7: document why `line_break__ward` excluded from xfn factory | **Accepted.** §2.7 note added. |
| L3 | Low | §2.6: cross-reference §1.4 linkage cost when calling both methods | **Accepted.** §2.6 parenthetical added. |
