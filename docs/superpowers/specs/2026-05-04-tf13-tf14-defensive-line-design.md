# PR-S27: TF-13 + TF-14 — Defending-GK Resolution + Defensive-Line Geometry

**Date:** 2026-05-04
**Scope:** silly-kicks 3.4.0
**Branch:** `pr-s27-defensive-line`

---

## 1. Overview

Two bundled features sharing the "defending-team-from-frames" theme:

- **TF-13** (~40 LOC): Frame-based defending-GK identification for all actions (fallback when events-based `defending_gk_player_id` is NaN).
- **TF-14** (~250 LOC): Per-frame defensive-line geometry (6 columns) + action-coupled features for VAEP integration.

**Total estimated LOC:** ~750 (implementation + tests + fixtures).

---

## 2. TF-13 — `defending_gk_from_frames`

### 2.1 Purpose

Resolve the defending team's goalkeeper `player_id` from tracking frames for every action. Standalone composable utility — callers use it for `fillna` on the events-based `defending_gk_player_id` or as a direct lookup.

### 2.2 Module Location

`silly_kicks/tracking/_gk_resolve.py` (new file).
Public re-export via `silly_kicks.tracking.features.defending_gk_from_frames`.

### 2.3 API

```python
def defending_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action defending-GK player_id resolved from tracking frames.

    For each action, links to the nearest frame (within tolerance), finds
    the opposing team's is_goalkeeper=True row, and returns that player_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype
        (object for kloppy/sportec, int64/Int64 for PFF).
        NaN where:
          - action couldn't link to any frame
          - no opposing-team GK found in linked frame
          - action.team_id is NaN
    """
```

### 2.4 Implementation Notes

- Reuses `link_actions_to_frames` internally (same tolerance semantics).
- For each linked (action_id, frame_id): filter frame rows where `is_goalkeeper=True AND team_id != action.team_id AND is_ball=False`.
- If multiple GKs found (substitution overlap): pick the one with lower player_id (deterministic tiebreak).
- Does NOT modify actions — pure Series output.
- Decorated with nothing (not an enrichment helper; it's a Series-returning utility).

### 2.5 Composition Pattern

```python
# Typical usage: fill NaN from events-based resolution
actions = add_pre_shot_gk_context(actions)
gk_from_tracking = defending_gk_from_frames(actions, frames)
actions["defending_gk_player_id"] = actions["defending_gk_player_id"].fillna(gk_from_tracking)
```

---

## 3. TF-14 — Defensive-Line Per-Frame Utility

### 3.1 Purpose

Compute back-line geometry for both teams per frame. Foundational primitive consumed by:
- Action-coupled VAEP features (this PR)
- GKDV stack TF-15..19 (future)
- TF-4 line-break detection (future)
- Pitch control TF-7 (future)

### 3.2 Module Location

`silly_kicks/tracking/_defensive_line.py` (new file).
Public re-export via `silly_kicks.tracking.features.compute_defensive_line`.

### 3.3 API

```python
def compute_defensive_line(
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
    adaptive_max_n: int = 5,
) -> pd.DataFrame:
    """Per-(period_id, frame_id, team_id): 6 back-line geometry columns.

    Computes for BOTH teams. home_team_id determines goal assignment
    (must match the value used in play_left_to_right).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
        Must be LTR-normalized (play_left_to_right applied).
    home_team_id : int | str
        Home team identifier. After LTR normalization:
        - home_team_id defends goal at x=0 (back-line = lowest-x outfield players)
        - other team defends goal at x=105 (back-line = highest-x outfield players)
    n : int | Literal["adaptive"], default 4
        Target back-line player count (3, 4, or 5), clamped to available
        outfield players (minimum 3). Or "adaptive" for x-gap clustering
        auto-detection per (frame, team).
    adaptive_max_n : int, default 5
        Upper bound for adaptive N. Must be in {3, 4, 5}.
        Only used when n="adaptive".

    Returns
    -------
    pd.DataFrame
        Columns: period_id, frame_id, team_id,
        defensive_line_x, back_line_high_x, compactness_x,
        lateral_width, max_lateral_gap, back_n_count.
        One row per (period_id, frame_id, team_id).
        All 6 feature columns are NaN where team has <3 valid outfield players.

    Raises
    ------
    ValueError
        If n is an int outside {3, 4, 5}.
        If adaptive_max_n is outside {3, 4, 5}.
        If frames is missing required columns.
        If frames contain non-"ltr" team_attacking_direction values
        (LTR normalization guard).
    """
```

### 3.4 Column Definitions

| Column | Type | Unit | Definition |
|--------|------|------|------------|
| `defensive_line_x` | float64 | m | Mean x of back-line players (absolute pitch coordinate) |
| `back_line_high_x` | float64 | m | x of the most advanced back-line player (furthest from own goal). Approximates the offside line when the GK is behind the defensive line (typical case); NOT a law-compliant offside computation when a sweeper-keeper is ahead of the line. |
| `compactness_x` | float64 | m | x-spread: `max(x) - min(x)` among back-line players. 0 = perfectly flat |
| `lateral_width` | float64 | m | y-spread: `max(y) - min(y)` among back-line players |
| `max_lateral_gap` | float64 | m | Largest y-gap between y-sorted adjacent back-line players. Captures "holes" |
| `back_n_count` | Int64 | count | Players in back line (3/4/5 for adaptive; equals clamped `n` for fixed). Nullable Int64 — NaN when <3 outfield. |

**NaN semantics:** All 6 columns are NaN when team has <3 valid outfield players in that frame (valid = non-NaN x coordinate).

### 3.5 Algorithm

**Per (period_id, frame_id, team_id):**

1. **Select outfield players:** `is_ball=False AND is_goalkeeper=False AND team_id=T AND x.notna()`
2. **Threshold check:** If fewer than 3 valid outfield players → emit NaN row for all 6 columns.
3. **Goal assignment:**
   - If `team_id == home_team_id`: defends x=0; sort players by x ascending (closest to x=0 first)
   - Else: defends x=105; sort players by x descending (closest to x=105 first)
4. **Player selection:**
   - **Fixed N:** take first `min(N, available_outfield)` sorted players. `n` is a target clamped to available players (floor of 3 enforced by threshold check in step 2).
   - **Adaptive:** see section 3.5.1 below.
5. **Compute 6 columns** from selected players' (x, y) coordinates:
   - `defensive_line_x = mean(selected_x)`
   - `back_line_high_x = max(selected_x)` for home team (furthest from x=0); `min(selected_x)` for away team (furthest from x=105)
   - `compactness_x = max(selected_x) - min(selected_x)`
   - `lateral_width = max(selected_y) - min(selected_y)`
   - `max_lateral_gap = max(diff(sorted(selected_y)))` where diff is between consecutive y-sorted values
   - `back_n_count = len(selected)`

#### 3.5.1 Adaptive Algorithm

When `n="adaptive"`, determine N dynamically from x-gap analysis:

1. Let `P` = number of valid outfield players (already sorted by proximity to own goal).
2. Let `available_cuts` = range of cut-points to examine: indices `[2]→[3]`, `[3]→[4]`, `[4]→[5]` (0-indexed), filtered to those that exist given P. Specifically:
   - Cut at [2]→[3] exists if P >= 4 (splits into 3 vs rest)
   - Cut at [3]→[4] exists if P >= 5 (splits into 4 vs rest)
   - Cut at [4]→[5] exists if P >= 6 (splits into 5 vs rest)
3. Compute `gap[i] = |x[i+1] - x[i]|` for each available cut-point.
4. **Selection rule:**
   - If no cuts are available (P == 3): N = 3.
   - If only one cut exists (P == 4): N = 4. (Single cut-point provides no relative comparison — the "1.5× second max" rule can't apply. Default to N=4 as the most common formation for a 4-outfield-defender group.)
   - If multiple cuts exist: find the cut with the maximum gap. If `max_gap >= 1.5 * second_max_gap` → use the corresponding N (3, 4, or 5). Otherwise (no dominant gap), default to N = 4.
   - **Degenerate case (all gaps == 0.0):** All players at same x; default to N = 4.
5. Clamp final N to [3, min(`adaptive_max_n`, P)].

### 3.6 LTR Normalization Guard

On entry, `compute_defensive_line` checks:
- If `team_attacking_direction` column exists in frames AND contains any value other than `"ltr"` (excluding NaN), raise `ValueError("compute_defensive_line: frames must be LTR-normalized (play_left_to_right). Found non-'ltr' values in team_attacking_direction.")`.

This is a cheap O(1) check (just `series.dropna().unique()`) that catches the most common misuse (forgetting to normalize) without being overly restrictive (NaN direction is tolerated — some frames lack the column entirely).

### 3.7 Edge Cases

| Scenario | Behavior |
|----------|----------|
| <3 valid outfield players (red cards, partial coverage, NaN coords) | All 6 cols = NaN |
| Exactly 3 outfield players, n=4 requested | Use all 3; back_n_count=3 |
| Exactly 4 outfield, n=5 requested | Use all 4; back_n_count=4 |
| Player with NaN x or y | Excluded from selection (not counted toward threshold) |
| Ball rows | Excluded (is_ball filter) |
| GK rows | Excluded (is_goalkeeper filter) |
| Team not present in frame | No row emitted for that (frame, team) |
| All outfield players at identical x (degenerate) | All gaps = 0; adaptive defaults to N=4; compactness_x = 0 |
| n=2 or n=6 as int | ValueError raised |
| adaptive_max_n outside {3,4,5} | ValueError raised |

---

## 4. TF-14 — Action-Coupled Layer

### 4.1 Module Location

Action-coupled xfns and aggregator in `silly_kicks/tracking/features.py`.
Internal batch kernel in `silly_kicks/tracking/_kernels.py`.

### 4.2 Batch Kernel Pattern (addresses redundant-compute concern)

The per-frame `compute_defensive_line` is expensive (iterates all frames × teams). To avoid 18× redundant calls (6 xfns × 3 game-states), the action-coupled layer uses a **single batch kernel** that computes all 6 columns once and returns a multi-column DataFrame:

```python
# In _kernels.py
def _defensive_line_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.DataFrame:
    """All 6 defensive-line columns for the defending team at each action's linked frame.

    Calls compute_defensive_line ONCE on the full frames DataFrame, then
    joins on (period_id, frame_id, opposing_team_id) per action.

    Returns DataFrame aligned with actions.index, columns:
    defensive_line_x, back_line_high_x, compactness_x,
    lateral_width, max_lateral_gap, back_n_count.
    """
```

Each per-Series function (`defensive_line_x`, `back_line_high_x`, etc.) calls this batch kernel and picks one column. Within a single `compute_features` invocation, the xfn factory ensures the batch kernel is called only once (see section 4.4).

### 4.3 Per-Series Functions

Six functions, each returning the **defending team's** (opposing team's) value at the linked frame:

```python
def defensive_line_x(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
def back_line_high_x(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
def compactness_x(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
def lateral_width(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
def max_lateral_gap(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
def back_n_count(actions, frames, *, home_team_id, n=4) -> pd.Series: ...
```

All return NaN where action is unlinked or defending team has insufficient players.

### 4.4 VAEP xfn Integration — Single Multi-Column Transformer

To avoid both the `functools.partial` `__name__` crash (issue #1) and the 18× redundant compute (issue #2), the factory returns a **single `FrameAwareTransformer` that emits all 6 columns at once**:

```python
def defensive_line_xfns(
    home_team_id: int | str,
    *,
    n: int | Literal["adaptive"] = 4,
) -> list[FrameAwareTransformer]:
    """Build VAEP xfn list bound to a specific home_team_id.

    Returns a list with ONE element: a single FrameAwareTransformer that
    emits 6 × nb_states columns (e.g., defensive_line_x_a0, ..._a2,
    back_line_high_x_a0, ...). This ensures compute_defensive_line is
    called only once per game-state slot (3 total), not 18 times.

    The returned transformer is a named closure (not functools.partial),
    so it has a proper __name__ attribute compatible with lift_to_states
    and VAEP feature introspection.

    Usage in VAEP pipeline:
        xfns = tracking_default_xfns + defensive_line_xfns("team_A")
        X = compute_features(actions, xfns=xfns, frames=frames)
    """
    def _defensive_line_transformer(states, frames):
        \"\"\"Multi-column defensive-line xfn (6 cols × nb_states).\"\"\"
        out = pd.DataFrame(index=states[0].index)
        col_names = [
            "defensive_line_x", "back_line_high_x", "compactness_x",
            "lateral_width", "max_lateral_gap", "back_n_count",
        ]
        for i, slot in enumerate(states[:3]):
            batch = _defensive_line_at_actions(slot, frames, home_team_id=home_team_id, n=n)
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _defensive_line_transformer._frame_aware = True
    _defensive_line_transformer.__name__ = "defensive_line"
    return [_defensive_line_transformer]
```

**Key design choices:**
- Named closure (not `functools.partial`) → has `__name__`, no `AttributeError`.
- Single transformer emitting all 6 × 3 = 18 columns → `compute_defensive_line` called 3× total (once per game-state), not 18×.
- Factory returns `list[FrameAwareTransformer]` for API consistency with other `*_default_xfns` lists (consumers always do `xfns + defensive_line_xfns(...)`).

### 4.5 Aggregator

```python
@nan_safe_enrichment
def add_defensive_line(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
) -> pd.DataFrame:
    """Enrich actions with 6 defensive-line columns + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions with columns:
        - defensive_line_x (float64, m)
        - back_line_high_x (float64, m)
        - compactness_x (float64, m)
        - lateral_width (float64, m)
        - max_lateral_gap (float64, m)
        - back_n_count (Int64)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)

    Provenance column collision handling:
        If frame_id / time_offset_seconds / link_quality_score /
        n_candidate_frames already exist on the input actions DataFrame
        (from a prior add_action_context or add_pre_shot_gk_position call),
        they are SKIPPED (not re-added). The values are identical because
        all enrichments use the same link_actions_to_frames with the same
        tolerance. An assertion verifies value equality in debug mode
        (AssertionError if they somehow differ — pipeline integrity issue).
    """
```

---

## 5. File Layout

```
silly_kicks/tracking/
  _gk_resolve.py              NEW  (~40 LOC)
  _defensive_line.py           NEW  (~140 LOC)
  features.py                  EXTEND (+120 LOC: 6 per-series + aggregator + factory)
  _kernels.py                  EXTEND (+40 LOC: batch action-kernel)
  __init__.py                  EXTEND (re-exports)

tests/tracking/
  test_gk_resolve.py           NEW  (~80 LOC)
  test_defensive_line.py       NEW  (~180 LOC, per-frame kernel)
  test_defensive_line_features.py  NEW  (~100 LOC, action-coupled)
  _provider_inputs.py          EXTEND (+60 LOC: formation fixtures)

tests/invariants/
  test_invariant_defensive_line.py   NEW  (~70 LOC)
  test_invariant_gk_resolve.py       NEW  (~30 LOC)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests — Per-Frame Kernel (`test_defensive_line.py`)

| Test | Validates |
|------|-----------|
| `test_fixed_n4_basic` | 4 defenders in known positions → exact 6-col values |
| `test_fixed_n3_n5` | 3-at-the-back and 5-at-the-back formations |
| `test_adaptive_detects_4_back` | Clear 4v1 x-gap → adaptive picks N=4 |
| `test_adaptive_detects_5_back` | 5 clustered + clear gap before 6th → N=5 |
| `test_adaptive_detects_3_back` | 3 clustered + clear gap before 4th → N=3 |
| `test_adaptive_no_dominant_gap` | All gaps within 20% → defaults to N=4 |
| `test_adaptive_all_same_x` | Degenerate: all x identical → defaults to N=4 |
| `test_adaptive_exactly_3_outfield` | Only 3 players available → N=3 (no cuts to examine) |
| `test_adaptive_exactly_4_outfield` | Only cut [2]→[3] available → defaults to N=4 |
| `test_fewer_than_3_outfield_nan` | 2 outfield players → NaN row |
| `test_fixed_n_clamped_to_available` | n=4 but only 3 outfield → back_n_count=3 |
| `test_gk_excluded_from_line` | GK at x=2 doesn't appear in back-line |
| `test_both_teams_computed` | Output has rows for both teams per frame |
| `test_home_team_defends_x0` | Home back-line players selected from low-x |
| `test_away_team_defends_x105` | Away back-line players selected from high-x |
| `test_nan_coordinates_excluded` | Player with NaN x not counted toward threshold |
| `test_empty_frames` | Empty input → empty output |
| `test_multi_period_isolation` | Period boundary doesn't bleed |
| `test_invalid_n_raises` | n=2 or n=6 → ValueError |
| `test_invalid_adaptive_max_n_raises` | adaptive_max_n=10 → ValueError |
| `test_ltr_guard_raises` | Frames with "rtl" direction → ValueError |
| `test_ltr_guard_allows_nan` | Frames with NaN direction → no error |

### 6.2 Unit Tests — TF-13 (`test_gk_resolve.py`)

| Test | Validates |
|------|-----------|
| `test_resolves_opposing_gk` | Action team=A → returns team=B's GK player_id |
| `test_all_actions_not_just_shots` | Non-shot actions get resolution |
| `test_nan_when_no_gk_in_frame` | No is_goalkeeper=True on opposing team → NaN |
| `test_nan_when_unlinked` | No frame within tolerance → NaN |
| `test_nan_when_team_id_nan` | NaN team_id on action → NaN |
| `test_dtype_matches_frames_object` | Kloppy/Sportec object player_id → object output |
| `test_dtype_matches_frames_int64` | PFF int64 player_id → int64 output |
| `test_multi_gk_deterministic` | Two GKs in frame → lowest player_id picked |
| `test_tolerance_respected` | Frame beyond tolerance → NaN |

### 6.3 Unit Tests — Action-Coupled (`test_defensive_line_features.py`)

| Test | Validates |
|------|-----------|
| `test_action_gets_opposing_team_line` | Correct team's geometry returned |
| `test_unlinked_action_nan` | NaN for all 6 cols |
| `test_aggregator_column_count` | 6 feature + 4 provenance = 10 new cols |
| `test_aggregator_provenance_skip_if_exists` | Pre-existing provenance cols not duplicated |
| `test_nan_safe_enrichment` | NaN team_id → NaN output, no crash |
| `test_xfns_factory_produces_valid` | Factory returns list with 1 FrameAwareTransformer |
| `test_xfns_factory_has_name` | Transformer has `__name__ == "defensive_line"` |
| `test_xfns_in_compute_features` | End-to-end VAEP feature compute works |
| `test_xfns_column_count` | Factory transformer emits 6 × 3 = 18 columns |
| `test_batch_kernel_called_once` | Verify compute_defensive_line not called 18× (mock) |

### 6.4 Invariant Tests

**`test_invariant_defensive_line.py`:**
- `defensive_line_x ∈ [0, 105]` when not NaN
- `back_line_high_x ∈ [0, 105]` when not NaN
- Home team: `back_line_high_x >= defensive_line_x` (max >= mean, always true — but also: `back_line_high_x - defensive_line_x <= compactness_x`)
- Away team: `back_line_high_x <= defensive_line_x` (min <= mean — and same triangle inequality)
- `compactness_x >= 0`
- `lateral_width ∈ [0, 68]`
- `max_lateral_gap >= 0` and `max_lateral_gap <= lateral_width`
- `back_n_count ∈ {3, 4, 5}` when not NaN
- **Cross-team sanity (non-trivial):** For frames where both teams have valid lines: `|home_defensive_line_x + (105 - away_defensive_line_x)| / 105 < 1.0` — i.e., it's not the case that both teams' lines are near the same goal (one team would be in the other's half at minimum).
- **Triangle inequality:** `back_line_high_x - defensive_line_x <= compactness_x` (for home team; mirrored for away)

**`test_invariant_gk_resolve.py`:**
- Resolved player_id exists in frames with `is_goalkeeper=True`
- Resolved player_id belongs to opposing team (never action's own team)

### 6.5 Provider Integration Tests

Expand `tests/tracking/_provider_inputs.py` to synthesize frames with known defensive formations per provider (PFF, Sportec, Metrica, SkillCorner):

| Fixture requirement | Purpose |
|---------------------|---------|
| 4-at-the-back formation (known positions) | Validate fixed N=4 exact output |
| 5-at-the-back formation (clear clustering gap) | Validate adaptive N=5 detection |
| 3-at-the-back formation | Validate adaptive N=3 detection |
| Red-card scenario (2 outfield defenders) | Validate NaN threshold |
| Both teams with `is_goalkeeper=True` | Validate TF-13 per provider |
| Both teams with 4+ outfield players | Validate both-teams compute |
| Frame with NaN coordinates on some players | Validate exclusion logic |

Parametrize across providers; assert:
- Column dtypes match provider schema (object vs int64/Int64)
- No crashes on provider-specific edge cases
- Physical plausibility of all outputs

### 6.6 Golden-Master Expected-Output Test

One deterministic synthetic fixture with hand-computed expected values for all 6 columns + TF-13 output. Pin exact float values (4 decimal places) to catch numerical regression. Include:
- A 4-back home team at known x positions
- A 4-back away team at known x positions (mirrored)
- Expected `defensive_line_x`, `back_line_high_x`, `compactness_x`, `lateral_width`, `max_lateral_gap`, `back_n_count` for each

---

## 7. Academic Attribution (NOTICE)

```
## Defensive Line Geometry (TF-14)

Hernandez-Rodriguez, A., et al. (2025). "Prediction-based evaluation of
    back-four defense with spatial control in soccer." arXiv:2511.06191.
    (Fixed N=4 selection; defensive line height absolute/relative;
    stretch index; minimum-4-defenders threshold.)

Nakashima, Y., et al. (2025). "Analysis of Line Break prediction models for
    detecting defensive breakthrough in football." arXiv:2511.00121.
    (Second-last-defender offside-line definition; line-break detection.)

FIFA Enhanced Football Intelligence (2022). "Defensive line height and team
    length." FIFA Training Centre, World Cup 2022.
    (Deepest-outfield-player definition; GK exclusion convention.)

Forcher, L., Altmann, S., Forcher, L., Jekauc, D., & Kempe, M. (2022).
    "The use of player tracking data to analyze defensive play in
    professional soccer - A scoping review." International Journal of
    Sports Science & Coaching, 17(6), 1567-1592.
    (Survey of defensive tracking metrics; line height definition.)
```

---

## 8. Non-Goals / Deferred

- **Line-break detection** (TF-4): per-action coupled feature using defensive line + attacker positions. Separate PR.
- **Adaptive algorithm tuning against real data**: Ship N=4 default first; adaptive empirical validation in a follow-up if needed.
- **`add_pre_shot_gk_context` auto-fill integration**: Callers compose manually via `fillna(defending_gk_from_frames(...))`. Auto-integration deferred to avoid scope creep.
- **Per-period direction inference without `home_team_id`**: Not implemented. Explicit parameter required.
- **Law-compliant offside line**: Would require knowing GK position relative to line (sweeper-keeper case). `back_line_high_x` is an approximation. Full offside computation deferred.

---

## 9. Dependencies

- No new runtime dependencies.
- Consumes existing `is_goalkeeper` column (PR-S26, guaranteed reliable cross-provider).
- Consumes existing `link_actions_to_frames` / `_resolve_action_frame_context`.
- Consumes existing `play_left_to_right` contract (LTR normalization prerequisite).

---

## 10. Versioning

- **silly-kicks 3.4.0** (minor: new public API surface, no breaking changes).
- CHANGELOG section: `### Added` with exact column names per consumer-contract handshake pattern.

---

## Appendix A: Lakehouse Review Resolution Log

Issues raised in 2026-05-04 lakehouse session review, with resolutions:

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | `functools.partial` crashes `lift_to_states` (`__name__` absent) | Critical | Factory uses named closure, not partial. §4.4 revised. |
| 2 | 18× redundant `compute_defensive_line` via per-Series xfns | Critical | Single multi-column transformer; batch kernel called 3× total. §4.2/4.4 revised. |
| 3 | `back_n_count` int64 cannot hold NaN | Critical | Changed to Int64 (nullable). §3.4 revised. |
| 4 | Provenance column collision with other `add_*` enrichments | Critical | Skip-if-exists with debug-mode equality assertion. §4.5 revised. |
| 5 | Adaptive algorithm under-specified (can't detect 5-back; degenerate cases) | Medium | Full cut-point enumeration in §3.5.1; degenerate cases explicit. |
| 6 | `offside_x` naming implies law compliance | Medium | Renamed to `back_line_high_x` with explicit doc caveat. §3.4 revised. |
| 7 | No LTR normalization guard | Minor | Added §3.6: cheap check on `team_attacking_direction`. |
| 8 | Trivial invariant tests | Minor | Added cross-team sanity + triangle inequality invariants. §6.4 revised. |
| 9 | `adaptive_max_n` lacks validation | Minor | ValueError for values outside {3,4,5}. §3.3/3.7 revised. |
| 10 | Adaptive P=4 rule always selects N=3 (gap > 0 is universally true) | Medium | Default to N=4 for P=4 (single cut-point, no relative comparison possible). §3.5.1 step 4 revised. |
