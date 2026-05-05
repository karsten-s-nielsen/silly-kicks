# TF-4: Off-Ball Runs + Line-Break Detection — Design Spec

**Date:** 2026-05-05
**PR:** PR-S30
**Branch:** `pr-s30-tf4-off-ball-runs`
**Status:** Approved (rev 2 — post cross-session review)
**Size:** ~250–350 LOC + ~400 LOC tests

---

## 1. Overview

TF-4 bundles two sibling action-coupled tracking features that share the same
`(action x attacker positions x defending-team line)` shape:

1. **Off-ball runs** — per-attacking-teammate temporal analysis in the
   pre-action window.
2. **Line-break detection** — per-action x line-geometry coupling: is the
   action destination behind the defending back-line, and how many attackers
   are positioned there?

Bundling avoids duplicate plumbing. Both features consume the same tracking
frames and defensive-line geometry; line-break reuses `compute_defensive_line`
(TF-14, shipped in PR-S27).

## 2. References

- Spearman, W. (2018). "Beyond Expected Goals." *MIT Sloan SAC.* — Originator
  of the OBSO (Off-Ball Scoring Opportunity) framework; the off-ball-runs and
  line-break concepts implemented here are inspired by this framework.
- Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not All Passes Are
  Created Equal: Objectively Measuring the Risk and Reward of Passes in Soccer
  from Tracking Data." *KDD '17.* — Contextual passing risk/reward; §4 briefly
  mentions line-breaking passes as a qualitative formation feature.

**Note:** The off-ball-runs algorithm and line-break detection kernel are novel
implementations inspired by the general OBSO framework (Spearman 2018), not
direct reproductions of any specific paper section.

See NOTICE for full bibliographic citations.

## 3. Output Columns

Six columns total, split across two kernels:

### Off-ball runs (4 columns)

| Column | Type | Description |
|--------|------|-------------|
| `n_off_ball_runners_pre_window` | Int64 | Count of attacking teammates with displacement >= `min_displacement_m` in the pre-action window |
| `max_off_ball_run_displacement_pre_window` | float64 | Max displacement among qualifying runners (NaN if none) |
| `mean_off_ball_run_speed_pre_window` | float64 | Mean speed (displacement / `pre_seconds`) of qualifying runners (NaN if none) |
| `n_off_ball_runners_toward_goal_pre_window` | Int64 | Count of qualifying runners moving toward attacking goal (direction resolved via home_team_id) |

### Line-break (2 columns)

| Column | Type | Description |
|--------|------|-------------|
| `line_break` | boolean | Action destination crosses opposing team's `defensive_line_x` (True/False/pd.NA) — nullable boolean |
| `n_attackers_behind_line` | Int64 | Count of attacking teammates positioned beyond opposing defensive line at action time |

## 4. Parameters

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| `home_team_id` | (required) | Both | Coordinate-frame orientation — determines which direction is "toward goal" for each team and which team's defensive line to evaluate |
| `pre_seconds` | 1.5 | Off-ball runs | Observation window before action timestamp |
| `min_displacement_m` | 3.0 | Off-ball runs | Minimum displacement to qualify as a runner |
| `n` | 4 | Line-break | Back-line player count (passed through to `compute_defensive_line`) |

`pre_seconds` and `min_displacement_m` are engineering choices. Both are
parameterized and included in TF-24's Optuna calibration scope for empirical
grounding.

## 5. Kernel Architecture

### Coordinate-Frame Resolution (Critical)

Two different LTR conventions are in play:

- **SPADL play_left_to_right:** normalizes so the *action's team* attacks
  toward x=105 (per-action flip).
- **Tracking play_left_to_right:** normalizes so the *home team* attacks
  toward x=105 (per-period flip).

For **home-team** actions/players: both frames agree — x=105 is the
attacking direction, defensive_line_x is comparable to SPADL end_x directly.

For **away-team** actions/players: SPADL end_x and tracking x are in
*opposite* coordinate frames. Resolution rule:

```
spadl_def_line_x = defensive_line_x       if action_team == home_team_id
                 = 105 - defensive_line_x  if action_team != home_team_id
```

Similarly for attacker positions and toward-goal direction:
- Home-team players: toward goal = positive Δx in tracking frames.
- Away-team players: toward goal = negative Δx in tracking frames.

Both kernels receive `home_team_id` and each action's `team_id` to perform
the correct coordinate resolution.

### `_off_ball_runs_kernel(actions, frames, *, home_team_id, pre_seconds=1.5, min_displacement_m=3.0)`

**Precondition:** Frames must be LTR-normalized (home team attacks toward
x=105). Kernel raises `ValueError` if frames fail the
`team_attacking_direction` validation check (same guard used by
`compute_defensive_line`).

**Algorithm:**

1. Partition by `game_id` to prevent period_id collisions across games.
   (`slice_around_event` merges on `period_id` only — multi-game DataFrames
   would cross-contaminate without partitioning.)
2. `slice_around_event(actions, frames, pre_seconds=pre_seconds, post_seconds=0.0)`
   to get windowed frames per action.
3. For each action, identify attacking teammates: same `team_id` as actor,
   excluding actor's `player_id`, excluding ball rows.
4. Per teammate: compute displacement between position at first alive frame
   and last alive frame in the window:
   `sqrt((x_end - x_start)^2 + (y_end - y_start)^2)`.
5. Runner qualifies if `displacement >= min_displacement_m`.
6. Toward-goal filter: direction depends on team relative to home_team_id.
   - If runner's team == home_team_id: toward goal = `x_end - x_start > 0`.
   - If runner's team != home_team_id: toward goal = `x_end - x_start < 0`.
7. Aggregate per action: count, max displacement, mean speed, toward-goal count.

**Edge cases:**

- Dead-ball at action's linked frame (window-end) -> entire action NaN.
- Dead-ball frames *within* the pre-window (but alive at action time) ->
  excluded from displacement calculation; displacement computed from
  first/last alive frame per teammate in the window.
- No teammates in window -> 0 runners, NaN for max/mean.
- Single alive frame (no start/end pair for displacement) -> NaN.
- Missing `vx`/`vy` -> not needed; displacement is positional, not velocity-based.

### `_line_break_kernel(actions, frames, *, home_team_id, n=4)`

**Algorithm:**

1. `link_actions_to_frames()` for 1:1 nearest-frame linkage.
2. `compute_defensive_line(frames, home_team_id=home_team_id, n=n)` once on
   full frames.
3. Join defensive-line geometry to each action's linked frame, filtering to
   the **opposing team's** defensive line.
4. Convert defensive_line_x to the action-team's SPADL coordinate frame:
   - If `action_team == home_team_id`: `spadl_def_line_x = defensive_line_x`.
   - If `action_team != home_team_id`: `spadl_def_line_x = 105 - defensive_line_x`.
5. `line_break = True` when `end_x > spadl_def_line_x` (both now in the same
   coordinate frame where the action's team attacks toward x=105).
6. Count attacking teammates positioned beyond the opposing defensive line:
   - For home-team actions: teammates with `tracking_x > defensive_line_x`.
   - For away-team actions: teammates with `tracking_x < defensive_line_x`.
   (Both correctly identify "beyond" the opposing back-line in tracking coords.)

**Edge cases:**

- No defensive line computable (< 3 outfield opponents) -> `line_break` = `pd.NA`, `n_attackers_behind_line` = `pd.NA`.
- No linked frame -> `line_break` = `pd.NA`, `n_attackers_behind_line` = `pd.NA`.

## 6. Public API Surface

### Standalone Aggregators

```python
def add_off_ball_runs(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Add 4 off-ball-run columns to actions."""

def add_line_break(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
) -> pd.DataFrame:
    """Add 2 line-break columns to actions."""

def add_off_ball_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> pd.DataFrame:
    """Umbrella: add all 6 off-ball-run + line-break columns."""
```

All three aggregators decorated with `@nan_safe_enrichment`.

**Linkage provenance:** `add_line_break` uses `link_actions_to_frames`
internally but does **not** emit provenance columns (frame_id,
time_offset_seconds, etc.). Callers who need provenance should call
`add_defensive_line` or `add_action_context` first — those append provenance
with the standard skip-if-present guard. Avoids redundant provenance when
aggregators are chained.

### VAEP Factory

```python
def off_ball_context_xfns(
    home_team_id: int | str,
    *,
    n: int = 4,
    pre_seconds: float = 1.5,
    min_displacement_m: float = 3.0,
) -> list[Callable]:
    """VAEP feature transformer factory.

    Returns one _frame_aware transformer producing 18 columns
    (6 features x 3 action slots a0/a1/a2).
    """
```

Follows `defensive_line_xfns(home_team_id)` factory pattern: binds parameters
into closure, iterates `states[:3]`, column naming `{col}_a{i}`.

### No Per-Series Helpers

Per-Series helpers are **not provided** for TF-4 features. Rationale:

1. Off-ball runs produces 4 columns; line-break produces 2 columns.
   `lift_to_states` expects `helper(slot, frames) -> pd.Series` (one column).
   Multi-column features cannot be lifted through per-Series helpers.
2. The line-break kernel requires `home_team_id` which cannot be passed through
   `lift_to_states`'s `(actions, frames)` signature.
3. Precedent: `defensive_line_xfns` (also multi-column, also needs
   `home_team_id`) uses factory-only, no per-Series helpers.

Users access these features via:
- `add_off_ball_runs()` / `add_line_break()` / `add_off_ball_context()` for
  standalone use.
- `off_ball_context_xfns(home_team_id)` for VAEP pipeline integration.

### Module Layout

- Kernel code -> `silly_kicks/tracking/_off_ball_runs.py` (new file)
- Aggregators + xfn factory -> `silly_kicks/tracking/features.py` (appended)
- Public re-exports via `__init__.py` as needed.

Follows TF-5 (`_ball_carrier.py`) and TF-14 (`_defensive_line.py`) precedent
for standalone kernel modules.

## 7. Dependencies

**Internal (already shipped):**

- `slice_around_event()` from `tracking.utils` (PR-S20)
- `link_actions_to_frames()` from `tracking.utils` (PR-S20)
- `compute_defensive_line()` from `tracking._defensive_line` (PR-S27, TF-14)
- `@nan_safe_enrichment` from `_nan_safety` (PR-S17)
- `derive_goalkeepers()` from `tracking._gk_identification` (PR-S26, TF-13) —
  required: `compute_defensive_line` validates that frames contain
  `is_goalkeeper` column (line 73 of `_defensive_line.py`). Callers must ensure
  `derive_goalkeepers` has been run on frames before passing to line-break.

**No new external dependencies.**

## 8. Testing Strategy

### Layer 1: Unit tests — synthetic DataFrames (`test_off_ball_runs.py`)

| Test | Validates |
|------|-----------|
| `test_off_ball_runs_basic` | 3 attackers, 2 qualify -> correct counts, max, mean |
| `test_off_ball_runs_actor_excluded` | Actor's movement not counted |
| `test_off_ball_runs_opponent_excluded` | Opposing team excluded |
| `test_off_ball_runs_below_threshold` | All < threshold -> 0 runners, NaN max/mean |
| `test_off_ball_runs_toward_goal_home` | Home-team runners: positive Δx = toward goal |
| `test_off_ball_runs_toward_goal_away` | Away-team runners: negative Δx = toward goal |
| `test_off_ball_runs_dead_ball` | Dead ball -> NaN |
| `test_off_ball_runs_no_teammates` | Lone attacker -> 0 runners |
| `test_off_ball_runs_single_frame` | One frame only -> NaN |
| `test_off_ball_runs_custom_params` | Non-default params respected |
| `test_line_break_crosses_line_home` | Home-team action: end_x past line -> True |
| `test_line_break_crosses_line_away` | Away-team action: coordinate flip correct |
| `test_line_break_does_not_cross` | end_x short -> False |
| `test_line_break_no_defensive_line` | < 3 opponents -> pd.NA |
| `test_n_attackers_behind_line_home` | Home-team: count attackers with x > line |
| `test_n_attackers_behind_line_away` | Away-team: count attackers with x < line |
| `test_line_break_no_linked_frame` | No linked frame -> pd.NA |
| `test_empty_frames` | Empty -> empty with correct columns |

### Layer 2: Aggregator integration (`test_off_ball_runs.py`)

| Test | Validates |
|------|-----------|
| `test_add_off_ball_runs_columns` | 4 columns, correct dtypes |
| `test_add_line_break_columns` | 2 columns, correct dtypes (boolean + Int64) |
| `test_add_off_ball_context_columns` | Umbrella adds all 6 |
| `test_off_ball_context_xfns_shape` | Factory -> 1 transformer -> 18 columns |
| `test_xfns_frame_aware_marker` | `_frame_aware = True` |

### Layer 3: VAEP pipeline integration (`test_off_ball_runs.py`)

| Test | Validates |
|------|-----------|
| `test_vaep_introspection` | 10-row dummy gamestate -> NaN columns, no crash |

### Layer 4: Provider fixture tests (`test_off_ball_runs_providers.py`)

Parametrized over Sportec, Metrica, PFF (+ SkillCorner via kloppy if fixture
exists). Uses existing slim-parquet fixtures from `tests/tracking/_provider_inputs.py`.

| Test | Validates |
|------|-----------|
| `test_off_ball_runs_provider_{provider}` | Non-crash, correct columns, >=1 non-NaN row |
| `test_line_break_provider_{provider}` | Non-crash, correct columns, >=1 valid row |
| `test_off_ball_context_provider_{provider}` | Umbrella -> all 6 columns |

**Fixture regeneration:** If existing fixtures lack teammate density for
meaningful coverage, regenerate via `_provider_inputs.py` synthesizer — ensure
>=2 attacking teammates per frame with non-zero displacement, and >=1 action
whose `end_x` crosses the defensive line.

### Layer 5: Invariant tests (`test_off_ball_runs_invariants.py`)

| Invariant | Validates |
|-----------|-----------|
| `n_off_ball_runners >= 0` | Non-negative count |
| `n_off_ball_runners >= n_toward_goal` | Subset relationship |
| `max_displacement >= min_displacement_m` when runners > 0 | Threshold respected |
| `mean_speed >= 0` when not NaN | Non-negative speed |
| `n_attackers_behind_line >= 0` | Non-negative count |
| `line_break=True -> end_x past spadl_def_line_x` | Definition consistency (coordinate-resolved) |

### NaN-safety

Aggregators (`add_off_ball_runs`, `add_line_break`, `add_off_ball_context`)
decorated `@nan_safe_enrichment` -> auto-discovered by
`tests/test_enrichment_nan_safety.py`.

### CI

All tests run in regular suite (`-m "not e2e"`). No external data dependencies.

## 9. NOTICE & TODO Updates

### NOTICE

Add to "Mathematical / Methodological References":

- Spearman, W. (2018). "Beyond Expected Goals." *MIT Sloan SAC.* — OBSO
  framework; off-ball-runs and line-break concepts.
- Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not All Passes Are
  Created Equal." *KDD '17.* — Contextual passing risk/reward; line-breaking
  pass concept (§4 formation clustering).

### TODO.md

- Delete TF-4 row from "On Deck" (ships in this PR; CHANGELOG is the
  per-release record).
- Update TF-24 notes to include: `plus TF-4's off_ball_runs parameters
  (pre_seconds, min_displacement_m)`.
- Bump header date and current-release version.

## 10. Consumer Handshake

### CHANGELOG entry (Added)

Six new columns:
- `n_off_ball_runners_pre_window`
- `max_off_ball_run_displacement_pre_window`
- `mean_off_ball_run_speed_pre_window`
- `n_off_ball_runners_toward_goal_pre_window`
- `line_break`
- `n_attackers_behind_line`

Three new aggregators: `add_off_ball_runs`, `add_line_break`, `add_off_ball_context`.

One new VAEP factory: `off_ball_context_xfns(home_team_id)`.
