# TF-28 + TF-29: DAS Adapter & VAEP Design-Space Variants

**Date:** 2026-05-06
**Target release:** silly-kicks 3.8.0
**Branch:** `pr-s32-das-vaep-variants` (two commits: TF-28, then TF-29)

---

## 1. TF-28 --- DAS Adapter

### 1.1 Motivation

Dangerous Accessible Space (DAS) is a physics-based possession value model
that simulates pass completion maps using ball/player motion kinematics, then
integrates valued completion maps into team-level and per-player "accessible
open space" and "dangerous accessible space" metrics. Published peer-reviewed
in Journal of Big Data 2026 (Bischofberger & Baca); validated on OJN-EPV
benchmark (74% accuracy, 80.4% excluding ball height tests); fitted on only 3
matches of open data.

Key advantage over learning-based EPV: generalizes to hypotheticals (move a
player 5 m and trust the output --- physics-based, local features only). This
makes DAS a natural complement to pitch control (TF-7) and a prerequisite for
GKDV counterfactuals (TF-15/TF-19).

The `accessible-space` PyPI package (MIT, v2.0.15) provides the full
implementation. silly-kicks provides a thin adapter mapping the 20-column
tracking schema to the library's API.

### 1.2 Module Layout

Single private module `silly_kicks/tracking/_das.py` with public symbols
re-exported from `silly_kicks/tracking/__init__.py`.

If the module grows (e.g., DAS counterfactuals for GKDV), promotion to a
`tracking/das/` subpackage is possible without breaking the public API ---
the re-exports in `tracking/__init__.py` stay identical.

### 1.3 Coordinate Transform

silly-kicks tracking schema uses `[0, 105] x [0, 68]`.
`accessible-space` uses `[-52.5, 52.5] x [-34, 34]`.

Private helper:
- `_to_das_coords(frames: pd.DataFrame) -> pd.DataFrame` --- shift `x - 52.5`,
  `y - 34.0`. Velocity columns (`vx`, `vy`) are unaffected (same units, same
  direction). Returns a copy; does not mutate the input.

No `_from_das_coords` inverse --- none of the three public functions return
coordinates (AS/DAS are scalar areas, xC is a probability). Add the inverse
if a future function needs it.

### 1.4 Public API --- Three Wrapper Functions

All three accept `**kwargs` passthrough to the underlying library (simulation
params like `n_angles`, `n_v0`, `chunk_size`, etc.) so users can tune without
silly-kicks needing to mirror every parameter.

Each wrapper unpacks the library's named-tuple return value and discards the
internal bookkeeping fields:

- `get_dangerous_accessible_space` returns `ReturnValueDAS(acc_space, das,
  frame_index, player_index, simulation_result, dangerous_result)` --- 6
  fields; wrapper uses `acc_space` and `das` only.
- `get_individual_dangerous_accessible_space` returns
  `ReturnValueIndividualDAS(acc_space, das, player_acc_space, player_das,
  frame_index, player_index, simulation_result, dangerous_result)` --- 8
  fields; wrapper uses `player_acc_space` and `player_das`.
- `get_expected_pass_completion` returns `ReturnValueXC(xc,
  event_frame_index, tracking_frame_index, tracking_player_index,
  simulation_result)` --- 5 fields; wrapper uses `xc` only.

**Note:** Named-tuple types and field names verified against
`accessible-space` v2.0.15. These are internal to the library and not part
of a documented stable API. The `<3` upper bound (§1.8) limits exposure;
any version bump within `>=2.0,<3` that renames fields will be caught by
the unit tests in §3.1 before reaching production.

#### `get_das(frames, *, use_progress_bar=False, **kwargs) -> pd.DataFrame`

Calls `accessible_space.get_dangerous_accessible_space`.

Returns a DataFrame indexed to match input frames with columns:
- `AS` (float64) --- accessible space per (frame, team).
- `DAS` (float64) --- dangerous accessible space per (frame, team).

#### `get_individual_das(frames, *, use_progress_bar=False, **kwargs) -> pd.DataFrame`

Calls `accessible_space.get_individual_dangerous_accessible_space`.

Returns a DataFrame with per-player columns:
- `AS` (float64) --- per-player accessible space.
- `DAS` (float64) --- per-player dangerous accessible space.

#### `get_xc(passes, frames, *, **kwargs) -> pd.DataFrame`

Calls `accessible_space.get_expected_pass_completion`.

`passes` is a SPADL actions DataFrame filtered to passes; end coordinates come
from `end_x` / `end_y`. Returns a DataFrame with:
- `xC` (float64) --- expected pass completion probability per pass.

### 1.5 Input Validation and Column Mapping

Internal `_build_column_map()` maps silly-kicks tracking schema columns to
`accessible-space` parameter names:

| silly-kicks column | accessible-space param |
|--------------------|----------------------|
| `x` | `x_col` |
| `y` | `y_col` |
| `vx` | `vx_col` |
| `vy` | `vy_col` |
| `player_id` | `player_col` |
| `team_id` | `team_col` |
| `frame_id` | `frame_col` |
| `period_id` | `period_col` |
| `team_in_possession` | `team_in_possession_col` |

**Required column validation** --- the adapter validates upfront (before
calling into `accessible-space`) and raises with actionable messages:

1. **`vx` / `vy`:** If missing, raise:
   ```
   ValueError("DAS requires velocity columns ('vx', 'vy'). "
              "Call derive_velocities() or smooth_frames() first.")
   ```

2. **`team_in_possession`:** If missing, raise:
   ```
   ValueError("DAS requires a 'team_in_possession' column. "
              "Call derive_team_in_possession(frames, carrier_df) to add it.")
   ```

Ball rows: `accessible-space` expects a ball player_id sentinel. The adapter
filters `is_ball == True` rows and maps them to `ball_player_id="ball"`.

### 1.6 `derive_team_in_possession` Helper

`infer_ball_carrier` (TF-5) returns a per-frame aggregation with columns
`ball_carrier_player_id`, `ball_carrier_team_id`, `ball_carrier_distance_m`
--- it does **not** add a `team_in_possession` column to the tracking frames.
The derivation is a 3-step merge-and-rename that should not be left to the
user.

**Placement:** `tracking/_ball_carrier.py`, adjacent to `infer_ball_carrier`
which it consumes. The function has no DAS dependency --- it is a general
tracking helper usable by any future feature needing team-in-possession
(OBSO, space creation, etc.). Re-exported from `tracking/__init__.py`
alongside existing `infer_ball_carrier`.

Signature:
```python
def derive_team_in_possession(
    frames: pd.DataFrame,
    carrier: pd.DataFrame,
) -> pd.DataFrame:
```

Steps:
1. Merge `carrier[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]`
   into `frames` on `(game_id, period_id, frame_id)`.
2. Rename `ball_carrier_team_id` -> `team_in_possession`.
3. Return the enriched copy.

Frames where no carrier was inferred get `team_in_possession = NaN`.
`accessible-space` handles NaN possession gracefully (treats as contested).

### 1.7 Action-Coupled Layer

Following the established pattern (ADR-005, `pitch_control_at_action` from
PR-S31, `link_actions_to_frames` linkage):

- `das_at_action(actions, frames, *, **kwargs)` ---
  per-action DAS from the linked frame. Uses `link_actions_to_frames` (not
  `slice_around_event`) to get exactly 1 frame per action, matching the
  `pitch_control_at_action` precedent.
- `add_das(actions, frames, **kwargs)` --- aggregator adding columns:
  `das_team`, `das_opponent`, `das_diff`. Uses team-level `get_das` (not
  per-player), matching the `add_pitch_control` pattern.
  `das_diff = das_team - das_opponent`.
- `das_xfns` --- VAEP-compatible xfn list factory via `_frame_aware` marker
  dispatch. Produces three named Series per action:
  - `das_team` --- DAS for the acting team at the linked frame.
  - `das_opponent` --- DAS for the opposing team.
  - `das_diff` --- `das_team - das_opponent`.

  Names follow the `<concept>_<perspective>` convention established by
  `pitch_control_at_ball__<method>` (PR-S31). No `__method` suffix needed
  since DAS has a single model (unlike pitch control's three flavors).
  Tolerates missing columns during `feature_column_names` 10-row dummy
  introspection (silent NaN).

### 1.8 Optional Dependency

`accessible-space` declared in `[das]` extra in `pyproject.toml`:
```toml
[project.optional-dependencies]
das = ["accessible-space>=2.0,<3"]
```

Upper bound `<3` guards against breaking API changes (the adapter depends on
specific function signatures and return-tuple field names).

Lazy import at function call time:
```python
try:
    import accessible_space
except ImportError as e:
    raise ImportError(
        "accessible-space is required for DAS features. "
        "Install with: pip install 'silly-kicks[das]'"
    ) from e
```

### 1.9 Computational Profile

Empirical benchmark (2026-05-06, accessible-space 2.0.15, 22 players + ball):

| Scope | Time |
|-------|------|
| 1 frame | ~28 ms |
| 1 action-coupled match (~1,500 actions) | ~42 s |
| 1 match at 25 fps (135,000 frames) | ~62 min |
| Season action-coupled (380 matches) | ~4.4 h |
| Season all-frame (380 matches) | ~394 h |

Action-coupled mode is production-practical for laptop/cluster use. All-frame
mode at season scale needs parallelization (Spark/Dask). Both modes are
available; the computational caveat is documented, not gated.

### 1.10 References

Bischofberger, J., & Baca, A. (2026). "Dangerous accessible space: a unified
model of space and value in team sports." *Journal of Big Data*, 13, 76.
Package: [accessible-space on PyPI](https://pypi.org/project/accessible-space/).

---

## 2. TF-29 --- VAEP Design-Space Variants

### 2.1 Motivation

The DTAI Sports "Three Key Design Decisions for Possession State Value Models"
blog series (Cascioli, Robberechts, Van Tente & Davis 2024--2025) --- from the
VAEP creators themselves --- experimentally compares VAEP / PV / OBV / g+
design choices on StatsBomb Big 5 2015/16 with leave-one-league-out CV.

Two findings are actionable for silly-kicks:

1. **Window choice materially affects player rankings** (Part 2). Action-based
   ranks defenders higher; possession-based doubles shot weight in offensive
   rating. Different windows are different lenses, not better/worse.

2. **`goalscore` is the only feature correlated with team strength** (Part 3).
   Correlation ~0.5 vs < 0.15 for all others. Removing it shifts top-25
   Jaccard by 0.20 but costs only ~0.002 AUC. Trade-off is the user's call.

silly-kicks already covers Part 1 (target variable: goals vs xG) via
`xg_column` param, and validates HybridVAEP's result-leakage fix.

This item fills the remaining gaps: windowing variants and goalscore bias
control. **No defaults change.**

### 2.2 Windowing Variants --- `vaep/labels.py`

#### API Change

```python
def scores(
    actions: pd.DataFrame,
    nr_actions: int = 10,
    xg_column: str | None = None,
    *,
    window: Literal["action", "possession", "time"] = "action",
    window_seconds: float = 15.0,
) -> pd.DataFrame:
```

Same signature change for `concedes()`.

Three window modes for three code paths. The DTAI blog's "extended possession"
vs "naive possession" distinction is handled upstream by how the user
configures `add_possessions`, not by the labels code (see § 2.2.1).

#### Behavior Per Mode

**`"action"` (default):** Current behavior, unchanged. Uses `nr_actions`.
Backward compatible --- all existing call sites work without modification.

**`"possession"`:** Looks ahead within the same `possession_id` value.
Requires `possession_id` column in `actions`; raises:
```
ValueError("window='possession' requires a 'possession_id' column. "
           "Call add_possessions() first.")
```
`nr_actions` is ignored. If the caller passes a non-default `nr_actions`
(i.e., `nr_actions != 10`) alongside `window != "action"`, emit:
```python
warnings.warn(
    f"nr_actions={nr_actions} is ignored when window={window!r}; "
    f"only window='action' uses nr_actions",
    UserWarning,
    stacklevel=2,
)
```

**`"time"`:** Looks ahead within `window_seconds` of the current action's
`time_seconds`, bounded by `period_id` (no cross-period bleed). Requires
`time_seconds` column (already in SPADL schema). `nr_actions` is ignored (same
warning as `"possession"` if non-default). Default `window_seconds=15.0` per
DTAI evidence. `window_seconds` is ignored when `window != "time"`.

#### 2.2.1 DTAI Possession Modes and `add_possessions` Configuration

The DTAI Part 2 blog defines two possession flavors:

- **Naive possession** (xT-style): ends after any unsuccessful on-the-ball
  action or opponent touch. Map to `add_possessions` with defaults:
  ```python
  add_possessions(actions)  # merge_brief_opposing_actions=0
  ```

- **Extended possession** (OBV/g+-style): requires the defensive team to
  "earn enough control" before ending a possession; merges brief opponent
  touches back into the containing chain. Map to:
  ```python
  add_possessions(
      actions,
      merge_brief_opposing_actions=2,
      brief_window_seconds=2.0,
      defensive_transition_types=("interception", "clearance"),
  )
  ```
  This combination merges up to 2 consecutive opponent actions within 2
  seconds back into the possession, and prevents interceptions/clearances
  alone from triggering a team-change boundary --- requiring the opponent
  to earn sustained control.

Both produce a `possession_id` column; the labels code consumes it
identically via `window="possession"`. The distinction is documented in the
`window="possession"` docstring with the two recommended `add_possessions`
configurations above.

#### Implementation Strategy

- `"action"`: existing shift-loop, untouched.
- `"possession"`: vectorized groupby on `(game_id, possession_id)` --- for
  each action, check if any goal/own-goal exists in the same possession group
  at a later index. Forward-looking cummax within group.
- `"time"`: per-period vectorization using `np.searchsorted` on sorted
  `time_seconds`. For each action at time `t`, binary-search for the window
  boundary `t + window_seconds` to find the range of actions within the
  window, then check for any goal/own-goal in that range. O(n log n) per
  period; bounded by `period_id` to prevent cross-period bleed.

  **Sorted-input precondition:** `np.searchsorted` requires monotonically
  non-decreasing `time_seconds` within each period. SPADL actions are
  conventionally sorted by `(game_id, period_id, action_id)` with
  `time_seconds` monotonic within a period. The helper asserts this:
  ```python
  assert (grp["time_seconds"].diff().iloc[1:] >= 0).all(), \
      "time_seconds must be non-decreasing within each period"
  ```
  Using a debug assertion (not a production raise) since this is a
  precondition on well-formed SPADL input, not a user-facing validation.

  **Boundary semantics:** strict inequality. An action at time `t` scores
  if a goal exists at `goal_time` where `goal_time - t < window_seconds`
  (strictly less than). An action exactly `window_seconds` before a goal
  does NOT score.

All modes support the `xg_column` variant (xG-weighted labels). The internal
dispatch branches early on `window` and delegates to mode-specific private
helpers.

### 2.3 Goalscore-Free xfn Lists

In `vaep/base.py`, adjacent to `xfns_default`:
```python
xfns_default_no_goalscore = [x for x in xfns_default if x is not fs.goalscore]
```

In `vaep/hybrid.py`, adjacent to `hybrid_xfns_default`:
```python
hybrid_xfns_default_no_goalscore = [
    x for x in hybrid_xfns_default if x is not fs.goalscore
]
```

Both re-exported from `vaep/__init__.py`.

**No default changes.** `xfns_default` and `hybrid_xfns_default` keep
`goalscore`. The no-goalscore variants are opt-in for users who want
team-strength-decorrelated player rankings per DTAI Part 3.

### 2.4 References

DTAI Sports blog series:
- [Intro](https://dtai.cs.kuleuven.be/sports/blog/three-key-design-decisions-for-possession-state-value-models:-an-experimental-analysis/)
- [Part 1](https://dtai.cs.kuleuven.be/sports/blog/an-experimental-analysis-of-possession-state-value-models:-part-1/) (target variable)
- [Part 2](https://dtai.cs.kuleuven.be/sports/blog/an-experimental-analysis-of-possession-state-value-models:-part-2/) (windowing)
- [Part 3](https://dtai.cs.kuleuven.be/sports/blog/an-experimental-analysis-of-possession-state-value-models:-part-3/) (goalscore bias)

---

## 3. Testing Strategy

TDD throughout: tests written first (red), implementation makes them green.
Regenerate provider fixtures from lakehouse (Sportec/Metrica/SkillCorner)
or local PFF data when existing fixtures lack needed cases.

### 3.1 TF-28 Tests

#### Unit tests (`tests/tracking/test_das.py`)

- Coordinate transform: `_to_das_coords` shifts correctly (spot-check
  known values: `(0,0)` -> `(-52.5,-34)`, `(105,68)` -> `(52.5,34)`).
- Column mapping correctness: mapping dict covers all required
  `accessible-space` params.
- Missing `team_in_possession` column raises `ValueError`.
- Missing `vx` / `vy` columns raises `ValueError`.
- Missing `accessible-space` import raises `ImportError` with helpful message.
- `derive_team_in_possession` merges correctly on synthetic carrier df;
  frames without carrier match get `NaN`.
- `das_xfns` factory: `_frame_aware` marker present;
  `feature_column_names` 10-row dummy introspection works (silent NaN).

#### Per-provider e2e tests (`tests/tracking/test_das_e2e.py`)

Parametrized across all tracking providers (Sportec, Metrica, SkillCorner,
PFF) using existing slim-parquet synthesizer fixtures from
`tests/tracking/_provider_inputs.py`. Regenerate from lakehouse/local data if
fixtures lack needed cases (e.g., velocity columns, goals, possession labels).

Full pipeline per provider:
1. Load fixture -> convert -> preprocess (smooth + derive_velocities) ->
   infer_ball_carrier -> `derive_team_in_possession` -> `get_das`.
2. Same pipeline -> `get_individual_das`.
3. Link to SPADL actions via `link_actions_to_frames` -> `das_at_action` ->
   `add_das` -> assert aggregator columns.
4. `get_xc` on pass-filtered actions + linked frames.

Per-provider assertions:
- Output shape matches input frame/action count.
- No all-NaN columns (silent regression guard).
- `n_valid_das >= 1` per provider.
- Column dtypes are float64.

#### Invariant tests (`tests/invariants/test_das_invariants.py`)

Parametrized across providers:
- `AS >= 0`, `DAS >= 0` (space non-negative).
- `AS >= DAS` per frame (danger-weighting can only reduce).
- `xC in [0, 1]` (probability bounds).
- `das_team >= 0`, `das_opponent >= 0` at action-coupled level.
- `das_diff = das_team - das_opponent` exact equality.

#### Edge cases

- Frame with all players stationary (`vx=vy=0`) --- valid AS/DAS, no crash.
- Frame with NaN ball position --- NaN output, no crash.
- Frame with fewer than 22 players --- library handles gracefully.
- Single-frame input (1 action) --- no off-by-one.
- Mixed `player_id` dtypes (int64 Sportec vs object kloppy) --- adapter
  handles both.

### 3.2 TF-29 Tests

#### Unit tests (`tests/vaep/test_labels_windowing.py`)

- `window="action"` backward compat: identical output to current `scores()` /
  `concedes()` on same input.
- `window="possession"` missing `possession_id` -> `ValueError`.
- `window="time"` missing `time_seconds` -> `ValueError`.
- `window="time"` + `window_seconds` not provided uses default 15.0.
- `nr_actions=5` with `window="possession"` emits `UserWarning`.
- `nr_actions=5` with `window="time"` emits `UserWarning`.
- `nr_actions=10` (default) with `window != "action"` does NOT warn.

#### Hand-crafted fixture tests

- **`"possession"`:** 12-action fixture, 3 possession chains, goal in chain 1
  at action 4, goal in chain 3 at action 11. Assert exact boolean vector for
  both `scores` and `concedes`.
- **`"time"`:** 10-action fixture with `time_seconds`
  [0, 3, 7, 12, 18, 20, 21, 25, 30, 35], goal at action 5 (t=20).
  `window_seconds=5.0` -> actions where `goal_time - action_time <
  window_seconds` score (strict inequality). Action at t=15.0 does NOT
  score (20 - 15 = 5.0, not strictly less than 5.0). Action at t=15.01
  scores (20 - 15.01 = 4.99 < 5.0). In the fixture, action at t=18
  scores (20 - 18 = 2 < 5), action at t=12 does NOT (20 - 12 = 8 >= 5).
- **Cross-period boundary:** goal in period 2 does NOT bleed into period 1
  labels for any window mode.
- **Own-goal handling:** own-goal in possession chain credits the opponent.
- **xG variant:** `xg_column` works with all window modes, not just `"action"`.

#### E2e against real SPADL data (`tests/vaep/test_labels_windowing_e2e.py`)

Parametrized across event providers using existing converter fixtures
(StatsBomb WC2018 H5 + any committed provider fixtures). Regenerate fixtures
from lakehouse/local data if needed.

1. Convert -> `add_names` -> `add_possessions` ->
   `scores(window="possession")` / `concedes(window="possession")`.
2. Same with `window="time"`, `window_seconds=15.0`.
3. Assert: output shape matches actions, all values are bool (or float for xG),
   no crashes on real data.
4. Sanity: `scores.sum() > 0` and `concedes.sum() > 0` (real matches have
   goals).
5. Parity: `window="action"` output identical to existing `scores()` call
   (regression guard).

#### Goalscore-free xfn tests

- `xfns_default_no_goalscore` has `len(xfns_default) - 1` entries.
- `fs.goalscore not in xfns_default_no_goalscore`.
- All other entries identical and in same order.
- Same three assertions for `hybrid_xfns_default_no_goalscore`.
- `feature_column_names` introspection works with both no-goalscore lists (no
  crash, fewer columns than default).

---

## 4. Deliverables Checklist

- [ ] `silly_kicks/tracking/_das.py` --- DAS adapter module
- [ ] `silly_kicks/tracking/_ball_carrier.py` --- add
      `derive_team_in_possession` (general tracking helper)
- [ ] `tracking/__init__.py` re-exports: `get_das`, `get_individual_das`,
      `get_xc`, `das_at_action`, `add_das`, `das_xfns`,
      `derive_team_in_possession`
- [ ] `pyproject.toml` --- `[das]` optional extra (`accessible-space>=2.0,<3`)
- [ ] `vaep/labels.py` --- `window` + `window_seconds` params on `scores()`
      and `concedes()`
- [ ] `vaep/base.py` --- `xfns_default_no_goalscore`
- [ ] `vaep/hybrid.py` --- `hybrid_xfns_default_no_goalscore`
- [ ] `vaep/__init__.py` --- re-export new symbols
- [ ] `tests/tracking/test_das.py` --- unit tests
- [ ] `tests/tracking/test_das_e2e.py` --- per-provider e2e
- [ ] `tests/invariants/test_das_invariants.py` --- physical invariants
- [ ] `tests/vaep/test_labels_windowing.py` --- unit + hand-crafted fixtures
- [ ] `tests/vaep/test_labels_windowing_e2e.py` --- real-data e2e
- [ ] NOTICE file --- Bischofberger & Baca 2026 + DTAI blog series references
- [ ] CHANGELOG.md --- 3.8.0 entry
- [ ] TODO.md --- delete TF-28 and TF-29 rows
