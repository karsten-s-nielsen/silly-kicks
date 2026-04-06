# Phase 3c: Converter Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vectorize all `apply(axis=1)` hot paths, decompose the Wyscout god module, cache config DataFrames, optimize atomic conversion, and fix the gamestates pandas 3.0 bug.

**Architecture:** Each converter's row-wise `apply()` dispatch is replaced with a two-phase approach: (1) pre-flatten nested dicts/lists into flat DataFrame columns at conversion start, then (2) use `np.select` condition/choice arrays over those flat columns. The Wyscout module is split into 3 files. Config DataFrame factories get `@lru_cache`. Gamestates uses vectorized shift + boundary detection. Atomic conversion defers concat+sort to a single final pass.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new dependencies.

**Working directory:** `D:\Development\karstenskyt__silly-kicks\`

**Sources:** Design spec at `docs/specs/2026-04-06-phase3c-converter-rewrite-design.md`

---

## File Structure

```
Modified:
  silly_kicks/spadl/config.py              — add @lru_cache to 3 functions
  silly_kicks/atomic/spadl/config.py       — add @lru_cache to actiontypes_df
  silly_kicks/vaep/features.py             — vectorize gamestates()
  silly_kicks/spadl/statsbomb.py           — flatten extra, np.select, vectorize coords
  silly_kicks/spadl/opta.py                — explode qualifiers, np.select
  silly_kicks/spadl/wyscout.py             — slim to entry point + constants (~150 lines)
  silly_kicks/atomic/spadl/base.py         — deferred single sort
  docs/DEFERRED.md                         — mark items resolved

Created:
  silly_kicks/spadl/_wyscout_events.py     — event fix pipeline (moved from wyscout.py)
  silly_kicks/spadl/_wyscout_mappings.py   — vectorized SPADL mapping functions
```

---

## Task 1: Config DataFrame Caching (O-15)

**Files:**
- Modify: `silly_kicks/spadl/config.py`
- Modify: `silly_kicks/atomic/spadl/config.py`

- [ ] **Step 1: Write the test**

Add to `tests/spadl/test_schema.py`:

```python
def test_config_df_caching():
    """O-15: Config DataFrame factories should return cached instances."""
    import silly_kicks.spadl.config as spadlcfg
    assert spadlcfg.actiontypes_df() is spadlcfg.actiontypes_df()
    assert spadlcfg.results_df() is spadlcfg.results_df()
    assert spadlcfg.bodyparts_df() is spadlcfg.bodyparts_df()


def test_atomic_config_df_caching():
    """O-15: Atomic config DataFrame factories should return cached instances."""
    import silly_kicks.atomic.spadl.config as atomicconfig
    assert atomicconfig.actiontypes_df() is atomicconfig.actiontypes_df()
    assert atomicconfig.bodyparts_df() is atomicconfig.bodyparts_df()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_schema.py::test_config_df_caching tests/spadl/test_schema.py::test_atomic_config_df_caching -v`
Expected: FAIL — `is` identity check fails because each call creates a new DataFrame

- [ ] **Step 3: Add `@lru_cache` to SPADL config**

In `silly_kicks/spadl/config.py`, add `import functools` at the top (after `import pandas as pd`), then decorate all three functions:

```python
import functools

# ... (field_length, field_width, lists, dicts stay the same)

@functools.lru_cache(maxsize=None)
def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type."""
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])


@functools.lru_cache(maxsize=None)
def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type."""
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


@functools.lru_cache(maxsize=None)
def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type."""
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])
```

- [ ] **Step 4: Add `@lru_cache` to atomic config**

In `silly_kicks/atomic/spadl/config.py`, add `import functools` and decorate `actiontypes_df`:

```python
import functools

# ... (existing code)

@functools.lru_cache(maxsize=None)
def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each Atomic-SPADL action type."""
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])
```

Note: `bodyparts_df` is aliased from `_spadl.bodyparts_df` which is already cached.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_schema.py -v`
Expected: all PASS

---

## Task 2: Gamestates Vectorization (O-8)

**Files:**
- Modify: `silly_kicks/vaep/features.py:85-94`

This task fixes the 4 pre-existing test failures caused by `groupby().apply(as_index=False)` dropping key columns in pandas 3.0.

- [ ] **Step 1: Verify the 4 tests currently fail**

Run: `python -m pytest tests/vaep/test_features.py::test_same_index tests/vaep/test_features.py::test_time tests/vaep/test_features.py::test_player_possession_time tests/atomic/test_atomic_features.py::test_same_index -v`
Expected: 4 FAIL with `KeyError: "['period_id'] not in index"`

- [ ] **Step 2: Replace gamestates() implementation**

In `silly_kicks/vaep/features.py`, replace lines 85-94 (the `gamestates` function body from `if nb_prev_actions < 1:` through `return states`):

```python
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    # Precompute group-first values once for boundary filling
    first_in_group = actions.groupby(["game_id", "period_id"], sort=False).transform("first")
    for i in range(1, nb_prev_actions):
        prev = actions.shift(i)
        # Detect period/game boundaries: where shifted row crosses a group
        boundary = (
            (actions["game_id"] != actions["game_id"].shift(i))
            | (actions["period_id"] != actions["period_id"].shift(i))
        )
        # At boundaries, fill with the first row of the current group
        prev[boundary] = first_in_group[boundary]
        prev.index = actions.index.copy()
        states.append(prev)
    return states
```

Key changes:
- Removed `groupby().apply()` entirely — no more lambda, no `as_index` issue
- Uses `actions.shift(i)` globally, then detects boundary rows where the shift crossed a game/period break
- At boundary rows, fills from `first_in_group` (the first row of each group, computed once via `transform("first")`)
- The `prev.index = actions.index.copy()` ensures index alignment (matching old behavior)

- [ ] **Step 3: Run the 4 previously-failing tests**

Run: `python -m pytest tests/vaep/test_features.py::test_same_index tests/vaep/test_features.py::test_time tests/vaep/test_features.py::test_player_possession_time tests/atomic/test_atomic_features.py::test_same_index -v`
Expected: 4 PASS

- [ ] **Step 4: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass (previously 74 pass + 4 fail → now 78 pass)

---

## Task 3: StatsBomb Converter Vectorization (O-2b, O-5b, O-6)

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py` (major rewrite of internals)

This is the largest single task. The converter has 4 `apply(axis=1)` calls and 1 Python for-loop that all need vectorizing. The approach:
1. Add a `_flatten_extra()` helper that extracts ~20 nested dict fields into flat columns
2. Replace `_insert_interception_passes` to use the flat column instead of apply
3. Replace `_get_end_location` apply with vectorized fillna chain
4. Replace `_convert_locations` for-loop with numpy array ops
5. Replace `_parse_event` apply with `np.select` chains for type_id, result_id, bodypart_id
6. Replace `_infer_xy_fidelity_versions` to avoid `apply(pd.Series)`

**Key principle:** Every if/elif branch in the existing `_parse_*` functions translates to a condition in `np.select`. The flat columns (`_pass_type`, `_shot_outcome`, etc.) serve as the vectorized equivalents of the nested dict lookups.

- [ ] **Step 1: Run existing StatsBomb tests to confirm they pass before changes**

Run: `python -m pytest tests/spadl/test_statsbomb.py -v`
Expected: 6 PASS, 1 SKIP

- [ ] **Step 2: Add `_flatten_extra` helper**

Add this function to `silly_kicks/spadl/statsbomb.py` (after the constants, before `convert_to_actions`):

```python
def _flatten_extra(events: pd.DataFrame) -> pd.DataFrame:
    """Extract nested extra dict fields into flat columns for vectorized dispatch."""
    extra = events["extra"]

    def _safe_get(series: pd.Series, *keys: str) -> pd.Series:
        """Vectorized nested dict get: series.str.get(k1).str.get(k2)..."""
        result = series
        for key in keys:
            result = result.str.get(key, default={})
        # Replace empty dicts/None with actual None for cleaner downstream logic
        return result.where(result.apply(lambda x: x is not None and x != {}), other=None)

    events["_pass_type"] = _safe_get(extra, "pass", "type", "name")
    events["_pass_height"] = _safe_get(extra, "pass", "height", "name")
    events["_pass_cross"] = extra.str.get("pass", {}).str.get("cross")
    events["_pass_outcome"] = _safe_get(extra, "pass", "outcome", "name")
    events["_pass_body_part"] = _safe_get(extra, "pass", "body_part", "name")
    events["_pass_end_location"] = extra.str.get("pass", {}).str.get("end_location")
    events["_shot_type"] = _safe_get(extra, "shot", "type", "name")
    events["_shot_outcome"] = _safe_get(extra, "shot", "outcome", "name")
    events["_shot_body_part"] = _safe_get(extra, "shot", "body_part", "name")
    events["_shot_end_location"] = extra.str.get("shot", {}).str.get("end_location")
    events["_dribble_outcome"] = _safe_get(extra, "dribble", "outcome", "name")
    events["_gk_type"] = _safe_get(extra, "goalkeeper", "type", "name")
    events["_gk_outcome"] = _safe_get(extra, "goalkeeper", "outcome", "name")
    events["_gk_body_part"] = _safe_get(extra, "goalkeeper", "body_part", "name")
    events["_foul_card"] = _safe_get(extra, "foul_committed", "card", "name")
    events["_duel_type"] = _safe_get(extra, "duel", "type", "name")
    events["_duel_outcome"] = _safe_get(extra, "duel", "outcome", "name")
    events["_interception_outcome"] = _safe_get(extra, "interception", "outcome", "name")
    events["_clearance_body_part"] = _safe_get(extra, "clearance", "body_part", "name")
    events["_carry_end_location"] = extra.str.get("carry", {}).str.get("end_location")
    return events
```

- [ ] **Step 3: Vectorize `_insert_interception_passes`**

Replace the `is_interception_pass` apply with a check on the flat column. In `_insert_interception_passes`, replace:

```python
def is_interception_pass(x: dict) -> bool:
    return x.get("extra", {}).get("pass", {}).get("type", {}).get("name") == "Interception"

df_events_interceptions = df_events[df_events.apply(is_interception_pass, axis=1)].copy()
```

with:

```python
mask = df_events["_pass_type"] == "Interception"
df_events_interceptions = df_events[mask].copy()
```

**Important:** `_flatten_extra` must be called BEFORE `_insert_interception_passes` in `convert_to_actions`. Move the call order so `_flatten_extra(events)` happens right after `events["extra"] = events["extra"].fillna({})`.

- [ ] **Step 4: Vectorize `_get_end_location`**

Replace the row-wise apply:
```python
end_location = events[["location", "extra"]].apply(_get_end_location, axis=1)
```

with a vectorized fillna chain using the flat columns:
```python
end_location = events["_pass_end_location"].fillna(
    events["_shot_end_location"]
).fillna(
    events["_carry_end_location"]
).fillna(
    events["location"]
)
```

Remove the old `_get_end_location` function.

- [ ] **Step 5: Vectorize `_convert_locations`**

Replace the Python for-loop in `_convert_locations` with numpy array operations:

```python
def _convert_locations(locations: pd.Series, fidelity_version: int) -> npt.NDArray[np.float64]:
    """Convert StatsBomb locations to SPADL coordinates (vectorized)."""
    cell_side = 0.1 if fidelity_version == 2 else 1.0
    crc = cell_side / 2

    # Build a 2D array from the Series of lists; handle None/NaN
    loc_list = locations.tolist()
    xy_raw = np.array(
        [loc[:2] if isinstance(loc, list) and len(loc) >= 2 else [np.nan, np.nan] for loc in loc_list],
        dtype=float,
    )
    # Detect 3-element goal-frame coordinates (shot end locations)
    is_three = np.array(
        [isinstance(loc, list) and len(loc) == 3 for loc in loc_list]
    )
    y_offset = np.where(is_three, 0.05, crc)

    coordinates = np.empty((len(locations), 2), dtype=float)
    coordinates[:, 0] = (xy_raw[:, 0] - crc) / _SB_FIELD_LENGTH * spadlconfig.field_length
    coordinates[:, 1] = spadlconfig.field_width - (xy_raw[:, 1] - y_offset) / _SB_FIELD_WIDTH * spadlconfig.field_width
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, spadlconfig.field_length)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, spadlconfig.field_width)
    return coordinates
```

- [ ] **Step 6: Vectorize `_infer_xy_fidelity_versions`**

Replace `events.location.apply(pd.Series)` with:
```python
locations = pd.DataFrame(events.location.dropna().tolist())
```

- [ ] **Step 7: Replace `_parse_event` with vectorized np.select**

This is the core rewrite. Replace the `_parse_event` apply call and all 11 `_parse_*` functions with three vectorized functions that each return a Series:

```python
def _vectorized_type_id(events: pd.DataFrame) -> pd.Series:
    """Vectorized type_id assignment using np.select on flat columns."""
    tn = events["type_name"]
    aid = spadlconfig.actiontype_id

    # Pass subtypes
    is_pass = tn.isin(["Pass"])
    pass_type = events["_pass_type"]
    pass_height = events["_pass_height"]
    pass_cross = events["_pass_cross"].fillna(False)
    pass_outcome = events["_pass_outcome"]

    conditions = [
        # Pass types (ordered by specificity)
        is_pass & (pass_type == "Free Kick") & ((pass_height == "High Pass") | pass_cross),
        is_pass & (pass_type == "Free Kick"),
        is_pass & (pass_type == "Corner") & ((pass_height == "High Pass") | pass_cross),
        is_pass & (pass_type == "Corner"),
        is_pass & (pass_type == "Goal Kick"),
        is_pass & (pass_type == "Throw-in"),
        is_pass & pass_cross,
        is_pass & pass_outcome.isin(["Injury Clearance", "Unknown"]),
        is_pass,
        # Non-pass types
        tn == "Dribble",
        tn == "Carry",
        tn == "Foul Committed",
        tn == "Interception",
        tn == "Miscontrol",
        tn == "Own Goal Against",
        tn == "Clearance",
        # Shot subtypes
        (tn == "Shot") & (events["_shot_type"] == "Free Kick"),
        (tn == "Shot") & (events["_shot_type"] == "Penalty"),
        tn == "Shot",
        # Duel subtypes
        (tn == "Duel") & (events["_duel_type"] == "Tackle"),
        tn == "Duel",
        # Goalkeeper subtypes
        (tn == "Goal Keeper") & (events["_gk_type"] == "Shot Saved"),
        (tn == "Goal Keeper") & events["_gk_type"].isin(["Collected", "Keeper Sweeper"]),
        (tn == "Goal Keeper") & (events["_gk_type"] == "Punch"),
        tn == "Goal Keeper",
    ]
    choices = [
        aid["freekick_crossed"],
        aid["freekick_short"],
        aid["corner_crossed"],
        aid["corner_short"],
        aid["goalkick"],
        aid["throw_in"],
        aid["cross"],
        aid["non_action"],  # Injury Clearance / Unknown pass
        aid["pass"],
        aid["take_on"],
        aid["dribble"],
        aid["foul"],
        aid["interception"],
        aid["bad_touch"],
        aid["bad_touch"],  # Own Goal Against
        aid["clearance"],
        aid["shot_freekick"],
        aid["shot_penalty"],
        aid["shot"],
        aid["tackle"],
        aid["non_action"],  # non-tackle duels
        aid["keeper_save"],
        aid["keeper_claim"],
        aid["keeper_punch"],
        aid["non_action"],  # other goalkeeper events
    ]
    return pd.Series(
        np.select(conditions, choices, default=aid["non_action"]),
        index=events.index,
        dtype="int64",
    )
```

Write `_vectorized_result_id(events)` and `_vectorized_bodypart_id(events)` following the same pattern — translate every if/elif branch from the old `_parse_*_event` functions into np.select conditions. Read the existing functions carefully to capture every branch.

Then in `convert_to_actions`, replace:
```python
actions[["type_id", "result_id", "bodypart_id"]] = events[["type_name", "extra"]].apply(
    _parse_event, axis=1, result_type="expand"
)
```
with:
```python
actions["type_id"] = _vectorized_type_id(events)
actions["result_id"] = _vectorized_result_id(events)
actions["bodypart_id"] = _vectorized_bodypart_id(events)
```

Remove all the old `_parse_*` functions after the vectorized replacements are working.

- [ ] **Step 8: Run all StatsBomb tests**

Run: `python -m pytest tests/spadl/test_statsbomb.py -v`
Expected: all PASS

- [ ] **Step 9: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass

---

## Task 4: Opta Converter Vectorization (O-2c)

**Files:**
- Modify: `silly_kicks/spadl/opta.py`

- [ ] **Step 1: Run existing Opta tests to confirm they pass**

Run: `python -m pytest tests/spadl/test_opta.py -v`
Expected: 6 PASS, 1 SKIP

- [ ] **Step 2: Add qualifier explosion and np.select for type_id**

At the top of `convert_to_actions` (after input validation), add qualifier explosion:

```python
    # Pre-explode qualifier dict into boolean columns for vectorized dispatch
    _USED_QUALIFIER_IDS = [1, 2, 3, 5, 6, 9, 15, 20, 21, 26, 28, 32, 72, 107, 124, 155, 168, 238]
    for qid in _USED_QUALIFIER_IDS:
        events[f"q_{qid}"] = events["qualifiers"].apply(lambda q, k=qid: k in q)
```

Then replace the three apply calls (lines 85-91) with vectorized np.select functions. Write `_vectorized_type_id(events)`, `_vectorized_result_id(events)`, `_vectorized_bodypart_id(events)` that each translate the existing if/elif chain from `_get_type_id`, `_get_result_id`, `_get_bodypart_id` into `np.select` using `events["type_name"]`, `events["outcome"]`, and the `q_*` boolean columns.

The conditions and choices must exactly reproduce the original logic. For example, `_get_type_id`'s pass branch becomes:

```python
is_pass = tn.isin(["pass", "offside pass"])
conditions = [
    events["q_238"],  # fairplay → non_action
    is_pass & events["q_107"],  # throw_in
    is_pass & events["q_5"] & (events["q_2"] | events["q_1"] | events["q_155"]),  # freekick_crossed
    is_pass & events["q_5"],  # freekick_short
    is_pass & events["q_6"] & events["q_2"],  # corner_crossed
    is_pass & events["q_6"],  # corner_short
    is_pass & events["q_2"],  # cross
    is_pass & events["q_124"],  # goalkick
    is_pass,  # pass (default for pass events)
    tn == "take on",
    (tn == "foul") & (events["outcome"] == False),  # noqa: E712
    tn == "tackle",
    tn.isin(["interception", "blocked pass"]),
    tn.isin(["miss", "post", "attempt saved", "goal"]) & events["q_9"],  # shot_penalty
    tn.isin(["miss", "post", "attempt saved", "goal"]) & events["q_26"],  # shot_freekick
    tn.isin(["miss", "post", "attempt saved", "goal"]),  # shot
    tn == "save",
    tn == "claim",
    tn == "punch",
    tn == "keeper pick-up",
    tn == "clearance",
    tn == "card",
    (tn == "ball touch") & (events["outcome"] == False),  # noqa: E712
]
```

Replace the three apply calls with:
```python
actions["type_id"] = _vectorized_type_id(events)
actions["result_id"] = _vectorized_result_id(events)
actions["bodypart_id"] = _vectorized_bodypart_id(events)
```

Remove the old `_get_type_id`, `_get_result_id`, `_get_bodypart_id` functions.

- [ ] **Step 3: Run Opta tests**

Run: `python -m pytest tests/spadl/test_opta.py -v`
Expected: all PASS

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass

---

## Task 5: Wyscout File Split + Tag/Position Optimization (A1, O-3, O-4)

**Files:**
- Modify: `silly_kicks/spadl/wyscout.py` (slim to ~150 lines)
- Create: `silly_kicks/spadl/_wyscout_events.py` (~400 lines)
- Create: `silly_kicks/spadl/_wyscout_mappings.py` (~200 lines)

This task splits the 1033-line wyscout.py into 3 files and optimizes tag expansion + position extraction. No dispatch vectorization yet (that's Task 6).

- [ ] **Step 1: Run existing Wyscout tests to confirm they pass**

Run: `python -m pytest tests/spadl/test_wyscout.py -v`
Expected: all pass

- [ ] **Step 2: Create `_wyscout_events.py`**

Move these functions from `wyscout.py` to `silly_kicks/spadl/_wyscout_events.py`:
- `_fix_wyscout_events`
- `_create_shot_coordinates`
- `_convert_duels`
- `_insert_interceptions`
- `_add_offside_variable`
- `_convert_touches`
- `_convert_simulations`
- All coordinate fix helpers from `_fix_actions`: `_fix_actions`, `_fix_foul_coordinates`, `_fix_clearance_coordinates`, `_fix_opta_to_wyscout_coordinates`, `_adjust_goalkick_result`, etc.

The new file's imports should reference constants from `wyscout.py` (e.g., `from .wyscout import _WS_TYPE_DUEL, _WS_SUBTYPE_*`).

- [ ] **Step 3: Optimize `_get_tagsdf` (O-4)**

In `wyscout.py` (where `_get_tagsdf` and `wyscout_tags` remain), replace the 60x serial `apply()` loop:

```python
tags = events.tags.apply(_get_tag_set)
tagsdf = pd.DataFrame()
for tag_id, column in wyscout_tags:
    tagsdf[column] = tags.apply(lambda x, tag=tag_id: tag in x)
```

with a batch DataFrame constructor:

```python
tags = events.tags.apply(_get_tag_set)
tagsdf = pd.DataFrame(
    {column: tags.apply(lambda s, t=tag_id: t in s) for tag_id, column in wyscout_tags}
)
```

This is still N×60 work but builds the DataFrame in one constructor call instead of 60 separate column assignments.

- [ ] **Step 4: Vectorize `_make_new_positions` (O-3)**

Replace the `_make_position_vars` apply in `_wyscout_events.py` (or `wyscout.py`, wherever it ends up) with vectorized list indexing:

```python
def _make_new_positions(events: pd.DataFrame) -> pd.DataFrame:
    pos_list = events["positions"].tolist()
    empty_pos = {"x": None, "y": None}
    start_pos = [p[0] if len(p) >= 1 else empty_pos for p in pos_list]
    end_pos = [p[1] if len(p) >= 2 else (p[0] if len(p) >= 1 else empty_pos) for p in pos_list]
    events["start_x"] = pd.Series([p.get("x") for p in start_pos], index=events.index, dtype=float)
    events["start_y"] = pd.Series([p.get("y") for p in start_pos], index=events.index, dtype=float)
    events["end_x"] = pd.Series([p.get("x") for p in end_pos], index=events.index, dtype=float)
    events["end_y"] = pd.Series([p.get("y") for p in end_pos], index=events.index, dtype=float)
    events = events.drop("positions", axis=1)
    return events
```

Remove the old `_make_position_vars` function.

- [ ] **Step 5: Create `_wyscout_mappings.py` (stub)**

Create `silly_kicks/spadl/_wyscout_mappings.py` and move these functions from `wyscout.py`:
- `_create_df_actions`
- `_determine_bodypart_id`
- `_determine_type_id`
- `_determine_result_id`
- `_remove_non_actions`

For now, move them as-is (still using `apply`). Task 6 will vectorize them.

The new file imports constants from `wyscout.py`:
```python
from .wyscout import (
    _WS_TYPE_TAKE_ON, _WS_TYPE_DUEL, _WS_TYPE_FOUL, _WS_TYPE_OFFSIDE,
    _WS_TYPE_PASS, _WS_TYPE_GK, _WS_TYPE_SHOT,
    _WS_SUBTYPE_CROSS, _WS_SUBTYPE_THROW_IN, _WS_SUBTYPE_CORNER,
    # ... all needed constants
)
```

- [ ] **Step 6: Update `wyscout.py` to import from new modules**

In `wyscout.py`, replace the moved function bodies with imports:
```python
from ._wyscout_events import _fix_wyscout_events, _make_new_positions, _fix_actions
from ._wyscout_mappings import _create_df_actions
```

The `convert_to_actions` function stays in `wyscout.py` as the entry point.

- [ ] **Step 7: Run Wyscout tests**

Run: `python -m pytest tests/spadl/test_wyscout.py -v`
Expected: all PASS (behavior unchanged, just restructured)

---

## Task 6: Wyscout Dispatch Vectorization (O-2)

**Files:**
- Modify: `silly_kicks/spadl/_wyscout_mappings.py`

- [ ] **Step 1: Replace `_determine_type_id` with np.select**

The current `_determine_type_id` function uses `apply(axis=1)` over event rows, checking boolean tag columns and integer type_id/subtype_id. Since all the tag booleans are already DataFrame columns (created by `_get_tagsdf`), this translates directly to `np.select`:

```python
def _vectorized_type_id(df_events: pd.DataFrame) -> pd.Series:
    """Vectorized SPADL type_id assignment for Wyscout events."""
    import numpy as np
    aid = spadlconfig.actiontype_id
    tid = df_events["type_id"]
    sid = df_events["subtype_id"]

    conditions = [
        df_events["fairplay"],
        df_events["own_goal"],
        (tid == _WS_TYPE_PASS) & (sid == _WS_SUBTYPE_CROSS),
        tid == _WS_TYPE_PASS,
        sid == _WS_SUBTYPE_THROW_IN,
        (sid == _WS_SUBTYPE_CORNER) & df_events["high"],
        sid == _WS_SUBTYPE_CORNER,
        sid == _WS_SUBTYPE_FK_CROSSED,
        sid == _WS_SUBTYPE_FK_SHORT,
        sid == _WS_SUBTYPE_GOALKICK,
        (tid == _WS_TYPE_FOUL) & ~sid.isin([_WS_SUBTYPE_HAND_FOUL, _WS_SUBTYPE_LATE_CARD_FOUL, _WS_SUBTYPE_OUT_OF_GAME_FOUL, _WS_SUBTYPE_VIOLENT_FOUL]),
        tid == _WS_TYPE_SHOT,
        sid == _WS_SUBTYPE_PENALTY,
        sid == _WS_SUBTYPE_FK_SHOT,
        (tid == _WS_TYPE_GK) & (sid == _WS_SUBTYPE_GK_CLAIM),
        (tid == _WS_TYPE_GK) & (sid == _WS_SUBTYPE_GK_PUNCH),
        tid == _WS_TYPE_GK,
        sid == _WS_SUBTYPE_CLEARANCE,
        (sid == _WS_SUBTYPE_TOUCH) & df_events["not_accurate"],
        sid == _WS_SUBTYPE_ACCELERATION,
        df_events["take_on_left"] | df_events["take_on_right"],
        df_events["sliding_tackle"],
        df_events["interception"] & sid.isin([_WS_TYPE_TAKE_ON, _WS_SUBTYPE_AIR_DUEL, _WS_SUBTYPE_GROUND_ATT_DUEL, _WS_SUBTYPE_GROUND_DEF_DUEL, _WS_SUBTYPE_GROUND_LOOSE_BALL, _WS_SUBTYPE_TOUCH]),
    ]
    choices = [
        aid["non_action"],
        aid["bad_touch"],
        aid["cross"],
        aid["pass"],
        aid["throw_in"],
        aid["corner_crossed"],
        aid["corner_short"],
        aid["freekick_crossed"],
        aid["freekick_short"],
        aid["goalkick"],
        aid["foul"],
        aid["shot"],
        aid["shot_penalty"],
        aid["shot_freekick"],
        aid["keeper_claim"],
        aid["keeper_punch"],
        aid["keeper_save"],
        aid["clearance"],
        aid["bad_touch"],
        aid["dribble"],
        aid["take_on"],
        aid["tackle"],
        aid["interception"],
    ]
    return pd.Series(np.select(conditions, choices, default=aid["non_action"]), index=df_events.index, dtype="int64")
```

Write `_vectorized_result_id` and `_vectorized_bodypart_id` similarly, translating the existing `_determine_result_id` and `_determine_bodypart_id` if/elif chains.

- [ ] **Step 2: Update `_create_df_actions` to use vectorized functions**

Replace:
```python
df_actions["bodypart_id"] = df_events.apply(_determine_bodypart_id, axis=1)
df_actions["type_id"] = df_events.apply(_determine_type_id, axis=1)
df_actions["result_id"] = df_events.apply(_determine_result_id, axis=1)
```
with:
```python
df_actions["bodypart_id"] = _vectorized_bodypart_id(df_events)
df_actions["type_id"] = _vectorized_type_id(df_events)
df_actions["result_id"] = _vectorized_result_id(df_events)
```

Remove the old `_determine_*` functions.

- [ ] **Step 3: Run Wyscout tests**

Run: `python -m pytest tests/spadl/test_wyscout.py -v`
Expected: all PASS

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all pass

---

## Task 7: Atomic Conversion Deferred Sort (O-13)

**Files:**
- Modify: `silly_kicks/atomic/spadl/base.py`

- [ ] **Step 1: Run existing atomic tests**

Run: `python -m pytest tests/atomic/ -m "not e2e" -v`
Expected: all pass

- [ ] **Step 2: Refactor to deferred single sort**

Rename `_extra_from_passes` → `_compute_pass_extras`, `_extra_from_shots` → `_compute_shot_extras`, `_extra_from_fouls` → `_compute_foul_extras`. Each function should:
- Accept the original actions DataFrame
- Return ONLY the extras DataFrame (not the concatenated result)
- NOT sort or renumber

Then update `convert_to_atomic`:

```python
def convert_to_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    atomic_actions = actions.copy()
    # Compute all extras from the original actions (before any injection)
    pass_extras = _compute_pass_extras(atomic_actions)
    shot_extras = _compute_shot_extras(atomic_actions)
    foul_extras = _compute_foul_extras(atomic_actions)
    # Single concat + sort + renumber
    atomic_actions = pd.concat(
        [atomic_actions, pass_extras, shot_extras, foul_extras],
        ignore_index=True, sort=False,
    )
    atomic_actions = atomic_actions.sort_values(
        ["game_id", "period_id", "action_id"]
    ).reset_index(drop=True)
    atomic_actions["action_id"] = range(len(atomic_actions))
    # Add dribbles after sorting (needs correct order)
    atomic_actions = _add_dribbles(atomic_actions)
    atomic_actions = _convert_columns(atomic_actions)
    atomic_actions = _simplify(atomic_actions)
    return _finalize_output(atomic_actions, ATOMIC_SPADL_COLUMNS)
```

Each `_compute_*_extras` function removes its internal `pd.concat + sort_values + action_id renumber` and just returns the extras DataFrame.

- [ ] **Step 3: Run atomic tests**

Run: `python -m pytest tests/atomic/ -m "not e2e" -v`
Expected: all pass

---

## Task 8: DEFERRED.md Update + Final Verification

**Files:**
- Modify: `docs/DEFERRED.md`

- [ ] **Step 1: Update DEFERRED.md**

Mark these items as resolved:

```
## Phase 3c: Converter Rewrite (2026-04-06)

- **O-2** (Wyscout 3x apply): RESOLVED — replaced with np.select
- **O-2b** (StatsBomb apply): RESOLVED — replaced with np.select
- **O-2c** (Opta 3x apply): RESOLVED — replaced with np.select
- **O-3** (Wyscout position extraction): RESOLVED — vectorized list indexing
- **O-4** (Wyscout 55x tag apply): RESOLVED — batch DataFrame constructor
- **O-5b** (StatsBomb dispatch dict per row): RESOLVED — module-level np.select
- **O-6** (StatsBomb coordinate for-loop): RESOLVED — numpy vectorized
- **O-8** (gamestates groupby.apply): RESOLVED — vectorized shift + boundary detection
- **O-12** (kloppy dispatch): RECLASSIFIED — kloppy EventDataset API, not actionable
- **O-13** (atomic 3x concat+sort): RESOLVED — single deferred sort
- **O-15** (config DataFrame caching): RESOLVED — @lru_cache
- **A1** (Wyscout god module): RESOLVED — split into 3 files
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass (78+ tests, 0 failures)

- [ ] **Step 3: Verify no apply(axis=1) remains in converter code**

Run: `grep -n "apply.*axis=1" silly_kicks/spadl/statsbomb.py silly_kicks/spadl/opta.py silly_kicks/spadl/wyscout.py silly_kicks/spadl/_wyscout_mappings.py`
Expected: zero matches
