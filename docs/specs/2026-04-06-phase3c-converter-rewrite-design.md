# Phase 3c: Converter Rewrite — Design Spec

> **Goal:** Vectorize all `apply(axis=1)` hot paths in every converter,
> decompose the Wyscout god module, cache config DataFrames, optimize
> atomic conversion, and fix the gamestates pandas 3.0 bug. Resolves
> deferred items A1, O-2, O-2b, O-2c, O-3, O-4, O-5b, O-6, O-8,
> O-12, O-13, O-15.

## Context

Phase 3b established typed I/O contracts — every converter now returns
`(DataFrame, ConversionReport)` with guaranteed columns and dtypes via
`_finalize_output()`. Phase 3c rewrites the converter internals for
performance without changing these contracts.

The codebase has 12+ `apply(axis=1)` calls across 3 converter modules, a
Python for-loop in StatsBomb coordinate conversion, 60 serial `apply()` passes
for Wyscout tag expansion, 3x concat+sort in atomic conversion, uncached config
DataFrame factories, and a pandas 3.0 compat bug in gamestates.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vectorization strategy | Pre-flatten nested dicts/lists to columns, then `np.select` | Consistent across all 3 DataFrame-based converters |
| Wyscout decomposition | 3-file moderate split | Enough isolation without over-splitting |
| Gamestates | Full vectorize (not quick compat fix) | Fixes bug + resolves O-8, one less item for Phase 3d |
| Kloppy | No change | Python loop is intrinsic to kloppy `EventDataset` API |
| Tag expansion optimization | Batch DataFrame constructor (save `MultiLabelBinarizer` for later) | Removes 60 serial `apply()` calls; bitmap optimization deferred |

## 1. Config DataFrame Caching (O-15)

### Problem

`actiontypes_df()`, `results_df()`, `bodyparts_df()` in `spadl/config.py` and
`atomic/spadl/config.py` reconstruct a fresh `pd.DataFrame` on every call. VAEP
feature functions call these repeatedly per game.

### Fix

Add `@functools.lru_cache(maxsize=None)` to all three functions in
`silly_kicks/spadl/config.py`. The atomic config delegates to SPADL's
`bodyparts_df` and re-implements `actiontypes_df()` — cache both.

```python
@functools.lru_cache(maxsize=None)
def actiontypes_df() -> pd.DataFrame:
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])
```

**Files:** `silly_kicks/spadl/config.py`, `silly_kicks/atomic/spadl/config.py`

## 2. Gamestates Vectorization (O-8)

### Problem

`vaep/features.py:gamestates()` uses `groupby().apply(as_index=False)` which
drops key columns (`game_id`, `period_id`) in pandas 3.0. This causes 4 test
failures.

### Fix

Replace `groupby().apply()` with vectorized shift + boundary detection:

```python
for i in range(1, nb_prev_actions):
    prev = actions.shift(i)
    # Detect period boundaries
    boundary = (
        (actions["game_id"] != actions["game_id"].shift(i))
        | (actions["period_id"] != actions["period_id"].shift(i))
    )
    # At boundaries, fill from group's first row
    first_in_group = actions.groupby(["game_id", "period_id"]).transform("first")
    prev[boundary] = first_in_group[boundary]
    states.append(prev)
```

This is O(N) vectorized, pandas-version-agnostic, and avoids the lambda
entirely.

**Files:** `silly_kicks/vaep/features.py`

## 3. StatsBomb Converter Vectorization (O-2b, O-5b, O-6)

### 3.1 Pre-flatten `extra` dict

At the top of `convert_to_actions()`, extract the ~15 nested fields actually
used into flat columns. Use safe `.str.get()` chains or a single-pass helper:

```python
def _flatten_extra(events: pd.DataFrame) -> pd.DataFrame:
    extra = events["extra"]
    events["_pass_outcome"] = extra.str.get("pass", {}).str.get("outcome", {}).str.get("name")
    events["_pass_type"] = extra.str.get("pass", {}).str.get("type", {}).str.get("name")
    events["_pass_height"] = extra.str.get("pass", {}).str.get("height", {}).str.get("name")
    events["_pass_cross"] = extra.str.get("pass", {}).str.get("cross")
    events["_pass_body_part"] = extra.str.get("pass", {}).str.get("body_part", {}).str.get("name")
    events["_pass_end_location"] = extra.str.get("pass", {}).str.get("end_location")
    events["_shot_outcome"] = extra.str.get("shot", {}).str.get("outcome", {}).str.get("name")
    events["_shot_type"] = extra.str.get("shot", {}).str.get("type", {}).str.get("name")
    events["_shot_body_part"] = extra.str.get("shot", {}).str.get("body_part", {}).str.get("name")
    events["_shot_end_location"] = extra.str.get("shot", {}).str.get("end_location")
    events["_dribble_outcome"] = extra.str.get("dribble", {}).str.get("outcome", {}).str.get("name")
    events["_gk_type"] = extra.str.get("goalkeeper", {}).str.get("type", {}).str.get("name")
    events["_gk_outcome"] = extra.str.get("goalkeeper", {}).str.get("outcome", {}).str.get("name")
    events["_gk_body_part"] = extra.str.get("goalkeeper", {}).str.get("body_part", {}).str.get("name")
    events["_foul_card"] = extra.str.get("foul_committed", {}).str.get("card", {}).str.get("name")
    events["_duel_type"] = extra.str.get("duel", {}).str.get("type", {}).str.get("name")
    events["_duel_outcome"] = extra.str.get("duel", {}).str.get("outcome", {}).str.get("name")
    events["_interception_outcome"] = extra.str.get("interception", {}).str.get("outcome", {}).str.get("name")
    events["_clearance_body_part"] = extra.str.get("clearance", {}).str.get("body_part", {}).str.get("name")
    events["_carry_end_location"] = extra.str.get("carry", {}).str.get("end_location")
    return events
```

Note: `.str.get()` on a Series of dicts is a vectorized operation — no `apply()`.

### 3.2 Replace `_parse_event` with `np.select`

The 11 parser functions become condition/choice arrays. Example for `type_id`:

```python
type_name = events["type_name"]
conditions = [
    type_name == "Pass",
    type_name == "Dribble",
    type_name == "Carry",
    ...
]
type_choices = [
    _resolve_pass_type(events),      # returns Series of type_id ints
    spadlconfig.actiontype_id["take_on"],
    spadlconfig.actiontype_id["dribble"],
    ...
]
actions["type_id"] = np.select(conditions, type_choices, default=spadlconfig.actiontype_id["non_action"])
```

Each `_resolve_*_type()` helper returns a vectorized Series using `np.select`
on the flat `_pass_type`, `_pass_height`, `_pass_cross` columns. Same pattern
for `result_id` and `bodypart_id`.

### 3.3 Vectorize `_get_end_location`

Replace the row-wise `apply` with:
```python
end_loc = events["_pass_end_location"].fillna(
    events["_shot_end_location"]
).fillna(
    events["_carry_end_location"]
).fillna(
    events["location"]
)
```

### 3.4 Vectorize `_convert_locations` (O-6)

Replace Python for-loop with numpy array operations:
```python
arr = np.array(locations.tolist(), dtype=float)  # shape (N, 2) or (N, 3)
xy = arr[:, :2]
is_three = np.array([len(loc) == 3 if isinstance(loc, list) else False for loc in locations])
y_offset = np.where(is_three, 0.05, cell_relative_center)
coordinates = np.empty((len(locations), 2), dtype=float)
coordinates[:, 0] = (xy[:, 0] - cell_relative_center) / _SB_FIELD_LENGTH * field_length
coordinates[:, 1] = field_width - (xy[:, 1] - y_offset) / _SB_FIELD_WIDTH * field_width
```

### 3.5 Vectorize `is_interception_pass`

Replace `df_events.apply(is_interception_pass, axis=1)` with:
```python
mask = events["_pass_type"] == "Interception"
```

**Files:** `silly_kicks/spadl/statsbomb.py`

## 4. Opta Converter Vectorization (O-2c)

### 4.1 Pre-explode qualifiers

At conversion start, extract the ~16 qualifier IDs actually checked into
boolean columns:

```python
_USED_QUALIFIER_IDS: list[int] = [1, 2, 3, 5, 6, 9, 15, 20, 21, 26, 28, 32, 72, 107, 124, 155, 168, 238]

for qid in _USED_QUALIFIER_IDS:
    events[f"q_{qid}"] = events["qualifiers"].apply(lambda q, k=qid: k in q)
```

### 4.2 Replace `_get_type_id` with `np.select`

The if/elif chain on `eventname` + qualifier booleans becomes:

```python
tn = events["type_name"]
conditions = [
    events["q_238"],                                    # fairplay → non_action
    (tn.isin(["pass", "offside pass"])) & events["q_107"],  # throw_in
    (tn.isin(["pass", "offside pass"])) & events["q_5"] & (events["q_2"] | events["q_1"] | events["q_155"]),  # freekick_crossed
    ...
]
```

### 4.3 Replace `_get_result_id` and `_get_bodypart_id` similarly

Same `np.select` pattern over `type_name` + qualifier booleans.

**Files:** `silly_kicks/spadl/opta.py`

## 5. Wyscout Decomposition + Vectorization (A1, O-2, O-3, O-4)

### 5.1 File split

| File | Contents | ~Lines |
|------|----------|--------|
| `wyscout.py` | Entry point, constants, registries, `wyscout_tags` list | ~150 |
| `_wyscout_events.py` | Event fixes: `_fix_wyscout_events`, `_convert_duels`, `_insert_interceptions`, `_add_offside_variable`, `_convert_simulations`, `_convert_touches`, `_create_shot_coordinates` | ~400 |
| `_wyscout_mappings.py` | SPADL mapping: `_create_df_actions`, vectorized `_determine_type_id`, `_determine_result_id`, `_determine_bodypart_id`, `_remove_non_actions` | ~200 |

The coordinate extraction (`_make_new_positions`) moves to `_wyscout_events.py`
and is vectorized there.

### 5.2 Tag expansion optimization (O-4)

Replace 60 serial `apply()` calls with a single batch DataFrame constructor:

```python
tag_sets = events.tags.apply(_get_tag_set)  # 1 apply call
tagsdf = pd.DataFrame(
    {name: tag_sets.apply(lambda s, t=tid: t in s) for tid, name in wyscout_tags}
)
```

**Future optimization (not this phase):** Replace with `MultiLabelBinarizer` or
a set-to-bitmap approach for O(N) single-pass expansion.

### 5.3 Position extraction optimization (O-3)

Replace `_make_position_vars` apply with vectorized list indexing:

```python
pos_list = events["positions"].tolist()
start_pos = [p[0] if len(p) >= 1 else {"x": None, "y": None} for p in pos_list]
end_pos = [p[1] if len(p) >= 2 else p[0] if len(p) >= 1 else {"x": None, "y": None} for p in pos_list]
events["start_x"] = pd.Series([p["x"] for p in start_pos], dtype=float)
events["start_y"] = pd.Series([p["y"] for p in start_pos], dtype=float)
events["end_x"] = pd.Series([p["x"] for p in end_pos], dtype=float)
events["end_y"] = pd.Series([p["y"] for p in end_pos], dtype=float)
```

### 5.4 Vectorize dispatch functions (O-2)

Replace `_determine_type_id`, `_determine_result_id`, `_determine_bodypart_id`
(all `apply(axis=1)`) with `np.select` chains on the pre-existing boolean tag
columns (`fairplay`, `own_goal`, `left_foot`, `right_foot`, `head/body`,
`sliding_tackle`, `interception`, etc.) and `type_id`/`subtype_id` integers.

The logic is already column-based in the row-wise functions — the conversion to
`np.select` is mechanical.

**Files:** `silly_kicks/spadl/wyscout.py`, `silly_kicks/spadl/_wyscout_events.py` (new), `silly_kicks/spadl/_wyscout_mappings.py` (new)

## 6. Atomic Conversion Optimization (O-13)

### Problem

`_extra_from_passes`, `_extra_from_shots`, `_extra_from_fouls` each do
`pd.concat + sort_values + renumber action_id`. Three intermediate sorts are
wasted.

### Fix

Restructure to compute all three extras against the original actions DataFrame,
then do a single final concat + sort + renumber:

```python
pass_extras = _compute_pass_extras(actions)
shot_extras = _compute_shot_extras(actions)
foul_extras = _compute_foul_extras(actions)
actions = pd.concat([actions, pass_extras, shot_extras, foul_extras], ignore_index=True, sort=False)
actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
actions["action_id"] = range(len(actions))
```

The `_extra_from_shots` function currently uses `actions.shift(-1)` after
pass extras have been injected. For the deferred approach, `_compute_shot_extras`
must read from the original actions — and needs to handle the "next action"
lookup from the original (pre-injection) DataFrame. Since shot extras check
`next_actions.type_id` for corner/goalkick/keeper types, and pass extras insert
*between* existing actions (at `action_id + 0.1`), the next-action from the
original DataFrame is still the correct "next real action." This is safe.

The `_add_dribbles` call happens after concat+sort, as before.

**Files:** `silly_kicks/atomic/spadl/base.py`

## 7. Kloppy — No Change (O-12)

The kloppy converter iterates a kloppy `EventDataset` object in a Python
for-loop. This is the kloppy domain API — not a pandas `apply()`. The dispatch
dict is already at module level after Phase 3b. No optimization is possible
without changing the kloppy API.

**O-12 is reclassified as "not actionable" in DEFERRED.md.**

## 8. Testing Strategy

### Behavioral equivalence

All existing converter tests (inline + any fixture-based) must pass unchanged.
The vectorized code must produce identical output to the row-wise code. The
Phase 3b output contract tests (`test_output_contract.py`) verify column
presence, dtypes, and values.

### Gamestates regression

The 4 currently-failing tests (`test_same_index` ×2, `test_time`,
`test_player_possession_time`) become the acceptance criteria. They must pass.

### New tests

- **Config caching**: Verify `actiontypes_df() is actiontypes_df()` (same
  object identity — lru_cache returns the cached instance).
- **Wyscout decomposition**: Verify that importing `from silly_kicks.spadl.wyscout import convert_to_actions` still works after the file split.
- **Coordinate conversion equivalence**: A parameterized test comparing for-loop
  vs. vectorized `_convert_locations` on a batch of known coordinates.

### What is NOT tested

- Performance benchmarks are not enforced in CI. Vectorization correctness is
  tested; speedup is validated manually.

## 9. Files Modified/Created

```
Modified:
  silly_kicks/spadl/config.py              — add @lru_cache
  silly_kicks/atomic/spadl/config.py       — add @lru_cache
  silly_kicks/vaep/features.py             — vectorize gamestates()
  silly_kicks/spadl/statsbomb.py           — flatten extra, np.select, vectorize coords
  silly_kicks/spadl/opta.py                — explode qualifiers, np.select
  silly_kicks/spadl/wyscout.py             — slim down to entry point + constants
  silly_kicks/atomic/spadl/base.py         — deferred single sort
  docs/DEFERRED.md                         — mark 12 items resolved

Created:
  silly_kicks/spadl/_wyscout_events.py     — event fix pipeline
  silly_kicks/spadl/_wyscout_mappings.py   — vectorized SPADL mapping
```

## 10. What This Does NOT Change

- **I/O contracts from Phase 3b** — all converters still return
  `tuple[pd.DataFrame, ConversionReport]` with guaranteed columns/dtypes
- **SPADL vocabulary** — no new action types, results, or bodyparts
- **Public API** — same function signatures, same imports
- **Kloppy internals** — no change (O-12 reclassified)
- **VAEP model code** — no changes to fit/predict/rate
