# Phase 3b: Input/Output Contract — Design Spec

> **Goal:** Establish typed, documented, guaranteed I/O contracts for all
> converters. Eliminate luxury-lakehouse workarounds #4 (dtype coercions),
> #5 (column allowlisting), #6 (add_names column re-injection). Create
> foundation for eliminating workarounds #1-3 in Phase 3c.

## Context

silly-kicks inherits socceraction's implicit contracts: undocumented input
requirements, untyped output, pandera-enforced strict schemas that reject
caller-added columns, and silent `non_action` event drops. luxury-lakehouse
compensates with 6 workarounds (adapters, coercions, allowlists, re-injections).

Phase 3b makes these contracts explicit, typed, and guaranteed.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Schema enforcement | Drop pandera, use plain Python constants + optional `validate_spadl()` | Removes pandera + multimethod deps, unblocks numpy>=2.0, eliminates strict-mode column rejection |
| Output dtypes | Enforce int64/float64 for all numeric columns; `str` for `original_event_id` | Eliminates luxury-lakehouse's 7 dtype coercions |
| ID column types | int64 for StatsBomb/Wyscout/Opta; object (str) for kloppy | kloppy receives string UUIDs from upstream providers we don't control |
| Input validation | Document + validate per provider (column presence check) | Clear error messages instead of cryptic KeyErrors |
| Nothing Left Behind | Explicit mapping/excluded/unrecognized registries + ConversionReport | Every source event has an accounted fate |
| Converter return type | `tuple[pd.DataFrame, ConversionReport]` | Structured audit trail without changing the DataFrame |
| add_names() contract | Preserves all extra columns; documented guarantee | Eliminates match_id re-injection workaround |

## 1. Schema Constants (replaces pandera)

### 1.1 SPADL Schema

Replace `silly_kicks/spadl/schema.py` contents with:

```python
"""SPADL output schema — plain Python constants.

These constants define the guaranteed output contract of convert_to_actions().
They replace the pandera DataFrameModel that previously served this role.
"""

import dataclasses

SPADL_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "original_event_id": "object",  # always str
    "action_id": "int64",
    "period_id": "int64",
    "time_seconds": "float64",
    "team_id": "int64",
    "player_id": "int64",
    "start_x": "float64",
    "start_y": "float64",
    "end_x": "float64",
    "end_y": "float64",
    "type_id": "int64",
    "result_id": "int64",
    "bodypart_id": "int64",
}

# Name columns added by add_names() — not present in raw converter output
SPADL_NAME_COLUMNS: dict[str, str] = {
    "type_name": "object",
    "result_name": "object",
    "bodypart_name": "object",
}

# Valid value ranges
SPADL_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id": (1, 5),
    "time_seconds": (0, float("inf")),
    "start_x": (0, 105.0),
    "start_y": (0, 68.0),
    "end_x": (0, 105.0),
    "end_y": (0, 68.0),
}
```

### 1.2 Atomic-SPADL Schema

Replace `silly_kicks/atomic/spadl/schema.py` with the same pattern:

```python
ATOMIC_SPADL_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "original_event_id": "object",
    "action_id": "int64",
    "period_id": "int64",
    "time_seconds": "float64",
    "team_id": "int64",
    "player_id": "int64",
    "x": "float64",
    "y": "float64",
    "dx": "float64",
    "dy": "float64",
    "type_id": "int64",
    "bodypart_id": "int64",
}
```

### 1.3 Kloppy Schema Variant

kloppy's converter receives string UUIDs from upstream providers for ID columns.
A separate constant specifies kloppy's output dtypes:

```python
KLOPPY_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "game_id": "object",      # overrides: may be string
    "team_id": "object",      # overrides: may be string UUID
    "player_id": "object",    # overrides: may be string UUID
}
```

### 1.4 Validation Utility

Add `validate_spadl(df, schema=SPADL_COLUMNS)` to `spadl/utils.py`:

- Missing columns → raises `ValueError` (hard failure — data is unusable)
- Dtype mismatches → emits `warnings.warn` (soft failure — callers may have
  intentionally widened types, e.g. kloppy string IDs)
- Value range violations → emits `warnings.warn` (soft failure — coordinates
  slightly out of bounds are common in real data)
- Returns the DataFrame unchanged (for chaining)
- Extra columns beyond the schema → silently accepted

This function is **not called automatically** by converters. It is a utility for
callers who want to verify a DataFrame conforms to SPADL. The converters enforce
the contract by construction (see Section 2).

## 2. Guaranteed Output Contract

### 2.1 `_finalize_output()` Shared Utility

Add to `spadl/utils.py`:

```python
def _finalize_output(
    df: pd.DataFrame,
    schema: dict[str, str] = SPADL_COLUMNS,
) -> pd.DataFrame:
    """Project to declared columns and enforce dtypes."""
    # Select exactly the declared columns (drops any internal working columns)
    result = df[list(schema.keys())].copy()
    # Cast to declared dtypes
    for col, dtype in schema.items():
        result[col] = result[col].astype(dtype)
    return result
```

Every converter calls `_finalize_output(df_actions)` as its last step before
returning. This guarantees:

- Exactly the declared columns appear in output (no extras, no missing)
- Every column has the declared dtype (no more float64 ints, no dict columns)
- `original_event_id` is always `str` (via `astype("object")` after explicit
  `str()` conversion in each converter)

### 2.2 Atomic Output

`convert_to_atomic()` calls `_finalize_output(df, ATOMIC_SPADL_COLUMNS)`.

### 2.3 add_names() Contract

`add_names()` already preserves extra columns. Phase 3b adds:

- Explicit docstring guarantee: "All columns not in the SPADL schema are
  preserved unchanged."
- A test proving this (see Section 5)
- Remove the `DataFrame[SPADLSchema]` return type annotation (plain
  `pd.DataFrame`)

## 3. Input Validation

### 3.1 Per-Provider Input Constants

Each converter module defines its expected input columns:

**StatsBomb** (`spadl/statsbomb.py`):
```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "timestamp",
    "team_id", "player_id", "type_name", "location", "extra",
}
```

**Wyscout** (`spadl/wyscout.py`):
```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "milliseconds",
    "team_id", "player_id", "type_id", "subtype_id",
    "positions", "tags",
}
```

**Opta** (`spadl/opta.py`):
```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "minute", "second",
    "team_id", "player_id", "type_name", "outcome",
    "start_x", "start_y", "end_x", "end_y", "qualifiers",
}
```

**Kloppy** — not applicable (takes `EventDataset`, not `DataFrame`).

### 3.2 Entry Validation

Each DataFrame-based converter validates at the top of `convert_to_actions()`:

```python
def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> ...:
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="StatsBomb")
    ...
```

`_validate_input_columns()` (shared in `spadl/utils.py`):
- Missing columns → `ValueError` with clear message naming the missing columns
- Extra columns → silently accepted (callers may have enriched data)
- Dtype mismatches → no enforcement (input data is messy; converters handle
  coercion internally)

### 3.3 Evolution in Phase 3c

These input constants document the *current* expectations. Phase 3c (converter
rewrite) may change what columns are expected. The validation infrastructure
carries forward; only the column sets change.

## 4. Nothing Left Behind — Mapping Registry

### 4.1 Event Fate Categories

Every source event type has exactly one of three fates:

| Fate | Meaning | Audit |
|------|---------|-------|
| **Mapped** | Converted to one or more SPADL actions | Counted in `ConversionReport.mapped_counts` |
| **Excluded** | Intentionally dropped — has no on-ball action equivalent | Counted in `ConversionReport.excluded_counts` |
| **Unrecognized** | Not in mapped or excluded sets — dropped with warning | Counted in `ConversionReport.unrecognized_counts` + `warnings.warn()` |

### 4.2 Per-Provider Registries

Each converter module defines explicit frozensets:

```python
_MAPPED_EVENT_TYPES: frozenset[str] = frozenset({...})
_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset({...})
```

The dispatch function checks membership:
- In `_MAPPED_EVENT_TYPES` → route to parser
- In `_EXCLUDED_EVENT_TYPES` → count, skip
- In neither → count as unrecognized, `warnings.warn()`, skip

### 4.3 ConversionReport Dataclass

```python
@dataclasses.dataclass(frozen=True)
class ConversionReport:
    """Audit trail for convert_to_actions()."""
    provider: str
    total_events: int
    total_actions: int
    mapped_counts: dict[str, int]
    excluded_counts: dict[str, int]
    unrecognized_counts: dict[str, int]

    @property
    def has_unrecognized(self) -> bool:
        return len(self.unrecognized_counts) > 0
```

Lives in `silly_kicks/spadl/schema.py` (alongside the column constants).

### 4.4 Return Signature

All converters return a tuple:

```python
def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
) -> tuple[pd.DataFrame, ConversionReport]:
```

Callers that don't need the report can unpack with `_`:
```python
actions, _ = statsbomb.convert_to_actions(events, home_team_id)
```

### 4.5 StatsBomb Registry (illustrative)

```python
_MAPPED_EVENT_TYPES: frozenset[str] = frozenset({
    "Pass", "Dribble", "Carry", "Foul Committed", "Duel",
    "Interception", "Shot", "Own Goal Against", "Goal Keeper",
    "Clearance", "Miscontrol",
})

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset({
    # Meta-events with no on-ball action equivalent in SPADL
    "Pressure", "Ball Receipt*", "Block", "50/50",
    "Substitution", "Starting XI", "Tactical Shift",
    "Referee Ball-Drop", "Half Start", "Half End",
    "Injury Stoppage", "Player On", "Player Off", "Error",
    "Shield", "Camera On", "Camera off",
    "Bad Behaviour", "Ball Recovery",
})
```

Registries for Wyscout and Opta follow the same pattern, keyed on their
respective event type identifiers (string names for Opta, integer type_ids
for Wyscout).

## 5. Testing Strategy

### 5.1 Schema Contract Tests (`tests/spadl/test_schema.py`)

- `SPADL_COLUMNS` has exactly 14 entries
- `ATOMIC_SPADL_COLUMNS` has exactly 13 entries
- `validate_spadl()` accepts valid DataFrame, rejects missing columns
- `validate_spadl()` warns on dtype mismatch (does not raise)

### 5.2 Output Guarantee Tests (per converter)

- Output has exactly the columns in `SPADL_COLUMNS` (no more, no less)
- Each column has the declared dtype
- No `non_action` type_id in output
- `original_event_id` is always `str` dtype

### 5.3 Input Validation Tests (per converter)

- Missing required column → `ValueError` with provider name and missing column
- Extra columns accepted without error

### 5.4 ConversionReport Tests

- `total_events == sum(mapped) + sum(excluded) + sum(unrecognized)`
- Known test event types appear in `mapped_counts`
- Fabricated unknown event type appears in `unrecognized_counts`
- `has_unrecognized` property works correctly

### 5.5 add_names() Preservation Test

- Add custom column `"my_custom_col"` to SPADL DataFrame
- Call `add_names()` → assert `"my_custom_col"` is still present with same values

### 5.6 Pandera Removal Verification

- `pandera` not importable (not in dependencies)
- No `import pandera` in any source file
- `multimethod` not importable

## 6. Dependency Changes

### Remove
- `pandera[mypy]` (currently in `[dev]` extras or core deps)
- `multimethod<2.0` (pandera transitive dependency)

### No New Dependencies
All replacements use stdlib (`dataclasses`, `warnings`) and pandas/numpy.

## 7. Files Modified

```
Modified:
  silly_kicks/spadl/schema.py           — pandera model → plain constants
  silly_kicks/atomic/spadl/schema.py    — pandera model → plain constants
  silly_kicks/spadl/utils.py            — add _finalize_output, _validate_input_columns, validate_spadl
  silly_kicks/spadl/statsbomb.py        — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/wyscout.py          — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/opta.py             — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/kloppy.py           — mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/__init__.py         — update exports (remove SPADLSchema, add SPADL_COLUMNS, validate_spadl, ConversionReport)
  silly_kicks/atomic/spadl/base.py      — update call to _finalize_output
  silly_kicks/atomic/spadl/utils.py     — add_names docstring + remove pandera types
  silly_kicks/atomic/spadl/__init__.py  — update exports
  silly_kicks/vaep/base.py             — update type annotations
  silly_kicks/atomic/vaep/base.py      — update type annotations
  pyproject.toml                        — remove pandera/multimethod deps

Created:
  tests/spadl/test_schema.py            — schema contract tests
  tests/spadl/test_output_contract.py   — output guarantee tests per converter
  tests/spadl/test_input_validation.py  — input validation tests
  tests/spadl/test_conversion_report.py — ConversionReport tests

Modified (tests):
  tests/conftest.py                     — remove pandera fixture shim
  tests/spadl/test_statsbomb.py         — update for tuple return
  tests/spadl/test_wyscout.py           — update for tuple return
  tests/spadl/test_opta.py              — update for tuple return
  tests/vaep/test_features.py           — remove pandera type annotations
  tests/vaep/test_vaep.py              — remove pandera type annotations
```

## 8. What This Does NOT Change

- **Converter internals** — dispatch logic, coordinate transforms, dribble
  insertion all stay as-is. Phase 3c rewrites these.
- **SPADL vocabulary** — actiontypes, results, bodyparts lists unchanged.
- **Public function signatures** — same parameters, same names. Only the return
  type changes (plain DataFrame + ConversionReport instead of
  DataFrame[SPADLSchema]).
- **kloppy's EventDataset input** — stays architecturally different (DEFERRED A15).
- **Feature/label/VAEP functions** — no changes except removing pandera type
  annotations.

## 9. Luxury-Lakehouse Impact

After Phase 3b, luxury-lakehouse can:

| Workaround | Action |
|-----------|--------|
| #4 `_clean_spadl_for_spark` dtype coercions | **Delete entirely** — output dtypes are guaranteed |
| #5 `_spadl_cols` allowlist + guards | **Delete entirely** — output columns are guaranteed |
| #6 `match_id` re-injection after add_names() | **Delete** — add_names() preserves extras |
| #1 `_raw_extra_json` double API call | Partially addressed (input docs); full fix in 3c |
| #3 Adapter modules | Partially addressed (input docs); full fix in 3c |
| #7 competition/season injection | Unchanged — by-design per-game scope |
