# Block-detection converter columns (`shot_blocked` + `cross_blocked`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two nullable-boolean SPADL output columns — `shot_blocked` and `cross_blocked` — emitted by every converter (real value where the provider encodes it, `pd.NA` where it does not), so TF-51's `shot_block` rule and bravery metric read a stable cross-provider signal instead of re-deriving it.

**Architecture:** Both columns join the shared `SPADL_COLUMNS` schema (dtype `"boolean"`), which propagates to every provider schema via `{**SPADL_COLUMNS, ...}` spread. A shared `_blocked_flag(n, applicable, blocked)` helper builds the nullable-boolean column with the correct 3-valued semantics (`True`/`False` on shot/cross rows, `pd.NA` elsewhere). Each converter sets both columns before its `_finalize_output(...)` call; `_finalize_output` is a strict projection (`df[cols]`) so a missing column is a `KeyError`, which is why the schema change and the all-converter emission land in one commit (Task 2). Feasible providers then get their real mask (Tasks 3–8); infeasible ones (Opta, SkillCorner) and the deferred `cross_blocked` (StatsBomb) stay `pd.NA`.

**Tech Stack:** Python, pandas (nullable `"boolean"` extension dtype), numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-22-block-detection-converter-columns-design.md`

**Scope note:** additive only — no existing column or value changes → **no VAEP/tracking retrain**. C4-free (converter columns, not aggregators). The atomic converter uses its own `ATOMIC_SPADL_COLUMNS` and drops non-atomic extras, so it is **unaffected**.

---

### Task 1: Shared `_blocked_flag` nullable-boolean helper

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (add `_blocked_flag` near `_finalize_output`, ~line 1559)
- Test: `tests/spadl/test_utils.py` (or create `tests/spadl/test_blocked_flag.py` if `test_utils.py` absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/spadl/test_blocked_flag.py
import numpy as np
import pandas as pd

from silly_kicks.spadl.utils import _blocked_flag


def test_all_na_when_not_applicable():
    col = _blocked_flag(3)
    assert str(col.dtype) == "boolean"
    assert col.isna().all()


def test_true_false_on_applicable_na_elsewhere():
    # rows 0,2 are shots (applicable); row 0 blocked, row 2 not; row 1 is a non-shot.
    applicable = np.array([True, False, True])
    blocked = np.array([True, False, False])
    col = _blocked_flag(3, applicable=applicable, blocked=blocked)
    assert str(col.dtype) == "boolean"
    assert col[0] is True or col[0] == True  # noqa: E712  blocked shot
    assert pd.isna(col[1])  # non-shot -> NA, never False
    assert col[2] == False  # noqa: E712  shot, not blocked


def test_nan_in_blocked_coerced_to_false_not_true():
    # P-3: astype(bool) would turn NaN -> True; the helper must coerce NaN -> False.
    applicable = np.array([True, True])
    blocked = pd.array([True, pd.NA], dtype="boolean")  # a NaN-bearing blocked signal
    col = _blocked_flag(2, applicable=applicable, blocked=blocked)
    assert col[0] == True  # noqa: E712
    assert col[1] == False  # noqa: E712  NA -> False, NOT True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/spadl/test_blocked_flag.py -v`
Expected: FAIL with `ImportError: cannot import name '_blocked_flag'`

- [ ] **Step 3: Write the helper**

```python
# silly_kicks/spadl/utils.py  (add near _finalize_output)
def _blocked_flag(
    n: int,
    *,
    applicable: "np.ndarray | pd.Series | None" = None,
    blocked: "np.ndarray | pd.Series | None" = None,
) -> pd.arrays.BooleanArray:
    """Build a length-``n`` nullable-boolean ``"boolean"`` block-flag column.

    - ``applicable=None`` -> every row ``pd.NA`` (the provider cannot encode the signal).
    - otherwise -> ``blocked`` (True/False) on the rows where ``applicable`` is True,
      ``pd.NA`` everywhere else. A non-shot / non-cross row is *not applicable* -> ``pd.NA``,
      never ``False`` (spec Column-contract §3: ``pd.NA`` = "not a shot/cross OR unknown provider").

    ``applicable`` must be a clean boolean mask; ``blocked`` may carry NA -> it is coerced to
    ``False`` (P-3: a bare ``np.asarray(blocked, dtype=bool)`` would turn NA -> True, the repo's
    documented ``astype(bool)`` string/NaN trap).
    """
    values = np.full(n, None, dtype=object)
    if applicable is not None:
        mask = np.asarray(applicable, dtype=bool)
        blocked_clean = pd.array(blocked, dtype="boolean").fillna(False).to_numpy(dtype=bool)
        values[mask] = blocked_clean[mask]
    return pd.array(values, dtype="boolean")
```

(Confirm `import numpy as np` and `import pandas as pd` are already present at the top of `utils.py` — they are.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/spadl/test_blocked_flag.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/spadl/utils.py tests/spadl/test_blocked_flag.py
git commit -m "feat(spadl): _blocked_flag nullable-boolean block-flag helper (TF-51 prereq)"
```

---

### Task 2: Schema foundation — register the columns and scaffold `pd.NA` emission in all 8 converters

This is the one coupled task: `SPADL_COLUMNS` gains both keys, so **every** converter's strict-projection `_finalize_output` requires them present. We emit `pd.NA` everywhere here (real masks come in Tasks 3–8), keeping the suite green.

**Files:**
- Modify: `silly_kicks/spadl/schema.py` (add 2 keys to `SPADL_COLUMNS`, ~line 24)
- Modify: `silly_kicks/reflection.py` (add 2 `"invariant"` keys, ~line 186; bump the "32 columns" docstring)
- Modify (emit `pd.NA`): `gradientsports.py`, `statsbomb.py`, `wyscout.py`/`_wyscout_mappings.py`, `sportec.py`, `metrica.py`, `kloppy.py`, `opta.py`, `skillcorner.py`
- Test: `tests/spadl/test_schema.py:16` (14→16), `tests/test_reflection.py` (32→34)

- [ ] **Step 1: Add both columns to the base schema**

```python
# silly_kicks/spadl/schema.py  — inside SPADL_COLUMNS, after "bodypart_id": "int64",
    "type_id": "int64",
    "result_id": "int64",
    "bodypart_id": "int64",
    # Block-detection columns (TF-51 prereq). Nullable: True/False on shot/cross rows
    # where the provider encodes a block, pd.NA on non-shot/non-cross rows AND on
    # providers that cannot encode the signal (Opta, SkillCorner). See the block-detection spec.
    "shot_blocked": "boolean",
    "cross_blocked": "boolean",
```

- [ ] **Step 2: Declare both as invariant in the reflection registry**

```python
# silly_kicks/reflection.py — inside _SPADL_REFLECTION_KINDS, alongside the other
# provider-variant "invariant" entries (~line 186, after "tackle_loser_team_id"):
    "tackle_loser_team_id": "invariant",
    "shot_blocked": "invariant",   # TF-51 prereq: boolean, not geometric -> invariant under reflection
    "cross_blocked": "invariant",
```

Also bump the module docstring count in `reflection.py` (the line reading `32 columns: the 14 canonical, ...`) from `32`/`7 provider-variant` to `34`/`9 provider-variant`.

- [ ] **Step 3: Emit `pd.NA` in every converter, immediately before its `_finalize_output` call.** Use `_blocked_flag(len(<df>))` (all-NA). Import it in each module: `from .utils import _blocked_flag` (add to the existing `from .utils import _finalize_output` line, or a new import).

Insert points (each is right before the finalize call; `n` = length of the DataFrame being finalized):

**gradientsports.py** — in the dict literal at ~line 534–563 (`pd.DataFrame({...})`), add two keys built from the `events`-length. Since `_blocked_flag` needs `n`, compute it once before the dict: `n_ev = len(events)` and add `"shot_blocked": _blocked_flag(n_ev), "cross_blocked": _blocked_flag(n_ev),`. The empty-input fast path (line ~506, `pd.DataFrame({col: [] for col in GRADIENTSPORTS_SPADL_COLUMNS.keys()})`) auto-includes them (schema-driven) — no edit.

**statsbomb.py** — after `actions["bodypart_id"] = _vectorized_bodypart_id(events)` (~line 275), before the `type_id != non_action` filter (~277):
```python
actions["shot_blocked"] = _blocked_flag(len(actions))
actions["cross_blocked"] = _blocked_flag(len(actions))
```

**_wyscout_mappings.py** — inside `_create_df_actions`, after `df_actions["result_id"] = ...` (~line 284):
```python
df_actions["shot_blocked"] = _blocked_flag(len(df_actions))
df_actions["cross_blocked"] = _blocked_flag(len(df_actions))
```
(Import `_blocked_flag` into `_wyscout_mappings.py`.)

**sportec.py** — in the `_build_raw_actions` dict literal (~line 1004–1024), add `"shot_blocked": _blocked_flag(n), "cross_blocked": _blocked_flag(n),` (use the same row-count variable the dict uses; the empty path `_empty_raw_actions` iterating `SPORTEC_SPADL_COLUMNS.items()` auto-includes them). Also add the two keys to the synthesized-GK-distribution dict (~line 1121–1139) as `_blocked_flag(<its length>)` so those rows are `pd.NA` (they are never shots).

**metrica.py** — in the `_build_raw_actions` dict literal (~line 511–527), add `"shot_blocked": _blocked_flag(n), "cross_blocked": _blocked_flag(n),`; add the same to the synthesized-GK-pass dict (~line 595–611). Empty path auto-covered.

**kloppy.py** — add a helper and merge it into the per-event dict (mirrors `_get_end_location`):
```python
def _get_blocked_flags(event) -> dict[str, object]:
    return {"shot_blocked": pd.NA, "cross_blocked": pd.NA}
```
and in the loop (~line 213), add `**_get_blocked_flags(event)` alongside `**_get_end_location(event)`. (Real `ShotResult.BLOCKED` wiring lands in Task 8.) After `pd.DataFrame(actions)` the columns are object-`NA`; `_finalize_output`'s cast to `"boolean"` normalizes them.

**opta.py** — immediately before `_finalize_output` (~line 219):
```python
actions["shot_blocked"] = _blocked_flag(len(actions))
actions["cross_blocked"] = _blocked_flag(len(actions))
```

**skillcorner.py** — immediately before `_finalize_output` (~line 581), **after** the `pd.concat(parts, ...)` (~535) and `_add_dribbles`/`to_spadl_ltr` (do NOT add to the native dict at 449–468 — the `[actions.columns]` reindex at lines 531/533 would `KeyError` on the derived frames):
```python
actions["shot_blocked"] = _blocked_flag(len(actions))
actions["cross_blocked"] = _blocked_flag(len(actions))
```

- [ ] **Step 4: Update the count/column-set assertions (P-2: enumerated — most auto-adjust, three are hardcoded)**

Every provider column-set test compares against `list(X_SPADL_COLUMNS.keys())`, so it **auto-adjusts** once the schema includes the two columns — no edit needed at `test_gradientsports.py:178,200`, `test_opta.py:48,212`, `test_metrica.py:70,81`, `test_sportec.py:58,70,1183`, `test_kloppy.py:127,193`, `test_wyscout.py:61,66,385`, `test_statsbomb.py:124,247,254`, `test_skillcorner_e2e.py:51`, `test_schema.py:114,117`. Only these need manual edits:

```python
# tests/spadl/test_schema.py:16
assert len(SPADL_COLUMNS) == 16
```
```python
# tests/test_reflection.py:235  (test_meta_every_known_spadl_column_declares_a_kind)
assert len(known) == 34, f"expected the measured 34-column surface, got {len(known)}"
```
```python
# tests/spadl/test_skillcorner.py:271 — assert set(actions.columns) == expected_cols
# Inspect the `expected_cols` construction just above: if it is a HARDCODED set, add
# "shot_blocked" and "cross_blocked" to it; if it is set(SKILLCORNER_SPADL_COLUMNS.keys())
# it auto-adjusts (no edit).
```

- [ ] **Step 5: Run the schema + reflection tests**

Run: `python -m pytest tests/spadl/test_schema.py tests/test_reflection.py -v`
Expected: PASS

- [ ] **Step 6: Run the full converter suite; fix any column-set/snapshot assertions**

Run: `python -m pytest tests/spadl tests/ -m "not e2e" -k "spadl or converter or convert" -q`
Expected: PASS. If any test asserts an exact `list(actions.columns)` or a golden snapshot/hash, add `shot_blocked`, `cross_blocked` (both `pd.NA`) to its expectation / regenerate the snapshot. (Most converter tests assert specific column *values*, not the set, so few if any will need this.)

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/spadl/schema.py silly_kicks/reflection.py silly_kicks/spadl/*.py tests/spadl/test_schema.py tests/test_reflection.py
git commit -m "feat(spadl): register shot_blocked/cross_blocked schema columns; all converters emit pd.NA (TF-51 prereq)"
```

---

### Task 3: Gradient Sports — real `shot_blocked` + `cross_blocked` masks

GS `shot_outcome_type=="B"` = blocked shot; `cross_outcome_type=="B"` = blocked cross (pining-probed: 6/39 crosses on real WC2022). `possession_event_type` "SH"=shot, "CR"=cross.

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py` (dict literal ~line 534–563; synthesized rows ~677–716)
- Test: `tests/spadl/test_gradientsports.py`

- [ ] **Step 1: Write the failing test** (pattern mirrors the existing `test_shot_result_mapping[B-fail]` and `TestGradientsportsCrossDispatch`)

```python
# tests/spadl/test_gradientsports.py
def test_shot_blocked_true_on_B_shot():
    from silly_kicks.spadl import gradientsports
    df = _df_minimal_pass()  # existing helper in this test module
    df.loc[0, "possession_event_type"] = "SH"
    df.loc[0, "shot_outcome_type"] = "B"
    actions, _ = gradientsports.convert_to_actions(df, home_team_id=_HOME)
    row = actions[actions["type_name"].str.contains("shot")].iloc[0]
    assert row["shot_blocked"] == True  # noqa: E712
    assert pd.isna(row["cross_blocked"])  # a shot row -> cross_blocked NA


def test_cross_blocked_true_on_B_cross():
    from silly_kicks.spadl import gradientsports
    df = _df_minimal_pass()
    df.loc[0, "possession_event_type"] = "CR"
    df.loc[0, "cross_outcome_type"] = "B"
    actions, _ = gradientsports.convert_to_actions(df, home_team_id=_HOME)
    row = actions[actions["type_name"] == "cross"].iloc[0]
    assert row["cross_blocked"] == True  # noqa: E712
    assert pd.isna(row["shot_blocked"])


def test_setpiece_cross_blocked_is_na():
    # P-4: a set-piece corner delivery (CR + set_piece corner code) dispatches to corner_crossed,
    # which is OUT of open-play cross scope -> cross_blocked stays pd.NA even with outcome "B".
    from silly_kicks.spadl import gradientsports
    df = _df_minimal_pass()
    df.loc[0, "possession_event_type"] = "CR"
    df.loc[0, "set_piece_type"] = "C"  # corner (confirm the code vs gradientsports.py:193-195)
    df.loc[0, "cross_outcome_type"] = "B"
    actions, _ = gradientsports.convert_to_actions(df, home_team_id=_HOME)
    row = actions[actions["type_name"] == "corner_crossed"].iloc[0]
    assert pd.isna(row["cross_blocked"])
```

(Use the real minimal-event + `_HOME` helpers already in `tests/spadl/test_gradientsports.py`; `_df_minimal_pass` and the cross-dispatch pattern at `TestGradientsportsCrossDispatch` show the exact fixture shape.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_gradientsports.py -k "shot_blocked_true or cross_blocked_true" -v`
Expected: FAIL (both are currently `pd.NA` from Task 2)

- [ ] **Step 3: Replace the Task-2 `pd.NA` with the real masks.** Before the dict literal (~line 534), compute the masks from `events` (aligned to the dict rows). **P-4 — `cross_blocked` is scoped to the open-play `cross` SPADL type only** (spec BD-3): a `pe=="CR"` event that is a set-piece corner/freekick delivery dispatches to `corner_crossed`/`freekick_crossed` and must stay `pd.NA`, so exclude those using the **same set-piece codes as the cross dispatch at `gradientsports.py:193-195`**:

```python
_pe = events["possession_event_type"].fillna("").to_numpy()
_sp = events["set_piece_type"].fillna("").to_numpy()
_shot_outcome = events["shot_outcome_type"].fillna("").to_numpy()
_cross_outcome = events["cross_outcome_type"].fillna("").to_numpy()
_n_ev = len(events)
# open-play cross = a CR possession-event that is NOT a set-piece delivery.
# _GS_SETPIECE_CROSS_CODES = the set_piece_type letters the dispatch uses for corner/freekick
# crosses at gradientsports.py:193-195 (e.g. {"C", "F"} — CONFIRM against that exact code).
_open_play_cross = (_pe == "CR") & ~np.isin(_sp, _GS_SETPIECE_CROSS_CODES)
```
In the dict literal, replace the two Task-2 entries with:
```python
    "shot_blocked": _blocked_flag(_n_ev, applicable=(_pe == "SH"), blocked=(_shot_outcome == "B")),
    "cross_blocked": _blocked_flag(_n_ev, applicable=_open_play_cross, blocked=(_cross_outcome == "B")),
```

- [ ] **Step 4: Reset the block flags on synthesized rows.** The foul/cross-goal-shot synthesis (~677–716) `.copy()`s parent rows, so a synthesized row would inherit the parent's flags. After the `pd.concat` that adds `synth_mask` rows, set them to `pd.NA` (they are not real block observations):

```python
# after the synthesis pd.concat (~line 716), on the combined `actions`:
_synth = actions["is_synthetic"].to_numpy()
actions.loc[_synth, "shot_blocked"] = pd.NA
actions.loc[_synth, "cross_blocked"] = pd.NA
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/spadl/test_gradientsports.py -k "shot_blocked or cross_blocked" -v`
Expected: PASS

- [ ] **Step 6: Full GS suite (no regressions)**

Run: `python -m pytest tests/spadl/test_gradientsports.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/spadl/gradientsports.py tests/spadl/test_gradientsports.py
git commit -m "feat(spadl): gradientsports shot_blocked/cross_blocked from crossOutcomeType/shotOutcomeType B (TF-51 prereq)"
```

---

### Task 4: StatsBomb — real `shot_blocked` (cross deferred)

`shot.outcome.name == "Blocked"` (already parsed into `events["_shot_outcome"]`). `cross_blocked` stays `pd.NA` (BD-2: the `related_events` join is deferred). Real fixture `7298.json` has 12 blocked shots.

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py` (~line 275)
- Test: `tests/spadl/test_statsbomb.py` (+ the real-fixture loader in `tests/invariants/_loaders.py:63`)

- [ ] **Step 1: Write the failing test** (real fixture)

```python
# tests/spadl/test_statsbomb.py
def test_shot_blocked_true_on_real_blocked_shot():
    from tests.invariants._loaders import load_statsbomb
    actions = load_statsbomb("7298")  # 12 real blocked shots
    shots = actions[actions["type_name"] == "shot"]
    assert (shots["shot_blocked"] == True).sum() == 12  # noqa: E712
    # every shot row is True/False (never NA on a provider that encodes it)
    assert shots["shot_blocked"].notna().all()
    # non-shots + the deferred cross column are NA
    assert actions["cross_blocked"].isna().all()
```

(Confirm the `load_statsbomb` helper signature in `tests/invariants/_loaders.py:63`; it returns the converted actions.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_statsbomb.py -k shot_blocked -v`
Expected: FAIL (currently all `pd.NA`)

- [ ] **Step 3: Replace the Task-2 `shot_blocked` `pd.NA` with the real mask** (leave `cross_blocked` as `_blocked_flag(len(actions))`):

```python
actions["shot_blocked"] = _blocked_flag(
    len(actions),
    applicable=(events["type_name"] == "Shot").to_numpy(),
    blocked=(events["_shot_outcome"] == "Blocked").to_numpy(),
)
actions["cross_blocked"] = _blocked_flag(len(actions))  # deferred (BD-2)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/spadl/test_statsbomb.py -k shot_blocked -v`
Expected: PASS

- [ ] **Step 5: Full StatsBomb suite**

Run: `python -m pytest tests/spadl/test_statsbomb.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/statsbomb.py tests/spadl/test_statsbomb.py
git commit -m "feat(spadl): statsbomb shot_blocked from shot.outcome Blocked; cross deferred (TF-51 prereq)"
```

---

### Task 5: Wyscout — real `shot_blocked` + `cross_blocked` (tag 2101)

Tag 2101 → `df_events["blocked"]` (shot+pass-shared). Scope to shots (`type_id == _WS_TYPE_SHOT`) and crosses (`_WS_TYPE_PASS` & `_WS_SUBTYPE_CROSS`). No real fixture → synthetic events. Emit inside `_create_df_actions` (the column-projection at line 280 drops `blocked` otherwise).

**Files:**
- Modify: `silly_kicks/spadl/_wyscout_mappings.py` (~line 284, inside `_create_df_actions`)
- Test: `tests/spadl/test_wyscout.py`

- [ ] **Step 1: Write the failing test** (synthetic shot + cross carrying tag 2101)

```python
# tests/spadl/test_wyscout.py
def test_shot_and_cross_blocked_from_tag_2101():
    from silly_kicks.spadl import wyscout
    events = pd.DataFrame([
        _make_shot_event(tags=[{"id": 2101}]),   # blocked shot   (see helper note below)
        _make_cross_event(tags=[{"id": 2101}]),   # blocked cross
        _make_shot_event(tags=[]),                 # non-blocked shot
    ])
    actions, _ = wyscout.convert_to_actions(events, home_team_id=_HOME)
    shots = actions[actions["type_name"] == "shot"]
    crosses = actions[actions["type_name"] == "cross"]
    assert (shots["shot_blocked"] == True).sum() == 1   # noqa: E712
    assert (shots["shot_blocked"] == False).sum() == 1  # noqa: E712
    assert (crosses["cross_blocked"] == True).all()
    assert shots["cross_blocked"].isna().all()  # a shot -> cross_blocked NA
```

Build `_make_shot_event` / `_make_cross_event` following the existing `_make_pass_event` helper (`tests/spadl/test_wyscout.py:9-26`) — a shot uses the shot type id and a cross uses `"subtype_id": 80` (`_WS_SUBTYPE_CROSS`), each with a `"tags"` list.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_wyscout.py -k blocked -v`
Expected: FAIL (currently `pd.NA`)

- [ ] **Step 3: Replace the Task-2 lines in `_create_df_actions` (after `df_actions["result_id"] = ...`, ~line 284)** with the real masks (use `df_events`, still index-aligned; `_WS_TYPE_SHOT`/`_WS_TYPE_PASS`/`_WS_SUBTYPE_CROSS` are already imported):

```python
df_actions["shot_blocked"] = _blocked_flag(
    len(df_actions),
    applicable=(df_events["type_id"] == _WS_TYPE_SHOT).to_numpy(),
    blocked=df_events["blocked"].to_numpy(),
)
df_actions["cross_blocked"] = _blocked_flag(
    len(df_actions),
    applicable=(
        (df_events["type_id"] == _WS_TYPE_PASS) & (df_events["subtype_id"] == _WS_SUBTYPE_CROSS)
    ).to_numpy(),
    blocked=df_events["blocked"].to_numpy(),
)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/spadl/test_wyscout.py -k blocked -v`
Expected: PASS

- [ ] **Step 5: Full Wyscout suite**

Run: `python -m pytest tests/spadl/test_wyscout.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/_wyscout_mappings.py tests/spadl/test_wyscout.py
git commit -m "feat(spadl): wyscout shot_blocked/cross_blocked from tag 2101 (TF-51 prereq)"
```

---

### Task 6: Sportec/DFL — real `shot_blocked` (minus own-team deflections)

`shot_outcome_type == "blocked"`, but `False` where `shot_outcome_blocked_by_own_team` is truthy (own-team deflection ≠ opponent block). `cross_blocked` = `pd.NA`. Real fixture `idsse/per_period_match.parquet`: 4 blocked shots, exactly one with `blocked_by_own_team="true"` (event `18226500001292` → `shot_blocked==False`).

**Files:**
- Modify: `silly_kicks/spadl/sportec.py` (`_build_raw_actions`, dict literal ~line 1004)
- Test: `tests/spadl/test_sportec.py` (+ `tests/invariants/_loaders.py:291` real loader)

- [ ] **Step 1: Write the failing test**

```python
# tests/spadl/test_sportec.py
def test_shot_blocked_excludes_own_team_deflection():
    from tests.invariants._loaders import load_sportec_native_per_period
    actions = load_sportec_native_per_period()
    shots = actions[actions["type_name"].str.contains("shot")]
    # 4 shot_outcome=="blocked" rows, one is an own-team deflection -> 3 True, that one False
    assert (shots["shot_blocked"] == True).sum() == 3   # noqa: E712
    assert (shots["shot_blocked"] == False).sum() >= 1  # noqa: E712  (incl. the own-team deflection)
    assert actions["cross_blocked"].isna().all()
```

(Confirm the loader name/signature at `tests/invariants/_loaders.py:291`.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_sportec.py -k shot_blocked -v`
Expected: FAIL

- [ ] **Step 3: Replace the Task-2 `pd.NA` in the `_build_raw_actions` dict.** Before the dict (~line 1004), compute (`is_shot` ~905, `shot_outcome` ~915, `_opt` closure ~808 are all in scope):

```python
_own_team_block = (
    _opt("shot_outcome_blocked_by_own_team", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
)
_shot_blocked_mask = (shot_outcome == "blocked") & ~_own_team_block
```
In the dict literal, replace the two entries:
```python
    "shot_blocked": _blocked_flag(n, applicable=is_shot, blocked=_shot_blocked_mask),
    "cross_blocked": _blocked_flag(n),  # no DFL blocked-cross field
```
(Use the same `n`/`is_shot` names the surrounding dict uses.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/spadl/test_sportec.py -k shot_blocked -v`
Expected: PASS

- [ ] **Step 5: Full Sportec suite + the parse-port parity gate**

Run: `python -m pytest tests/spadl/test_sportec.py tests/providers/sportec -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/sportec.py tests/spadl/test_sportec.py
git commit -m "feat(spadl): sportec shot_blocked from shot_outcome blocked, excl own-team deflection (TF-51 prereq)"
```

---

### Task 7: Metrica — real `shot_blocked` (subtype endswith BLOCKED)

`subtype.str.endswith("BLOCKED")` (exact-token per BD-3; robust to `HEAD-BLOCKED`). `cross_blocked` = `pd.NA` (structural). Real fixture `metrica/per_period_match.parquet`: 1 BLOCKED shot (event 668, Player5).

**Files:**
- Modify: `silly_kicks/spadl/metrica.py` (`_build_raw_actions`, dict literal ~line 511)
- Test: `tests/spadl/test_metrica.py` (+ `tests/invariants/_loaders.py:319` real loader)

- [ ] **Step 1: Write the failing test**

```python
# tests/spadl/test_metrica.py
def test_shot_blocked_true_on_blocked_subtype():
    from tests.invariants._loaders import load_metrica_native_per_period
    actions = load_metrica_native_per_period()
    shots = actions[actions["type_name"].str.contains("shot")]
    assert (shots["shot_blocked"] == True).sum() == 1  # noqa: E712  (event 668)
    assert shots["shot_blocked"].notna().all()
    assert actions["cross_blocked"].isna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_metrica.py -k shot_blocked -v`
Expected: FAIL

- [ ] **Step 3: Replace the Task-2 `pd.NA` in the `_build_raw_actions` dict.** Before the dict (~line 511) (`is_shot` ~470, `sub_raw` ~380 in scope):

```python
_shot_blocked_mask = np.array([s.endswith("BLOCKED") for s in sub_raw])
```
Replace the dict entries:
```python
    "shot_blocked": _blocked_flag(n, applicable=is_shot, blocked=_shot_blocked_mask),
    "cross_blocked": _blocked_flag(n),  # structural: failed crosses are untyped BALL LOST
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/spadl/test_metrica.py -k shot_blocked -v`
Expected: PASS

- [ ] **Step 5: Full Metrica suite**

Run: `python -m pytest tests/spadl/test_metrica.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/metrica.py tests/spadl/test_metrica.py
git commit -m "feat(spadl): metrica shot_blocked from BLOCKED subtype (TF-51 prereq)"
```

---

### Task 8: Kloppy gateway — real `shot_blocked` (`ShotResult.BLOCKED`)

Wire the real flag into `_get_blocked_flags` (added in Task 2). Real fixture `kloppy/metrica_events.json`: 3 blocked shots (assert on `team_id` — `player` is `None` on the sample).

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py` (`_get_blocked_flags`)
- Test: `tests/spadl/test_kloppy.py` (uses the `metrica_dataset` fixture, `tests/spadl/conftest.py:26`)

- [ ] **Step 1: Write the failing test**

```python
# tests/spadl/test_kloppy.py
def test_shot_blocked_true_on_blocked_shot(metrica_dataset):
    from silly_kicks.spadl import kloppy
    actions, _ = kloppy.convert_to_actions(metrica_dataset)
    shots = actions[actions["type_name"] == "shot"]
    assert (shots["shot_blocked"] == True).sum() == 3  # noqa: E712
    assert shots["shot_blocked"].notna().all()
    assert actions["cross_blocked"].isna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/spadl/test_kloppy.py -k shot_blocked -v`
Expected: FAIL

- [ ] **Step 3: Wire the real flag into `_get_blocked_flags`** (`ShotEvent`/`ShotResult` already imported):

```python
def _get_blocked_flags(event) -> dict[str, object]:
    shot_blocked = isinstance(event, ShotEvent) and event.result == ShotResult.BLOCKED
    return {"shot_blocked": shot_blocked, "cross_blocked": pd.NA}
```

(A non-shot event yields `shot_blocked=False` here, but non-shot rows are filtered/typed away and, per contract, should read `pd.NA` — `_finalize_output` casts the mixed `False`/`NA` column to `"boolean"`; to enforce non-shot = NA, restrict at the dict: return `{"shot_blocked": (True if event.result == ShotResult.BLOCKED else False), ...}` only inside the `ShotEvent` branch and `pd.NA` otherwise:)

```python
def _get_blocked_flags(event) -> dict[str, object]:
    if isinstance(event, ShotEvent):
        return {"shot_blocked": event.result == ShotResult.BLOCKED, "cross_blocked": pd.NA}
    return {"shot_blocked": pd.NA, "cross_blocked": pd.NA}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/spadl/test_kloppy.py -k shot_blocked -v`
Expected: PASS

- [ ] **Step 5: Full kloppy suite**

Run: `python -m pytest tests/spadl/test_kloppy.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/kloppy.py tests/spadl/test_kloppy.py
git commit -m "feat(spadl): kloppy shot_blocked from ShotResult.BLOCKED (TF-51 prereq)"
```

---

### Task 9: Cross-provider contract test + Opta/SkillCorner all-NA + additivity guard

Lock the contract: every converter emits both columns as dtype `"boolean"`; Opta + SkillCorner (+ StatsBomb `cross_blocked`) are all-`pd.NA`; and no *other* column changed.

**Files:**
- Test: `tests/spadl/test_block_detection_contract.py` (new)
- Test (Opta/SkillCorner presence): `tests/spadl/test_opta.py`, `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write the contract + all-NA tests**

```python
# tests/spadl/test_block_detection_contract.py
import pandas as pd


def _assert_block_columns(actions):
    for col in ("shot_blocked", "cross_blocked"):
        assert col in actions.columns, f"missing {col}"
        assert str(actions[col].dtype) == "boolean", f"{col} dtype {actions[col].dtype}"


def test_all_real_fixture_providers_emit_boolean_block_columns():
    # the committed-real-fixture providers, via the shared invariant loaders
    from tests.invariants._loaders import (
        load_statsbomb,
        load_sportec_native_per_period,
        load_metrica_native_per_period,
    )

    for actions in (
        load_statsbomb("7298"),
        load_sportec_native_per_period(),
        load_metrica_native_per_period(),
    ):
        _assert_block_columns(actions)
```

```python
# tests/spadl/test_opta.py  — Opta has no committed fixture; use the existing synthetic events
def test_opta_block_columns_all_na():
    from silly_kicks.spadl import opta
    actions, _ = opta.convert_to_actions(_synthetic_opta_events(), home_team_id=_HOME)
    assert str(actions["shot_blocked"].dtype) == "boolean"
    assert actions["shot_blocked"].isna().all()
    assert actions["cross_blocked"].isna().all()
```

```python
# tests/spadl/test_skillcorner.py — assert on BOTH the basic AND derived fixtures
def test_skillcorner_block_columns_all_na_basic_and_derived():
    from silly_kicks.spadl import skillcorner
    for events, meta in (_load_basic_fixture(), _load_derived_fixture()):  # existing helpers
        actions, _ = skillcorner.convert_to_actions(events, meta)
        assert str(actions["shot_blocked"].dtype) == "boolean"
        assert actions["shot_blocked"].isna().all()
        assert actions["cross_blocked"].isna().all()
```

(The `derived_actions.csv` case is load-bearing — it exercises the `[actions.columns]` merge that would have crashed if the columns were added to the native dict; Task 2 inserts them after `pd.concat`, so this must pass.)

- [ ] **Step 2: Run to verify pass** (Opta/SkillCorner already emit NA from Task 2)

Run: `python -m pytest tests/spadl/test_block_detection_contract.py tests/spadl/test_opta.py tests/spadl/test_skillcorner.py -k "block or na" -v`
Expected: PASS

- [ ] **Step 3: Additivity — the honest guarantee (P-1).** The no-retrain claim rests on **(a) additive-only edits** — every change adds a *new* column via `_blocked_flag`; no existing column's code path is touched, so existing values cannot change *by construction* — and **(b) the existing golden/parity fixtures** (regenerated additively in Task 2 Step 6) catching any accidental drift. It does **not** rest on a column-name check. Add only a **presence** assertion (the honest scope of a structural test — it cannot prove value-invariance):

```python
def test_block_columns_present_and_additive(metrica_dataset):
    from silly_kicks.spadl import kloppy
    actions, _ = kloppy.convert_to_actions(metrica_dataset)
    assert {"shot_blocked", "cross_blocked"} <= set(actions.columns)
    # every pre-existing canonical column still present (additive: nothing dropped or renamed)
    for col in kloppy.KLOPPY_SPADL_COLUMNS:
        if col not in ("shot_blocked", "cross_blocked"):
            assert col in actions.columns
```

Value-invariance is guaranteed by (a)+(b) above and verified by the **full existing suite** in Step 4 — not by this test.

- [ ] **Step 4: Run the full suite (no regressions anywhere)**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: PASS

- [ ] **Step 5: Lint + type check (Shift Left)**

Run: `ruff check silly_kicks/spadl silly_kicks/reflection.py && ruff format --check silly_kicks/spadl && pyright silly_kicks/spadl/utils.py silly_kicks/spadl/schema.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add tests/spadl/test_block_detection_contract.py tests/spadl/test_opta.py tests/spadl/test_skillcorner.py
git commit -m "test(spadl): block-detection cross-provider contract + Opta/SkillCorner all-NA + additivity (TF-51 prereq)"
```

---

## Notes for the executor

- **Branch:** `pr-s<NN>-block-detection-columns` off `main` (do not claim the PR-S/ADR number until commit-prep).
- **Verification discipline (BD-1):** the plan's real-data assertions cover GS (synthetic), StatsBomb/sportec/metrica/kloppy (committed real fixtures). Wyscout is synthetic-only and Opta is NA-only — both are honestly recorded as mechanism-only in the spec; do not fabricate a real assertion for them.
- **No retrain / C4-free:** additive columns only; do not touch any aggregator, xfn list, or the C4 count.
- **ADR:** a new cross-provider output-column convention is ADR-worthy — draft the ADR at commit-prep (may fold into the TF-51 ADR); confirm the number then.
- **Owner-gated GS real-data e2e (spec §6, owner-run — NOT in the CI-green flow): IMPLEMENTED** in `tests/spadl/test_gradientsports_block_e2e.py` (there is no committed real GS fixture, so the Task-3 GS assertions are synthetic; this is the real-data validation). An `@pytest.mark.e2e` test pulls real WC2022 GS match 10502 via the pining loader (`scripts/_loader_pining`, owner token), re-derives the raw B-shot / B-cross counts from the feed, and asserts the converter surfaces `shotOutcomeType=="B"` → `shot_blocked` and `crossOutcomeType=="B"` → `cross_blocked` (open-play), plus `pd.NA` on non-shot/non-cross rows. Skips where the owner token is absent (public CI). Verified passing (26.8 s).
