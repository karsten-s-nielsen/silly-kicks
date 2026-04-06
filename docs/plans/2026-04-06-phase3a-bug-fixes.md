# Phase 3a: Bug Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 6 confirmed bugs inherited from socceraction v1.5.3 plus pandas 3.0 compatibility, with tests proving each fix.

**Architecture:** Each bug fix is independent — a failing test, then a minimal fix, then verification. No structural changes. The converter architecture stays as-is (Phase 3c rewrites converters). These are correctness fixes on the existing code.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new dependencies.

**Working directory:** `D:\Development\karstenskyt__silly-kicks\`

**Sources:** Upstream issues #507, #950, #784, #831, #37/D44, #946. Research triage from 2026-04-06.

---

## File Structure

```
Tests (create/modify):
  tests/vaep/test_features.py          — Bug 1 (empty game), Bug 2 (atomic actiontype)
  tests/spadl/test_opta.py             — Bug 3 (cards)
  tests/atomic/test_atomic_spadl.py    — Bug 4 (shot "out")  [create new]
  tests/spadl/test_wyscout.py          — Bug 5 (keeper differentiation)
  tests/spadl/test_statsbomb.py        — Bug 6 (fillna)

Source (modify):
  silly_kicks/vaep/features.py         — Bug 1 fix
  silly_kicks/atomic/vaep/features.py  — Bug 2 fix
  silly_kicks/spadl/opta.py            — Bug 3 fix
  silly_kicks/atomic/spadl/base.py     — Bug 4 fix
  silly_kicks/spadl/wyscout.py         — Bug 5 fix (D44)
  silly_kicks/spadl/statsbomb.py       — Bug 6 fix
```

---

## Task 1: Bug #507 — Empty game crash in `gamestates()`

**Files:**
- Modify: `silly_kicks/vaep/features.py:93`
- Test: `tests/vaep/test_features.py`

The `gamestates()` function crashes with `IndexError` when a game/period group has zero rows, because the lambda calls `x.iloc[0]` unconditionally.

- [ ] **Step 1: Write the failing test**

Add to `tests/vaep/test_features.py`:

```python
def test_gamestates_empty_dataframe():
    """Bug #507: gamestates should not crash on empty input."""
    import silly_kicks.spadl.config as spadlcfg

    empty_actions = pd.DataFrame(
        columns=[
            "game_id", "period_id", "action_id", "time_seconds",
            "team_id", "player_id",
            "start_x", "start_y", "end_x", "end_y",
            "type_id", "result_id", "bodypart_id",
            "type_name", "result_name", "bodypart_name",
        ]
    )
    result = fs.gamestates(empty_actions, nb_prev_actions=3)
    assert len(result) == 3
    for gs in result:
        assert len(gs) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/vaep/test_features.py::test_gamestates_empty_dataframe -v`
Expected: FAIL with `IndexError: single positional indexer is out-of-bounds`

- [ ] **Step 3: Write minimal fix**

In `silly_kicks/vaep/features.py:93`, change:

```python
lambda x: x.shift(i, fill_value=float("nan")).fillna(x.iloc[0])
```

to:

```python
lambda x: x.shift(i, fill_value=float("nan")).fillna(x.iloc[0]) if len(x) > 0 else x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/vaep/test_features.py::test_gamestates_empty_dataframe -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all 37+ tests pass (this test adds 1 new)

---

## Task 2: Bug #950 — `actiontype` feature wrong for Atomic-SPADL

**Files:**
- Modify: `silly_kicks/vaep/features.py:182,203`
- Modify: `silly_kicks/atomic/vaep/features.py`
- Test: `tests/atomic/test_atomic_features.py`

The `actiontype` function hardcodes `spadlcfg.actiontypes` (23 types). When called on Atomic-SPADL actions (33 types), types 23-32 are silently dropped/miscategorized.

- [ ] **Step 1: Write the failing test**

Add to `tests/atomic/test_atomic_features.py`:

```python
def test_actiontype_includes_atomic_types():
    """Bug #950: actiontype must use atomic config when given atomic actions."""
    import silly_kicks.atomic.spadl.config as atomicspadl
    from silly_kicks.atomic.vaep.features import actiontype

    # Create a single action with an atomic-only type (e.g., "receival" = type_id 23)
    receival_id = atomicspadl.actiontype_id["receival"]
    actions = pd.DataFrame({
        "game_id": [1],
        "period_id": [1],
        "action_id": [0],
        "type_id": [receival_id],
    })
    result = actiontype(actions)
    assert "type_receival" in result.columns or result["type_id"].iloc[0] == receival_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/atomic/test_atomic_features.py::test_actiontype_includes_atomic_types -v`
Expected: FAIL — `actiontype` uses the 23-type SPADL vocab, not the 33-type atomic vocab

- [ ] **Step 3: Write minimal fix**

The fix is to make `actiontype` accept the config module as a parameter so the atomic version can pass its own config. In `silly_kicks/vaep/features.py`, modify the `actiontype` function (line ~166) to accept an optional `spadl_config` parameter:

```python
def actiontype(actions: Actions, spadl_config: Any = None) -> pd.DataFrame:
    if spadl_config is None:
        spadl_config = spadlcfg
    # Use spadl_config.actiontypes instead of spadlcfg.actiontypes
```

Then in `silly_kicks/atomic/vaep/features.py`, instead of re-exporting `actiontype` unchanged, create a wrapper:

```python
import silly_kicks.atomic.spadl.config as _atomicspadl
from silly_kicks.vaep.features import actiontype as _base_actiontype

def actiontype(actions):
    return _base_actiontype(actions, spadl_config=_atomicspadl)
```

Read both files to understand the exact current signatures before implementing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/atomic/test_atomic_features.py::test_actiontype_includes_atomic_types -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 3: Bug #784 — Opta converter silently drops card events

**Files:**
- Modify: `silly_kicks/spadl/opta.py:124-181`
- Test: `tests/spadl/test_opta.py`

The `_get_type_id` function has no branch for card events — they fall through to `non_action` and are silently discarded.

- [ ] **Step 1: Write the failing test**

Add to `tests/spadl/test_opta.py`:

```python
def test_convert_card_events():
    """Bug #784: Opta cards should produce yellow_card/red_card actions, not non_action."""
    import silly_kicks.spadl.config as spadlcfg

    # Yellow card event
    yellow_type_id = spadlcfg.opta._get_type_id("card", True, {})
    assert yellow_type_id == spadlcfg.actiontype_id["non_action"], "Before fix: cards are non_action"
```

Wait — this test would pass before the fix (it asserts the broken behavior). Instead, write the test for the DESIRED behavior:

```python
def test_opta_card_events_not_dropped():
    """Bug #784: Card events should not be silently dropped."""
    from silly_kicks.spadl.opta import _get_type_id
    import silly_kicks.spadl.config as spadlcfg

    # A card event with outcome=True (yellow card shown)
    result = _get_type_id("card", True, {})
    assert result != spadlcfg.actiontype_id["non_action"], "Card events must not be dropped"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/spadl/test_opta.py::test_opta_card_events_not_dropped -v`
Expected: FAIL — `_get_type_id("card", ...)` returns `non_action`

- [ ] **Step 3: Write minimal fix**

In `silly_kicks/spadl/opta.py`, in `_get_type_id` function (around line 124-181), add handling for card events before the catch-all `else`. Cards in Opta use event name `"card"`. The SPADL vocabulary has `yellow_card` (result_id 4) and `red_card` (result_id 5). Since cards are captured as results (not types) in SPADL, add them as `foul` type with the appropriate result. Read the function to find the right insertion point.

Note: The exact mapping depends on how Opta encodes yellow vs red. Read the existing Opta qualifier patterns to determine how to distinguish them. Qualifier 31 = "involved player" and qualifier 32 = "red card" in Opta. If qualifier 32 is present → `red_card` result; otherwise → `yellow_card` result.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/spadl/test_opta.py::test_opta_card_events_not_dropped -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 4: Bug #831 — Atomic-SPADL missing "out" for blocked/saved shots

**Files:**
- Modify: `silly_kicks/atomic/spadl/base.py:135`
- Test: `tests/atomic/test_atomic_spadl.py` (create new)

Blocked/saved shots don't generate an `out` action because the next action is a keeper event or recovery, not a corner/goalkick.

- [ ] **Step 1: Write the failing test**

Create `tests/atomic/test_atomic_spadl.py`:

```python
import pandas as pd
import pytest
from silly_kicks.atomic.spadl.base import convert_to_atomic
import silly_kicks.spadl.config as spadlcfg


def test_blocked_shot_produces_out():
    """Bug #831: Blocked/saved shots must produce an atomic 'out' action."""
    actions = pd.DataFrame({
        "game_id": [1, 1],
        "period_id": [1, 1],
        "action_id": [0, 1],
        "time_seconds": [10.0, 11.0],
        "team_id": [1, 2],
        "player_id": [101, 201],
        "start_x": [90.0, 10.0],
        "start_y": [34.0, 34.0],
        "end_x": [100.0, 15.0],
        "end_y": [34.0, 34.0],
        "type_id": [spadlcfg.actiontype_id["shot"], spadlcfg.actiontype_id["keeper_save"]],
        "result_id": [spadlcfg.result_id["fail"], spadlcfg.result_id["success"]],
        "bodypart_id": [spadlcfg.bodypart_id["foot"], spadlcfg.bodypart_id["head"]],
    })
    atomic = convert_to_atomic(actions)
    # After a failed shot followed by a keeper_save, there should be an atomic "out"
    # action inserted between them
    import silly_kicks.atomic.spadl.config as atomicconfig
    out_id = atomicconfig.actiontype_id["out"]
    assert out_id in atomic["type_id"].values, "Blocked shots must produce an 'out' atomic action"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/atomic/test_atomic_spadl.py::test_blocked_shot_produces_out -v`
Expected: FAIL — no `out` action in atomic output

- [ ] **Step 3: Write minimal fix**

In `silly_kicks/atomic/spadl/base.py`, modify `_extra_from_shots` (around line 135). Currently `out` only detects shots followed by corner/goalkick. Extend the detection to also include shots followed by keeper actions (`keeper_save`, `keeper_claim`, `keeper_punch`, `keeper_pick_up`):

```python
keeper_actions = next_type.isin([
    _spadl.actiontype_id["keeper_save"],
    _spadl.actiontype_id["keeper_claim"],
    _spadl.actiontype_id["keeper_punch"],
    _spadl.actiontype_id["keeper_pick_up"],
])
out = shot & (next_corner_goalkick | keeper_actions) & samegame & sameperiod
```

Read the file first to understand the full logic before modifying.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/atomic/test_atomic_spadl.py::test_blocked_shot_produces_out -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 5: Bug #37/D44 — Wyscout keeper_claim/punch/pick_up differentiation

**Files:**
- Modify: `silly_kicks/spadl/wyscout.py:714`
- Test: `tests/spadl/test_wyscout.py`

All Wyscout goalkeeper events (`type_id == 9`) are unconditionally mapped to `keeper_save`. The Wyscout data contains `subtype_id` values that distinguish between save, claim, punch — these are currently ignored.

Wyscout subtype IDs for goalkeeper events:
- `90` = reflexes / save attempt → `keeper_save`
- `91` = save attempt (variant) → `keeper_save`
- `92` = claim → `keeper_claim`
- `93` = punch → `keeper_punch`
- No explicit pick_up subtype in Wyscout

Also: Wyscout aerial duels by a goalkeeper (`type_id == 1`, `subtype_id == 10`, where the player is the GK) should map to `keeper_claim` — but this requires knowing the player's position, which may not be available in the event data. For now, handle only the subtype routing on `type_id == 9`.

- [ ] **Step 1: Add Wyscout GK subtype constants**

In `silly_kicks/spadl/wyscout.py`, add new constants near the existing `_WS_SUBTYPE_*` definitions:

```python
_WS_SUBTYPE_GK_REFLEXES: int = 90
_WS_SUBTYPE_GK_SAVE: int = 91
_WS_SUBTYPE_GK_CLAIM: int = 92
_WS_SUBTYPE_GK_PUNCH: int = 93
```

- [ ] **Step 2: Write the failing test**

Add to `tests/spadl/test_wyscout.py`:

```python
def test_wyscout_keeper_claim_differentiation():
    """Bug #37/D44: Wyscout GK events must differentiate keeper_claim/punch."""
    import silly_kicks.spadl.config as spadlcfg
    from silly_kicks.spadl.wyscout import _determine_type_id

    # Create a minimal event dict that looks like a keeper claim (subtype 92)
    event = pd.Series({
        "type_id": 9,
        "subtype_id": 92,
        "fairplay": False,
        "own_goal": False,
        "high": False,
        "take_on_left": False,
        "take_on_right": False,
        "sliding_tackle": False,
        "interception": False,
    })
    result = _determine_type_id(event)
    assert result == spadlcfg.actiontype_id["keeper_claim"]


def test_wyscout_keeper_punch_differentiation():
    """Bug #37/D44: Wyscout GK punch events must map to keeper_punch."""
    import silly_kicks.spadl.config as spadlcfg
    from silly_kicks.spadl.wyscout import _determine_type_id

    event = pd.Series({
        "type_id": 9,
        "subtype_id": 93,
        "fairplay": False,
        "own_goal": False,
        "high": False,
        "take_on_left": False,
        "take_on_right": False,
        "sliding_tackle": False,
        "interception": False,
    })
    result = _determine_type_id(event)
    assert result == spadlcfg.actiontype_id["keeper_punch"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_wyscout.py::test_wyscout_keeper_claim_differentiation tests/spadl/test_wyscout.py::test_wyscout_keeper_punch_differentiation -v`
Expected: FAIL — both return `keeper_save`

- [ ] **Step 4: Fix `_determine_type_id` for GK sub-routing**

In `silly_kicks/spadl/wyscout.py`, find the line (currently ~714):

```python
elif event["type_id"] == _WS_TYPE_GK:
    action_type = "keeper_save"
```

Replace with:

```python
elif event["type_id"] == _WS_TYPE_GK:
    subtype = event["subtype_id"]
    if subtype == _WS_SUBTYPE_GK_CLAIM:
        action_type = "keeper_claim"
    elif subtype == _WS_SUBTYPE_GK_PUNCH:
        action_type = "keeper_punch"
    else:
        action_type = "keeper_save"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_wyscout.py -v`
Expected: all Wyscout tests pass

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 6: Bug #946 — pandas 3.0 `fillna(inplace=True)` compatibility

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py:85`
- Test: `tests/spadl/test_statsbomb.py`

`events["extra"].fillna({}, inplace=True)` is deprecated in pandas 2.x and will break in pandas 3.0.

- [ ] **Step 1: Write the test**

This is a behavioral test — the fix should not change output, just compatibility. Add to any appropriate test file (or inline verify):

```python
def test_statsbomb_extra_fillna_no_inplace():
    """Bug #946: fillna must not use inplace=True (pandas 3.0 compat)."""
    import ast
    import inspect
    from silly_kicks.spadl import statsbomb

    source = inspect.getsource(statsbomb.convert_to_actions)
    assert "inplace=True" not in source, "inplace=True is deprecated in pandas 2.x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/spadl/test_statsbomb.py::test_statsbomb_extra_fillna_no_inplace -v`
Expected: FAIL — source contains `inplace=True`

- [ ] **Step 3: Fix the fillna call**

In `silly_kicks/spadl/statsbomb.py:85`, change:

```python
events["extra"].fillna({}, inplace=True)
```

to:

```python
events["extra"] = events["extra"].fillna({})
```

Note: `fillna({})` replaces NaN values in the `extra` column with empty dicts. The explicit assignment form is pandas 3.0 compatible.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/spadl/test_statsbomb.py::test_statsbomb_extra_fillna_no_inplace -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 7: Final Verification and Deferred Update

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass (37 original + 5-7 new tests)

- [ ] **Step 2: Update docs/DEFERRED.md**

Remove the items that were fixed:
- Remove D44 from the luxury-lakehouse TODO reference (now fixed in silly-kicks)
- Add a note: "Phase 3a complete — 6 upstream bugs fixed"

Move any partially-addressed items (e.g., #37 keeper_pick_up still not produced for Wyscout aerial duels) to the Phase 3c section.

- [ ] **Step 3: Commit (with user approval)**

Present the diff summary to the user and wait for commit approval.
