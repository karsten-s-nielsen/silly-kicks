# Derive End Coordinates + GK Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `end_x == start_x` for pass-class SPADL actions on single-position providers and wire `defending_gk_from_frames()` as fallback for GK feature resolution.

**Architecture:** Replace `_fix_clearances()` in `base.py` with `_derive_end_coordinates()` that uses a source-data guard (`end == start` check) + type filter + period-safe groupby shift. Wire the existing `defending_gk_from_frames()` as a `.fillna()` fallback in `add_pre_shot_gk_context()`. Commit the paired IDSSE tracking fixture extracted from the lakehouse.

**Tech Stack:** pandas, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-15-derive-end-coordinates-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `silly_kicks/spadl/base.py` | Modify | Delete `_fix_clearances`, add `_derive_end_coordinates` |
| `silly_kicks/spadl/sportec.py:120,656` | Modify | Import + call site swap |
| `silly_kicks/spadl/gradientsports.py:469-559` | Modify | Insert `_derive_end_coordinates` before foul synthesis, delete post-LTR block |
| `silly_kicks/spadl/skillcorner.py:29,415` | Modify | Import + call site swap |
| `silly_kicks/spadl/statsbomb.py:11,291` | Modify | Import + call site swap |
| `silly_kicks/spadl/opta.py:13,214` | Modify | Import + call site swap |
| `silly_kicks/spadl/wyscout.py:11,314` | Modify | Import + call site swap |
| `silly_kicks/spadl/metrica.py:72,276` | Modify | Import + call site swap |
| `silly_kicks/spadl/kloppy.py:47,220` | Modify | Import + call site swap |
| `silly_kicks/spadl/utils.py:705-710` | Modify | Insert GK fallback wiring |
| `tests/spadl/test_derive_end_coordinates.py` | Create | Unit tests for `_derive_end_coordinates` |
| `tests/spadl/test_end_coord_integration.py` | Create | Integration tests per converter + dribble regression |
| `tests/spadl/test_gk_fallback_integration.py` | Create | Paired IDSSE GK fallback test |
| `tests/datasets/idsse/paired_tracking.parquet` | Create | Paired tracking fixture (already extracted) |

---

### Task 1: Unit tests for `_derive_end_coordinates`

**Files:**
- Create: `tests/spadl/test_derive_end_coordinates.py`

- [ ] **Step 1: Write the unit test file**

```python
"""Unit tests for _derive_end_coordinates (Bug #7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import _derive_end_coordinates


def _make_actions(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal SPADL-shaped DataFrame from row dicts."""
    defaults = {
        "game_id": 1,
        "period_id": 1,
        "team_id": 100,
        "player_id": 200,
        "bodypart_id": 0,
        "result_id": 1,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


class TestPassClassDerivation:
    """Pass-class types get next-event end coordinates."""

    def test_pass_gets_next_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 50.0, "end_y": 30.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["tackle"],
             "time_seconds": 12.0, "start_x": 70.0, "start_y": 40.0,
             "end_x": 70.0, "end_y": 40.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(70.0)
        assert result.loc[0, "end_y"] == pytest.approx(40.0)

    def test_cross_gets_next_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["cross"],
             "time_seconds": 10.0, "start_x": 80.0, "start_y": 5.0,
             "end_x": 80.0, "end_y": 5.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["shot"],
             "time_seconds": 12.0, "start_x": 95.0, "start_y": 34.0,
             "end_x": 95.0, "end_y": 34.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(95.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_throw_in_gets_next_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["throw_in"],
             "time_seconds": 10.0, "start_x": 60.0, "start_y": 0.0,
             "end_x": 60.0, "end_y": 0.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 11.0, "start_x": 65.0, "start_y": 10.0,
             "end_x": 65.0, "end_y": 10.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(65.0)
        assert result.loc[0, "end_y"] == pytest.approx(10.0)

    def test_clearance_gets_next_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["clearance"],
             "time_seconds": 10.0, "start_x": 15.0, "start_y": 34.0,
             "end_x": 15.0, "end_y": 34.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 13.0, "start_x": 55.0, "start_y": 50.0,
             "end_x": 55.0, "end_y": 50.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(55.0)
        assert result.loc[0, "end_y"] == pytest.approx(50.0)

    def test_goalkick_gets_next_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["goalkick"],
             "time_seconds": 10.0, "start_x": 5.0, "start_y": 34.0,
             "end_x": 5.0, "end_y": 34.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 12.0, "start_x": 40.0, "start_y": 20.0,
             "end_x": 40.0, "end_y": 20.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(40.0)
        assert result.loc[0, "end_y"] == pytest.approx(20.0)


class TestExclusions:
    """Types NOT in the derive set keep end = start."""

    def test_shot_keeps_end_equals_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["shot"],
             "time_seconds": 10.0, "start_x": 90.0, "start_y": 34.0,
             "end_x": 90.0, "end_y": 34.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["keeper_save"],
             "time_seconds": 11.0, "start_x": 104.0, "start_y": 34.0,
             "end_x": 104.0, "end_y": 34.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(90.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_tackle_keeps_end_equals_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["tackle"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 50.0, "end_y": 30.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 11.0, "start_x": 55.0, "start_y": 35.0,
             "end_x": 55.0, "end_y": 35.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(50.0)
        assert result.loc[0, "end_y"] == pytest.approx(30.0)

    def test_keeper_save_keeps_end_equals_start(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["keeper_save"],
             "time_seconds": 10.0, "start_x": 104.0, "start_y": 34.0,
             "end_x": 104.0, "end_y": 34.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["clearance"],
             "time_seconds": 11.0, "start_x": 100.0, "start_y": 30.0,
             "end_x": 100.0, "end_y": 30.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(104.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)


class TestSourceDataGuard:
    """Rows where source already provided end != start are NOT overwritten."""

    def test_pass_with_source_end_preserved(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 65.0, "end_y": 35.0},  # source provided different end
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["tackle"],
             "time_seconds": 12.0, "start_x": 70.0, "start_y": 40.0,
             "end_x": 70.0, "end_y": 40.0},
        ])
        result = _derive_end_coordinates(actions)
        # Source end_x=65.0 must be preserved, NOT overwritten with 70.0
        assert result.loc[0, "end_x"] == pytest.approx(65.0)
        assert result.loc[0, "end_y"] == pytest.approx(35.0)

    def test_clearance_with_source_end_preserved(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["clearance"],
             "time_seconds": 10.0, "start_x": 15.0, "start_y": 34.0,
             "end_x": 55.0, "end_y": 50.0},  # source provided different end
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 13.0, "start_x": 60.0, "start_y": 20.0,
             "end_x": 60.0, "end_y": 20.0},
        ])
        result = _derive_end_coordinates(actions)
        # Source end preserved, not overwritten with 60.0/20.0
        assert result.loc[0, "end_x"] == pytest.approx(55.0)
        assert result.loc[0, "end_y"] == pytest.approx(50.0)


class TestPeriodBoundary:
    """Last action per period keeps end = start (no cross-period contamination)."""

    def test_last_action_period_1_not_contaminated(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 2700.0, "start_x": 80.0, "start_y": 34.0,
             "end_x": 80.0, "end_y": 34.0, "period_id": 1},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 0.5, "start_x": 50.0, "start_y": 34.0,
             "end_x": 50.0, "end_y": 34.0, "period_id": 2},
        ])
        result = _derive_end_coordinates(actions)
        # Period 1 last action keeps end = start (not contaminated by P2 start)
        assert result.loc[0, "end_x"] == pytest.approx(80.0)
        assert result.loc[0, "end_y"] == pytest.approx(34.0)

    def test_period_2_action_gets_next_within_period(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 2700.0, "start_x": 80.0, "start_y": 34.0,
             "end_x": 80.0, "end_y": 34.0, "period_id": 1},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 0.5, "start_x": 50.0, "start_y": 34.0,
             "end_x": 50.0, "end_y": 34.0, "period_id": 2},
            {"action_id": 2, "type_id": spadlconfig.actiontype_id["tackle"],
             "time_seconds": 2.0, "start_x": 55.0, "start_y": 40.0,
             "end_x": 55.0, "end_y": 40.0, "period_id": 2},
        ])
        result = _derive_end_coordinates(actions)
        # P2 first pass gets next action within P2
        assert result.loc[1, "end_x"] == pytest.approx(55.0)
        assert result.loc[1, "end_y"] == pytest.approx(40.0)


class TestEdgeCases:
    """Empty and single-row DataFrames."""

    def test_empty_dataframe(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 50.0, "end_y": 30.0},
        ]).iloc[0:0]
        result = _derive_end_coordinates(actions)
        assert len(result) == 0

    def test_single_action_keeps_end(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 50.0, "end_y": 30.0},
        ])
        result = _derive_end_coordinates(actions)
        assert result.loc[0, "end_x"] == pytest.approx(50.0)
        assert result.loc[0, "end_y"] == pytest.approx(30.0)

    def test_does_not_mutate_input(self):
        actions = _make_actions([
            {"action_id": 0, "type_id": spadlconfig.actiontype_id["pass"],
             "time_seconds": 10.0, "start_x": 50.0, "start_y": 30.0,
             "end_x": 50.0, "end_y": 30.0},
            {"action_id": 1, "type_id": spadlconfig.actiontype_id["tackle"],
             "time_seconds": 12.0, "start_x": 70.0, "start_y": 40.0,
             "end_x": 70.0, "end_y": 40.0},
        ])
        original_end_x = actions.loc[0, "end_x"]
        _derive_end_coordinates(actions)
        assert actions.loc[0, "end_x"] == original_end_x
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `python -m pytest tests/spadl/test_derive_end_coordinates.py -v --tb=short 2>&1 | head -30`
Expected: FAIL with `ImportError: cannot import name '_derive_end_coordinates' from 'silly_kicks.spadl.base'`

- [ ] **Step 3: Implement `_derive_end_coordinates` in `base.py`**

Replace `_fix_clearances` (lines 13-20) with:

```python
# Type IDs for pass-class actions where the ball physically travels to a
# different location.  Used by _derive_end_coordinates to overwrite
# placeholder end_x/end_y with the next action's start position.
_DERIVE_END_TYPE_IDS: frozenset[int] = frozenset({
    spadlconfig.actiontype_id["pass"],             # 0
    spadlconfig.actiontype_id["cross"],             # 1
    spadlconfig.actiontype_id["throw_in"],          # 2
    spadlconfig.actiontype_id["freekick_crossed"],  # 3
    spadlconfig.actiontype_id["freekick_short"],    # 4
    spadlconfig.actiontype_id["corner_crossed"],    # 5
    spadlconfig.actiontype_id["corner_short"],      # 6
    spadlconfig.actiontype_id["clearance"],         # 18
    spadlconfig.actiontype_id["goalkick"],          # 22
})


def _derive_end_coordinates(actions: pd.DataFrame) -> pd.DataFrame:
    """Derive end_x/end_y from next action's start for pass-class types.

    Only overwrites rows where the source data did not provide a separate
    end coordinate (detected by ``end_x == start_x AND end_y == start_y``).
    Period-safe: uses ``groupby("period_id").shift(-1)`` so the last action
    per period keeps its original end coordinates.

    Replaces the former ``_fix_clearances`` with a broader type set, a
    source-data guard, and period-boundary safety.
    """
    if len(actions) == 0:
        return actions
    actions = actions.copy()

    needs_derivation = (
        actions["type_id"].isin(_DERIVE_END_TYPE_IDS)
        & (actions["end_x"] == actions["start_x"])
        & (actions["end_y"] == actions["start_y"])
    )

    next_start_x = actions.groupby("period_id")["start_x"].shift(-1)
    next_start_y = actions.groupby("period_id")["start_y"].shift(-1)

    mask = needs_derivation & next_start_x.notna()
    actions.loc[mask, "end_x"] = next_start_x[mask].values
    actions.loc[mask, "end_y"] = next_start_y[mask].values
    return actions
```

Keep `_fix_clearances` temporarily (it is still imported by other modules). We will remove it in the converter-sweep task.

- [ ] **Step 4: Run unit tests — expect PASS**

Run: `python -m pytest tests/spadl/test_derive_end_coordinates.py -v --tb=short`
Expected: 16 PASSED

- [ ] **Step 5: Run existing test suite — no regressions**

Run: `python -m pytest tests/spadl/ -m "not e2e" -v --tb=short -q 2>&1 | tail -5`
Expected: all existing tests pass (function exists alongside `_fix_clearances`)

---

### Task 2: Converter sweep — replace `_fix_clearances` with `_derive_end_coordinates`

**Files:**
- Modify: `silly_kicks/spadl/sportec.py:120,656`
- Modify: `silly_kicks/spadl/statsbomb.py:11,291`
- Modify: `silly_kicks/spadl/opta.py:13,214`
- Modify: `silly_kicks/spadl/wyscout.py:11,314`
- Modify: `silly_kicks/spadl/metrica.py:72,276`
- Modify: `silly_kicks/spadl/skillcorner.py:29,415`
- Modify: `silly_kicks/spadl/kloppy.py:47,220`
- Modify: `silly_kicks/spadl/base.py` (delete `_fix_clearances`)

- [ ] **Step 1: Update sportec.py import and call site**

At line 120, change:
```python
from .base import _add_dribbles, _fix_clearances
```
to:
```python
from .base import _add_dribbles, _derive_end_coordinates
```

At line 656, change:
```python
        actions = _fix_clearances(raw_actions)
```
to:
```python
        actions = _derive_end_coordinates(raw_actions)
```

- [ ] **Step 2: Update statsbomb.py import and call site**

At line 11, change:
```python
from .base import _add_dribbles, _fix_clearances
```
to:
```python
from .base import _add_dribbles, _derive_end_coordinates
```

At line 291, change:
```python
    actions = _fix_clearances(actions)
```
to:
```python
    actions = _derive_end_coordinates(actions)
```

- [ ] **Step 3: Update opta.py import and call site**

At lines 13-14, change:
```python
from .base import (
    _add_dribbles,
    _fix_clearances,
    min_dribble_length,
)
```
to:
```python
from .base import (
    _add_dribbles,
    _derive_end_coordinates,
    min_dribble_length,
)
```

At line 214, change:
```python
    actions = _fix_clearances(actions)
```
to:
```python
    actions = _derive_end_coordinates(actions)
```

- [ ] **Step 4: Update wyscout.py import and call site**

At lines 9-12, change:
```python
from .base import (
    _add_dribbles,
    _fix_clearances,
)
```
to:
```python
from .base import (
    _add_dribbles,
    _derive_end_coordinates,
)
```

At line 314, change:
```python
    actions = _fix_clearances(actions)
```
to:
```python
    actions = _derive_end_coordinates(actions)
```

- [ ] **Step 5: Update metrica.py import and call site**

At line 72, change:
```python
from .base import _add_dribbles, _fix_clearances
```
to:
```python
from .base import _add_dribbles, _derive_end_coordinates
```

At line 276, change:
```python
        actions = _fix_clearances(raw_actions)
```
to:
```python
        actions = _derive_end_coordinates(raw_actions)
```

- [ ] **Step 6: Update skillcorner.py import and call site**

At line 29, change:
```python
from .base import _add_dribbles, _fix_clearances
```
to:
```python
from .base import _add_dribbles, _derive_end_coordinates
```

At line 415, change:
```python
    actions = _fix_clearances(actions)
```
to:
```python
    actions = _derive_end_coordinates(actions)
```

- [ ] **Step 7: Update kloppy.py import and call site**

At line 47, change:
```python
from .base import _add_dribbles, _fix_clearances
```
to:
```python
from .base import _add_dribbles, _derive_end_coordinates
```

At line 220, change:
```python
    df_actions = _fix_clearances(df_actions)  # type: ignore[reportArgumentType]  # kloppy API varies by version
```
to:
```python
    df_actions = _derive_end_coordinates(df_actions)  # type: ignore[reportArgumentType]  # kloppy API varies by version
```

- [ ] **Step 8: Delete `_fix_clearances` from base.py**

Remove lines 13-20 (the entire `_fix_clearances` function).

- [ ] **Step 9: Run full test suite — no regressions**

Run: `python -m pytest tests/spadl/ -m "not e2e" -v --tb=short -q 2>&1 | tail -10`
Expected: all tests pass. The source-data guard ensures providers with explicit end coords are unchanged.

- [ ] **Step 10: Verify no remaining references to `_fix_clearances`**

Run: `grep -r "_fix_clearances" silly_kicks/ tests/`
Expected: no matches (all references replaced).

---

### Task 3: Gradient Sports converter — pre-foul-synthesis derivation + delete post-LTR block

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py`

- [ ] **Step 1: Add import of `_derive_end_coordinates` at top of gradientsports.py**

After the existing imports from base (the file currently does NOT import `_fix_clearances`), add an import. Find the import block near the top of the file and add:

```python
from .base import _derive_end_coordinates
```

- [ ] **Step 2: Insert `_derive_end_coordinates` call before foul synthesis**

At line 468 (after the tackle winner/loser block ends, before the foul handling comment block at line 469), insert:

```python
    # ------------------------------------------------------------------
    # Derive end_x/end_y from next-action start for pass-class types.
    # Must run BEFORE foul synthesis: synthesized foul rows interleave
    # via 0.5-offset sort key and would intercept the shift(-1) chain.
    # ------------------------------------------------------------------
    actions = _derive_end_coordinates(actions)
```

- [ ] **Step 3: Delete the post-LTR end-coordinate derivation block**

Delete lines 542-559 (the block starting with `# end_x / end_y from next-action start_x/y within same period.`):

```python
    # ------------------------------------------------------------------
    # end_x / end_y from next-action start_x/y within same period.
    # Last row of each period falls back to its own start_x/y.
    # ------------------------------------------------------------------
    if len(actions) > 0:
        next_start_x = actions["start_x"].shift(-1)
        next_start_y = actions["start_y"].shift(-1)
        same_period = actions["period_id"].eq(actions["period_id"].shift(-1))
        actions["end_x"] = np.where(
            same_period & next_start_x.notna(),
            next_start_x,
            actions["start_x"],
        )
        actions["end_y"] = np.where(
            same_period & next_start_y.notna(),
            next_start_y,
            actions["start_y"],
        )
```

- [ ] **Step 4: Run Gradient Sports test suite**

Run: `python -m pytest tests/spadl/test_gradientsports.py -v --tb=short`
Expected: all existing GS tests pass.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/spadl/ -m "not e2e" -v --tb=short -q 2>&1 | tail -10`
Expected: all tests pass.

---

### Task 4: Integration tests — end coordinates per converter + dribble regression

**Files:**
- Create: `tests/spadl/test_end_coord_integration.py`

- [ ] **Step 1: Write integration test file**

```python
"""Integration tests: _derive_end_coordinates across converters (Bug #7)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

_IDSSE_DIR = Path(__file__).parent.parent / "datasets" / "idsse"
_SB_DIR = Path(__file__).parent.parent / "datasets" / "statsbomb"


# -----------------------------------------------------------------------
# Sportec / IDSSE: single-position provider — pass-class gets end != start
# -----------------------------------------------------------------------


class TestSportecEndCoordinates:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        from silly_kicks.spadl.sportec import convert_to_actions

        events = pd.read_parquet(_IDSSE_DIR / "per_period_match.parquet")
        gk_ids: set[str] | None = None
        if "play_goal_keeper_action" in events.columns:
            gk_ids = set(
                events.loc[events["play_goal_keeper_action"].notna(), "player_id"]
                .dropna().astype(str).tolist()
            )
        actions, _report = convert_to_actions(
            events,
            home_team_id="home",
            home_team_start_left=False,  # home attacks LEFT in P1
            goalkeeper_ids=gk_ids,
        )
        return actions

    def test_pass_class_majority_end_neq_start(self, actions: pd.DataFrame):
        pass_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("pass", "cross", "throw_in", "freekick_crossed",
                      "freekick_short", "corner_crossed", "corner_short",
                      "goalkick")
        }
        pass_actions = actions[actions["type_id"].isin(pass_type_ids)]
        has_different_end = (
            (pass_actions["end_x"] != pass_actions["start_x"])
            | (pass_actions["end_y"] != pass_actions["start_y"])
        )
        # Majority should have end != start; only period-boundary last
        # actions may still have end == start.
        ratio = has_different_end.mean()
        assert ratio > 0.90, f"Only {ratio:.1%} of pass-class actions have end != start"

    def test_shots_keep_end_equals_start(self, actions: pd.DataFrame):
        shot_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("shot", "shot_penalty", "shot_freekick")
        }
        shots = actions[actions["type_id"].isin(shot_type_ids)]
        if len(shots) == 0:
            pytest.skip("No shots in fixture")
        same_end = (
            (shots["end_x"] == shots["start_x"])
            & (shots["end_y"] == shots["start_y"])
        )
        assert same_end.all(), "Shot end coordinates should equal start"

    def test_dribble_count_decreases(self, actions: pd.DataFrame):
        dribble_id = spadlconfig.actiontype_id["dribble"]
        n_dribbles = (actions["type_id"] == dribble_id).sum()
        # Pre-fix: 708 dribbles (639 spurious). Post-fix: ~69 legitimate.
        # Use generous upper bound to allow for minor variation.
        assert n_dribbles < 200, (
            f"Expected < 200 dribbles after fix, got {n_dribbles}"
        )
        assert n_dribbles > 0, "Should still have some legitimate dribbles"


# -----------------------------------------------------------------------
# StatsBomb: source-provided end coordinates preserved (regression guard)
# -----------------------------------------------------------------------


class TestStatsBombEndCoordinatesPreserved:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        import json

        from silly_kicks.spadl.statsbomb import convert_to_actions

        raw_path = _SB_DIR / "raw" / "events" / "7584.json"
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Adapt raw StatsBomb events to DataFrame format expected by converter.
        _top_level_keys = {
            "id", "index", "period", "timestamp", "minute", "second",
            "type", "possession", "possession_team", "play_pattern",
            "team", "player", "position", "location", "duration",
            "under_pressure", "off_camera", "out", "related_events",
            "tactics",
        }
        events = pd.DataFrame(
            [
                {
                    "event_id": e["id"],
                    "game_id": 7584,
                    "period_id": e["period"],
                    "timestamp": e["timestamp"],
                    "minute": e["minute"],
                    "second": e["second"],
                    "type_id": e["type"]["id"],
                    "type_name": e["type"]["name"],
                    "possession": e.get("possession"),
                    "possession_team_id": e.get("possession_team", {}).get("id"),
                    "possession_team_name": e.get("possession_team", {}).get("name"),
                    "play_pattern_id": e.get("play_pattern", {}).get("id"),
                    "play_pattern_name": e.get("play_pattern", {}).get("name"),
                    "team_id": e.get("team", {}).get("id"),
                    "team_name": e.get("team", {}).get("name"),
                    "player_id": e.get("player", {}).get("id"),
                    "player_name": e.get("player", {}).get("name"),
                    "position_id": e.get("position", {}).get("id"),
                    "position_name": e.get("position", {}).get("name"),
                    "location": e.get("location"),
                    "duration": e.get("duration"),
                    "under_pressure": e.get("under_pressure"),
                    "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
                }
                for e in raw
            ]
        )
        home_team_id = int(events["team_id"].dropna().iloc[0])
        actions, _ = convert_to_actions(
            events,
            home_team_id=home_team_id,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
        )
        return actions

    def test_passes_have_source_end_coordinates(self, actions: pd.DataFrame):
        pass_id = spadlconfig.actiontype_id["pass"]
        passes = actions[actions["type_id"] == pass_id]
        has_different_end = (
            (passes["end_x"] != passes["start_x"])
            | (passes["end_y"] != passes["start_y"])
        )
        # StatsBomb provides explicit pass.end_location for virtually all passes.
        ratio = has_different_end.mean()
        assert ratio > 0.95, (
            f"StatsBomb passes should have source end coords; only {ratio:.1%} do"
        )

    def test_clearances_with_source_end_preserved(self, actions: pd.DataFrame):
        clearance_id = spadlconfig.actiontype_id["clearance"]
        clearances = actions[actions["type_id"] == clearance_id]
        if len(clearances) == 0:
            pytest.skip("No clearances in fixture")
        has_different_end = (
            (clearances["end_x"] != clearances["start_x"])
            | (clearances["end_y"] != clearances["start_y"])
        )
        # StatsBomb provides end coords for ~99.6% of clearances.
        # Guard ensures these are NOT overwritten (unlike old _fix_clearances).
        ratio = has_different_end.mean()
        assert ratio > 0.90, (
            f"StatsBomb clearances should keep source end coords; only {ratio:.1%} do"
        )


# -----------------------------------------------------------------------
# Gradient Sports: shots / tackles / keeper_saves must keep end == start
# -----------------------------------------------------------------------

_GS_DIR = Path(__file__).parent.parent / "datasets" / "gradientsports"


class TestGradientSportsExcludedTypesKeepEnd:
    @pytest.fixture()
    def actions(self) -> pd.DataFrame:
        from tests.spadl.test_gradientsports import _load_synthetic_events

        from silly_kicks.spadl.gradientsports import convert_to_actions

        events = _load_synthetic_events()
        actions, _ = convert_to_actions(
            events,
            home_team_id=100,
            home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        return actions

    def test_shots_tackles_keeper_saves_end_equals_start(
        self, actions: pd.DataFrame,
    ):
        """Shots, tackles, keeper_saves should NOT get next-event end coords."""
        excluded_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("shot", "shot_penalty", "shot_freekick",
                      "tackle", "keeper_save")
        }
        excluded = actions[actions["type_id"].isin(excluded_type_ids)]
        if len(excluded) == 0:
            pytest.skip("No shot/tackle/keeper_save in GS fixture")
        same_end = (
            (excluded["end_x"] == excluded["start_x"])
            & (excluded["end_y"] == excluded["start_y"])
        )
        assert same_end.all(), (
            f"GS shots/tackles/keeper_saves should keep end==start; "
            f"{(~same_end).sum()}/{len(excluded)} have different end coords"
        )
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/spadl/test_end_coord_integration.py -v --tb=short`
Expected: 6 PASSED

---

### Task 5: Commit paired tracking fixture

**Files:**
- Create: `tests/datasets/idsse/paired_tracking.parquet` (already extracted)

- [ ] **Step 1: Verify the fixture exists and has expected shape**

Run:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('tests/datasets/idsse/paired_tracking.parquet')
assert len(df) > 10000, f'Too few rows: {len(df)}'
assert set(df['period'].unique()) == {1, 2}, f'Expected periods 1,2: {df[\"period\"].unique()}'
assert df['is_goalkeeper'].sum() > 1000, f'Too few GK rows: {df[\"is_goalkeeper\"].sum()}'
print(f'OK: {len(df)} rows, {df[\"frame\"].nunique()} frames, {df[\"player_id\"].nunique()} players, {df[\"is_goalkeeper\"].sum()} GK rows')
"
```
Expected: `OK: 18194 rows, 827 frames, 22 players, 1654 GK rows`

- [ ] **Step 2: Update IDSSE README with paired_tracking section**

Append to `tests/datasets/idsse/README.md`:

```markdown

---

## `paired_tracking.parquet` (PR-S42 / silly-kicks 3.15.0)

Tracking frames from match `J03WMX` covering two time windows aligned with
events in `per_period_match.parquet`. Enables paired events+tracking testing
for Bug #2 (GK fallback) and Bug #7 (end-coordinate derivation).

### Provenance

- Source: same DFL DataHub free-sample data, pulled from
  `soccer_analytics.dev_gold.fct_tracking_frames` via Databricks.
- Match identifier: `J03WMX` (Bundesliga; public DFL competition
  identifier).
- Time windows:
  - P1: 90.0 - 107.0s (4 events: 3 Play + 1 ShotAtGoal)
  - P2: 624.0 - 640.0s (9 events: 3 Play + 1 ThrowIn + 2 OtherBallAction + 1 TacklingGame + 1 ShotAtGoal)
- Row count: ~18,194 (22 players x ~827 frames).
- GK players: `DFL-OBJ-0002DR` (away), `DFL-OBJ-0002HE` (home).
- File size: ~518 KB.
- Extraction script: `scripts/extract_paired_idsse_fixture.py`.

### License

Same DFL DataHub free-sample license as `per_period_match.parquet`.
Test-only fixture excluded from the published wheel.
```

---

### Task 6: GK fallback wiring in `add_pre_shot_gk_context`

**Files:**
- Modify: `silly_kicks/spadl/utils.py:705-710`

- [ ] **Step 1: Insert the GK fallback between lines 705 and 707**

After line 705 (`sorted_actions["defending_gk_player_id"] = defending_gk_player_id`), insert:

```python
    # Fallback: fill NaN defending_gk_player_id from tracking frames.
    # DFL/Sportec events rarely produce keeper_save actions, so the
    # events-based lookback above leaves most shots with NaN.  The
    # frame-based resolver finds the opposing team's is_goalkeeper=True
    # row in the nearest tracking frame.
    if frames is not None:
        from silly_kicks.tracking._gk_resolve import defending_gk_from_frames

        gk_series = defending_gk_from_frames(sorted_actions, frames)
        sorted_actions["defending_gk_player_id"] = (
            sorted_actions["defending_gk_player_id"].fillna(gk_series)
        )
```

The existing `if frames is not None:` block at line 710 (which imports `add_pre_shot_gk_position` and `add_pre_shot_gk_angle`) remains unchanged below this insertion.

- [ ] **Step 2: Run existing GK context tests**

Run: `python -m pytest tests/spadl/test_add_pre_shot_gk_context.py -v --tb=short`
Expected: all existing tests pass (the fallback is a no-op when `frames is None`).

---

### Task 7: GK fallback integration test with paired IDSSE fixture

**Files:**
- Create: `tests/spadl/test_gk_fallback_integration.py`

- [ ] **Step 1: Write the integration test**

```python
"""Integration test: GK fallback via defending_gk_from_frames (Bug #2).

Uses paired IDSSE events (per_period_match.parquet) + tracking
(paired_tracking.parquet) to verify that shots get defending_gk_player_id
populated from tracking frames when events-based lookback finds no
keeper_save.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.sportec import convert_to_actions
from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking import TRACKING_FRAMES_COLUMNS

_IDSSE_DIR = Path(__file__).parent.parent / "datasets" / "idsse"
_PAIRED_TRACKING = _IDSSE_DIR / "paired_tracking.parquet"


def _load_tracking_frames() -> pd.DataFrame:
    """Load paired tracking fixture and reshape for defending_gk_from_frames.

    This is a PARTIAL reshape — only the columns accessed by
    defending_gk_from_frames / link_actions_to_frames are mapped.
    The fixture is NOT fully SPORTEC_TRACKING_FRAMES_COLUMNS compliant
    (missing z, ball_state, team_attacking_direction, confidence,
    visibility, is_goalkeeper_source, speed_source).
    """
    raw = pd.read_parquet(_PAIRED_TRACKING)
    # Lakehouse column names -> silly-kicks tracking schema.
    # NB: fixture has both "team" ("home"/"away") and "team_id"
    # (DFL CLU IDs like "DFL-CLU-00000G").  Sportec convert_to_actions
    # produces actions with team_id = "home"/"away", so we must use the
    # "team" column as team_id to match.
    rename = {
        "match_id": "game_id",
        "period": "period_id",
        "frame": "frame_id",
        "timestamp_seconds": "time_seconds",
        "team": "team_id",  # "home"/"away" — matches action team_id
        "x": "x",
        "y": "y",
        "ball_x": "ball_x",
        "ball_y": "ball_y",
        "speed_ms": "speed",
    }
    frames = raw.rename(columns=rename)
    # Add missing columns with defaults.
    if "vx" not in frames.columns:
        frames["vx"] = np.nan
    if "vy" not in frames.columns:
        frames["vy"] = np.nan
    if "ax" not in frames.columns:
        frames["ax"] = np.nan
    if "ay" not in frames.columns:
        frames["ay"] = np.nan
    if "is_ball" not in frames.columns:
        frames["is_ball"] = False
    # Ensure dtypes match schema expectations.
    frames["game_id"] = frames["game_id"].astype(str)
    frames["period_id"] = frames["period_id"].astype("int64")
    frames["frame_id"] = frames["frame_id"].astype("int64")
    frames["time_seconds"] = frames["time_seconds"].astype("float64")
    frames["player_id"] = frames["player_id"].astype(str)
    frames["team_id"] = frames["team_id"].astype(str)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    return frames


@pytest.fixture()
def paired_data():
    """Load paired events + tracking for match J03WMX."""
    if not _PAIRED_TRACKING.exists():
        pytest.skip("Paired tracking fixture not available")

    events = pd.read_parquet(_IDSSE_DIR / "per_period_match.parquet")
    gk_ids: set[str] | None = None
    if "play_goal_keeper_action" in events.columns:
        gk_ids = set(
            events.loc[events["play_goal_keeper_action"].notna(), "player_id"]
            .dropna().astype(str).tolist()
        )
    actions, _report = convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=False,  # home attacks LEFT in P1
        goalkeeper_ids=gk_ids,
    )
    frames = _load_tracking_frames()
    return actions, frames


class TestGkFallbackPopulatesDefendingGk:
    """Bug #2: add_pre_shot_gk_context fills NaN GK IDs from tracking."""

    def test_shots_within_tracking_window_have_gk(self, paired_data):
        actions, frames = paired_data
        enriched = add_pre_shot_gk_context(actions, frames=frames)

        shot_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("shot", "shot_penalty", "shot_freekick")
        }
        shots = enriched[enriched["type_id"].isin(shot_type_ids)]

        # Tracking covers P1 ts=90-107 and P2 ts=624-640.
        # Find shots within those windows.
        p1_shots = shots[
            (shots["period_id"] == 1)
            & (shots["time_seconds"] >= 90.0)
            & (shots["time_seconds"] <= 107.0)
        ]
        p2_shots = shots[
            (shots["period_id"] == 2)
            & (shots["time_seconds"] >= 624.0)
            & (shots["time_seconds"] <= 640.0)
        ]
        covered_shots = pd.concat([p1_shots, p2_shots])
        assert len(covered_shots) >= 2, (
            f"Expected >= 2 shots in tracking windows, got {len(covered_shots)}"
        )

        has_gk = covered_shots["defending_gk_player_id"].notna()
        assert has_gk.all(), (
            f"Shots in tracking window missing defending_gk_player_id: "
            f"{covered_shots[~has_gk][['period_id', 'time_seconds']].to_dict()}"
        )

    def test_resolved_gk_is_opposing_team(self, paired_data):
        actions, frames = paired_data
        enriched = add_pre_shot_gk_context(actions, frames=frames)

        shot_type_ids = {
            spadlconfig.actiontype_id[t]
            for t in ("shot", "shot_penalty", "shot_freekick")
        }
        shots = enriched[enriched["type_id"].isin(shot_type_ids)]

        # Known GK IDs from the tracking fixture.
        home_gk = "DFL-OBJ-0002HE"
        away_gk = "DFL-OBJ-0002DR"

        for _, shot in shots.iterrows():
            gk_id = shot["defending_gk_player_id"]
            if pd.isna(gk_id):
                continue  # Shot outside tracking window
            shooter_team = shot["team_id"]
            # Home shooter -> away GK defends; away shooter -> home GK defends.
            if shooter_team == "home":
                assert str(gk_id) == away_gk, (
                    f"Home shot at t={shot['time_seconds']:.1f}: "
                    f"expected away GK {away_gk}, got {gk_id}"
                )
            else:
                assert str(gk_id) == home_gk, (
                    f"Away shot at t={shot['time_seconds']:.1f}: "
                    f"expected home GK {home_gk}, got {gk_id}"
                )
```

- [ ] **Step 2: Run GK fallback integration test**

Run: `python -m pytest tests/spadl/test_gk_fallback_integration.py -v --tb=short`
Expected: 2 PASSED

---

### Task 8: Lint + type check + full test suite

**Files:** (no new files — verification only)

- [ ] **Step 1: Run ruff format check**

Run: `python -m ruff format --check silly_kicks/ tests/spadl/test_derive_end_coordinates.py tests/spadl/test_end_coord_integration.py tests/spadl/test_gk_fallback_integration.py`
Expected: all files already formatted. If not, fix with `python -m ruff format`.

- [ ] **Step 2: Run ruff lint**

Run: `python -m ruff check silly_kicks/ tests/spadl/test_derive_end_coordinates.py tests/spadl/test_end_coord_integration.py tests/spadl/test_gk_fallback_integration.py`
Expected: no errors.

- [ ] **Step 3: Run pyright**

Run: `uv run pyright silly_kicks/spadl/base.py silly_kicks/spadl/sportec.py silly_kicks/spadl/gradientsports.py silly_kicks/spadl/utils.py`
Expected: 0 errors.

- [ ] **Step 4: Run full test suite (non-e2e)**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short -q 2>&1 | tail -15`
Expected: all tests pass.

---

### Task 9: Version bump + CHANGELOG

**Files:**
- Modify: `silly_kicks/__init__.py` (or wherever `__version__` is defined)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version to 3.15.0**

Find the `__version__` assignment and change from `"3.14.1"` to `"3.15.0"`.

- [ ] **Step 2: Add CHANGELOG entry**

Add at the top of `CHANGELOG.md` under a new `## 3.15.0` section:

```markdown
## 3.15.0

### Added

- `_derive_end_coordinates()` shared function in `base.py` — derives `end_x/end_y`
  from next action's `start_x/start_y` for pass-class SPADL types on providers whose
  event format carries only a single position per event (DFL/Sportec, Gradient Sports).
  Source-data guard (`end == start` check) prevents overwriting explicit end coordinates
  on providers like StatsBomb, Opta, Wyscout, Metrica, SkillCorner.
- GK fallback in `add_pre_shot_gk_context()`: fills NaN `defending_gk_player_id` from
  tracking frames via `defending_gk_from_frames()` when events-based lookback finds no
  keeper_save (fixes DFL/Sportec shots getting NULL GK features).
- Paired IDSSE tracking fixture (`tests/datasets/idsse/paired_tracking.parquet`) for
  testing both fixes against real data.

### Fixed

- Sportec/IDSSE: pass, cross, throw_in, freekick, corner, goalkick, clearance actions
  now have correct `end_x/end_y` derived from next event (previously all had `end == start`).
- Gradient Sports: end-coordinate derivation now type-filtered (shots/tackles/keeper_saves
  keep `end == start`) and runs before foul synthesis to avoid interleaving corruption.
- All providers: clearance end-coordinate derivation gains period-boundary safety
  (`groupby("period_id")` instead of bare `shift(-1)`).
- Sportec: spurious dribble count reduced (~639 eliminated on IDSSE fixture) because
  pass-class actions now have correct end coordinates.

### Removed

- `_fix_clearances()` from `base.py` — replaced by `_derive_end_coordinates()`.
```
