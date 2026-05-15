# SkillCorner Events SPADL Converter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated SPADL converter for SkillCorner event data (`dynamic_events.csv`), making SkillCorner a first-class event provider alongside StatsBomb, Opta, Sportec, Gradient Sports, and Wyscout.

**Architecture:** Two-file converter (`skillcorner.py` + `_skillcorner_inference.py`) following the Gradient Sports dedicated DataFrame-input converter pattern. Native actions from `player_possession` rows via `end_type`/`game_interruption_before` dispatch; derived defensive actions from `start_type` column + OBE cross-referencing; keeper saves from shot-to-GK sequence detection. Possession-perspective coordinates rescaled from centered meters to SPADL 0-105 x 0-68 frame.

**Tech Stack:** pandas, numpy, silly-kicks SPADL infrastructure (`_finalize_output`, `_add_dribbles`, `_fix_clearances`, `to_spadl_ltr`, `ConversionReport`)

**Spec:** `docs/superpowers/specs/2026-05-14-skillcorner-events-converter-design.md`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `silly_kicks/spadl/skillcorner.py` | Public API (`convert_to_actions`), coordinate transform, action dispatch, body part dispatch |
| Create | `silly_kicks/spadl/_skillcorner_inference.py` | Derived action logic: `start_type`-based interceptions, OBE tackle enrichment, keeper saves |
| Modify | `silly_kicks/spadl/schema.py` | Add `SKILLCORNER_SPADL_COLUMNS` |
| Modify | `silly_kicks/spadl/__init__.py` | Add `skillcorner` + `SKILLCORNER_SPADL_COLUMNS` to public re-exports |
| Create | `tests/datasets/skillcorner/match_metadata.json` | Minimal match JSON fixture (pitch_length, pitch_width, teams, periods) |
| Create | `tests/datasets/skillcorner/basic_possessions.csv` | Synthetic CSV fixture: happy-path dispatch (pass, shot, cross, clearance, foul, set pieces) |
| Create | `tests/datasets/skillcorner/derived_actions.csv` | Synthetic CSV fixture: OBE + start_type interceptions, keeper saves, dual-action production |
| Create | `tests/spadl/test_skillcorner.py` | Unit tests: contract, dispatch, coordinates, body part, end coordinates, provenance |
| Create | `tests/spadl/test_skillcorner_inference.py` | Unit tests: start_type interceptions, OBE tackle upgrade, keeper saves, dual-action ordering |
| Create | `tests/invariants/test_skillcorner_invariants.py` | Physical invariants: shot high-x, clearance low-x, coordinate bounds |

---

### Task 1: Schema Constant + Re-exports

**Files:**
- Modify: `silly_kicks/spadl/schema.py:41-46`
- Modify: `silly_kicks/spadl/__init__.py`
- Test: `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/spadl/test_skillcorner.py`:

```python
"""SkillCorner SPADL converter tests."""

import pytest

from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS, SKILLCORNER_SPADL_COLUMNS


class TestSkillcornerSchema:
    """Schema constant structure."""

    def test_extends_kloppy_spadl_columns(self):
        for col, dtype in KLOPPY_SPADL_COLUMNS.items():
            assert col in SKILLCORNER_SPADL_COLUMNS
            assert SKILLCORNER_SPADL_COLUMNS[col] == dtype

    def test_has_action_provenance_column(self):
        assert "action_provenance" in SKILLCORNER_SPADL_COLUMNS
        assert SKILLCORNER_SPADL_COLUMNS["action_provenance"] == "object"

    def test_exactly_one_extra_column(self):
        extra = set(SKILLCORNER_SPADL_COLUMNS) - set(KLOPPY_SPADL_COLUMNS)
        assert extra == {"action_provenance"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestSkillcornerSchema -v`
Expected: FAIL with `ImportError: cannot import name 'SKILLCORNER_SPADL_COLUMNS'`

- [ ] **Step 3: Add SKILLCORNER_SPADL_COLUMNS to schema.py**

Add after the `GRADIENTSPORTS_SPADL_COLUMNS` docstring (after line 78 in `silly_kicks/spadl/schema.py`):

```python
SKILLCORNER_SPADL_COLUMNS: dict[str, str] = {
    **KLOPPY_SPADL_COLUMNS,
    "action_provenance": "object",
}
"""SkillCorner SPADL output schema: KLOPPY_SPADL_COLUMNS (object-dtype IDs) +
``action_provenance`` column (``"native"`` or ``"derived"``). Derived actions
include ``start_type``-based interceptions/recoveries, OBE-enriched tackles,
keeper saves, and synthetic dribbles. See spec §6."""
```

- [ ] **Step 4: Update __init__.py re-exports**

In `silly_kicks/spadl/__init__.py`:

Add `"SKILLCORNER_SPADL_COLUMNS"` to the `__all__` list (after `"SPORTEC_SPADL_COLUMNS"`).

Add `SKILLCORNER_SPADL_COLUMNS` to the import from `.schema`:

```python
from .schema import (
    GRADIENTSPORTS_SPADL_COLUMNS,
    SKILLCORNER_SPADL_COLUMNS,
    SPADL_COLUMNS,
    SPORTEC_SPADL_COLUMNS,
    ConversionReport,
)
```

Add `"skillcorner"` to `__all__` (after `"opta"`).

Add to the lazy-import block at the bottom (after the kloppy try/except):

```python
try:
    from . import skillcorner
except ImportError:
    pass
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestSkillcornerSchema -v`
Expected: 3 PASSED

---

### Task 2: Synthetic Test Fixtures

**Files:**
- Create: `tests/datasets/skillcorner/__init__.py`
- Create: `tests/datasets/skillcorner/match_metadata.json`
- Create: `tests/datasets/skillcorner/basic_possessions.csv`
- Create: `tests/datasets/skillcorner/derived_actions.csv`

- [ ] **Step 1: Create directory and __init__.py**

```bash
mkdir -p tests/datasets/skillcorner
touch tests/datasets/skillcorner/__init__.py
```

- [ ] **Step 2: Create match_metadata.json**

Create `tests/datasets/skillcorner/match_metadata.json`:

```json
{
  "id": 9999999,
  "pitch_length": 105,
  "pitch_width": 68,
  "home_team": {"id": "team_home", "name": "Home FC"},
  "away_team": {"id": "team_away", "name": "Away FC"},
  "periods": [
    {"id": 1, "start_frame": 0, "end_frame": 27000},
    {"id": 2, "start_frame": 27001, "end_frame": 54000}
  ]
}
```

- [ ] **Step 3: Create basic_possessions.csv**

Create `tests/datasets/skillcorner/basic_possessions.csv` — a synthetic fixture with the minimum columns the converter needs, covering every dispatch branch. Each row is a `player_possession` event except where noted.

The CSV must have these columns (subset of the 294-column schema): `event_type`, `event_id`, `period`, `time_start`, `minute_start`, `second_start`, `team_id`, `player_id`, `x_start`, `y_start`, `x_end`, `y_end`, `start_type`, `end_type`, `game_interruption_before`, `game_interruption_after`, `is_header`, `hand_pass`, `player_targeted_x_reception`, `player_targeted_y_reception`, `player_targeted_third_pass`, `player_targeted_channel_pass`.

Rows (all `event_type=player_possession` unless stated):

| Row | event_type | event_id | period | time_start | team_id | player_id | x_start | y_start | x_end | y_end | start_type | end_type | game_interruption_before | game_interruption_after | is_header | hand_pass | player_targeted_x_reception | player_targeted_y_reception | player_targeted_third_pass | player_targeted_channel_pass | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | player_possession | 8_0 | 1 | 00:10.0 | team_home | p1 | 5.0 | 3.0 | 8.0 | 4.0 | pass_reception | pass | | | False | False | 15.0 | 10.0 | middle_third | center | Pass with targeted reception |
| 2 | player_possession | 8_1 | 1 | 00:15.0 | team_home | p2 | 15.0 | 10.0 | 18.0 | 12.0 | pass_reception | pass | | | False | False | | | | | Pass without targeted reception (fallback) |
| 3 | player_possession | 8_2 | 1 | 00:20.0 | team_home | p3 | 40.0 | -30.0 | 42.0 | -28.0 | pass_reception | pass | | | False | False | 45.0 | -25.0 | attacking_third | wide_left | Cross (attacking_third + wide) |
| 4 | player_possession | 8_3 | 1 | 00:25.0 | team_away | p12 | 35.0 | 5.0 | 38.0 | 4.0 | pass_reception | shot | | goal_for | False | False | | | | | Shot → goal |
| 5 | player_possession | 8_4 | 1 | 00:30.0 | team_home | p4 | 30.0 | 2.0 | 33.0 | 1.0 | pass_reception | shot | | | False | False | | | | | Shot → no goal |
| 6 | player_possession | 8_5 | 1 | 00:35.0 | team_away | p13 | 30.0 | 2.0 | 28.0 | 0.0 | pass_reception | shot | | | True | False | | | | | Headed shot |
| 7 | player_possession | 8_6 | 1 | 00:40.0 | team_home | p5 | -20.0 | -5.0 | -15.0 | -3.0 | pass_reception | clearance | | | False | False | | | | | Clearance |
| 8 | player_possession | 8_7 | 1 | 00:45.0 | team_home | p6 | 10.0 | 8.0 | 12.0 | 7.0 | pass_reception | foul_suffered | | free_kick_for | False | False | | | | | Foul suffered |
| 9 | player_possession | 8_8 | 1 | 00:50.0 | team_home | p7 | 10.0 | 8.0 | 15.0 | 5.0 | pass_reception | pass | free_kick_for | | False | False | 20.0 | 3.0 | middle_third | center | Free kick short |
| 10 | player_possession | 8_9 | 1 | 00:55.0 | team_home | p8 | -40.0 | -10.0 | -35.0 | -8.0 | pass_reception | pass | goal_kick_for | | False | False | -20.0 | 0.0 | defensive_third | center | Goal kick |
| 11 | player_possession | 8_10 | 1 | 01:00.0 | team_home | p9 | 5.0 | -33.0 | 8.0 | -30.0 | pass_reception | pass | throw_in_for | | False | False | 15.0 | -25.0 | middle_third | half_space_left | Throw-in |
| 12 | player_possession | 8_11 | 1 | 01:05.0 | team_home | p10 | 45.0 | 30.0 | 48.0 | 28.0 | pass_reception | pass | corner_for | | False | False | 48.0 | 5.0 | attacking_third | center | Corner crossed |
| 13 | player_possession | 8_12 | 1 | 01:10.0 | team_home | p11 | 10.0 | -5.0 | 12.0 | -3.0 | pass_reception | possession_loss | | | False | False | | | | | Possession loss → non_action |
| 14 | player_possession | 8_13 | 1 | 01:15.0 | team_home | p1 | -45.0 | 0.0 | -42.0 | 2.0 | pass_reception | pass | | | False | True | -30.0 | 5.0 | defensive_third | center | GK hand pass |
| 15 | player_possession | 8_14 | 1 | 01:20.0 | team_home | p2 | 5.0 | 5.0 | 8.0 | 3.0 | pass_reception | unknown | | | False | False | | | | | Unknown → non_action |

Write the CSV with these values. Columns not listed above should be empty/NaN.

- [ ] **Step 4: Create derived_actions.csv**

Create `tests/datasets/skillcorner/derived_actions.csv` with both `player_possession` and `on_ball_engagement` rows:

| Row | event_type | event_id | period | time_start | team_id | player_id | x_start | y_start | x_end | y_end | start_type | end_type | game_interruption_before | game_interruption_after | is_header | hand_pass | player_targeted_x_reception | player_targeted_y_reception |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | player_possession | pp_1 | 1 | 00:10.0 | team_home | p1 | 5.0 | 3.0 | 8.0 | 4.0 | pass_reception | pass | | | False | False | 15.0 | 10.0 |
| 2 | player_possession | pp_2 | 1 | 00:15.0 | team_away | p12 | 15.0 | 10.0 | 18.0 | 12.0 | pass_interception | pass | | | False | False | 25.0 | 8.0 |
| 3 | on_ball_engagement | obe_1 | 1 | 00:14.8 | team_away | p12 | 14.0 | 9.0 | | | | direct_regain | | | | | | |
| 4 | player_possession | pp_3 | 1 | 00:20.0 | team_home | p3 | 20.0 | 5.0 | 22.0 | 3.0 | recovery | pass | | | False | False | 30.0 | 5.0 |
| 5 | on_ball_engagement | obe_2 | 1 | 00:19.5 | team_home | p3 | 19.0 | 4.0 | | | | direct_regain | | | | | | |
| 6 | player_possession | pp_4 | 1 | 00:25.0 | team_home | p4 | 30.0 | 2.0 | 33.0 | 1.0 | pass_reception | shot | | | False | False | | |
| 7 | player_possession | pp_5 | 1 | 00:27.0 | team_away | p15 | -40.0 | 0.0 | -38.0 | 2.0 | recovery | pass | | | False | False | -30.0 | 5.0 |
| 8 | player_possession | pp_6 | 1 | 00:30.0 | team_home | p5 | 10.0 | 8.0 | 12.0 | 7.0 | throw_in_interception | pass | | | False | False | 20.0 | 10.0 |
| 9 | player_possession | pp_7 | 1 | 00:35.0 | team_away | p13 | 5.0 | 3.0 | 8.0 | 4.0 | pass_interception | pass | | | False | False | 15.0 | 5.0 |

Row 2: `pass_interception` + OBE `direct_regain` within 2s (row 3) → produces **tackle** (derived, upgraded from interception) then **pass** (native). OBE player/coords used for the tackle.

Row 4: `recovery` + OBE `direct_regain` within 2s (row 5) → produces **tackle** (derived, upgraded) then **pass** (native).

Row 6-7: Shot with no `goal_for`, then next possession is opponent team → produces **keeper_save** (derived) for row 7's player.

Row 8: `throw_in_interception` with no OBE match → produces **interception** (derived, from start_type) then **pass** (native).

Row 9: `pass_interception` with no OBE within 2s → produces **interception** (derived) then **pass** (native).

- [ ] **Step 5: Verify fixtures load correctly**

Run a quick smoke check:

```bash
python -c "
import pandas as pd, json
df = pd.read_csv('tests/datasets/skillcorner/basic_possessions.csv')
print(f'basic: {len(df)} rows, {len(df.columns)} cols')
df2 = pd.read_csv('tests/datasets/skillcorner/derived_actions.csv')
print(f'derived: {len(df2)} rows, {len(df2.columns)} cols')
with open('tests/datasets/skillcorner/match_metadata.json') as f:
    m = json.load(f)
print(f'metadata: pitch={m[\"pitch_length\"]}x{m[\"pitch_width\"]}')
"
```

Expected: `basic: 15 rows`, `derived: 9 rows`, `metadata: pitch=105x68`

---

### Task 3: Coordinate Transform + Time Parsing Helpers

**Files:**
- Create: `silly_kicks/spadl/skillcorner.py` (partial — helpers only)
- Test: `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write failing tests for coordinate transform and time parsing**

Add to `tests/spadl/test_skillcorner.py`:

```python
import numpy as np
import pandas as pd


class TestCoordinateTransform:
    """Spec §5.3: centered meters → SPADL 0-105 x 0-68 frame."""

    def test_center_spot_maps_to_pitch_center(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([0.0]),
            y=pd.Series([0.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 52.5) < 0.01
        assert abs(y.iloc[0] - 34.0) < 0.01

    def test_positive_corner_maps_to_top_right(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([52.5]),
            y=pd.Series([34.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01
        assert abs(y.iloc[0] - 68.0) < 0.01

    def test_negative_corner_maps_to_bottom_left(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([-52.5]),
            y=pd.Series([-34.0]),
            pitch_length=105,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 0.0) < 0.01
        assert abs(y.iloc[0] - 0.0) < 0.01

    def test_non_standard_pitch_rescales(self):
        """104m pitch: half_length=52, so x=52 → rescaled to 105.0."""
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([52.0]),
            y=pd.Series([34.0]),
            pitch_length=104,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01
        assert abs(y.iloc[0] - 68.0) < 0.01

    def test_106m_pitch_rescales(self):
        """106m pitch: half_length=53, so x=53 → 105.0."""
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([53.0]),
            y=pd.Series([34.0]),
            pitch_length=106,
            pitch_width=68,
        )
        assert abs(x.iloc[0] - 105.0) < 0.01

    def test_nan_propagates(self):
        from silly_kicks.spadl.skillcorner import _transform_coords

        x, y = _transform_coords(
            x=pd.Series([float("nan")]),
            y=pd.Series([float("nan")]),
            pitch_length=105,
            pitch_width=68,
        )
        assert pd.isna(x.iloc[0])
        assert pd.isna(y.iloc[0])


class TestTimeParsing:
    """Spec §5.5: parse time_start MM:SS.d string."""

    def test_simple_time(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["00:01.8"]))
        assert abs(result.iloc[0] - 1.8) < 0.01

    def test_multi_minute(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["12:34.5"]))
        assert abs(result.iloc[0] - (12 * 60 + 34.5)) < 0.01

    def test_zero_fraction(self):
        from silly_kicks.spadl.skillcorner import _parse_time_start

        result = _parse_time_start(pd.Series(["05:00.0"]))
        assert abs(result.iloc[0] - 300.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestCoordinateTransform tests/spadl/test_skillcorner.py::TestTimeParsing -v`
Expected: FAIL with `ImportError: cannot import name '_transform_coords'`

- [ ] **Step 3: Implement helpers in skillcorner.py**

Create `silly_kicks/spadl/skillcorner.py`:

```python
"""SkillCorner SPADL converter.

Converts SkillCorner ``dynamic_events.csv`` DataFrames to SPADL actions.

SkillCorner events are possession-centric: the primary event type is
``player_possession`` (one row per possession phase). Defensive actions
(interceptions, tackles) are derived from the ``start_type`` column and
cross-referenced with ``on_ball_engagement`` rows. Keeper saves are
inferred from shot-to-GK possession sequences.

Coordinate system: attacking-direction-normalized centered meters
(origin at center spot, positive x toward the goal being attacked).
This is ``POSSESSION_PERSPECTIVE`` — the same as StatsBomb/Wyscout.
Rescaled to SPADL 0-105 x 0-68 frame using pitch dimensions from
``match_metadata``.

See spec: ``docs/superpowers/specs/2026-05-14-skillcorner-events-converter-design.md``
"""

from __future__ import annotations

import pandas as pd


def _transform_coords(
    x: pd.Series,
    y: pd.Series,
    pitch_length: int | float,
    pitch_width: int | float,
) -> tuple[pd.Series, pd.Series]:
    """Rescale centered meters to SPADL 0-based frame.

    Parameters
    ----------
    x, y : pd.Series
        Coordinates in centered meters (origin at center spot).
    pitch_length, pitch_width : int or float
        Actual pitch dimensions from match_metadata.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(x_spadl, y_spadl)`` in SPADL [0, 105] x [0, 68] frame.
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2
    x_out = (x / half_length) * 52.5 + 52.5
    y_out = (y / half_width) * 34.0 + 34.0
    return x_out, y_out


def _parse_time_start(time_start: pd.Series) -> pd.Series:
    """Parse ``MM:SS.d`` time strings to float seconds.

    Parameters
    ----------
    time_start : pd.Series
        String series in ``"MM:SS.d"`` format (e.g. ``"12:34.5"``).

    Returns
    -------
    pd.Series
        Float64 seconds since period start.
    """
    parts = time_start.str.split(":", expand=True)
    minutes = parts[0].astype("float64")
    seconds = parts[1].astype("float64")
    return minutes * 60 + seconds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestCoordinateTransform tests/spadl/test_skillcorner.py::TestTimeParsing -v`
Expected: 9 PASSED

---

### Task 4: Body Part + Cross Detection Dispatch

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py`
- Test: `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/spadl/test_skillcorner.py`:

```python
class TestBodyPartDispatch:
    """Spec §7.1.1: is_header → head, hand_pass → other, else → foot."""

    def test_default_is_foot(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([False]),
            hand_pass=pd.Series([False]),
        )
        assert result[0] == spadlconfig.bodypart_id["foot"]

    def test_header(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([True]),
            hand_pass=pd.Series([False]),
        )
        assert result[0] == spadlconfig.bodypart_id["head"]

    def test_hand_pass(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([False]),
            hand_pass=pd.Series([True]),
        )
        assert result[0] == spadlconfig.bodypart_id["other"]

    def test_header_takes_priority_over_hand(self):
        from silly_kicks.spadl.skillcorner import _dispatch_bodypart

        result = _dispatch_bodypart(
            is_header=pd.Series([True]),
            hand_pass=pd.Series([True]),
        )
        assert result[0] == spadlconfig.bodypart_id["head"]


class TestCrossDetection:
    """Spec §3.2: native channel/third columns with spatial fallback."""

    def test_attacking_third_wide_left_is_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["attacking_third"]),
            channel=pd.Series(["wide_left"]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([10.0]),
        )
        assert result.iloc[0] is True or result.iloc[0] == True  # noqa: E712

    def test_attacking_third_center_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["attacking_third"]),
            channel=pd.Series(["center"]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([34.0]),
        )
        assert not result.iloc[0]

    def test_middle_third_wide_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series(["middle_third"]),
            channel=pd.Series(["wide_right"]),
            start_x_spadl=pd.Series([50.0]),
            start_y_spadl=pd.Series([60.0]),
        )
        assert not result.iloc[0]

    def test_nan_columns_fall_back_to_spatial(self):
        """When native columns are NaN, use spatial heuristic."""
        from silly_kicks.spadl.skillcorner import _is_cross

        # In attacking third (>70) and wide channel (<15)
        result = _is_cross(
            third=pd.Series([None]),
            channel=pd.Series([None]),
            start_x_spadl=pd.Series([80.0]),
            start_y_spadl=pd.Series([10.0]),
        )
        assert result.iloc[0]

    def test_nan_columns_not_in_zone_is_not_cross(self):
        from silly_kicks.spadl.skillcorner import _is_cross

        result = _is_cross(
            third=pd.Series([None]),
            channel=pd.Series([None]),
            start_x_spadl=pd.Series([50.0]),
            start_y_spadl=pd.Series([34.0]),
        )
        assert not result.iloc[0]
```

Add this import at the top of the test file:

```python
from silly_kicks.spadl import config as spadlconfig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestBodyPartDispatch tests/spadl/test_skillcorner.py::TestCrossDetection -v`
Expected: FAIL with `ImportError: cannot import name '_dispatch_bodypart'`

- [ ] **Step 3: Implement dispatch helpers**

Add to `silly_kicks/spadl/skillcorner.py`:

```python
import numpy as np

from . import config as spadlconfig


def _dispatch_bodypart(
    is_header: pd.Series,
    hand_pass: pd.Series,
) -> np.ndarray:
    """Map SkillCorner body part booleans to SPADL bodypart_id.

    Priority: is_header > hand_pass > default foot.
    """
    return np.select(
        [is_header.fillna(False).astype(bool), hand_pass.fillna(False).astype(bool)],
        [spadlconfig.bodypart_id["head"], spadlconfig.bodypart_id["other"]],
        default=spadlconfig.bodypart_id["foot"],
    )


def _is_cross(
    third: pd.Series,
    channel: pd.Series,
    start_x_spadl: pd.Series,
    start_y_spadl: pd.Series,
) -> pd.Series:
    """Detect crosses using native SC columns with spatial fallback.

    A pass is a cross when it originates in the attacking third from a
    wide channel. Uses ``player_targeted_third_pass`` /
    ``player_targeted_channel_pass`` when available (~98%), falling back
    to a coordinate heuristic for NaN rows.
    """
    has_native = third.notna() & channel.notna()
    native_cross = (third == "attacking_third") & channel.isin({"wide_left", "wide_right"})
    spatial_cross = (start_x_spadl > 70.0) & ((start_y_spadl < 15.0) | (start_y_spadl > 53.0))
    return (has_native & native_cross) | (~has_native & spatial_cross)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestBodyPartDispatch tests/spadl/test_skillcorner.py::TestCrossDetection -v`
Expected: 9 PASSED

---

### Task 5: Derived Action Inference Module

**Files:**
- Create: `silly_kicks/spadl/_skillcorner_inference.py`
- Create: `tests/spadl/test_skillcorner_inference.py`

- [ ] **Step 1: Write failing tests for start_type interceptions**

Create `tests/spadl/test_skillcorner_inference.py`:

```python
"""SkillCorner derived action inference tests."""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig


class TestStartTypeInterceptions:
    """Spec §4.1: start_type-based interception detection."""

    def test_pass_interception_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "pass_interception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]
        assert result.iloc[0]["player_id"] == "p12"

    def test_recovery_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "recovery"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_throw_in_interception_produces_interception(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "throw_in_interception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_pass_reception_produces_no_defensive_action(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_a"],
            "player_id": ["p1", "p2"],
            "start_type": ["pass_reception", "pass_reception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame(columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"])
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 0


class TestOBETackleUpgrade:
    """Spec §4.1: OBE direct_regain upgrades interception → tackle."""

    def test_interception_upgraded_to_tackle_with_obe(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "pass_interception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame({
            "period": [1],
            "time_seconds": [14.8],
            "team_id": ["team_b"],
            "player_id": ["p13"],
            "end_type": ["direct_regain"],
            "x_start": [14.0],
            "y_start": [9.0],
        })
        result = infer_defensive_actions(pp, obe)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]
        assert result.iloc[0]["player_id"] == "p13"
        assert abs(result.iloc[0]["start_x"] - 14.0) < 0.01

    def test_recovery_upgraded_to_tackle_with_obe(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "recovery"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame({
            "period": [1],
            "time_seconds": [14.5],
            "team_id": ["team_b"],
            "player_id": ["p14"],
            "end_type": ["direct_regain"],
            "x_start": [16.0],
            "y_start": [11.0],
        })
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]

    def test_no_upgrade_when_obe_too_far_in_time(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "pass_interception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame({
            "period": [1],
            "time_seconds": [12.0],
            "team_id": ["team_b"],
            "player_id": ["p13"],
            "end_type": ["direct_regain"],
            "x_start": [14.0],
            "y_start": [9.0],
        })
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]

    def test_indirect_regain_does_not_upgrade(self):
        from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p12"],
            "start_type": ["pass_reception", "pass_interception"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        })
        obe = pd.DataFrame({
            "period": [1],
            "time_seconds": [14.8],
            "team_id": ["team_b"],
            "player_id": ["p13"],
            "end_type": ["indirect_regain"],
            "x_start": [14.0],
            "y_start": [9.0],
        })
        result = infer_defensive_actions(pp, obe)
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["interception"]


class TestKeeperSaves:
    """Spec §4.2: shot → opponent possession = keeper_save."""

    def test_shot_followed_by_opponent_produces_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p15"],
            "end_type": ["shot", "pass"],
            "x_start": [30.0, -40.0],
            "y_start": [2.0, 0.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 1
        assert result.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]
        assert result.iloc[0]["player_id"] == "p15"
        assert result.iloc[0]["result_id"] == spadlconfig.result_id["success"]

    def test_shot_followed_by_goal_no_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_a"],
            "player_id": ["p1", "p2"],
            "end_type": ["shot", "pass"],
            "game_interruption_after": ["goal_for", None],
            "x_start": [30.0, 5.0],
            "y_start": [2.0, 3.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_shot_followed_by_same_team_no_keeper_save(self):
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_a"],
            "player_id": ["p1", "p2"],
            "end_type": ["shot", "pass"],
            "x_start": [30.0, 5.0],
            "y_start": [2.0, 3.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_goal_kick_against_no_keeper_save(self):
        """goal_kick_against = shot missed wide/over bar — not a save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p15"],
            "end_type": ["shot", "pass"],
            "game_interruption_after": ["goal_kick_against", None],
            "x_start": [30.0, -40.0],
            "y_start": [2.0, 0.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 0

    def test_corner_for_produces_keeper_save(self):
        """corner_for after shot = deflected behind goal line — plausible save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p15"],
            "end_type": ["shot", "pass"],
            "game_interruption_after": ["corner_for", None],
            "x_start": [30.0, -40.0],
            "y_start": [2.0, 0.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 1

    @pytest.mark.parametrize("gi_after_val", [
        "free_kick_against",
        "throw_in_against",
        "throw_in_for",
    ])
    def test_non_save_gi_after_excluded(self, gi_after_val):
        """gi_after values that indicate non-save outcomes produce no keeper_save."""
        from silly_kicks.spadl._skillcorner_inference import infer_keeper_saves

        pp = pd.DataFrame({
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [25.0, 27.0],
            "team_id": ["team_a", "team_b"],
            "player_id": ["p1", "p15"],
            "end_type": ["shot", "pass"],
            "game_interruption_after": [gi_after_val, None],
            "x_start": [30.0, -40.0],
            "y_start": [2.0, 0.0],
        })
        result = infer_keeper_saves(pp)
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_skillcorner_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silly_kicks.spadl._skillcorner_inference'`

- [ ] **Step 3: Implement _skillcorner_inference.py**

Create `silly_kicks/spadl/_skillcorner_inference.py`:

```python
"""SkillCorner derived action inference.

Produces SPADL actions that are not directly mapped from ``player_possession``
dispatch but inferred from cross-referencing event types:

- Defensive actions (interceptions, tackles) from ``start_type`` + OBE
- Keeper saves from shot → opponent-possession sequences

All returned DataFrames have partial SPADL columns (at minimum:
``period_id``, ``time_seconds``, ``team_id``, ``player_id``, ``start_x``,
``start_y``, ``type_id``, ``result_id``, ``bodypart_id``,
``action_provenance``). The caller merges them into the main action stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as spadlconfig

_OBE_TEMPORAL_WINDOW: float = 2.0  # seconds


def _is_defensive_start_type(start_type: pd.Series) -> pd.Series:
    """True for start_type values that indicate a possession win."""
    return start_type.str.endswith("_interception", na=False) | (start_type == "recovery")


def infer_defensive_actions(
    pp: pd.DataFrame,
    obe: pd.DataFrame,
) -> pd.DataFrame:
    """Infer interceptions and tackles from start_type + OBE cross-referencing.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``event_id``, ``period``, ``time_seconds``,
        ``team_id``, ``player_id``, ``start_type``, ``x_start``, ``y_start``.
    obe : pd.DataFrame
        ``on_ball_engagement`` rows.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.

    Returns
    -------
    pd.DataFrame
        Derived defensive actions with partial SPADL columns.
    """
    defensive_mask = _is_defensive_start_type(pp["start_type"])
    if not defensive_mask.any():
        return pd.DataFrame()

    rows: list[dict] = []
    obe_regains = obe[obe["end_type"] == "direct_regain"] if len(obe) > 0 else pd.DataFrame()

    for idx in pp.index[defensive_mask]:
        row = pp.loc[idx]
        period = row["period"]
        t = row["time_seconds"]

        # Default: interception from start_type
        action_type = spadlconfig.actiontype_id["interception"]
        player = row["player_id"]
        team = row["team_id"]
        x = row["x_start"]
        y = row["y_start"]

        # OBE upgrade: check for direct_regain within temporal window + same team
        if len(obe_regains) > 0:
            candidates = obe_regains[
                (obe_regains["period"] == period)
                & (obe_regains["team_id"] == row["team_id"])
                & ((obe_regains["time_seconds"] - t).abs() <= _OBE_TEMPORAL_WINDOW)
            ]
            if len(candidates) > 0:
                best = candidates.iloc[(candidates["time_seconds"] - t).abs().argmin()]
                action_type = spadlconfig.actiontype_id["tackle"]
                player = best["player_id"]
                team = best["team_id"]
                x = best["x_start"]
                y = best["y_start"]

        rows.append({
            "event_id": row["event_id"],
            "period_id": int(period),
            "time_seconds": float(t) - 0.01,  # just before the native action
            "team_id": team,
            "player_id": player,
            "start_x": float(x),
            "start_y": float(y),
            "end_x": float(x),
            "end_y": float(y),
            "type_id": action_type,
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": spadlconfig.bodypart_id["foot"],
            "action_provenance": "derived",
        })

    return pd.DataFrame(rows)


def infer_keeper_saves(pp: pd.DataFrame) -> pd.DataFrame:
    """Infer keeper saves from shot → opponent-possession sequences.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.
        Optional: ``game_interruption_after``.

    Returns
    -------
    pd.DataFrame
        Derived keeper_save actions with partial SPADL columns.
    """
    rows: list[dict] = []
    end_types = pp["end_type"].to_numpy()
    team_ids = pp["team_id"].to_numpy()
    gia = pp["game_interruption_after"].to_numpy() if "game_interruption_after" in pp.columns else np.full(len(pp), None)

    for i in range(len(pp) - 1):
        if end_types[i] != "shot":
            continue
        # Only infer save for on-target shots: gi_after is NaN (plausible save)
        # or corner_for (deflected behind goal line by GK/defender).
        # Skip: goal_for (scored), goal_kick_against (missed wide/over bar),
        # free_kick_against (foul), throw_in_* (unusual — not a save).
        # NOTE: keeper save is attributed to whoever starts the next possession,
        # which is typically an outfield player (e.g. CB taking the goal kick),
        # NOT the actual GK. This is a data limitation — SkillCorner does not
        # tag saves natively.
        if gia[i] == "goal_for":
            continue
        if not (pd.isna(gia[i]) or gia[i] == "corner_for"):
            continue
        if team_ids[i] == team_ids[i + 1]:
            continue

        next_row = pp.iloc[i + 1]
        rows.append({
            "period_id": int(next_row["period"]),
            "time_seconds": float(next_row["time_seconds"]) - 0.01,
            "team_id": next_row["team_id"],
            "player_id": next_row["player_id"],
            "start_x": float(next_row["x_start"]),
            "start_y": float(next_row["y_start"]),
            "end_x": float(next_row["x_start"]),
            "end_y": float(next_row["y_start"]),
            "type_id": spadlconfig.actiontype_id["keeper_save"],
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": spadlconfig.bodypart_id["foot"],
            "action_provenance": "derived",
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_skillcorner_inference.py -v`
Expected: 16 PASSED

---

### Task 6: Main Converter — `convert_to_actions`

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py`
- Test: `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write failing tests for the main converter**

Add to `tests/spadl/test_skillcorner.py`:

```python
import json
from pathlib import Path

from silly_kicks.spadl.schema import SKILLCORNER_SPADL_COLUMNS, ConversionReport

_FIXTURE_DIR = Path(__file__).parent.parent / "datasets" / "skillcorner"


def _load_basic_fixture():
    events = pd.read_csv(_FIXTURE_DIR / "basic_possessions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        metadata = json.load(f)
    return events, metadata


class TestConvertToActionsContract:
    """Contract: return shape, schema, dtypes."""

    def test_returns_tuple(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        result = convert_to_actions(events, meta)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], ConversionReport)

    def test_output_schema_matches(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        expected_cols = set(SKILLCORNER_SPADL_COLUMNS.keys())
        assert set(actions.columns) == expected_cols

    def test_dtypes_match_schema(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        for col, expected_dtype in SKILLCORNER_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, (
                f"Column {col}: expected {expected_dtype}, got {actions[col].dtype}"
            )

    def test_provenance_values(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        valid = {"native", "derived"}
        assert set(actions["action_provenance"].unique()) <= valid

    def test_conversion_report_provider(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        _, report = convert_to_actions(events, meta)
        assert report.provider == "skillcorner"


class TestActionDispatch:
    """Spec §3.1: end_type + game_interruption_before dispatch."""

    def test_pass_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "pass" in named["type_name"].values

    def test_shot_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "shot" in named["type_name"].values

    def test_goal_is_success(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        # Row 4 in fixture is a goal (game_interruption_after=goal_for)
        goal_shots = shots[shots["result_id"] == spadlconfig.result_id["success"]]
        assert len(goal_shots) >= 1

    def test_clearance_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "clearance" in named["type_name"].values

    def test_goalkick_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "goalkick" in named["type_name"].values

    def test_throw_in_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "throw_in" in named["type_name"].values

    def test_foul_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "foul" in named["type_name"].values

    def test_cross_dispatched(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "cross" in named["type_name"].values

    def test_possession_loss_is_non_action(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")
        assert "non_action" in named["type_name"].values

    def test_header_bodypart(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        head_actions = actions[actions["bodypart_id"] == spadlconfig.bodypart_id["head"]]
        assert len(head_actions) >= 1

    def test_hand_pass_bodypart(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        other_actions = actions[actions["bodypart_id"] == spadlconfig.bodypart_id["other"]]
        assert len(other_actions) >= 1


class TestEndCoordinates:
    """Spec §5.4: pass uses player_targeted_x_reception, fallback to x_end."""

    def test_pass_with_targeted_reception_uses_reception(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        # Row 1 is a pass with targeted_x_reception=15.0 vs x_end=8.0
        # After transform (105m pitch): targeted → (15/52.5)*52.5+52.5 = 67.5
        # x_end would be (8/52.5)*52.5+52.5 = 60.5
        passes = actions[actions["type_id"] == spadlconfig.actiontype_id["pass"]]
        first_pass = passes.iloc[0]
        # end_x should be based on targeted reception (15.0), not x_end (8.0)
        expected_end_x = (15.0 / 52.5) * 52.5 + 52.5  # = 67.5
        assert abs(first_pass["end_x"] - expected_end_x) < 0.5

    def test_pass_without_targeted_reception_uses_x_end(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_basic_fixture()
        actions, _ = convert_to_actions(events, meta)
        # Row 2 has no targeted_x_reception, x_end=18.0
        passes = actions[actions["type_id"] == spadlconfig.actiontype_id["pass"]]
        # Find the pass corresponding to row 2 (second pass, player p2)
        p2_passes = passes[passes["player_id"] == "p2"]
        if len(p2_passes) > 0:
            expected_end_x = (18.0 / 52.5) * 52.5 + 52.5  # = 70.5
            assert abs(p2_passes.iloc[0]["end_x"] - expected_end_x) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestConvertToActionsContract -v`
Expected: FAIL with `ImportError: cannot import name 'convert_to_actions'`

- [ ] **Step 3: Implement convert_to_actions**

Add to `silly_kicks/spadl/skillcorner.py` (extending what was created in Tasks 3-4):

```python
from .base import _add_dribbles, _fix_clearances
from .orientation import POSSESSION_PERSPECTIVE, to_spadl_ltr
from .schema import SKILLCORNER_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output
from ._skillcorner_inference import infer_defensive_actions, infer_keeper_saves


def convert_to_actions(
    events: pd.DataFrame,
    match_metadata: dict,
    *,
    preserve_native: bool = False,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert SkillCorner dynamic_events.csv to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Full ``dynamic_events.csv`` as a DataFrame (all 294 columns;
        the converter selects what it needs).
    match_metadata : dict
        Parsed ``match.json`` dict. Required keys: ``pitch_length``,
        ``pitch_width``, ``home_team`` (with ``id`` sub-key).
    preserve_native : bool, default False
        When True, attach ``original_event_id`` (the SC ``event_id``)
        as an extra column.

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions and conversion audit trail.

    Examples
    --------
    Convert a single match::

        import pandas as pd, json
        from silly_kicks.spadl import skillcorner

        events = pd.read_csv("dynamic_events.csv", low_memory=False)
        with open("match.json") as f:
            meta = json.load(f)
        actions, report = skillcorner.convert_to_actions(events, meta)
        assert not report.has_unrecognized

    See NOTICE for full bibliographic citations.
    """
    pitch_length = match_metadata["pitch_length"]
    pitch_width = match_metadata["pitch_width"]
    home_team_id = str(match_metadata["home_team"]["id"])

    # --- Filter to player_possession rows with valid actors ---
    pp = events[events["event_type"] == "player_possession"].copy()
    pp = pp[pp["player_id"].notna() & pp["team_id"].notna()].copy()
    pp["team_id"] = pp["team_id"].astype(str)
    pp["player_id"] = pp["player_id"].astype(str)

    # OBE rows for tackle enrichment
    obe = events[events["event_type"] == "on_ball_engagement"].copy()
    if len(obe) > 0:
        obe["team_id"] = obe["team_id"].astype(str)
        obe["player_id"] = obe["player_id"].astype(str)

    # --- Time parsing ---
    pp["time_seconds"] = _parse_time_start(pp["time_start"])
    if len(obe) > 0 and "time_start" in obe.columns:
        obe["time_seconds"] = _parse_time_start(obe["time_start"])

    total_pp = len(pp)

    # --- Coordinate transform (start) ---
    sx, sy = _transform_coords(
        pp["x_start"].astype("float64"),
        pp["y_start"].astype("float64"),
        pitch_length, pitch_width,
    )

    # --- End coordinates: per-action-type strategy ---
    # Default: use x_end/y_end (carrier's end position)
    raw_end_x = pp["x_end"].astype("float64")
    raw_end_y = pp["y_end"].astype("float64")

    # For passes/crosses/set-pieces: prefer player_targeted_x_reception
    has_targeted = pp["player_targeted_x_reception"].notna()
    use_targeted = has_targeted & (pp["end_type"] == "pass")

    end_x_raw = raw_end_x.copy()
    end_y_raw = raw_end_y.copy()
    end_x_raw[use_targeted] = pp.loc[use_targeted, "player_targeted_x_reception"].astype("float64")
    end_y_raw[use_targeted] = pp.loc[use_targeted, "player_targeted_y_reception"].astype("float64")

    ex, ey = _transform_coords(end_x_raw, end_y_raw, pitch_length, pitch_width)

    # --- Body part dispatch ---
    bodypart_arr = _dispatch_bodypart(
        pp["is_header"] if "is_header" in pp.columns else pd.Series(False, index=pp.index),
        pp["hand_pass"] if "hand_pass" in pp.columns else pd.Series(False, index=pp.index),
    )

    # --- Action type + result dispatch ---
    gi_before = pp["game_interruption_before"].fillna("") if "game_interruption_before" in pp.columns else pd.Series("", index=pp.index)
    gi_after = pp["game_interruption_after"].fillna("") if "game_interruption_after" in pp.columns else pd.Series("", index=pp.index)
    end_type = pp["end_type"].fillna("") if "end_type" in pp.columns else pd.Series("", index=pp.index)

    # Cross detection
    third_col = pp["player_targeted_third_pass"] if "player_targeted_third_pass" in pp.columns else pd.Series(dtype="object", index=pp.index)
    channel_col = pp["player_targeted_channel_pass"] if "player_targeted_channel_pass" in pp.columns else pd.Series(dtype="object", index=pp.index)
    cross_mask = _is_cross(third_col, channel_col, sx, sy)

    # Next-possession lookups for result logic
    next_team = pp["team_id"].shift(-1)
    same_team_next = (pp["team_id"] == next_team).fillna(False)

    # Short corner/freekick detection: next action same team within 15m
    next_sx = sx.shift(-1)
    next_sy = sy.shift(-1)
    dist_to_next = np.sqrt((sx - next_sx) ** 2 + (sy - next_sy) ** 2)
    is_short = same_team_next & (dist_to_next < 15.0)

    # --- Vectorized dispatch ---
    # Priority 1: set pieces from game_interruption_before
    # Exclude shots — a free kick or corner that ends with a shot is a shot, not a set piece pass
    is_goalkick = gi_before == "goal_kick_for"
    is_corner = (gi_before == "corner_for") & (end_type != "shot")
    is_throw_in = gi_before == "throw_in_for"
    is_freekick = (gi_before == "free_kick_for") & (end_type != "shot")

    # Priority 2: end_type
    is_shot = end_type == "shot"
    is_pass = (end_type == "pass") & ~cross_mask
    is_cross_action = (end_type == "pass") & cross_mask
    is_clearance = end_type == "clearance"
    # NOTE: foul is attributed to the fouled player (possession holder), not the
    # fouler. Other providers (StatsBomb, Wyscout, Gradient Sports) attribute to the
    # fouler. SkillCorner's foul_suffered is from the victim's perspective and OBE
    # foul_committed cross-referencing is not implemented. This affects per-player
    # VAEP foul credit in cross-provider analyses.
    is_foul = end_type == "foul_suffered"

    # Priority 3: residuals
    is_possession_loss = end_type == "possession_loss"
    is_unknown = end_type == "unknown"

    type_id_arr = np.select(
        [
            is_goalkick,
            is_corner & is_short,
            is_corner & ~is_short,
            is_throw_in,
            is_freekick & is_short,
            is_freekick & ~is_short,
            is_shot,
            is_cross_action,
            is_pass,
            is_clearance,
            is_foul,
            is_possession_loss,
            is_unknown,
        ],
        [
            spadlconfig.actiontype_id["goalkick"],
            spadlconfig.actiontype_id["corner_short"],
            spadlconfig.actiontype_id["corner_crossed"],
            spadlconfig.actiontype_id["throw_in"],
            spadlconfig.actiontype_id["freekick_short"],
            spadlconfig.actiontype_id["freekick_crossed"],
            spadlconfig.actiontype_id["shot"],
            spadlconfig.actiontype_id["cross"],
            spadlconfig.actiontype_id["pass"],
            spadlconfig.actiontype_id["clearance"],
            spadlconfig.actiontype_id["foul"],
            spadlconfig.actiontype_id["non_action"],
            spadlconfig.actiontype_id["non_action"],
        ],
        default=spadlconfig.actiontype_id["non_action"],
    )

    # Result dispatch
    is_goal = gi_after == "goal_for"
    result_id_arr = np.select(
        [
            is_goalkick,
            is_clearance,
            is_foul,
            is_shot & is_goal,
            is_shot & ~is_goal,
            same_team_next,
            ~same_team_next,
        ],
        [
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["fail"],
            spadlconfig.result_id["success"],
            spadlconfig.result_id["fail"],
        ],
        default=spadlconfig.result_id["fail"],
    )

    # --- Build native actions DataFrame ---
    game_id = str(match_metadata.get("id", "unknown"))
    actions = pd.DataFrame({
        "game_id": game_id,
        "original_event_id": pp["event_id"].astype("object").values,
        "action_id": np.arange(len(pp), dtype="int64"),
        "period_id": pp["period"].astype("int64").values,
        "time_seconds": pp["time_seconds"].values,
        "team_id": pp["team_id"].values,
        "player_id": pp["player_id"].values,
        "start_x": sx.values,
        "start_y": sy.values,
        "end_x": ex.values,
        "end_y": ey.values,
        "type_id": type_id_arr,
        "result_id": result_id_arr,
        "bodypart_id": bodypart_arr,
        "action_provenance": "native",
    })

    # --- Derived actions ---
    # Prepare OBE for inference (transform coordinates)
    if len(obe) > 0:
        obe_for_inference = obe.copy()
    else:
        obe_for_inference = pd.DataFrame(
            columns=["period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]
        )

    pp_for_inference = pp[["event_id", "period", "time_seconds", "team_id", "player_id", "start_type", "x_start", "y_start"]].copy()

    defensive = infer_defensive_actions(pp_for_inference, obe_for_inference)
    if len(defensive) > 0:
        # Transform defensive action coordinates
        d_sx, d_sy = _transform_coords(
            pd.Series(defensive["start_x"].values, dtype="float64"),
            pd.Series(defensive["start_y"].values, dtype="float64"),
            pitch_length, pitch_width,
        )
        defensive["start_x"] = d_sx.values
        defensive["start_y"] = d_sy.values
        defensive["end_x"] = d_sx.values
        defensive["end_y"] = d_sy.values
        defensive["game_id"] = game_id
        defensive["original_event_id"] = defensive["event_id"]
        defensive["action_id"] = 0  # will be re-indexed

    keeper_saves_pp = pp[["event_id", "period", "time_seconds", "team_id", "player_id", "end_type", "x_start", "y_start"]].copy()
    if "game_interruption_after" in pp.columns:
        keeper_saves_pp["game_interruption_after"] = pp["game_interruption_after"].values

    ks = infer_keeper_saves(keeper_saves_pp)
    if len(ks) > 0:
        ks_sx, ks_sy = _transform_coords(
            pd.Series(ks["start_x"].values, dtype="float64"),
            pd.Series(ks["start_y"].values, dtype="float64"),
            pitch_length, pitch_width,
        )
        ks["start_x"] = ks_sx.values
        ks["start_y"] = ks_sy.values
        ks["end_x"] = ks_sx.values
        ks["end_y"] = ks_sy.values
        ks["game_id"] = game_id
        ks["original_event_id"] = pd.NA
        ks["action_id"] = 0

    # Merge all actions
    parts = [actions]
    if len(defensive) > 0:
        parts.append(defensive[actions.columns])
    if len(ks) > 0:
        parts.append(ks[actions.columns])

    actions = pd.concat(parts, ignore_index=True)
    actions = actions.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    actions["action_id"] = np.arange(len(actions), dtype="int64")

    # --- Post-processors ---
    actions = _fix_clearances(actions)
    actions = _add_dribbles(actions)

    # Mark dribbles as derived
    dribble_mask = actions["type_id"] == spadlconfig.actiontype_id["dribble"]
    actions.loc[dribble_mask, "action_provenance"] = "derived"

    # --- LTR normalization (no-op for possession perspective) ---
    actions = to_spadl_ltr(
        actions, input_convention=POSSESSION_PERSPECTIVE, home_team_id=home_team_id,
    )

    # --- ConversionReport ---
    from collections import Counter

    mapped_counts: dict[str, int] = {}
    id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
    for tid in actions["type_id"].to_numpy():
        name = id_to_name.get(int(tid), "unknown")
        mapped_counts[name] = mapped_counts.get(name, 0) + 1

    # NOTE: on_ball_engagement rows are consumed for tackle enrichment but are not
    # directly mapped to native actions, so they appear in excluded_counts alongside
    # passing_option and off_ball_run. This matches other converters' semantics where
    # "excluded" means "not directly mapped to a SPADL action row".
    excluded_types = events[~events["event_type"].isin({"player_possession"})]["event_type"]
    excluded_counts = dict(Counter(excluded_types))

    # Field names verified against schema.py ConversionReport dataclass:
    # mapped_counts, excluded_counts, unrecognized_counts (not _types)
    report = ConversionReport(
        provider="skillcorner",
        total_events=total_pp,
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts={},
    )

    # --- Finalize output ---
    extra = ["original_event_id"] if preserve_native else None
    actions = _finalize_output(actions, SKILLCORNER_SPADL_COLUMNS, extra_columns=extra)

    return actions, report
```

- [ ] **Step 4: Run all converter tests**

Run: `python -m pytest tests/spadl/test_skillcorner.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run linter + type checker**

Run: `python -m ruff check silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py`
Run: `python -m ruff format --check silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py`

Fix any issues before proceeding.

---

### Task 7: Derived Actions Integration Tests

**Files:**
- Test: `tests/spadl/test_skillcorner.py`

- [ ] **Step 1: Write tests for dual-action production and derived action integration**

Add to `tests/spadl/test_skillcorner.py`:

```python
def _load_derived_fixture():
    events = pd.read_csv(_FIXTURE_DIR / "derived_actions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        metadata = json.load(f)
    return events, metadata


class TestDerivedActions:
    """Spec §4: dual-action production, start_type interceptions, OBE tackles, keeper saves."""

    def test_interception_with_obe_upgraded_to_tackle(self):
        """Row 2 + OBE row 3: pass_interception + direct_regain → tackle."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        tackles = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]]
        assert len(tackles) >= 1

    def test_recovery_with_obe_upgraded_to_tackle(self):
        """Row 4 + OBE row 5: recovery + direct_regain → tackle."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        tackles = actions[actions["type_id"] == spadlconfig.actiontype_id["tackle"]]
        assert len(tackles) >= 2  # both row 2 and row 4

    def test_interception_without_obe_stays_interception(self):
        """Rows 8-9: interceptions with no nearby OBE → interception (not tackle)."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        interceptions = actions[actions["type_id"] == spadlconfig.actiontype_id["interception"]]
        assert len(interceptions) >= 1

    def test_keeper_save_after_shot(self):
        """Rows 6-7: shot followed by opponent possession → keeper_save."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        ks = actions[actions["type_id"] == spadlconfig.actiontype_id["keeper_save"]]
        assert len(ks) >= 1
        assert (ks["action_provenance"] == "derived").all()

    def test_dual_action_interception_before_native(self):
        """Spec §4 dual-action: derived interception ordered before native pass."""
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        named = actions.merge(spadlconfig.actiontypes_df(), how="left")

        # Find interceptions/tackles (derived) and check they precede a pass
        derived = named[named["action_provenance"] == "derived"]
        defensive = derived[derived["type_name"].isin({"interception", "tackle"})]
        for idx in defensive.index:
            pos = actions.index.get_loc(idx)
            if pos < len(actions) - 1:
                next_action = named.iloc[pos + 1]
                # The native action should follow immediately
                assert next_action["action_provenance"] == "native"

    def test_derived_actions_have_provenance(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        derived = actions[actions["action_provenance"] == "derived"]
        # Should have tackles, interceptions, keeper_saves, and dribbles
        assert len(derived) >= 3

    def test_dribbles_are_derived(self):
        from silly_kicks.spadl.skillcorner import convert_to_actions

        events, meta = _load_derived_fixture()
        actions, _ = convert_to_actions(events, meta)
        dribbles = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
        if len(dribbles) > 0:
            assert (dribbles["action_provenance"] == "derived").all()
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/spadl/test_skillcorner.py::TestDerivedActions -v`
Expected: All PASS (may require fixture tuning if spatial parameters don't trigger dribble synthesis)

---

### Task 8: Physical-Invariant Tests

**Files:**
- Create: `tests/invariants/test_skillcorner_invariants.py`

- [ ] **Step 1: Write invariant tests**

Create `tests/invariants/test_skillcorner_invariants.py`:

```python
"""Physical invariants for SkillCorner SPADL converter.

Verify that converted actions satisfy spatial properties that must hold
regardless of the specific match data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.skillcorner import convert_to_actions

_FIXTURE_DIR = Path(__file__).parent.parent / "datasets" / "skillcorner"


@pytest.fixture
def actions():
    events = pd.read_csv(_FIXTURE_DIR / "basic_possessions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        meta = json.load(f)
    actions, _ = convert_to_actions(events, meta)
    return actions


class TestSpatialInvariants:
    """Coordinates must be in valid SPADL range."""

    def test_start_x_in_range(self, actions):
        valid = actions["start_x"].dropna()
        assert (valid >= -1.0).all(), f"start_x below -1: {valid[valid < -1.0].tolist()}"
        assert (valid <= 106.0).all(), f"start_x above 106: {valid[valid > 106.0].tolist()}"

    def test_start_y_in_range(self, actions):
        valid = actions["start_y"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 69.0).all()

    def test_shots_in_attacking_half(self, actions):
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        if len(shots) > 0:
            assert (shots["start_x"] > 52.5).all(), (
                f"Shot(s) in defensive half: {shots['start_x'].tolist()}"
            )

    def test_clearances_in_defensive_half(self, actions):
        clearances = actions[actions["type_id"] == spadlconfig.actiontype_id["clearance"]]
        if len(clearances) > 0:
            assert (clearances["start_x"] < 52.5).all(), (
                f"Clearance(s) in attacking half: {clearances['start_x'].tolist()}"
            )

    def test_goalkick_in_defensive_half(self, actions):
        goalkicks = actions[actions["type_id"] == spadlconfig.actiontype_id["goalkick"]]
        if len(goalkicks) > 0:
            assert (goalkicks["start_x"] < 30.0).all(), (
                f"Goalkick(s) too far from own goal: {goalkicks['start_x'].tolist()}"
            )


class TestTemporalInvariants:
    """Time must be monotonic within periods."""

    def test_time_seconds_non_negative(self, actions):
        assert (actions["time_seconds"] >= 0).all()

    def test_time_seconds_monotonic_within_period(self, actions):
        for period in actions["period_id"].unique():
            period_actions = actions[actions["period_id"] == period]
            times = period_actions["time_seconds"].values
            # Allow equal timestamps but not decreasing
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
                f"Non-monotonic time_seconds in period {period}"
            )
```

- [ ] **Step 2: Run invariant tests**

Run: `python -m pytest tests/invariants/test_skillcorner_invariants.py -v`
Expected: All PASS

---

### Task 9: Full Test Suite Run + Lint

**Files:** All created/modified files

- [ ] **Step 1: Run the complete test suite (excluding e2e)**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: All existing tests still pass + all new tests pass

- [ ] **Step 2: Run ruff lint + format**

Run: `python -m ruff check silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py tests/spadl/test_skillcorner.py tests/spadl/test_skillcorner_inference.py tests/invariants/test_skillcorner_invariants.py`
Run: `python -m ruff format --check silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py tests/spadl/test_skillcorner.py tests/spadl/test_skillcorner_inference.py tests/invariants/test_skillcorner_invariants.py`

Fix any issues.

- [ ] **Step 3: Run pyright**

Run: `python -m pyright silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py`

Fix any type errors.

- [ ] **Step 4: Verify no regressions**

Run: `python -m pytest tests/ -m "not e2e" --tb=short -q`
Expected: Same pass count as before + new tests, zero failures.

---

### Task 10: e2e Tests (local only, marked e2e)

**Files:**
- Create: `tests/spadl/test_skillcorner_e2e.py`

- [ ] **Step 1: Write e2e tests**

Data layout: each match lives in `{SKILLCORNER_SAMPLE_DIR}/{match_id}/`. The pining-for-the-data API serves artifacts as `{match_id}_dynamic_events.csv` and `{match_id}_match.json`. The fixture loader globs for `*dynamic_events.csv` / `*match.json` to handle both bare and prefixed filenames. The 10 match IDs are the pining-for-the-data A-League set downloaded in the brainstorming phase.

Create `tests/spadl/test_skillcorner_e2e.py`:

```python
"""SkillCorner SPADL converter e2e tests.

Requires the 10 A-League matches downloaded from pining-for-the-data API
at C:\\Users\\Karsten\\AppData\\Local\\Temp\\skillcorner_sample\\
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import SKILLCORNER_SPADL_COLUMNS
from silly_kicks.spadl.skillcorner import convert_to_actions

_SAMPLE_DIR = Path(os.environ.get(
    "SKILLCORNER_SAMPLE_DIR",
    r"C:\Users\Karsten\AppData\Local\Temp\skillcorner_sample",
))

_MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]


def _find_artifact(match_dir: Path, suffix: str) -> Path | None:
    """Find artifact by suffix, handling both bare and {id}-prefixed names."""
    candidates = list(match_dir.glob(f"*{suffix}"))
    return candidates[0] if candidates else None


def _has_data():
    return _SAMPLE_DIR.exists() and any(
        _find_artifact(_SAMPLE_DIR / m, "dynamic_events.csv") is not None
        for m in _MATCH_IDS
    )


pytestmark = pytest.mark.e2e


@pytest.fixture(params=_MATCH_IDS)
def match_data(request):
    match_dir = _SAMPLE_DIR / request.param
    events_path = _find_artifact(match_dir, "dynamic_events.csv")
    meta_path = _find_artifact(match_dir, "match.json")
    if events_path is None or meta_path is None:
        pytest.skip(f"Match {request.param} not available at {match_dir}")
    events = pd.read_csv(events_path, low_memory=False)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return events, meta, request.param


@pytest.mark.skipif(not _has_data(), reason="SkillCorner sample data not available")
class TestSkillcornerE2E:
    def test_no_crash(self, match_data):
        events, meta, mid = match_data
        actions, report = convert_to_actions(events, meta)
        assert len(actions) > 0

    def test_schema_matches(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        assert set(actions.columns) == set(SKILLCORNER_SPADL_COLUMNS.keys())

    def test_no_unrecognized_events(self, match_data):
        events, meta, mid = match_data
        _, report = convert_to_actions(events, meta)
        assert not report.has_unrecognized, f"Match {mid}: {report.unrecognized_counts}"

    def test_shots_in_attacking_half(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        shots = actions[actions["type_id"] == spadlconfig.actiontype_id["shot"]]
        if len(shots) > 0:
            assert (shots["start_x"] > 52.5).all(), f"Match {mid}: shots in defensive half"

    def test_clearances_in_defensive_half(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        clearances = actions[actions["type_id"] == spadlconfig.actiontype_id["clearance"]]
        if len(clearances) > 0:
            assert (clearances["start_x"] < 52.5).all(), f"Match {mid}: clearances in attacking half"

    def test_monotonic_time_per_period(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        for period in actions["period_id"].unique():
            times = actions[actions["period_id"] == period]["time_seconds"].values
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
                f"Match {mid}: non-monotonic in period {period}"
            )

    def test_has_interceptions(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        interceptions = actions[actions["type_id"] == spadlconfig.actiontype_id["interception"]]
        # Every match should have some interceptions (from start_type)
        assert len(interceptions) > 0, f"Match {mid}: no interceptions"

    def test_coordinates_in_range(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        for col in ["start_x", "end_x"]:
            valid = actions[col].dropna()
            assert (valid >= -2.0).all(), f"Match {mid}: {col} below -2"
            assert (valid <= 107.0).all(), f"Match {mid}: {col} above 107"
        for col in ["start_y", "end_y"]:
            valid = actions[col].dropna()
            assert (valid >= -2.0).all(), f"Match {mid}: {col} below -2"
            assert (valid <= 70.0).all(), f"Match {mid}: {col} above 70"
```

- [ ] **Step 2: Run e2e tests locally**

Run: `python -m pytest tests/spadl/test_skillcorner_e2e.py -v`
Expected: 80 PASSED (8 tests x 10 matches)

- [ ] **Step 3: Fix any failures surfaced by real data**

If any e2e tests fail, diagnose root cause and fix in the converter. Re-run until all 80 pass.

---

### Task 11: Final Review + Commit Prep

**Files:** All

- [ ] **Step 1: Run /final-review skill**

Invoke the `mad-scientist-skills:final-review` skill. This is mandatory before the single commit.

- [ ] **Step 2: Address any findings from final-review**

Fix documentation drift, stale references, missing test updates, consistency issues.

- [ ] **Step 3: Run full quality gate**

```bash
python -m ruff check silly_kicks/ tests/
python -m ruff format --check silly_kicks/ tests/
python -m pyright silly_kicks/spadl/skillcorner.py silly_kicks/spadl/_skillcorner_inference.py
python -m pytest tests/ -m "not e2e" --tb=short -q
```

All must pass with zero errors.

- [ ] **Step 4: Stage and commit (after user approval)**

```bash
git add silly_kicks/spadl/skillcorner.py \
       silly_kicks/spadl/_skillcorner_inference.py \
       silly_kicks/spadl/schema.py \
       silly_kicks/spadl/__init__.py \
       tests/datasets/skillcorner/ \
       tests/spadl/test_skillcorner.py \
       tests/spadl/test_skillcorner_inference.py \
       tests/spadl/test_skillcorner_e2e.py \
       tests/invariants/test_skillcorner_invariants.py \
       docs/superpowers/specs/2026-05-14-skillcorner-events-converter-design.md \
       docs/superpowers/plans/2026-05-14-skillcorner-events-converter.md
```

Wait for explicit user approval before committing.
