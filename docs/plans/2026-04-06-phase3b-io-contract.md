# Phase 3b: Input/Output Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish typed, documented, guaranteed I/O contracts for all converters — drop pandera, enforce output dtypes, add input validation, implement the "Nothing Left Behind" mapping registry with ConversionReport.

**Architecture:** Replace pandera `DataFrameModel` schemas with plain Python dict constants (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`). Add a `ConversionReport` frozen dataclass returned alongside the DataFrame from every converter. Each converter gets explicit `_MAPPED_EVENT_TYPES`/`_EXCLUDED_EVENT_TYPES` frozensets, per-provider `EXPECTED_INPUT_COLUMNS`, and a call to `_finalize_output()` to guarantee output columns and dtypes. The pandera dependency (and its `multimethod<2.0` transitive) is removed entirely.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new dependencies.

**Working directory:** `D:\Development\karstenskyt__silly-kicks\`

**Sources:** Design spec at `docs/specs/2026-04-06-phase3b-io-contract-design.md`

---

## File Structure

```
Modified:
  silly_kicks/spadl/schema.py           — pandera model → plain constants + ConversionReport
  silly_kicks/atomic/spadl/schema.py    — pandera model → plain constants
  silly_kicks/spadl/utils.py            — add _finalize_output, _validate_input_columns, validate_spadl
  silly_kicks/spadl/statsbomb.py        — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/opta.py             — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/wyscout.py          — input validation, mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/kloppy.py           — mapping registry, ConversionReport, tuple return
  silly_kicks/spadl/__init__.py         — update exports
  silly_kicks/atomic/spadl/base.py      — update for _finalize_output + remove pandera
  silly_kicks/atomic/spadl/utils.py     — add_names contract + remove pandera
  silly_kicks/atomic/spadl/__init__.py  — update exports
  silly_kicks/vaep/features.py          — remove pandera imports/types
  silly_kicks/vaep/labels.py            — remove pandera imports/types
  silly_kicks/vaep/formula.py           — remove pandera imports/types
  silly_kicks/atomic/vaep/features.py   — remove pandera imports/types
  silly_kicks/atomic/vaep/labels.py     — remove pandera imports/types
  silly_kicks/atomic/vaep/formula.py    — remove pandera imports/types
  silly_kicks/xthreat.py                — remove pandera imports/types
  pyproject.toml                        — remove pandera/multimethod deps, unpin numpy
  tests/conftest.py                     — remove pandera shim
  tests/spadl/test_opta.py              — update for tuple return
  tests/spadl/test_wyscout.py           — update for tuple return
  tests/spadl/test_statsbomb.py         — update for tuple return
  tests/atomic/test_atomic_spadl.py     — update for tuple return indirectly (convert_to_actions callers)

Created:
  tests/spadl/test_schema.py            — schema constants + ConversionReport tests
  tests/spadl/test_output_contract.py   — output guarantee + input validation + add_names tests
```

---

## Task 1: Schema Constants + ConversionReport

**Files:**
- Modify: `silly_kicks/spadl/schema.py` (full rewrite)
- Modify: `silly_kicks/atomic/spadl/schema.py` (full rewrite)
- Create: `tests/spadl/test_schema.py`

- [ ] **Step 1: Write the tests**

Create `tests/spadl/test_schema.py`:

```python
"""Tests for SPADL and Atomic-SPADL schema constants and ConversionReport."""

import pytest
from silly_kicks.spadl.schema import (
    SPADL_COLUMNS,
    SPADL_NAME_COLUMNS,
    SPADL_CONSTRAINTS,
    KLOPPY_SPADL_COLUMNS,
    ConversionReport,
)
from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS


def test_spadl_columns_count():
    assert len(SPADL_COLUMNS) == 14


def test_atomic_spadl_columns_count():
    assert len(ATOMIC_SPADL_COLUMNS) == 13


def test_spadl_columns_are_strings():
    for col, dtype in SPADL_COLUMNS.items():
        assert isinstance(col, str)
        assert isinstance(dtype, str)


def test_spadl_name_columns():
    assert set(SPADL_NAME_COLUMNS.keys()) == {"type_name", "result_name", "bodypart_name"}


def test_kloppy_overrides_id_columns():
    assert KLOPPY_SPADL_COLUMNS["game_id"] == "object"
    assert KLOPPY_SPADL_COLUMNS["team_id"] == "object"
    assert KLOPPY_SPADL_COLUMNS["player_id"] == "object"
    # Non-ID columns are unchanged
    assert KLOPPY_SPADL_COLUMNS["start_x"] == SPADL_COLUMNS["start_x"]


def test_spadl_constraints_cover_coordinate_columns():
    for col in ["start_x", "start_y", "end_x", "end_y"]:
        assert col in SPADL_CONSTRAINTS


def test_conversion_report_creation():
    report = ConversionReport(
        provider="StatsBomb",
        total_events=100,
        total_actions=80,
        mapped_counts={"Pass": 50, "Shot": 10, "Duel": 20},
        excluded_counts={"Pressure": 15, "Substitution": 5},
        unrecognized_counts={},
    )
    assert report.provider == "StatsBomb"
    assert report.total_events == 100
    assert report.total_actions == 80
    assert not report.has_unrecognized


def test_conversion_report_has_unrecognized():
    report = ConversionReport(
        provider="StatsBomb",
        total_events=10,
        total_actions=5,
        mapped_counts={"Pass": 5},
        excluded_counts={"Pressure": 3},
        unrecognized_counts={"NewEventType": 2},
    )
    assert report.has_unrecognized


def test_conversion_report_is_frozen():
    report = ConversionReport(
        provider="test",
        total_events=0,
        total_actions=0,
        mapped_counts={},
        excluded_counts={},
        unrecognized_counts={},
    )
    with pytest.raises(AttributeError):
        report.provider = "other"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_schema.py -v`
Expected: FAIL — imports fail because schema.py still has pandera classes

- [ ] **Step 3: Replace SPADL schema.py**

Rewrite `silly_kicks/spadl/schema.py` to:

```python
"""SPADL output schema — plain Python constants.

These constants define the guaranteed output contract of convert_to_actions().
They replace the pandera DataFrameModel that previously served this role.
"""

import dataclasses


SPADL_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "original_event_id": "object",
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

SPADL_NAME_COLUMNS: dict[str, str] = {
    "type_name": "object",
    "result_name": "object",
    "bodypart_name": "object",
}

SPADL_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id": (1, 5),
    "time_seconds": (0, float("inf")),
    "start_x": (0, 105.0),
    "start_y": (0, 68.0),
    "end_x": (0, 105.0),
    "end_y": (0, 68.0),
}

KLOPPY_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "game_id": "object",
    "team_id": "object",
    "player_id": "object",
}


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
        """Return True if any unrecognized event types were encountered."""
        return len(self.unrecognized_counts) > 0
```

- [ ] **Step 4: Replace Atomic-SPADL schema.py**

Rewrite `silly_kicks/atomic/spadl/schema.py` to:

```python
"""Atomic-SPADL output schema — plain Python constants."""

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

ATOMIC_SPADL_NAME_COLUMNS: dict[str, str] = {
    "type_name": "object",
    "bodypart_name": "object",
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_schema.py -v`
Expected: PASS

---

## Task 2: Shared Utilities

**Files:**
- Modify: `silly_kicks/spadl/utils.py`
- Create: `tests/spadl/test_output_contract.py`

- [ ] **Step 1: Write the tests**

Create `tests/spadl/test_output_contract.py`:

```python
"""Tests for output contract utilities, input validation, and add_names preservation."""

import warnings

import pandas as pd
import pytest
import silly_kicks.spadl.config as spadlcfg
from silly_kicks.spadl.schema import SPADL_COLUMNS
from silly_kicks.spadl.utils import (
    _finalize_output,
    _validate_input_columns,
    validate_spadl,
)


def _make_valid_spadl_df(n: int = 3) -> pd.DataFrame:
    """Create a minimal valid SPADL DataFrame for testing."""
    return pd.DataFrame({
        "game_id": [1] * n,
        "original_event_id": ["ev1", "ev2", "ev3"][:n],
        "action_id": list(range(n)),
        "period_id": [1] * n,
        "time_seconds": [0.0, 1.0, 2.0][:n],
        "team_id": [100] * n,
        "player_id": [200] * n,
        "start_x": [50.0] * n,
        "start_y": [34.0] * n,
        "end_x": [60.0] * n,
        "end_y": [34.0] * n,
        "type_id": [spadlcfg.actiontype_id["pass"]] * n,
        "result_id": [spadlcfg.result_id["success"]] * n,
        "bodypart_id": [spadlcfg.bodypart_id["foot"]] * n,
    })


class TestFinalizeOutput:
    def test_selects_declared_columns_only(self):
        df = _make_valid_spadl_df()
        df["extra_col"] = "should_be_dropped"
        result = _finalize_output(df)
        assert list(result.columns) == list(SPADL_COLUMNS.keys())
        assert "extra_col" not in result.columns

    def test_enforces_dtypes(self):
        df = _make_valid_spadl_df()
        # Intentionally make game_id a float (simulating NaN-in-int issue)
        df["game_id"] = df["game_id"].astype(float)
        result = _finalize_output(df)
        assert result["game_id"].dtype == "int64"

    def test_original_event_id_is_object(self):
        df = _make_valid_spadl_df()
        result = _finalize_output(df)
        assert result["original_event_id"].dtype == object


class TestValidateInputColumns:
    def test_missing_column_raises(self):
        df = pd.DataFrame({"game_id": [1], "event_id": [1]})
        required = {"game_id", "event_id", "period_id"}
        with pytest.raises(ValueError, match="period_id"):
            _validate_input_columns(df, required, provider="Test")

    def test_extra_columns_accepted(self):
        df = pd.DataFrame({"game_id": [1], "event_id": [1], "bonus": [True]})
        required = {"game_id", "event_id"}
        _validate_input_columns(df, required, provider="Test")  # should not raise

    def test_error_message_includes_provider(self):
        df = pd.DataFrame({"game_id": [1]})
        required = {"game_id", "missing_col"}
        with pytest.raises(ValueError, match="StatsBomb"):
            _validate_input_columns(df, required, provider="StatsBomb")


class TestValidateSpadl:
    def test_valid_dataframe_passes(self):
        df = _make_valid_spadl_df()
        result = validate_spadl(df)
        assert result is df  # returns same object for chaining

    def test_missing_column_raises(self):
        df = _make_valid_spadl_df().drop(columns=["type_id"])
        with pytest.raises(ValueError, match="type_id"):
            validate_spadl(df)

    def test_dtype_mismatch_warns(self):
        df = _make_valid_spadl_df()
        df["game_id"] = df["game_id"].astype(float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_spadl(df)
            dtype_warnings = [x for x in w if "dtype" in str(x.message).lower()]
            assert len(dtype_warnings) >= 1

    def test_extra_columns_accepted(self):
        df = _make_valid_spadl_df()
        df["custom_col"] = "hello"
        validate_spadl(df)  # should not raise


class TestAddNamesPreservesExtraColumns:
    def test_extra_columns_preserved(self):
        from silly_kicks.spadl.utils import add_names

        df = _make_valid_spadl_df()
        df["my_custom_col"] = [10, 20, 30]
        result = add_names(df)
        assert "my_custom_col" in result.columns
        assert list(result["my_custom_col"]) == [10, 20, 30]

    def test_name_columns_added(self):
        from silly_kicks.spadl.utils import add_names

        df = _make_valid_spadl_df()
        result = add_names(df)
        assert "type_name" in result.columns
        assert "result_name" in result.columns
        assert "bodypart_name" in result.columns

    def test_atomic_add_names_preserves_extra_columns(self):
        from silly_kicks.atomic.spadl.utils import add_names as atomic_add_names
        import silly_kicks.atomic.spadl.config as atomicconfig

        df = pd.DataFrame({
            "game_id": [1],
            "original_event_id": ["ev1"],
            "action_id": [0],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": [100],
            "player_id": [200],
            "x": [50.0],
            "y": [34.0],
            "dx": [10.0],
            "dy": [0.0],
            "type_id": [atomicconfig.actiontype_id["pass"]],
            "bodypart_id": [atomicconfig.bodypart_id["foot"]],
            "my_custom_col": [42],
        })
        result = atomic_add_names(df)
        assert "my_custom_col" in result.columns
        assert result["my_custom_col"].iloc[0] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_output_contract.py -v`
Expected: FAIL — `_finalize_output`, `_validate_input_columns`, `validate_spadl` do not exist yet

- [ ] **Step 3: Implement the shared utilities**

Replace `silly_kicks/spadl/utils.py` with:

```python
"""Utility functions for working with SPADL dataframes."""

import warnings

import pandas as pd

from . import config as spadlconfig
from .schema import SPADL_COLUMNS, SPADL_CONSTRAINTS


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name, result name and bodypart name to a SPADL dataframe.

    All columns not in the SPADL schema are preserved unchanged.

    Parameters
    ----------
    actions : pd.DataFrame
        A SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with 'type_name', 'result_name' and
        'bodypart_name' appended.
    """
    return (
        actions.drop(columns=["type_name", "result_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")
        .merge(spadlconfig.results_df(), how="left")
        .merge(spadlconfig.bodyparts_df(), how="left")
        .set_index(actions.index)
    )


def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Perform all actions in the same playing direction.

    This changes the start and end location of each action, such that all actions
    are performed as if the team that executes the action plays from left to
    right.

    Parameters
    ----------
    actions : pd.DataFrame
        The SPADL actions of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    pd.DataFrame
        All actions performed left to right.

    See Also
    --------
    silly_kicks.vaep.features.play_left_to_right : For transforming gamestates.
    """
    ltr_actions = actions.copy()
    away_idx = actions.team_id != home_team_id
    for col in ["start_x", "end_x"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values
    for col in ["start_y", "end_y"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_width - actions[away_idx][col].values
    return ltr_actions


def _finalize_output(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Project to declared columns and enforce dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        The raw converter output DataFrame.
    schema : dict[str, str], optional
        Column name → dtype mapping. Defaults to SPADL_COLUMNS.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the declared columns and dtypes.
    """
    if schema is None:
        schema = SPADL_COLUMNS
    result = df[list(schema.keys())].copy()
    for col, dtype in schema.items():
        result[col] = result[col].astype(dtype)
    return result


def _validate_input_columns(
    df: pd.DataFrame,
    expected: set[str],
    provider: str,
) -> None:
    """Validate that a DataFrame has all expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to validate.
    expected : set[str]
        Set of required column names.
    provider : str
        Provider name for error messages.

    Raises
    ------
    ValueError
        If any expected columns are missing.
    """
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{provider} convert_to_actions: missing required columns: "
            f"{sorted(missing)}. Got: {sorted(df.columns)}"
        )


def validate_spadl(
    df: pd.DataFrame,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms to the SPADL schema.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema : dict[str, str], optional
        Column name → dtype mapping. Defaults to SPADL_COLUMNS.

    Returns
    -------
    pd.DataFrame
        The input DataFrame unchanged (for chaining).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if schema is None:
        schema = SPADL_COLUMNS
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing SPADL columns: {sorted(missing)}")
    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            warnings.warn(
                f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'",
                stacklevel=2,
            )
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_output_contract.py -v`
Expected: PASS (the add_names tests may fail because utils.py still imports old schema — see next step)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: Some failures from other files still importing pandera — this is expected at this point

---

## Task 3: Update Atomic-SPADL Utils + Atomic Base

**Files:**
- Modify: `silly_kicks/atomic/spadl/utils.py`
- Modify: `silly_kicks/atomic/spadl/base.py`

- [ ] **Step 1: Replace atomic utils.py**

Replace `silly_kicks/atomic/spadl/utils.py` with:

```python
"""Utility functions for working with Atomic-SPADL dataframes."""

import pandas as pd

from . import config as spadlconfig


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the type name and bodypart name to an Atomic-SPADL dataframe.

    All columns not in the Atomic-SPADL schema are preserved unchanged.

    Parameters
    ----------
    actions : pd.DataFrame
        An Atomic-SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The original dataframe with 'type_name' and 'bodypart_name' appended.
    """
    return (
        actions.drop(columns=["type_name", "bodypart_name"], errors="ignore")
        .merge(spadlconfig.actiontypes_df(), how="left")
        .merge(spadlconfig.bodyparts_df(), how="left")
        .set_index(actions.index)
    )


def play_left_to_right(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Perform all action in the same playing direction.

    This changes the location of each action, such that all actions
    are performed as if the team that executes the action plays from left to
    right.

    Parameters
    ----------
    actions : pd.DataFrame
        The SPADL actins of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    list(pd.DataFrame)
        All actions performed left to right.

    See Also
    --------
    silly_kicks.atomic.vaep.features.play_left_to_right : For transforming gamestates.
    """
    ltr_actions = actions.copy()
    away_idx = actions.team_id != home_team_id
    ltr_actions.loc[away_idx, "x"] = spadlconfig.field_length - actions[away_idx]["x"].values
    ltr_actions.loc[away_idx, "y"] = spadlconfig.field_width - actions[away_idx]["y"].values
    ltr_actions.loc[away_idx, "dx"] = -actions[away_idx]["dx"].values
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values
    return ltr_actions
```

- [ ] **Step 2: Update atomic base.py**

Replace `silly_kicks/atomic/spadl/base.py` imports and `convert_to_atomic` to remove pandera and add `_finalize_output`:

Replace the imports (lines 1-13) with:

```python
"""Implements a converter for regular SPADL actions to atomic actions."""

import pandas as pd

import silly_kicks.spadl.config as _spadl
from silly_kicks.spadl.base import _add_dribbles
from silly_kicks.spadl.utils import _finalize_output

from . import config as _atomicspadl
from .schema import ATOMIC_SPADL_COLUMNS
```

Replace the `convert_to_atomic` function (lines 16-36) with:

```python
def convert_to_atomic(actions: pd.DataFrame) -> pd.DataFrame:
    """Convert regular SPADL actions to atomic actions.

    Parameters
    ----------
    actions : pd.DataFrame
        A SPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The Atomic-SPADL dataframe.
    """
    atomic_actions = actions.copy()
    atomic_actions = _extra_from_passes(atomic_actions)
    atomic_actions = _add_dribbles(atomic_actions)
    atomic_actions = _extra_from_shots(atomic_actions)
    atomic_actions = _extra_from_fouls(atomic_actions)
    atomic_actions = _convert_columns(atomic_actions)
    atomic_actions = _simplify(atomic_actions)
    return _finalize_output(atomic_actions, ATOMIC_SPADL_COLUMNS)
```

Also remove the `from typing import cast` import line (no longer needed).

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/atomic/ -m "not e2e" -v --tb=short`
Expected: PASS (atomic tests should still work)

---

## Task 4: StatsBomb Converter — Registry + Tuple Return

**Files:**
- Modify: `silly_kicks/spadl/statsbomb.py`
- Modify: `tests/spadl/test_statsbomb.py`

- [ ] **Step 1: Write the test**

Add to `tests/spadl/test_statsbomb.py`:

```python
import pandas as pd
from silly_kicks.spadl.schema import ConversionReport


def _make_statsbomb_events() -> pd.DataFrame:
    """Minimal StatsBomb event DataFrame for testing."""
    return pd.DataFrame([
        {
            "game_id": 1,
            "event_id": "abc-123",
            "period_id": 1,
            "timestamp": "00:00:01.000",
            "team_id": 100,
            "player_id": 200,
            "type_name": "Pass",
            "location": [60.0, 40.0],
            "extra": {
                "pass": {
                    "end_location": [70.0, 40.0],
                    "outcome": {"name": "Complete"},
                    "height": {"name": "Ground Pass"},
                }
            },
        },
        {
            "game_id": 1,
            "event_id": "abc-456",
            "period_id": 1,
            "timestamp": "00:00:05.000",
            "team_id": 100,
            "player_id": 201,
            "type_name": "Pressure",
            "location": [50.0, 30.0],
            "extra": {},
        },
    ])


def test_statsbomb_returns_tuple():
    """Phase 3b: convert_to_actions returns (DataFrame, ConversionReport)."""
    events = _make_statsbomb_events()
    result = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert isinstance(result, tuple)
    assert len(result) == 2
    actions, report = result
    assert isinstance(actions, pd.DataFrame)
    assert isinstance(report, ConversionReport)


def test_statsbomb_conversion_report():
    events = _make_statsbomb_events()
    actions, report = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert report.provider == "StatsBomb"
    assert report.total_events == 2
    assert "Pass" in report.mapped_counts
    assert "Pressure" in report.excluded_counts
    assert not report.has_unrecognized


def test_statsbomb_unrecognized_event_warning():
    events = _make_statsbomb_events()
    # Add a fabricated unknown event type
    events = pd.concat([events, pd.DataFrame([{
        "game_id": 1,
        "event_id": "abc-789",
        "period_id": 1,
        "timestamp": "00:00:10.000",
        "team_id": 100,
        "player_id": 202,
        "type_name": "FutureEventType",
        "location": [50.0, 30.0],
        "extra": {},
    }])], ignore_index=True)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actions, report = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert report.has_unrecognized
    assert "FutureEventType" in report.unrecognized_counts
    unrecognized_warnings = [x for x in w if "unrecognized" in str(x.message).lower()]
    assert len(unrecognized_warnings) >= 1


def test_statsbomb_output_columns():
    from silly_kicks.spadl.schema import SPADL_COLUMNS

    events = _make_statsbomb_events()
    actions, _ = statsbomb.convert_to_actions(events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())
    for col, dtype in SPADL_COLUMNS.items():
        assert str(actions[col].dtype) == dtype, f"{col}: expected {dtype}, got {actions[col].dtype}"


def test_statsbomb_input_validation():
    df = pd.DataFrame({"game_id": [1]})  # missing most columns
    with pytest.raises(ValueError, match="StatsBomb"):
        statsbomb.convert_to_actions(df, home_team_id=100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/spadl/test_statsbomb.py -v`
Expected: FAIL — convert_to_actions still returns a plain DataFrame

- [ ] **Step 3: Update statsbomb.py**

Replace imports (lines 1-13) with:

```python
"""StatsBomb event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns
```

Add after the `_SB_FIELD_WIDTH` constant (line 16):

```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "timestamp",
    "team_id", "player_id", "type_name", "location", "extra",
}

_MAPPED_EVENT_TYPES: frozenset[str] = frozenset({
    "Pass", "Dribble", "Carry", "Foul Committed", "Duel",
    "Interception", "Shot", "Own Goal Against", "Goal Keeper",
    "Clearance", "Miscontrol",
})

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset({
    "Pressure", "Ball Receipt*", "Block", "50/50",
    "Substitution", "Starting XI", "Tactical Shift",
    "Referee Ball-Drop", "Half Start", "Half End",
    "Injury Stoppage", "Player On", "Player Off", "Error",
    "Shield", "Camera On", "Camera off",
    "Bad Behaviour", "Ball Recovery",
})
```

Replace the `convert_to_actions` function signature and return (lines 19-129) — change the return type and add input validation at the top, conversion report building at the bottom. The function body stays the same except:

1. Add at line 53 (after the docstring, before `actions = pd.DataFrame()`):
```python
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="StatsBomb")

    # Count source events by type for the ConversionReport
    _event_type_counts = Counter(events["type_name"])
```

2. Change the return type annotation to `-> tuple[pd.DataFrame, ConversionReport]:`

3. Replace the return statement (line 129) with:
```python
    actions = _finalize_output(actions)

    # Build ConversionReport
    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for etype, count in _event_type_counts.items():
        if etype in _MAPPED_EVENT_TYPES:
            mapped_counts[etype] = count
        elif etype in _EXCLUDED_EVENT_TYPES:
            excluded_counts[etype] = count
        else:
            unrecognized_counts[etype] = count
    if unrecognized_counts:
        warnings.warn(
            f"StatsBomb: {sum(unrecognized_counts.values())} unrecognized event types "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="StatsBomb",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report
```

Also ensure `original_event_id` is cast to str before `_finalize_output`:
At line 88, change:
```python
actions["original_event_id"] = events.event_id
```
to:
```python
actions["original_event_id"] = events.event_id.astype(str)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/spadl/test_statsbomb.py -v`
Expected: PASS

---

## Task 5: Opta Converter — Registry + Tuple Return

**Files:**
- Modify: `silly_kicks/spadl/opta.py`
- Modify: `tests/spadl/test_opta.py`

- [ ] **Step 1: Write the test**

Add to `tests/spadl/test_opta.py` (before the e2e section):

```python
from silly_kicks.spadl.schema import ConversionReport, SPADL_COLUMNS


def test_opta_returns_tuple():
    """Phase 3b: convert_to_actions returns (DataFrame, ConversionReport)."""
    event = pd.DataFrame([{
        "game_id": 1, "event_id": 100, "type_id": 1, "period_id": 1,
        "minute": 1, "second": 0, "team_id": 10, "player_id": 20,
        "outcome": False, "start_x": 50.0, "start_y": 50.0,
        "end_x": 60.0, "end_y": 50.0,
        "qualifiers": {124: True}, "type_name": "pass",
    }])
    result = opta.convert_to_actions(event, home_team_id=10)
    assert isinstance(result, tuple)
    actions, report = result
    assert isinstance(actions, pd.DataFrame)
    assert isinstance(report, ConversionReport)
    assert report.provider == "Opta"


def test_opta_output_columns():
    event = pd.DataFrame([{
        "game_id": 1, "event_id": 100, "type_id": 1, "period_id": 1,
        "minute": 1, "second": 0, "team_id": 10, "player_id": 20,
        "outcome": True, "start_x": 50.0, "start_y": 50.0,
        "end_x": 60.0, "end_y": 50.0,
        "qualifiers": {124: True}, "type_name": "pass",
    }])
    actions, _ = opta.convert_to_actions(event, home_team_id=10)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())
    for col, dtype in SPADL_COLUMNS.items():
        assert str(actions[col].dtype) == dtype, f"{col}: expected {dtype}, got {actions[col].dtype}"


def test_opta_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="Opta"):
        opta.convert_to_actions(df, home_team_id=10)
```

- [ ] **Step 2: Update existing tests for tuple return**

In `tests/spadl/test_opta.py`, update the existing tests. Change line 43:
```python
    action = opta.convert_to_actions(event, 0).iloc[0]
```
to:
```python
    actions, _ = opta.convert_to_actions(event, 0)
    action = actions.iloc[0]
```

Change line 72:
```python
    action = opta.convert_to_actions(event, 0).iloc[0]
```
to:
```python
    actions, _ = opta.convert_to_actions(event, 0)
    action = actions.iloc[0]
```

- [ ] **Step 3: Update opta.py**

Replace imports (lines 1-15) with:

```python
"""Opta event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Any

import pandas as pd

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
    min_dribble_length,
)
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns
```

Add after imports:

```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "minute", "second",
    "team_id", "player_id", "type_name", "outcome",
    "start_x", "start_y", "end_x", "end_y", "qualifiers",
}

_MAPPED_EVENT_TYPES: frozenset[str] = frozenset({
    "pass", "offside pass", "take on", "foul", "tackle",
    "interception", "blocked pass", "miss", "post",
    "attempt saved", "goal", "save", "claim", "punch",
    "keeper pick-up", "clearance", "card", "ball touch",
    "ball recovery",
})

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset({
    "end", "start", "formation change", "resume", "deleted event",
    "shield ball opp", "offside provoked", "player off",
    "player on", "player retired", "chance missed",
    "attendance", "referee stop", "referee drop ball",
    "50/50", "cross not claimed", "blocked pass",
    "goalkeeper position",
})
```

Change signature of `convert_to_actions` (line 18) to:
```python
def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> tuple[pd.DataFrame, ConversionReport]:
```

Add input validation at the top of the function body:
```python
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Opta")
    _event_type_counts = Counter(events["type_name"])
```

Replace the return at line 79 with:
```python
    actions = _finalize_output(actions)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for etype, count in _event_type_counts.items():
        if etype in _MAPPED_EVENT_TYPES:
            mapped_counts[etype] = count
        elif etype in _EXCLUDED_EVENT_TYPES:
            excluded_counts[etype] = count
        else:
            unrecognized_counts[etype] = count
    if unrecognized_counts:
        warnings.warn(
            f"Opta: {sum(unrecognized_counts.values())} unrecognized event types "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="Opta",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report
```

Remove the `from typing import cast` import and the `cast(DataFrame[SPADLSchema], ...)` wrapper.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/spadl/test_opta.py -v`
Expected: PASS

---

## Task 6: Wyscout Converter — Registry + Tuple Return

**Files:**
- Modify: `silly_kicks/spadl/wyscout.py`
- Modify: `tests/spadl/test_wyscout.py`

- [ ] **Step 1: Update existing tests for tuple return**

In `tests/spadl/test_wyscout.py`, update every call to `wy.convert_to_actions(...)`:

Line 30: `actions = wy.convert_to_actions(event, 1610)` → `actions, _ = wy.convert_to_actions(event, 1610)`

Line 93: `actions = wy.convert_to_actions(event, 1639)` → `actions, _ = wy.convert_to_actions(event, 1639)`

Line 136: `actions = wy.convert_to_actions(events, 3158)` → `actions, _ = wy.convert_to_actions(events, 3158)`

Line 191: `actions = wy.convert_to_actions(events, 3157)` → `actions, _ = wy.convert_to_actions(events, 3157)`

Add a test for the tuple return and input validation:

```python
from silly_kicks.spadl.schema import ConversionReport, SPADL_COLUMNS


def test_wyscout_returns_tuple():
    """Phase 3b: convert_to_actions returns (DataFrame, ConversionReport)."""
    event = pd.DataFrame([{
        "type_id": 8, "subtype_name": "Simple pass", "subtype_id": 85,
        "tags": [{"id": 1801}],
        "player_id": 1, "positions": [{"y": 50, "x": 50}, {"y": 60, "x": 60}],
        "game_id": 1, "type_name": "Pass", "team_id": 100,
        "period_id": 1, "milliseconds": 1000.0, "event_id": 1,
    }])
    result = wy.convert_to_actions(event, 100)
    assert isinstance(result, tuple)
    actions, report = result
    assert isinstance(report, ConversionReport)
    assert report.provider == "Wyscout"


def test_wyscout_output_columns():
    event = pd.DataFrame([{
        "type_id": 8, "subtype_name": "Simple pass", "subtype_id": 85,
        "tags": [{"id": 1801}],
        "player_id": 1, "positions": [{"y": 50, "x": 50}, {"y": 60, "x": 60}],
        "game_id": 1, "type_name": "Pass", "team_id": 100,
        "period_id": 1, "milliseconds": 1000.0, "event_id": 1,
    }])
    actions, _ = wy.convert_to_actions(event, 100)
    assert list(actions.columns) == list(SPADL_COLUMNS.keys())


def test_wyscout_input_validation():
    df = pd.DataFrame({"game_id": [1]})
    with pytest.raises(ValueError, match="Wyscout"):
        wy.convert_to_actions(df, home_team_id=100)
```

- [ ] **Step 2: Update wyscout.py**

Replace imports (lines 1-15) with:

```python
"""Wyscout event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Any, Optional

import pandas as pd

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_direction_of_play,
    min_dribble_length,
)
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns
```

Add after the `_WS_TAG_OWN_GOAL` constant:

```python
EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "milliseconds",
    "team_id", "player_id", "type_id", "subtype_id",
    "positions", "tags",
}

# Wyscout type_ids that can produce SPADL actions
_MAPPED_WS_TYPE_IDS: frozenset[int] = frozenset({
    _WS_TYPE_TAKE_ON,        # 0 — synthetic take-on/tackle
    _WS_TYPE_DUEL,           # 1
    _WS_TYPE_FOUL,           # 2
    7,                        # Others on the ball (can produce bad_touch)
    _WS_TYPE_PASS,           # 8
    _WS_TYPE_GK,             # 9
    _WS_TYPE_SHOT,           # 10
})

# Wyscout type_ids intentionally excluded (meta-events)
_EXCLUDED_WS_TYPE_IDS: frozenset[int] = frozenset({
    3,   # Free kick (captured via subtype on pass/shot events)
    4,   # Goal kick (captured via subtype)
    5,   # Interruption (half-time, end of game, etc.)
    _WS_TYPE_OFFSIDE,  # 6 — captured as pass result
})
```

Change signature of `convert_to_actions` to return a tuple:
```python
def convert_to_actions(events: pd.DataFrame, home_team_id: int) -> tuple[pd.DataFrame, ConversionReport]:
```

Add input validation at top of function body:
```python
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Wyscout")
    _event_type_counts = Counter(events["type_id"])
```

Replace the return (line 98, `return cast(DataFrame[SPADLSchema], actions)`) with:
```python
    actions = _finalize_output(actions)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for ws_type_id, count in _event_type_counts.items():
        label = str(ws_type_id)
        if ws_type_id in _MAPPED_WS_TYPE_IDS:
            mapped_counts[label] = count
        elif ws_type_id in _EXCLUDED_WS_TYPE_IDS:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Wyscout: {sum(unrecognized_counts.values())} unrecognized event type_ids "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="Wyscout",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report
```

Remove `from typing import cast` and `cast(DataFrame[SPADLSchema], ...)`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/spadl/test_wyscout.py -v`
Expected: PASS

---

## Task 7: Kloppy Converter — Registry + Tuple Return

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py`

- [ ] **Step 1: Update kloppy.py**

Replace imports (lines 1-48) with:

```python
"""Kloppy EventDataset to SPADL converter."""

import warnings
from collections import Counter
from typing import Optional, Union

import kloppy
import pandas as pd
from kloppy.domain import (
    BodyPart,
    CardType,
    CarryEvent,
    ClearanceEvent,
    CoordinateSystem,
    Dimension,
    DuelEvent,
    DuelResult,
    DuelType,
    Event,
    EventDataset,
    EventType,
    FoulCommittedEvent,
    GoalkeeperActionType,
    GoalkeeperEvent,
    InterceptionResult,
    MetricPitchDimensions,
    MiscontrolEvent,
    Orientation,
    Origin,
    PassEvent,
    PassResult,
    PassType,
    PitchDimensions,
    Provider,
    Qualifier,
    RecoveryEvent,
    SetPieceType,
    ShotEvent,
    ShotResult,
    TakeOnEvent,
    TakeOnResult,
    VerticalOrientation,
)
from packaging import version

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances
from .schema import KLOPPY_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output
```

Add after `_SUPPORTED_PROVIDERS`:

```python
_MAPPED_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.PASS,
    EventType.SHOT,
    EventType.TAKE_ON,
    EventType.CARRY,
    EventType.FOUL_COMMITTED,
    EventType.DUEL,
    EventType.CLEARANCE,
    EventType.MISCONTROL,
    EventType.GOALKEEPER,
    EventType.INTERCEPTION,
})

_EXCLUDED_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.GENERIC,
    EventType.RECOVERY,
    EventType.SUBSTITUTION,
    EventType.CARD,
    EventType.PLAYER_ON,
    EventType.PLAYER_OFF,
    EventType.BALL_OUT,
    EventType.FORMATION_CHANGE,
})
```

Change signature to:
```python
def convert_to_actions(
    dataset: EventDataset, game_id: Optional[Union[str, int]] = None
) -> tuple[pd.DataFrame, ConversionReport]:
```

Add event counting in the event loop:
```python
    _event_type_counts: Counter[EventType] = Counter()
    actions = []
    for event in new_dataset.events:
        _event_type_counts[event.event_type] += 1
        action = dict(...)
        actions.append(action)
```

Replace the return (line 129, `return cast(DataFrame[SPADLSchema], df_actions)`) with:
```python
    df_actions = _finalize_output(df_actions, KLOPPY_SPADL_COLUMNS)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for etype, count in _event_type_counts.items():
        label = etype.value if hasattr(etype, "value") else str(etype)
        if etype in _MAPPED_EVENT_TYPES:
            mapped_counts[label] = count
        elif etype in _EXCLUDED_EVENT_TYPES:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Kloppy: {sum(unrecognized_counts.values())} unrecognized event types "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="Kloppy",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(df_actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return df_actions, report
```

Remove `from typing import cast` and `cast(DataFrame[SPADLSchema], ...)`.

- [ ] **Step 2: Run kloppy test**

Run: `python -m pytest tests/spadl/test_kloppy.py -v`
Expected: SKIP (kloppy tests are all e2e)

---

## Task 8: Pandera Removal Sweep

**Files:**
- Modify: `silly_kicks/spadl/__init__.py`
- Modify: `silly_kicks/atomic/spadl/__init__.py`
- Modify: `silly_kicks/vaep/features.py`
- Modify: `silly_kicks/vaep/labels.py`
- Modify: `silly_kicks/vaep/formula.py`
- Modify: `silly_kicks/atomic/vaep/features.py`
- Modify: `silly_kicks/atomic/vaep/labels.py`
- Modify: `silly_kicks/atomic/vaep/formula.py`
- Modify: `silly_kicks/xthreat.py`
- Modify: `tests/conftest.py`
- Modify: `tests/vaep/test_features.py`
- Modify: `tests/vaep/test_labels.py`
- Modify: `tests/atomic/test_atomic_features.py`
- Modify: `tests/atomic/test_atomic_labels.py`
- Modify: `tests/test_xthreat.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update spadl/__init__.py**

Replace `silly_kicks/spadl/__init__.py` with:

```python
"""Implementation of the SPADL language."""

__all__ = [
    "opta",
    "statsbomb",
    "wyscout",
    "kloppy",
    "config",
    "SPADL_COLUMNS",
    "ConversionReport",
    "bodyparts_df",
    "actiontypes_df",
    "results_df",
    "add_names",
    "validate_spadl",
    "play_left_to_right",
]

from . import config, opta, statsbomb, wyscout
from .config import actiontypes_df, bodyparts_df, results_df
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import add_names, play_left_to_right, validate_spadl

try:
    from . import kloppy
except ImportError:
    pass
```

- [ ] **Step 2: Update atomic/spadl/__init__.py**

Replace `silly_kicks/atomic/spadl/__init__.py` with:

```python
"""Implementation of the Atomic-SPADL language."""

__all__ = [
    "convert_to_atomic",
    "ATOMIC_SPADL_COLUMNS",
    "bodyparts_df",
    "actiontypes_df",
    "add_names",
    "play_left_to_right",
]

from .base import convert_to_atomic
from .config import actiontypes_df, bodyparts_df
from .schema import ATOMIC_SPADL_COLUMNS
from .utils import add_names, play_left_to_right
```

- [ ] **Step 3: Remove pandera from vaep/features.py**

In `silly_kicks/vaep/features.py`, replace lines 8-17:
```python
from pandera.typing import DataFrame

import silly_kicks.spadl.config as spadlcfg
from silly_kicks.spadl.schema import SPADLSchema

SPADLActions = DataFrame[SPADLSchema]
Actions = DataFrame[SPADLSchema]
GameStates = list[Actions]
Features = DataFrame[Any]
FeatureTransfomer = Callable[[GameStates], Features]
```
with:
```python
import pandas as pd

import silly_kicks.spadl.config as spadlcfg

Actions = pd.DataFrame
GameStates = list[pd.DataFrame]
Features = pd.DataFrame
FeatureTransfomer = Callable[[GameStates], Features]
```

- [ ] **Step 4: Remove pandera from vaep/labels.py**

In `silly_kicks/vaep/labels.py`, replace lines 3-7:
```python
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import silly_kicks.spadl.config as spadl
from silly_kicks.spadl.schema import SPADLSchema
```
with:
```python
import pandas as pd

import silly_kicks.spadl.config as spadl
```

Change all `actions: DataFrame[SPADLSchema]` parameter types to `actions: pd.DataFrame`.

- [ ] **Step 5: Remove pandera from vaep/formula.py**

In `silly_kicks/vaep/formula.py`, replace lines 3-6:
```python
import pandas as pd  # type: ignore
from pandera.typing import DataFrame, Series

from silly_kicks.spadl.schema import SPADLSchema
```
with:
```python
import pandas as pd
```

Change all `DataFrame[SPADLSchema]` to `pd.DataFrame` and `Series[...]` to `pd.Series`.

- [ ] **Step 6: Remove pandera from atomic/vaep/features.py**

In `silly_kicks/atomic/vaep/features.py`, replace lines 7-11:
```python
from pandera.typing import DataFrame

import silly_kicks.atomic.spadl.config as atomicspadl
from silly_kicks.atomic.spadl import AtomicSPADLSchema
from silly_kicks.spadl import SPADLSchema
```
with:
```python
import pandas as pd

import silly_kicks.atomic.spadl.config as atomicspadl
```

Change all `DataFrame[AtomicSPADLSchema]` and `DataFrame[SPADLSchema]` to `pd.DataFrame`.

- [ ] **Step 7: Remove pandera from atomic/vaep/labels.py**

In `silly_kicks/atomic/vaep/labels.py`, replace lines 3-7:
```python
import pandas as pd
from pandera.typing import DataFrame

import silly_kicks.atomic.spadl.config as atomicspadl
from silly_kicks.atomic.spadl import AtomicSPADLSchema
```
with:
```python
import pandas as pd

import silly_kicks.atomic.spadl.config as atomicspadl
```

Change all `DataFrame[AtomicSPADLSchema]` to `pd.DataFrame`.

- [ ] **Step 8: Remove pandera from atomic/vaep/formula.py**

In `silly_kicks/atomic/vaep/formula.py`, replace lines 3-6:
```python
import pandas as pd
from pandera.typing import DataFrame, Series

from silly_kicks.atomic.spadl import AtomicSPADLSchema
```
with:
```python
import pandas as pd
```

Change all `DataFrame[AtomicSPADLSchema]` to `pd.DataFrame` and `Series[...]` to `pd.Series`.

- [ ] **Step 9: Remove pandera from xthreat.py**

In `silly_kicks/xthreat.py`, replace lines 7-12:
```python
import pandas as pd
from pandera.typing import DataFrame, Series
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.schema import SPADLSchema
```
with:
```python
import pandas as pd
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
```

Change all `DataFrame[SPADLSchema]` to `pd.DataFrame` and `Series[float]`/`Series[int]` to `pd.Series`.

- [ ] **Step 10: Update conftest.py**

Replace `tests/conftest.py` with:

```python
"""Configuration for pytest."""

import os
from collections.abc import Iterator

import pandas as pd
import pytest
from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


@pytest.fixture(scope="session")
def sb_worldcup_data() -> Iterator[pd.HDFStore]:
    hdf_file = os.path.join(
        os.path.dirname(__file__), "datasets", "statsbomb", "spadl-WorldCup-2018.h5"
    )
    store = pd.HDFStore(hdf_file, mode="r")
    yield store
    store.close()


@pytest.fixture(scope="session")
def spadl_actions() -> pd.DataFrame:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "spadl.json")
    return pd.read_json(json_file, orient="records")


@pytest.fixture(scope="session")
def atomic_spadl_actions() -> pd.DataFrame:
    json_file = os.path.join(os.path.dirname(__file__), "datasets", "spadl", "atomic_spadl.json")
    return pd.read_json(json_file, orient="records")
```

- [ ] **Step 11: Remove pandera from test files**

In `tests/vaep/test_features.py`, replace:
```python
from pandera.typing import DataFrame
```
with removal of that line, and change all `DataFrame[SPADLSchema]` types to `pd.DataFrame`.

In `tests/vaep/test_labels.py`, replace:
```python
from pandera.typing import DataFrame
```
with removal of that line, and change all `DataFrame[SPADLSchema]` types to `pd.DataFrame`.

In `tests/atomic/test_atomic_features.py`, replace:
```python
from pandera.typing import DataFrame
```
with removal of that line, and change all `DataFrame[...]` types to `pd.DataFrame`.

In `tests/atomic/test_atomic_labels.py`, replace:
```python
from pandera.typing import DataFrame
```
with removal of that line, and change types.

In `tests/test_xthreat.py`, replace:
```python
from pandera.typing import DataFrame, Series
```
with removal of that line, and change types.

In `tests/spadl/test_opta.py`, remove line 4:
```python
from silly_kicks.spadl import SPADLSchema
```

- [ ] **Step 12: Update pyproject.toml**

Remove pandera and multimethod from dependencies. Unpin numpy upper bound:

Change:
```toml
dependencies = [
    "pandas>=2.1.1",
    "numpy>=1.26.0,<2.0",  # pandera 0.17-0.19 uses np.string_ removed in numpy 2.0
    "scikit-learn>=1.3.1",
    # DEPENDENCY CHAIN: pandera 0.17.x uses pa.SchemaModel (renamed to DataFrameModel in 0.20).
    # pandera 0.17.2 -> multimethod <2.0 (multimethod 2.0 removed the 'overload' API).
    # TODO: evaluate migrating to pandera >=0.20 or dropping pandera entirely.
    "pandera>=0.17.2,<0.20",
    "multimethod<2.0",
]
```
to:
```toml
dependencies = [
    "pandas>=2.1.1",
    "numpy>=1.26.0",
    "scikit-learn>=1.3.1",
]
```

Remove the pandera filterwarnings in `[tool.pytest.ini_options]`:
```toml
filterwarnings = [
    "ignore::DeprecationWarning:pandera",
]
```
→ remove the entire `filterwarnings` section.

- [ ] **Step 13: Reinstall**

Run: `pip install -e ".[test]" && pip uninstall -y pandera multimethod`

- [ ] **Step 14: Run full test suite**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

---

## Task 9: Pandera Removal Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Verify no pandera imports remain**

Run: `grep -r "import pandera\|from pandera" silly_kicks/ tests/`
Expected: zero matches

- [ ] **Step 2: Verify pandera is not importable**

Run: `python -c "import pandera" 2>&1`
Expected: `ModuleNotFoundError` (confirming pandera is not installed)

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass

- [ ] **Step 4: Verify output dtype contract with a quick spot-check**

Run:
```python
python -c "
import pandas as pd
from silly_kicks.spadl.schema import SPADL_COLUMNS
from silly_kicks.spadl import opta

event = pd.DataFrame([{
    'game_id': 1, 'event_id': 100, 'type_id': 1, 'period_id': 1,
    'minute': 1, 'second': 0, 'team_id': 10, 'player_id': 20,
    'outcome': True, 'start_x': 50.0, 'start_y': 50.0,
    'end_x': 60.0, 'end_y': 50.0,
    'qualifiers': {124: True}, 'type_name': 'pass',
}])
actions, report = opta.convert_to_actions(event, home_team_id=10)
print('Columns match:', list(actions.columns) == list(SPADL_COLUMNS.keys()))
print('Report:', report)
for col, dtype in SPADL_COLUMNS.items():
    assert str(actions[col].dtype) == dtype, f'{col}: {actions[col].dtype} != {dtype}'
print('All dtypes match!')
"
```
Expected: `Columns match: True`, `All dtypes match!`

---

## Task 10: Update DEFERRED.md

**Files:**
- Modify: `docs/DEFERRED.md`

- [ ] **Step 1: Update DEFERRED.md**

Mark the following items as resolved by Phase 3b:
- A14 (pandera pervasive): RESOLVED — pandera removed entirely
- A18 (schema validators frozen): RESOLVED — resolves naturally with A14

Add a note at the top of the file:
```
## Phase 3b: I/O Contract (2026-04-06)
- A14 (pandera dependency): RESOLVED — replaced with plain schema constants
- A18 (frozen schema validators): RESOLVED — no longer applicable
- numpy upper bound removed (was pinned due to pandera)
```

- [ ] **Step 2: Run full test suite one final time**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Expected: all tests pass — Phase 3b complete
