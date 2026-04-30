# PFF FC events-data converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `silly_kicks.spadl.pff` — a first-class native PFF FC / Gradient Sports events → SPADL converter on par with sportec/metrica/wyscout. Hexagonal pure-function contract (events DataFrame in, SPADL DataFrame + ConversionReport out, zero I/O). Synthetic-only test fixtures. Atomic-SPADL + VAEP composability. Tracking deferred to a TODO.md entry. Ships as silly-kicks 2.6.0.

**Architecture:** New module `silly_kicks/spadl/pff.py` mirrors `sportec.py` shape: `EXPECTED_INPUT_COLUMNS` (~40 PFF-shaped flat columns), `np.select`-based dispatch over `(game_event_type, possession_event_type, set_piece_type)`, `PFF_SPADL_COLUMNS` schema (SPADL_COLUMNS + 4 nullable Int64 tackle-passthrough columns), per-period direction-of-play normalization (PFF coordinates are perspective-real, unlike all current providers), explicit `BC → dribble` mapping (skips `_add_dribbles` synthesis), foul row synthesis from the `fouls[]` array. Caller is responsible for parsing raw PFF JSON into the DataFrame; converter is pure.

**Tech Stack:** Python 3.10+, pandas (with `Int64` nullable extension dtype on tackle columns — first introduction in silly-kicks), numpy, pytest, ruff 0.15.7, pyright 1.1.395, pandas-stubs 2.3.3.260113.

**Spec:** `docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`

**Commit policy:** Per project CLAUDE.md memory — *literally ONE commit per branch; no WIP commits + squash; explicit user approval before that one commit*. This plan therefore does NOT have intermediate `git commit` steps. All changes accumulate in the working tree; a single commit at the very end (Task 33) gathers everything after user approval.

**Test count target:** baseline + ~50 net delta (≈30 unit/contract + ≈10 e2e on the synthetic match + ≈10 dispatch / cross-pipeline parity).

---

## Phase 1 — Setup

### Task 1: Create feature branch

**Files:**
- (none — git operation only)

- [ ] **Step 1.1: Verify clean working tree**

Run:
```bash
git status --short
```
Expected: only the pre-existing untracked items (`README.md.backup`, `uv.lock`) plus the spec file and this plan file written this session. No modified tracked files.

- [ ] **Step 1.2: Create and switch to the feature branch from main**

Run:
```bash
git checkout main
git pull
git checkout -b feat/pff-fc-events-converter
```
Expected: `Switched to a new branch 'feat/pff-fc-events-converter'`. Branch is on top of latest main (which includes the just-merged 2.5.0 commit `377c1b8`).

- [ ] **Step 1.3: Verify branch position**

Run:
```bash
git rev-parse --abbrev-ref HEAD && git log -1 --oneline
```
Expected: branch `feat/pff-fc-events-converter`; last commit is the 2.5.0 merge.

### Task 2: Capture pytest baseline

**Files:**
- (none — verification only)

- [ ] **Step 2.1: Run baseline pytest count**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: ends with `N passed` for some baseline `N` (record the exact number — subsequent expectations adjust by `+~50`).

---

## Phase 2 — Schema constant

### Task 3: Add `PFF_SPADL_COLUMNS` to `schema.py`

**Files:**
- Modify: `silly_kicks/spadl/schema.py`
- Test: `tests/spadl/test_schema.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/spadl/test_schema.py`:

```python
def test_pff_spadl_columns_extends_spadl_columns():
    """PFF_SPADL_COLUMNS is SPADL_COLUMNS + 4 tackle-passthrough columns."""
    from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS, SPADL_COLUMNS

    expected_extras = {
        "tackle_winner_player_id": "Int64",
        "tackle_winner_team_id":   "Int64",
        "tackle_loser_player_id":  "Int64",
        "tackle_loser_team_id":    "Int64",
    }
    # SPADL_COLUMNS appears verbatim, in order, at the front
    assert list(PFF_SPADL_COLUMNS.keys())[: len(SPADL_COLUMNS)] == list(SPADL_COLUMNS.keys())
    # 4 extras follow, in declared order
    extras_part = {k: PFF_SPADL_COLUMNS[k] for k in list(PFF_SPADL_COLUMNS.keys())[len(SPADL_COLUMNS):]}
    assert extras_part == expected_extras


def test_pff_spadl_columns_uses_int64_extension_dtype():
    """The 4 tackle-passthrough columns use pandas nullable Int64 (capital I)."""
    from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS
    for col in ("tackle_winner_player_id", "tackle_winner_team_id",
                "tackle_loser_player_id",  "tackle_loser_team_id"):
        assert PFF_SPADL_COLUMNS[col] == "Int64"
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/spadl/test_schema.py::test_pff_spadl_columns_extends_spadl_columns -v`
Expected: FAIL with `ImportError: cannot import name 'PFF_SPADL_COLUMNS'`.

- [ ] **Step 3.3: Add the constant to `schema.py`**

In `silly_kicks/spadl/schema.py`, after the `SPORTEC_SPADL_COLUMNS` block (which ends with the `"""..."""` docstring assertion), insert:

```python
PFF_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "tackle_winner_player_id": "Int64",
    "tackle_winner_team_id":   "Int64",
    "tackle_loser_player_id":  "Int64",
    "tackle_loser_team_id":    "Int64",
}
"""PFF SPADL output schema: SPADL_COLUMNS + 4 nullable Int64 tackle-actor
passthrough columns. NaN on rows where no challenge winner/loser is
identifiable (i.e., everywhere except CH events).

Identifier-conventions rationale (ADR-001) shared with SPORTEC_SPADL_COLUMNS.

Dtype departure from SPORTEC_SPADL_COLUMNS (which uses ``object`` strings):
PFF native player/team identifiers are integers, whereas kloppy hands sportec
strings. Using ``Int64`` (pandas nullable) preserves int-ness while allowing
NaN on non-tackle rows. Long-term unification of the two extended schemas
under a common name is a follow-up TODO."""
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `uv run pytest tests/spadl/test_schema.py -v`
Expected: both new tests pass; existing tests in the file still pass.

### Task 4: Export `PFF_SPADL_COLUMNS` from `silly_kicks.spadl`

**Files:**
- Modify: `silly_kicks/spadl/__init__.py`
- Test: `tests/spadl/test_schema.py`

- [ ] **Step 4.1: Write the failing test**

Append to `tests/spadl/test_schema.py`:

```python
def test_pff_spadl_columns_exported_from_top_level():
    """PFF_SPADL_COLUMNS is reachable from `silly_kicks.spadl`."""
    from silly_kicks.spadl import PFF_SPADL_COLUMNS
    assert "tackle_winner_player_id" in PFF_SPADL_COLUMNS
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `uv run pytest tests/spadl/test_schema.py::test_pff_spadl_columns_exported_from_top_level -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 4.3: Update the package exports**

In `silly_kicks/spadl/__init__.py`:

1. Add `"PFF_SPADL_COLUMNS"` to the `__all__` list (alphabetically, between `KLOPPY_SPADL_COLUMNS` if present and `SPADL_COLUMNS` — keep the existing order discipline).
2. Update the schema import line:

   Change:
   ```python
   from .schema import SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
   ```
   to:
   ```python
   from .schema import PFF_SPADL_COLUMNS, SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
   ```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `uv run pytest tests/spadl/test_schema.py -v`
Expected: all passing.

---

## Phase 3 — `_finalize_output` Int64 generalization

### Task 5: Verify `_finalize_output` already handles `Int64`

**Files:**
- Test: `tests/spadl/test_output_contract.py`

- [ ] **Step 5.1: Write a synthetic test**

Append to `tests/spadl/test_output_contract.py`:

```python
def test_finalize_output_supports_int64_extension_dtype():
    """_finalize_output handles pandas nullable Int64 schema entries.

    Required for PFF_SPADL_COLUMNS which uses Int64 on the four
    tackle-passthrough columns (NaN-bearing integer ids).
    """
    import pandas as pd
    from silly_kicks.spadl.utils import _finalize_output

    schema = {"a": "int64", "b": "Int64"}
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, pd.NA, 30]})

    out = _finalize_output(df, schema=schema)
    assert str(out["a"].dtype) == "int64"
    assert str(out["b"].dtype) == "Int64"
    assert out["b"].isna().tolist() == [False, True, False]
```

- [ ] **Step 5.2: Run test**

Run: `uv run pytest tests/spadl/test_output_contract.py::test_finalize_output_supports_int64_extension_dtype -v`

Expected outcome (one of two — both are valid; choose the matching branch):

- **PASS** — `np.dtype("Int64")` already round-trips to the pandas extension dtype on this pandas/numpy combination. No code change needed; **skip Step 5.3**.
- **FAIL** with a numpy/pandas dtype-coercion error — proceed to Step 5.3.

- [ ] **Step 5.3 (only if Step 5.2 failed): Generalize `_finalize_output` for extension dtypes**

In `silly_kicks/spadl/utils.py`, in the `_finalize_output` function, replace the dtype loop:

Old:
```python
    for col, dtype in schema.items():
        # ``np.dtype(dtype)`` narrows the schema's str dtype name to a typed
        # ``DtypeObj`` so pandas-stubs's ``astype`` overload set accepts it.
        result[col] = result[col].astype(np.dtype(dtype))
```

New:
```python
    # Pandas extension dtypes (e.g., "Int64" — nullable) are not numpy dtypes
    # and must be passed as the string name directly. numpy dtypes go through
    # np.dtype() for pandas-stubs typing.
    _PANDAS_EXTENSION_DTYPES: frozenset[str] = frozenset({"Int64", "Int32", "Int16", "Int8",
                                                          "UInt64", "UInt32", "UInt16", "UInt8",
                                                          "Float64", "Float32", "boolean", "string"})
    for col, dtype in schema.items():
        if dtype in _PANDAS_EXTENSION_DTYPES:
            result[col] = result[col].astype(dtype)
        else:
            result[col] = result[col].astype(np.dtype(dtype))
```

- [ ] **Step 5.4: Re-run the test**

Run: `uv run pytest tests/spadl/test_output_contract.py::test_finalize_output_supports_int64_extension_dtype -v`
Expected: PASS.

- [ ] **Step 5.5: Run the full output-contract test file to confirm no regressions**

Run: `uv run pytest tests/spadl/test_output_contract.py -v`
Expected: all passing (existing + new).

---

## Phase 4 — `pff.py` module scaffolding

### Task 6: Create `silly_kicks/spadl/pff.py` skeleton with `EXPECTED_INPUT_COLUMNS`

**Files:**
- Create: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py` (new)

- [ ] **Step 6.1: Write the failing contract tests**

Create `tests/spadl/test_pff.py`:

```python
"""PFF FC DataFrame SPADL converter tests."""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import pff as pff_mod
from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS

# Minimum set of input columns to construct a one-row test DataFrame.
# Mirrors the EXPECTED_INPUT_COLUMNS frozenset in pff.py.
_REQUIRED_COLS = sorted(pff_mod.EXPECTED_INPUT_COLUMNS)


def _df_minimal_pass() -> pd.DataFrame:
    """One-row open-play pass DataFrame; player 1 (home team 100) to player 2."""
    base = {col: [None] for col in _REQUIRED_COLS}
    overrides = {
        "game_id":               [10502],
        "event_id":              [1],
        "possession_event_id":   [1],
        "period_id":             [1],
        "time_seconds":          [10.5],
        "team_id":               [100],
        "player_id":             [1],
        "game_event_type":       ["OTB"],
        "possession_event_type": ["PA"],
        "set_piece_type":        ["O"],
        "ball_x":                [0.0],
        "ball_y":                [0.0],
        "pass_outcome_type":     ["C"],
        "body_type":             ["R"],
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    # Cast nullable-Int64 columns explicitly (the converter expects them).
    for col in ("possession_event_id", "player_id", "carry_defender_player_id",
                "challenger_player_id", "challenger_team_id",
                "challenge_winner_player_id", "challenge_winner_team_id"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


class TestPffContract:
    """Contract: return shape, schema, dtypes, no input mutation."""

    def test_returns_tuple_dataframe_conversion_report(self):
        events = _df_minimal_pass()
        result = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        actions, report = result
        assert isinstance(actions, pd.DataFrame)
        assert report.provider == "PFF"

    def test_output_schema_matches_pff_spadl_columns(self):
        events = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        assert list(actions.columns) == list(PFF_SPADL_COLUMNS.keys())

    def test_dtypes_match_schema(self):
        events = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        for col, expected in PFF_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected, (
                f"{col}: got {actions[col].dtype}, expected {expected}"
            )

    def test_empty_input_returns_empty_actions_with_schema(self):
        empty = pd.DataFrame({c: [] for c in _REQUIRED_COLS})
        actions, report = pff_mod.convert_to_actions(
            empty, home_team_id=100, home_team_start_left=True,
        )
        assert len(actions) == 0
        assert list(actions.columns) == list(PFF_SPADL_COLUMNS.keys())
        assert report.total_events == 0
        assert report.total_actions == 0

    def test_input_dataframe_not_mutated(self):
        events = _df_minimal_pass()
        original_columns = list(events.columns)
        original_len = len(events)
        _, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        assert list(events.columns) == original_columns
        assert len(events) == original_len


class TestPffRequiredColumns:
    """Missing any required input column must raise ValueError with column names."""

    @pytest.mark.parametrize("missing", _REQUIRED_COLS)
    def test_missing_required_column_raises(self, missing):
        events = _df_minimal_pass().drop(columns=[missing])
        with pytest.raises(ValueError, match=missing):
            pff_mod.convert_to_actions(
                events, home_team_id=100, home_team_start_left=True,
            )
```

- [ ] **Step 6.2: Run tests to verify they fail with ImportError**

Run: `uv run pytest tests/spadl/test_pff.py -v`
Expected: ALL FAIL with `ImportError: cannot import name 'pff' from 'silly_kicks.spadl'` (or similar).

- [ ] **Step 6.3: Create `silly_kicks/spadl/pff.py` with the input contract and a stub converter**

Create `silly_kicks/spadl/pff.py`:

```python
"""PFF FC / Gradient Sports DataFrame SPADL converter.

Converts already-flattened PFF events DataFrames (e.g., produced from the
public WC 2022 release JSON via ``pd.json_normalize`` + a roster join) to
SPADL actions.

PFF event vocabulary (recognized as input)
------------------------------------------

PFF events have a hierarchical shape: ``gameEvents`` envelope (high-level
game-event class: OTB / OUT / SUB / KICKOFF / END) + ``possessionEvents``
payload (detailed possession-event class: PA / SH / CR / CL / BC / CH / RE /
TC / IT) + ``fouls[]`` array. Each top-level event JSON list element flattens
to one row in the events DataFrame consumed here.

The converter dispatches on the tuple ``(game_event_type,
possession_event_type, set_piece_type)``. See the spec at
``docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`` § 4.4
for the full mapping table.

Coordinate system
-----------------

PFF source coordinates are pitch-centered meters (origin at center spot,
x ∈ ~[-52.5, 52.5], y ∈ ~[-34, 34]). The converter translates to SPADL's
bottom-left-origin meters (x ∈ [0, 105], y ∈ [0, 68]) and applies per-period
direction-of-play normalization so all teams attack left-to-right (the
standard SPADL invariant).

PFF coordinates reflect actual on-field direction (which switches between
periods); the converter therefore requires explicit direction parameters
(``home_team_start_left`` and, when ET is present, ``home_team_start_left_extratime``).
Both come from PFF metadata JSON (``homeTeamStartLeft``, ``homeTeamStartLeftExtraTime``).

ADR-001: identifier conventions are sacred (silly-kicks 2.0.0)
---------------------------------------------------------------

The converter never overrides ``team_id`` / ``player_id`` from the
on-the-ball actor (PFF ``gameEvents.playerId``). Tackle winner/loser
qualifier values (PFF ``challenge_winner_player_id`` / ``challenger_player_id``)
surface via dedicated output columns:

==========================  ============================================
Output column               PFF qualifier source
==========================  ============================================
``tackle_winner_player_id`` ``challenge_winner_player_id``
``tackle_winner_team_id``   ``challenge_winner_team_id``  (caller-supplied via roster join)
``tackle_loser_player_id``  derived: challenger_player_id if winner != challenger
                              else event row's player_id
``tackle_loser_team_id``    derived: same logic on team_id
==========================  ============================================

The output schema is :data:`silly_kicks.spadl.PFF_SPADL_COLUMNS`
(extends :data:`silly_kicks.spadl.SPADL_COLUMNS` with the 4 tackle columns).
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .schema import PFF_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

# ---------------------------------------------------------------------------
# Required input columns (raise ValueError if any are missing)
# ---------------------------------------------------------------------------
EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset({
    # Identification & timing
    "game_id", "event_id", "possession_event_id",
    "period_id", "time_seconds", "team_id", "player_id",
    # Event-class dispatch keys
    "game_event_type", "possession_event_type", "set_piece_type",
    # Ball position (PFF centered meters)
    "ball_x", "ball_y",
    # Body part / pass / cross qualifiers
    "body_type", "ball_height_type",
    "pass_outcome_type", "pass_type", "incompletion_reason_type",
    "cross_outcome_type", "cross_type", "cross_zone_type",
    # Shot qualifiers
    "shot_outcome_type", "shot_type", "shot_nature_type",
    "shot_initial_height_type", "save_height_type", "save_rebound_type",
    # Carry / dribble qualifiers
    "carry_type", "ball_carry_outcome", "carry_intent",
    "carry_defender_player_id",
    # Challenge / tackle qualifiers (PFF carries actor IDs only as players;
    # caller supplies team affiliation via roster join — see § 4.5 of spec)
    "challenge_type", "challenge_outcome_type",
    "challenger_player_id", "challenger_team_id",
    "challenge_winner_player_id", "challenge_winner_team_id",
    "tackle_attempt_type",
    # Clearance / rebound / GK / touch qualifiers
    "clearance_outcome_type", "rebound_outcome_type", "keeper_touch_type",
    "touch_outcome_type", "touch_type",
    # Foul (one PFF event row has at most one fouls[0] entry; flatten)
    "foul_type", "on_field_offense_type", "final_offense_type",
    "on_field_foul_outcome_type", "final_foul_outcome_type",
})


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert a flattened PFF events DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        PFF-shaped events DataFrame. Required columns:
        :data:`EXPECTED_INPUT_COLUMNS`.
    home_team_id : int
        PFF home-team identifier (``homeTeam.id`` from PFF metadata JSON).
    home_team_start_left : bool
        Whether the home team attacks toward the left goal in period 1
        (``homeTeamStartLeft`` from PFF metadata JSON). Drives per-period
        direction-of-play normalization.
    home_team_start_left_extratime : bool or None, default None
        Same flag for ET periods 3/4 (``homeTeamStartLeftExtraTime`` from
        PFF metadata JSON). Required only if the events span ET; raises
        ``ValueError`` if events have ``period_id`` ∈ {3, 4} but this is
        ``None``.
    preserve_native : list[str] or None, default None
        Optional input columns to passthrough into the output unchanged.

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions matching :data:`silly_kicks.spadl.PFF_SPADL_COLUMNS`
        and a ConversionReport audit trail.

    Raises
    ------
    ValueError
        If any column in :data:`EXPECTED_INPUT_COLUMNS` is missing from
        ``events``, or if ``period_id`` 3/4 rows exist but
        ``home_team_start_left_extratime`` is ``None``.

    Examples
    --------
    Convert a single match's events to SPADL::

        from silly_kicks.spadl import pff
        actions, report = pff.convert_to_actions(
            events,
            home_team_id=366,             # Netherlands in WC 2022 NED-USA
            home_team_start_left=True,    # from match metadata
        )
        assert not report.has_unrecognized
    """
    _validate_input_columns(events, set(EXPECTED_INPUT_COLUMNS), provider="PFF")
    _validate_preserve_native(events, preserve_native, provider="PFF",
                              schema=PFF_SPADL_COLUMNS)

    # Stub: empty output, empty report — Phase 5+ fills in real conversion.
    actions = pd.DataFrame({col: [] for col in PFF_SPADL_COLUMNS.keys()})
    # Initialize the four tackle columns as Int64 explicitly (empty NaN-only).
    for col in ("tackle_winner_player_id", "tackle_winner_team_id",
                "tackle_loser_player_id", "tackle_loser_team_id"):
        actions[col] = pd.array([], dtype="Int64")
    actions = _finalize_output(actions, schema=PFF_SPADL_COLUMNS)

    report = ConversionReport(
        provider="PFF",
        total_events=len(events),
        total_actions=len(actions),
        mapped_counts={},
        excluded_counts={},
        unrecognized_counts={},
    )
    return actions, report
```

- [ ] **Step 6.4: Run the contract tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffContract -v tests/spadl/test_pff.py::TestPffRequiredColumns -v`
Expected: contract tests pass (returns tuple, schema, dtypes, empty input, no mutation, missing-column ValueError). Other test classes will be added in later tasks.

- [ ] **Step 6.5: Lint check**

Run: `uv run ruff check silly_kicks/spadl/pff.py`
Expected: no errors. (Stub `np`, `Counter`, `warnings`, `spadlconfig` imports are present for forthcoming tasks; if ruff complains about unused imports, leave them — they will be consumed by Phase 5+. If ruff blocks the build, add `# noqa: F401` comments inline.)

---

## Phase 5 — Coordinate translation + direction-of-play

### Task 7: Coordinate translation (PFF centered meters → SPADL bottom-left)

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 7.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffCoordinateTranslation:
    """PFF centered meters → SPADL bottom-left meters."""

    def test_center_spot_translates_to_pitch_center(self):
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 0.0
        df.loc[0, "ball_y"] = 0.0
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # Home team in period 1 (start left=True) attacks right, no flip.
        # SPADL center: (52.5, 34.0).
        assert actions.iloc[0]["start_x"] == pytest.approx(52.5)
        assert actions.iloc[0]["start_y"] == pytest.approx(34.0)

    def test_corner_translates_to_pitch_corner(self):
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = -52.5  # PFF centered: left-side corner
        df.loc[0, "ball_y"] = -34.0
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # Home team period 1, no flip → SPADL (0, 0).
        assert actions.iloc[0]["start_x"] == pytest.approx(0.0)
        assert actions.iloc[0]["start_y"] == pytest.approx(0.0)
```

- [ ] **Step 7.2: Run tests to verify they fail**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffCoordinateTranslation -v`
Expected: FAIL — stub returns empty actions, so `actions.iloc[0]` raises `IndexError`.

- [ ] **Step 7.3: Implement the conversion pipeline (first slice — coords only, no dispatch yet)**

In `silly_kicks/spadl/pff.py`, replace the stub body of `convert_to_actions` (after the validation calls) with:

```python
    # ------------------------------------------------------------------
    # Per-period direction lookup (full implementation in Task 8 below)
    # ------------------------------------------------------------------
    if (events["period_id"].isin([3, 4]).any()
            and home_team_start_left_extratime is None):
        raise ValueError(
            "PFF convert_to_actions: events contain ET periods (period_id ∈ {3, 4}) "
            "but home_team_start_left_extratime was not provided. Set it explicitly to "
            "match metadata.homeTeamStartLeftExtraTime, or filter ET events out before calling."
        )
    home_attacks_right_per_period = {
        1: bool(home_team_start_left),
        2: not bool(home_team_start_left),
        3: bool(home_team_start_left_extratime),
        4: not bool(home_team_start_left_extratime) if home_team_start_left_extratime is not None else True,
        5: True,  # PSO — single-end; flip moot for SH+P
    }

    # ------------------------------------------------------------------
    # Coordinate translation (PFF centered → SPADL bottom-left meters)
    # ------------------------------------------------------------------
    actions = pd.DataFrame({
        "game_id":          events["game_id"].astype("int64"),
        "original_event_id": events["event_id"].astype("object"),
        "action_id":        np.arange(len(events), dtype="int64"),
        "period_id":        events["period_id"].astype("int64"),
        "time_seconds":     events["time_seconds"].astype("float64"),
        "team_id":          events["team_id"].astype("int64"),
        "player_id":        events["player_id"].astype("Int64").fillna(0).astype("int64"),
        # Coordinate translation
        "start_x":          (events["ball_x"] + 52.5).astype("float64"),
        "start_y":          (events["ball_y"] + 34.0).astype("float64"),
        "end_x":            (events["ball_x"] + 52.5).astype("float64"),  # filled by Task 19
        "end_y":            (events["ball_y"] + 34.0).astype("float64"),
        # Dispatch defaults; filled in Phase 6
        "type_id":          spadlconfig.actiontype_id["non_action"],
        "result_id":        spadlconfig.result_id["fail"],
        "bodypart_id":      spadlconfig.bodypart_id["foot"],
        # Tackle passthroughs (NaN by default; filled in Task 14)
        "tackle_winner_player_id": pd.array([pd.NA] * len(events), dtype="Int64"),
        "tackle_winner_team_id":   pd.array([pd.NA] * len(events), dtype="Int64"),
        "tackle_loser_player_id":  pd.array([pd.NA] * len(events), dtype="Int64"),
        "tackle_loser_team_id":    pd.array([pd.NA] * len(events), dtype="Int64"),
    })

    # ------------------------------------------------------------------
    # Per-period direction-of-play flip
    # ------------------------------------------------------------------
    period_attacks_right = actions["period_id"].map(home_attacks_right_per_period).astype("bool")
    is_home = actions["team_id"].eq(home_team_id)
    team_attacks_right = is_home == period_attacks_right
    flip_idx = (~team_attacks_right).values
    actions.loc[flip_idx, ["start_x", "end_x"]] = (
        spadlconfig.field_length - actions.loc[flip_idx, ["start_x", "end_x"]].values
    )
    actions.loc[flip_idx, ["start_y", "end_y"]] = (
        spadlconfig.field_width - actions.loc[flip_idx, ["start_y", "end_y"]].values
    )

    # ------------------------------------------------------------------
    # Finalize and report (dispatch counts will be filled in Phase 6)
    # ------------------------------------------------------------------
    actions = _finalize_output(actions, schema=PFF_SPADL_COLUMNS,
                               extra_columns=preserve_native)
    report = ConversionReport(
        provider="PFF",
        total_events=len(events),
        total_actions=len(actions),
        mapped_counts={},
        excluded_counts={},
        unrecognized_counts={},
    )
    return actions, report
```

- [ ] **Step 7.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py -v`
Expected: `TestPffContract` tests still pass (empty input yields empty output; minimal pass → 1 action). `TestPffCoordinateTranslation` tests pass (center spot → 52.5, 34.0; corner → 0, 0).

### Task 8: Direction-of-play flip — period 2 home flip + away period 1 flip

**Files:**
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 8.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffDirectionOfPlay:
    """All teams attack left-to-right after conversion (per-period flip)."""

    def test_home_period1_no_flip(self):
        """Home team, period 1, home_team_start_left=True → no flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25  # ¾ pitch toward right (home goal end in left)
        df.loc[0, "ball_y"] = 0.0
        df.loc[0, "team_id"] = 100  # home
        df.loc[0, "period_id"] = 1
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # SPADL bottom-left: (26.25 + 52.5, 34) = (78.75, 34). No flip.
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)

    def test_away_period1_flips(self):
        """Away team, period 1, home_team_start_left=True → away attacks left, flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "ball_y"] = 0.0
        df.loc[0, "team_id"] = 200  # away (any non-home id)
        df.loc[0, "period_id"] = 1
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # Pre-flip SPADL: 78.75. Away in P1 flips → 105 - 78.75 = 26.25.
        assert actions.iloc[0]["start_x"] == pytest.approx(26.25)

    def test_home_period2_flips(self):
        """Home team, period 2 → home attacks left in P2, flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "team_id"] = 100  # home
        df.loc[0, "period_id"] = 2
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # Pre-flip 78.75. Home P2 flips → 26.25.
        assert actions.iloc[0]["start_x"] == pytest.approx(26.25)

    def test_away_period2_no_flip(self):
        """Away team, period 2 → away attacks right in P2, no flip."""
        df = _df_minimal_pass()
        df.loc[0, "ball_x"] = 26.25
        df.loc[0, "team_id"] = 200  # away
        df.loc[0, "period_id"] = 2
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["start_x"] == pytest.approx(78.75)


class TestPffExtraTimeFallback:
    """ET data without explicit ET-direction param raises ValueError."""

    def test_period3_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            pff_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True,
            )

    def test_period4_event_without_extratime_param_raises(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 4
        with pytest.raises(ValueError, match="home_team_start_left_extratime"):
            pff_mod.convert_to_actions(
                df, home_team_id=100, home_team_start_left=True,
            )

    def test_period3_event_with_extratime_param_succeeds(self):
        df = _df_minimal_pass()
        df.loc[0, "period_id"] = 3
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
            home_team_start_left_extratime=True,
        )
        assert len(actions) == 1
```

- [ ] **Step 8.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffDirectionOfPlay tests/spadl/test_pff.py::TestPffExtraTimeFallback -v`
Expected: PASS (Task 7's implementation already has the per-period and ET fallback logic).

If any test fails, debug the period mapping logic in Task 7's implementation; do not move on until all 7 tests in this section + Task 7 tests pass.

---

## Phase 6 — Dispatch table

### Task 9: Body-part helper

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 9.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffBodyPart:
    """body_type → SPADL bodypart_id mapping."""

    @pytest.mark.parametrize("body_type, expected_name", [
        ("L", "foot_left"),
        ("R", "foot_right"),
        ("H", "head"),
        ("O", "other"),
        (None, "foot"),
    ])
    def test_body_type_dispatch(self, body_type, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "body_type"] = body_type
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        expected_id = spadlconfig.bodypart_id[expected_name]
        assert actions.iloc[0]["bodypart_id"] == expected_id
```

- [ ] **Step 9.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffBodyPart -v`
Expected: FAIL — bodypart_id is hardcoded to `"foot"` in the stub.

- [ ] **Step 9.3: Implement `_dispatch_bodypart`**

In `silly_kicks/spadl/pff.py`, before `convert_to_actions`, add:

```python
def _dispatch_bodypart(body_type: pd.Series) -> pd.Series:
    """Map PFF body_type codes to SPADL bodypart_id (vectorized).

    Mapping:
      L → foot_left, R → foot_right, H → head, O → other, null → foot.
    """
    mapping: dict[object, str] = {
        "L": "foot_left",
        "R": "foot_right",
        "H": "head",
        "O": "other",
    }
    name_series = body_type.map(mapping).fillna("foot")
    return name_series.map(spadlconfig.bodypart_id).astype("int64")
```

In `convert_to_actions`, replace the hardcoded `"bodypart_id": spadlconfig.bodypart_id["foot"]` line with:

```python
        "bodypart_id":      _dispatch_bodypart(events["body_type"]).values,
```

- [ ] **Step 9.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffBodyPart -v`
Expected: PASS for all 5 parameter rows.

### Task 10: Pass dispatch (PA + set-piece composition)

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 10.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffPassDispatch:
    """OTB+PA dispatched by set_piece_type."""

    @pytest.mark.parametrize("set_piece, expected_name", [
        ("O", "pass"),                # open play
        ("K", "pass"),                # kickoff (no SPADL kickoff type)
        ("F", "freekick_short"),
        ("C", "corner_short"),
        ("T", "throw_in"),
        ("G", "goalkick"),
    ])
    def test_pass_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        expected_id = spadlconfig.actiontype_id[expected_name]
        assert actions.iloc[0]["type_id"] == expected_id

    def test_pass_outcome_complete_is_success(self):
        df = _df_minimal_pass()
        df.loc[0, "pass_outcome_type"] = "C"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]

    def test_pass_outcome_fail_is_fail(self):
        df = _df_minimal_pass()
        df.loc[0, "pass_outcome_type"] = "F"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["fail"]
```

- [ ] **Step 10.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffPassDispatch -v`
Expected: FAIL — type_id is `non_action`, result_id is `fail` for all.

- [ ] **Step 10.3: Implement dispatch using `np.select`**

In `silly_kicks/spadl/pff.py`, before `convert_to_actions`, add:

```python
def _dispatch_actiontype_resultid(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized dispatch from (game_event_type, possession_event_type, set_piece_type)
    to (type_id, result_id) per row.

    Returns
    -------
    (type_id_arr, result_id_arr) : tuple of np.ndarray
        Both arrays have length len(events). type_id defaults to non_action;
        result_id defaults to fail. Subsequent dispatch passes (foul, tackle,
        rebound) refine these on specific rows.
    """
    ge = events["game_event_type"].fillna("").to_numpy()
    pe = events["possession_event_type"].fillna("").to_numpy()
    sp = events["set_piece_type"].fillna("").to_numpy()
    pass_out = events["pass_outcome_type"].fillna("").to_numpy()
    cross_out = events["cross_outcome_type"].fillna("").to_numpy()
    shot_out = events["shot_outcome_type"].fillna("").to_numpy()
    foul_outcome = events["final_foul_outcome_type"].fillna("").to_numpy()

    A = spadlconfig.actiontype_id  # local alias for table compactness
    R = spadlconfig.result_id

    # type_id dispatch: top-down priority (np.select picks first match)
    type_conds = [
        (ge == "OTB") & (pe == "PA") & (sp == "F"),
        (ge == "OTB") & (pe == "PA") & (sp == "C"),
        (ge == "OTB") & (pe == "PA") & (sp == "T"),
        (ge == "OTB") & (pe == "PA") & (sp == "G"),
        (ge == "OTB") & (pe == "PA"),                    # PA + O / K / unknown → pass
        (ge == "OTB") & (pe == "CR") & (sp == "F"),
        (ge == "OTB") & (pe == "CR") & (sp == "C"),
        (ge == "OTB") & (pe == "CR"),                    # CR + O / others → cross
        (ge == "OTB") & (pe == "SH") & (sp == "F"),
        (ge == "OTB") & (pe == "SH") & (sp == "P"),
        (ge == "OTB") & (pe == "SH"),                    # SH + O / K / others → shot
        (ge == "OTB") & (pe == "CL"),
        (ge == "OTB") & (pe == "BC"),
        (ge == "OTB") & (pe == "CH"),
        (ge == "OTB") & (pe == "RE"),
        (ge == "OTB") & (pe == "TC"),
    ]
    type_choices = [
        A["freekick_short"], A["corner_short"], A["throw_in"], A["goalkick"], A["pass"],
        A["freekick_crossed"], A["corner_crossed"], A["cross"],
        A["shot_freekick"], A["shot_penalty"], A["shot"],
        A["clearance"],
        A["dribble"],
        A["tackle"],
        A["keeper_save"],          # default; refined for keeper_pick_up in Task 13
        A["bad_touch"],
    ]
    type_id_arr = np.select(type_conds, type_choices, default=A["non_action"]).astype("int64")

    # result_id dispatch: pass / cross outcomes
    is_pass_class = (pe == "PA") | (pe == "CR")
    pass_success = is_pass_class & ((pass_out == "C") | (cross_out == "C"))

    # shot outcomes
    is_shot = pe == "SH"
    shot_goal = is_shot & (shot_out == "G")
    shot_owngoal = is_shot & (shot_out == "O")

    # foul card outcomes (final_foul_outcome_type)
    is_yellow = pd.Series(foul_outcome).str.startswith(("Y", "2Y")).fillna(False).to_numpy()
    is_red = pd.Series(foul_outcome).str.startswith(("R", "SR")).fillna(False).to_numpy()

    result_conds = [pass_success, shot_goal, shot_owngoal, is_yellow, is_red]
    result_choices = [R["success"], R["success"], R["owngoal"], R["yellow_card"], R["red_card"]]
    result_id_arr = np.select(result_conds, result_choices, default=R["fail"]).astype("int64")

    return type_id_arr, result_id_arr
```

In `convert_to_actions`, after the bodypart line, replace the two hardcoded type_id / result_id literals with a call:

Replace:
```python
        "type_id":          spadlconfig.actiontype_id["non_action"],
        "result_id":        spadlconfig.result_id["fail"],
        "bodypart_id":      _dispatch_bodypart(events["body_type"]).values,
```

with:
```python
        "type_id":          spadlconfig.actiontype_id["non_action"],   # filled below
        "result_id":        spadlconfig.result_id["fail"],              # filled below
        "bodypart_id":      _dispatch_bodypart(events["body_type"]).values,
```

(Keep the placeholders so the column exists.) Then immediately after the `actions = pd.DataFrame(...)` block but before the direction-flip, add:

```python
    type_id_arr, result_id_arr = _dispatch_actiontype_resultid(events)
    actions["type_id"] = type_id_arr
    actions["result_id"] = result_id_arr
```

- [ ] **Step 10.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffPassDispatch -v`
Expected: PASS for all 6 parametrized rows + the two outcome tests.

### Task 11: Cross dispatch + set-piece composition

**Files:**
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 11.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffCrossDispatch:
    """OTB+CR dispatched by set_piece_type."""

    @pytest.mark.parametrize("set_piece, expected_name", [
        ("O", "cross"),
        ("F", "freekick_crossed"),
        ("C", "corner_crossed"),
    ])
    def test_cross_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        expected_id = spadlconfig.actiontype_id[expected_name]
        assert actions.iloc[0]["type_id"] == expected_id

    def test_cross_outcome_uses_pass_outcome(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CR"
        df.loc[0, "cross_outcome_type"] = "C"
        df.loc[0, "pass_outcome_type"] = None
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id["success"]
```

- [ ] **Step 11.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffCrossDispatch -v`
Expected: PASS — Task 10's dispatch already covers CR rows.

If any fail, inspect `_dispatch_actiontype_resultid` priority order — `(pe == "CR") & (sp == "F")` must come before the default `(pe == "CR")`.

### Task 12: Shot dispatch + result mapping

**Files:**
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 12.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffShotDispatch:
    """OTB+SH dispatched by set_piece_type, results from shot_outcome_type."""

    @pytest.mark.parametrize("set_piece, expected_name", [
        ("O", "shot"),
        ("F", "shot_freekick"),
        ("P", "shot_penalty"),
    ])
    def test_shot_set_piece_composition(self, set_piece, expected_name):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "SH"
        df.loc[0, "set_piece_type"] = set_piece
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id[expected_name]

    @pytest.mark.parametrize("shot_outcome, expected_result", [
        ("G", "success"),
        ("O", "owngoal"),
        ("S", "fail"),
        ("B", "fail"),
        ("W", "fail"),
        ("M", "fail"),
        (None, "fail"),
    ])
    def test_shot_result_mapping(self, shot_outcome, expected_result):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "SH"
        df.loc[0, "shot_outcome_type"] = shot_outcome
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["result_id"] == spadlconfig.result_id[expected_result]
```

- [ ] **Step 12.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffShotDispatch -v`
Expected: PASS for all 10 parametrized rows.

### Task 13: Rebound (RE) → keeper_save / keeper_pick_up disambiguation

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 13.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffRebound:
    """RE events disambiguate by keeper_touch_type → keeper_save / keeper_pick_up."""

    def test_rebound_default_is_keeper_save(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = None
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_save"]

    def test_rebound_catch_class_is_keeper_pick_up(self):
        """Catch-class keeper_touch_type → keeper_pick_up."""
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "RE"
        df.loc[0, "keeper_touch_type"] = "C"  # catch — exemplar catch-class code
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["keeper_pick_up"]
```

NOTE: The exact PFF `keeper_touch_type` code letters for catch vs deflect are not enumerated by this spec; the test uses `"C"` as a placeholder catch code. When the synthetic match is authored during Task 19, enumerate the actual codes used (the generator script is the source of truth for the synthetic vocabulary). If `"C"` is not the catch indicator in your generator's authored vocabulary, update both the test in this task AND the `catch_class` set in step 13.3 simultaneously so the synthetic match exercises the keeper_pick_up branch.

- [ ] **Step 13.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffRebound -v`
Expected: first test PASSES (default RE is keeper_save), second test FAILS (refinement not in dispatch).

- [ ] **Step 13.3: Implement keeper_pick_up refinement**

In `silly_kicks/spadl/pff.py`, in `_dispatch_actiontype_resultid` after the `np.select` for `type_id_arr`, add:

```python
    # Refinement: RE rows with catch-class keeper_touch_type → keeper_pick_up
    keeper_touch = events["keeper_touch_type"].fillna("").to_numpy()
    catch_class = {"C"}  # extend at synthetic-match-generator time when real
                         # PFF keeper_touch_type vocabulary is enumerated
    is_catch = (pe == "RE") & np.isin(keeper_touch, list(catch_class))
    type_id_arr = np.where(is_catch, A["keeper_pick_up"], type_id_arr).astype("int64")
```

- [ ] **Step 13.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffRebound -v`
Expected: both pass.

### Task 14: Tackle (CH) — populate `tackle_winner_*` / `tackle_loser_*` columns

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 14.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffTackle:
    """OTB+CH → SPADL tackle, with winner/loser passthrough columns."""

    def _df_tackle(self, winner_id, winner_team_id):
        """Carrier (player 1, team 100) is challenged by player 5 (team 200)."""
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CH"
        df.loc[0, "challenger_player_id"] = 5
        df.loc[0, "challenger_team_id"] = 200
        df.loc[0, "challenge_winner_player_id"] = winner_id
        df.loc[0, "challenge_winner_team_id"] = winner_team_id
        df["challenger_player_id"] = df["challenger_player_id"].astype("Int64")
        df["challenger_team_id"] = df["challenger_team_id"].astype("Int64")
        df["challenge_winner_player_id"] = df["challenge_winner_player_id"].astype("Int64")
        df["challenge_winner_team_id"] = df["challenge_winner_team_id"].astype("Int64")
        return df

    def test_tackle_type_id_set(self):
        df = self._df_tackle(winner_id=5, winner_team_id=200)
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["tackle"]

    def test_tackle_winner_columns_populated_when_challenger_wins(self):
        """Challenger (5/200) wins → carrier (1/100) lost."""
        df = self._df_tackle(winner_id=5, winner_team_id=200)
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["tackle_winner_player_id"] == 5
        assert actions.iloc[0]["tackle_winner_team_id"] == 200
        assert actions.iloc[0]["tackle_loser_player_id"] == 1
        assert actions.iloc[0]["tackle_loser_team_id"] == 100

    def test_tackle_winner_columns_populated_when_carrier_holds(self):
        """Carrier (1/100) wins (== event_player_id) → challenger (5/200) lost."""
        df = self._df_tackle(winner_id=1, winner_team_id=100)
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["tackle_winner_player_id"] == 1
        assert actions.iloc[0]["tackle_winner_team_id"] == 100
        assert actions.iloc[0]["tackle_loser_player_id"] == 5
        assert actions.iloc[0]["tackle_loser_team_id"] == 200

    def test_tackle_passthrough_NaN_on_non_tackle_rows(self):
        """A pass row has NA on all four tackle columns."""
        df = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        for col in ("tackle_winner_player_id", "tackle_winner_team_id",
                    "tackle_loser_player_id", "tackle_loser_team_id"):
            assert pd.isna(actions.iloc[0][col]), f"{col} should be NA on a pass row"
```

- [ ] **Step 14.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffTackle -v`
Expected: `test_tackle_type_id_set` and `test_tackle_passthrough_NaN_on_non_tackle_rows` PASS;  `test_tackle_winner_columns_populated_*` FAIL — columns are still NA.

- [ ] **Step 14.3: Populate tackle passthrough columns**

In `silly_kicks/spadl/pff.py`, in `convert_to_actions`, after the `actions["type_id"] = type_id_arr` / `actions["result_id"] = result_id_arr` lines, add:

```python
    # ------------------------------------------------------------------
    # Tackle winner/loser passthrough (ADR-001)
    # ------------------------------------------------------------------
    is_tackle = (events["possession_event_type"].fillna("") == "CH").to_numpy()

    # winner is direct from PFF qualifier
    winner_pid = events["challenge_winner_player_id"].astype("Int64")
    winner_tid = events["challenge_winner_team_id"].astype("Int64")

    # loser: challenger lost iff winner != challenger; otherwise carrier (event row) lost.
    challenger_pid = events["challenger_player_id"].astype("Int64")
    challenger_tid = events["challenger_team_id"].astype("Int64")
    event_pid = events["player_id"].astype("Int64")
    event_tid = events["team_id"].astype("Int64")

    challenger_won = (winner_pid == challenger_pid).fillna(False).to_numpy()
    # When challenger_won → loser = event row (carrier lost)
    # When !challenger_won → loser = challenger
    loser_pid = pd.Series(
        np.where(challenger_won, event_pid.values, challenger_pid.values),
        index=events.index,
    ).astype("Int64")
    loser_tid = pd.Series(
        np.where(challenger_won, event_tid.values, challenger_tid.values),
        index=events.index,
    ).astype("Int64")

    actions["tackle_winner_player_id"] = pd.array(
        np.where(is_tackle, winner_pid.values, pd.NA), dtype="Int64",
    )
    actions["tackle_winner_team_id"] = pd.array(
        np.where(is_tackle, winner_tid.values, pd.NA), dtype="Int64",
    )
    actions["tackle_loser_player_id"] = pd.array(
        np.where(is_tackle, loser_pid.values, pd.NA), dtype="Int64",
    )
    actions["tackle_loser_team_id"] = pd.array(
        np.where(is_tackle, loser_tid.values, pd.NA), dtype="Int64",
    )
```

- [ ] **Step 14.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffTackle -v`
Expected: all 4 pass.

### Task 15: Clearance / dribble (BC) / bad_touch (TC)

**Files:**
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 15.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffClearanceDribbleTouchControl:
    """OTB+CL → clearance, OTB+BC → dribble, OTB+TC → bad_touch."""

    def test_clearance(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "CL"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["clearance"]

    def test_ball_carry(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "BC"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["dribble"]

    def test_touch_control(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "TC"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[0]["type_id"] == spadlconfig.actiontype_id["bad_touch"]
```

- [ ] **Step 15.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffClearanceDribbleTouchControl -v`
Expected: PASS — Task 10's dispatch already covers CL, BC, TC.

### Task 16: Foul row synthesis + card mapping

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 16.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffFoul:
    """Rows with foul_type non-null synthesize an extra SPADL foul action."""

    def test_foul_synthesizes_additional_action(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"  # any non-null value
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # Original PA row + synthesized foul row = 2
        assert len(actions) == 2
        assert actions["type_id"].tolist() == [
            spadlconfig.actiontype_id["pass"],
            spadlconfig.actiontype_id["foul"],
        ]

    def test_foul_yellow_card(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        df.loc[0, "final_foul_outcome_type"] = "Y"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        foul_row = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]].iloc[0]
        assert foul_row["result_id"] == spadlconfig.result_id["yellow_card"]

    def test_foul_red_card(self):
        df = _df_minimal_pass()
        df.loc[0, "foul_type"] = "STANDARD"
        df.loc[0, "final_foul_outcome_type"] = "R"
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        foul_row = actions[actions["type_id"] == spadlconfig.actiontype_id["foul"]].iloc[0]
        assert foul_row["result_id"] == spadlconfig.result_id["red_card"]

    def test_no_foul_no_synthesis(self):
        df = _df_minimal_pass()
        # foul_type left null
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert len(actions) == 1
```

- [ ] **Step 16.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffFoul -v`
Expected: `test_no_foul_no_synthesis` PASSES; the others FAIL — synthesis not yet implemented.

- [ ] **Step 16.3: Implement foul synthesis**

In `silly_kicks/spadl/pff.py`, in `convert_to_actions`, after the tackle-passthrough block (Task 14) and before the direction-of-play flip, add:

```python
    # ------------------------------------------------------------------
    # Foul row synthesis (one extra SPADL row when foul_type non-null)
    # ------------------------------------------------------------------
    foul_mask = events["foul_type"].notna().values
    if foul_mask.any():
        foul_rows = actions.loc[foul_mask].copy()
        foul_rows["type_id"] = spadlconfig.actiontype_id["foul"]
        # Result: derive from final_foul_outcome_type
        foul_outcome = events.loc[foul_mask, "final_foul_outcome_type"].fillna("").to_numpy()
        is_yellow = pd.Series(foul_outcome).str.startswith(("Y", "2Y")).fillna(False).to_numpy()
        is_red = pd.Series(foul_outcome).str.startswith(("R", "SR")).fillna(False).to_numpy()
        result_arr = np.where(is_yellow, spadlconfig.result_id["yellow_card"],
                              np.where(is_red, spadlconfig.result_id["red_card"],
                                       spadlconfig.result_id["success"]))
        foul_rows["result_id"] = result_arr.astype("int64")
        # Body part: foul defaults to foot
        foul_rows["bodypart_id"] = spadlconfig.bodypart_id["foot"]

        # Combine: insert foul rows immediately after their parent rows.
        # Use shifted action_ids — fouls get .5 offsets, then renumber dense.
        actions["__order__"] = np.arange(len(actions), dtype="float64")
        foul_rows["__order__"] = np.arange(len(actions))[foul_mask] + 0.5
        actions = pd.concat([actions, foul_rows], ignore_index=True)
        actions = actions.sort_values("__order__").reset_index(drop=True)
        actions = actions.drop(columns="__order__")
        actions["action_id"] = np.arange(len(actions), dtype="int64")
```

- [ ] **Step 16.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffFoul -v`
Expected: all 4 pass.

### Task 17: Exclusions + ConversionReport `mapped_counts` / `excluded_counts` / `unrecognized_counts`

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 17.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffExclusions:
    """OUT / SUB / period-boundary / OTB+IT events excluded from output;
    counts surface in ConversionReport.excluded_counts."""

    @pytest.mark.parametrize("ge_type", ["OUT", "SUB", "FIRSTKICKOFF", "SECONDKICKOFF", "END"])
    def test_excluded_game_event_types_drop_out(self, ge_type):
        df = _df_minimal_pass()
        df.loc[0, "game_event_type"] = ge_type
        df.loc[0, "possession_event_type"] = None
        actions, report = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get(ge_type) == 1

    def test_otb_plus_it_excluded(self):
        df = _df_minimal_pass()
        df.loc[0, "possession_event_type"] = "IT"
        actions, report = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert len(actions) == 0
        assert report.excluded_counts.get("OTB+IT") == 1


class TestPffReportCounts:
    """ConversionReport.mapped_counts uses SPADL action-type names."""

    def test_mapped_counts_uses_spadl_names(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "possession_event_type"] = "SH"
        df.loc[1, "shot_outcome_type"] = "G"
        df.loc[1, "time_seconds"] = 11.0
        actions, report = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert report.mapped_counts.get("pass") == 1
        assert report.mapped_counts.get("shot") == 1
```

- [ ] **Step 17.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffExclusions tests/spadl/test_pff.py::TestPffReportCounts -v`
Expected: FAIL — exclusion-row filtering not implemented; mapped_counts is empty.

- [ ] **Step 17.3: Implement exclusion filtering and report population**

In `silly_kicks/spadl/pff.py`, in `convert_to_actions`, restructure the early portion:

After the validation block, before the `actions = pd.DataFrame(...)` block, insert:

```python
    # ------------------------------------------------------------------
    # Exclusion filtering — drop rows whose (game_event_type,
    # possession_event_type) pair is in the documented excluded set.
    # ------------------------------------------------------------------
    ge_arr = events["game_event_type"].fillna("").to_numpy()
    pe_arr = events["possession_event_type"].fillna("").to_numpy()

    excluded_ge_types = {"OUT", "SUB", "FIRSTKICKOFF", "SECONDKICKOFF", "END"}
    excluded_pair_keys = {("OTB", "IT")}

    is_excluded_ge = np.isin(ge_arr, list(excluded_ge_types))
    is_excluded_pair = np.zeros(len(events), dtype=bool)
    for ge_, pe_ in excluded_pair_keys:
        is_excluded_pair |= (ge_arr == ge_) & (pe_arr == pe_)
    is_excluded = is_excluded_ge | is_excluded_pair

    # excluded_counts dict
    excluded_counts: Counter = Counter()
    for ge_t in ge_arr[is_excluded_ge]:
        excluded_counts[str(ge_t)] += 1
    for ge_, pe_ in excluded_pair_keys:
        n = int(((ge_arr == ge_) & (pe_arr == pe_)).sum())
        if n > 0:
            excluded_counts[f"{ge_}+{pe_}"] = n

    keep = ~is_excluded
    events = events.loc[keep].reset_index(drop=True)
```

(This filters `events` in place; the rest of `convert_to_actions` operates on the filtered DataFrame.)

Then near the end, replace the report construction with:

```python
    # ------------------------------------------------------------------
    # ConversionReport: mapped_counts uses SPADL action-type names
    # ------------------------------------------------------------------
    id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
    mapped_counts: Counter = Counter()
    for tid in actions["type_id"].to_numpy():
        name = id_to_name.get(int(tid), "non_action")
        if name == "non_action":
            continue
        mapped_counts[name] += 1

    # Unrecognized: rows mapped to non_action that were NOT in the excluded set
    # (already filtered out above) — these surface as (ge, pe) tuple-strings.
    unrecognized_counts: Counter = Counter()
    non_action_id = spadlconfig.actiontype_id["non_action"]
    if (actions["type_id"] == non_action_id).any():
        # Recover original ge/pe via the (already-filtered) events DataFrame
        # which is row-aligned with actions pre-foul-synthesis. Foul-synthesized
        # rows always have type_id != non_action, so they don't contribute here.
        # Use action_id<len(events) to identify pre-synthesis rows; safe because
        # foul-synthesized action_ids are appended via the renumber pass.
        # Simpler: re-run dispatch markers on the filtered events.
        pe_filtered = events["possession_event_type"].fillna("").to_numpy()
        ge_filtered = events["game_event_type"].fillna("").to_numpy()
        # An events row maps to non_action iff its (ge, pe) didn't hit any branch.
        # For our dispatch all OTB rows with known PE map; the residual is OTB+
        # an unmapped PE, or something else entirely.
        for i in range(len(events)):
            tid_first_action = actions[actions["original_event_id"] == events["event_id"].iloc[i]]
            if len(tid_first_action) > 0 and (tid_first_action["type_id"].iloc[0] == non_action_id):
                key = f"{ge_filtered[i]}+{pe_filtered[i]}"
                unrecognized_counts[key] += 1

    report = ConversionReport(
        provider="PFF",
        total_events=len(events) + sum(excluded_counts.values()),
        total_actions=len(actions),
        mapped_counts=dict(mapped_counts),
        excluded_counts=dict(excluded_counts),
        unrecognized_counts=dict(unrecognized_counts),
    )
    return actions, report
```

Note: `total_events` reports the original input event count, not the post-filter count. This matches the contract used by other converters (e.g., statsbomb counts every input row including excluded ones).

- [ ] **Step 17.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffExclusions tests/spadl/test_pff.py::TestPffReportCounts -v`
Expected: all pass.

- [ ] **Step 17.5: Re-run the full test_pff.py to ensure no regression**

Run: `uv run pytest tests/spadl/test_pff.py -v`
Expected: all tests in the file pass. Existing contract / coordinate / direction / dispatch / tackle / foul / exclusion tests all green.

---

## Phase 7 — End-coordinate fill

### Task 18: `end_x` / `end_y` from next-action start_x/y

**Files:**
- Modify: `silly_kicks/spadl/pff.py`
- Test: `tests/spadl/test_pff.py`

- [ ] **Step 18.1: Write failing tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffEndCoordinates:
    """end_x/end_y of each action equals start_x/start_y of the next action
    in the same period (chained-event semantics)."""

    def test_pass_end_is_next_start(self):
        df = pd.concat([_df_minimal_pass(), _df_minimal_pass()], ignore_index=True)
        df.loc[1, "event_id"] = 2
        df.loc[1, "possession_event_id"] = 2
        df.loc[1, "ball_x"] = 20.0  # second event 20m forward
        df.loc[1, "ball_y"] = 5.0
        df.loc[1, "time_seconds"] = 11.0
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        # First action end should match second action start
        assert actions.iloc[0]["end_x"] == pytest.approx(actions.iloc[1]["start_x"])
        assert actions.iloc[0]["end_y"] == pytest.approx(actions.iloc[1]["start_y"])

    def test_last_action_end_equals_start(self):
        """Last action has no successor — end falls back to its own start."""
        df = _df_minimal_pass()
        actions, _ = pff_mod.convert_to_actions(
            df, home_team_id=100, home_team_start_left=True,
        )
        assert actions.iloc[-1]["end_x"] == pytest.approx(actions.iloc[-1]["start_x"])
        assert actions.iloc[-1]["end_y"] == pytest.approx(actions.iloc[-1]["start_y"])
```

- [ ] **Step 18.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffEndCoordinates -v`
Expected: `test_pass_end_is_next_start` FAILS — end_x is currently set to the same row's start_x. `test_last_action_end_equals_start` PASSES (already the default).

- [ ] **Step 18.3: Implement end-coordinate fill**

In `silly_kicks/spadl/pff.py`, in `convert_to_actions`, AFTER the direction-of-play flip and BEFORE the `_finalize_output` call, add:

```python
    # ------------------------------------------------------------------
    # end_x / end_y from next-action start_x/y within same period.
    # Last row of each period falls back to its own start_x/y (already true).
    # ------------------------------------------------------------------
    if len(actions) > 0:
        next_start_x = actions["start_x"].shift(-1)
        next_start_y = actions["start_y"].shift(-1)
        same_period = actions["period_id"].eq(actions["period_id"].shift(-1))
        actions["end_x"] = np.where(
            same_period & next_start_x.notna(), next_start_x, actions["start_x"],
        )
        actions["end_y"] = np.where(
            same_period & next_start_y.notna(), next_start_y, actions["start_y"],
        )
```

- [ ] **Step 18.4: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffEndCoordinates -v`
Expected: both pass.

- [ ] **Step 18.5: Final regression check on the full pff test file**

Run: `uv run pytest tests/spadl/test_pff.py -v`
Expected: every test in the file passes.

---

## Phase 8 — Synthetic-match fixture

### Task 19: Create the synthetic-match generator script

**Files:**
- Create: `tests/datasets/pff/_generate_synthetic_match.py`
- Create: `tests/datasets/pff/synthetic_match.json`
- Create: `tests/datasets/pff/README.md`

- [ ] **Step 19.1: Create the directory**

Run:
```bash
mkdir -p tests/datasets/pff
```

- [ ] **Step 19.2: Write the generator script**

Create `tests/datasets/pff/_generate_synthetic_match.py`:

```python
"""Deterministic generator for the PFF FC synthetic match fixture.

Produces ``synthetic_match.json`` covering every dispatch row from
``silly_kicks/spadl/pff.py`` with ≥2× redundancy, plus exclusion / set-piece
/ result / body-part / tackle-winner / foul / card variations needed to
exercise the full converter contract.

Run:
    uv run python tests/datasets/pff/_generate_synthetic_match.py

The resulting JSON is committed and is the canonical test artifact. The
generator is not invoked from pytest; it is a maintainer-time tool.
"""

from __future__ import annotations

import json
from pathlib import Path

# Two synthetic teams. Player IDs 1-11 home, 12-22 away.
HOME_TEAM_ID = 100
AWAY_TEAM_ID = 200
HOME_TEAM_NAME = "Synthetic FC"
AWAY_TEAM_NAME = "Test United"
HOME_PLAYERS = list(range(1, 12))
AWAY_PLAYERS = list(range(12, 23))

OUTPUT_PATH = Path(__file__).parent / "synthetic_match.json"


def _ge_envelope(ge_type, *, period, time_s, team_id, player_id,
                 set_piece="O", home_team=True):
    """gameEvents skeleton."""
    return {
        "gameEventType": ge_type,
        "initialNonEvent": False,
        "startGameClock": int(time_s),
        "startFormattedGameClock": f"{int(time_s) // 60:02d}:{int(time_s) % 60:02d}",
        "period": period,
        "videoMissing": False,
        "teamId": team_id,
        "teamName": HOME_TEAM_NAME if home_team else AWAY_TEAM_NAME,
        "homeTeam": home_team,
        "playerId": player_id,
        "playerName": f"Player {player_id}",
        "touches": 1,
        "touchesInBox": 0,
        "setpieceType": set_piece,
        "earlyDistribution": False,
        "videoUrl": None,
        "endType": None,
        "outType": None,
        "subType": None,
        "playerOffId": None, "playerOffName": None, "playerOffType": None,
        "playerOnId": None, "playerOnName": None,
    }


def _empty_pe():
    """possessionEvents skeleton with all qualifier fields null."""
    return {
        "possessionEventType": None, "nonEvent": False,
        "gameClock": 0, "formattedGameClock": "00:00",
        "eventVideoUrl": None,
        "ballHeightType": None, "bodyType": None, "highPointType": None,
        "passerPlayerId": None, "passerPlayerName": None, "passType": None, "passOutcomeType": None,
        "crosserPlayerId": None, "crosserPlayerName": None,
        "crossType": None, "crossZoneType": None, "crossOutcomeType": None,
        "targetPlayerId": None, "targetPlayerName": None, "targetFacingType": None,
        "receiverPlayerId": None, "receiverPlayerName": None, "receiverFacingType": None,
        "defenderPlayerId": None, "defenderPlayerName": None,
        "blockerPlayerId": None, "blockerPlayerName": None,
        "deflectorPlayerId": None, "deflectorPlayerName": None,
        "failedBlockerPlayerId": None, "failedBlockerPlayerName": None,
        "failedBlocker2PlayerId": None, "failedBlocker2PlayerName": None,
        "accuracyType": None, "incompletionReasonType": None, "secondIncompletionReasonType": None,
        "linesBrokenType": None,
        "shooterPlayerId": None, "shooterPlayerName": None,
        "bodyMovementType": None, "ballMoving": None, "shotType": None, "shotNatureType": None,
        "shotInitialHeightType": None, "shotOutcomeType": None,
        "keeperPlayerId": None, "keeperPlayerName": None,
        "saveHeightType": None, "saveReboundType": None, "keeperTouchType": None,
        "glClearerPlayerId": None, "glClearerPlayerName": None,
        "badParry": None, "saveable": None,
        "clearerPlayerId": None, "clearerPlayerName": None, "clearanceOutcomeType": None,
        "carrierPlayerId": None, "carrierPlayerName": None,
        "ballCarrierPlayerId": None, "ballCarrierPlayerName": None,
        "ballCarryOutcome": None, "carryDefenderPlayerId": None, "carryDefenderPlayerName": None,
        "carryIntent": None, "carrySuccessful": None, "carryType": None,
        "challengerPlayerId": None, "challengerPlayerName": None,
        "challengeType": None, "challengeOutcomeType": None,
        "challengeWinnerPlayerId": None, "challengeWinnerPlayerName": None,
        "challengeKeeperPlayerId": None, "challengeKeeperPlayerName": None,
        "tackleAttemptType": None,
        "rebounderPlayerId": None, "rebounderPlayerName": None, "reboundOutcomeType": None,
        "touchPlayerId": None, "touchPlayerName": None,
        "touchType": None, "touchOutcomeType": None,
        "missedTouchPlayerId": None, "missedTouchPlayerName": None, "missedTouchType": None,
        "originateType": None, "opportunityType": None, "trickType": None,
        "createsSpace": None,
        "pressureType": None, "pressurePlayerId": None, "pressurePlayerName": None,
        "closingDownPlayerId": None, "closingDownPlayerName": None,
        "additionalDuelerPlayerId": None, "additionalDuelerPlayerName": None,
        "betterOptionPlayerId": None, "betterOptionPlayerName": None,
        "betterOptionTime": None, "betterOptionType": None,
        "movementPlayerId": None, "movementPlayerName": None,
        "positionPlayerId": None, "positionPlayerName": None,
        "homeDuelPlayerId": None, "homeDuelPlayerName": None,
        "awayDuelPlayerId": None, "awayDuelPlayerName": None,
        "dribblerPlayerId": None, "dribblerPlayerName": None, "dribbleType": None,
    }


def _empty_foul():
    return [{
        "onFieldCulpritPlayerId": None, "onFieldCulpritPlayerName": None,
        "finalCulpritPlayerId": None, "finalCulpritPlayerName": None,
        "victimPlayerId": None, "victimPlayerName": None,
        "foulType": None,
        "onFieldOffenseType": None, "finalOffenseType": None,
        "onFieldFoulOutcomeType": None, "finalFoulOutcomeType": None,
        "var": None, "varReasonType": None, "correctDecision": None,
    }]


def make_event(event_id, *, ge_type="OTB", pe_type=None, period=1, time_s=10.0,
               team_id=HOME_TEAM_ID, player_id=1, set_piece="O", home_team=True,
               ball_x=0.0, ball_y=0.0, **pe_overrides):
    """Build one PFF event row."""
    pe = _empty_pe()
    if pe_type is not None:
        pe["possessionEventType"] = pe_type
    pe.update(pe_overrides)
    return {
        "gameId": 99999,
        "gameEventId": event_id,
        "possessionEventId": event_id,  # 1:1 mapping for synthetic data
        "startTime": 200.0 + time_s,
        "endTime": 200.0 + time_s,
        "duration": 0.0,
        "eventTime": 200.0 + time_s,
        "sequence": float(event_id),
        "gameEvents": _ge_envelope(ge_type, period=period, time_s=time_s,
                                   team_id=team_id, player_id=player_id,
                                   set_piece=set_piece, home_team=home_team),
        "initialTouch": {"initialBodyType": "R", "initialHeightType": "G",
                         "facingType": "G", "initialTouchType": "S",
                         "initialPressureType": "N", "initialPressurePlayerId": None,
                         "initialPressurePlayerName": None},
        "possessionEvents": pe,
        "fouls": _empty_foul(),
        "grades": [],
        "stadiumMetadata": [],
        "homePlayers": [],
        "awayPlayers": [],
        "ball": [{"visibility": "VISIBLE", "x": ball_x, "y": ball_y, "z": 0.0}],
    }


def build_match():
    """Compose the synthetic match's events list with full dispatch coverage."""
    events: list[dict] = []
    eid = 1

    # ---------- KICKOFF + FIRST/SECOND HALF MARKERS (excluded) ----------
    events.append(make_event(eid, ge_type="FIRSTKICKOFF", pe_type=None,
                             period=1, time_s=0.0, ball_x=0.0, ball_y=0.0))
    eid += 1

    # ---------- PASS DISPATCH: open-play / kickoff / FK / corner / throw / GK ----------
    for sp, body, outcome, ball in [
        ("O", "R", "C", -10.0),  # open-play, right-foot, complete
        ("O", "L", "F", -5.0),   # open-play, left-foot, fail
        ("K", "R", "C", 0.0),    # kickoff (treated as pass)
        ("F", "R", "C", -25.0),  # freekick_short
        ("C", "R", "C", -45.0),  # corner_short
        ("T", "H", "C", -52.0),  # throw_in (head)
        ("G", "R", "C", -50.0),  # goalkick
    ]:
        events.append(make_event(eid, pe_type="PA", period=1, time_s=10.0 + eid,
                                  set_piece=sp, ball_x=ball, ball_y=0.0,
                                  bodyType=body, passOutcomeType=outcome))
        eid += 1

    # ---------- CROSS DISPATCH: open / FK / corner ----------
    for sp, outcome, ball in [
        ("O", "C", 30.0),
        ("F", "C", -20.0),
        ("C", "F", -45.0),
    ]:
        events.append(make_event(eid, pe_type="CR", period=1, time_s=20.0 + eid,
                                  set_piece=sp, ball_x=ball, ball_y=15.0,
                                  bodyType="R", crossOutcomeType=outcome))
        eid += 1

    # ---------- SHOT DISPATCH: open / FK / penalty + result mapping ----------
    for sp, outcome, ball in [
        ("O", "G", 40.0),  # goal
        ("O", "S", 35.0),  # saved
        ("O", "B", 30.0),  # blocked
        ("O", "W", 38.0),  # wide
        ("F", "S", -20.0), # free-kick shot
        ("P", "G", 41.0),  # penalty
        ("O", "O", -10.0), # own goal (defender shooting at own goal)
    ]:
        events.append(make_event(eid, pe_type="SH", period=1, time_s=30.0 + eid,
                                  set_piece=sp, ball_x=ball, ball_y=0.0,
                                  bodyType="R", shotOutcomeType=outcome))
        eid += 1

    # ---------- CLEARANCE / DRIBBLE / TOUCH-CONTROL ----------
    events.append(make_event(eid, pe_type="CL", period=1, time_s=50.0,
                              ball_x=-30.0, ball_y=0.0, bodyType="H"))
    eid += 1
    for outcome in ["R", "L"]:  # retained, lost
        events.append(make_event(eid, pe_type="BC", period=1, time_s=51.0 + eid,
                                  ball_x=10.0, ball_y=0.0, bodyType="R",
                                  ballCarryOutcome=outcome))
        eid += 1
    events.append(make_event(eid, pe_type="TC", period=1, time_s=53.0,
                              ball_x=15.0, ball_y=0.0, bodyType="O"))
    eid += 1

    # ---------- CHALLENGE: tackle winner = challenger AND tackle winner = carrier ----------
    # Carrier=1 (home), challenger=12 (away), winner = challenger → carrier loses
    events.append(make_event(eid, pe_type="CH", period=1, time_s=55.0,
                              team_id=HOME_TEAM_ID, player_id=1, ball_x=20.0, ball_y=0.0,
                              bodyType="R",
                              challengerPlayerId=12, challengerPlayerName="Player 12",
                              challengeWinnerPlayerId=12,
                              challengeWinnerPlayerName="Player 12"))
    eid += 1
    # Carrier=2 (home), challenger=13 (away), winner = carrier → challenger loses
    events.append(make_event(eid, pe_type="CH", period=1, time_s=56.0,
                              team_id=HOME_TEAM_ID, player_id=2, ball_x=22.0, ball_y=0.0,
                              bodyType="R",
                              challengerPlayerId=13, challengerPlayerName="Player 13",
                              challengeWinnerPlayerId=2,
                              challengeWinnerPlayerName="Player 2"))
    eid += 1

    # ---------- REBOUND: keeper save (default) and keeper pick-up ----------
    events.append(make_event(eid, pe_type="RE", period=1, time_s=60.0,
                              team_id=AWAY_TEAM_ID, player_id=12, home_team=False,
                              ball_x=-45.0, ball_y=0.0, bodyType="O",
                              keeperTouchType="P"))  # parry → keeper_save
    eid += 1
    events.append(make_event(eid, pe_type="RE", period=1, time_s=61.0,
                              team_id=AWAY_TEAM_ID, player_id=12, home_team=False,
                              ball_x=-45.0, ball_y=0.0, bodyType="O",
                              keeperTouchType="C"))  # catch → keeper_pick_up
    eid += 1

    # ---------- FOULS: yellow card + red card + plain ----------
    e = make_event(eid, pe_type="PA", period=1, time_s=65.0, ball_x=0.0, ball_y=0.0,
                    bodyType="R", passOutcomeType="C")
    e["fouls"] = [{
        **_empty_foul()[0],
        "foulType": "STANDARD",
        "onFieldCulpritPlayerId": 12, "onFieldCulpritPlayerName": "Player 12",
        "victimPlayerId": 1, "victimPlayerName": "Player 1",
        "finalFoulOutcomeType": "Y",
    }]
    events.append(e); eid += 1

    e = make_event(eid, pe_type="PA", period=1, time_s=66.0, ball_x=10.0, ball_y=0.0,
                    bodyType="R", passOutcomeType="C")
    e["fouls"] = [{**_empty_foul()[0], "foulType": "STANDARD",
                    "finalFoulOutcomeType": "R"}]
    events.append(e); eid += 1

    # ---------- EXCLUDED EVENTS (drives excluded_counts non-trivially) ----------
    events.append(make_event(eid, ge_type="OUT", pe_type=None, period=1,
                             time_s=70.0, ball_x=-52.0, ball_y=20.0))
    eid += 1
    events.append(make_event(eid, ge_type="SUB", pe_type=None, period=1, time_s=72.0))
    eid += 1
    events.append(make_event(eid, ge_type="OTB", pe_type="IT",
                             period=1, time_s=73.0, ball_x=10.0, ball_y=0.0))
    eid += 1

    # ---------- PERIOD 2: KICKOFF + small set of events to exercise direction flip ----------
    events.append(make_event(eid, ge_type="SECONDKICKOFF", pe_type=None,
                             period=2, time_s=0.0, ball_x=0.0, ball_y=0.0))
    eid += 1
    # Repeat the dispatch coverage minimally in period 2 to verify cross-period stability.
    for sp, body, outcome, ball in [
        ("O", "R", "C", 10.0),
        ("O", "L", "F", -5.0),
    ]:
        events.append(make_event(eid, pe_type="PA", period=2, time_s=10.0 + eid,
                                  set_piece=sp, ball_x=ball, ball_y=0.0,
                                  bodyType=body, passOutcomeType=outcome))
        eid += 1
    events.append(make_event(eid, pe_type="SH", period=2, time_s=30.0,
                              set_piece="O", ball_x=42.0, ball_y=0.0,
                              bodyType="R", shotOutcomeType="G"))
    eid += 1

    # ---------- END markers ----------
    events.append(make_event(eid, ge_type="END", pe_type=None, period=2,
                             time_s=2700.0, ball_x=0.0, ball_y=0.0))
    eid += 1

    return events


def main():
    events = build_match()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(events)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 19.3: Run the generator**

Run:
```bash
uv run python tests/datasets/pff/_generate_synthetic_match.py
```
Expected: prints `Wrote N events to .../synthetic_match.json` where N is in the 30–50 range. The file `tests/datasets/pff/synthetic_match.json` exists.

- [ ] **Step 19.4: Write the dataset README**

Create `tests/datasets/pff/README.md`:

```markdown
# PFF FC test fixtures

These files support the silly-kicks PFF events-data converter test suite (`tests/spadl/test_pff.py`).

## Files

| File | Origin | License |
|---|---|---|
| `synthetic_match.json` | Hand-authored / mechanically generated by the local generator script | None (every byte is original synthetic data; no PFF-copyrighted content) |
| `_generate_synthetic_match.py` | Companion generator | Source-only; runs locally with `uv run python tests/datasets/pff/_generate_synthetic_match.py` |

## Synthetic-only policy

Real PFF FC / Gradient Sports data licensing for redistributable slices is
**pending**. Until confirmed in writing, this directory contains only synthetic
hand-authored fixtures — no real WC 2022 events. This matches the policy used
for the kloppy-vendored sportec_events.xml fixture in `tests/datasets/kloppy/`
(also synthetic).

If/when PFF licensing is confirmed for small attributed slices, real-data
fixtures may be added here as **additive** test surface; the synthetic suite
remains the canonical CI test surface either way.

## Regenerating

Run the generator:

```bash
uv run python tests/datasets/pff/_generate_synthetic_match.py
```

The output is deterministic for a given source-file checksum of the generator
script — small generator edits → small JSON edits, reviewable in a PR diff.

## Coverage

The synthetic match exercises:

- Every dispatch row from `silly_kicks/spadl/pff.py` § 4.4 (per spec) at least once.
- Every set-piece composition (kickoff, open play, corner, free kick, throw-in, goal kick, penalty).
- Every result_id mapping (success, fail, owngoal, yellow_card, red_card).
- Every body_type mapping (L, R, H, O, null).
- All exclusion classes (OUT, SUB, FIRSTKICKOFF, SECONDKICKOFF, END, OTB+IT).
- Both periods, with realistic time progression.
- Tackle winner ≠ challenger AND tackle winner = carrier — both cases.
- Dribble outcomes: retained (R) and lost (L).

## Wheel-exclusion

This directory is excluded from the published `silly-kicks` wheel via
`[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in
`pyproject.toml` (which packages only the `silly_kicks/` source tree).
```

---

## Phase 9 — Synthetic-match e2e

### Task 20: e2e: load synthetic JSON, run convert_to_actions, assert report contents

**Files:**
- Modify: `tests/spadl/test_pff.py`

- [ ] **Step 20.1: Add a helper to load and flatten the synthetic JSON**

Append to `tests/spadl/test_pff.py`, near the top (after imports, before `_df_minimal_pass`):

```python
from pathlib import Path
import json

_SYNTHETIC_FIXTURE = Path(__file__).parent.parent / "datasets" / "pff" / "synthetic_match.json"


def _load_synthetic_events() -> pd.DataFrame:
    """Load the synthetic match JSON and flatten into the EXPECTED_INPUT_COLUMNS shape.

    This is the exact reference helper documented in the converter docstring —
    callers building events DataFrames from raw PFF JSON follow the same shape.
    """
    with _SYNTHETIC_FIXTURE.open("r", encoding="utf-8") as f:
        events_json = json.load(f)

    # ---- json_normalize pulls ge.* and pe.* fields flat ----
    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        fouls = (ev.get("fouls") or [{}])
        f0 = fouls[0] if fouls else {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}

        rows.append({
            "game_id":               ev["gameId"],
            "event_id":              ev["gameEventId"],
            "possession_event_id":   ev.get("possessionEventId"),
            "period_id":             ge.get("period"),
            "time_seconds":          ge.get("startGameClock") or 0.0,
            "team_id":               ge.get("teamId"),
            "player_id":             ge.get("playerId"),
            "game_event_type":       ge.get("gameEventType"),
            "possession_event_type": pe.get("possessionEventType"),
            "set_piece_type":        ge.get("setpieceType"),
            "ball_x":                ball.get("x"),
            "ball_y":                ball.get("y"),
            "body_type":             pe.get("bodyType"),
            "ball_height_type":      pe.get("ballHeightType"),
            "pass_outcome_type":     pe.get("passOutcomeType"),
            "pass_type":             pe.get("passType"),
            "incompletion_reason_type": pe.get("incompletionReasonType"),
            "cross_outcome_type":    pe.get("crossOutcomeType"),
            "cross_type":            pe.get("crossType"),
            "cross_zone_type":       pe.get("crossZoneType"),
            "shot_outcome_type":     pe.get("shotOutcomeType"),
            "shot_type":             pe.get("shotType"),
            "shot_nature_type":      pe.get("shotNatureType"),
            "shot_initial_height_type": pe.get("shotInitialHeightType"),
            "save_height_type":      pe.get("saveHeightType"),
            "save_rebound_type":     pe.get("saveReboundType"),
            "carry_type":            pe.get("carryType"),
            "ball_carry_outcome":    pe.get("ballCarryOutcome"),
            "carry_intent":          pe.get("carryIntent"),
            "carry_defender_player_id": pe.get("carryDefenderPlayerId"),
            "challenge_type":        pe.get("challengeType"),
            "challenge_outcome_type": pe.get("challengeOutcomeType"),
            "challenger_player_id":  pe.get("challengerPlayerId"),
            "challenger_team_id":    None,  # synthetic data: we don't carry team_id on challenger; fill below
            "challenge_winner_player_id": pe.get("challengeWinnerPlayerId"),
            "challenge_winner_team_id": None,  # filled by roster join below
            "tackle_attempt_type":   pe.get("tackleAttemptType"),
            "clearance_outcome_type": pe.get("clearanceOutcomeType"),
            "rebound_outcome_type":  pe.get("reboundOutcomeType"),
            "keeper_touch_type":     pe.get("keeperTouchType"),
            "touch_outcome_type":    pe.get("touchOutcomeType"),
            "touch_type":            pe.get("touchType"),
            "foul_type":             f0.get("foulType"),
            "on_field_offense_type": f0.get("onFieldOffenseType"),
            "final_offense_type":    f0.get("finalOffenseType"),
            "on_field_foul_outcome_type": f0.get("onFieldFoulOutcomeType"),
            "final_foul_outcome_type": f0.get("finalFoulOutcomeType"),
        })
    df = pd.DataFrame(rows)

    # ---- Roster join for challenge team affiliations (synthetic teams: 1-11=100, 12-22=200) ----
    def _team_for(pid):
        if pd.isna(pid): return pd.NA
        return 100 if 1 <= int(pid) <= 11 else 200

    df["challenger_team_id"] = df["challenger_player_id"].map(_team_for)
    df["challenge_winner_team_id"] = df["challenge_winner_player_id"].map(_team_for)

    # ---- Cast nullable Int64 columns the converter expects ----
    for col in ("possession_event_id", "player_id", "carry_defender_player_id",
                "challenger_player_id", "challenger_team_id",
                "challenge_winner_player_id", "challenge_winner_team_id"):
        df[col] = df[col].astype("Int64")
    df["game_id"] = df["game_id"].astype("int64")
    df["event_id"] = df["event_id"].astype("int64")
    df["period_id"] = df["period_id"].astype("int64")
    df["team_id"] = df["team_id"].astype("int64")
    df["time_seconds"] = df["time_seconds"].astype("float64")
    df["ball_x"] = df["ball_x"].astype("float64")
    df["ball_y"] = df["ball_y"].astype("float64")
    return df
```

- [ ] **Step 20.2: Write the synthetic-match e2e tests**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffSyntheticMatchE2E:
    """End-to-end conversion against the committed synthetic match fixture."""

    def test_synthetic_match_converts_with_no_unrecognized(self):
        events = _load_synthetic_events()
        actions, report = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        assert report.has_unrecognized is False, (
            f"Unexpected unrecognized vocabulary: {report.unrecognized_counts}"
        )
        assert report.total_actions > 20  # comfortably above ge_envelope minimum

    def test_synthetic_match_dispatch_coverage(self):
        """Every documented dispatch row produces at least one action."""
        events = _load_synthetic_events()
        actions, report = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        expected_action_types = {
            "pass", "freekick_short", "corner_short", "throw_in", "goalkick",
            "cross", "freekick_crossed", "corner_crossed",
            "shot", "shot_freekick", "shot_penalty",
            "clearance", "dribble", "tackle", "bad_touch",
            "keeper_save", "keeper_pick_up",
            "foul",
        }
        produced = set(report.mapped_counts.keys())
        missing = expected_action_types - produced
        assert not missing, f"Synthetic match missing dispatch coverage: {missing}"

    def test_synthetic_match_excluded_counts_non_trivial(self):
        events = _load_synthetic_events()
        _, report = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        assert report.excluded_counts.get("OUT") == 1
        assert report.excluded_counts.get("SUB") == 1
        assert report.excluded_counts.get("FIRSTKICKOFF") == 1
        assert report.excluded_counts.get("SECONDKICKOFF") == 1
        assert report.excluded_counts.get("END") == 1
        assert report.excluded_counts.get("OTB+IT") == 1

    def test_synthetic_match_yields_one_goal_action(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        from silly_kicks.spadl import config as spc
        is_shot = actions["type_id"].isin([spc.actiontype_id["shot"],
                                            spc.actiontype_id["shot_penalty"],
                                            spc.actiontype_id["shot_freekick"]])
        is_goal = actions["result_id"] == spc.result_id["success"]
        assert int((is_shot & is_goal).sum()) >= 2  # at least open-play + penalty goals

    def test_synthetic_match_yields_yellow_and_red_cards(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        from silly_kicks.spadl import config as spc
        foul_actions = actions[actions["type_id"] == spc.actiontype_id["foul"]]
        assert (foul_actions["result_id"] == spc.result_id["yellow_card"]).any()
        assert (foul_actions["result_id"] == spc.result_id["red_card"]).any()

    def test_synthetic_match_tackle_winner_columns_populated(self):
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        from silly_kicks.spadl import config as spc
        tackles = actions[actions["type_id"] == spc.actiontype_id["tackle"]]
        assert len(tackles) >= 2
        # At least one tackle should have winner != event-row's player_id
        winners_diff_from_actor = (tackles["tackle_winner_player_id"] != tackles["player_id"]).any()
        winners_eq_actor = (tackles["tackle_winner_player_id"] == tackles["player_id"]).any()
        assert winners_diff_from_actor and winners_eq_actor
```

- [ ] **Step 20.3: Run the e2e tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffSyntheticMatchE2E -v`
Expected: all 6 tests pass.

If any test fails, iterate: usually the synthetic match needs another event or two of the missing dispatch class. Edit `_generate_synthetic_match.py`, regenerate, re-run.

### Task 21: Atomic-SPADL composability test

**Files:**
- Modify: `tests/spadl/test_pff.py`

- [ ] **Step 21.1: Write the atomic composability test**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffAtomicComposability:
    """PFF SPADL output composes cleanly with Atomic-SPADL."""

    def test_atomic_conversion_runs_without_error(self):
        from silly_kicks import atomic
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        # convert_to_atomic_spadl is the canonical entry point
        atomic_actions = atomic.spadl.convert_to_atomic_spadl(actions)
        assert len(atomic_actions) > 0
        # Atomic SPADL columns should include type_id, period_id, x, y, etc.
        for col in ("game_id", "period_id", "time_seconds", "team_id",
                    "player_id", "type_id"):
            assert col in atomic_actions.columns

    def test_atomic_add_possessions_runs(self):
        from silly_kicks import atomic
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        atomic_actions = atomic.spadl.convert_to_atomic_spadl(actions)
        with_poss = atomic.spadl.add_possessions(atomic_actions)
        assert "possession_id" in with_poss.columns
```

- [ ] **Step 21.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffAtomicComposability -v`
Expected: pass.

If `convert_to_atomic_spadl` is named differently, check `silly_kicks/atomic/__init__.py` and update — function name resolution at implementation time. Do not stop here on naming differences; resolve and move on.

### Task 22: VAEP composability test

**Files:**
- Modify: `tests/spadl/test_pff.py`

- [ ] **Step 22.1: Write VAEP composability test**

Append to `tests/spadl/test_pff.py`:

```python
class TestPffVaepComposability:
    """PFF SPADL output composes cleanly with VAEP labels."""

    def test_vaep_labels_scores(self):
        from silly_kicks import vaep
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        labels = vaep.labels.scores(actions, nr_actions=5)
        assert len(labels) == len(actions)
        # At least one row should have scores=True (synthetic match has goals)
        assert int(labels["scores"].sum()) > 0

    def test_vaep_labels_concedes_runs(self):
        from silly_kicks import vaep
        events = _load_synthetic_events()
        actions, _ = pff_mod.convert_to_actions(
            events, home_team_id=100, home_team_start_left=True,
        )
        labels = vaep.labels.concedes(actions, nr_actions=5)
        assert len(labels) == len(actions)
```

- [ ] **Step 22.2: Run tests**

Run: `uv run pytest tests/spadl/test_pff.py::TestPffVaepComposability -v`
Expected: pass. If VAEP signatures differ in the released codebase, inspect `silly_kicks/vaep/labels.py` and adapt.

### Task 23: Cross-provider parity test parametrize add

**Files:**
- Modify: `tests/spadl/test_cross_provider_parity.py`

- [ ] **Step 23.1: Inspect the existing cross-provider parity file**

Run: `uv run pytest tests/spadl/test_cross_provider_parity.py -v --co -q | head -30`
Expected: list of parity tests + the parametrize structure they use.

- [ ] **Step 23.2: Add PFF as a parametrize entry**

Open `tests/spadl/test_cross_provider_parity.py`. Find the parametrize `provider_data` (or similar) fixture/parametrize that lists the existing providers (statsbomb, opta, wyscout, sportec, metrica). Add a PFF entry that:

1. Loads `_load_synthetic_events()` from `test_pff.py` (import-from-sibling pattern, OR inline a small loader equivalent — pick the path that fits the existing file's structure best).
2. Calls `pff.convert_to_actions(events, home_team_id=100, home_team_start_left=True)`.
3. Returns the actions DataFrame for the parity assertions.

If the file uses a fixture-of-fixtures pattern, the addition is one new entry in the parametrize/fixture list. If it uses inline dataframes per test, the addition is symmetric to the sportec entry.

**Code template (adapt to the actual file structure):**

```python
@pytest.fixture
def pff_actions():
    """PFF actions from the synthetic match fixture."""
    from tests.spadl.test_pff import _load_synthetic_events
    from silly_kicks.spadl import pff as pff_mod
    events = _load_synthetic_events()
    actions, _ = pff_mod.convert_to_actions(
        events, home_team_id=100, home_team_start_left=True,
    )
    return actions
```

Then add `("pff", pff_actions)` (or equivalent) to the parametrize list.

- [ ] **Step 23.3: Run the parity tests**

Run: `uv run pytest tests/spadl/test_cross_provider_parity.py -v`
Expected: all parity tests pass for PFF (boundary metrics, coverage metrics, schema compliance, add_possessions cross-validation).

If a parity test fails specifically for PFF, examine the failure — it usually flags a dispatch / dtype / schema-shape divergence we need to fix in `pff.py` (NOT in the parity test).

---

## Phase 10 — Module exports + Public API Examples gate

### Task 24: Update `silly_kicks/spadl/__init__.py`

**Files:**
- Modify: `silly_kicks/spadl/__init__.py`

- [ ] **Step 24.1: Add the import + export**

In `silly_kicks/spadl/__init__.py`:

1. In `__all__`, add `"pff"` alphabetically (between `opta` and `play_left_to_right`).
2. In the imports block:

   Change:
   ```python
   from . import config, opta, statsbomb, wyscout
   ```
   to:
   ```python
   from . import config, opta, pff, statsbomb, wyscout
   ```

- [ ] **Step 24.2: Verify the public surface**

Run:
```bash
uv run python -c "from silly_kicks.spadl import pff; print(pff.convert_to_actions.__doc__[:200])"
```
Expected: prints the first 200 chars of the docstring (which includes "Convert a flattened PFF events DataFrame to SPADL actions.").

- [ ] **Step 24.3: Run the public-API Examples-coverage CI gate**

Run: `uv run pytest tests/test_public_api_examples.py -v`
Expected: PASS. (If it fails for `pff.convert_to_actions`, add an `Examples::` block to the docstring matching the canonical style — see `silly_kicks/spadl/utils.py::add_possessions` and `boundary_metrics` for canonical patterns. Task 6 already includes a basic `Examples` block; the gate may demand a slightly different format — refine accordingly.)

---

## Phase 11 — Docs example notebook

### Task 25: Create `docs/examples/pff_wc2022_walkthrough.py`

**Files:**
- Create: `docs/examples/pff_wc2022_walkthrough.py`
- Create: `docs/examples/README.md`

- [ ] **Step 25.1: Verify the docs/examples directory does not yet exist**

Run:
```bash
ls docs/examples 2>&1 || echo "(does not exist yet — create it)"
```
Expected: directory does not exist.

- [ ] **Step 25.2: Create the directory**

Run:
```bash
mkdir -p docs/examples
```

- [ ] **Step 25.3: Write the walkthrough script**

Create `docs/examples/pff_wc2022_walkthrough.py`:

```python
"""End-to-end walkthrough: PFF FC WC 2022 → SPADL → Atomic-SPADL → VAEP labels.

This script is **documentation, not test**. It runs against a user-supplied
local PFF directory and demonstrates how to use the silly-kicks public API.

Usage:
    uv run python docs/examples/pff_wc2022_walkthrough.py /path/to/PFF/WC2022/

The PFF directory is expected to contain:
- `Event Data/<match_id>.json`
- `Metadata/<match_id>.json`
- `Rosters/<match_id>.json`

This file is excluded from automated test discovery (it lives outside `tests/`)
and is not invoked by pytest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from silly_kicks import atomic, vaep
from silly_kicks.spadl import boundary_metrics, coverage_metrics, pff


# Match 10502 = NED-USA, 2022 R16. Edit to point at any match in the directory.
DEFAULT_MATCH_ID = 10502


def load_pff_events(pff_dir: Path, match_id: int) -> tuple[pd.DataFrame, dict]:
    """Load and flatten one PFF match's event data into the EXPECTED_INPUT_COLUMNS shape.

    Returns
    -------
    (events_df, metadata)
        events_df : pd.DataFrame
            Flat DataFrame matching pff.EXPECTED_INPUT_COLUMNS.
        metadata : dict
            Match metadata extracted from Metadata/<match_id>.json (used to
            populate home_team_id and home_team_start_left).
    """
    with (pff_dir / "Event Data" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        events_json = json.load(f)
    with (pff_dir / "Metadata" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        meta_list = json.load(f)
    metadata = meta_list[0] if isinstance(meta_list, list) else meta_list

    with (pff_dir / "Rosters" / f"{match_id}.json").open("r", encoding="utf-8") as f:
        roster_json = json.load(f)
    # Build player_id -> team_id index from roster
    pid_to_team = {int(r["player"]["id"]): int(r["team"]["id"]) for r in roster_json}

    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        fouls = (ev.get("fouls") or [{}])
        f0 = fouls[0] if fouls else {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}

        challenger_pid = pe.get("challengerPlayerId")
        winner_pid = pe.get("challengeWinnerPlayerId")

        rows.append({
            "game_id":               int(ev["gameId"]),
            "event_id":              int(ev["gameEventId"]),
            "possession_event_id":   ev.get("possessionEventId"),
            "period_id":             int(ge.get("period") or 0),
            "time_seconds":          float(ge.get("startGameClock") or 0.0),
            "team_id":               int(ge.get("teamId") or 0),
            "player_id":             ge.get("playerId"),
            "game_event_type":       ge.get("gameEventType"),
            "possession_event_type": pe.get("possessionEventType"),
            "set_piece_type":        ge.get("setpieceType"),
            "ball_x":                ball.get("x"),
            "ball_y":                ball.get("y"),
            "body_type":             pe.get("bodyType"),
            "ball_height_type":      pe.get("ballHeightType"),
            "pass_outcome_type":     pe.get("passOutcomeType"),
            "pass_type":             pe.get("passType"),
            "incompletion_reason_type": pe.get("incompletionReasonType"),
            "cross_outcome_type":    pe.get("crossOutcomeType"),
            "cross_type":            pe.get("crossType"),
            "cross_zone_type":       pe.get("crossZoneType"),
            "shot_outcome_type":     pe.get("shotOutcomeType"),
            "shot_type":             pe.get("shotType"),
            "shot_nature_type":      pe.get("shotNatureType"),
            "shot_initial_height_type": pe.get("shotInitialHeightType"),
            "save_height_type":      pe.get("saveHeightType"),
            "save_rebound_type":     pe.get("saveReboundType"),
            "carry_type":            pe.get("carryType"),
            "ball_carry_outcome":    pe.get("ballCarryOutcome"),
            "carry_intent":          pe.get("carryIntent"),
            "carry_defender_player_id": pe.get("carryDefenderPlayerId"),
            "challenge_type":        pe.get("challengeType"),
            "challenge_outcome_type": pe.get("challengeOutcomeType"),
            "challenger_player_id":  challenger_pid,
            "challenger_team_id":    pid_to_team.get(int(challenger_pid)) if challenger_pid is not None else None,
            "challenge_winner_player_id": winner_pid,
            "challenge_winner_team_id": pid_to_team.get(int(winner_pid)) if winner_pid is not None else None,
            "tackle_attempt_type":   pe.get("tackleAttemptType"),
            "clearance_outcome_type": pe.get("clearanceOutcomeType"),
            "rebound_outcome_type":  pe.get("reboundOutcomeType"),
            "keeper_touch_type":     pe.get("keeperTouchType"),
            "touch_outcome_type":    pe.get("touchOutcomeType"),
            "touch_type":            pe.get("touchType"),
            "foul_type":             f0.get("foulType"),
            "on_field_offense_type": f0.get("onFieldOffenseType"),
            "final_offense_type":    f0.get("finalOffenseType"),
            "on_field_foul_outcome_type": f0.get("onFieldFoulOutcomeType"),
            "final_foul_outcome_type": f0.get("finalFoulOutcomeType"),
        })
    df = pd.DataFrame(rows)

    # Cast nullable Int64 columns
    for col in ("possession_event_id", "player_id", "carry_defender_player_id",
                "challenger_player_id", "challenger_team_id",
                "challenge_winner_player_id", "challenge_winner_team_id"):
        df[col] = df[col].astype("Int64")

    return df, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pff_dir", type=Path,
                        help="Path to the PFF FC WC 2022 directory")
    parser.add_argument("--match-id", type=int, default=DEFAULT_MATCH_ID)
    args = parser.parse_args()

    print(f"Loading match {args.match_id} from {args.pff_dir}...")
    events, metadata = load_pff_events(args.pff_dir, args.match_id)
    print(f"  Loaded {len(events)} events.")

    home_team_id = int(metadata["homeTeam"]["id"])
    home_team_start_left = bool(metadata["homeTeamStartLeft"])
    home_team_start_left_extratime = (
        bool(metadata["homeTeamStartLeftExtraTime"])
        if metadata.get("homeTeamStartLeftExtraTime") is not None else None
    )

    actions, report = pff.convert_to_actions(
        events,
        home_team_id=home_team_id,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    print(f"  Converted to {report.total_actions} SPADL actions.")
    if report.has_unrecognized:
        print(f"  WARNING: unrecognized vocabulary: {report.unrecognized_counts}")
    print(f"  Action-type counts: {report.mapped_counts}")

    # Atomic-SPADL composition
    atomic_actions = atomic.spadl.convert_to_atomic_spadl(actions)
    print(f"  Atomic-SPADL: {len(atomic_actions)} atomic actions.")

    # Coverage metrics
    cov = coverage_metrics(actions, expected_action_types={"pass", "shot", "tackle", "cross"})
    print(f"  Coverage: {cov}")

    # Boundary metrics
    bm = boundary_metrics(actions)
    print(f"  Boundary: {bm}")

    # VAEP scores label
    scores = vaep.labels.scores(actions, nr_actions=10)
    print(f"  VAEP scores label: {int(scores['scores'].sum())} positive rows.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 25.4: Write the docs/examples README**

Create `docs/examples/README.md`:

```markdown
# silly-kicks examples

End-to-end usage examples for silly-kicks public API.

Each script is **documentation, not a test**. They run against user-supplied
data directories (paths passed at the command line) and demonstrate the
canonical conversion → enrichment → labels pipeline for a given provider.

## Scripts

| Script | Provider | Demonstrates |
|---|---|---|
| `pff_wc2022_walkthrough.py` | PFF FC | JSON parsing → SPADL → Atomic-SPADL → coverage / boundary metrics → VAEP labels |

## Convention for new examples

When adding a new provider walkthrough:

1. Name: `<provider>_<dataset_or_release>_walkthrough.py`.
2. Take dataset path as a CLI argument (no hard-coded paths, no env-var
   gating, no test-discovery hooks).
3. Demonstrate the full pipeline: events → SPADL → Atomic-SPADL → coverage_metrics → boundary_metrics → VAEP labels.
4. Print progress to stdout — these run interactively.
```

- [ ] **Step 25.5: Verify the script imports cleanly**

Run:
```bash
uv run python -c "import importlib.util; spec = importlib.util.spec_from_file_location('w', 'docs/examples/pff_wc2022_walkthrough.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```
Expected: prints `OK`. (We do NOT run the script with real data — that requires a PFF directory; only verify the module imports.)

---

## Phase 12 — Release artifacts

### Task 26: Bump version to 2.6.0

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 26.1: Locate the version line**

Run:
```bash
grep -n "^version" pyproject.toml
```
Expected: prints `version = "2.5.0"` (or similar — record the line number).

- [ ] **Step 26.2: Bump to 2.6.0**

Edit `pyproject.toml`: change `version = "2.5.0"` → `version = "2.6.0"`. Use the Edit tool, NOT a sed command.

- [ ] **Step 26.3: Verify**

Run:
```bash
uv run python -c "import silly_kicks; print(silly_kicks.__version__)"
```
Expected: prints `2.6.0` (assuming the package re-exports `__version__` from pyproject; if the file uses a different mechanism, follow that mechanism).

### Task 27: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 27.1: Locate the changelog**

Run:
```bash
ls CHANGELOG.md && head -30 CHANGELOG.md
```
Expected: file exists; format follows Keep-a-Changelog or similar.

- [ ] **Step 27.2: Add the 2.6.0 entry**

In `CHANGELOG.md`, immediately after the heading line and before the existing `## [2.5.0]` block, insert:

```markdown
## [2.6.0] — 2026-04-30

### Added

- New first-class PFF FC / Gradient Sports event-data converter:
  `silly_kicks.spadl.pff.convert_to_actions`. Hexagonal pure-function
  contract (events DataFrame in, SPADL DataFrame + ConversionReport out,
  zero I/O). Output schema `PFF_SPADL_COLUMNS` extends `SPADL_COLUMNS`
  with 4 nullable Int64 tackle-passthrough columns
  (`tackle_winner_player_id`, `tackle_winner_team_id`,
  `tackle_loser_player_id`, `tackle_loser_team_id`) per ADR-001.
- Per-period direction-of-play normalization parameters
  (`home_team_start_left`, `home_team_start_left_extratime`) — first
  silly-kicks converter requiring perspective-real coordinate handling.
- Public-API Examples coverage extended to include the new converter.
- `tests/datasets/pff/` directory with synthetic match fixture and
  deterministic generator.
- `docs/examples/pff_wc2022_walkthrough.py` end-to-end pipeline
  demonstration (documentation, not test).
- `TODO.md` entry for the deferred tracking-namespace expansion,
  capturing verified lakehouse prior art and library-native architectural
  rules.

### Changed

- `silly_kicks.spadl._finalize_output` recognizes pandas extension dtypes
  (e.g., `Int64`) on schema entries — small surface-area generalization,
  fully backwards-compatible with existing object/int64 dtype handling.
```

- [ ] **Step 27.3: Verify the diff is clean**

Run:
```bash
git diff CHANGELOG.md | head -40
```
Expected: shows a clean prepend, no other changes.

### Task 28: TODO.md entry

**Files:**
- Modify: `TODO.md` (or create if absent)

- [ ] **Step 28.1: Inspect current TODO.md state**

Run:
```bash
ls TODO.md && cat TODO.md
```
Expected: empty or near-empty file (per project memory, TODO.md has been empty since 2.5.0).

- [ ] **Step 28.2: Insert the tracking-namespace TODO**

Replace `TODO.md` with the full TODO content from § 6 of the spec
(`docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`) —
the section starting with `# Open Items` and ending at `### Validation plan`'s
final bullet.

(The spec contains the canonical text. The implementation plan does not
duplicate it here — read the spec § 6 and copy-paste verbatim, removing the
spec's surrounding markdown code-fence wrapper.)

- [ ] **Step 28.3: Verify**

Run:
```bash
head -20 TODO.md
```
Expected: starts with `# Open Items` and `## Tracking namespace — silly_kicks.tracking.*`.

---

## Phase 13 — Final lint / type / test pass

### Task 29: Lint

**Files:**
- (none — verification only)

- [ ] **Step 29.1: Run ruff over the modified package and tests**

Run:
```bash
uv run ruff check silly_kicks/ tests/
```
Expected: clean. If ruff flags anything in `pff.py`, fix at the source — do not add `# noqa` comments unless it's a legitimately unavoidable case (e.g., per-file-ignores already declared in pyproject for ML naming).

- [ ] **Step 29.2: Run ruff format check**

Run:
```bash
uv run ruff format --check silly_kicks/ tests/ docs/examples/
```
Expected: clean. If formatting is off, run `uv run ruff format silly_kicks/ tests/ docs/examples/` and re-run the check.

### Task 30: Type check

**Files:**
- (none — verification only)

- [ ] **Step 30.1: Run pyright on the package**

Run:
```bash
uv run pyright silly_kicks/spadl/pff.py silly_kicks/spadl/schema.py silly_kicks/spadl/__init__.py silly_kicks/spadl/utils.py
```
Expected: 0 errors. If pyright complains about Int64-related typing, refer to the existing handling in `utils.py::_finalize_output` and follow its discipline.

### Task 31: Full test suite

**Files:**
- (none — verification only)

- [ ] **Step 31.1: Run the full non-e2e test suite**

Run:
```bash
uv run pytest tests/ -m "not e2e" --tb=short -q
```
Expected: passes; new test count = baseline + ~50 (record exact delta and confirm against the target).

- [ ] **Step 31.2: Run the e2e suite (no-op; e2e is local-data, not CI)**

Run:
```bash
uv run pytest tests/ -m "e2e" --tb=short -q
```
Expected: passes the existing e2e suite (unchanged by this PR).

### Task 32: Manual smoke against the synthetic match

**Files:**
- (none — verification only)

- [ ] **Step 32.1: Run the converter end-to-end on the synthetic match and inspect output**

Run:
```bash
uv run python -c "
import json, pandas as pd
from pathlib import Path
from silly_kicks.spadl import pff

# Load synthetic
with open('tests/datasets/pff/synthetic_match.json', encoding='utf-8') as f:
    j = json.load(f)
print(f'Loaded {len(j)} synthetic events.')

# Use the test-helper loader pattern from test_pff.py — for the smoke run we
# inline a minimal version.
import sys; sys.path.insert(0, 'tests/spadl')
from test_pff import _load_synthetic_events

events = _load_synthetic_events()
actions, report = pff.convert_to_actions(events, home_team_id=100, home_team_start_left=True)
print(f'Actions: {len(actions)}')
print(f'Report: {report}')
print(f'Action-type counts: {sorted(report.mapped_counts.items())}')
print(f'Excluded: {sorted(report.excluded_counts.items())}')
print(f'Unrecognized: {report.unrecognized_counts}')
assert not report.has_unrecognized, 'Unexpected unrecognized vocabulary'
"
```
Expected: prints reasonable counts; no `unrecognized_counts` entries; >= 20 actions.

---

## Phase 14 — Single-commit finalization (after explicit user approval)

### Task 33: Commit (gated on user approval)

**Files:**
- (git operation only — no source changes)

- [ ] **Step 33.1: Capture working-tree summary**

Run:
```bash
git status --short && echo "---" && git diff --stat
```
Expected: shows the full set of created/modified files:
- new: `silly_kicks/spadl/pff.py`
- modified: `silly_kicks/spadl/schema.py`
- modified: `silly_kicks/spadl/__init__.py`
- modified: `silly_kicks/spadl/utils.py` (only if Step 5.3 was needed)
- new: `tests/spadl/test_pff.py`
- new: `tests/datasets/pff/_generate_synthetic_match.py`
- new: `tests/datasets/pff/synthetic_match.json`
- new: `tests/datasets/pff/README.md`
- modified: `tests/spadl/test_cross_provider_parity.py`
- modified: `tests/spadl/test_schema.py`
- modified: `tests/spadl/test_output_contract.py` (only if Step 5.1 added a test)
- new: `docs/examples/pff_wc2022_walkthrough.py`
- new: `docs/examples/README.md`
- new: `docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`
- new: `docs/superpowers/plans/2026-04-30-pff-fc-events-converter.md`
- modified: `pyproject.toml` (version bump)
- modified: `CHANGELOG.md`
- modified: `TODO.md`

- [ ] **Step 33.2: Hand to user for explicit commit approval**

**STOP here.** Per repo policy (CLAUDE.md memory: "literally ONE commit per branch; no WIP commits + squash; explicit approval before that one commit"), implementation does NOT commit autonomously. Present the summary to the user; await explicit "commit it" approval; then proceed to Step 33.3.

- [ ] **Step 33.3 (only on explicit user approval): Stage and commit**

Run:
```bash
git add silly_kicks/spadl/pff.py silly_kicks/spadl/schema.py silly_kicks/spadl/__init__.py silly_kicks/spadl/utils.py
git add tests/spadl/test_pff.py tests/spadl/test_cross_provider_parity.py tests/spadl/test_schema.py tests/spadl/test_output_contract.py
git add tests/datasets/pff/
git add docs/examples/pff_wc2022_walkthrough.py docs/examples/README.md
git add docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md
git add docs/superpowers/plans/2026-04-30-pff-fc-events-converter.md
git add pyproject.toml CHANGELOG.md TODO.md
```

Then commit using a HEREDOC:

```bash
git commit -m "$(cat <<'EOF'
feat(spadl): add first-class PFF FC events-data converter — silly-kicks 2.6.0

New silly_kicks.spadl.pff converter: hexagonal pure-function contract
(events DataFrame in, SPADL + ConversionReport out, zero I/O), mirrors
sportec/metrica shape. Output schema PFF_SPADL_COLUMNS extends
SPADL_COLUMNS with 4 nullable Int64 tackle-passthrough columns per
ADR-001.

First silly-kicks converter requiring perspective-real coordinate handling
(PFF coordinates switch direction between periods); two new direction-of-play
parameters (home_team_start_left, home_team_start_left_extratime) carry
the metadata-derived flip information.

Tests: 30+ unit/contract + dispatch tests + 6 e2e tests against a
hand-authored synthetic match fixture (~30-50 events covering every
dispatch row from the spec § 4.4 + every set-piece composition + every
result_id + every body_type + tackle winner-vs-challenger both directions
+ exclusion classes). PFF licensing for redistributable real-data slices
remains pending; synthetic-only is the canonical CI surface.

Tracking-namespace expansion deferred to TODO.md with verified lakehouse
prior art (3 providers / 20 matches / ~38M player-frames in
soccer_analytics.dev_gold.fct_tracking_frames as of 2026-04-30) and
library-native architectural rules.

Spec: docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md
Plan: docs/superpowers/plans/2026-04-30-pff-fc-events-converter.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 33.4: Verify the commit**

Run:
```bash
git log -1 --stat
```
Expected: shows the commit at HEAD with the full file set from Step 33.1.

- [ ] **Step 33.5: Hand back to user for PR approval**

**Do not push or create a PR autonomously.** Per repo policy, the user controls when to push and when to open the PR. Present the commit summary; await explicit "push and open PR" approval.
