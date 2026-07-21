# Vector-Reflection Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make coordinate reflection transform every column according to its declared kind, so a point reflection can no longer leave velocities, smoothed positions, or direction labels silently untransformed.

**Architecture:** One new public module `silly_kicks/reflection.py` provides `reflect()` (registry-driven) for schema-bearing tables and `reflect_columns()` (explicit, kind-aware) for derived feature columns. Every reflection call site migrates onto them — see spec §4.6, which gives a precise breakdown (eleven places apply a reflection; two more are defective by omission) rather than a headline count, because the inventory has been wrong three times. Guards are per-row invariance tests with non-vacuity partners, plus a registry-completeness meta-assertion — which is where fail-closed lives, per spec §4.5. `reflect()` does **not** raise on undeclared columns at runtime; it treats them as `invariant` and warns only on a geometry-shaped name.

**Tech Stack:** Python 3.10-3.14, pandas, numpy. No new dependencies. Plain-dict schemas (no pandera, per house convention).

**Spec:** `docs/superpowers/specs/2026-07-19-vector-reflection-consistency-design.md`

**Target:** 4.55.0 / PR-S122 / ADR-045

---

## Background the implementer needs

Read the spec first. The three facts that matter most:

1. **A 180° point reflection is `x -> 105-x` AND `y -> 68-y`** (ADR-028). It negates vectors
   (`vx -> -vx`), leaves magnitudes alone (`speed`), and swaps direction labels (`ltr <-> rtl`).
2. **`vx`/`vy`/`x_smoothed`/`y_smoothed` are NOT in `TRACKING_FRAMES_COLUMNS`**
   (`silly_kicks/tracking/schema.py:9-32`). They are added later by `preprocess`. This is why
   every reflection site forgot them.
3. **Guards must be per-row, never aggregate.** Defect D2's mean bias is only -1.1% (rows
   overstate and understate in near-equal measure) and D3's is -0.002. A mean-comparison test
   passes cleanly on broken code.

### Before you paste any "before" block

**The quoted before-blocks in Tasks 9, 9b and 9c have their trailing
`# type: ignore[reportAttributeAccessIssue]` comments STRIPPED**, despite one of them carrying
a "verified verbatim" assurance. The real lines are, e.g. `spadl/utils.py:1546`:

```python
        ltr_actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values  # type: ignore[reportAttributeAccessIssue]
```

An exact-match edit against the stripped text fails on all four sites. Read the current line
before replacing it, and do not trust a quoted block in this document to be byte-accurate —
the fifth review pass found anchor slips as well (`_kernels.py:696`→695, `sportec.py:157`→156,
`_shape_graph.py:931-932`→932-933). Navigate by symbol name, not by line number.

### Branch setup

```bash
git checkout main && git pull
git checkout -b pr-s121-vector-reflection
```

Do NOT use a worktree (house convention). One commit per task is fine locally; the branch is
squashed on merge.

---

## File structure

| File | Responsibility |
|---|---|
| `silly_kicks/reflection.py` | **new.** `ReflectionKind`, the two registries, `reflect()`, `reflect_columns()`. The single source of truth for what a reflection does to a column. |
| `silly_kicks/tracking/_action_orientation.py` | `reproject_to_action_ltr` reimplemented over `reflect_columns`, gaining vector params. |
| `silly_kicks/tracking/utils.py` | `_resolve_action_frame_context._reproject_rows` passes vectors (**D1 fix**); `play_left_to_right` routes through `reflect()`. |
| `silly_kicks/tracking/features.py` | `_build_ball_xy_v_per_action` re-projects the ball (**D2 fix**). |
| `silly_kicks/tracking/direction.py` | both reflection legs route through `reflect()`. |
| `silly_kicks/spadl/utils.py` | `play_left_to_right` routes through `reflect()` with the SPADL registry. |
| `silly_kicks/spadl/orientation.py` | **D8, added in the fifth review pass.** `_mirror_absolute_frame` and `_mirror_per_period` — the two branches of public `to_spadl_ltr`, called by nine converters — route through `reflect()`. |
| `tests/test_reflection.py` | **new.** Unit tests for the primitive + registry completeness meta-assertion. |
| `tests/tracking/test_point_reflection_invariance.py` | **new.** Per-row invariance across every site, each with a non-vacuity partner. |

---

## Task 1: The `reflection` module — kinds and registries

**Files:**
- Create: `silly_kicks/reflection.py`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reflection.py
"""Tests for silly_kicks.reflection (ADR-045)."""
from __future__ import annotations

import warnings  # used by Task 3's warn/silence tests

import numpy as np
import pandas as pd
import pytest

# NOTE: Tasks 2 and 3 each extend this import line as they add reflect_columns / reflect.
# Import only what exists now, or Step 4 below fails with ImportError instead of passing.
from silly_kicks.reflection import SPADL_REFLECTION_KINDS, TRACKING_REFLECTION_KINDS


def test_registries_declare_the_kinds_that_matter():
    assert TRACKING_REFLECTION_KINDS["x"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["y"] == "point_y"
    assert TRACKING_REFLECTION_KINDS["vx"] == "vector_x"
    assert TRACKING_REFLECTION_KINDS["vy"] == "vector_y"
    # speed is a MAGNITUDE -- reflecting it would be the inverse defect.
    assert TRACKING_REFLECTION_KINDS["speed"] == "magnitude"
    # z is invariant: a reflection in the pitch plane does not change height.
    assert TRACKING_REFLECTION_KINDS["z"] == "invariant"
    assert TRACKING_REFLECTION_KINDS["team_attacking_direction"] == "direction_label"
    # The columns the old API structurally could not express:
    assert TRACKING_REFLECTION_KINDS["x_smoothed"] == "point_x"
    assert TRACKING_REFLECTION_KINDS["y_smoothed"] == "point_y"
    # SPADL side, including the ADR-025 enrichment columns.
    assert SPADL_REFLECTION_KINDS["start_x"] == "point_x"
    assert SPADL_REFLECTION_KINDS["end_y"] == "point_y"
    assert SPADL_REFLECTION_KINDS["enriched_start_x"] == "point_x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.reflection'`

- [ ] **Step 3: Create the module with kinds and registries**

```python
# silly_kicks/reflection.py
"""Coordinate reflection with per-column transform semantics (ADR-045).

A 180-degree point reflection about the pitch centre acts differently on different
KINDS of quantity:

===================  ==========================  ================================
kind                 example columns             transform
===================  ==========================  ================================
``point_x``          ``x``, ``start_x``          ``x -> FIELD_LENGTH - x``
``point_y``          ``y``, ``end_y``            ``y -> FIELD_WIDTH - y``
``vector_x``         ``vx``                      negated
``vector_y``         ``vy``                      negated
``magnitude``        ``speed``                   unchanged
``direction_label``  ``team_attacking_direction``  ``"ltr" <-> "rtl"``
``invariant``        ids, timestamps, ``z``      unchanged
===================  ==========================  ================================

Before this module each reflection site enumerated an x/y column list by hand and
had **no way to express** that a column was a vector. Any column not on the list --
``vx``/``vy`` and ``x_smoothed``/``y_smoothed``, neither of which is in
``TRACKING_FRAMES_COLUMNS`` -- rode through untransformed and silently wrong.

See ADR-045 and NOTICE.
"""

from __future__ import annotations

import re
import types
import warnings
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

# STAGED DELIBERATELY. `__all__` must list only names that EXIST at this commit: ruff F822
# ("undefined name in __all__") fails otherwise, and Task 1 Step 5 is a commit that the
# verification checklist lints. Task 2 appends "reflect_columns"; Task 3 appends "reflect".
# Same reason `Sequence` is NOT imported here -- it is first used by reflect_columns in
# Task 2, and an unused import is ruff F401.
__all__ = [
    "ATOMIC_SPADL_REFLECTION_KINDS",
    "GEOMETRIC_NAME",
    "SPADL_REFLECTION_KINDS",
    "TRACKING_REFLECTION_KINDS",
    "ReflectionKind",
    "UndeclaredGeometricColumnWarning",
]


class UndeclaredGeometricColumnWarning(UserWarning):
    """A column with a geometry-shaped name reached a reflection without a declared kind.

    Its own category so a consumer can escalate JUST this to an error (ADR-045 section 4.5):

        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)

    That escalation is how a consumer who fully controls its column universe -- the lakehouse
    does -- gets fail-closed behaviour, without the library imposing it on callers whose
    universe is unbounded by construction (``preserve_native``). Follows the ADR-041
    precedent in ``tracking/_warnings.py``: separate categories so silencing a routine
    notice cannot silence genuine misuse.
    """

# NOTE: pitch dimensions are resolved from silly_kicks.spadl.config at CALL time, not imported
# at module scope. `from silly_kicks.spadl.config import field_length` would snapshot the value
# at first import, making this module the one reader that cannot see a reassignment -- every
# other site in the repo reads them late-bound by attribute access (spadl/utils.py:1546,
# atomic/spadl/utils.py:1130, atomic/vaep/features.py:166). A silent producer/consumer
# divergence introduced by the module whose purpose is to eliminate silent producer/consumer
# divergence would be a poor start. Resolving inside the call also keeps `reflection` free of
# any package-level dependency, matching the id_compat precedent that justifies its placement.


def _pitch_dims(
    field_length: float | None, field_width: float | None
) -> tuple[float, float]:
    from silly_kicks.spadl import config as spadlconfig

    return (
        spadlconfig.field_length if field_length is None else field_length,
        spadlconfig.field_width if field_width is None else field_width,
    )

ReflectionKind = Literal[
    "point_x",
    "point_y",
    "vector_x",
    "vector_y",
    "magnitude",
    "direction_label",
    "invariant",
]

_DIRECTION_SWAP = {"ltr": "rtl", "rtl": "ltr"}

# Registries are exposed as read-only mappings: a plain dict would let a consumer do
# TRACKING_REFLECTION_KINDS["x"] = "invariant" and silently disable reflection process-wide.
# The Mapping[str, ReflectionKind] parameter type already accepts MappingProxyType.

_TRACKING_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry ---
    "x": "point_x",
    "y": "point_y",
    "z": "invariant",
    # --- preprocess-added (NOT in TRACKING_FRAMES_COLUMNS -- the original blind spot) ---
    "vx": "vector_x",
    "vy": "vector_y",
    "x_smoothed": "point_x",
    "y_smoothed": "point_y",
    # invariant BY DECISION, not oversight: the tag names the preprocess config that
    # produced x_smoothed/y_smoothed, and after a reflection those outputs are mirrored
    # while the tag still claims the original config. D3b establishes that the natural
    # mitigation (re-run smoothing) silently no-ops on the tag check
    # (preprocess/_smoothing.py:100-103), so the staleness is recorded here rather than
    # papered over. Reflecting a config tag would be meaningless; renaming it is out of scope.
    "_preprocessed_with": "invariant",
    # --- magnitudes ---
    "speed": "magnitude",
    # --- labels / identity / provenance ---
    "team_attacking_direction": "direction_label",
    "game_id": "invariant",
    "period_id": "invariant",
    "frame_id": "invariant",
    "time_seconds": "invariant",
    "frame_rate": "invariant",
    "player_id": "invariant",
    "team_id": "invariant",
    "is_ball": "invariant",
    "is_goalkeeper": "invariant",
    "speed_source": "invariant",
    "ball_state": "invariant",
    "confidence": "invariant",
    "visibility": "invariant",
    "source_provider": "invariant",
    "is_goalkeeper_source": "invariant",
}
"""Transform kind per tracking-frame column. Covers TRACKING_FRAMES_COLUMNS plus the
preprocess-added columns. Completeness is CI-gated (tests/test_reflection.py)."""

_SPADL_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry ---
    "start_x": "point_x",
    "start_y": "point_y",
    "end_x": "point_x",
    "end_y": "point_y",
    # ADR-025 restart-coordinate enrichment -- these would ride through unmirrored today.
    "enriched_start_x": "point_x",
    "enriched_start_y": "point_y",
    "enriched_end_x": "point_x",
    "enriched_end_y": "point_y",
    "start_coord_source": "invariant",
    "end_coord_source": "invariant",
    "start_coord_confidence": "invariant",
    "end_coord_confidence": "invariant",
    # --- canonical identity / typing (SPADL_COLUMNS) ---
    "game_id": "invariant",
    "original_event_id": "invariant",
    "action_id": "invariant",
    "period_id": "invariant",
    "time_seconds": "invariant",
    "team_id": "invariant",
    "player_id": "invariant",
    "type_id": "invariant",
    "result_id": "invariant",
    "bodypart_id": "invariant",
    # --- add_names() output. add_names attaches these routinely, and a DECLARED column is
    #     the only one whose kind is guaranteed right, so they belong here. ---
    "type_name": "invariant",
    "result_name": "invariant",
    "bodypart_name": "invariant",
    # --- provider-variant columns, union over the four *_SPADL_COLUMNS dicts ---
    "action_provenance": "invariant",   # kloppy family
    "is_synthetic": "invariant",        # gradientsports (ADR-018 synthesized rows)
    "result_source": "invariant",       # skillcorner (ADR-024 native/inferred/stopgap)
    "tackle_winner_player_id": "invariant",  # sportec (ADR-001 qualifier-derived)
    "tackle_winner_team_id": "invariant",
    "tackle_loser_player_id": "invariant",
    "tackle_loser_team_id": "invariant",
}
"""Transform kind per SPADL action column.

32 columns: the 14 canonical, the 3 ``add_names`` outputs, the 7 provider-variant columns
(union over ``*_SPADL_COLUMNS``), and the 8 ADR-025 enrichment columns. Completeness is a CI
contract (tests/test_reflection.py), NOT a runtime one -- ``preserve_native``
(``spadl/utils.py:1651``) lets a caller attach arbitrarily-named provider fields, so the SPADL
column universe is unbounded by construction and no registry can enumerate it at runtime.
See ADR-045 section 4.5. Verified by union over ``silly_kicks.spadl.schema`` on 2026-07-19.

Every provider-variant column is an identifier or provenance token; none is geometric.
"""

_ATOMIC_SPADL_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    # --- geometry: atomic-SPADL carries a POINT plus a DISPLACEMENT VECTOR ---
    "x": "point_x",
    "y": "point_y",
    "dx": "vector_x",
    "dy": "vector_y",
    # --- identity / typing (ATOMIC_SPADL_COLUMNS + ATOMIC_SPADL_NAME_COLUMNS) ---
    "game_id": "invariant",
    "original_event_id": "invariant",
    "action_id": "invariant",
    "period_id": "invariant",
    "time_seconds": "invariant",
    "team_id": "invariant",
    "player_id": "invariant",
    "type_id": "invariant",
    "bodypart_id": "invariant",
    "type_name": "invariant",
    "bodypart_name": "invariant",
}
"""Transform kind per atomic-SPADL column (15: 13 canonical + 2 name columns).

``dx``/``dy`` are the clearest vector columns in the codebase, and
``atomic/spadl/utils.py:1129-1133`` ALREADY negates them correctly. That site is not being
fixed -- it is being migrated, so the contract lives in ONE place instead of eleven.
"""

# Freeze the three registries: a mutable export would let any consumer do
# TRACKING_REFLECTION_KINDS["x"] = "invariant" and silently disable reflection process-wide.
#
# PRIVATE dict, PUBLIC proxy, one declared type per name. Rebinding a name declared
# `dict[str, ReflectionKind]` to a MappingProxyType does NOT type-check --- measured with the
# repo's pyright: `Type "MappingProxyType[str, K]" is not assignable to declared type
# "dict[str, K]"`. A name has one declared type, so the dict literals above must be the
# PRIVATE `_`-prefixed names and these are the public ones.
TRACKING_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(
    _TRACKING_REFLECTION_KINDS
)
SPADL_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(
    _SPADL_REFLECTION_KINDS
)
ATOMIC_SPADL_REFLECTION_KINDS: Mapping[str, ReflectionKind] = types.MappingProxyType(
    _ATOMIC_SPADL_REFLECTION_KINDS
)

# The one geometric-name pattern every ADR-045 guard shares (see tests/test_reflection.py).
# Published so the conformance guards cannot drift from a private copy.
GEOMETRIC_NAME = re.compile(r"^([vd]?[xy]|[xy]_.*|.*_[xy]|.*_smoothed)$")
"""Fully-anchored, .match()-safe. Covers bare (``x``, ``vx``, ``dx``), prefix
(``x_centered``, ``x_smoothed``) and suffix (``defensive_line_x``, ``enriched_start_x``)
forms.

MEASURED LIMITS -- do NOT restate the earlier "zero misses, zero false positives" claim, which
was false against real repo columns::

    team_shape_centroid_x_attacking            False   (infix axis token)
    defending_centroid_vx                      False   (infix axis token)
    team_shape_defensive_line_height_attacking False   (an x-position with NO axis token)

Tolerable only because this pattern never DECIDES anything (ADR-045 section 4.5): library-owned
columns are covered by the registries, complete by enumeration, and this pattern only reports
on passenger columns and drives the conformance guards. Widening it to catch infix forms trades
false negatives for false positives (``max_x_velocity``), and ADR-043's lesson is that a name
heuristic must not be the enforcement mechanism."""
```

The three dict literals are PRIVATE (`_`-prefixed, annotated `dict[str, ReflectionKind]`); the
public names are the `MappingProxyType` rebinds shown above. This is not style: a name has one
declared type, and rebinding a name declared `dict[str, K]` to a `MappingProxyType` does not
type-check under the repo's pyright (measured: *"Type `MappingProxyType[str, K]` is not
assignable to declared type `dict[str, K]`"*).

- [ ] **Step 3b: Run the import-cycle gate NOW, not at the end**

```bash
python -m pytest tests/test_no_import_cycles.py -q
```
Expected: PASS. `reflection.py` will be imported by `spadl/`, `atomic/`, `tracking/` and
`vaep/`, and it resolves `spadl.config` inside a function precisely to stay dependency-free.
CLAUDE.md notes this gate exists because "the Fork-A cycle closed twice and is invisible to the
ordinary suite" — running it fifteen commits later would surface a cycle long after the commit
that introduced it.

- [ ] **Step 3c: Register the new public module in the two gates that auto-discover it**

A new public top-level module trips two anti-rot meta-assertions **on this commit**, not later.
Both were verified against the working tree:

1. `tests/test_public_api_examples.py::test_derived_surface_is_fully_accounted_for` derives the
   public surface at run time and asserts `discovered - set(_PUBLIC_MODULE_FILES)` is empty. Its
   own docstring names the precedent: *"`silly_kicks/id_compat.py` and `silly_kicks/gkdv/` both
   reached the surface this release and only a human noticing kept the registry near-honest."*
   Add `"silly_kicks/reflection.py"` to `_PUBLIC_MODULE_FILES` (`:137`).

   This gate also requires an `Examples` section on each public symbol. `reflect` and
   `reflect_columns` already carry one in Tasks 2–3; check whether the gate's symbol discovery
   reaches `GEOMETRIC_NAME`, the three registries and `UndeclaredGeometricColumnWarning`, and if
   so give each an example or defer the SYMBOL in `_EXAMPLES_DEBT` with a note. Deferring a
   whole module is deliberately not expressible.

2. `tests/test_no_import_cycles.py::_PACKAGES` (`:28`) — add `silly_kicks.reflection` so it is
   subprocess-imported standalone. The meta-assertion at `:63` (`len(_PACKAGES) >= 8`) does not
   pin membership, so a missing entry is silent; add it deliberately.

Run: `python -m pytest tests/test_public_api_examples.py tests/test_no_import_cycles.py -q`
Expected: PASS.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/reflection.py tests/test_reflection.py
git commit -m "feat(reflection): declare per-column reflection kinds (ADR-045)"
```

---

## Task 2: `reflect_columns` — the explicit, kind-aware primitive

**Files:**
- Modify: `silly_kicks/reflection.py`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the failing test**

First extend the import line at the top of `tests/test_reflection.py`:

```python
from silly_kicks.reflection import (
    SPADL_REFLECTION_KINDS,
    TRACKING_REFLECTION_KINDS,
    reflect_columns,
)
```

Then append to `tests/test_reflection.py`:

```python
def _frame():
    return pd.DataFrame(
        {
            "x": [10.0, 90.0],
            "y": [20.0, 60.0],
            "vx": [3.0, -4.0],
            "vy": [-1.0, 2.0],
            "speed": [np.hypot(3.0, 1.0), np.hypot(4.0, 2.0)],
        }
    )


def test_reflect_columns_applies_each_kind_correctly():
    df = _frame()
    mask = np.array([True, False])
    out = reflect_columns(
        df, mask, point_x=["x"], point_y=["y"], vector_x=["vx"], vector_y=["vy"]
    )
    # masked row: point reflected, vector negated
    assert out.loc[0, "x"] == pytest.approx(95.0)
    assert out.loc[0, "y"] == pytest.approx(48.0)
    assert out.loc[0, "vx"] == pytest.approx(-3.0)
    assert out.loc[0, "vy"] == pytest.approx(1.0)
    # unmasked row: untouched
    assert out.loc[1, "x"] == pytest.approx(90.0)
    assert out.loc[1, "vx"] == pytest.approx(-4.0)
    # speed is a magnitude and was not listed -> unchanged on BOTH rows
    pd.testing.assert_series_equal(out["speed"], df["speed"])


def test_reflect_columns_is_pure():
    df = _frame()
    before = df.copy(deep=True)
    out = reflect_columns(df, np.array([True, True]), point_x=["x"])
    pd.testing.assert_frame_equal(df, before)  # ADR-033: no input mutation
    assert out is not df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'reflect_columns'`

- [ ] **Step 3: Implement `reflect_columns`**

Add `import numpy as np` to the module imports (``_as_mask`` calls ``np.asarray`` at RUNTIME, so `from __future__ import annotations` does not cover it -- without this every migrated tracking call site raises `NameError: name 'np' is not defined` on its first ndarray mask). Then append to `silly_kicks/reflection.py`:

```python
def _as_mask(mask: pd.Series | np.ndarray, index: pd.Index) -> np.ndarray:
    """Align a boolean mask to ``index``.

    A duplicated index does NOT raise on ``reindex`` (measured, pandas 2.3.3) -- a
    same-shaped duplicate aligns positionally and a SUBSET source silently broadcasts a
    wrong mask. Both are worse than an error, so check explicitly rather than relying on
    reindex to complain.
    """
    if isinstance(mask, pd.Series):
        if not index.is_unique and not mask.index.equals(index):
            raise ValueError(
                "reflect: cannot align a mask to a non-unique index unless the mask carries "
                "exactly that index; reindex would silently broadcast. Pass a positional "
                "ndarray mask, or de-duplicate the index."
            )
        return mask.reindex(index, fill_value=False).to_numpy(dtype=bool)
    return np.asarray(mask, dtype=bool)


def reflect_columns(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    point_x: Sequence[str] = (),
    point_y: Sequence[str] = (),
    vector_x: Sequence[str] = (),
    vector_y: Sequence[str] = (),
    direction_label: Sequence[str] = (),
    field_length: float | None = None,
    field_width: float | None = None,
) -> pd.DataFrame:
    """Point-reflect the masked rows, transforming each column by its stated KIND.

    The explicit sibling of :func:`reflect`, for tables with no declared schema
    (computed feature outputs). The caller states what each column IS; unlisted
    columns are left alone.

    Pure: returns a new frame and never mutates ``df`` (ADR-033).

    Examples
    --------
    Reflect a position and its velocity together::

        import numpy as np, pandas as pd
        from silly_kicks.reflection import reflect_columns
        df = pd.DataFrame({"x": [10.0], "vx": [3.0]})
        out = reflect_columns(df, np.array([True]), point_x=["x"], vector_x=["vx"])
        print(float(out.loc[0, "x"]), float(out.loc[0, "vx"]))
        # 95.0 -3.0
    """
    out = df.copy()
    m = _as_mask(mask, out.index)
    if not m.any():
        return out
    fl, fw = _pitch_dims(field_length, field_width)

    for col in point_x:
        if col in out.columns:
            out.loc[m, col] = fl - out[col].to_numpy(dtype="float64")[m]
    for col in point_y:
        if col in out.columns:
            out.loc[m, col] = fw - out[col].to_numpy(dtype="float64")[m]
    for col in (*vector_x, *vector_y):
        if col in out.columns:
            out.loc[m, col] = -out[col].to_numpy(dtype="float64")[m]
    for col in direction_label:
        if col in out.columns:
            out.loc[m, col] = out.loc[m, col].map(lambda v: _DIRECTION_SWAP.get(v, v))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/reflection.py tests/test_reflection.py
git commit -m "feat(reflection): add kind-aware reflect_columns primitive"
```

---

## Task 3: `reflect` — registry-driven and fail-closed

**Files:**
- Modify: `silly_kicks/reflection.py`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the failing test**

First extend the import line at the top of `tests/test_reflection.py` to add `reflect`:

```python
from silly_kicks.reflection import (
    SPADL_REFLECTION_KINDS,
    TRACKING_REFLECTION_KINDS,
    reflect,
    reflect_columns,
)
```

Then append to `tests/test_reflection.py`:

```python
def test_reflect_uses_the_registry_and_covers_the_blind_spot_columns():
    df = pd.DataFrame(
        {
            "x": [10.0], "y": [20.0], "z": [1.5],
            "vx": [3.0], "vy": [-1.0], "speed": [np.hypot(3.0, 1.0)],
            "x_smoothed": [10.5], "y_smoothed": [20.5],
            "team_attacking_direction": ["ltr"],
            "player_id": ["p1"],
        }
    )
    out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    assert out.loc[0, "x"] == pytest.approx(95.0)
    assert out.loc[0, "vx"] == pytest.approx(-3.0)
    assert out.loc[0, "x_smoothed"] == pytest.approx(94.5)   # the D3b blind spot
    assert out.loc[0, "y_smoothed"] == pytest.approx(47.5)
    assert out.loc[0, "z"] == pytest.approx(1.5)             # height is invariant
    assert out.loc[0, "speed"] == pytest.approx(np.hypot(3.0, 1.0))  # magnitude
    assert out.loc[0, "team_attacking_direction"] == "rtl"
    assert out.loc[0, "player_id"] == "p1"


def test_reflect_warns_on_an_undeclared_GEOMETRIC_column():
    """ADR-045 section 4.5: an undeclared column is treated as invariant -- correct for a
    passenger -- but a GEOMETRY-shaped name is the suspicious case, so it warns."""
    from silly_kicks.reflection import UndeclaredGeometricColumnWarning

    df = pd.DataFrame({"x": [10.0], "mystery_x": [5.0]})
    with pytest.warns(UndeclaredGeometricColumnWarning, match="mystery_x"):
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    # Warned, but still treated as invariant -- the library must not guess a kind.
    assert out.loc[0, "mystery_x"] == pytest.approx(5.0)


def test_reflect_is_SILENT_on_an_undeclared_non_geometric_column():
    """The load-bearing half. `preserve_native` surfaces caller-chosen provider fields
    (spadl/utils.py:1651) whose names are unbounded BY CONSTRUCTION, and `invariant` is the
    CORRECT treatment for them. Warning here would be spam on a supported first-party
    feature, which is why the earlier `on_unknown="raise"` default was withdrawn."""
    df = pd.DataFrame({"x": [10.0], "possession": [7]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning fails this test
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
    assert out.loc[0, "possession"] == 7
    assert out.loc[0, "x"] == pytest.approx(95.0)


def test_reflect_escalates_to_an_error_via_the_warning_filter():
    """How a consumer that DOES control its column universe gets fail-closed."""
    from silly_kicks.reflection import UndeclaredGeometricColumnWarning

    df = pd.DataFrame({"x": [10.0], "mystery_x": [5.0]})
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)
        with pytest.raises(UndeclaredGeometricColumnWarning, match="mystery_x"):
            reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)


def test_reflect_on_unknown_raise_is_available_explicitly():
    """Retained as a greppable per-call opt-in; nothing in silly-kicks passes it."""
    df = pd.DataFrame({"x": [10.0], "possession": [7]})
    with pytest.raises(ValueError, match="possession"):
        reflect(
            df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS, on_unknown="raise"
        )


def test_reflect_extra_kinds_is_the_documented_escape_hatch():
    df = pd.DataFrame({"x": [10.0], "custom_vx": [2.0]})
    out = reflect(
        df,
        np.array([True]),
        kinds=TRACKING_REFLECTION_KINDS,
        extra_kinds={"custom_vx": "vector_x"},
    )
    assert out.loc[0, "custom_vx"] == pytest.approx(-2.0)


def test_extra_kinds_is_add_only_and_may_not_override_the_registry():
    """A call site must not be able to locally redefine a column's semantics."""
    df = pd.DataFrame({"x": [10.0]})
    with pytest.raises(ValueError, match="may not override"):
        reflect(
            df,
            np.array([True]),
            kinds=TRACKING_REFLECTION_KINDS,
            extra_kinds={"x": "invariant"},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'reflect'`

- [ ] **Step 3: Implement `reflect`**

Append to `silly_kicks/reflection.py`:

```python
def reflect(
    df: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    kinds: Mapping[str, ReflectionKind],
    extra_kinds: Mapping[str, ReflectionKind] | None = None,
    on_unknown: Literal["warn", "raise", "ignore"] = "warn",
    field_length: float | None = None,
    field_width: float | None = None,
) -> pd.DataFrame:
    """Point-reflect the masked rows of a schema-bearing table, by declared kind.

    An undeclared column is treated as ``invariant`` -- the correct treatment for a
    caller-attached passenger column -- and WARNS
    (:class:`UndeclaredGeometricColumnWarning`) only if its name is geometry-shaped
    (:data:`GEOMETRIC_NAME`). Supply ``extra_kinds`` to declare one properly.

    Fail-closed lives in the CI registry-completeness meta-assertion, not here
    (ADR-045 section 4.5). Three reasons, and the first is decisive: ``to_spadl_ltr``
    calls this from INSIDE nine converters on a frame already carrying the caller's
    ``preserve_native`` columns, with no reachable ``extra_kinds``, so raising there
    has no remedy. Second, ALL EIGHT catalogued ADR-045 defects were library-owned
    columns that the meta-assertion catches and a runtime raise adds nothing to.
    Third, a per-site policy split would recreate D3 -- two same-named orienters with
    divergent contracts -- one layer up.

    A consumer that fully controls its column universe gets fail-closed with::

        warnings.filterwarnings("error", category=UndeclaredGeometricColumnWarning)

    ``on_unknown="raise"`` is retained as a greppable per-call opt-in (it raises on
    ANY undeclared column, geometric or not); ``"ignore"`` silences the check
    entirely. No silly-kicks code path passes either.

    Pure: returns a new frame and never mutates ``df`` (ADR-033).

    Examples
    --------
    Reflect tracking frames, velocities included::

        import numpy as np, pandas as pd
        from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, reflect
        df = pd.DataFrame({"x": [10.0], "vx": [3.0]})
        out = reflect(df, np.array([True]), kinds=TRACKING_REFLECTION_KINDS)
        print(float(out.loc[0, "x"]), float(out.loc[0, "vx"]))
        # 95.0 -3.0
    """
    # extra_kinds is ADD-ONLY: it declares columns the registry does not know. Silently
    # overriding a registry declaration would let a call site redefine semantics locally,
    # which is how divergent conventions (ADR-045 D3) start.
    extra = dict(extra_kinds or {})
    collisions = sorted(set(extra) & set(kinds))
    if collisions:
        raise ValueError(
            f"reflect: extra_kinds may not override registry declarations; collision(s) "
            f"{collisions}. extra_kinds is for columns the registry does not know."
        )
    # `dict(kinds)` + `.update`, NOT `{**kinds, **extra}`. Measured under the repo's pyright
    # (1.1.409, 2026-07-20): the `{**mapping}` spread WIDENS the value type
    # `ReflectionKind` (a Literal) to `str`, so the annotated assignment raises
    # `reportAssignmentType` ("dict[str, str] is not assignable to dict[str, ReflectionKind]").
    # `dict()` of a `Mapping[str, ReflectionKind]` preserves the value type, and `.update` of a
    # dict does not re-widen the declared type -> zero pyright errors, no cast, no ignore.
    # (`extra` is already `dict[str, ReflectionKind]` from the line above; collisions have
    # raised, so update order is irrelevant.)
    resolved: dict[str, ReflectionKind] = dict(kinds)
    resolved.update(extra)

    undeclared = [c for c in df.columns if c not in resolved]
    if undeclared and on_unknown == "raise":
        raise ValueError(
            f"reflect: undeclared column(s) {sorted(undeclared)}. Every column must declare a "
            f"reflection kind so it cannot be silently left untransformed (ADR-045). Add it to "
            f"the registry, or pass extra_kinds={{'<col>': '<kind>'}}."
        )
    if undeclared and on_unknown == "warn":
        # Gate on the NAME, not on mere absence. A passenger column is legitimately
        # undeclared and `invariant` is the right answer for it, so warning on every
        # unknown would be noise on `preserve_native`'s supported output -- and noise is
        # how a real signal gets filtered away. Only a geometry-shaped name is suspicious.
        suspicious = sorted(c for c in undeclared if GEOMETRIC_NAME.match(c))
        if suspicious:
            warnings.warn(
                f"reflect: undeclared column(s) {suspicious} have geometry-shaped names but no "
                f"declared reflection kind, so they were left UNTRANSFORMED. If any is a "
                f"coordinate or a vector this is the ADR-045 defect class. Declare it in the "
                f"registry, or pass extra_kinds={{'<col>': '<kind>'}}. To make this an error: "
                f"warnings.filterwarnings('error', "
                f"category=UndeclaredGeometricColumnWarning).",
                UndeclaredGeometricColumnWarning,
                stacklevel=2,
            )

    buckets: dict[str, list[str]] = {
        "point_x": [], "point_y": [], "vector_x": [], "vector_y": [], "direction_label": []
    }
    for col in df.columns:
        kind = resolved.get(col)
        # `kind is not None` is required: `kind in buckets` does NOT narrow the Optional for
        # pyright, and `buckets[kind]` then errors on "None not assignable to str".
        if kind is not None and kind in buckets:
            buckets[kind].append(col)

    return reflect_columns(
        df,
        mask,
        point_x=buckets["point_x"],
        point_y=buckets["point_y"],
        vector_x=buckets["vector_x"],
        vector_y=buckets["vector_y"],
        direction_label=buckets["direction_label"],
        field_length=field_length,
        field_width=field_width,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/reflection.py tests/test_reflection.py
git commit -m "feat(reflection): add fail-closed registry-driven reflect()"
```

---

## Task 4: Registry-completeness meta-assertion

Prevents rot: a newly added column must declare a kind or CI fails. Same idiom as
`PURITY_ENTRIES` / `PUBLIC_ID_SCALAR_ENTRIES`.

**Files:**
- Modify: `tests/test_reflection.py`

- [ ] **Step 1: Write the completeness gate** (NOT a red-green step -- Task 1 pre-declares
      every column, so this is expected to pass immediately; it is the anti-rot gate, and the
      kind-plausibility gate in Step 4 is what actually has teeth)

Append to `tests/test_reflection.py`:

```python
def test_meta_every_known_tracking_column_declares_a_kind():
    """A new frame column must declare a reflection kind or this fails. Anti-rot."""
    from silly_kicks.tracking.schema import (
        GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
        KLOPPY_TRACKING_FRAMES_COLUMNS,
        TRACKING_FRAMES_COLUMNS,
    )

    known = (
        set(TRACKING_FRAMES_COLUMNS)
        | set(KLOPPY_TRACKING_FRAMES_COLUMNS)
        | set(GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS)
        # preprocess-added -- the columns that caused ADR-045 in the first place
        | {"vx", "vy", "x_smoothed", "y_smoothed", "_preprocessed_with"}
    )
    missing = sorted(known - set(TRACKING_REFLECTION_KINDS))
    assert not missing, (
        f"columns without a declared reflection kind: {missing}. Add them to "
        f"TRACKING_REFLECTION_KINDS (ADR-045) -- an undeclared column is exactly how "
        f"vx/vy went untransformed."
    )


def test_meta_every_known_spadl_column_declares_a_kind():
    """Built by UNION over the real constants, not a hardcoded list.

    This is the PRIMARY fail-closed mechanism in ADR-045 (section 4.5): runtime raising was
    withdrawn, so a column that is not declared here rides through as invariant with at most
    a warning. CI is therefore the only place that catches an undeclared LIBRARY-owned
    column -- and it catches it when the column is ADDED to a schema, not on the first
    production run that happens to reflect it. A hardcoded literal would stay green when a
    fifth provider variant is added -- exactly the rot this guards.
    """
    from silly_kicks.spadl import schema as S

    provider_variants = set().union(
        *(set(getattr(S, n)) for n in dir(S) if n.endswith("_SPADL_COLUMNS"))
    )
    known = (
        provider_variants                      # 14 canonical + 7 provider-specific
        | set(S.SPADL_NAME_COLUMNS)            # type_name / result_name / bodypart_name
        | {                                    # ADR-025 restart-coordinate enrichment
            "enriched_start_x", "enriched_start_y", "enriched_end_x", "enriched_end_y",
            "start_coord_source", "end_coord_source",
            "start_coord_confidence", "end_coord_confidence",
        }
    )
    missing = sorted(known - set(SPADL_REFLECTION_KINDS))
    assert not missing, (
        f"columns without a declared reflection kind: {missing}. Under on_unknown='raise' "
        f"these RAISE in production, they do not pass through."
    )
    assert len(known) == 32, f"expected the measured 32-column surface, got {len(known)}"
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_reflection.py -x -q -k meta`
Expected: PASS — Task 1 pre-declares all 32/25/15 columns. A FAILURE names a column added since this plan was written; classify it per Step 3.

- [ ] **Step 3: CLASSIFY each missing column by kind — do not default to `"invariant"`**

**Read this before adding anything.** `magnitude` and `invariant` are behaviourally identical
— both are no-ops in `reflect_columns`. So the completeness gate proves a column is *declared*;
it can never prove it is declared *correctly*. Blanket-declaring missing columns `"invariant"`
would make CI green while a geometric column rides through unreflected — **D1/D2 reconstituted,
now with a registry blessing it.**

Classify instead: identifiers, timestamps and provenance tokens are `invariant`; anything
positional is `point_x`/`point_y`; anything directional is `vector_x`/`vector_y`; scalar
distances/areas/speeds are `magnitude`. If a name reads geometric you must either declare a
geometric kind or add a justification entry in Step 4's allowlist.

- [ ] **Step 4: Add the kind-plausibility gate**

Completeness alone is insufficient (Step 3). Append to `tests/test_reflection.py`:

```python
import re

from silly_kicks.reflection import ATOMIC_SPADL_REFLECTION_KINDS

# ONE shared pattern, used by every NAME-BASED guard in this PR (Task 4 and Task 12b's three
# name-based sites). Task 12b's site 4 (_reproject_team_shape) is deliberately NOT name-based --
# GEOMETRIC_NAME cannot see its infix column names, so it is gated behaviourally instead.
#
# It is FULLY ANCHORED and applied with .match(). An earlier draft used
#     r"^v?[xy]$|_x$|_y$"
# with .match(), which anchors at position 0 -- so the `_x$` branch required the name to
# START with "_x". Measured: defensive_line_x -> False, x_centered -> False,
# enriched_start_x -> False. Two of the three conformance guards were therefore
# UNCONDITIONALLY PASSING, and the third's non-vacuity partner used the four bare names
# (x/y/vx/vy) that do match, which kept it reporting green. That is this PR's own defect
# class inside this PR's own guards -- a check that appears to run and silently does not.
#
# Verified against the real column names (2026-07-19): 20 should-match, 16 should-not,
# zero misses, zero false positives. Covers bare (x, vx, dx), prefix (x_centered,
# x_smoothed), and suffix (defensive_line_x, enriched_start_x) forms.
from silly_kicks.reflection import GEOMETRIC_NAME as _GEOMETRIC_NAME

# Columns whose NAME reads geometric but whose kind legitimately is not. Every entry needs a
# reason -- this allowlist is the visible record of a deliberate judgement.
#
# EMPTY, and measured so. An earlier draft listed the four ADR-025 provenance columns
# (start_coord_source / end_coord_source / start_coord_confidence / end_coord_confidence).
# All four fail GEOMETRIC_NAME.match -- none ends in _x/_y or begins with x_/y_ -- so they
# exempted NOTHING while reading as four considered judgements, and
# test_meta_the_plausibility_allowlist_actually_exempts_something FAILED on them. Measured
# 2026-07-20:
#
#     start_coord_source      False
#     end_coord_source        False
#     start_coord_confidence  False
#     end_coord_confidence    False
#
# Also measured: ZERO columns across the three registries are geometric-named AND declared
# invariant/magnitude, so there is nothing to exempt. An empty registry is the honest state
# and matches the STRUCTURAL_CONSTANTS precedent (CLAUDE.md records that one as "currently
# EMPTY" rather than deleting it). The three meta-tests below stay armed: they now guard
# against a FUTURE inert or stale entry rather than blessing four present ones.
_JUSTIFIED_NON_GEOMETRIC: dict[str, str] = {}


def test_meta_no_geometric_looking_column_is_declared_inert():
    """Guards the failure mode Step 3 warns about: a geometric column declared invariant.

    NOT the AST lint ADR-043 deleted. That failed because a safe and an unsafe id compare are
    the identical AST and only provenance separates them, so no syntactic rule could ever see
    the difference. Here the surface is a small enumerated dict we maintain (~72 keys), the
    exemptions are a visible allowlist with reasons, and spec 4.7.4's conformance guard already
    uses this same regex. This is the complete-by-enumeration idiom, not the heuristic lint.
    """
    for registry in (
        TRACKING_REFLECTION_KINDS,
        SPADL_REFLECTION_KINDS,
        ATOMIC_SPADL_REFLECTION_KINDS,
    ):
        for col, kind in registry.items():
            if _GEOMETRIC_NAME.match(col) and kind in {"invariant", "magnitude"}:
                assert col in _JUSTIFIED_NON_GEOMETRIC, (
                    f"{col!r} is declared {kind!r} but its name reads geometric. Declare the "
                    f"real kind, or add a justification to _JUSTIFIED_NON_GEOMETRIC."
                )


def test_meta_the_plausibility_allowlist_is_not_stale():
    """Every exemption must still correspond to a live registry entry."""
    declared = (
        set(TRACKING_REFLECTION_KINDS)
        | set(SPADL_REFLECTION_KINDS)
        | set(ATOMIC_SPADL_REFLECTION_KINDS)
    )
    stale = sorted(set(_JUSTIFIED_NON_GEOMETRIC) - declared)
    assert not stale, f"exemptions for columns that no longer exist: {stale}"


def test_meta_the_plausibility_allowlist_actually_exempts_something():
    """An exemption for a name the pattern never matches is decoration.

    Staleness alone cannot detect this: the key can exist in a registry and still be
    outside the gate's reach, in which case the allowlist entry does nothing and the
    reader is misled into thinking a judgement was made.
    """
    inert = [c for c in _JUSTIFIED_NON_GEOMETRIC if not _GEOMETRIC_NAME.match(c)]
    assert not inert, (
        f"allowlist entries that the geometric pattern never matches: {inert}. They exempt "
        f"nothing -- delete them, or fix the pattern."
    )


def test_meta_the_plausibility_gate_would_actually_reject_a_bad_declaration():
    """BOTH-SIDES partner. The gate fires on nothing in the current registries (by design --
    nothing is mis-declared), so prove it CAN fire. Without this, the anchoring bug that made
    it match nothing at all would have been invisible."""
    would_be_caught = {
        "ghost_gk_x": "invariant",       # suffix form
        "receiver_y": "magnitude",       # suffix form
        "x_smoothed": "invariant",       # prefix form -- a REAL registry column
        "dx": "invariant",               # atomic displacement vector
        "defensive_line_x": "invariant", # derived geometry
    }
    for col, _kind in would_be_caught.items():
        assert _GEOMETRIC_NAME.match(col), (
            f"{col!r} must be recognised as geometric or the gate cannot protect it"
        )
        assert col not in _JUSTIFIED_NON_GEOMETRIC

    # And the inverse: genuinely non-geometric names must NOT trip it, or the gate is noise.
    for col in ("speed", "back_n_count", "lateral_width", "team_id", "_preprocessed_with"):
        assert not _GEOMETRIC_NAME.match(col), f"{col!r} is not geometric; pattern too broad"
```

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: PASS. A failure names a column you declared inert that looks geometric — fix the
declaration, do not extend the allowlist without a real reason.

- [ ] **Step 5: Run the full reflection suite**

Run: `python -m pytest tests/test_reflection.py -x -q`
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/reflection.py tests/test_reflection.py
git commit -m "test(reflection): gate registry completeness against the real schemas"
```

---

## Task 5: D1 — fix `bekkers_pi` velocity re-projection (LIVE DEFECT)

**Files:**
- Modify: `silly_kicks/tracking/_action_orientation.py:180-204`
- Modify: `silly_kicks/tracking/utils.py:868-878`
- Test: `tests/tracking/test_point_reflection_invariance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_point_reflection_invariance.py
"""Per-row point-reflection invariance across every reflection site (ADR-045).

PER-ROW, never aggregate: D2's mean bias is -1.1% and D3's is -0.002 because rows
over- and under-state in near-equal measure. A mean-comparison gate passes cleanly
on broken code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import add_pressure_on_actor

FL, FW = 105.0, 68.0


def _scenario():
    actions = pd.DataFrame([{
        "game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 10.0,
        "team_id": "H", "player_id": "h1", "start_x": 60.0, "start_y": 40.0,
        "end_x": 70.0, "end_y": 40.0, "type_name": "pass", "result_name": "success",
    }])
    rows = [
        {"player_id": "h1",  "team_id": "H", "x": 60.0,  "y": 40.0, "vx": 2.0,  "vy": 0.0},
        {"player_id": "a1",  "team_id": "A", "x": 63.0,  "y": 41.0, "vx": -5.0, "vy": -1.0},
        {"player_id": "a2",  "team_id": "A", "x": 66.0,  "y": 38.0, "vx": -4.0, "vy": 1.5},
        {"player_id": "hgk", "team_id": "H", "x": 5.0,   "y": 34.0, "vx": 0.0,  "vy": 0.0},
        {"player_id": "agk", "team_id": "A", "x": 100.0, "y": 34.0, "vx": 0.0,  "vy": 0.0},
    ]
    frames = pd.DataFrame(rows)
    frames["game_id"] = 1
    frames["period_id"] = 1
    frames["frame_id"] = 250
    frames["time_seconds"] = 10.0
    frames["is_ball"] = False
    frames["source_provider"] = "snapshot"
    frames["is_goalkeeper"] = frames["player_id"].isin(["hgk", "agk"])
    frames["speed"] = np.hypot(frames["vx"], frames["vy"])
    frames["team_attacking_direction"] = np.where(frames["team_id"] == "H", "ltr", "rtl")
    ball = {
        "player_id": None, "team_id": None, "x": 60.0, "y": 40.0, "vx": 2.0, "vy": 0.0,
        "game_id": 1, "period_id": 1, "frame_id": 250, "time_seconds": 10.0,
        "is_ball": True, "is_goalkeeper": False, "speed": 2.0,
        "team_attacking_direction": None, "source_provider": "snapshot",
    }
    frames = pd.concat([frames, pd.DataFrame([ball])], ignore_index=True)
    return actions, frames


def _mirror(actions, frames, *, complete: bool):
    """Physically mirror the FRAME. Actions are already LTR so they do not change.

    complete=True  -> positions AND velocities AND labels (the true physical mirror)
    complete=False -> positions only (the historical, incomplete mirror)
    """
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    if complete:
        f["vx"] = -f["vx"]
        f["vy"] = -f["vy"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map(
        {"ltr": "rtl", "rtl": "ltr"}
    )
    return actions.copy(), f


def _pressure(a, f):
    out = add_pressure_on_actor(a, frames=f, methods=("bekkers_pi",))
    col = next(c for c in out.columns if c.startswith("pressure_on_actor__bekkers"))
    return float(out.iloc[0][col])


def test_bekkers_pressure_is_invariant_under_a_complete_physical_mirror():
    a, f = _scenario()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=True)
    assert _pressure(am, fm) == pytest.approx(base, abs=1e-6)


def test_nonvacuity_an_incomplete_mirror_would_be_caught():
    """The guard's discriminating power: a positions-only mirror MUST differ.

    Without this, the invariance test above would pass just as happily on code that
    reflects nothing at all.
    """
    a, f = _scenario()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=False)
    assert abs(_pressure(am, fm) - base) > 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`

Expected: **BOTH tests fail.** Measured on the pre-fix working tree with this exact fixture:

```
base = 0.914107   complete_mirror = 0.000000   incomplete_mirror = 0.914107
  invariance  |base - complete|   = 0.914107  -> FAIL
  non-vacuity |base - incomplete| = 0.000000  -> FAIL (vacuous)
```

The non-vacuity partner is *vacuous* pre-fix, and that is expected rather than a fixture bug:
with velocities un-re-projected, mirroring positions and then re-projecting them back returns
the base configuration exactly, so the incomplete-mirror arm is indistinguishable from base.
Both tests go green together at Step 5. Do not "fix" the non-vacuity test to pass at this step.

(An earlier draft claimed the non-vacuity test "PASSES already"; a cross-session review claimed
instead that the invariance test could not go green until Task 6 because of the un-re-projected
ball. Both were wrong — the ball term enters through a `max()` with the actor term, and since
the fixture places the ball exactly on the actor, that max is dominated by the actor leg in
both arms. Measured: after Task 5 alone, invariance = 0.000000 PASS, non-vacuity = 0.914107
PASS. Tasks 5 and 6 therefore stay separate.)

- [ ] **Step 3: Give `reproject_to_action_ltr` vector parameters**

In `silly_kicks/tracking/_action_orientation.py`, replace the body of
`reproject_to_action_ltr` (currently `:192-204`) and extend its signature:

```python
def reproject_to_action_ltr(
    df: pd.DataFrame,
    flip_mask: pd.Series,
    *,
    x_cols: list[str],
    y_cols: list[str],
    vx_cols: list[str] | None = None,
    vy_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the named columns re-projected where ``flip_mask``.

    Positions map ``x -> 105 - x`` / ``y -> 68 - y``; velocity columns are NEGATED
    (a 180-degree point reflection reverses a vector). NaN is preserved.
    ``flip_mask`` is reindexed to ``df`` (missing -> False).

    ``vx_cols``/``vy_cols`` exist because a positions-only re-projection silently
    produced velocity that contradicted its own positions (ADR-045 D1).
    """
    from silly_kicks.reflection import reflect_columns

    return reflect_columns(
        df,
        flip_mask,
        point_x=x_cols,
        point_y=y_cols,
        vector_x=vx_cols or [],
        vector_y=vy_cols or [],
    )
```

- [ ] **Step 4: Pass the velocity columns at the D1 call site**

In `silly_kicks/tracking/utils.py`, in `_reproject_rows` (currently `:874`), replace:

```python
        return reproject_to_action_ltr(rows, row_flip, x_cols=["x"], y_cols=["y"])
```

with:

```python
        # ADR-045: velocities MUST be negated alongside positions. Omitting them made
        # _pressure_bekkers read action-LTR positions against frame-convention velocity,
        # modelling away defenders as running backwards (-38.9% on away actions).
        #
        # x_smoothed/y_smoothed are enumerated too, and that is NOT belt-and-braces:
        # derive_velocities REQUIRES them (preprocess/_velocity.py:41 raises without
        # them), so every frame that carries vx/vy -- i.e. every frame where this fix
        # matters at all -- also carries the smoothed pair. Enumerating x/y/vx/vy alone
        # would leave a mirrored position sitting next to an unmirrored copy of itself,
        # which is D3b reconstituted inside D1's own fix.
        return reproject_to_action_ltr(
            rows,
            row_flip,
            x_cols=["x", "x_smoothed"],
            y_cols=["y", "y_smoothed"],
            vx_cols=["vx"],
            vy_cols=["vy"],
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`
Expected: PASS (2 passed) — measured `invariance 0.000000`, `non-vacuity 0.914107`.

- [ ] **Step 6: Expose the flip on the context for Task 6 to reuse**

`_resolve_action_frame_context` already computes `flip_by_action` (`utils.py:861-866`, dedupe
included). Task 6 must READ it rather than recompute `acting_team_attacks_rtl` independently —
two computations of the same orientation decision in two modules is the producer/consumer drift
this PR exists to remove, and it would add a second full-table groupby per call.

Add the field to `ActionFrameContext` in `silly_kicks/tracking/feature_framework.py`. It must be
**REQUIRED — no default** — which means declaring it *before* the defaulted `defending_gk_rows`:

```python
    flip_by_action: pd.Series
    defending_gk_rows: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
```

**Do not give it `default_factory=pd.Series` for style-consistency with `defending_gk_rows`.**
The two are not analogous. An empty `defending_gk_rows` means "no GK data" — a legitimate degrade
the kernels handle. An empty `flip_by_action` means "reflect nothing": `out["action_id"].map(empty)`
is all-NaN, `.fillna(False)` makes it all-False, and the ball is silently never re-projected —
**D2 restored, with no error.** Requiring the field removes that failure mode.

- [ ] **Step 6b: Update the SEVEN other constructors — this is not optional**

An earlier draft of this plan claimed the dataclass "is frozen with exactly one constructor
(`utils.py:880`), so requiring the field costs nothing". **That is false.** Measured
2026-07-20:

```
$ grep -rn "ActionFrameContext(" silly_kicks/ tests/ --include=*.py
silly_kicks/tracking/utils.py:880
tests/tracking/test_kernels.py:56,117,142,168,184
tests/tracking/test_pressure_andrienko.py:70
tests/tracking/test_pressure_bekkers.py:132
```

Eight constructors, and none of the seven test ones passes `flip_by_action`. Adding it as a
required field raises `TypeError: __init__() missing 1 required positional argument` in all
seven.

**Why this matters beyond the mechanical fix:** the breakage stays latent through Tasks 6–13
(none of them runs those files) and first surfaces at **Task 14 Step 1**, whose stop-condition
reads *"failures confined to pressure snapshot/golden tests. Any other failure is a real
regression — stop and investigate."* The implementer would then halt on damage this plan caused,
with nine tasks of unrelated diff in between. Fix it here, in the same commit that adds the
field.

Add `flip_by_action=pd.Series(dtype=bool)` to each of the seven test constructors — an empty
Series is correct in a test fixture that exercises no flip, and the "never default it" rule
above is about the PRODUCTION constructor, where an empty value silently means "reflect
nothing".

Run: `python -m pytest tests/tracking/test_kernels.py tests/tracking/test_pressure_andrienko.py tests/tracking/test_pressure_bekkers.py -q`
Expected: PASS.

(`default_factory=pd.Series` would also construct a dtype-less empty Series, which emits a
pandas 2.x `FutureWarning`.)

and document it in the class docstring's Attributes block:

```
    flip_by_action : pd.Series
        Indexed by ``action_id``, True where the acting team attacks right-to-left and the
        sampled frame geometry must be point-reflected into action-LTR (ADR-028). Computed
        once here so every consumer re-projects on the SAME decision (ADR-045).
```

Then pass it in the constructor call at `utils.py:880-886`:

```python
        flip_by_action=flip_by_action,
```

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/tracking/_action_orientation.py silly_kicks/tracking/utils.py \
        silly_kicks/tracking/feature_framework.py \
        tests/tracking/test_point_reflection_invariance.py
git commit -m "fix(tracking): negate velocities in action-LTR re-projection (ADR-045 D1)"
```

---

## Task 6: D2 — re-project the ball row (LIVE DEFECT)

The ball is currently not reflected **at all** — position included. Measured actor-to-ball
distance at the same linked frame: home median 6.13 m, away median 62.13 m.

**Files:**
- Modify: `silly_kicks/tracking/features.py:908-923`
- Test: `tests/tracking/test_point_reflection_invariance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_point_reflection_invariance.py`:

```python
def test_ball_row_is_reprojected_into_action_ltr():
    """The ball must land near the actor for an away action, not at its mirror image."""
    from silly_kicks.tracking.features import _build_ball_xy_v_per_action
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    a, f = _scenario()
    # Make the acting team the AWAY side so the action requires re-projection.
    a["team_id"] = "A"
    a["player_id"] = "a1"
    a["start_x"] = FL - 63.0
    a["start_y"] = FW - 41.0

    ctx = _resolve_action_frame_context(a, f)
    ball = _build_ball_xy_v_per_action(a, f, ctx)

    # Ball sits at (60, 40) in frame coords -> (45, 28) in the away action's LTR frame.
    assert float(ball.iloc[0]["x"]) == pytest.approx(FL - 60.0, abs=1e-6)
    assert float(ball.iloc[0]["y"]) == pytest.approx(FW - 40.0, abs=1e-6)
    assert float(ball.iloc[0]["vx"]) == pytest.approx(-2.0, abs=1e-6)
```

Then add the END-TO-END guard, which the Task 5 fixture cannot provide:

```python
def _scenario_ball_leads():
    """Ball near the defenders, defenders FAR from the actor.

    Moving the ball alone is not enough, and an earlier draft of this plan got this wrong.
    It kept Task 5's defenders at x=63/66 against an actor at x=60 and only displaced the
    ball -- but at that separation p_to_actor SATURATES, so np.maximum(p_to_actor, p_to_ball)
    (_kernels.py:695) is won by the actor leg at EVERY ball position. Measured on that
    fixture, sweeping the ball across the whole pitch:

        ball (60.0, 40.0) -> 0.9141067043
        ball (70.0, 39.0) -> 0.9141067043     <- the earlier draft's "ball leads"
        ball (20.0, 10.0) -> 0.9141067043
        ball (100.0, 60.0) -> 0.9141067043
        ball (5.0, 5.0)   -> 0.9141067043

    Zero sensitivity at any placement. So its non-vacuity partner FAILED, and the invariance
    test it guarded would have shipped VACUOUS -- the guard for D2, the one live defect.

    The fix is to move the DEFENDERS out to x=92/95 so p_to_actor is unsaturated and the
    ball leg can win the max. Measured on THIS fixture: ball-on-actor 0.0011294381 vs
    ball-near-defenders 0.9504238642, a swing of 0.9492944261.
    """
    a, f = _scenario()
    f = f.copy()
    f.loc[f["player_id"] == "a1", ["x", "y"]] = [92.0, 40.0]
    f.loc[f["player_id"] == "a2", ["x", "y"]] = [95.0, 38.0]
    is_ball = f["is_ball"].astype(bool)
    f.loc[is_ball, "x"] = 90.0
    f.loc[is_ball, "y"] = 39.0
    return a, f


def _scenario_ball_on_actor():
    """_scenario_ball_leads with the ball moved back onto the actor. Same defenders."""
    a, f = _scenario_ball_leads()
    f = f.copy()
    is_ball = f["is_ball"].astype(bool)
    f.loc[is_ball, "x"] = 60.0
    f.loc[is_ball, "y"] = 40.0
    return a, f


def test_end_to_end_invariance_covers_the_ball_leg():
    a, f = _scenario_ball_leads()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=True)
    assert _pressure(am, fm) == pytest.approx(base, abs=1e-6)


def test_nonvacuity_the_ball_leg_actually_drives_this_fixture():
    """Proves the fixture discriminates: the ball leg must WIN the max here, otherwise
    test_end_to_end_invariance_covers_the_ball_leg is just Task 5's test again.

    Measured: 0.9504238642 (ball near defenders) vs 0.0011294381 (ball on actor).
    On the earlier draft's fixture this difference was exactly 0.0000000000.
    """
    ball_ahead = _pressure(*_scenario_ball_leads())
    on_actor = _pressure(*_scenario_ball_on_actor())
    assert abs(ball_ahead - on_actor) > 0.1, (
        "moving the ball did not change pressure -- the ball leg is not driving this "
        "fixture, so it cannot guard D2"
    )


def test_nonvacuity_an_incomplete_mirror_is_caught_on_the_ball_fixture():
    """Both-sides partner for the ball fixture specifically.

    Measured POST-fix: complete mirror 0.9504238642 (== base), incomplete mirror
    0.0000000981. PRE-fix the incomplete arm reads 0.0011294381 -- non-zero either way,
    so unlike Task 5's partner this one is non-vacuous in BOTH states.
    """
    a, f = _scenario_ball_leads()
    base = _pressure(a, f)
    am, fm = _mirror(a, f, complete=False)
    assert abs(_pressure(am, fm) - base) > 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`
Expected: FAIL — `test_ball_row_is_reprojected_into_action_ltr` (x is 60.0, expected 45.0) and
`test_end_to_end_invariance_covers_the_ball_leg`.

Measured on `_scenario_ball_leads`, all three states, so you can tell a real red from a broken
fixture:

```
                              PRE-fix       D1 only       D1+D2 (target)
base (home, flip=False)     0.9504238642  0.9504238642   0.9504238642
complete mirror             0.0000000000  0.0011294381   0.9504238642
incomplete mirror           0.0011294381  0.0000000000   0.0000000981

INVARIANCE  |base-complete| 0.9504238642  0.9492944261   0.0000000000   <- must reach ~0
NON-VACUITY |base-incompl.| 0.9492944261  0.9504238642   0.9504237661   <- must stay > 1e-3
```

Two things to read off that table:

1. **The fixture isolates D2 correctly.** After Task 5 alone the invariance test is still red
   (0.9492944261). It goes green only when Task 6 lands. An earlier draft's fixture went green
   at Task 5 and could never have guarded D2.
2. **`base` is byte-identical across all three columns** (0.9504238642). That is the spec's
   "home rows bit-identical" claim, executable here rather than eyeballed in a golden diff —
   the base arm is a home action, `flip=False`, and neither fix touches it.

If `test_end_to_end_invariance_covers_the_ball_leg` PASSES before the fix, the fixture is not
driving the max — fix the fixture, do not proceed.

- [ ] **Step 3: Re-project the ball inside the helper**

In `silly_kicks/tracking/features.py`, replace the last two lines of
`_build_ball_xy_v_per_action` (currently `:922-923`):

```python
    merged = pointers_with_period.merge(ball_rows, on=["period_id", "frame_id"], how="left")
    return merged[["action_id", "x", "y", "vx", "vy"]]
```

with:

```python
    merged = pointers_with_period.merge(ball_rows, on=["period_id", "frame_id"], how="left")
    out = merged[["action_id", "x", "y", "vx", "vy"]]

    # ADR-045 D2: the ball never entered ActionFrameContext, so _reproject_rows never
    # saw it -- it reached _pressure_bekkers in FRAME coordinates while the defenders
    # were in action-LTR. Position AND velocity both need re-projecting here.
    #
    # Read the flip from ctx (Task 5 put it there); do NOT recompute
    # acting_team_attacks_rtl. One orientation decision, one place -- otherwise the ball
    # and the players agree only by coincidence, which is the drift this PR removes.
    row_flip = out["action_id"].map(ctx.flip_by_action).fillna(False).astype(bool)
    row_flip.index = out.index
    return reproject_to_action_ltr(
        out, row_flip, x_cols=["x"], y_cols=["y"], vx_cols=["vx"], vy_cols=["vy"]
    )
```

with `from silly_kicks.tracking._action_orientation import reproject_to_action_ltr` hoisted to
module scope. `frames` remains in the signature (it still sources `ball_rows`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Confirm the atomic mirror inherits the fix**

Run: `python -m pytest tests/atomic -q -k pressure`
Expected: PASS. `silly_kicks/atomic/tracking/features.py:993` imports the same helper, so no
separate change is needed. If any atomic test fails, it is because it froze a pre-fix value —
regenerate it, do not revert the fix.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/features.py tests/tracking/test_point_reflection_invariance.py
git commit -m "fix(tracking): re-project the ball row into action-LTR (ADR-045 D2)"
```

---

## Task 7: D3/D3b — `play_left_to_right` via the registry (LATENT — divergent contract, not a live miscomputation)

**LATENT everywhere currently known.** No library producer labels home `"rtl"`, so
`play_left_to_right` is a measured no-op in-library; and the lakehouse — the consumer an earlier
draft named as the live case — reaches orientation through `orient_frames_to_ltr_by_geometry`,
which already negates `vx`/`vy`. Verified in cross-session review; the "live for external
consumers" label is **withdrawn** and must not reach ADR-045.

Still worth fixing: two public orienters with divergent vector semantics is a real defect, and
`play_left_to_right`'s docstring (`utils.py:118-120`) actively claims it mirrors "ALL rows in
that period", which it does not.

**Files:**
- Modify: `silly_kicks/tracking/utils.py:174-182`
- Test: `tests/tracking/test_point_reflection_invariance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_point_reflection_invariance.py`:

```python
def test_play_left_to_right_negates_velocity_and_reflects_smoothed_positions():
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    f["team_attacking_direction"] = np.where(
        f["is_ball"], None, np.where(f["team_id"] == "H", "rtl", "ltr")
    )
    f["x_smoothed"] = f["x"] + 0.5
    f["y_smoothed"] = f["y"] + 0.5

    out = play_left_to_right(f, "H")

    assert float(out.loc[0, "x"]) == pytest.approx(FL - 60.0)
    assert float(out.loc[0, "vx"]) == pytest.approx(-2.0)        # D3
    assert float(out.loc[0, "x_smoothed"]) == pytest.approx(FL - 60.5)  # D3b
    assert float(out.loc[0, "y_smoothed"]) == pytest.approx(FW - 40.5)
    # speed is a magnitude -- must NOT change
    assert float(out.loc[0, "speed"]) == pytest.approx(float(f.loc[0, "speed"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py::test_play_left_to_right_negates_velocity_and_reflects_smoothed_positions -x -q`
Expected: FAIL — `vx` is 2.0, expected -2.0.

- [ ] **Step 3: Route the flip through `reflect()`**

In `silly_kicks/tracking/utils.py`, replace lines `174-182` **inclusive** (from
`period_flip = ...` through `return out`) with:

```python
    period_flip = out["period_id"].isin(rtl_periods).to_numpy()

    # ADR-045: reflect by DECLARED KIND. Previously this transformed x/y only, so vx/vy
    # (a vector), x_smoothed/y_smoothed (a point pair) and the direction label rode through
    # untransformed -- none is in TRACKING_FRAMES_COLUMNS, so all were invisible to a
    # schema-driven author. `direction_label` handles the ltr<->rtl swap: ball rows carry a
    # null label and _DIRECTION_SWAP.get(None, None) is None, so the swap is already a no-op
    # on them and no player/ball split is needed.
    return reflect(out, period_flip, kinds=TRACKING_REFLECTION_KINDS)
```

and hoist the import to module scope with the others at the top of `utils.py`:

```python
from silly_kicks.reflection import TRACKING_REFLECTION_KINDS, reflect
```

Two things to note:

- **`on_unknown` is left at its `"warn"` default.** This is a public entry point. An external
  caller carrying an undeclared column keeps working: the column is treated as `invariant`,
  and it warns only if the name is geometry-shaped (spec §4.5). A consumer wanting hard
  failure escalates the category via `filterwarnings`. An earlier draft defaulted this to
  `"raise"`; that was withdrawn in the fifth review pass — see spec §4.5 and Task 9d.
- **Behaviour delta to accept deliberately:** a ball row that anomalously carried `"ltr"`
  instead of `None` would now have its label swapped, where previously it would not. This is
  arguably more correct, but it is a change — record it in ADR-045.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Verify the no-op path is unchanged**

Run: `python -m pytest tests/tracking -q -k "play_left_to_right or orient"`
Expected: PASS — library frames label home `"ltr"`, so `rtl_periods` is empty and the
function still returns early unchanged.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/utils.py tests/tracking/test_point_reflection_invariance.py
git commit -m "fix(tracking): reflect velocity and smoothed positions in play_left_to_right (ADR-045 D3)"
```

---

## Task 8: D4 — `finalize_orientation`, both legs together

Both legs must change: on a wrong-flag match the flag leg flips positions and the geometric
backstop flips them back (net identity) while negating `vx`/`vy` **once**.

**Not "both or neither".** An earlier draft said that; the fifth review pass measured it false.
"Neither" leaves an 8 m/s-scale kinematic inconsistency between positions and velocities on the
composed path — only **both** is zero-error across all four cases (correct-flag / wrong-flag ×
backstop-fires / does-not). Implement both; do not carry the slogan into ADR-045.

**Files:**
- Modify: `silly_kicks/tracking/direction.py:359-360`
- Test: `tests/tracking/test_point_reflection_invariance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/tracking/test_point_reflection_invariance.py`:

```python
def test_finalize_orientation_flag_leg_negates_velocity():
    """direction.py:284-289 already negates; the flag leg at :359-360 did not."""
    import silly_kicks.tracking.direction as D

    df = pd.DataFrame({
        "game_id": [1], "period_id": [1], "frame_id": [1], "time_seconds": [0.0],
        "player_id": ["p1"], "team_id": ["H"], "is_ball": [False], "is_goalkeeper": [False],
        "x": [20.0], "y": [10.0], "vx": [3.0], "vy": [-2.0], "speed": [np.hypot(3.0, 2.0)],
        "team_attacking_direction": ["ltr"],
    })
    out = D._flip_frames_by_flag(df, np.array([True]))
    assert float(out.loc[0, "x"]) == pytest.approx(FL - 20.0)
    assert float(out.loc[0, "vx"]) == pytest.approx(-3.0)
    assert float(out.loc[0, "speed"]) == pytest.approx(float(df.loc[0, "speed"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py::test_finalize_orientation_flag_leg_negates_velocity -x -q`
Expected: FAIL — `AttributeError: module has no attribute '_flip_frames_by_flag'`

- [ ] **Step 3: Extract the flag leg into a named helper and route it through `reflect`**

In `silly_kicks/tracking/direction.py`, replace lines `359-360`:

```python
    out.loc[flip_mask, "x"] = _PITCH_LENGTH_M - out.loc[flip_mask, "x"]
    out.loc[flip_mask, "y"] = _PITCH_WIDTH_M - out.loc[flip_mask, "y"]
```

with a call to a new module-level helper:

```python
    out = _flip_frames_by_flag(out, flip_mask)
```

and add the helper near the top of the module, after the imports:

```python
def _flip_frames_by_flag(out: pd.DataFrame, flip_mask) -> pd.DataFrame:
    """Point-reflect flagged rows, vectors included (ADR-045 D4).

    The geometric leg at :284-289 already negates vx/vy. This leg did not, so the two
    legs had divergent vector semantics. They COMPOSE on a wrong-flag match (flag leg
    flips, geometric backstop flips back -> net identity), which is correct only when
    BOTH apply complete reflections.
    """
    return reflect_columns(
        out, flip_mask, point_x=["x"], point_y=["y"], vector_x=["vx"], vector_y=["vy"]
    )
```

with the import hoisted to module scope alongside the existing ones.

**Why `reflect_columns` here and `reflect` at the three `play_left_to_right` sites.** This is
the spec §4.5 rule applied, not an exception to it: *schema-bearing table -> `reflect()`;
derived/unschema'd -> `reflect_columns()`*. `finalize_orientation` runs inside the adapter's
**construction window**, on a pre-canonical working frame — `sportec.py:129-131` and
`gradientsports.py:121-123` are `out = raw_frames.copy()` plus two derived columns, so `out`
still carries `x_centered`, `y_centered`, `speed_native` and (for GS) the entire raw bronze
column set. The schema projection is ~20 lines later (`sportec.py:156`). `reflect()` here would
report every one of those raw bronze columns as undeclared on **every sportec and GS
conversion** — a warning storm on correct behaviour, and under the withdrawn `"raise"` default
it would have been a hard break on the IDSSE path all of this spec's measurements came from.

`reflect_columns` is the right instrument for a different reason too: the caller genuinely
knows what these columns are, and there is no schema to consult.

Rejected alternative: filtering `out` to the declared subset and calling `reflect()` on that.
It reads like it preserves fail-closed but does not — an undeclared *geometric* column would be
silently skipped, which is the original defect shape. `reflect_columns` is honest about the same
outcome and needs no new behaviour.

The rule is: **schema-bearing table → `reflect()`; derived or pre-canonical → `reflect_columns()`.**
Note this is now a rule about *which registry applies*, not about *where it is safe to raise* —
the fifth review pass withdrew runtime raising entirely (spec §4.5), and also measured that the
earlier framing was wrong on its own terms: `orient_frames_to_ltr` (`utils.py:295`) and
`spadl/utils.play_left_to_right` (`:1543`) are **not** schema-projected either, and neither
exposes an `extra_kinds` parameter.

**No `extra_kinds` override on `team_attacking_direction`.** An earlier draft forced it to
`"invariant"` here to protect the label. It protects nothing: `direction.py:362` executes
`out["team_attacking_direction"] = None` on the very next statement, discarding whatever this
function did to it. Verified against the working tree.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py -x -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Verify the ADR-035 composition regression still holds**

Run: `python -m pytest tests/tracking/test_adapter_extra_time_orientation.py -q`
Expected: PASS — the two legs must still compose to identity on a wrong-flag match.

- [ ] **Step 5b: Add the tripwire that announces when D4 goes live**

Spec §5's "no converter output changes" rests on an UNTESTED invariant: D4 is unreachable only
because the adapter schema projections drop `vx`/`vy` (`sportec.py:156`,
`gradientsports.py:147`). Nothing states that. Append to
`tests/tracking/test_point_reflection_invariance.py`:

```python
def test_adapter_schemas_exclude_velocity_so_D4_stays_unreachable():
    """ADR-045 D4 is latent ONLY because the adapter schema projection drops vx/vy.

    The day velocity is added to a *_TRACKING_FRAMES_COLUMNS, D4 goes LIVE and the
    finalize_orientation flag leg starts mattering. Fail loudly then, rather than
    discovering it from a wrong number.
    """
    from silly_kicks.tracking.schema import (
        GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS,
        KLOPPY_TRACKING_FRAMES_COLUMNS,
        SPORTEC_TRACKING_FRAMES_COLUMNS,
        TRACKING_FRAMES_COLUMNS,
    )

    for name, cols in [
        ("TRACKING", TRACKING_FRAMES_COLUMNS),
        ("KLOPPY", KLOPPY_TRACKING_FRAMES_COLUMNS),
        ("SPORTEC", SPORTEC_TRACKING_FRAMES_COLUMNS),
        ("GRADIENTSPORTS", GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS),
    ]:
        assert not ({"vx", "vy"} & set(cols)), (
            f"{name}_TRACKING_FRAMES_COLUMNS now carries velocity. ADR-045 D4 is no longer "
            f"latent: finalize_orientation's flag leg re-projects real velocity data. Confirm "
            f"the fix is in place and update spec section 5's 'no converter output changes'."
        )
```

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/direction.py tests/tracking/test_point_reflection_invariance.py
git commit -m "fix(tracking): negate velocity in the finalize_orientation flag leg (ADR-045 D4)"
```

---

## Task 9: SPADL-side `play_left_to_right`

Closes the latent trap on ADR-025 enrichment columns.

**Files:**
- Modify: `silly_kicks/spadl/utils.py:1492`
- Test: `tests/spadl/test_play_left_to_right_enrichment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/spadl/test_play_left_to_right_enrichment.py
"""ADR-045: enrichment coordinate columns must mirror with the canonical ones."""
from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl.utils import play_left_to_right


def test_enriched_restart_coordinates_are_mirrored():
    actions = pd.DataFrame([{
        "game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 0.0,
        "team_id": 2, "player_id": 7,
        "start_x": 10.0, "start_y": 20.0, "end_x": 30.0, "end_y": 40.0,
        "enriched_start_x": 10.0, "enriched_start_y": 20.0,
        "enriched_end_x": 30.0, "enriched_end_y": 40.0,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
    }])
    out = play_left_to_right(actions, home_team_id=1)  # acting team is AWAY -> mirrored
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_end_y"] == pytest.approx(28.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/spadl/test_play_left_to_right_enrichment.py -x -q`
Expected: FAIL — `enriched_start_x` is still 10.0.

- [ ] **Step 3: Route the mirror through `reflect()`**

In `silly_kicks/spadl/utils.py`, the existing body is exactly this (`:1541-1549`, verified
verbatim — an earlier draft of this plan named `away_mask`, `ids_differ` and `out`, none of
which exist):

```python
    from silly_kicks.id_compat import ids_match

    ltr_actions = actions.copy()
    away_idx = ~ids_match(actions.team_id, home_team_id)
    for col in ["start_x", "end_x"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values
    for col in ["start_y", "end_y"]:
        ltr_actions.loc[away_idx, col] = spadlconfig.field_width - actions[away_idx][col].values
    return ltr_actions
```

Replace it with:

```python
    from silly_kicks.id_compat import ids_match

    # ADR-045: reflect by declared kind so enrichment coordinate columns (ADR-025
    # enriched_start_x/_y, enriched_end_x/_y) mirror alongside the canonical four.
    #
    # NO .fillna(False) HERE. An earlier draft added one with a comment claiming
    # ids_match returns a nullable boolean and that filling keeps an unresolvable
    # team_id UNMIRRORED. Both halves are wrong, measured 2026-07-20:
    #   ids_match ends in _as_bool (id_compat.py:272-274) = .fillna(False).astype(bool),
    #   so it returns non-nullable np.bool_ and the fillna is DEAD; and because NA
    #   resolves to False INSIDE ids_match, `~` sends it to the away branch -- an NA
    #   team_id IS mirrored, the opposite of what that comment said.
    # That NA-as-away behaviour is intentional and matches orientation.py:210-218,
    # which warns explicitly against "hardening" it. Do not add a .notna() guard.
    away_idx = ~ids_match(actions.team_id, home_team_id)
    return reflect(actions, away_idx, kinds=SPADL_REFLECTION_KINDS)
```

with `from silly_kicks.reflection import SPADL_REFLECTION_KINDS, reflect` at module scope.

`on_unknown` stays at its `"warn"` default. Task 1's `SPADL_REFLECTION_KINDS` must still carry
all 32 columns including `type_name`/`result_name`/`bodypart_name` — not to avoid a raise, but
because a *declared* column is the only one whose kind is guaranteed correct, and `add_names`
attaches those routinely.

**Two Hyrum-class notes for the migration section of ADR-045**, both on a documented public
boundary helper (`:1502-1507`):

1. **Undeclared columns are treated as `invariant` and may warn.** A caller attaching its own
   column — a join key, a debug field — keeps working. If its name is geometry-shaped it warns
   (`UndeclaredGeometricColumnWarning`) and should be declared via `extra_kinds`. This replaces
   an earlier draft's "now raises", withdrawn in the fifth review pass: `preserve_native`
   (`spadl/utils.py:1651`) lets callers attach arbitrarily-named provider fields, so no registry
   can enumerate the SPADL column universe and a raise had no reachable remedy at Task 9d's
   converter-internal seam.
2. **Duplicated-index behaviour is UNCHANGED on this path.** An earlier draft of this note
   claimed a duplicated action index "now raises". It does not, and `reindex` never did --
   measured on pandas 2.3.3, a same-shaped duplicate aligns positionally and a subset source
   silently broadcasts. `_as_mask` (Task 2) raises only when the index is non-unique AND the
   mask carries a different index; here `away_idx` derives from `actions.team_id`, so it carries
   exactly `actions.index` and the guard never fires. Nothing to migrate. Do not copy the
   retracted claim into ADR-045.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/spadl/test_play_left_to_right_enrichment.py -x -q`
Expected: PASS

- [ ] **Step 5: Verify no SPADL regression**

Run: `python -m pytest tests/spadl -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/utils.py tests/spadl/test_play_left_to_right_enrichment.py
git commit -m "fix(spadl): mirror enrichment coordinates in play_left_to_right (ADR-045)"
```

---

## Task 9b: Atomic-SPADL `play_left_to_right` (site 6 of 8)

`atomic/spadl/utils.py:1129-1133` is a **third correct implementation** of the contract
(alongside `direction.py:284-289`): it negates `dx`/`dy` while point-reflecting `x`/`y`. It is
not broken. Migrate it anyway — leaving one hand-enumerated site is how the next divergence
starts, and the two SPADL siblings already differ (standard carries `enriched_*`, atomic carries
`dx`/`dy`), which is the exact divergence shape of D3.

**Files:**
- Modify: `silly_kicks/atomic/spadl/utils.py:1126-1134`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the failing meta-assertion**

Append to `tests/test_reflection.py`:

```python
def test_meta_every_known_atomic_spadl_column_declares_a_kind():
    from silly_kicks.atomic.spadl import schema as A
    from silly_kicks.reflection import ATOMIC_SPADL_REFLECTION_KINDS

    known = set(A.ATOMIC_SPADL_COLUMNS) | set(A.ATOMIC_SPADL_NAME_COLUMNS)
    missing = sorted(known - set(ATOMIC_SPADL_REFLECTION_KINDS))
    assert not missing, f"columns without a declared reflection kind: {missing}"
    assert len(known) == 15, f"expected the measured 15-column surface, got {len(known)}"
    # dx/dy are VECTORS -- this is the property the migration must not lose.
    assert ATOMIC_SPADL_REFLECTION_KINDS["dx"] == "vector_x"
    assert ATOMIC_SPADL_REFLECTION_KINDS["dy"] == "vector_y"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflection.py -x -q -k atomic`
Expected: FAIL — `ImportError: cannot import name 'ATOMIC_SPADL_REFLECTION_KINDS'` (it is added
in Task 1; if Task 1 is already done this passes and you may proceed).

- [ ] **Step 3: Migrate the call site**

In `silly_kicks/atomic/spadl/utils.py`, replace `:1128-1134`:

```python
    ltr_actions = actions.copy()
    away_idx = ~ids_match(actions.team_id, home_team_id)
    ltr_actions.loc[away_idx, "x"] = spadlconfig.field_length - actions[away_idx]["x"].values
    ltr_actions.loc[away_idx, "y"] = spadlconfig.field_width - actions[away_idx]["y"].values
    ltr_actions.loc[away_idx, "dx"] = -actions[away_idx]["dx"].values
    ltr_actions.loc[away_idx, "dy"] = -actions[away_idx]["dy"].values
    return ltr_actions
```

with:

```python
    # ADR-045: one seam. This site was already CORRECT -- it is migrated, not fixed, so the
    # point/vector contract lives in the registry rather than in hand-written copies.
    # No .fillna(False): ids_match already returns non-nullable bool (see Task 9).
    away_idx = ~ids_match(actions.team_id, home_team_id)
    return reflect(actions, away_idx, kinds=ATOMIC_SPADL_REFLECTION_KINDS)
```

and add at module scope:

```python
from silly_kicks.reflection import ATOMIC_SPADL_REFLECTION_KINDS, reflect
```

- [ ] **Step 4: Verify byte-identical behaviour**

Run: `python -m pytest tests/atomic -q`
Expected: PASS with no golden changes. This site was already correct, so **any** value change
means the migration is wrong — investigate, do not regenerate.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/reflection.py silly_kicks/atomic/spadl/utils.py tests/test_reflection.py
git commit -m "refactor(atomic): route atomic play_left_to_right through the reflection seam (ADR-045)"
```

---

## Task 9c: D7 — the two VAEP `play_left_to_right` helpers (sites 7 and 8 of 8; in-place contract preserved)

Found in cross-session review, not by the audit. `vaep/features/core.py:134` and
`atomic/vaep/features.py:119` are the 4th and 5th same-named functions; the audit carried three
and dropped these two.

**These are NOT drop-in migrations.** Both mutate the caller's frames in place and return the
same `gamestates` objects; `reflect()` is pure (ADR-033). A naive swap converts in-place to pure
and silently breaks any caller relying on the mutation. Compute purely, **assign back**.

**Files:**
- Modify: `silly_kicks/vaep/features/core.py:186-194`
- Modify: `silly_kicks/atomic/vaep/features.py:163-170`
- Test: `tests/test_reflection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reflection.py`:

```python
def test_vaep_play_left_to_right_mirrors_enrichment_and_stays_in_place():
    from silly_kicks.vaep.features import play_left_to_right as vaep_ltr

    df = pd.DataFrame([{
        "game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 0.0,
        "team_id": 2, "player_id": 7,
        "start_x": 10.0, "start_y": 20.0, "end_x": 30.0, "end_y": 40.0,
        "enriched_start_x": 10.0, "enriched_start_y": 20.0,
        "enriched_end_x": 30.0, "enriched_end_y": 40.0,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
    }])
    states = [df]
    out = vaep_ltr(states, home_team_id=1)   # acting team is AWAY -> mirrored

    assert out[0].loc[0, "start_x"] == pytest.approx(95.0)
    assert out[0].loc[0, "enriched_start_x"] == pytest.approx(95.0)   # the latent trap
    # IN-PLACE contract preserved: the caller's own frame was mutated and returned.
    assert out[0] is df
    assert df.loc[0, "enriched_start_x"] == pytest.approx(95.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflection.py -x -q -k vaep`
Expected: FAIL — `enriched_start_x` is still 10.0.

- [ ] **Step 3: Migrate the body, keep the mutation**

In `silly_kicks/vaep/features/core.py`, replace `:189-194`:

```python
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = spadlcfg.field_length - actions[away_idx][col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = spadlcfg.field_width - actions[away_idx][col].values
    return gamestates
```

with:

```python
    # ADR-045: single seam for the transform, but the IN-PLACE contract is preserved --
    # reflect() is pure, so assign its result back into the caller's own frame. Converting
    # this to pure would silently break any caller relying on the mutation.
    for actions in gamestates:
        reflected = reflect(actions, away_idx, kinds=SPADL_REFLECTION_KINDS)
        # Assign back ONLY the columns whose kind can change. Writing every column would
        # upcast untouched integer columns to float (reflect_columns computes in float64) and
        # would replace every underlying array, which is a Hyrum surface on a function whose
        # entire contract is that it mutates in place. The registry already knows which
        # columns are inert.
        for col in reflected.columns:
            if SPADL_REFLECTION_KINDS.get(col) not in ("invariant", "magnitude"):
                actions[col] = reflected[col]
    return gamestates
```

and the same shape in `silly_kicks/atomic/vaep/features.py:165-169`, using
`ATOMIC_SPADL_REFLECTION_KINDS` (its `dx`/`dy` negation is already correct — this is a
migration, not a fix).

Add the imports at module scope in both files.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflection.py tests/atomic tests/invariants -q`
Expected: PASS. `tests/invariants/test_vaep_geometric_sanity.py` and
`test_play_left_to_right_id_dtype.py` exercise these paths — **any value change on the four
canonical coordinates means the migration is wrong**, since only the enrichment columns should
newly move.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/vaep/features/core.py silly_kicks/atomic/vaep/features.py \
        tests/test_reflection.py
git commit -m "fix(vaep): mirror enrichment coords in play_left_to_right, in-place preserved (ADR-045 D7)"
```

---

## Task 9d: D8 — the SPADL orienter (`to_spadl_ltr`), missed by the audit AND by its correction

Found in the fifth review pass. `spadl/orientation.py` carries two more hand-enumerated point
reflections that neither the original audit nor the cross-session review that added Tasks 9c
caught — even though `spadl/orientation.py` is in `_PUBLIC_MODULE_FILES` and the repo's own
`tests/invariants/test_play_left_to_right_id_dtype.py` is docstringed *"the `play_left_to_right`
/ `to_spadl_ltr` family (ADR-019)"*.

**More reachable than Task 9's site.** `to_spadl_ltr` is called by nine converters
(`gradientsports.py:725`, `kloppy.py:242`, `metrica.py:275`, `opta.py:209`,
`skillcorner.py:548`, `sportec.py:650`, `statsbomb.py:286`, `wyscout.py:321`).

**Latent, not live:** at the converter seam the frame carries only the canonical coordinates.
It goes live for any caller routing enrichment-bearing actions through public `to_spadl_ltr`.

**Files:**
- Modify: `silly_kicks/spadl/orientation.py:222-225` and `:272-275`
- Test: `tests/spadl/test_play_left_to_right_enrichment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/spadl/test_play_left_to_right_enrichment.py`:

```python
def test_to_spadl_ltr_absolute_frame_mirrors_enrichment_columns():
    from silly_kicks.spadl.orientation import ABSOLUTE_FRAME_HOME_RIGHT, to_spadl_ltr

    actions = pd.DataFrame([{
        "game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 0.0,
        "team_id": 2, "player_id": 7,
        "start_x": 10.0, "start_y": 20.0, "end_x": 30.0, "end_y": 40.0,
        "enriched_start_x": 10.0, "enriched_start_y": 20.0,
        "enriched_end_x": 30.0, "enriched_end_y": 40.0,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
    }])
    out = to_spadl_ltr(
        actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=1
    )
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)   # the D8 trap
    assert out.loc[0, "enriched_end_y"] == pytest.approx(28.0)


def test_to_spadl_ltr_per_period_mirrors_enrichment_columns():
    from silly_kicks.spadl.orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr

    actions = pd.DataFrame([{
        "game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 0.0,
        "team_id": 1, "player_id": 7,
        "start_x": 10.0, "start_y": 20.0, "end_x": 30.0, "end_y": 40.0,
        "enriched_start_x": 10.0, "enriched_start_y": 20.0,
        "enriched_end_x": 30.0, "enriched_end_y": 40.0,
        "type_id": 0, "result_id": 1, "bodypart_id": 0,
    }])
    # home team, period where home attacks LEFT -> row must mirror
    out = to_spadl_ltr(
        actions, input_convention=PER_PERIOD_ABSOLUTE, home_team_id=1,
        home_attacks_right_per_period={1: False},
    )
    assert out.loc[0, "start_x"] == pytest.approx(95.0)
    assert out.loc[0, "enriched_start_x"] == pytest.approx(95.0)


def test_to_spadl_ltr_preserves_NA_team_as_away():
    """BOTH-SIDES partner for the NA semantics the migration must not change.

    orientation.py:210-218 documents NA-as-away explicitly and warns against
    'hardening' it with .notna(), because that would split the two mirror functions'
    semantics apart. Pin it so the migration cannot drift.
    """
    from silly_kicks.spadl.orientation import ABSOLUTE_FRAME_HOME_RIGHT, to_spadl_ltr

    actions = pd.DataFrame([
        {"game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": None, "player_id": 7, "start_x": 10.0, "start_y": 20.0,
         "end_x": 30.0, "end_y": 40.0, "type_id": 0, "result_id": 1, "bodypart_id": 0},
    ])
    out = to_spadl_ltr(
        actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=1
    )
    assert out.loc[0, "start_x"] == pytest.approx(95.0), "NA team_id must mirror as AWAY"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/spadl/test_play_left_to_right_enrichment.py -x -q`
Expected: FAIL on the two enrichment tests (`enriched_start_x` is still 10.0). The NA test
should already PASS — it pins existing behaviour, and a failure there means you have misread
the current semantics, not found a bug.

- [ ] **Step 3: Route both branches through `reflect()`**

In `silly_kicks/spadl/orientation.py`, replace the enumerated loops in
`_mirror_absolute_frame` (`:222-225`):

```python
    for col in ("start_x", "end_x"):
        out.loc[away_idx, col] = spadlconfig.field_length - out.loc[away_idx, col].to_numpy()
    for col in ("start_y", "end_y"):
        out.loc[away_idx, col] = spadlconfig.field_width - out.loc[away_idx, col].to_numpy()
    return out
```

with:

```python
    # ADR-045 D8: reflect by declared kind so ADR-025 enrichment coordinates mirror
    # alongside the canonical four. The NA-as-away semantics documented above are
    # UNCHANGED -- away_idx is computed exactly as before and merely handed to reflect().
    return reflect(out, away_idx, kinds=SPADL_REFLECTION_KINDS)
```

and the identical shape in `_mirror_per_period` (`:272-275`), passing `mirror_idx`. Keep every
line above each loop untouched — the `ids_match` NA handling, the missing-period `raise`, and
the `if not ....any(): return out` early exits all stay.

Add at module scope:

```python
from silly_kicks.reflection import SPADL_REFLECTION_KINDS, reflect
```

**Both masks are `np.ndarray`** (`.to_numpy()` at `:219` and a boolean expression at `:268`),
so `_as_mask` takes the positional branch and no index alignment is involved.

**`on_unknown` stays at its `"warn"` default, and this site is WHY that default changed.**
`to_spadl_ltr` is called from inside nine converters on a frame already carrying the caller's
`preserve_native` columns, and its signature has no `extra_kinds`. Under the withdrawn
`"raise"` default this call would have broken every converter with no remedy reachable by the
caller. See spec §4.5.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/spadl/test_play_left_to_right_enrichment.py -x -q`
Expected: PASS.

- [ ] **Step 5: Verify no converter regression**

Run: `python -m pytest tests/spadl tests/invariants -q`
Expected: PASS with **no value changes**. Only enrichment columns should newly move, and no
converter emits them at this seam — so any canonical-coordinate change means the migration is
wrong. Investigate; do not regenerate a golden over it.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/spadl/orientation.py tests/spadl/test_play_left_to_right_enrichment.py
git commit -m "fix(spadl): mirror enrichment coordinates in to_spadl_ltr (ADR-045 D8)"
```

---

## Task 10: D5 — decide the shape-graph lateral-label question

This is the one **open question**, not a known fix. Do not guess.

**Files:**
- Read: `silly_kicks/tracking/_shape_graph.py:919-932`, the TF-39 entry in `NOTICE`
- Modify: `silly_kicks/tracking/_shape_graph.py` (only if the decision says so)

**Default to Step 3 (pitch-absolute + docstring) unless the TF-39 source says otherwise.**
`infer_positions` has **no in-library consumer** — verified, the only reference is its own
docstring example at `:906` — so the cheapest correct move is to document the existing
behaviour rather than change it. Step 2 exists for the case where TF-39 clearly intends
team-relative labels.

- [ ] **Step 1: Establish intent**

Read `_shape_graph.py`'s `infer_positions` docstring and the TF-39 attribution in `NOTICE`
(Sotudeh 2026). Answer one question: is the lateral `L/LC/C/RC/R` label **team-relative**
(the rtl team's "left" is the opposite touchline) or **pitch-absolute**?

Note `x = -x` at `:920-921` is a *sort-direction negation*, NOT `105 - x` — it reverses level
ordering rather than mapping into a canonical frame. That is why this is not self-evidently a
defect.

- [ ] **Step 2: If TEAM-RELATIVE — negate `y` AND `face_centers_y`, together**

Two traps here, both of which an earlier draft of this plan fell into:

1. **Use `-y`, not `_PITCH_WIDTH_M - y`.** The existing transform is `x = -x`, a negation about
   the origin, not a reflection about the centre line. `68 - y` is a different map and would not
   be the consistent partner.
2. **`face_centers_y` must be negated too.** The x path already flips both (`:920-921` for `x`,
   `:932-933` for `face_centers_x`). `_assign_levels_horizontal` assigns L/R by comparing values
   against `max(face_centers)`/`min(face_centers)`, so negating `y` while leaving
   `face_centers_y` unflipped compares against unflipped boundaries — **a new defect, worse than
   the one being fixed.**

```python
    # after `x = -x` at :920-921
        y = -y
    # ... and alongside the face_centers_x flip at :932-933
    if attacking_direction < 0 and len(face_centers_y) > 0:
        face_centers_y = -face_centers_y
```

Also resolve, and state in the docstring, what the level assigners should receive: both
`_assign_levels_vertical` and `_assign_levels_horizontal` are currently passed the raw,
**unflipped** `positions` array alongside the flipped scalars. That inconsistency already exists
on the x side; adding a y flip doubles it.

Add a test asserting a team-relative label mirrors under a direction flip.

- [ ] **Step 3: If PITCH-ABSOLUTE (the expected outcome) — document, change no behaviour**

**The decision procedure is unfalsifiable as posed, and the ADR must say so.** `infer_positions`
has no consumer, so there is nothing to validate either answer against; "read the TF-39 intent"
derives the answer from a paper attribution rather than from a behaviour anyone depends on.
Take pitch-absolute **by default in the absence of a consumer**, and record it in ADR-045 as
settled-by-default rather than as resolved intent. If a consumer ever appears, the docstring is
the thing it will contradict — and that contradiction is the signal.

Add to the `infer_positions` docstring:

```
    The lateral label is PITCH-ABSOLUTE: ``y`` is deliberately not mirrored for a
    reversed attacking direction. The ``x`` negation reverses level ORDERING only.
    Settled by default (ADR-045 D5): this function has no consumer, so no behaviour
    validates either convention. A future consumer that needs team-relative labels
    should change this and negate ``y`` AND ``face_centers_y`` together.
```

Add a both-sides test asserting the label does NOT mirror under a direction flip, so the chosen
convention is pinned rather than incidental.

Separately worth a Chesterton pass — not in this PR: whether a consumer-less `infer_positions`
should exist at all. This is the moment that question surfaced; record it, do not act on it here.

- [ ] **Step 4: Commit**

```bash
git add silly_kicks/tracking/_shape_graph.py
git commit -m "fix(tracking): settle shape-graph lateral-label orientation (ADR-045 D5)"
```

---

## Task 11: D6 — correct the false ADR-042 claim

**Files:**
- Modify: `docs/superpowers/adrs/ADR-042-tf35-off-ball-run-valuation.md:92-94`
- Modify: `CLAUDE.md` (the ADR-042 bullet)

- [ ] **Step 1: Correct the ADR**

Replace "TF-4 was the last module in the ACTION-COUPLED GEOMETRY layer keyed on home/away
identity" with:

```
TF-4 was re-keyed onto `acting_team_attacks_rtl`. This did NOT eliminate identity-keying
from the action-coupled geometry layer -- other action-coupled aggregators still take
`home_team_id` by design. The earlier "last module" phrasing was wrong (ADR-045 D6).
```

- [ ] **Step 2: Apply the same correction to the CLAUDE.md ADR-042 bullet**

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/adrs/ADR-042-tf35-off-ball-run-valuation.md CLAUDE.md
git commit -m "docs: correct the ADR-042 last-identity-keyed-module claim (ADR-045 D6)"
```

---

## Task 12: Orienter-divergence guard (spec §4.7.3)

D3 existed because two sibling orienters silently disagreed. Pin them together.

**Files:**
- Modify: `tests/tracking/test_point_reflection_invariance.py`

- [ ] **Step 1: Write the test**

Append to `tests/tracking/test_point_reflection_invariance.py`:

```python
def test_the_two_orienters_agree_on_vector_semantics():
    """play_left_to_right and orient_frames_to_ltr_by_geometry must transform
    identically. They diverged for the whole life of ADR-045 D3: one negated
    velocity, the other did not."""
    from silly_kicks.tracking.direction import orient_frames_to_ltr_by_geometry
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    # Home keeper at HIGH x -> the geometric anchor says this period is mis-oriented,
    # and the label says the same, so both orienters must flip the same rows.
    f.loc[f["player_id"] == "hgk", "x"] = 100.0
    f.loc[f["player_id"] == "agk", "x"] = 5.0
    f["team_attacking_direction"] = np.where(
        f["is_ball"], None, np.where(f["team_id"] == "H", "rtl", "ltr")
    )

    by_flag = play_left_to_right(f, "H")
    by_geom = orient_frames_to_ltr_by_geometry(f, home_team_id="H")

    for col in ("x", "y", "vx", "vy", "speed"):
        pd.testing.assert_series_equal(
            by_flag[col].reset_index(drop=True),
            by_geom[col].reset_index(drop=True),
            check_names=False,
            obj=f"orienter divergence on {col!r}",
        )


def test_orienters_are_pinned_to_PHYSICS_not_merely_to_each_other():
    """Non-vacuity partner for the divergence guard.

    The correctness argument across the pair is TRANSITIVE and load-bearing: the previous
    test pins the two orienters EQUAL to each other, and this one pins ONE of them to the
    physical answer. Together they pin both. Delete either and the remaining one is
    satisfiable by two siblings that are wrong in the same way.

    Agreement between two siblings is NOT correctness: if both were reverted to
    positions-only they would still agree, and the divergence test would pass. "Both wrong in
    the same way" is precisely the successor state this PR creates by consolidating them onto
    one seam, so the guard must anchor to the physical answer as well.

    The fixture is deliberately y-ASYMMETRIC. A y-symmetric probe is how ADR-041's incomplete
    [:, ::-1] repair survived its first fix; Task 13 guards that for grids, and this guards it
    here.
    """
    from silly_kicks.tracking.utils import play_left_to_right

    _a, f = _scenario()
    f = f.copy()
    f.loc[f["player_id"] == "hgk", "x"] = 100.0
    f.loc[f["player_id"] == "agk", "x"] = 5.0
    # y-asymmetric by construction: every row sits off the centre line, and vy is non-zero,
    # so an unnegated vy or an unmirrored y is observable.
    f["y"] = f["y"] + 12.0
    f.loc[~f["is_ball"].astype(bool), "vy"] = 3.0
    # keep the fixture physically coherent: speed is a MAGNITUDE and is asserted
    # unchanged below, so it must actually equal hypot(vx, vy) going in.
    f["speed"] = np.hypot(f["vx"], f["vy"])
    f["team_attacking_direction"] = np.where(
        f["is_ball"], None, np.where(f["team_id"] == "H", "rtl", "ltr")
    )

    out = play_left_to_right(f, "H")

    # The physical answer, computed independently of either orienter.
    for i in range(len(f)):
        assert out.iloc[i]["x"] == pytest.approx(FL - f.iloc[i]["x"])
        assert out.iloc[i]["y"] == pytest.approx(FW - f.iloc[i]["y"])
        assert out.iloc[i]["vx"] == pytest.approx(-f.iloc[i]["vx"])
        assert out.iloc[i]["vy"] == pytest.approx(-f.iloc[i]["vy"])
        assert out.iloc[i]["speed"] == pytest.approx(f.iloc[i]["speed"])  # magnitude
    # Non-vacuity: the fixture must actually be y-asymmetric, or the y assertions are free.
    assert (f["y"] - FW / 2).abs().min() > 1.0, "fixture is y-symmetric -- assertions are vacuous"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py::test_the_two_orienters_agree_on_vector_semantics -x -q`
Expected: PASS (Task 7 already aligned them). If it FAILS, Task 7 is incomplete — fix Task 7,
do not weaken this test.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_point_reflection_invariance.py
git commit -m "test(tracking): pin the two orienters to identical vector semantics (ADR-045)"
```

---

## Task 12b: Call-site conformance guard (spec §4.7.4)

`reflect_columns` skips a named-but-absent column (`if col in out.columns`). That tolerance is
**necessary** — frames without `derive_velocities` legitimately have no `vx` — but it means the
two LIVE defects are fixed on a path whose failure mode is structurally identical to the original
bug: the caller enumerates, and a miss is silent. Task 4 gates the registries; nothing gates the
enumerating call sites. This test is the durable output of the whole PR.

**Four call sites, gated two ways.** `reproject_to_action_ltr` has FOUR call sites (measured:
`utils.py:874`, `_kernels.py:879`, `features.py:2026`, `features.py:2433`). Sites 1-3 are gated
by the shared `GEOMETRIC_NAME` pattern (Task 4). **Site 4 — `_reproject_team_shape`
(`features.py:2026`) — cannot use it:** its column names are infix (`team_shape_centroid_x_attacking`)
and one carries no axis token at all (`team_shape_defensive_line_height_attacking`), all
`GEOMETRIC_NAME.match() == False` (measured). So site 4 is gated **behaviourally** — mirror
invariance over auto-discovered columns — which is stronger anti-rot than a name pattern anyway.
(`features.py:2433` is inert — a two-column scratch frame renamed and enumerated on the next line;
verified, no gate needed.) The site-4 gate also closes a measured live gap: the pre-existing
`test_team_shape_centroids_mirror_invariant` is **vacuous on the y-axis** (its `_scenario` centroid
sits ~1 m off centre, so `68−y` is a near-identity), so today the team-shape y-reflection is
effectively untested. The site-4 both-sides partner is the executable proof it now is.

**Files:**
- Modify: `tests/tracking/test_point_reflection_invariance.py` (sites 1-3)
- Modify: `tests/tracking/test_action_ltr_mirror_invariance.py` (site 4 — it needs that file's
  `_ghost_scenario`/`_mirror`/`HOME`/`AWAY`/`FW`)

- [ ] **Step 1: Write the test**

Append to `tests/tracking/test_point_reflection_invariance.py`:

```python
from silly_kicks.reflection import GEOMETRIC_NAME as _GEOMETRY_COL


def test_every_geometry_column_on_the_context_is_enumerated_for_reprojection():
    """A geometry column that reaches the context but is not enumerated at the
    re-projection call site is silently left in frame coordinates -- ADR-045 D1/D2 exactly."""
    from silly_kicks.tracking.utils import _resolve_action_frame_context

    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    a, f = _scenario()
    # RUN THE REAL PREPROCESS. Without it the fixture carries no x_smoothed/y_smoothed and
    # this guard is blind to exactly the columns it exists to catch -- and its non-vacuity
    # assertion below would be satisfied by the very enumeration it is checking, which is
    # how a guard reports green while covering nothing. derive_velocities REQUIRES the
    # smoothed pair (preprocess/_velocity.py:41), so this ordering is forced.
    f = derive_velocities(smooth_frames(f))

    # defending_gk_rows is EMPTY without this: _resolve_action_frame_context (utils.py:851-855)
    # returns an empty frame when the column is absent, and an `if rows.empty: continue` would
    # then skip a third of the surface this guard claims to cover -- silently.
    a = a.assign(defending_gk_player_id="agk")
    ctx = _resolve_action_frame_context(a, f)

    # What _reproject_rows actually enumerates (keep in sync with utils.py:874).
    enumerated = {"x", "y", "vx", "vy", "x_smoothed", "y_smoothed"}

    for name in ("actor_rows", "opposite_rows_per_action", "defending_gk_rows"):
        rows = getattr(ctx, name)
        assert not rows.empty, (
            f"{name} is empty -- this guard would silently cover only part of its surface. "
            f"Fix the fixture; do not skip the surface."
        )
        geometry = {c for c in rows.columns if _GEOMETRY_COL.match(c)}
        missed = geometry - enumerated
        assert not missed, (
            f"{name} carries geometry column(s) {sorted(missed)} that _reproject_rows does not "
            f"re-project. Either enumerate them at utils.py:874 or declare why they are exempt."
        )
        # Non-vacuity: the fixture must actually exercise the columns we claim to cover.
        # x_smoothed/y_smoothed are the load-bearing half -- they are the ones an earlier
        # draft's fixture lacked, which made this assertion satisfiable by the enumeration
        # it was supposed to be checking.
        assert {"x", "y", "vx", "vy", "x_smoothed", "y_smoothed"} <= geometry, (
            f"{name} fixture does not carry x/y/vx/vy/x_smoothed/y_smoothed -- this guard "
            f"would pass vacuously. Did preprocess actually run on the fixture?"
        )


def test_defensive_line_reprojection_enumerates_every_geometry_column():
    """Site 2 of 4. _kernels.py:879 passes y_cols=[] -- a live assumption that nothing
    lateral is ever added to the defensive-line output. Gate it."""
    from silly_kicks.tracking.features import add_defensive_line

    a, f = _scenario()
    # home_team_id is KEYWORD-ONLY AND REQUIRED (features.py:1186). An earlier draft called
    # add_defensive_line(a, frames=f) and would have died with TypeError before asserting
    # anything.
    out = add_defensive_line(a, frames=f, home_team_id="H")
    added = set(out.columns) - set(a.columns)

    # What _kernels.py:879 enumerates. Keep in sync.
    enumerated = {"defensive_line_x", "back_line_high_x"}
    # compactness_x is a SPAN (a difference of two x values), so it is flip-invariant --
    # documented at _kernels.py:876-877. Derived by measuring the real add_defensive_line
    # output against the pattern, not guessed: the geometry-matching columns are exactly
    # {defensive_line_x, back_line_high_x, compactness_x}.
    exempt = {"compactness_x": "span (difference of x values) -- flip-invariant"}

    geometry = {c for c in added if _GEOMETRY_COL.match(c)} - set(exempt)
    missed = geometry - enumerated
    assert not missed, (
        f"add_defensive_line emits geometry column(s) {sorted(missed)} that _kernels.py:879 "
        f"does not re-project (it passes y_cols=[]). Enumerate them, or add a documented "
        f"exemption."
    )
    # BOTH-SIDES: this guard was unconditionally passing for a full revision cycle because the
    # pattern matched nothing. Prove it sees the real columns.
    assert enumerated <= {c for c in added if _GEOMETRY_COL.match(c)}, (
        "the guard cannot see defensive_line_x / back_line_high_x -- it is vacuous"
    )


def test_finalize_orientation_enumerates_every_geometry_column_it_owns():
    """Site 3 of 4, found in review. finalize_orientation runs on a PRE-canonical frame that
    carries live geometric columns it does not reflect.

    gradientsports.py:121-123 does `out = raw_frames.copy()` then derives x/y from
    x_centered/y_centered, so x_centered/y_centered reach the flip UNREFLECTED. That is benign
    today only because they are dead after the projection at gradientsports.py:147 -- and
    nothing asserts they stay dead. The fix's own boundary would otherwise ship carrying the
    original defect shape.
    """
    import silly_kicks.tracking.direction as D

    df = pd.DataFrame({
        "game_id": [1], "period_id": [1], "frame_id": [1], "time_seconds": [0.0],
        "player_id": ["p1"], "team_id": ["H"], "is_ball": [False], "is_goalkeeper": [False],
        "x": [20.0], "y": [10.0], "vx": [3.0], "vy": [-2.0], "speed": [np.hypot(3.0, 2.0)],
        "x_centered": [-32.5], "y_centered": [-24.0],   # the adapter scratch columns
        "team_attacking_direction": ["ltr"],
    })
    out = D._flip_frames_by_flag(df, np.array([True]))

    enumerated = {"x", "y", "vx", "vy"}
    # Adapter scratch, exempt ONLY because both adapters project them away before the frame
    # becomes canonical (sportec.py:157, gradientsports.py:147).
    scratch = {"x_centered", "y_centered"}

    # BOTH-SIDES first: this guard existed for a revision cycle while matching NOTHING (the
    # pattern was .match()-anchored, and x_centered is a PREFIX form that the round-two
    # proposed fix would still have missed). Prove it sees them before trusting the result.
    assert {c for c in df.columns if _GEOMETRY_COL.match(c)} >= scratch | enumerated, (
        "the guard cannot see the adapter scratch columns -- it is vacuous"
    )

    unreflected = {
        c for c in df.columns
        if _GEOMETRY_COL.match(c) and c not in enumerated and out[c].equals(df[c])
    }
    assert unreflected <= scratch, (
        f"geometry column(s) {sorted(unreflected - scratch)} pass through "
        f"finalize_orientation unreflected and are not documented as adapter scratch."
    )
    assert scratch <= unreflected, (
        "x_centered/y_centered were reflected -- if that is now intended, update this guard "
        "and confirm the adapters still project them away"
    )
    # The rule this encodes: any frame reaching a reflect_columns call must have every
    # geometry-named column either enumerated or explicitly exempted with a reason.
    # x_centered/y_centered are exempt ONLY because both adapters project them away
    # (sportec.py:157, gradientsports.py:147) before the frame becomes canonical.
```

**Site 4 goes in a DIFFERENT FILE.** Sites 1-3 above append to `test_point_reflection_invariance.py`
(they use its `_scenario`). Site 4 needs `_ghost_scenario`, `_mirror(a, f)` (the two-arg mirror,
NOT this file's `_mirror(a, f, *, complete)`), and the `HOME`/`AWAY`/`FW` constants — all of which
live in `tests/tracking/test_action_ltr_mirror_invariance.py`, alongside the vacuous
`test_team_shape_centroids_mirror_invariant` it strengthens. **Append the two site-4 tests THERE,
not here.** (Do not import cross-module — that file already has every symbol.)

Append to `tests/tracking/test_action_ltr_mirror_invariance.py` (verified 2026-07-20 to already
import `numpy as np`, `pandas as pd`, `pytest`, and `add_team_shape` at module scope — add
nothing, just append the two functions):

```python
def test_team_shape_reprojection_is_mirror_invariant_over_ALL_columns():
    """Site 4 of 4. _reproject_team_shape (features.py:2026) hand-enumerates
    _TEAM_SHAPE_X_COLS / _TEAM_SHAPE_Y_COLS. GEOMETRIC_NAME CANNOT SEE these infix names
    (`team_shape_centroid_x_attacking` -> .match() is False -- measured), so unlike sites
    1-3 this gate must be BEHAVIOURAL, not name-based: under a physical mirror every emitted
    team-shape column must be invariant in action-LTR. Auto-discovering `added` (not a hand
    list) is the anti-rot half -- a FUTURE lateral column that _reproject_team_shape forgets
    to enumerate would break this without any name signal.

    Uses _ghost_scenario (attack lateralised low), NOT _scenario. Measured 2026-07-20: on
    _scenario the acting-team centroid_y sits ~1 m off the centre line, so 68-y is a near
    identity and the y-reflection is UNTESTED -- the pre-existing
    test_team_shape_centroids_mirror_invariant is vacuous on the y-axis (disabling
    _TEAM_SHAPE_Y_COLS leaves its assertions green). _ghost_scenario's action 1 is an AWAY
    action (flip=True) with centroid_y ~= 51 (17 m off centre), so the y-axis carries real
    signal. The both-sides partner below proves it.
    """
    from silly_kicks.tracking.features import add_team_shape

    a, f = _ghost_scenario()
    am, fm = _mirror(a, f)
    base = add_team_shape(a, f, home_team_id=HOME)      # action 1: away, flip=True (reprojects)
    mir = add_team_shape(am, fm, home_team_id=AWAY)     # action 1: home, flip=False (raw)

    # NON-VACUITY: the acting-team centroid must be genuinely off the centre line, or the
    # y-axis is untested exactly as the pre-existing test is. Measured base value ~= 51.
    b1 = base[base["action_id"] == 1].iloc[0]
    assert abs(float(b1["team_shape_centroid_y_attacking"]) - FW / 2) > 3.0, (
        "acting centroid_y is within 3 m of the centre line -- the y-reflection is not "
        "exercised (this is the vacuity measured in test_team_shape_centroids_mirror_invariant)"
    )

    # ANTI-ROT: EVERY added column invariant under the mirror. No name pattern. A lateral
    # column riding through _reproject_team_shape unreflected differs between the flip=True
    # (base) and flip=False (mir) representations of the same physical scene.
    #
    # NA-SAFE by necessity: team-shape emits nullable columns (a degenerate hull / absent
    # second inter-line gap on this fixture is pd.NA, not np.nan). `pd.isna` covers BOTH; a
    # bare `np.isnan` raises TypeError on pd.NA, and `pd.NA == pytest.approx(...)` raises
    # "boolean value of NA is ambiguous" -- measured 2026-07-20, this is a real crash, not a
    # style note. Two shared-NA columns are skipped; 22 numeric columns are checked.
    m1 = mir[mir["action_id"] == 1].iloc[0]
    added = sorted(set(base.columns) - set(a.columns))
    checked = 0
    for col in added:
        bv, mv = b1[col], m1[col]
        if pd.isna(bv) or pd.isna(mv):
            assert pd.isna(bv) and pd.isna(mv), (
                f"team-shape column {col!r} is NA on one side only (base={bv}, mir={mv}) -- "
                f"a mirror should not create or destroy a value"
            )
            continue
        assert float(bv) == pytest.approx(float(mv), abs=1e-6), (
            f"team-shape column {col!r} is not mirror-invariant (base={bv}, mir={mv}) -- a "
            f"lateral quantity is riding through _reproject_team_shape unreflected"
        )
        checked += 1
    assert checked >= 20, f"only {checked} columns actually compared -- fixture may be degenerate"


def test_team_shape_gate_fails_when_the_y_reprojection_is_disabled():
    """BOTH-SIDES partner for site 4, and the executable record of the vacuity finding.
    Disabling _TEAM_SHAPE_Y_COLS must BREAK the mirror-invariance above; if it does not, the
    scenario is not y-asymmetric enough and the gate is vacuous. Measured: the ON delta is 0
    and the OFF delta is ~34 on _ghost_scenario action 1."""
    import silly_kicks.tracking.features as _F
    from silly_kicks.tracking.features import add_team_shape

    a, f = _ghost_scenario()
    am, fm = _mirror(a, f)
    orig = _F._TEAM_SHAPE_Y_COLS
    try:
        _F._TEAM_SHAPE_Y_COLS = []  # a y re-projection that never happens
        base = add_team_shape(a, f, home_team_id=HOME)
        mir = add_team_shape(am, fm, home_team_id=AWAY)
    finally:
        _F._TEAM_SHAPE_Y_COLS = orig
    b = float(base[base["action_id"] == 1].iloc[0]["team_shape_centroid_y_attacking"])
    m = float(mir[mir["action_id"] == 1].iloc[0]["team_shape_centroid_y_attacking"])
    assert abs(b - m) > 1.0, (
        "disabling the y re-projection did not break mirror-invariance -- the fixture is not "
        "y-asymmetric enough, so the site-4 invariance gate is vacuous on the y-axis"
    )
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_point_reflection_invariance.py tests/tracking/test_action_ltr_mirror_invariance.py -x -q -k "conformance or enumerated or team_shape"`
Expected: PASS (sites 1-4 + the site-4 both-sides partner). If a site FAILS it has found a real
gap — enumerate the column, do not widen `enumerated`. `test_team_shape_gate_fails_when_the_y_reprojection_is_disabled`
is a both-sides partner: it must be GREEN (it asserts the gate CAN fail), and it monkeypatches a
module global inside a `try/finally` — confirm `_TEAM_SHAPE_Y_COLS` is restored.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_point_reflection_invariance.py tests/tracking/test_action_ltr_mirror_invariance.py
git commit -m "test(tracking): gate that context geometry columns are all re-projected (ADR-045)"
```

---

## Task 13: Grid-reflection shape assertion (spec §4.6)

**CUT — do not implement. The proposed test was dead, and the coverage already exists.**

An earlier draft of this plan proposed appending a test that built a `np.arange(12).reshape(3,4)`
grid and asserted `grid[::-1, ::-1] != grid[:, ::-1]`. That test **imports no repository code**.
It asserts that numpy slicing behaves like numpy slicing, so it cannot fail for any change
anyone could make to silly-kicks — a guard with no discriminating power, in a PR whose entire
thesis is that a computation which appears to run and silently does not is the defect class.

The spec's §4.6 promise of "a shape assertion that both axes are reversed" is also not
implementable as stated: slicing reversal is **shape-invariant**, so a shape assertion can
never distinguish `[:, ::-1]` from `[::-1, ::-1]`. And a value assertion on the default grid is
vacuous too, because the synthetic EPV ramp is y-symmetric — which is exactly why the ADR-041
defect survived its first repair.

**The real guard already exists and is behavioural.**
`tests/tracking/test_obso_orientation.py:158`, `class TestEpvIsReflectedOnBothAxes`, whose
docstring records the same history:

> *"The first repair of DEFECT A flipped the EPV grid on the x axis alone, which is exact only
> for a y-symmetric grid — true of the synthetic ramp default, and approximately true of a
> fitted xT surface, which is precisely why it survived the x-axis tests."*

It injects a deliberately y-ASYMMETRIC EPV grid through the real `add_obso` and asserts the
away team's threat is read from the correct half of the pitch. That is a test of the code, on
the failure mode, with discriminating power.

- [ ] **Step 1: Confirm the existing guard covers this and record the decision**

Run: `python -m pytest tests/tracking/test_obso_orientation.py -q -k BothAxes`
Expected: PASS.

Amend spec §4.6 to drop the "shape assertion" promise and point at this class instead. Record
in ADR-045 that the grid reflections are guarded **behaviourally** by an asymmetric-grid probe,
not by a shape assertion — and why a shape assertion cannot work.

No new test, no code change, no commit for this task.

---

## Task 14: Regenerate pressure goldens

`bekkers_pi` away values legitimately change. Goldens must move; that is the point.

**Files:**
- Modify: pressure snapshot fixtures via `scripts/regenerate_pressure_snapshot_shas.py`

- [ ] **Step 1: Confirm only pressure snapshots fail**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: failures confined to pressure snapshot/golden tests. **Any other failure is a real
regression — stop and investigate, do not regenerate over it.**

- [ ] **Step 2: Regenerate**

Run: `python scripts/regenerate_pressure_snapshot_shas.py`

- [ ] **Step 3: Make the home-row claim EXECUTABLE, not an eyeball**

A manual diff inspection is not a gate. Append to
`tests/tracking/test_point_reflection_invariance.py`:

```python
def test_home_acting_actions_never_enter_reprojection():
    """The fix re-projects only rows whose acting team attacks rtl. Home rows must be
    byte-identical -- expressible today, no golden diff required."""
    a, f = _scenario()
    base = _pressure(a, f)          # home acting -> flip False -> early return

    # Mutate ONLY away-row velocities; the home action's value must not move.
    f2 = f.copy()
    away = f2["team_id"] == "A"
    f2.loc[away, "vx"] = f2.loc[away, "vx"] * 3.0
    assert _pressure(a, f2) != pytest.approx(base, abs=1e-9), (
        "away velocities do not affect this action -- fixture cannot discriminate"
    )

    # And the home-acting path never enters re-projection at all.
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
    assert not bool(acting_team_attacks_rtl(a, f).iloc[0])
```

- [ ] **Step 4: Add the owner-gated two-provider e2e (spec §6 generality gap)**

This PR forces a rho retrain, a deep-zone gate re-run and a ~5.5 h x 8-worker lakehouse drain
on evidence from a **single** match. §2 D1's home/away asymmetry is directly expressible, so
close the gap with measurement instead of argument.

Create `tests/tracking/test_bekkers_home_away_asymmetry_e2e.py`:

```python
"""Owner-gated: the ADR-045 velocity fix must remove an unjustified home/away asymmetry.

Pre-fix measurement (IDSSE DFL-MAT-J03WMX): away mean 0.2897 vs home 0.4848, ratio 0.60.
Post-fix: 0.4325 vs 0.4848, ratio 0.89. Football gives no reason for a large systematic gap
between home and away pressure; a low ratio is the fingerprint of the defect.
"""
import pytest

_MIN_RATIO = 0.80   # pre-fix 0.60 fails; post-fix 0.89 passes


@pytest.mark.e2e
@pytest.mark.parametrize("provider", ["idsse", "gradientsports"])
def test_home_away_bekkers_asymmetry_is_within_tolerance(provider):
    ...  # load one match per provider via scripts._loader_pining, add_pressure_on_actor,
         # split by acting-team direction, assert mean(away)/mean(home) >= _MIN_RATIO
```

Run on **two** providers (IDSSE plus one Gradient Sports or SkillCorner match). Fill in the
loader body following `scripts/_loader_pining.load_matches`; it is owner-gated so it does not
run in CI.

- [ ] **Step 5: Full suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: regenerate pressure goldens for the ADR-045 velocity fix"
```

---

## Task 15: ADR-045, CLAUDE.md, version bump

**Files:**
- Create: `docs/superpowers/adrs/ADR-045-vector-reflection-consistency.md`
- Modify: `CLAUDE.md`, `pyproject.toml`, `silly_kicks/__init__.py`

- [ ] **Step 1: Write ADR-045**

Use `docs/superpowers/adrs/ADR-TEMPLATE.md`. Record:

- The defect family and its root cause: reflection APIs that enumerate columns and **cannot
  express kind**. Frame it as a missing shared seam, not missing physics — four independent
  places already implemented the correct contract.
- **D1–D8** with their measured impact. D1 and D2 are the only LIVE ones; say so.
- **The §4.5 decision and its reversal.** Fail-closed lives in the CI meta-assertion, not the
  runtime call. Give all three reasons: `preserve_native` makes the SPADL column universe
  unbounded by construction so runtime completeness is unachievable; `to_spadl_ltr` is called
  converter-internally with no reachable `extra_kinds`, so a raise there has no remedy; and
  **zero of the eight catalogued defects involved a caller-owned column**, so runtime raising
  would have caught none of them while breaking the one case it reaches. State that strictness
  is available to consumers via `filterwarnings("error", category=...)`.
- **The residual hole, plainly:** a third-party caller attaching a geometric column whose name
  `GEOMETRIC_NAME` misses gets silent `invariant` treatment. Scope limit, not a guarantee.
- The D4 both-legs-compose constraint — but **not** as "both or neither". That sentence is
  measurably false: "neither" leaves an 8 m/s-scale kinematic inconsistency, and only "both" is
  zero-error in all four cases. The plan's *implementation* is correct; the justifying sentence
  was not.
- The per-row-never-aggregate guard rule, with the numbers that force it (D2 mean bias -1.1%,
  D3 -0.002).
- **Counts as measurements, not assertions.** Quote spec §4.6's breakdown (eleven places apply
  a reflection; two more are defective by omission) and paste the `grep` that produced it. Do
  not repeat the audit's "96 sites examined / exhaustive" claim — the inventory has now been
  wrong three times, and §1 explains why.
- `GEOMETRIC_NAME`'s **measured blind spots** (infix axis tokens; one column with no axis token
  at all), not a coverage claim.
- D5 recorded as **settled-by-default**, not resolved intent — `infer_positions` has no
  consumer, so nothing validates either answer.
- One sentence on why there is no `angle` kind (spec §4.3).
- The grid reflections are guarded **behaviourally** by
  `tests/tracking/test_obso_orientation.py::TestEpvIsReflectedOnBothAxes`; a shape assertion
  cannot work, because slicing reversal is shape-invariant.

- [ ] **Step 2: Add the CLAUDE.md bullet**

Follow the existing PR-S### bullet style. State the retrain/re-materialize consequences.

- [ ] **Step 3: Bump the version in all three places**

`pyproject.toml` `version = "4.55.0"`, plus `silly_kicks/__init__.py`. Verify:

```bash
grep -rn "4\.53\.0" pyproject.toml silly_kicks/__init__.py
```
Expected: no output.

- [ ] **Step 4: Lint, type-check, full suite**

```bash
ruff check . && ruff format --check . && pyright && python -m pytest tests/ -m "not e2e" -q
```
Expected: all clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(adr): ADR-045 vector-reflection consistency; bump 4.55.0"
```

---

## Task 16: Downstream consequences (owner-run, NOT in this PR)

These require compute the PR cannot do. Record them in the PR body as follow-ups.

> **These are NOT independent — they are one ordered pass.** This PR and the shipped 4.52.0
> xT-EPV / TF-35 work both force a lakehouse action-context recompute. An AC-1 cold drain is
> ~5.5 h across 8 workers plus a staging rebuild and a `rederive_synced_marts --rebuild`;
> running it twice is a full working day of compute and two windows of mart churn. The chains
> also overlap: `bekkers_pi` -> rho retrain -> deep-zone gate -> `xt_gk_v2` materialization,
> which is already queued behind the rho retrain.
>
> **Pinned order:** ship both library changes -> **one** wheel bump -> **one** AC drain ->
> **one** rho retrain (both variants) -> re-run the deep-zone gate **once**, on corrected
> pressure. Put this ordering in the PR body explicitly so it is not scheduled as three
> independent follow-ups.

- [ ] **Step 1: Re-run the xT-GK v2 deep-zone gate**

`scripts/validate_xtgk_possession_value.py` pins `pressure_on_actor__bekkers_pi`
(`:47`). Its GO-leaning verdict was measured on broken pressure and must be re-run.

- [ ] **Step 2: Retrain rho, both variants**

`scripts/train_gk_retention.py` uses `bekkers_pi` as a feature. Retrain `default` and
`skillcorner`; both must still clear the calibration gate (`ece<=0.10`,
`|slope-1|<=0.25`) per `tests/xtgk/test_retention_bundle_calibration.py`.

- [ ] **Step 3: Lakehouse re-materialize**

`fct_action_context.pressure_on_actor__bekkers_pi` for all providers.

---

## Verification checklist

- [ ] `python -m pytest tests/ -m "not e2e" -q` passes
- [ ] `ruff check .` and `ruff format --check .` clean (run separately — house convention)
- [ ] `pyright` clean (bare `pyright`, full package scope)
- [ ] Home-team pressure values are byte-identical; only away values moved
- [ ] Every new invariance test has a non-vacuity partner that fails on a positions-only mirror
- [ ] Registry-completeness meta-assertions pass against the real schemas
- [ ] `/final-review` run, including C4 regeneration

Added by the fifth review pass — each corresponds to a defect it found:

- [ ] `silly_kicks/reflection.py` is registered in `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` **and** `tests/test_no_import_cycles.py::_PACKAGES` (Task 1 Step 3c)
- [ ] All seven test `ActionFrameContext(...)` constructors pass `flip_by_action` (Task 5 Step 6b)
- [ ] `tests/spadl` and `tests/invariants` show **no value change** after Task 9d — only enrichment columns may newly move
- [ ] `_JUSTIFIED_NON_GEOMETRIC` is empty, and its three meta-tests pass (Task 4)
- [ ] Task 6's ball fixture measurably discriminates: `|ball_near_defenders - ball_on_actor| > 0.1`
- [ ] Task 6's invariance test is still RED after Task 5 alone (it must isolate D2, not ride Task 5)
- [ ] No document or docstring claims `GEOMETRIC_NAME` has "zero misses, zero false positives"
- [ ] No document claims a headline site count; §4.6's breakdown is quoted with its `grep`
- [ ] ADR-045 does **not** say "both or neither" for D4
- [ ] A `preserve_native` conversion emits **no** `UndeclaredGeometricColumnWarning`
