# SB360 Coverage Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish, by execution rather than by reading, which of silly-kicks' 33 tracking aggregators (plus `gkdv`, `xtgk` v2 and `spadl.add_restart_coordinates`) produce meaningful output on StatsBomb 360 freeze-frames — and lock the answer against silent rot.

**Architecture:** A paired synthetic fixture (freeze-frame Leg A built by the real producer, velocity-bearing Leg B with identical linked-frame positions) drives every aggregator twice per axis. A machine **observation** is re-derived on every CI run and locked; a human **adjudication** carries the judgement and its rationale. A namespaced state vocabulary with its own completeness gate prevents the design acquiring another hole. A separate network-gated driver measures real 360 coverage.

**Tech Stack:** pytest, pandas, numpy, `scripts/_driver.py` (ADR-052 corpus seam), `scripts/_provenance.py` (ADR-037), optional `statsbombpy`.

**Spec:** `docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md` (rev 5)

## Global Constraints

- **No library behaviour change.** Nothing under `silly_kicks/` is modified. This is audit-only; every finding is recorded, none is fixed.
- **The `snapshots` contract is not extended.** `snapshot_to_tracking_frames` is called, never edited. The `visible_area` polygon rides a side table keyed by `action_id` inside the harness.
- **Registry gates land RED.** Enumeration commits before verdicts, so each gate is observed failing. A gate written after its own repair arrives green and is never seen to work (ADR-051).
- **Observations are executed, never declared.** Every recorded observation is produced by running the aggregator.
- **Vocabulary is namespaced** — `call_outcome.*`, `row_class.*`, `observation.*`, `adjudication.*`, `kind.*`, `applicability.*` — declared once in `tests/sb360/_vocabulary.py`.
- **Docs commit first.** `scripts/_provenance.py` counts untracked files as dirty, so the spec and this plan must be in the branch's first commit or Task 9's driver `SystemExit`s before doing any work.
- **Tolerance:** default `rtol=1e-9`, `atol=1e-12`, float columns only. Counts and booleans compare exactly. Cross-leg dtype mismatch is a loud failure, never an implicit cast.
- **Python:** no `__init__.py` in `tests/scripts/`; `tests/sb360/` does get one (mirrors `tests/tracking/`).
- **Version:** bump to the next number from `main` (tree reads `4.73.0` → `4.74.0`); confirm no collision at PR time.

## File Structure

| Path | Responsibility |
|---|---|
| `tests/sb360/__init__.py` | Package marker (mirrors `tests/tracking/`) |
| `tests/sb360/_vocabulary.py` | The namespaced state vocabulary — single source |
| `tests/sb360/_compare.py` | Row classification, column aggregation, dtype/tolerance dispatch |
| `tests/sb360/_fixture.py` | Versioned paired Leg A / Leg B builder |
| `tests/sb360/_probes.py` | Applicability-class perturbation probes |
| `tests/sb360/_registry.py` | `Sb360Entry` dataclass, `SB360_ENTRIES`, `_entry()` helper |
| `tests/sb360/_entries/` | Per-family entry modules (mirrors `tests/tracking/_mirror_entries/`) |
| `tests/sb360/_harness.py` | Runs one entry on both legs → observations |
| `tests/sb360/test_*.py` | The gates |
| `scripts/build_sb360_coverage.py` | Layer B corpus driver |
| `tests/scripts/test_build_sb360_coverage.py` | Driver unit tests |
| `docs/research/sb360_coverage/` | Report artifact |

**Phase boundary:** Tasks 1–8 are Layer A and are independently shippable — they deliver the behaviour matrix and every CI gate with no network dependency. Tasks 9–11 are Layer B and can be deferred to a follow-on PR without invalidating Layer A.

## Execution notes — cycle 1 (Tasks 1–4 complete)

Three mechanisms differ from what this plan specified, each because **execution measured
something reading could not**. Tasks 5–11 must be written against the code, not against the
task text above.

| What the plan said | What the tree does, and why |
|---|---|
| `_cast_ids(frames, id_dtype)` | Frames use `_cast_frame_ids`, which asks **`frame_id_dtype()`** — a cached one-row probe through the real producer. A frame set carries a **ball row with NaN ids** and numpy `int64` cannot hold NaN, so the producer's concat must widen; forcing `int64` raises `IntCastingNaNError`. The widened dtype is **not a constant** (see finding #1 below), so it is derived, never tabulated. |
| `_player_layout -> pd.DataFrame`, walked with `iterrows()` | Returns `list[dict]`, walked with `enumerate`. `iterrows()`' index is `Hashable`, needing an `int()` cast at every use; pyright covers `tests` (`pyproject.toml:227`) and this idiom alone produced most of an initial **54 errors**. |
| `times = np.arange(t0 - half, ...)` | `times = t0 + np.arange(-n, n+1) * step`. `np.arange` from a large origin accumulates float drift and put the anchor **2.8e-12** off `t0`, making per-linked-frame position equality approximate. Integer-step construction makes it exactly **0.0**. |

**Measured constants** (use these; do not re-derive): window allowlist finds 4 parameters, max
`1.5` → `required_neighbourhood_seconds() == 3.0`. Leg A is **138** rows (6 actions × 23), Leg B
**8418**. Audited surface is **33 `add_*` + 5 boundary = 38**.

### Audit finding #1 (library, real) — recorded not fixed, belongs in the Task 8 report

`snapshot_to_tracking_frames`' id-column dtype is **pandas-version-dependent**. Measured on
both interpreters, same call:

| snapshots id dtype | pandas 2.3.3 (`.venv`, py3.10) | pandas 3.0.3 (`.venv312`, py3.12) |
|---|---|---|
| `int64` | `float64` | `float64` |
| **`Int64`** | **`Int64`** | **`Float64`** |
| `object` | `object` | `object` |

The frame set carries a ball row with NA ids, so the producer's `pd.concat`
(`_snapshot.py:172`) must widen — and *what it widens a nullable integer to changed in
pandas 3*. This is the `FutureWarning` that call emits, materialising. Hyrum's law: any
consumer pinning the `Int64` output dtype breaks on pandas 3. The repo already tracks a
sibling (DAS all-NaN on pandas 3).

**How it surfaced, and the rule it produced:** `.venv` passed; `.venv312` failed the full suite
with `assert 'Float64' == 'Int64'`, because an earlier draft **hardcoded a table measured on one
venv**. A measurement taken on one interpreter is not a constant. **Run the full suite on
`.venv312`, never only `.venv`.**

**RED gate shape:** the two surface gates are `@pytest.mark.xfail(strict=True)`, so they are
observed failing now AND turn into XPASS→failure the moment Task 7 populates the registry,
forcing deletion of the marker. Same mechanism as `MirrorEntry.known_defect`. Task 7 must
delete both markers.

### ⚠ Adjudication caveat for Task 7 — `differs` has TWO possible causes

The legs necessarily carry **different total frame counts** (Leg A 138 rows, Leg B 8418): Leg A
is one frame per action, Leg B a ±3 s neighbourhood. Measured consequence:

| Stub | Observation |
|---|---|
| reads `speed` | `all_nan` (Leg A speed is NaN by construction) |
| reads only `x`, aggregated over **all** frames | **`differs`** |

So a feature that reads frames **globally** rather than through `links` observes `differs` for a
purely structural reason, with no velocity dependence whatsoever. Every registered aggregator
takes `links=` and should sample only its linked frame — but that is an assumption about 33
functions, not a guarantee.

**Task 7 must therefore establish, per `differs` column, WHICH cause applies** before writing
`silent_degrade`. The mandatory rationale is where that goes, and this is precisely why the
design refuses to let a machine adjudicate. A cheap discriminator: re-run the entry against a
Leg B truncated to only its anchor frames — if `differs` disappears, the cause was frame-count,
not velocity.

---

### Task 1: State vocabulary and its completeness gate

The anti-rot spine. Lands first so every later table is checked against it. Five consecutive spec revisions acquired the same defect — a state introduced at one level and not propagated into every table claiming completeness — and this gate is what stops the sixth.

**Files:**
- Create: `tests/sb360/__init__.py`
- Create: `tests/sb360/_vocabulary.py`
- Test: `tests/sb360/test_vocabulary_completeness.py`

**Interfaces:**
- Consumes: nothing
- Produces: `CALL_OUTCOMES`, `ROW_CLASSES`, `OBSERVATIONS`, `KINDS`, `ADJUDICATIONS`, `APPLICABILITY` (all `frozenset[str]`); `OBSERVATION_KIND: dict[str, str]`; `ADMISSIBLE_FROM: dict[str, frozenset[str]]`; `PRECEDENCE: tuple[tuple[int, str], ...]`; `ROW_CLASS_CONSUMERS: dict[str, frozenset[str]]`; `ROW_CLASSIFICATION_PRECONDITION: str`; `SHARED_NAMES: frozenset[str]`; `RATIONALE_ALWAYS: frozenset[str]`; `RATIONALE_CONDITIONAL: dict[str, str]`

- [ ] **Step 1: Create the package marker**

```python
# tests/sb360/__init__.py
"""SB360 coverage audit gates. See docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md."""
```

- [ ] **Step 2: Write the failing completeness gate**

```python
# tests/sb360/test_vocabulary_completeness.py
"""The gate that stops the design acquiring a sixth hole.

Five spec revisions acquired the SAME defect -- a state introduced at one level and not
propagated into every table that claims completeness. Reviewing again finds the next
instance; it does not stop the next instance. This does.
"""

from __future__ import annotations

from tests.sb360 import _vocabulary as V


def test_every_call_outcome_is_placed():
    """Each call outcome is a precedence rule OR the declared row-classification precondition."""
    in_precedence = {name for _, name in V.PRECEDENCE}
    placed = in_precedence | {V.ROW_CLASSIFICATION_PRECONDITION}
    missing = V.CALL_OUTCOMES - placed
    assert not missing, (
        f"call_outcome(s) {sorted(missing)} appear in no precedence rule and are not the "
        f"declared row-classification precondition. This is the rev-4 defect: raises_b was "
        f"defined at Level 1 and absent from the table claiming to be the complete procedure."
    )


def test_every_row_class_is_consumed():
    consumed = set().union(*V.ROW_CLASS_CONSUMERS.values()) if V.ROW_CLASS_CONSUMERS else set()
    assert set(V.ROW_CLASS_CONSUMERS) == set(V.ROW_CLASSES), (
        f"ROW_CLASS_CONSUMERS keys {sorted(V.ROW_CLASS_CONSUMERS)} != ROW_CLASSES "
        f"{sorted(V.ROW_CLASSES)}"
    )
    assert consumed <= V.OBSERVATIONS, f"unknown observations referenced: {sorted(consumed - V.OBSERVATIONS)}"
    unconsumed = [rc for rc, obs in V.ROW_CLASS_CONSUMERS.items() if not obs]
    assert not unconsumed, f"row_class(es) {sorted(unconsumed)} are consumed by no precedence rule"


def test_every_observation_carries_a_kind():
    producible = {name for _, name in V.PRECEDENCE}
    missing = producible - set(V.OBSERVATION_KIND)
    assert not missing, f"observation(s) {sorted(missing)} carry no kind"
    bad = {o: k for o, k in V.OBSERVATION_KIND.items() if k not in V.KINDS}
    assert not bad, f"unknown kind(s): {bad}"


def test_adjudicated_and_budgeted_observations_are_admissible_somewhere():
    reachable = set().union(*V.ADMISSIBLE_FROM.values()) if V.ADMISSIBLE_FROM else set()
    for obs, kind in V.OBSERVATION_KIND.items():
        if kind in {"adjudicated", "budgeted"}:
            assert obs in reachable, (
                f"observation {obs!r} has kind {kind!r} so it reaches the registry, but no "
                f"adjudication admits it -- it would be unadjudicatable"
            )


def test_terminal_observations_are_absent_from_admissibility():
    reachable = set().union(*V.ADMISSIBLE_FROM.values()) if V.ADMISSIBLE_FROM else set()
    for obs, kind in V.OBSERVATION_KIND.items():
        if kind == "terminal_fixture_failure":
            assert obs not in reachable, (
                f"observation {obs!r} is a terminal fixture failure and must never reach the "
                f"registry, but an adjudication admits it"
            )


def test_every_adjudication_is_reachable():
    assert set(V.ADMISSIBLE_FROM) == set(V.ADJUDICATIONS), (
        f"ADMISSIBLE_FROM keys != ADJUDICATIONS: "
        f"{sorted(set(V.ADMISSIBLE_FROM) ^ set(V.ADJUDICATIONS))}"
    )
    orphans = [adj for adj, obs in V.ADMISSIBLE_FROM.items() if not obs]
    assert not orphans, f"adjudication(s) {sorted(orphans)} are reachable from no observation"


def test_shared_names_are_declared():
    """A name in two vocabularies is deliberate or it is a bug. Nothing in between."""
    overlap = V.CALL_OUTCOMES & V.OBSERVATIONS
    assert overlap == V.SHARED_NAMES, (
        f"call_outcome/observation name overlap {sorted(overlap)} != declared "
        f"SHARED_NAMES {sorted(V.SHARED_NAMES)}. Rev 4 had observation 'raises' and "
        f"adjudication 'raises' meaning different things; namespacing plus this "
        f"assertion is what makes reuse expressible rather than accidental."
    )
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python -m pytest tests/sb360/test_vocabulary_completeness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.sb360._vocabulary'`

- [ ] **Step 4: Write the vocabulary**

```python
# tests/sb360/_vocabulary.py
"""The SB360 audit's state vocabulary -- declared ONCE, namespaced by kind.

Namespacing is not tidiness. Rev 4 of the spec had an observation named ``raises`` and an
adjudication named ``raises`` denoting DIFFERENT things, while ``raises_b`` appeared in two
vocabularies denoting the SAME thing. A flat name set conflates the first and cannot express
that the second is deliberate.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

# --- call_outcome.* -------------------------------------------------------------------
CALL_OUTCOMES: frozenset[str] = frozenset({"raises_a", "raises_b", "both_succeeded"})

#: The one call outcome that is NOT a precedence rule: it is the gate into row classification.
ROW_CLASSIFICATION_PRECONDITION: str = "both_succeeded"

# --- row_class.* ----------------------------------------------------------------------
ROW_CLASSES: frozenset[str] = frozenset(
    {"row_identical", "row_differs", "row_nan_a", "row_nan_b", "row_nan_both"}
)

# --- observation.* --------------------------------------------------------------------
OBSERVATIONS: frozenset[str] = frozenset(
    {
        "raises_a",
        "raises_b",
        "leg_b_declined",
        "no_signal",
        "all_nan",
        "partial_nan",
        "differs",
        "identical",
    }
)

#: Names shared between call_outcome and observation because the state passes straight
#: through unchanged. DECLARED, not incidental -- see test_shared_names_are_declared.
SHARED_NAMES: frozenset[str] = frozenset({"raises_a", "raises_b"})

# --- kind.* ---------------------------------------------------------------------------
KINDS: frozenset[str] = frozenset({"terminal_fixture_failure", "budgeted", "adjudicated"})

OBSERVATION_KIND: dict[str, str] = {
    "raises_a": "adjudicated",
    "raises_b": "terminal_fixture_failure",
    "leg_b_declined": "terminal_fixture_failure",
    "no_signal": "budgeted",
    "all_nan": "adjudicated",
    "partial_nan": "adjudicated",
    "differs": "adjudicated",
    "identical": "adjudicated",
}

# --- precedence -----------------------------------------------------------------------
#: (rank, observation). First match wins. Ranks 1-2 are mutually exclusive by their Level 1
#: definitions (raises_a is "Leg A raised, Leg B irrelevant"), so their relative order is
#: immaterial; 2 follows 1 so this cannot read as disagreeing with Level 1.
PRECEDENCE: tuple[tuple[int, str], ...] = (
    (1, "raises_a"),
    (2, "raises_b"),
    (3, "leg_b_declined"),
    (4, "no_signal"),
    (5, "all_nan"),
    (6, "partial_nan"),
    (7, "differs"),
    (8, "identical"),
)

#: Which precedence rules read each row class. row_nan_both is consumed by no_signal (via the
#: informative-set denominator) and by every rule that excludes it from that denominator.
ROW_CLASS_CONSUMERS: dict[str, frozenset[str]] = {
    "row_identical": frozenset({"identical"}),
    "row_differs": frozenset({"differs"}),
    "row_nan_a": frozenset({"all_nan", "partial_nan"}),
    "row_nan_b": frozenset({"leg_b_declined"}),
    "row_nan_both": frozenset({"no_signal"}),
}

# --- adjudication.* -------------------------------------------------------------------
ADJUDICATIONS: frozenset[str] = frozenset(
    {"works", "silent_degrade", "differs_by_design", "honest_nan", "not_exercised", "raises"}
)

ADMISSIBLE_FROM: dict[str, frozenset[str]] = {
    "works": frozenset({"identical"}),
    "silent_degrade": frozenset({"differs", "partial_nan"}),
    "differs_by_design": frozenset({"differs", "partial_nan"}),
    "honest_nan": frozenset({"all_nan", "partial_nan"}),
    "not_exercised": frozenset({"no_signal"}),
    "raises": frozenset({"raises_a"}),
}

#: Adjudications that ALWAYS require a written rationale.
RATIONALE_ALWAYS: frozenset[str] = frozenset(
    {"silent_degrade", "differs_by_design", "not_exercised"}
)

#: Adjudications that require a rationale only under a stated condition.
RATIONALE_CONDITIONAL: dict[str, str] = {
    # Loosening a tolerance converts `differs` into `identical`, which manufactures a
    # rationale-free `works`. This is the half of that mitigation that lives in the vocabulary.
    "works": "tolerance is non-default",
    "honest_nan": "observation is partial_nan",
}

# --- applicability.* ------------------------------------------------------------------
APPLICABILITY: frozenset[str] = frozenset(
    {"region_support", "no_support", "support_data_defined"}
)
```

- [ ] **Step 5: Run the gate to verify it passes**

Run: `python -m pytest tests/sb360/test_vocabulary_completeness.py -v`
Expected: 7 passed

- [ ] **Step 6: Prove the gate is not vacuous**

Temporarily delete the `(2, "raises_b")` entry from `PRECEDENCE`, re-run, and confirm `test_every_call_outcome_is_placed` FAILS naming `raises_b`. Then restore it. This reproduces the exact rev-4 defect and confirms the gate catches it.

Run: `python -m pytest tests/sb360/test_vocabulary_completeness.py::test_every_call_outcome_is_placed -v`
Expected (with the line deleted): FAIL — `call_outcome(s) ['raises_b'] appear in no precedence rule`

- [ ] **Step 7: Commit**

```bash
git add tests/sb360/__init__.py tests/sb360/_vocabulary.py tests/sb360/test_vocabulary_completeness.py
git commit -m "test(sb360): state vocabulary + completeness gate"
```

---

### Task 2: Comparison primitives

**Files:**
- Create: `tests/sb360/_compare.py`
- Test: `tests/sb360/test_compare.py`

**Interfaces:**
- Consumes: `tests/sb360/_vocabulary.py`
- Produces:
  - `DEFAULT_RTOL: float = 1e-9`, `DEFAULT_ATOL: float = 1e-12`
  - `classify_row(a, b, *, is_float: bool, rtol: float, atol: float) -> str` → a `row_class`
  - `aggregate_column(counts: dict[str, int]) -> str` → an `observation`
  - `compare_column(leg_a: pd.Series, leg_b: pd.Series, *, rtol, atol) -> tuple[str, dict[str, int]]`
  - `DtypeMismatch(AssertionError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sb360/test_compare.py
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from tests.sb360 import _vocabulary as V
from tests.sb360._compare import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DtypeMismatch,
    ShapeMismatch,
    aggregate_column,
    classify_row,
    compare_column,
)

_NAN = float("nan")


def test_row_classification_is_exhaustive_over_the_finite_nan_grid():
    """Every (Leg A, Leg B) combination lands in exactly one declared row class."""
    values = [1.0, 2.0, _NAN]
    seen = set()
    for a, b in itertools.product(values, values):
        cls = classify_row(a, b, is_float=True, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
        assert cls in V.ROW_CLASSES, f"({a}, {b}) produced undeclared class {cls!r}"
        seen.add(cls)
    assert seen == V.ROW_CLASSES, f"unreached row classes: {sorted(V.ROW_CLASSES - seen)}"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1.0, 1.0, "row_identical"),
        (1.0, 2.0, "row_differs"),
        (_NAN, 1.0, "row_nan_a"),
        (1.0, _NAN, "row_nan_b"),
        (_NAN, _NAN, "row_nan_both"),
    ],
)
def test_row_classification_cases(a, b, expected):
    assert classify_row(a, b, is_float=True, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL) == expected


def test_non_float_compares_exactly():
    """A tolerance on an integer count would silently absorb an off-by-one."""
    assert classify_row(3, 4, is_float=False, rtol=1.0, atol=1.0) == "row_differs"
    assert classify_row(3, 3, is_float=False, rtol=0.0, atol=0.0) == "row_identical"


def test_aggregation_precedence_leg_b_declined_beats_everything_else():
    counts = {"row_identical": 100, "row_differs": 0, "row_nan_a": 5, "row_nan_b": 1, "row_nan_both": 3}
    assert aggregate_column(counts) == "leg_b_declined"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"row_nan_both": 4}, "no_signal"),
        ({"row_nan_a": 3, "row_nan_both": 2}, "all_nan"),
        ({"row_nan_a": 3, "row_identical": 2}, "partial_nan"),
        ({"row_differs": 1, "row_identical": 5}, "differs"),
        ({"row_identical": 5}, "identical"),
        # The sparse-domain column: NaN in BOTH legs off-domain, identical on-domain.
        # Rev 4 tightened `identical` and orphaned exactly this.
        ({"row_identical": 2, "row_nan_both": 40}, "identical"),
    ],
)
def test_aggregation_cases(counts, expected):
    full = {rc: 0 for rc in V.ROW_CLASSES}
    full.update(counts)
    assert aggregate_column(full) == expected


def test_aggregation_is_total_over_reachable_tallies():
    """No tally of row classes falls through to a default. The rev-2/rev-4 hole, asserted."""
    for combo in itertools.product([0, 1, 2], repeat=len(V.ROW_CLASSES)):
        counts = dict(zip(sorted(V.ROW_CLASSES), combo))
        if sum(counts.values()) == 0:
            continue
        obs = aggregate_column(counts)
        assert obs in V.OBSERVATIONS, f"tally {counts} produced undeclared observation {obs!r}"


def test_dtype_mismatch_fails_loudly_rather_than_casting():
    """int64 vs object is the ADR-019 trap: an implicit cast makes a real defect read identical."""
    a = pd.Series([1, 2, 3], dtype="int64")
    b = pd.Series(["1", "2", "3"], dtype="object")
    with pytest.raises(DtypeMismatch, match="int64.*object"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_all_nan_upcast_is_not_a_dtype_mismatch():
    """An integer count that DECLINES on freeze-frames is the desirable `all_nan` outcome.

    pandas cannot hold NaN in int64, so Leg A upcasts to float64 against an int64 Leg B. A
    naive dtype-first guard aborts the audit on exactly the honest-degradation case it most
    wants to record.
    """
    a = pd.Series([np.nan, np.nan, np.nan], dtype="float64")
    b = pd.Series([1, 2, 3], dtype="int64")
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "all_nan"
    assert counts["row_nan_a"] == 3


def test_partial_nan_upcast_is_not_a_dtype_mismatch():
    """The ADJACENT case, and the one an all-NaN-only exemption misses.

    An integer column declining on SOME rows leaves BOTH legs populated with different
    declared dtypes -- so a guard exempting only the all-NaN case fires here, aborting on
    `partial_nan`, which the spec calls the expected outcome on the visibility axis.
    """
    a = pd.Series([1.0, np.nan, 3.0], dtype="float64")
    b = pd.Series([1, 2, 3], dtype="int64")
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"
    assert counts["row_nan_a"] == 1
    assert counts["row_identical"] == 2


def test_bool_column_declining_to_object_is_not_a_dtype_mismatch():
    """pandas has no NaN in numpy bool either, so a declining bool column becomes object."""
    a = pd.Series([True, None, False], dtype="object")
    b = pd.Series([True, True, False], dtype="bool")
    obs, _ = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"


def test_numeric_versus_string_still_raises():
    """The exemption must not swallow the trap it exists for -- partially-NaN or not."""
    a = pd.Series([1.0, np.nan, 3.0], dtype="float64")
    b = pd.Series(["1", "2", "3"], dtype="object")
    with pytest.raises(DtypeMismatch, match="numeric.*other"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_length_mismatch_fails_rather_than_truncating():
    """zip() truncates to the shorter series and reports a confident observation computed
    from a PREFIX -- the audit's core primitive carrying the defect class the audit exists
    to find."""
    a = pd.Series([1.0, 2.0])
    b = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ShapeMismatch, match="2 rows.*3 rows"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_index_mismatch_fails_even_at_equal_length():
    """Equal length is not enough: a re-indexed leg would compare row i against a different
    action entirely, and every value would still line up positionally."""
    a = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    b = pd.Series([1.0, 2.0, 3.0], index=[0, 2, 1])
    with pytest.raises(ShapeMismatch, match="index"):
        compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)


def test_compare_column_returns_observation_and_tally():
    a = pd.Series([1.0, np.nan, 3.0])
    b = pd.Series([1.0, 2.0, 3.0])
    obs, counts = compare_column(a, b, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
    assert obs == "partial_nan"
    assert counts["row_nan_a"] == 1
    assert counts["row_identical"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sb360/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.sb360._compare'`

- [ ] **Step 3: Implement**

```python
# tests/sb360/_compare.py
"""Row classification and column aggregation for the paired-leg comparison.

Two levels, because a raise is not a row property: if a call raises there is no output frame,
so there are no rows to classify. Call outcomes are resolved by the harness BEFORE this module
is reached; everything here assumes ``both_succeeded``.
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


class DtypeMismatch(AssertionError):
    """Legs disagree on a column's dtype. Never reconciled silently -- see ADR-019."""


class ShapeMismatch(AssertionError):
    """Legs disagree on row count or index. Comparing anyway would use a silent prefix."""


_NUMERIC_INFERRED = frozenset({"integer", "floating", "mixed-integer-float", "decimal", "complex"})
_BOOL_INFERRED = frozenset({"boolean"})


def _value_kind(s: pd.Series) -> str | None:
    """Kind of a column's NON-NULL values, or None when it has none.

    Content-inferred rather than dtype-declared, so a NaN-forced upcast (int64 -> float64,
    bool -> object) cannot reach the comparison at all.
    """
    vals = s.dropna()
    if vals.empty:
        return None
    inferred = pd.api.types.infer_dtype(vals, skipna=True)
    if inferred in _NUMERIC_INFERRED:
        return "numeric"
    if inferred in _BOOL_INFERRED:
        return "boolean"
    return "other"


def classify_row(a, b, *, is_float: bool, rtol: float, atol: float) -> str:
    # NOTE: `math.isclose` semantics, NOT numpy's. math uses a SYMMETRIC
    # max(rel_tol*max(|a|,|b|), abs_tol); np.isclose uses the asymmetric atol + rtol*|b|.
    # Immaterial at 1e-9, but do not assume numpy behaviour when tuning a per-column override.
    a_nan = pd.isna(a)
    b_nan = pd.isna(b)
    if a_nan and b_nan:
        return "row_nan_both"
    if a_nan:
        return "row_nan_a"
    if b_nan:
        return "row_nan_b"
    if is_float:
        same = math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    else:
        same = bool(a == b)
    return "row_identical" if same else "row_differs"


def aggregate_column(counts: dict[str, int]) -> str:
    """Collapse a row-class tally to one observation, by declared precedence.

    ``row_nan_both`` rows are UNINFORMATIVE -- neither leg said anything -- so they leave the
    denominator rather than counting as agreement. That single choice handles both the
    unexercised column (all uninformative -> ``no_signal``) and the sparse-domain column
    (uninformative off-domain, compared on-domain -> its real observation). An earlier draft
    instead TIGHTENED ``identical`` and orphaned the second case entirely.
    """
    get = lambda rc: int(counts.get(rc, 0))  # noqa: E731

    if get("row_nan_b"):
        return "leg_b_declined"

    total = sum(int(v) for v in counts.values())
    informative = total - get("row_nan_both")
    if informative == 0:
        return "no_signal"

    nan_a = get("row_nan_a")
    if nan_a == informative:
        return "all_nan"
    if nan_a:
        return "partial_nan"
    if get("row_differs"):
        return "differs"
    return "identical"


def compare_column(
    leg_a: pd.Series, leg_b: pd.Series, *, rtol: float, atol: float
) -> tuple[str, dict[str, int]]:
    """Compare one column across legs. Shape and dtype are checked BEFORE any value comparison."""
    if len(leg_a) != len(leg_b):
        raise ShapeMismatch(
            f"leg A has {len(leg_a)} rows, leg B has {len(leg_b)} rows. Refusing to compare: "
            f"zip() would truncate to the shorter and report a confident observation computed "
            f"from a PREFIX. An aggregator dropping unlinked actions is the likely cause."
        )
    if not leg_a.index.equals(leg_b.index):
        raise ShapeMismatch(
            "legs have equal length but a different index, so row i would be compared against "
            "a different action. Re-align before comparing."
        )

    # Compare the KIND of the actual values, never the declared dtype.
    #
    # The trap this guard exists for is ADR-019's int64-vs-object -- a numeric-versus-
    # non-numeric difference. int64 vs float64 is not that, and it is exactly what a NaN
    # forces: pandas cannot hold NaN in int64, so an integer column that declines on SOME
    # Leg A rows upcasts to float64 while Leg B stays int64. A declared-dtype guard fires on
    # that, aborting the audit on `partial_nan` -- which the spec calls the EXPECTED outcome
    # on the visibility axis, not an edge case.
    #
    # Inferring the kind from non-null values subsumes all-NaN, partial-NaN and the real trap
    # under ONE rule, instead of accruing an exemption per case. An earlier draft exempted
    # all-NaN only and landed one case short of the one it was reasoning about.
    kind_a, kind_b = _value_kind(leg_a), _value_kind(leg_b)
    if kind_a is not None and kind_b is not None and kind_a != kind_b:
        raise DtypeMismatch(
            f"leg A values are {kind_a} (dtype {leg_a.dtype}), leg B values are {kind_b} "
            f"(dtype {leg_b.dtype}). Not reconciled: an implicit cast is how a real ADR-019 "
            f"dtype defect reads as `identical`."
        )

    # A leg with no values contributes only row_nan_a/row_nan_both, so no value comparison
    # runs and the reference dtype is safely taken from the populated leg.
    is_float = pd.api.types.is_float_dtype(
        leg_a.dtype if kind_a is not None else leg_b.dtype
    )
    counts = {
        "row_identical": 0,
        "row_differs": 0,
        "row_nan_a": 0,
        "row_nan_b": 0,
        "row_nan_both": 0,
    }
    for a, b in zip(leg_a.to_numpy(), leg_b.to_numpy()):
        counts[classify_row(a, b, is_float=is_float, rtol=rtol, atol=atol)] += 1
    return aggregate_column(counts), counts
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sb360/test_compare.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add tests/sb360/_compare.py tests/sb360/test_compare.py
git commit -m "test(sb360): row classification + column aggregation primitives"
```

---

### Task 3: Paired fixture

**Files:**
- Create: `tests/sb360/_fixture.py`
- Test: `tests/sb360/test_fixture.py`

**Interfaces:**
- Consumes: `silly_kicks.tracking.snapshot_to_tracking_frames`
- Produces:
  - `FIXTURE_VERSION: str` — bump on any fixture change; appears in every lock failure message
  - `HOME_TEAM_ID`, `AWAY_TEAM_ID`
  - `build_leg_a(*, roster: str = "full", id_dtype: str = "int64") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` → `(actions, frames, links)`
  - `build_leg_b(*, roster: str = "full", id_dtype: str = "int64") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - `required_neighbourhood_seconds() -> float`
  - `visible_area_side_table(...) -> pd.DataFrame` — `action_id` → polygon; harness-only, never passed to `snapshot_to_tracking_frames`
  - roster tokens: `"full"`, `"gk_absent"`, `"defender_absent"`

- [ ] **Step 1: Write the failing fixture preconditions**

```python
# tests/sb360/test_fixture.py
"""Fixture preconditions. Each is load-bearing for a verdict, so each asserts rather than hopes."""

from __future__ import annotations

import numpy as np
import pytest

from tests.sb360 import _fixture as F


def test_leg_a_is_built_by_the_real_producer():
    """Leg A must come from snapshot_to_tracking_frames, so the fixture cannot drift from it
    and so the audit exercises the code path real SB360 data will hit."""
    _, frames, _ = F.build_leg_a()
    assert (frames["source_provider"] == "snapshot").all()
    assert (frames["speed_source"] == "unavailable").all()
    assert "vx" not in frames.columns and "vy" not in frames.columns


def test_positions_match_at_linked_frames():
    """The velocity axis holds ROSTER and POSITION fixed; only kinematics vary. Otherwise a
    verdict confounds position with velocity and is unattributable."""
    _, frames_a, links_a = F.build_leg_a()
    _, frames_b, links_b = F.build_leg_b()

    for action_id in links_a["action_id"]:
        fid_a = links_a.loc[links_a["action_id"] == action_id, "frame_id"].iloc[0]
        fid_b = links_b.loc[links_b["action_id"] == action_id, "frame_id"].iloc[0]
        a = frames_a[(frames_a["frame_id"] == fid_a) & (~frames_a["is_ball"])]
        b = frames_b[(frames_b["frame_id"] == fid_b) & (~frames_b["is_ball"])]
        a = a.sort_values("player_id").reset_index(drop=True)
        b = b.sort_values("player_id").reset_index(drop=True)
        assert list(a["player_id"]) == list(b["player_id"]), f"roster differs at action {action_id}"
        np.testing.assert_allclose(a["x"].to_numpy(), b["x"].to_numpy(), atol=1e-12)
        np.testing.assert_allclose(a["y"].to_numpy(), b["y"].to_numpy(), atol=1e-12)


def test_window_discovery_actually_finds_the_known_window():
    """Non-vacuity for the scan itself. `add_actor_pre_window.pre_seconds` (features.py:864)
    is the one window we know exists; if the scan misses it, the fixture length is a fallback
    dressed as a measurement."""
    found = F.discovered_windows()
    assert "add_actor_pre_window.pre_seconds" in found, (
        f"scan missed the known window. Found: {sorted(found)}"
    )
    assert F.required_neighbourhood_seconds() >= 2.0 * found["add_actor_pre_window.pre_seconds"]
    # The allowlist must stay an allowlist: tau_seconds is an influence-DECAY constant, not a
    # frame window, and a substring heuristic admits it.
    assert not [k for k in found if k.endswith(".tau_seconds")], (
        f"scan admitted a decay constant: {sorted(k for k in found if 'tau' in k)}"
    )


def test_leg_b_neighbourhood_covers_the_longest_enumerated_window():
    """If Leg B is shorter than a feature's window, that feature is NaN in BOTH legs ->
    no_signal -> not_exercised, and its structurally_impossible annotation becomes
    INADMISSIBLE. The distinction is then silently lost."""
    _, frames_b, links_b = F.build_leg_b()
    required = F.required_neighbourhood_seconds()
    for action_id in links_b["action_id"]:
        fid = links_b.loc[links_b["action_id"] == action_id, "frame_id"].iloc[0]
        t0 = frames_b.loc[frames_b["frame_id"] == fid, "time_seconds"].iloc[0]
        period = frames_b.loc[frames_b["frame_id"] == fid, "period_id"].iloc[0]
        same_period = frames_b[frames_b["period_id"] == period]
        before = t0 - same_period["time_seconds"].min()
        after = same_period["time_seconds"].max() - t0
        assert before >= required, f"action {action_id}: only {before}s before, need {required}s"
        assert after >= required, f"action {action_id}: only {after}s after, need {required}s"


def test_leg_b_motion_is_non_degenerate():
    """Constant velocity makes every acceleration-dependent quantity identically zero in BOTH
    legs, which reads as a false `works`."""
    _, frames_b, _ = F.build_leg_b()
    players = frames_b[~frames_b["is_ball"]]
    moving = players[players["player_id"] == players["player_id"].iloc[0]].sort_values("time_seconds")
    speeds = moving["speed"].to_numpy()
    headings = np.arctan2(moving["vy"].to_numpy(), moving["vx"].to_numpy())
    assert np.nanstd(speeds) > 1e-3, f"speed is constant (std={np.nanstd(speeds)}) -- degenerate"
    assert np.nanstd(headings) > 1e-3, f"heading is constant (std={np.nanstd(headings)}) -- degenerate"


@pytest.mark.parametrize("id_dtype", ["int64", "Int64", "object"])
def test_id_dtype_parameterization_round_trips(id_dtype):
    """A hand-built fixture silently picks one id dtype and can mask a real ADR-019 defect."""
    _, frames_a, _ = F.build_leg_a(id_dtype=id_dtype)
    _, frames_b, _ = F.build_leg_b(id_dtype=id_dtype)
    assert str(frames_a["team_id"].dtype) == id_dtype
    assert frames_a["team_id"].dtype == frames_b["team_id"].dtype


@pytest.mark.parametrize("roster", ["full", "gk_absent", "defender_absent"])
def test_roster_variants_hold_velocity_fixed(roster):
    """The visibility axis varies roster at FIXED velocity -- the mirror of the velocity axis."""
    _, frames_a, _ = F.build_leg_a(roster=roster)
    players = frames_a[~frames_a["is_ball"]]
    if roster == "gk_absent":
        assert not players["is_goalkeeper"].any(), "gk_absent variant still contains a keeper"
    elif roster == "full":
        assert players["is_goalkeeper"].sum() >= 2, "full variant should carry both keepers"
    assert (frames_a["speed_source"] == "unavailable").all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sb360/test_fixture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.sb360._fixture'`

- [ ] **Step 3: Implement the fixture**

Build a 6-action scene (pass, cross, shot, goalkick, dribble, throw_in) across 22 players plus ball, in SPADL action-LTR. Leg A goes through the real producer; Leg B is a 10 Hz frame set spanning `required_neighbourhood_seconds()` either side of each action, carrying `vx`/`vy`/`speed` from a non-degenerate trajectory whose position at the linked frame equals Leg A's exactly.

```python
# tests/sb360/_fixture.py
"""Paired Leg A / Leg B fixture for the SB360 audit.

Leg A is built by CALLING ``snapshot_to_tracking_frames`` -- never hand-assembled -- so the
fixture cannot drift from the producer and the audit exercises the path real SB360 data hits.

FIXTURE_VERSION is surfaced in every observation-lock failure message: the lock pins the
fixture as well as the library, so "the fixture changed" and "the library regressed" must be
distinguishable at the point of failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import link_actions_to_frames, snapshot_to_tracking_frames

FIXTURE_VERSION = "sb360-fixture-1"

HOME_TEAM_ID = 1
AWAY_TEAM_ID = 2

_FRAME_HZ = 10.0
_GAME_ID = 7
_PERIOD = 1

#: (action_id, type_name, team_id, player_id, start_x, start_y, end_x, end_y, time_seconds)
_ACTIONS = [
    (0, "pass", HOME_TEAM_ID, 10, 52.5, 34.0, 70.0, 40.0, 300.0),
    (1, "cross", HOME_TEAM_ID, 11, 88.0, 8.0, 98.0, 34.0, 320.0),
    (2, "shot", HOME_TEAM_ID, 12, 95.0, 34.0, 105.0, 34.0, 340.0),
    (3, "goalkick", AWAY_TEAM_ID, 20, 5.5, 34.0, 45.0, 20.0, 360.0),
    (4, "dribble", HOME_TEAM_ID, 13, 60.0, 50.0, 68.0, 52.0, 380.0),
    (5, "throw_in", AWAY_TEAM_ID, 21, 40.0, 68.0, 48.0, 55.0, 400.0),
]


#: ALLOWLIST of parameter names that are genuinely temporal frame windows.
#:
#: A denylist was tried and is wrong on both sides. Matching any name containing "seconds"
#: admitted `add_gk_influence.tau_seconds` and `add_player_influence.tau_seconds`, which are
#: influence-DECAY constants rather than frame windows -- so the scan measured the wrong set
#: and returned the right answer by luck. Worse, a denylist is open at the top: a future
#: `timeout_seconds=600.0` anywhere in `tracking.__all__` would silently inflate Leg B into
#: the millions of rows. Fail-safe defaults, the rule the repo applies elsewhere.
#:
#: Measured 2026-08-04 across `tracking.__all__`: `add_actor_pre_window.pre_seconds=0.5`,
#: `add_elastic_sync.window_seconds=1.0`, `add_off_ball_context.pre_seconds=1.5`,
#: `add_off_ball_runs.pre_seconds=1.5` -> max 1.5.
_WINDOW_PARAM_NAMES = frozenset({"pre_seconds", "post_seconds", "window_seconds"})

#: Upper bound on a single discovered window. Leg B is 10 Hz x 6 actions x 23 rows, so its
#: row count scales as ~5500 x this value; 10 s is ~55k rows, 600 s would be ~3.3M.
_MAX_PLAUSIBLE_WINDOW_S = 10.0


def discovered_windows() -> dict[str, float]:
    """Every temporal-window default across the enumerated aggregators, by ``func.param``."""
    import inspect

    import silly_kicks.tracking as T

    found: dict[str, float] = {}
    for fn_name in (n for n in T.__all__ if n.startswith("add_")):
        fn = getattr(T, fn_name)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        for pname, p in params.items():
            if pname not in _WINDOW_PARAM_NAMES:
                continue
            if isinstance(p.default, (int, float)) and not isinstance(p.default, bool):
                found[f"{fn_name}.{pname}"] = float(p.default)
    return found


def required_neighbourhood_seconds() -> float:
    """Longest window among ALL enumerated features, read from the library, never hardcoded.

    Scans every ``add_*`` rather than one function, because the claim is "longest among
    enumerated features" and reading a single signature does not establish it. Both bounds
    are asserted: an empty result means the scan silently returned a fallback, and an
    implausible maximum means it admitted something that is not a frame window.
    """
    windows = discovered_windows()
    assert windows, (
        f"no temporal-window parameter discovered across tracking.__all__ from allowlist "
        f"{sorted(_WINDOW_PARAM_NAMES)}. The fixture would fall back to a hardcoded value "
        f"while claiming to read the library. A new window parameter must be added here."
    )
    longest = max(windows.values())
    assert longest <= _MAX_PLAUSIBLE_WINDOW_S, (
        f"longest discovered window is {longest}s (> {_MAX_PLAUSIBLE_WINDOW_S}s): "
        f"{ {k: v for k, v in windows.items() if v == longest} }. Leg B would balloon to "
        f"roughly {int(5500 * longest):,} rows. Either the allowlist admitted a "
        f"non-window parameter, or the fixture needs a deliberate redesign."
    )
    # 2x headroom so a boundary-inclusive implementation still has frames to consume.
    return 2.0 * longest


def _player_layout(roster: str) -> pd.DataFrame:
    """22 players in a plausible shape, one keeper per side."""
    rows = []
    for i in range(11):
        rows.append(
            {
                "player_id": 10 + i,
                "team_id": HOME_TEAM_ID,
                "is_goalkeeper": i == 0,
                "base_x": 5.0 if i == 0 else 30.0 + (i % 4) * 18.0,
                "base_y": 34.0 if i == 0 else 8.0 + (i % 5) * 13.0,
            }
        )
    for i in range(11):
        rows.append(
            {
                "player_id": 20 + i,
                "team_id": AWAY_TEAM_ID,
                "is_goalkeeper": i == 0,
                "base_x": 100.0 if i == 0 else 60.0 + (i % 4) * 11.0,
                "base_y": 34.0 if i == 0 else 10.0 + (i % 5) * 12.0,
            }
        )
    out = pd.DataFrame(rows)
    if roster == "gk_absent":
        out = out[~out["is_goalkeeper"]]
    elif roster == "defender_absent":
        # Drop one outfield away player far from the action: the extreme-member case.
        out = out[out["player_id"] != 24]
    return out.reset_index(drop=True)


def _actions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": aid,
                "game_id": _GAME_ID,
                "period_id": _PERIOD,
                "time_seconds": t,
                "team_id": team,
                "player_id": pid,
                "type_name": tname,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
            }
            for aid, tname, team, pid, sx, sy, ex, ey, t in _ACTIONS
        ]
    )


def _offset(action_id: int, player_index: int, t: float) -> tuple[float, float]:
    """Non-degenerate trajectory: speed AND heading both vary with time.

    A constant-velocity path zeroes every acceleration-dependent quantity in BOTH legs, which
    the comparison would read as `identical` and the audit would record as `works`.
    """
    w = 0.35 + 0.05 * player_index
    return (
        3.0 * np.sin(w * t + action_id),
        2.0 * np.sin(0.5 * w * t + 0.7 * player_index),
    )


def _cast_ids(df: pd.DataFrame, id_dtype: str) -> pd.DataFrame:
    out = df.copy()
    for col in ("team_id", "player_id"):
        if col in out.columns:
            out[col] = out[col].astype(id_dtype)
    return out


def build_leg_a(*, roster: str = "full", id_dtype: str = "int64"):
    """Freeze-frame leg: one synthetic frame per action, via the real producer."""
    # Cast BEFORE the producer sees it, not after. Casting only on the way out means the
    # producer always runs on one dtype combination and the parameterization never exercises
    # the ADR-019 `ids_match` path it exists to probe.
    actions = _cast_ids(_actions_frame(), id_dtype)
    layout = _player_layout(roster)

    snap_rows = []
    for aid, _, _, _, _, _, _, _, t in _ACTIONS:
        for idx, p in layout.iterrows():
            dx, dy = _offset(aid, idx, t)
            snap_rows.append(
                {
                    "action_id": aid,
                    "team_id": p["team_id"],
                    "player_id": p["player_id"],
                    "is_goalkeeper": bool(p["is_goalkeeper"]),
                    "x": float(p["base_x"] + dx),
                    "y": float(p["base_y"] + dy),
                }
            )
    snapshots = _cast_ids(pd.DataFrame(snap_rows), id_dtype)
    frames, links = snapshot_to_tracking_frames(snapshots, actions)
    return actions, frames, links


def build_leg_b(*, roster: str = "full", id_dtype: str = "int64"):
    """Velocity-bearing leg: same positions at the linked frame, plus a real neighbourhood."""
    actions = _cast_ids(_actions_frame(), id_dtype)
    layout = _player_layout(roster)
    half = required_neighbourhood_seconds()
    step = 1.0 / _FRAME_HZ

    rows = []
    frame_id = 0
    for aid, _, _, _, sx, sy, _, _, t0 in _ACTIONS:
        times = np.arange(t0 - half, t0 + half + step / 2, step)
        for t in times:
            for idx, p in layout.iterrows():
                # Position matches Leg A exactly at t == t0 by construction.
                dx, dy = _offset(aid, idx, t)
                # Analytic derivative of _offset, so velocity is consistent with position.
                w = 0.35 + 0.05 * idx
                vx = 3.0 * w * np.cos(w * t + aid)
                vy = 2.0 * 0.5 * w * np.cos(0.5 * w * t + 0.7 * idx)
                rows.append(
                    {
                        "game_id": _GAME_ID,
                        "period_id": _PERIOD,
                        "frame_id": frame_id,
                        "time_seconds": float(t),
                        "frame_rate": _FRAME_HZ,
                        "player_id": p["player_id"],
                        "team_id": p["team_id"],
                        "is_ball": False,
                        "is_goalkeeper": bool(p["is_goalkeeper"]),
                        "x": float(p["base_x"] + dx),
                        "y": float(p["base_y"] + dy),
                        "z": np.nan,
                        "speed": float(np.hypot(vx, vy)),
                        "vx": float(vx),
                        "vy": float(vy),
                        "speed_source": "derived",
                        "ball_state": "alive",
                        "team_attacking_direction": "ltr",
                        "confidence": np.nan,
                        "visibility": np.nan,
                        "source_provider": "synthetic",
                        "is_goalkeeper_source": "native",
                    }
                )
            rows.append(
                {
                    "game_id": _GAME_ID,
                    "period_id": _PERIOD,
                    "frame_id": frame_id,
                    "time_seconds": float(t),
                    "frame_rate": _FRAME_HZ,
                    "player_id": np.nan,
                    "team_id": np.nan,
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": float(sx),
                    "y": float(sy),
                    "z": np.nan,
                    "speed": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed_source": "derived",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": np.nan,
                    "visibility": np.nan,
                    "source_provider": "synthetic",
                    "is_goalkeeper_source": "native",
                }
            )
            frame_id += 1

    frames = _cast_ids(pd.DataFrame(rows), id_dtype)

    # Link with the REAL linker, for the same reason Leg A uses the real producer: a
    # hand-built five-column table drifts the moment the linkage contract changes, and the
    # LinkReport gives a free assertion that every action actually found a frame.
    links, report = link_actions_to_frames(actions, frames)
    assert len(links) == len(actions), (
        f"only {len(links)}/{len(actions)} actions linked in Leg B "
        f"(report={report}). An unlinked action produces NaN geometry that would be "
        f"misread as a library property."
    )
    return actions, frames, links


def visible_area_side_table(*, fraction: float = 1.0) -> pd.DataFrame:
    """Synthetic camera polygons, keyed by ``action_id``.

    A HARNESS-ONLY side table. The ``snapshots`` contract is NOT extended and
    ``snapshot_to_tracking_frames`` is NOT modified -- stating the seam here closes the route
    by which scope creeps into a public contract.
    """
    rows = []
    for aid, _, _, _, sx, _, _, _, _ in _ACTIONS:
        half_len = 52.5 * fraction
        rows.append(
            {
                "action_id": aid,
                "polygon": [
                    (max(0.0, sx - half_len), 0.0),
                    (min(105.0, sx + half_len), 0.0),
                    (min(105.0, sx + half_len), 68.0),
                    (max(0.0, sx - half_len), 68.0),
                ],
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sb360/test_fixture.py -v`
Expected: all passed. If `test_positions_match_at_linked_frames` fails, the anchor-frame detection in `build_leg_b` is off by a half-step — check `is_anchor` tolerance against `step`.

- [ ] **Step 5: Commit**

```bash
git add tests/sb360/_fixture.py tests/sb360/test_fixture.py
git commit -m "test(sb360): versioned paired Leg A/Leg B fixture"
```

---

### Task 4: Registry schema and RED surface gate

**Files:**
- Create: `tests/sb360/_registry.py`
- Create: `tests/sb360/_entries/__init__.py`
- Test: `tests/sb360/test_registry_surface.py`

**Interfaces:**
- Consumes: `_vocabulary`, `_compare`
- Produces:
  - `AxisVerdict` dataclass: `observation: str`, `adjudication: str`, `rationale: str | None`, `counts: dict[str, int] | None`
  - `Sb360Entry` dataclass: `name`, `call`, `columns: tuple[str, ...]`, `velocity: dict[str, AxisVerdict]`, `visibility: dict[str, AxisVerdict]`, `applicability: dict[str, str]`, `tolerances: dict[str, tuple[float, float]]`, `tolerance_basis: dict[str, str]`, `structurally_impossible: dict[str, str]`
  - `SB360_ENTRIES: dict[str, Sb360Entry]`
  - `_entry(...)` helper
  - `NOT_EXERCISED_BUDGET: int` — pre-registered locked count
  - `audited_surface() -> set[str]`

- [ ] **Step 1: Write the failing surface gate**

```python
# tests/sb360/test_registry_surface.py
"""Both-directions pin of the SB360 registry to the public surface.

Lands RED: the registry is empty, so every export is unregistered. A gate written after its
own repair arrives green and is never observed to work (ADR-051).
"""

from __future__ import annotations

import silly_kicks.tracking as T
from tests.sb360 import _vocabulary as V
from tests.sb360._registry import (
    NOT_EXERCISED_BUDGET,
    SB360_ENTRIES,
    VISIBILITY_ROSTERS,
    audited_surface,
    iter_verdicts,
)


def _public_add_star() -> set[str]:
    return {n for n in T.__all__ if n.startswith("add_")}


def test_every_public_add_star_is_registered():
    missing = _public_add_star() - set(SB360_ENTRIES)
    assert not missing, (
        f"{len(missing)} public add_* export(s) carry no SB360 verdict: {sorted(missing)}. "
        f"Register them or CI stays red -- this is the anti-rot property the 2026-05-27 "
        f"compatibility table lacked."
    )


def test_no_registry_entry_names_a_missing_export():
    extra = set(SB360_ENTRIES) - audited_surface()
    assert not extra, f"registry names non-exported function(s): {sorted(extra)}"


def test_every_visibility_roster_has_its_own_slot():
    for name, entry in SB360_ENTRIES.items():
        assert set(entry.visibility) == set(VISIBILITY_ROSTERS), (
            f"{name}: visibility keys {sorted(entry.visibility)} != "
            f"{sorted(VISIBILITY_ROSTERS)}. Each roster needs its own verdict -- a shared "
            f"slot cannot represent a feature that survives a missing outfielder and "
            f"collapses on a missing keeper."
        )


def test_every_verdict_is_admissible_from_its_observation():
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, verdict in iter_verdicts(entry):
            admissible = V.ADMISSIBLE_FROM[verdict.adjudication]
            assert verdict.observation in admissible, (
                f"{name}.{col} ({axis}/{roster}): adjudication {verdict.adjudication!r} is "
                f"not admissible from observation {verdict.observation!r}"
            )


def test_rationales_are_present_where_required():
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, v in iter_verdicts(entry):
            needs = v.adjudication in V.RATIONALE_ALWAYS
            if v.adjudication == "honest_nan" and v.observation == "partial_nan":
                needs = True
            if v.adjudication == "works" and col in entry.tolerances:
                needs = True
            if needs:
                assert v.rationale, (
                    f"{name}.{col} ({axis}/{roster}): adjudication {v.adjudication!r} "
                    f"requires a written rationale and has none"
                )


def test_tolerance_overrides_carry_a_basis():
    for name, entry in SB360_ENTRIES.items():
        for col in entry.tolerances:
            assert entry.tolerance_basis.get(col), (
                f"{name}.{col}: tolerance override with no basis. Loosening a tolerance "
                f"converts `differs` into `identical` and manufactures a `works` verdict."
            )


def test_structural_impossibility_co_occurs_with_all_nan_or_raises():
    """Checked on EVERY axis. A structurally impossible feature cannot become possible
    because a defender left the frame."""
    for name, entry in SB360_ENTRIES.items():
        for axis, roster, col, v in iter_verdicts(entry):
            if col not in entry.structurally_impossible:
                continue
            assert v.observation in {"all_nan", "raises_a"}, (
                f"{name}.{col} ({axis}/{roster}) is annotated structurally_impossible but "
                f"observes {v.observation!r}. The annotation is falsifiable by construction "
                f"-- this is the contradiction."
            )


def test_not_exercised_count_is_within_its_locked_budget():
    actual = sum(
        1
        for e in SB360_ENTRIES.values()
        for _axis, _roster, _col, v in iter_verdicts(e)
        if v.adjudication == "not_exercised"
    )
    assert actual == NOT_EXERCISED_BUDGET, (
        f"{actual} not_exercised verdict(s) against a locked budget of "
        f"{NOT_EXERCISED_BUDGET}. A fixture inadequacy must be acknowledged deliberately, "
        f"never allowed to grow quietly (ADR-052: a bounded pass logs what it dropped)."
    )
```

- [ ] **Step 2: Run to verify it fails RED**

Run: `python -m pytest tests/sb360/test_registry_surface.py -v`
Expected: FAIL — `33 public add_* export(s) carry no SB360 verdict`. **Record the exact count in the commit message**; it is the enumeration this gate exists to hold.

- [ ] **Step 3: Implement the registry schema (deliberately empty)**

```python
# tests/sb360/_registry.py
"""The SB360 verdict registry.

Each entry carries TWO independent observation/adjudication pairs -- velocity and visibility --
because a feature can be sound on one axis and fabricated on the other.

The CI gate locks the OBSERVATION, never the adjudication. Repair a function and the
observation changes, CI fails, and the human judgement is forced to be revisited. Locking the
adjudication instead would pretend a machine can adjudicate; locking neither is the rot the
2026-05-27 compatibility table demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import silly_kicks.tracking as _T


@dataclass(frozen=True)
class AxisVerdict:
    observation: str
    adjudication: str
    rationale: str | None = None
    counts: dict[str, int] | None = None


#: Roster variants the visibility axis sweeps. Each gets its OWN verdict slot: a feature can
#: survive a missing outfielder and collapse on a missing keeper, which is the entire reason
#: both are swept. An earlier draft had ONE visibility dict for both rosters, so whenever they
#: genuinely differed -- the case of interest -- one parametrization had to fail.
VISIBILITY_ROSTERS: tuple[str, ...] = ("gk_absent", "defender_absent")


@dataclass(frozen=True)
class Sb360Entry:
    name: str
    call: Callable  # (actions, frames, links, home_team_id) -> pd.DataFrame
    columns: tuple[str, ...]
    velocity: dict[str, AxisVerdict] = field(default_factory=dict)
    #: roster -> column -> verdict
    visibility: dict[str, dict[str, AxisVerdict]] = field(default_factory=dict)
    applicability: dict[str, str] = field(default_factory=dict)
    #: col -> {"extreme": delta, "near": delta}. Recorded so a zero-movement classification is
    #: VISIBLE: a `no_support` derived from two zero deltas is indistinguishable from a probe
    #: that silently failed to perturb anything.
    applicability_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    #: col -> (rtol, atol). Absent means the default; presence means an override.
    tolerances: dict[str, tuple[float, float]] = field(default_factory=dict)
    tolerance_basis: dict[str, str] = field(default_factory=dict)
    structurally_impossible: dict[str, str] = field(default_factory=dict)


SB360_ENTRIES: dict[str, Sb360Entry] = {}

#: Pre-registered count of (entry, axis, column) triples adjudicated `not_exercised`.
#: Raised only with a recorded reason; it is a budget, not a tally.
NOT_EXERCISED_BUDGET = 0


def _entry(
    name,
    call,
    columns,
    *,
    velocity=None,
    visibility=None,
    applicability=None,
    applicability_deltas=None,
    tolerances=None,
    tolerance_basis=None,
    structurally_impossible=None,
) -> None:
    SB360_ENTRIES[name] = Sb360Entry(
        name=name,
        call=call,
        columns=tuple(columns),
        velocity=velocity or {},
        visibility=visibility or {},
        applicability=applicability or {},
        applicability_deltas=applicability_deltas or {},
        tolerances=tolerances or {},
        tolerance_basis=tolerance_basis or {},
        structurally_impossible=structurally_impossible or {},
    )


def iter_verdicts(entry):
    """Yield ``(axis, roster, column, verdict)`` for every recorded verdict.

    THE single iteration seam. Every gate walks the registry through this, so a gate cannot
    silently disagree with the schema about how verdicts are keyed -- which is exactly how the
    one-visibility-dict-for-two-rosters defect arose.
    """
    for col, v in entry.velocity.items():
        yield ("velocity", "full", col, v)
    for roster, cols in entry.visibility.items():
        for col, v in cols.items():
            yield ("visibility", roster, col, v)


def audited_surface() -> set[str]:
    """Every name this audit is allowed to register.

    `tracking.__all__`'s add_* exports, plus the enumerated boundary cases that consume frames
    without living in `tracking`.
    """
    names = {n for n in _T.__all__ if n.startswith("add_")}
    names |= {
        "gkdv.build_ghost_frames",
        "gkdv.delta_das",
        "gkdv.delta_threat_suppression",
        "xtgk.compute_xt_gk_v2",
        "spadl.add_restart_coordinates",
    }
    return names


def _load_entry_modules() -> None:
    """Import every per-family entry module for its registration side effects."""
    import importlib
    import pkgutil

    from tests.sb360 import _entries

    for mod in pkgutil.iter_modules(_entries.__path__):
        importlib.import_module(f"tests.sb360._entries.{mod.name}")


_load_entry_modules()
```

```python
# tests/sb360/_entries/__init__.py
"""Per-family SB360 registry entries. Mirrors tests/tracking/_mirror_entries/."""
```

- [ ] **Step 4: Run and confirm the gate is still RED for the right reason**

Run: `python -m pytest tests/sb360/test_registry_surface.py -v`
Expected: `test_every_public_add_star_is_registered` FAILS listing all 33; the other six PASS vacuously over an empty registry. That asymmetry is correct — they become meaningful in Task 7.

- [ ] **Step 5: Commit the RED gate**

```bash
git add tests/sb360/_registry.py tests/sb360/_entries/__init__.py tests/sb360/test_registry_surface.py
git commit -m "test(sb360): registry schema + RED surface gate (33 unregistered)"
```

---

### Task 5: Harness

**Files:**
- Create: `tests/sb360/_harness.py`
- Test: `tests/sb360/test_harness.py`

**Interfaces:**
- Consumes: `_compare`, `_fixture`, `_registry`
- Produces:
  - `run_axis(entry, *, axis: str, roster: str = "full") -> dict[str, AxisVerdict]` — observations only, adjudication left `""`
  - `CallOutcomeError(AssertionError)` — raised for `raises_b` and `leg_b_declined`

- [ ] **Step 1: Write the failing harness tests**

```python
# tests/sb360/test_harness.py
from __future__ import annotations

import pandas as pd
import pytest

from tests.sb360 import _vocabulary as V
from tests.sb360._harness import CallOutcomeError, run_axis
from tests.sb360._registry import Sb360Entry


def _stub(cols: dict[str, list]):
    def call(actions, frames, links, home_team_id):
        out = actions.copy()
        for c, vals in cols.items():
            out[c] = vals
        return out

    return call


def test_observation_is_produced_by_execution():
    entry = Sb360Entry(name="stub", call=_stub({"m": [1.0] * 6}), columns=("m",))
    got = run_axis(entry, axis="velocity")
    assert got["m"].observation in V.OBSERVATIONS
    assert got["m"].adjudication == "", "the harness observes; it must not adjudicate"


def test_leg_b_raise_is_a_fixture_failure_not_a_library_property():
    calls = {"n": 0}

    def call(actions, frames, links, home_team_id):
        calls["n"] += 1
        if calls["n"] == 2:  # Leg B
            raise RuntimeError("leg B blew up")
        return actions.assign(m=1.0)

    entry = Sb360Entry(name="stub", call=call, columns=("m",))
    with pytest.raises(CallOutcomeError, match="raises_b"):
        run_axis(entry, axis="velocity")


def test_leg_a_raise_is_recorded_as_an_observation():
    def call(actions, frames, links, home_team_id):
        raise ValueError("nope")

    entry = Sb360Entry(name="stub", call=call, columns=("m",))
    got = run_axis(entry, axis="velocity")
    assert got["m"].observation == "raises_a"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sb360/test_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.sb360._harness'`

- [ ] **Step 3: Implement**

```python
# tests/sb360/_harness.py
"""Runs one registry entry on both legs and returns OBSERVATIONS only.

The harness never adjudicates. That separation is the whole design: the machine half is
re-derived and locked on every CI run, the human half carries the judgement and its rationale.
"""

from __future__ import annotations

from tests.sb360 import _fixture as F
from tests.sb360._compare import DEFAULT_ATOL, DEFAULT_RTOL, compare_column
from tests.sb360._registry import AxisVerdict


class CallOutcomeError(AssertionError):
    """A fixture-integrity failure: `raises_b` or `leg_b_declined`. Never a library property."""


def run_axis(entry, *, axis: str, roster: str = "full") -> dict[str, AxisVerdict]:
    # BOTH legs take the same roster. The velocity axis holds roster fixed and varies
    # kinematics; the visibility axis holds kinematics fixed and varies roster. Passing
    # different rosters to the two legs would vary both at once and make every verdict
    # unattributable -- the confound Layer B's 2x2 was built to avoid.
    effective_roster = roster if axis == "visibility" else "full"
    actions_a, frames_a, links_a = F.build_leg_a(roster=effective_roster)
    actions_b, frames_b, links_b = F.build_leg_b(roster=effective_roster)

    try:
        out_a = entry.call(actions_a, frames_a, links_a, F.HOME_TEAM_ID)
    except Exception:
        return {c: AxisVerdict(observation="raises_a", adjudication="") for c in entry.columns}

    try:
        out_b = entry.call(actions_b, frames_b, links_b, F.HOME_TEAM_ID)
    except Exception as exc:
        raise CallOutcomeError(
            f"{entry.name}: call outcome `raises_b` on fixture {F.FIXTURE_VERSION} -- Leg B "
            f"raised where Leg A succeeded ({exc!r}). This is a FIXTURE defect and is never "
            f"recorded as a library property."
        ) from exc

    result: dict[str, AxisVerdict] = {}
    for col in entry.columns:
        rtol, atol = entry.tolerances.get(col, (DEFAULT_RTOL, DEFAULT_ATOL))
        obs, counts = compare_column(out_a[col], out_b[col], rtol=rtol, atol=atol)
        if obs == "leg_b_declined":
            raise CallOutcomeError(
                f"{entry.name}.{col}: observation `leg_b_declined` on fixture "
                f"{F.FIXTURE_VERSION} -- {counts['row_nan_b']} of "
                f"{sum(counts.values())} rows are NaN on Leg B where Leg A is finite. "
                f"The richer leg yielded less; the comparison is broken for those rows. "
                f"Sample row classes: {counts}."
            )
        result[col] = AxisVerdict(observation=obs, adjudication="", counts=counts)
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sb360/test_harness.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/sb360/_harness.py tests/sb360/test_harness.py
git commit -m "test(sb360): paired-leg harness producing observations"
```

---

### Task 6: Applicability probes

**Files:**
- Create: `tests/sb360/_probes.py`
- Test: `tests/sb360/test_applicability_probes.py`

**Interfaces:**
- Consumes: `_fixture`
- Produces: `derive_applicability(entry, column: str) -> tuple[str, dict[str, float]]` — returns the class plus each probe's measured delta

- [ ] **Step 1: Write the failing probe tests**

```python
# tests/sb360/test_applicability_probes.py
"""The class is DERIVED, not declared. A human picking one of three categories would put a
declaration inside the locked half of the registry -- the exact conflation this design exists
to prevent."""

from __future__ import annotations

import numpy as np

from tests.sb360 import _vocabulary as V
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import Sb360Entry


def _hull_like(actions, frames, links, home_team_id):
    """Support defined BY the visible players: sensitive to an extreme member."""
    players = frames[~frames["is_ball"]]
    span = players["x"].max() - players["x"].min()
    return actions.assign(m=float(span))


def _region_like(actions, frames, links, home_team_id):
    """Fixed query region from action geometry: indifferent to a player never inside it."""
    players = frames[~frames["is_ball"]]
    near = players[(players["x"] - 52.5).abs() < 5.0]
    return actions.assign(m=float(len(near)))


def _scalar_like(actions, frames, links, home_team_id):
    return actions.assign(m=actions["start_x"].astype(float))


def test_extreme_displacement_identifies_data_defined_support():
    entry = Sb360Entry(name="hull", call=_hull_like, columns=("m",))
    cls, deltas = derive_applicability(entry, "m")
    assert cls == "support_data_defined", f"got {cls} with deltas {deltas}"
    assert deltas["extreme"] > 0, "probe 1 must measurably move, or the classification is vacuous"


def test_near_displacement_separates_region_from_scalar():
    region_cls, region_deltas = derive_applicability(
        Sb360Entry(name="region", call=_region_like, columns=("m",)), "m"
    )
    scalar_cls, scalar_deltas = derive_applicability(
        Sb360Entry(name="scalar", call=_scalar_like, columns=("m",)), "m"
    )
    assert region_cls == "region_support", f"got {region_cls} with {region_deltas}"
    assert scalar_cls == "no_support", f"got {scalar_cls} with {scalar_deltas}"
    assert np.isclose(scalar_deltas["extreme"], 0.0)
    assert np.isclose(scalar_deltas["near"], 0.0)


def test_every_class_is_producible():
    """Vocabulary invariant 7, exercised rather than asserted in prose."""
    produced = set()
    for call in (_hull_like, _region_like, _scalar_like):
        cls, _ = derive_applicability(Sb360Entry(name="x", call=call, columns=("m",)), "m")
        produced.add(cls)
    assert produced == V.APPLICABILITY, f"unreached classes: {sorted(V.APPLICABILITY - produced)}"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sb360/test_applicability_probes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.sb360._probes'`

- [ ] **Step 3: Implement**

```python
# tests/sb360/_probes.py
"""Derive a column's visible-area applicability class by perturbation.

Both probes move player POSITIONS at fixed roster and fixed polygon, so neither collapses into
the other. Masking by polygon would BE roster variation and would discriminate nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.sb360 import _fixture as F

#: A player this far from the action is outside any plausible query region.
_EXTREME_SHIFT_M = 12.0
_NEAR_SHIFT_M = 3.0
_MOVED_DELTA = 1e-9


def _value(entry, column: str, frames: pd.DataFrame, actions, links) -> float:
    out = entry.call(actions, frames, links, F.HOME_TEAM_ID)
    return float(pd.to_numeric(out[column], errors="coerce").fillna(0.0).sum())


def _shift(frames: pd.DataFrame, *, extreme: bool) -> pd.DataFrame:
    out = frames.copy()
    players = out[~out["is_ball"]]
    if players.empty:
        return out
    centre_x = players["x"].mean()
    dist = (players["x"] - centre_x).abs()
    target = dist.idxmax() if extreme else dist.idxmin()
    shift = _EXTREME_SHIFT_M if extreme else _NEAR_SHIFT_M
    direction = 1.0 if extreme else -1.0
    out.loc[target, "x"] = float(np.clip(out.loc[target, "x"] + direction * shift, 0.0, 105.0))
    return out


def derive_applicability(entry, column: str) -> tuple[str, dict[str, float]]:
    """Return ``(applicability_class, {"extreme": delta, "near": delta})``.

    Probe 1 runs FIRST and WINS. A feature can satisfy both, and data-defined support is the
    dangerous property: it is the one where a coverage fraction reads as reassurance while
    being circular, because the hull over visible players is 100% observed by construction.
    """
    actions, frames, links = F.build_leg_a()
    base = _value(entry, column, frames, actions, links)

    extreme_delta = abs(_value(entry, column, _shift(frames, extreme=True), actions, links) - base)
    near_delta = abs(_value(entry, column, _shift(frames, extreme=False), actions, links) - base)
    deltas = {"extreme": extreme_delta, "near": near_delta}

    if extreme_delta > _MOVED_DELTA:
        return "support_data_defined", deltas
    if near_delta > _MOVED_DELTA:
        return "region_support", deltas
    return "no_support", deltas
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/sb360/test_applicability_probes.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/sb360/_probes.py tests/sb360/test_applicability_probes.py
git commit -m "test(sb360): applicability-class perturbation probes"
```

---

### Task 7: Populate the registry — the audit itself

This is where the findings come from. Every observation is produced by running the aggregator; every adjudication is written by a human reading the feature.

**Files:**
- Create: `tests/sb360/_entries/test_context.py`, `_gk.py`, `_space.py`, `_shape.py`, `_models.py`, `_boundary.py`
- Create: `tests/sb360/test_axis_locks.py`
- Modify: `tests/sb360/_registry.py` (`NOT_EXERCISED_BUDGET` only)

**Interfaces:**
- Consumes: `_harness.run_axis`, `_probes.derive_applicability`, `_registry._entry`
- Produces: a populated `SB360_ENTRIES` covering `audited_surface()`

- [ ] **Step 1: Write the observation-lock gate**

```python
# tests/sb360/test_axis_locks.py
"""The lock. Re-derives every observation and asserts it against the registry.

Repair `add_gk_influence` and the observation flips, CI fails, and the adjudication is forced
to be revisited. Locking the KEY SET alone -- the rev-1 defect -- would let a repaired function
keep a stale verdict while CI stayed green.
"""

from __future__ import annotations

import pytest

from tests.sb360 import _fixture as F
from tests.sb360._harness import run_axis
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import SB360_ENTRIES

#: (axis, roster). The velocity axis holds roster fixed; the visibility axis varies it at
#: fixed kinematics. Both visibility rosters are checked because a feature can survive a
#: missing outfielder and collapse on a missing keeper.
_AXES = [("velocity", "full"), ("visibility", "gk_absent"), ("visibility", "defender_absent")]


@pytest.mark.parametrize(("axis", "roster"), _AXES)
@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_observations_match_the_registry(name, axis, roster):
    entry = SB360_ENTRIES[name]
    observed = run_axis(entry, axis=axis, roster=roster)
    # Each visibility ROSTER has its own slot: a feature can survive a missing outfielder and
    # collapse on a missing keeper, so resolving both rosters to one dict would force a
    # failure whenever they genuinely differ -- the case both rosters exist to expose.
    recorded = entry.velocity if axis == "velocity" else entry.visibility[roster]
    for col, got in observed.items():
        expected = recorded[col].observation
        assert got.observation == expected, (
            f"{name}.{col} ({axis}/{roster}): observed {got.observation!r}, registry says "
            f"{expected!r}. Fixture {F.FIXTURE_VERSION}. If the FIXTURE changed, bump "
            f"FIXTURE_VERSION and re-record; if the LIBRARY changed, re-adjudicate. "
            f"Row classes: {got.counts}."
        )


@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_tolerance_overrides_target_float_columns_only(name):
    """A tolerance on an integer count is meaningless and would absorb an off-by-one."""
    import pandas as pd

    entry = SB360_ENTRIES[name]
    if not entry.tolerances:
        pytest.skip("no overrides")
    actions, frames, links = F.build_leg_a()
    out = entry.call(actions, frames, links, F.HOME_TEAM_ID)
    for col in entry.tolerances:
        assert pd.api.types.is_float_dtype(out[col].dtype), (
            f"{name}.{col} has a tolerance override but dtype is {out[col].dtype} -- "
            f"non-float columns compare exactly"
        )


@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_no_signal_is_acknowledged_on_every_axis(name):
    """Per-column liveness, per AXIS -- the spec's wording. An earlier draft inspected only
    entry.velocity, so a column dead on the visibility axis passed silently. Set equality
    guarantees stability, not meaningfulness: dead columns lock as `identical` forever."""
    entry = SB360_ENTRIES[name]
    for axis, roster, col, v in iter_verdicts(entry):
        if v.observation == "no_signal":
            assert v.adjudication == "not_exercised", (
                f"{name}.{col} ({axis}/{roster}): observed no_signal but is not adjudicated "
                f"not_exercised -- an unexercised column must be acknowledged, not absorbed"
            )


@pytest.mark.parametrize("name", sorted(SB360_ENTRIES))
def test_applicability_class_matches_the_registry(name):
    entry = SB360_ENTRIES[name]
    for col, expected in entry.applicability.items():
        got, deltas = derive_applicability(entry, col)
        assert got == expected, (
            f"{name}.{col}: probes derived {got!r}, registry says {expected!r}. "
            f"Measured deltas: {deltas}. Fixture {F.FIXTURE_VERSION}."
        )
        # A zero-delta `no_support` is indistinguishable from a probe that failed to run.
        # Any OTHER class is a positive claim and must be backed by a measurable movement.
        if expected != "no_support":
            probe = "extreme" if expected == "support_data_defined" else "near"
            assert deltas[probe] > 0.0, (
                f"{name}.{col}: class {expected!r} recorded but the {probe} probe moved "
                f"nothing ({deltas}). The classification would be vacuous."
            )
        assert entry.applicability_deltas.get(col) == pytest.approx(deltas, rel=1e-6), (
            f"{name}.{col}: recorded deltas {entry.applicability_deltas.get(col)} != "
            f"measured {deltas}. Deltas are recorded so a zero-movement classification is "
            f"visible rather than inferred."
        )


def test_the_canary_proves_the_legs_are_distinguishable():
    """Non-vacuity. NOT a `differs` canary -- that would require naming a silently-degrading
    column in advance, which is the audit's OUTPUT. `actor_speed` is the MODEL CITIZEN: NaN on
    Leg A (_snapshot.py:122 sets speed=NaN; _kernels.py:80,88 fills only where notna) against
    finite on Leg B. Anything other than `identical` proves distinguishability."""
    entry = SB360_ENTRIES["add_action_context"]
    observed = run_axis(entry, axis="velocity")
    got = observed["actor_speed"].observation
    assert got != "identical", (
        f"actor_speed observed {got!r} -- the legs are not distinguishable, so every "
        f"`identical` verdict in this audit is vacuous. Row classes: "
        f"{observed['actor_speed'].counts}."
    )
```

- [ ] **Step 2: Run the harness once per aggregator to READ the observations**

Write this scratch file (it is a tool, not a deliverable — do not commit it):

```python
# scratch_probe_sb360.py
"""Print measured observations + applicability per aggregator, for transcription.

The registry records what this prints. Nothing in it is predicted.
"""

from __future__ import annotations

import inspect
import sys
import traceback

import silly_kicks.tracking as T
from tests.sb360 import _fixture as F
from tests.sb360._harness import run_axis
from tests.sb360._probes import derive_applicability
from tests.sb360._registry import Sb360Entry


def wrap(fn):
    """Pass links/home_team_id only where the signature accepts them."""
    params = inspect.signature(fn).parameters

    def call(actions, frames, links, home_team_id):
        kwargs = {}
        if "links" in params:
            kwargs["links"] = links
        if "home_team_id" in params:
            kwargs["home_team_id"] = home_team_id
        return fn(actions, frames, **kwargs)

    return call


def emitted_columns(fn) -> list[str]:
    actions, frames, links = F.build_leg_a()
    out = wrap(fn)(actions, frames, links, F.HOME_TEAM_ID)
    return [c for c in out.columns if c not in actions.columns]


def main(names: list[str]) -> None:
    for name in names:
        fn = getattr(T, name)
        print(f"\n=== {name} ===")
        try:
            cols = emitted_columns(fn)
        except Exception:
            print("  LEG A RAISED during column discovery:")
            traceback.print_exc(limit=3)
            continue
        print(f"  columns: {cols}")
        entry = Sb360Entry(name=name, call=wrap(fn), columns=tuple(cols))
        for axis, roster in (("velocity", "full"), ("visibility", "gk_absent"),
                             ("visibility", "defender_absent")):
            try:
                observed = run_axis(entry, axis=axis, roster=roster)
            except Exception as exc:
                print(f"  {axis}/{roster}: HARNESS RAISED -> {exc}")
                continue
            for col, v in observed.items():
                print(f"  {axis}/{roster} {col}: {v.observation}  counts={v.counts}")
        for col in cols:
            try:
                cls, deltas = derive_applicability(entry, col)
                print(f"  applicability {col}: {cls}  deltas={deltas}")
            except Exception as exc:
                print(f"  applicability {col}: PROBE RAISED -> {exc}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv or sorted(n for n in T.__all__ if n.startswith("add_")))
```

Run it per family so the output stays readable:

```bash
python scratch_probe_sb360.py add_action_context add_pressure_on_actor add_actor_pre_window
```

Expected: a column list, three axis readings and an applicability class per column. **Transcribe these values verbatim into the entry modules.** Where `wrap` does not fit a signature (an aggregator needing an extra required argument), write that aggregator's wrapper by hand in its entry module rather than loosening `wrap` — a generic wrapper that silently guesses is how a wrong call becomes a recorded verdict.

Run `python -m pytest tests/sb360/test_axis_locks.py -v -k <family>` after each family lands, so a wrong transcription surfaces immediately rather than at the end.

- [ ] **Step 3: Write one entry module per family**

Group by compute home, not alphabetically, so a reviewer reads related features together. Example shape (`tests/sb360/_entries/_gk.py`):

```python
"""GK-domain entries. The collaboration's motivating surface."""

from __future__ import annotations

from silly_kicks.tracking import add_gk_influence, add_pre_shot_gk_position
from tests.sb360._registry import AxisVerdict, _entry


def _call_gk_influence(actions, frames, links, home_team_id):
    return add_gk_influence(actions, frames, links=links, home_team_id=home_team_id)


_entry(
    "add_gk_influence",
    _call_gk_influence,
    columns=(...),  # exact emitted columns, read from the executed output
    velocity={
        # RECORD THE MEASURED OBSERVATION, never the predicted one. This example is why:
        # every draft predicted `differs` here (add_gk_influence zero-fills vx/vy at
        # :205-206 and :406-408 and never consults the marker). MEASURED: `all_nan`, cause
        # isolated as velocity. It DECLINES rather than fabricating -> `honest_nan`. The
        # zero-fill is real and IS reached; a reachable code path is not evidence about the
        # value it produces.
        "...": AxisVerdict(
            observation="<measured>",
            adjudication="<written by a human>",
            rationale="<why this is fabrication rather than a weaker-but-valid model>",
        ),
    },
    visibility={...},
    applicability={...},
)
```

Repeat for every name in `audited_surface()`. `add_gradientsports_player_ids` is a jersey helper; record its expected observation as a **prediction that fails loudly if violated**, not as an exception carved out of the surface.

- [ ] **Step 4: Set the `not_exercised` budget to the measured count**

Modify `NOT_EXERCISED_BUDGET` in `_registry.py` to the number of `not_exercised` verdicts actually recorded, with a comment naming each one and why the fixture cannot exercise it.

- [ ] **Step 5: Run every Layer A gate green**

Run: `python -m pytest tests/sb360/ -v`
Expected: all passed, including the previously-RED surface gate.

- [ ] **Step 6: Run the full suite for regressions**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: no new failures. This adds tests only — a failure elsewhere means the audit imported something with a side effect.

- [ ] **Step 7: Commit**

```bash
git add tests/sb360/
git commit -m "test(sb360): populate verdict registry for the full audited surface"
```

---

### Task 8: Layer A report artifact

**Files:**
- Create: `docs/research/sb360_coverage/README.md`
- Create: `docs/research/sb360_coverage/behaviour_matrix.md`
- Create: `scripts/render_sb360_matrix.py`

**Interfaces:**
- Consumes: `SB360_ENTRIES`
- Produces: a rendered matrix — one row per (function, column, axis) with observation, adjudication, rationale, applicability class, tolerance

- [ ] **Step 1: Write the renderer**

```python
# scripts/render_sb360_matrix.py
"""Render the SB360 behaviour matrix from the verdict registry.

No provenance guard: this reads a COMMITTED registry and writes a doc, so there is no corpus
work whose inputs could be misattributed. Layer B (scripts/build_sb360_coverage.py) measures
real data and DOES take require_clean_tree.
"""

from __future__ import annotations

import argparse
import pathlib

from tests.sb360._fixture import FIXTURE_VERSION
from tests.sb360._registry import SB360_ENTRIES


def render() -> str:
    lines = [
        "# SB360 behaviour matrix",
        "",
        f"Fixture: `{FIXTURE_VERSION}`. Observations are re-derived and locked on every CI run;",
        "adjudications are human judgements carrying a written rationale.",
        "",
        "| Function | Column | Axis | Observation | Adjudication | Applicability | Rationale |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(SB360_ENTRIES):
        e = SB360_ENTRIES[name]
        for axis in ("velocity", "visibility"):
            for col, v in sorted(getattr(e, axis).items()):
                lines.append(
                    f"| `{name}` | `{col}` | {axis} | `{v.observation}` | "
                    f"`{v.adjudication}` | {e.applicability.get(col, '')} | "
                    f"{v.rationale or ''} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render and read the output**

Run: `python scripts/render_sb360_matrix.py --out docs/research/sb360_coverage/behaviour_matrix.md`
Expected: file written. Read it — this is the audit's answer to the original question.

- [ ] **Step 3: Write the README**

Cover: what the audit establishes, the observation/adjudication distinction, the count of each adjudication, and — prominently — every `silent_degrade` finding with its rationale, since those are the actionable results.

- [ ] **Step 4: Commit**

```bash
git add scripts/render_sb360_matrix.py docs/research/sb360_coverage/
git commit -m "docs(sb360): Layer A behaviour matrix + findings"
```

---

### Task 9: Layer B driver

> **COMPLETE. The code block in Step 4 below is the pre-implementation DRAFT and is now stale —
> read `scripts/build_sb360_coverage.py`, not this.** It is kept for the reasoning in its
> comments, not as a description of the shipped driver. A plan that inlines a full
> implementation rots by construction; this note is the fence.
>
> What the shipped driver does differently, all of it found by running against real data:
>
> | Draft | Shipped, and why |
> |---|---|
> | `actions["type_name"]` | **SPADL emits `type_id`.** The draft raised `KeyError: 'type_name'` on the first real match — the synthetic fixture carries `type_name` as a convenience column and the draft was written against that shape. Now `_type_id_to_name()`. |
> | one keeper rate | **Two.** `keeper AND NOT teammate` is definitionally zero on goal kicks and saves, where the keeper IS the actor. `_acting_side_gk_visible` added. |
> | per-frame metrics only | **`frame_existence_rate`, counted from the ACTION side.** Per-frame metrics cannot see an action that got no frame — for goal kicks that is the whole story. |
> | no retry | **`_retry`**, mirroring `validate_shot_goalmouth_sb.py`, which needed it at ~17 calls; this driver makes ~72. |
> | `assert` in `resolve_competition` | **`CompetitionMismatchError`.** `S101` is exempt for `tests/**` only, and an assert validating EXTERNAL data vanishes under `python -O`. |
> | `--out` required, example under `docs/research/` | **Defaults to top-level `sb360_coverage_shards/`.** The anchored `/*_shards/` glob at `.gitignore:90` does NOT cover a nested path — verified with `git check-ignore` — so the draft's example would have dirtied the tree and blocked the driver's own resume. |
> | `DROP_CELL` sentinel | **`tuple[bool, list | None]`** — a module-level `object()` sentinel does not type-narrow. |

**Files:**
- Create: `scripts/build_sb360_coverage.py`
- Test: `tests/scripts/test_build_sb360_coverage.py`

**Interfaces:**
- Consumes: `scripts/_driver.for_each`, `scripts/_provenance.{git_provenance, require_clean_tree}`
- Produces: per-match shards; `resolve_competition(comp_id, season_id) -> dict` with asserted names

- [ ] **Step 1: Write the failing driver tests**

```python
# tests/scripts/test_build_sb360_coverage.py
"""No __init__.py in this directory -- it would shadow the `scripts` namespace package."""

from __future__ import annotations

import pytest

import scripts.build_sb360_coverage as mod


def test_competition_ids_are_resolved_and_name_asserted():
    """Prose verification does not survive an upstream renumber."""
    catalogue = [
        {"competition_id": 44, "season_id": 107, "competition_name": "Major League Soccer",
         "season_name": "2023", "competition_gender": "male"},
    ]
    got = mod.resolve_competition(44, 107, catalogue=catalogue, expect_name="Major League Soccer")
    assert got["competition_gender"] == "male"

    with pytest.raises(AssertionError, match="expected .*Major League Soccer"):
        mod.resolve_competition(44, 107, catalogue=catalogue, expect_name="Bundesliga")


def test_missing_competition_raises_rather_than_sampling_silently():
    with pytest.raises(AssertionError, match="not found"):
        mod.resolve_competition(999, 1, catalogue=[], expect_name="x")


def test_driver_offers_allow_dirty_and_calls_require_clean_tree_from_main():
    """ADR-037, CI-gated by tests/scripts/test_provenance_wiring.py."""
    import ast
    import inspect

    src = inspect.getsource(mod)
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {
        n.func.id
        for n in ast.walk(main_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "require_clean_tree" in called
    assert "--allow-dirty" in src
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/scripts/test_build_sb360_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_sb360_coverage'`

- [ ] **Step 3: Confirm the statsbombpy return shapes BEFORE writing against them**

The driver's per-type breakdown and its keeper metric both depend on the exact shape of the
360 payload, and a wrong assumption here silently collapses the spec's headline Layer B
metric rather than failing. Confirm, do not assume:

```bash
.venv312/Scripts/python -c "
from statsbombpy import sb
ms = sb.matches(competition_id=44, season_id=107, fmt='dict')
mid = sorted(ms)[0] if isinstance(ms, dict) else sorted(m['match_id'] for m in ms)[0]
fr = sb.frames(match_id=mid, fmt='dict')
print('frames type:', type(fr))
first = next(iter(fr.values())) if isinstance(fr, dict) else fr[0]
print('frame keys:', sorted(first))
print('freeze_frame[0] keys:', sorted(first['freeze_frame'][0]))
print('visible_area len + sample:', len(first['visible_area']), first['visible_area'][:6])
ev = sb.events(match_id=mid, fmt='dict')
print('events type:', type(ev))
e0 = next(iter(ev.values())) if isinstance(ev, dict) else ev[0]
print('event keys sample:', sorted(e0)[:12])
"
```

Record the output in the commit message. Expected, per the SB360 open-data schema: each frame
record carries `event_uuid`, `visible_area` and `freeze_frame`, and **not** the event type —
type comes from joining `sb.events` on `event_uuid` ↔ `id`. Each `freeze_frame` player carries
`teammate`, `actor`, `keeper`, `location`. `visible_area` is a **flat** `[x0, y0, x1, y1, …]`
list in StatsBomb's **120×80** frame. If any of that differs, adjust Step 4 before writing it.

- [ ] **Step 4: Implement the driver**

```python
# scripts/build_sb360_coverage.py
"""Measure real StatsBomb 360 freeze-frame coverage across the three-cell design.

Corpus pass: adopts the ADR-052 seam (`for_each`, per-match shards, resumable) and the
ADR-037 provenance rule (`require_clean_tree` in `main()`, before any corpus work).

`statsbombpy` is imported lazily inside `work` so `--help` and the unit tests never need it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from scripts._driver import for_each
from scripts._provenance import git_provenance, require_clean_tree

#: (competition_id, season_id) -> expected name. Asserted at run time: prose verification
#: does not survive an upstream renumber.
EXPECTED_NAMES = {
    (72, 107): "Women's World Cup",
    (43, 106): "FIFA World Cup",
    (44, 107): "Major League Soccer",
}

#: SPADL types where the DEFENDING keeper is the metric's subject. These are SPADL names, and
#: they are matched against REAL SPADL actions produced by the converter -- never against
#: StatsBomb's own type taxonomy, which does not contain them.
#:
#: Verified in `silly_kicks/spadl/statsbomb.py`: StatsBomb has no `Cross` type (it is
#: `pass_cross == True`, :463), no `Goal Kick` type (it is `pass_type == "Goal Kick"`, :459),
#: and no penalty/freekick shot types (those are `shot.type.name`). Its actual keeper type is
#: `"Goal Keeper"` (:41). Matching this constant against StatsBomb names would therefore hit
#: only `pass` and `shot` -- i.e. most of the match -- while goal kicks, the spec's named
#: GK-domain event, folded into `Pass` and were never flagged.
GK_DOMAIN_TYPES = ("shot", "shot_penalty", "shot_freekick", "cross", "goalkick", "keeper_save")

#: Keys the converter reads from the top level; everything else rides in `extra`.
#: Mirrors `tests/test_xthreat_statsbomb_e2e.py::_adapt`.
_TOP_LEVEL_KEYS = frozenset({"id", "period", "timestamp", "team", "player", "type", "location"})


def _adapt_events(events: list[dict], match_id: int) -> "pd.DataFrame":  # noqa: F821
    """Raw StatsBomb event dicts -> the silly-kicks converter's input contract."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "type_name": (e.get("type") or {}).get("name"),
                "location": e.get("location"),
                "extra": {k: v for k, v in e.items() if k not in _TOP_LEVEL_KEYS},
            }
            for e in events
        ]
    )


def resolve_competition(comp_id: int, season_id: int, *, catalogue, expect_name: str) -> dict:
    """Resolve one competition/season and assert its NAME, not just its id."""
    for row in catalogue:
        if row["competition_id"] == comp_id and row["season_id"] == season_id:
            actual = row["competition_name"]
            assert actual == expect_name, (
                f"competition {comp_id}/{season_id} resolved to {actual!r}, expected "
                f"{expect_name!r} -- upstream ids have drifted; sampling would be silent"
            )
            return row
    raise AssertionError(f"competition {comp_id}/{season_id} not found in catalogue")


def _load_catalogue() -> list[dict]:
    from statsbombpy import sb  # noqa: PLC0415  -- lazy: --help must not need it

    comps = sb.competitions(fmt="dict")
    return list(comps.values()) if isinstance(comps, dict) else list(comps)


#: StatsBomb's pitch, NOT SPADL's 105x68. `visible_area` is delivered in this frame, so the
#: fraction must be normalised by this area -- dividing by 105*68 yields ~1.34 for a
#: fully-visible frame, i.e. a "fraction" above 1 reported to the club.
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0


def _values(payload):
    """statsbombpy returns dict-keyed-by-id or a list depending on call and version."""
    return list(payload.values()) if isinstance(payload, dict) else list(payload)


def measure_match(match) -> "pd.DataFrame":  # noqa: F821
    """One match -> tidy per-(SPADL action_type) coverage rows. Rates carry denominators."""
    import pandas as pd
    from statsbombpy import sb  # noqa: PLC0415

    from silly_kicks.spadl.statsbomb import convert_to_actions

    comp_id, season_id, match_id, home_team_id = match

    # Frame records carry event_uuid, visible_area and freeze_frame -- NOT the event type.
    # Without a join every frame buckets as "unknown" and the spec's headline metric
    # collapses to one row.
    #
    # The join runs through the REAL converter, for two reasons. The spec asks for coverage
    # "per SPADL action type", and StatsBomb's own taxonomy cannot express those types (see
    # GK_DOMAIN_TYPES). And it exercises the converter -- the path NWSL data will take -- for
    # the same reason Leg A is built by the real producer rather than by hand.
    events = _values(sb.events(match_id=match_id, fmt="dict"))
    actions, _report = convert_to_actions(_adapt_events(events, match_id), home_team_id)
    # statsbomb.py:235 sets original_event_id = events.event_id.astype(str).
    type_by_uuid = dict(
        zip(actions["original_event_id"].astype(str), actions["type_name"].astype(str))
    )

    per_type: dict[str, dict[str, float]] = {}
    for ff in _values(sb.frames(match_id=match_id, fmt="dict")):
        players = ff.get("freeze_frame") or []
        # "unmapped" rather than "unknown": a frame whose event the converter dropped as a
        # non_action is a REAL category, not a failed lookup, and conflating the two would
        # hide a broken join inside a legitimate bucket.
        type_name = type_by_uuid.get(str(ff.get("event_uuid")), "unmapped")

        # The DEFENDING keeper, which is what the spec asks for -- `keeper` alone answers
        # "a keeper is visible", a different question. freeze-frame flags are relative to the
        # ACTOR, so the defending keeper is keeper AND NOT teammate.
        defending_gk_visible = any(p.get("keeper") and not p.get("teammate") for p in players)

        bucket = per_type.setdefault(
            type_name,
            {
                "n_events": 0,
                "n_defending_gk_visible": 0,
                "sum_visible": 0.0,
                "sum_visible_area": 0.0,
            },
        )
        bucket["n_events"] += 1
        bucket["n_defending_gk_visible"] += int(defending_gk_visible)
        bucket["sum_visible"] += len(players)
        bucket["sum_visible_area"] += _visible_fraction(ff.get("visible_area") or [])

    rows = []
    for type_name, b in per_type.items():
        n = b["n_events"]
        rows.append(
            {
                "competition_id": comp_id,
                "season_id": season_id,
                "match_id": match_id,
                "action_type": type_name,
                # Denominators travel WITH every rate. A rate alone invites a reader to treat
                # an 8-event cell as an 800-event one.
                "n_events": n,
                "n_defending_gk_visible": b["n_defending_gk_visible"],
                "defending_gk_visible_rate": b["n_defending_gk_visible"] / n if n else float("nan"),
                # Roster completeness -- the honest quantity for a data-defined support, where
                # a coverage fraction is circular because the hull over visible players is
                # 100% observed by construction.
                "mean_players_visible": b["sum_visible"] / n if n else float("nan"),
                "mean_visible_pitch_fraction": b["sum_visible_area"] / n if n else float("nan"),
                "is_gk_domain": type_name.lower() in GK_DOMAIN_TYPES,
            }
        )
    return pd.DataFrame(rows)


def _visible_fraction(flat: list[float]) -> float:
    """Shoelace over StatsBomb's flat [x0, y0, x1, y1, ...], normalised by the SB pitch."""
    if len(flat) < 6:
        return 0.0
    xs, ys = flat[0::2], flat[1::2]
    n = len(xs)
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)))
    return area / (SB_PITCH_LENGTH * SB_PITCH_WIDTH)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--competitions", nargs="+", default=["72:107", "43:106", "44:107"])
    ap.add_argument("--matches-per-cell", type=int, default=8)
    ap.add_argument("--match-ids-json", type=pathlib.Path, default=None)
    ap.add_argument("--list-matches", action="store_true")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    # FIRST, before any corpus work: `git rev-parse HEAD` returns the same SHA whether or not
    # the tree is modified, so a driver stamping the bare SHA records a commit that does not
    # describe the code that ran (ADR-037).
    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    cells = [tuple(int(p) for p in c.split(":")) for c in args.competitions]
    catalogue = _load_catalogue()
    selected = [
        resolve_competition(c, s, catalogue=catalogue, expect_name=EXPECTED_NAMES[(c, s)])
        for c, s in cells
    ]

    matches = _iter_matches(selected, args)
    if args.list_matches:
        print(json.dumps([m[2] for m in matches]))
        return

    res = for_each(
        matches,
        key=lambda m: f"{m[0]}_{m[1]}_{m[2]}",
        work=measure_match,
        shard_root=args.out,
        token_inputs={
            "competitions": sorted(f"{c}:{s}" for c, s in cells),
            "matches_per_cell": args.matches_per_cell,
            "schema": "sb360-coverage-1",
        },
        tag=args.tag,
        label="match",
    )
    (args.out / f"manifest_{args.tag}.json").write_text(
        json.dumps({**res.manifest(), **prov}, indent=2), encoding="utf-8"
    )
    # CorpusPassResult (scripts/_driver.py:511-518) carries shard_dir, attempted, skipped,
    # failed, failures, counters, keys, counters_unrecorded. There is NO `processed` -- and
    # because main() is only AST-inspected by the unit tests, an AttributeError here would
    # land AFTER three cells x eight matches had been pulled over the network.
    print(
        f"attempted={res.attempted} processed={res.attempted - res.skipped - res.failed} "
        f"skipped={res.skipped} failed={res.failed}"
    )


def _iter_matches(selected, args):
    """STREAM, never list() -- the ADR-052 rule; a match pull is expensive per item.

    Yields ``(competition_id, season_id, match_id, home_team_id)``. The home team id is
    carried because `convert_to_actions` requires it and re-fetching the match row inside
    `work` would repeat a network call per item.
    """
    from statsbombpy import sb  # noqa: PLC0415

    override = json.loads(args.match_ids_json.read_text()) if args.match_ids_json else None
    for row in selected:
        c, s = row["competition_id"], row["season_id"]
        ids = override.get(f"{c}:{s}") if override else None
        if override and not ids:
            # A partition naming NO ids for a cell must DROP it. An empty list and an absent
            # key both being falsy is exactly how a worker loads the unsliced corpus in full.
            continue
        matches = {m["match_id"]: m for m in _values(sb.matches(competition_id=c, season_id=s, fmt="dict"))}
        if ids is None:
            ids = sorted(matches)[: args.matches_per_cell]
        for mid in ids:
            home = (matches[mid].get("home_team") or {})
            home_id = home.get("home_team_id", home.get("id")) if isinstance(home, dict) else home
            yield (c, s, mid, home_id)


if __name__ == "__main__":
    main()
```

`CorpusPassResult` (`scripts/_driver.py:511-518`) carries exactly `shard_dir`, `attempted`, `skipped`, `failed`, `failures`, `counters`, `keys`, `counters_unrecorded` — **there is no `processed`**, which is why the print above derives it. Re-verify those field names against the file before running, and use `res.manifest()`, never a hand-written `manifest_fields(...)`, which silently takes the old `counters_unrecorded=0` default and reports a complete corpus of nothing.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/scripts/test_build_sb360_coverage.py -v`
Expected: 3 passed

- [ ] **Step 5: Verify `--help` is safe**

Run: `python scripts/build_sb360_coverage.py --help`
Expected: usage text, no side effects, no files written. 16 public `scripts/*.py` have no argparse, so `--help` runs `main()` and rewrites committed goldens — confirm this one is not the seventeenth.

- [ ] **Step 6: Add the network-gated e2e smoke test**

```python
# append to tests/scripts/test_build_sb360_coverage.py
import pytest


@pytest.mark.e2e
def test_measure_match_against_real_open_360():
    """Deselected from the normal suite: network + slow. Self-skips without statsbombpy."""
    pytest.importorskip("statsbombpy")
    import scripts.build_sb360_coverage as m

    catalogue = m._load_catalogue()
    row = m.resolve_competition(44, 107, catalogue=catalogue, expect_name="Major League Soccer")
    assert row["competition_gender"] == "male"

    from statsbombpy import sb

    matches = {mm["match_id"]: mm for mm in m._values(sb.matches(competition_id=44, season_id=107, fmt="dict"))}
    match_id = sorted(matches)[0]
    home = matches[match_id]["home_team"]
    home_id = home.get("home_team_id", home.get("id")) if isinstance(home, dict) else home
    out = m.measure_match((44, 107, match_id, home_id))

    assert len(out) > 0, "no freeze-frames returned -- 360 may be absent for this match"
    assert out["n_events"].sum() > 0
    # Rates must never be reported without their denominator.
    assert {"n_events", "n_defending_gk_visible", "defending_gk_visible_rate"} <= set(out.columns)
    # Normalised by the 120x80 StatsBomb pitch, so a fully-visible frame is 1.0, not 1.34.
    assert out["mean_visible_pitch_fraction"].between(0.0, 1.0).all()

    # The SPADL join must actually resolve, or the per-type breakdown is one bucket and the
    # spec's headline Layer B metric has silently vanished.
    types = set(out["action_type"])
    assert types != {"unmapped"}, "event_uuid -> SPADL type join resolved nothing"
    assert "pass" in types, f"no SPADL passes -- converter output looks wrong: {sorted(types)}"

    # NON-VACUOUS GK-domain check. `out["is_gk_domain"].any()` would pass trivially on any
    # match containing a shot; `goalkick` is the spec's named GK-domain event and can ONLY
    # appear if the converter ran, since StatsBomb encodes it as pass_type == "Goal Kick"
    # rather than as a type of its own.
    assert "goalkick" in types, (
        f"no SPADL goalkick found -- the converter did not run, or the join key is wrong. "
        f"Types seen: {sorted(types)}"
    )
    gk_rows = out[out["is_gk_domain"]]
    assert len(gk_rows) > 0 and set(gk_rows["action_type"]) <= set(m.GK_DOMAIN_TYPES)
```

Run: `python -m pytest tests/scripts/test_build_sb360_coverage.py -m e2e -v` (requires network + `statsbombpy` in `.venv312`)
Expected: PASS, or SKIP without `statsbombpy`.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_sb360_coverage.py tests/scripts/test_build_sb360_coverage.py
git commit -m "feat(scripts): SB360 real-data coverage driver (ADR-052 seam)"
```

---

### Task 10: Layer B run and coverage report

**Files:**
- Create: `docs/research/sb360_coverage/coverage.md`
- Modify: `docs/research/sb360_coverage/README.md`

- [ ] **Step 1: Install `statsbombpy` into `.venv312`, never `.venv`**

```bash
.venv312/Scripts/python -m pip install statsbombpy
```

- [ ] **Step 2: Commit everything first**

`require_clean_tree` counts untracked files as dirty. An uncommitted doc or an `--output-dir` inside the repo makes the driver `SystemExit` before doing any work.

- [ ] **Step 3: Run the three cells**

```bash
.venv312/Scripts/python scripts/build_sb360_coverage.py \
  --competitions 72:107 43:106 44:107 --matches-per-cell 8 \
  --out docs/research/sb360_coverage/_shards
```

Expected: `[i/n]` progress lines, one shard per match, resumable on interrupt.

- [ ] **Step 4: Write the coverage report**

Report the three cells with the two contrasts stated as bounded claims, never as a point estimate: **MLS 2023 bounds the tier axis, WWC 2023 bounds the sex axis, and NWSL sits at a combination the open data has no observation for.** Include every rate's denominator. Stamp generation date, competition/season IDs *and resolved names*, the full match-ID list, `statsbombpy` version, `run_commit` and `run_tree_dirty`.

- [ ] **Step 5: Commit**

```bash
git add docs/research/sb360_coverage/
git commit -m "docs(sb360): real-data coverage across the three-cell design"
```

---

### Task 11: Release wiring

**Files:**
- Modify: `CHANGELOG.md`, `silly_kicks/__init__.py:7`, `pyproject.toml:7`, `TODO.md`

- [ ] **Step 1: Confirm the version has not been taken**

```bash
git fetch --tags && git tag --list "v4.7*"
grep -n "__version__" silly_kicks/__init__.py
grep -n "^version" pyproject.toml
```

Expected: both read `4.73.0` today. **The version lives in TWO places** — `silly_kicks/__init__.py:7` (`__version__ = "4.73.0"`) and `pyproject.toml:7` (`version = "4.73.0"`) — and both must move together or the installed package reports a version the metadata contradicts.

- [ ] **Step 2: Bump both files and write the CHANGELOG entry**

Record: the audit's headline counts per adjudication; every `silent_degrade` finding; that this is **additive, no retrain trigger, C4-free** (no new aggregator, documented count unchanged); and the follow-on items the audit deliberately did not fix — the `providers/statsbomb/` parse port, the `visible_area` library seam, and extending `velocity_unavailable_by_design` beyond its two current consumers.

- [ ] **Step 3: Add the follow-ons to `TODO.md`**

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md silly_kicks/__init__.py pyproject.toml TODO.md
git commit -m "chore: release wiring for the SB360 coverage audit"
```
