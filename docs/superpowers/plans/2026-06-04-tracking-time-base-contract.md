# Tracking time-base contract + loud low-coverage guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the silly-kicks per-period `time_seconds` contract explicit and enforced, and make a low action↔frame link-coverage outcome loud (per-period, warn-by-default, opt-in raise) with a diagnostic that names a suspected time-base mismatch.

**Architecture:** One pure detector (`_diagnose_time_base`) feeds two surfaces — the linker's `on_low_coverage` guard (primary, lazy, per-period) and a public `validate_time_base` affordance (the real guard for consumers that pre-filter actions by time before linking). `LinkReport` gains per-period link rates. Documentation + convention-pinning tests turn the period-relative invariant into an enforced contract.

**Tech Stack:** Python 3.10, pandas 2.3, numpy 2.x; pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-04-tracking-time-base-contract-design.md` (reviewed; 2 HIGH + 3 MEDIUM incorporated).

---

## ⚠️ Project-specific execution rules (override the skill defaults)

1. **NO per-task commits.** This project's standing rule (user memory `feedback-no-standalone-doc-commits`) is: one feature branch, **a single commit at the end**, PR when fully tested. The TDD steps below use checkboxes for tracking but **do not commit between tasks**. Only Task 9 commits.
2. **Commit is sentinel-gated.** `git commit` triggers `~/.claude/hooks/git_commit_guard.py`. **Do NOT create the `~/.claude-git-approval` sentinel yourself.** In Task 9, present the command + full diff and HOLD until the user explicitly authorizes the commit (memory `feedback-never-create-sentinel-without-approval`).
3. **Branch is already created:** `feat/tracking-time-base-contract` (the spec lives there, untracked).
4. **Lint trio before the commit** (memory): `ruff check silly_kicks/ tests/ scripts/` + `ruff format --check silly_kicks/ tests/ scripts/` + `pyright silly_kicks/` (whole package), on pinned versions (`ruff==0.15.7`, `pyright==1.1.409`).
5. **Run tests unpiped** (or `; echo $?`) — piped exit codes lie. Read the actual `N passed` summary line.
6. **Venv:** the uv-managed `.venv` (CPython 3.10.19). Activate it / use `python -m pytest`.

---

## File Structure

**Modified:**
- `silly_kicks/tracking/schema.py` — add `TimeBaseDiagnosis` frozen dataclass; add `per_period_link_rate` field to `LinkReport` (last field, `default_factory=dict`).
- `silly_kicks/tracking/utils.py` — add `MISMATCH_OVERLAP_FLOOR` const, `_diagnose_time_base`, `_format_diagnosis`, `validate_time_base`, `_enforce_link_coverage`; modify `link_actions_to_frames` (new kwargs + per-period rate + guard); add time-base-contract docstring notes to `link_actions_to_frames` + `slice_around_event`.
- `silly_kicks/tracking/__init__.py` — export `validate_time_base` + `TimeBaseDiagnosis`.
- `silly_kicks/tracking/gradientsports.py`, `silly_kicks/tracking/sportec.py` — convention note in `convert_to_frames` docstrings.
- `silly_kicks/tracking/schema.py`, `silly_kicks/spadl/schema.py` — `time_seconds` convention comment on the schema dict entry.
- `CLAUDE.md` — one architecture line.
- Version sites (Task 9): `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`.

**Created:**
- `tests/tracking/test_time_base_contract.py` — detector, `validate_time_base`, linker guard, laundering keystone, backcompat.
- `tests/spadl/test_time_seconds_convention.py` — convention-pinning lock tests (Opta, StatsBomb, GS events pass-through).
- `docs/superpowers/adrs/ADR-017-time-base-contract-link-coverage-guard.md`.

---

## Task 1: `LinkReport.per_period_link_rate` field + per-period population

Foundational — the guard (Task 4) reads this. Per-period rate **must** come from `merged_all` (retains `period_id`), not the returned `pointers` (drops it).

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (LinkReport dataclass, ~line 128-162)
- Modify: `silly_kicks/tracking/utils.py` (`link_actions_to_frames`, ~line 282-302; empty early-return ~line 222)
- Test: `tests/tracking/test_time_base_contract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tracking/test_time_base_contract.py` with the shared fixtures + first test:

```python
"""Tests for the time-base contract + low-coverage guard (spec 2026-06-04)."""

import warnings

import pandas as pd
import pytest

from silly_kicks.tracking.utils import link_actions_to_frames


def _frame_row(period_id, frame_id, t):
    return {
        "game_id": 1, "period_id": period_id, "frame_id": frame_id, "time_seconds": t,
        "frame_rate": 25.0, "player_id": 7, "team_id": 100, "is_ball": False,
        "is_goalkeeper": False, "x": 50.0, "y": 34.0, "z": float("nan"), "speed": 5.0,
        "speed_source": "native", "ball_state": "alive", "team_attacking_direction": "ltr",
        "confidence": None, "visibility": None, "source_provider": "gradientsports",
    }


def _action_row(action_id, period_id, t):
    return {
        "game_id": 1, "action_id": action_id, "period_id": period_id, "time_seconds": t,
        "team_id": 100, "player_id": 7, "type_id": 0, "result_id": 1, "bodypart_id": 0,
        "start_x": 50.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0,
    }


def test_per_period_link_rate_populated():
    # p1: 2 actions both on a frame (linked). p2: 2 actions, both on frames (linked).
    # Healthy match -> no warning, no kwargs that don't exist yet.
    frames = pd.DataFrame(
        [_frame_row(1, 0, 0.0), _frame_row(1, 1, 1.0), _frame_row(2, 2, 0.0), _frame_row(2, 3, 1.0)]
    )
    actions = pd.DataFrame(
        [_action_row(0, 1, 0.0), _action_row(1, 1, 1.0), _action_row(2, 2, 0.0), _action_row(3, 2, 1.0)]
    )
    _, report = link_actions_to_frames(actions, frames)
    assert report.per_period_link_rate == {1: 1.0, 2: 1.0}


def test_link_report_positional_construction_backcompat():
    """LOW: lock that the existing 7-positional-arg LinkReport construction still works
    (the new per_period_link_rate field is added LAST with a default). The empty-actions
    early return in link_actions_to_frames relies on this."""
    from silly_kicks.tracking.schema import LinkReport

    rpt = LinkReport(3, 2, 1, 0, {"gradientsports": 0.67}, 0.05, 0.2)  # 7 positional args
    assert rpt.per_period_link_rate == {}
    assert rpt.link_rate == 2 / 3
```

> **Note (cross-task test hygiene, lakehouse MEDIUM):** the per-period *non-laundering* field assertion and the GS-10503 laundering keystone both need the `on_low_coverage` kwarg, which does not exist until Task 4 — so they live in **Task 4**, not here. Task 1 leaves only green tests in the suite, keeping the between-task subagent reviews clean.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_time_base_contract.py::test_per_period_link_rate_populated -v`
Expected: FAIL — `AttributeError: 'LinkReport' object has no attribute 'per_period_link_rate'`.

- [ ] **Step 3a: Add the `per_period_link_rate` field to `LinkReport`**

In `silly_kicks/tracking/schema.py`, inside the `LinkReport` dataclass, add the field **last** (after `tolerance_seconds`) so positional construction stays valid:

```python
    tolerance_seconds: float

    per_period_link_rate: dict[int, float] = dataclasses.field(default_factory=dict)
    """period_id -> linked / actions-in-that-period. Computed from the internal
    per-period merge (NOT the returned pointers, which drop period_id), so a
    catastrophically-unlinked period is never laundered behind a healthy one.
    Empty for an empty-actions call. See ADR-017."""
```

(`dataclasses` is already imported at the top of `schema.py`.)

- [ ] **Step 3b: Populate it in `link_actions_to_frames`**

In `silly_kicks/tracking/utils.py`, after `merged_all` is built (currently ~line 259) and before the `report = LinkReport(...)` construction (~line 293), compute the per-period rate from `merged_all`:

```python
    per_period_link_rate: dict[int, float] = {
        int(p): float(s.notna().mean()) for p, s in merged_all.groupby("period_id")["frame_id"]
    }
```

Then pass it into the report constructor (add the kwarg to the existing `LinkReport(...)` call):

```python
    report = LinkReport(
        n_actions_in=n_in,
        n_actions_linked=n_linked,
        n_actions_unlinked=n_unlinked,
        n_actions_multi_candidate=n_multi,
        per_provider_link_rate=per_provider,
        max_time_offset_seconds=max_off,
        tolerance_seconds=tolerance_seconds,
        per_period_link_rate=per_period_link_rate,
    )
```

The empty-actions early return (~line 222) keeps its positional `LinkReport(0, 0, 0, 0, {}, 0.0, tolerance_seconds)` — the new field defaults to `{}`.

- [ ] **Step 4: Run the Task-1 tests to verify they pass**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k "per_period_link_rate_populated or positional_construction" -v`
Expected: both PASS. (No red tests linger — the laundering/keystone tests are in Task 4.)

- [ ] **Step 5: Verify existing link tests still pass (backcompat)**

Run: `python -m pytest tests/test_tracking_utils_link.py -v`
Expected: all PASS (the new field is additive with a default).

---

## Task 2: `TimeBaseDiagnosis` dataclass + `_diagnose_time_base` pure detector

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (new dataclass)
- Modify: `silly_kicks/tracking/utils.py` (const + detector + formatter)
- Test: `tests/tracking/test_time_base_contract.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tracking/test_time_base_contract.py`:

```python
from silly_kicks.tracking.utils import MISMATCH_OVERLAP_FLOOR, _diagnose_time_base


def test_diagnosis_flags_disjoint_period():
    # p1 conforming (overlap ~1); p2 actions absolute [2700,5835], frames relative [0,3142].
    frames = pd.DataFrame(
        [_frame_row(1, 0, 0.0), _frame_row(1, 1, 2823.0)]
        + [_frame_row(2, 2, 0.0), _frame_row(2, 3, 3142.0)]
    )
    actions = pd.DataFrame(
        [_action_row(0, 1, 1.0), _action_row(1, 1, 2822.0)]
        + [_action_row(2, 2, 2700.0), _action_row(3, 2, 5835.0)]
    )
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods == (2,)
    assert diag.per_period_overlap_fraction[2] == pytest.approx((3142.0 - 2700.0) / (5835.0 - 2700.0), rel=1e-6)
    assert diag.per_period_overlap_fraction[2] < MISMATCH_OVERLAP_FLOOR
    assert diag.per_period_overlap_fraction[1] >= MISMATCH_OVERLAP_FLOOR
    assert "period 2" in diag.message


def test_diagnosis_no_mismatch_when_overlapping():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, 90.0)])
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods == ()


def test_diagnosis_worst_first_ordering():
    # p2 fully disjoint (overlap 0), p3 partial-but-below-floor.
    frames = pd.DataFrame(
        [_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0),
         _frame_row(2, 2, 0.0), _frame_row(2, 3, 10.0),
         _frame_row(3, 4, 0.0), _frame_row(3, 5, 100.0)]
    )
    actions = pd.DataFrame(
        [_action_row(0, 1, 10.0), _action_row(1, 1, 90.0),     # p1 fine
         _action_row(2, 2, 900.0), _action_row(3, 2, 1000.0),  # p2 fully disjoint
         _action_row(4, 3, 95.0), _action_row(5, 3, 600.0)]    # p3 small overlap < floor
    )
    diag = _diagnose_time_base(actions, frames)
    assert diag.suspected_mismatch_periods[0] == 2  # worst (overlap 0) first
    assert set(diag.suspected_mismatch_periods) == {2, 3}


def test_diagnosis_single_action_degenerate_span():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    in_range = pd.DataFrame([_action_row(0, 1, 50.0)])
    out_range = pd.DataFrame([_action_row(0, 1, 500.0)])
    assert _diagnose_time_base(in_range, frames).per_period_overlap_fraction[1] == 1.0
    assert _diagnose_time_base(out_range, frames).per_period_overlap_fraction[1] == 0.0


def test_diagnosis_nan_tolerant():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, float("nan"))])
    diag = _diagnose_time_base(actions, frames)  # must not raise
    assert diag.per_period_action_range[1] == (10.0, 10.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k diagnosis -v`
Expected: FAIL — `ImportError: cannot import name 'MISMATCH_OVERLAP_FLOOR'` / `_diagnose_time_base`.

- [ ] **Step 3a: Add `TimeBaseDiagnosis` to `schema.py`**

In `silly_kicks/tracking/schema.py`, after the `LinkReport` class, add:

```python
@dataclasses.dataclass(frozen=True)
class TimeBaseDiagnosis:
    """Per-period action-vs-frame time-range diagnosis (time-base mismatch hypothesis).

    Produced by ``silly_kicks.tracking.utils._diagnose_time_base`` and surfaced by
    ``validate_time_base`` and the ``link_actions_to_frames`` low-coverage guard.
    A *cause hypothesis* for low link coverage, distinct from the *symptom*
    (low link rate). See ADR-017.

    Attributes:
        per_period_action_range: period_id -> (min, max) action time_seconds.
        per_period_frame_range: period_id -> (min, max) frame time_seconds
            (absent for a period that has actions but no frames).
        per_period_overlap_fraction: period_id -> fraction of the action span
            covered by the frame span (1.0 = frames fully span; 0.0 = disjoint).
        suspected_mismatch_periods: periods with overlap < MISMATCH_OVERLAP_FLOOR,
            ordered worst-first (lowest overlap first).
        message: human-readable summary enumerating suspected periods worst-first.
    """

    per_period_action_range: dict[int, tuple[float, float]]
    per_period_frame_range: dict[int, tuple[float, float]]
    per_period_overlap_fraction: dict[int, float]
    suspected_mismatch_periods: tuple[int, ...]
    message: str

    @property
    def has_suspected_mismatch(self) -> bool:
        return len(self.suspected_mismatch_periods) > 0
```

- [ ] **Step 3b: Add the const, detector, and formatter to `utils.py`**

In `silly_kicks/tracking/utils.py`, update the import from `.schema` (currently `from .schema import LinkReport`) to:

```python
from .schema import LinkReport, TimeBaseDiagnosis
```

Add the module constant near the top (after imports):

```python
MISMATCH_OVERLAP_FLOOR: float = 0.2
"""Per-period action/frame range overlap below this is flagged a suspected
time-base mismatch (period-relative vs absolute). Decoupled from the linker's
min_link_rate: this governs the *cause hypothesis*, not the *symptom*. 0.2 is
specific to near-disjoint ranges (the GS bug was ~0.14) and stays quiet on
ordinary sparsity. See ADR-017."""
```

Add the detector + formatter (place them after `link_actions_to_frames`, before `_count_candidates_within_tolerance`):

```python
def _diagnose_time_base(actions: pd.DataFrame, frames: pd.DataFrame) -> TimeBaseDiagnosis:
    """Pure per-period action-vs-frame time-range diagnosis. No warn/raise/I/O.

    Vectorized: per-period ranges via a single groupby().agg on each side
    (NOT the iterrows pattern in _count_candidates_within_tolerance). NaN
    time_seconds rows are dropped before computing ranges.
    """
    a = actions[["period_id", "time_seconds"]].dropna(subset=["time_seconds"])
    f = frames[["period_id", "time_seconds"]].dropna(subset=["time_seconds"])
    a_rng = a.groupby("period_id")["time_seconds"].agg(["min", "max"])
    f_rng = f.groupby("period_id")["time_seconds"].agg(["min", "max"])

    per_action: dict[int, tuple[float, float]] = {}
    per_frame: dict[int, tuple[float, float]] = {}
    overlap_frac: dict[int, float] = {}
    suspected: list[int] = []

    for p in a_rng.index:
        a_min, a_max = float(a_rng.loc[p, "min"]), float(a_rng.loc[p, "max"])
        per_action[int(p)] = (a_min, a_max)
        if p in f_rng.index:
            f_min, f_max = float(f_rng.loc[p, "min"]), float(f_rng.loc[p, "max"])
            per_frame[int(p)] = (f_min, f_max)
            span = a_max - a_min
            if span <= 0.0:  # degenerate single-point action span
                frac = 1.0 if (f_min <= a_min <= f_max) else 0.0
            else:
                overlap = max(0.0, min(a_max, f_max) - max(a_min, f_min))
                frac = overlap / span
        else:
            frac = 0.0  # actions in this period but no frames at all
        overlap_frac[int(p)] = frac
        if frac < MISMATCH_OVERLAP_FLOOR:
            suspected.append(int(p))

    suspected.sort(key=lambda p: overlap_frac[p])  # worst (lowest overlap) first
    message = _format_diagnosis(per_action, per_frame, overlap_frac, tuple(suspected))
    return TimeBaseDiagnosis(per_action, per_frame, overlap_frac, tuple(suspected), message)


def _format_diagnosis(
    per_action: dict[int, tuple[float, float]],
    per_frame: dict[int, tuple[float, float]],
    overlap_frac: dict[int, float],
    suspected: tuple[int, ...],
) -> str:
    """Human-readable summary; enumerates suspected periods worst-first."""
    if not suspected:
        return "no time-base mismatch detected (all periods overlap)"
    parts = []
    for p in suspected:
        a_min, a_max = per_action[p]
        if p in per_frame:
            f_min, f_max = per_frame[p]
            frames_desc = f"frames [{f_min:g}, {f_max:g}]"
        else:
            frames_desc = "no frames"
        parts.append(
            f"period {p}: actions [{a_min:g}, {a_max:g}] vs {frames_desc} "
            f"— near-disjoint (overlap {overlap_frac[p]:.2f})"
        )
    return (
        "; ".join(parts)
        + "; suspected period-relative/absolute time-base mismatch "
        "(time_seconds must be period-relative; see the time-base contract)"
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k diagnosis -v`
Expected: all 5 `diagnosis` tests PASS.

---

## Task 3: `validate_time_base` public affordance + exports

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (public wrapper)
- Modify: `silly_kicks/tracking/__init__.py` (`__all__` + imports)
- Test: `tests/tracking/test_time_base_contract.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from silly_kicks.tracking import TimeBaseDiagnosis, validate_time_base


def _mismatch_inputs():
    frames = pd.DataFrame([_frame_row(2, 0, 0.0), _frame_row(2, 1, 3142.0)])
    actions = pd.DataFrame([_action_row(0, 2, 2700.0), _action_row(1, 2, 5835.0)])
    return actions, frames


def test_validate_time_base_raises_by_default():
    actions, frames = _mismatch_inputs()
    with pytest.raises(ValueError, match="time-base mismatch"):
        validate_time_base(actions, frames)


def test_validate_time_base_warn_returns_diagnosis():
    actions, frames = _mismatch_inputs()
    with pytest.warns(UserWarning, match="time-base mismatch"):
        diag = validate_time_base(actions, frames, on_mismatch="warn")
    assert isinstance(diag, TimeBaseDiagnosis)
    assert diag.suspected_mismatch_periods == (2,)


def test_validate_time_base_ignore_silent():
    actions, frames = _mismatch_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail
        diag = validate_time_base(actions, frames, on_mismatch="ignore")
    assert diag.has_suspected_mismatch


def test_validate_time_base_clean_inputs_no_raise():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 100.0)])
    actions = pd.DataFrame([_action_row(0, 1, 10.0), _action_row(1, 1, 90.0)])
    diag = validate_time_base(actions, frames)  # default raise, but nothing to raise on
    assert not diag.has_suspected_mismatch
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k validate_time_base -v`
Expected: FAIL — `ImportError: cannot import name 'validate_time_base'`.

- [ ] **Step 3a: Add `validate_time_base` to `utils.py`**

Add after `_format_diagnosis`:

```python
def validate_time_base(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> TimeBaseDiagnosis:
    """Pre-link assertion that actions + frames share a per-period time base.

    silly_kicks' canonical ``time_seconds`` convention is **period-relative**
    (resets to 0 each period; see the link_actions_to_frames docstring). This
    helper runs the pure per-period range diagnosis and, on a suspected
    mismatch, raises (default), warns, or returns silently.

    **This is the primary guard for any consumer that pre-filters / windows /
    batches actions by time before linking.** ``link_actions_to_frames``'s own
    ``on_low_coverage`` guard only sees the actions that reach it — a pre-filter
    that drops out-of-range actions upstream leaves the linker with
    ~100%-linkable survivors and the guard silent (exactly how the original GS
    period-2 bug stayed invisible). Call this on the **unfiltered** inputs at
    work-unit entry. See ADR-017.

    Parameters
    ----------
    actions, frames : pd.DataFrame
        SPADL actions / long-form tracking frames (need ``period_id`` +
        ``time_seconds``).
    on_mismatch : {"raise", "warn", "ignore"}, default "raise"
        Policy when a suspected mismatch is found. Default ``"raise"`` — an
        explicitly-invoked assertion should fail loud (the asymmetry with the
        linker's ``warn`` default is intentional).

    Returns
    -------
    TimeBaseDiagnosis
        The per-period diagnosis (returned in all policies, including "raise"
        when no mismatch is found).

    Raises
    ------
    ValueError
        If ``on_mismatch="raise"`` and a suspected mismatch is found.

    Examples
    --------
    >>> from silly_kicks.tracking import validate_time_base
    >>> diag = validate_time_base(actions, frames, on_mismatch="warn")
    >>> diag.has_suspected_mismatch  # doctest: +SKIP
    """
    diag = _diagnose_time_base(actions, frames)
    if diag.has_suspected_mismatch:
        if on_mismatch == "raise":
            raise ValueError(f"validate_time_base: {diag.message}")
        if on_mismatch == "warn":
            warnings.warn(f"validate_time_base: {diag.message}", UserWarning, stacklevel=2)
    return diag
```

Ensure `Literal` is imported in `utils.py`. Check the top of the file; if absent, add:

```python
from typing import Literal
```

- [ ] **Step 3b: Export from `__init__.py`**

In `silly_kicks/tracking/__init__.py`:
- Add `"TimeBaseDiagnosis",` and `"validate_time_base",` to `__all__` (keep alphabetical: `TimeBaseDiagnosis` near the dataclasses block; `validate_time_base` after `utils`).
- Add `TimeBaseDiagnosis` to the `from .schema import (...)` block.
- Add `validate_time_base` to the `from .utils import (...)` block.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k validate_time_base -v`
Expected: 4 PASS.

- [ ] **Step 5: Import-surface sanity**

Run: `python -c "import silly_kicks.tracking as t; print(t.validate_time_base, t.TimeBaseDiagnosis)"`
Expected: prints both objects, no error.

---

## Task 4: `link_actions_to_frames` low-coverage guard (per-period, warn-default)

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (`link_actions_to_frames` signature + guard + message helper)
- Test: `tests/tracking/test_time_base_contract.py`

- [ ] **Step 1: Write the failing tests (incl. the laundering keystone)**

Append:

```python
def test_low_coverage_warns_by_default():
    # p2 fully unlinked (actions far from p2 frame).
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with pytest.warns(UserWarning, match="period 2"):
        link_actions_to_frames(actions, frames)


def test_laundering_keystone_per_period_not_aggregate():
    """GS 10503 shape: aggregate > 0.5 but p2 < 0.5. Must fire on p2.

    p1: 100 actions all linked. p2: 100 actions, 19 linked. Aggregate = 59.5%
    (> 0.5) — a match-aggregate guard would stay SILENT. Per-period p2 = 19%.
    This test fails if anyone reduces the guard to the match aggregate.
    """
    frame_rows = [_frame_row(1, i, float(i)) for i in range(100)]
    frame_rows += [_frame_row(2, 100 + i, float(i)) for i in range(100)]
    frames = pd.DataFrame(frame_rows)
    action_rows = [_action_row(i, 1, float(i)) for i in range(100)]            # p1 all on frames
    action_rows += [_action_row(100 + i, 2, float(i)) for i in range(19)]      # p2 first 19 on frames
    action_rows += [_action_row(200 + i, 2, 9000.0 + i) for i in range(81)]    # p2 other 81 far away
    actions = pd.DataFrame(action_rows)

    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    aggregate = report.n_actions_linked / report.n_actions_in
    assert aggregate > 0.5                      # would launder under a match-agg guard
    assert report.per_period_link_rate[2] < 0.5

    with pytest.warns(UserWarning, match="period 2"):
        link_actions_to_frames(actions, frames)  # default warn fires on p2


def test_low_coverage_raise():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with pytest.raises(ValueError, match="period 2"):
        link_actions_to_frames(actions, frames, on_low_coverage="raise")


def test_low_coverage_ignore_silent_but_report_populated():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[2] == 0.0


def test_healthy_match_silent():
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(1, 1, 1.0), _frame_row(2, 2, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 1, 1.0), _action_row(2, 2, 0.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails
        link_actions_to_frames(actions, frames)  # default warn, but healthy -> silent


def test_low_coverage_message_carries_time_base_hint():
    # p2 low coverage AND near-disjoint ranges -> message names the mismatch.
    frames = pd.DataFrame([_frame_row(2, 0, 0.0), _frame_row(2, 1, 3142.0)])
    actions = pd.DataFrame([_action_row(i, 2, 2700.0 + 50.0 * i) for i in range(50)])
    with pytest.warns(UserWarning, match="time-base mismatch"):
        link_actions_to_frames(actions, frames)


def test_sparsity_warns_without_mismatch_hint():
    # Tighten min_link_rate; ranges overlap (uniform sparsity) -> warn but NO mismatch claim.
    frames = pd.DataFrame([_frame_row(1, i, float(i)) for i in range(0, 100, 10)])  # every 10s
    actions = pd.DataFrame([_action_row(i, 1, float(i)) for i in range(100)])       # every 1s
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        link_actions_to_frames(actions, frames, min_link_rate=0.8, tolerance_seconds=0.2)
    msgs = [str(w.message) for w in caught]
    assert any("period 1" in m for m in msgs)
    assert all("time-base mismatch" not in m for m in msgs)


def test_per_period_link_rate_is_not_laundered_field():
    # p1 fully linked, p2 fully UNLINKED. Field reflects per-period truth (moved from Task 1).
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[1] == 1.0
    assert report.per_period_link_rate[2] == 0.0


def test_low_coverage_warning_blames_caller_not_linker_internals():
    """HIGH (lakehouse): stacklevel must point at THIS call site, not utils.py.

    The warn lives in the _enforce_link_coverage helper (one frame below
    link_actions_to_frames), so stacklevel must be 3. A value of 2 would set
    w.filename to silly_kicks/tracking/utils.py — the exact 'into the linker
    internals' outcome the spec forbids. Message-only assertions cannot catch
    this; assert the warning's filename is this test module.
    """
    frames = pd.DataFrame([_frame_row(1, 0, 0.0), _frame_row(2, 1, 0.0)])
    actions = pd.DataFrame([_action_row(0, 1, 0.0), _action_row(1, 2, 5000.0)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        link_actions_to_frames(actions, frames)  # default warn
    assert caught, "expected a low-coverage UserWarning"
    assert caught[0].filename == __file__, (
        f"warning blamed {caught[0].filename}, expected the caller ({__file__}); "
        "stacklevel is wrong"
    )


def test_guard_integration_fidelity_realistic_two_period_match():
    """LOW (lakehouse): one 'as e2e as possible' test exercising the full link +
    guard on a realistic 2-period fixture — p1 healthy, p2 a period-relative-vs-
    absolute time-base mismatch (frames reset to 0, actions on the match clock)."""
    frames = pd.DataFrame(
        [_frame_row(1, i, float(i) * 30.0) for i in range(40)]            # p1 frames 0..1170s
        + [_frame_row(2, 40 + i, float(i) * 30.0) for i in range(40)]     # p2 frames RESET to 0..1170s
    )
    actions = pd.DataFrame(
        [_action_row(i, 1, float(i) * 30.0) for i in range(40)]                  # p1 actions on frames
        + [_action_row(40 + i, 2, 1500.0 + float(i) * 30.0) for i in range(40)]  # p2 actions ABSOLUTE
    )
    _, report = link_actions_to_frames(actions, frames, on_low_coverage="ignore")
    assert report.per_period_link_rate[1] == pytest.approx(1.0)
    assert report.per_period_link_rate[2] < 0.2
    with pytest.warns(UserWarning, match="time-base mismatch"):
        link_actions_to_frames(actions, frames)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -k "low_coverage or laundering or healthy or sparsity or is_not_laundered or integration_fidelity" -v`
Expected: FAIL — `on_low_coverage`/`min_link_rate` are unexpected kwargs.

- [ ] **Step 3a: Extend the signature**

In `silly_kicks/tracking/utils.py`, change the `link_actions_to_frames` signature to:

```python
def link_actions_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    tolerance_seconds: float = 0.2,
    *,
    min_link_rate: float = 0.5,
    on_low_coverage: Literal["warn", "raise", "ignore"] = "warn",
) -> tuple[pd.DataFrame, LinkReport]:
```

- [ ] **Step 3b: Add the guard after the report is built**

At the end of `link_actions_to_frames`, replace `return pointers, report` with the guard + return:

```python
    _enforce_link_coverage(
        actions, frames, report,
        min_link_rate=min_link_rate, on_low_coverage=on_low_coverage,
    )
    return pointers, report
```

Add the helper (after `validate_time_base`):

```python
def _enforce_link_coverage(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    report: LinkReport,
    *,
    min_link_rate: float,
    on_low_coverage: Literal["warn", "raise", "ignore"],
) -> None:
    """Per-period low-coverage policy for link_actions_to_frames. See ADR-017."""
    if on_low_coverage == "ignore" or not report.per_period_link_rate:
        return
    offending = {p: r for p, r in report.per_period_link_rate.items() if r < min_link_rate}
    if not offending:
        return

    diag = _diagnose_time_base(actions, frames)  # lazy: only on a tripped guard
    suspected = set(diag.suspected_mismatch_periods)
    worst_first = sorted(offending, key=lambda p: offending[p])

    def _line(p: int) -> str:
        n_total = int((actions["period_id"] == p).sum())
        n_unlinked = int(round((1.0 - offending[p]) * n_total))
        msg = (
            f"link_actions_to_frames: period {p} link_rate {offending[p]:.2f} "
            f"({n_total} actions, {n_unlinked} unlinked) below min_link_rate {min_link_rate:g}."
        )
        if p in suspected:
            a_min, a_max = diag.per_period_action_range[p]
            frng = diag.per_period_frame_range.get(p)
            frames_desc = f"frames [{frng[0]:g}, {frng[1]:g}]" if frng else "no frames"
            msg += (
                f" period {p}: actions [{a_min:g}, {a_max:g}] vs {frames_desc} — "
                f"near-disjoint (overlap {diag.per_period_overlap_fraction[p]:.2f}); "
                "suspected period-relative/absolute time-base mismatch. "
                "See the time-base contract in the docstring."
            )
        return msg

    if on_low_coverage == "raise":
        raise ValueError(" ".join(_line(p) for p in worst_first))
    for p in worst_first:  # one warning per offending period (deduped per period)
        # stacklevel=3: warn site is _enforce_link_coverage (1) -> link_actions_to_frames (2)
        # -> the user's call site (3). NOT 2 — that would blame the linker's own internals.
        # (Contrast validate_time_base, which warns in its own body, so stacklevel=2 is correct
        # there. The project's "stacklevel=2" convention means "point at the user"; the literal
        # value depends on call-nesting depth.)
        warnings.warn(_line(p), UserWarning, stacklevel=3)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_time_base_contract.py -v`
Expected: all tests in the file PASS (incl. the laundering keystone and the earlier laundering/per-period tests from Task 1 that used `on_low_coverage="ignore"`).

- [ ] **Step 5: Backcompat re-check**

Run: `python -m pytest tests/test_tracking_utils_link.py -v`
Expected: all PASS (these use healthy single-period fixtures → no warning; the new kwargs default safely).

---

## Task 5: Convention-pinning lock tests (make §4.1 enforced — Opta + StatsBomb)

These **lock existing behavior** — they PASS against current code; a future converter refactor that silently emits absolute `time_seconds` makes them FAIL.

**Scope (lakehouse-reviewed):** these tests guard the converters whose `time_seconds` **arithmetic the library owns** — **Opta** (`opta.py:166` subtracts period offsets) and **StatsBomb** (`statsbomb.py:237` parses period-elapsed `timestamp`). They do **not** meaningfully guard GradientSports: GS `time_seconds` is a **verbatim pass-through** (`gradientsports.py:416`), so feeding period-relative in and asserting it out can never catch a GS absolute-time regression — GS time originates **upstream in the lakehouse**. The real GS guard is lakehouse-side (`validate_time_base` at work-unit entry + the lakehouse boundary test). This scoping is documented in ADR-017 (Task 6) so coverage is not overstated. Sportec/kloppy are also pass-through (lower drift risk); not included.

**Fixtures reuse proven builders** (lakehouse MEDIUM): mirror the exact event-dict shapes from `tests/spadl/test_opta.py` and `tests/spadl/test_statsbomb.py` (`_make_statsbomb_events`) so the lock tests track *convention* changes, not input-schema drift.

**Files:**
- Create: `tests/spadl/test_time_seconds_convention.py`

- [ ] **Step 1: Write the lock tests (Opta + StatsBomb only)**

```python
"""Convention-pinning lock tests: SPADL time_seconds is PERIOD-RELATIVE.

silly_kicks' canonical convention is that ``time_seconds`` resets to 0 at the
start of each period (NOT absolute match-clock). These tests turn that prose
contract (spec 2026-06-04 §4.1 / ADR-017) into enforced behavior for the
converters whose time arithmetic the library OWNS: a future refactor that
emits absolute, continuous-across-periods time makes them fail.

Scope: Opta + StatsBomb only. GradientSports time_seconds is a verbatim
pass-through (gradientsports.py:416) originating upstream in the lakehouse, so
it is guarded lakehouse-side (validate_time_base), not here. See ADR-017.
"""

import pandas as pd

from silly_kicks.spadl import opta, statsbomb


def _opta_event(event_id, period_id, minute, second):
    # Mirrors the event dict in tests/spadl/test_opta.py (proven-accepted shape).
    return {
        "game_id": 318175, "event_id": event_id, "type_id": 1, "period_id": period_id,
        "minute": minute, "second": second, "timestamp": "2010-01-27 19:47:14",
        "player_id": 8786, "team_id": 157, "outcome": True,
        "start_x": 50.0, "start_y": 50.0, "end_x": 60.0, "end_y": 50.0,
        "assist": False, "keypass": False, "qualifiers": {1: True}, "type_name": "pass",
    }


def test_opta_time_seconds_is_period_relative():
    # P1 at 02:14 (134s). P2 at 47:00 absolute = 02:00 into the 2nd half (120s relative).
    events = pd.DataFrame([_opta_event(1, 1, 2, 14), _opta_event(2, 2, 47, 0)])
    actions, _ = opta.convert_to_actions(events, home_team_id=157)
    p1 = actions.loc[actions["period_id"] == 1, "time_seconds"].iloc[0]
    p2 = actions.loc[actions["period_id"] == 2, "time_seconds"].iloc[0]
    assert p1 == 134.0
    assert p2 == 120.0          # period-relative: 47min − 45min = 2min
    assert p2 < p1              # absolute would give 2820s >> 134s; period-relative resets


def _sb_event(event_id, period_id, timestamp):
    # Mirrors _make_statsbomb_events() in tests/spadl/test_statsbomb.py (minimal accepted shape).
    return {
        "game_id": 1, "event_id": event_id, "period_id": period_id, "timestamp": timestamp,
        "team_id": 100, "player_id": 200, "type_name": "Pass",
        "location": [60.0, 40.0],
        "extra": {"pass": {"end_location": [70.0, 40.0], "outcome": {"name": "Complete"},
                           "height": {"name": "Ground Pass"}}},
    }


def test_statsbomb_time_seconds_is_period_relative():
    events = pd.DataFrame([
        _sb_event("abc-1", 1, "00:02:14.000"),
        _sb_event("abc-2", 2, "00:01:00.000"),  # 1 min into the 2nd half
    ])
    actions, _ = statsbomb.convert_to_actions(
        events, home_team_id=100, xy_fidelity_version=1, shot_fidelity_version=1
    )
    p1 = actions.loc[actions["period_id"] == 1, "time_seconds"].iloc[0]
    p2 = actions.loc[actions["period_id"] == 2, "time_seconds"].iloc[0]
    assert p1 == 134.0
    assert p2 == 60.0   # period-relative; an absolute clock would be ~2760s
    assert p2 < p1
```

- [ ] **Step 2: Run — expect PASS (lock current behavior)**

Run: `python -m pytest tests/spadl/test_time_seconds_convention.py -v`
Expected: both PASS. **If either FAILS, that is a real finding** — a converter is not period-relative; stop and report it (do not "fix" the test to match). If the converters reject the mirrored fixture (input-schema changed since `test_opta.py`/`test_statsbomb.py`), re-mirror from the current sibling test — do not hand-roll new columns.

---

## Task 6: Documentation — convention notes + ADR-017 + CLAUDE.md

**Files:**
- Modify: `silly_kicks/tracking/utils.py` (docstrings: `link_actions_to_frames`, `slice_around_event`)
- Modify: `silly_kicks/tracking/gradientsports.py`, `silly_kicks/tracking/sportec.py` (`convert_to_frames` docstrings)
- Modify: `silly_kicks/tracking/schema.py`, `silly_kicks/spadl/schema.py` (`time_seconds` comment)
- Create: `docs/superpowers/adrs/ADR-017-time-base-contract-link-coverage-guard.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the time-base contract note to `link_actions_to_frames` docstring**

In the `link_actions_to_frames` docstring (Parameters / Notes), add:

```
    Time-base contract
    ------------------
    ``actions`` and ``frames`` MUST share a per-period time base. silly_kicks'
    canonical convention is that ``time_seconds`` is **seconds since the start
    of its period, resetting to 0 each period** — NOT absolute match-clock /
    continuous across periods. Linking is per-period (merge_asof within each
    ``period_id``), so cross-period continuity is irrelevant, but a period whose
    actions and frames use different origins (e.g. period-relative frames vs
    absolute actions) will not link. Low per-period coverage triggers
    ``on_low_coverage``. For consumers that pre-filter actions by time before
    linking, call :func:`validate_time_base` on the unfiltered inputs — the
    guard here cannot see actions a pre-filter has already dropped. See ADR-017.
```

Add to the Parameters section docs for `min_link_rate` and `on_low_coverage` (default 0.5; per-period worst; warn/raise/ignore; rationale: warn because low coverage is a quality continuum, not a structurally-impossible input).

- [ ] **Step 2: Add a one-line convention note to `slice_around_event` docstring**

In `slice_around_event`, add to the Notes/Parameters: "Assumes the per-period ``time_seconds`` convention (resets each period); see ``link_actions_to_frames`` / ADR-017."

- [ ] **Step 3: Add convention note to the tracking converter docstrings**

In `silly_kicks/tracking/gradientsports.py` and `silly_kicks/tracking/sportec.py` `convert_to_frames` docstrings (Parameters for `raw_frames` / a Notes block):

```
    ``raw_frames["time_seconds"]`` must be **period-relative** (seconds since the
    start of each period, resetting to 0 — NOT absolute match-clock). This is
    silly_kicks' canonical convention (matches the events converters) and what
    :func:`silly_kicks.tracking.utils.link_actions_to_frames` requires. See ADR-017.
```

- [ ] **Step 4: Add the schema comments**

In `silly_kicks/tracking/schema.py` `TRACKING_FRAMES_COLUMNS`, add a comment on the `time_seconds` entry; same in `silly_kicks/spadl/schema.py` `SPADL_COLUMNS`:

```python
    "time_seconds": "float64",  # PERIOD-RELATIVE: seconds since the start of the period, resets to 0 each period (ADR-017)
```

- [ ] **Step 5: Write ADR-017**

Create `docs/superpowers/adrs/ADR-017-time-base-contract-link-coverage-guard.md` mirroring the ADR-010 structure:
- **Title:** "ADR-017: Period-relative `time_seconds` contract + loud per-period link-coverage guard".
- **Status:** "Accepted — pending implementation (silly-kicks X.Y.0)" (mirror ADR-010's wording; fill the version reconciled in Task 9). Confirm **017** is still the next free ADR number at commit time (ADR-015 = the TF-17 PR-C causal harness, ADR-016 = the parallel ghost-gk serve-mean — both spoken-for; if either has landed a *different* 017, renumber).
- **Deciders:** Karsten S. Nielsen, Claude Opus 4.8 (1M); luxury-lakehouse AC-1 session.
- **Context:** the GS period-2 ~81% silent loss; root cause (period-relative frames vs absolute actions); the source-verified finding that silly_kicks is period-relative (`opta.py:166`, `statsbomb.py:237`); the per-period-scoped linker.
- **Decision:** (1) canonical convention is period-relative, documented on converters/linker/schema + pinned by lock tests; (2) `link_actions_to_frames` gains a per-period `min_link_rate`/`on_low_coverage` guard (warn default); (3) `MISMATCH_OVERLAP_FLOOR=0.2` cause-hypothesis decoupled from the symptom threshold; (4) public `validate_time_base` as the guard for pre-filtering consumers; (5) reject library-owned GS bronze time-normalization (hexagonal I/O boundary).
- **Alternatives considered:** absolute convention (rejected — conflicts with every events converter); raise-by-default (rejected — low coverage is a continuum, breaks legitimately-partial matches); match-aggregate threshold (rejected — launders GS 10503's 60.6% over p2's 19%); silent opt-in (rejected — protects exactly the discard-the-report population that needs it least).
- **Consequences:** positive (no silent majority-drop; enforced contract); negative/Hyrum (callers under `-W error` now fail on degraded matches — intended shift-left); neutral (lakehouse opts up to raise + min_link_rate≈0.9 + wires `validate_time_base` at work-unit entry).
- **Scope note (record explicitly, lakehouse MEDIUM):** the convention-pinning lock tests (Task 5) enforce the contract only for converters whose `time_seconds` arithmetic the library owns (**Opta, StatsBomb**). **GradientSports `time_seconds` is a verbatim pass-through (`gradientsports.py:416`) originating upstream in the lakehouse — it is guarded lakehouse-side (`validate_time_base` at work-unit entry + the lakehouse boundary test), NOT by these library tests.** State this so coverage is not overstated.
- **Related:** ADR-010 (fail-loud precedent), the spec, this plan.

- [ ] **Step 6: Add the CLAUDE.md architecture line**

In `CLAUDE.md`, amend the tracking-namespace "Linkage primitive" sentence to note the contract + guard, e.g. append:

> PR-S## adds the period-relative `time_seconds` contract (ADR-017): `link_actions_to_frames` gains per-period `min_link_rate`/`on_low_coverage` (warn-default) coverage guard + `LinkReport.per_period_link_rate`; public `validate_time_base` + `TimeBaseDiagnosis` pre-link affordance for time-filtering consumers; `MISMATCH_OVERLAP_FLOOR=0.2` time-base-mismatch diagnostic.

- [ ] **Step 7: Verify docs don't break imports/doctests**

Run: `python -m pytest tests/tracking/test_time_base_contract.py tests/spadl/test_time_seconds_convention.py -v`
Expected: all PASS (docstring edits are inert).

---

## Task 7: Full local verification (lint trio + suite)

**Files:** none (verification only)

- [ ] **Step 1: ruff check**

Run: `ruff check silly_kicks/ tests/ scripts/`
Expected: `All checks passed!` (fix any I001 import-sort with `ruff check --fix`; ensure `__all__` stays sorted).

- [ ] **Step 2: ruff format check**

Run: `ruff format --check silly_kicks/ tests/ scripts/`
Expected: clean. If it reports files, run `ruff format silly_kicks/ tests/ scripts/` and re-check.

- [ ] **Step 3: pyright (whole package)**

Run: `pyright silly_kicks/`
Expected: `0 errors`. (Common spots: the `Literal` import; `dict[int, tuple[float, float]]` annotations; the `merged_all.groupby(...)` Series typing — add `# type: ignore[...]` matching the file's existing idiom only if pyright flags a pandas-stubs limitation.)

- [ ] **Step 4: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all pass (read the actual `N passed` line — do not pipe to `tail`). Note the count.

- [ ] **Step 5: Targeted re-confirm**

Run: `python -m pytest tests/tracking/test_time_base_contract.py tests/spadl/test_time_seconds_convention.py tests/test_tracking_utils_link.py -v`
Expected: all PASS.

---

## Task 8: C4 check (likely no-op)

**Files:** possibly `docs/c4/architecture.{dsl,html}`

- [ ] **Step 1: Check whether the `tracking` container description needs an edit**

The C4 `tracking` container description enumerates trained models + an aggregator count, and the linkage primitive. This change adds **no** new model / KDE backend / aggregator — but it does extend the linkage primitive's contract. Inspect `docs/c4/architecture.dsl` for the `tracking` container description string.

Run: `grep -n "link" docs/c4/architecture.dsl` (or open it).

- [ ] **Step 2: If the description references the linker's behavior, add the contract token; else skip**

If the description mentions linkage/coverage, add a short token (e.g. "period-relative time-base contract + coverage guard") and regenerate via the `mad-scientist-skills:c4` pipeline (structurizr.war → plantuml.jar → c4_assemble.py), confirming the token appears in the embedded SVG + DSL panel. If the description does not enumerate linker behavior, **no regen needed** — record "C4: no structural change" for the final review.

---

## Task 9: Version bump + single commit + PR (sentinel-gated)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`

- [ ] **Step 1: Reconcile the version number AND the ADR number**

Run: `git fetch origin && git log origin/main --oneline -8`
Determine the next free minor. The parallel session has 4.12.0 + ADR-016 (ghost-gk serve-mean) in flight. If 4.12.0 has landed on `origin/main`, use **4.13.0**; else **4.12.0**. Confirm the chosen number is unused in `CHANGELOG.md`.

Also reconcile the **ADR number** at the same time: confirm `docs/superpowers/adrs/ADR-017-*.md` does not now collide with a landed parallel ADR-017. ADR-015 (TF-17 PR-C) and ADR-016 (ghost-gk serve-mean) are spoken-for; if either parallel branch landed a *different* ADR at 017, bump this ADR (and every in-repo `ADR-017` reference: the ADR filename/title, CLAUDE.md line, docstrings, CHANGELOG) to the next free number before committing.

- [ ] **Step 2: Bump all 5 sites**

Edit `pyproject.toml` (`version = "X.Y.0"`), `silly_kicks/__init__.py` (`__version__`), prepend a `CHANGELOG.md` section (feature summary: contract docs, per-period guard, `validate_time_base`, `TimeBaseDiagnosis`, `per_period_link_rate`; Hyrum note: callers under `-W error` now fail on degraded matches), update `TODO.md`, then:

Run: `uv lock`
Expected: `uv.lock` updates the silly-kicks version pin only.

- [ ] **Step 3: Re-run the lint trio + targeted tests after the bump**

Run: `ruff check silly_kicks/ tests/ scripts/ && ruff format --check silly_kicks/ tests/ scripts/ && python -m pytest tests/tracking/test_time_base_contract.py -q`
Expected: clean + pass.

- [ ] **Step 4: Stage + present the diff — HOLD for commit authorization**

Run: `git add -A && git status && git --no-pager diff --staged --stat`

**STOP.** Present the staged diff summary + the proposed commit message to the user. Do **NOT** create `~/.claude-git-approval` yourself. Wait for the user to either (a) explicitly authorize you to create the sentinel for this specific commit, or (b) run the commit themselves from the CLI.

Proposed commit message (write to a temp file via the Write tool, commit with `git commit -F`, per the memory note — never inline `-m` with apostrophes):

```
feat(tracking): period-relative time_seconds contract + per-period link-coverage guard -- silly-kicks X.Y.0 (ADR-017)

Documents and enforces the canonical period-relative time_seconds convention;
adds a per-period min_link_rate/on_low_coverage guard to link_actions_to_frames
(warn default), LinkReport.per_period_link_rate, a public validate_time_base +
TimeBaseDiagnosis affordance for time-filtering consumers, and convention-pinning
lock tests. Resolves the GradientSports period-2 silent-data-loss class.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- [ ] **Step 5: After the commit lands — push + PR**

Run: `git push -u origin feat/tracking-time-base-contract`
Then `gh pr create --title "..." --body-file <tmp>` (body via the Write tool; end with the Claude Code footer). Push/PR are chat-approval only (not sentinel-gated).

---

## Self-Review (completed by plan author; updated after lakehouse plan review)

**Spec coverage:** §4.1 docs → Task 6; §4.2 detector → Task 2; §4.3 linker guard + per_period_link_rate-from-merged_all → Tasks 1+4; §4.4 validate_time_base + pre-filter framing → Task 3 (docstring) + Task 6; §5 tests incl. laundering keystone + convention-pinning → Tasks 1, 4, 5; §6 edge cases (empty actions, zero-frame period, all-unlinked) → Task 1 default field + Task 4 tests; §7 housekeeping (ADR-017, CLAUDE.md, C4, version) → Tasks 6, 8, 9. All §8 resolutions reflected (MISMATCH_OVERLAP_FLOOR=0.2 Task 2; validate raise-default Task 3; worst-first Task 2).

**Lakehouse plan-review items folded in:** HIGH `stacklevel=3` (Task 4 helper + `__file__` assertion test); MEDIUM cross-task red test moved Task 1→Task 4 (Task 1 now leaves only green tests); MEDIUM converter fixtures reuse the proven `test_opta.py`/`_make_statsbomb_events` shapes; MEDIUM GS lock test dropped as vacuous + GS-guarded-lakehouse-side recorded in ADR-017 + Task 5 scope; LOW integration-fidelity test (Task 4), positional-backcompat lock (Task 1), ADR-number reconciliation (Task 9), unused-helper/misleading-note removed, ADR status wording.

**Placeholder scan:** no deferred fixture bodies remain — Task 5 ships complete Opta + StatsBomb tests mirrored from proven sibling builders. No TBDs.

**Type consistency:** `TimeBaseDiagnosis` fields/property (`has_suspected_mismatch`, `suspected_mismatch_periods: tuple`, `per_period_overlap_fraction`, `per_period_action_range`, `per_period_frame_range`) are used identically across Tasks 2/3/4. `MISMATCH_OVERLAP_FLOOR`, `_diagnose_time_base`, `_format_diagnosis`, `_enforce_link_coverage`, `validate_time_base`, `link_actions_to_frames(..., min_link_rate, on_low_coverage)`, `LinkReport.per_period_link_rate` are named consistently throughout.
