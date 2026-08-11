# SB360 spine: pandas coverage, snapshot dtype, and reclaiming audit coverage — Implementation Plan

> **EXECUTED as 4.79.0. Kept as-written; the unticked `- [ ]` boxes are ORIGINAL, not outstanding.**
> Tasks 1–9 are all done. The plan is retained as the record of what was planned, so the delta
> against the CHANGELOG stays legible — see the companion spec's header annotation for the three
> claims that turned out false.
>
> Two places where execution diverged from this document, both deliberate and both because the
> written step was WRONG rather than merely incomplete:
>
> * **Task 7 Step 1's positive control** asserted `value_changed is True` for a reconstructed
>   pre-4.76.0 ghost fixture. That couples a gate to a defect CONTINUING TO EXIST — 4.76.0 repaired
>   the ghost path, so it now refuses rather than changing its value. Implemented as a split
>   control instead (planted case + "surfaced, not cleared"), which is the same correction this repo
>   already made once when `test_at_least_one_column_was_adjudicated_a_fabrication` broke.
> * **Task 8's fix-list is EMPTY.** The plan reasonably assumed the discriminator would convict
>   something. It convicted 0 of 24 — after two revisions of the instrument, the first two of which
>   measured the wrong quantity. The Global Constraint *"`NOT_EXERCISED_BUDGET` can only RISE"* held,
>   and for the reason given.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Declare and assert the pandas coverage CI already has by accident, answer the deferred
`snapshot_to_tracking_frames` dtype question on both pandas majors, reclaim SB360 audit coverage by
adding a roster rather than widening one, and replace a 24-file grep with a behavioural
discriminator that decides which velocity fixtures actually need fixing.

**Architecture:** Four independent items sharing one branch. Items 1→2 are ordered (2's value is
that it runs on both majors; 1 is what stops that silently ceasing to be true). Items 3 and 4 are
independent of both and of each other. Nothing here touches `_kernels.py`, `utils.py` or
`tests/scripts/`, which the parallel `cycleb-artifact-contracts` session owns.

**Tech Stack:** Python 3.10+, pandas (2.x and 3.x), pytest, PyYAML (already a test dep — used by the
existing CI wiring guards), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-09-sb360-spine-pandas-coverage-and-fixture-widening-design.md`
(rev 2, commit `502d4b0`). Read §2 before Task 1 and §3 Item 3 before Task 5.

## Global Constraints

- **ONE commit at the end.** Tasks commit as they go for resumability; the final task squashes to a
  single commit via `git reset --soft` before anything is proposed. Never propose a commit without
  explicit owner approval.
- **Feature branch** `sb360-pandas-coverage-and-fixture-widening`, already created off `main` @
  `a29ae0f`. Do not merge.
- **Do not claim a version number.** `main` is 4.77.1; the next number is taken at commit-prep.
- **Every new gate must be observed RED** against a reintroduced defect before it is accepted, and
  **the mutation must match the HAZARD, not the implementation**. Where a task names the mutation,
  use that one.
- **Nothing deferred silently.** Anything not done lands in `TODO.md` with its reason, in the same
  commit.
- **Lint at CI scope only:** `python -m ruff check silly_kicks/ tests/ scripts/` and
  `python -m ruff format --check silly_kicks/ tests/ scripts/`. Never `ruff check .`.
- **pyright runs bare:** `python -m pyright`. Neither tool is on PATH; use `python -m`.
- **Full suite:** `python -m pytest tests/ -m "not e2e" -q`. Takes ~19 minutes — run it in the
  background and keep working.
- **`NOT_EXERCISED_BUDGET` can only RISE in this cycle.** It counts `(entry, axis, roster, column)`
  tuples (`tests/sb360/_registry.py:156`, equality-asserted at
  `tests/sb360/test_registry_surface.py:158`), and the registry holds 35 entries × 2 rosters. A
  third roster ADDS tuples. Any task claiming a drop is wrong.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tests/test_ci_pandas_span_wired.py` | **Create.** Structural guard: the resolved CI leg set straddles the Python 3.11 boundary. | 2 |
| `.github/workflows/ci.yml` | **Modify.** Per-leg pandas artifact + aggregation job — ONLY if Task 1 decides it earns its place. | 3 |
| `tests/tracking/test_snapshot_id_dtype_across_pandas.py` | **Create.** Behavioural dtype differential for `snapshot_to_tracking_frames`. | 4 |
| `tests/sb360/_fixture.py` | **Modify** `_player_layout` (`:110-142`): add the `gk_one_end` branch. | 5 |
| `tests/sb360/_registry.py` | **Modify.** `VISIBILITY_ROSTERS` (`:24`), `NOT_EXERCISED_BUDGET` (`:156`), and the new coverage metric. | 5, 6 |
| `tests/sb360/_regenerate.py` | **Modify** `AXES` (`:39`). | 5 |
| `tests/sb360/test_axis_locks.py` | **Modify** `_AXES` (`:24`). | 5 |
| `tests/sb360/_entries/*.py` | **Modify.** 35 entries gain a `gk_one_end` block inside `visibility` — REGENERATED, not hand-written. | 5 |
| `tests/sb360/test_registry_surface.py` | **Modify.** Assert the new coverage metric beside the budget. | 6 |
| `scripts/audit_velocity_fixtures.py` | **Create.** The velocity discriminator. Reports, fixes nothing. | 7 |
| `tests/scripts/test_audit_velocity_fixtures.py` | **Create.** Positive control for the discriminator. | 7 |
| `TODO.md`, `CHANGELOG.md`, `CLAUDE.md` | **Modify.** Corrections and findings. | 9 |

---

## Task 1: DECISION — does the CI aggregation job earn its existence?

**This task produces a written decision, not code.** It exists because the spec's Item 1 half (1)
was authored in response to review and has had zero adversarial reading. Deciding by building is
cheaper than arguing: the cost is measurable in ten minutes.

**Files:**
- Create: `docs/superpowers/plans/_decisions/2026-08-09-ci-aggregation-job.md` (scratch; folded into
  the spec in Task 9)

**Interfaces:**
- Produces: a GO/NO-GO consumed by Task 3. On NO-GO, Task 3 is skipped and Task 9 must edit the
  spec's §5 gate list and §7 ADR criterion, both of which currently assume the job exists.

- [ ] **Step 1: State what the job catches that the structural guard does not**

Write the two hazards down explicitly:

1. **Structural guard (Task 2) catches:** someone edits `ci.yml` so the leg set no longer straddles
   Python 3.11 — by axis deletion, by `exclude`, or by changing `include`.
2. **Aggregation job would additionally catch:** the leg set is unchanged but the *resolved pandas*
   collapses to one major anyway. Concretely: pandas raises its minimum Python above 3.11, or a
   pandas 4 lands and 3.11/3.12 both jump to it, or a transitive pin caps pandas on one leg.

The question is whether hazard (2) is worth a CI job. Note it is **not** hypothetical in kind — this
repo has one measured instance of a silent pandas-3 behaviour change (DAS all-NaN) — but that
instance was a *behaviour* change, which neither guard catches; only a test does.

- [ ] **Step 2: Measure the cost — write the job, count the lines**

Draft it (do not commit):

```yaml
      - name: Record resolved pandas major
        run: |
          python -c "import pandas, pathlib; pathlib.Path('pandas-major.txt').write_text(pandas.__version__.split('.')[0])"
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: pandas-major-${{ matrix.os }}-${{ matrix.python-version }}
          path: pandas-major.txt

  pandas-span:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          pattern: pandas-major-*
          path: majors/
      - name: The matrix must span both pandas majors
        run: |
          python - <<'PY'
          import pathlib, sys
          majors = {p.read_text().strip() for p in pathlib.Path("majors").rglob("pandas-major.txt")}
          if not majors:
              sys.exit("::error::no pandas-major artifacts found -- the recording step did not run")
          if len(majors) < 2:
              sys.exit(f"::error::CI resolved pandas major(s) {sorted(majors)} on every leg. The "
                       f"differential coverage this repo relies on is GONE. See ADR/spec: the "
                       f"coverage was never declared, only inherited from the Python matrix.")
          print(f"pandas majors covered: {sorted(majors)}")
          PY
```

Record the actual line count and the added CI wall-clock (one extra job, ~20s).

- [ ] **Step 3: Decide, and write the decision with its reason**

Recommended GO, on these grounds — but the decider owns this:

- The job is the ONLY place the span is observed; Task 2 asserts a *proxy* (the Python boundary),
  and the spec's own §2 argues against inferring what can be observed.
- It fails loudly with a message naming the property, which is what a future reader needs.
- Cost is one short job, no new dependency, no effect on the test legs' wall-clock.

Grounds for NO-GO, if the decider prefers: hazard (2) requires an upstream ecosystem change that
would be noisy anyway, and CI surface has cost beyond its line count.

**On NO-GO:** record it, skip Task 3, and add to Task 9 the edits to spec §5 (drop half (1) from the
gate list) and §7 (the ADR criterion currently says "if the guard asserts a property other code must
respect" — with only the structural half, it does not).

- [ ] **Step 4: Commit the decision**

```bash
git add docs/superpowers/plans/_decisions/2026-08-09-ci-aggregation-job.md
git commit -m "docs(plan): decide the CI pandas-span aggregation job"
```

---

## Task 2: The structural pandas-span guard

**Files:**
- Create: `tests/test_ci_pandas_span_wired.py`
- Reference (do not modify): `tests/test_ci_slow_gating_wired.py` — the house pattern for parsing
  `ci.yml`; it reads `matrix.get("include", [])` rather than trusting the axes, which is exactly the
  mistake this guard must not repeat.

**Interfaces:**
- Consumes: nothing.
- Produces: `resolved_legs(matrix) -> list[dict]` — a module-level helper returning the concrete leg
  set (os × python-version − exclude + include). Task 3 does not use it; it is internal.

- [ ] **Step 1: Write the failing test**

```python
"""Structural guard: CI's leg set must still span both pandas majors.

`pyproject.toml` pins `pandas>=2.1.1,!=3.0.4` with NO upper bound, so pip resolves the newest
compatible pandas per interpreter, and pandas 3 requires Python >= 3.11. Measured on run
31316804815: ubuntu-3.10 -> pandas 2.3.3, every other leg -> 3.0.5.

That coverage is REAL but ACCIDENTAL -- nothing declared it. This guard declares it.

It asserts over the RESOLVED LEG SET, never the `python-version` axis: legs are
os x python-version MINUS `exclude` PLUS `include`, and `exclude` is already the pruning mechanism
in use (two windows legs). Excluding ubuntu/3.10 collapses the span while leaving "3.10" in the
axis -- an axis-based assertion would pass.
"""

from __future__ import annotations

import itertools
import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

#: pandas 3 requires Python >= 3.11, so a leg below it resolves pandas 2 and a leg at or above it
#: resolves pandas 3. This is the ASSUMPTION that makes the structural check a valid proxy; if
#: pandas changes its minimum Python, this constant is what must move -- not the assertion.
_PANDAS3_MIN_PY = (3, 11)


def _pyver(leg: dict) -> tuple[int, ...]:
    return tuple(int(p) for p in str(leg["python-version"]).split("."))


def resolved_legs(matrix: dict) -> list[dict]:
    """os x python-version, MINUS `exclude`, PLUS `include` -- GitHub's own resolution order."""
    base = [
        {"os": os_, "python-version": py}
        for os_, py in itertools.product(matrix["os"], matrix["python-version"])
    ]
    for ex in matrix.get("exclude", []):
        base = [leg for leg in base if not all(leg.get(k) == v for k, v in ex.items())]
    for inc in matrix.get("include", []):
        for leg in base:
            if all(leg.get(k) == v for k, v in inc.items() if k in leg):
                leg.update(inc)
    return base


def test_ci_leg_set_spans_both_pandas_majors() -> None:
    matrix = yaml.safe_load(_CI.read_text(encoding="utf-8"))["jobs"]["test"]["strategy"]["matrix"]
    legs = resolved_legs(matrix)
    below = [leg for leg in legs if _pyver(leg) < _PANDAS3_MIN_PY]
    at_or_above = [leg for leg in legs if _pyver(leg) >= _PANDAS3_MIN_PY]

    assert below and at_or_above, (
        f"CI's resolved leg set no longer straddles Python {_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}, "
        f"so every leg resolves the SAME pandas major and the differential coverage this repo "
        f"relies on is gone. legs={[(l['os'], l['python-version']) for l in legs]}. "
        f"ASSUMPTION: pandas 3 requires Python >= {_PANDAS3_MIN_PY[0]}.{_PANDAS3_MIN_PY[1]}. If "
        f"pandas changed that, fix _PANDAS3_MIN_PY -- do NOT delete this assertion, and do not "
        f"'fix' it by moving the boundary to match a matrix that lost its old leg."
    )


def test_resolved_legs_honours_exclude_not_just_the_axis() -> None:
    """Non-vacuity for the resolver itself: `exclude` must actually remove a leg.

    Without this, `resolved_legs` could ignore `exclude` entirely and the guard above would still
    pass on today's matrix -- while missing the likeliest way the span gets destroyed.
    """
    matrix = {
        "os": ["ubuntu-latest"],
        "python-version": ["3.10", "3.12"],
        "exclude": [{"os": "ubuntu-latest", "python-version": "3.10"}],
    }
    legs = resolved_legs(matrix)
    assert [leg["python-version"] for leg in legs] == ["3.12"]
```

- [ ] **Step 2: Run it and verify it PASSES on the current matrix**

Run: `python -m pytest tests/test_ci_pandas_span_wired.py -v`
Expected: 2 passed. (This guard is written against a healthy matrix, so it starts green — the RED
observation is Step 3, which is the part that matters.)

- [ ] **Step 3: Observe it RED via `exclude` — the hazard, not the implementation**

Write a throwaway script (scratchpad, not committed) that copies `ci.yml`, appends
`{os: ubuntu-latest, python-version: "3.10"}` to `matrix.exclude`, runs only
`test_ci_leg_set_spans_both_pandas_majors` against the mutated copy, restores, and prints the
verdict. Do **not** mutate by deleting `"3.10"` from the axis — that mutation matches the
implementation rather than the hazard, and it is the reason an earlier draft of this guard was
wrong.

Expected: FAIL, with the message naming the leg set.

Record the pasted output in the task notes. A gate whose red case was never observed is not accepted.

- [ ] **Step 4: Lint and commit**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format silly_kicks/ tests/ scripts/
git add tests/test_ci_pandas_span_wired.py
git commit -m "test(ci): assert the resolved leg set spans both pandas majors"
```

---

## Task 3: The pandas-span aggregation job — ONLY IF TASK 1 SAID GO

**Skip this task entirely on NO-GO.** If skipped, say so explicitly in Task 9's notes; do not leave
it silently undone.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_ci_pandas_span_wired.py` (add the wiring assertions below)

**Interfaces:**
- Consumes: Task 1's GO decision; Task 2's test module (assertions are appended to it).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Add the recording step and aggregation job to `ci.yml`**

Use the YAML drafted in Task 1 Step 2 verbatim. Pin both actions to the SHAs already used elsewhere
in the repo (`upload-artifact@043fb46d…`, `download-artifact@3e5f45b2…`) — this repo pins every
action by SHA and a floating tag would be inconsistent.

- [ ] **Step 2: Add the wiring assertions**

```python
def test_the_aggregation_job_exists_and_needs_test() -> None:
    """Without `needs: test` the job runs before the artifacts exist and passes vacuously."""
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    job = wf["jobs"].get("pandas-span")
    assert job is not None, "the pandas-span aggregation job is gone; the span is only proxied now"
    assert job["needs"] == "test"


def test_every_test_leg_records_its_pandas_major() -> None:
    """The aggregation job asserts over a UNION; a leg that records nothing shrinks it silently."""
    wf = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    steps = wf["jobs"]["test"]["steps"]
    assert any("Record resolved pandas major" in str(s.get("name", "")) for s in steps)
    assert any(
        "upload-artifact" in str(s.get("uses", "")) and "pandas-major" in str(s.get("with", {}).get("name", ""))
        for s in steps
    )
```

- [ ] **Step 3: Verify the aggregation script logic locally**

The job's Python runs in CI, so verify its logic here rather than discovering it on a push. Write a
scratchpad script that extracts the heredoc body from `ci.yml` (do NOT retype it) and runs it against
three fabricated `majors/` trees: `{2,3}` → exit 0; `{3}` → exit non-zero; empty → exit non-zero.

Expected: `PASS / FAIL / FAIL`. Paste the output into the task notes.

- [ ] **Step 4: Lint and commit**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
git add .github/workflows/ci.yml tests/test_ci_pandas_span_wired.py
git commit -m "ci: record each leg's pandas major and assert the union spans both"
```

---

## Task 3b: DECIDE the ADR question — NOW, not at commit-prep

Spec §7 says this decision *"gets its own plan step, immediately after Item 1's guard is working"*,
and gives the reason: *"a deferred decision with no owner does not get made — it gets discovered at
commit-prep, when the cheap moment to write an ADR has passed."* An earlier draft of this plan
routed it to Task 9 Step 3, which **is** commit-prep. That is the deferral §7 exists to prevent.

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-<slug>.md` — ONLY on a YES.

- [ ] **Step 1: Apply the spec's own criterion to the guard as built**

> If the guard ends up asserting a property other code must respect (a declared pandas span), it is
> a contract and earns an ADR; if it only pins CI's own configuration, it is a wiring guard like
> `test_ci_slow_gating_wired.py` and does not.

Answer it against what Tasks 2 and 3 actually produced, not what they were planned to produce.
Relevant input: whether Task 1 said GO. With the aggregation job the guard asserts a property of the
delivered ARTIFACT set; without it, only of `ci.yml`.

- [ ] **Step 2: Write the ADR, or record the NO with its reason**

On YES: take the next free ADR number (list `docs/superpowers/adrs/` — do not assume) and follow
`ADR-TEMPLATE.md`. On NO: one paragraph in the task notes, carried into Task 9's CHANGELOG entry so
the decision is visible rather than absent.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/adrs/
git commit -m "docs(adr): decide whether the pandas-span guard is a contract"
```

---

## Task 4: The `snapshot_to_tracking_frames` dtype differential test

**Files:**
- Create: `tests/tracking/test_snapshot_id_dtype_across_pandas.py`
- Reference: `silly_kicks/tracking/_snapshot.py` (locate `snapshot_to_tracking_frames`; confirm its
  exact import path from `silly_kicks.tracking` before writing the test)

**Interfaces:**
- Consumes: `silly_kicks.tracking.snapshot_to_tracking_frames`, `silly_kicks.id_compat.ids_match`.
- Produces: nothing consumed downstream.

**Read first:** spec §3 Item 2. The assertion is **behavioural** — that `id_compat` comparisons keep
working — NOT a dtype literal. A dtype-literal test passes or fails on whatever pandas returns, which
is precisely what left this question unverifiable for two cycles.

- [ ] **Step 1: Confirm the API before writing against it**

Run:
```bash
python -c "import silly_kicks.tracking as T, inspect; print(inspect.signature(T.snapshot_to_tracking_frames))"
```
Record the real signature. If it differs from what this task assumes, adapt the test — do not adapt
the API.

- [ ] **Step 2: Write the failing test**

```python
"""`snapshot_to_tracking_frames` id dtypes, across whichever pandas the leg resolves.

ADR-055 dropped a dtype PIN as unimplementable: `TRACKING_FRAMES_COLUMNS` declares `int64` for
`player_id`/`team_id`, the ball row is NA in both, and `int64` cannot hold NA
(`IntCastingNaNError` on every snapshot). The residual question -- does the behaviour DIFFER across
pandas majors? -- was never checked, and TODO wrongly recorded that CI had no pandas-3 leg.

So this asserts the property consumers actually depend on (ADR-019): ids surviving the snapshot must
still compare equal, via `id_compat`, to the same ids in their source form. That holds or fails
identically on both majors, and the leg's pandas is reported in the failure message so a divergence
names itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as T
from silly_kicks.id_compat import ids_match


def _snapshot_rows(id_dtype: str):
    """Two players per event. Required columns per the docstring: action_id, team_id,
    is_goalkeeper, x, y (player_id optional -- supplied here because it is under test)."""
    return pd.DataFrame(
        {
            "action_id": [0, 0],
            "player_id": np.array([7, 9]) if id_dtype == "numpy_int" else pd.array([7, 9], dtype=id_dtype),
            "team_id": np.array([1, 2]) if id_dtype == "numpy_int" else pd.array([1, 2], dtype=id_dtype),
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "is_goalkeeper": [True, False],
        }
    )


def _actions():
    """`snapshot_to_tracking_frames` derives game_id/period_id/time_seconds and the BALL position
    from the actions frame -- the ball row it synthesizes is where the NA ids come from."""
    return pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [10.0],
            "start_x": [50.0],
            "start_y": [34.0],
        }
    )


# NUMPY INT is the case with a recorded 2.3.3 measurement ("the concat yields float64 for the
# as-built numeric-int fixture"), so it is the one the differential most needs. It is built with
# `np.array`, NOT `pd.array(..., dtype="int64")` -- the latter yields a NumpyExtensionArray, and
# while the resulting Series dtype is int64 either way, `np.array` removes the question entirely.
@pytest.mark.parametrize("id_dtype", ["numpy_int", "Int64", "object"])
def test_ids_survive_the_snapshot_comparably(id_dtype: str) -> None:
    frames, _links = T.snapshot_to_tracking_frames(_snapshot_rows(id_dtype), _actions())

    for col, probe in (("player_id", 7), ("team_id", 1)):
        matched = ids_match(frames[col], probe)
        assert matched.any(), (
            f"id {probe!r} in column {col!r} no longer compares equal after the snapshot on "
            f"pandas {pd.__version__} with source dtype {id_dtype!r}. Result dtype was "
            f"{frames[col].dtype!r}. This is the ADR-019 property consumers depend on, NOT a "
            f"dtype literal -- if pandas changed the concat result, that is the finding."
        )


def test_the_ball_row_stays_NA_rather_than_becoming_a_sentinel() -> None:
    """A non-NA sentinel bypasses `pd.isna` routing and crashes downstream opponent guards
    (ADR-027). The synthesized ball row belongs to no team and holds no player."""
    frames, _links = T.snapshot_to_tracking_frames(_snapshot_rows("Int64"), _actions())
    ball = frames[frames["is_ball"].astype("boolean").fillna(False).astype(bool)]
    assert len(ball) == 1, f"expected one synthesized ball row, got {len(ball)}"
    assert ball["team_id"].isna().all(), (
        f"the ball row's team_id is not NA on pandas {pd.__version__} -- an absent team became a "
        f"value, which ADR-027 records as a crash source in downstream opponent guards"
    )
```

- [ ] **Step 3: Run it and record the measurement on BOTH majors available locally**

Run: `python -m pytest tests/tracking/test_snapshot_id_dtype_across_pandas.py -v`

Then repeat on the pandas-3 environment. **One already exists** — the throwaway venv built during
4.77.1 at
`<scratchpad>/sbvenv` (pandas 3.0.5). If it is gone, rebuild: `python -m venv sbvenv && sbvenv/Scripts/python -m pip install -e . pyarrow`.

Record BOTH results and the resolved `pd.__version__` for each. **Assert the majors actually
DIFFER before believing the comparison** — the repo venv should report 2.x and `sbvenv` 3.x
(3.0.5 when built). A stale venv resolving pandas 2 yields a same-major 'differential' that
proves nothing, which is this cycle's own subject applied to its own instrument. This is the
measurement the item exists to produce.

- [ ] **Step 4: Observe it RED against a behaviour-breaking mutation**

Mutate `snapshot_to_tracking_frames` to drop the ball row's NA (e.g. `fillna(0)` on the id columns)
and confirm `test_the_ball_row_stays_NA_rather_than_becoming_a_sentinel` fails.

**Do NOT attempt a "stringify the ids" mutation** — the assertion is that `id_compat` still matches,
and `id_compat` matches across string/numeric by design, so the test survives it. An earlier spec
draft offered exactly that example; it was deleted for this reason.

- [ ] **Step 5: If the two majors DIVERGE, land it as a strict xfail**

Only if Step 3 showed divergence. Mark the failing parametrization:

```python
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED DIVERGENCE across pandas majors: on 2.x <record exact behaviour>, on 3.x "
        "<record exact behaviour>. Strict so the marker must be deleted when the behaviour is "
        "fixed rather than rotting into an exemption. Repair scoped in TODO.md."
    ),
)
```
and add the `TODO.md` row in Task 9. If they AGREE, skip this step and say so.

- [ ] **Step 6: Lint and commit**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
git add tests/tracking/test_snapshot_id_dtype_across_pandas.py
git commit -m "test(tracking): pin snapshot id comparability across pandas majors"
```

---

## Task 5: The `gk_one_end` roster

**Files:**
- Modify: `tests/sb360/_fixture.py:137-141` (the `_player_layout` roster branches)
- Modify: `tests/sb360/_registry.py:24` (`VISIBILITY_ROSTERS`)
- Modify: `tests/sb360/_regenerate.py:39` (`AXES`)
- Modify: `tests/sb360/test_axis_locks.py:24` (`_AXES`)
- Modify: `tests/sb360/_entries/*.py` (35 entries gain a `gk_one_end` key inside `visibility`) —
  **REGENERATED via `tests/sb360/_regenerate.py`, never hand-written**
- Modify: `tests/sb360/_registry.py:156` (`NOT_EXERCISED_BUDGET`, which will RISE)

**Interfaces:**
- Consumes: nothing.
- Produces: the roster name `"gk_one_end"` in `VISIBILITY_ROSTERS`, consumed by Task 6's metric.

**Read first:** spec §3 Item 3. `gk_absent` is a real visibility axis and the only case exercising
the both-absent refusal path — this task ADDS a roster, it does not widen one.

- [ ] **Step 1: PIN the `gk_absent` slice before touching anything**

CI re-derives verdicts, so a leak into the wrong axis would be absorbed into the new baseline
silently. Capture the slice first:

```bash
python -c "
import json
from tests.sb360._registry import SB360_ENTRIES, iter_verdicts
# NOTE: _load_entry_modules() already runs at module scope (_registry.py:222) -- calling it
# again is redundant. If SB360_ENTRIES is empty here, that auto-invocation is what broke.
slice_ = {}
for name, e in SB360_ENTRIES.items():
    for axis, roster, col, v in iter_verdicts(e):
        if roster == 'gk_absent':
            slice_[f'{name}|{col}'] = [v.observation, v.adjudication]
print(json.dumps(slice_, indent=2, sort_keys=True))
" > gk_absent_before.json
wc -l gk_absent_before.json
```
Keep `gk_absent_before.json` in the scratchpad (NOT the repo — an untracked file in the repo makes
every artifact driver refuse on a dirty tree).

**Then assert the pin is non-empty, or Step 6 compares nothing and says UNCHANGED.** `diff` of two
`{}` files succeeds, and the capture can silently produce `{}` if the inline import fails on
`sys.path` or the roster filter matches nothing:

```bash
python -c "
import json, sys
d = json.load(open(r'<scratchpad>/gk_absent_before.json'))
sys.exit(0 if len(d) > 100 else f'pin holds only {len(d)} verdicts -- the capture FAILED, and '
                                f'Step 6 would print UNCHANGED having compared nothing')
"
```
35 entries each contributing roughly one verdict per column puts the real figure well above 100;
record the actual number.

- [ ] **Step 2: Add the roster branch to `_player_layout`**

In `tests/sb360/_fixture.py`, after the `defender_absent` branch:

```python
    elif roster == "gk_one_end":
        # ONE keeper visible, the other off-frame -- the modal SB360 freeze-frame (the defending
        # keeper is in-frame 97.7%/92.2% of the time on shots while the acting side's usually is
        # not), and the case that breaks `gk_absent`'s degeneracy.
        #
        # Keeping the HOME keeper (base_x 5.0) makes team 1 RESOLVE to x=0. Team 2 falls to the
        # outfield rung, whose mean base_x is 76.5 (its ten outfielders sit at 60/71/82/93 by
        # `60 + (i % 4) * 11`), which is above the 52.5 midline, so it guesses x=105. The two ends
        # DIFFER, the map is non-degenerate, and `attacked_goal` resolves -- which is what makes
        # the five `add_cover_shadows` columns exercisable again.
        rows = [r for r in rows if not (r["is_goalkeeper"] and r["team_id"] == AWAY_TEAM_ID)]
```

- [ ] **Step 3: Verify the map is actually non-degenerate BEFORE regenerating 35 entries**

```bash
python -c "
from silly_kicks.tracking import resolve_defended_goals
from tests.sb360._fixture import build_leg_a
_actions, frames, _links = build_leg_a(roster='gk_one_end')
gm = resolve_defended_goals(frames)
print('resolved:', dict(gm.resolved))
print('guessed :', dict(gm.guessed))
print('unresolved:', gm.unresolved)
"
```
Expected: two entries across `resolved` + `guessed` with **different** end values (one 0.0, one
105.0). If both are 105.0 the roster has not broken the degeneracy — stop and fix the layout before
going further. Paste the output into the task notes.

- [ ] **Step 4: Register the roster in all three declaration sites**

```python
# tests/sb360/_registry.py:24
VISIBILITY_ROSTERS: tuple[str, ...] = ("gk_absent", "defender_absent", "gk_one_end")

# tests/sb360/_regenerate.py:39
AXES = (
    ("velocity", "full"),
    ("visibility", "gk_absent"),
    ("visibility", "defender_absent"),
    ("visibility", "gk_one_end"),
)

# tests/sb360/test_axis_locks.py:24
_AXES = [
    ("velocity", "full"),
    ("visibility", "gk_absent"),
    ("visibility", "defender_absent"),
    ("visibility", "gk_one_end"),
]
```

- [ ] **Step 5: Regenerate the entries, then adjudicate**

```bash
python tests/sb360/_regenerate.py
python -m pytest tests/sb360/ -q
```

The regenerator writes machine OBSERVATIONS; a human writes each `adjudication` and `rationale`
(ADR-053 — a machine cannot tell *fabricated* from *legitimately different*). Work through the new
`gk_one_end` blocks with `tests/sb360/_adjudicate.py` as the guide.

**The five `add_cover_shadows` columns are the point of this task.** They must come back
non-`not_exercised` under `gk_one_end`. Name them individually in the task notes:
`n_blocked_receivers`, `n_potential_receivers`, `blocking_score`, `blocked_threat_fraction`,
`max_single_defender_blocking_score`. (The sixth, `max_single_defender_player_id`, is
`not_exercised` for an unrelated reason — no pressing sequence in the fixture — and is expected to
stay that way.)

- [ ] **Step 6: Assert the `gk_absent` slice is BYTE-IDENTICAL**

Re-run Step 1's command into `gk_absent_after.json` and diff:

```bash
diff gk_absent_before.json gk_absent_after.json && echo "gk_absent UNCHANGED (required)"
```
Any difference means the roster construction leaked into the wrong axis. That is a **defect, not a
rebaseline** — fix the leak, do not accept the new values.

- [ ] **Step 7: Raise `NOT_EXERCISED_BUDGET` with a per-tuple reason**

It will RISE (a third roster adds tuples). Set the measured value and extend the docstring above it
with the enumerated new `not_exercised` tuples and why each is unexercised. An unexplained rise is
not acceptable; the constant's own docstring already demands a recorded reason.

- [ ] **Step 8: Run the sb360 suite and commit**

```bash
python -m pytest tests/sb360/ -q
python -m ruff check silly_kicks/ tests/ scripts/
git add tests/sb360/
git commit -m "test(sb360): add the gk_one_end roster to reclaim cover-shadow coverage"
```

---

## Task 6: DECISION + build — `columns_exercised_on_no_roster`

**This task begins with a decision.** The metric was invented to replace an impossible criterion and
has had no adversarial reading: it does not exist, and its shape, location and computation are
unexamined.

**Files:**
- Modify: `tests/sb360/_registry.py` (add the helper)
- Modify: `tests/sb360/test_registry_surface.py` (assert it beside the budget)

**Interfaces:**
- Consumes: `iter_verdicts`, `SB360_ENTRIES`, `VISIBILITY_ROSTERS` from Task 5.
- Produces: `columns_exercised_on_no_roster() -> set[tuple[str, str]]` returning `(entry, column)`
  pairs that are `not_exercised` under EVERY visibility roster.

- [ ] **Step 1: Decide the shape, and record why**

The question the metric answers is *"which columns are exercised NOWHERE?"* — a per-column property
across the roster sweep, which is what the coverage claim was always about. `NOT_EXERCISED_BUDGET`
counts per-roster tuples and cannot express it.

Recommended shape, but the decider owns it:

- **A computed SET, not a locked int.** A set names its members, so a regression says *which* column
  went dark. A bare count reproduces the budget's weakness — you learn something moved, not what.
- **Live in `_registry.py`** beside `iter_verdicts`, because it is a projection of the registry and
  belongs with the other seams over it.
- **Asserted in `test_registry_surface.py`** beside the budget, against a small locked set — so it
  is a pin, not a report.

Alternative if the decider prefers minimalism: an int constant mirroring the budget. Cheaper, and
strictly worse for diagnosis. Record whichever is chosen.

- [ ] **Step 2: Write the failing test**

```python
def test_no_column_is_unexercised_on_every_roster_except_the_recorded_ones() -> None:
    """A column `not_exercised` under EVERY visibility roster is exercised NOWHERE.

    This is the metric the gk_one_end cycle is actually about. `NOT_EXERCISED_BUDGET` counts
    per-ROSTER tuples, so adding a roster can only raise it -- it cannot express "this column is
    now covered somewhere", which is the coverage claim.
    """
    dark = columns_exercised_on_no_roster()
    assert dark == _EXPECTED_DARK_COLUMNS, (
        f"columns exercised on NO roster changed.\n"
        f"  newly dark: {sorted(dark - _EXPECTED_DARK_COLUMNS)}\n"
        f"  newly lit : {sorted(_EXPECTED_DARK_COLUMNS - dark)}\n"
        f"A column going dark is a coverage regression. A column lighting up is this cycle's "
        f"goal -- update the expectation and say which roster covered it."
    )
```

- [ ] **Step 3: Run it, see it fail (the helper does not exist)**

Run: `python -m pytest tests/sb360/test_registry_surface.py -k unexercised -v`
Expected: FAIL with `NameError`/`ImportError`.

- [ ] **Step 4: Implement the helper**

```python
def columns_exercised_on_no_roster() -> set[tuple[str, str]]:
    """``(entry, column)`` pairs adjudicated ``not_exercised`` under EVERY visibility roster.

    The complement of the coverage claim: a column here is exercised NOWHERE in the visibility
    sweep, whatever the per-roster budget says. Columns absent from a roster's dict are treated as
    unexercised for that roster -- an absent verdict is not evidence of coverage.

    SCOPE, because two numbers over one registry WILL be compared by someone: this walks the
    VISIBILITY rosters only and ignores the velocity axis, while ``NOT_EXERCISED_BUDGET`` counts
    every ``(entry, axis, roster, column)`` tuple including velocity. They are not comparable and
    neither is a subset of the other.
    """
    dark: set[tuple[str, str]] = set()
    for name, entry in SB360_ENTRIES.items():
        for col in entry.columns:
            verdicts = [entry.visibility.get(r, {}).get(col) for r in VISIBILITY_ROSTERS]
            if all(v is None or v.adjudication == "not_exercised" for v in verdicts):
                dark.add((name, col))
    return dark
```

- [ ] **Step 5: Set the expectation to the MEASURED value — which is UNCHANGED by this cycle**

**Do not look for the five cover-shadow columns here; they were never in this set.** Measured on the
pre-change registry, the set is exactly four members:

```
('add_cover_shadows', 'max_single_defender_player_id')
('add_press_commitment', 'press_commitment')
('add_press_commitment', 'press_commitment_closing_speed')
('add_xshot_occurrence', 'xshot_occurrence')
```

The five are `honest_nan` under `defender_absent` (`_entries/_space.py:96-101`), so `all(... ==
"not_exercised")` is already False for them and they cannot "light up". **This pin is expected to
register ZERO change from this cycle** — it is a standing regression guard, not the deliverable. The
deliverable is Task 6b's named-five assertion.

Set `_EXPECTED_DARK_COLUMNS` to the four measured members. If `gk_one_end` adds a member, that is a
NEW column dark on all three rosters and needs a recorded reason before the expectation moves.

- [ ] **Step 6: Observe RED — plant a dark column**

Temporarily flip one `gk_one_end` verdict to `not_exercised` for a column that is otherwise dark on
the other two rosters, and confirm the test fails naming that column. Restore. This tests the PIN,
not the cycle — which is exactly what the pin is for.

- [ ] **Step 7: Commit**

```bash
python -m pytest tests/sb360/ -q
git add tests/sb360/_registry.py tests/sb360/test_registry_surface.py
git commit -m "test(sb360): pin which columns are exercised on no roster at all"
```

---

## Task 6b: The cycle's ACTUAL success criterion, as a real assertion

Task 5 Step 5 required the five columns to come back non-`not_exercised` under `gk_one_end`, but
only as a task note — and a note is not a gate. Neither aggregate can express it (Task 6 Step 5
explains why), so it is asserted directly.

**Files:**
- Modify: `tests/sb360/test_registry_surface.py`

**Interfaces:**
- Consumes: `SB360_ENTRIES` from Task 5's regenerated registry.
- Produces: nothing.

- [ ] **Step 1: Write the assertion**

```python
#: The five columns ADR-055 sent dark on `gk_absent` by making `add_cover_shadows`
#: keeper-dependent. Reclaiming them is the entire point of the `gk_one_end` roster, so the claim
#: is asserted rather than noted. The sixth column, `max_single_defender_player_id`, is
#: deliberately absent: it is `not_exercised` for an UNRELATED reason (the fixture has no pressing
#: sequence) and is expected to stay that way on every roster.
_RECLAIMED_BY_GK_ONE_END = (
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
)


def test_gk_one_end_reclaims_the_cover_shadow_columns() -> None:
    """The `gk_one_end` roster exists to make these five exercisable again.

    Under `gk_absent` both keepers are gone, `resolve_defended_goals` guesses BOTH teams at x=105,
    `attacked_goal` refuses the degenerate map, and every leg goes NaN for a roster-driven reason.
    With one keeper visible the ends differ, so the columns carry real observations.
    """
    entry = SB360_ENTRIES["add_cover_shadows"]
    verdicts = entry.visibility.get("gk_one_end", {})
    assert verdicts, "add_cover_shadows has no gk_one_end block -- the roster was not regenerated"

    unexercised = sorted(
        col
        for col in _RECLAIMED_BY_GK_ONE_END
        if verdicts.get(col) is None or verdicts[col].adjudication == "not_exercised"
    )
    assert not unexercised, (
        f"gk_one_end did not reclaim {unexercised}. The roster's whole purpose is a NON-degenerate "
        f"goal map: one keeper visible so the two ends differ. If these are still unexercised, "
        f"check that _player_layout drops only the AWAY keeper and that resolve_defended_goals "
        f"returns two DIFFERENT ends on this roster (Task 5 Step 3)."
    )
```

- [ ] **Step 2: Run it before Task 5's regeneration to see it FAIL**

If Task 5 is already done, `git stash` its registry changes, run, restore.

Run: `python -m pytest tests/sb360/test_registry_surface.py -k reclaims -v`
Expected: FAIL — no `gk_one_end` block exists. Paste the output.

- [ ] **Step 3: Run it after, and see it PASS**

Expected: PASS. If it fails, the roster did not break the degeneracy — go back to Task 5 Step 3
rather than weakening this assertion.

- [ ] **Step 4: Commit**

```bash
git add tests/sb360/test_registry_surface.py
git commit -m "test(sb360): assert gk_one_end reclaims the five cover-shadow columns"
```

---

## Task 7: The velocity discriminator (reports; fixes nothing)

**Files:**
- Create: `scripts/audit_velocity_fixtures.py`
- Create: `tests/scripts/test_audit_velocity_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `classify(path) -> dict` with keys `claims`, `reaches_consumer`, `value_changed`,
  `deltas`. Task 8 consumes the report to build its fix-list.

**Read first:** spec §3 Item 4. **24 is not a defect count.** A fixture that never reaches a velocity
consumer is correct as written.

**Note on `tests/scripts/`:** do NOT add an `__init__.py` there — it shadows the `scripts` namespace
package and produces ~10 collection errors. Import as `import scripts.audit_velocity_fixtures as mod`.

- [ ] **Step 1: Write the positive-control test FIRST**

The discriminator is an instrument, and an instrument that reports "nothing found" is
indistinguishable from a broken one. ADR-053/4.76.0 already identified two fixtures that declared
`speed_source="native"` with no `vx`/`vy` and reached a scored model on 5-of-26 imputed features:
`tests/tracking/test_ghost_gk_orientation.py` and `tests/tracking/test_action_ltr_mirror_invariance.py`.
Both have since been FIXED, so reconstruct their pre-4.76.0 shape as fixtures inside the test.

```python
def test_the_discriminator_surfaces_the_two_known_ADR053_fixtures(tmp_path):
    """Positive control. These two are the measured instances of the defect this instrument
    hunts; if it cannot see them, a 'nothing found' result means nothing."""
    planted = tmp_path / "test_planted.py"
    planted.write_text(_PRE_4760_GHOST_FIXTURE_SOURCE, encoding="utf-8")
    verdict = mod.classify(planted)
    assert verdict["claims"] is True
    assert verdict["reaches_consumer"] is True
    assert verdict["value_changed"] is True, (
        "the discriminator did not detect that supplying velocity changes the scored value on a "
        "fixture reconstructed from the known 4.76.0 defect -- the instrument is broken, and any "
        "'no fixtures affected' conclusion from it is worthless"
    )
```

- [ ] **Step 2: Run it, see it fail (no module yet)**

Run: `python -m pytest tests/scripts/test_audit_velocity_fixtures.py -v`
Expected: FAIL, `ModuleNotFoundError: scripts.audit_velocity_fixtures`.

- [ ] **Step 3: Implement the discriminator**

Requirements it must satisfy, each stated because omitting it produces a false all-clear:

1. **Report A/B/C counts** — files referencing `speed_source`; files claiming `native`/`derived`;
   claiming-with-no-`vx`. The spec deliberately pins no numbers because they drift; the script is
   the source of truth.
2. **Supply a PERTURBING velocity in step 2, never `vx = vy = 0`.** Zero velocity produces no change
   in many consumers even where imputation matters, making "no change" indistinguishable from "the
   probe did nothing" — and this repo already names a `vx=vy=0` fixture as a defect, not a
   convenience.
3. **Record the deltas beside each verdict**, mirroring `applicability_deltas` in
   `tests/sb360/_registry.py` (*"Recorded so a zero-movement classification is VISIBLE"*).
4. **`--out` writes a JSON report**; no `--out`, print to stdout. It reads only committed test
   sources and runs no corpus pass, so it does **not** need `require_clean_tree` — say so in the
   module docstring, mirroring `render_sb360_matrix.py`'s exemption reasoning.

- [ ] **Step 4: Run the positive control until green**

Run: `python -m pytest tests/scripts/test_audit_velocity_fixtures.py -v`
Expected: PASS.

- [ ] **Step 5: Run the discriminator over the whole test tree and record the report**

```bash
python scripts/audit_velocity_fixtures.py --out <scratchpad>/velocity_audit.json
```
Paste the A/B/C counts and the `value_changed` list into the task notes. **This list is Task 8's
scope** — not the grep's 24.

- [ ] **Step 6: Commit**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
git add scripts/audit_velocity_fixtures.py tests/scripts/test_audit_velocity_fixtures.py
git commit -m "test(scripts): add the velocity-fixture discriminator with a positive control"
```

---

## Task 8: Fix only the fixtures the discriminator convicted

**Files:**
- Modify: whichever fixtures Task 7 Step 5 listed under `value_changed` (unknown until then — that
  is the point)

**Interfaces:**
- Consumes: Task 7's report.
- Produces: nothing consumed downstream.

- [ ] **Step 1: For each convicted fixture, write the failing test first**

Each fix ships with a test that fails on the PRE-fix fixture — the same rule Task 7 applies to the
instrument. Supply real `vx`/`vy` matching the declared `speed_source`, or change the declaration to
`unavailable` if the fixture genuinely has no kinematics (`TRACKING_CATEGORICAL_DOMAINS["speed_source"]`
has three tokens, and `unavailable` is a builder DECLARING that its source has no temporal history —
deliberately distinct from a NULL `speed_source`).

- [ ] **Step 2: Fix, run, and confirm each test now passes**

Run the affected test files individually first, then `python -m pytest tests/ -m "not e2e" -q`.

- [ ] **Step 3: Record the ones you did NOT fix**

A fixture that claims velocity, reaches a consumer, and whose value does **not** change is left
alone — and that outcome goes in the task notes and Task 9's `TODO.md` entry. "We checked and it
didn't matter" is a finding, not an absence.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: supply real kinematics to the fixtures the discriminator convicted"
```

---

## Task 9: Documentation, corrections, and the single commit

**Files:**
- Modify: `TODO.md`, `CHANGELOG.md`, `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-09-…-design.md` (only if Task 1 said NO-GO)

- [ ] **Step 1: Correct the false `TODO.md` pandas row**

Replace the claim that CI has no pandas-3 environment with the measured table from spec §2, and note
that the guard now declares the span.

- [ ] **Step 2: Add the new TODO rows**

- The `schema.py` `int64` boundary defect (spec §7b): `player_id`/`team_id` declared `int64` while
  the ball row is NA in both, two adapter variants already override them to `object`, and one
  producer cannot satisfy it at all. `Int64` is the durable fix; it moves goldens, so it is its own
  cycle. Note it is the reason Task 4 exists.
- Any Task 4 divergence (with the xfail reference), any Task 8 unfixed-but-checked fixtures, and —
  if Task 1 said NO-GO — the unclosed hazard the aggregation job would have caught.

- [ ] **Step 3: Update `CHANGELOG.md` and `CLAUDE.md`**

CHANGELOG: a new section for this cycle. **Do not claim a version number** — use a placeholder
heading and take the number at commit-prep.

CLAUDE.md: add the durable contract only if one emerged. **The ADR decision itself was made in
Task 3b** — this step only RECORDS its outcome. Do not restate per-release narrative here.

- [ ] **Step 4: If Task 1 said NO-GO, fix the spec's dangling cross-references**

Spec §5's gate list and §7's ADR criterion both assume the aggregation job exists. Edit both. This
cycle has already rotted one cross-reference; do not add another.

- [ ] **Step 5: Full verification**

```bash
python -m pytest tests/ -m "not e2e" -q     # ~19 min; run in background
python -m pyright
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
```
All must be clean. Record the pass count.

- [ ] **Step 6: Squash to ONE commit**

```bash
git reset --soft $(git merge-base HEAD origin/main)
git status --short          # review every file before writing the message
```
Write the message to a FILE and use `git commit -F` — never `-m` with backticks or apostrophes; the
shell ate a fragment of a commit message doing exactly that during 4.77.1.

- [ ] **Step 7: STOP**

Do not commit-and-push. Present the diff, the verification results, and the two decisions from Tasks
1 and 6 to the owner, and wait for explicit approval.

---

## Self-Review Notes

**Spec coverage.** §3 Item 1 → Tasks 1–3; Item 2 → Task 4; Item 3 → Tasks 5–6; Item 4 → Tasks 7–8;
§4 (out of scope) → untouched by every task; §5 non-vacuity → Tasks 2.3, 3.3, 4.4, 6.6, 7.1; §6
conflict re-check → the command lives in the spec and is re-run before merging, not here; §7 ADR
decision → **Task 3b** (NOT Task 9 — §7 explicitly forbids deferring it to commit-prep); §7b →
Task 9 Step 2.

**Known gap, stated rather than hidden:** Task 8's file list cannot be enumerated in advance — it is
produced by Task 7. This is deliberate (the spec forbids treating the grep's 24 as a fix-list), but
it means Task 8 is the one task whose size is unknown until execution. If the convicted list is
large, stop and re-scope with the owner rather than fixing 24 fixtures under momentum.

**Type consistency.** `resolved_legs(matrix) -> list[dict]` (Task 2) is internal and unused
elsewhere. `classify(path) -> dict` (Task 7) is consumed by Task 8 by name. `columns_exercised_on_no_roster() -> set[tuple[str, str]]`
(Task 6) is consumed only by its own test. `VISIBILITY_ROSTERS` gains `"gk_one_end"` in Task 5 and is
read by Task 6's helper — the one cross-task name dependency, and it is exact.
