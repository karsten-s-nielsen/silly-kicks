# TF-30 cover shadows — invariant repair + identity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Make TF-30's monotonicity invariant able to fail, settle both clamps on evidence, correct the
five glossary entries, and — behind a measured gate — stop discarding per-defender identity.

**Architecture:** Test repair + documentation correction + one additive aggregator-only column. **No API
change, no shipped-column value change, no retrain.** C4-free (count stays 32).

**Tech stack:** pytest, numpy, pandas. No new dependency.

**Spec:** `docs/superpowers/specs/2026-07-27-tf30-cover-shadow-invariant-and-identity-design.md` (rev 3,
two review rounds applied). Section references below are to that spec.

**Numbers:** version / PR-S / ADR are assigned at **commit-prep only** (Task 12). Do not write one
anywhere before then — three collisions in two days came from exactly that.

---

## File structure

| File | Change | Kind |
|---|---|---|
| `tests/invariants/test_cover_shadow_invariants.py` | fixture scope + immutability; repaired invariant; plant; repaired low-score test | Modify |
| `tests/tracking/test_cover_shadows.py` | second-clamp assertion; `fernandez_bornn` run; identity tests | Modify |
| `silly_kicks/tracking/_cover_shadows.py` | `_CS_COL_NAMES` split; tolerances; identity on both paths | Modify |
| `silly_kicks/tracking/features.py` | emit identity in `add_cover_shadows` only | Modify |
| `silly_kicks/feature_glossary.py` | correct 5 entries; add identity entry | Modify |
| `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md` | append RQ3 evidence | Modify |
| `scripts/measure_cover_shadow_argmax_agreement.py` | owner-run agreement measurement | Create |

**Ordering — and the split point.** Tasks 1–7 are the invariant/glossary work and stand alone. Tasks
8–11 are the identity column, which is **blocked on an owner-run measurement** (§5.1). **If that run
cannot be scheduled, drop Tasks 8–11 and ship 1–7** — spec O4. Keep that boundary clean: nothing in
1–7 may depend on anything in 8–11.

---

## Task 0: Branch and baseline

- [ ] **Step 1: Branch — with a NUMBER-FREE name**

```bash
git -C "D:/Development/karstenskyt__silly-kicks" fetch origin
git -C "D:/Development/karstenskyt__silly-kicks" status --short
git -C "D:/Development/karstenskyt__silly-kicks" rev-list --left-right --count HEAD...origin/main
git -C "D:/Development/karstenskyt__silly-kicks" checkout -b tf30-cover-shadow-invariant
```

> **Deliberately no `pr-sNN-` prefix.** The previous cycle branched as `pr-s133-…`, the number was taken
> mid-flight, and the branch had to be renamed or `git push` fails with
> `src refspec does not match any`. A number-free branch name cannot collide. The PR **title** carries
> the assigned number at Task 12.

Expect `0	0` from `rev-list`. No `git reset --hard` — verify sync, do not enforce it.

- [ ] **Step 2: Baseline**

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q --benchmark-skip 2>&1 | tail -3
.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py -q --durations=6 2>&1 | tail -10
```

Record both. The second is the §3.7 baseline — measured **4.51 s**, 5 setups (~0.45 s each warm,
2.35 s cold). Task 1 must not regress it.

---

## Task 1: Fixture scope + immutability

**Files:** Modify `tests/invariants/test_cover_shadow_invariants.py`

The fixture is function-scoped and re-runs for all five tests. Module-scoping saves ~1.8 s — more than
Task 2's new assertion costs. But a shared, non-copied DataFrame is a classic order-dependence flake
(spec §3.7 / MINOR 5), so the copy is not optional.

- [ ] **Step 1: Write the guard test FIRST — ONE test, no ordering assumption**

```python
from tests.tracking import _cover_shadow_inputs as _csi


def test_per_test_fixture_is_a_copy(cover_shadow_result):
    """Memoizing must not hand the SAME object to two tests.

    A shared, non-copied DataFrame is a classic order-dependence flake: a `.loc` assignment,
    an `inplace=True` sort, or an added column in one test silently changes every later one.
    This risk is INTRODUCED by memoizing, so it is guarded rather than assumed away.
    """
    assert cover_shadow_result is not _csi.cover_shadow_result()
```

> **An earlier draft used a MUTATOR/OBSERVER PAIR** — one test writing `__scratch__`, a second
> asserting its absence "in a test that runs after it alphabetically or by file order". **That guard was
> itself order-dependent** and would pass vacuously and forever under `pytest -k`, running the file's
> tests individually (what an implementer does while iterating), `-x` aborting early, `pytest-randomly`,
> or a rename of either test. A guard against silent order-dependence that is itself silently
> order-dependent is the exact vacuity class this cycle exists to remove — and the draft's own hedge
> ("alphabetically **or** by file order") was the tell that the ordering was not pinned. The identity
> check above needs no ordering at all. Keep a mutation pair as documentation if you like; it cannot be
> the guard.

- [ ] **Step 2: Run — verify RED for the stated reason**

Run: `.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py -q -k copy`
Expected **after Step 3's scope change but before adding `.copy()`**: `test_per_test_fixture_is_a_copy`
FAILS on the identity assertion (the two fixtures return the same object). See it fail once, then add
the copy.

- [ ] **Step 3: Implement — a SHARED HELPER MODULE, memoized, copied per test**

> ### ⚠ Why a helper module and not a fixture (BLOCKER, review 3)
>
> An earlier draft put these fixtures in `tests/invariants/test_cover_shadow_invariants.py` and had
> Task 5 and four of Task 9's tests — which live in **`tests/tracking/test_cover_shadows.py`** — request
> them. **A fixture defined in a test MODULE is visible only inside that module.** Verified: there is
> **no `tests/invariants/conftest.py`**, and `tests/tracking/test_cover_shadows.py` defines **zero**
> fixtures. Those tasks would have died at collection with `fixture 'cover_shadow_raw' not found`.
>
> **A shared `tests/conftest.py` fixture is NOT the fix**: at `scope="module"` it builds **once per
> consuming module** — two builds, not one — which silently invalidates Task 12's measured budget. It
> would also mean widening the root `fitted_xt` that `tests/vaep/`, `tests/tracking/` and
> `tests/invariants/` all consume, and two tests already call `fitted_xt.interpolator()`
> (`test_cover_shadows.py:640`, `test_gk_influence.py:514`).
>
> **The established pattern is a private helper module beside `_provider_inputs.py`** — verified
> importable across directories today (`tests/invariants/`, `tests/calibration/` and others already do
> `from tests.tracking._provider_inputs import …`; both `tests/` and `tests/tracking/` have
> `__init__.py`). Memoizing at MODULE level makes the build **session-wide — once, not once per
> module**, which is strictly better than either conftest option.

Create `tests/tracking/_cover_shadow_inputs.py`:

```python
"""Shared TF-30 cover-shadow test inputs (see the plan's Task 1).

Lives beside _provider_inputs.py because BOTH tests/invariants/ and tests/tracking/ consume it, and
a fixture defined in a test MODULE is invisible to a sibling directory. Cross-directory import of
this package is the established pattern here.

Builds are memoized at MODULE level, so the expensive chain runs ONCE PER SESSION -- not once per
consuming module, which is what a shared conftest fixture at scope="module" would have cost.

CALLERS MUST .copy() BEFORE MUTATING. The thin fixtures in each test file do that; the copy is what
Task 1's guard test pins.
"""

from __future__ import annotations

import functools

import numpy as np

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions


@functools.cache
def fitted_xt():
    """Module-local xT, byte-identical to the conftest fixtures.

    Deliberately NOT the root `tests/conftest.py:44` fixture: this module must not depend on the
    scope of a fixture shared with tests/vaep/ and tests/tracking/. It does NOT `.fit()` -- see the
    degenerate-xT trap in the Self-Review.
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@functools.cache
def prepared_frames_and_actions():
    """(frames, actions, home_team_id) -- the shared expensive chain. ~0.17 s, once per session."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from silly_kicks.tracking.utils import play_left_to_right

    frames = load_provider_frames("sportec")
    frames = smooth_frames(frames)
    frames = derive_velocities(frames)
    home_team_id = frames[~frames["team_id"].isna()]["team_id"].iloc[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return frames, synthesize_actions(frames), home_team_id


@functools.cache
def cover_shadow_result():
    """add_cover_shadows output. ~0.45 s, once per session. COPY BEFORE MUTATING."""
    from silly_kicks.tracking.features import add_cover_shadows

    frames, actions, home_team_id = prepared_frames_and_actions()
    return add_cover_shadows(actions, frames, fitted_xt(), home_team_id=home_team_id)
```

Then a thin fixture in **each** consuming test file:

```python
import pytest

from tests.tracking import _cover_shadow_inputs as _csi


@pytest.fixture
def cover_shadow_result():
    """Per-test COPY of the session-memoized build. The copy is the whole point."""
    return _csi.cover_shadow_result().copy()
```

Task 1's guard test then compares against the memoized object:

```python
def test_per_test_fixture_is_a_copy(cover_shadow_result):
    assert cover_shadow_result is not _csi.cover_shadow_result()
```

⚠ No `ScopeMismatch` is possible with this design — there is no fixture-scope graph to satisfy. That
whole class of problem disappears with the fixtures.

- [ ] **Step 4: Run, verify GREEN and FASTER**

```bash
.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py -q --durations=6
```
Expected: all pass; **one** expensive setup instead of five; total below the 4.51 s baseline.

---

## Task 2: The repaired invariant + the plant

**Files:** Modify `tests/invariants/test_cover_shadow_invariants.py`

The core of the cycle. §3.1 / §3.2.

- [ ] **Step 1: Add `cover_shadow_raw` to the SHARED HELPER (not to this test file)**

The existing build returns only the aggregator's clamped output. The repaired assertion needs the
unclamped pair, which means calling `compute_blocking_score` per action. Mirror the aggregator's own
per-action frame resolution (`features.py:3672-3695`) so the test scores the same frames.

⚠ **This goes in `tests/tracking/_cover_shadow_inputs.py`, alongside Task 1's builders** — Task 5
(in `tests/tracking/test_cover_shadows.py`) consumes it, and a fixture in the invariants module would
be invisible there. Each consuming file gets a thin `@pytest.fixture` wrapper returning
`_csi.cover_shadow_raw()`.

```python
@functools.cache
def cover_shadow_raw():
    """(threat_original, threat_unblocked) per action, UNCLAMPED. Memoized; treat as read-only.

    Mirrors add_cover_shadows' own action->frame resolution so the test scores exactly the
    frames production does.
    """
    import pandas as pd

    # NOTE the import path: link_actions_to_frames lives in tracking/utils.py and is re-exported
    # from silly_kicks.tracking. There is NO silly_kicks.tracking.linkage module -- an earlier
    # draft of this plan imported one and would have failed at collection.
    from silly_kicks.tracking import link_actions_to_frames
    from silly_kicks.tracking._cover_shadows import compute_blocking_score

    frames, actions, home_team_id = prepared_frames_and_actions()
    xt = fitted_xt()

    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    rows = []
    for _idx, row in actions.iterrows():
        aid, tid = row["action_id"], row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        try:
            frame_data = frame_groups.get_group((row["period_id"], int(float(fid_raw))))
        except KeyError:
            continue
        res = compute_blocking_score(frame_data, tid, xt, home_team_id=home_team_id)
        rows.append((aid, frame_data, tid, res))
    return {"rows": rows, "home_team_id": home_team_id, "xt": xt}
```

Thin wrapper in **each** consuming test file (`tests/invariants/test_cover_shadow_invariants.py` and
`tests/tracking/test_cover_shadows.py`):

```python
@pytest.fixture
def cover_shadow_raw():
    return _csi.cover_shadow_raw()   # read-only; do not mutate
```

- [ ] **Step 2: Write the repaired assertion + the plant**

**First, define the tolerance ONCE — in the library, in THIS task.**

Add to `silly_kicks/tracking/_cover_shadows.py` (Task 2, **not** Task 8):

```python
# "How negative may numerical integration make the raw threat difference." This is the floor below
# which the clamp at :907 is doing nothing but numerical hygiene, so it is a statement about this
# module's numerics and belongs here rather than in a test file. Calibrated against measured threat
# differences of order +3.79.
TOL_INVARIANT = 1e-9
```

> ⚠ **It must live in Tasks 1–7, and there must be exactly one of it.** An earlier draft defined it
> twice — once in `tests/invariants/…` (Task 2) and once in `_cover_shadows.py` (**Task 8**) — while
> `tests/tracking/test_cover_shadows.py` (Tasks 3 and 5) *used* it with no import. Two consequences,
> either fatal: Task 3 would `NameError` on first run, and Tasks 3/5 would depend on Task 8 across
> **the split point**, so dropping Tasks 8–11 per O4 — the documented expected outcome — would ship
> tests referencing an undefined name. Every test module that uses it **imports it explicitly**:
> `from silly_kicks.tracking._cover_shadows import TOL_INVARIANT`.

Then in the test file:

```python
from silly_kicks.tracking._cover_shadows import TOL_INVARIANT

# The plant must clear the tolerance by a real margin, not merely dip below zero -- otherwise a
# shrinking fixture could leave it "passing" on float noise.
_PLANT_MARGIN = 1e-6


def test_blocking_score_monotone_on_RAW_fields(cover_shadow_raw):
    """Removing defenders cannot decrease threat -- asserted on the UNCLAMPED difference.

    The shipped `blocking_score` column is clamped at _cover_shadows.py:907, so the previous
    version of this test (`assert blocking_score >= -1e-9`) was green by construction and had
    never checked the property it named.
    """
    assert cover_shadow_raw["rows"], "fixture produced no scoreable actions"
    for aid, _frame, _tid, res in cover_shadow_raw["rows"]:
        assert res.threat_unblocked - res.threat_original >= -TOL_INVARIANT, aid


def test_a_negative_difference_is_reachable(cover_shadow_raw):
    """THE WITNESS -- a permanent canary, NOT a substitute for seeing the guard fail (Step 4).

    Renamed from "..._assertion_can_actually_fail", which overclaimed: this test calls
    compute_blocking_score with DIFFERENT arguments (defenders_to_remove=[attacker_id]) and a
    DIFFERENT assertion than the guard, so it demonstrates a negative difference is REACHABLE.
    It does not demonstrate that test_blocking_score_monotone_on_RAW_fields fails when the
    production path breaks -- and it exercises the explicit defenders_to_remove branch
    (:885-886) while the real test's rows come from auto-identification (:866-884), so a
    regression in _classify_man_markers lives in a branch this never touches.

    Direction AND target both matter (spec 3.2). Dropping a NON-dangerous attacker can leave
    the counted cell set nearly unchanged -- its cells reassign to neighbours, some of them
    dangerous, so cells can even be ADDED -- and the plant silently becomes a no-op. Dropping a
    DANGEROUS attacker that owns non-zero per_receiver threat is guaranteed negative: no cell is
    ever added (the vacated region redistributes only among existing generators) so the counted
    set weakly shrinks, AND attacking pitch control weakly falls.
    """
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.tracking._cover_shadows import compute_blocking_score

    xt, home_team_id = cover_shadow_raw["xt"], cover_shadow_raw["home_team_id"]

    planted = 0
    for _aid, frame_data, tid, _res in cover_shadow_raw["rows"]:
        # Identify a DANGEROUS attacker owning non-zero threat, via the real internal.
        # `attacking_team_id` is POSITIONAL (pitch_control/_dispatch.py:31-33) -- there is no
        # `team=` keyword; an earlier draft of this plan passed one and would have TypeError'd.
        surface = cs.compute_pitch_control(frame_data, tid, method="spearman")
        _total, per_receiver = cs._voronoi_threat(
            surface, xt, frame_data, attacking_team_id=tid, home_team_id=home_team_id
        )
        targets = [pid for pid, thr in per_receiver.items() if thr > 0.0]
        if not targets:
            continue

        # The plant: remove an ATTACKER through the defender-removal seam.
        res = compute_blocking_score(
            frame_data, tid, xt, home_team_id=home_team_id, defenders_to_remove=[targets[0]]
        )
        assert res.threat_unblocked - res.threat_original < -_PLANT_MARGIN, (
            "plant did not go negative -- it has degenerated into a no-op and proves nothing"
        )
        planted += 1

    assert planted > 0, (
        "no fixture action offered a dangerous attacker with non-zero per_receiver threat. "
        "That is a FIXTURE INADEQUACY TO FIX, not a plant to weaken or skip."
    )
```

- [ ] **Step 3: Run — both must behave**

```bash
.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py \
  -k "monotone_on_RAW_fields or negative_difference_is_reachable" --collect-only -q | tail -1
.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py -q \
  -k "monotone_on_RAW_fields or negative_difference_is_reachable"
```

**Eyeball `2 tests collected` from the first command.** `-k` cannot error on a typo — it silently
deselects — so a mistyped term reports a green run of fewer tests than intended. The `--collect-only`
line costs nothing and makes that whole defect class unrepeatable rather than re-caught each round.

⚠ **Use the post-rename names.** An earlier draft still said `-k "RAW or can_actually_fail"` after
Step 2 renamed the witness. `-k` does **not** error on a term that matches nothing, so that expression
silently ran **one** test and reported success — the implementer would see green believing both ran.

Expected: **both pass.** That means the invariant holds on the fixture **and** a negative difference is
*reachable*. It does **not** yet mean the guard would catch a regression — only Step 4 establishes that,
via the branch the witness never touches. If the witness fails with "degenerated into a no-op", the
target selection is wrong — fix it, do not loosen `_PLANT_MARGIN`.

- [ ] **Step 4: OBSERVE THE REAL GUARD GO RED — one-off, manual, recorded**

The witness is not this. The repo rule is that a repaired guard must be *seen* to fail, and only this
step does that for `test_blocking_score_monotone_on_RAW_fields` itself.

1. In `cover_shadow_raw`, **temporarily** point the `compute_blocking_score` call at attacker ids —
   reuse the witness's own selection rather than hunting for one by hand, so the single manual step in
   this plan is mechanical:

   ```python
   surface = cs.compute_pitch_control(frame_data, tid, method="spearman")
   _t, per_receiver = cs._voronoi_threat(
       surface, xt, frame_data, attacking_team_id=tid, home_team_id=home_team_id
   )
   targets = [pid for pid, thr in per_receiver.items() if thr > 0.0]
   res = compute_blocking_score(
       frame_data, tid, xt, home_team_id=home_team_id, defenders_to_remove=[targets[0]]
   )
   ```
2. Run `.venv/Scripts/python.exe -m pytest tests/invariants/test_cover_shadow_invariants.py -q -k RAW`
3. **Observe it RED.** Paste the failure line into the commit message.
4. **Revert the edit.** Re-run; green.

If it does *not* go red, the guard is still not measuring what it claims and the task is not done.

- [ ] **Step 5: Delete the vacuous original**

Remove `test_blocking_score_non_negative` (`:31-34`). Superseded — keeping both would leave a
green-by-construction test beside the real one.

- [ ] **Step 6: Document what the guard does and does NOT cover**

Measured: min raw difference **+3.792** against `TOL_INVARIANT = 1e-9` — **nine orders of headroom**
across 9 actions. Say so in the docstring: this catches **gross** breakage (sign flip, wrong team,
degenerate grid) and will **not** catch a subtle monotonicity violation that only appears in geometry
the fixture does not contain. That is a fine place to be and far better than a tautology — but the next
reader must not over-trust it. Same honesty §3.5 asks of the clamp.

- [ ] **Step 7: Commit**

```bash
git add tests/invariants/test_cover_shadow_invariants.py
git commit -m "test(tracking): make the TF-30 monotonicity invariant able to fail

Observed RED before the fix: <paste the failure line from Step 4>."
```

---

## Task 3: The second clamp (§1.2 / §3.3)

**Files:** Modify `tests/tracking/test_cover_shadows.py`

- [ ] **Step 1: Write the test — through the REAL internals**

```python
def test_per_blocker_delta_is_non_negative_before_the_clamp():
    """The SECOND clamp (_cover_shadows.py:1109) is measured, not assumed.

    Asserted through the real `_lane_int_probs` + `_lane_received_batched` (signature verified:
    returns (p_blocked_full, p_received_full, p_received_loo)). Recomputing new_recv/old_recv in
    the test would assert the TEST's arithmetic, not the shipped code's -- a vacuous fixture.

    Summed over the three lanes, NOT per lane: _cover_shadows.py:1101-1109 accumulates
    old_recv/new_recv across center/left/right and clamps the TOTAL, so a per-lane assertion
    tests a stronger property than the code relies on and could fail for non-defects.
    """
    # Imports go IN this block, after the docstring -- round 2's BLOCKER 2 was a NameError from a
    # constant used one task away from where it was introduced. (Imports placed BEFORE a docstring
    # demote it to a bare string expression; __doc__ becomes None.)
    import numpy as np

    from silly_kicks.tracking._cover_shadows import (
        TOL_RECEPTION,
        _lane_int_probs,
        _lane_received_batched,
    )

    ...  # build lane geometry from a fixture frame, then:
    old_recv = 0.0
    new_recv = np.zeros(n_lb)
    for lane in (center, left, right):
        p_int_def, p_int_att, _t_ball, p_ctrl = _lane_int_probs(
            lane, lb_pos, lb_vel, att_pos, att_vel, params=params
        )
        _pb, base_rec, loo_rec = _lane_received_batched(p_int_def, p_int_att, p_ctrl)
        old_recv += base_rec
        new_recv += loo_rec

    assert (new_recv - old_recv >= -TOL_RECEPTION).all()
```


**`TOL_RECEPTION` is a THIRD tolerance, not a reuse.** Add it beside the other two in
`_cover_shadows.py` (Task 2, so it stays in the 1–7 block):

```python
# "How negative may float error make a summed RECEPTION-PROBABILITY difference." Distinct from
# TOL_INVARIANT, which is calibrated against threat differences of order +3.79; new_recv/old_recv
# are probabilities summed over three lanes, so they are O(1). Reusing one constant across two
# scales is exactly what the TOL_INVARIANT / TOL_ATTRIB split exists to prevent -- the same
# reasoning, applied one level down.
TOL_RECEPTION = 1e-12
```

Build the lane geometry by mirroring `_cover_shadows.py:1086-1097` exactly.

- [ ] **Step 2: Run** — `.venv/Scripts/python.exe -m pytest tests/tracking/test_cover_shadows.py -q -k "before_the_clamp"`

- [ ] **Step 3: Record the result** in the test docstring — *"measured non-negative on N (lane, blocker)
pairs"* — per §3.5. A measured no-op reads differently from assumed hygiene.

- [ ] **Step 4: Commit**

---

## Task 4: Repair the misnamed test (§3.4)

**Files:** Modify `tests/invariants/test_cover_shadow_invariants.py`

`test_zero_blocked_implies_low_score` promises "low score" and asserts only non-negative.

⚠ **The existing test is a METHOD of `TestCoverShadowInvariants` (`:55-66`), not a module-level
function.** Replace it **in place** inside the class, or delete the method and add the module-level
function — but do **one** of those explicitly. An earlier draft showed a module-level function with no
deletion step: implemented literally, both would exist, pytest would collect both, and the vacuous
class method would stay green **in the task written to remove it**. Task 2 Step 5 deletes its
counterpart explicitly; this task needs the same. The same applies to Task 2's new module-level tests
sitting beside that class.

- [ ] **Step 1: Make it assert what its name claims (and DELETE the old one)**

```python
def test_zero_blocked_implies_low_score(cover_shadow_result):
    """When n_blocked_receivers == 0, blocking_score is small RELATIVE to the blocked population.

    The previous version asserted only non-negative and said so in its own comment -- the name
    promised a property nothing checked.
    """
    df = cover_shadow_result
    zero = df[df["n_blocked_receivers"] == 0]["blocking_score"].dropna()
    nonzero = df[df["n_blocked_receivers"] > 0]["blocking_score"].dropna()
    # NO pytest.skip here. An earlier draft skipped when either population was empty -- but at
    # 9-10 actions that is a live possibility, and a skip turns this test straight back into the
    # vacuity it was written to remove. Task 2's plant takes the opposite line ("a FIXTURE
    # INADEQUACY TO FIX, not a plant to weaken or skip") and the same standard applies here.
    assert len(zero) > 0 and len(nonzero) > 0, (
        "fixture lacks both populations -- FIX THE FIXTURE, do not skip"
    )
    # Directional evidence, NOT a calibrated check: these may be ~3 and ~6 rows. Passing says
    # little; failing is a real finding (see below).
    assert zero.mean() < nonzero.mean()
```

> If this FAILS, that is a **finding**, not a reason to rename the test back. Report it; it would mean
> the lane-level classification and the Voronoi integral disagree more than the design assumes.

- [ ] **Step 2: Run and report** — if red, STOP and report rather than weakening.

- [ ] **Step 3: Commit**

---

## Task 5: `fernandez_bornn` — run it (§3.6)

**Files:** Modify `tests/tracking/test_cover_shadows.py`

D1 rests on §1.3, which is argued only for `spearman`/`voronoi`. Do **not** attempt the proof — for a
Gaussian-influence-field model with logistic normalisation that is a research task. The empirical check
falsifies cheaply.

- [ ] **Step 1: Write the per-method run**

```python
@pytest.mark.parametrize("method", ["spearman", "voronoi", "fernandez_bornn"])
def test_raw_difference_non_negative_per_method(method, cover_shadow_raw):
    """D1's scope, settled empirically for fernandez_bornn rather than by extending the proof.

    The repaired invariant test only exercises whichever method its fixture uses, so this needs
    its own explicit per-method run.

    Iterates EVERY action cover_shadow_raw resolved, so the N recorded in Step 2 is real. An
    earlier draft scored a single frame_data -- N would have been 1, and D1 (keep both clamps)
    would have rested, for a supported method, on one frame.
    """
    from silly_kicks.tracking._cover_shadows import TOL_INVARIANT, compute_blocking_score

    rows = cover_shadow_raw["rows"]
    # NON-EMPTINESS FIRST. Everything below is vacuous without it, and empty is a LIVE mode:
    # cover_shadow_raw skips actions with no team_id / no pointer / no frame group (mirroring
    # features.py:3679-3692, which is why it resolves 9 of 10 today). A fixture or linkage change
    # empties it silently -- three parametrized tests then go green having measured NOTHING, and
    # Step 2 records "verified empirically on N actions" with N = 0. D1 (keep both clamps) would
    # rest on zero measurements, in the task that produces D1's only evidence.
    assert rows, "fixture produced no scoreable actions -- FIX THE FIXTURE, do not skip"

    xt, home_team_id = cover_shadow_raw["xt"], cover_shadow_raw["home_team_id"]
    n = 0
    for aid, frame_data, tid, _res in rows:
        res = compute_blocking_score(
            frame_data, tid, xt, home_team_id=home_team_id, method=method
        )
        assert res.threat_unblocked - res.threat_original >= -TOL_INVARIANT, (method, aid)
        n += 1
    # The N that reaches the Step 2 docstring must be the N this test actually measured.
    assert n == len(rows) >= 9, f"scored {n} of {len(rows)} rows"
```

> **PRE-VERIFIED while writing this plan — all three methods HOLD.** Run against the sportec fixture,
> 9 actions each:
>
> | method | n | min raw difference |
> |---|---|---|
> | `spearman` | 9 | **+3.79** |
> | `voronoi` | 9 | **+47.16** |
> | `fernandez_bornn` | 9 | **+29.43** |
>
> So D1's scope question is settled: the "fails → D1 changes, ADR required" branch is **not** expected
> to fire. Still implement and commit the test — a pre-verification by the plan author is not a
> committed regression guard, and the recorded N must come from the shipped test.

- [ ] **Step 2: Run and record the OUTCOME, either way**

- **Holds** → record in the glossary + `_cover_shadows` docstring: *"verified empirically on N actions
  for `fernandez_bornn`; argued structurally for `spearman`/`voronoi`"* — an honest **mixed-provenance**
  statement.
- **Fails** → a genuine finding. **D1 changes**: the clamp masks real negatives on a supported
  configuration. STOP and report; do not proceed to Task 6's glossary wording.

⚠ **No API change** (D6). Whatever the result, `compute_blocking_score` keeps accepting all three
methods — rejecting or warning on one would be an API break strictly worse than the column changes D2
forbids.

- [ ] **Step 3: Commit**

---

## Task 6: Glossary correction (§4)

**Files:** Modify `silly_kicks/feature_glossary.py`

**FIVE** entries, not four: `:1027`, `:1034`, `:1041`, `:1048`, and **`n_potential_receivers` at
`:1055`** — outside the range an earlier draft cited, which is how it was miscounted.

- [ ] **Step 1: Amend the three SCORING entries only**

`blocking_score`, `max_single_defender_blocking_score`, and `blocked_threat_fraction` gain: (a)
non-negativity is **by construction**, with Task 5's provenance split; (b) ours **cannot** express
"this defender's positioning made things worse", unlike the paper's SoccerMap-CNN counterfactual.

Leave the two **counts** alone — non-negativity there is trivial and the caveat would be noise.
Leave `higher_is_better=None` on all five (direction flips by perspective); that is decided, not
forgotten.

- [ ] **Step 2: Run the glossary gates**

```bash
.venv/Scripts/python.exe -m pytest tests/test_feature_glossary_coverage.py -q
```

- [ ] **Step 3: Commit**

---

## Task 7: Append RQ3 evidence to the design doc (§7)

**Files:** Modify `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md`

RQ3 stays scoped out; this **strengthens** the existing reason with new evidence — the headline
"789 of 822" is circular (defenders are optimised *into* the Cone Corridor, then success is scored as
"is a defender in the cone", using their **worst** model at 7.3% recall; the real result is threat
reduced in 75% of 63,037 snapshots, i.e. worse a quarter of the time, sign only, no magnitude/CI/placebo).

**No new ADR** — an ADR records a change of decision, not agreement with a recorded one.

- [ ] **Step 1: Append.** - [ ] **Step 2: Commit.**

> ### ⛳ SPLIT POINT
> Tasks 0–7 stand alone and are shippable. Tasks 8–11 are the identity column, **blocked on an
> owner-run measurement**. If that cannot be scheduled this cycle, **stop here and go to Task 12** —
> spec O4.

---

## Task 8: The identity column — implementation

**Files:** Modify `silly_kicks/tracking/_cover_shadows.py`, `silly_kicks/tracking/features.py`

- [ ] **Step 1: Add the attribution tolerance (§5.3)**

`TOL_INVARIANT` was already added in **Task 2** (it must not live here — see that task's note on the
split point). This task adds only:

```python
# "How small is NOT an attribution." A different question from TOL_INVARIANT -- they must not
# silently share a constant: a strict `<= 0` would still name a defender when max_def is 1e-14.
#
# PROVISIONAL -- set from Task 11's measured max_def distribution, NOT reasoned into place.
# Two things are currently unknown: (a) the accumulation noise floor (score_per_blocker sums
# recv_xt * delta across receivers and three lanes, and measured threat differences are O(+3.79),
# so rounding over a 50x32 grid sum can plausibly reach 1e-13 or higher -- 1e-12 may sit barely
# above the floor or below it); (b) whether a defender genuinely screening a low-xT lane produces a
# legitimately tiny max_def that this would NA out. Setting it by assertion is the same mistake the
# 0.9 agreement threshold was corrected for in spec review 2.
TOL_ATTRIB = 1e-12  # PROVISIONAL until Task 11 reports the distribution
```

- [ ] **Step 2: Split `_CS_COL_NAMES` (D3)**

```python
# Numeric columns the VAEP factory consumes (features.py reads this one).
_CS_COL_NAMES = [
    "n_blocked_receivers", "n_potential_receivers", "blocking_score",
    "blocked_threat_fraction", "max_single_defender_blocking_score",
]
# Aggregator-only additions. NEVER appended to _CS_COL_NAMES: features.py:3784 feeds that list
# straight into cover_shadow_xfns, and a player-id column would put a non-numeric value into VAEP
# feature matrices. Same split as `das_source` (ADR-043).
_CS_AGGREGATOR_ONLY_COLS = ["max_single_defender_player_id"]
```

- [ ] **Step 3: Emit identity on ALL THREE RETURN SITES, with the §5.2 NA rule**

⚠ **`_compute_cover_shadow_dict` returns the 5-key dict from THREE places, not one** — verified:

| Line | Branch | Identity |
|---|---|---|
| `:975-981` | `n_potential == 0` | `None` |
| `:1019-1026` | no lane blockers | `None` |
| `:1113-1119` | main path (exact / cheap) | per Step 3 below |

`add_cover_shadows` reads `cs["max_single_defender_player_id"]` **by key per action**
(`features.py:3710+`), so missing the two early returns is a **`KeyError` on the first action that
hits one** — not a silent gap. Spec §5.2 names both sites; an earlier draft of this plan lost them in
translation and edited only the main path.

Main-path emission:

Exact path (`:1040-1052`) — sentinel is `None`, never index 0, assigned only on strict improvement:

```python
    if detailed:
        max_def = 0.0
        max_def_pid = None
        for d_pid in lane_blocker_ids:
            d_result = compute_blocking_score(...)
            if d_result.blocking_score > max_def:
                max_def = d_result.blocking_score
                max_def_pid = d_pid
```

Cheap path (`:1112`):

```python
        max_def = float(score_per_blocker.max()) if n_lb > 0 else 0.0
        max_def_pid = (
            lane_blocker_ids[int(score_per_blocker.argmax())]
            if n_lb > 0 and max_def > TOL_ATTRIB
            else None
        )
```

> **Why the `max_def > TOL_ATTRIB` guard is load-bearing:** `score_per_blocker = np.zeros(n_lb)`
> accumulates only clamped non-negative deltas, so when no lane is meaningfully affected **every entry
> stays 0.0** and `argmax()` returns **index 0** — naming a defender who did nothing. Verified:
> `np.zeros(4).argmax() == 0`.

- [ ] **Step 4: Emit in `add_cover_shadows` only** (`features.py`, beside the existing `out[...]`
assignments), with

```python
id_compat.restore_id_dtype(values, frames["player_id"].dtype)
```

⚠ **`frames`, not `actions`.** The values come from `frames` — `lane_blocker_ids` is a list-comp over
`defenders_outfield["player_id"]` (`_cover_shadows.py:1017`). Both repo precedents restore the
**frames** dtype and one names it a shared rule: `_ball_carrier.py:171→:331`,
`_gk_resolve.py:131→:191`, and `features.py:310-311` — *"Restore the frames dtype (shared rule — see
restore_id_dtype)."* An earlier draft used `actions["player_id"].dtype`, which Task 9's dtype test
breaks on by construction: it casts `f2["player_id"]` to string and leaves `a2["player_id"]` numeric,
so string ids would be fed into a numeric target — a raise or a silently all-NaN column, inside the
test written to prove ADR-019 compliance.

**Do not** touch the `cover_shadow_xfns` path.

- [ ] **Step 5: Run** — `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "cover_shadow" -q`

---

## Task 9: Identity gates

**Files:** Modify `tests/tracking/test_cover_shadows.py`

> ⚠ **These are written out in full, deliberately.** An earlier draft of this plan shipped all five as
> docstring-only / `...` bodies. **Verified: pytest reports those PASSED** (a body that is only a
> docstring returns `None`, asserts nothing). A plan whose whole purpose is deleting
> green-by-construction tests would have delivered five of them — and its own placeholder scan missed
> it by grepping for the `...` token while four of the five were docstring-only. Same
> heuristic-vs-enumeration failure ADR-043 records. **If any assertion below is hard to write, that is
> information about the design — do not leave the body empty.**

- [ ] **Step 1: Write all five, WITH assertions**

```python
def test_identity_is_na_wherever_there_is_no_attribution(cover_shadow_result):
    """NA wherever max_def <= TOL_ATTRIB -- INCLUDING n_lb > 0 rows whose max is 0.

    An earlier draft of the spec said NA "exactly where n_lb == 0", which would have asserted NA
    is ABSENT everywhere else -- i.e. required the fabricated player-id and failed if anyone later
    fixed it. The gate would have enforced the defect.
    """
    from silly_kicks.tracking._cover_shadows import TOL_ATTRIB

    df = cover_shadow_result
    no_attrib = df["max_single_defender_blocking_score"].fillna(0.0) <= TOL_ATTRIB
    has_attrib = ~no_attrib & df["max_single_defender_blocking_score"].notna()
    assert df.loc[no_attrib, "max_single_defender_player_id"].isna().all()
    # Both directions: NA must not be the answer everywhere, or the column is useless.
    assert has_attrib.any(), "fixture has no attributable action -- FIX THE FIXTURE"
    assert df.loc[has_attrib, "max_single_defender_player_id"].notna().all()


def test_identity_is_the_argmax_over_all_lane_blockers():
    """Under detailed=True the identity is EXACTLY checkable: max_def is
    `max(compute_blocking_score(remove=[d]).blocking_score for d in lane_blocker_ids)`
    (_cover_shadows.py:1040-1052), so max_def_pid is that argmax by construction.

    An earlier draft asserted only `compute_blocking_score(remove=[named]).blocking_score > 0.0`.
    Three defects, all of which this replaces: (1) it never compared the named defender to ANY
    other, so a column naming the second-best passed -- exactly what D4/section 5.1 exist to
    prevent; (2) it mixed paths, taking the identity from the CHEAP score_per_blocker
    (detailed=False) and the value from the EXACT PC counterfactual -- two different quantities,
    while the docstring claimed "the same array"; (3) `> 0.0` is a one-sided check on the CLAMPED
    :907 value -- the very shape section 3.1 exists to repair, reintroduced inside a guard for the
    new column.

    The cheap path's identity is covered by the section 5.1 owner measurement (Task 11). That
    division is the right one; the earlier draft blurred it.
    """
    import silly_kicks.tracking._cover_shadows as cs
    from tests.tracking import _cover_shadow_inputs as _csi

    frames, actions, home_team_id = _csi.prepared_frames_and_actions()
    xt = _csi.fitted_xt()
    checked = 0
    for frame_data, passer_xy, tid in _iter_scoreable(frames, actions):
        d = cs._compute_cover_shadow_dict(
            frame_data, passer_xy, tid, xt, home_team_id=home_team_id, detailed=True
        )
        if d is None or d["max_single_defender_blocking_score"] <= cs.TOL_ATTRIB:
            continue
        pid = d["max_single_defender_player_id"]
        blockers = _lane_blocker_ids(frame_data, tid, home_team_id)
        assert len(blockers) >= 2, "single-blocker action cannot discriminate an argmax"
        scores = {
            b: cs.compute_blocking_score(
                frame_data, tid, xt, home_team_id=home_team_id, defenders_to_remove=[b]
            ).blocking_score
            for b in blockers
        }
        assert scores[pid] == d["max_single_defender_blocking_score"]
        assert all(scores[pid] >= s for s in scores.values())
        checked += 1
    assert checked > 0, (
        "no action with >=2 lane blockers and a non-zero max -- FIXTURE INADEQUACY TO FIX"
    )


def test_identity_survives_numeric_and_string_player_ids():
    """ADR-019: source-dtype passthrough, and the SAME player named either way."""
    from silly_kicks.tracking.features import add_cover_shadows
    from tests.tracking import _cover_shadow_inputs as _csi

    frames, actions, home_team_id = _csi.prepared_frames_and_actions()
    fitted_xt = _csi.fitted_xt()
    num = add_cover_shadows(actions, frames, fitted_xt, home_team_id=home_team_id)

    f2, a2 = frames.copy(), actions.copy()
    f2["player_id"] = f2["player_id"].astype("string")
    f2["team_id"] = f2["team_id"].astype("string")
    a2["team_id"] = a2["team_id"].astype("string")
    strv = add_cover_shadows(a2, f2, fitted_xt, home_team_id=str(home_team_id))

    left = num["max_single_defender_player_id"].dropna().astype(str).tolist()
    right = strv["max_single_defender_player_id"].dropna().astype(str).tolist()
    assert left == right, "the identity changed under a pure dtype change"
    assert len(left) > 0, "no attributable action -- FIX THE FIXTURE"


def test_cover_shadow_xfns_do_not_leak_the_identity_column():
    """VAEP feature matrices stay numeric: the identity is an aggregator column only.

    Mirrors tests/tracking/test_das.py:1933 (the das_source precedent, ADR-043).
    """
    from silly_kicks.tracking.features import cover_shadow_xfns
    from silly_kicks.vaep.features import feature_column_names
    from tests.tracking import _cover_shadow_inputs as _csi

    _frames, _actions, home_team_id = _csi.prepared_frames_and_actions()
    xfns = cover_shadow_xfns(_csi.fitted_xt(), home_team_id=home_team_id)
    names = feature_column_names(xfns, nb_prev_actions=3)
    assert not [n for n in names if "max_single_defender_player_id" in n]
    # Non-vacuity: the factory must actually be emitting its numeric columns.
    assert any("blocking_score" in n for n in names)


def test_identity_is_not_wired_to_a_constant():
    """Non-vacuity: must name a DIFFERENT player than lane_blocker_ids[0] on >= 1 action.

    If no fixture action produces a non-zero argmax at an index other than 0, that is a FIXTURE
    INADEQUACY TO FIX, not a test to skip.
    """
    import silly_kicks.tracking._cover_shadows as cs
    from tests.tracking import _cover_shadow_inputs as _csi

    frames, actions, home_team_id = _csi.prepared_frames_and_actions()
    xt = _csi.fitted_xt()
    off_zero = 0
    for frame_data, passer_xy, tid in _iter_scoreable(frames, actions):
        d = cs._compute_cover_shadow_dict(
            frame_data, passer_xy, tid, xt, home_team_id=home_team_id
        )
        if d is None or d["max_single_defender_player_id"] is None:
            continue
        blockers = _lane_blocker_ids(frame_data, tid, home_team_id)
        if blockers and d["max_single_defender_player_id"] != blockers[0]:
            off_zero += 1
    assert off_zero > 0, (
        "identity always names the first lane blocker -- either it is wired to index 0, or the "
        "fixture cannot discriminate. FIX THE FIXTURE; do not skip."
    )
```

Two small module-level helpers in the test file. `_iter_scoreable` mirrors `features.py:3672-3695`
(action→frame resolution). `_lane_blocker_ids` mirrors `_cover_shadows.py:1004-1017` **exactly** —
**EXECUTED and verified against the fixture, do not paraphrase it**:

```python
def _lane_blocker_ids(frame_data, attacking_team_id, home_team_id):
    """The candidate set the production path scores. Mirrors _cover_shadows.py:1004-1017."""
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.id_compat import ids_match, same_id

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id))
        & (~players["is_goalkeeper"].astype(bool))
    ]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    goal_x_own = 105.0 if same_id(attacking_team_id, home_team_id) else 0.0
    man_markers = cs._classify_man_markers(
        defenders_outfield, attackers, goal_x_own=goal_x_own, params=cs.CoverShadowParams()
    )
    return [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]
```

> ⚠ `_classify_man_markers(defenders, attackers, *, goal_x_own, params)` — **not**
> `(frame, team_id, home_team_id=...)`. A first draft of this probe guessed the latter and
> `TypeError`d immediately.

**MEASURED on the fixture before hand-off** — the assertions below are known reachable, not hoped:

| Quantity | Value |
|---|---|
| Actions with a non-zero `detailed=True` max | **9** |
| Lane blockers per action | **10** (so the `>= 2` requirement never trips) |
| `max_def` reproduces as `max(compute_blocking_score(remove=[b]))` | **9 / 9** |

- [ ] **Step 2: Run — and OBSERVE `test_identity_is_not_wired_to_a_constant` GO RED**

A constant-wired column is exactly what that test claims to catch, so it must be seen to catch one.
Temporarily hard-code the cheap path's identity to `lane_blocker_ids[0]`, run, **observe RED**, revert.
Paste the failure line into the commit message. Without this, that test carries the same
unproven-guard status the whole cycle exists to remove.

- [ ] **Step 3: Commit.**

---

## Task 10: Glossary + NOTICE for the new column

**Files:** Modify `silly_kicks/feature_glossary.py`

> **Split across Tasks 10 and 11 — the number does not exist yet.** An earlier draft had this entry
> land "with Task 11's number" while ordered *before* Task 11, which produces it. Step 1 lands the
> entry; Step 1b (after Task 11) appends the measurement.

- [ ] **Step 1: land the entry now, WITHOUT the number.** Add `max_single_defender_player_id` —
`unit` per the closed vocab, `emitting_module=_M_COVER_SHADOWS`, `attribution=_A_CASCIOLI`,
`higher_is_better=None`. Docstring states exactness under `detailed=True` and that the cheap path is
approximate. `NOTICE:511-523` already carries Cascioli + Spearman — the column attaches to the existing
block, no new citation. Glossary + NOTICE land in **this same commit** (4.59.0 rule).

- [ ] **Step 1b (AFTER Task 11): append the measured agreement number** to the docstring and glossary
entry. Until it exists the entry must not imply one.

- [ ] **Step 2: Run the coverage gate.** - [ ] **Step 3: Commit.**

---

## Task 11: The agreement measurement — OWNER-RUN, prerequisite (§5.1)

**Files:** Create `scripts/measure_cover_shadow_argmax_agreement.py`

> **BLOCKING.** The column does not merge until this number exists and is recorded. Shipping on the
> caveat alone is exactly the weaker standard D4 was written to reject.

- [ ] **Step 1: Write the script.** For each qualifying action (≥ 2 lane blockers and
`max_def > TOL_ATTRIB`), compute the identity on **both** paths and record:

1. agreement rate **with an interval**;
2. **the value gap at disagreements** — agreement rate alone is the wrong decision input, since a
   disagreement between near-tied defenders is harmless where the same rate with large gaps is serious;
3. **the full `max_def` distribution** — including the rows excluded by the `> TOL_ATTRIB` filter. The
   script already iterates every action, so this is free, and it is what sets `TOL_ATTRIB` (Task 8):
   look for the gap between the zero cluster and the smallest genuinely non-zero values. If no such gap
   exists, that is itself the finding — it means "no attribution" and "small attribution" are not
   separable and the NA rule needs rethinking rather than a tighter constant.

- [ ] **Step 2: Owner runs it** on provider / `@e2e` data (pining GS WC2022), target **n in the
hundreds**. ⚠ Needs `PINING_FOR_THE_DATA_TOKEN` from env — never hardcode.

- [ ] **Step 3: Apply the PRE-REGISTERED rule**

| Outcome | Action |
|---|---|
| agreement ≥ **0.9** at **n ≥ 100** | ship as specified; record the number + gap distribution |
| below either | do **not** ship silently — gate the column to `detailed=True` (NA on the cheap path) or drop it, as an explicit owner decision |
| run not schedulable | drop Tasks 8–11, ship 0–7 (spec O4) |

0.9 is a **stated engineering threshold, not derived** — "a consumer reading `..._player_id` assumes it
is usually right". Named before the number was seen; do not move it afterwards.

- [ ] **Step 4: Commit** the script + the recorded number.

---

## Task 12: Commit-prep — STOP for owner approval

- [ ] **Step 1: Full lint trio**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright
```
Re-run the WHOLE trio after any fix.

- [ ] **Step 2: Full suite — SAME marker set as the Task 0 baseline**

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q --benchmark-skip
```

⚠ Task 0 baselines with `-m "not e2e and not slow"`; an earlier draft compared against
`-m "not e2e"` here — **different populations, so the comparison was not valid**. Use the marker set
above for the comparison, then optionally run the full `-m "not e2e"` separately as an absolute check.

**Measured budget — both files, so a moved number is interpretable rather than mysterious:**

| Where | Addition | Measured |
|---|---|---|
| `tests/invariants/` | shared prep 0.17 + raw scoring pass 0.59 + witness 0.09 | **+0.85 s** |
| `tests/invariants/` | module-scoping saving | **−1.8 s** |
| **invariants net** | | **≈ −0.95 s (faster)** |
| `tests/tracking/` | Task 5: 3 methods × 9 actions ≈ 54 PC grids | **+0.87 s** |
| `tests/tracking/` | Tasks 3 + 9 | small; measure and record |

So the invariants file gets **faster** and the whole-suite number moves by roughly **+0.9 s** from the
`tests/tracking/` side. An earlier draft budgeted only the invariants file while the command below runs
the whole suite — Task 5 alone was uncounted at ~54 PC grids.

If the invariants file is **slower**, the shared `prepared_frames_and_actions` fixture is probably not
actually shared.

- [ ] **Step 3: Merge from main; assign the registers NOW, not before**

```bash
git fetch origin && git merge origin/main
```
Then read `origin/main`'s `pyproject.toml` **and** check
`../karstenskyt__silly-kicks_part-deux` — it had **4.65.0** committed locally at spec time. Assign
version / PR-S / ADR only at this point.

- [ ] **Step 4: Version bump ×5 + docs** — `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`,
`CHANGELOG.md`, `TODO.md` (retire the TF-30 (a) row). Update `CLAUDE.md`'s tracking bullet.
**No new ADR** (§7) unless Task 5 falsified D1, in which case one IS required.

- [ ] **Step 5: part-deux decision (spec O3).** All four touched files were **byte-identical** at spec
time. This PR diverges them unless propagated — decide deliberately and record the decision.

- [ ] **Step 6: C4** — expected C4-free (count stays 32). Verify by running `/c4`, do not assert it.
⚠ Descriptions are capped at **200 chars** (`tests/test_c4_dsl_description_cap.py`).

- [ ] **Step 7: `/final-review`**

- [ ] **Step 8: STOP.** Report and await approval. Do NOT commit the release, push, tag, or open a PR
without it.

---

## Self-Review

**Spec coverage:** §3.1 → Task 2; §3.2 plant → Task 2; §3.3 second clamp → Task 3; §3.4 → Task 4;
§3.6 `fernandez_bornn` → Task 5; §3.7 runtime/scope → Tasks 0, 1, 12; §4 glossary → Task 6; §5.1
measurement → Task 11; §5.2 NA → Tasks 8, 9; §5.3 tolerances → Task 8; §5.4 mechanics → Task 8;
§6 gates → Tasks 1, 2, 3, 9; §7 RQ3 → Task 7; O3 part-deux → Task 12; O4 split → the SPLIT POINT.
All D1–D6 map to a task.

**Placeholder scan — and the scan itself was wrong in rev 2.** It certified that Task 9's assertions
were "fully specified" when **there were none**: four of its five tests had docstring-only bodies and
one was `...`. Verified: **pytest reports both forms PASSED**. The scan grepped for the `...` token and
never asked "does this body assert anything" — a name-shaped heuristic missing a whole class, which is
the ADR-043 lesson recurring inside the document that cites it.

Current state: Task 9's five tests are written out with real assertions. **Task 3 Step 1 is the only
remaining elision** — its assertion, access route and summed-not-per-lane rule are fully specified and
the elided part is mirroring `_cover_shadows.py:1086-1097`, referenced by line rather than duplicated.
Every other step carries executable code or an exact command.

**Rule for any future revision of this plan: a test body that contains no `assert` is a placeholder,
whatever it is made of.**

**Type/name consistency:** `TOL_INVARIANT` / `TOL_ATTRIB` / `_PLANT_MARGIN` / `_CS_AGGREGATOR_ONLY_COLS`
/ `max_single_defender_player_id` / `max_def_pid` are used identically throughout.

**Seams verified against source before writing** — signatures, not assumptions:
`compute_blocking_score(frame, attacking_team_id, xt, *, home_team_id, defenders_to_remove=None,
method=..., params=None, pitch_control_cache=None)`; `BlockingScoreResult(blocking_score,
threat_original, threat_unblocked)` (`:908`); `_lane_int_probs(...) -> (p_int_def, p_int_att, t_ball,
p_ctrl)` (`:358-366`); `_lane_received_batched(...) -> (p_blocked_full, p_received_full,
p_received_loo)` (`:463-467`); `restore_id_dtype(values, source_dtype)` (`id_compat.py:198`); the
per-action frame resolution at `features.py:3672-3695`; the xfns read at `features.py:3784`.

**Task 2 was EXECUTED against real fixture data before this plan was handed over.** Measured on the
sportec fixture with the real `fitted_xt`:

| Quantity | Value |
|---|---|
| Scoreable actions | **9** (of 10; one has no resolvable frame) |
| Min raw `threat_unblocked - threat_original` | **+3.792** — invariant holds, comfortably |
| Actions where the plant goes negative past `_PLANT_MARGIN` | **9 / 9** |

So the plant is usable and the fixture is adequate — §3.2's "fixture inadequacy" branch does **not**
fire, and Task 2 can be implemented as written.

⚠ **One trap, hit while verifying, that an implementer will hit too.** A first run showed
`min raw = 0.000e+00` on every action and the plant firing **0/9** — which reads exactly like fixture
inadequacy. It was not: the harness had built xT via `ExpectedThreat(l=16, w=12).fit(actions)` on 10
synthetic actions, producing a degenerate grid where all threat is zero, so every counterfactual was a
no-op. The real fixture (`tests/tracking/conftest.py:29-36`) does **not** fit — it sets
`xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))` manually. **If Task 2 or Task 3 shows all-zero
threat, suspect the xT grid before concluding the fixture is inadequate.**

**Review round 1 applied in full (1 blocker / 4 major / 4 minor).** Two of its findings were
**measured rather than accepted**, and one measurement contradicted the review:

- **MAJOR 2 said the file "will very likely get slower".** Measured: shared prep **0.17 s** + raw
  scoring pass **0.59 s** + witness **0.09 s** = **0.85 s added** against the ~1.8 s module-scoping
  saving → **net ~0.95 s FASTER**. The review predicted the witness would be "the most expensive thing
  in the file" (~27 PC grids); it is the **cheapest** item at 0.09 s, because each grid on this frame
  subset is milliseconds. The *process* point stood — the additions genuinely were uncounted and
  Task 12 asserted non-regression on that uncounted basis — so the shared-fixture fix is adopted and
  the budget is now stated as a measured number.
- **MAJOR 3's `fernandez_bornn` question is pre-answered** — all three methods hold (Task 5 table).

**Two defects in this plan's own code, found by executing it and already fixed above:**

1. `compute_pitch_control(frame, team=tid, ...)` — `attacking_team_id` is **positional**
   (`pitch_control/_dispatch.py:31-33`); the keyword form `TypeError`s.
2. `from silly_kicks.tracking.linkage import link_actions_to_frames` — **there is no `linkage`
   module.** It lives in `tracking/utils.py` and is re-exported from `silly_kicks.tracking`; the wrong
   path would have failed at collection.

**Known risks left to the implementer, deliberately:** (1) `fitted_xt`'s scope — Task 1 Step 3 may hit
`ScopeMismatch` and the fix depends on whether anything mutates it; (2) Task 4 may legitimately go red,
which is a finding to report, not a test to weaken; (3) Task 5 may falsify D1, which changes the
glossary wording and adds an ADR.
