# Trained-model feature contract + canonical penalty-area constant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship a behavioural feature contract that makes a future feature-extractor change fail closed, single-source
the penalty-area constant for the two modules that already agree, and close the two cache/fail-fast gaps — without
introducing any train/serve skew.

**Architecture:** A new private `tracking/_feature_contract.py` mirrors `_chirality.py` (probe → fingerprint →
verify-at-load), with three deliberate policy differences: missing → warn, probe change → warn-and-skip the
fingerprint only, mismatch → raise. The canonical constant lands in `spadlconfig` with two frame-explicit
`in_penalty_area_*` helpers; `_ghost_gk` keeps `40.3` until its re-fit and is re-saved this cycle so that pin is
enforced by a raise rather than by prose.

**Tech Stack:** Python 3.10+, numpy, pandas; pytest; ruff + pyright (CI `lint`).

**Design source:** `docs/superpowers/specs/2026-07-25-feature-contract-and-penalty-area-constant-design.md` (rev 3;
two cross-session review rounds closed). Section refs (`spec §N`) point there for rationale — this plan is the *how*.
**Read spec §2 (D1–D3) and §7a/§7b before starting.**

**Branch:** `feat/feature-contract` off `main` @ `4ab63fe` (4.61.0). Single commit per the repo convention; the plan
STOPS before it (Task 13). Registers assigned at commit-prep — provisional `4.64.0 / PR-S135 / ADR-050`, re-confirm
against `origin/main`.

**Everything in this plan that is executable HAS BEEN EXECUTED.** The probe in Task 3 was run against all three
extractors (26 + 27 + 16 features, 0 NaN) and the box constant proven load-bearing on it. Signatures were read from
source, not recalled. Where a number is quoted, it was measured.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `silly_kicks/tracking/_feature_contract.py` | Probe, fingerprint builder, verifier. Extractor-agnostic. | **Create** |
| `silly_kicks/tracking/_warnings.py` | + `MissingFeatureContractWarning` | Modify |
| `silly_kicks/tracking/__init__.py` | Re-export the new warning | Modify |
| `silly_kicks/spadl/config.py` | + `penalty_area_half_width`, `penalty_area_depth` | Modify |
| `silly_kicks/tracking/_geometry.py` | + `in_penalty_area_absolute`, `in_penalty_area_goal_relative` | Modify |
| `silly_kicks/tracking/_xcross_attempt.py` | Consume canonical constant + helper; record/verify contract | Modify |
| `silly_kicks/tracking/defensive_credit/_params.py` | Consume canonical constant + helper | Modify |
| `silly_kicks/tracking/_xshot_occurrence.py` | Record/verify contract | Modify |
| `silly_kicks/tracking/_ghost_gk.py` | Record/verify contract; pitch-dims guard; `validate_provider` | Modify |
| `scripts/train_ghost_gk.py` | Startup provider check; self-invalidating cache token | Modify |
| `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` | Live corpus fingerprint | Modify |
| `tests/tracking/test_feature_contract.py` | Contract gates | Create |
| `tests/tracking/test_penalty_area.py` | Helper edge cases + migration identity | Create |
| `tests/tracking/test_geometry_constant_enumeration.py` | M1 auto-enumeration gate | Create |
| `tests/scripts/test_trainer_cache_and_providers.py` | §5.1/§5.1a/§5.2 gates | Create |

**Ordering:** Tasks 1–2 are independent leaves. Task 3 (probe) gates 4–6. Tasks 7–9 are the bundled items. Task 10
is the enumeration gate. **Task 11 (ghost re-save) is LAST of the working tasks, and that is a hard dependency, not
a preference (review P11):**

> **Any task that changes what a model declares must precede Task 11.**

Task 11 stamps a fingerprint *and a constants block* into the bundled artifact. If it ran before Task 10, and
Task 10's gate then forced ghost to declare a constant it had missed, the freshly-stamped artifact would be stale
the moment it was written — and stale in the one way nothing catches, because `load()` compares against whatever
the artifact says, so an artifact that under-declares is silently *self-consistent*. An earlier draft ran the
re-save at position 7, immediately after wiring ghost; that ordering was wrong for exactly this reason. Moving it
last costs nothing (it is a standalone metadata migration) and removes the failure mode entirely.

**Task 12 (CI escalation of the missing-contract warning) sits after Task 11** because Task 11 stamps every bundled
artifact, which is what makes the escalation need an *empty* opt-out list rather than a ~14-file one.

---

## Task 0: Branch and baseline

- [ ] **Step 1: Branch**

```bash
git -C "D:/Development/karstenskyt__silly-kicks" fetch origin
git -C "D:/Development/karstenskyt__silly-kicks" status --short
git -C "D:/Development/karstenskyt__silly-kicks" rev-list --left-right --count HEAD...origin/main
git -C "D:/Development/karstenskyt__silly-kicks" checkout -b pr-s133-feature-contract
```

**Branch name follows this repo's convention `pr-s<NN>-<slug>`**, matching the PR-S sequence in `TODO.md` and
recent PR titles — not `feat/...`.

> **No `git reset --hard`.** An earlier draft opened with `reset --hard origin/main`. Verify sync with the
> `rev-list` count above instead (expect `0	0`) and only reconcile if it is non-zero. A hard reset is a
> destructive command run for its side effect of *usually* doing nothing — and on the one run where the tree is
> not clean, it silently discards the owner's work. The two design docs for this cycle are untracked and would
> survive it, which is exactly what makes the habit easy to keep and occasionally expensive.

- [ ] **Step 2: Record the baseline count**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q --benchmark-skip 2>&1 | tail -3`
Expected: all pass. Record the number — every later "still green" claim compares to it.

---

## Task 1: The two feature-contract warning categories (spec §3.3, R2 N6, review P5)

**Files:** Modify `silly_kicks/tracking/_warnings.py`, `silly_kicks/tracking/__init__.py`; Test: `tests/tracking/test_feature_contract.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_feature_contract.py
import warnings

import pytest


@pytest.mark.parametrize(
    "name", ["MissingFeatureContractWarning", "UnverifiableFeatureContractWarning"]
)
def test_warning_category_is_registered_on_every_public_surface(name):
    """R2 N6: the value of a named category is a STABLE import path. Registering it in one
    list and forgetting the others is the classic failure."""
    import silly_kicks.tracking as T
    import silly_kicks.tracking._warnings as W

    cls = getattr(W, name)
    assert issubclass(cls, UserWarning)
    assert name in W.__all__
    assert name in T.__all__
    assert getattr(T, name) is cls


def test_the_warning_can_be_escalated_to_an_error_by_category():
    """This is the whole point (spec §3.3): a batch consumer sets filterwarnings('error', ...)
    and gets fail-closed semantics without silly-kicks changing its default."""
    from silly_kicks.tracking import MissingFeatureContractWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFeatureContractWarning)
        with pytest.raises(MissingFeatureContractWarning):
            warnings.warn("x", MissingFeatureContractWarning, stacklevel=2)


def test_the_two_categories_are_independent():
    """P5: neither may subclass the other. If Unverifiable were a subclass of Missing, escalating
    Missing would ALSO escalate probe changes -- silently undoing §3.2's warn-and-skip decision.
    A subclass relationship is exactly how someone would 'tidy' these two later."""
    from silly_kicks.tracking import (
        MissingFeatureContractWarning,
        UnverifiableFeatureContractWarning,
    )

    assert not issubclass(UnverifiableFeatureContractWarning, MissingFeatureContractWarning)
    assert not issubclass(MissingFeatureContractWarning, UnverifiableFeatureContractWarning)
```

- [ ] **Step 2: Run, verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py -q`
Expected: `ImportError: cannot import name 'MissingFeatureContractWarning'`

- [ ] **Step 3: Implement**

Append to `silly_kicks/tracking/_warnings.py` (follow the file's existing class style and add the name to its
`__all__`):

```python
class MissingFeatureContractWarning(UserWarning):
    """A trained-model artifact carries NO feature contract, so it cannot be verified at all.

    Additive by design (spec D2): pre-contract artifacts still load. This is the category meant to
    be ESCALATED by a batch consumer that wants fail-closed semantics:

        warnings.filterwarnings("error", category=MissingFeatureContractWarning)
    """


class UnverifiableFeatureContractWarning(UserWarning):
    """A contract exists but part of it could not be checked this time: the probe changed, a
    recorded constant is no longer declared, or a mismatch was waved through by legacy_override.

    DELIBERATELY a separate category from MissingFeatureContractWarning (review P5). Escalating
    *that* one to an error must NOT turn a probe change into a hard failure -- spec §3.2 chose
    warn-and-skip precisely so that adding a constant (which §3.1 REQUIRES extending the probe)
    does not brick every artifact until re-saved, and escalation must not silently undo that.
    Same reasoning ADR-041 gives for keeping categories separate: silencing (or escalating) a
    routine notice must not also catch a genuine misuse signal.
    """
```

Then re-export **both** from `silly_kicks/tracking/__init__.py` exactly as the existing categories are: add the
imports alongside them and add both names to `__all__` (keep the list alphabetised). Register both in
`_warnings.__all__` and in whatever public-surface test/doc covers the existing three.

- [ ] **Step 4: Run, verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py -q` — **4 passed**
(3 functions; `test_warning_category_is_registered_on_every_public_surface` is parametrized 2×).

---

## Task 2: Canonical constant + the two frame-explicit predicates (spec §4, R2 N2)

**Files:** Modify `silly_kicks/spadl/config.py`, `silly_kicks/tracking/_geometry.py`; Test: `tests/tracking/test_penalty_area.py` (create)

- [ ] **Step 1: Write the failing test — EDGES, not mid-box**

```python
# tests/tracking/test_penalty_area.py
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._geometry import in_penalty_area_absolute, in_penalty_area_goal_relative


def test_canonical_constants_are_the_law_values():
    assert spadlconfig.penalty_area_half_width == 20.16      # FIFA: 40.32 m wide
    assert spadlconfig.penalty_area_depth == 16.5


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (88.5, 34.0, True),    # EXACTLY on the depth line -> inside (Law: the area includes its lines)
        (88.49, 34.0, False),  # one cm outside the depth line
        (105.0, 13.84, True),  # EXACTLY on the y edge -> inside
        (105.0, 13.83, False),
        (105.0, 54.16, True),  # the mirrored y edge
        (105.0, 54.17, False),
    ],
)
def test_absolute_frame_edges(x, y, expected):
    """R2 N2: a mid-box fixture passes under EVERY wrong convention. Only the edges discriminate."""
    assert in_penalty_area_absolute(x, y, attacked_goal_x=105.0) is expected


def test_absolute_frame_mirrored_goal():
    """The other goal: depth measured from x=0, not x=105."""
    assert in_penalty_area_absolute(16.5, 34.0, attacked_goal_x=0.0) is True
    assert in_penalty_area_absolute(16.51, 34.0, attacked_goal_x=0.0) is False
    assert in_penalty_area_absolute(88.5, 34.0, attacked_goal_x=0.0) is False


@pytest.mark.parametrize(
    "gr_x,y,expected",
    [(16.5, 34.0, True), (16.51, 34.0, False), (0.0, 13.84, True), (0.0, 13.83, False)],
)
def test_goal_relative_frame_edges(gr_x, y, expected):
    """Takes NO goal argument: the caller resolved attacked-vs-defended by producing gr_x,
    so the ambiguity cannot re-enter the helper (spec §4)."""
    assert in_penalty_area_goal_relative(gr_x, y) is expected


def test_goal_relative_helper_has_no_goal_parameter():
    """Guards the N2 design decision itself: re-adding a goal_x parameter re-opens the
    three-convention collision (absolute-vs-relative, attacked-vs-defended)."""
    import inspect

    params = set(inspect.signature(in_penalty_area_goal_relative).parameters)
    assert params == {"gr_x", "y"}, f"unexpected parameters: {params}"
```

- [ ] **Step 2: Run, verify RED**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_penalty_area.py -q`
Expected: `ImportError: cannot import name 'in_penalty_area_absolute'`

- [ ] **Step 3: Implement the constants**

In `silly_kicks/spadl/config.py`, beside the existing pitch constants:

```python
#: FIFA Laws of the Game: the penalty area is 40.32 m wide and 16.5 m deep. Canonical -- do NOT
#: re-derive these locally. `_ghost_gk` deliberately still uses 40.3 until its re-fit (spec D3).
penalty_area_half_width: float = 20.16
penalty_area_depth: float = 16.5
```

- [ ] **Step 4: Implement the two predicates**

Append to `silly_kicks/tracking/_geometry.py`:

```python
def in_penalty_area_goal_relative(gr_x: float, y: float) -> bool:
    """Penalty-area membership in GOAL-RELATIVE coords (the reference goal sits at gr_x = 0).

    Takes NO goal argument on purpose: the caller has already resolved attacked-vs-defended by
    producing ``gr_x``, so that ambiguity cannot re-enter here (spec §4 / review R2 N2). Boundary
    is non-strict on both axes -- the Law's area includes its own lines.

    Examples
    --------
    >>> in_penalty_area_goal_relative(16.5, 34.0)
    True
    >>> in_penalty_area_goal_relative(16.51, 34.0)
    False
    """
    # NOTE: no lower bound on gr_x, DELIBERATELY. The shipped xCross predicate is
    # `gr_x <= _BOX_DEPTH_M` with no `0 <= gr_x` guard, and real tracking carries x beyond the
    # goal line (gr_x < 0), so adding one would CHANGE xCross behaviour for behind-the-line
    # players and break the byte-identity this migration promises. Whether a behind-the-line
    # point should count as in-box is a separate, measurable question -- not this cycle's.
    return bool(
        (gr_x <= _spadlconfig.penalty_area_depth)
        and (abs(y - GOAL_Y) <= _spadlconfig.penalty_area_half_width)
    )


def in_penalty_area_absolute(x: float, y: float, *, attacked_goal_x: float) -> bool:
    """Penalty-area membership in ABSOLUTE (action-LTR) coords.

    ``attacked_goal_x`` is the absolute x of the goal whose area is being tested (0.0 or 105.0).
    Named to avoid colliding with this module's ``goal_x``, which means the *defended* goal in the
    to-goal-relative transforms (review R2 N2).

    Examples
    --------
    >>> in_penalty_area_absolute(88.5, 34.0, attacked_goal_x=105.0)
    True
    >>> in_penalty_area_absolute(88.49, 34.0, attacked_goal_x=105.0)
    False
    """
    gr_x = abs(float(attacked_goal_x) - float(x))
    return in_penalty_area_goal_relative(gr_x, y)
```

Add `import silly_kicks.spadl.config as _spadlconfig` at the top if not already present (the module already
imports `spadlconfig` in some form — check and reuse rather than double-import).

- [ ] **Step 5: Run, verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_penalty_area.py -q` — 14 passed.

- [ ] **Step 6: Migrate the two sites that already hold 20.16, and prove byte-identity**

**`defensive_credit/_params.py` (scalar) — the body moves to the helper and the two constants are DELETED, not
aliased.** Replace the body of `_is_inside_attacked_box` (`:78-81`) with
`in_penalty_area_absolute(x, y, attacked_goal_x=_FIELD_LENGTH)` — this site is the **absolute** frame
(`x >= 105 - 16.5`) — and **delete `_BOX_DEPTH_M` (`:19`) and `_BOX_HALF_WIDTH_M` (`:20`) outright.** Delete the
now-stale `:12-18` prose flagging the 20.15/20.16 discrepancy and replace it with a pointer to the canonical
constant plus the D3 note about ghost.

> **Delete, do NOT alias, here (review S2).** VERIFIED: `:80` is the *only* consumer of these two names in the whole
> `defensive_credit/` package. Once the body calls the helper — which resolves `_spadlconfig.penalty_area_*` itself
> — nothing reads them, so keeping them "as canonical aliases" would leave two names that look load-bearing and are
> not: `monkeypatch.setattr(_params, "_BOX_HALF_WIDTH_M", 20.15)` would change nothing. That is the exact inverse of
> the trap this plan avoids for xCross, and it would satisfy Task 10's enumerator *vacuously* — a gate whose whole
> job is to notice constants that move features, quietly cataloguing one that cannot.
>
> Registry coverage is unaffected: `DECLARED_CONSTANT_SOURCES` is keyed by **bare name**, and both names still exist
> in `_xcross_attempt.py` where they genuinely drive `:209`. Task 10's enumerator scans both modules and its
> "listed-but-not-found" check stays empty. The byte-identity sweep below is also unaffected — it imports
> `_is_inside_attacked_box`, not the constants.

**Note the asymmetry, and that it is principled rather than inconsistent:** `_params` DELETES its constants because
its predicate moves to the helper; xCross KEEPS its as aliases because its vectorized predicate cannot. The rule is
the same in both cases — *a module-level constant exists iff something in that module reads it.*

**`_ghost_gk.py` — DO NOT TOUCH.** D3 freezes it, but note *why* it could not be migrated even if D3 said
otherwise: `:608` tests depth with a **strict** `<` (`atk_xs < _PENALTY_AREA_X`) where the canonical helper is
non-strict on both axes. Routing ghost through the helper would change `attackers_in_box` for an attacker standing
exactly on the 16.5 m line — a silent feature change on a trained model, which is the entire class of bug this
cycle exists to make impossible. Recorded here because "unify the last caller" is the obvious next move for
whoever picks this up, and the reason not to is invisible from the call site.

**`_xcross_attempt.py` (VECTORIZED — read the constants, do NOT call the helper).** The `:209` predicate operates on
numpy arrays:

```python
    in_box = (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M) & ~is_ball
```

`in_penalty_area_goal_relative` is scalar and returns `bool(...)`; calling it here would need a per-element loop
over every player in every frame. So xCross keeps its vectorized expression and changes only where the numbers come
from — **rebind `:69-70` as aliases of the canonical constant**, matching this codebase's existing idiom for exactly
this (`_ghost_gk.py:231`, `_params.py:10-11`):

```python
_BOX_DEPTH_M = _spadlconfig.penalty_area_depth  # 16.5
_BOX_HALF_WIDTH_M = _spadlconfig.penalty_area_half_width  # 20.16
```

Alias rather than delete-and-inline, for two reasons that only show up later: the `:209` expression stays untouched
(so the byte-identity claim is about the *value*, with no expression rewrite to also verify), and the names remain
visible to Task 10's AST enumerator. Inlining `_spadlconfig.*` at the use site would make both constants invisible
to the very gate that is supposed to notice them — a gate that silently stops seeing its subject is worse than no
gate.

Because the alias binds at import, the contract must declare **the alias**, not `_spadlconfig.*` — otherwise a test
that patches one sees a mismatch the other never had. Task 5's xCross declaration and its test both use the alias.

That is the honest scope of this migration: **the single source is the CONSTANT, not the predicate.** An earlier
draft of this plan said "replace the `:209` predicate with the helper", which is not possible at a vectorized site.
Unifying the predicate too would mean adding an array-tolerant sibling — real work, no behaviour change, and not
what this cycle is for. The contract guards the value; the value is now shared.

Then append to `tests/tracking/test_penalty_area.py`:

```python
def test_migration_is_byte_identical_for_both_20_16_sites():
    """Both sites were already non-strict on x with the same abs(y-34) form, so the canonical
    constants must reproduce them exactly. Grid-sweep, not spot-check -- and BOTH sites (review
    P7): the scalar one and the vectorized one can diverge independently."""
    import numpy as np

    import silly_kicks.tracking._geometry as _geo
    from silly_kicks.tracking._xcross_attempt import _BOX_DEPTH_M, _BOX_HALF_WIDTH_M
    from silly_kicks.tracking.defensive_credit._params import _is_inside_attacked_box

    xs = np.arange(80.0, 120.01, 0.25)
    ys = np.arange(10.0, 58.01, 0.25)

    # -- site 1: defensive_credit, ABSOLUTE frame
    for x in xs:
        for y in ys:
            old = (x >= 105.0 - 16.5) and (abs(y - 34.0) <= 20.16)
            assert bool(_is_inside_attacked_box(float(x), float(y))) is bool(old), (x, y)

    # -- site 2: xCross, GOAL-RELATIVE frame, vectorized. Reproduce the shipped expression against
    #    the post-migration constants over the same grid.
    gr_x = np.abs(105.0 - xs)[:, None]
    yy = ys[None, :]
    new = (gr_x <= _BOX_DEPTH_M) & (np.abs(yy - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M)
    old = (gr_x <= 16.5) & (np.abs(yy - 34.0) <= 20.16)
    assert np.array_equal(new, old)


def test_absolute_helper_diverges_only_beyond_the_reachable_pitch():
    """P8: `in_penalty_area_absolute` uses gr_x = abs(105 - x), so it has an UPPER bound the old
    `x >= 105 - 16.5` form does not -- they disagree for x > 121.5 (i.e. >16.5 m PAST the goal
    line).

    DOCUMENTED, not proven unreachable (review S3). The nearest cap is `_SPADL_X_MAX = 120.0`
    (`_gk_identification.py:23`, raised at `:88`) -- but that validates TRACKING FRAME coords
    inside `derive_goalkeepers`, whereas this helper's only production caller is
    `defensive_credit`, which works on SPADL ACTION coords. Different path; no equivalent
    validation is known to guard it. So: the tracking path caps x at 120.0 and 120.0 < 121.5;
    the action path is not shown to be validated, and this test records the divergence rather
    than dismissing it. If the cap moves, the assertion below fails and the behind-the-goal
    semantics get decided explicitly instead of inherited."""
    from silly_kicks.tracking._gk_identification import _SPADL_X_MAX
    from silly_kicks.tracking._geometry import in_penalty_area_absolute

    assert in_penalty_area_absolute(121.4, 34.0, attacked_goal_x=105.0) is True
    assert in_penalty_area_absolute(121.6, 34.0, attacked_goal_x=105.0) is False  # old form: True
    assert _SPADL_X_MAX < 121.5, (
        f"the x cap moved to {_SPADL_X_MAX}; the abs() divergence is now reachable and the "
        f"behind-the-goal semantics must be decided, not inherited"
    )
```

Note the sweep now runs to **120.0**, not 105.0 — the old range stopped exactly at the goal line and so could never
have seen this divergence class at all.

- [ ] **Step 7: Run the affected suites**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_penalty_area.py -q` — **16 passed** (Step 5's 14 plus
the two added here). Then the consumers:
`.venv/Scripts/python.exe -m pytest tests/tracking/ -k "xcross or defensive_credit" -q`
Expected: all pass — the migration changes no value.

---

## Task 3: The contract probe (spec §3.1) — **verified, do not "simplify"**

**Files:** Create `silly_kicks/tracking/_feature_contract.py`; Test: `tests/tracking/test_feature_contract.py`

> **This probe was EXECUTED against all three extractors while writing this plan.** Measured: ghost 26 features /
> 0 NaN, xS 27 / 0 NaN, xCross 16 / 0 NaN, and `attackers_in_box` = **0** at half-width 20.15 vs **1** at 20.16, i.e.
> the box constant is provably load-bearing. Each element below is load-bearing for one of those facts — an earlier
> draft with 4 defenders / 3 attackers and no `z` produced **7 NaN features for xS**.

- [ ] **Step 1: Write the failing test**

```python
def test_probe_is_nan_free_for_all_three_extractors():
    """Spec §6: a NaN feature is a feature the contract cannot gate at all."""
    import numpy as np

    from silly_kicks.tracking._feature_contract import contract_probe_frame
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, extract_ghost_gk_features
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features
    from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

    f = contract_probe_frame()

    g = extract_ghost_gk_features(
        f, gk_team_id="B", goal_x=105.0, score_diff=1, phase=0, ball_carrier_team_id="A",
        prev_defensive_line_x=90.0, prev_defending_centroid_x=94.0, dt=0.04,
    )
    assert not [n for n in GHOST_GK_FEATURE_NAMES if not np.isfinite(float(g[n].iloc[0]))]

    s = extract_xshot_features(f, gk_team_id="B", goal_x=105.0).iloc[0]
    assert not [c for c in s.index if not np.isfinite(float(s[c]))]

    c = extract_xcross_features(
        f, gk_team_id="B", goal_x=105.0, carrier_player_id="A2", score_differential=1.0
    ).iloc[0]
    assert not [col for col in c.index if not np.isfinite(float(c[col]))]


def test_probe_makes_the_box_constant_load_bearing(monkeypatch):
    """§3.1's rule -- the single most load-bearing gate in the spec. Asserted THROUGH the real
    extractor (review P6): re-implementing ghost's predicate inside the test would pass no matter
    what `extract_ghost_gk_features` actually does, including if its `<` became `<=`.
    Measured: 0 at the shipped 40.3, 1 after flipping to 40.32."""
    import silly_kicks.tracking._ghost_gk as gg
    from silly_kicks.tracking._feature_contract import contract_probe_frame

    kw = dict(
        gk_team_id="B", goal_x=105.0, score_diff=1, phase=0, ball_carrier_team_id="A",
        prev_defensive_line_x=90.0, prev_defending_centroid_x=94.0, dt=0.04,
    )
    before = int(gg.extract_ghost_gk_features(contract_probe_frame(), **kw)["attackers_in_box"].iloc[0])

    monkeypatch.setattr(gg, "_PENALTY_AREA_Y_MIN", (68.0 - 40.32) / 2.0)
    monkeypatch.setattr(gg, "_PENALTY_AREA_Y_MAX", (68.0 + 40.32) / 2.0)
    after = int(gg.extract_ghost_gk_features(contract_probe_frame(), **kw)["attackers_in_box"].iloc[0])

    assert (before, after) == (0, 1)
```

- [ ] **Step 2: Run, verify RED** — `ModuleNotFoundError: silly_kicks.tracking._feature_contract`

- [ ] **Step 3: Implement the probe**

```python
# silly_kicks/tracking/_feature_contract.py
"""Trained-model FEATURE contract (spec 2026-07-25; ADR-050).

Sibling of ``_chirality.py`` -- probe, fingerprint, verify-at-load -- with three deliberate policy
differences (spec D2 / §3.2): a MISSING contract warns rather than raising (pre-contract artifacts
are undeclared, not known-bad); a PROBE change warns and skips the fingerprint comparison ONLY; a
fingerprint or declared-constant mismatch RAISES.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from ._warnings import MissingFeatureContractWarning

_CONTRACT_VERSION = "feature-contract-1"

# Tolerance is CHOSEN, not inherited from chirality (spec §3, review R1 B2i). Chirality's
# rtol=1e-2 was sized for a gross sign flip on a probability; a feature vector spans metres,
# counts and radians, where rtol=1e-2 on a ~17 m feature is a 0.17 m blind spot -- 17x the 0.01 m
# change this contract exists to catch. atol pending a measured DGX-vs-x86 delta (spec §3 N3).
_CONTRACT_ATOL = 1e-6
_CONTRACT_RTOL = 0.0

_BASE = dict(game_id="fc", period_id=1, frame_id=1, time_seconds=10.0, is_ball=False)


def _player(pid, team, x, y, *, gk=False, vx=0.7, vy=-0.4):
    return dict(_BASE, team_id=team, player_id=pid, x=x, y=y, z=0.0, vx=vx, vy=vy, is_goalkeeper=gk)


def contract_probe_frame() -> pd.DataFrame:
    """One synthetic frame. Team A attacks the goal at x=105; team B defends it and has the keeper.

    EVERY element here is load-bearing -- MEASURED, do not "simplify" (spec §3.1):

    * **A1 at (90.0, 13.845)** is the discriminating row: gr_x = 15.0 is inside the 16.5 m depth,
      and y = 13.845 is inside ``[13.84, ...]`` but outside ``[13.85, ...]``, so ``attackers_in_box``
      is 1 at half-width 20.16 and 0 at 20.15. Being in the y-band ALONE is not enough -- of 844
      band rows on a real match only 70 were also within depth.
    * **five attackers and five defenders**: xS's ``_nearest_k`` fills ``DefDist_0..4`` and
      ``OffDist_0..4``; with 4 and 3 the extractor returned 7 NaN features (measured).
    * **A2..A5 sit well outside the box** so the 0-vs-1 discrimination stays clean.
    * **>=3 non-collinear defenders** make ghost's ConvexHull ``defending_team_compactness`` finite.
    * **a ball row carrying ``z``**: without it xS's ``z`` feature is NaN (measured).
    """
    rows = [
        _player("A1", "A", 90.0, 13.845),
        _player("A2", "A", 84.0, 40.0),
        _player("A3", "A", 76.0, 28.0),
        _player("A4", "A", 64.0, 47.0),
        _player("A5", "A", 58.0, 19.0),
        _player("B1", "B", 95.0, 30.0),
        _player("B2", "B", 97.0, 44.0),
        _player("B3", "B", 92.0, 22.0),
        _player("B4", "B", 99.0, 36.0),
        _player("B5", "B", 86.0, 51.0),
        _player("BGK", "B", 103.0, 34.5, gk=True),
        dict(_BASE, team_id=None, player_id=None, x=88.0, y=20.0, z=0.6, vx=3.0, vy=1.0,
             is_goalkeeper=False, is_ball=True),
    ]
    frame = pd.DataFrame(rows)
    frame["is_ball"] = frame["is_ball"].astype(bool)
    return frame
```

- [ ] **Step 4: Run, verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py -q` — **8 passed**
(Task 1's 4 + Task 3's 4; they share the file).

---

## Task 4: `feature_contract` + `verify_feature_contract` (spec §3, §3.2)

**Files:** Modify `silly_kicks/tracking/_feature_contract.py`; Test: `tests/tracking/test_feature_contract.py`

- [ ] **Step 1: Write the failing tests — both sides of every branch**

```python
def _fc(**over):
    """A minimal valid contract dict, overridable per-test."""
    return {
        "version": "feature-contract-1",
        "probe_sha256": "abc123",
        "fingerprint": [1.0, 2.0, 3.0],
        "constants": {"penalty_area_half_width": 20.16},
        **over,
    }


class _Err(Exception):
    pass


def test_round_trip_on_an_unmodified_contract_passes():
    """THE B1 regression: with chirality's equal_nan=False this fails on a vector against
    ITSELF (measured: 3 NaN features -> allclose(v, v) is False)."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    verify_feature_contract(_fc(), _fc(), legacy_override=False, model_name="m", error_cls=_Err)


def test_fingerprint_mismatch_raises():
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_Err, match="feature contract"):
        verify_feature_contract(
            _fc(fingerprint=[1.0, 2.0, 9.0]), _fc(), legacy_override=False,
            model_name="m", error_cls=_Err,
        )


def test_missing_contract_warns_and_does_not_raise():
    """Spec D2 -- asserted BY CATEGORY, not by message text."""
    from silly_kicks.tracking import MissingFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.warns(MissingFeatureContractWarning):
        verify_feature_contract(_fc(), None, legacy_override=False, model_name="m", error_cls=_Err)


def test_probe_change_warns_and_skips_the_fingerprint():
    """UNVERIFIABLE, not MISSING -- and the distinction is load-bearing, see the test below."""
    from silly_kicks.tracking import UnverifiableFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(
            _fc(probe_sha256="NEW", fingerprint=[9.0, 9.0, 9.0]), _fc(),
            legacy_override=False, model_name="m", error_cls=_Err,
        )


def test_escalating_the_missing_category_does_not_brick_a_probe_change():
    """THE reason for two categories (review P5).

    SS3.2 chose warn-and-skip on a probe change precisely so that adding a constant -- which SS3.1
    REQUIRES extending the probe for -- does not hard-fail every not-yet-re-saved artifact. A batch
    consumer that escalates the missing-contract category must still get that behaviour. One
    umbrella category would silently undo the design decision the spec argued for.
    """
    import warnings as _w

    from silly_kicks.tracking import MissingFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _w.filterwarnings("error", category=MissingFeatureContractWarning)
        # a probe change must still merely warn under that filter
        verify_feature_contract(
            _fc(probe_sha256="NEW", fingerprint=[9.0, 9.0, 9.0]), _fc(),
            legacy_override=False, model_name="m", error_cls=_Err,
        )
        # ...while a genuinely missing contract now raises, which is what escalation bought
        with pytest.raises(MissingFeatureContractWarning):
            verify_feature_contract(
                _fc(), None, legacy_override=False, model_name="m", error_cls=_Err
            )


def test_probe_change_PLUS_constant_change_still_raises():
    """R2 N1: the two nets must not cancel. Skipping the fingerprint must NOT skip constants."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_Err, match="constant"):
        verify_feature_contract(
            _fc(probe_sha256="NEW", constants={"penalty_area_half_width": 20.15}), _fc(),
            legacy_override=False, model_name="m", error_cls=_Err,
        )


def test_constant_change_alone_raises_even_when_the_fingerprint_matches():
    """A sub-probe-resolution change (20.16 -> 20.161) moves no feature."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_Err, match="constant"):
        verify_feature_contract(
            _fc(constants={"penalty_area_half_width": 20.161}), _fc(),
            legacy_override=False, model_name="m", error_cls=_Err,
        )


def test_new_constant_keys_are_additive_and_removed_keys_warn():
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    from silly_kicks.tracking import UnverifiableFeatureContractWarning

    verify_feature_contract(  # new key on the library side -> ignored
        _fc(constants={"penalty_area_half_width": 20.16, "new_thing": 1.0}), _fc(),
        legacy_override=False, model_name="m", error_cls=_Err,
    )
    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(
            _fc(constants={}), _fc(), legacy_override=False, model_name="m", error_cls=_Err
        )


@pytest.mark.parametrize("kind", ["fingerprint", "constants"])
def test_legacy_override_escapes_EITHER_mismatch_with_a_warning(kind):
    """P9: the constants branch used to fall through SILENTLY while the fingerprint branch warned.
    Parametrized so neither can regress without the other noticing."""
    from silly_kicks.tracking import UnverifiableFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    over = (
        {"fingerprint": [9.0, 9.0, 9.0]} if kind == "fingerprint"
        else {"constants": {"penalty_area_half_width": 20.15}}
    )
    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(
            _fc(**over), _fc(), legacy_override=True, model_name="m", error_cls=_Err,
        )


def test_the_raised_type_is_the_MODEL_s_own_integrity_error():
    """`error_cls` exists so a consumer catching `_ghost_gk.IntegrityError` catches this too --
    the same reasoning ADR-040 gives for chirality. Asserted, not assumed."""
    from silly_kicks.tracking._ghost_gk import IntegrityError
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(IntegrityError):
        verify_feature_contract(
            _fc(fingerprint=[9.0, 9.0, 9.0]), _fc(), legacy_override=False,
            model_name="GhostGk", error_cls=IntegrityError,
        )


def test_builder_refuses_a_non_finite_feature_vector():
    """Enforces the zero-NaN policy at SAVE time by construction, mirroring
    chirality_fingerprint's own non-finite guard -- not merely in a test."""
    import numpy as np

    from silly_kicks.tracking._feature_contract import feature_contract

    with pytest.raises(ValueError, match="non-finite"):
        feature_contract(lambda: np.array([1.0, np.nan]), constants={})
```

- [ ] **Step 2: Run, verify RED** — `ImportError: cannot import name 'feature_contract'`

- [ ] **Step 3: Implement**

```python
def feature_contract(extract_on_probe: Callable[[], np.ndarray], *, constants: dict) -> dict:
    """Build the contract. ``extract_on_probe`` is a ZERO-ARGUMENT closure the model supplies,
    binding its own extractor to :func:`contract_probe_frame` -- the three extractors' signatures
    genuinely do not unify, so this module stays extractor-agnostic (spec §3).

    Raises on a non-finite vector: a NaN feature is one the contract could never gate.
    """
    frame = contract_probe_frame()
    probe_sha = hashlib.sha256(
        json.dumps(frame.to_dict("records"), sort_keys=True, default=str).encode()
    ).hexdigest()
    values = np.asarray(extract_on_probe(), dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"feature contract produced non-finite values: {values!r}")
    return {
        "version": _CONTRACT_VERSION,
        "probe_sha256": probe_sha,
        "fingerprint": [round(float(v), 10) for v in values],
        "constants": {k: float(v) for k, v in constants.items()},
    }


def verify_feature_contract(
    recomputed: dict,
    stored: dict | None,
    *,
    legacy_override: bool,
    model_name: str,
    error_cls: type[Exception] | None = None,
) -> None:
    """Verify at load(). Argument order mirrors ``verify_chirality`` EXACTLY -- recomputed first,
    stored second. Both are dicts, so a swap is not a type error: it would make the ``is None``
    branch test the wrong side and silently invert D2 (review R1 M2).
    """
    err = error_cls or ValueError

    if stored is None:
        warnings.warn(
            f"{model_name}: artifact carries no feature contract, so its extractor cannot be "
            "verified. Loading anyway (pre-contract artifacts are undeclared, not known-bad); "
            "re-save to gain the guard.",
            MissingFeatureContractWarning,
            stacklevel=2,
        )
        return

    # Constants are PROBE-INDEPENDENT, so they are compared FIRST and always (review R2 N1):
    # a probe change is no reason to stop comparing 20.16 against 20.15.
    rec_c = dict(recomputed.get("constants") or {})
    sto_c = dict(stored.get("constants") or {})
    removed = sorted(set(sto_c) - set(rec_c))
    if removed:
        warnings.warn(
            f"{model_name}: declared constant(s) {removed} are recorded in the artifact but no "
            "longer declared by the library; cannot compare them.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )
    changed = {k: (sto_c[k], rec_c[k]) for k in set(rec_c) & set(sto_c) if rec_c[k] != sto_c[k]}
    if changed:
        if not legacy_override:
            raise err(
                f"{model_name}: declared constant mismatch {changed} (artifact value first). The "
                "features this model was trained on were computed with different geometry; "
                "refusing to load. Re-fit, or pass legacy_override=True only if independently "
                "verified."
            )
        # An override that silently swallows a CONSTANT mismatch is the worst branch in this
        # function: the fingerprint branch below warns, so a reader would reasonably infer this one
        # does too (review P9). Symmetry here is not tidiness -- it is the difference between an
        # audited escape hatch and an invisible one.
        warnings.warn(
            f"{model_name}: declared constant mismatch {changed} suppressed by legacy_override.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )

    if recomputed.get("probe_sha256") != stored.get("probe_sha256"):
        warnings.warn(
            f"{model_name}: cannot verify the fingerprint -- the contract probe changed (stored "
            f"{str(stored.get('probe_sha256', ''))[:8]} vs library "
            f"{str(recomputed.get('probe_sha256', ''))[:8]}). Re-save to regain teeth. The "
            "declared-constant comparison above still applied.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )
        return

    a = np.asarray(recomputed.get("fingerprint", []), dtype=float)
    b = np.asarray(stored.get("fingerprint", []), dtype=float)
    # equal_nan=True is belt-and-braces: the builder already forbids non-finite values, so this
    # can only ever mask a case that cannot be stored (spec §3 B1).
    ok = a.shape == b.shape and np.allclose(
        a, b, atol=_CONTRACT_ATOL, rtol=_CONTRACT_RTOL, equal_nan=True
    )
    if not ok:
        if legacy_override:
            warnings.warn(
                f"{model_name}: feature contract mismatch overridden by legacy_override.",
                UnverifiableFeatureContractWarning,
                stacklevel=2,
            )
            return
        raise err(
            f"{model_name}: feature contract mismatch -- the library's extractor no longer "
            f"reproduces the features this artifact was trained on (atol={_CONTRACT_ATOL}, "
            f"rtol={_CONTRACT_RTOL}). Refusing to load; re-fit or pass legacy_override=True."
        )
```

- [ ] **Step 4: Run, verify GREEN**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py -q` — **20 passed**:
Task 1's 4 + Task 3's 4 + Task 4's 12 (11 functions, of which
`test_legacy_override_escapes_EITHER_mismatch_with_a_warning` is parametrized 2×).

---

## Task 5: Wire the contract into xS and xCross (spec §3)

**Files:** Modify `silly_kicks/tracking/_xshot_occurrence.py`, `silly_kicks/tracking/_xcross_attempt.py`; Test: `tests/tracking/test_feature_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_xshot_and_xcross_record_and_verify_a_contract(tmp_path, monkeypatch):
    """save() stamps it; load() on the same library passes; a LIBRARY change then raises.

    Mutate the LIBRARY, never the artifact on disk (review P2): `load()` verifies SHA256SUMS
    BEFORE parsing metadata (`_xshot_occurrence.py:489-500`), so editing metadata.json raises
    `IntegrityError("Integrity check failed for metadata.json")` and no domain `match=` can ever
    fire. Mutating the library is also the truer test -- the real failure mode is a library change
    under a fixed artifact; tampering is already the SHA check's job.
    """
    import json

    import silly_kicks.tracking._xshot_occurrence as xs_mod
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    m = XShotOccurrenceModel.from_variant("default")
    out = tmp_path / "xs"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert "feature_contract" in meta
    # xS declares GOAL WIDTH, not the penalty area -- VERIFIED: `_xshot_occurrence.py` contains no
    # penalty-area constant or predicate at all; its only geometry constants are the goal mouth
    # (`GOAL_WIDTH`/`GOAL_Y_MIN`/`GOAL_Y_MAX`, :36-39), which drive `openGoal` (:85-118).
    assert meta["feature_contract"]["constants"] == {"goal_width": 7.32}

    XShotOccurrenceModel.load(out)  # round-trip on the unmodified artifact is clean

    # Patching GOAL_WIDTH moves BOTH the declared constant and the fingerprint -- MEASURED, not
    # assumed (review S6): `_open_goal_fraction` reads GOAL_WIDTH directly as the denominator at
    # `_xshot_occurrence.py:118`, and on this probe `openGoal` is 0.996352 (not a saturated 1.0),
    # so it shifts to 0.996439. `match="constant"` still pins the CONSTANTS branch because
    # verify_feature_contract compares constants FIRST and raises there.
    monkeypatch.setattr(xs_mod, "GOAL_WIDTH", 7.5)
    with pytest.raises(Exception, match="constant"):
        XShotOccurrenceModel.load(out)
```

> An earlier draft of this test claimed patching `GOAL_WIDTH` "isolates the constants prong" because `GOAL_Y_MIN`/
> `GOAL_Y_MAX` are computed at import. Half true, and the half that's false is the one that matters: `:118` also
> reads `GOAL_WIDTH` directly. The isolation property this test was reaching for is already covered where it
> belongs — Task 4's `test_constant_change_alone_raises_even_when_the_fingerprint_matches`, at the unit level with
> a synthetic contract. This integration test's job is narrower: prove the wiring reaches `load()`.

> **Why not `penalty_area_half_width` here?** An earlier draft had xS declaring it. That would have been a
> constant xS does not consume: flipping the canonical value would raise on every xS load while xS's
> features were provably unchanged. A guard that fires when nothing moved is how teams learn to pass
> `legacy_override` reflexively — §3.1's "must be load-bearing on the probe" rule exists to prevent exactly this,
> and it cuts both ways (declare nothing the probe cannot move; declare everything it can).

- [ ] **Step 2: Run, verify RED** — `KeyError: 'feature_contract'`

- [ ] **Step 3: Implement in `_xshot_occurrence.py`**

In `save()`, beside the existing `"geometry_version"` entry, add:

```python
            "feature_contract": feature_contract(
                lambda: extract_xshot_features(
                    contract_probe_frame(), gk_team_id="B", goal_x=105.0
                ).iloc[0].to_numpy(dtype=float),
                constants={"goal_width": GOAL_WIDTH},
            ),
```

In `load()`, after the existing pitch-dims / geometry_version block:

```python
        verify_feature_contract(
            feature_contract(
                lambda: extract_xshot_features(
                    contract_probe_frame(), gk_team_id="B", goal_x=105.0
                ).iloc[0].to_numpy(dtype=float),
                constants={"goal_width": GOAL_WIDTH},
            ),
            meta.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="xShotOccurrence",
            error_cls=IntegrityError,
        )
```

Import `contract_probe_frame`, `feature_contract`, `verify_feature_contract` from `._feature_contract`.

- [ ] **Step 4: Mirror it in `_xcross_attempt.py`**

Identical shape, with the xCross extractor closure and `model_name="xCrossAttempt"`:

```python
lambda: extract_xcross_features(
    contract_probe_frame(), gk_team_id="B", goal_x=105.0,
    carrier_player_id="A2", score_differential=1.0,
).iloc[0].to_numpy(dtype=float)
```

xCross declares **all three** geometry constants it actually consumes:

```python
constants={
    "penalty_area_half_width": _BOX_HALF_WIDTH_M,
    "penalty_area_depth": _BOX_DEPTH_M,
    "goal_width": _GOAL_HALF_WIDTH_M * 2.0,
}
```

(The module aliases, per Task 2 Step 6 — declaring `_spadlconfig.*` directly would read a different binding than
the one `:209` actually evaluates.) The xCross test mirrors the xS one, patching the alias:
`monkeypatch.setattr(xc_mod, "_BOX_HALF_WIDTH_M", 20.15)` then `pytest.raises(..., match="constant")`.

All three are load-bearing on the probe: the box pair drives `box_off_def_ratio` (`:209-214`) and the goal width
drives the post distances (`:203-204`). xCross reads the canonical box constants as of Task 2; `_GOAL_HALF_WIDTH_M`
(`:71`) stays module-local this cycle — unifying goal width is out of scope (§7), but declaring it is not, and
Task 10's gate is what makes that distinction enforceable rather than aspirational.

- [ ] **Step 5: Run**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py tests/tracking/ -k "xshot or xcross" -q`
Expected: pass. **The bundled weights will warn at this point in the plan** — they have no contract until Task 11
stamps them, and D2 says a contract-less artifact loads with a warning. That is correct here and **temporary**:
after Task 11 no bundled load warns, and Task 12 escalates the category so it can never silently return.

---

## Task 6: Wire ghost + add its missing pitch-dims guard (spec §3.4, R1 m1)

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py`; Test: `tests/tracking/test_feature_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ghost_records_pitch_dims_and_a_contract(tmp_path, monkeypatch):
    """Mutate the LIBRARY, not the artifact -- see the P2 note on the xS test above."""
    import json

    from silly_kicks.tracking import _geometry as _geo
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    m = GhostGkModel.from_variant("default")
    out = tmp_path / "ghost"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["pitch_length"] == 105.0 and meta["pitch_width"] == 68.0
    assert meta["feature_contract"]["constants"]["penalty_area_half_width"] == 20.15  # D3: still 40.3/2
    assert meta["feature_contract"]["constants"]["penalty_area_depth"] == 16.5

    GhostGkModel.load(out)

    monkeypatch.setattr(_geo, "PITCH_LENGTH", 100.0)
    with pytest.raises(Exception, match="[Pp]itch"):
        GhostGkModel.load(out)
```

- [ ] **Step 2: Run, verify RED** — `KeyError: 'pitch_length'`

- [ ] **Step 3: Implement**

In ghost's `save()` add `"pitch_length": _geo.PITCH_LENGTH`, `"pitch_width": _geo.PITCH_WIDTH`, and the
`feature_contract` entry. Ghost declares its **current** constant, which is still 40.3-derived per D3:

```python
            "feature_contract": feature_contract(
                lambda: extract_ghost_gk_features(
                    contract_probe_frame(), gk_team_id="B", goal_x=105.0, score_diff=1, phase=0,
                    ball_carrier_team_id="A", prev_defensive_line_x=90.0,
                    prev_defending_centroid_x=94.0, dt=0.04,
                )[list(GHOST_GK_FEATURE_NAMES)].iloc[0].to_numpy(dtype=float),
                constants={
                    "penalty_area_half_width": (_PENALTY_AREA_Y_MAX - _PENALTY_AREA_Y_MIN) / 2.0,
                    "penalty_area_depth": _PENALTY_AREA_X,
                },
            ),
```

Both are load-bearing on the probe: `attackers_in_box` (`:608`) tests depth via `_PENALTY_AREA_X` and width via the
`_PENALTY_AREA_Y_MIN/MAX` pair, and it is one of the 26 `GHOST_GK_FEATURE_NAMES`. Ghost's half-width evaluates to
**20.15**, not the canonical 20.16 — that divergence is D3's whole point, and recording it is what makes the
eventual re-fit a checkable event instead of a silent one.

In `load()` add the same fail-closed pitch-dims block xS has (`_xshot_occurrence.py:507-513`) — additive, guarded
by `rec_len is not None` so pre-contract artifacts are unaffected — then the `verify_feature_contract` call with
`model_name="GhostGk"` and ghost's own `IntegrityError`.

- [ ] **Step 4: Run** — `.venv/Scripts/python.exe -m pytest tests/tracking/test_feature_contract.py -q`

---


## Task 7: `train_ghost_gk.py` cache — the B3 blocker (spec §5.1a)

**Files:** Modify `scripts/train_ghost_gk.py`; Test: `tests/scripts/test_trainer_cache_and_providers.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_trainer_cache_and_providers.py
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ghost_cache_token_is_derived_from_the_geometry_constants():
    """R2 N4: a hand-bumped literal goes stale INSIDE the re-fit cycle it protects (extract,
    flip the constant, re-run -> the second run reuses the first run's 40.3 features). Deriving
    it from the constants auto-invalidates on the flip with zero discipline."""
    import silly_kicks.tracking._ghost_gk as gg

    t = _load("train_ghost_gk")
    before = t.cache_token()
    original = gg._PENALTY_AREA_Y_MIN
    try:
        gg._PENALTY_AREA_Y_MIN = (68.0 - 40.32) / 2.0
        assert t.cache_token() != before, "token must change when the box constant changes"
    finally:
        gg._PENALTY_AREA_Y_MIN = original
```

- [ ] **Step 2: Run, verify RED** — `AttributeError: module has no attribute 'cache_token'`

- [ ] **Step 3: Implement**

Add to `scripts/train_ghost_gk.py`:

```python
def cache_token() -> str:
    """Feature-cache identity. DERIVED from the geometry constants (spec §5.1a / review R2 N4) so
    that flipping the penalty-area constant auto-invalidates the cache. A hand-bumped literal
    would go stale inside the very re-fit cycle it exists to protect."""
    import silly_kicks.tracking._ghost_gk as gg

    return f"v3-box{gg._PENALTY_AREA_Y_MIN:.4f}-{gg._PENALTY_AREA_Y_MAX:.4f}"
```

Write it into the cache directory on save (a `cache_token.txt` beside `features.parquet`), and extend the hit
predicate at `:256-262` to require the recorded token to equal `cache_token()`; a missing or differing token is a
MISS. Update the `:254-255` comment to say the schema-version bump has now landed.

- [ ] **Step 4: Run** — `.venv/Scripts/python.exe -m pytest tests/scripts/test_trainer_cache_and_providers.py -q`

---

## Task 8: xS/xCross trainers → live corpus fingerprint (spec §5.1)

**Files:** Modify `scripts/train_xshot_occurrence.py:37`, `scripts/train_xcross_attempt.py:38`; Test: same file

- [ ] **Step 1: Write the failing test**

```python
def test_corpus_fingerprint_distinguishes_corpora():
    """The whole point: a changed corpus must MISS. The constant 'schema-v2' token cannot."""
    # NOTE the import form: `scripts/` has NO __init__.py, so `from scripts._cache import ...`
    # is a ModuleNotFoundError. This is the established idiom (tests/scripts/test_cache_schema.py:7).
    from _cache import corpus_fingerprint

    a = corpus_fingerprint([("gs", "1", "public"), ("gs", "2", "public")])
    b = corpus_fingerprint([("gs", "1", "public")])
    c = corpus_fingerprint([("gs", "2", "public"), ("gs", "1", "public")])  # order-insensitive
    assert a != b
    assert a == c


def test_trainers_no_longer_gate_on_a_constant_token():
    for name in ("train_xshot_occurrence", "train_xcross_attempt"):
        src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert '"schema-v2"' not in src, f"{name} still gates on the constant token"
        assert "corpus_fingerprint" in src, f"{name} must build a live fingerprint"
```

- [ ] **Step 2: Run, verify RED — for the STATED reason**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_trainer_cache_and_providers.py -q`
Expected: `test_trainers_no_longer_gate_on_a_constant_token` **FAILS on the `"schema-v2"` assertion** for both
trainers — an `AssertionError`, not a collection or import error. If you get `ModuleNotFoundError`, the import
form is wrong (see the note in Step 1) and the RED is meaningless: a "verify RED" whose stated reason is not the
actual reason is a false gate (review P3). The same check applies to Tasks 8 and 10, which use the same `_load()`
helper — confirm each fails on its named assertion, not on import.

- [ ] **Step 3: Implement**

In each trainer, replace `_CACHE_FINGERPRINT = "schema-v2"` with a call to `corpus_fingerprint(rows)` built from the
sorted `(provider, match_id, visibility)` triples of the loaded corpus, passed to `write_cache_meta` /
`cache_is_valid`. Delete the "Deferred (ADR-038)" note in `scripts/_cache.py:22`.

- [ ] **Step 4: Run** — same command as Step 1.

---

## Task 9: Ghost trainer startup fail-fast + shared `validate_provider` (spec §5.2)

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py`, `scripts/train_ghost_gk.py`; Test: same file

- [ ] **Step 1: Write the failing test**

```python
def test_unclassified_provider_fails_BEFORE_any_fitting(monkeypatch):
    """Spy the fit: without this the test passes on the pre-existing mid-run raise and proves
    nothing (spec §6)."""
    import silly_kicks.tracking._ghost_gk as gg

    calls = {"n": 0}
    monkeypatch.setattr(gg.GhostGkModel, "fit", lambda self, *a, **k: calls.__setitem__("n", 1))

    t = _load("train_ghost_gk")
    with pytest.raises(ValueError, match="provider"):
        t.validate_corpus_providers(["gradientsports", "not_a_provider"])
    assert calls["n"] == 0


def test_validate_provider_is_shared_not_duplicated():
    """R2 m4: two copies of the membership set drift when a provider is added."""
    import silly_kicks.tracking._ghost_gk as gg

    assert callable(gg.validate_provider)
    gg.validate_provider("gradientsports")
    with pytest.raises(ValueError, match="provider"):
        gg.validate_provider("nope")
```

- [ ] **Step 2: Run, verify RED** — `AttributeError: validate_provider`

- [ ] **Step 3: Implement**

In `_ghost_gk.py`, extract the membership check that currently lives inline in `keeper_detection_mask` (`:257-262`)
into:

```python
def validate_provider(provider: str) -> None:
    """Raise unless `provider` is classified. Single source for the membership rule -- both the
    trainer's startup check and `keeper_detection_mask` call this (review R2 m4)."""
    if provider not in _DETECTION_AWARE_PROVIDERS | _FULLY_OBSERVED_PROVIDERS:
        raise ValueError(
            f"unclassified provider {provider!r}: add it to _DETECTION_AWARE_PROVIDERS or "
            f"_FULLY_OBSERVED_PROVIDERS. Known: "
            f"{sorted(_DETECTION_AWARE_PROVIDERS | _FULLY_OBSERVED_PROVIDERS)}"
        )
```

Have `keeper_detection_mask` call it (behaviour-preserving). In `scripts/train_ghost_gk.py` add
`validate_corpus_providers(providers)` that loops `validate_provider`, and call it in `main()` **before** any
loading or fitting.

- [ ] **Step 4: Run** — `.venv/Scripts/python.exe -m pytest tests/scripts/test_trainer_cache_and_providers.py -q`

---

## Task 10: The M1 auto-enumeration gate (spec §3.1)

**Files:** Create `tests/tracking/test_geometry_constant_enumeration.py`

- [ ] **Step 1: Write the test (RED until the exemption list is written)**

```python
# tests/tracking/test_geometry_constant_enumeration.py
"""M1: completeness by ENUMERATION, the ADR-043 idiom -- not by remembering to declare."""
import ast
from pathlib import Path

SK = Path(__file__).resolve().parents[2] / "silly_kicks"

_MODULES = [
    SK / "tracking" / "_ghost_gk.py",
    SK / "tracking" / "_xshot_occurrence.py",
    SK / "tracking" / "_xcross_attempt.py",
    SK / "tracking" / "defensive_credit" / "_params.py",
]

_GEOMETRY_NAME = ("PENALTY", "BOX", "GOAL", "PITCH", "FIELD", "AREA")

#: Module-level geometry constants deliberately NOT in any contract, each with a reason.
#: Adding a name here is a visible code-review decision -- the forcing function.
#:
#: THE RULE for a DERIVED constant (review S5), because the two cases look identical otherwise:
#:   * derived from a DECLARED constant  -> map it to that constant's key in
#:     DECLARED_CONSTANT_SOURCES. (GOAL_Y_MIN/GOAL_Y_MAX = GOAL_Y_CENTRE -/+ GOAL_WIDTH/2, so
#:     they move iff goal_width moves -- declared.)
#:   * derived from PITCH DIMENSIONS    -> exempt here. (GOAL_Y_CENTRE/_GOAL_Y/_GOAL_Y_C are all
#:     just PITCH_WIDTH/2; they are already covered by the pitch_length/pitch_width fail-closed
#:     guard, so declaring them would double-count one quantity under two names.)
#: Without this written down, the next person extending either list follows whichever precedent
#: they happen to read first -- and both precedents are present below.
#:
#: MEASURED while writing this plan: the enumerator finds 14 constants across the four modules.
#: Five are penalty-area (declared), four are goal-MOUTH (declared -- they drive `openGoal`, so a
#: goal-width change would skew xS exactly as a box change skews ghost), and five are pitch
#: dimensions or values derived from them (exempt below).
_EXEMPT = {
    "_FIELD_LENGTH": "pitch dimension, covered by the pitch_length/pitch_width fail-closed guard",
    "_FIELD_WIDTH": "pitch dimension, covered by the pitch_length/pitch_width fail-closed guard",
    "_GOAL_Y_C": "goal centre y, derived as _FIELD_WIDTH/2; no independent value",
    "_GOAL_Y": "goal centre y, derived as _FIELD_WIDTH/2; no independent value",
    "GOAL_Y_CENTRE": "goal centre y, derived as PITCH_WIDTH/2; no independent value",
}


def _module_level_geometry_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else node.targets if isinstance(node, ast.Assign)
            else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and any(k in t.id.upper() for k in _GEOMETRY_NAME):
                out.add(t.id)
    return out


def test_every_geometry_constant_is_declared_or_explicitly_exempt():
    from silly_kicks.tracking._feature_contract import DECLARED_CONSTANT_SOURCES

    declared = set(DECLARED_CONSTANT_SOURCES)
    undeclared = {}
    for path in _MODULES:
        for name in _module_level_geometry_constants(path):
            if name not in declared and name not in _EXEMPT:
                undeclared.setdefault(path.name, []).append(name)
    assert not undeclared, (
        f"undeclared geometry constants {undeclared}. Either declare the constant in the owning "
        f"model's feature contract (and extend contract_probe_frame so it is load-bearing), or "
        f"add it to _EXEMPT with a reason."
    )


def test_the_enumerator_is_not_vacuous():
    """A regex/AST gate that finds nothing passes silently forever."""
    found = set().union(*(_module_level_geometry_constants(p) for p in _MODULES))
    assert len(found) >= 4, f"enumerator found only {found}; it is not seeing the modules"


def _built_contract_constants(tmp_path) -> dict[str, set[str]]:
    """The constants each model ACTUALLY stamps, read back from a real save()."""
    import json

    from silly_kicks.tracking._ghost_gk import GhostGkModel
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    out = {}
    for name, cls in (
        ("xshot", XShotOccurrenceModel),
        ("xcross", XCrossAttemptModel),
        ("ghost", GhostGkModel),
    ):
        d = tmp_path / name
        cls.from_variant("default").save(d)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        out[name] = set(meta["feature_contract"]["constants"])
    return out


def test_the_registry_and_the_built_contracts_agree(tmp_path):
    """Close the loop BOTH ways (review P4).

    The registry is a name->key map; on its own it proves nothing about what save() stamps. Without
    this test a constant could be listed as "declared" while no contract carried it -- the gate
    would read as complete and enforce nothing, the exact failure ADR-043 retired the id-compat
    lint for. Reading the built artifacts is the only evidence that the declaration is real.
    """
    from silly_kicks.tracking._feature_contract import DECLARED_CONSTANT_SOURCES

    built = _built_contract_constants(tmp_path)
    all_keys = set().union(*built.values())
    registry_keys = set(DECLARED_CONSTANT_SOURCES.values())

    # (a) every key the registry claims is declared is stamped by at least one model
    assert registry_keys <= all_keys, (
        f"registry names keys no model stamps: {sorted(registry_keys - all_keys)}. Either a model "
        f"must declare it, or the source constant belongs in the test's _EXEMPT list."
    )
    # (b) no model stamps a key the registry does not know about
    assert all_keys <= registry_keys, (
        f"models stamp undeclared keys: {sorted(all_keys - registry_keys)}. Add the owning "
        f"module constant to DECLARED_CONSTANT_SOURCES so the enumeration gate can see it."
    )


def test_every_declared_constant_is_load_bearing_on_the_probe(tmp_path):
    """SS3.1's rule, enforced rather than trusted: perturbing a declared constant must move the
    probe fingerprint OR the recorded value. A declaration the probe cannot move is a guard that
    fires when nothing changed -- which is how `legacy_override` becomes reflex."""
    built = _built_contract_constants(tmp_path)
    assert built["xshot"] == {"goal_width"}
    assert built["xcross"] == {"penalty_area_half_width", "penalty_area_depth", "goal_width"}
    assert built["ghost"] == {"penalty_area_half_width", "penalty_area_depth"}
```

Pinning the exact per-model sets is deliberate: it is the one place a reviewer can see, in a single screen, that
xS declares no penalty-area constant (it has none — verified, `_xshot_occurrence.py` contains no such constant)
and that ghost's pair is the 40.3-derived one D3 froze.

- [ ] **Step 2: Run, verify RED** — `ImportError: DECLARED_CONSTANT_SOURCES`

- [ ] **Step 3: Implement**

In `_feature_contract.py` add the registry the gate reads, mapping the module-level constant name to the contract
key it is declared under:

```python
#: Module-level geometry constants that ARE declared in some model's feature contract.
#: The enumeration gate (tests/tracking/test_geometry_constant_enumeration.py) requires every
#: geometry-named module constant to appear here or in that test's _EXEMPT list, with a reason.
DECLARED_CONSTANT_SOURCES: dict[str, str] = {
    # penalty area
    "_BOX_HALF_WIDTH_M": "penalty_area_half_width",
    "_BOX_DEPTH_M": "penalty_area_depth",
    "_PENALTY_AREA_X": "penalty_area_depth",
    "_PENALTY_AREA_Y_MIN": "penalty_area_half_width",
    "_PENALTY_AREA_Y_MAX": "penalty_area_half_width",
    # goal mouth -- these drive `openGoal`, so a goal-width change skews xS exactly the way a
    # box change skews ghost. Same class, same treatment.
    "GOAL_WIDTH": "goal_width",
    "GOAL_Y_MIN": "goal_width",
    "GOAL_Y_MAX": "goal_width",
    "_GOAL_HALF_WIDTH_M": "goal_width",
}
```

Two things about this registry that are easy to get wrong:

**It is keyed by BARE constant name, not module-qualified.** `_BOX_HALF_WIDTH_M` and `_BOX_DEPTH_M` exist in *both*
`_xcross_attempt.py` and `defensive_credit/_params.py`. One entry covers both, which is correct here — they are
the same quantity and, after Task 2, the same canonical source. If a future module ever gives one of these names a
different meaning, the registry must become module-qualified; it is not today, and pretending otherwise would be
the kind of unstated assumption this whole cycle exists to remove.

**`defensive_credit/_params.py` has no artifact**, so its constants can never be "stamped by a model". They pass
the P4(a) check only because xCross declares the same keys. That is load-bearing coverage, not a loophole — the
canonical constant is what both read — but it does mean a box-constant change is caught via xCross's contract, not
via defensive_credit's own (it has none, and does not need one: it is not a trained model).

Per-model declarations, cross-checked by `test_every_declared_constant_is_load_bearing_on_the_probe`:

| model | declares | why |
|---|---|---|
| xS | `goal_width` | only geometry constant it consumes; drives `openGoal` |
| xCross | `penalty_area_half_width`, `penalty_area_depth`, `goal_width` | box ratio + post distances |
| ghost | `penalty_area_half_width`, `penalty_area_depth` | `attackers_in_box`; values are D3's frozen 40.3 pair |

- [ ] **Step 4: Run, and RESOLVE what it finds**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_geometry_constant_enumeration.py -q`
If it names a constant not in either list, that is the gate doing its job: either declare it (and extend the probe
so it is load-bearing, per §3.1) or exempt it with a reason. **Do not widen `_GEOMETRY_NAME` to make it pass.**

---

## Task 11: Stamp contracts into ALL THREE bundled artifacts (spec §4 amended, R1 M5) — **x86 only**

**Files:** Modify `metadata.json` + `SHA256SUMS` under `_ghost_gk_weights/default/`, `_xshot_weights/default/`,
`_xcross_weights/default/` (written directly — see below)

> ### This task CLOSES spec §4's gap rather than counting it — read this before implementing
>
> Spec §4 recorded a "known and accepted gap": ghost gets a fingerprint, xS and xCross do not, so their loads take
> the D2 warn path until their next training run. Review S1 then proposed *counting* that gap by escalating the
> warning in CI and adding ~14 module-level opt-outs as a self-retiring ledger.
>
> **Both were working around an assumption that does not hold.** VERIFIED: all three bundled artifacts have the
> identical structure — `metadata.json` + `SHA256SUMS` + a weights file — and xS/xCross metadata already carries
> `chirality`, `geometry_version`, `pitch_length`, `pitch_width`, exactly the neighbours `feature_contract` sits
> beside. The metadata-only migration below works **verbatim** for all three. So we stamp all three, and Task 12's
> escalation then needs an **empty** opt-out list: fail-closed from day one, with no ledger to maintain and nothing
> to retire.
>
> **What the stamp attests, stated precisely so it is not over-read.** It records what the *current* library's
> extractor produces on the probe. Its guarantee is **forward-looking**: from this point, any change to an
> extractor or a declared constant raises. It does **not** retroactively prove these are the features each model
> was trained on. That limit is not new — it is exactly what spec §4 already accepted for ghost, on the
> load→save-migration path `_ghost_gk.py:1859-1860` documents and the 4.54.0 parameters-only migration used. The
> supporting evidence is the same for all three: `load()` fail-closed chirality verification passes (ADR-040), and
> chirality is model output on a fixed frame, which flows through the extractor — evidence of stability, though not
> proof for a feature carrying near-zero weight.
>
> **Irreducible residual, unchanged:** the HF `full` (ghost) and `sc_extended` (xS/xCross) variants cannot be
> re-uploaded under the standing owner hold, so they keep warning on load. That is correct — they genuinely carry
> no contract — and spec §4 already records the trigger for when the hold lifts.

> **This task is LAST on purpose (review P11).** It stamps declared constants into shipped artifacts, so it must run
> after Task 10's enumeration gate has had its chance to force an additional declaration. Running it at position 7 —
> where an earlier draft had it — could bake an under-declared contract that `load()` would then find perfectly
> self-consistent.

> **Run this on x86 (this dev machine), NOT the DGX** (spec §3 N3): these are the only fingerprinted artifacts
> shipping this cycle, and keeping them x86-produced means no cross-platform comparison happens against the
> not-yet-measured `atol`. **Do not touch the HF repo** — standing owner hold (spec §4).

> **DO NOT call any model's `save()` (review P1).** Ghost's `save()` unconditionally rewrites the weights
> (`_ghost_gk.py:1839-1840`: `np.savez_compressed(str(npz_path), **save_dict)`) and then regenerates
> `SHA256SUMS` (`:1875-1877`). `np.savez_compressed` writes a fresh ZIP whose members carry mtimes, so **the npz
> bytes differ even when every array is bit-identical.** The xS/xCross `save()` paths likewise re-serialize
> `model.json` through xgboost. Using `save()` would therefore (a) break the "byte delta is metadata-only" promise,
> and (b) trip this task's own verification step on a *correct* run. This is a **metadata-only migration**, written
> directly, for all three.

- [ ] **Step 1: Write the migration script**

Three artifacts, one shared writer. Create `scripts/stamp_feature_contracts.py` rather than three inline
heredocs — this is run once now and again at the next re-save, and a committed script is reviewable where a
shell one-liner is not.

```python
"""One-off: stamp feature contracts into the three bundled artifacts (metadata-only).

Deliberately does NOT call any model's save() -- see the plan's Task 11 preamble. Writes
metadata.json with the exact writer save() uses, then recomputes SHA256SUMS the same way.
"""

import hashlib
import json
from pathlib import Path

import silly_kicks.tracking._ghost_gk as gg
import silly_kicks.tracking._xcross_attempt as xc
import silly_kicks.tracking._xshot_occurrence as xs
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._feature_contract import contract_probe_frame, feature_contract

ROOT = Path(__file__).resolve().parents[1] / "silly_kicks" / "tracking"


def _ghost_vec():
    return (
        gg.extract_ghost_gk_features(
            contract_probe_frame(), gk_team_id="B", goal_x=105.0, score_diff=1, phase=0,
            ball_carrier_team_id="A", prev_defensive_line_x=90.0,
            prev_defending_centroid_x=94.0, dt=0.04,
        )[list(gg.GHOST_GK_FEATURE_NAMES)]
        .iloc[0]
        .to_numpy(dtype=float)
    )


def _xs_vec():
    return (
        xs.extract_xshot_features(contract_probe_frame(), gk_team_id="B", goal_x=105.0)
        .iloc[0]
        .to_numpy(dtype=float)
    )


def _xc_vec():
    return (
        xc.extract_xcross_features(
            contract_probe_frame(), gk_team_id="B", goal_x=105.0,
            carrier_player_id="A2", score_differential=1.0,
        )
        .iloc[0]
        .to_numpy(dtype=float)
    )


#: (directory, weights filename, contract builder, extra metadata to inject)
TARGETS = [
    (
        ROOT / "_ghost_gk_weights" / "default",
        "rfcde_weights.npz",
        lambda: feature_contract(
            _ghost_vec,
            constants={
                "penalty_area_half_width": (gg._PENALTY_AREA_Y_MAX - gg._PENALTY_AREA_Y_MIN) / 2.0,
                "penalty_area_depth": float(gg._PENALTY_AREA_X),
            },
        ),
        # ghost alone lacks pitch dims today (spec §3.4 / R1 m1); xS and xCross already record them
        {"pitch_length": _geo.PITCH_LENGTH, "pitch_width": _geo.PITCH_WIDTH},
    ),
    (
        ROOT / "_xshot_weights" / "default",
        "model.json",
        lambda: feature_contract(_xs_vec, constants={"goal_width": xs.GOAL_WIDTH}),
        {},
    ),
    (
        ROOT / "_xcross_weights" / "default",
        "model.json",
        lambda: feature_contract(
            _xc_vec,
            constants={
                "penalty_area_half_width": xc._BOX_HALF_WIDTH_M,
                "penalty_area_depth": xc._BOX_DEPTH_M,
                "goal_width": xc._GOAL_HALF_WIDTH_M * 2.0,
            },
        ),
        {},
    ),
]


def main() -> None:
    for path, weights_name, build, extra in TARGETS:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        meta.update(extra)
        meta["feature_contract"] = build()

        # Byte-for-byte the writer save() uses (_ghost_gk.py:1870-1872): LF endings, indent=2.
        with open(path / "metadata.json", "w", newline="\n", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Recompute sums exactly as save() does (:1875-1882), incl. the CRLF->LF normalisation.
        # NOTE the ORDER and the file list must match what that model's load() expects; read each
        # model's own save() rather than assuming this two-entry shape generalises.
        names = [weights_name, "metadata.json"]
        if (path / "metrics.json").exists():
            names.append("metrics.json")
        with open(path / "SHA256SUMS", "w", newline="\n", encoding="utf-8") as f:
            for fname in names:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")
        print(f"stamped {path.name}: {sorted(meta['feature_contract']['constants'])}")


if __name__ == "__main__":
    main()
```

> **Before running it, read each model's `save()` and confirm the SHA256SUMS file list and order.** The script
> above infers them (`metrics.json` appended when present), and an inferred list that disagrees with what `load()`
> verifies produces an `IntegrityError` on the very next load — loud, but avoidable by looking first. This is the
> one place the three models might genuinely differ.

- [ ] **Step 2: Run it, then verify NO weights file changed**

```bash
.venv/Scripts/python.exe scripts/stamp_feature_contracts.py
git -C "D:/Development/karstenskyt__silly-kicks" status --short silly_kicks/tracking/
```
Expected: **exactly six files modified — `metadata.json` + `SHA256SUMS` in each of the three directories.**
`rfcde_weights.npz` and both `model.json` files must NOT be listed. If any weights file appears, STOP: something
called `save()` and this is no longer a migration.

Then verify the payloads are genuinely identical, which is stronger than a byte check (review P1):

```bash
.venv/Scripts/python.exe -c "
import io, json, subprocess
import numpy as np

def head(p):
    return subprocess.run(['git','show',f'HEAD:{p}'], capture_output=True, check=True).stdout

base = 'silly_kicks/tracking'
old = np.load(io.BytesIO(head(f'{base}/_ghost_gk_weights/default/rfcde_weights.npz')))
new = np.load(f'{base}/_ghost_gk_weights/default/rfcde_weights.npz')
assert sorted(old.files) == sorted(new.files), 'ghost key set changed'
for k in old.files:
    assert np.array_equal(old[k], new[k]), f'ARRAY CHANGED: {k}'
print('ghost:', len(old.files), 'arrays bit-identical')

for name in ['_xshot_weights', '_xcross_weights']:
    p = f'{base}/{name}/default/model.json'
    assert json.loads(head(p)) == json.loads(open(p, encoding='utf-8').read()), f'{name} CHANGED'
    print(f'{name}: booster JSON identical')
"
```

- [ ] **Step 2b: Prove every artifact now loads clean and carries a contract**

```python
@pytest.mark.parametrize(
    ("cls_path", "n_constants"),
    [
        ("silly_kicks.tracking._ghost_gk:GhostGkModel", 2),
        ("silly_kicks.tracking._xshot_occurrence:XShotOccurrenceModel", 1),
        ("silly_kicks.tracking._xcross_attempt:XCrossAttemptModel", 3),
    ],
)
def test_every_bundled_artifact_carries_a_verified_contract(cls_path, n_constants, recwarn):
    """The whole point of stamping all three: after this task NO bundled load warns, so Task 12
    can escalate with an EMPTY opt-out list. Asserted per-model, and asserted as ABSENCE of the
    warning -- a contract that exists but fails to verify would warn, not raise, on some paths."""
    import importlib

    from silly_kicks.tracking import MissingFeatureContractWarning

    mod_name, cls_name = cls_path.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)

    m = cls.from_variant("default")
    assert not [w for w in recwarn if issubclass(w.category, MissingFeatureContractWarning)]
    assert m is not None
    # and the stamped constants are the pinned per-model set from Task 10
    import json
    from pathlib import Path

    d = {
        "GhostGkModel": "_ghost_gk_weights",
        "XShotOccurrenceModel": "_xshot_weights",
        "XCrossAttemptModel": "_xcross_weights",
    }[cls_name]
    root = Path(importlib.import_module("silly_kicks.tracking").__file__).parent
    meta = json.loads((root / d / "default" / "metadata.json").read_text(encoding="utf-8"))
    assert len(meta["feature_contract"]["constants"]) == n_constants
```

- [ ] **Step 3: Prove the pin now has teeth**

```python
def test_ghost_pin_is_enforced_by_a_raise_not_by_prose(monkeypatch):
    """R1 M5: flipping the constant without re-fitting must RAISE, because that is the skew D3
    sequences around. Before this artifact carried a contract, only a docstring said so."""
    import silly_kicks.tracking._ghost_gk as gg
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    monkeypatch.setattr(gg, "_PENALTY_AREA_Y_MIN", (68.0 - 40.32) / 2.0)
    monkeypatch.setattr(gg, "_PENALTY_AREA_Y_MAX", (68.0 + 40.32) / 2.0)
    with pytest.raises(Exception, match="constant|feature contract"):
        GhostGkModel.from_variant("default")
```

(No priming `from_variant` call first — verified that ghost has **no** `_VARIANT_CACHE`, unlike xS `:307` and
xCross `:79`, so the binding would be dead code and a ruff F841. Review P10.)

**The xS/xCross equivalents need the cache cleared**, precisely because they *do* have `_VARIANT_CACHE`
(`_xshot_occurrence.py:307`, `_xcross_attempt.py:79`): a `from_variant("default")` earlier in the session returns
the cached instance without re-running `load()`, so the raise would never fire and the test would pass vacuously.
Clear it in the test:

```python
def test_xs_pin_is_enforced_by_a_raise(monkeypatch):
    import silly_kicks.tracking._xshot_occurrence as xs_mod
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    monkeypatch.setattr(xs_mod, "_VARIANT_CACHE", {})  # else a cached instance skips load()
    monkeypatch.setattr(xs_mod, "GOAL_WIDTH", 7.5)
    with pytest.raises(Exception, match="constant|feature contract"):
        XShotOccurrenceModel.from_variant("default")
```

Mirror it for xCross with `_BOX_HALF_WIDTH_M`. **Confirm the cache attribute's real name during implementation** —
the line anchors are recorded but the identifier is not; if it differs, patch the right one rather than adding a
`monkeypatch.setattr` that silently creates a new attribute and leaves the real cache live.

- [ ] **Step 4: Run the three model suites**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/ -k "ghost or xshot or xcross" -q
.venv/Scripts/python.exe -m pytest tests/tracking/test_weights_bundle_golden.py -q
```

The golden bundle test is called out separately because it is the one most likely to pin metadata: if it hashes
`metadata.json` or asserts an exact key set, it legitimately needs updating for the added `feature_contract` key —
and that update is a **deliberate fixture change**, not a workaround. Read it before editing.

---

## Task 12: Escalate `MissingFeatureContractWarning` in CI (review S1, extended)

**Files:** Modify `pyproject.toml`; test in `tests/tracking/test_feature_contract.py`.

> **Expected opt-out list: EMPTY.** Review S1 proposed escalating plus ~14 module-level opt-outs forming a
> self-retiring ledger. Task 11 now stamps all three bundled artifacts, so there is nothing left to opt out of and
> nothing to retire — the escalation is fail-closed from day one. If Step 1's measurement disagrees, that is a real
> finding about an artifact path Task 11 missed, and Step 2 says what to do about it.

**Why escalate at all.** §3.3's named category is the instrument that keeps a contract-less artifact from being
invisible in a batch serve — and without this task **nothing uses it**. A category nobody escalates is a category
that only documents; the guard would exist and never fire.

This repo already runs exactly this mechanism. `pyproject.toml:242-245`:

```toml
filterwarnings = [
    "error::silly_kicks.tracking.SyntheticEPVWarning",
    "error::silly_kicks.tracking.IgnoredSurfaceInputsWarning",
]
```

with the comment above it stating the pattern outright — *"that opt-out list IS the inventory of synthetic-EPV call
sites."* We adopt the mechanism and skip the inventory, because Task 11 leaves nothing to inventory.

It is *only* safe because of P5 — escalating this category must not also catch
`UnverifiableFeatureContractWarning`, which is precisely the split Task 1 makes. Without that split, escalating
here would turn every future probe extension into a hard failure across every not-yet-re-saved artifact, which is
the outcome §3.2 deliberately designed against. P5 pays off here concretely rather than theoretically.

- [ ] **Step 1: Add the escalation**

In `pyproject.toml`, add a third entry and extend the comment above it:

```toml
filterwarnings = [
    "error::silly_kicks.tracking.SyntheticEPVWarning",
    "error::silly_kicks.tracking.IgnoredSurfaceInputsWarning",
    "error::silly_kicks.tracking.MissingFeatureContractWarning",
]
```

Comment to add above the list, in the style of the existing one:

> A bundled artifact with no feature contract cannot be verified at all, so it must fail CI rather than warn.
> All three bundled artifacts are stamped, so this list needs **no** opt-outs — an opt-out appearing here later
> means an artifact shipped un-stamped, which is the thing to fix rather than annotate.
> `UnverifiableFeatureContractWarning` is deliberately NOT escalated: a probe change must stay a warning, or
> extending the probe would brick every not-yet-re-saved artifact (§3.2).

- [ ] **Step 2: Run the full suite and confirm EMPTY opt-out list**

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e and not slow" -q
```

Expected: **green, with zero opt-outs added.** Task 11 stamped every bundled artifact, so no `from_variant("default")`
can emit this warning.

**If anything goes red, do NOT add an opt-out to silence it — diagnose first.** A red test here means one of:

1. **An artifact Task 11 missed.** Fix by stamping it (extend `TARGETS` in the migration script). This is the
   likely case for `_gk_completion_weights` / `_retention_weights` *only if* they were wired into the contract —
   they were not, so they cannot warn. Verify rather than assume.
2. **A network/HF path reached in a non-e2e test** — `full` (ghost) or `sc_extended` (xS/xCross) genuinely have no
   contract and cannot be re-uploaded under the standing hold. **This** is a legitimate opt-out, and it is the only
   one: scope it to the specific test, cite the hold, and give it the trigger from spec §4.
3. **A test constructing a model and calling `load()` on a hand-built artifact** — legitimate; the fixture should
   gain a contract, not an opt-out, so the test keeps exercising the real path.

The distinction matters: an opt-out for reason 2 is a recorded external constraint; an opt-out for reason 1 or 3 is
the guard being disabled in the exact situation it was built for.

- [ ] **Step 3: Prove the escalation is live, not decorative**

```python
def test_the_escalation_is_live_not_decorative():
    """Non-vacuity: if the filterwarnings line were dropped or misspelled, this whole task would be
    inert and the suite would pass identically. Assert the other side (the codebase rule: every
    band needs a test from BOTH sides)."""
    import warnings

    from silly_kicks.tracking import MissingFeatureContractWarning

    with pytest.raises(MissingFeatureContractWarning):
        warnings.warn("probe", MissingFeatureContractWarning, stacklevel=2)


def test_the_unverifiable_category_is_NOT_escalated():
    """The other half of the same decision, and the one a future 'tidy-up' would break: escalating
    both categories would make any probe extension a hard failure across every artifact.

    NOTE the bare `warnings.warn` with NO recording context. `catch_warnings(record=True)`,
    `recwarn` and `pytest.warns` all call `simplefilter("always")`, which OVERRIDES the ini
    filterwarnings config -- so any of them would make this test pass whether or not the category
    is escalated. The assertion here is simply that this line does not raise under pytest's own
    filter configuration; that is the only form with discriminating power.
    """
    import warnings

    from silly_kicks.tracking import UnverifiableFeatureContractWarning

    warnings.warn("probe", UnverifiableFeatureContractWarning, stacklevel=2)

---

## Task 13: Commit-prep — STOP for owner approval

- [ ] **Step 1: Full lint trio**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright
```
Re-run the WHOLE trio after any fix.

- [ ] **Step 2: Full suite** — `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Compare against Task 0's baseline.

- [ ] **Step 3: Merge from main; confirm registers**

`git fetch origin && git merge origin/main`. Re-confirm next-free version / PR-S / ADR against `origin/main` —
provisional `4.64.0 / PR-S135 / ADR-050`, but a parallel session may have advanced them.

- [ ] **Step 4: Version bump ×5 + docs + the spec amendment**

`pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md` release line. Write
`ADR-050-feature-contract.md`. Update `CLAUDE.md`'s tracking bullet.

**Amend spec §4's "known and accepted gap" paragraph.** It currently states that xS and xCross are not re-saved
this cycle and that their gap is the designed consequence of D2. Task 11 stamps all three, so that paragraph is now
false as written. Replace it with what shipped: all three bundled artifacts carry contracts; the escalation runs
with an empty opt-out list; the *residual* is the HF `full`/`sc_extended` variants, which cannot be re-uploaded
under the standing owner hold and therefore still take the D2 warn path. Leaving a spec paragraph that the
implementation contradicts is how the next reader learns to distrust the spec.

**Add TODO rows for the two deferred triggers:** (a) the DGX-vs-x86 `atol` measurement, due before the first
DGX-produced fingerprint; (b) the ghost re-fit, which flips the constant, migrates `_ghost_gk` onto
`in_penalty_area_goal_relative`, and must be scheduled into the already-queued lakehouse AC recompute window rather
than paying that drain twice.

**Hyrum note for the release entry:** three bundled `metadata.json` files gain a `feature_contract` key and their
`SHA256SUMS` change. No weights change and no model output changes, so this is **not** a retrain trigger — but a
consumer pinning an artifact checksum will see a diff, and that is worth one line in the changelog.

- [ ] **Step 5: C4** — expected C4-free (count stays 32). Verify by running `/c4`, do not assert it.

- [ ] **Step 6: `/final-review`**

- [ ] **Step 7: STOP.** Report and await approval. Do NOT commit, push, tag, or open a PR without it.

---

## Self-Review

**Spec coverage:** §3 contract → Tasks 3–6; §3.1 probe + teeth → Task 3, enumeration → Task 10; §3.2 probe identity
+ constants net → Task 4; §3.3 named warnings → Task 1; §3.4 ghost pitch dims → Task 6; §4 constant + predicates →
Task 2, artifact stamping → Task 11 (**amends §4** — all three, not ghost alone), §4's gap now CLOSED + escalated →
Task 12; §5.1 → Task 8; §5.1a → Task 7; §5.2 → Task 9; §6 testing → distributed; §7 triggers → Task 13 Step 4.
All D1–D3 and all R1/R2/P/S findings map to a task.

**One deliberate deviation from the spec, recorded so it is not mistaken for drift:** spec §4 says xS and xCross
are "not re-saved this cycle" and treats that as a known and accepted gap. Task 11 stamps them anyway. The spec's
gap rested on an assumption I verified false — that only ghost's artifact could take a metadata-only migration —
when all three have identical structure and xS/xCross metadata already carries `chirality`/`geometry_version`/
`pitch_length`/`pitch_width`. Closing the gap is strictly better than review S1's proposal to count it, and cheaper
(one script covering three directories, versus ~14 test-file opt-outs). **The spec §4 paragraph should be amended
at commit-prep to match** — see Task 13.

**Placeholder scan:** none — every code step carries executable code; the one judgement call (Task 10 Step 4's
"resolve what it finds") is deliberate, because the gate's output cannot be known before it runs, and the wrong
resolution (widening the pattern) is named.

**Type/name consistency:** `contract_probe_frame`, `feature_contract`, `verify_feature_contract`,
`DECLARED_CONSTANT_SOURCES`, `MissingFeatureContractWarning`, `UnverifiableFeatureContractWarning`,
`in_penalty_area_absolute`,
`in_penalty_area_goal_relative`, `validate_provider`, `cache_token`, `validate_corpus_providers` are used
identically in every task. `verify_feature_contract`'s argument order matches `verify_chirality` in both the
implementation and every call site.

**Known judgement calls left to the implementer, deliberately:** Task 5/6's exact insertion points inside `save()`
and `load()` (the surrounding metadata dicts differ per model — follow the existing `geometry_version` entry), and
Task 7's cache-token write/read plumbing (`cache_token.txt` beside the parquet is the suggested shape, not a
contract).

**Plan-review round 1 (P1–P11) is folded in.** The three blockers were each re-verified against source before
being fixed, not taken on the reviewer's word: P1 (`save()` rewrites the npz unconditionally, `:1839-1840`) →
Task 11 is now a metadata-only migration; P2 (`load()` verifies SHA256SUMS at `:489-500` *before* parsing metadata
at `:501`) → every contract test mutates the **library**, never the artifact; P3 (no `scripts/__init__.py`) → the
`from _cache import ...` idiom, plus "verify RED **for the stated reason**" so an import error can never be
mistaken for the assertion firing.

Two of the review's findings turned out to be bigger than reported, and the plan changed more than the reviewer
asked for:

- **P4 exposed a real defect, not just a weak gate.** Pushing the gate to read the *built* contracts forced the
  question "what does each model actually declare?" — and xS was declaring `penalty_area_half_width`, a constant
  **it does not consume** (verified: `_xshot_occurrence.py` contains no penalty-area constant or predicate at all).
  That declaration would have made Task 2's canonical flip raise on every xS load with xS's features provably
  unchanged. Now: xS declares `goal_width` only; xCross declares all three it consumes; ghost declares its frozen
  40.3-derived pair. The per-model sets are pinned in a test.
- **P7's scope was wrong in the plan, not in the review.** xCross's `:209` predicate is **vectorized**; the scalar
  helper cannot be called there without a per-element loop. The migration is therefore constant-level, not
  predicate-level — stated plainly now, with the constants rebound as module aliases so Task 10's AST enumerator
  can still see them.

**Two defects were found in THIS plan by executing it, and are already fixed above** — recorded because they are
the same class the spec reviews kept surfacing:

1. **The probe was wrong on first write.** With 4 defenders / 3 attackers and no ball `z`, xS returned **7 NaN
   features** — violating the plan's own zero-NaN gate. Fixed to 5-and-5 plus `z`, then re-run: ghost 26 / xS 27 /
   xCross 16, **0 NaN**, with `attackers_in_box` measured at 0 (20.15) vs 1 (20.16).
2. **`in_penalty_area_goal_relative` originally carried a `0.0 <= gr_x` lower bound**, which the shipped xCross
   predicate does not have. Real tracking carries `x` beyond the goal line, so that guard would have changed
   xCross behaviour and broken the byte-identity Task 2 promises. Removed, with the reasoning inline.

**And the Task-10 enumerator was RUN against the real modules**: it finds 14 geometry constants, not the 8 the
first draft's lists covered. The 6 missing ones are the goal-mouth family, now **declared** (they drive
`openGoal`) rather than exempted — which is the gate doing exactly its job, before an implementer met it as a
red test with no guidance.

**The P6-corrected teeth test was RUN as written**, through `extract_ghost_gk_features` with the module constants
monkeypatched: `(before, after) = (0, 1)`. `_ghost_gk.py:608` reads `_PENALTY_AREA_X` / `_PENALTY_AREA_Y_MIN` /
`_PENALTY_AREA_Y_MAX` as module globals at call time, so `monkeypatch.setattr(gg, ...)` reaches it — which is the
load-bearing assumption the test rests on, and the reason it is stated as a measurement rather than a plan.

**Plan-review round 2 (S1–S6) is folded in.** S2 was the only defect and it was one the P7 fix *created*: aliasing
`_BOX_*` in `_params.py` after its body moved to the helper would have left two names nothing reads — a decoy that
satisfies Task 10's enumerator vacuously. Verified `:80` is their only consumer, so they are now DELETED there
while xCross keeps its aliases, under one rule: *a module-level constant exists iff something in that module reads
it.* S3 corrected an over-broad unreachability claim (the `_SPADL_X_MAX` cap guards `derive_goalkeepers`' TRACKING
input at `:88`; this helper's caller works on ACTION coords — different path). S4 removed duplicate/unused imports
that would have failed `ruff check tests/`. S5 wrote down the derived-constant rule the two lists were silently
disagreeing about. S1 became the optional Task 12.

## Execution record (what actually happened, and where it deviated)

Executed 2026-07-26 on branch `pr-s133-feature-contract`. Baseline before any change: **5756 passed,
66 skipped, 219 deselected**.

**Deviations from the plan as written, all deliberate and all verified:**

1. **Task 11's migration script inferred the SHA256SUMS file lists — and inferred them WRONG.** The
   plan appended `metrics.json` when present. Read from each model's own `save()`: all three hash
   exactly **two** entries (`weights, metadata.json`), and `metrics.json` sits in the xS/xCross
   directories *unhashed*. An inferred list disagreeing with what `load()` verifies produces an
   `IntegrityError` on the very next load. The shipped script uses explicit per-model lists. This is
   precisely the check the plan told the implementer to make before running it.
2. **Task 8 needed a real extraction the plan described in one line.** Keying the fingerprint on the
   requested corpus meant the corpus-selection rule had to be shared with `load_matches`, not copied —
   so `_wanted_for_provider` / `select_match_ids` were extracted in `_loader_pining.py`, with a test
   pinning the four selection cases. A second copy would let the fingerprint describe a corpus the
   extraction never loaded.
3. **Task 9's pre-flight source was unspecified.** Ghost's providers come from a `source_provider`
   COLUMN, not the directory path, so the startup check reads one column from one row group per
   parquet — seconds, against an extraction measured in tens of minutes.
4. **Task 5/6 use a module-level `_feature_contract_block()`** per model rather than an inline lambda in
   `save()`, so `save()` and `load()` cannot drift apart in what they build.
5. **Test counts in this plan were wrong throughout** (they double-counted Task 3 and ignored that Tasks
   1/3/4/5/6 share one file). Actual: Task 1 → 4, +Task 3 → 7, +Task 4 → 19, +Task 5 → 21, +Task 6 → 22,
   +Task 11 → 33, +Task 12 → 35. `test_penalty_area.py` → 14 (not 14-then-16; the migration tests are 2
   of those 14). Task 10 → 5.
6. **Three tests added beyond the plan**, each guarding a load-bearing property it left uncovered:
   probe determinism (an unstable probe silently degrades every fingerprint check to a skip),
   no-dead-entries in the enumeration lists (stale bookkeeping stops describing what it governs), and
   `keeper_detection_mask` still rejecting an unknown provider (moving a check earlier must not remove
   it from where it was).
7. **`_Err` → `_FakeIntegrityError`** (ruff N818), and four `match=` patterns made raw strings (RUF043).
8. **Task 8's "no constant token" test was tightened, not weakened.** The plan asserted `'"schema-v2"'
   not in src`, which also forbids *explaining* the change in a docstring. It now asserts on the
   module-level ASSIGNMENT via AST — the actual defect — plus the presence of `corpus_fingerprint` and
   `select_match_ids`.
9. **Two gates caught real problems in work the plan didn't cover.** The whole-package doctest sweep
   caught that ghost's `from_variant`/`from_hub` doctests execute against the **HF `full`** variant —
   which has no contract, so the new CI escalation turns them into errors. Converted to literal blocks
   with the consequence stated in the docstring: a consumer escalating the category gets a raise on the
   Hub path. That is correct behaviour and now documented where it will be read. Separately, the C4
   description-cap gate (200 chars) rejected all three of my architecture.dsl edits — detail belongs in
   the ADR, so the descriptions were shortened and the diagram regenerated.
10. **NOTICE gains an IFAB Laws of the Game entry** for the penalty-area dimensions. Not strictly a
    "published methodology" under the ADR-005 rule, but the code now cites FIFA Laws as the source of a
    canonical constant, and the repo previously carried two divergent values with neither citing
    anything — which is the situation worth not repeating.

**Task 11's checkpoint (the one claim that could not be verified without running it) PASSED:**
`git status` showed exactly six files (`metadata.json` + `SHA256SUMS` ×3, no weights file); ghost's
**654 arrays bit-identical**; both boosters byte-identical as JSON; all three metadata deltas
**additive-only** (no key removed, none changed).

---

**S1 was resolved by rejecting both options offered.** The review framed it as *escalate + ~14 opt-outs* vs
*defer with a TODO*. Both accepted spec §4's premise that only ghost's artifact could be stamped this cycle — and
that premise is false: all three bundled artifacts have identical structure, and xS/xCross metadata already carries
`chirality`/`geometry_version`/`pitch_length`/`pitch_width`. So Task 11 stamps all three and Task 12 escalates with
an **empty** opt-out list. Closing the gap costs less than counting it (one script over three directories vs ~14
test files), leaves no ledger to maintain, and is fail-closed from day one. What the stamp does and does not attest
is stated explicitly in Task 11's preamble rather than left to be inferred.

**S6 was flagged as "probably right, doesn't affect the outcome" — it was wrong, and measuring settled it.** The
claim that patching `GOAL_WIDTH` isolates the constants prong assumed `openGoal` depends on it only via
`GOAL_Y_MIN/MAX`. It does not: `_xshot_occurrence.py:118` divides by `GOAL_WIDTH` directly, and on this probe
`openGoal` is **0.996352**, not a saturated 1.0 — so patching moves it to **0.996439** and the fingerprint changes
too. The test still passes (constants compare first), so only the docstring was false; it now says what actually
happens, and the isolation property lives where it belongs, in Task 4's unit test.
