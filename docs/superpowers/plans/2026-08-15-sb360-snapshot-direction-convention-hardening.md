# SB360 snapshot direction-convention hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the SB360 snapshot direction convention (both teams `"ltr"` = never re-projected) with a named constant, a behavioral test, and a documentation sweep, so a future reader cannot "fix" it into the ADR-028 mixed-frame defect.

**Architecture:** Additive doc/test hardening of an already-correct convention. `snapshot_to_tracking_frames` labels both teams `"ltr"` because a freeze-frame is already in SPADL action-LTR; `acting_team_attacks_rtl` therefore returns a resolved all-`False` (no-flip) mask, which every ADR-028 geometry consumer gates re-projection on. Nothing about behaviour changes — this cycle documents and pins the convention and corrects stale documentation of it.

**Tech Stack:** Python, pandas (nullable-boolean + `Int64` semantics), pytest, `silly_kicks.id_compat` (dtype-safe id comparison).

## Global Constraints

- **One feature branch, one commit, one PR.** No commit steps appear in this plan; the user commits once, on their own approval. The uncommitted spec + this plan ride that same commit (an untracked doc makes provenance drivers treat the tree as dirty).
- **No behaviour change, no retrain, no re-materialization, no public-surface change.** If any test that currently passes starts failing after a code edit here, STOP and investigate — do not change `silly_kicks/tracking/_snapshot.py`'s output.
- **No golden / C4 / trained-model artifacts are touched.**
- **Run the full suite on BOTH interpreters:** `.venv` (py3.10, pandas 2.3.3) and `.venv312` (py3.12, pandas 3.0.5). The new test exercises nullable-boolean `.all()`/`.any()` + `Int64` `.loc` masking, which differ across pandas majors (copy-on-write is NOT the hazard — `frames.copy()` is an explicit deep copy).
- **Lint at CI scope only:** `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, and bare `python -m pyright`. Never `ruff check .`.
- **Confirm no incidental warning under `-W error`** (the suite may run `filterwarnings=error`): the new test's frames are fully oriented on both legs, so no `OrientationUnresolvedWarning`; `_snapshot.py` has a concat-all-NA `FutureWarning` history — confirm none trips.
- **Version / PR numbers are assigned at commit-prep**, after merging `origin/main` (the parallel keeper-box cycle also bumps the version; only the `TODO.md` "Release" header line and CHANGELOG collide).

---

### Task 1: New behavioral test — `test_snapshot_actions_are_never_reprojected`

Pins the *meaning* of the uniform-`"ltr"` labelling that `test_constant_columns` pins the *value* of. This is a **characterization test** (green on current code — the convention is already correct), plus a **non-vacuity mutation probe** (NOT red-first — there is no fix to drive; see spec §5).

**Files:**
- Modify: `tests/tracking/test_snapshot.py` (add one test function; reuses existing `actions_3` + `snapshots_combined` fixtures)

**Interfaces:**
- Consumes: `actions_3` (fixture; action `id=11` is `team_id=200`, the away action) and `snapshots_combined` (fixture; teams 100 & 200 in one frame at `action_id=10`) — both already defined at the top of `tests/tracking/test_snapshot.py`.
- Consumes: `silly_kicks.tracking._action_orientation.acting_team_attacks_rtl(actions, frames) -> pd.Series` (nullable `"boolean"`; `<NA>` = unresolved, `False` = resolved-no-flip); `silly_kicks.tracking._snapshot.snapshot_to_tracking_frames`; `silly_kicks.id_compat.ids_match(series, scalar) -> pd.Series` (non-nullable `np.bool_`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the test function** at the end of `tests/tracking/test_snapshot.py`.

```python
def test_snapshot_actions_are_never_reprojected(actions_3, snapshots_combined):
    """Uniform 'ltr' means acting_team_attacks_rtl resolves BOTH teams to a RESOLVED no-flip.

    Pins the MEANING of the labelling that test_constant_columns pins the VALUE of. A snapshot is
    already in SPADL action-LTR, so the flip mask acting_team_attacks_rtl returns is the input EVERY
    ADR-028 geometry consumer gates its re-projection on -- an all-False (resolved) mask is exactly
    what "never re-projected" means. A future change to per-team directions would flip away-team
    actions; this test fails first, and its mutation leg proves it would.

    Cross-module: this lives in test_snapshot.py for fixture reuse but asserts a property of
    _action_orientation; a future move of acting_team_attacks_rtl touches a test in the snapshot file.

    Accepted limit (spec section 8): this pins the SEAM's contract -- acting_team_attacks_rtl returns
    a resolved no-flip mask for a snapshot -- NOT the guarantee that every consumer keeps routing
    through that seam. The module is the documented SSOT for re-projection, so the seam is the right
    altitude for a doc-hardening cycle.
    """
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
    from silly_kicks.tracking._snapshot import snapshot_to_tracking_frames

    away = ids_match(actions_3["team_id"], 200)
    assert away.any()  # premise: the load-bearing away action EXISTS (guards emptiness-vacuity)

    frames, _links = snapshot_to_tracking_frames(snapshots_combined, actions_3)
    flip = acting_team_attacks_rtl(actions_3, frames)

    # Property: every action RESOLVES (not <NA>) and never flips -> SB360 is never re-projected.
    assert flip.notna().all()
    assert not flip.any()

    # Non-vacuity: the per-team "fix" this guards against WOULD flip the away action, from a RESOLVED
    # False to a RESOLVED True. Re-assert notna on the MUTATED frame -- nullable-boolean .all() is
    # skipna=True, so without this an <NA> away action would pass .all() silently.
    per_team = frames.copy()
    per_team.loc[ids_match(per_team["team_id"], 200), "team_attacking_direction"] = "rtl"
    flip_mut = acting_team_attacks_rtl(actions_3, per_team)
    assert flip_mut.notna().all()      # still fully resolved post-mutation
    assert flip_mut[away].all()        # away flips to True
    assert not flip_mut[~away].any()   # home unchanged (still False)
```

- [ ] **Step 2: Run the new test, expect PASS** (characterization — the convention is already correct).

Run: `python -m pytest "tests/tracking/test_snapshot.py::test_snapshot_actions_are_never_reprojected" -v`
Expected: PASS.

- [ ] **Step 3: Prove the mutation probe is non-vacuous.** Temporarily neutralize the mutation — change the `per_team.loc[...] = "rtl"` line to a no-op (e.g. `per_team = frames.copy()` with the `.loc` assignment removed) — and re-run.

Run: `python -m pytest "tests/tracking/test_snapshot.py::test_snapshot_actions_are_never_reprojected" -v`
Expected: FAIL at `assert flip_mut[away].all()` (with no mutation, `flip_mut[away]` is resolved all-`False`, so `.all()` is `False`). This proves the mutation is what moves the away action. **Then restore the `.loc[...] = "rtl"` line** and re-run → PASS.

- [ ] **Step 4: Run the whole file on both interpreters, expect PASS.**

Run: `python -m pytest tests/tracking/test_snapshot.py -v` (in `.venv`), then the same in `.venv312`.
Expected: PASS on both.

---

### Task 2: Named constant `_SNAPSHOT_ATTACKING_DIRECTION` + breadcrumb

Replaces the two duplicated `"ltr"` literals with one module-level constant whose comment points to the authority. Pure refactor — no output change; the existing `test_constant_columns` is the regression guard.

**Files:**
- Modify: `silly_kicks/tracking/_snapshot.py` (add the constant after the imports ~`:14`; use it at `:131` and `:163`)

**Interfaces:**
- Consumes: nothing new.
- Produces: module-level `_SNAPSHOT_ATTACKING_DIRECTION = "ltr"`. Its own comment NAMES the pinning test `test_snapshot_actions_are_never_reprojected` (Task 1); the test does NOT reference the constant back — the coupling is one-directional and Tasks 1 and 2 are independent.

- [ ] **Step 1: Confirm the regression guard is green** before changing anything.

Run: `python -m pytest "tests/tracking/test_snapshot.py::test_constant_columns" -v`
Expected: PASS (it asserts both teams get `"ltr"`).

- [ ] **Step 2: Add the constant immediately after the imports** (after line `from .schema import ...`, before `def snapshot_to_tracking_frames`). Placement is above first use on purpose (spec §4.1): the constant exists for its rationale comment, so it must precede `:131`, not sit at the `_ID_COLUMNS` position (`:192`, below the function).

```python
#: BOTH teams are labelled with this ONE value on purpose: a snapshot shares its event's SPADL
#: action-LTR frame, so it is already action-LTR and the geometry layer must NEVER re-project it
#: (ADR-028). This is the accepted-convention case in ``validate_period_directions`` -- NOT the
#: rejected single-team self-contradiction (that guard raises only when ONE team carries both
#: directions in a period). Flipping to per-team directions reintroduces the ADR-028 mixed-frame
#: defect on all SB360 input. Pinned by ``test_snapshot_actions_are_never_reprojected``.
_SNAPSHOT_ATTACKING_DIRECTION = "ltr"
```

- [ ] **Step 3: Replace both literals.** At the player-row dict (currently `"team_attacking_direction": "ltr",` at `:131`) and the ball-row dict (currently `"team_attacking_direction": "ltr",` at `:163`), change the value to the constant:

```python
            "team_attacking_direction": _SNAPSHOT_ATTACKING_DIRECTION,
```

Both occurrences are **byte-identical** (12-space indent), so this is the one edit that REQUIRES `replace_all=True` (a single-occurrence edit errors with "old_string not unique") — in deliberate contrast to Task 3 Step 3, where the two comments differ by indentation and must be edited separately. The surrounding `speed_source` comment blocks stay untouched.

- [ ] **Step 4: Confirm no behaviour change on both interpreters.**

Run: `python -m pytest "tests/tracking/test_snapshot.py::test_constant_columns" "tests/tracking/test_snapshot.py::test_snapshot_actions_are_never_reprojected" -v` (in `.venv`, then `.venv312`)
Expected: PASS on both. If `test_constant_columns` fails, the refactor changed output — revert and investigate (it must not).

---

### Task 3: Stale-comment sweep (6 files) + authority citation fix (2 sites)

Documentation-only, **per-site judgment, not a find-replace** (spec §7). Three categories: flatly-stale (4 sites claim `validate_period_directions` *rejects* blanket-`"ltr"` — false), imprecise-but-valid (1 site), correct-in-context (1 site — a clarifying aside only, do NOT rewrite). Plus two rotted `_snapshot.py:92` citations → the symbol.

**Files:**
- Modify: `tests/tracking/conftest_id_dtype.py:95-97`
- Modify: `tests/tracking/test_aggregator_column_liveness.py:116-119`
- Modify: `tests/tracking/test_defensive_line.py:101-102` and `:123-124` (two identical comments)
- Modify: `tests/tracking/test_off_ball_runs.py:67-71`
- Modify: `tests/tracking/test_off_ball_runs_orientation.py:8-11` (docstring) and `:153` (citation)
- Modify: `tests/vaep/test_hybrid_with_tracking.py:17-22` (aside only)
- Modify: `silly_kicks/tracking/_action_orientation.py:56` (citation)

**Interfaces:** none (comments/docstrings only).

- [ ] **Step 1 — `conftest_id_dtype.py`.** Replace the comment above `team_attacking_direction=...`:

Old:
```python
        # ADR-041: per-TEAM direction. Team 5 is home; a blanket "ltr" labels both teams as
        # attacking the same way, which is physically impossible and is now rejected by
        # validate_period_directions.
```
New:
```python
        # ADR-041: per-TEAM direction. Team 5 is home; a blanket "ltr" would silently make
        # acting_team_attacks_rtl return no-flip for the away team and mis-orient its geometry.
        # (validate_period_directions does NOT reject a uniform "ltr" -- it raises only when a
        # SINGLE team self-contradicts in a period; snapshot frames use uniform "ltr" by convention.)
```

- [ ] **Step 2 — `test_aggregator_column_liveness.py`.** Replace:

Old:
```python
        # ADR-041: per-TEAM direction. Team 5 is home (all actions are team 5, so the
        # acting team never flips and every aggregator's values are unchanged); team 6
        # attacks the other way. A blanket "ltr" is physically impossible -- two teams
        # cannot attack the same way -- and validate_period_directions now rejects it.
```
New:
```python
        # ADR-041: per-TEAM direction. Team 5 is home (all actions are team 5, so the
        # acting team never flips and every aggregator's values are unchanged); team 6
        # attacks the other way. A blanket "ltr" would silently make acting_team_attacks_rtl
        # return no-flip for team 6 and mis-orient its geometry. (validate_period_directions
        # does NOT reject uniform "ltr" -- it raises only on a SINGLE team self-contradicting.)
```

- [ ] **Step 3 — `test_defensive_line.py` (TWO occurrences, DIFFERENT indentation).** The comment appears twice with the SAME text but different leading whitespace: once at **12 spaces** (~`:101-102`, the away-GK block) and once at **16 spaces** (~`:123-124`, nested in the away-outfield `for` loop). A single `replace_all` with one exact string will NOT match both — do two separate edits, preserving each site's indentation. The 12-space form:

Old:
```python
            # ADR-041: away attacks the OTHER way. Labelling both teams "ltr" is
            # physically impossible and is now rejected by validate_period_directions.
```
New:
```python
            # ADR-041: away attacks the OTHER way. Labelling both teams "ltr" would silently
            # make acting_team_attacks_rtl return no-flip for the away team and mis-orient its
            # geometry -- NOT a validate_period_directions raise (it accepts uniform "ltr" and
            # rejects only a SINGLE team self-contradicting).
```
The second occurrence is identical text at **16 spaces** of indentation — apply the same replacement there, indented to match. Confirm both are done: after the edits, `grep -n "physically impossible" tests/tracking/test_defensive_line.py` must return nothing.

- [ ] **Step 4 — `test_off_ball_runs.py`.** Replace:

Old:
```python
                    # ADR-041: per-team direction, NOT a blanket "ltr". Labelling both
                    # teams "ltr" is physically impossible (two teams cannot attack the
                    # same way) and is now rejected by validate_period_directions. The
                    # old blanket label is exactly why toward_goal could not be re-keyed
                    # onto the frames' own direction.
```
New:
```python
                    # ADR-041: per-team direction, NOT a blanket "ltr". Labelling both
                    # teams "ltr" would silently make acting_team_attacks_rtl return no-flip
                    # for the away team and mis-orient its geometry -- which is exactly why
                    # toward_goal could not be re-keyed onto the frames' own direction until
                    # per-team labels were used. (validate_period_directions accepts uniform
                    # "ltr"; it rejects only a SINGLE team self-contradicting.)
```

- [ ] **Step 5 — `test_off_ball_runs_orientation.py` docstring (`:8-11`).** Imprecise-but-valid: keep the re-key-safety point, fix the "physically impossible" framing. Replace:

Old:
```
The re-key was only safe once ``validate_period_directions`` started rejecting frames whose
per-team labels are physically impossible: ``_validate_ltr`` alone accepts every row being
``"ltr"`` (it merely requires that ``"ltr"`` appears), and on such frames
``acting_team_attacks_rtl`` silently resolves to "no flip" for the away team.
```
New:
```
The re-key was only safe once ``validate_period_directions`` began rejecting the genuinely-broken
case -- a SINGLE team carrying both "ltr" and "rtl" in one period. It deliberately does NOT reject a
uniform "ltr" (that is an accepted convention, e.g. snapshot frames): ``_validate_ltr`` accepts it
too (it merely requires that "ltr" appears), and on such a frame ``acting_team_attacks_rtl`` silently
resolves to "no flip" for the away team -- the mis-orientation the per-team labels below avoid.
```

- [ ] **Step 6 — `test_off_ball_runs_orientation.py:153` citation.** Remove the rotted line-number:

Old:
```
    worked: ``snapshot_to_tracking_frames`` (``_snapshot.py:92``, uniform "ltr" because
```
New:
```
    worked: ``snapshot_to_tracking_frames`` (uniform "ltr" because
```

- [ ] **Step 7 — `test_hybrid_with_tracking.py:17-22`: CORRECT AS-IS, add a clarifying aside only.** This describes a *real two-team scene* where uniform-`"ltr"` genuinely IS a fixture bug, and it does NOT make the false guard claim — do NOT rewrite it. Append one sentence to preempt conflation with the snapshot convention:

Old (end of the comment block):
```python
# actions -- 101 of them away actions. The fixture therefore never exercised the ADR-028
# re-projection path at all.
```
New:
```python
# actions -- 101 of them away actions. The fixture therefore never exercised the ADR-028
# re-projection path at all. (This is a REAL two-team scene, where uniform "ltr" is a fixture
# bug; contrast snapshot_to_tracking_frames, where uniform "ltr" is the CORRECT convention
# because a freeze-frame is already action-LTR -- see _SNAPSHOT_ATTACKING_DIRECTION.)
```

- [ ] **Step 8 — `_action_orientation.py:56` authority citation.** Remove the rotted line-number (this is the authority the whole cycle canonizes; the citation must not itself rot):

Old:
```
    * **A different convention** -- ``snapshot_to_tracking_frames`` (``_snapshot.py:92``)
      labels every player ``"ltr"`` because snapshot frames are ALREADY in SPADL action-LTR,
      so "never flip" is the correct reading, not a contradiction.
```
New:
```
    * **A different convention** -- ``snapshot_to_tracking_frames`` labels every player
      ``"ltr"`` because snapshot frames are ALREADY in SPADL action-LTR, so "never flip" is
      the correct reading, not a contradiction.
```

- [ ] **Step 9: Run the affected test files, expect PASS** (comments/docstrings do not change behaviour, but confirm nothing was structurally broken).

Run: `python -m pytest tests/tracking/test_aggregator_column_liveness.py tests/tracking/test_defensive_line.py tests/tracking/test_off_ball_runs.py tests/tracking/test_off_ball_runs_orientation.py tests/vaep/test_hybrid_with_tracking.py -q`
Expected: PASS (or normal skips).

Do NOT run pytest directly on `tests/tracking/conftest_id_dtype.py` — it is a fixture-only helper module (a `conftest_*` suffix, NOT the auto-loaded `conftest.py`) with zero test functions, so pytest exits 5 ("no tests collected"), which some CI treats as failure. Its comment edit is behaviour-neutral and is exercised by the files above (which consume its fixture) and the full Task 6 sweep. Confirm both edited non-test modules still parse: `python -c "import silly_kicks.tracking._action_orientation"` and `python -c "import tests.tracking.conftest_id_dtype"`.

---

### Task 4: Rip out the retracted `TODO.md` row (strikethrough is not used)

The SB360 Tech-Debt section carried a **struck-through**, retracted row — the 2026-08-13 owner
retraction of the goal-kick-coverage constraint. The repo does **not** use strikethrough (owner,
2026-08-15: "we do not use strike through ever, rip them out"), and a struck-through retracted
non-action row is exactly the "completed item" TODO grooming removes. So the whole bullet is deleted,
not folded in. Its measurement is preserved in `docs/research/sb360_coverage/` and the memory topic
file. This **subsumes** the earlier `n=16` figure-reconcile (that dispersion line lived inside the
removed bullet).

**Files:**
- Modify: `TODO.md` (delete the retracted SB360 goal-kick-coverage bullet)

- [ ] **Step 1: Delete the whole bullet.** Remove every line from the one starting `- **~~SB360
  goal-kick` up to (not including) the next bullet `- **\`sportec_slim.parquet\``. Robust approach —
  match by start/end markers, not by the special-char body (en-dashes, `~~`):

```python
import pathlib
p = pathlib.Path("TODO.md")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(i for i, ln in enumerate(lines) if ln.startswith("- **~~SB360 goal-kick"))
end = next(i for i, ln in enumerate(lines) if ln.startswith("- **`sportec_slim.parquet`"))
del lines[start:end]
p.write_text("".join(lines), encoding="utf-8")
```

- [ ] **Step 2: Confirm no strikethrough remains** anywhere in the file.

Run: `grep -c "~~" TODO.md`
Expected: `0`. Also confirm the lakehouse bullet ("...Not answerable from this repo.") now directly precedes the `sportec_slim.parquet` bullet.

---

### Task 5: Release mechanics — version bump + CHANGELOG

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md` ("Release" header line at `:5`)

- [ ] **Step 1: Determine the version.** After the branch is synced with `origin/main` at commit-prep, take the next-free version. This repo takes one **minor** bump per cycle (4.80.0 → 4.81.0 → 4.82.0), so this is most likely **4.82.0** — NOT a patch (`4.81.1`). The parallel keeper-box cycle also bumps; verify the current top version in `CHANGELOG.md` on `main` and take the next unused `X.Y.0`. Use the resolved number everywhere below (referred to as `X.Y.Z`).

- [ ] **Step 2: Bump the four version sites** to `X.Y.Z`: `pyproject.toml` (`version = `), `silly_kicks/__init__.py` (`__version__ = `), `uv.lock` (**hand-edit** the single `version = "..."` line under the `[[package]]` / `name = "silly-kicks"` stanza — currently `uv.lock:3626`; do NOT run `uv lock`, which would re-resolve the carefully-pinned deps — confirm this matches how the parallel keeper-box cycle edits it so the two don't diverge at merge), and the `TODO.md:5` "Release" line.

- [ ] **Step 3: Add the CHANGELOG entry** under a new `## X.Y.Z` heading (keyed by the assigned PR number `PR-Snnn`):

```markdown
## X.Y.Z

### Hardened
- **SB360 snapshot direction convention is now named, tested, and correctly documented (PR-Snnn).**
  `snapshot_to_tracking_frames` labels both teams `team_attacking_direction="ltr"` because a
  freeze-frame is already in SPADL action-LTR, so `acting_team_attacks_rtl` returns a resolved
  all-`False` (no-flip) mask and SB360 is never re-projected (ADR-028). The value is now a named
  constant `_SNAPSHOT_ATTACKING_DIRECTION` with a pointer to the authority in
  `validate_period_directions`, pinned by `test_snapshot_actions_are_never_reprojected` (both-teams
  resolved-no-flip + a non-vacuity mutation). Corrected six stale test comments that claimed
  `validate_period_directions` *rejects* a blanket `"ltr"` (it accepts uniform `"ltr"`; it raises
  only on a single team self-contradicting) and two rotted `_snapshot.py:92` citations. Groomed the
  SB360 Tech-Debt section: removed the retracted goal-kick-coverage-constraint row (strikethrough is
  not used in this repo; the measurement is preserved in `docs/research/sb360_coverage/`). **Doc/test
  only — no behaviour change, no retrain, no re-materialization, no public-surface change.**
```

---

### Task 6: Final CI-faithful verification

**Files:** none (verification only).

- [ ] **Step 1: Full non-e2e suite on `.venv` (py3.10).**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all pass (normal skips OK). Paste the summary line.

- [ ] **Step 2: Full non-e2e suite on `.venv312` (py3.12).**

Run the same under `.venv312`.
Expected: all pass. (This is the leg that exercises pandas 3 nullable-boolean/`Int64` semantics.)

- [ ] **Step 3: Confirm no incidental warning under `-W error`** for the new test and the snapshot module.

Run: `python -m pytest "tests/tracking/test_snapshot.py" -W error -q`
Expected: PASS (no `OrientationUnresolvedWarning`, no concat-all-NA `FutureWarning`). If a benign warning trips, narrow the `-W error` scope or filter that specific category at the call site — do NOT silence a real one.

- [ ] **Step 4: Lint + types at CI scope.**

Run: `python -m ruff check silly_kicks/ tests/ scripts/` ; `python -m ruff format --check silly_kicks/ tests/ scripts/` ; `python -m pyright`
Expected: clean. Paste the exit-coded output.

---

## Self-review — spec coverage

- **§4.1 named constant** (above first use, pointer comment) → Task 2. ✅
- **§4.2 behavioral test** (C1-fixed: `away.any()` + `notna().all()` on both legs + mutation) → Task 1. ✅
- **§4.3 rip out the struck-through retraction row** → Task 4. ✅
- **§7 stale-comment sweep** (6 files, per-site judgment: 4 flatly-stale / 1 imprecise-but-valid / 1 correct-in-context) + **C3 citation fix** (`_action_orientation.py:56` + `test_off_ball_runs_orientation.py:153`) → Task 3. ✅
- **§4.5 release mechanics** (patch bump, 5 sites, CHANGELOG; no ADR/CLAUDE.md) → Task 5. ✅
- **§5 verification** (characterization-not-red-first; dual-major run; `-W error`; lint/pyright at CI scope) → Task 1 Step 3, Task 6. ✅
- **§8 accepted limit** (test protects the seam contract, not that all consumers use the seam) → recorded as a distinct "Accepted limit" note in the test docstring (Task 1). ✅
