# SB360 snapshot direction-convention hardening — design

- **Date**: 2026-08-15
- **Status**: **Ready to implement** — review round 1 (8 findings, all accepted) and round 2 ("ready
  to implement"; 2 minor items folded in: constant placement above first use, and the corroborating
  `test_obso_orientation_e2e` site + a phrasing-robustness note on the §7 completeness). Verified
  against the code throughout. Not committed.
- **Size**: Dunkin'+ (additive — a named constant, one test, doc/comment edits across 6 test files +
  two stale-citation fixes; no behaviour change, no retrain, no new public surface). One feature
  branch, one commit, one PR.
- **Follows**: ADR-028 (per-action geometry re-projection), ADR-041 (`validate_period_directions`
  narrowing), ADR-051 (D3 direction-from-identity close-out). No new ADR — this hardens existing
  decisions, it does not make one.
- **Owner-directed input**: the `TODO.md` SB360 retraction was a pre-existing uncommitted edit the
  owner first said to *"fold into whatever comes next"*, then (2026-08-15) directed be **ripped out
  entirely** — the repo does not use strikethrough, and a retracted non-action row is not tech debt.

---

## 1. Executive summary

`snapshot_to_tracking_frames` (the SB360 freeze-frame → tracking-frame port) labels **both teams**
`team_attacking_direction="ltr"`. This is **correct** — a StatsBomb freeze-frame already shares its
event's SPADL action-LTR frame, so it must never be re-projected — but it is *exactly* the shape the
codebase elsewhere calls "physically impossible: two teams cannot attack the same way." A future
reader reconciling those two could "fix" `_snapshot.py` into per-team directions and thereby
reintroduce the **ADR-028 mixed-frame defect** on all SB360 input.

This cycle closes the **narrow residual gap** left after the 4.80.0 D3 cycle:

1. **A named constant** `_SNAPSHOT_ATTACKING_DIRECTION` giving the load-bearing value one home, with a
   pointer to the authoritative rationale (revised from a duplicated comment per review C4).
2. **A behavioral test** locking the *consequence* — `acting_team_attacks_rtl` yields resolved no-flip
   for a two-team snapshot, with a non-vacuity mutation that is itself guarded against `<NA>` and
   emptiness (revised per review C1).
3. **Rip out** the struck-through `TODO.md` retraction row (the repo does not use strikethrough); this subsumes the earlier `n=16` reconcile.
4. **A directory-agnostic, per-site stale-comment sweep** across all of `tests/` (6 files, not the 4
   originally scoped), plus fixing the rotted `_snapshot.py:92` citation inside the very authority this
   spec canonizes (review C2/C3).
5. **Release mechanics** (version bump, CHANGELOG).

---

## 2. Background — and the premise correction

The original framing (a 2026-08-12 triage note) was that the `_snapshot.py` `"ltr"` convention was
*"recorded nowhere"* and an *unguarded live trap*. **That was true on 2026-08-12 but is now largely
false**: the 4.80.0 D3 cycle (ADR-051 / ADR-041) documented and partially guarded it. The value of
this cycle rests on the *corrected*, narrower picture, stated here in full.

### 2.1 Verified current state (reviewer independently re-verified each; evidence in-line)

| Fact | Location | Verified behaviour |
|---|---|---|
| Both teams labelled `"ltr"` | `_snapshot.py:131` (player rows), `:163` (ball row) | Uniform `"ltr"`; the adjacent comment there is `speed_source` (`:123–128`/`:155–160`), not direction |
| The guard **accepts** uniform-`"ltr"` | `_action_orientation.py:45–127` (`validate_period_directions`) | Raises **only** when a *single* team carries both `"ltr"` and `"rtl"` in one period (per-team `set()` with `len > 1`, `:114–127`). Uniform-`"ltr"` → each team `{"ltr"}` → no raise |
| Rationale **is** documented (but with a rotted pointer — see §7/C3) | `_action_orientation.py:56–58` | Names `snapshot_to_tracking_frames` as an accepted "different convention" — **but cites `_snapshot.py:92`, which is now `if len(action_meta) == 0:`.** The literals live at `:131`/`:163`. This citation is fixed in this cycle. |
| Behaviour correct end-to-end | `acting_team_attacks_rtl`, `_action_orientation.py:173–292` | For blanket-`"ltr"` both teams resolve to **`False`** (resolved, no-flip) at `:289–292` — *not* `<NA>` |
| Output value test-pinned | `tests/tracking/test_snapshot.py:150` (`test_constant_columns`) | Asserts both teams (100 & 200) get `"ltr"`. A per-team "fix" **would fail this** |
| Convention exercised through a kernel | `tests/tracking/test_off_ball_runs_orientation.py:153, 178` | **Caveat (C2/C3):** this file's module docstring (`:9`) repeats the stale framing and its `:153` citation carries the same rotted `_snapshot.py:92`. Cited here as evidence *and fixed in §7*, so the citation no longer lands a reader on a contradicting docstring. |
| Convention **already documented correctly** in a second place | `tests/tracking/test_obso_orientation_e2e.py::test_oracle_discriminates` (surfaced by review round 2) | Builds `frames.assign(team_attacking_direction="ltr")` and its docstring calls this *"exactly the state in which `acting_team_attacks_rtl` resolves to 'no flip' for both teams"* — the right mechanism, no false guard claim. Independent corroboration that the convention is understood; **not stale, out of scope**. |

### 2.2 The residual gap (what this cycle fixes)

- **(a) No single, structural home for the value.** The literals are duplicated at `:131`/`:163` with
  only a `speed_source` comment nearby; the direction rationale lives one module away.
- **(b) The existing pin's intent is not stated.** `test_snapshot.py:150` pins the *value* under a
  docstring reading *"Verify NaN/constant columns per spec."* A future author introducing per-team
  directions could just *edit the assertion*, not realising uniform-`"ltr"` is load-bearing.
- **(c) The documentation that *does* exist is partly stale.** Six test-comment sites and one authority
  citation describe the pre-ADR-041 guard (§7).

---

## 3. Scope

**In scope** (one branch, one commit, one PR):

- `silly_kicks/tracking/_snapshot.py` — the named constant `_SNAPSHOT_ATTACKING_DIRECTION` + its
  pointer comment, used at `:131` and `:163`.
- `silly_kicks/tracking/_action_orientation.py` — fix the rotted `_snapshot.py:92` citation in the
  `validate_period_directions` docstring (C3). One-symbol edit; no behaviour change.
- `tests/tracking/test_snapshot.py` — the new behavioral test.
- **Stale-comment sweep across all of `tests/` (C2), per-site judgment (not find-replace):**
  `tests/tracking/conftest_id_dtype.py:96`, `test_aggregator_column_liveness.py:118–119`,
  `test_defensive_line.py:102,124`, `test_off_ball_runs.py:68`, `test_off_ball_runs_orientation.py:9`
  (+ its `:153` citation, C3), and `tests/vaep/test_hybrid_with_tracking.py:20`.
- `TODO.md` (rip out the struck-through retraction row), `CHANGELOG.md`, version files.

**Out of scope** (declined or blocked): `sportec_slim.parquet` mirror repair; `visible_area` wiring;
the four `UNAUDITABLE_BOUNDARY` points; lakehouse adoption of `providers/statsbomb`; the Sergio
licensed-data questions; extending the coverage pass past 22 matches.

---

## 4. Deliverables

### 4.1 Named constant `_SNAPSHOT_ATTACKING_DIRECTION` (revised per review C4)

Instead of a full note at `:131` + a pointer at `:163` (which replicated the file's own
duplicated-`speed_source`-comment smell), promote the load-bearing value to a module-level constant —
`_ID_COLUMNS` already establishes module-level constants as this file's idiom:

```python
#: BOTH teams are labelled with this ONE value on purpose: a snapshot shares its event's SPADL
#: action-LTR frame, so it is already action-LTR and the geometry layer must NEVER re-project it
#: (ADR-028). This is the accepted-convention case in `validate_period_directions` -- NOT the rejected
#: single-team self-contradiction (that guard raises only when ONE team carries both directions).
#: Flipping to per-team directions reintroduces the ADR-028 mixed-frame defect on all SB360 input.
#: Pinned by `test_snapshot_actions_are_never_reprojected`.
_SNAPSHOT_ATTACKING_DIRECTION = "ltr"
```

Used at both `:131` and `:163`: `"team_attacking_direction": _SNAPSHOT_ATTACKING_DIRECTION`.

**Placement (review round 2, item 1):** define the constant **immediately after the imports (before
`snapshot_to_tracking_frames`, i.e. ~`:14`)**, NOT at the existing `_ID_COLUMNS` position (`:192`,
below the function). The constant's whole purpose is the rationale comment; the file's first use is
`:131`, so a definition at `:192` would land the rationale ~60 lines below the thing it explains.
Runtime is identical either way — this is pure discoverability, which is the constant's reason to
exist.

Rationale for adopting C4: it DRYs the literal, gives the local rationale exactly one home (consistent
with this spec's own single-source principle), and makes "both teams identical" structurally obvious —
breaking the convention now requires *reassigning the constant*, a far louder edit than deleting a
comment. The comment is deliberately a **pointer + the one local fact** (accepted-convention, not
single-team-contradiction), NOT a restatement of the full mechanism — the mechanism's single home
stays `validate_period_directions`. The known tension (it becomes the only named constant among
inlined literals `"alive"`/`"snapshot"`/`"native"` in the same constructor) is accepted: the direction
literal is the one member this whole spec argues is semantically load-bearing, so special treatment is
warranted. The duplicated-comment approach is recorded as the rejected alternative (§6).

### 4.2 New behavioral test — `test_snapshot_actions_are_never_reprojected` (revised per review C1)

Home: `tests/tracking/test_snapshot.py`. Reuses `actions_3` + `snapshots_combined`; the load-bearing
case is **action 11 (`team_id=200`) — an away action** that resolves from the shared frame because
direction resolution is period-keyed.

The original draft had a **vacuity hole (C1)**: `.all()` on a nullable-boolean is `skipna=True`, so an
`<NA>` result would silently pass `(flip_mut[away] == True).all()`, and `notna()` was asserted only on
the un-mutated leg — so the mutation leg proved "the value moved" but not "moved from a resolved
`False` to a resolved `True`." Fixed by guarding emptiness and re-asserting resolution post-mutation,
and by using the `.all()`/`.any()` form throughout (which also settles open-Q2 — the form is preferred
because it removes the `== True/False`-on-nullable trap, not merely for the lint):

```python
def test_snapshot_actions_are_never_reprojected(actions_3, snapshots_combined):
    """Uniform 'ltr' means acting_team_attacks_rtl resolves BOTH teams to a RESOLVED no-flip.

    This pins the MEANING of the labelling that test_constant_columns pins the VALUE of. A snapshot is
    already in SPADL action-LTR, so the flip mask acting_team_attacks_rtl returns is the input EVERY
    ADR-028 geometry consumer gates its re-projection on -- an all-False (resolved) mask is exactly
    what "never re-projected" means. A future change to per-team directions would flip away-team
    actions; this test fails first, and its mutation leg proves it would.
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
    # False to a RESOLVED True. Re-assert notna on the MUTATED frame (C1: closes the skipna hole --
    # without it an <NA> away action passes `.all()` silently).
    per_team = frames.copy()
    per_team.loc[ids_match(per_team["team_id"], 200), "team_attacking_direction"] = "rtl"
    flip_mut = acting_team_attacks_rtl(actions_3, per_team)
    assert flip_mut.notna().all()      # still fully resolved post-mutation
    assert flip_mut[away].all()        # away flips to True
    assert not flip_mut[~away].any()   # home unchanged (still False)
```

Notes:
- `ids_match` (ADR-019), not raw `== 200`: the frame's ball row has `team_id=<NA>`, so a raw mask is
  NA-bearing and `.loc[]` rejects it; `ids_match` returns a plain `np.bool_` Series (verified by the
  reviewer).
- **Cross-module coupling (C6):** this test lives in `test_snapshot.py` for fixture reuse but asserts a
  property of `_action_orientation`. A brief docstring line records that, so a future move of
  `acting_team_attacks_rtl` is understood to touch a test in the snapshot file. The name asserts the
  *consequence* ("never re-projected"); the docstring makes explicit that the all-False flip mask IS
  the seam every consumer's re-projection is gated on, so the name is honest. We deliberately do **not**
  add a `reproject_to_action_ltr(...)` no-op assertion (the reviewer's optional suggestion): re-projection
  is applied to frame-sampled positions, not to an action's own coords, so calling it on `actions_3`
  would exercise `reproject`'s all-False branch rather than anything snapshot-specific — the flip mask
  is the honest assertion point.
- Non-slow, non-version-sensitive → runs on **all** CI legs (ADR-023; must NOT be `slow`).

### 4.3 Remove the retracted row (`TODO.md`) — strikethrough is not used

The SB360 Tech-Debt bullet carrying the 2026-08-13 owner retraction was a **struck-through**
row. The repo does **not** use strikethrough (owner, 2026-08-15: "we do not use strike
through ever, rip them out"), so the whole bullet is **ripped out** rather than folded in — a
struck-through, retracted, non-action row is exactly the "completed item" TODO grooming removes. The
measurement it carried is preserved in `docs/research/sb360_coverage/` and the memory topic file. This
**subsumes** the earlier `n=16` figure-reconcile: that dispersion line lived inside the removed bullet,
so there is nothing left to annotate.

### 4.4 The stale-comment sweep (§7) and the citation fix (C3) — see §7.

### 4.5 Release mechanics

- **Minor** version bump across the five sites (`pyproject.toml`, `__init__.py`, `uv.lock`, CHANGELOG
  heading, `TODO.md` "Release" line) — this repo takes one minor bump per cycle (4.80.0 → 4.81.0 →
  4.82.0), not a patch. The exact number is resolved at commit-prep after merging `origin/main`.
- **CHANGELOG entry** by PR number: the constant, the test, the retracted-row removal, the doc sweep +
  citation fix. Explicitly: no retrain, no re-materialization, no public-surface change.
- **Single commit** (house rule; the spec+plan ride the code commit — an untracked doc makes every
  provenance driver treat the tree as dirty).
- **No ADR**, **no CLAUDE.md contract** (ADR-028/041/051 already carry the convention).

---

## 5. Verification plan (revised per review C5/C7/C8)

- **This is a characterization test + a non-vacuity mutation probe, NOT red-first (C5).** The convention
  is already correct, so the new test is **green on current code** (it documents behaviour, it does not
  drive a fix). Its non-vacuity comes from the mutation leg (team 200 → `"rtl"` must flip the away
  action to a *resolved* True), not from red-first. Confirm the probe is real by reverting the mutation
  to a no-op and observing the away assertion fail. (If the repo ran a mutation-testing tool the manual
  leg would stand in for it; it does not, so the manual leg is the guarantee.)
- **Dual-major run is kept, but for the right reason (C7).** `per_team = frames.copy()` is an explicit
  deep copy in every pandas version, so `frames` is never mutated regardless of copy-on-write — CoW is
  *not* the hazard here. The real cross-major risk the `.venv312` run guards is nullable-boolean
  `.all()`/`.any()` semantics + `Int64` `.loc` masking. Run on both `.venv` (3.10) and `.venv312`.
- **Confirm no incidental warning under `-W error` (C8).** The test relies on full resolution (both
  legs are oriented — the mutated leg is a valid per-team labelling), so no `OrientationUnresolvedWarning`
  should fire. `_snapshot.py` has a history of concat-with-all-NA `FutureWarning` (CHANGELOG.md:409;
  `sb360_coverage/README.md:184`); confirm nothing incidental trips if the suite runs
  `filterwarnings=error`. Likely fine — one confirming line.
- **Full suite** green: `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- **Lint at CI scope**: `ruff check` / `ruff format --check` on `silly_kicks/ tests/ scripts/`; bare
  `pyright`.
- **No golden/C4/model artifacts touched** → no re-materialization.

---

## 6. Rejected alternatives

- **Duplicated comment at both literals (the original draft; rejected per C4).** Full note at `:131` +
  pointer at `:163` replicated the file's own duplicated-`speed_source`-comment smell and gave the
  rationale two homes that can drift. The named constant single-sources the value and makes the
  "both teams identical" invariant structural. Superseded by §4.1.
- **Test approach B — strengthen `test_constant_columns`'s comment only, no new test.** Annotates the
  value-pin without locking the consequence and adds no failing-side proof.
- **Test approach C — end-to-end through a geometry aggregator.** Heavier and overlaps existing
  orientation coverage; `acting_team_attacks_rtl` is the single seam all re-projection flows through, so
  testing it directly is the right altitude. (C6's optional `reproject_to_action_ltr` assertion is a
  lighter cousin of this and rejected for the reason in §4.2.)
- **Restate the full rationale at the `_snapshot.py` site.** Duplicates the `validate_period_directions`
  authority; the constant comment stays a pointer.
- **Bundle the `sportec_slim.parquet` mirror repair.** Declined by the owner for this cycle.

---

## 7. Stale-comment sweep — DECIDED (fix in this cycle), widened, per-site (review C2/C3)

Open-Q1 is resolved: **fix in this cycle** (single commit, so no follow-up split is available anyway).
The review showed my original "four sibling comments / ~4–5 edits" estimate was too narrow. The true
surface across **all of `tests/`** (verified by `grep -rniE 'physically impossible|two teams
cannot|now rejects|attack the same' tests/`, one false positive at `conftest_id_scalar.py:280`
discarded) is **6 files**, and the correction is **per-site judgment, not a find-replace** — the sites
are wrong in different ways:

**Flatly stale** — claim `validate_period_directions` *rejects* blanket-`"ltr"` (it does not; it rejects
only single-team self-contradiction):
- `tests/tracking/conftest_id_dtype.py:96`
- `tests/tracking/test_aggregator_column_liveness.py:118–119`
- `tests/tracking/test_defensive_line.py:102, 124`
- `tests/tracking/test_off_ball_runs.py:68`

**Imprecise-but-carrying-a-valid-point — fix carefully, do not delete the point:**
- `tests/tracking/test_off_ball_runs_orientation.py:9` (module docstring). It makes a *valid*
  architectural claim (the TF-4 re-key was safe once `validate_period_directions` began rejecting
  *single-team contradictions*) but phrases it as "per-team labels are physically impossible," which
  reads as "rejects blanket-`"ltr"`." Reword to name the single-team-contradiction case precisely,
  preserving the re-key-safety point. This is the file §2.1 cites as evidence, so fixing it closes the
  loop where my own citation lands a reader on a contradicting docstring.

**Correct-in-its-own-context — needs a clarifying half-sentence, possibly no change:**
- `tests/vaep/test_hybrid_with_tracking.py:20`. In a **real two-team tracking scene**, a blanket
  `"ltr"` genuinely *is* physically wrong, and this comment attributes the consequence to the correct
  mechanism (`acting_team_attacks_rtl` resolving all-False). It repeats the "physically impossible"
  framing, but for a real scene, not a snapshot. Read it at fix time and decide: it may be correct as-is
  and only need a half-sentence distinguishing the snapshot convention, NOT a correction. **Do not
  blindly "fix" a correct comment.**

**Citation fix (C3), same goal:**
- `silly_kicks/tracking/_action_orientation.py:56` — replace the rotted `_snapshot.py:92` with the
  symbol `snapshot_to_tracking_frames`. This is the authority this spec canonizes; leaving a rotted
  line-number in it contradicts §4.1's own "line numbers rot" principle.
- `tests/tracking/test_off_ball_runs_orientation.py:153` — same stale `_snapshot.py:92`, same fix.

**Completeness is robust to phrasing, not just to one grep pattern (review round 2).** Because the
"6 files" surface rests on a single verb pattern (`physically impossible|two teams cannot|now
rejects|attack the same`), a stale comment using other words could slip past it. Two broader,
differently-worded sweeps were run to close that gap: (1) `grep -rn 'validate_period_directions'
tests/` returns only the 5 tracking sites already listed (the vaep site correctly does not name the
guard) — no new guard-naming comment; (2) a blanket/uniform/both-teams-`"ltr"` sweep surfaced two
further sites, both **read and found correct-in-context, not stale**:
`tests/tracking/test_obso_orientation_e2e.py::test_oracle_discriminates` (documents uniform-`"ltr"` →
no-flip with the right mechanism — see §2.1) and `tests/tracking/test_line_breaking.py:358` (a plain
SPADL-vs-LTR coordinate statement, unrelated to the guard). So the widened sweep is complete for the
**defect class**, not merely for the phrase.

All of the above is one "don't mislead a future reader about the direction convention" goal; splitting
it from the constant/test would be artificial, and the single-commit structure settles it.

---

## 8. Risks & non-goals

- **Risk: the constant's comment rots.** Mitigated by pointing at symbols (`validate_period_directions`,
  the test name) not line numbers, and keeping the mechanism single-sourced.
- **Risk: the test is vacuous.** Closed by C1's fix — `away.any()` (emptiness), `notna().all()` on BOTH
  legs (skipna), and a mutation that must move the away action from resolved-False to resolved-True.
- **Risk: the widened sweep "fixes" a correct comment.** Mitigated by §7's per-site judgment — the
  `test_hybrid_with_tracking.py` site is explicitly flagged as possibly-correct-in-context.
- **Non-goal: changing any behaviour.** The convention is already correct. If the new test does not pass
  on current code, stop and investigate — do not change `_snapshot.py`.
- **Accepted limit (review round 2, for the record).** The test protects the *seam's contract* —
  `acting_team_attacks_rtl` returns an all-False resolved mask for a snapshot — not the *guarantee that
  every consumer keeps routing through that seam*. Nothing fails if a future consumer bypasses
  `acting_team_attacks_rtl`. This is the right altitude for a doc-hardening cycle: the module is the
  documented SSOT for re-projection and all 7 known call sites route through it (§6 rejects the
  aggregator-level alternative for this reason). Recorded so the boundary is explicit, not discovered.

---

## 9. Open questions — RESOLVED in review round 1

1. **§7 scope** → **Fix in this cycle, widened to all of `tests/` (6 files), per-site judgment** (C2),
   plus the `_snapshot.py:92` citation fix in the authority + `test_off_ball_runs_orientation.py:153`
   (C3). Single commit, so no follow-up split exists.
2. **Assertion style** → **`.all()`/`.any()` form**, because it closes the C1 skipna/emptiness hole,
   not merely for the E712 lint.
3. **Breadcrumb placement** → **moot**: C4 replaces the asymmetric duplicated comment with one named
   constant used at both sites. (The underlying fact stands and is verified: `acting_team_attacks_rtl`
   reads only non-ball rows at `_action_orientation.py:249`, so the ball-row direction at `:163` is
   never consulted — the constant is consumed identically regardless.)

---

## 10. Discipline notes (non-vacuity)

- The test's mutation leg is guarded from BOTH sides *and* against its own vacuity (emptiness + skipna),
  which is the specific hole review C1 caught — a spec whose thesis is non-vacuity had a vacuity hole in
  its one test, now closed.
- The stale-comment sweep is per-site, not mechanical: a directory-agnostic *scope* with per-site
  *judgment*, because "correct in a real-scene context" and "flatly stale about the guard" need opposite
  edits.
- The citation fix proves the spec's own "line numbers rot; point at symbols" principle on the very
  authority it canonizes.
