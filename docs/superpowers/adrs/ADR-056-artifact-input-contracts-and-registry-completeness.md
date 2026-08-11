# ADR-056 — Artifact input contracts and registry completeness (Cycle B)

**Status:** Accepted (4.78.0, PR-S147).
**Spec:** `docs/superpowers/specs/2026-08-05-cycle-b-artifact-input-contracts-design.md`
**Plan:** `docs/superpowers/plans/2026-08-05-cycle-b-artifact-input-contracts.md`

---

## Context

Two defect classes, found while auditing PR 5's own closeout (4.74.0) and related only on the surface.

**A research artifact records `run_commit` — *when* it ran, never *what its numbers depend on*.** So
"is this still valid?" is archaeology. PR 5 had to answer it by hand: `causal/opportunities.py`
builds its covariates from the very extractors PR 5 changed, and every arm config carries `theta`
and `GK_theta` — bearings, exactly what the chirality fix negates. The answer happened to be
reassuring, but nothing about the artifacts made it *derivable*.

**Registry gates were guarded by floors, and a floor cannot detect an omission.**
`tests/scripts/test_provenance_wiring.py` pinned its population with a hand-maintained tuple whose
only anti-rot assertion was `assert len(ARTIFACT_DRIVERS) >= 6`. It passed at 18 entries while
`validate_xcross_causal` was absent and its artifact carried no provenance at all. The same shape
recurred in three more places: ADR-020's dup-`action_id` gate (`>= 21`, with a comment recording it
had already been hand-bumped once), ADR-003's NaN-safety registry (three floors), and three COPIES
of one leakage-guard discovery rule (`>= 10` against a real population of 19).

## Decision

### D1 — Derive the population; justify the exclusions

Every registry gate derives its own population structurally and asserts it EXACTLY, in both
directions. Exemptions are a **three-bucket** shape, not a debt list:

| bucket | meaning | assertion |
|---|---|---|
| the registry | enrolled and derivable | equality, both directions |
| `_NOT_A_DRIVER` | matched the rule, correctly excluded | reason required; names must exist |
| `_UNDERIVABLE` | genuinely enrolled but invisible to the rule | **asserted EMPTY** |

`_UNDERIVABLE` closes the blind spot the other two cannot see: a script that is neither derivable
NOR enrolled is absent from every set, and the equality still holds. That is only unreachable while
every enrolled driver IS derivable — so the day it stops being true, the gate says so. Landed at
**22 enrolled drivers, 5 reasoned exemptions, `_UNDERIVABLE` empty**, and a `SWEPT` registry of
exactly **19** default xfn lists.

### D2 — Single-source the UNIVERSE; let the predicates differ

Asserting "every corpus-walking artifact driver is in ADR-052's population" is **tautological** the
moment both gates call the same predicate over the same script set — it cannot fail. What *can* fail
is one gate re-growing its own `glob`, after which the two universes drift with nothing relating
them. So the reconciliation is structural, not a set relation: `tests/scripts/_script_population.py`
owns the one AST walk, and a gate pins that neither consumer re-grows a glob.

**Content predicates read CODE, never PROSE.** `string_literals()` excludes docstrings because three
separate source-text scans were fooled during this cycle's review — `make_ghost_gk_golden` matched a
`_weights` rule solely through a module docstring, `render_sb360_matrix` matched a `_provenance` scan
through the sentence "No provenance guard, deliberately", and `regenerate_gs_et_native_gk` matched
through a docstring saying it MIRRORS the loader. None carried the literal in code. A dedicated test
pins that the exclusion stays load-bearing, and a second assertion pins that the probe still probes
something — without it, a docstring edit makes the first assertion vacuously true.

### D3 — Contracts declare SYMBOLS, never their current contents

`scripts/_input_contract.py`'s `declare_inputs()` records which symbols a driver's numbers depend on
— covariate tuples, extractor module identity, `GEOMETRY_VERSION` — and digests them. When
`SHOT_ARM_CONFOUNDERS` gains a column the digest moves without anyone editing the driver. That is the
difference between this and "a human writes a list": the residual failure mode is "forgot to
reference a symbol at all", which is narrow and visible, rather than "typed a list that went stale".

Deliberately the same shape as ADR-050's `feature_contract`, and **warn, never raise** — an artifact
is not a serving path.

**KNOWN LIMIT, declared rather than discovered: this catches code drift, not under-declaration.** A
driver that never references a symbol digests stably forever. The cycle then produced a live
near-miss for exactly that limit: because the mechanism declares MODULES BY NAME
(`models=("silly_kicks.tracking._ghost_gk.GhostGkModel",)`), the concurrent
`sb360-degradation-and-port` change to ghost refusal would NOT have moved any digest. It cannot
affect GS, so it is not the escalation trigger — but it is the clearest evidence of where that
trigger will come from.

### D4 — The output side is a separate gate, on the WIDENED glob

The source-side rule says "do not produce artifacts from a dirty tree"; the output-side gate says
"artifacts must carry provenance". They are different assertions, and a stamper needs only the
second — which is why `stamp_feature_contracts` is `_NOT_A_DRIVER` while its OUTPUT is policed. The
gate walks `docs/research/**/*.json` (18 files) rather than the 7 the spec assumed.

### D5 — Provenance had FOUR conventions, not the two §2 predicted

Measured across `docs/research/**/*.json`:

| convention | count |
|---|---|
| top-level `run_commit` | 12 |
| nested `_provenance.commit` | 2 (the rc4 orientation measurements) |
| `training_commit` | the bundled weights, outside `docs/research/` |
| nothing at all | 3 |

Plus one artifact that is not a JSON object at all. The gate picks one canonical shape per surface
and records divergences WITH the location of their real provenance, rather than back-stamping
artifacts to fit a shape invented after they were produced.

### D6 — A fixture generator that cannot reproduce its own fixture is not a generator

`tests/datasets/gradientsports/_generate_synthetic_match.py` emitted **51** events against a
committed **54**. The three missing ones were the ADR-018 goal-capture events — the RE+G own goal,
the CR+G cross-goal and the `nonEvent` disallowed shot — appended directly to the JSON when goal
capture landed, with the generator never updated. **Any regeneration silently deleted all three**,
including the two that `test_owngoal_crossgoal_captured_disallowed_excluded` asserts. Invisible
because nothing in CI ever runs the generator; its own docstring calls it "a maintainer-time tool".

Repaired by reproducing all 54 **verbatim first** — artifacts included: a truncated `ball` dict with
no `visibility`/`z`, `startTime`/`endTime`/`eventTime` holding the raw clock instead of
`200.0 + time_s`, a `startFormattedGameClock` reading `"00:12"` at clocks of 1000/1100/1200 s, and a
trailing newline `main()` never wrote. Byte-identity against the git blob was the acceptance
evidence: **a generator that "improves" the artifact in the same change that restores it cannot show
which of the two edits moved the file.** Only then was the fixture reshaped, and the reshape was
verified purely additive (all 54 prior events byte-unchanged, relative order preserved).

### D7 — The reshape target is the MEASURED distribution, and it makes the guard fire

The fixture carried shots from ONE team only (team 100: 8 in period 1, 1 in period 2), so
`detect_input_convention` returned `convention=None, confidence="low"` on the
fewer-than-two-reliable-groups clause and the converter's `validate_input_convention` deferred
**silently**. CI ran this provider's guard down neither branch — not agreement, not raise.

The binding constraint was therefore never per-group shot count: the fixture already had 10 shots in
one group, AT the `high` threshold. Raising counts, which an earlier reading of the plan called for,
would not have made CI see the case.

Reshaped to 2a's measured distribution (`docs/research/gs_input_convention/`, 64 GS matches): both
teams on OPPOSITE ends within a period, SWAPPING between periods. **Five** shots per group, not ten,
because that is the representative tier — only **6 of 64** real matches reach two `high`-reliable
(>= 10) groups while **50 of 64** reach two at `medium`. Measured both sides:

    before   convention=None            confidence=low     (deferred)
    after    PER_PERIOD_ABSOLUTE        confidence=medium  (classifies, and AGREES)

So CI is green because the pipeline is correct, not because nothing was checked. Three properties
are pinned: the detector classifies and agrees; conversion emits no convention warning on the real
fixture; and the guard RAISES on a planted, genuinely mis-declared convention through the real
converter. The middle one was verified non-vacuous by confirming the same filter DOES raise on the
mis-declared data.

**Deliberately NOT fixed here: `detect_input_convention` rule 1 misfires on sparse groups.** Rule 1
tests `(reliable['side'] == 'high').all()` on groups already filtered to `n >= 5`, so the filter
drops the counter-examples and on a sparse match the rule fires on effectively one team's data
(2 of 36 GS matches).

> **CORRECTION (ADR-059, 4.79.0): the diagnosis in the sentence above is WRONG, and the fix it
> implies would not have worked.** The symptom is not "effectively ONE team's data" — measured, the
> survivor set spans TWO teams (51 in P2, 366 in P1), so the `>= 2 distinct teams` guard the On-Deck
> row prescribed permits the misfire unchanged. It would have shipped, reviewed clean against its own
> rationale, and left the defect live. The real failure is that under `PER_PERIOD_ABSOLUTE` the
> surviving observations are exactly what you would expect — **the evidence does not DISCRIMINATE
> between the hypotheses**, because the observations that would reveal the swap are the ones the
> `n >= 5` filter removed. ADR-059 requires a configuration an absolute convention could not have
> produced, and defers otherwise. Left in place as the historical record: a plausible diagnosis,
> stated with a real measurement attached, that a second measurement overturned.

The fix precedent sits four lines below in the same function — TF-22's guard on
the ABSOLUTE branch (that part HELD: it is ADR-059's clause (b), now single-sourced as
`_a_team_spans_periods`). It is out of this cycle's charter, touches 6 providers and 23 test call sites,
and risks silently downgrading StatsBomb/SkillCorner to ambiguous — a coverage loss that shows up as
a gate quietly not checking rather than as a red test. Registered as the top On-Deck item with its
full measurement.

### D8 — A synthesized row must carry its OWN end, not its parent's derived one

Exposed by D7's reshape. Both GS synthesis sites `.copy()` a pass-class parent AFTER
`_derive_end_coordinates` has rewritten that parent's end to the next action's start, then relabel
the copy `foul` or `shot` — **neither of which is in `_DERIVE_END_TYPE_IDS`**, i.e. both keep
`end == start` by contract. Copying wholesale lands a pass-class destination on a type that must not
have one, and reads downstream as a shot or foul that travelled.

Measured on the committed fixture: **all three** synthesized rows carried an inherited end. The two
fouls had been wrong since the row was introduced — their parents are mid-period passes, so a next
action always existed — and were invisible because
`test_shots_tackles_keeper_saves_end_equals_start` does not include `foul` in its type set. The
cross-goal shot read correctly only by accident: its parent was the last period-1 event surviving
exclusion, so derivation had nothing to reach for. Adding any later period-1 event exposes it.

Single-sourced onto `_reset_synthetic_end`, because the *reason* is what needed single-sourcing. The
regression test asserts BOTH kinds are present (non-vacuity) and that the parent cross STILL carries
its derived end — the fix must not disable derivation wholesale.

## Consequences

* **GS conversion values change for synthesized rows only.** Cross-goal shots and synthesized foul
  rows now carry `end == start`. Real (non-synthetic) rows are byte-identical. Small-N but
  corpus-wide: in real data a cross-goal is essentially never period-last, so the shot case was
  wrong every time. A GS consumer persisting `end_x`/`end_y` for synthesized rows should
  re-materialize.
* No retrain otherwise. Phases 1 and 2 are gates; the two bundled attempt models gained a
  `training_commit` metadata stamp with `model.json` hashes UNCHANGED (ghost-GK already carried one
  from ADR-050).
* Four research artifacts now declare an `input_contract`; the rest are recorded with the location of
  their real provenance rather than back-stamped.
* C4 unchanged — this cycle adds no action-coupled aggregator.

## What we got wrong

* **The spec's independent source for K2 was the wrong surface, and its headline finding an
  artifact.** It paired `dir()` against `features.__all__` and cited a four-way disagreement as proof
  the independent source was needed. All four names are exported at PACKAGE level; the disagreement
  was an artifact of comparing against a narrower surface. Building it as written would have
  manufactured four false findings and four pointless `__all__` edits.
* **Two of three pin surfaces would have been vacuous.** `silly_kicks.spadl.utils` has no `__all__`
  at all, and neither module surface exports any `add_*`.
* **The spec's three `_UNDERIVABLE` entries were an artifact of a too-narrow rule.** It measured
  `calibrate_xt_bandwidth`, `train_gk_completion` and `train_gk_retention` as underivable by any
  `--out`-keyed rule and concluded the bucket must be non-empty on day one. Broadening the predicate
  to any `--*out*` flag plus a bundled-weights clause makes all three derivable, so the bucket landed
  EMPTY — which is what lets `test_UNDERIVABLE_is_empty` be an assertion rather than a waiver.
* **A warn-only gate passes whether or not the warning fires.** The first draft put the comparison
  inline in the test body; a reviewer reimplemented it INVERTED with `warnings.warn` DELETED and both
  tests still passed. Root cause: the detector was not a function, so nothing else could call it.
* **`ADR-054` was claimed mid-flight by a concurrent session**, in 23 files. The response was to
  DE-number to a slug, not to renumber to the next free slot — reaching for the next slot re-enters
  the same race. Two number types needed two sweeps: a stale `4.76.0` VERSION claim was hiding in a
  comment, invisible to any search for the ADR number.
* **A non-counter dict destroyed a corpus aggregate.** `input_contract()` was placed in a per-worker
  manifest, which is a cross-worker COUNTER surface; `aggregate_manifests` called `int()` on every
  dict value and raised — after ~60 h of completed work. Both defects fixed: the contract moved to
  the cited artifact, and a dict is now treated as counters only if every value is numeric, otherwise
  dropped and REPORTED.
