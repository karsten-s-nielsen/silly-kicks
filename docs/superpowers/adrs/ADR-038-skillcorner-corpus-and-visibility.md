# ADR-038: SkillCorner corpus expansion, visibility surfacing, and expanded-corpus retrain registration

| Field | Value |
|---|---|
| **Date** | 2026-07-14 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; TF-19 GKDV cycle (silly-kicks session) |
| **Supersedes / amends** | amends ADR-024 (SkillCorner keeper-origin), ADR-034 (native SkillCorner builder); ADR-011 paired-test lifecycle re-registered, not overturned |
| **Source spec** | `docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md` (rev 5; five review rounds) |

## Context

The pining API's SkillCorner listing grew from 10 to 108 matches. This ADR records the
decisions that make the 98 new matches reachable and *safely classified*, surface the
`is_detected` flag the pipeline had been discarding, fix two coordinate defects that block
routing SkillCorner through the native bronze builder, and — before any fit sees the new data
— **register** the rules by which an expanded corpus is admitted. It **cites the spec for full
detail rather than duplicating it**; the sections below fix the load-bearing facts, constants,
and contracts in one durable place. Every figure was measured; the probes and raw outputs are
in `docs/research/skillcorner_corpus/`.

PR sequencing (spec §8): **PR-A** (this ADR ships with it) = loader + taxonomy + native
SkillCorner route + pitch fix + visibility + the registered-protocol machinery — **code and
tests only, no weights**. Owner DGX runs follow (Stage A re-baseline → Stage B expansion →
the ghost-GK keeper-CV pair). **PR-B** bundles the final weights. This ADR is PR-A.

## (1) The 98 are owner-tier, not public

`visibility: "private"`, licence *"Restricted; redistribution not permitted"*, S3 prefix
`skillcorner/_private/…`. They are LaLiga 23/24 (37) + 24/25 (34) + UCL 24/25 (14) + 23/24
(13), with **Real Madrid in all 98**. The **public arm therefore stays at 17 matches** (10
SkillCorner open-data + 7 IDSSE), so the prior 4.9.0 / 4.18.0 public-vs-full paired verdicts
are **not invalidated** — nothing was added to the arm they were decided on. What the 98 can
expand is the **owner/full arm: 81 → 179 matches**. The artifacts are the same SkillCorner V3
product as the public 10 (identical frame/player keys, 10 fps exactly, ball `z` present, the
294-column event taxonomy unchanged) — only the containers (CSV→Parquet, JSONL→gzipped JSON
array) and the manifest key names differ.

## (2) The licensing control — keyed on `visibility`, never on provider name

The 98 carry provider `skillcorner`. Both trainers classified public-vs-owner by *provider
name* (`_PUBLIC_PROVIDERS = {"skillcorner", "idsse"}`), so wiring the 98 in naively would
absorb them into the **public** arm — and the model shipped *because* it is reproducible from
redistributable data would be trained on restricted data. This is a **compliance** control,
not an engineering preference, and it lands whether or not the 98 are ever admitted.

**Verified trace (the landmine, alive in the one place a naive fix would not look).**
`_PUBLIC_PROVIDERS` has **six** sites, and one sets the *shipped artifact's label*:
`provset <= _PUBLIC_PROVIDERS` at `train_xshot_occurrence.py:306` (mirror
`train_xcross_attempt.py:391`; both pre-PR at 4.47.0 / e33fca9, deleted here). Trace an `sc_extended`-shaped run — providers `{skillcorner,
idsse}` with restricted SkillCorner rows and no Gradient Sports: `two_candidate` is False, the
`else` branch runs, `provset ⊆ _PUBLIC_PROVIDERS` is True, and a model trained on restricted
data ships **labelled `"public"`**. Verified in code.

**Fix.** `_PUBLIC_PROVIDERS` is **deleted outright** (all six sites — bypassing it would let
the name-keyed rule creep back). Classification moves to `scripts/_corpus.py`:

- `is_public_row` — **fail-closed**: an unknown or missing `visibility` is treated as
  *restricted*. A new match can never silently enter the public arm.
- `artifact_label` — the shipped label is derived from the **visibility composition of the
  ship mask**, not from provider names: `public` iff every row is `visibility == "public"`;
  `sc_extended` iff it contains restricted SkillCorner rows and no Gradient Sports; `full` iff
  it contains Gradient Sports rows.
- `assert_public_corpus` — the public arm must resolve to exactly the known 17 (10 SkillCorner
  ids + 7 IDSSE ids); drift fails the run loudly.

The compliance gate `test_a_restricted_corpus_NEVER_ships_a_public_label` is **driven
red-first against today's code**, which fails it. Missing `visibility` ⇒ restricted is
asserted at the label path, not merely the arm split.

## (3) The clamp split — tracking calls the affine map, never the events clamp

`spadl/skillcorner.py::_transform_coords` scales THEN clamps (`.clip(0,105)/.clip(0,68)`). For
**events** the clamp is harmless — an action's location is on-pitch by construction. For
**tracking** it is destructive, because tracking is full of *legitimately off-pitch*
positions. Measured on match 1886347 (956,076 player rows, 43,458 ball rows): **11.31% of ball
rows** and 0.71% of player rows would be snapped, by up to **9.00 m**, and 1,391 ball rows
(3.2%) lie **beyond the goal line**. A ball nine metres behind the goal becomes a ball on the
goal line — **goal-vs-save, erased** — while its `z` is untouched, so it also acquires an
impossible height-on-the-line. The clamp is unconditional; it would fire on all 108 matches,
including the 97 already at 105×68 that this change was meant to leave untouched.

**Decision.** Split the pure affine map from the clamp:

- `_scale_to_spadl(x, y, L, W)` — the affine part, **no clamp**; the single-sourced coordinate
  truth.
- `_transform_coords` = `_scale_to_spadl` + clamp, unchanged, and remains what the **events**
  converter calls.
- `tracking/skillcorner.py` imports and calls **`_scale_to_spadl` only**.

Tracking inherits the events' *geometry* without inheriting a domain assumption that is false
for tracking. The mutation that kills the guard is named: route tracking through the clamping
`_transform_coords` and the off-pitch-survival test fails.

## (4) Pitch-dimension scaling — single-source the events transform, fail closed on missing dims

The native builder scaled by a fixed `+52.5 / +34.0` offset with **no pitch-dimension input**,
so on a non-105×68 pitch it mis-placed the goal line: 104/106 m → 0.5 m, 103 m → 1.0 m, 101 m →
**2.0 m**. **Four of the ten public matches are 104/106 m**, and the lakehouse consumes this
error today. The correct transform is neither a new one nor kloppy's — it is the **affine part
of the SkillCorner events converter** (`_scale_to_spadl`), because action↔frame co-location
(ADR-028) is the contract that matters. Missing `pitch_length`/`pitch_width` now **RAISE**
(fail-closed); a silent 105×68 default would reproduce the exact defect being fixed, and a
warning is invisible in a DGX batch log. A caller that genuinely knows its pitch is standard
passes `assume_standard_pitch=True` explicitly.

**The 104 m choice rests on PROVENANCE, not on agreement.** Single-sourcing guarantees
*consistency* (tracking equals events), not *correctness* — if events are wrong, both are
wrong together, unfalsifiably. 104 m is what SkillCorner *declares* in `pitch_length` and what
our SPADL events have always used; choosing it preserves the events contract. Kloppy's map is
**non-affine** and assumes a shorter effective length (~103.48 m measured mirror-invariantly
on 956,076 rows, diverging 0.263 m at the goal line; a clean affine fit leaves ~0.14 m
residual, so "effective length" is itself a fit artefact — **nobody has characterised what
kloppy's SkillCorner transform actually does**). Whether SkillCorner's declared length is the
one its coordinates are normalised against is an **open question carried to the SkillCorner
email** (§ open items); until answered, 104 is the registered choice, recorded *as a choice*.

## (5) The detection finding — `is_detected` was in the feed all along

The SkillCorner feed carries a per-player `is_detected` flag (present in **both** container
schemas). The native builder maps it to `visibility`; the **kloppy gateway hard-codes `None`**,
so the research/pining path had been discarding it. Measured on real data: **goalkeepers are
detected in only 19.6% of frames** (outfield 66.6%) — i.e. ~80% of SkillCorner keeper
positions are *interpolator output*, not observation. The GK is the least-detected player
because the broadcast camera follows the ball. This **quantitatively vindicates** two rules
adopted on judgement: the registered "GKDV measurement runs on Gradient Sports frames only",
and ADR-024/PR-S104's distrust of SkillCorner keeper origins. Routing the pining SkillCorner
path onto the native builder (§ decision below) surfaces `visibility` (and recovers `ball_z`)
so the ~19.6% of frames where the keeper is *actually* detected can be used, and the ~80% that
are not can be excluded from targets that depend on real keeper positions (§7).

## (6) S1 recalibration + the per-match rate-gate — and its pinned blindness

The within-pitch invariant only *warned and counted* (invisible in a batch log); its
systematic backstop was a **deferred** rate-gate; and `_TOL_BALL = 30.0 m` sat 3× above the
largest real ball excursion (9.00 m), so it could never fire. Calibrated on the known-good
public 10 (10.0 M rows; calibrating on the 98 would be circular), under the correct §4
transform:

- `_TOL_BALL`: **30.0 → 15.0 m** (public-10 max is 9.00 m; ~67% headroom; zero public rows
  exceed it).
- The deferred per-match rate-gate is **implemented**: `player_frac(>3 m) > 0.005` **or**
  `ball_frac(>10 m) > 0.0005` → the match is **EXCLUDED** (not warned). Margins: 5.8× over the
  worst clean public match (`player_frac` 0.00086); a catastrophic sign/origin break measures
  0.34139, exceeding the player threshold by ~68×. It is mutation-tested three ways
  (catastrophic break excluded; clean match not; the limitation below pinned).

**PINNED LIMITATION — the gate CANNOT detect a pitch-dimension error.** Transforming a real
105 m match as though it were 101 m (the 2 m goal-line error, the worst case in this corpus)
moves `player_frac(>3 m)` from 0.00047 to **0.00095** — *inside* the clean-band worst of
0.00086. Neither can action↔frame co-location see it: events and tracking read the same
metadata, so a wrong `pitch_length` moves them **together**. The only instruments for pitch
dimensions are **provenance (§4) and asking SkillCorner**. Registering a gate that appeared to
cover this, and did not, would be worse than registering none — so the limitation is stated and
pinned by a test (`test_a_pitch_dimension_error_is_INVISIBLE_to_this_gate`) on purpose.

## (7) The registered protocol — locked before any fit sees data

`scripts/_paired.py` is the **executable statement of the rules** (pure functions, table-tested,
no I/O); the prose here fixes the intent. All of it is fixed *now*; results are reported against
it whatever they say.

**Fixed-sequence three-candidate paired test.** Candidates: `public` (17), `sc_extended`
(public + 98), `full` (public + 98 + Gradient Sports 64 = 179). The ship rule is the unchanged
sign-consistency criterion (`Δ_k` strictly positive in ≥ K−1 of K folds AND mean Δ strictly
positive, on the common public held-out fold). The sequence is: test `sc_extended` first; if it
clears it is the provisional ship; test `full` next, which displaces `sc_extended` only if it
clears against `public` **and** the per-fold `full`-vs-`sc_extended` contrast clears the same
rule (**ties go to less data**). If `sc_extended` fails, stop and ship `public` — `full` cannot
ship on this registration even if its deltas would have cleared (that branch is recorded as a
finding triggering a new registration, not a silent loss). The fixed order holds the effective
error rate at the historical single-test level; `sc_extended`-first is justified a priori by the
same-product hypothesis (identical V3 schema, 10 fps, clean geometry).

**Tuning is NESTED inside the outer CV — the M4 fix.** The historical protocol fit every
candidate at public-optimal hyperparameters, handicapping added-data arms (*"more data looks
worse"*). Tuning each candidate once on its full data instead would introduce the *opposite*
bias: `public` would tune on exactly the 17 matches that ARE the evaluation universe — maximal
selection leakage favouring `public`, deciding what ships. So for each outer fold *k*, every
candidate is tuned on its own training data **with fold *k*'s public games excluded**, then
fitted at those parameters and scored on fold *k*. No candidate's hyperparameters ever see the
fold they are scored on (`n_trials=50`, `seed=42`, identical per arm and fold). The
shared-params contrast is still reported (never decides) for comparability with the 4.9.0 /
4.18.0 records. **Power ceiling, registered:** 17 public matches, ~3.4 held-out per fold — the
honest limit on every paired verdict, old and new; the expansion does not fix it (the new data
is not public), and it is stated in the model cards.

**Ghost-GK — detected-keeper targets, keeper-grouped CV, paired admission.** Ghost-GK's target
is the keeper's position; on SkillCorner ~80% of that is interpolator output, so it trains
**only on frames where the keeper was actually detected** (`scripts/_ghost_domain.py`), with a
provider allowlist that **fail-closes** — a detection-aware provider (`skillcorner`) with null
`visibility` raises rather than silently training on the interpolator. CV is `GroupKFold(5)`
grouped by **keeper `player_id`** (half the new cohort's keeper-slots are three Real Madrid
keepers, so match-grouped folds would leak Courtois across train/test). Admission is a **paired
per-fold sign-consistency** rule (ΔMAE on detected-keeper held-out frames only, over a **common
keeper domain** = the 81-match baseline's keepers minus any who appear in the 98; a hard
assertion that no test-fold keeper appears in the 98), replacing the never-costed 0.05 m band.
The surviving-domain size is reported *before* Stage B; too small → underpowered and reported
as such.

**The interpolator-tell refusal is RETIRED (a reported diagnostic instead).** Writing the
implementation proved it dead by construction: admission already *requires* detected-only
improvement, so the refusal branch is only reachable when the fall-through returns the same
verdict. Its mechanism vanished when rev 3 moved training to detected-keeper targets only (the
model never sees an interpolated target). It is replaced by all-frames and interpolated-only
deltas that are *published alongside* the detected-only deltas that decide — informative,
deciding nothing. A "divergence" variant is also rejected: interpolated positions are smoother
and thus easier to predict, so almost any model improves more there, and such a rule would
refuse models that are genuinely better on real keepers.

**Staged retrain (the confound control).** §5's route changes the SkillCorner frames
themselves, so **Stage A** re-baselines xS, xCross **and ghost-GK** on the *same 81-match
corpus* under the new pipeline (apples-to-apples against the 2026-07-13 run), and **Stage B**
fits the three candidates on 179. If Stage A alone moves the verdict, that is a finding about
the pipeline, recorded as such. A registered `ball_z`-leakage check (z-missingness by provider,
each model's z split frequency) runs before Stage B — SkillCorner was the only provider missing
`z`, so the expectation is the channel *shrinks*, but it is measured either way.

**Selection-bias limitation (registered).** SkillCorner detects the keeper when the camera sees
him, which correlates with the ball being near him — so detected-keeper frames over-represent
the *engaged* keeper, precisely the regime GKDV cares less about. Before Stage B is interpreted,
the bias is **characterised** (ball-to-keeper distance and keeper depth, detected vs undetected)
as a measurement and a model-card limitation, not a gate. Choosing observed-but-biased targets
over unbiased-but-fabricated ones is the right trade, recorded as a trade.

## (8) Hyrum surface / consequences

- **Pitch-dimension scaling (§4).** The **lakehouse** re-materializes SkillCorner frames: on
  non-105×68 pitches geometry moves by up to **2.0 m**. It is a **correctness fix** — the
  previous geometry was wrong.
- **SkillCorner → native builder (§5).** The **research/pining** SkillCorner frames change
  (native route: geometry moves ≈ 0.26 m at the goal line on the four non-105 public matches,
  and `visibility` + `ball_z` appear). New training inputs → the owner runs the **Stage A
  re-baseline** before any expansion is judged.
- **`visibility` surfaced.** The pining `xt_gk` SkillCorner keeper-origin resolution now
  activates for the ~19.6% of frames where the keeper is actually detected. No lakehouse impact
  (it already had `visibility`).
- **Corpus taxonomy + ghost-GK size-gate.** Trainer-only; no external impact; closes a
  compliance hole.
- **No change to the shipped library API**, and **no weights ship in this PR** — every item is a
  bronze/loader concern or a correctness fix behind an existing surface. Weights land in PR-B,
  after the owner Stage A/B runs. C4 count stays 28 (no new action-coupled aggregator, backend,
  or trained model in this PR).

## Deviations & findings during implementation

Recorded honestly; each is either a deliberate departure from the spec or a fact the spec's
fixtures had hidden.

**(a) The cache fingerprint is a constant schema token, not the spec-registered live per-corpus
hash.** `scripts/_cache.py` invalidates every pre-Task-11 cache (the load-bearing need — the
DGX caches were already populated under the old schema), but a constant token does **not** detect
corpus *drift* within the same schema. Mitigation, and the DGX runbook: **use a fresh
`--output-dir` per corpus**. The live fingerprint (sorted `(provider, match_id, visibility)`
triples, computable from the cached providers + match_ids + `match_visibility`) is a registered
follow-up. This is a deliberate deviation from spec §3.2's live fingerprint.

**(b) The SkillCorner V3 `timestamp` is a broadcast-clock STRING**, not a float. It is
`"MM:SS.s"`, continuous across periods; the plan's float fixtures hid it. `_sc_timestamp_seconds`
parses it, and the native builder then subtracts `_PERIOD_START_SECONDS` per period → the
period-relative clock the linkage contract (ADR-017) requires (verified).

**(c) The action↔frame co-location e2e's naive 26.98 m median was an orientation confound, not a
geometry error.** SPADL actions are per-acting-team-LTR; the loader emits `absolute_frame`
frames (matching the old kloppy default). The gap is a **180° team-keyed orientation** artefact,
not a coordinate-system bug — orientation-resolved, the residual is **1.196 m < 2 m**. A real
single-axis y-mirror would have shown as a 32–50 m flip and been caught. `ball_z` is recovered
(99.4% non-zero, max 13.97 m). `orientation3_result.txt` carries the same-player 0.0000 m
identity-or-180° reconciliation behind this.

**(d) Unrostered-player guard.** A `player_id` absent from the team-sheet (referee / tracking
artefact) is **dropped**, not stamped `team_id="None"` — a stamped placeholder would corrupt
every team-based feature. This matches the old kloppy path.

**(e) The ghost-GK detected-only filter fail-closes on a genuinely-unknown provider** — correct: it
forces classification before training on possibly-interpolated targets. The registered retrain
corpus (`skillcorner` + `idsse` + `gradientsports`) is all-classified. `metrica` is classified
**fully-observed** (full optical tracking, no detection flag → the mask is all-True, so pre-PR
metrica ghost-training keeps working); its exclusion from the registered GKDV corpora is a separate
corpus-composition decision (Tier-2 anonymized, no roster). Test fixtures were moved to a classified
provider. A fail-fast-**at-startup** refinement (vs mid-run) for a genuinely-unknown provider is a
follow-up.

## References

Spec: `docs/superpowers/specs/2026-07-14-skillcorner-corpus-and-visibility-design.md` (rev 5).
Evidence: `docs/research/skillcorner_corpus/`. Related ADRs: 011 (trained-model lifecycle,
paired test), 017 (period-relative clock), 019 (id-dtype), 024 / PR-S104 (SkillCorner
keeper-origin distrust), 028 (action-LTR), 031 (kloppy CS pin), 034 (native SkillCorner/Metrica
builders), 037 (TF-19 GKDV cycle — PR-A of which this is a sibling). Modules: `scripts/_corpus.py`,
`scripts/_paired.py`, `scripts/_ghost_domain.py`, `scripts/_cache.py`;
`silly_kicks/spadl/skillcorner.py`, `silly_kicks/tracking/skillcorner.py`.
