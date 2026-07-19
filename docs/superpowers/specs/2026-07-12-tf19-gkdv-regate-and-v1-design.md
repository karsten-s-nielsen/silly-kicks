# TF-19 GKDV — Gate Re-run + Physics-Arms v1 — Design

**Date**: 2026-07-12
**Amended**: 2026-07-18 (see "Amendment 2026-07-18" below)
**Reviewed**: 2026-07-18, cross-session review **round 1** — 3 blockers + 10 should-fixes,
ALL accepted and applied (`[REVIEW ROUND 1 …]` tags); the review independently re-verified
every Chesterton hazard in amendment cause (A) against the shipped code and confirmed each
holds. **Round 2** — 3 blockers + 3 should-fixes, ALL accepted and applied
(`[REVIEW ROUND 2 …]` tags). **Round 2's headline: two of round 1's own fixes interacted to
falsify a third claim** (the B1 outcome change made §2.2's "constructible purely as builder
arguments" false), and the PR-3/PR-3b split from B2 was fixed at its instance but not
propagated to §6.4 Layer 4, §7 or §8. Both are recorded in place rather than silently
corrected, because the failure mode — *a fix that invalidates a verified claim elsewhere* —
is the one this document's amendment convention is least able to catch on its own.
**Status**: PR-1 and PR-2 SHIPPED; PR-3 (gkdv package) not started — this document is the
PR-3 design, amended to current reality and awaiting review
**Scope decisions (owner)**: re-gate the attempt-probability arms first AND build the
gate-independent physics arms in parallel; frozen rule governs the xCross probe re-run,
the new xS probe registers its own rule; Approach A architecture (new `silly_kicks/gkdv/`
package + pre-planned promotions).

---

## Amendment 2026-07-18 — what changed and why

This spec was written before any of it shipped. **PR-1 landed as 4.47.0 (ADR-037) and PR-2
as 4.51.0 (ADR-040)**, the xCross decision-table row FIRED, and two pieces of information
arrived that post-date the document. The sections below are amended in place; every change
carries an inline **[AMENDED 2026-07-18]** tag so a reviewer can find them without a diff.
Amendments are grouped by cause:

**(A) The code diverged from the spec during PR-1 — correcting the spec, not the code.**
Three are Chesterton's-Fence hazards where a PR-3 author following this document literally
would *undo* a deliberate hardening or duplicate a frozen function: the xS ratio prong was
STRENGTHENED (§3.1), the decision table lives in `_model_eval.py` and NOT in the
PR-3-owned `_validate.py` (§3.5), and the probe/engine boundary became a
**typed cross-package data contract** rather than a function call (§2.2, §4).

**(B) Facts on the ground moved.** The retrains ran and the cross verdict is recorded
(§3.5, §9); the ghost MAE figures are superseded (§3.1); the C4 aggregator count is 29 after
TF-49, not 28 (§2.3); the known-failure xS e2e passed on the corrected weights (§9).

**(C) Two mechanisms in §5 were verified against source and found mis-stated** — the channel
through which the keeper enters the threat arm, and the sufficiency of the DAS pinning guard.
Both reductions survive; the stated reasons did not. One is a live silent-Δ≡0 hazard.

**(D) A rival hypothesis for the attempt arm's sub-floor result.** The gate's recorded
consequence assumes **H1: the models are under-featured on GK position.** A sibling
possession-value metric in this codebase was independently shown to read inert because it
was **mis-specified** — it rewarded a behaviour good keepers do not perform — and a
cross-check on our own GS-WC2022 and SkillCorner cohorts reproduced the generalizable half:
the behaviour that predicts possession retention is not rewarded by our shipped `xt_gk`
column (per-keeper correlation ≈0 on GS, negative-trending on SkillCorner;
`docs/research/` cross-check, 2026-07-18). That raises **H2: the deterrent construct is
measured on the wrong outcome axis** — keepers may not suppress attempt *probability*;
they may deny space and lanes, with the attempt made anyway from a worse position.
A code check found this is not idle speculation: **only 2 of the 27 faithful xS features
respond to keeper position at all** (§3.1). §6 as written cannot distinguish H1 from H2,
and the shipped verdict function structurally encodes H1. §6 is amended to make the
distinction a pre-registered read-off rather than a post-hoc argument.

---

## Executive summary

TF-19 (GKDV — GK Deterrent Value) is the Layer-3 capstone of the GKDV research program:
a per-frame measure of how much the defending goalkeeper's actual position depresses
opponent attempt probabilities relative to a league-average "ghost" GK in the same frame
state. It has been gated since 4.18.0 by `tf19_ready=False` from the xCross GK
substitution probe. This investigation (twice cross-session-reviewed) established three
facts about the shipped Layer-2 models — none of which is "the gate is stale":
(a) **the weights are chirality-mis-served in production today** — trained in a
y-mirrored convention and served on y-correct frames, xS reads 12 of 27 features
sign-inconsistently (xCross 3 of 16) on every y-correct provider, for consumers opted
into `pre_shot_gk_full_default_xfns`, and `load()` is structurally blind to it — a
correctness bug that mandates the retrain regardless of any gate outcome; (b) **the
paired test that excluded GS from training was a mixed-chirality comparison** (clean
native GS superimposed on mirrored SC + internally-corrupted IDSSE), so the corpus
decision must be re-taken on corrected frames; (c) **the 4.18.0 probe measurement
itself was NOT attenuated** — a uniformly mirrored world is an isometry of the frames-
only feature space, so the frozen gate is EXPECTED to hold (~0.001, ~10× under the
floor) and no cycle outcome is premised on it flipping. The shot arm has never been
measured at all; the model-free causal null (~83% clean GS spells) is expected to
stand for the cross arm. Detail and the retraction record: §1.2.

This cycle therefore runs two tracks:

1. **Re-gate track** — retrain xS/xCross/ghost-GK on the corrected corpus as a
   correctness fix (re-running the pre-registered public-vs-full paired test, since
   GS's exclusion was decided on a mixed-chirality comparison), harden `load()` with a
   chirality fingerprint, re-run the frozen xCross probe apples-to-apples, add the
   never-measured xS substitution probe under a newly registered dose-banded,
   placebo-banded rule, and run a causal shot arm. A pre-registered decision table
   converts the results into the attempt-arm verdict; the expected outcome is that the
   cross gate holds and the value of the track is the correctness fix, the clean
   paired test, and the first xS measurement.
2. **Physics-arms track** — ship GKDV v1 on the two gate-independent counterfactual
   arms already named in the TODO entry (ΔDAS and ΔGK-threat-suppression — the
   blocking-score idea in its algebraically reduced, honestly-named form, §5), built on a shared
   ghost-substitution engine, with a pre-registered validation harness (keeper-level
   ICC primary; amended expected-sign panel; conceded-xG exploratory). Arms are reported
   separately per keeper; no cross-arm composite in v1.

Honest-reporting discipline throughout (ADR-036 precedent): all gate constants locked in
code before any run; results reported as they land; verdicts recorded either way.

**[AMENDED 2026-07-18] Status of the two tracks.**
Track 1 (re-gate) is **COMPLETE except the xS probe**. PR-1 shipped 4.47.0; the DGX retrains
ran; PR-2 shipped 4.51.0 with corrected weights, fail-closed chirality enforcement on
`load()`, and an xgboost 2.x/3.x `base_score` guard (a real serialization bug the chirality
golden caught). Prediction (c) held: the frozen xCross probe did NOT flip — it strengthened
to ratio 2.21× (clearing the ≥2.0 prong) but missed the 0.01 absolute floor at 0.009697, so
`tf19_ready=false` stands and `regate_verdict(arm='cross', …)` returned **`gated_clean_fail`**.
The correctness fix, the clean paired test, and a sound instrument are the delivered value,
exactly as the summary anticipated. **The xS arm remains unmeasured**: its probe is
dose-banded and consumes ghost targets from the §4 engine, which is PR-3. It is therefore
*PR-3-gated*, not merely pending.
Track 2 (physics arms) is **entirely unbuilt** — `silly_kicks/gkdv/` does not exist. PR-3 is
this document's remaining scope.

---

## 1. Context

### 1.1 TF-19 definition (TODO.md)

For every frame in possession in the final third, compute
`Δ_attempt(action) = P(action | actual_GK) − P(action | ghost_GK)` for
action ∈ {shot, cross, key_pass}, weight by outcome value, and sum across the build-up
window. Negative GKDV ⇒ deterrent effect. Depends on TF-15 (gk_influence), TF-16 (xS),
TF-17 (xCross), TF-18 (ghost-GK) — all shipped. The TODO entry also names two non-model
counterfactual arms: `Δ_DAS = DAS(actual_GK) − DAS(ghost_GK)` (post TF-28) and
`Δ_blocking = blocking_score(actual_GK) − blocking_score(ghost_GK)` (post TF-30).
(The Δ_blocking formulation is quoted from the TODO as-is; §5 ships it in its renamed,
algebraically reduced two-surface form — do not implement the four-surface version.)

### 1.2 The stale gate

- **Gate record** (`silly_kicks/tracking/_xcross_weights/default/metrics.json`):
  `gk_median_abs_delta=0.00107` vs nearest-defender `0.00041` (ratio 2.6×, passes) but
  below the 0.01 absolute floor by ~9.3× ⇒ `tf19_ready=false`. Rule frozen at
  `_xcross_eval.py:25-26` (`TF19_PROBE_RATIO=2.0`, `TF19_PROBE_ABS_FLOOR=0.01`),
  unit-pinned by `tests/tracking/test_xcross_eval.py`.
- **What is actually wrong with the shipped weights (corrected after cross-session
  review + source verification; the original "attenuated probe" mechanism is RETRACTED)**:
  1. **PRODUCTION MIS-SERVING (primary — a correctness bug, not a research question).**
     The models were trained in a y-mirrored convention and are served on y-correct
     frames: GS was always native-clean (mis-served from day one), SC/Metrica are clean
     since 4.29.0/4.30.0. Under the mirror, xCross negates 3/16 features
     (`ball_theta`/`gk_theta`/`gk_lateral_offset`) and **xS negates 12/27** (θ, GK_θ,
     and all 10 Def/OffAngle bearings) — those features are read sign-inconsistently on
     every y-correct provider today. Blast radius: `xcross_attempt_xfns` +
     `xshot_occurrence_xfns` are wired into `pre_shot_gk_full_default_xfns` (+ atomic
     mirror), not `tracking_default_xfns` — opted-in consumers only, but for them the
     columns are wrong. `load()`'s guard cannot catch this (pitch-dims fail-closed;
     `geometry_version` warn-only and metadata-vs-constant — structurally blind to input
     chirality). The retrain is therefore a **correctness fix with a Hyrum/retrain
     trigger**, independent of any gate outcome; PR-2 additionally records a
     **y-convention/chirality fingerprint in `metadata.json` and hardens `load()` to
     fail closed on mismatch** (same class as the existing pitch-dimension guard).
  2. **The 4.18.0 probe measurement itself was NOT mirror-attenuated.** Verified: both
     extractors are frames-only (no action coordinate enters), the goal map is x-only,
     `GOAL_Y=34` sits on the mirror axis, all domain gates and the displacement panel
     are y-symmetric, and the probe frames came from the same extraction stream as
     training — a uniformly mirrored world is an isometry of the feature space, and a
     boosted tree learns the conjugate partitions. The measured 0.00107 is what
     corrected frames would have produced, up to float noise.
  3. **Residual training-data defect (secondary, magnitude unknown)**: IDSSE (~41% of
     the shipped training matches) was NOT a clean mirror — its frames were internally
     corrupted (ball at the wrong pitch END at shot times, an x-error no mirror
     absorbs; ~5.7 m post-flip residual; root cause never isolated). Ball-anchored
     features, the wide-area domain gate, and carrier inference were noised for the
     affected IDSSE frames. Ghost-GK's `gk_y` label was additionally mirrored for
     17/81 of its training matches.
  4. **The paired test that excluded GS was a mixed-chirality comparison.** The full
     candidate superimposed clean native GS on mirrored SC + corrupted IDSSE and lost
     all 5 folds (verified: pure data-effect at shared public-optimal params) — sign
     conventions genuinely conflicted ACROSS providers in that candidate. This is the
     defensible reason to re-run the paired test on corrected frames (§3.4); GS's 64
     clean elite-keeper matches may enter.
  Weight provenance: xS 4.9.0, xCross 4.18.0, ghost-GK 4.14.0 — none retrained after
  the fixes. The probe's 200 sample frames were almost certainly SkillCorner (the
  4.18.0 provider order is unrecorded — a provenance gap this cycle closes).
- **Expectation (pre-registered, honest)**: a corrected-frame retrain should be
  EXPECTED to reproduce a probe result near 0.00107 — still ~10× under the frozen
  floor. The isometry argument removes any basis for predicting a flip. If the gate
  must pass, the honest lever is the one 4.18.0 already named — GK feature
  engineering — which is out of scope for this cycle. The physics-arms v1 does not
  depend on the gate either way.
- **What is NOT stale**: the causal-harness null (`gk_clears_placebo_band=False`,
  ablation shift 0.0179 inside placebo p95 0.0239). It is model-free (propensity
  matching on raw frame features) and ~83% of its 23,966 spells were y-clean GS. Its
  0.0179 shift cleared the 0.01 floor — the placebo band was the binding criterion.
  The cross-arm causal verdict is expected to stand after the retrain.
- **Never measured**: the shot arm. No xS substitution probe or xS-side causal ablation
  exists anywhere — theoretically the strongest deterrence arm, and the single most
  decision-relevant missing number.
- **Precedent for gate flips on data quality**: the ADR-019 canonical-id fix already
  flipped `gk_clears_placebo_band` once (spurious True → correct False).

### 1.3 Lessons imported from xT-GK v2 (ADR-036, 4.45.0)

**Resolution (4.46.0, PR-S113, published 2026-07-12)**: the resolved-origin re-run
CONFIRMED the contamination-biases-ICC-toward-zero mechanism — **the keeper-flat leg of
the 4.45.0 verdict did not survive the fix**. Leg-2 (corrected coords + retrained ρ):
GS v2 ICC −0.0020 → **+0.0256** (now above v1's 0.0193); SC 0.0109 → **0.0147**
(≈ v1's 0.0176). The outcome-AUC leg stands (GS −0.1474 / SC −0.0268; v2 remains not
construct-validated on that lens, now on trustworthy numbers). TF-19's data path was
never affected (frames-based; nothing routes through `flat_zones`, `load_xtgk_cohort`,
or ρ), and 4.46.0 touched only `xtgk/` + its loaders — §1.2's surviving findings (the
production chirality mis-serve, the mixed-chirality paired test, the IDSSE residual
corruption) are unaffected by it. Consequences for this design:

- Keeper-flatness downgrades from "recurring failure mode" to "primary risk with a
  measured counter-example": clean GK-distribution geometry carries a small but real
  between-keeper signal, and the action-level ICC instrument is validated as sensitive
  in both directions (it detected the contamination AND the recovery). Keeper-level
  discrimination stays a PRIMARY pre-registered criterion.
- **Effect-size anchor (new)**: observed between-keeper variance shares on clean
  GK-distribution metrics are ~0.015–0.026 (39–54 keepers, min-n 20). GKDV's ICC
  null-band and min-n should be designed to resolve signals of that order — a
  pre-registered gate that cannot detect ICC ≈ 0.02 would fail keepers with real
  signal by construction.
- CV-of-per-keeper-means is a near-zero-mean artifact — excluded as a criterion;
  action/frame-level ICC with unbalanced-n correction is the instrument.
- A composite can degrade below its best single term (v2 component-AUC ladder) —
  GKDV v1 ships arms separately; composites are a validated follow-up.
- Absolute-effect floors must be calibrated to the intrinsic magnitude of the quantity
  before registration; scale-free relative criteria + placebo bands are the honest
  idiom for small-probability quantities.
- Stratification/conditioning variables need non-degenerate support checks before a
  failed gate is read as absence of signal (the andrienko_oval 52%-exact-zero STOP);
  pressure, where used, is pinned to `bekkers_pi`.

### 1.4 Novelty check (2026-07-12)

The 2026-05-01 literature gap substantially holds: no published GK-evaluation framework
measures positioning-as-deterrent. Closest adjacents: DEFCON-GNN (arXiv:2512.10355,
per-defender EPV-reduction, prevention-focused, not GK-positional); Groom et al.
(arXiv:2601.00748, role-conditioned ghosts for corner defenders — TF-46); and the xS
paper itself (arXiv:2512.00203 v2) now suggests mirroring its framework "to quantify
shot and goal suppression with credit for goalkeeper positioning" — the gap is being
circled, which argues for landing GKDV v1 sooner rather than later.

### 1.5 Housekeeping corrections bundled into this cycle

- TODO.md line 28 "UNBLOCKED"-vs-gate contradiction: FIXED in 4.46.0 (landed) — the
  entry now correctly reads "GATED, not unblocked" and flags the xS arm as unmeasured.
  PR-1 updates that prose once more to point at THIS re-gate cycle as the path to a
  fresh verdict.
- Ter Stegen comes off the sweeper-keeper expected-sign panel (0 WC2022 minutes);
  in-corpus replacements: Alisson (4 starts), Neuer (3), Onana (1).
- Metrica is excluded from all GKDV corpora (keeper-ID contamination; the
  derive-once-per-match follow-up has not shipped). It is not in the pining corpus
  anyway (81 = 64 GS + 10 SkillCorner + 7 IDSSE).

---

## 2. Architecture (Approach A)

### 2.1 New package `silly_kicks/gkdv/`

Hexagonal, mirrors `xtgk/`; NOT imported by bare `import silly_kicks`.

```
silly_kicks/gkdv/
  __init__.py     # public surface
  _engine.py      # ghost-substitution counterfactual engine
  _arms.py        # ΔDAS + ΔGK-threat-suppression delta computers; Δattempt slot reserved
  _metric.py      # per-frame → per-keeper aggregation (frames-resolved GK player_id)
  _validate.py    # pre-registered validation harness (pure; I/O in scripts/)
```

### 2.2 Pre-planned promotions executed this cycle

- **Probe core** `tracking/_xcross_eval.py` → `tracking/_model_eval.py` (stays PRIVATE
  to `tracking/`): model-agnostic `gk_substitution_probe(model, extract_fn, domain_fn,
  frames, ...)`. The xCross wrapper stays byte-equivalent (frozen constants stay where
  `test_xcross_eval.py` pins them); a new xS wrapper carries its own registered
  constants. Full public promotion of the eval home stays deferred until a consumer
  outside `silly_kicks` exists (deliberate, narrower reading of the docstring's
  "2nd consumer" trigger).
  **[AMENDED 2026-07-18 — SHIPPED SIGNATURE DIFFERS; do not "restore" the one above.]**
  PR-1 shipped **string dispatch**, not callable injection:
  `substitution_deltas(model, frames, *, arm: str, mode: str, targets=None, …)` with
  `arm ∈ {"xcross","xs"}` resolved internally by `_resolve_extractor(arm)` /
  `_extract_kwargs(arm, …)`, plus a `PROBE_WRAPPERS` registry
  (`tracking/_model_eval.py`). This is house-style dispatch (matches `xthreat`'s
  transition family and the pressure-method surface) and it is what keeps the
  `tracking → gkdv` import direction clean. **Consequence PR-3 must accept:** `gkdv`
  CANNOT register a new probe arm by passing callables — a future Δattempt arm requires
  an edit inside `tracking/_model_eval.py`. That is a deliberate cost of the layering
  rule, not an oversight.
  **The load-bearing layering contract PR-1 introduced, absent from this document as
  written:** the probe consumes ghost positions **as DATA** — a `targets` DataFrame that
  crosses the package boundary and is contract-validated fail-loud — precisely so
  `tracking/` never imports `gkdv/`. §4 is amended accordingly: the engine does not call
  the probe and the probe does not call the engine; **the engine emits a frame satisfying
  `_TARGET_COLUMNS` and the runner (in `scripts/`) joins them.**
- **`silly_kicks/_causal/` → public `silly_kicks/causal/`** (ADR-015's "one move"):
  `matching.py` unchanged; `opportunities.py` parameterized with the FULL builder
  surface enumerated NOW (the current signature provably cannot express its own second
  consumer): `extractor_fn`, `domain_fn`, treatment `type_ids`, outcome `type_ids` +
  **`result_ids`** (§3.3), confounder lists, window seconds. Acceptance criterion,
  tested: the §3.3 shot-arm configuration is constructible purely as builder arguments.
  The xCross configuration is preserved as the default-constants path so
  `tests/causal/` known-truth gates stay green unmodified, and a regression guard
  asserts the parameterized builder reproduces the xCross default byte-identically.
  The newly-public `causal/` registers in the Examples/`_PUBLIC_MODULE_FILES` gate
  from day one — the same treatment §7 mandates for `gkdv/` (no new xtgk-style gap).
  **[AMENDED 2026-07-18 — SHIPPED COMPLETE in PR-1; this bullet is no longer PR-3 work.
  Added at review round 1 (S10), which correctly noted the sibling bullet got a SHIPPED
  tag and this one did not.]** Verified present: `silly_kicks/causal/{__init__,matching,
  opportunities}.py` (`_causal/` gone), with the **full builder surface** landed as a
  frozen `OpportunityConfig` plus `xcross_config()` / `shot_arm_config()`; `config=None`
  reproduces the legacy xCross path byte-identically (regression-gated); and the package
  is registered in `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`. **This
  matters beyond bookkeeping: §6.4 Layer 2 depends on constructing its configuration
  purely as builder arguments, and that acceptance criterion is exactly what makes Layer 2
  buildable without reopening `causal/`.**
  **[REVIEW ROUND 2 — N1 CORRECTION: the sentence above is now FALSE and is retained only
  to show what changed.]** Round 1's B1 fix replaced Layer 2's outcome with an
  unconditional **spatially-filtered** indicator (`Y_close_attempt`), and the shipped
  `OpportunityConfig` / `_label_outcome` have **no location axis** — verified in source.
  **Layer 2 therefore DOES reopen `causal/`, by exactly one additive field**
  (`outcome_max_distance_m: float | None = None`, legacy-preserving), scoped to PR-3b. The
  rest of the promotion's value stands: treatment, outcome types/results, window,
  confounders and extractor are all still expressible as builder arguments, so this is a
  one-field extension rather than a redesign.

### 2.3 Documentation / infra impact

New ADR-037 (GKDV composition + re-gate policy, covers both tracks). ADR-015 status →
promoted. C4 DSL edit + regen (new `gkdv` element, same treatment as `xtgk`); gkdv adds
no `add_*` action-coupled aggregator, so the cap-tested count is **UNCHANGED by this cycle**
(definition: `len(add_* in tracking.__all__) − 1`, excluding the roster helper
`add_gradientsports_player_ids`). **[AMENDED 2026-07-18]** the baseline was 28 at spec
time and is **29** after TF-49 packing (4.50.0, ADR-039). **PR-3 must preserve whatever
`tracking.__all__` yields at PR-3 time — re-derive it, do not trust any figure written
here.** **[REVIEW ROUND 1 — B3]** the earlier wording "must PRESERVE 29" is deleted: it
becomes FALSE if the parallel session's PR-S119 (which adds `add_off_ball_run_values`,
29→30) lands first. The re-derive instruction is the durable form.
Measured on the current DSL: the `tracking` container description is **191/200 chars —
9 characters of headroom** — and contains the literal string `29 action-coupled
aggregators`; two other boxes sit at **exactly 200**. So adding the gkdv/ADR reference
means **trimming, not appending** (`test_c4_dsl_description_cap`; that gate has already
bitten once, in 4.51.0). See §9.1 for the resulting cross-session merge hazard. NOTICE
entries: Le et al. 2017 (ghosting); DEFCON-GNN comparator. TODO reconciliation rides in
PR-1 (no standalone doc commits).

### 2.4 Ownership boundary

No modifications to `scripts/_loader_pining.py` or `scripts/calibrate_*`. Retrain
scripts get only minimal re-gate edits (probe-sample provenance recording). The
validation script reads keeper names directly from the Drive roster JSONs rather than
plumbing names through the shared loader.

---

## 3. Re-gate track

### 3.1 xS substitution probe (new)

- Generalized core over attacking-third frames (xS's `_ATTACKING_THIRD_M=35.0`
  predicate), frames-level substitution of the defending-GK row (engine-consistent even
  though xS's GK dependence is only `GK_r`/`GK_theta`).
- **[AMENDED 2026-07-18] FEATURE-CHANNEL ASYMMETRY — registered as an a-priori
  expectation BEFORE the xS probe runs. Changes NO threshold; changes how a fail is
  READ.** A source check confirmed and sharpened the parenthetical above. Of the 27
  faithful xS features, **exactly 2 respond to keeper position** (`GK_r`, `GK_theta`).
  `openGoal` is computed from a defender array built as
  `defending = players[is_gk_team & (~players_is_gk)]` — the keeper is **excluded as an
  occluder**, which is also physically questionable (a keeper on his line does not
  reduce "open goal") — and the same GK-excluded array feeds all 10 `DefDist_k` /
  `DefAngle_k` pairs. **The consequence for the ratio prong is a structural bias against
  the GK**: the paired-vector nearest-defender control moves a player who enters up to
  11 of 27 features, while the GK move enters 2 of 27, and prong 1 compares the two
  medians directly. **Therefore: a low or failing xS reading is PREDICTED BY THE FEATURE
  DESIGN and must NOT be read as "the keeper does not deter."** It is evidence about the
  instrument. This is registered here so the reading is pre-committed rather than
  argued after the number lands; see §6 for how H1 (under-featured) is separated from
  H2 (wrong outcome axis), and §10 for the `openGoal` keeper-exclusion follow-up (a
  change there is a full ADR-011 retrain cycle and is NOT in PR-3).
- **[AMENDED 2026-07-18] Prong asymmetry with the cross arm, never previously stated**:
  the xS rule has **no absolute-magnitude floor**. xCross gates on ratio AND
  `TF19_PROBE_ABS_FLOOR = 0.01` — and the floor was the *only* prong it missed
  (0.009697). xS gates on ratio + dose-response + support floors alone. An xS "pass" and
  an xCross "pass" are therefore **not the same standard of evidence**, and given the
  2-of-27 channel above, an xS pass on a relative prong alone should be reported with
  that caveat attached rather than treated as symmetric corroboration. Do NOT add a
  floor to xS now — the constants are frozen and pre-registered; record the asymmetry
  in the report instead.
- **The accuracy-vs-magnitude paradox (stated up front, it shapes the rule)**: the
  ghost's held-out MAE is 1.07 m **[AMENDED 2026-07-18: superseded — the Stage-B
  179-match retrain bundled in 4.51.0 serves 1.108 m (full) / 1.185 m (default). The
  argument is unaffected and in fact strengthens: both figures remain below the frozen
  2 m step. Quote the model card at PR-3 time rather than any number in this document.]**
  — BELOW the frozen panel's smallest step (2 m). A
  ghost good enough to be credible is, by construction, usually too close to the
  actual keeper to move a piecewise-constant boosted surface; retraining the ghost on
  corrected data will shrink typical displacements further. A naive median over all
  ghost-substituted frames therefore under-perturbs by design and cannot be read as
  "the surface is GK-insensitive" — the deterrent signal lives in the displacement
  TAIL where the actual keeper deviates from the league-average ghost.
- **Registered rule (constants in code, PR-1, before any run). The `ready` boolean is
  the AND of every registered prong below — nothing here is advisory prose**:
  1. Ratio prong: **[AMENDED 2026-07-18 — the shipped rule is STRONGER than this
     registration; the code is correct and MUST NOT be relaxed back to match the
     original sentence.]** As shipped in PR-1: `gk_med ≥ XS_PROBE_RATIO ×
     max(nearest_def_median, placebo_p95)` — twice the LARGER of the nearest-defender
     control and the placebo p95 band, on the gated stratum, under paired-vector
     controls (see 2). The strengthening is recorded in ADR-037 item (13), which also
     records that the separate explicit `gk_med > placebo_p95` conjunct was dropped as
     *implied by* the strengthened form. The original registration read "≥ 2.0× the
     nearest-defender control" only. `XS_PROBE_RATIO = 2.0` is frozen
     (`tracking/_model_eval.py`); a PR-3 author who "fixes" the code to match the old
     prose would silently weaken a pre-registered gate.
  2. **Placebo prong, fully mechanized (the shipped probe's control band is ALREADY
     the degenerate case — `random_band_median_abs_delta = 0.0` in the 4.18.0
     record)**: the nearest-defender control AND the placebo outfielder are displaced
     by the SAME per-frame ghost-displacement vector on the SAME banded frames
     (paired-vector controls — no dose mismatch by construction); the placebo
     statistic is the IDENTICAL gated functional computed per replicate; the band is
     the p95 over R registered placebo replicates (R a registered constant). An
     M2-analog non-degeneracy guard: placebo p95 = 0, or placebo zero-fraction above a
     registered ceiling ⇒ "no valid placebo comparison" — fail-closed, never a pass.
  3. **Dose-banded go/no-go, gated on the TRUSTED stratum**: the registered statistic
     is the median |ΔP(shot)| among frames with |ghost − actual| ≥ 2 m that are
     UNCLAMPED and INSIDE the ghost's training box — OOD/clamped frames are
     report-only, so a registered PASS cannot be driven by the ghost's least-trusted
     outputs. A registered **minimum band n and per-stratum minimum** apply: shortfall
     ⇒ the §3.5 outcome "unmeasurable at this dose" (a distinct verdict, NOT a fail —
     the §1.3 support-check lesson). A 2/3/4 m threshold ladder is reported; only the
     registered 2 m band gates. The full unbanded distribution AND the fixed ±2–4 m
     panel are reported for comparability, never gated on.
  4. **Dose-response prong (registered constant, ANDed into `ready`), cluster-EXACT
     (plan-review N1 amendment)**: per-game Spearman ρ of |ΔP| against |ghost − actual|,
     then a sign-flip permutation test across GAME-level ρ's (mean ρ > 0, p < .05) —
     ragged games handled natively, nothing truncated, the game is the unit of
     inference. Constant-response games count as ρ = 0 (measured flat). Three states:
     ok / flat / underpowered (fewer than the registered minimum of measurable games) —
     a band pass with a FLAT test is "band-pass-overridden-by-flat-dose-response"; a
     band pass with an UNDERPOWERED test routes to "unmeasurable at this dose" (low
     power must neither manufacture the flat verdict nor let a band pass stand alone).
  5. **Zero-inflation: reported diagnostic, NOT a gate (reasoned reversal of the
     earlier ANDed-prong registration, plan-review round)**: zeros have two causes and
     only the CONTROLS disambiguate them — dead controls are already caught fail-closed
     as `no_valid_placebo`, so past that gate an all-zero GK band can only mean the
     keeper does not move the surface, which is a CLEAN FAIL (the cycle's expected,
     publishable outcome), never "unmeasurable". A ceiling was also provably
     outcome-inert for passes (zero-fraction > 0.5 forces median 0 → fail already).
     The per-band exact-zero fraction is reported alongside every verdict — it is what
     makes a fail interpretable.
  6. **Estimand, stated honestly**: observed displacement = keeper deviation − ghost
     error, and the two are not separable per frame (MAE 1.07 m is stated alongside
     every banded number). The registered estimand: "among frames where
     |ghost − actual| ≥ 2 m within the ghost's trusted domain, does moving the keeper
     to the ghost position change P(shot)?" — a deviating-keeper conditional, not an
     all-frames average; the aggressive-keeper expected-sign panel lives in exactly
     this stratum.
- **Instrument validation (CI-gated, ships with PR-1)**: a discriminating-power
  meta-test with a MIXED-dependence planted model (GK-dominant + defender dependence +
  noise — a GK-only model like p = σ(gk_r) zeroes the nearest-defender control and
  trips the M2 guard for reasons unrelated to discriminating power) plus
  fixture-validity preconditions asserted INSIDE the meta-test (control > 0, placebo
  p95 > 0, gated-band zero-fraction < 1). The planted model must make the probe PASS;
  a GK-blind model must make it FAIL specifically on the ratio/band prongs. Without
  this, a null result is uninterpretable (the 4.46.0 lesson: an invariance/threshold
  instrument is only evidence if it demonstrably fails when it should).
- **Two probe reports, one gated (sequencing/comparability)**: the GATED run is on GS
  frames (§3.4) — a different provider/cohort/fps than the 4.18.0 SC probe sample, so
  the "reproduces ≈0.00107" expectation is restated as population-shifted, not
  same-sample; an SC same-population comparison run is additionally reported-not-gated
  — the only leg the float-noise prediction applies to.

### 3.2 xCross probe re-run

Byte-equivalent wrapper, frozen rule; the frozen-gate verdict stays diagnostics-frozen,
but the re-run OUTPUT gains report-only zero-fraction and dose-response fields so the
§3.5 table does not compare a diagnostics-rich xS verdict against a diagnostics-blind
xCross one. Probe-sample provider/match ids are recorded into `metrics.json` via a
schema-tested field (closing the 4.18.0 provenance gap with an assertion, not prose).
The same two-report split as §3.1 applies: gated GS run (population-shifted
expectation), SC same-population comparison reported-not-gated.

### 3.3 Causal shot arm (re-registered after review — the prior outcome was
inexpressible by the label machinery)

The previous registration ("goal within the 6 s window, strictly post-treatment") is
WITHDRAWN: `_label_outcome` filters by type only and the window is strictly post-anchor
— in SPADL a goal IS the successful shot at the anchor, so that outcome measured rebound
goals (or, with the labeler reused unchanged, shot flurries). Registered instead:

- Domain = attacking-third predicate; treatment = shot types (`shot`, `shot_freekick`,
  `shot_penalty`); **outcome Y = a SUCCESSFUL outcome-type action in the
  ANCHOR-INCLUSIVE 6 s window (`ts ≥ anchor`, `result_id == success`)** — for treated
  spells that is the anchor shot's own goal or a rebound goal; for CONTROLS (anchored
  at entry) a within-window conversion. (Second re-registration, plan-review P1: the
  own-result-only form made control Y ≡ 0 by construction — controls have no anchor
  action — rendering the ATT confounder-invariant and the entanglement gate
  structurally dead.) The labeler gains result-conditioned + anchor-inclusive axes as
  NAMED scope with known-truth tests: a saved anchor shot yields Y=0, a scored one
  Y=1, and a builder-level fixture asserts `Y.var() > 0 among Z==0` — the instrument
  is validated at the layer it defends. Confounders = a PR-1-recorded xS-side confounder list (xS has
  no `_CONFOUNDERS` constant to reuse — this is a fresh registered decision) + the GK
  block with missing-indicator.
- **What this instrument measures (relabeled honestly)**: the GK block enters ONLY as
  confounders in the propensity for treatment; nothing counterfactually varies the GK.
  The `gk_ablation_shift`-vs-placebo verdict is therefore registered as
  **"GK-confounder entanglement"** — supportive context for the Δattempt arm, NOT a
  causal deterrence estimate — and §3.5's column is named accordingly. (A
  deterrence-flavored causal design — treatment = an upstream event such as
  final-third entry, outcome = shot ATTEMPT — is noted for the follow-up spec, out of
  scope here.)
- Placebo idiom per §6's clustering fix: match-level (or cluster-bootstrap) placebo
  band, NOT row-i.i.d. permutation; plus a positive-control known-truth test in
  `tests/causal/` proving the ablation shift DETECTS a planted GK-confounding signal
  (the same instrument-validation discipline §3.1 demands of the probe — currently
  only the null is pinned).

### 3.4 Retrains (owner-run, DGX)

All three models on the corrected corpus (post-4.29.0/4.30.0 geometry, ADR-035 GS-ET
corrections, 4.28.0 real ball-z):

- **xS** and **xCross**: same HPO objectives, seeds, and protocols; each RE-RUNS the
  pre-registered public-vs-full paired test — GS's exclusion was decided on a
  mixed-chirality comparison (§1.2.4), and its 64 clean elite-keeper matches may now
  enter. If the paired verdict flips, the bundled `default` changes provider
  composition (recorded in the model card + metrics). **If GS is admitted to
  training**, the probe/causal measurement frames are drawn from GS matches HELD OUT
  of the admitted training folds (or the verdict is gated on the held-out stratum) —
  boosted-tree responsiveness on memorized partitions is not the served regime.
- **Ghost-GK**: forced retrain — mirrored `gk_y` labels for 17/81 matches; it is the
  counterfactual engine for every arm including the physics arms. Carrier params
  re-recorded; artifact version bump. Its model card also gets a correctness fix in
  PR-2: the card claims the training filter is "GK outside penalty area during active
  defensive actions"; the code is a pure geometric box (x∈[0,30], y∈[18,50], no action
  condition, and 30 m ≈ 2× the penalty area) — the card's own label-domain line is
  already correct, the prose sentence is not.
- **Chirality guard (split across PR-1/PR-2 — the sequencing matters)**: the
  fingerprint EMISSION lands in PR-1's retrain-script/`save()` edit surface (the DGX
  retrains run between PR-1 and PR-2 and must write it — otherwise the artifacts need
  hand-patching); `load()` hardening lands in PR-2. The fingerprint is DERIVED from
  the frames pipeline or behavioral (model output on a canonical asymmetric probe
  frame), never a training-script-written self-declaration a mislabeled artifact
  would satisfy. Missing fingerprint (i.e. every pre-PR-2 artifact — exactly the
  mis-served ones) ⇒ fail-closed with an explicit legacy override flag. Coverage
  includes **ghost-GK** (whose `gk_y` labels were the actually-mirrored ones and on
  which every arm depends), not just xS/xCross. §7 carries a test that `load()`
  RAISES on a mismatched and on a missing fingerprint.
- **Measurement runs on GS frames only (registered)**: the kloppy gateway hardcodes
  `visibility: None` for player and ball rows, so SkillCorner frames on the pining
  path cannot distinguish an observed keeper from a smoothed/extrapolated one — a
  GK-substitution statistic measured there measures the smoother. The native SC
  converter populates the schema's `visibility` column, but re-routing the pining
  loader is not this cycle's surface (user-mediated follow-up: preserve visibility in
  the gateway, or route SC through the native bronze converter). Until then: probes,
  causal runs, and gates are measured on GS frames (native-clean, elite keepers,
  29.97 fps); training-corpus composition remains whatever the re-run paired test
  selects, with the SC smoothed-keeper caveat recorded in the model cards.

### 3.5 Decision table (registered in ADR-037; the verdict is mechanical)

The two arms are gated INDEPENDENTLY — each against its own probe and its own
GK-confounder-entanglement run (§3.3 — supportive context, not a causal deterrence
estimate). GKDV v1 (physics arms) ships regardless of every row below. **The table is
implemented as a pure function in `_validate.py` with a parametrized test over every
row** — "mechanical" is a code property, not a prose claim.

**[AMENDED 2026-07-18 — WRONG MODULE, and the mistake is actionable.]** The verdict
function shipped in PR-1 as **`regate_verdict` in `silly_kicks/tracking/_model_eval.py`**,
not in `_validate.py`. `_validate.py` is a **different, PR-3-owned module** holding the
physics arms' per-arm expected-direction constants (ADR-037 item 7) and does not exist
yet. A PR-3 author reading this line literally would create `_validate.py` and either
duplicate or re-home an already-frozen, already-tested verdict function. **Do neither:
`regate_verdict` stays where it is; `_validate.py` is new and separate.**

**[AMENDED 2026-07-18 — STATUS] The `cross | fail | any` row has FIRED.** Stage-B figures:
`gk_median_abs_delta = 0.009697` vs `nearest_def = 0.004380` → ratio 2.21× (clears the
≥2.0 prong) but under `TF19_PROBE_ABS_FLOOR = 0.01` → `tf19_ready = false`, and
`regate_verdict(arm='cross', probe_verdict='fail', entanglement='inside_band')` returns
**`gated_clean_fail`**. Only the four `shot` rows remain live. **Reproducibility gap PR-3
must close:** the bundled `_xcross_weights/default/metrics.json` records the *Stage A*
probe (0.002417 / ratio ≈1.41×), NOT the Stage-B figures that produced the verdict — those
exist only as prose in `docs/research/tf19_pr2/decision_table.md`, sourced from an SSH read
of the DGX box. A pre-registered gate whose firing numbers are not reproducible from the
repository is a provenance hole: **PR-3 must bundle the Stage-B probe record as a
machine-readable artifact.** **[REVIEW ROUND 1 — S6]** the earlier "…or state explicitly
which path is authoritative" escape hatch is **deleted**: naming the prose table
authoritative does not close the hole, because the numbers remain transcribed prose rather
than something recomputable from the repo — which fails this document's own ADR-036
honest-reporting discipline. The xgboost-version pin (§6.4 disclosures) rides with this
artifact: an unpinned probe number is non-citeable given the 2.x/3.x `base_score`
divergence PR-2 had to guard.

| arm | its probe | GK-confounder entanglement | consequence for that arm |
|---|---|---|---|
| shot | pass (registered §3.1 rule, all prongs) | clears | Δattempt shot arm joins the follow-up composition spec |
| shot | pass | inside band | joins, with "surface-responsive, confounder-entanglement unconfirmed" recorded |
| shot | band pass, flat dose-response | any | **band-pass-overridden-by-flat-dose-response** — stays gated; routed to GK feature engineering, NOT read as clean fail |
| shot | insufficient support (band n / stratum n / placebo degenerate) | any | **unmeasurable at this dose** — stays gated; routed to sampling/ghost follow-up, not to "no signal". An all-zero GK band with LIVE controls is NOT this row — it is the clean fail below |
| shot | any probe verdict | degenerate (no positivity / empty overlap) | probe verdict governs; entanglement recorded as "unmeasured" (a real harness outcome, not an error) |
| shot | instrument-invalid (meta-test red) | any | verdict VOID — fix the instrument first |
| shot | fail (prongs evaluated, not met) | any | stays gated (on GK feature engineering) — the clean fail |
| cross | pass (frozen rule) | clears | Δattempt cross arm joins |
| cross | pass | inside band | joins, with the same caveat |
| cross | fail | any | stays gated — the §1.2 expectation; recorded as the clean verdict |

**[AMENDED 2026-07-18] JOINT ROW — the table gates the arms independently and has no
branch for "both gated". That branch is now LIVE, not hypothetical**: cross is already
`gated_clean_fail`, so any shot outcome other than a pass realizes it immediately.
Registered consequence, consistent with what this document already commits to elsewhere
(line: "GKDV v1 ships regardless of every row below"):

| joint state | registered consequence |
|---|---|
| BOTH arms gated (any combination of fail / flat-dose / unmeasurable) | The Δattempt composition follow-up spec is **NOT commissioned**. GKDV v1 ships on the **physics arms alone**, and the attempt track routes wholesale to GK feature engineering (§3.1 feature-channel asymmetry names the first target). This is a registered outcome, not a programme failure — see §6 for the H1/H2 read-off that determines *which kind* of gated it is. |

Optionally expose as a pure `regate_composition(shot_verdict, cross_verdict, …)` beside
`regate_verdict`; if added, the existing per-arm rows must stay byte-identical and the
existing frozen enums must not be mutated (both are fail-loud `frozenset`s).

---

## 4. Ghost-substitution engine

`build_ghost_frames(frames, *, model, home_team_id, carrier=None, params=GkdvParams())
→ (counterfactual_frames, provenance, GkdvReport)` in `gkdv/_engine.py`.

**[AMENDED 2026-07-18] The engine and the probe are decoupled by a typed DATA contract,
not by function calls.** PR-1 shipped the probe consuming ghost positions as a `targets`
DataFrame so that `tracking/` never imports `gkdv/`. Read every "the probe uses the
engine's provenance" statement below through that lens: **the engine EMITS a frame; a
runner in `scripts/` passes it to `substitution_deltas(..., mode="targets", targets=…)`.**
The shipped contract is `_TARGET_COLUMNS` in `tracking/_model_eval.py`, validated
fail-loud by `_validate_targets`, and it differs from §4.6 below on four points that PR-3
must reconcile — §4.6 is amended in place. This is the single most load-bearing change
PR-1 made to this design.

1. **Domain**: alive ball, team in possession attacking, ball within 35 m of the
   attacked goal, defending-team GK row present. Frames with a missing/NaN GK block are
   **dropped-and-counted, never scored as Δ=0** (a zero delta from a missing keeper
   reads as "no deterrence" and biases keeper aggregates toward the null).
2. **Pinned factual context, computed once**: `infer_ball_carrier` with the ghost
   model's recorded `carrier_params`; `derive_team_in_possession`; ONE goal-map
   instance (GK-mean-x rule) reused for both the defended-goal flip and GK-team
   identification. Nothing is re-derived on counterfactual frames (avoids
   carrier-hysteresis / goal-map drift contaminating the delta).
3. **Ghost positions — via a NEW positions-only serving seam in `_ghost_gk.py`**
   (`serve_ghost_gk_positions()` or `compute_ghost_gk(..., density=False)`), which
   single-sources feature extraction, the 4.12.1 duplicate-(frame, gk_team) collapse,
   `predict_mean`, and the 4.22.1 physical-pitch clamp, while skipping the KDE density
   pass (positions, not spread — the density pass is the entire cost of
   `compute_ghost_gk`). gkdv NEVER imports tracking underscore-privates directly:
   calling the extractor raw would silently drop the dup-collapse (recreating the
   exact merge-inflation bug 4.12.1 fixed, on GK-substitution frames) and fork the
   clamp as a drift-prone copy. The gkdv→tracking import surface is an explicit
   allowlist pinned by a test; the dependency direction (gkdv → tracking public
   seams only, never the reverse) is recorded in ADR-037.
   **[AMENDED 2026-07-18 — the seam must EMIT two flags, not merely apply the clamp.]**
   None of this exists yet: there is no `serve_ghost_gk_positions`, no `density=` kwarg,
   and `compute_ghost_gk` applies the 4.22.1 clamp as a whole-array `np.clip` behind a
   single "one or more" batch warning, discarding which rows were clamped. The probe
   REQUIRES `ghost_clamped` per row and **non-null** (`_validate_targets` raises on a
   null, because `bool(NaN)` is `True` and would silently shrink the trusted stratum).
   Likewise `ghost_out_of_box` does not exist anywhere — `GRID_X_MAX = 30.0` is a bare
   module constant used only as a training-label filter, and the flag **must be evaluated
   on goal-relative x BEFORE the write-back flip in (4)**. So the seam's contract is:
   *serve positions, apply the clamp, and return per-row `ghost_clamped` +
   `ghost_out_of_box` alongside* — additive provenance in the house `*_source` idiom,
   no value change, no retrain.
4. **Write-back**: goal-relative → frame coords: `x = gr_x` if the defended goal is at
   x=0 else `105 − gr_x`; y unchanged. Consumes `compute_ghost_gk`-style goal-relative
   output, never `add_ghost_gk`'s action-LTR columns (ADR-028).
5. **Velocity policy (registered) + sensitivity variant**: the ghost keeps the factual
   GK's vx/vy — minimal-intervention counterfactual; the ghost model predicts position
   only. Known bias, concentrated exactly in the gated stratum: both arms are
   velocity-dependent (`get_das` requires vx/vy; Spearman TTI uses velocity), so a
   teleported-but-still-moving ghost projects the ACTUAL keeper's momentum from the
   ghost position — for aggressive-keeper frames the deep-placed ghost carries outward
   velocity, shrinking the actual-vs-ghost contrast in the ≥2 m band. A
   reported-not-gated sensitivity variant (ghost velocity zeroed and/or scaled, on a
   sample) is registered: stable → recorded; unstable → documented limitation.
   **[AMENDED 2026-07-18, REVIEW ROUND 1 — S3: this is now GATING, superseding
   "reported-not-gated" in the sentence above.]** §6.4 row 2 halts on a zeroed-velocity
   variant that FLIPS the ratio (`physics_velocity_confounded`), and the two statements
   contradicted each other. Gating is the correct resolution **by this bullet's own
   analysis**: the bias is concentrated exactly in the gated stratum, so a sign flip there
   is not a footnote — it means the arm's headline number is an artifact of the velocity
   policy. Reading: *stable → recorded and continue; FLIPS → halt and re-run under the
   registered policy; unstable-but-does-not-flip → documented limitation.*
6. **Provenance per frame**: ghost x/y, |ghost − actual| displacement, GK `player_id` +
   `is_goalkeeper_source`, clamped flag, ghost out-of-training-box flag (goal-relative
   x beyond the 30 m label hull — OOD-served), drop reason. The displacement
   distribution doubles as the xS probe's calibration input, and the OOD/clamped flags
   feed the probe's §3.1(3)/(6) stratification. The provenance frame's keying and its
   inclusion of DROPPED frames are part of the return contract (two independent
   consumers: the probe and `_metric`); `GkdvReport` ECHOES the `GkdvParams` used, and
   the validation script writes them into the `gkdv_v1` artifacts — registration
   without traceability is not registration.

   **[AMENDED 2026-07-18] The provenance frame is NOT itself the targets frame — four
   concrete mismatches with the shipped `_TARGET_COLUMNS`. PR-3 emits a small ADAPTER
   (provenance → targets); it must not try to make one frame serve both roles.**
   1. **Names**: the probe requires `target_x` / `target_y`, not "ghost x/y".
   2. **Dropped frames**: `_validate_targets` requires `target_x`/`target_y` FINITE on
      every row — a targets frame containing dropped frames **raises**. Provenance keeps
      them (it must, for drop accounting); the adapter filters them out.
   3. **Keying**: the probe requires **exactly one row per `(game_id, period_id,
      frame_id)`** — three keys, no team. But `compute_ghost_gk` keys per
      `(frame, gk_team)` and writes ghosts for **BOTH** teams' keepers, so a naive
      pass-through duplicates every frame and trips the uniqueness check. The adapter
      must select the **defending-team** keeper per frame, using the §4.2 pinned goal
      map — not a re-derivation.
   4. **Displacement is not consumed**: the shipped probe RECOMPUTES `displacement_m`
      internally as `target − actual GK` read off the frame's own GK row, rather than
      reading the provenance column. The engine's displacement is therefore a
      **planning/reporting** quantity (it tells you offline whether the ≥2 m band is
      populated at all), not a probe input. Do not wire it as one.

   **[REVIEW ROUND 1 — S7 accepted] The adapter is the load-bearing seam; name it and test
   it.** Everything else in this spec earns a test; the adapter was prose. Register it as
   **`gkdv._engine.provenance_to_targets`** with a contract test asserting all four points
   above — column names; `target_x`/`target_y` finite on EVERY row; **exactly one row per
   `(game_id, period_id, frame_id)`**; defending-team selection via the §4.2 pinned goal
   map — plus a **red-first test that a naive both-teams pass-through RAISES**. Mismatch #3
   is the dangerous one: a pass-through either trips the uniqueness check or, worse,
   silently selects the WRONG keeper. That test is the executable form of this amendment.
   Note the engine therefore returns **two views**, not one: the full provenance frame
   (drops included, keyed per frame + gk_team, consumed by `_metric`) and the
   defending-team-only, drop-free, 3-key-unique targets projection (consumed by the probe).

Rules honored: counterfactual frames NEVER routed through `PitchControlCache` (ADR-008
canonical-only contract; the factual side may use one); all id seams through
`_id_compat` (ADR-019); PR-1 hardens `extract_xshot_features`' raw `==` team-id
compares to canonical ids (latent ADR-019 gap, fixed regardless of TF-19).

**[AMENDED 2026-07-18] The GHOST extractor's raw id compares were NOT hardened and are on
the PR-3 critical path.** PR-1 fixed `extract_xshot_features`; `extract_ghost_gk_features`
still uses raw `==`/`!=` on team ids — and §4.3 routes **every ghost position** through it.
On Gradient Sports frames `team_id` is nullable `Int64` (ADR-027) while other providers
carry object strings, so a raw compare yields **empty defending/attacking splits and a
corrupt feature row rather than an error** — the same silent-null shape this cycle exists
to eliminate. CLAUDE.md records this as an out-of-scope latent gap citing
`_ghost_gk.py:488-490`; **that citation is line-stale** (the compares now live at 521-523,
584 and 771). **PR-3 must promote this from "recorded gap" to in-scope work**: route them
through `_id_compat`, and re-derive the line numbers rather than trusting either source.

**[AMENDED 2026-07-18] Risk ordering in §4.5 is now inverted by evidence.** The velocity
policy is presented above as the main known distortion in the gated stratum. With the
§3.1 feature-channel asymmetry confirmed in source (2 of 27 xS features see the keeper)
and a ghost MAE (~1.1 m) well below the 2 m dose floor, **the dominant risk to the xS arm
is the feature channel, not ghost momentum.** Both remain registered; the reports must
state this ordering so a `fail` is not attributed to the velocity policy by default.

---

## 5. Physics arms (`gkdv/_arms.py`)

Both arms are defined in attacker-value units, `actual − ghost`, so **negative =
deterrent** uniformly.

- **ΔDAS**: `get_das(factual) − get_das(counterfactual)` on the attacking team's
  dangerous accessible space; `team_in_possession` pinned from the factual context on
  both sides. Lazy `[das]` import guard — the arm skips-with-report if accessible-space
  is absent; reuses `_das.py`'s frame preparation (pyarrow/object-dtype handling
  included).

  **[AMENDED 2026-07-18 — PINNING `team_in_possession` IS NOT SUFFICIENT, and the
  mandated API cannot pin what actually needs pinning. This is a live Δ-corruption
  hazard, not a nit.]** accessible-space infers playing direction **per period** via a
  discrete argmin over a position-dependent statistic —
  `x_mean = groupby(team)[x].mean(); smaller_x_team = x_mean.idxmin()`. Pinning
  `team_in_possession` supplies the grouping KEY only; it does not constrain the mean-x
  comparison that produces the DIRECTION. Moving the keeper changes his team's mean x
  (a 4 m GK displacement shifts an 11-player mean by ~0.36 m), so the factual and ghost
  legs can infer **different directions** and the resulting Δ is not a counterfactual at
  all. Required amendment: **(a)** compute the attacking direction ONCE on the factual
  FULL frames via `_das._pin_attacking_direction` and pass that SAME pinned column to
  BOTH legs — neither leg may infer; **(b)** route the arm through
  `get_individual_das(..., attacking_direction_col=…)` summed per team (the established
  library pattern), OR extend `get_das` with an `attacking_direction_col` passthrough
  that overrides its hardcoded `infer_attacking_direction=True` — a small `_das.py`
  change that must be scoped INTO PR-3 if `get_das` is kept.

  **[AMENDED 2026-07-18] What ΔDAS actually measures — an interpretation limit that
  belongs in the reports.** `is_goalkeeper` is **not passed to accessible-space at all**
  (`_das.py` `_COLUMN_MAP` forwards x, y, vx, vy, player, team, frame, period,
  team_in_possession — no keeper flag). accessible-space therefore treats the keeper as a
  generic player with outfield locomotor parameters, so **ΔDAS is the accessible-space
  consequence of relocating one anonymous defender who happens to be the keeper**; no
  keeper-specific reach, handling, or control advantage is modelled. ΔGK-threat-suppression,
  by contrast, weights the keeper at `lambda_gk = 3.0` inside pitch control. **The two arms
  are therefore not on a common keeper-modelling footing** — which reinforces the existing
  decision to report them separately in v1, and any future composite must resolve the
  asymmetry first. Register `lambda_gk` alongside the DAS stride in `GkdvParams` so both
  arms' keeper treatment is visible in one place.
  **[REVIEW ROUND 1 — S9 accepted] Registering `lambda_gk` is not enough — sensitivity-test
  it.** Since Δthreat's GK sensitivity is inherited ENTIRELY from `lambda_gk = 3.0`, any
  Δthreat result is partly a statement about that weight, and if the two arms disagree a
  reader cannot tell a finding from a weight artifact. Register a **reported-not-gated
  `lambda_gk` sensitivity leg** (e.g. 1.0 / 3.0 / 5.0) alongside the primary — the same
  idiom §4.5 uses for the velocity policy, applied to the parameter this amendment just
  identified as governing the arm.
- **ΔGK-threat-suppression** (renamed from "Δcover-shadow" after source verification —
  the arm is NOT lane-specific and is specced in its algebraically reduced form):
  `compute_blocking_score(…, defenders_to_remove=[gk])` at two GK positions has its
  removal legs cancel exactly (frames differing only in the GK row are identical once
  the GK is removed), so the two-call form reduces to
  `threat_pc(ghost frames) − threat_pc(actual frames)` — where `threat_pc` is the
  xT-weighted Voronoi pitch-control threat integral (`_voronoi_threat`), the ONLY
  channel through which the GK enters `compute_blocking_score` at all (the lane/TTI
  model excludes goalkeepers by construction, and the explicit-removal path never ran
  it anyway).
  **[AMENDED 2026-07-18 — THE REDUCTION SURVIVES BUT THIS MECHANISM IS MIS-STATED; a
  PR-3 author debugging a zero Δ would look in the wrong function.]** `_voronoi_threat`
  is **not** the channel: its own body selects only ATTACKERS (receiver set, Voronoi
  seeds, dangerous-receiver filter) and the *defending* keeper — the player GKDV
  substitutes — appears nowhere in its selection logic. The keeper enters solely through
  the `surface` argument, i.e. through **`compute_pitch_control`**, where GK rows are
  retained and up-weighted by `lambda_gk = 3.0`. So the correct statement is:
  *`threat_pc` is the only GK-sensitive non-cancelling term, and its GK sensitivity is
  inherited entirely from the pitch-control surface it integrates.* The algebraic
  reduction to two calls is unaffected; only the reason is corrected. Practical
  consequence: **the arm's sensitivity is governed by `lambda_gk`**, which is why it is
  registered in `GkdvParams` above.
  The arm therefore computes `threat_pc` directly on the full actual and
  ghost frames — two pitch-control surfaces, not four, no removal, no
  `max(…, 0.0)` clamp — via a `compute_threat_pc()` facade added to
  `_cover_shadows.py` (no reimplementation of `_voronoi_threat`; gkdv imports the
  facade, never the private).
  **The REPORTED value — sign stated explicitly; an earlier draft printed it
  inverted**: `Δ = threat_pc(actual frames) − threat_pc(ghost frames)`, the
  **NEGATION of the Δ_blocking reduction** (blocking_score is defense-positive; this
  arm reports attacker-value): a deterrent actual keeper suppresses attacker threat,
  so threat_pc(actual) < threat_pc(ghost) and Δ < 0 = deterrent, matching the section
  convention. ADR-037 carries a worked numeric example; a red-first planted-polarity
  fixture asserts BOTH arms go negative for an obviously-deterrent actual keeper; the
  per-arm expected directions are registered as `_validate.py` constants (not TODO
  prose, which PR-1 rewrites).
  A GK-INCLUSIVE lane-obstruction arm would be a real code change to `lane_control`
  and is explicitly out of scope for v1. Requires a caller-injected fitted
  `ExpectedThreat` (ADR-022 fail-closed pattern; no self-fit; unfitted-grid guard).
  Comparing the same quantity at two GK positions keeps this a REPOSITIONING
  counterfactual — it sidesteps the monotone-LOO structural-zero limitation documented
  in the 4.24.0 space-creation work.
  **Silent-zero guards (both verified live hazards)**: (i) the arm's API does NOT
  accept `pitch_control_cache` and both legs use fresh local surface computations —
  the cache key excludes player positions, so a shared cache would serve the ghost
  frame the ACTUAL frame's surface and Δ ≡ 0 silently; a red-first test pins that the
  arm's deltas are non-zero on a fixture where the surfaces must differ. (ii) every
  GK-row identification in the arm routes through `_id_compat` (ADR-019) — the
  library's own removal path uses a raw `.isin` that silently removes nothing on a
  dtype-mismatched id, the same Δ≡0 failure shape.

**Cost control**: the DAS arm simulates accessible space per frame × 2; eligible frames
are sampled at a registered per-possession stride (`GkdvParams`, frozen dataclass). The
default stride value is chosen at plan time from a measured per-frame cost budget and is
locked before the owner validation run; sampled counts reported, never silent.

**Aggregation (`gkdv/_metric.py`)**: per-keeper aggregates by the **frames-resolved GK
`player_id`** (the per-frame provenance §4.6 already emits). The earlier `player_key`
mandate is struck: `player_key` does not exist anywhere in the library (it is a
lakehouse gold-mart, action-grain column surfaced only by the Databricks loader), the
NULL-taker problem it solves is an ACTIONS-side artifact that cannot occur here (frames
carry `player_id` on every row, and a goal-kick is structurally outside GKDV's
in-possession final-third domain), and a pure library module must not depend on a gold
join. Cross-provider re-keying, if ever wanted, is consumer-edge work. Keeper-NAME
resolution for the expected-sign panel stays in the owner-run script via the Drive
roster JSONs (§6). **The per-keeper aggregate FUNCTIONAL is registered, not just the
key**: mean AND median are both reported, the registered gate reads the mean; per-arm
per-keeper exact-zero fractions appear in the report (ΔDAS is exactly 0 whenever the
displacement moves no accessible-space boundary, and §3.1 predicts small displacements
dominate), and a minimum NONZERO observation count per keeper applies for gate
inclusion. **Aggregation is grain-agnostic** (observation-level values keyed by the
resolved GK id + optional weights — the future Δattempt arm is a build-up-WINDOW
quantity, a different grain than per-frame), and the §4 domain predicate is a
`GkdvParams` argument defaulting to the registered v1 domain (the cross arm's domain
is `_in_wide_area`, not the 35 m predicate) — both recorded in ADR-037 as
forward-compat decisions. Arms reported SEPARATELY in v1; no z-normed cross-arm
composite (registered follow-up, gated on the arms validating individually).

---

## 6. Validation harness

Pure functions in `gkdv/_validate.py`; I/O in `scripts/validate_gkdv.py`; constants
locked in code before the owner run; verdicts to `docs/research/gkdv_v1/`.

1. **Keeper discrimination (PRIMARY)**: ICC(1) on sampled-frame arm values grouped by
   the frames-resolved GK `player_id` (§5), unbalanced-n correction, with a
   **match-block permutation null band** — whole matches' frames reassigned to keepers,
   NEVER frame-level shuffling (frames within possession/match are strongly
   autocorrelated; a fully-exchangeable frame-level null is anti-conservative — real
   data with zero keeper effect and ordinary match structure would sit above it: a
   guaranteed-significant instrument). **Clustering floors, registered**: keeper
   inclusion requires ≥2 matches (for a single-match keeper, keeper ≡ match, so
   between-keeper variance mechanically absorbs between-match variance) AND a
   possession floor alongside the 20-frame floor (min-n=20 was an ACTIONS-unit
   convention from xtgk; 20 strided FRAMES can be a handful of possessions in one
   match). The same clustering fix applies to both causal legs: match-level (or
   cluster-bootstrap) placebo bands, not row-i.i.d. permutation over spells clustered
   in ~81 matches. `icc_one_way`/`keeper_spread` are lifted
   from `scripts/xtgk_v2_keeper_discrimination.py` into
   **`silly_kicks/_group_metrics.py`** — a LIBRARY home, not a script: the wheel ships
   only `silly_kicks/`, so anything the lakehouse imports cannot live in `scripts/`
   (precedent: `silly_kicks/_calibration_metrics.py`, whose docstring records the same
   lift). Module concept, stated: `_group_metrics.py` = domain-free grouped statistics
   (ICC, spread, permutation band, power sim); `gkdv/_validate.py` = registered
   constants + verdict logic.
   **[REVIEW ROUND 1 — S8: the stated rationale and the chosen visibility disagree; PR-3
   must RESOLVE this rather than inherit it.]** The argument for a library home is *"the
   wheel ships only `silly_kicks/`, so anything the lakehouse imports cannot live in
   `scripts/`"* — i.e. it is justified **by downstream consumption** — yet the proposed
   module is underscore-**private**. A private module a downstream consumer imports is a
   Hyrum contract with no stability promise, and the cited precedent
   (`_calibration_metrics.py`) has exactly the same shape. Pick one, explicitly:
   **(a)** make it PUBLIC (`silly_kicks/group_metrics.py`, in `_PUBLIC_MODULE_FILES` + the
   Examples gate from day one — the treatment §7 already mandates for `gkdv/` and
   `causal/`) if a lakehouse import is genuinely intended; or **(b)** keep it private and
   restate the rationale as **gkdv's own** import need (gkdv cannot import from
   `scripts/`), stating explicitly that the lakehouse must NOT import it. Do not ship the
   current incoherent pairing.
   Knock-ons: `tests/xtgk/test_keeper_discrimination.py`
   AND `scripts/xtgk_v2_keeper_discrimination.py` itself are re-pointed in the same PR
   (single-sourcing is the point of the precedent); `keeper_spread` gets a
   group-neutral name at lift time; lift from CURRENT main (4.46.0 modified that
   script).
   **Coordination**: the 4.46.0 cycle (which actively used that script) COMPLETED and
   published 2026-07-12 — the lift is unblocked; re-confirm via the user only if another
   xtgk cycle is in flight when PR-3 starts.
   CV-of-per-keeper-means is explicitly excluded as a criterion.
   **Power analysis (pre-registered, runs BEFORE the owner validation) — plasmode, not
   i.i.d.**: real strided frames from the actual corpus, match-level label permutation,
   injected keeper effects; report the power CURVE at ICC = 0.015 / 0.020 / 0.026 (the
   §1.3 anchor is a range, and SC's 0.0147 sits below a single 0.02 point); the gate is
   registered only if detection at the anchor (α = 0.05, match-block null) is ≥ 0.8 —
   otherwise floors/sampling are adjusted FIRST. An i.i.d. simulation would inherit
   none of the clustering and could pass while the real instrument is simultaneously
   underpowered and anti-conservative. The power simulator is a pure `_validate`
   function with a fast CI smoke.
2. **Expected-sign panel (amended)**: pre-registered aggressive panel (Alisson, Neuer;
   Onana falls to the §6.1 ≥2-match rule — decided NOW, not post hoc) vs a
   pre-registered line-keeper panel; rank-sum test across the GS starting keepers, in
   the per-arm expected DIRECTION registered as `_validate.py` constants (§5 — both
   arms negative-for-deterrent). Panel names are resolved to `player_id`s ONCE from
   the Drive roster JSONs, the id lists registered as `_validate.py` constants,
   fail-loud on any corpus miss, roster provenance (path + hash) recorded in the
   artifacts — runtime name resolution against a mutable external store is a
   silent-shrink hazard on a pre-registered panel. ADR-037 states the panel is
   supportive-only; each arm is an independent pre-registered hypothesis (no
   family-wise claim from the battery).
3. **Conceded-xG correlation**: exploratory, reported-not-gated (1–7 matches/keeper).
   **[AMENDED 2026-07-18]** This is the harness's ONLY external outcome axis, and §6.4
   makes the axis question first-order — so it can no longer carry the whole burden of
   "does this arm relate to anything outside the model family". §6.4's Layer 2 supplies
   a properly powered, model-free external test; §6.3 stays exploratory as written.

Domain-shift caveats stated in the reports: (a) if the paired test keeps GS out of
training, GKDV scores GS frames with SC+IDSSE-trained ghost/attempt models; (b) the
ghost's training-label domain is a pure geometric box (x∈[0,30] m, y∈[18,50], NO
action/defensive-phase condition — the model card's "outside penalty area during
active defensive actions" sentence is wrong and is fixed in PR-2): ordinary sweeper
range 16.5–30 m is in-domain, but keepers beyond 30 m are OOD-served (4.22.1 clamp),
and truncating training at 30 m biases the ghost conservative — which INCREASES
|ghost − actual| for aggressive keepers and so preserves the expected-sign test's
direction, at the cost of the panel's most extreme frames being the least-trusted
ghost outputs.

### [AMENDED 2026-07-18] 6.4 Separating H1 from H2 — PROPOSED, needs sign-off before registration

> **Review status**: this subsection is NEW and is the one part of this amendment that is
> a *design proposal* rather than a correction of record. It registers new constants and
> proposes amending shipped routing. **It must be signed off before any constant is
> written into code** — that is the whole point of pre-registration.
> **Review round 2 (2026-07-18) applied**: the outcome definitions were corrected again —
> `Y_attempt`'s labeller citation was a phantom (N2), `Y_far_attempt` is now an explicit
> PARTITION so the coherence arithmetic holds on multi-attempt spells (N4), `D` is
> landmark-defined rather than cohort-derived (N5), and **Layer 2 is NOT constructible from
> the shipped builder surface — it needs one additive `causal/` field, now scoped to
> PR-3b** (N1). **Layer 4 moved OUT of this subsection's PR-3b scope into PR-3** (N3(a)),
> because it guards §6.1's ICC, which ships in PR-3.
> **Review round 1 (2026-07-18) applied**: the decider's outcome was re-specified after a
> post-treatment-conditioning error was found in it (B1 — see the callout in Layer 2);
> Layer 0 gained a training-support precondition (S1); Layer 2 gained a positive control,
> an overlap diagnostic and a stated estimand (S5); three constants are marked as
> **placeholders requiring derivation** (S4); the table's precedence is now test-mandated
> (S2); and the velocity-variant contradiction with §4.5 is resolved in favour of gating
> (S3). **Still unsigned-off.**

**The problem.** §6.1–6.3 measure whether the *physics arms* discriminate keepers. The
attempt arm is adjudicated somewhere else entirely — by `evaluate_xs_probe`, a pooled
median-magnitude/ratio/dose statistic that **never groups by keeper**. So the two arms are
judged by different instruments, on different statistics, with different groupings, and no
comparison between them is licensed. Worse, `regate_verdict` structurally encodes **H1**:
its `fail` row routes unconditionally to "GK feature engineering", and the function's
signature `(arm, probe_verdict, entanglement)` has no input through which **H2** could ever
be expressed. As written, this cycle *cannot* return "the axis is wrong" — only "the
features are thin". Given §3.1's confirmed 2-of-27 feature channel, that is a live risk of
mis-diagnosis, not a philosophical worry.

**The design principle.** A flat attempt-model probe is **inadmissible** as evidence about
the world, because it is predicted by the feature contract. H2 must therefore be reachable
ONLY through a **model-free** test, never from a flat xS or xCross reading. Everything
below follows from that one commitment.

**Layer 0 — instrument validity, fires before every other gate.** Evaluate each channel at
three doses on the same frames: *realistic* (ghost − actual, |δ| ≥ 2 m, trusted stratum),
*ladder* (the existing `XS_PROBE_DOSE_LADDER`), and **saturating** (keeper → goal-line
centre; keeper → goal-relative x = 30 m — both inside the ghost training hull). Registered
rule, per arm independently: if the saturating median |Δ| is not ≥ 5× that arm's own
realistic median **and** not > that arm's own placebo p95 → **`instrument_void`**. This is
the answer to the shared-low trap: it converts "everything read low" from a threshold
judgement into a named diagnosis. An instrument that cannot respond to a keeper teleported
onto his own goal line is broken, and no hypothesis claim may be made from it.

> **[REVIEW ROUND 1 — S1 accepted: the saturating dose can void a HEALTHY instrument.]**
> "Inside the ghost training hull" is the wrong support test, because **the ghost's hull is
> not the attempt models' support.** xS/xCross are boosted trees — piecewise-constant and
> flat-extrapolating — so if the xS training corpus holds few attacking-third frames with
> the keeper on his own goal line, the tree returns its nearest leaf and the response is
> flat **for coverage reasons, not brokenness**. Row 0 would then fire `instrument_void` on
> a working instrument and halt all H1/H2 inference. This is the same OOD-vs-defect
> confusion §3.1 is careful about everywhere else. **Required:** register a
> training-support diagnostic at each imposed position (e.g. the count/fraction of training
> frames within a neighbourhood of the imposed GK location) and make row 0 **conditional on
> adequate support**. Where support is inadequate the verdict is
> **`saturating_dose_unsupported`** — a distinct outcome routing to corpus/sampling work —
> NOT `instrument_void`.

**Layer 1 — responsiveness, comparable but NOT decisive.** Run every channel under two
registered regimes, assigned in advance so they cannot be swapped post hoc: **Regime I**
(imposed dose — the discriminating regime) and **Regime O** (observed ghost — the shipped
metric). Compute one dimensionless statistic per channel by reusing the shipped idiom
verbatim (`gk_med ≥ RATIO × max(nd_med, placebo_p95)`), extended to the physics arms with
**paired-vector controls** (nearest defender + R random outfielders displaced by the
identical per-frame vector). **No cross-axis magnitude comparison, ever** — an
attempt-probability delta and a square-metre delta are not commensurable, and
z-normalising them is a category error. Record that as a stated non-goal in the ADR.

**Layer 2 — THE DECIDER: model-free treatment, two outcomes, one matched sample.** This is
the only construction in which H1 and H2 make *opposite* predictions.
- **Treatment**: keeper depth at final-third-spell entry, binarised at goal-relative
  **x = 16.5 m — the penalty-area line**. Law-defined, data-independent, provably untuned.
  (A ghost-model-derived treatment was considered and rejected: it would trust the very
  model whose blindness is in question.)
- **Outcomes**, estimated on the SAME matched sample with the SAME estimator. **BOTH ARE
  UNCONDITIONAL — defined on every spell, nothing conditioned on a post-treatment
  variable.**
  - `Y_attempt` = 1 if a shot attempt occurs in the spell, else 0. **[REVIEW ROUND 2 — N2]
    Produced by `causal.opportunities._label_outcome`, configured via `OpportunityConfig`
    — NOT by `_occurrence_labels`.** The earlier citation `causal/_occurrence_labels` was
    a phantom on both counts: no such module exists in `causal/`, and the real
    `silly_kicks/tracking/_occurrence_labels.py::_build_occurrence_labels(frames_index,
    events, *, horizon, frame_team_col, …)` is a **frames-level** label builder for
    TRAINING the xS/xCross occurrence models — a different package, a different grain, and
    a different purpose. Wiring it into a spell-level causal ATT would be exactly the class
    of error §3.5's `_validate.py`-vs-`_model_eval.py` catch exists to prevent. Do not
    restore the old pointer.
  - `Y_close_attempt` = 1 if an attempt occurs in-spell **whose SPADL origin
    (`start_x`,`start_y`) lies within D metres of the attacked goal centre** (action-LTR:
    `hypot(105 − start_x, 34 − start_y) ≤ D`), else 0 — **and 0 also when no attempt occurs
    at all.**
  - **[REVIEW ROUND 2 — N4] `Y_far_attempt` := `Y_attempt ∧ ¬Y_close_attempt`** — stated as
    an explicit PARTITION, not "the complement". Reported coherence check, not a gate: if
    `Y_attempt` is null while `Y_close_attempt` is negative, `Y_far_attempt` must be
    correspondingly positive — attempts displaced outward, not destroyed; close AND far
    both negative under a null total is incoherent and voids the read. The partition
    definition is load-bearing because **multi-attempt spells are real here** (§3.3
    explicitly contemplates rebounds and shot flurries, which is why the outcome window was
    re-registered twice): under the looser reading "an attempt occurs beyond D", a spell
    containing both a close and a far attempt would count in BOTH indicators, so
    `ATT(close) + ATT(far) ≠ ATT(attempt)` and "correspondingly positive" would not be
    licensed arithmetic.
  - **No xG** — silly-kicks ships no xG model and the decider must not depend on an
    unvalidated external artifact. Secondaries reported only:
    `shot_on_target_derived`, TF-48 `shot_crossing_y/z`.
  - **[REVIEW ROUND 2 — N1: this outcome is NOT constructible from the shipped builder
    surface. Layer 2 DOES reopen `causal/`, by one additive field.]** `_label_outcome`
    (`causal/opportunities.py`) filters on `game_id`, `period_id`, `team_id`, `type_id`, an
    optional `result_id` and the time window — **there is no spatial predicate**, and
    `OpportunityConfig` exposes **no location axis** (its fields are types, result ids,
    window seconds, anchor-inclusivity, exposure/spell seconds, confounders, gk_block,
    `domain` (a SPELL domain, not an outcome filter), and extractor — all verified in
    source). `Y_close_attempt` therefore requires outcome geometry INSIDE the labeller.
    Required work, and it belongs to **PR-3b**:
    1. add `outcome_max_distance_m: float | None = None` (None = legacy, no spatial
       filter) to `OpportunityConfig`, plus the corresponding geometry in `_label_outcome`;
    2. default it so `xcross_config()` / `shot_arm_config()` stay **byte-identical** — the
       existing `config=None` regression guard already covers this;
    3. list it in §9's PR-3b scope (done).
    **Why this was missed, worth recording:** round 1's B1 fix CAUSED it. The original
    `Y_quality` was a distance measured on the anchor shot's own `start_x/start_y` —
    computed OUTSIDE the builder, needing no builder support — so §2.2's "constructible
    purely as builder arguments" claim was TRUE when written. Replacing it with an
    unconditional spatially-filtered *indicator* pushed the geometry inside
    `_label_outcome`. **Two correct fixes, taken together, broke a third claim** — and the
    claim had been verified at the level of "the config class exists with a full surface"
    rather than "can it express THIS outcome". That is round 1's own rule (quote the
    assertion body, not the registration) applied to a config surface.
  - **[REVIEW ROUND 2 — N5] `D` is LANDMARK-defined, not cohort-derived.** Registered
    primary: **D = 16.5 m**, the penalty-area line — the same Law-defined landmark as the
    treatment, keeping the ENTIRE decider data-independent, and approximating the
    inside/outside-the-box distinction that shot-quality analysis conventionally uses.
    Registered robustness leg: **D = 11 m** (the penalty spot), reported alongside. The
    earlier "derive from a cohort quantile" instruction is **withdrawn as the primary**:
    although a marginal quantile is treatment-blind and outcome-contrast-blind (so it was
    not p-hacking), it would have been the *only* constant in the decider that is not
    landmark-defined, and it sits on the row that decides H2. The observed close/far split
    at both radii is still **REPORTED**, as a non-degeneracy diagnostic — never as the
    means of choosing D.

  > **[REVIEW ROUND 1, 2026-07-18 — B1 accepted; this replaced a mis-identified outcome.]**
  > The original draft used `Y_quality` = shot distance from goal centre **conditional on
  > an attempt occurring**. That is a **post-treatment conditioning error**: `attempt` is a
  > descendant of the treatment, so `E[Y_quality | Z=1, attempt=1] − E[Y_quality | Z=0,
  > attempt=1]` is not a causal effect of keeper depth on shot quality no matter how well
  > the confounders are adjusted — it is contaminated by selection into the attempt
  > stratum. The failure mode points the WRONG WAY and manufactures H2's signature: if a
  > high keeper suppresses marginal (low-quality) attempts, then among *realised* attempts
  > the treated group is selected toward higher quality, so a quality "effect" appears with
  > zero causal effect on quality. Worse, row 7's own precondition springs the trap — a
  > null `Y_attempt` means "not detectably different from zero", not "zero", and the
  > selection bias scales with the TRUE attempt effect, not the detected one. So the row
  > most likely to fire was exactly the row where conditioning looks harmless. The
  > registered negative control does not cover this (it detects confounding, not
  > post-treatment selection), and neither Layer 0 nor Layer 3 touches it.
  > The unconditional joint outcome above is also a **better statement of H2**: the
  > hypothesis's own mechanism — "the attempt is made anyway from a worse position" —
  > predicts precisely `Y_attempt` null AND `Y_close_attempt` negative.
  > If a continuous distance measure is still wanted, keep it as a **reported secondary
  > with an explicit selection caveat**, or add Lee bounds; it must never be the outcome
  > row 7 fires on.
- **Estimator**: `causal/matching.py` ATT + Abadie–Imbens SEs, cluster-aware
  `placebo_shift` at match level. All shipped, unchanged.
- **Confounders**: defensive line height and compactness (`compute_defensive_line`, TF-14
  — the dominant confound: a high line both invites attempts and permits a high keeper);
  score differential and time remaining; ball r/θ to goal; defenders between ball and goal;
  carrier pressure (`bekkers_pi`); team fixed effect or a within-match estimand.
- **Negative control**: the identical ATT with an outcome the keeper's entry position
  cannot influence (e.g. a throw-in conceded by the attacking team in-window). If it fires,
  the world test is confounded and the whole leg is VOID.
- **[REVIEW ROUND 1 — S5 accepted] POSITIVE control + overlap + estimand, all required.**
  As drafted, the DECIDER had weaker instrument validation than the arms it adjudicates —
  §3.1 mandates a full discriminating-power meta-test (planted model must pass, GK-blind
  model must fail) and §3.3 mandates a known-truth positive control, while Layer 2 had only
  a negative control. Hold Layer 2 to §3.1's standard:
  (i) a **planted-effect fixture the ATT must DETECT** — a synthetic cohort with a known
  keeper-depth effect on each outcome, red-first, mirroring `tests/causal/`'s known-truth
  gates; (ii) a **positivity/overlap diagnostic on the 16.5 m binarisation** (propensity
  overlap + standardised mean differences before/after matching, the ADR-015 idiom already
  used by the xCross causal harness — a treatment split with no overlap silently estimates
  nothing); (iii) an explicit statement of the **target population** the ATT estimates over
  (treated spells, i.e. high-keeper entries), since that is what the verdict generalises to.
  §7's rule applies verbatim: every counterfactual needs a non-vacuity assertion that it
  actually moved something.

**Layer 3 — remedy routing, and it runs FIRST because it is nearly free.** A
*feature-headroom probe*: append the GK to the xS defender array, recompute
`openGoal_with_GK` and `DefDist_0_with_GK`, and measure how far the ghost substitution
moves those **feature values**. Pure geometry, no fitting, no GPU. Registered threshold:
median |Δ openGoal_with_GK| ≥ 0.02 **(PLACEHOLDER — must be derived before registration;
see the derivation duty below)**. This answers "is there anything for a retrained model
to see?" before a GPU is booked.

**Layer 4 — behavioural anchoring (the guard the sibling metric's failure teaches).**
**[REVIEW ROUND 2 — N3(a): Layer 4 SHIPS IN PR-3, not PR-3b, despite living in this
subsection.]** It gates a **PR-3** deliverable — §6.1's ICC is the PRIMARY criterion and
ships in PR-3 — so leaving Layer 4 in PR-3b would have PR-3 ship the primary criterion
without the guard this spec says must precede its interpretation, and anyone running §6.1
between the two PRs would get a number the spec forbids interpreting, with nothing saying
so. Layer 4 is also the cheapest item in §6.4 (terciles plus a mean-signed-δx comparison —
no causal machinery, no new constants beyond the 0.5 m separation). It therefore moves into
PR-3 alongside the ICC it guards.
Before ANY ICC is interpreted: split keepers into terciles by arm value; the top and bottom
terciles must differ in mean signed goal-relative δx by ≥ 0.5 m. If they do not, the arm is
not tracking a behaviour keepers actually vary, and its ICC is reported **`uninterpretable`**
rather than as evidence. This is precisely the check that would have caught a metric
rewarding a behaviour elite practitioners do not perform.

**Pre-registered decision rule** — pure function `gkdv_discrimination_verdict(...)` in
`gkdv/_validate.py`, parametrized-tested over every row (the `regate_verdict` discipline).
Evaluated strictly in order; the first firing row returns. `live` ≡ `|ATT|/SE ≥ 2` AND
`|ATT| >` its own match-clustered placebo p95.

| # | condition | verdict | routing |
|---|---|---|---|
| 0 | any arm's saturating positive control dead | `instrument_void` | fix the instrument; no H1/H2 claim |
| 1 | physics arm fails tercile separability (≥0.5 m) | `arm_not_behaviourally_anchored` | redesign the arm; do NOT report its ICC |
| 2 | zeroed-velocity variant flips the ratio | `physics_velocity_confounded` | re-run under the registered velocity policy — **[REVIEW ROUND 1, S3] this is GATING, and §4.5 is amended to match** |
| 3 | negative-control outcome significant | `world_test_confounded` | redesign Layer 2; H2 neither supported nor refuted |
| 4 | max ratio < 2.0 on every channel at **imposed** dose | `dose_inadequate` | ghost/sampling work; evidence for nothing |
| 5 | matched n < N_min **(PLACEHOLDER — derive; see below)** OR plasmode power < 0.80 at ICC 0.015–0.026 | `underpowered` | a null here is NOT evidence |
| 6 | `Y_attempt` live | **`H1_supported`** | GK feature engineering, **scoped by Layer 3**: headroom ≥0.02 → costed retrain; <0.02 → `H1_supported_remedy_exhausted`, escalate model class |
| 7 | `Y_attempt` null AND `Y_close_attempt` live-negative (coherence check passes) | **`H2_supported`** | **retire Δattempt from GKDV; ship the physics/quality arms; amend the TF-19 definition.** NOT "fixable by features" |
| 8 | both null, instrument live, powered | `construct_falsified` | the deterrent construct is refuted at realistic dose — a publishable result |
| 9 | both live | `both_axes_live` | composition gets its own spec |
| 10 | otherwise | `indeterminate` | enumerated; no silent fallthrough |

**Registered asymmetry — the single most important line in this subsection.** H2 is
reachable **only** through row 7. It can never be inferred from a flat xS or xCross probe,
because §3.1's feature contract predicts that flatness. Encode this as a named constant
with a docstring citing the GK-excluded defender array.

**Registration disclosures** (the ADR-036 honesty discipline, applied to ourselves):
- Do **NOT** touch `TF19_PROBE_ABS_FLOOR`. 4.51.0's `gated_clean_fail` stands as the
  shipping gate. Any new floor is a *separately named* GKDV criterion.
- If a base-rate-relative criterion is registered, **disclose in the same commit** what it
  would do to the already-recorded verdict, and record the incommensurability it exposes:
  a fixed 0.01 floor is ~32% of xCross's positive rate but ~4.6% of xS's, and the xS rule
  carries no absolute floor at all — so any side-by-side presentation of the two arms'
  numbers is misleading without that note.
- **`regate_verdict`'s routing needs amending** (ADR-037): `gated_clean_fail` must stop
  routing *unconditionally* to GK feature engineering, since that hard-codes H1.
- Pin the resolved xgboost version into the probe artifact and schema-assert it — the
  2.x/3.x `base_score` divergence PR-2 guarded makes an unpinned probe number non-citeable.

**[REVIEW ROUND 1 — S4 accepted] DERIVATION DUTY: three constants are placeholders and
must be derived from measured quantities BEFORE registration.** This spec is otherwise
rigorous about deriving thresholds (the ICC band ← §1.3's measured 0.015–0.026; the 2 m
dose ← the ghost's ~1.1 m MAE), and §1.3's own lesson is that *absolute-effect floors must
be calibrated to the intrinsic magnitude of the quantity before registration*. These three
were stated bare and are not yet registrable:
- **`N_min`** (row 5): derive from the plasmode simulator — it already generates real
  strided frames with injected effects; report the matched-n at which ATT power reaches
  0.80 at the ICC anchor, and register THAT.
- **`median |Δ openGoal_with_GK|` threshold** (Layer 3): **state `openGoal`'s units and
  observed range first** (a reader cannot currently tell whether 0.02 is generous or
  unreachable), then set the threshold as a stated fraction of that range.
- ~~**`D`** (the `Y_close_attempt` radius): derive from a cohort quantile.~~
  **[REVIEW ROUND 2 — N5: WITHDRAWN.]** `D` is now landmark-defined (16.5 m primary, 11 m
  robustness leg) and is therefore NOT a placeholder — see Layer 2. It is listed here only
  so the change is visible from the derivation duty. The close/far split remains a
  reported non-degeneracy diagnostic.

**Sequencing consequence.** Run Layer 3 (geometry only) and Layer 2 (model-free, shipped
`causal/`) **before** booking the GPU. Either may settle its question outright, and if both
land first the owner run *confirms* rather than *discovers* — the condition under which
pre-registration is worth anything.

**[REVIEW ROUND 1 — S2 accepted] The table's PRECEDENCE must be tested, not just its rows.**
"Parametrized-tested over every row" pins each row's mapping; it does NOT pin the ordering,
and the conditions genuinely overlap (rows 4 and 5 can both hold; rows 1–2 can co-fire with
6–7). Add precedence tests built on inputs that satisfy **≥2 rows simultaneously**,
asserting the earlier row wins. The same duty applies to §3.5's table wherever its rows can
co-fire. This is the difference between "the table is tested" and "the table's semantics
are tested".

---

## 7. Testing

- **Construct/CI guards (synthetic, all matrix legs)**: ghost-substituted-at-actual ⇒
  deltas exactly 0 (identity) PLUS a value-realistic nonzero-displacement fixture
  asserting nonzero deltas (a no-change test must exercise the path that can change the
  value); mirror-invariance over the engine AND the model extractors (magnitudes
  invariant, signed features flip) — extending the `test_y_blast_radius_ab.py` idiom to
  `extract_xshot_features` / `extract_xcross_features` / ghost-GK extraction;
  write-back flip tested at both goal directions; drop-reason accounting with
  **conservation asserted** (scored + dropped == total; the fixture plants ≥1 frame
  per §4.1 drop reason) and tests that the clamped AND out-of-box flags each FIRE (a
  stub returning only in-box positions leaves the §3.1 strata machinery dead forever);
  purity (caller frames never mutated); a **scored-frame GK `player_id` non-null
  invariant** (replacing the earlier player_key contract idea — see §5); the **probe
  discriminating-power meta-test** (§3.1: MIXED-dependence planted model PASSES,
  GK-blind model FAILS on the ratio/band prongs, with fixture-validity preconditions
  asserted in-test); the **planted-polarity sign test** (§5: both arms negative for an
  obviously-deterrent actual keeper); the **shared-cache silent-zero guard** (§5:
  non-zero deltas on a fixture where the two legs' surfaces provably differ); a
  **chirality-guard test** (`load()` RAISES on mismatched AND on missing fingerprint,
  legacy override exercised); and a **seeded planted-ICC e2e chain test**
  (frames → engine → arms → `_metric` → `_validate`: 2–3 synthetic keepers, one
  systematically offset, keeper separation above the match-block null + golden
  aggregates; slow-marked per ADR-023).
  Mirror tests carry in-test preconditions that every signed feature is NONZERO
  pre-flip and assert EXACT negation (not `change > 1e-6`); the engine-level mirror
  test uses a synthetic y-equivariant model stub (the real booster is not
  equivariant). ΔDAS determinism is verified at plan time; if `accessible_space` is
  stochastic, a seed is registered in `GkdvParams` and threaded through both legs so
  the identity assertion stays exact.
- **Probe**: generalized core vs xCross wrapper byte-equivalence against a GOLDEN
  report captured from PRE-refactor code on a committed synthetic fixture (else the
  test proves the new code equals itself), with pinned compared fields and a stated
  cross-platform float policy; xS wrapper constants pinned; the every-wrapper-has-a-
  pinned-rule-test meta-assertion iterates an explicit `PROBE_WRAPPERS` registry
  (without a registry it cannot discover an unregistered wrapper). Prose contracts
  become assertions: probe-sample provenance ids in `metrics.json` (schema test),
  sampled-counts-never-silent, and the §8 high-drop-rate warn threshold as a pinned
  constant.
- **Causal promotion**: parameterized builder reproduces the xCross default
  byte-identically; `tests/causal/` known-truth gates stay green unmodified.
- **Examples gate**: every public `gkdv` export carries a house-style `Examples`
  docstring AND the `gkdv` modules register in
  `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` from day one (the xtgk
  package shipped outside that gate — a gap noted in the 4.46.0 spec §7; gkdv does not
  recreate it).
- **Slow-marking**: ADR-023 discipline — platform-invariant smokes/parity may be
  `slow`; golden/numeric and cheap behavioral contracts run on all legs.
- **[REVIEW ROUND 2 — N3(b)] THIS LIST SPANS BOTH PRs AND MUST BE PARTITIONED.** As
  written it reads as one undifferentiated block, so a PR-3 author will take all of it as
  scope. Label every bullet **PR-3** or **PR-3b** at plan time. The split follows the §9
  item-4 registration: probe meta-test, planted-polarity arm fixture, shared-cache guard,
  chirality guard, planted-ICC chain, Layer-4 anchoring tests, `provenance_to_targets`
  contract tests, and the Examples/`_PUBLIC_MODULE_FILES` registration → **PR-3**;
  `gkdv_discrimination_verdict` row + PRECEDENCE tests, Layer-0 saturating-dose and
  support-diagnostic tests, Layer-2 planted-effect positive control and overlap diagnostic,
  Layer-3 headroom probe tests, and the `causal/` legacy-byte-identity check → **PR-3b**.
  Note the both-sides-band amendment below deliberately spans both.
- **[AMENDED 2026-07-18] The non-vacuity mandate has a hole at the two §6 statistics that
  lack a planted counterpart, and it is the anti-conservative direction that is missing.**
  The match-block null is currently exercised only in the DETECT direction (a planted-ICC
  fixture must separate *above* the band). There is **no test that the band is correctly
  SIZED** — i.e. that a genuine no-effect fixture lands *INSIDE* it. Without that, an
  anti-conservative band ships undetected, which is the exact hazard §6.1 spends a
  paragraph justifying the match-block design against ("a guaranteed-significant
  instrument"). Add the null-direction test: no-effect fixture ⇒ inside the band. The same
  applies to §6.4's placebo bands. **Rule of thumb this cycle keeps re-learning: every
  band needs a test from BOTH sides, and every counterfactual needs a non-vacuity
  assertion that it actually moved something** — four separate silent-null defects across
  this programme (a y-inversion, a fabricated grid origin, an identity-keyed cache, and a
  mirrored external-provider event frame) have all had this shape.

## 8. Error handling

Fail-loud: unfitted/missing `ExpectedThreat` or ghost model; all-null
`team_attacking_direction` (raise, route caller to the ADR-029 orient helpers);
two-team guard. Degrade-with-report: zero eligible frames → empty result + report;
`[das]` absent → DAS arm skipped-with-report; high drop rates warn and are tallied.

**[AMENDED 2026-07-18] Degrade-gap: the ARMS have no "unmeasurable at this dose" analogue.**
§8 covers *zero eligible frames*, and §3.1 gives the probe an explicit floor-and-return for
a degenerate counterfactual — but the arms have no run-level guard. A run whose ghost
displacements are all ≈0 produces arm values ≈0, a full report, and **no warning**; §5's
minimum-nonzero-observation rule gates keeper *inclusion*, not the run verdict. That is the
§1.3 support-check lesson in a new place, and it is the same silent-null shape as everything
else this amendment corrects. Add a run-level degrade verdict for the arms mirroring the
probe's, and note it composes with §6.4 Layer 0 (which catches the stronger case where even
a *saturating* dose fails to move the arm).
**[REVIEW ROUND 2 — N3(c)] What PR-3 ships ALONE**: the run-level arm degrade verdict, on
the OBSERVED ghost displacements only — **without** the Layer-0 saturating-dose check,
which lands in PR-3b. So between the two PRs the guard catches "the ghost barely moved" but
NOT "the arm cannot respond even to a teleported keeper". PR-3's reports must say so rather
than implying full instrument validation.

---

## 9. PR sequencing

Each PR bumps + tags + publishes (per-PR convention; no version/ADR numbers reserved —
next-free at release time). **Sequencing**: the parallel resolved-origin cycle landed as
4.46.0 (ADR-036 amendment, no new ADR) — this cycle's versions are next-free after it,
ADR-037 is free, the TODO line-28 collision is resolved, and the
`xtgk_v2_keeper_discrimination.py` lift is unblocked (§6.1).

1. **PR-1 (re-gate code)**: `_model_eval` generalization + xS probe with registered
   rule + `causal/` promotion with the FULL builder surface (§2.2, incl. the §3.3
   result-conditioned outcome axis) + xS extractor canonical-id hardening +
   probe-sample provenance + **chirality-fingerprint EMISSION in the
   retrain-script/`save()` surface** (§3.4 — the DGX runs must write it) + the §3.5
   verdict function + TF-19 TODO-entry updates (Ter Stegen; status prose) + ADR-037
   (incl. the gkdv→tracking dependency rule and the §5 worked sign example) + ADR-015
   update + C4 edit. **The probe core is a pure function over pre-substituted inputs**
   (frames/ΔP/provenance as data; composition lives in scripts/) — `tracking/` never
   imports `gkdv/`.
2. **Owner DGX runs, SPLIT (the xS dose-banded probe needs the §4 engine, which is
   PR-3)**: after PR-1 — the three retrains (paired GS test re-run), the frozen xCross
   probe re-run, and both causal arms; after PR-3 — the xS dose-banded probe (ghost
   substitution + OOD strata come from the engine).
3. **PR-2 (weights + verdicts)**: new bundled weights ×3, fresh gate results, decision-
   table verdict recorded. Hyrum flags: retrained ghost-GK changes served `ghost_gk_*`
   (lakehouse re-materializes); xS/xCross weight changes shift
   `pre_shot_gk_full_default_xfns` columns for opted-in consumers (retrain trigger for
   them).
4. **PR-3 (gkdv package, parallel after PR-1)**: engine (consuming the §4 tracking
   serving seams) + arms + metric + validation code + tests + NOTICE. Parallelism
   rule: gkdv tests assert numerics against SYNTHETIC/fixture models only, never the
   bundled `default` weights (else green/red depends on PR-2 merge order); version and
   ADR numbers resolve by rebase order at release time per the no-reservation policy.
   **[AMENDED 2026-07-18] Keep the parallelism rule; its rationale changed but did not
   expire.** As a *sequencing* constraint it is spent — PR-2 has merged. As a *design
   principle* it must be RETAINED, for three reasons this document could not have known:
   (a) the `sc_extended` variant is **HF-only and the Hub repos do not exist yet**, so any
   test pinning a non-default variant is still order-dependent on an owner upload;
   (b) PR-2's `load()` chirality enforcement is **fail-closed**, so a test touching
   bundled weights now couples gkdv's CI to artifact metadata integrity; (c) the bundled
   weights will move again if the attempt track routes to GK feature engineering.
   Synthetic/fixture models keep gkdv's suite independent of all three.
   **[REVIEW ROUND 1 — B2 accepted: §6.4 roughly DOUBLED this item's scope and the
   sequencing was never amended to match. PR-3 is therefore SPLIT.]** §6.4 adds, all
   nominally inside "validation code": Layer 0 saturating-dose machinery per arm; Layer 1
   dual-regime responsiveness with paired-vector controls extended to the physics arms;
   Layer 2, a complete causal study (treatment, two outcomes, estimator config, six
   confounder families, negative AND positive controls, overlap diagnostic); Layer 3 a
   feature-headroom probe; Layer 4 tercile anchoring; an 11-row
   `gkdv_discrimination_verdict` with precedence tests; and a plasmode power simulator.
   That is a second PR's worth of work — and §6.4 is explicitly **unsigned-off**, so it
   cannot gate the package landing. Registered split:
   - **PR-3 (package)**: `gkdv/` skeleton, `_engine.py` (incl. the named
     `provenance_to_targets` adapter + its contract tests), the three `_ghost_gk` serving
     seams, `_arms.py`, `_metric.py`, the `_group_metrics.py` lift, **§6.1–6.3 validation**,
     **plus §6.4 Layer 4 (behavioural anchoring) — [REVIEW ROUND 2, N3(a)] it guards §6.1's
     ICC, which ships here, and is the cheapest item in §6.4**, tests, NOTICE, C4.
   - **PR-3b (discrimination harness)**: §6.4 Layers **0–3** + `gkdv_discrimination_verdict`
     (incl. its precedence tests) + the derived constants, **after sign-off** —
     **plus [REVIEW ROUND 2, N1] the one additive `causal/` field
     `OpportunityConfig.outcome_max_distance_m` + its `_label_outcome` geometry +
     the byte-identical-legacy regression check**, which no earlier scope listed. The owner
     validation run and PR-4 follow PR-3b, not PR-3 — Layers 2 and 3 are the cheap decisive
     experiments and must precede the GPU booking (§6.4 sequencing).
   - **[REVIEW ROUND 2 — N6] Both PRs write `gkdv/_validate.py`.** PR-3 ships the §6.1–6.3
     constants (and Layer 4's) there; **PR-3b EXTENDS that module, it does not rewrite it**,
     and the constants PR-3 froze stay **byte-identical**. Stated explicitly because §3.5's
     amendment already had to warn against a PR author re-homing an
     already-frozen verdict function.
5. **Owner validation run → PR-4**: `docs/research/gkdv_v1/` findings, gate verdicts,
   TODO/ADR amendments. If the re-gate passed, the Δattempt arm gets its own follow-up
   spec (composition + outcome-value weighting are NOT designed in this cycle).

### [AMENDED 2026-07-18] 9.1 Status, TODO obligations, and a live collision risk

**Status.** Items 1–3 are DONE (PR-1 = 4.47.0; DGX retrains; PR-2 = 4.51.0). Of item 2's
owner runs, the retrains, the frozen xCross probe re-run and both causal arms have run;
**only the xS dose-banded probe remains, and it is PR-3-gated** exactly as item 2
anticipated. Items 4–5 are the remaining work.

**TODO.md obligations for PR-3** (house rule: no standalone doc commits, so these fold
into the code PR):
- **Line 50 (GKDV research-program paragraph) carries stale PRE-retrain gate numbers**:
  it still says the re-gate is "in flight (PR-S114)" and quotes median `0.00107`,
  "2.59× the nearest-defender control", "misses the absolute floor by ~10×". The
  authoritative post-PR-2 figures are `0.009697`, ratio ≈2.21×, and a ~10% *relative*
  miss on the floor prong alone. Keep "the xS arm has never been measured at all" — it
  is still true — but re-scope it to **PR-3-gated** rather than merely pending.
- **Lines 58–68 (the known-failure `test_xshot_gradientsports_e2e` entry) have met the
  entry's own stated removal condition** — it was re-run on 4.51.0 on 2026-07-18 and
  **PASSED** (1021 s). Remove it, but **record the caveat rather than deleting silently**:
  the run used local xgboost 2.1.4 while the artifact was produced under 3.2.0, and PR-2
  shipped a `base_score` 2.x/3.x guard for exactly that skew — so state the xgboost
  version the pass was obtained under.
- Header "Current release: silly-kicks 4.50.0" → 4.51.0; the "98 owner-tier SkillCorner
  matches" entry should become "weights landed 4.51.0; `sc_extended` is HF-only and the
  Hub upload is the remaining owner action"; the TF-19 entry proper still describes the
  validation strategy as needing design.

**⚠ COLLISION RISK — highest-severity, needs user mediation before PR-3 starts.** A
parallel session is taking items from TODO.md's *Course-derived candidates* table. Its
**"Course-derived validation/QA bundle"** row overlaps PR-3 directly:
- item (f) proposes adding xGChain "as a near-zero-cost extra baseline for the xT-GK v2
  construct-validity harness" — that harness is **`scripts/xtgk_v2_keeper_discrimination.py`**,
  the exact file §6.1 mandates re-pointing at the new `silly_kicks/_group_metrics.py`, in
  the same PR as the lift;
- item (c) builds on `silly_kicks/_calibration_metrics.py`, the named precedent the
  `_group_metrics.py` lift is modelled on.
Two sessions editing the same script — one re-homing its functions, one adding a baseline
to it — is a merge conflict on a file whose whole point is single-sourcing. §6.1 already
requires re-confirming via the user if another xtgk cycle is in flight when PR-3 starts;
**this is that condition, and it is met.** Resolve ownership of that file before PR-3
touches it.

**⚠ [REVIEW ROUND 1 — B3] SECOND COLLISION, measured, and this spec could not see it: the
C4 `tracking` box.** The parallel session is executing **PR-S119** (real-xT EPV wiring +
TF-35 run valuation, targeting 4.52.0). It adds a new `add_*` aggregator
(`add_off_ball_run_values`) and its release task takes the C4 count **29 → 30**. Measured
on the current DSL: the `tracking` container description is **191/200 characters** and
contains the literal `29 action-coupled aggregators`; two other boxes are at **exactly
200**. Therefore **both PRs must edit the same near-cap string**: PR-S119 changes 29→30
(net zero characters), while TF-19 PR-3 must fit a gkdv reference into **9 characters of
headroom**. The conflict is a merge conflict on a **gate-enforced** line where resolution
is NOT mechanical — whoever lands second must decide which of the other session's prose to
delete in order to fit, under a red CI gate.
Smaller shared surfaces between the two PRs: both append to
`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` (trivial conflict, but both must
expect it), and both take a next-free version number (no-reservation policy, so fine).
**Recommended resolution — needs the owner, alongside the collision above**: agree a merge
ORDER, and agree that whoever lands second **re-derives the aggregator count and re-trims
the box**, rather than either session pre-writing a number.

---

## 10. Out of scope / deferred

- **[AMENDED 2026-07-18] `openGoal` keeper-exclusion**: §3.1 records that the xS
  `openGoal` feature and all 10 `DefDist`/`DefAngle` pairs are computed from a
  GK-EXCLUDED defender array, which is both the mechanical cause of the 2-of-27 channel
  and arguably physically wrong (a keeper on his line does not reduce "open goal").
  Changing it is a **feature-contract change requiring a full ADR-011 retrain cycle** and
  is explicitly NOT in PR-3. It is the first-named target if §6.4 returns
  `H1_supported` with headroom ≥ 0.02. Tracked here so the finding is not lost between
  cycles.
- **key_pass arm**: no P(key_pass) model exists and "key pass" is not a SPADL type; a
  derived-label xKeyPass sibling is a full ADR-011 lifecycle — explicitly deferred,
  tracked in the TODO entry.
- **Δattempt composition + outcome-value weighting**: verdict-gated; own spec cycle.
  The outcome-value function choice (realized vs xG vs xT) inherits the xT-GK v2
  V-reward fork and is deliberately not pre-decided here.
- **Cross-arm composite GKDV**: gated on per-arm validation.
- **Metrica**: excluded until derive-once-per-match GK identification ships.
- **Public model-eval home**: deferred until an external consumer exists.
- **Lakehouse migration** of any GKDV columns: separate follow-up, user-mediated.

## 11. References

Le et al. 2017 (ghosting); arXiv:2505.11841 (causal crossing); arXiv:2512.10355
(DEFCON-GNN); arXiv:2512.00203 (xS; v2 suppression suggestion); arXiv:2601.00748
(Groom et al., role-conditioned ghosting); Bischofberger & Baca 2026 (DAS); Cascioli et
al. 2025 (cover shadows). ADRs: 008, 011, 015, 016, 019, 022, 023, 024, 028, 029, 031,
036. Gate record: `_xcross_weights/default/metrics.json`; causal record:
`docs/research/xcross_causal/`.
