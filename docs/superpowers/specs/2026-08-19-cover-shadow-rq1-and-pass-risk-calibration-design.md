# Cover-shadow RQ1 + pass-risk calibration — real-data validation cycle (design)

**Date:** 2026-08-19
**Status:** design draft, for review
**Scope:** TF-30(b) (Cascioli et al. 2025 §RQ1) + Course-QA item (c) (pass-risk calibration), combined into
one reported-not-gated validation cycle on GS WC2022.
**Sources:** `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md` (§7.3/§9.2 — the shipped
smoke test and its stated fixture limitation); TODO On-Deck rows "TF-30 (b)" and "Course-derived
validation/QA bundle (c)".

> This cycle **measures** two already-shipped predictors against real pass outcomes. It ships **no**
> library behaviour change, **no** retrain, and **no** gate on the numbers — it produces two auditable
> research artifacts. The σ/λ recalibration it enables, and the failed-pass expected-receiver model it
> would ideally consume, are BOTH deferred and separately tracked (§9).

---

## 1. Purpose

Two shipped predictors have never been checked against real pass outcomes:

- **Cover-shadow lane blocking** (`_cover_shadows`, TF-30(a), shipped 4.67.0) ships with the paper's
  σ=0.20 / λ=4.3 defaults and a **10–60% predicted-block-rate smoke test** — explicitly *not* a
  precision/recall measurement, because (per the design doc §9.2) "it requires ground-truth pass
  outcomes we don't have in fixtures." True of fixtures; **false of real data**, where SPADL `result_id`
  carries the outcome.
- **`pitch_control_at_target`** (shipped) is used as a pass-viability signal but its **calibration** as a
  pass-success predictor has never been measured on real data.

Both are **continuous predictors of pass viability evaluated against `result_id`** — the same evaluation
harness, corpus, and outcome join. Combining them is not a bundle of convenience: it is one measurement
question asked of two surfaces, so one corpus pass produces two artifacts.

## 2. The metric hierarchy — the leakage-free anchor leads; the continuous scores are caveated

An earlier draft led with ROC-AUC as "the cleaner headline." That is **backwards**, for a reason specific
to how pass outcomes are observed.

**The failed-pass target is outcome-selected.** A completed pass has an observed receiver — open by
selection (it completed) → low `p_blocked`. A failed pass has **no observed intended receiver**, so it is
scored at the release-frame `end_xy` — but `end_xy` for a failure is *where the pass was lost*,
disproportionately at the defender that caused the loss → high `p_blocked`. So failed passes carry high
`p_blocked` and completed passes low `p_blocked` **by construction of the outcome, not the model's skill**.
Release-frame defenders do not remove this — the leak is in *target selection*, and it is **intrinsic to
using failed passes at all** (using `end_xy` for both classes would not fix it — a failure endpoint is
defender-adjacent regardless). Every metric that reads a failed pass — **AUC, reliability slope, recall,
balanced accuracy** — is therefore **leakage-inflated / optimistic**. The same mechanism hits Driver B
(pitch control at a failure's interception endpoint is low *because* a defender is there), so its continuous
scores carry the identical caveat (§5).

The hierarchy that follows:

1. **Headline — leakage-free.** The completed-pass **false-positive rate** = `P(predicted-blocked | completed)`.
   It conditions ONLY on completed passes (real receivers, open-by-selection), touches no failed-pass proxy,
   and directly measures over-prediction: how often the model calls a lane blocked that the pass actually
   completed. Reported for the `center` / `mean` / `max` lane aggregations, not `mean` alone (§4).
2. **Optimistic, caveated — for paper comparison + as the recalibration baseline.** ROC-AUC(`is_fail`,
   `p_blocked`), the reliability slope, and the binary recall / balanced accuracy — reported *with* the §6
   leakage caveat, never as the trustworthy result.
3. **Baseline, not a verdict.** `ece(is_fail, p_blocked)` is large *by construction* — `p_blocked` is
   `P(lane screened)`, a different quantity and scale from `P(fail)` — so it is the **pre-recalibration
   baseline the σ/λ cycle drives down**, not evidence of miscalibration. The reliability *slope* (does
   empirical failure rate rise monotonically with `p_blocked`?) is the informative shape read, itself an
   optimistic monotonicity check under the failed-pass leakage.

The continuous framing still earns its keep — the `p_blocked → empirical-failure-rate` **reliability curve
is the objective the deferred σ/λ recalibration (TF-24) optimises against** — but a caveat must travel to
that handoff: the curve is computed on the **attempted-pass distribution** (open-biased), while
cover-shadow's intended use is scoring **decision-time / counterfactual lanes, many never attempted**. So
recalibrating σ/λ against it tunes the model for the *observable slice*; TF-24 must know it is optimising a
selection-biased objective and not over-fit it (restated at the §9 handoff).

**Scope boundary — this cycle measures OVER-PREDICTION, not DETECTION (R1).** Both drivers' leakage-free
legs (Driver A's completed-pass false-positive rate; Driver B's completed-pass false-alarm rate) measure
only ONE direction: how often the model flags a lane blocked/low-control that the pass nonetheless completed
— i.e. specificity on completed passes. **Recall / sensitivity — detecting real blocks/failures — is not
cleanly measurable here**, because it needs the failed-pass class, which is both leaked (above) and
confounded (§6), until the deferred Power-2017 receiver model lands. So the clean headline answers *"does the
model over-predict?"* and NOT *"does it detect real blocks?"* — and the paper's headline (recall 36.9%) is
exactly the direction this cycle can only measure optimistically. A reader must not take the clean headline
as a full validation; the artifacts state this in one sentence.

## 3. Architecture

**Two `scripts/validate_*.py` drivers sharing one corpus-loading helper**, mirroring the established
convention (8 existing `validate_*` drivers). Each driver is independently shardable, registered, and
provenance-stamped; each emits its own `docs/research/` artifact.

```
scripts/
  _rq_corpus.py                     # shared: pining GS WC2022 -> (actions, frames, links) per match
  validate_cover_shadow_rq1.py      # Driver A
  validate_pass_risk_calibration.py # Driver B
docs/research/
  cover_shadow_rq1/                 # Driver A artifact
  pass_risk_calibration/            # Driver B artifact
```

- **Corpus:** GS WC2022 via `scripts/_loader_pining` (owner-tier), full 25 Hz measured-velocity tracking.
  (SB360 is deliberately NOT used — this cycle is non-SB360; and GS carries real velocities, so the 4.85.0
  velocity-less lift is irrelevant here.) Cohort extensible later; GS WC2022 is the pinnable start.
- **Per-match work is the expensive part** (parse a full tracking frame per match), so both drivers use
  `scripts/_driver.py::for_each` (ADR-052) — one shard per match into a fingerprinted generation
  directory, resume-safe, conservation- and injectivity-asserted.
- **No library code change is required.** The drivers consume existing seams: the `_cover_shadows` per-lane
  primitive, `pitch_control_at_target`, `resolve_next_touch_receiver`, `_calibration_metrics`
  (`ece`/`reliability_slope`), and sklearn `roc_auc_score` (already a runtime dep). Any private seam
  consumed is recorded in `docs/PRIVATE_CONSUMERS.md`; no new public API unless a thin public wrapper is
  preferred at implementation time.

## 4. Driver A — cover-shadow RQ1 validation

**Unit of analysis:** one played pass (SPADL pass/cross actions with a resolvable target and a linked
release frame).

**Per pass:**
- `passer_xy = (start_x, start_y)` (the release position).
- **Target (`receiver_xy`):**
  - **completed pass** → the receiver's release-frame position, via the existing `resolve_next_touch_receiver`
    (identifies the receiving teammate) looked up in the linked frame. This leg is leakage-free — the
    receiver is observed, and the lane evaluated is the one the pass was actually played into.
  - **failed pass** → the release-frame `end_xy` proxy (the intended receiver is unobserved). This target is
    **outcome-selected** — a failure ends where it was lost, disproportionately defender-adjacent — so it
    inflates every metric that reads it (§2). The leak is in target *selection*, not in the defender frame,
    and it is intrinsic to using failed passes. The assumption is stated in the artifact; the rigorous fix
    (a Power-2017 expected-receiver model) is deferred and tracked (§9), and the failed-pass-dependent
    metrics are re-run through it when it lands.
- **Defenders:** from the linked release frame.
- **Score:** run the `_cover_shadows` per-lane primitive → `is_blocked_majority` (binary; the paper's
  majority rule, `n_blocked ≥ 2` over center/left/right) and the three continuous aggregations
  **`p_blocked_center` / `p_blocked_mean` / `p_blocked_max`**. The continuous discrimination is reported for
  ALL THREE — privileging `mean` would be an unexamined aggregation choice that could bury a strong
  center-lane signal (M3); `mean` stays the basis for the binary majority rule only.
- **Outcome:** `is_fail = (result_id == fail)`.

**Metrics reported (hierarchy per §2):**
- **Headline — leakage-free.** The completed-pass **false-positive rate** = `P(is_blocked_majority | completed)`,
  conditioned ONLY on completed passes (real receivers) so no failed-pass proxy enters it (M2). Reported for
  `center` / `mean` / `max`. This is the trustworthy number and leads the artifact.
- **Optimistic, caveated** (read failed passes → leakage-inflated, §2/§6): `roc_auc_score(is_fail, p_blocked)`
  for all three aggregations; the binary `is_blocked_majority` × `is_fail` recall / specificity / balanced
  accuracy; the reliability slope. Each carries the leakage caveat inline; none is the headline.
- **Recalibration baseline (not a verdict):** `ece(is_fail, p_blocked_mean)` + the `p_blocked → empirical-
  failure-rate` reliability curve — the object the σ/λ cycle drives down, with the §9 selection-bias caveat
  (M4).
- **Paper reconciliation** (this is the ONLY place the confusion matrix earns its keep): the README MUST
  carry the one-line comparison to the independently-recomputed Cascioli Appendix-B rates (Lane Control
  Majority recall 36.9% / precision 22.0%; All 34.2% / 28.8% — **recomputed, not trusting the handoff
  table**), e.g. "our recall X vs the paper's 36.9%; consistent / differs because Y." A confusion matrix
  with no comparison sentence is clutter and is not shipped (Q1).

## 5. Driver B — pass-risk calibration

**Unit of analysis:** the same played passes.

**Per pass:** predictor = `pitch_control_at_target` (the acting team's pitch control at the pass target),
outcome = `is_success = (result_id == success)`. **Target handling is identical to Driver A** (completed →
receiver's release-frame position; failed → release-frame `end_xy`), so **the §2 failed-pass leakage applies
here too**: pitch control at a failure's interception endpoint is low *because* a defender is there, inflating
any metric that reads a failed pass.

**Metrics reported (same hierarchy as Driver A — symmetric clean headline, R1):**
- **Headline — leakage-free false-alarm rate.** `P(control < τ | completed)` for τ ∈ {0.1, 0.2, 0.3}: of
  passes that *succeeded*, the fraction the control model flagged low-control. Completed-only (clean
  targets), it is the direct symmetric analog of Driver A's `P(predicted-blocked | completed)` and the
  trustworthy number. (Control *among completed passes* on its own is NOT a metric — completed passes are all
  successes, so there is no within-class outcome contrast; the false-alarm rate is what carries content.)
- **Optimistic, caveated** (read failed passes → inflated, §2/§6): `roc_auc_score(is_success, control)` +
  reliability curve + `ece(is_success, control)` + `reliability_slope(is_success, control)`.
- **Low-control COMPLETION band is CONTAMINATED, not the headline.** `P(success | control < τ)` at
  τ ≈ 0.1 / 0.2 / 0.3 (a 3-point curve, Q2) — the "technically complete, functionally lost" read — mixes
  failed passes, which cluster at low control via the `end_xy` selection (§2). Reported caveated and kept
  distinct from the clean false-alarm headline above; the two are never conflated.

## 6. Limitations — stated in both artifacts (this is the whole point of reported-not-gated)

- **Selection bias (unfixable, fundamental).** Only *attempted* passes are observed, biased toward open
  lanes; a blocked lane that was never attempted has no ground-truth outcome. Precision is therefore a
  **lower bound** — a good passer threading a genuinely screened lane scores as a false positive by
  construction.
- **Failed-pass target is outcome-selected (the core caveat, §2).** `end_xy` for a failure is
  defender-adjacent by construction, so AUC / recall / specificity / slope (Driver A) and the AUC/curve
  (Driver B) are **optimistic bounds**, not clean reads; the completed-pass false-positive rate (§4 headline)
  and completed-only control are the ONLY leakage-free legs. The deferred expected-receiver model (§9)
  supplies a leakage-free failed-pass target; re-run those metrics through it then.
- **Screening ≠ failure; control ≠ completion.** See §2 — the reliability curves are mappings, not proofs
  of calibrated outcome probabilities.
- **Recall is confounded by failure cause.** A failed pass is not necessarily a screened-lane event (bad
  touch, receiver marked, error elsewhere) — a second reason, beyond the §2 target leakage, that the
  failed-pass legs are optimistic. The **completed-pass false-positive rate is the only trustworthy leg**;
  AUC / recall / balanced-accuracy are paper-comparability reads under both caveats.

## 7. Driver discipline (mandatory, CI-checked where the gate exists)

- **ADR-037 provenance (`scripts/_provenance.py`, CI-gated since 4.65.0):** each driver calls
  `require_clean_tree(git_provenance(), ...)` in `main()` before any corpus work, stamps `run_commit` +
  `run_tree_dirty`, offers `--allow-dirty` (dev only; the artifact still records `dirty: true`), and never
  shells out to `git rev-parse`. Auto-discovered and enforced by `tests/scripts/test_provenance_wiring.py`.
- **ADR-052 sharding:** `for_each(items, key=match_id, work=, shard_root=, token_inputs=)`; declared
  `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION` pinned together; conservation + injectivity asserted;
  an empty result still writes a shard.
- **ADR-056 registry + input contract:** both drivers enrolled in `ARTIFACT_DRIVERS` (the population gate
  asserts membership exactly); `declare_inputs()` digests the covariate/param identity that the numbers
  depend on — here the `CoverShadowParams` (σ, λ) and lane rule, the pitch-control method + params, and
  `GEOMETRY_VERSION` — so a param change moves the digest.
- **Corpus visibility:** GS WC2022 is owner-tier; the artifacts carry the correct ship-mask label via
  `scripts/_corpus.py` (fail-closed).

## 8. Testing

- **Metric wiring (unit, in CI):** synthetic labelled sets with hand-derived AUC / precision / ECE prove
  each metric is wired correctly (a perfectly-separating score → AUC 1.0; a constant score → AUC ~0.5 and
  `reliability_slope` NaN).
- **Non-vacuity (asserted IN the driver, fails the run):** `n_passes` above a floor **set relative to GS
  WC2022's known pass count** (not a round-number placeholder, so a half-empty run trips it — Q2); **both**
  outcome classes present; **both** completed and failed passes present; **enough completed passes for the
  leakage-free headline** to be estimable; the lane primitive / pitch-control returned finite scores for a
  non-trivial fraction. A green run cannot be an empty one — the recurring silent-null trap this codebase
  names repeatedly.
- **Sharding guards (unit):** `for_each` conservation + injectivity; schema-token completeness.
- **e2e (owner-run, GS data, `@e2e`, not CI):** the full GS WC2022 pass, producing both artifacts on a
  clean tree.

## 9. Deferred / out of scope (each with its home)

- **σ/λ recalibration — deferred, retrain trigger, its OWN cycle (TF-24).** This cycle *supplies the
  objective* (Driver A's reliability curve); it does not change `CoverShadowParams`. Kept separate because
  a param change moves cover-shadow outputs → VAEP retrain trigger, a different PR discipline than a
  reported-not-gated artifact. **Handoff caveat (M5):** the reliability-curve objective is computed on the
  attempted-pass distribution (open-biased), while cover-shadow scores decision-time / counterfactual lanes
  (many never attempted). σ/λ tuned to it fits the *observable slice*; the artifact MUST carry this at the
  handoff so TF-24 does not over-fit it — handling the selection bias (reweighting, or an explicit
  attempted-only-scope statement) is TF-24's problem, flagged here so it is not discovered late.
- **Failed-pass expected-receiver (Power 2017) — deferred, its OWN On-Deck item** (added 2026-08-19; shared
  primitive with TF-51 Track B). A trained model (ADR-011 weights/chirality/training pipeline), deliberately
  NOT folded into this validation cycle. Until it lands, Driver A recall uses the §4 release-frame proxy;
  re-run recall through it when built.
- **Paper RQ2 (SoccerMap CNN threat surfaces) / RQ3 (positioning optimiser)** — out of scope (design doc
  §1 + Appendix A); not silly-kicks-shaped.
- **The rest of the Course-QA bundle** (xT solver oracle, magnitude anchors, retention diff, repeatability,
  window convention) — separate items; only item (c) shares this cycle's corpus/harness.

## 10. Deliverables & success criteria

- **Two research artifacts** under `docs/research/{cover_shadow_rq1, pass_risk_calibration}/`, each with
  `metrics.json` (the numbers), a `README.md` (framing + the §6 limitations verbatim + the required
  paper-reconciliation sentence, §4), reliability-curve figures, and clean-tree provenance + corpus bound
  (n matches, n passes) recorded **inside** the artifact.
- **Cover-shadow:** the 10–60% smoke band is replaced by — in priority order — the **leakage-free
  completed-pass false-positive rate** (headline, `center`/`mean`/`max`), the caveated continuous AUC +
  reliability slope, the ECE recalibration baseline, and the paper-reconciliation sentence beside the
  recomputed Cascioli rates.
- **Pass-risk:** the caveated pitch-control AUC + calibration, the completed-only clean leg, and the
  low-control 3-point band.
- Both **non-vacuous** and **reported, never gated** — the artifacts inform the σ/λ recalibration decision;
  they do not themselves pass or fail CI on their numbers. **Each states in one sentence that it validates
  OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall) — the latter awaits the deferred
  receiver model (§2 scope boundary)** — so the clean headline is not read as a full validation.

## 11. C4 / retrain

C4-free (no new aggregator/backend/model — two `scripts/` drivers, aggregator count unchanged). **No
retrain** (no library behaviour change; existing features consumed as-is). Its own ADR at implementation
time recording the reported-not-gated decision, the **leakage-aware metric hierarchy** (the completed-pass
false-positive anchor leads; failed-pass metrics are caveated-optimistic), and the two deferrals.
