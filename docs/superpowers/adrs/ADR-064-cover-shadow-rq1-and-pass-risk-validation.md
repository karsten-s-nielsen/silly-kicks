# ADR-064: Cover-shadow RQ1 + pass-risk calibration -- a reported-not-gated real-data validation cycle

| Field | Value |
|---|---|
| **Date** | 2026-08-19 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |

## Context

Two shipped predictors have never been checked against real pass outcomes: the cover-shadow lane
model (`_cover_shadows.lane_control`, TF-30(a), σ=0.20/λ=4.3 paper defaults + only a 10--60%
block-rate smoke test) and `pitch_control_at_target` (used as a pass-viability signal, calibration
unmeasured). Both are continuous predictors of pass viability evaluated against SPADL `result_id` on
the GS WC2022 corpus. Spec: `docs/superpowers/specs/2026-08-19-cover-shadow-rq1-and-pass-risk-calibration-design.md`.

## Decisions

**1. Reported, never gated.** The cycle produces two auditable research artifacts
(`docs/research/{cover_shadow_rq1, pass_risk_calibration}/`); it changes no library behaviour, ships
no retrain, and CI does not assert its numbers. Non-vacuity is asserted IN the driver (fails the run),
never in CI.

**2. The metric hierarchy is leakage-aware -- the completed-pass anchor leads.** The failed-pass target
is the release-frame `end_xy`, which is *outcome-selected* (a failure ends where it was lost,
defender-adjacent by construction), so every metric that reads a failed pass -- AUC, reliability slope,
recall, balanced accuracy -- is optimistic. The leakage-free headline is therefore the completed-pass
**false-positive rate** `P(predicted-blocked | completed)` (Driver A) / **false-alarm rate**
`P(control < τ | completed)` (Driver B). ECE is the pre-recalibration baseline, not a calibration
verdict (`p_blocked` is P(screened), a different quantity from P(fail)). **Scope: this cycle measures
OVER-PREDICTION, not DETECTION** -- recall needs the failed-pass class, which is leaked and confounded,
so the paper's headline (Cascioli majority recall 36.9%) is exactly the direction we can only measure
optimistically. Both artifacts state this in one sentence.

**3. One corpus pass, two artifacts (build-driver + persisted-table consumers).** `build_rq_pass_scores`
is the expensive, `for_each`-sharded corpus pass (per-pass lane + pitch-control scores); the two thin
`validate_*` consumers read the persisted `pass_scores.parquet` (+ its `manifest.json`) and each compute
+ write their artifact. This realises the spec's "one corpus pass" intent (two independent `for_each`
drivers would each re-load GS) and lets the metric framing iterate without re-loading -- the codebase's
established "an expensive corpus pass is its own shardable driver; the consumer takes the table" pattern.
Each consumer REFUSES a dirty / missing / commit-mismatched upstream manifest (ADR-037 extended to every
artifact a number derives from).

**4. Corpus scope is `pass`/`cross`; Driver A's headline is PASS-ONLY.** `lane_control` models
GROUND-lane screening and crosses are aerial (the ball clears ground defenders), so Driver A leads with
the pass-only cut and reports pass+cross only as the paper-comparable secondary (with the required
one-line Cascioli reconciliation, recomputed from their Appendix B, not the handoff table). Short
set-pieces are excluded (spec §4). Driver B keeps all passes (spatial control is not aerial-specific).

**5. Owner-tier licensing: aggregate-only, ship-mask-labeled.** GS WC2022 is owner-tier. The raw
per-pass positions in `pass_scores.parquet` are NEVER committed (owner-run / gitignored `--out`); only
the consumers' aggregate `metrics.json` + `README.md` land under `docs/research/`, labeled via
`scripts/_corpus.artifact_label(providers={"gradientsports"}, all_public=False)`.

## Consequences

- **Two commits, non-squash.** Commit 1 lands the code + docs (clean tree so the driver can run); the
  three drivers are then run locally on the GS WC2022 pining corpus (stamping commit 1's SHA); commit 2
  lands the aggregate artifacts. Non-squash so the artifacts' `run_commit` resolves.
- **No retrain, C4-free** (three `scripts/` drivers; aggregator count unchanged). `_cover_shadows.lane_control`
  is consumed as a private seam (recorded in `docs/PRIVATE_CONSUMERS.md`).
- **Two deferrals, each with a home.** (a) The σ/λ recalibration is TF-24's own cycle -- this artifact
  *supplies its objective* (Driver A's reliability curve), which is **selection-biased to attempted
  passes**; the handoff carries that caveat so TF-24 does not over-fit the observable slice. (b) The
  Power-2017 expected-receiver model (the leakage-free failed-pass target) is its own On-Deck item,
  shared with TF-51 Track B; recall is re-run through it when it lands.
- **KNOWN LIMIT.** The clean, leakage-free result is a single direction (over-prediction). A future
  reader must not read the headline as a full validation of the cover-shadow model.
- **FINDING (real run) -- the discriminating score is the margin, not the magnitude.** The model's
  decision compares `p_blocked` to `p_received` **per lane**, so the continuous cover-shadow score is the
  **margin / `n_blocked` count** the majority rule thresholds -- NOT the absolute `p_blocked` intensity,
  whose AUC on GS WC2022 is ~0.51 (no discrimination) even optimistically, while the native binary
  majority rule reaches balanced accuracy ~0.68 (matching Cascioli's 68%). The shard therefore stores
  `p_received_{center,left,right}` + `n_blocked` (schema `rq-scores-2`); the artifact leads its optimistic
  AUC with `n_blocked` / mean-margin and keeps the absolute `p_blocked` AUC alongside as the "magnitude
  alone fails" comparison. The FP-rate headline is unaffected (it reads the binary verdict).
- **NaN-safety (fixed after the first real run).** ~0.8% of `control` and a stray `p_blocked` are
  non-finite on real GS data (degenerate geometry / unlinked actions); the library `ece`/`reliability_slope`
  (`np.polyfit`) raise on NaN, so every score-consuming metric in `_rq_metrics` drops non-finite scores
  first. `p_blocked` is an unbounded intensity (real max ~2.3), so the binning clips it into the last bin.
