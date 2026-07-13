# TF-19 GKDV — Gate Re-run + Physics-Arms v1 — Design

**Date**: 2026-07-12
**Status**: Approved design, pre-plan
**Scope decisions (owner)**: re-gate the attempt-probability arms first AND build the
gate-independent physics arms in parallel; frozen rule governs the xCross probe re-run,
the new xS probe registers its own rule; Approach A architecture (new `silly_kicks/gkdv/`
package + pre-planned promotions).

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

### 2.3 Documentation / infra impact

New ADR-037 (GKDV composition + re-gate policy, covers both tracks). ADR-015 status →
promoted. C4 DSL edit + regen (new `gkdv` element, same treatment as `xtgk`); gkdv adds
no `add_*` action-coupled aggregator, so the cap-tested count **stays 28** (definition:
`len(add_* in tracking.__all__) − 1`, excluding the roster helper
`add_gradientsports_player_ids` — raw 29 = counted 28). NOTICE
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
- **The accuracy-vs-magnitude paradox (stated up front, it shapes the rule)**: the
  ghost's held-out MAE is 1.07 m — BELOW the frozen panel's smallest step (2 m). A
  ghost good enough to be credible is, by construction, usually too close to the
  actual keeper to move a piecewise-constant boosted surface; retraining the ghost on
  corrected data will shrink typical displacements further. A naive median over all
  ghost-substituted frames therefore under-perturbs by design and cannot be read as
  "the surface is GK-insensitive" — the deterrent signal lives in the displacement
  TAIL where the actual keeper deviates from the league-average ghost.
- **Registered rule (constants in code, PR-1, before any run). The `ready` boolean is
  the AND of every registered prong below — nothing here is advisory prose**:
  1. Ratio prong: GK median |ΔP(shot)| ≥ 2.0× the nearest-defender control, on the
     gated stratum, under paired-vector controls (see 2).
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

---

## 4. Ghost-substitution engine

`build_ghost_frames(frames, *, model, home_team_id, carrier=None, params=GkdvParams())
→ (counterfactual_frames, provenance, GkdvReport)` in `gkdv/_engine.py`.

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
6. **Provenance per frame**: ghost x/y, |ghost − actual| displacement, GK `player_id` +
   `is_goalkeeper_source`, clamped flag, ghost out-of-training-box flag (goal-relative
   x beyond the 30 m label hull — OOD-served), drop reason. The displacement
   distribution doubles as the xS probe's calibration input, and the OOD/clamped flags
   feed the probe's §3.1(3)/(6) stratification. The provenance frame's keying and its
   inclusion of DROPPED frames are part of the return contract (two independent
   consumers: the probe and `_metric`); `GkdvReport` ECHOES the `GkdvParams` used, and
   the validation script writes them into the `gkdv_v1` artifacts — registration
   without traceability is not registration.

Rules honored: counterfactual frames NEVER routed through `PitchControlCache` (ADR-008
canonical-only contract; the factual side may use one); all id seams through
`_id_compat` (ADR-019); PR-1 hardens `extract_xshot_features`' raw `==` team-id
compares to canonical ids (latent ADR-019 gap, fixed regardless of TF-19).

---

## 5. Physics arms (`gkdv/_arms.py`)

Both arms are defined in attacker-value units, `actual − ghost`, so **negative =
deterrent** uniformly.

- **ΔDAS**: `get_das(factual) − get_das(counterfactual)` on the attacking team's
  dangerous accessible space; `team_in_possession` pinned from the factual context on
  both sides. Lazy `[das]` import guard — the arm skips-with-report if accessible-space
  is absent; reuses `_das.py`'s frame preparation (pyarrow/object-dtype handling
  included).
- **ΔGK-threat-suppression** (renamed from "Δcover-shadow" after source verification —
  the arm is NOT lane-specific and is specced in its algebraically reduced form):
  `compute_blocking_score(…, defenders_to_remove=[gk])` at two GK positions has its
  removal legs cancel exactly (frames differing only in the GK row are identical once
  the GK is removed), so the two-call form reduces to
  `threat_pc(ghost frames) − threat_pc(actual frames)` — where `threat_pc` is the
  xT-weighted Voronoi pitch-control threat integral (`_voronoi_threat`), the ONLY
  channel through which the GK enters `compute_blocking_score` at all (the lane/TTI
  model excludes goalkeepers by construction, and the explicit-removal path never ran
  it anyway). The arm therefore computes `threat_pc` directly on the full actual and
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
   constants + verdict logic. Knock-ons: `tests/xtgk/test_keeper_discrimination.py`
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

## 8. Error handling

Fail-loud: unfitted/missing `ExpectedThreat` or ghost model; all-null
`team_attacking_direction` (raise, route caller to the ADR-029 orient helpers);
two-team guard. Degrade-with-report: zero eligible frames → empty result + report;
`[das]` absent → DAS arm skipped-with-report; high drop rates warn and are tallied.

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
5. **Owner validation run → PR-4**: `docs/research/gkdv_v1/` findings, gate verdicts,
   TODO/ADR amendments. If the re-gate passed, the Δattempt arm gets its own follow-up
   spec (composition + outcome-value weighting are NOT designed in this cycle).

---

## 10. Out of scope / deferred

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
