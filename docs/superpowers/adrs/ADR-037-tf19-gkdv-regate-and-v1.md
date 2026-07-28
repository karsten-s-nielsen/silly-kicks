# ADR-037: TF-19 GKDV — attempt-arm re-gate + physics-arms v1

| Field | Value |
|---|---|
| **Date** | 2026-07-12 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; TF-19 GKDV re-gate cycle (silly-kicks session) |
| **Supersedes / amends** | none (ADR-011 gate consequence re-opened, not overturned) |
| **Source spec** | `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md` (twice cross-session-reviewed) |

## Context

TF-19 (GKDV — GK Deterrent Value) is the Layer-3 capstone of the GKDV research
program: a per-frame measure of how much the defending goalkeeper's actual position
depresses opponent attempt probabilities relative to a league-average "ghost" GK in the
same frame state. It has been gated since 4.18.0 by `tf19_ready=false` from the xCross GK
substitution probe (`_xcross_weights/default/metrics.json`). A twice-reviewed
investigation established that the gate is **not stale** — it is measured on
**chirality-mis-served** weights, which is a correctness bug independent of any gate
outcome. This ADR records the two-track decision, the corrected findings, and the
registered rules/verdict machinery. It **cites the spec for full detail rather than
duplicating it**; the sections below fix the load-bearing constants, tables, and
contracts in one durable place.

PR sequencing (spec §9): PR-1 = re-gate CODE (this ADR ships with it); owner DGX runs
between PRs; PR-2 = weights + verdicts; PR-3 = the `gkdv/` package (engine + physics
arms + validation); PR-4 = owner validation findings.

## Decision (1) — two tracks, owner scope choices

This cycle runs two tracks in parallel, per owner scope decisions:

1. **Re-gate track** — retrain xS/xCross/ghost-GK on the corrected corpus as a
   **correctness fix** (not merely a gate re-run), re-run the pre-registered
   public-vs-full paired test (GS's exclusion was decided on a mixed-chirality
   comparison), harden `load()` with a behavioral chirality fingerprint, re-run the
   frozen xCross probe apples-to-apples, add the **never-measured xS substitution probe**
   under a newly registered dose-banded / placebo-banded rule, and run a causal shot arm.
   A pre-registered decision table (`regate_verdict`) converts the results into the
   attempt-arm verdict.
2. **Physics-arms track** — ship GKDV v1 on the two gate-independent counterfactual arms
   already named in the TODO entry (ΔDAS + ΔGK-threat-suppression), built on a shared
   ghost-substitution engine with a pre-registered validation harness. Arms are reported
   **separately per keeper**; no cross-arm composite in v1.

Owner scope choices: re-gate the attempt arms first **AND** build the gate-independent
physics arms in parallel; the frozen rule governs the xCross probe re-run while the new
xS probe registers its own rule; Approach A architecture (new `silly_kicks/gkdv/`
package + pre-planned promotions of the probe core and the causal port). GKDV v1 (physics
arms) ships **regardless** of every attempt-arm verdict row below.

## (2) Corrected findings — the retraction record

The original 4.18.0 read of the gate was that a y-mirrored convention **attenuated the
probe measurement**. That mechanism is **RETRACTED** after source verification and
cross-session review. The corrected findings (spec §1.2):

1. **Production mis-serving (primary — a correctness bug, not a research question).** The
   models were trained in a y-mirrored convention and are served on y-correct frames (GS
   was always native-clean → mis-served from day one; SC/Metrica are clean since
   4.29.0/4.30.0, ADR-031). Under the mirror, **xCross negates 3/16 features**
   (`ball_theta`/`gk_theta`/`gk_lateral_offset`) and **xS negates 12/27** (θ, GK_θ, and
   all 10 Def/OffAngle bearings) — read sign-inconsistently on every y-correct provider,
   for consumers opted into `pre_shot_gk_full_default_xfns` (+ atomic mirror). `load()`'s
   guard cannot catch it (pitch-dims fail-closed; `geometry_version` is warn-only and
   metadata-vs-constant — structurally blind to input chirality). The retrain is
   therefore a **correctness fix with a Hyrum/retrain trigger, independent of any gate
   outcome**.
2. **The 4.18.0 probe measurement itself was NOT mirror-attenuated.** Both extractors are
   frames-only (no action coordinate enters), the goal map is x-only, `GOAL_Y=34` sits on
   the mirror axis, and every domain gate / displacement panel is y-symmetric — a
   uniformly mirrored world is an **isometry** of the frames-only feature space, and a
   boosted tree learns the conjugate partitions. The measured 0.00107 is what corrected
   frames would have produced, up to float noise. The frozen cross gate is therefore
   **EXPECTED to hold** (~0.001, ~10× under the 0.01 floor); no cycle outcome is premised
   on it flipping.
3. **Residual IDSSE training-data defect (secondary, magnitude unknown).** IDSSE (~41% of
   the shipped training matches) was NOT a clean mirror — its frames were internally
   corrupted (ball at the wrong pitch END at shot times; an x-error no mirror absorbs;
   ~5.7 m post-flip residual; root cause never isolated). Ghost-GK's `gk_y` label was
   additionally mirrored for 17/81 of its training matches.
4. **The paired test that excluded GS was a mixed-chirality comparison.** The full
   candidate superimposed clean native GS on mirrored SC + corrupted IDSSE and lost all 5
   folds — sign conventions genuinely conflicted ACROSS providers in that candidate. This
   is the defensible reason to re-run the paired test on corrected frames; GS's 64 clean
   elite-keeper matches may now enter.

What is NOT stale: the causal-harness null (`gk_clears_placebo_band=False`) — model-free,
~83% y-clean GS spells — expected to stand for the cross arm after the retrain.

## (3) Registered xS probe rule (constants locked in code before any owner run)

The `ready` boolean is the AND of every prong; all constants live in
`silly_kicks/tracking/_model_eval.py`, locked in PR-1 (values copied here verbatim):

| Constant | Value | Role |
|---|---|---|
| `XS_PROBE_RATIO` | `2.0` | GK median |ΔP| ≥ 2.0× the control (prong 1) |
| `XS_PROBE_DOSE_M` | `2.0` | gated band: `|ghost − actual| ≥ 2 m`, trusted stratum only (prong 3) |
| `XS_PROBE_DOSE_LADDER` | `(2.0, 3.0, 4.0)` | reported ladder; only the 2 m band gates |
| `XS_PROBE_MIN_BAND_N` | `100` | minimum frames in the gated band |
| `XS_PROBE_MIN_STRATUM_N` | `50` | minimum frames in the trusted (unclamped, in-box) stratum |
| `XS_PROBE_PLACEBO_REPLICATES` | `20` | R paired-vector placebo replicates (prong 2) |
| `XS_PROBE_PLACEBO_BAND_PCT` | `95.0` | placebo band = p95 over replicate medians |
| `XS_PROBE_MAX_PLACEBO_ZERO_FRACTION` | `0.95` | non-degeneracy ceiling (fail-closed) |
| `XS_PROBE_DOSE_RESPONSE_ALPHA` | `0.05` | game-level sign-flip permutation p (prong 4) |
| `XS_PROBE_DOSE_RESPONSE_PERMUTATIONS` | `999` | permutation count |
| `XS_PROBE_MIN_GAME_N` | `10` | frames a game needs to contribute a per-game ρ |
| `XS_PROBE_MIN_GAMES` | `8` | measurable games needed for the dose test to be POWERED |

The frozen xCross constants are unchanged and stay where `tests/tracking/test_xcross_eval.py`
pins them: `TF19_PROBE_RATIO=2.0`, `TF19_PROBE_ABS_FLOOR=0.01`.

**Instrument-validation discipline (registered):** *no gate this cycle is an instrument
until it detects a planted signal under the actual clustering structure and the actual
control construction.* A discriminating-power meta-test (mixed-dependence planted model
PASSES; GK-blind model FAILS on the ratio/band prongs; fixture-validity preconditions
asserted in-test) ships with PR-1 — without it a null result is uninterpretable (the
4.46.0 lesson).

## (4) Verdict table (§3.5) — mechanical, implemented as `regate_verdict`

The two arms are gated INDEPENDENTLY, each against its own probe and its own
GK-confounder-entanglement run (§(5) — supportive context, not a causal deterrence
estimate). The table is a pure function in `silly_kicks/tracking/_model_eval.py`
(`regate_verdict(*, arm, probe_verdict, entanglement)`), test-parametrized over every row
— "mechanical" is a code property, not a prose claim.

| arm | its probe | GK-confounder entanglement | consequence for that arm |
|---|---|---|---|
| shot | pass (registered §(3) rule, all prongs) | clears | Δattempt shot arm joins the follow-up composition spec |
| shot | pass | inside band | joins, with "surface-responsive, confounder-entanglement unconfirmed" recorded |
| shot | band pass, flat dose-response | any | **band-pass-overridden-by-flat-dose-response** — stays gated; routed to GK feature engineering, NOT read as clean fail |
| shot | insufficient support (band n / stratum n / placebo degenerate) | any | **unmeasurable at this dose** — stays gated; routed to sampling/ghost follow-up, not to "no signal". An all-zero GK band with LIVE controls is NOT this row — it is the clean fail below |
| shot | any probe verdict | degenerate (no positivity / empty overlap) | probe verdict governs; entanglement recorded as "unmeasured" (a real harness outcome, not an error) |
| shot | instrument-invalid (meta-test red) | any | verdict VOID — fix the instrument first |
| shot | fail (prongs evaluated, not met) | any | stays gated (on GK feature engineering) — the clean fail |
| cross | pass (frozen rule) | clears | Δattempt cross arm joins |
| cross | pass | inside band | joins, with the same caveat |
| cross | fail | any | stays gated — the §(2) expectation; recorded as the clean verdict |

## (5) Causal shot arm — re-registration (SECOND time) + honest relabel

The prior registration ("goal within a 6 s window, strictly post-treatment") was
withdrawn once because the label machinery could not express it (a goal IS the successful
shot at the anchor in SPADL). A **first** re-registration used an own-result-only outcome;
review (plan-review P1) found that form **structurally degenerate for controls** — control
`Y ≡ 0` by construction (controls have no anchor action), which made the ATT
confounder-invariant and the entanglement gate dead. The **SECOND and final**
registration (implemented in `silly_kicks/causal/opportunities.py::shot_arm_config`):

- Domain = attacking-third predicate; treatment = shot types
  (`shot`, `shot_freekick`, `shot_penalty`); **outcome Y = a SUCCESSFUL outcome-type
  action in the ANCHOR-INCLUSIVE 6 s window** (`ts ≥ anchor`, `result_id == success`) —
  for treated spells that is the anchor shot's own goal or a rebound goal; for CONTROLS
  (anchored at entry) a within-window conversion. The builder gains
  `outcome_result_ids` and `outcome_window_anchor_inclusive` as **named** config axes,
  with known-truth tests (a saved anchor shot → `Y=0`, a scored one → `Y=1`, and a
  builder-level `Y.var() > 0 among Z==0` fixture — the instrument validated at the layer
  it defends).
- Confounders = the fresh xS-side list `SHOT_ARM_CONFOUNDERS =
  ("r","theta","speed","openGoal","DefDist_0","DefDist_1")` (xS has no `_CONFOUNDERS`
  constant to reuse — a PR-1-recorded decision) + the GK block `("GK_r","GK_theta")` with
  missing-indicator.
- **Honest relabel:** the GK block enters ONLY as confounders in the propensity for
  treatment; nothing counterfactually varies the GK, so the
  `gk_ablation_shift`-vs-placebo verdict is **"GK-confounder entanglement"** — supportive
  context for the Δattempt arm, NOT a causal deterrence estimate. §(4)'s column is named
  accordingly. Placebo idiom is match-level whole-cluster reassignment (a permutation
  over clusters — `_cluster_reassign`, not a cluster bootstrap), never row-i.i.d.
  permutation, plus a positive-control known-truth test in `tests/causal/`.

## (6) Dependency direction — gkdv → tracking, public seams only

`silly_kicks/gkdv/` (lands PR-3) imports `silly_kicks/tracking/` **public seams only,
never the reverse** and never a tracking underscore-private. In particular the ghost
positions are drawn from a positions-only serving seam in `_ghost_gk.py` (single-sourcing
the feature extraction, the 4.12.1 duplicate-`(frame, gk_team)` collapse, `predict_mean`,
and the 4.22.1 physical-pitch clamp); calling the extractor raw would silently drop the
dup-collapse and fork the clamp. The gkdv→tracking import surface is an explicit allowlist
**pinned by a test that lands in PR-3**. `tracking/` never imports `gkdv/`: the probe core
in `_model_eval.py` is a pure function over **pre-substituted inputs** (frames / ΔP /
provenance as data), so composition lives in `scripts/`.

## (7) ΔGK-threat-suppression — sign convention (worked example)

Both physics arms are defined in attacker-value units, `actual − ghost`, so **negative =
deterrent** uniformly. The ΔGK-threat-suppression arm (renamed from "Δcover-shadow"; the
four-surface `compute_blocking_score` removal legs cancel exactly, reducing to two
`threat_pc` surfaces — no removal, no `max(…,0)` clamp) reports the **NEGATION** of the
TODO's `Δ_blocking` reduction, because `blocking_score` is defense-positive while the arm
reports attacker value. Worked numeric example (registered in a red-first planted-polarity
fixture):

`threat_pc(actual)=0.30, threat_pc(ghost)=0.42 ⇒ Δ = −0.12 < 0 = deterrent; the Δ_blocking reduction equals +0.12 (defense-positive) — the arm reports the NEGATION`

A deterrent actual keeper suppresses attacker threat, so `threat_pc(actual) < threat_pc(ghost)`
and Δ is below zero = deterrent, matching the section convention. Per-arm expected
directions are `_validate.py` constants (PR-3), never TODO prose.

## (8) ADR-025 interplay (for the future engine)

The physics-arms engine and any future Δattempt scoring operate on **transient
scoring-time views** — ghost-substituted frame copies handed to a compute and discarded.
ADR-025's fence (never mutate canonical `start_x`/`end_x`; provenance as side-band
columns) is **not breached**: the counterfactual/ghost frames are never written back to
any mart. This mirrors 4.46.0/PR-S113's `apply_resolved_gk_geometry` resolution — a
scoring-time override on a copy is compatible with the never-persist-canonical contract;
policy stays at the edge, the engine stays provenance-free.

## (9) Chirality fingerprint decision

The guard is **behavioral, not a self-declaration**: `silly_kicks/tracking/_chirality.py`
computes `chirality_fingerprint(predict_on_frame)` = the model's own extraction + predict
on a fixed, deliberately y-ASYMMETRIC synthetic frame (`canonical_probe_frame`, goal at
x=105, every row off the y=34 mirror axis), returning `{version, frame_sha256, outputs}`.
A mislabeled artifact cannot satisfy it (a y-mirrored model produces different outputs).

- **PR-1**: emission only — all three `save()` paths (xS, xCross, ghost-GK) write the
  block into `metadata.json`. Coverage includes ghost-GK, whose `gk_y` labels were the
  actually-mirrored ones and on which every arm depends.
- **PR-2**: `load()` hardening — fail-closed on a mismatched fingerprint AND on a
  **missing** one (every pre-PR-2 artifact = exactly the mis-served ones), with an
  explicit **legacy override flag**. `tests/` assert `load()` RAISES on both.

**Recorded PR-2 hardening notes (Task-8 quality review):** (i) `chirality_fingerprint`
has no finiteness assert today; a NaN output would json-serialize as nonstandard `NaN`
and PR-2's `==` enforcement would always mismatch — PR-2 adds
`if not np.all(np.isfinite(outputs)): raise`. (ii) In xS/xCross `save()`, `model.json` is
written **before** the metadata dict is built, so a raising `_chirality_block` (only
reachable for an already-serve-broken model) leaves a partial artifact without
`SHA256SUMS`; this is fail-closed at load and is **recorded as acceptable**.

## (10) ruthless-efficiency boundary

Shared training/tuning logic lives upstream in `ruthless-efficiency` (floor `>=0.2.1`).
This cycle adds **zero** tuning logic to silly-kicks: the train scripts are touched only
for chirality-fingerprint emission and probe-sample provenance. Model-EVALUATION
machinery (probes/verdicts) is domain-specific and stays here (the `_xcross_eval.py`
docstring records the split); grouped validation statistics land in
`silly_kicks/_group_metrics.py` in PR-3 (validation ≠ training/tuning → not ruthless). If
the owner-run paired-test re-run ever needs HPO-harness changes they go upstream to
ruthless first, never forked here.

## (11) Zero-inflation prong — REVERSED to a reported diagnostic (was an ANDed gate)

Spec §3.1(5), as amended in the plan-review round, **reverses** the earlier
spec-review-round registration that ANDed a zero-inflation ceiling into `ready`. The
prong is now a **reported diagnostic, never a gate.** Reasoning: zeros have two causes and
only the CONTROLS disambiguate them — dead controls are already caught fail-closed as
`no_valid_placebo`, so **past that gate** an all-zero GK band can only mean the keeper
does not move the surface, which is a **CLEAN FAIL** (`gk_med = 0` → `band_pass` False),
the cycle's expected, publishable outcome — never "unmeasurable". A ceiling was also
provably outcome-inert for passes (zero-fraction > 0.5 forces median 0 → fail already).
`_model_eval.py` therefore carries **no** gated-band zero-fraction ceiling; the per-band
exact-zero fraction is reported alongside every verdict (it is what makes a fail
interpretable). `XS_PROBE_MAX_PLACEBO_ZERO_FRACTION=0.95` remains — but it guards the
**placebo** non-degeneracy (prong 2), a different quantity.

## (12) Recorded latent gap — ghost-GK extractor id compares (out of PR-1 scope)

`extract_ghost_gk_features` compares team ids with raw `==`/`!=` at
`silly_kicks/tracking/_ghost_gk.py:488-490` — the same ADR-019 class that Task 7 fixes in
the xS extractor. It is **out of PR-1 scope** and left as-is under Chesterton's Fence: the
canonical-frame chirality fingerprint is unaffected (single-provider dtypes match in
practice), and the ghost-GK retrain path is owner-run. Recorded here so a future cycle
that touches ghost-GK feature extraction closes it deliberately, not by accident.

## (13) Ratio prong — strengthened to `max(nearest_def, placebo_p95)`

`evaluate_xs_probe` computes
`band_pass = gk_med >= XS_PROBE_RATIO * max(nd_med, placebo_p95)` — a **deliberate
strengthening** over the spec's nearest-defender-only prong: the GK must clear twice the
LARGER of the nearest-defender control and the placebo band. The redundant explicit
`gk_med > placebo_p95` conjunct is **dropped** — it is implied by `ratio ≥ 2` together
with the `placebo_p95 > 0` non-degeneracy guard (a value ≥ 2× a positive band already
exceeds the band).

## (14) `no_valid_placebo → unmeasurable_at_dose` conflation is DELIBERATE

`regate_verdict` maps BOTH `unmeasurable_at_dose` (support shortfall) and
`no_valid_placebo` (degenerate control construction) to the single verdict string
`"unmeasurable_at_dose"`. This conflation is **intentional**: at the verdict layer both
outcomes have the same consequence — the arm **stays gated** and is NOT read as "no
signal". The **probe report** (`evaluate_xs_probe`) preserves the distinction (it returns
the two verdicts separately, with `dose_state` and the placebo diagnostics), because the
**follow-ups differ** (support → sampling/ghost work; degenerate placebo → control
construction). One string at the go/no-go layer, full detail in the report.

## (15) Paired-vector off-pitch policy — score, never clamp

In targets-mode the single per-frame paired vector `GK → target` is applied identically to
the GK and every control. **Registered policy:** a control displaced off-pitch is scored
**as-is, never clamped** — clamping would break the paired-vector equal-magnitude
guarantee (the whole point of the control construction). The off-pitch fraction over the
control rows (nearest_def + placebo_out) is reported as `off_pitch_control_fraction`
(report-only, feeds no verdict) alongside the verdict; the physical-pitch clamp lives
upstream on the ghost *position* (4.22.1), not on the control displacement vector.

## Owner runs (DGX, between PR-1 and PR-2)

Recorded runbook constraint (Task-6b): **an owner DGX re-run that admits GS to training
MUST keep held-out probe matches.** The trainer fails loud (`_gated_probe_matches`
`SystemExit`) if GS is admitted and no held-out probe matches remain; the runbook is to
pass `--probe-providers` a provider list whose matches are excluded from the training
folds. The gated probe/causal measurement frames are GS-only (native-clean, elite
keepers, 29.97 fps; the kloppy gateway hardcodes `visibility: None`, so SkillCorner frames
would measure the smoother) — drawn from GS matches **held out** of the admitted training
folds, so boosted-tree responsiveness on memorized partitions is not read as the served
regime.

## Consequences

- **Retrain trigger (PR-2, correctness).** New xS/xCross/ghost-GK weights change served
  values for consumers opted into `pre_shot_gk_full_default_xfns` (+ atomic mirror) and
  ghost-GK's `ghost_gk_*`; the lakehouse re-materializes. Not a forced default-VAEP
  retrain (these are opt-in xfn lists).
- **No new action-coupled aggregator this cycle.** The C4-tested count **stays 28**
  (definition: `len(add_* in tracking.__all__) − 1`, excluding the roster helper
  `add_gradientsports_player_ids`; raw 29 = counted 28). The `gkdv/` package (PR-3) adds a
  new C4 element but no `add_*`.
- **Public surface added PR-1.** `silly_kicks/causal/` (promoted from `_causal/`, ADR-015)
  with the full `OpportunityConfig` builder surface; `tracking/_model_eval.py` stays
  PRIVATE to `tracking/` (full public promotion of the eval home is deferred until an
  external consumer exists). `causal/` deliberately gets **no C4 container**: it is a
  maintainer-run research/validation harness (never imported by `silly_kicks/__init__`),
  not a runtime feature package — the cycle's one new C4 element is reserved for `gkdv/`
  (PR-3, same treatment as `xtgk`).

## Amendment (4.58.0, PR-S129) — xS-probe placebo v2 (relevance-matched defender null)

**Date:** 2026-07-23. **Status:** Accepted. **Source spec:**
`docs/superpowers/specs/2026-07-23-tf19-xs-placebo-v2-design.md` (twice cross-session-reviewed); plan
`docs/superpowers/plans/2026-07-23-tf19-xs-placebo-v2.md`.

**Problem.** TF-19 PR-3b Part A (4.55.4, PR-S126) ran the registered xS-arm GK-substitution probe and
returned `no_valid_placebo → unmeasurable_at_dose` — not for lack of a GK effect (dose-responsive, ~3.1×
the nearest-defender control) but because the probe's *secondary* null control, a **random outfielder**, was
degenerate (`placebo_p95 = 0.0`, zero-fraction 0.66). That gate fires before the clustered dose-response
ever runs, so the effect's *significance* was never tested.

**Decision.** Register a NEW variant `xs-dose-banded-v2` ALONGSIDE the frozen v1 (v1's rule / constants /
`evaluate_xs_probe` untouched; the record reports both verdicts). The ONLY change is the placebo pool:
random outfielder → the **model-relevant defenders** (ball-nearest defenders, minus the carrier-nearest
`nearest_def`, mirroring the xS extractor's 5-nearest-defender reference). Attackers are excluded from the
gated pool — the nearest attacker is the shooter, so gating on it would inflate the placebo through
attacking geometry — but survive as a reported, carrier-excluded, non-gating `attacker_diag` population.

**Honest consequence (settled in review).** The defender placebo is a *weaker* control than `nearest_def`
(farther from the ball), so it is **inert in the ratio prong** (`gk_med ≥ 2·max(nearest_def, placebo_p95)`
pins to `nearest_def`). Its role is (1) to clear the instrument-validity `no_valid_placebo` gate with a
principled, non-degenerate control (which the random pool could not) and (2) to be a reportable fair null —
NOT to move the bar. With the gate cleared and the ratio near-certain to pass, **v2's genuine open question
is the clustered dose-response permutation**, which v1 never reached.

**Blindness by discipline, made auditable.** The pool + constants land in a **lock commit**; the ~64-match
GS run happens only after it and records the lock-commit hash in `metrics.json`, so the git DAG shows
constants-locked-before-run. The run is a post-lock owner/DGX step (`--variant both`), reported under
`docs/research/tf19_pr3b_xs_v2/` — NOT part of the lock commit.

**Surfaces.** `substitution_deltas(..., placebo="random"|"model_relevant_def")` (default `"random"` =
byte-identical v1); private `_model_relevant_def_pool` / `_attacker_diag_pool`; `xs_substitution_probe_v2`
(reuses `evaluate_xs_probe` verbatim, relabels `rule`); `PROBE_WRAPPERS["xs_v2"]` (constants identical to
`xs` + `placebo_pool`); driver `scripts/validate_xs_probe.py` `--variant {v1,v2,both}` + `--lock-commit`.

**Scope.** Research instrument in `tracking/_model_eval.py` (stays PRIVATE) — in no default xfn list, no
VAEP consumer. **C4-free** (no new action-coupled aggregator; the 4.57.0 count of 31 is unchanged) and **no
retrain trigger** (xS/ghost weights untouched — v2 only re-reads them through a different placebo pool). v1
is byte-identical (frozen suite + a pre-refactor numeric pin). No new methodological reference (placebo
redesign within the existing ADR-037 probe; xS attribution arXiv:2512.00203 unchanged).

**Run result (4.60.0, PR-S131) — the deliverable, `docs/research/tf19_pr3b_xs_v2/`.** The ~64-match GS
probe ran on the DGX **from the lock commit `78ffc70`** (blindness verified: `metrics.json` records
`lock_commit == run_commit == 78ffc70`). **v1 = `no_valid_placebo`** (reproduces the 4.55.4 PR-3b baseline
exactly; degenerate random placebo, `placebo_p95 = 0.0`). **v2 = `pass` → re-gate `joins_with_caveat`.** The
methodology performed as designed: the defender placebo cleared the `no_valid_placebo` gate
(`placebo_p95 = 0.00057`, live) yet is **inert in the ratio** (weaker than `nearest_def = 0.00503`); the ratio
prong passed (`gk_med 0.01548`, `3.08×`); and the genuine decider — the **clustered dose-response permutation —
is significant, ρ = 0.436 / p = 0.001 across all 64 games** (dose-responsive 2 m→4 m: 0.0155→0.0222). The
`joins_with_caveat` re-gate reflects an `inside_band` entanglement input: the GK→shot-occurrence effect is real
and dose-responsive, joining the metric, but not cleanly isolable from the xS positional confounders.

> **CORRECTION (TF-19 sign-off package, F6).** That entanglement value was **not a measurement of this arm**.
> `regate_verdict` consults `entanglement` ONLY when the probe verdict is `pass`; every run before v2 expected a
> fail, so `scripts/validate_xs_probe.py` carried it as a hard-coded default — annotated in its own source as
> *"inert unless the probe surprises with `pass`"* — carried forward from the CROSS arm's §2 registration. The
> v2 probe surprised with `pass`, and the parameter documented as inert became the one that decided
> `joins` vs `joins_with_caveat`. `scripts/validate_xshot_causal.py` measures the quantity properly and had
> never been run; PR-2's own decision table recorded the shot row as *"(not run — PR-3-gated)"*. Nothing shipped
> was overclaimed — `joins_with_caveat` is the CONSERVATIVE branch, and a measured `clears` would have produced
> the stronger `joins` — but the attribution was false and is corrected here.
>
> **AMENDMENT (4.63.0) — the blindness discipline is now MECHANICALLY ENFORCED, not merely
> registered.** This ADR's citeability rests on artifacts recording `lock_commit == run_commit`, but
> nothing checked that the recorded commit described the code that actually ran: `git rev-parse HEAD`
> returns the same SHA whether or not the tree is modified. A corpus pass was launched from a tree
> with three modified drivers while HEAD read clean — the artifacts would have carried a
> verifiable-looking FALSE provenance, which is strictly worse than none, and it would have
> propagated (the arm-values table feeds the §6.1 ICC number, so a clean SHA on the power metrics
> would have laundered a dirty input). `scripts/_provenance.py` makes an artifact-writing run
> REFUSE a dirty tree, naming the dirty files and the SHA that would have been recorded; absent git
> counts as dirty, never clean; `--allow-dirty` permits a dev run but the artifact still records
> `dirty: true`. The escape hatch must never launder the fact. This converts the xS arm from PR-3b's `unmeasurable_at_dose` dead-end into a real,
citeable `pass` — a materially more positive read on the attempt axis than the prior "leans H2/abandon"; it
directly informs the §6.4 Part B go/no-go (still owner-gated). **Deliverable ships the research artifact only —
no `silly_kicks/` code change; the wheel is byte-identical to 4.59.0.**

## Amendment (4.65.0) — an inestimable replicate is COUNTED, and the corpus pass is its own driver

**Date:** 2026-07-27. **Status:** Accepted. Both decisions come from ONE measured failure on the
first full §6.1 corpus run.

**What happened.** `run_signoff_power.py` built its Layer 2 spells INLINE, walked all 64 matches
over **8.7h**, then raised in the cheap analysis step immediately after — losing every spell,
because nothing had been written to disk. The raise came from `fit_propensity`
(`This solver needs samples of at least 2 classes`): a cluster resample had drawn no treated unit.

**Measured cause, not a silent null.** A 6-match probe returned `gk_finite_frac = 1.0` (the GK
covariate block is fully populated — no ADR-015-style NaN defect) with `Z` prevalence **0.0039**
(15 treated of 3,811 spells; keeper depth at spell entry median 3.73 m, p99 14.04 m). At that
prevalence a 500-row draw has a **14%** chance of containing no treated unit, and the run made
~1,200 such draws. The treatment is RARE, not absent.

**Decision 1 — degenerate replicates are scored as non-detections and COUNTED.**
`att_power_curve` returns a new `n_degenerate_by_size`. A single-treatment-class resample is a
positivity failure at that size: the ATT is not estimable, so it has detected nothing. It stays in
the DENOMINATOR — excluding it would condition on estimability and inflate the curve exactly where
the design is weakest — and it is reported, because power 0.2 with most replicates inestimable is a
different claim from power 0.2 with none. **A size whose degenerate count approaches `n_replicates`
reports an inestimable design at that n, and must not be read as a weak effect.** Byte-identical on
any input that was already estimable; the guard only fires where the old code raised.

**Decision 2 — an expensive corpus pass is its own shardable, resumable, partitionable driver.**
`scripts/build_layer2_spells.py` (per-match shards on completion, `--match-ids-json`,
`--list-matches`) produces the table; `run_signoff_power.py --spells` consumes it. The arm-values
pass already had this shape and survived its 64-match run twice; the power pass did not, and that
asymmetry is what made an 8.7h loss possible. Shared reconciliation lives in `scripts/_partition.py`
(it has been wrong once already) and now also checks **commit consistency** across workers. Two
corollaries of N writers on one output dir: combined tables are written via a private temp +
`os.replace`, and the consumer REFUSES a dirty upstream, a **missing** upstream manifest
(unprovenanced == dirty), or an upstream blended from different commits — extending the 4.63.0
provenance control from "this run's tree" to "every artifact this number derives from".

**Consequence for §6.1, stated BEFORE the run.** ~36k spells over 64 matches implies roughly **140
treated units** corpus-wide. ATT power at the registered relative anchors is likely to come back
**below 0.8**, in which case §6.1's own rule applies: adjust floors/sampling FIRST and do **not**
register the gate. The registered 16.5 m threshold is **not** retuned to raise prevalence — it is
Law-defined and data-independent precisely so the decider stays untuned; changing it is a
re-registration decision, not an implementation one.

## References

Spec: `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md`. Le et al.
2017 (ghosting); Cao et al. 2025, arXiv:2505.11841 (causal crossing); Kim et al. 2025,
arXiv:2512.10355 (DEFCON-GNN comparator); Pipping et al. 2026, arXiv:2512.00203 (xS);
Groom et al. 2026, arXiv:2601.00748 (role-conditioned ghosting, TF-46). Related ADRs: 008,
011, 015, 016, 019, 022, 023, 024, 025, 028, 029, 031, 036. Gate record:
`_xcross_weights/default/metrics.json`; causal record: `docs/research/xcross_causal/`.
