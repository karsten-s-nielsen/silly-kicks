# Cover-shadow σ/λ discrimination re-tuning + expected-receiver model — design

| Field | Value |
|---|---|
| **Date** | 2026-08-20 |
| **Status** | Ready to execute — spec reviews 1–2 + plan reviews 1–2 + final review incorporated; D1 RESOLVED (public SB360 + conditional GS-owner variants); ADR-066 |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |
| **Supersedes/uses** | ADR-064 (RQ1 validation); ADR-011 (bundle); ADR-060 (prefer-incumbent); ADR-009 (**amended** §3.4); ADR-052/037 (corpus driver + provenance); `PreprocessConfig.for_provider` (precedent §3.4) |

## 0. What review 1 changed (rev 2)

The receiver model trains on **completed** passes but is used on **failed** passes, and completed↔failed
differ *because failure is the selection*. So (H1) completed-pass accuracy cannot bound failed-pass error;
(H2) a lane-pressure feature makes the resulting error **directional** — the model prefers open targets, so
failed-pass `p_blocked` is biased *low*, feeding the σ/λ objective a low target; (H3) σ/λ is a global
constant, and a value fit on one provider/frame-rate (GS 25 Hz) mis-profiles every other provider — the
exact error this cycle criticizes. Rev 2 answers all three (trajectory-weak-labelled failed-pass
validation; a lane-pressure ablation the apply-gate honours; per-provider `for_provider` apply), reframes
the σ/λ work as **discrimination re-tuning** not magnitude "recalibration" (M1 — magnitude calibration
needs counterfactual block ground truth we don't have), and treats the **honest-null as the likely,
acceptable outcome** (L2 — the incumbent σ/λ scored 0.764 in ADR-064; it is not demonstrably broken).

## 0b. What review 2 changed (rev 3)

The trajectory-validation subset (§3.1b) is **not random** — it is the *easy tail* of failed passes: a clear
pre-interception trajectory is exactly where the intended receiver is easiest to infer AND where the
geometric proxy is already near-ceiling. So (R1) its accuracy is an **upper bound**, not "the trust number,"
and a ship-gate "beat the proxy on this subset" is evaluated where beating the proxy is hardest and least
informative — a null there means *"unvalidatable where the model would matter; ties the proxy on the easy
cases,"* NOT "the model has no value." Rev 3 scopes that meaning (§3.4/§3.5), restricts the σ/λ objective's
intercepted leg to the validated subset so use-population = validation-population (R2), states the residual
bias the gates leave unmeasured (R3 — the ablation covers only the lane-pressure channel; the
position/velocity/bearing covariate shift on hard failures stays unmeasured), names how the weak-label is
weak (R4 — leading/through-balls run *onto* the ball, not at the trajectory endpoint), **pre-registers every
gate threshold as a number** (R5, §3.6), and adds a GS next-action-tagging reliability check (R6, §5).

## 1. Context & goal

ADR-064 (4.87.0) validated the cover-shadow model on GS WC2022 and left two tracked follow-ups, done here
as **one cycle**: adapt `CoverShadowParams.sigma=0.20 / lambda_ctrl=4.3` (`_cover_shadows.py:103`) to our
measured-velocity profile, and remove the RQ1 recall leak (a failed pass's target is the outcome-selected
release-frame `end_xy`, `_rq_corpus.py:47`) via a model that infers the *intended* receiver.

**Why one coherent cycle:** the receiver model de-leaks the σ/λ *objective*, not only the recall —
sequenced receiver-first, the same de-leaked target feeds both. **This is exploratory:** the goal is to
learn whether a measured-velocity σ/λ discriminates pass failure better than Cascioli's estimated-velocity
default, with keeping the incumbent as a fully acceptable result.

## 2. Internal sequencing (single branch, minimal commits)

1. Build the **expected-receiver model** (§3.1) + its **trajectory-weak-labelled failed-pass validation
   set** (§3.1b).
2. **De-leak** the RQ harness, failure-mode-conditional (§3.2).
3. Re-tune **σ/λ** against the de-leaked discrimination objective + the lane-pressure ablation (§3.3).
4. **Conditional, per-provider apply** gated on receiver-validity AND bias AND noise (§3.4).
5. Re-run **RQ1 recall** with the failed-pass validity check + robustness band (§3.5).

Commits provenance-driven (~2–3, like the 4.81.0 ghost-refit); **non-squash** merge so weight
`metrics.json` `run_commit` stamps resolve. Split pinned at commit-prep, minimised.

## 3. Components

### 3.1 Expected-receiver model (`_receiver.py` + `_receiver_weights/`)

Per-candidate binary scorer (mirrors xShot/xCross): for each passing-team teammate present in the release
frame, score `P(intended)` from **pre-pass state + release kinematics ONLY**; argmax = intended receiver,
full ranking retained (TF-51 Track B wants the distribution). **Never the end/loss location** — the
outcome-selected quantity we are removing (L1 gate, §5). Release direction/velocity IS allowed (the
*attempt*, observed identically for completed and failed passes).

**Trained** on completed passes (`resolve_next_touch_receiver`, `spadl/utils.py:1299`, = observed receiver
ground truth), match-stratified `GroupKFold`. **Corpus = SB360 open-data (public, D1)** — **positions ONLY**
(no leakage-free release direction on velocity-less freeze frames; the pass-event angle is origin→end,
banned). The public bundled default, applied on GS (train-SB360 / serve-GS transfer, validated on GS via
§3.1b); the GS-owner variant adds the ball-release-velocity direction + closing speed (§D1, Task 6b). Bundle = ADR-011 (numpy/JSON + `SHA256SUMS` + chirality +
feature contract; pickle-free, sklearn-free inference). SB360 loader reuses `scripts/_sb_raw` +
`load_statsbomb_matches` (ADR-062).

**Geometric proxy** (nearest teammate to the release ray within a cone): a feature, the **must-beat
baseline**, and the **fallback** — a model that fails to beat it on the failed-pass validation set (§3.1b,
*not* the completed-pass set) is not bundled; we ship the proxy and record the null (xShot 4.9.0 precedent).

**H2 — lane-pressure is a declared bias risk.** "Lane pressure to the candidate" is informative but
circular: the model uses it to pick the target, then we measure `p_blocked` (lane pressure) at that target,
so a completed-pass-trained model prefers open targets and biases failed-pass `p_blocked` low. It is kept
as a feature but **ablated** (§3.3), and the apply-gate (§3.4) refuses a σ/λ shift the ablation attributes
to it.

**Public surface:** `resolve_intended_receiver(actions, frames, *, model=None) -> Series[player_id]`
(`model=None` → geometric proxy) + `intended_receiver_positions(...)`.

### 3.1b Trajectory-weak-labelled failed-pass validation set (H1 — the load-bearing addition)

The leakage ban is a **training-feature** constraint; **validation** may use the observed post-release
trajectory, which the model never sees. For **intercepted** failures whose ball travelled a usable distance
toward a teammate before being cut out, the early trajectory weak-labels the intended lane/receiver. This
yields a **direct, if partial, failed-pass accuracy estimate** — the number H1 says is otherwise missing.
**R1 — this subset is the easy tail, so its accuracy is an UPPER BOUND, not an estimate.** A clear
pre-interception trajectory is where the intended receiver is easiest to infer; the uncovered failures
(foot-blocked, immediately-out, contested-target-with-no-clear-lane) are the hard cases the σ/λ objective
still applies the model to, and the model's covered-subset accuracy overstates its accuracy there. The
covered subset is also where the geometric proxy is near-ceiling, so it is the **region of smallest
incremental model value** — consequential for the ship-gate (§3.4/§3.5). **R4 — the weak-label is itself
noisy:** "nearest teammate to the trajectory" mislabels leading passes / through-balls (the receiver runs
*onto* the ball, not at the trajectory endpoint), concentrated in exactly the progressive passes most worth
getting right, so accuracy-vs-trajectory-label conflates model error with label error. Mitigation: label
against a **forward-projected** meeting point (ball path × teammate run), not the endpoint; report the
weak-label's own failure mode so the number is not over-read. Coverage fraction is reported; the
completed-pass top-k is demoted to a **training-fit diagnostic**, never a trust bound.

**M2/M3 — with the D1 public corpus this set now validates a DOUBLE transfer on the easy tail.** The
deployed inference is SB360-completed → GS-failed (cross-provider + cross-frame-rate *and* completed→failed),
so a conjunct-1 pass licenses **neither** transfer on the hard subset (R1's scoped-null, extended to the
cross-provider leg). Separately, SB360's visible area truncates the **negative** candidate set (the model
ranks among *visible* teammates in training but *all* teammates at GS serve — a ranking-distribution shift
orthogonal to the positive being visible); the SB360-train vs GS-serve candidate-count distributions are
recorded (Task 6/8 `metrics.json`) and carried here as a caveat.

### 3.2 De-leaked, failure-mode-conditional RQ harness (`_rq_corpus.py`)

The failure mode is derivable from the **next SPADL action** (opponent interception/tackle ⇒ intercepted;
throw-in/goal-kick ⇒ out). (M2) **Intercepted** failures → receiver-model intended target (end is
defender-selected). **Overhit/out** failures → a trajectory-informed target (end is empty space, not
defender-selected, and carries overshot intent the receiver model is blind to) — not the blanket ban rev 1
wrote. `target_source` vocabulary gains `{intended_receiver, trajectory, geometric_proxy}`; shard schema →
`rq-scores-3` (ADR-052 completeness assertion on the new columns).

### 3.3 σ/λ discrimination re-tuning objective (`calibration/_cover_shadow_objective.py`)

New TF-24 objective sweeping `(sigma, lambda_ctrl)`. **This re-tunes σ/λ SHAPE for discrimination; it does
NOT calibrate `p_blocked` MAGNITUDE** (M1 — that needs counterfactual block ground truth we lack; stated in
the manifest).

- **Primary = de-leaked discrimination:** maximise failed-vs-completed **margin-AUC** with failed-pass
  `p_blocked` at the (failure-mode-conditional) intended target — the intercepted leg on the
  trajectory-validated subset (R2/§3.4), with a full-intercepted-population re-tune as a reported sensitivity.
- **Constraint:** the completed-pass **FP rate** must not rise above the incumbent's (no buying AUC by
  over-blocking).
- **H2 ablation (first-class output):** the σ/λ argmax computed **with vs without** lane-pressure in the
  receiver features; the gap is the share of the σ/λ shift attributable to the open-target bias.
- **Cross-check = `AugmentedVaepBrier`** (proper scoring rule, magnitude-sensitive via the VAEP model) —
  does the re-tuned σ/λ also help downstream? Reported, not primary.
- **R3 — the ablation is only ONE channel of the bias.** Removing lane-pressure does not make a
  completed-trained model unbiased on failed passes: position/velocity/bearing still carry the covariate
  shift (completed passes go to open, reachable teammates). So "minority attributable to lane-pressure" ≠
  "unbiased target." The two gates together cover the **lane-pressure channel** (ablation) and the **easy-
  subset transfer** (§3.1b); the **hard-subset, non-lane-pressure bias stays unmeasured** — declared in the
  manifest as the honest residual ("bounded two channels", not "ruled out the bias").
- **Out-of-play failures are NOT in the objective** (Low-1): they are empty-space, low-`p_blocked` by
  construction — not a blocking phenomenon — so σ/λ's discrimination class is completed vs *blocked*
  (intercepted-validated) failures. The recall (§3.5) still spans them; the objective does not.
- Still **attempted-pass-conditional** (a blocked, never-attempted pass is unobservable) AND
  **model-conditional** — both in the manifest.

### 3.4 Conditional, PER-PROVIDER apply (H3; ADR-009 amendment + ADR-060 + `for_provider`)

The harness **recommends** σ/λ + manifest (ADR-009: harness never mutates defaults). **R2 — the σ/λ
objective's intercepted-failure leg is restricted to the trajectory-validated subset (§3.1b), so
use-population = validation-population**; a full-intercepted-population re-tune is reported alongside as a
sensitivity (if they diverge, the extrapolation is flagged, not applied). A deliberate apply then moves σ/λ
**only if ALL hold** (thresholds pre-registered, §3.6), else keeps the incumbent (honest null):
1. **Receiver-validity (H1):** coverage ≥ `MIN_COVERAGE` AND the model beats the proxy on the validated
   subset by ≥ `MIN_RECEIVER_MARGIN`;
2. **Bias (H2):** the lane-pressure ablation attributes < `MAX_BIAS_SHARE` of the σ/λ shift to the
   open-target channel;
3. **Noise + effect size (ADR-060):** `exceeds_noise_floor` clears both floors.

**R1 — what a FAILED gate means is scoped, not silent.** Because the validated subset is the easy tail
where the proxy is near-ceiling (§0b), a conjunct-1 miss is reported as *"unvalidatable where the model
would matter; ties the proxy on the easy cases"* — NOT "the model has no value." The apply outcome is
recorded as one of `{applied, null:unvalidatable, null:biased, null:within-noise}`, so a null carries its
reason.

**Per-provider, never global (H3):** `CoverShadowParams` gains `for_provider` (additive, mirrors
`PreprocessConfig.for_provider`); a cleared GS re-tune sets **GS's** σ/λ, other providers keep the incumbent
until each is re-tuned on its own velocity profile. **Cost stays small** (verified: no bundled silly-kicks
model uses cover-shadow features; `cover_shadow_xfns` is opt-in, not in `tracking_default_xfns`) — apply =
the `for_provider` entry + a **consumer** VAEP-retrain trigger *for GS-cover-shadow consumers only*; no
weight re-bundle.

### 3.5 Recall re-run + honesty (`validate_cover_shadow_rq1.py`)

1. **recall/precision/BA** with the failure-mode-conditional target;
2. **failed-pass validity (H1/R1)** — trajectory-set accuracy + coverage, reported as an **upper bound on
   the easy tail**, not "the trust number"; the uncovered fraction is stated as the region the number does
   not reach;
3. **robustness band (M3)** — recall under trained-model vs geometric proxy, framed as *do two
   same-failure-mode extrapolations agree* (both can be wrong the same way, per H2/R3), NOT as evidence the
   inference is correct.

### 3.6 Pre-registered gate thresholds (R5)

Because the cycle's credibility rests on the honest-null being reached *honestly* rather than via a movable
bar, every §3.4 threshold is a **named constant fixed before the run**; changing one after seeing results
invalidates the gate, and any pre-run change is recorded with its reason. Proposed defaults (owner/plan may
re-pin *before* the run):

| Constant | Default | Meaning |
|---|---|---|
| `MIN_COVERAGE` | 0.30 | trajectory-validated subset ≥ 30% of intercepted failures, else conjunct-1 is unmeasurable → null |
| `MIN_RECEIVER_MARGIN` | 0.05 | model top-1 accuracy beats the proxy by ≥ 5 pts on the validated subset (hard: proxy near-ceiling there, R1) |
| `MAX_BIAS_SHARE` | 0.50 | < half the σ/λ shift attributable to the lane-pressure channel (the "minority" of R2/H2) |
| effect size / noise | ADR-060 `MIN_EFFECT_SIZE` + `exceeds_noise_floor` | existing prefer-incumbent floors |

## 4. Corpus, provenance, redistribution

GS WC2022 via pining (owner-tier), run **locally**; ADR-052 `for_each` shards + ADR-037 provenance;
aggregate-only, ship-mask-labeled artifacts; raw positions gitignored; weights carry `metrics.json`
provenance (→ non-squash). Receiver weights follow corpus-visibility (owner-tier → `stores_training_data:false`).
**Foundation (4.89.0 / ADR-065):** the receiver labels (`resolve_next_touch_receiver`) and the failure-mode
split (next SPADL action) rest on the now-guaranteed chronological `action_id`; consumer code reading
persisted marts sorts via `_sort_actions_chronological_or_action_id` (a mart may carry a non-chronological
`action_id`). GS's real WC2022 feed is byte-identical under the fix (not a retrain trigger) but the GS
converter now requires a `start_time` input column — the pining loader supplies it.

## 5. Testing & validation

- **Receiver model:** beats the geometric proxy on the **failed-pass** validation set (ship gate, §3.1b);
  **L1 leakage guard = output-invariance** — perturb the end location, assert the receiver output is
  byte-identical (not the disjunctive "raises or ignores"); ADR-011 chirality + feature-contract load;
  in-domain, no-constant-feature fixture (ADR-032).
- **σ/λ objective:** pure, deterministic, cache-equivalent (1e-9); discrimination moves when σ/λ move
  (non-vacuity); FP-constraint rejects an over-blocking point; the ablation output is present and non-trivial.
- **Apply gate:** all three conjuncts exercised from BOTH sides (a clears-all case and, per conjunct, a
  case that fails only that conjunct and correctly yields the null); the null path leaves `for_provider`
  and the global default byte-identical.
- **Per-provider:** a GS re-tune does not change any other provider's resolved σ/λ.
- **Harness:** `target_source` vocabulary incl. `trajectory`; failure-mode routing (interception vs out);
  `rq-scores-3` completeness assertion; the robustness band differs from the leaked baseline on real away data.
- **R6 — GS failure-mode tagging reliability (pre-run):** a driver step that verifies GS's next-action
  tagging cleanly separates interception from out (a residual/ambiguous rate below a pinned bar), because a
  mislabel routes a pass to the wrong target definition (model vs trajectory) and silently mixes the
  objective's two legs. If GS tagging is not clean enough, the failure-mode split is reported as unreliable
  and the intercepted/out legs are not mixed.
- **R5 — pre-registration wiring:** the §3.6 constants live in one module, are referenced (not literals) by
  the apply gate, and a test pins that the gate reads them — so the bar cannot be moved silently.
- Purity / mirror / id-dtype / liveness / glossary registries updated for new public surface.

## 6. Consequences

- **VAEP retrain trigger (consumer-side, GS-scoped, conditional)** — only if the GS apply clears all gates,
  only for GS cover-shadow consumers. No silly-kicks bundled-model change. Null path → no trigger.
- **New public surface:** `resolve_intended_receiver` / `intended_receiver_positions` + receiver bundle;
  a calibration objective; `CoverShadowParams.for_provider`.
- **ADRs:** new **ADR-066**; **amends ADR-009** (gated in-cycle per-provider apply); uses ADR-011/060/052/037.
- **C4:** confirm whether the receiver module registers a new container (mirror xShot/xCross).
- **Deferred:** the **public** bundled receiver variant + **TF-51 Track B** consumption; per-provider σ/λ
  for non-GS providers (each its own re-tune).

## 7. Open decisions for review 2
- **D1 — RESOLVED (owner): public variant in-cycle.** The bundled deliverable is a **public** receiver
  model trained on **SB360 open-data** (freeze frames carry the pre-pass positions with real team ids per
  ADR-062; ~510 matches; features = **positions ONLY** — no leakage-free release direction exists on
  velocity-less freeze frames (the pass-event angle is origin→end, banned), which also keeps the model
  provider-portable; visible-area partial-observation is a caveat, but a
  completed pass's receiver is near the action and typically visible). It is the redistributable primitive
  TF-51 Track B reuses, and it is what the GS harness applies (validated on GS via the trajectory set,
  §3.1b — a train-SB360 / serve-GS transfer, acceptable because intended-receiver geometry is far less
  provider-specific than σ/λ noise profiles). **The velocity question and the deployment question are measured on
  DIFFERENT populations (M-A resolution):** a **velocity ablation on held-out COMPLETED passes**
  (GS-positions-only vs GS-positions+velocity — full ground truth, NOT the failed easy tail where positions
  already resolve the receiver and velocity's marginal value is ~0 by construction) answers whether velocity
  carries intended-receiver signal; a separate **deployment gate on the failed validated subset**
  (SB360-public vs GS-owner) decides which variant to serve on GS, with the R1 caveat that a non-decisive
  gate is "unmeasurable on the easy tail," not "velocity adds nothing." The GS-owner variant is bundled
  in-cycle iff the deployment gate is decisive, else a recorded follow-up (variant-keyed like
  `GkCompletion.variant_key_for_provider`).
- **D2 — RESOLVED (review 1 / M1):** discrimination re-tuning primary + FP constraint + VAEP cross-check;
  renamed from "recalibration".
- **H3 — RESOLVED (owner):** per-provider via `for_provider`.
- **Commit split** — provenance-driven, finalized at commit-prep.
