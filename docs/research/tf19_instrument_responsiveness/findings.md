# TF-19 physics-arm instrument-validity + responsiveness — F1b / native-DAS re-run (findings)

**Reported-not-gated.** Re-materialized for the combined-cycle completion (silly-kicks 4.128.0 /
ADR-106 float32 frames + ADR-107/108 native DAS), superseding the 4.111.0 / PR-S182 / ADR-089 both-axes
run. The gkdv physics arms were re-computed against the re-fit ghost-GK variants on float32-stored
frames with the native DAS engine, and this sign-off re-run confirms the instrument verdict is
unchanged and the named-keeper face-validity still holds 2 of 2.

- **run_commit:** `b62c1f24a7a9e3361ce416b402ed27da4a59b9e6` (M; tree clean on the artifact).
- **Corpus:** 179 matches — Gradient Sports WC2022 (64) + SkillCorner (108) + IDSSE/Sportec (7); the
  delta_das arm over **241,715 scored domain frames** (116 resolved keepers; 744 keeper teams, 0
  unresolved).
- **Registered thresholds (locked):** SATURATING_MULTIPLE=5.0, PHYSICS_ARM_PROBE_RATIO=2.0, R=3,
  MIN_DOMAIN_FRAMES=200, REGIME_I_LADDER_M=2.0, REALISTIC_MIN_DISP_M=2.0.
- **Reduce note:** `reduce_mode = driver-reduce-only` — the pooled verdicts come from the driver's own
  `--reduce-only` combine over exactly the 179-match corpus shards (`partition = corpus179`), which
  counts each match once and refuses an unaccounted key (the amendment-2 overlap guard), so no manual
  reduce is needed.

## Headline — unchanged from A+2

**ΔDAS remains a WEAK instrument for keeper deterrence**, now confirmed on the larger multi-provider
corpus under the both-axes weights.

| Layer | Verdict |
|---|---|
| Layer 0 — instrument validity | `instrument_void` |
| Layer 1 — responsiveness | `not_responsive` |
| Threat arm (`delta_threat_suppression`) | `arm_unscoreable` (no loadable ExpectedThreat; the package ships no xG model) |

Pooled |ΔDAS| medians (attacker-value units; **negative = deterrent** for the signed metric):

| quantity | value | meaning |
|---|---|---|
| realistic (shipped ghost dose) | 0.4731 | the actual keeper-vs-ghost displacement |
| saturating (keeper on goal line) | 0.1254 | maximal keeper displacement (dose, not effect) |
| ladder 2 m (keeper) | 0.0577 | the imposed responsiveness dose |
| nearest-defender control | 0.3420 | one outfielder moved by the keeper's vector |
| single-outfielder placebo p95 | 2.8779 | R single-player placebos |

**Interpretation.** Accessible space (DAS) is dominated by the outfield frontier, so relocating the one
deep keeper — even onto the goal line (saturating 0.13) — perturbs it *less* than moving a random
outfielder by the same vector (placebo p95 2.88). The instrument fails both the 5×-realistic and the
placebo-band legs (Layer 0 `instrument_void`) and the keeper move is not specifically responsive
(Layer 1 `not_responsive`). The float32 + native-DAS re-run did **not** change this conclusion — it is the probe
working as designed, detecting that ΔDAS is not the right arm for keeper deterrence, not a null
"no-effect" claim. The threat arm would be the relevant signal but is unscoreable here.

## Named-keeper face validity — improved to 2 of 2 (PRE-REGISTERED prior, locked 2026-08-29)

Prior: {"Alisson": "negative", "Neuer": "negative"} (deterrent = negative ΔDAS). Caveated-and-excluded:
{"Ter Stegen": "0_min", "Onana": "descriptive_only"}.

| keeper | expected | observed (mean ΔDAS) | meets prior |
|---|---|---|---|
| Alisson | negative | negative (−0.501) | YES |
| Neuer | negative | negative (−0.017) | YES |

**2 of 2 confirmed** (held from the 4.111.0 both-axes run). Alisson remains a clear deterrent (mean
−0.501); Neuer stays marginally negative (mean −0.017, was −0.034 under the both-axes weights), still
matching the prior. Neuer's mean is small and its median is marginally positive (+0.061), so this is
face-validity corroboration, not a strong measurement — consistent with the weak-instrument headline.

Census: **116 resolved keepers, 74 gate-eligible** (min_nonzero=20, min_games=2), **38 of the eligible
match the expected-negative sign** (≈51 %, near chance — the weak-instrument signature); 0 unresolved
keeper frames; Layer-4 behavioural anchoring: `uninterpretable`.

## Per-keeper signed ΔDAS

The full 116-keeper sign table (Regime-O realistic dose, signed, |displacement| ≥ 2.0 m subset) is the
machine-readable deliverable `named_keeper_signs.parquet`, keyed by `player_id` with columns `mean`,
`median`, `n`, `n_nonzero`, `n_games`, `gate_eligible`, `expected_direction`, `observed_sign`,
`sign_matches_expected` (no keeper names in the artifact; the Alisson = 32 / Neuer = 4602 mapping is the
owner-injected map, applied externally). The `observed_sign` column is **mean-based**; because ΔDAS is
zero-dominated and heavy-tailed, the mean and median can disagree in sign for a keeper (Neuer: mean
−0.017, median +0.061), so a single named-keeper eye-test is face-validity only, not a measurement (the
parquet carries `median` for inspection). The most-deterrent eligible keepers reach mean ΔDAS ≈ −3.2;
the least, ≈ +3.6.

## Caveats
- **Reported-not-gated:** these numbers flip no gate and trigger no retrain.
- **ΔDAS only:** the threat arm is `arm_unscoreable` here; ΔDAS is NaN on velocity-less providers (SB360) by construction (ADR-063).
- The sign table is over the ≥ 2.0 m realistic-displacement subset (spec §4.1) — sharpens, does not invert, the deterrent sign; it is not the no-floor `build_gkdv_arm_values` population.
- **Gradient Sports keeper clamp (ADR-083):** GS tracking pins the keeper at 27.5 m from goal, so the GS actual-keeper leg is bounded; the SkillCorner/IDSSE cohorts are unaffected.
