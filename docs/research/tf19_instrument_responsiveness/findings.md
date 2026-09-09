# TF-19 physics-arm instrument-validity + responsiveness — both-axes re-run (findings)

**Reported-not-gated.** Re-run for the TF-60 ghost-GK **both-axes convention** re-fit (silly-kicks
4.111.0 / PR-S182 / ADR-089), superseding the 4.104.0 / ADR-082 A+2 run (which was Gradient Sports
WC2022 only, on the old x-only weights). The gkdv physics arms were re-materialized against the five
re-fit ghost-GK variants and this sign-off re-run confirms the instrument verdict is unchanged and the
named-keeper face-validity improved.

- **run_commit:** `a867f9b326b0ef71ed000185292b698e93183d32` (tree clean; the re-fit-weights commit).
- **Corpus:** 179 matches — Gradient Sports WC2022 (64) + SkillCorner (108) + IDSSE/Sportec (7); the
  delta_das arm over **241,716 scored domain frames** (116 resolved keepers).
- **Registered thresholds (locked):** SATURATING_MULTIPLE=5.0, PHYSICS_ARM_PROBE_RATIO=2.0, R=3,
  MIN_DOMAIN_FRAMES=200, REGIME_I_LADDER_M=2.0, REALISTIC_MIN_DISP_M=2.0.
- **Reduce note:** the pooled verdicts were produced by a manual reduce (`harness/reduce_tf19_manual.py`)
  applying the driver's own reduce functions over exactly the 179-match corpus shards — the driver's
  unpartitioned `main()` reduce would re-stream the full provider manifest (>179). Scientific content is
  identical (same functions); see `metrics.json` `reduce_mode`.

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
| realistic (shipped ghost dose) | 0.4646 | the actual keeper-vs-ghost displacement |
| saturating (keeper on goal line) | 0.1216 | maximal keeper displacement (dose, not effect) |
| ladder 2 m (keeper) | 0.0563 | the imposed responsiveness dose |
| nearest-defender control | 0.3385 | one outfielder moved by the keeper's vector |
| single-outfielder placebo p95 | 2.8346 | R single-player placebos |

**Interpretation.** Accessible space (DAS) is dominated by the outfield frontier, so relocating the one
deep keeper — even onto the goal line (saturating 0.12) — perturbs it *less* than moving a random
outfielder by the same vector (placebo p95 2.83). The instrument fails both the 5×-realistic and the
placebo-band legs (Layer 0 `instrument_void`) and the keeper move is not specifically responsive
(Layer 1 `not_responsive`). The both-axes re-fit did **not** change this conclusion — it is the probe
working as designed, detecting that ΔDAS is not the right arm for keeper deterrence, not a null
"no-effect" claim. The threat arm would be the relevant signal but is unscoreable here.

## Named-keeper face validity — improved to 2 of 2 (PRE-REGISTERED prior, locked 2026-08-29)

Prior: {"Alisson": "negative", "Neuer": "negative"} (deterrent = negative ΔDAS). Caveated-and-excluded:
{"Ter Stegen": "0_min", "Onana": "descriptive_only"}.

| keeper | expected | observed (mean ΔDAS) | meets prior |
|---|---|---|---|
| Alisson | negative | negative (−0.470) | YES |
| Neuer | negative | negative (−0.034) | YES |

**2 of 2 confirmed** (was 1 of 2 on the GS-only x-only run). Alisson remains a clear deterrent; Neuer
**flipped** from a marginal +0.015 (A+2) to a marginal −0.034 under the both-axes weights + fuller
corpus, now matching the prior. Both flips are small, so this is face-validity corroboration, not a
strong measurement — consistent with the weak-instrument headline.

Census: **116 resolved keepers, 74 gate-eligible** (min_nonzero=20, min_games=2), **35 of the eligible
match the expected-negative sign** (≈47 %, near chance — the weak-instrument signature); 0 unresolved
keeper frames; Layer-4 behavioural anchoring: `uninterpretable`.

## Per-keeper signed ΔDAS

The full 116-keeper sign table (Regime-O realistic dose, signed, |displacement| ≥ 2.0 m subset) is the
machine-readable deliverable `named_keeper_signs.parquet`; only Alisson (32) and Neuer (4602) carry a
resolved `keeper_name` (the owner-injected map). The `sign` column is **mean-based**; because ΔDAS is
zero-dominated and heavy-tailed, the mean and median can disagree in sign for a keeper, so a single
named-keeper eye-test is face-validity only, not a measurement (the parquet carries `median` for
inspection). The most-deterrent eligible keepers reach mean ΔDAS ≈ −2.6; the least, ≈ +3.0.

## Caveats
- **Reported-not-gated:** these numbers flip no gate and trigger no retrain.
- **ΔDAS only:** the threat arm is `arm_unscoreable` here; ΔDAS is NaN on velocity-less providers (SB360) by construction (ADR-063).
- The sign table is over the ≥ 2.0 m realistic-displacement subset (spec §4.1) — sharpens, does not invert, the deterrent sign; it is not the no-floor `build_gkdv_arm_values` population.
- **Gradient Sports keeper clamp (ADR-083):** GS tracking pins the keeper at 27.5 m from goal, so the GS actual-keeper leg is bounded; the SkillCorner/IDSSE cohorts are unaffected.
