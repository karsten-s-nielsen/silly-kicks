# TF-19 A+2 — physics-arm instrument-validity + responsiveness (findings)

**Reported-not-gated.** silly-kicks 4.104.0 / PR-S175 / ADR-082.
Produced by `scripts/build_tf19_instrument_responsiveness.py` (see `metrics.json` for the machine-readable verdicts + provenance).

- **run_commit:** `07a88f69ce12ab07a229678f6f1c48db56893072` (tree clean)
- **Corpus:** Gradient Sports WC2022, 64 matches, delta_das arm over **123,430 scored domain frames**.
- **Registered thresholds:** SATURATING_MULTIPLE=5.0, PHYSICS_ARM_PROBE_RATIO=2.0, R=3, MIN_DOMAIN_FRAMES=200, REGIME_I_LADDER_M=2.0, REALISTIC_MIN_DISP_M=2.0.

## Headline

**ΔDAS is a WEAK instrument for keeper deterrence on this corpus.**

| Layer | Verdict |
|---|---|
| Layer 0 — instrument validity | `instrument_void` |
| Layer 1 — responsiveness | `not_responsive` |
| Threat arm (`delta_threat_suppression`) | `arm_unscoreable` (no loadable ExpectedThreat; the package ships no xG model) |

Pooled |ΔDAS| medians (attacker-value units; **negative = deterrent** for the signed metric):

| quantity | value | meaning |
|---|---|---|
| realistic (shipped ghost dose) | 0.4133 | the actual keeper-vs-ghost displacement |
| saturating (keeper on goal line) | 0.1249 | maximal keeper displacement (dose, not effect) |
| ladder 2 m (keeper) | 0.0562 | the imposed responsiveness dose |
| nearest-defender control | 0.3499 | one outfielder moved by the keeper's vector |
| single-outfielder placebo p95 | 2.9380 | R single-player placebos |

**Interpretation.** Accessible space (DAS) is dominated by the outfield frontier, so relocating the one deep keeper — even onto the goal line (saturating 0.12) — perturbs it *less* than moving a random outfielder by the same vector (placebo p95 2.94). The instrument therefore fails both the 5×-realistic and the placebo-band legs (Layer 0 `instrument_void`), and the keeper move is not specifically responsive (Layer 1 `not_responsive`). This is the probe working as designed — it is *detecting* that ΔDAS is not the right arm for keeper deterrence, not a null "no effect" claim about keepers. The threat arm would be the relevant signal but is unscoreable here (needs an xG-calibrated ExpectedThreat silly-kicks does not ship).

## Named-keeper face validity (PRE-REGISTERED prior, locked 2026-08-29 (before the owner run))

Prior: {"Alisson": "negative", "Neuer": "negative"} (deterrent = negative ΔDAS). Caveated-and-excluded: {"Ter Stegen": "0_min", "Onana": "descriptive_only"}.

| keeper | expected | observed | meets prior |
|---|---|---|---|
| Alisson | negative | negative | YES |
| Neuer | negative | positive | no |

**1 of 2 confirmed.** Alisson scores as a deterrent; Neuer scores ≈zero — consistent with a weak instrument (no clean deterrent pattern across keepers).

Census: 40 resolved keepers, 32 gate-eligible (min_nonzero=20, min_games=2), 15 of the eligible match the expected-negative sign; Layer-4 behavioural anchoring: `uninterpretable`.

## Per-keeper signed ΔDAS (most-deterrent first)

The sign table over the Regime-O realistic dose (signed; |displacement| ≥ 2.0 m subset). Machine-readable copy: `named_keeper_signs.parquet`.

The `sign` column is **mean-based**. Because ΔDAS is zero-dominated and heavy-tailed, the mean and median disagree in sign for **9 of the 40 keepers** (e.g. Yann Sommer mean −0.423 / median +0.001; Keylor Navas mean −0.344 / median +0.018) — the mean reads deterrent while the median sits at ≈zero. This **reinforces** the weak-instrument headline (there is no clean per-keeper deterrent signal), and it is why a single named-keeper eye-test is face-validity only, not a measurement. The parquet carries the `median` column for inspection.

| player_id | keeper | n | n_nonzero | n_games | mean ΔDAS | sign | matches_prior | gate_eligible |
|---|---|---|---|---|---|---|---|---|
| 12653 | Vanja Milinković-Savić | 3011 | 373 | 3 | -0.970 | negative | yes | yes |
| 13941 | Saad Al-Sheeb | 981 | 96 | 1 | -0.853 | negative | yes | no |
| 11241 | Andries Noppert | 5245 | 418 | 5 | -0.739 | negative | yes | yes |
| 1341 | Unai Simon | 2266 | 188 | 4 | -0.646 | negative | yes | yes |
| 13799 | Shuichi Gonda | 4733 | 271 | 4 | -0.592 | negative | yes | yes |
| 32 | Alisson | 3316 | 256 | 4 | -0.548 | negative | yes | yes |
| 13871 | Devis Epassy | 2786 | 237 | 2 | -0.501 | negative | yes | yes |
| 4691 | Yann Sommer | 3417 | 307 | 3 | -0.423 | negative | yes | yes |
| 13854 | Dominik Livakovic | 7583 | 630 | 7 | -0.399 | negative | yes | yes |
| 3841 | Keylor Navas | 5289 | 301 | 3 | -0.344 | negative | yes | yes |
| 462 | Wayne Hennessey | 2606 | 234 | 2 | -0.245 | negative | yes | yes |
| 144 | Emiliano Martínez | 5424 | 365 | 7 | -0.220 | negative | yes | yes |
| 2059 | Edouard Mendy | 3260 | 351 | 4 | -0.164 | negative | yes | yes |
| 3343 | Guillermo Ochoa | 2256 | 206 | 3 | -0.151 | negative | yes | yes |
| 14034 | Lawrence Ati Zigi | 3291 | 344 | 3 | -0.114 | negative | yes | yes |
| 3968 | Steve Mandanda | 612 | 62 | 1 | -0.059 | negative | yes | no |
| 8042 | Wojciech Szczesny | 6711 | 276 | 4 | -0.020 | negative | yes | yes |
| 4602 | Manuel Neuer | 2033 | 202 | 3 | +0.015 | positive | no | yes |
| 201 | Danny Ward | 1385 | 175 | 1 | +0.026 | positive | no | no |
| 6 | Ederson | 553 | 58 | 1 | +0.109 | positive | no | no |
| 1672 | Thibaut Courtois | 3011 | 159 | 3 | +0.122 | positive | no | yes |
| 13987 | Mohammed Al-Owais | 3963 | 497 | 3 | +0.125 | positive | no | yes |
| 61 | Hugo Lloris | 5538 | 311 | 6 | +0.126 | positive | no | yes |
| 13924 | Sergio Rochet | 3254 | 201 | 3 | +0.165 | positive | no | yes |
| 13968 | Alireza Beiranvand | 1155 | 122 | 1 | +0.170 | positive | no | no |
| 13942 | Meshaal Barsham | 1425 | 95 | 2 | +0.192 | positive | no | yes |
| 940 | Matt Turner | 2930 | 239 | 4 | +0.199 | positive | no | yes |
| 4656 | Gregor Kobel | 745 | 58 | 1 | +0.231 | positive | no | no |
| 13969 | Hossein Hosseini | 2571 | 277 | 2 | +0.266 | positive | no | yes |
| 14013 | Aymen Dahmen | 2773 | 187 | 3 | +0.322 | positive | no | yes |
| 8018 | Andre Onana | 952 | 106 | 1 | +0.367 | positive | no | no |
| 13862 | Munir | 1074 | 125 | 1 | +0.383 | positive | no | no |
| 1785 | Bono | 4926 | 435 | 6 | +0.438 | positive | no | yes |
| 8538 | Diogo Costa | 3387 | 358 | 5 | +0.492 | positive | no | yes |
| 435 | Mathew Ryan | 5189 | 344 | 4 | +0.680 | positive | no | yes |
| 200 | Kasper Schmeichel | 2096 | 133 | 3 | +0.928 | positive | no | yes |
| 169 | Jordan Pickford | 3346 | 270 | 5 | +0.948 | positive | no | yes |
| 14031 | Milan Borjan | 2753 | 270 | 3 | +1.017 | positive | no | yes |
| 13935 | Hernan Galindez | 1258 | 91 | 3 | +1.185 | positive | no | yes |
| 13907 | Seun-gyu Kim | 4326 | 422 | 4 | +2.252 | positive | no | yes |

## Caveats
- **Reported-not-gated:** these numbers flip no gate and trigger no retrain.
- **ΔDAS only:** the threat arm is `arm_unscoreable` here; ΔDAS is NaN on velocity-less providers (SB360) by construction (ADR-063).
- The sign table is over the ≥ 2.0 m realistic-displacement subset (spec §4.1) — sharpens, does not invert, the deterrent sign; it is not the no-floor `build_gkdv_arm_values` population.
