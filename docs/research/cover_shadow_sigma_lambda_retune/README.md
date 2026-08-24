# Cover-shadow σ/λ discrimination re-tune — GS decision (ADR-066, TF-30 b)

**Outcome: keep the incumbent σ/λ (0.20 / 4.3). No library change.** The σ/λ re-tune found only a
marginal, boundary-degenerate optimum on Gradient Sports WC2022; per this *reported-not-gated* cycle,
the incumbent is retained. `silly_kicks.tracking._cover_shadows._PROVIDER_COVER_SHADOW_PARAMS` has no
GS entry; `CoverShadowParams()` is unchanged.

Aggregate statistics only (σ/λ, AUC, gain, rates) — no raw per-pass GS data (owner-tier).

## The pipeline (all owner-run, local pining)

1. **De-leak.** The bundled public `ReceiverModel` (4.91.0) replaces the outcome-selected `end_xy`
   target of each intercepted failed pass with its inferred intended receiver
   (`build_rq_pass_scores --receiver-model default`; run at clean `1018dc0`, de-leak path byte-identical
   after the 4.90.x merge). 61,944 passes / 64 matches; 9,546 failed passes de-leaked. **De-leaked
   majority recall 0.224** (vs the leaked-target legs of the 4.87.0 headline).

2. **Receiver-validity (apply conjunct 1).** On the trajectory-validated intercepted subset
   (`coverage = 0.450`, 4,327 of 9,609 interceptions), pooled over 64 matches:

   | | top-1 |
   |---|---|
   | public receiver | 0.266 |
   | geometric proxy | 0.093 |
   | **receiver_margin** | **0.173** |

   The receiver **decisively beats** the geometric proxy (≫ the 0.05 floor) — so receiver-validity
   passes, and the apply gate does *not* short-circuit. (`pooled_top1 0.26600` reproduces the owner
   deployment's public-model number exactly, cross-validating the measurement.)

3. **σ/λ discrimination sweep.** `CoverShadowDiscriminationObjective` — maximize the failed-vs-completed
   margin-AUC subject to the completed-pass FP rate not exceeding the incumbent's — over σ ∈ [0.02…0.10],
   λ ∈ [1.5…3.5] (an extended-LOW grid; the original coarse grid argmax pinned to its low edge, and a
   10-match probe confirmed the optimum sits below it). The lane-pressure ablation re-runs the sweep on
   targets de-leaked by a `{ball_dist, space}` receiver retrained without `lane_pressure`.

   | | σ / λ | margin-AUC |
   |---|---|---|
   | incumbent | 0.20 / 4.3 | 0.585 |
   | candidate (with lane-pressure) | 0.04 / **1.5** | 0.595 |
   | candidate (without lane-pressure) | 0.06 / 1.5 | — |
   | gain (64-fold) | | **+0.0098** |

   `ablation_share 0.177`, `noise_ok true` → the pre-registered `decide_apply` returns **`applied`**.

## Why it is NOT applied (the owner decision)

`decide_apply` tests receiver-validity, bias and noise — but **not** whether the argmax is a *bounded*
optimum. Three signals say the measured candidate is not a valid re-tune:

- **λ is boundary-degenerate.** The λ argmax pins to the grid minimum (1.5) and moves *further out* as
  data grows (10-match probe λ=2.0 → 64-match λ=1.5) — an unbounded descent, not a bounded optimum.
- **The gain is marginal** — +0.0098 margin-AUC (≈1 pp), barely over the 0.005 effect-size floor.
- **The parameter change is large and partly bias-driven** — σ 0.20 → 0.04 is a near-unblurred
  cover-shadow, and removing the receiver's `lane_pressure` feature shifts the σ argmax (0.04 → 0.06),
  i.e. some of the shift *is* the open-target bias the ablation exists to catch (ablation 0.177, no
  longer 0.0, though still under the 0.50 bar).

Shipping σ=0.04 for +0.0098 AUC on an edge-clipped, partly-biased optimum is not a defensible default.
Boundary-degeneracy is a validity failure outside the pre-registered conjuncts; the honest landing for
a reported-not-gated cycle is **keep the incumbent**. The full sweep stands as the evidence.

## Reproduction

`build_rq_pass_scores --receiver-model default` (de-leak) → the receiver-validity + σ/λ sweep
measurements (pooled `receiver_failed_pass_accuracy` and `CoverShadowDiscriminationObjective.argmax` +
`lane_pressure_shift_share` + `select_recommended_point` for the ADR-060 noise test). Aggregate results
in `sweep_summary.json`. Non-GS per-provider σ/λ remains deferred (each its own re-tune on its own
velocity profile). See ADR-066.
