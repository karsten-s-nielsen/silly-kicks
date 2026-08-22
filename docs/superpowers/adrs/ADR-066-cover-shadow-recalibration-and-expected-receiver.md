# ADR-066: Cover-shadow σ/λ discrimination re-tuning + expected-receiver model

| Field | Value |
|---|---|
| **Date** | 2026-08-20 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner); drafted with Claude (Opus 4.8) |
| **Uses / amends** | **amends ADR-009** (gated in-cycle apply); uses ADR-011/060/052/037/062; supersedes-not the ADR-064 RQ1 cycle |

## Context

ADR-064 (4.87.0) validated the cover-shadow model on GS WC2022 and left two tracked follow-ups: adapt
`CoverShadowParams.sigma=0.20 / lambda_ctrl=4.3` to our measured-velocity profile, and remove the RQ1
recall leak (a failed pass's target was the outcome-selected release-frame `end_xy`). Spec:
`docs/superpowers/specs/2026-08-20-cover-shadow-recalibration-and-expected-receiver-design.md`.

## Decisions

**1. Expected-receiver model (public SB360 default + conditional GS-owner variant).** A per-candidate
logistic `ReceiverModel` (ADR-011 bundle: SHA + chirality + feature contract; sklearn-free serve) infers the
intended receiver from PRE-PASS state ONLY. The **only leakage-free release direction is the ball's release
velocity** (GS-only); the pass-event angle is origin→end and is BANNED, so the **public feature set is
positions-only** (leakage-free on velocity-less SB360 freeze frames), and the **owner (GS) variant adds** the
velocity-derived direction + closing speed. Leakage is pinned by an **output-invariance** guard: perturbing
the end location leaves the features byte-identical.

**2. This is exploratory σ/λ DISCRIMINATION re-tuning, not magnitude "recalibration" (M1).** margin-AUC
re-tunes the σ/λ SHAPE; magnitude calibration needs counterfactual block ground truth we lack. The incumbent
σ/λ scored 0.764 (ADR-064) and is not demonstrably broken — the **honest-null is the likely, acceptable
outcome**, reached through named, pre-registered gates.

**3. The failed-pass validity is a trajectory-weak-labelled UPPER BOUND on the easy tail (H1/R1).** The
covered subset (clear-trajectory interceptions) is where the intended receiver is easiest to infer AND the
proxy is near-ceiling, so a non-decisive gate reads *"unvalidatable where the model would matter,"* NOT "no
value." The receiver serves on GS via a **train-SB360 / serve-GS + completed→failed double transfer** (M3),
validated only on the easy tail; SB360's visible area truncates the negative candidate set (M2, recorded).

**4. The apply is three-conjunct, per-provider, gated on pre-registered thresholds (H2/H3/R5; ADR-060).**
`decide_apply` moves σ/λ only if receiver-validity (coverage ≥ `MIN_COVERAGE` AND margin ≥
`MIN_RECEIVER_MARGIN`) AND bias (lane-pressure ablation share < `MAX_BIAS_SHARE`) AND ADR-060
`exceeds_noise_floor` all hold; else an outcome in `{null:unvalidatable, null:biased, null:within-noise}`.
Thresholds are referenced constants, never literals. The apply is **per-provider** via
`CoverShadowParams.for_provider` — a σ/λ fit on one provider's velocity profile is a global default for NONE
(H3) — so every null path leaves the library byte-identical, and `applied` is a small committed GS constant.

**5. ADR-009 amendment.** The harness RECOMMENDS + emits a manifest (never mutates defaults); the DELIBERATE,
gated per-provider apply lives in the same cycle. Small + gated (no bundled-model re-train; `cover_shadow_xfns`
is opt-in), so the consumer VAEP-retrain trigger is GS-scoped and conditional.

## Consequences

- **New public surface:** `resolve_intended_receiver` / `intended_receiver_positions` + `ReceiverModel`
  bundle; `CoverShadowParams.for_provider`; a `calibration` σ/λ objective.
- **Reported-not-gated, no library change unless the gate clears** (honest-null likely). C4-free (no new
  aggregator/subpackage). The de-leaked harness is opt-in (`build_rq_pass_scores --receiver-model`);
  `model=None` is 4.87.0-byte-identical.
- **Deferred, tracked:** non-GS per-provider σ/λ (each its own re-tune); TF-51 Track B consumes the receiver
  primitive.
- Attribution: Power et al. 2017 (receiver); Cascioli et al. 2025 (σ/λ). See `NOTICE`.
