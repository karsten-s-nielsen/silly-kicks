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

## Amendment (post-review, real-data) — SB360's identity gap forced a per-provider labeling design

The original premise — "train the public model on SB360" — was **falsified on real pining data** before the
first bundle. Two defects, invisible to the unit fixtures (which gave freeze-frame rows the *same real ids*
as the actions and always a ball row), surfaced on the first match per provider:

1. **SB360 carries no player identity** (ADR-062): `snapshot_to_tracking_frames` numbers freeze-frame rows,
   so the candidate id is a row index while the observed receiver is a real `player_id` — **0 positive
   labels** on 482 passes. Id-based labeling is structurally incompatible with SB360.
2. **A ball-less GS frame** (~3.1%) crashed the owner feature extractor (`_ball_release_dir`), unguarded
   per-pass, aborting the whole match.

The design response (green-lit in a design review, whose bindings this records):

- **Per-provider labeling strategy, decided at frame-set granularity — NEVER per-pass** (Q4).
  `labeling_strategy_for_provider`: identity providers → `"id"` (clean); SB360 → `"trajectory"`. An identity
  provider's pass whose receiver has no in-frame id match is **dropped and counted, never trajectory-guessed**
  — a per-pass "id-match else trajectory" fallback would silently mislabel a GS pass whose receiver ran
  off-frame, eroding the clean-label leg the pool trusts.
- **Trajectory label anchored on the next same-team touch** (Q1): the label's *identity* and its *reception
  point* derive from ONE action, not two. Reuses the `trajectory_weak_labels` ray idea.
- **Drop-on-ambiguous with a tighter, SEPARATE label lane-width** (Q2, `_LABEL_LANE_WIDTH_M`): a pass with no
  candidate clearly on the ray is dropped — never labeled all-zero (a false negative on the true receiver),
  never guessed onto the nearest visible teammate (a confident mislabel on visibility-truncated SB360).
- **SB360-primary + GS earns inclusion via a held-out gate** (Q3, `pooling_gate`): GroupKFold on the primary
  games, pool added to TRAIN only, kept iff it does not regress the primary held-out top-1. Naive pooling of
  a clean and a noisy source at different candidate regimes (M2) is unjustified; the gate's outcome and the
  pool's coverage are recorded in the manifest. `--pool-provider` opts it in.
- **Fail-soft on a per-frame gap, loud on a misconfiguration** (Q5, `NoReleaseDirectionError`): a ball-less
  frame is a per-frame gap → skip/NaN (train and serve), counted; a *missing `vx`/`vy` column* is the owner
  variant routed to a velocity-less provider → a LOUD `KeyError`, never swallowed. This is the fail-loud
  discipline (ADR-051/ADR-010) — *not* ADR-043, which is GKDV v1. **All FIVE `geometric_proxy_receiver`
  callers handle it** — `extract_candidate_rows`, `rank`, `resolve_intended_receiver`,
  `intended_receiver_positions`, and (surfaced only on the real GS deployment pass, not the fixtures) the
  M-A(ii) `receiver_failed_pass_accuracy` proxy, which skips the ball-less frame because the deployment gate
  scores on `top1`, not the proxy.

**Load-bearing consequences** a future maintainer will ask "why?" about: the public model is trained on the
serve distribution (SB360) with trajectory labels; GS is pooled in only if the held-out gate clears (outcome
recorded per run); label coverage is reported so drop-on-ambiguous thinning is visible against the
`--min-passes` floor. **The passer/actor artifact (review Finding A-F1), MEASURED not assumed:** on
identity-less frames the passer's real id can't match a synthetic frame row, so id-exclusion leaves the ACTOR
as an acting-team candidate at ~the release, ~on the pass ray — where, *if annotated forward of the release*,
it would win a trajectory label (perp≈0) and mislabel the pass onto the passer. Measured on real SB360 (match
4047626, 482 passes): the actor sits at the release EXACTLY (displacement 0.0, `proj>0` in **0.0%** of
passes), so the `proj≤0` forward filter already prevented any win — the defect is LATENT, not active.
`_split_players` now also EXCLUDES the acting-team candidate within `_PASSER_EXCLUSION_M` of the release
(identity-less only, gated on id-exclusion having matched nothing) — data-cleanliness + defence against
annotation noise. A genuinely *forward* actor (never observed) would need the freeze-frame `actor` flag
propagated through the port (a schema change ADR-062's numbered-rows contract does not carry); recorded as the
escalation trigger, not a deferred defect. **Two provenance/robustness fixes in the same pass (Findings
B-F1/F2):** a pool that contributes ZERO rows no longer flips `providers_trained`/visibility (the empty-pool
tie is gated on `len(pool_rows)`), and an empty primary raises the `vacuous training set` `SystemExit` rather
than a `KeyError` on a column-less reconcile. The leakage contract is unchanged: labels may read the outcome
(they always must); features stay pre-pass, now pinned by a **driver-layer** guard.
