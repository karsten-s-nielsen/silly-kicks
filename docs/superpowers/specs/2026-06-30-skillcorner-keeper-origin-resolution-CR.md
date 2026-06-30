# Change request: SkillCorner keeper-origin resolution (broadcast-tracking domain fix)

**Date:** 2026-06-30 · **From:** xT-GK analysis side (Karsten) · **For:** silly-kicks session to plan + implement (TDD)
**Type:** change request — requirements + acceptance. The *how* is yours to plan; this is the *what* and *why*.
**Cross-repo context (read for depth):**
`karstenskyt__luxury-lakehouse/docs/investigations/2026-06-29-skillcorner-keeper-origin-coordinate-scatter.md`
(full investigation) · `…/docs/superpowers/specs/2026-06-30-skillcorner-keeper-origin-rebuild-and-access-tier-completion.md`
(lakehouse spec — §1.3 ownership, the boundary) · this repo: `docs/superpowers/specs/2026-06-29-xtgk-pre-jeff-verification-handoff.md`.

> **⚑ Superseded in part (4.37.0, analysis-side approved):** real-data validation narrowed the **S3** distrust
> scope to **goal-kicks only**. The native origin of an **open-play GK pass/throw IS the keeper** (measured 0.4 m
> vs the detected keeper — the ball is at the keeper at release), so those keep native (no `unresolved` discard);
> only goal-kicks (native = broadcast ball ~14–20 m downfield) are distrusted. An ADR-028 home/away re-projection
> of the detected keeper was also added. See ADR-024's 4.37.0 amendment + the design doc's post-validation banner.

## Why this is silly-kicks' (the boundary, already agreed)
Every part below is **general broadcast-tracking domain/geometry**, not warehouse logic — any silly-kicks consumer
ingesting SkillCorner broadcast tracking hits the same problem. It lives where the domain already lives
(`resolve_gk_geometry`, the SkillCorner frame converter), consistent with the deleted lakehouse-side orientation net
(TF-23). The lakehouse stays thin: it adopts this release, plumbs one provenance enum to its mart, and renders
`unresolved` as NULL — none of which is your concern here. **Do not** reach into warehouse/`access_tier`/mart
plumbing.

## The bug (data-grounded)
SkillCorner GK-distribution origins are **scattered across the pitch** (goal-kick `start_x` min 0.8 / max 98.4 /
SD 23.2; own-box rate 51% vs ~100% for every other provider; passes 60% own-box). The SPADL action's native origin is
the **broadcast ball-detection event location, not the keeper's position** — and `resolve_gk_geometry` trusts it
(`xt_gk_origin_source=native`, conf 1.0) because it's non-NaN. This corrupts `xt_gk` base/DZV and the keeper
pressure/PEV (computed at the wrong position).

Key supporting facts (verified against the data):
- **Keeper is identifiable + tracked**: `is_goalkeeper` / lineup role resolves the keeper; bronze
  `skillcorner_tracking` carries per-player **`is_visible`** (real detection) + `ball_is_detected` + native coords.
- **Detection is partial (broadcast)**: keeper detected ~24.7% of frames overall, **~58% at the keeper's own action
  frames, ~70% within ±1s**. Reliable *when detected* (position near the correct goal, correct half-switch).
- **Coordinate convention**: SkillCorner native tracking is **center-origin** (x∈[−52.5,+52.5], y∈[−34,+34]) and
  **physical** (ends switch each half). Bronze legitimately has players to ±60 native (keepers behind the goal line).

## The four changes

### S1 — Coordinate transform correctness in the SkillCorner frame converter (`convert_to_frames`)
Transform SkillCorner native (center-origin ±52.5) → SPADL [0,105]×[0,68] + home-LTR. **Within-pitch invariant:** a
correct transform keeps players within the pitch except a **few-metre tolerance** for legitimately off-pitch players
(keepers behind the goal line). **Gross** off-pitch (well beyond tolerance) must be impossible under a correct
transform → if it occurs, **fail loud** (assertion pointing upstream), **never silently clamp**. (Assess whether the
current converter already transforms correctly — this may be a verification + invariant rather than a from-scratch
transform.) Scope: the **AC/xt_gk frame path** (`convert_to_frames`); the lakehouse's separate `fct_tracking_frames`
transform is theirs (L4), out of scope here.

### S2 — Carry the per-player detection bit through the frame model
`convert_to_frames` must preserve the SkillCorner per-player detection signal (`is_visible`) on the frame model —
**not interpolate it away** — so the resolver can distinguish a real keeper detection from a held/interpolated
position. (The downstream lakehouse mart currently loses this; the bronze has it.)

### S3 — Tiered keeper-origin resolution in `resolve_gk_geometry`
Stop trusting a non-NaN **broadcast ball-event coordinate** as a keeper origin. Resolve keeper-action origins by a
fallback ladder, **by action type** (detection is only ~58–70%, so one rule is dishonest):
1. keeper **detected at / within ±1s** of the action → **tracked keeper position** (transformed). *(best)*
2. else **goal-kick** → **rule-point prior ≈(5.5, 34)** — reliable; goal kicks are always taken from the goal area
   (the existing `goalkick_prior` path).
3. else **open-play pass / throw, no detection** → **`unresolved`** (flag) — no honest prior; **never impute a guess**.

Emit provenance per action: `xt_gk_origin_source ∈ {tracking_gk, goalkick_prior, unresolved}`. This **generalizes the
GS NaN-imputation you already do** — the new judgment is that a *non-NaN* native origin is not automatically a keeper
position for a broadcast provider ("is this a keeper position or a ball-detection artifact?" is domain logic).
**Confirmed by the analysis side:** tier-3 is flag-`unresolved`, no impute (do NOT add a weaker prior).

### S4 — Loud validation (defense-in-depth)
A *native* goal-kick origin implausibly far from goal (beyond the penalty area) should **fail loud / flag**, so a
future provider feeding ball-location-as-origin can't pass silently. Belongs next to the resolver. (Tracked as the
companion to S3; not a separate workstream.)

## Acceptance
- SkillCorner goal-kick origins ≈ 100% own-box; pass origins localize; the scatter SD collapses.
- Frame coords within the pitch (few-metre tolerance); **gross** off-pitch fails loud (no silent clamp).
- `unresolved` subset is produced + countable, **never imputed**; provenance enum populated correctly per tier.
- **Regression gate:** GS / idsse / metrica keeper-origin resolution + values **unchanged** (this is SkillCorner-only;
  GS's existing `native`/`tracking_gk`/`goalkick_prior` behavior must be byte-identical).
- A golden/representative test exercising each tier (detected→tracked, goal-kick→prior, no-detection→unresolved).

## Sequencing / ownership notes
- **You ship first**, as a silly-kicks release — it's the prerequisite the lakehouse adopts (version bump → recompute).
- **Not yours:** the `xt_gk_origin_source` mart enum + `unresolved`→NULL rendering + `access_tier` + the
  `fct_tracking_frames` re-point (lakehouse L4). Keep this change to the converter + resolver.
- Open for you to decide in planning: the exact ±1s detection window + nearest-detected-frame snapping rule; whether
  S1 is a fix or a verification+invariant; test fixture shape (use production-realistic SkillCorner geometry per the
  pre-Jeff handoff's test-parity note).
