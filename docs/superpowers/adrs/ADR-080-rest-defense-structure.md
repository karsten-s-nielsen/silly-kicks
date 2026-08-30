# ADR-080: Rest-defense structure metrics as a new `restdefense/` package (TF-60 PR1)

| Field | Value |
|---|---|
| **Date** | 2026-08-30 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

Rest defense (*Restverteidigung*) is the defensive rearguard an **in-possession** team keeps while
attacking, to blunt the opponent's counter after a ball loss. silly-kicks already owns every
*ingredient* — pitch control (TF-7), DAS (TF-28), team shape (TF-31/44), defensive line (TF-14),
team-in-possession (TF-5), packing (TF-49), and a deep GK stack (TF-13/15/18) — but had **no
team-level rest-defense structure metric** and no public/emitted goal-side counting primitive (the
only prior art is `GhostGkModel`'s private `defenders_behind_ball`). A targeted literature survey
(Forcher 2023, Peters 2025, Memmert 2022, Dash 2025) supplies validated KPIs; a separate finding is
that **no publication couples GK depth + defensive-line height + danger-of-space-behind into one
quantity** — the gap TF-60 fills.

The constraint that shapes the design is the provider spread: the family must work on both continuous
tracking (Sportec / GS / SkillCorner / Metrica) and **StatsBomb-360 freeze-frames**, which are single
snapshots per on-ball event — no inter-event frames, no velocity, FOV-cropped, anonymous players. The
one grain available on both is the in-possession team's **on-ball action grid**, which is also
silly-kicks' native action-centric grain.

TF-60 is decomposed into five cycles (PR1 descriptive Layer-1 KPIs; PR2 danger-behind-line valuation;
PR3 GK counterfactual arms; PR4 a trained ghost-outfield model; PR5 the outfield arm). This ADR
records **PR1**.

## Decision

Ship rest defense as a **new hexagonal `silly_kicks/restdefense/` package** (a new C4 container,
mirroring `gkdv/` / `xtgk/`), consuming `silly_kicks.tracking` public seams only, sampled at the
in-possession team's on-ball **action grid**, oriented entirely via `resolve_defended_goals` →
`GoalMap` (ADR-055; never team identity). PR1 delivers the **descriptive Layer-1 structure KPIs**
(`compute_rest_defense` / `summarize_rest_defense`) plus one new public goal-side counting primitive
(`count_goalside`) — additive, **no VAEP retrain**, in no default xfn list, and shipping the
*descriptive* KPIs, **not** the source papers' weak success-prediction models (Forcher AUC ≈ 0.60).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Tracking-resident (no new package) | fewer files | would reimplement gkdv's DAS direction-pin / identity-cache differencing (a documented silent-bug seam); breaks the gkdv/xtgk precedent | rejected — a composite metric subsystem is its own package (owner-confirmed) |
| B. Per-frame continuous sampling | denser | impossible on SB360 (no inter-event frames) | rejected — the action grid subsumes it and unifies both provider classes |
| C. Ship the papers' predictive success model | novelty | weak (AUC ≈ 0.60), off-charter | rejected — ship the descriptive KPIs (Forcher's own recommendation direction) |
| D. **New `restdefense/` package, action-grid, descriptive Layer-1 (chosen)** | hexagonal; SB360-compatible; reuses gkdv delta seams in later cycles | +1 C4 container; SB360 coverage is often partial | — |

## Consequences

### Positive

- A first team-level rest-defense structure metric family, on all providers (tracking-optimal, SB360
  best-effort), with the GK folded into the rearguard as a first-class agent — the literature gap.
- A reusable public `count_goalside` primitive (the numerical-superiority / behind-the-ball count).
- Additive: no existing feature changes, no VAEP retrain; later cycles (PR2–PR5) each depend only on
  the prior cycle's public surface.

### Negative

- One more C4 container to model and keep pinned to the code.
- The honest SB360 ceiling: the deep rearguard is exactly the region a ball-centred broadcast FOV
  crops out when the ball is advanced (the moment rest defense matters most), so every count/region
  metric carries the ADR-077 FOV-observability companions and a count at `observed_fraction < 1` is a
  **lower bound, never a measurement**.

### Neutral

- Calibratable defaults (`min_ball_advance_m`, `zone_depth_m`, `danger_field_weight`) ship un-tuned
  with an empty `for_provider` override map (the `CoverShadowParams` pattern, ADR-066); a per-provider
  tune is a separate gated apply PR (ADR-009), never this cycle.
- Three honesty-preserving edge decisions (the `/review-impl` gate's IMPL-02/03/04 CONSIDERs,
  owner-approved 2026-08-30): (IMPL-02) `rd_geometry_source` is a **three**-valued closed vocabulary
  `{resolved, guessed, unresolved}` — a `GoalMap` `allow_guess` outfield-mean fallback is labelled
  `guessed`, not `resolved`, because its defended-goal end is an inference (load-bearing on FOV-cropped
  SB360); (IMPL-03) `GoalEndUnresolvedError` is caught at the orchestrator edge as defence-in-depth
  behind the primary per-team pre-filter (policy at the edge, engine pure — ADR-055); (IMPL-04) a
  non-two-team frame set yields `pd.NA` numerical superiority rather than a silent A-count, since the
  absent opponent count would fabricate a 0 that reads A's whole rearguard as "superiority" (ADR-027,
  NA-never-a-0).
- `rd_compactness_x` (rearguard x-spread) and `rd_width` (rearguard lateral/y width) are BOTH the back
  line from `compute_defensive_line`; `rd_depth` is the WHOLE-TEAM `compute_team_shape.team_length` (a
  back-line depth would merely duplicate `rd_compactness_x` — a flat line has ~no independent depth — so
  the team's vertical stretch is the informative counter-vulnerability signal). This is "Option B",
  owner-ratified 2026-08-30 after the `/review-impl` gate flagged that the spec's "rearguard subset"
  wording did not match a first draft that shipped `rd_width` as whole-team; spec §7.1 is corrected to
  match, and this ADR records the owner's decision rather than asserting it.

## Related

- **Specs:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-30-tf60-rest-defense-pr1-layer1.md`
- **ADRs:** builds on ADR-055 (`GoalMap` orientation), ADR-019 (`id_compat`), ADR-068/073 (no-rescan +
  sub-quadratic guard), ADR-062/077 (FOV companions), ADR-063 (velocity tiers), ADR-078 (keeper
  identity), ADR-053 (SB360 audit), ADR-048 (feature glossary), ADR-005 (attribution). Reuses `gkdv`
  public delta seams in later cycles.
- **External references:** Forcher et al. (2023) JSSM; Peters et al. (2025) IJPAS; Memmert et al.
  (2022) MLSA; Dash et al. (2025) arXiv:2511.06191; FIFA (2022) Enhanced Football Intelligence. See
  `NOTICE`.

## Notes

PR1 also folds in a pre-existing NOTICE correction discovered while touching the same reference block:
arXiv:2511.06191 was mis-attributed to "Herold et al. (2022)" (it is Dash et al. 2025, the back-four
spatial-control paper `compute_defensive_line` implements) and arXiv:2511.00121 to "Forcher et al.
(2022)" (it is Yagi et al. 2025, a line-break paper). Both corrected, verified against arxiv.org.
