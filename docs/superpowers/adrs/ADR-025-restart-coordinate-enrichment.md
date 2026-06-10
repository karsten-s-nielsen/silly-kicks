# ADR-025: General restart-coordinate enrichment (Phase 1, additive)

| Field | Value |
|---|---|
| **Date** | 2026-06-10 |
| **Status** | Accepted |
| **Deciders** | Karsten (with Claude); two independent review sessions |

## Context

Real Gradient Sports data (owner-tier, WC2022) ships a large fraction of set-piece / restart events
with a NaN coordinate. A live lakehouse probe (2026-06-10, `bronze.spadl_actions`, ~9.74M actions)
quantified it: NaN coordinates are **almost entirely a Gradient Sports phenomenon** — StatsBomb (73%
of the corpus), Wyscout (25%), and SkillCorner have **0.00%** NaN coordinates; within GS the gap is
**concentrated in restart types that all have a Law-defined restart location** (goal-kick 60%,
free-kick-short 35%, corner 30–34%, throw-in 27%, penalty 23% NaN-start). The xT-GK release (4.21.0)
closed this *for goal-kicks only*, inside the xT-GK valuation, via a private `resolve_gk_geometry`
helper (ADR-024). The TODO follow-up asked to promote that helper into a general, codebase-wide
coordinate-enrichment feature.

Changing the *canonical* `start_x`/`end_x` columns is a Hyrum/retrain trigger for every trained model
(VAEP, xT, calibration), shifts every golden test, and changes the coordinate data contract. That
blast radius must not be paid up front, before any consumer needs the canonical change.

## Decision

Ship **Phase 1 only: an additive, zero-retrain enrichment** — a public
`silly_kicks.spadl.add_restart_coordinates` (frames-optional) that imputes coordinates for all
Law-fixed-spot restart types (goal-kick, penalty, corner, throw-in) and emits them as **new**,
provenance-tagged columns (`enriched_start_x/_y`, `enriched_end_x/_y`, `start_coord_source` /
`end_coord_source`, `start_coord_confidence` / `end_coord_confidence`), **never mutating** the
canonical coordinate columns. The goal-kick-scoped `resolve_gk_geometry` is consolidated to delegate
to a single general engine `resolve_restart_geometry` (parameterised by `impute_types`) so its 4
internal callers + the xT-GK completion path stay byte-identical — no retrain. The canonical
promotion (Phase 2) is a deferred, separate PR with the coordinated retrain.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Heavy now: replace canonical `start_x/end_x` | downstream metrics improve immediately | retrains VAEP/xT/calibration, shifts every golden, new data contract — huge blast radius before any consumer needs it | premature; mirrors the exact reason the xT-GK release scoped it out |
| B. Keep two independent resolvers (no consolidation) | zero risk to the frozen path | two coordinate resolvers that drift apart — the thing the TODO's "decoupled so extraction is cheap" was meant to avoid | divergence risk |
| C. (chosen) Additive Phase 1 + single engine via `impute_types` + tripwire at the edge | immediate opt-in value, zero retrain, single source of truth, frozen path byte-identical AND zero-cost | the canonical win waits for Phase 2 | — |

The single-engine consolidation initially risked two problems: (1) the geometry tripwire leaking
warnings onto the frozen `resolve_gk_geometry`/`compute_xt_gk` path, and (2) wasted `_tracking_ball_xy`
work on that hot path. Both were resolved by one lever — the engine's `impute_types` parameter (the
shim passes `(goalkick,)` → the engine does zero non-goalkick work, needs no revert) — plus moving the
tripwire to the `add_restart_coordinates` edge (the engine is pure: no warnings, no reverts).

## Consequences

### Positive

- Any consumer can read enriched restart coordinates + per-row provenance/confidence with **no model
  retrain** and no change to existing behavior. The lakehouse's events-only path (`frames=None`) gets
  population-aggregate coverage; the frames path gets position-sensitive imputation.
- One coordinate-resolution code path (`resolve_restart_geometry`); `resolve_gk_geometry` is a thin,
  byte-identical shim over it (parity-gated by a committed golden snapshot).
- Phase 2 (canonical promotion) becomes mechanical: copy `enriched_* → canonical *` + retrain +
  golden re-baseline. The enriched columns are the promotion contract.

### Negative

- Two public entry points with slightly different behavior: `resolve_restart_geometry` (raw, no
  tripwire) vs `add_restart_coordinates` (applies the tripwire). Documented.
- Events-only position-sensitive value is essentially nil under the recommended `≥0.7` filter (the
  only non-native tier there is the ≤0.5 rule-point) — stated honestly; position-sensitive value
  needs tracking frames.
- `freekick_short` (the single largest events-only-unresolved NaN bucket) is intentionally left
  unresolved — it has no Law-defined spot.

### Neutral

- New released `*_coord_source` enum values (7): `native` / `tracking_ball` / `tracking_gk` /
  `restart_prior` / `next_event` / `unresolved` / `tripwire_reverted` (origin only). Once released,
  these are Hyrum-observable.
- Destinations are NOT tripwire-guarded in Phase 1 (the regions are origin-framed); a Phase-2
  candidate.

## Phase-2 promotion recipe (for the future apply-PR)

1. Copy `enriched_* → canonical start_x/start_y/end_x/end_y` (or flip a converter flag).
2. Coordinated retrain: VAEP, xT, calibration. Re-baseline every coordinate-touching golden test.
3. Promote the geometry tripwire from warn/edge-policy to a hard converter-level gate.
4. **`enriched_*` is NaN for `unresolved`/`tripwire_reverted` rows** — but those rows had NaN *native*
   coordinates anyway, and `native` rows are never reverted, so a `coalesce(enriched, native)`-style
   promotion is a no-op regression there. Do NOT write a guard that assumes `enriched_*` is always
   finite.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-10-general-restart-coordinate-enrichment-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-10-general-restart-coordinate-enrichment.md`
- **ADRs:** builds on ADR-024 (xT-GK / `resolve_gk_geometry`); follows the ADR-009 recommend-then-apply
  split pattern; preserves ADR-001 (converter identifier conventions — the fix lives at the enrichment
  seam, not in converters); reuses ADR-003 (`@nan_safe_enrichment`), ADR-005 §5 (lazy spadl→tracking
  import), ADR-018 (geometry-tripwire pattern), ADR-019 (`is_ball` coercion / id-dtype safety).
- **External references:** Laws of the Game restart locations (goal area, penalty spot, corner arc,
  touchline).

## Notes

Live-probe evidence (2026-06-10, `soccer_analytics.bronze.spadl_actions`): per-provider NaN-start% —
statsbomb 0.00, wyscout 0.00, skillcorner 0.00, gradientsports 5.40, idsse 1.90, metrica 3.69. The
"degrades EVERY coordinate-consuming metric across the corpus" framing in the original TODO is **not**
what the data shows — the gap is a GS set-piece phenomenon, which is why a Law-geometry prior is
defensible. Aggregator count unchanged (27) → C4-free.
