# ADR-032: retire the dead `pitch_control_at_ball` column, re-aim pitch control to the action destination

| Field | Value |
|---|---|
| **Date** | 2026-06-16 |
| **Status** | Accepted (PR-S96, silly-kicks 4.31.0) |
| **Deciders** | Karsten (with Claude); lakehouse session (cross-session spec + plan review) |

## Context

`silly_kicks.tracking.features.pitch_control_at_action` sampled the pitch-control (PPCF) surface at the
action's `(start_x, start_y)` — a "proxy for ball position." Because the action *happens at the ball*, the
query was always ~0 m from the frame ball, where the **Spearman PPCF is the 0.5 reaction-time fallback** (the
ball reaches near cells before any player's reaction time, so no player accrues control). The emitted
`pitch_control_at_ball__spearman` was therefore ~0.5 for every well-linked action — **informationally
dead**, declared a `STRUCTURAL_CONSTANTS` entry in the liveness gate and flagged to the lakehouse in the
4.24.0 changelog.

**A latent ADR-028 bug was masked by the degeneracy.** `pitch_control_at_action` sampled the action-LTR
`start_x/start_y` against a surface built from absolute-frame (home-attacks-right) frame positions, with **no
per-action re-projection**. For away-team actions those conventions are a 180° point reflection apart, so the
query hit the wrong cell — harmless *only* because near-ball is 0.5 in both conventions. This is why ADR-028
could list pitch_control as "untouched": the degeneracy hid it. Any meaningful (non-near-ball) sample point
resurrects it as a real away-team bug.

Spec: `docs/superpowers/specs/2026-06-16-pitch-control-at-target-design.md`.

## Decision (Option A — retire and replace)

1. **Retire `pitch_control_at_ball__<method>`** (the 4.24.0 lean-contract pattern for dead columns): drop the
   column, the `STRUCTURAL_CONSTANTS` entry, and the near-ball-degeneracy invariant test.
2. **Add `pitch_control_at_target__<method>`** sampled at the action **destination** `(end_x, end_y)`.
   Ball-travel-time to the destination is strictly positive → players can contest → non-degenerate, and the
   value answers a real question: *does the acting team control the destination of this action?* All three
   methods (`spearman`/`fernandez_bornn`/`voronoi`) move to `at_target`.
3. **Re-project the query into the frame convention (mandatory).** Per-action flip via
   `acting_team_attacks_rtl(actions, frames)` + `reproject_to_action_ltr` on the query point (the 180°
   reflection is involutive, so applying it to the action-LTR `(end_x, end_y)` yields the absolute-frame
   point the surface is keyed in). The cached per-frame surface stays absolute-frame (`PitchControlCache` key
   unchanged) — only the query flips. Atomic mirror synthesizes `end_x=x+dx, end_y=y+dy` and delegates, so
   the re-aim + re-projection apply uniformly.

**Localized scope.** The degeneracy was specific to *sampling at the ball*; the other PPCF consumers
(`obso`, `cover_shadows`, `gk_influence`, `player_influence`, `space_creation`) sample the surface at their
own relevant points and are untouched. The rejected potential-control model variant (changing the surface
near the ball) would have touched every consumer — disproportionate to reviving one column.

**Semantics — kept uniform, interpreted per type.** For passes/crosses/carries `at_target` is open-play
control of the destination; for **shots** the destination is the shot target (GK/defender-dominated), so it
reads as target-cell contestation; **in-place** actions (`end ≈ start ≈ ball`) read ~0.5 by construction
(no spatial destination — honest). Shots are kept (not excluded — no NaN special case absent from the old
all-action column); a model conditions on the per-type interpretation via `type_id`. The docstring states this.

## Correctness gate (the TDD core)

Mirror-symmetry alone is **necessary but not sufficient** — a symmetric-wrong / cancelling-double-flip
projection makes both mirrored frames agree on the same wrong value and passes a symmetry-only test. So the
headline guard is a **ground-truth, hand-computable, ASYMMETRIC + EXTREME** fixture: the destination cell is
acting-team-controlled (≈1.0) while its 180° reflection is opponent-controlled (≈0.0), run for both a home
and an away action. The away case lands on the reflection (≈0.0) under a wrong-direction flip → RED; correct
→ ≈1.0. A **multi-action mixed home/away** test (one call, ≥2 rows) pins the per-action flip *vectorization*
the production path uses (single-action tests can't catch a row-alignment bug). Confirmed during execution:
mirror-invariance *passed* on the degenerate 0.5 state, concretely demonstrating it is not the correctness
gate. The liveness gate carries a hard **off-ball-destination precondition** (≥2 actions with
`dist(end, ball) > 10 m`) so it can't silently lose its teeth on a fixture refactor.

## Consequences

- **VAEP/tracking + calibration retrain trigger.** The column goes from a dead ~0.5 constant to a live
  destination-control signal (and away-team values are corrected). The silly-kicks calibration
  `_features.py` augmented-VAEP Brier feature set + the lakehouse feature set re-materialize.
- **Lakehouse adoption is a full column-lifecycle MIGRATION, not a find-replace.** The
  `pitch_control_at_ball__*` → `pitch_control_at_target__*` rename lands across **AC** (`schema.py`,
  `enrich.py`, `action_context.py`, `oracle_map.py`, `tracking_context.py`) **and DEFCON**
  (`defcon_lite*.py` + the schema-parity tests): bronze drop+add migration + runner, dbt rename, Lakebase
  reshape, HF republish, DEFCON-parity tests, a forced AC recompute. **4.31.0 is BREAKING and ATOMIC with
  the migration — NOT a currency pin-bump** (running AC against 4.31.0 before the migration → KeyError).
  Batch the recompute with the parked Metrica y-fix recompute; A/B-validate the new feature (Brier improves /
  no regress), don't blind-adopt.
- **C4-free:** no new aggregator/container; the action-coupled aggregator count stays 28.
- References: ADR-028 (per-action LTR re-projection — the helper reused here), ADR-008 (`PitchControlCache`),
  ADR-005 §8 (`<feature>__<method>` naming).

## Alternatives rejected

- **Potential-control / near-ball PPCF model variant** — changes the surface for every consumer; needs broad
  re-validation; disproportionate to reviving one column.
- **Integration window past the reaction time** — more machinery, fuzzier meaning than "the destination."
- **Redefine `at_ball` in place** — the name would be a lie if it sampled the destination; retiring dead
  columns is the established pattern.
