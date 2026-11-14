# ADR-133: One cross definition for every provider; the tracking layer decides the delivery

| Field | Value |
|---|---|
| **Date** | 2026-11-13 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner decisions 1-10 and the implementation-review rulings, 2026-11-13) |
| **Review record** | independent reviews A and B: spec r1 to r6, plan r1 to r3, implementation r1 (`D:\Development\_reviews\2026-11-13-cross-definition-*`) |
| **Supersedes / amends** | supersedes in part ADR-127 (the Sportec `play_height` corner discriminator decides only `cross_native` now); amends ADR-046 (`cross_blocked` is keyed to the provider's own cross) |
| **Source spec** | `docs/superpowers/specs/2026-11-13-cross-definition-design.md` (r6) |

## Context

The SPADL `cross` type meant something different for every provider. On the staged xshot187 corpus the open-play
crosses per match were SkillCorner 101, Gradient Sports 26.5 and IDSSE 4 (medians), while shots per match agreed
(20-25). SkillCorner typed every wide final-third pass a cross (our own zone rule); IDSSE typed only the passes
the DFL annotation called crosses. The xshot187 xCross refit failed the ADR-132 per-provider skill gates because
each provider's cross rate differed from the pooled one, not because the model ranked badly. Set-piece deliveries
were inconsistent too (`corner_crossed` per match: GS 7, IDSSE 7, SkillCorner 2).

No vendor-independent cross standard exists. The Common Data Format paper (Anzer et al. 2025) uses the cross as its
example of providers disagreeing. Vidal-Codina et al. (2022) detect crosses from tracking with a cross zone and the
next controller in the penalty area.

## Decision

1. **One geometric rule in every converter** (`spadl.base.classify_deliveries`). An open-play pass (`pass` or
   `cross`) is a `cross` when it starts wide (outside the penalty-area width), at or beyond `CROSS_START_X`, and the
   next on-ball action in the same period starts inside the penalty area (mirrored into the passer's frame for an
   opponent). A corner or free kick is `*_crossed` under the delivery condition alone, and `*_short` otherwise.
   The rule runs after `to_spadl_ltr` and before `_add_dribbles` in every converter.
2. **The provider's own label stays** in the new SPADL column `cross_native` (open play and set pieces). It is
   `<NA>` wherever the provider gave no label: every SkillCorner row (its old rule was ours), Metrica set pieces
   (positional, no crossed/short split), converter-synthesized GK rows, and a kloppy dataset whose deserializer
   emits no `PassType` qualifier.
3. **`cross_blocked` is keyed to `cross_native`** on open-play rows. Its meaning does not change; bravery reads
   `cross_native` so its rates keep their denominator.
4. **The tracking layer decides the delivery** (TF-86 `add_cross_deliveries`): `cross_attempt` (the ball path enters
   the penalty area, or an opponent blocks it within a radius of the release while it heads in) and `cross_height`
   (aerial above the ADR-125 1.8 m threshold; honest `<NA>` unless the ball z is measured). The release is aligned
   through ELASTIC (TF-43), taken at the nearest ball frame, and checked against the action start; the path ends at
   the first ball contact by another player (ADR-124/125). An unanswerable path, or an unknown action start, is
   `<NA>`, never a guess. It covers open play and set pieces, one match per call.
5. **The xCross label is version 2**: a same-team `cross_attempt` in the horizon, all heights, blocked included;
   unknown horizons leave training. Model metadata records `xcross_label_version`; causal opportunities select by it.
6. **TF-46 `corner_kind`** reads `cross_attempt` when present (`corner_kind_source`: tracking, event, gi).
7. **Event-only limits, stated**: without tracking, a blocked cross and a delivery followed by a restart are `pass`
   (a restart-ended set piece `*_short`), and SkillCorner events miss defenders' in-box touches. The converter
   docstrings record each provider's measured recall.
8. **Retrains are owed**, not done in this cycle: xCross, PassCompletionModel, XSuccessModel and the TF-46 corner
   weights (`docs/research/cross-definition/caller-sweep.md`). Part Deux runs them after the release.

## Evidence

The measured values come from `scripts/build_cross_definition.py` on the clean commit that introduces this ADR; the
second commit of the cycle records them here and in `docs/research/cross-definition/`. The spec's audits (§5) used
approximate path ends and are not cited as results.

## Consequences

- SPADL values change for every provider: `cross` / `pass` counts, `corner_*` / `freekick_*` typing and the new
  `cross_native` column. Consumers that pin column sets or set-piece counts must adapt (CHANGELOG Hyrum notice).
- `compute_bravery` refuses actions without `cross_native` (owner-accepted loud break): SPADL produced before this
  release must be re-converted.
- The xCross label needs `cross_attempt`; `build_xcross_labels` refuses without it, and the staged corpora are
  restaged through `add_cross_deliveries` before the refit.
- SkillCorner event-only corner typing becomes worse (its events miss in-box defensive touches); tracking decides it.
