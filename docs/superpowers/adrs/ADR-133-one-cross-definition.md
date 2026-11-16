# ADR-133: One cross definition for every provider; the tracking layer decides the delivery

| Field | Value |
|---|---|
| **Date** | 2026-11-13 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner decisions and review rulings, 2026-11-13 to 2026-11-15) |
| **Review record** | independent reviews A and B (`D:\Development\_reviews\2026-11-1[345]-cross-definition-*`) |
| **Supersedes / amends** | supersedes in part ADR-127 (the Sportec `play_height` corner discriminator decides only `cross_native` now); amends ADR-046 (`cross_blocked` is keyed to the provider's own cross) |
| **Source spec** | `docs/superpowers/specs/2026-11-13-cross-definition-design.md` |

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
4. **The tracking layer decides the delivery** (TF-86 `add_cross_deliveries`): `cross_attempt` (True when the ball
   path enters the penalty area; `<NA>`, cut out, when it does not, an opponent's first contact comes within 1.0 s of
   the release, and the initial direction points into the box or cannot be resolved; False otherwise; decision 10)
   and `cross_height`
   (aerial above the ADR-125 1.8 m threshold; honest `<NA>` unless the ball z is measured). The release is aligned
   through ELASTIC (TF-43), taken at the nearest ball frame, and checked against the action start; the path ends at
   the first ball contact by another player (ADR-124/125). An unanswerable path, or an unknown action start, is
   `<NA>`, never a guess. It covers open play and set pieces, one match per call.
5. **The xCross label is version 2**: a same-team `cross_attempt` in the horizon, all heights; unknown horizons
   (an unanswerable or cut-out attempt) leave training, and a causal spell with an unknown treatment leaves the
   opportunity table. Model metadata records `xcross_label_version`; causal opportunities select by it.
6. **TF-46 `corner_kind`** reads `cross_attempt` when present (`corner_kind_source`: tracking, event, gi).
7. **Event-only limits, stated**: without tracking, a blocked cross and a delivery followed by a restart are `pass`
   (a restart-ended set piece `*_short`), and SkillCorner events miss defenders' in-box touches. The tracking layer
   covers the restart-ended delivery and the in-box touches; a cross cut out near the release stays `<NA>` there too
   (decision 10). The converter
   docstrings record each provider's measured recall.
8. **Retrains are owed**, not done in this cycle: xCross, PassCompletionModel, XSuccessModel and the TF-46 corner
   weights, and (decision 9) every artifact trained on Gradient Sports or SkillCorner actions or frame links: xShot,
   the ghost-GK and ghost-outfield models, and both xT-GK retention variants
   (`docs/research/cross-definition/caller-sweep.md`). Part Deux runs them after the release.
9. **Native event timing (owner ruling 2026-11-14).** Where a provider times its events from its own tracking
   clock, the SPADL row carries that time, and the release uses it instead of ELASTIC:
   - SkillCorner: a native row is the on-ball action at the possession END (`time_end`, `x_end`/`y_end`), not the
     reception at its start; start-type derived rows and keeper saves stay at the start.
   - Gradient Sports: `time_seconds` is the possession event's `event_time` minus its period's kickoff (PFF's own
     frame sync within 0.067 s), period-relative (ADR-017); the converter sorts by it (ADR-065).
   - A per-row provenance column `time_source` (`"tracking"` / `"event"`, ADR-054): GS, SkillCorner and Metrica are
     `"tracking"`; synthetic dribbles and every other provider are `"event"`. `add_cross_deliveries` releases a
     `"tracking"` row at its own time (step 0).
   - ELASTIC splits episodes by the ADR-130 `stoppage_state` when the frames carry it.
   - ELASTIC's score does not change: a location term lowered IDSSE ground-truth accuracy at every weight.
10. **A cross cut out by an opponent is unknown (owner rulings 2026-11-15).** Against PFF's code B on
    847 GS open-play crosses, the tracking block call had precision 0.14 at a base rate of 0.20, and no radius or
    contact-strength gate gave it signal (spec section 5.2; probes `D:\Temp\cross-def\block_probe*.py`). The rule:
    no box entry, an opponent's first contact within `cut_out_window_s` (1.0 s, fixed), and an initial direction into
    the box or unresolved -> `cross_attempt` `<NA>`. `block_radius_m` and its calibration are removed.

## Evidence

The measured values come from `scripts/build_cross_definition.py`, run on a clean committed tree. The cycle's last
commit records them here and in `docs/research/cross-definition/`. The spec's audits (§5) used approximate path ends
and are not cited as results.

The first runs, on the commit that introduced this ADR, showed that the first calibration rules measured alignment
and detection error: they gave a 15 m release match and a 20 m block radius. Spec r7 §7.4 replaced the rules (the
release match from the IDSSE clean-sync reference, the block sample limited to contacts within the fixed 1.0 s
block window) and widened the start-line candidates to 80 m. The recorded values come from the r7 rules.

Run 2 under the r7 rules fired the release-match owner trigger: the IDSSE match (4.67 m) turned 28.2 % of Gradient
Sports and 31.7 % of SkillCorner ELASTIC-confident rows into `<NA>`. The root cause was the event timing, not the
match (decision 9; `docs/research/cross-definition/native-timing.md`): SkillCorner placed each action at the
reception, and Gradient Sports timed it with a whole-second clock at the start of the on-ball sequence. At the native
times, 1.2 % (GS) and 4.3 % (SkillCorner) of rows lie beyond 4.67 m. The values are recorded from the four runs on
the commit that carries decision 9, over a corpus re-staged with it.

## Consequences

- SPADL values change for every provider: `cross` / `pass` counts, `corner_*` / `freekick_*` typing and the new
  `cross_native` column. Consumers that pin column sets or set-piece counts must adapt (CHANGELOG Hyrum notice).
- `compute_bravery` refuses actions without `cross_native` (owner-accepted loud break): SPADL produced before this
  release must be re-converted.
- The xCross label needs `cross_attempt`; `build_xcross_labels` refuses without it, and the staged corpora are
  restaged through `add_cross_deliveries` before the refit.
- SkillCorner event-only corner typing becomes worse (its events miss in-box defensive touches); tracking decides it.
- Native timing: Gradient Sports and SkillCorner actions move in time and (SkillCorner) in space, so every time-based frame link
  on them moves to the action's own moment, and their `action_id` renumbers (an ADR-065 cross-table migration). The
  GS converter needs `event_time`, `start_time` and the kickoff rows; SkillCorner needs `time_end`, `x_end`, `y_end`
  (`docs/PRIVATE_CONSUMERS.md`).
