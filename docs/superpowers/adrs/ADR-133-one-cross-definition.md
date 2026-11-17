# ADR-133: One open-play cross definition for every provider, at the action's own time

| Field | Value |
|---|---|
| **Date** | 2026-11-16 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner); scope and design by the supervisor directive of 2026-11-16, which the owner delegated |
| **Review record** | independent reviews A and B of this implementation, 2026-11-16, two rounds (`2026-11-16-cross-definition-open-play-impl-A.md`, `-impl-B.md`, `-impl-r2-A.md`, `-impl-r2-B.md` in the owner's shared reviews folder) |
| **Amends** | ADR-046 (`cross_blocked` is keyed to the provider's own cross); ADR-017 and ADR-065 (the GS time base) |

## Context

The SPADL `cross` type meant something different for every provider. On the staged xshot187 corpus (187 matches)
the open-play crosses per match were SkillCorner 101, Gradient Sports 26.5 and IDSSE 4 (medians), while shots per
match agreed (20-25). SkillCorner typed every wide final-third pass a cross (a zone rule in our converter); IDSSE
typed only the passes the DFL annotation called crosses. A model trained across providers, such as xCross, then
learned each provider's labelling, not the football: the xshot187 xCross refit failed the ADR-132 per-provider
skill gates.

## Decision

1. **One geometric rule for open play.** `spadl.base.classify_deliveries` types an open-play pass-class row
   (`pass` or `cross`) a `cross` when it starts wide (outside the penalty-area width), at or beyond
   `CROSS_START_X`, and the next on-ball action in the same period starts inside the penalty area (mirrored into
   the passer's frame when the team differs). Otherwise the row is a `pass`. The next on-ball action skips
   `non_action` and `foul`; a restart as the next action never delivers; a row whose own or next team is
   missing, or whose coordinates are NaN, never delivers.
2. **Every converter calls it in one place:** after `to_spadl_ltr`, before `_add_dribbles`. The kloppy and
   SkillCorner converters move `to_spadl_ltr` ahead of `_add_dribbles` for this. SkillCorner's zone rule
   (`_is_cross`) is deleted.
3. **The provider's own label is kept** in the new SPADL column `cross_native` (nullable boolean): True or False
   on open-play pass-class rows and on corner and free-kick rows where the provider labels deliveries, `<NA>`
   elsewhere and on providers with no label (SkillCorner). It is opt-in for atomic SPADL (`preserve_native`).
   Set pieces are not re-typed, so on corner and free-kick rows the column equals the type in this cycle.
4. **`cross_blocked` and bravery read `cross_native`.** `cross_blocked` applies where `cross_native` is True on
   a pass-class row; TF-51 bravery counts an open-play cross faced by `cross_native`. So on a provider with its
   own label the block rate keeps its meaning and its denominator when the type changes. In a game with no known
   open-play label (SkillCorner, a kloppy dataset without `PassType`) the open-play count is `<NA>`, never 0
   (ADR-027). Set-piece bravery still counts `corner_crossed`/`freekick_crossed` by type and does not change.
5. **`CROSS_START_X`** comes from a pre-declared rule: among {52.5, 60, 65, 70, 75, 80}, the line with the
   highest equal-weight mean F1 against the native label of StatsBomb (open data) and Gradient Sports; a tie
   within 0.005 goes to the larger line (`scripts/build_cross_definition.py`).
6. **The action sits at its own moment (native event timing).** A SkillCorner native row is the on-ball action at
   the possession end: `time_end` and `x_end`/`y_end` (required inputs), not the reception; a pass without a
   targeted receiver ends at the next possession's reception. A Gradient Sports row is timed by its possession
   event's `event_time` minus its period's kickoff `start_time` (required inputs), not the whole-second
   `startGameClock`, and the converter sorts by that time. The pining loader's GS frames are period-relative
   (`periodElapsedTime`), the clock the actions use.

**Why decision 6 belongs to the cross definition.** A label that reads the frames before an action (the xCross
label: a positive frame is one within 1.0 s before a cross) depends on the action's time, not only its type. On
`main` a GS action sits a median 1.29 s before its possession event (the native converted actions of the 64
WC2022 matches) and a SkillCorner action at the reception, so the label
marks frames before the crosser has the ball, and how many depends on the provider. The share of training frames
with a positive label (`tracking._xcross_attempt.prepare_xcross_training_data` at commit `2a8e5ef`, label version
1, horizon 1.0 s, 187 matches; a read-only measurement outside the repository: rows 1 and 2 on the xshot187 stage
made by the `main` converters, row 3 on the stage made from this release's code commit `f5fc2e2b`, with its staged
types as they are):

| Converters | GS | IDSSE | SkillCorner | largest / smallest |
|---|---|---|---|---|
| `main`, vendor labels | 2.27 % | 0.61 % | 7.61 % | 12.5 |
| `main` + the open-play rule | 1.41 % | 3.81 % | 2.68 % | 2.70 |
| native timing + the open-play rule | 3.76 % | 3.81 % | 4.63 % | 1.23 |

Positive frames per cross: GS 6.4 without native timing and 17.5 with it; SkillCorner 5.3 and 9.0; IDSSE 14.7 in
both.

## Alternatives considered

| Option | Result on xshot187 | Why not |
|---|---|---|
| Keep the vendor labels | provider medians 4 to 101, ratio about 25 | The type then measures the vendor, not the play |
| A row-wise rule on the row's own `end_x`, `end_y` in the box | ratio 1.64 (2.00 with native timing) | Fails the 1.5 bar: a provider's end point is its own convention |
| A tracking label (the ball path decides) | ratio 1.14; F1 against the GS label 0.523, against 0.448 for the event rule | No material gain in provider consistency, and it makes the label depend on tracking data |
| **The event rule on the next on-ball action (chosen)** | medians GS 14, IDSSE 16, SkillCorner 17; ratio 1.21 | Meets the 1.5 bar with event data alone |
| The rule without native timing | crosses per match agree (1.21), label rate ratio 2.70 | The type agrees, but the label marks the wrong frames, differently per provider |

## Evidence

`docs/research/cross-definition/` holds the start-line measurement (`cross_definition.json`: the F1 table per
candidate line, the selected line 75.0, the per-provider medians and the ratio bar) and its memo `start-line.md`,
run on xshot187 staged from this release's converters and the pinned StatsBomb open-data snapshot; and the
native-timing evidence
(`native-timing.md`, `native-timing-metrics.json`: GS events-only time equal to PFF's frame sync within 0.067 s
on 86,067 possession events; the SkillCorner kick a median 0.86 m from the raw end point).

## Consequences

- Open-play `cross` and `pass` counts change for every provider; the set-piece types do not. A reader that
  needs the provider's own label reads `cross_native` (a release notice names every change; Hyrum).
- GS and SkillCorner action times, SkillCorner action starts and both providers' `action_id` order change, so
  every time-based frame link on them reads a new frame (an ADR-065 cross-table migration for stored tables).
- Retrains owed: xCross, PassCompletionModel and XSuccessModel (they read the type); xShot, the ghost-GK and
  ghost-outfield models, both xT-GK retention variants and the TF-46 corner models (they train on GS or
  SkillCorner action times or links). `docs/research/cross-definition/caller-sweep.md` lists the readers.
- Atomic SPADL drops `cross_native` unless `preserve_native` asks for it.

## Limits

- The rule finds about 35 % of PFF's crosses on Gradient Sports (precision 0.63): many PFF crosses do not reach
  the box in the event stream.
- A blocked cross and a delivery that ends in a restart are typed `pass`: the event data has no delivery end.
- Set-piece types stay provider-dependent in this release: the rule re-types open play only.
- The SkillCorner label rate stays 1.23 times the GS rate with native timing; the cause is not found.
- Whether the ADR-132 per-provider gates pass needs the refit; this release does not show it.
