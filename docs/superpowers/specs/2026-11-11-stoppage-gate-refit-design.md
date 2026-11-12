# Stoppage gate and xShot/xCross refit — design

Status: r8 (2026-11-11; the build's names in 4.1, 4.5 and 5; r4 plus section 6 corrected for the owner's launch order), r4 APPROVED by the owner ("approved, proceed"), with the split of section 6. r3: r2 was approved by A and B with SHOULD FIX items (A: SGR-SPEC-08, 09; B: SGR-07, 08);
r3 answers them and carries the owner's ruling (a) on TF-58. r1 was reviewed by A (REQUEST CHANGES: SGR-SPEC-01 to
07) and B (REQUEST CHANGES: SGR-01 to 06). r2 carries two owner rulings of the same day (Gradient Sports included; refit certification by the fresh
acceptance gates) and answers every finding (table at the end). Branch `tf-46-arm2-clean` on `origin/main`
`2a0cb92b`. It replaces the withdrawn re-cut specs of 2026-11-10 for this cycle.

## 1. Intent

The xShot and xCross models train on tracking frames. A frame where the ball is out of play must not be a training
row. Main drops frames with `ball_state != "alive"`. Only IDSSE/Sportec frames record the dead ball. SkillCorner and
Gradient Sports frames do not: their `ball_state` is always `"alive"` (`silly_kicks/tracking/skillcorner.py`;
`scripts/_loader_pining.py:1318`). Main still lists Gradient Sports as observed; the owner ruled on 2026-10-26 that
it is not (TF-46 Part 2, section 27.15, G4).

SkillCorner is 108 of 115 matches of `xshot115` and 10 of 17 of `xshot17`. `xcross_probe` (2 Gradient Sports
matches) gives the xCross trainer a gated statistic. So today both models train, and xCross is checked, on dead-ball
frames.

This cycle marks the dead-ball frames of SkillCorner and Gradient Sports, keeps them out of training, and refits
both models on HF. Owner scope (2026-11-11): the stoppage gate, the data it needs, the label exclusion, the refit;
Gradient Sports included. Minimal and clean, written new on main. Not in this cycle: the rest of the converter
recut, the other tracking corrections, the research regeneration, the corner-driver migration, the Databricks
severing.

## 2. The rule

Each frame gets `stoppage_state` in {`live`, `dead`, `unknown`}. `dead` and `unknown` frames are not training rows.
The aim: no dead frame is `live`. The cost is live frames marked `dead` or `unknown`.

**All providers whose frames do not record the dead ball (SkillCorner, Gradient Sports):**

1. **Ball out (Law 9).** A frame whose ball is beyond a touch line or goal line by more than `BALL_OUT_MARGIN_M =
   0.11` (the ball radius: the whole ball has crossed the line) is `dead`. A smaller margin is the safe side: noise
   along a line makes a live frame `dead`, which costs one training row. From that frame, every frame up to the next on-ball event
   (SkillCorner: the next possession's `frame_start`; Gradient Sports: the next action's time) is `dead`, and that
   next event's own span (SkillCorner: the possession; Gradient Sports: up to the following action) is `unknown`,
   because the restart lies inside it. The test reads the canonical frames, which are in SPADL coordinates on a
   105 x 68 m pitch for both providers (`silly_kicks/tracking/skillcorner.py` scales by the match pitch size;
   `silly_kicks/tracking/gradientsports.py:97`): out is `x < -m`, `x > 105 + m`, `y < -m` or `y > 68 + m`. No
   separate pitch-size helper is needed. A frame without a ball position takes no state from this rule.
2. **Period edges.** The frames of a period before its first on-ball event and after its last are `dead`. The period
   is the tracking's own period field, never the event label (7 of 895 SkillCorner matches carry a wrong period
   label on the second-half kick-off; TF-46 Part 2, section 26).
3. **A frame with no frame id match or no event coverage** is `unknown`.

**SkillCorner, from the event markers.** `player_possession` events carry `game_interruption_after` (play stops
after this possession) and `game_interruption_before` (this possession is the restart), and the frame ids
`frame_start` and `frame_end` of the tracking. Possessions are paired by frame order:

| Possession `a`, next possession `b` | Frames | State |
|---|---|---|
| `a` has `after`, `b` has `before` | `[a.frame_end, b.frame_start)` | `dead` |
| | `[b.frame_start, b.frame_end]` (the taker: the kick is inside) | `unknown` |
| only one of the two has its marker | `[a.frame_end, b.frame_end]` | `unknown` |

**Gradient Sports, from its events.** Gradient Sports emits native restart actions with their own time. For each
restart action `r` (`throw_in`, `freekick_crossed`, `freekick_short`, `corner_crossed`, `corner_short`, `goalkick`,
`shot_penalty`, `shot_freekick`) the frames in `[previous action time, r.time)` are `dead`. For each goal (a shot
type with result `success`, or result `owngoal`) the frames in `[goal time, next action time)` are `dead`. Times
are period-relative on both sides (ADR-017).

**IDSSE and Sportec** keep `ball_state`: `dead` where it is `"dead"`, else `live`.

A frame takes the most restrictive state of all rules (`unknown` over `dead` over `live`).

**What the rules cannot see.** The claim is: no frame with a tracked ball out of play is `live`. A stop with the
ball inside the pitch (a foul) or not tracked (about a quarter of SkillCorner frames) is found only by the event
rules. When a marker is late (the measurement found `a.frame_end` after the ball left) or absent, those frames stay
`live` unless the ball is out. The rate at which the markers miss a tracked ball-out estimates that rest. Rule
version 3 had the same limit.

**Measured (2 SkillCorner matches, 119,324 frames; `D:\Temp\tf46-recut\port\sc_rule_check.py`).** With the ball out
by more than 0.11 m: 9,992 frames; the marker rule alone left 139 `live`, in 15 runs (1.4 %). With 0.5 m: 8,231 and
94 in 12 runs. The misses are mostly the last frames of a possession that ends in a throw-in (the ball is out
before `frame_end`), and some after a shot or a pass with no marker. Rule 1 marks all of them `dead`.

**The cost.** In those 2 matches the `dead` share is 0.41 and 0.51, the `unknown` share 0.12 and 0.02. So about half
of the SkillCorner frames leave training. A match has about 55 to 60 minutes of play in 90 minutes, plus the
frames before and after each half.

## 3. Differences from the owner's rule version 3 (2026-10-26/27)

| Rule version 3 | This design | Effect |
|---|---|---|
| SkillCorner restart and goal rows made by the converter (`_skillcorner_restarts`, `_skillcorner_goals`, about 970 lines) | The raw markers, read at staging; the converter does not change | Fewer lines; the actions table does not change |
| A resolved kick time ends a SkillCorner span | No kick resolver: the taker's possession is `unknown` | More live frames lost; never a dead frame `live` |
| The ball-out run starts a span inside `(previous action, restart)` | A ball-out frame is `dead` anywhere, up to the next on-ball event | Also catches stoppages that no marker records |
| A goal rule and a penalty rule by name | A goal is a marker pair (`goal_for`, then `goal_against`), or a Gradient Sports goal action; a penalty is a marker pair or a native `shot_penalty` | A scored penalty that GI does not record is caught only by rule 1 or rule 2 |
| A reader without `stoppage_state` raises | The trainers fail closed on SkillCorner and Gradient Sports without it; other readers keep main's behaviour | Outside the two trainers only TF-58 changes (next row) |
| Gradient Sports out of `_DEAD_BALL_OBSERVED_PROVIDERS` | The same (owner ruling (a), 2026-11-11: apply it everywhere) | `silly_kicks/coordination/_windows.py:526` (`resolve_stoppages`, TF-58, the one other reader): in `auto` mode a Gradient Sports match takes the event intervals in place of `ball_state`, which is constant `"alive"` there and so split nothing; `mode="ball_state"` now raises for it. The committed TF-58 artifacts that hold Gradient Sports matches go stale: a `TODO.md` row records them as owed a re-run (owner yes, 2026-11-11); this cycle does not re-run them. `tests/tracking/test_dead_ball_taxonomy.py:15` changes with the set |
| Old bundles refused at `load()` until the refit | No refusal; the refit replaces them in this cycle | No refusal fixtures |

## 4. Design

1. **`silly_kicks/tracking/_stoppages.py` (new).** Pure functions; `STOPPAGE_RULE_VERSION = 1`:
   - `skillcorner_stoppage_spans(events) -> StoppageSpans` and `event_stoppage_spans(actions) -> StoppageSpans`
     (Gradient Sports): the spans and the on-ball events that rules 1 and 2 need (r8: the names of the build);
   - `with_stoppage_state(frames, *, spans) -> DataFrame` (private, not exported, so no `add_*` registry takes it
     in): adds `stoppage_state` and
     `stoppage_source` (`events`, `ball_out`, `period_edge`, `no_coverage`, `ball_state`). It never writes `ball_state` (ruling R-C2).
2. **`silly_kicks/tracking/_provider_visibility.py`:** Gradient Sports leaves `_DEAD_BALL_OBSERVED_PROVIDERS`
   (owner rulings 2026-10-26 and 2026-11-11 (a)). Its readers on main: `coordination/_windows.py:526` (TF-58; see
   section 3) and `tests/tracking/test_dead_ball_taxonomy.py`. The `TODO.md` row for the stale TF-58 artifacts is
   written in C1.
3. **Staging.** `scripts/_loader_pining.py` calls the functions for every match (it holds the raw SkillCorner events
   and the match metadata there). `scripts/materialize_tc3_frames.py`: `_SHARD_SCHEMA_VERSION = "tc3-frames-4"`. Three
   runs on the DGX through pining, each with `--match-ids-json` built from the current lock's unit list: `xshot115`
   (108 SkillCorner + 7 IDSSE), `xshot17` (10 + 7), and the 2 Gradient Sports probe matches (canonical frames,
   SPADL coordinates, for all three), then
   `scripts/hf_jobs/repackage.py` `flat_to_pergame` for `xcross_probe`. Before `datasets record`, each staged unit
   list must equal the current lock's units for that dataset; a difference stops the staging. `tc3` and `sb360` keep
   their pins.
4. **Training.** `prepare_xshot_training_data` and `prepare_xcross_training_data` keep a frame only where
   `stoppage_state == "live"` when the column exists, else main's `ball_state` rule. A frame whose label horizon
   `[t, t + horizon]` reaches an `unknown` frame is dropped; one that reaches a `dead` frame is kept (no shot or
   cross starts in dead time). Both trainers' extract stages fail closed on a SkillCorner or Gradient Sports match
   without `stoppage_state`. `metrics.json` records, per provider, the frames dropped as `dead`, as `unknown`, and by
   the horizon, and the labels inside dropped frames.
5. **Provenance.** `metrics.json` (`stoppage`) and the model `metadata.json` (`stoppage_provenance`) record the
   rule version and the
   `stoppage_source` counts per provider.
6. **Refit.** `Campaign.mode` (`rematerialize` default, `refit`) in `scripts/hf_jobs/campaign.py`. In `refit` mode
   `verify` (owner ruling 2026-11-11): no `_integrity` "reproduces" check; each arm passes on its own acceptance
   gates (the `acceptance` block that the trainers write); the head-to-head or CV comparison with the committed model
   is run and reported, not gated. One campaign file, `scripts/hf_jobs/campaigns/tf46_refit.toml`: xShot and xCross,
   tiers `public` and `sc_extended`, variants `default` and `position_only`.
7. **A failed acceptance gate is a stop.** The session reports to the owner. No pre-built withdrawal path.
8. **Records.** `ADR-130` (the stoppage gate and the refit; the number is free on main), a `CHANGELOG.md` entry, the
   model cards.

## 5. Tests (written first)

- `skillcorner_stoppage_spans`: each row of the table; an unpaired marker on each side; an empty table.
- `event_stoppage_spans`: each restart type; a goal; an own goal; the first and the last action of a period.
- `with_stoppage_state`: rule 1 on a SkillCorner and on a Gradient Sports frame table (a ball-out frame inside a live
  span, and the frames after it; a ball 0.10 m and 0.12 m beyond each line); rule 2 with a wrong event period label;
  rule 3; the precedence; `ball_state` is never changed (purity gate, ADR-033).
- TF-58: `resolve_stoppages` on a Gradient Sports match in `auto` mode returns the event evidence; `mode="ball_state"`
  raises.
- `_DEAD_BALL_OBSERVED_PROVIDERS` holds `sportec` and `idsse` only.
- Both `prepare_*` functions: `dead`, `unknown` and horizon drops; without the column the output is byte-identical
  to main.
- Each trainer's extract fails on a SkillCorner or Gradient Sports match without `stoppage_state`.
- `load_campaign` refuses an unknown mode; `verify` in `refit` mode runs no `_integrity` and passes or fails on the
  acceptance gates alone.
- Before staging, on the DGX (read-only): the shipped functions on 20 SkillCorner matches and the 2 Gradient Sports
  probe matches. It must report zero `live` frames with the ball out by more than 0.11 m (a failure is a code
  defect). It reports the `dead`, `unknown` and `live` shares, and the ball-out frames that the event rules alone
  missed (the marker miss rate of section 2), for the owner.

## 6. Operations, commits and the split with Part Deux (owner ruling 2026-11-11)

This cycle ends at the merge of one PR with two commits. The refit runs in the Part Deux session's launch 9.

- **C1 (this cycle):** code, tests, `scripts/hf_jobs/campaigns/tf46_refit.toml`, the ADR, the docs, the `TODO.md` row
  of section 3. `cycle1.toml` is not changed (r7; owner ruling to the Part Deux session, "Option 1", 2026-11-11:
  it stays whole). Before it: the final review, the A/B implementation review, the diff shown to the owner, his yes.
- **Staging (this cycle, after C1 is pushed; each step with the owner's yes):** the three materializer runs on the
  DGX at the pushed C1, the uploads (about 6.4 GB of new revisions; room under the storage rule today: 22.7 GB),
  `datasets record`.
- **C2 (this cycle):** the dataset lock. Then the merge by a merge commit, with the owner's yes.
- **Launch 9 (Part Deux, after the merge; owner ruling "Option 1", 2026-11-11), in this order, each step with the
  owner's yes:** (1) one driver run of `tf46_refit.toml` in `refit` mode, then `retrieve`, `verify` and the adoption
  of the refit weights in a commit of the Part Deux cycle (C3; a failed acceptance gate stops at the owner);
  (2) then one driver run of `cycle1.toml`, unchanged, at that commit, as a full parity launch: xShot and xCross
  must reproduce the adopted refit weights on the re-staged data, every other artifact must equal launch 7;
  (3) the launch record, the Hub publish of the `sc_extended` models and the release. The owner's `TODO.md` row on
  the launch-8 timings gates the launches.
- **The handover note** (this cycle, with C1): the instances, the gates, the expected differences, and what a failed
  gate means.

## 7. Findings of the r1 reviews

| Finding | Answer |
|---|---|
| A SGR-SPEC-01, B SGR-03 (Gradient Sports) | Owner ruling: included (sections 1, 2, 4.2, 4.3) |
| A SGR-SPEC-02, B SGR-02 (dead frames left live; no test that can fail) | Rules 1 to 3; frame order, not the period label; the measurement; the pre-staging check (sections 2, 5) |
| A SGR-SPEC-03 (differences from version 3) | Section 3 |
| A SGR-SPEC-04 (staging path) | Section 4.3 |
| A SGR-SPEC-05, B SGR-04 (refit certification) | Owner ruling: the fresh acceptance gates (section 4.6) |
| A SGR-SPEC-06, B SGR-05 (counts of dropped rows) | Section 4.4 |
| A SGR-SPEC-07 (missing frame id) | Rule 3 |
| B SGR-01 (commit gates) | Section 6 |
| B SGR-06 (ADR, changelog) | Section 4.8 |
| A SGR-SPEC-08, B SGR-08 (TF-58 changes) | Owner ruling (a); section 3, section 4.2, section 5 |
| A SGR-SPEC-09, B SGR-09 (pitch size, coordinate frame) | Rule 1: canonical SPADL frames for both providers; section 5 tests both |
| A SGR-SPEC-11, B SGR-07 (the margin) | 0.11 m, measured; section 2 |
| A SGR-SPEC-10, SGR-SPEC-12 (the cost; the late marker) | Section 2 |
