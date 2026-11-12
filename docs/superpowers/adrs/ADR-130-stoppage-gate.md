# ADR-130: The stoppage gate for xShot/xCross training, and the refit campaign mode

| Field | Value |
|---|---|
| **Date** | 2026-11-11 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner rulings 2026-10-26 and 2026-11-11); reviewers A and B (spec r1 to r4, plan r1 and r2) |

## Context

The xShot and xCross models train on tracking frames. A frame where the ball is out of play must not be a training
row: no shot or cross can start there. The training preparers dropped frames with `ball_state != "alive"`. Only
IDSSE/Sportec frames record the dead ball. SkillCorner and Gradient Sports frames carry `"alive"` on every frame
(`silly_kicks/tracking/skillcorner.py`; the Gradient Sports loader in `scripts/_loader_pining.py`). SkillCorner is 108
of 115 matches of `xshot115` and 10 of 17 of `xshot17`; `xcross_probe` (2 Gradient Sports matches) gives the xCross
trainer a gated statistic. So both models trained, and xCross was checked, on dead-ball frames.

An earlier design (TF-46 rule version 3, 2026-10-26/27) derived the dead-ball spans through about 970 new converter
lines (SkillCorner restart and goal rows) and a kick-time resolver. It never merged. The owner restarted the work on
2026-11-11: minimal and clean, on main.

## Decision

1. **`silly_kicks/tracking/_stoppages.py`** (private) marks each frame `live`, `dead` or `unknown` in two new columns,
   `stoppage_state` and `stoppage_source`; it never writes `ball_state`. `STOPPAGE_RULE_VERSION = 1`. The rules, the
   most restrictive one wins (`unknown` over `dead` over `live`):
   - SkillCorner: the `game_interruption_after` / `game_interruption_before` markers of consecutive possessions, by
     frame id: a marker pair makes the gap `dead` and the taker's possession `unknown` (the kick is inside it); one
     marker alone makes the gap and the next possession `unknown`.
   - Gradient Sports: a native restart action makes `[previous action, restart)` `dead`; a goal makes
     `[goal, next action)` `dead`.
   - Every provider whose frames do not record the dead ball: a ball beyond a line by more than 0.11 m (the ball
     radius, Law 9) is `dead`, and so is every frame up to the next on-ball event, whose own span is `unknown`. Frames
     of a period before its first and after its last on-ball event are `dead` (the period is the frames' own field);
     a period with no on-ball event is `unknown`.
   - IDSSE and Sportec: the state comes from `ball_state`.
   The claim: no frame with a tracked ball out of play is `live`. A stop with the ball inside the pitch (a foul) or not
   tracked is found only by the event rules; the rate at which the event rules miss a tracked ball-out estimates that
   rest. Measured through the loader on 20 SkillCorner matches of `xshot115` and the 2 Gradient Sports probe matches
   (1,165,636 frames): of 73,190 frames with the ball out, none is `live`, and 3.9 % were caught only by the
   ball-out rule. The cost per SkillCorner match: `dead` 0.18 to 0.37, `unknown` 0.02 to 0.04 of the frames;
   per Gradient Sports match: `dead` 0.39 and 0.40, `unknown` 0.03.
2. **Gradient Sports leaves `_DEAD_BALL_OBSERVED_PROVIDERS`** (owner rulings 2026-10-26 and 2026-11-11 (a)). TF-58
   `resolve_stoppages` therefore takes the event intervals for a Gradient Sports match in `auto` mode, in place of a
   `ball_state` that split nothing, and `mode="ball_state"` raises for it. The committed TF-58 artifacts that hold
   Gradient Sports matches are stale; a `TODO.md` row records them as owed a re-run.
3. **Staging.** `scripts/_loader_pining.py` adds the two columns to every tracking provider's frames;
   `scripts/materialize_tc3_frames.py` moves the shard schema token to `tc3-frames-4`. `xshot115`, `xshot17` and
   `xcross_probe` are re-staged; `tc3` and `sb360` keep their pins.
4. **Training.** `prepare_xshot_training_data` and `prepare_xcross_training_data` keep a frame only where
   `stoppage_state == "live"` and no `unknown` frame lies within its label horizon (`training_frame_mask`); without the
   column they behave as before. Both trainers fail closed on a SkillCorner or Gradient Sports match without the
   column (extract and probe stages), and record `stoppage` (the rule version and per-provider counts) in
   `metrics.json` and as `stoppage_provenance` in the bundle's `metadata.json` (written only when set, so an older
   bundle re-saves byte-identical).
5. **`Campaign.mode`** (`rematerialize` default, `refit`). In `refit` mode `verify` skips the `_integrity`
   "reproduces" check and passes each arm on its own acceptance gates (the trainers' `acceptance` block); the
   comparison with the committed model is run and reported, not gated (owner ruling 2026-11-11).
   `scripts/hf_jobs/campaigns/tf46_refit.toml` holds the 8 xShot/xCross instances in `refit` mode.

## Consequences

- The refit runs in the Part Deux session's launch 9, which also adopts the weights and publishes the Hub models
  (owner ruling 2026-11-11). `cycle1.toml` stays whole: after the adoption, its full parity launch must reproduce the
  adopted xShot/xCross weights on the re-staged data (owner ruling "Option 1", 2026-11-11).
- A frame table without the stoppage columns trains as before, for IDSSE/Sportec only; for SkillCorner and Gradient
  Sports the trainers refuse it.
- Not done here: the converter recut, a kick-time resolver, the other tracking corrections of TF-46, the corner
  drivers, the research regeneration.
