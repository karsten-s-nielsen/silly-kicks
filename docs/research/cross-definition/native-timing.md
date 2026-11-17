# Cross definition (ADR-133): native event timing

This memo holds the evidence for the native event timing of Gradient Sports and SkillCorner (ADR-133, decision 6).
The measurement scripts are read-only and stay off-repo with the author, together with their outputs. The numbers
come from the staged xshot187 corpus (`run_commit` `0cc07e6c`) and the pining cache, before the native-timing code.
`native-timing-metrics.json` next to this memo holds them in machine-readable form (its `today` keys are the
converters on `main`, its `native` keys the native-timing converters). Its `library_commit` `b12c0fd9` is a commit of
the branch `feat/cross-definition` (draft PR #306), where the measurement ran.

## 1. Why the action time was wrong

- **SkillCorner.** The converter placed each `player_possession` row at the possession start (the reception). The
  kick is at the possession end: the ball leaves the foot a median 0.86 m from the raw end point, 93.7 % within
  4.67 m (`sc_kick_check.py`, 98 matches, 10,315 rows).
- **Gradient Sports.** `time_seconds` came from `startGameClock`, a whole-second clock at the start of the on-ball
  sequence, a median 1.59 s before the kick. PFF's own synchronisation (the tracking frames name the possession
  event they start) is ground truth. The events-only time `eventTime − kickoff startTime` matches it within
  0.067 s on 86,067 possession events (`gs_truth_check.py`).

A label that reads the frames before the action time (the xCross label) therefore marks frames before the crosser has
the ball, and the number of marked frames per cross differs by provider (ADR-133, the label-rate table).

## 2. Measurements

| Measure | Result | Script |
|---|---|---|
| GS converted events before their period's kickoff | 0 in 64 matches | `gs_prekick_clock.py` |
| GS `periodGameClockTime − periodElapsedTime` against 0/2700/5400/6300 s, every match and period | within 0.43 frames; none above one frame | `gs_prekick_clock.py` |
| Frame-link shift (`link_actions_to_frames`, tolerance 0.2 s), `main` against native timing | the table below | `frame_link_shift.py` |

| Provider | Native actions | Linked `main` / native | Linked-frame shift q10 / q50 / q90 | Shift above 1 s | Ball to action start q50 / q90, `main` → native |
|---|---|---|---|---|---|
| GS (62 matches; game events with one record) | 30,048 | 86.2 % / 87.8 % | 0.1 / 0.5 / 0.93 s | 4.1 % | 2.46 / 8.06 m → 0.0 / 0.22 m |
| SkillCorner (98 matches) | 119,765 | 100 % / 100 % | 0.0 / 1.1 / 3.7 s | 52.1 % | 1.15 / 2.72 m → 1.49 / 2.81 m |

- **GS.** The links move a median 0.5 s later, and the ball at the linked frame moves onto the event location. The
  rows are restricted to game events with one record: the staged actions key a row by `gameEventId` only, so a
  multi-record game event cannot be mapped to its own `eventTime` here. The converter uses each record's own time.
- **SkillCorner.** Half of all links move by more than 1 s, from the reception to the kick. The ball-to-start
  distance does not improve, because the old start was the reception point at the reception time: the link was
  self-consistent but described the wrong moment of the action. At the kick time the ball has left the foot by
  about 0.2 s.

Every module that links actions to frames by time reads the new times (`git grep -l link_actions_to_frames`); for
GS and SkillCorner actions each frame-aware `add_*` now reads the frame at the action's own moment. The caller
sweep (seams 5 and 6) classifies them.

## 3. The SkillCorner end of an untargeted pass

With the row at the kick, a SkillCorner pass-class row without a targeted receiver would end at the next action's
start, which is now the next possession's kick, not its reception. On the 118 cached SkillCorner matches
(`sb23_measure.py`, raw `player_possession` rows in reception order) 16,753 rows take this path: 15,685 untargeted
passes and 1,068 clearances, 13.4 % of all possessions. The next kick lies a median 0.88 m and a P95 15.6 m from the
next reception; on 49.1 % of the rows more than 1 m. These rows end at the next reception, and the receiver's carry
to the kick becomes a synthetic dribble.

## 4. What this does not measure

- The value change of each frame-aware feature: the shift says which frame each feature reads, not how much its
  value moves. The owed retrains (caller sweep) carry that.
- GS game events with more than one record (the shift table): their per-record times are tested in the converter,
  not measured here.
