# Cross definition (ADR-133): native event timing

This memo holds the evidence for spec r8 (decision 11, §5.1 and §6.6) and the plan r5 Task 12 measurements. The measurement scripts are read-only and stay off-repo with the author, together with their outputs. The numbers come from the staged xshot187 corpus (`run_commit` `0cc07e6c`) and the pining cache, before the r8 code. `native-timing-metrics.json` next to this memo holds them in machine-readable form.

## 1. Why the release was misaligned

Run 2 on commit 2 (`b12c0fd9`, line 75) fired the release-match owner trigger: the IDSSE match (4.67 m) turned 28.2 % of GS and 31.7 % of SkillCorner ELASTIC-confident rows into `<NA>`. The tail has two causes (spec §5.1):

- **SkillCorner.** The converter places each `player_possession` row at the possession start (the reception). The kick is at the possession end. ELASTIC finds the kick: its release ball lies 0.85 m (median) from the raw kick point, 93.7 % within 4.67 m (`sc_kick_check.py`, 98 matches, 10,315 rows).
- **Gradient Sports.** `time_seconds` comes from `startGameClock`, a whole-second clock at the start of the on-ball sequence. PFF's own synchronisation (the tracking frames name the possession event they start) is ground truth. The events-only time `eventTime − kickoff startTime` matches it within 0.067 s on 86,067 possession events (`gs_truth_check.py`). ELASTIC lands within 0.5 s of it for 41.6 % of all 71,623 pass and cross rows (`r8_checks.py`).

## 2. Task 12 measurements

| Step | Measure | Result | Script |
|---|---|---|---|
| 1 | Metrica: `start.time` against `start.frame / 25` on `tests/datasets/kloppy/metrica_events.json` (the tracking fixture holds only a generator, so the frame-rate fallback applies) | 3,620 of 3,620 events, maximum deviation 0.0 s. Metrica emits `"tracking"`. | inline (ledger) |
| 2 | GS converted events before their period's kickoff | 0 in 64 matches | `gs_prekick_clock.py` |
| 2 | GS `periodGameClockTime − periodElapsedTime` against 0/2700/5400/6300 s, every match and period | within 0.43 frames; none above one frame | `gs_prekick_clock.py` |
| 3 | Frame-link shift (`link_actions_to_frames`, tolerance 0.2 s), today against r8 | the table below | `frame_link_shift.py` |

### The frame-link shift (Step 3)

| Provider | Native actions | Linked today / r8 | Linked-frame shift q10 / q50 / q90 | Shift above 1 s | Ball to action start q50 / q90, today → r8 |
|---|---|---|---|---|---|
| GS (62 matches; game events with one record) | 30,048 | 86.2 % / 87.8 % | 0.1 / 0.5 / 0.93 s | 4.1 % | 2.46 / 8.06 m → 0.0 / 0.22 m |
| SkillCorner (98 matches) | 119,765 | 100 % / 100 % | 0.0 / 1.1 / 3.7 s | 52.1 % | 1.15 / 2.72 m → 1.49 / 2.81 m |

- **GS.** The links move a median 0.5 s later, and the ball at the linked frame moves onto the event location. The GS rows are restricted to game events with one record: the staged actions key a row by `gameEventId` only, so a multi-record game event cannot be mapped to its own `eventTime` here. The converter itself uses each record's own time.
- **SkillCorner.** Half of all links move by more than 1 s, from the reception to the kick. The ball-to-start distance does not improve, because today's start is the reception point at the reception time: the link is self-consistent but describes the wrong moment of the action. At the kick time the ball has left the foot by about 0.2 s (spec §5.1).

### The consumers (Step 4)

Every module that links actions to frames by time reads the new times. 38 files mention `link_actions_to_frames` (the definer `tracking/utils.py`, its re-export and docs among them; `git grep -l link_actions_to_frames` reproduces the list). The frame-aware `add_*` functions in the linking modules are: `add_action_context`, `add_actor_pre_window`, `add_cover_shadows`, `add_das`, `add_defensive_credit`, `add_defensive_line`, `add_elastic_sync`, `add_ghost_gk`, `add_gk_influence`, `add_line_break`, `add_obso`, `add_off_ball_context`, `add_off_ball_run_values`, `add_off_ball_runs`, `add_packing`, `add_pausa`, `add_pitch_control`, `add_player_influence`, `add_pre_shot_gk_angle`, `add_pre_shot_gk_position`, `add_press_commitment`, `add_pressure_on_actor`, `add_shape_graph`, `add_shot_goalmouth`, `add_space_creation`, `add_structural_pass`, `add_sync_score`, `add_team_shape`, `add_visible_area_coverage`, `add_xcross_attempt`, `add_xshot_occurrence`. For GS and SkillCorner actions, each of them now reads the frame at the action's own moment. The caller sweep (seams 5–8) classifies them.

### The SkillCorner end of an untargeted pass (spec r9, review B CD-SB-23)

Under r8 a SkillCorner pass-class row without a targeted receiver ended at the next action's start, and that start is now the next possession's kick, not its reception. On the 118 cached SkillCorner matches (`sb23_measure.py`, raw `player_possession` rows in reception order) 16,753 rows take this path: 15,685 untargeted passes and 1,068 clearances, 13.4 % of all possessions. The next kick lies a median 0.88 m and a P95 15.6 m from the next reception; on 49.1 % of the rows more than 1 m. Spec r9 ends these rows at the next reception, and the receiver's carry to the kick becomes a synthetic dribble.

## 3. What this does not measure

- The value change of each frame-aware feature: the shift says which frame each feature reads, not how much its value moves. The owed retrains (caller sweep) carry that.
- GS game events with more than one record (Step 3 table): their per-record times are tested in the converter, not measured here.
