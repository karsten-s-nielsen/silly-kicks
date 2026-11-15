# One cross definition for every provider — design

- **Status:** r9 (2026-11-14). The owner approved r1 and decisions 7–10. r2 answered review B round 1; r3 answered review A round 1; r4 answered CD-SB-08, CD-SPEC-08 and CD-SPEC-09; r5 answered CD-SB-09 and CD-SB-10; r6 answers CD-SPEC-10. r7 is the calibration amendment (Section 7.4, owner rulings of 2026-11-14), made after the first DGX runs on commit 1. r8 is the native-timing amendment (decision 11, Sections 5.1 and 6.6), made after run 2 on commit 2 fired the release-match owner trigger. r9 (owner ruling 2026-11-14, review B CD-SB-23) fixes the SkillCorner end rule of 6.6 A against the r8 start.
- **Branch:** `feat/cross-definition`, off `main` `0cc07e6c` (owner decision).
- **ADR:** ADR-133, written in this cycle.
- **Origin:** handoff from the Part Deux session (xshot187 refit, PR #305).

## 1. Intent and why

The SPADL `cross` type means different things for different providers. On the staged xshot187 corpus (187 matches), the open-play crosses per match are:

| Provider | Native rule today | Median crosses per match | Range |
|---|---|---|---|
| SkillCorner | our zone rule `_is_cross` (`spadl/skillcorner.py:138-154`): every wide pass in the final third | 101 | 40–235 |
| Gradient Sports | the provider's annotation (`OTB`/`CR`) | 26.5 | 12–57 |
| IDSSE (Sportec) | `play_height == "cross"` or `play_flat_cross` (`spadl/sportec.py:861-868`) | 4 | 1–5 |

Shots per match agree across the three providers (20–25), so the matches are comparable. Only the cross label differs, by about 25x.

Because of this, the xshot187 xCross refit failed the ADR-132 per-provider skill gates. The model ranks well for every provider. Its calibration fails where a provider's cross rate is far from the pooled rate (GS: Brier skill −3.2 %; IDSSE: −32.8 %).

The goal: one definition of a cross, used the same way for every provider, that does not depend on how a vendor labels its events.

## 2. What a cross is (owner's coaching definition)

- It is aimed at the penalty area. "Early service" counts: a ball from further back, played at the front of the box.
- It usually comes from wide, outside the width of the penalty area.
- It is in the air. A ground cutback is not a cross; a lifted cutback for a header is one.
- Its intent is an immediate chance, not keeping the ball, building up or switching play.

## 3. Prior art (summary)

- No vendor-independent standard exists. The Common Data Format paper (Anzer et al., arXiv 2505.15820) uses the cross as its example of providers disagreeing.
- Wyscout: "a ball played from the offensive flanks aimed towards a teammate in the area in front of the opponent's goal". It splits crosses into high, low and blocked.
- Opta: a cross is separate from a pass. "Pull back – a pass inside the penalty area which is pulled back" is a pass.
- Vidal-Codina et al. 2022 (arXiv 2202.00804; MIT and FIFA) detect events from tracking alone. A ball release is a cross when three things hold:
  - It starts in a "cross zone".
  - The next player in control, from either team, is in the penalty area.
  - At least one attacker is in the penalty area.
  - The paper benchmarks against Sportec (STS) events.
- The Twelve course (Deep Learning and AI in Sports, Module 3) uses `|y_start| > 15 & |x_end| > 40` (centre origin). The course calls this a teaching simplification. Course code is cohort-licensed and is not copied.

## 4. Owner decisions (2026-11-13)

1. **Two layers.**
   - The converters use one geometric rule for every provider.
   - The tracking layer adds the cross attempt and its height.
   - The provider's own cross flag stays as a separate column.
2. **The xCross attempt label counts all heights.** Height is an attribute, not a filter.
3. **A blocked cross counts as an attempt.** The tracking layer sees the ball path. Event-only data cannot, and Section 6.4 states this limit.
4. **`cross_height` is an honest NaN where the ball height is only estimated.** For SkillCorner today, `z_source` is `"estimated"`.
5. **Set-piece deliveries use the same delivery rule** (Section 6.1b). They differ by provider too. Median deliveries per match on xshot187:
   - `corner_crossed`: GS 7, IDSSE 7, SkillCorner 2.
   - `freekick_crossed`: GS 3, IDSSE 0, SkillCorner 11.
6. **Retrains are owed, not done in this cycle.** The plan lists every affected artifact with evidence. Part Deux has paused its refitting and will include these retrains when it resumes, after this release.
7. **A restart-ended delivery is a `pass` in the converters** (6.1 condition 4). The tracking layer decides whether it was a cross attempt. Section 7.3 measures the event-only undercount.
8. **The start line comes from the mean F1 over vendors with a full native cross label** (StatsBomb and GS, equal weight; 7.2). No single vendor sets it.
9. **r2 additions approved:**
   - `cross_native` also keeps the provider's own label on set pieces.
   - The bravery cross reads move to `cross_native == True`.
   - An unknown xCross label is excluded from training.
   - `cross_height` reuses the ADR-125 aerial threshold (`BallContactParams.aerial_z_m`, 1.8 m).
   - The tracking path ends at the first detected ball contact.
10. **Tracking decides set pieces.** `cross_attempt` also covers corner and free-kick rows (6.3). TF-46 corner routines and the xCross set-piece runs read it when frames exist. The converters keep the one event rule: a restart-ended set piece is `_short`, and each provider's event-only recall is recorded as a limit (6.4). Evidence (audit 5, `audit5_tracking_vs_event.py`):
   - The event rule finds 94 % of GS corner deliveries whose ball path enters the box, but only 43 % of SkillCorner's. SkillCorner's event data misses defenders' touches in the box.
   - SkillCorner's event-only corner typing gets worse (62 % `corner_crossed` today, 22 % under the rule). The release notice says so.
11. **Native event timing, in this cycle (2026-11-14, r8).** Run 2 on commit 2 fired the release-match owner trigger: the IDSSE match (4.67 m) turned 28.2 % of GS and 31.7 % of SkillCorner ELASTIC-confident rows into `<NA>`. The owner paused the calibration, asked for the root cause and for a spike, and then ruled: fix the timing in this cycle, at the gold standard. The question the owner answered stated that the fix changes all GS and SkillCorner SPADL, needs a corpus re-stage, and makes the models trained on those providers owed a retrain; decision 6 widens accordingly (review B C2). Section 5.1 gives the evidence; Section 6.6 gives the design.

## 5. Evidence (audit on the DGX, 2026-11-13)

The scripts are in `D:\Temp\cross-def\`; the per-match CSVs are on the DGX in `~/cross-def/`. The run is read-only.

Audit 2 tests the geometric rule (Section 6.1, start line 65 m) with two choices of end location:

| Provider | Native | End = provider end | End = next on-ball action start | Agreement of the two ends |
|---|---|---|---|---|
| GS | 26.5 | 17 | **16.5** | 98.4 % |
| IDSSE | 4 | 19 | **19** | 100 % |
| SkillCorner | 101 | 13 | **21** | 84.1 % |

- The next on-ball action excludes `non_action`, `foul` and synthetic dribbles only, as in 6.1. On the finalized staged actions, every IDSSE and SkillCorner `dribble` comes from `_add_dribbles`, and every GS `dribble` is a native carry, because GS never calls `_add_dribbles`. Audit 2b (`audit2b_synthetic_only.py`) applies that rule and gives the same medians: 16.5 / 19 / 21. Audit 2 excluded every dribble and gave the same medians.
- SkillCorner has no ball end for a pass without a targeted receiver. The converter then uses `player_targeted_x_reception` or the carrier's position (`spadl/skillcorner.py:292-306`). A failed cross therefore never "ends" in the box. The next on-ball action start removes this provider effect.
- Audit 1: moving the start line from 52.5 m to 70 m changes the medians by 1.5 to 3 crosses per match.
- Review A measured how often a geometric cross (audit-2b rule, 65 m) is followed by an opponent goal kick: GS 3.1 % (32 of 1049), IDSSE 9.9 % (14 of 142), SkillCorner 3.8 % (96 of 2534). The goal kick starts in the six-yard box, which mirrors into the passer's penalty area. Decision 7 removes this effect.
- Audit 3 checks tracking ball height over the cross path (release to next on-ball action, at most 4 s). It used a 1.5 m threshold; the design now uses 1.8 m (decision 9), and the plan re-measures:

| Provider | `z_source` | Crosses | Share with peak > 1.5 m | Median peak |
|---|---|---|---|---|
| GS | measured | 1046 | 52 % | 1.62 m |
| IDSSE | measured | 142 | 62 % | 3.13 m |
| SkillCorner | **estimated** | 2534 | 43 % | **0.75 m** |

SkillCorner's estimated height reads low, which supports decision 4.

### 5.1 Release timing (r8)

The scripts are `D:\Temp\cross-def\align_tail_{1..5}.py`, `sc_kick_check.py`, `gs_native_check.py`, `gs_truth_check.py` and `spike_loc.py`; the outputs are in `~/cross-def/align/` on the DGX. The rows are run 2 on commit 2 (`b12c0fd9`, line 75): the applicable ELASTIC-confident rows. "Tail" means `release_dist_m` above 4.67 m, the IDSSE match of run 2.

**The tail is systematic.** It is smooth (no second mode) and it is spread over the matches: the per-match tail share at the 10th/50th/90th percentile is GS 19/27/39 %, SkillCorner 25/32/40 % and IDSSE 2/6/7 %. Two different causes produce it.

**SkillCorner: the converter places the action at the possession start.** `spadl/skillcorner.py` (since `6fcc3141`, 2026-05-15, PR-S40, silly-kicks 3.14.0; spec `2026-05-14-skillcorner-events-converter-design` section 4) builds one SPADL row per `player_possession` from `time_start` and `x_start`/`y_start`. That is the reception. The pass, shot or clearance happens at the possession end (`time_end`, `frame_end`, `x_end`/`y_end`). The raw dynamic events carry all of these fields (pining cache). On 98 matches (10,315 rows):

| Measure | Value |
|---|---|
| ELASTIC release ball to the raw kick point (`x_end` through the converter's own start transform), median / P95 | 0.85 m / 6.3 m |
| Rows within 4.67 m of the kick point | 93.7 % (tail rows: 86.1 %) |
| ELASTIC frame minus raw `frame_end`, median | −2 frames (0.2 s) |
| Ball at raw `frame_end` to the kick point, median / P95 (the native release of 6.6 A) | 1.42 m / 3.21 m |
| Tail rows: carry from the SPADL start to the kick point, median; possession length, median | 8.1 m; 2.8 s |

ELASTIC finds the kick. The tail is the carry between the reception and the kick. The same defect makes `classify_deliveries` judge a SkillCorner cross at the reception point, and every time-based frame link reads the reception moment.

**Gradient Sports: the event time is a whole-second clock at the start of the on-ball sequence, and ELASTIC picks wrong frames.** The pining loader (`scripts/_loader_pining.py:1165`, since `0a76f529`, 2026-05-29, PR-B, silly-kicks 3.28.0) sets `time_seconds` from `gameEvents.startGameClock`. It loads the possession event's own `eventTime` as `event_time` but uses it only to order fouls. The raw GS tracking frames carry PFF's own synchronisation: `videoTimeMs`, `periodGameClockTime`, `game_event_id`, `possession_event_id` and `possession_event.start_frame`. On 62 matches:

| Measure | Value |
|---|---|
| Events-only time (`eventTime` − the period's kickoff `startTime`, compared on the match clock: + 0/2700/5400/6300 s) against the tracking `periodGameClockTime` at PFF's `start_frame`, 86,067 possession events | per-match median ≤ 0.033 s, 99th percentile ≤ 0.067 s (2 frames at 29.97 Hz), none above 0.1 s |
| Ball to the event start at that time, median / P95; share above 4.67 m | 0.0 m / 0.98 m; 0.8 % |
| Today's `time_seconds` minus that time, 10th/50th/90th percentile | −4.23 / −1.59 / −0.28 s |
| ELASTIC release minus that time: within 0.1 s / within 0.5 s / above 3 s | 14.6 % / 49.0 % / 20.4 % |

Every one of the 64 cached GS matches has a kickoff event (`FIRSTKICKOFF` … `FOURTHKICKOFF`) in every period 1–4 it plays, and every one of the 144,541 events has an `eventTime`. One match (3833) also has a period-0 `G` game marker, which the converter excludes (6.6 B).

**Why ELASTIC fails on GS.** Two causes, measured separately:
- *Episodes.* ELASTIC aligns per in-play episode, a maximal run of `ball_state == "alive"` frames (`_inplay_segments`). GS and SkillCorner frames are `"alive"` on every frame (the loader and `tracking/skillcorner.py:281` hardcode it; ADR-130). So each half is one episode, the candidate window is the whole half ± 5 s, the score has no time term, and both gap penalties are 0. A match to a touch by the same player tens of seconds away costs nothing: 52.5 % of the GS tail rows moved more than 3 s from the event time, 25.5 % more than 10 s. The staged frames already carry the real dead-ball state in `stoppage_state` (ADR-130). Against PFF's ground truth (`r8_checks.py`; 71,623 GS pass and cross rows with a truth frame; ELASTIC-confident rows only count as matched):

| ELASTIC on GS | matched | within 0.1 s of truth | within 0.5 s | off by more than 3 s |
|---|---|---|---|---|
| today's times and episodes | 78.2 % | 12.7 % | 41.6 % | 12.6 % |
| r8 times (6.6 B), today's episodes | 78.2 % | 12.7 % | 41.6 % | 12.6 % |
| r8 times, episodes from `stoppage_state` (6.6 D; spans from the r8 action times) | 75.9 % | 12.9 % | 42.2 % | 8.9 % |

Episodes from `stoppage_state` cut the gross mismatches (> 3 s) by 29 % (8,994 → 6,374 rows); most of them become honest non-matches (matched 78.2 % → 75.9 %), and the correct matches rise a little (29,789 → 30,197 within 0.5 s). The r8 times alone change nothing, because with one episode per half ELASTIC reads no per-event time. In the run-2 rows the same change moves the GS tail from 28.2 % to 23.3 %, but review A showed that most of that drop is rows leaving the confident set (`elastic_share` 0.754 → 0.698), not better matches. The IDSSE benchmark cannot test this change: IDSSE records `ball_state`, so `stoppage_state` copies it.
- *Touch choice.* Inside an episode, ELASTIC often matches an earlier touch by the same player, usually the reception: for the tail rows within 3 s of the event time, the ball reaches the event start 0.3–3 s after ELASTIC's frame in 76 % of cases.

**The spike rejected a location term.** Adding a ball-to-event-start term to ELASTIC's score (`spike_loc.py`) cut the GS tail, but that gain is partly circular, and it lowered the accuracy on the only ground truth we hold, the CC-BY IDSSE benchmark of TF-57 (Kim et al., 3 matches; the harness reproduces the published W2 0.862):

| ELASTIC variant | GS `elastic_share` | GS tail | IDSSE W2 | IDSSE exact |
|---|---|---|---|---|
| as committed | 0.754 | 28.2 % | 0.862 | 0.584 |
| episodes from `stoppage_state` | 0.698 | 23.3 % | 0.862 | 0.584 |
| + location term, weight 1.6, 5 m | 0.639 | 7.4 % | 0.854 | 0.569 |
| + location term, weight 4, 5 m | 0.577 | 0.0 % | 0.787 | 0.519 |
| + location term, weight 4, 10 m | 0.653 | 4.4 % | 0.833 | 0.543 |

**Are the native times the release? (review B CD-SB-13).** The GS ball-to-start distance at PFF's own frame is near-tautological: PFF's event location is its own ball at its own frame. The independent signal is the passer's own resolved touch from the ADR-124/125 detector (`passer_contact_dt`, the touch time minus the release, searched in [−0.5 s, +1.0 s]). `native_preview.py` runs the enricher's diagnostics on the run-2 rows with the release at the native time (GS: PFF truth; SkillCorner: `time_end` through the converter's own parse and period re-base, start at the kick point) and ELASTIC switched off:

| Rows (line 75) | release distance q50 / P95 | > 4.67 m | `cross_attempt` unknown | passer touch found | touch − release q05 / q50 / q95 |
|---|---|---|---|---|---|
| GS, native | 0.0 / 1.15 m | 1.2 % | 12.1 % | 50.4 % | −0.43 / −0.23 / +0.23 s |
| SkillCorner, native | 1.46 / 4.21 m | 4.3 % | 4.3 % | 78.4 % | −0.50 / −0.20 / +0.10 s |
| GS, ELASTIC (run 2) | 2.13 / 21.6 m | 28.2 % | 43.7 % | 77.2 % | −0.43 / −0.17 / +0.43 s |
| SkillCorner, ELASTIC (run 2) | 2.55 / 15.8 m | 31.7 % | 44.3 % | 86.9 % | −0.50 / −0.10 / +0.20 s |
| IDSSE, ELASTIC (run 2) | 1.63 / 4.67 m | 5.0 % | 15.3 % | 93.6 % | −0.40 / 0.00 / +0.04 s |

By row class (review A C21; SkillCorner restarts start at the taker's `x_end`, not at the placed ball), the share of native rows beyond 4.67 m is: SkillCorner pass 3.4 % (n 7,432), cross 4.6 % (1,498), free kick 6.9 % (1,830), corner 7.5 % (373; P95 7.08 m); GS pass 0.8 %, cross 1.8 %, corner 1.9 %, free kick 2.1 % (the short kinds alone: `corner_short` 3.0 %, n 100; `freekick_short` 2.2 %, n 996; review A C24). The restart start is the taker's position at the kick, not the ball's spot; that difference reaches the corner P95 but stays far below the 20 % owner-trigger bar of 7.4.

The native times sit about 0.2 s after the detector's passer touch, for both providers. The SkillCorner value agrees with the −2 frames between ELASTIC's frame and `frame_end`. The release is therefore the provider's own event moment, slightly after the touch, not the touch itself. The passer exclusion of 6.3 already removes that touch from the path. A first contact by another player inside those 0.2 s would be missed; 7.4 reports this check and has an owner trigger for it. For GS the detector finds the passer's touch on only half of the native rows; the IDSSE figure (93.6 %, at 0.0 s) is partly circular, because ELASTIC's candidates are touch frames.

**Conclusion.** GS and SkillCorner events are timed from tracking by the provider. Their native times are better than the alignment: GS ground truth is within 0.067 s, and the SkillCorner kick frame is 0.2 s from ELASTIC's. For these rows ELASTIC adds error: on GS it lands within 0.5 s of the truth for 49.0 % of the run-2 ELASTIC-confident delivery rows, and for 41.6 % of all 71,623 pass and cross rows with a truth frame, where an unmatched row counts as a miss (review A C25). IDSSE events are timed by hand, and ELASTIC stays the right tool for them.

## 6. Design

### 6.1 Converter layer: the geometric `cross`

One shared function in `silly_kicks/spadl/base.py` re-types the open-play pass-class rows of every converter. A row is a `cross` when all of these hold:

1. **Open play.** Its type is `pass` or `cross` before re-typing. Corners and free kicks follow 6.1b; `throw_in` and `goalkick` do not change.
2. **Wide start.** `start_y < W/2 − penalty_area_half_width` or `start_y > W/2 + penalty_area_half_width`. This uses `spadlconfig.penalty_area_half_width` (20.16 m) and `field_width`.
3. **Advanced start.** `start_x >= CROSS_START_X`. The value is chosen once by the rule in Section 7.2, from {52.5, 60, 65, 70, 75, 80} (r7).
4. **Delivery into the box.** The start of the next on-ball action in the same period is inside the penalty area, in the passer's frame:
   - The next on-ball action is the next row in the converter's chronological order that is not `non_action` or `foul`. Rows that share a timestamp keep the converter's own tie order: `sort_actions_chronologically` (stable mergesort, ADR-065) in most converters, and the `sort_values` at `skillcorner.py:545` for SkillCorner.
   - **Restarts.** When the next on-ball action is a restart — a type in `_SET_PIECE_RESTART_TYPE_NAMES` (`spadl/utils.py:22`), `shot_penalty` or `shot_freekick` — the ball left play or the game stopped, and the event data has no delivery end. Condition 4 then fails, so the row is a `pass` (decision 7). The tracking layer decides the attempt from the ball path.
   - At the run position below, synthetic dribbles do not exist yet. Native dribbles and carries (StatsBomb `Carry`, GS `OTB`/`BC`, the Opta recovery dribble) are on-ball actions. GS rows with `is_synthetic` True count by their type, so a synthetic `foul` is skipped and a synthetic shot is an on-ball action.
   - When its team differs from the passer's team, mirror it into the passer's LTR frame: `(L − x, W − y)`.
   - The penalty area is `x >= field_length − penalty_area_depth` and `|y − W/2| <= penalty_area_half_width`, from `spadlconfig` (16.5 m and 20.16 m).

A row that meets 1 but not 2–4 becomes a `pass`. The rule reads only the SPADL columns, so it is the same for every provider.

### 6.1b Converter layer: set-piece deliveries

The same function re-types corners and free kicks with the delivery condition (6.1 condition 4) alone:
- A corner is `corner_crossed` when the next on-ball action starts inside the penalty area, and `corner_short` otherwise.
- A free kick is `freekick_crossed` under the same condition, and `freekick_short` otherwise. There is no wide-start condition, because a central free kick lofted into the box is a delivery.
- **Restart-ended set pieces.** When the next on-ball action is a restart (6.1 condition 4), the set piece is `_short`. This is the event-only limit of decision 10, not a judgement that the delivery was short. The tracking layer decides the delivery (6.3).
- Penalties, throw-ins, goal kicks and shot-typed free kicks (`shot_freekick`) do not change.
- This supersedes the Sportec `play_height` discriminator for corners (ADR-127). ADR-133 records the supersession, and ADR-127 gets a status note.

**Where it runs.** The position is defined by its precondition, not by a neighbouring call:
- The rows are in canonical SPADL LTR, after `to_spadl_ltr` (ADR-006).
- The rows are in their final chronological order.
- `_add_dribbles` has not run yet.

Per converter (line numbers at `0cc07e6c`):

| Converter | Position |
|---|---|
| `statsbomb.py` | between `to_spadl_ltr` (293) and `_add_dribbles` (301) |
| `opta.py` | between `to_spadl_ltr` (219) and `_add_dribbles` (227) |
| `wyscout.py` | between `to_spadl_ltr` (330) and `_add_dribbles` (337) |
| `metrica.py` | between `to_spadl_ltr` (285) and `_add_dribbles` (292) |
| `sportec.py` | between `to_spadl_ltr` (660) and `_add_dribbles` (667) |
| `kloppy.py` | `to_spadl_ltr` (239) moves before `_add_dribbles` (231); the rule runs between them |
| `skillcorner.py` | `to_spadl_ltr` (557) moves before `_add_dribbles` (550); the rule runs between them |
| `gradientsports.py` | after `to_spadl_ltr` (856), before `_finalize_output`; GS never calls `_add_dribbles` |

The kloppy and SkillCorner reorder must be byte-identical for everything except the new re-typing:
- `_add_dribbles` reads only same-team pairs of rows.
- `to_spadl_ltr` mirrors a team's rows in a period as one block, so distances and times within a same-team pair do not change.
- The plan proves this with a regression test on the existing converter fixtures before the rule lands.

**Converters in scope.** Each native rule stops deciding the SPADL type and moves to `cross_native` (6.2):
- `statsbomb.py`: `_open_play_cross_mask` (`pass.cross`). It stays as the `cross_native` source and the `cross_blocked` key; only its use in `_vectorized_type_id` goes.
- `opta.py`: `q_2`.
- `wyscout.py` with `_wyscout_mappings.py`: subtype 80, and `_WS_SUBTYPE_FK_CROSSED` for free kicks.
- `kloppy.py`: `PassType.CROSS`.
- `metrica.py`: `CROSS` subtype.
- `gradientsports.py`: `CR`.
- `sportec.py`: `play_height` and `play_flat_cross`, and the ADR-127 corner discriminator.
- `skillcorner.py`: `_is_cross` is deleted, because it is our rule, not a vendor label.
- `providers/sportec/parse.py:243` does not change. It is the bronze column prefix map (`"Cross"` → the `cross_*` columns such as `cross_side`), not a type decision. The Sportec converter reads those columns for `cross_native`.

### 6.2 Converter layer: `cross_native`

Add a new SPADL column `cross_native`, a nullable `boolean`: the provider's own "crossed" label for every delivery that 6.1 or 6.1b re-types.
- On open-play pass-class rows: `True` where the provider labels the pass a cross, `False` otherwise.
- On corner and free-kick rows: `True` where the provider's own split said "crossed", `False` where it said "short". For example: opta `q_2`, wyscout `_WS_SUBTYPE_FK_CROSSED`, the ADR-127 `play_height` discriminator, StatsBomb `pass.cross` or `High Pass` on set pieces.
- `<NA>` on every other row, and on every row of a provider that has no native label. SkillCorner's zone rule is ours, not the vendor's, so SkillCorner is `<NA>`.

The column keeps the old signal available for open play and set pieces (Hyrum, decision 1). It is the benchmark for Section 7. It joins `SPADL_COLUMNS` and the per-provider schema dicts (`spadl/schema.py`). In atomic SPADL it is opt-in through `convert_to_atomic(preserve_native=…)`, the same way as `cross_blocked`.

`cross_blocked` (ADR-046) stays keyed to the provider's cross. It is applicable where `cross_native` is `True` on an open-play row, and `<NA>` elsewhere. Its meaning does not change.

**Bravery** (`defensive_credit/_bravery.py`, and atomic `compute_bravery` at `atomic/tracking/features.py:471,477`). Every cross read moves from the SPADL type to `cross_native == True`, so each output keeps its meaning:
- the block rate `b_open` (`:68`);
- `n_open_play_crosses_faced` (`:64`);
- the headline `_known_domain` (`:113-114`);
- `n_set` (`:65`), on the set-piece rows.

The atomic path threads `cross_native` the same way it threads `cross_blocked`.

**Causal opportunities** (`causal/opportunities.py:96-103`). This reads the model metadata `cross_types` and treats `"cross"` as the SPADL type. It reads `xcross_label_version`: version 2 selects `cross_attempt == True` rows, and version 1 (or no key) keeps the SPADL type.

### 6.3 Tracking layer: `cross_attempt` and `cross_height`

A new pure enricher `add_cross_deliveries(actions, frames, *, ...)` in `silly_kicks/tracking/_cross_delivery.py` adds two columns to actions. The name differs clearly from the xCross model's `add_xcross_attempt`. It follows the house rules for frame-consuming `add_*` functions (Section 9). It runs on finalized actions, so it never looks up a "next action". The ball path is defined from the frames alone.

**The release.** Event clocks and tracking clocks are not aligned for every provider. Review A measured that the ball is more than 5 m from the action start at the event time for 4041 of 6461 GS rows and 2463 of 4596 IDSSE rows; audit 5 could judge only the aligned rows. The release is therefore aligned through the TF-43 seam (`tracking/_elastic_sync.py`), not read from `time_seconds`, unless the event time is already a tracking time (r8, Section 6.6):
0. **Native time (r8).** Where the row's `time_source` is `"tracking"`, the release is `time_seconds`. ELASTIC is not consulted for that row (Section 5.1: there it adds error).
1. **Aligned frame.** The enricher uses `elastic_frame_id` when the action matched with `elastic_confidence >= ElasticSyncParams.min_confidence` (0.5).
   - When the actions already carry `elastic_frame_id` and `elastic_confidence` (from `add_elastic_sync`), it reads them.
   - Otherwise it calls `align_events_to_frames(actions, frames)` once per call.
2. **Unaligned action.** For an action with no confident match, the release is `time_seconds`.
3. **Release match, for every row.** At the release frame, the ball must be within `RELEASE_MATCH_M` of the action start. Otherwise the release is misaligned and the path is unanswerable (`<NA>`, unknown). This is the condition audit 5 used ("sync too poor to judge"). The plan sets `RELEASE_MATCH_M` from the 7.3 data.

**The ball path.** The release is the aligned release above. The path runs from the release to the first resolved ball contact by a player other than the passer at or after `release + RELEASE_GAP_S`, capped at `release + 4 s`. Excluding the passer means an event time earlier than the true kick cannot end the path at the passer's own touch.
- The contact comes from the ADR-124/125 detector (`detect_ball_contacts`, `first_contact_in_window`). The detector is reused, not reimplemented.
- `RELEASE_GAP_S` skips the passer's own release contact.
- With no resolved contact in the cap, the path ends at the cap.
- **Coverage.** The path is answerable when finite ball positions cover at least `MIN_BALL_COVERAGE` of the path's frames. Otherwise it is unanswerable.

**`cross_attempt`** (nullable `boolean`). Its value for each row class:

| Row class | Value |
|---|---|
| open-play pass-class row (`pass` or `cross`) released in the cross zone (6.1 conditions 1–3), path answerable | `True` or `False` by the test below |
| the same row, path unanswerable | `<NA>` (unknown) |
| open-play pass-class row released outside the cross zone | `False` |
| corner or free-kick row (`corner_*`, `freekick_*`), path answerable (decision 10) | `True` or `False` by the test below |
| the same set-piece row, path unanswerable | `<NA>` (unknown) |
| every other row (shots, `shot_freekick`, `shot_penalty`, throw-ins, goal kicks, dribbles native or synthetic, defensive actions) | `<NA>` (not applicable) |

Set-piece rows have no zone condition: a corner starts wide by definition, and a central free kick lofted into the box is a delivery (6.1b).

The test is `True` when either holds:
- the ball path enters the penalty area, or
- the first contact is by an opponent (a block), within `BLOCK_RADIUS_M` of the release and within `BLOCK_WINDOW_S` (1.0 s) after the release, and the initial ball direction points into the penalty area. The initial direction is the mean ball velocity over the first `DIRECTION_WINDOW_S` after the release.
- `BLOCK_WINDOW_S` (r7) is fixed, not calibrated, like the 4 s path cap. A block deflects the ball at once. At cross speeds of 15–25 m/s the ball travels 15–25 m in 1 s, so the window does not cap the radius. It excludes a first contact that comes after the ball has stopped or turned back, which is not the block. The author chose 1.0 s after seeing the run-3 contact times of the 49 GS blocked crosses (provisional parameters): 32 fall at 0.17–0.90 s, then nothing until 1.37 s (Section 7.4). The gap is not clean at the final parameters: on the run-2 rows within a 5.0 m release match (n = 59, review A CD-IMPL-09) there are contacts at 1.03, 1.10 and 1.17 s. The 95th percentile does not depend much on the window there: 7.89, 7.71, 7.43 and 7.34 m at 0.9, 1.0, 1.2 and 1.5 s.

All geometry is in the action-LTR frame (ADR-028), with the direction from the GoalMap (ADR-055).

**`cross_height`** (category `"aerial"` / `"ground"`). On rows where `cross_attempt` is `True`:
- It is `"aerial"` when the peak ball z over the path is above `BallContactParams.aerial_z_m` (1.8 m, ADR-125), and `"ground"` otherwise. The threshold is the one constant of the contact detector, not a second one.
- It is `<NA>` where the ball `z_source` is not `"measured"` (decision 4). This is the same rule as the detector's `first_contact_aerial_source`.
- It is `<NA>` on every other row.

### 6.4 The event-only limit

Without tracking, two kinds of delivery cannot be seen as a cross:
- a blocked cross, because the block ends the delivery outside the box;
- a restart-ended delivery, because the event data has no end (decision 7).

6.1 types both as `pass`. A restart-ended set piece is `_short` (6.1b). SkillCorner's event data also misses defenders' touches in the box, so the event rule finds only 43 % of its corner deliveries (decision 10). ADR-133 and the `classify_deliveries` docstring state these limits once. Each converter docstring points to `classify_deliveries`, names its provider's own label and states that provider's measured recall; the SkillCorner docstring also names the missing in-box touches (owner, 2026-11-14 size audit). The tracking layer covers all of them.

**TF-46 corners.** `corners/_identify.py:95` sets `corner_kind` from the SPADL type, over the existing domain `CORNER_KIND_VALUES = {"direct", "short", "unclear"}` (`corners/_columns.py:25`). The domain does not change. The decision order is:
1. **Delivery.** When the actions carry `cross_attempt`, a corner is `"direct"` where `cross_attempt` is True and `"short"` where it is False. Where it is `<NA>`, or the column is absent, the SPADL type decides, as today (`corner_short` → `"short"`, else `"direct"`).
2. **GI refinement.** `_apply_gi_refinement` (`_identify.py:106`) then runs as today and may set `"unclear"`. It wins over step 1, because it says that the first possession after the corner is not confirmed, which no delivery evidence answers.

A new column `corner_kind_source` records what decided: `"tracking"` (step 1 from `cross_attempt`), `"event"` (step 1 from the SPADL type), or `"gi"` (step 2 set `"unclear"`). This follows ADR-054, because the value source changes. `_identify.py:103` reindexes to `CORNER_IDENTIFY_COLUMNS`, so `corner_kind_source` joins that dict. It gets a closed vocabulary `CORNER_KIND_SOURCE_VALUES` in `corners/_columns.py`, tested in `test_columns.py`; otherwise it would be dropped without an error. `_encode.py:99` and `build_corner_routines.py:36` read `corner_kind` unchanged: they keep `"direct"` corners and skip `"short"` and `"unclear"`.

### 6.5 The xCross label

`build_xcross_labels` (`tracking/_xcross_attempt.py:353`) labels a frame 1 when a same-team cross event occurs within the horizon. The cross events change as follows:

| Entry in `cross_types` | Rows that count as cross events |
|---|---|
| `"cross"` (the default) | rows with `cross_attempt == True`: open play, all heights, blocked included |
| `"corner_crossed"`, `"freekick_crossed"` (sensitivity runs) | corner rows (or free-kick rows) with `cross_attempt == True` (decision 10) |

- **Unknown attempts.** A frame gets an unknown label when its horizon holds no cross event and holds a same-team row of a selected kind whose path is unanswerable (6.3 table, "unknown" rows). The open-play kind is selected by `"cross"`; a set-piece kind is selected by its `cross_types` entry. A not-applicable `<NA>` (a shot, a dribble, a throw-in, a goal kick, or a set-piece kind not selected by `cross_types`) never makes a frame unknown. It is excluded from training and from the evaluation folds. `prepare_xcross_training_data` reports the excluded count for each match. It never labels such a frame 0.
- **Missing column.** Every `cross_types` entry reads `cross_attempt`. When the actions have no `cross_attempt` column, `build_xcross_labels` raises `ValueError` for any `cross_types`, and the message names `add_cross_deliveries`. There is no fallback to the SPADL type.
- **Label version.** The model metadata gains the key `xcross_label_version`. Value 2 means this definition. An artifact without the key was trained on the SPADL type `cross` and reads as version 1.
- The refit itself is Part Deux's work, after this release.

### 6.6 Native event timing (r8, decision 11)

Four changes. Each one is provider-neutral in the shared code; the provider facts live in the converter that owns them.

**A. SkillCorner: the action at the kick.** In `spadl/skillcorner.py`, every native `player_possession` row (the rows typed from `end_type` and `game_interruption_before`: pass, shot, clearance, set-piece deliveries, `foul_suffered`, `possession_loss`, `unknown`) takes its `time_seconds` from `time_end` and its start from `x_end`/`y_end`, through the same parse, the same period re-base (`_parse_time_start`, `_to_period_relative`, ADR-017) and the same transform as today's start. The derived rows that happen at the possession start keep `time_start` and `x_start`/`y_start`: the `start_type` interceptions and recoveries (`infer_defensive_actions`) and the keeper saves (`infer_keeper_saves`). `_add_dribbles` (already called, `spadl/skillcorner.py:538`) then adds the carry from the previous action's end to the kick point; today the reception and the kick are one point, so the converter adds almost no carries (review A C18). `time_end`, `x_end` and `y_end` become required input columns; a row whose `time_end` is missing gets a NaN time, never one filled from `time_start`; its start stays `x_end`/`y_end`. The converter reports such rows with one `UserWarning` (`stacklevel=2`) that names their count; nothing else would surface them, because `_assert_chronological_action_id` checks finite times only (review B CD-SB-21).

The end coordinates, per type (review B CD-SB-15):
- **pass-class rows** (`pass`, `cross`, set-piece deliveries, `throw_in`, `goalkick`, `clearance`): `player_targeted_x_reception`/`y_reception` where present (passes and set pieces, as today); otherwise the reception point of the next `player_possession` row (its `x_start`/`y_start`, the next possession in reception order within the period), the ball's destination. The receiver's carry from that reception to their kick then follows as a synthetic dribble (`_add_dribbles`). At the last possession of a period there is no next reception; the end equals the start and `_derive_end_coordinates` (`spadl/base.py:83`) applies as before (r9, owner ruling 2026-11-14, review B CD-SB-23). r8 left these rows to `_derive_end_coordinates`, whose "next action's start" is now the next KICK, not the reception: the pass then ended at the receiver's kick and the receiver's carry disappeared. Measured on the 118 cached SkillCorner matches (`sb23_measure.py`): 16,753 such rows (untargeted passes 15,685, clearances 1,068; 13.4 % of `player_possession` rows); the next kick lies a median 0.88 m and a P95 15.6 m from the next reception, more than 1 m on 49.1 % of them;
- **`shot`, `foul`, `non_action`** (`foul_suffered`, `possession_loss`, `unknown`): the end equals the start. SkillCorner records no ball destination for these. This is the Gradient Sports convention (`spadl/gradientsports.py:311`). Today the end of such a row is the carrier's end position, which becomes its start.

**B. Gradient Sports: the possession event's own time, period-relative.** In `spadl/gradientsports.py`, `time_seconds` becomes the possession event's time in seconds since its period's kickoff: `event_time − kickoff_start_time[period]`. This is period-relative, the canonical SPADL convention (`spadl/schema.py:14`, ADR-017 decision 1; ADR-017 rejected the absolute match clock as its option A). `event_time` is the event record's top-level `eventTime` and `start_time` its top-level `startTime` (the pining loader's names, `scripts/_loader_pining.py:1168-1169`). `kickoff_start_time[period]` is the `start_time` of that period's kickoff game event (`FIRSTKICKOFF`, `SECONDKICKOFF`, `THIRDKICKOFF`, `FOURTHKICKOFF`). The converter reads the anchors before its exclusion filter, which drops the kickoff rows (`spadl/gradientsports.py:522`).
- **Which periods need an anchor.** Every period that holds a converted action. A period 1–4 action without its kickoff raises a `ValueError` that names the period; there is no silent fallback to `startGameClock`. The GS feed carries the penalty shootout inside period 4, after `FOURTHKICKOFF` (measured: match 10511 has 19 period-4 shots, clock to 7825 s), and every one of the 64 cached matches has a kickoff in every period 1–4. A caller that splits the shootout into period 5 (the lakehouse maps a period 5) gets period 5 anchored at the first `start_time` of its period-5 rows, because a shootout has no kickoff (review B CD-SB-12). A period with no converted action needs no anchor: match 3833 has one period-0 event, a `G` game marker, which the converter already excludes.
- **An event before its kickoff (review B CD-SB-18).** No converted event in the 64 cached matches has an `event_time` before its period's kickoff `start_time` (`gs_prekick_clock.py`: 0). SPADL documents `time_seconds` as non-negative (`spadl/schema.py:43-45`). A converted event before its kickoff therefore raises a `ValueError` that names the event and the period: it is a data defect, not a time to keep.
- **Order (review A CD-SPEC-11).** The converter computes the new `time_seconds` before its top sort (`spadl/gradientsports.py:612-637`) and sorts by it: `sort_actions_chronologically(events, tiebreak=("start_time", "possession_event_id", "event_id"))`, whose default key is `(game_id, period_id, time_seconds)`. The tiebreak makes the order a pure function of content (ADR-065). Today's order (`startGameClock`, then `start_time`) disagrees with the new time on some adjacent pairs: review A counted 74 inversions in 44 of 64 matches before the converter's row filter (worst −14.1 s). After the converter's exclusion filter, 89 of 89,776 converted rows change rank, in 27 of 64 matches (`r8_checks.py`). Sorting by the new time therefore reorders those rows, and GS `action_id` renumbers. That is an ADR-065 cross-table migration, and the Hyrum notice names it.
- **Option D becomes dead.** Option D imputes a NaN input `time_seconds` (`startGameClock`, which is null on the dedicated FOUL events) by an ffill ordered by `start_time` (`spadl/gradientsports.py:365-410`). Under 6.6 B the output time reads `event_time`, which every event carries (144,541 of 144,541, the FOUL events included). The imputation and the input `time_seconds` then reach no output. The plan removes the imputation (Chesterton's fence: its reason, a null `startGameClock`, no longer reaches any output) and drops `time_seconds` from the required GS input columns; a supplied `time_seconds` is ignored. A converted row whose `event_time` is NaN raises a `ValueError`.
- **The pining loader's GS frames move to the period-relative clock** (a pre-existing ADR-017 defect, fixed here because the re-stage needs it). `_build_gradientsports` (`scripts/_loader_pining.py:1324`) takes the frame `time_seconds` from `periodGameClockTime`, the match clock, although GS frames must be period-relative (`tracking/gradientsports.py:70`). It goes unnoticed today only because the loader's actions carry the match clock too (`startGameClock`). The frames switch to `periodElapsedTime`, the raw tracking's period-relative clock: on the raw tracking, `periodGameClockTime − periodElapsedTime` is 0, 2700, 5400 and 6300 s in periods 1–4, within 0.43 frames on every one of the 64 cached matches and every period (`gs_prekick_clock.py`; review B C2 and CD-SB-22). The 5.1 accuracy carries over unchanged: the check compared both sides on the match clock, which differs from the period-relative clock by that constant per period. The re-stage checks it again on its own output and stops on a deviation above one frame.

**C. A provenance column `time_source`.** Every converter emits `time_source` (dtype `object`, closed vocabulary `TIME_SOURCE_VALUES = ("tracking", "event")`, validated in `_finalize_output`): `"tracking"` where the provider times its events from its own tracking clock, `"event"` otherwise. `"tracking"` holds only against that provider's own frames; with frames from another source the column says nothing about them (review A C20). It follows ADR-054: the value of a downstream column (the release frame) changes with it. GS (B) and SkillCorner (A) emit `"tracking"` on native rows. Metrica emits `"tracking"` only if the plan measures that its `start_time_s` equals the tracking frame time of `Start Frame` on the committed sample (plan Task 12); otherwise `"event"`. Every other converter (StatsBomb, Opta, Wyscout, Sportec/IDSSE) emits `"event"`. kloppy emits `"event"` for every dataset: kloppy's event model does not record whether a deserializer took its timestamps from tracking, and a guess per deserializer would be a provider branch the converter cannot verify (review B C4). Derived rows take the value of the row they come from; synthetic dribbles take `"event"` (their time is interpolated). The column joins `SPADL_COLUMNS` and the per-provider schema dicts after `cross_native`, is declared non-geometric in the reflection registry, and is carried by `convert_to_atomic` through `preserve_native`. The release step 0 of 6.3 reads it; a missing column means `"event"` for every row.

**D. ELASTIC episodes from `stoppage_state`.** In `tracking/_elastic_sync.py`, `_inplay_segments` reads `stoppage_state` when the frames carry it (`"live"` = in play; `"dead"` and `"unknown"` = not in play), and `ball_state` otherwise. ADR-130 built `stoppage_state` for exactly the providers whose frames record no dead-ball state. For frames from observed providers, `with_stoppage_state(spans=None)` copies `ball_state`, so their episodes do not change. GS `stoppage_state` comes from event-derived spans (`scripts/_loader_pining.py:638`), so the GS episodes move with 6.6 B's new event times; the re-stage recomputes them (review B C1). It applies to every ELASTIC user (`add_elastic_sync`, `link_actions_to_frames_elastic`), not only to this enricher. What it buys is measured, and it is modest (Section 5.1): on GS it cuts ELASTIC's gross mismatches (> 3 s from the truth) by 29 %, mostly by turning them into honest non-matches. It does not make ELASTIC accurate on GS: within 0.5 s it stays near 42 %. For GS and SkillCorner the release therefore uses the native time (step 0), not ELASTIC.

**What stays as it is.** ELASTIC's score (no location or time term; Section 5.1 rejected the location term), its parameters, and the time-based linker `link_actions_to_frames`. The linker reads `time_seconds`, so A and B move every GS and SkillCorner time-based frame link to the kick. The plan measures that shift (plan Task 12) and lists the frame-aware consumers it reaches.

## 7. Validation

### 7.1 Tests

- **Rule:** unit tests of the shared rule, testing every band from both sides: start line, box edges and wide edges, the mirrored next action, and the period boundary.
- **Converters:** one test per converter that a native cross with a short delivery becomes `pass`, and that a native pass delivered into the box from wide becomes `cross`.
- **Set pieces:** corner and free-kick tests both sides of the delivery condition.
- **Direction invariance:** the same delivery is typed the same way when the team attacks left-to-right and right-to-left, in both periods, for every converter. This includes kloppy and SkillCorner after their reorder.
- **Reorder regression:** kloppy and SkillCorner output on the existing fixtures is unchanged by the `to_spadl_ltr` move, apart from the new re-typing.
- **xCross label:** the label for each row class in the 6.5 table, the unknown-label exclusion, and the missing-column refusal. A frame whose horizon holds only a non-zone pass or a shot keeps a known label.
- **Restarts:** a wide delivery followed by an opponent goal kick, a corner, or a throw-in is a `pass` in the converters. A corner and a free kick followed by a restart (out for a goal kick; a foul in the box, then `shot_penalty`) are `_short`. The tracking layer types each of them from the path.
- **TF-46 corners:** `corner_kind` and `corner_kind_source` with and without `cross_attempt`, including the `<NA>` fallback and the GI refinement winning over a tracking decision. `build_corner_routines` selects a corner that tracking decided as `"direct"`, and skips one that tracking decided as `"short"`.
- **Passer exclusion:** an event time earlier than the true kick does not end the path at the passer's touch.
- **Release alignment:** a confident ELASTIC match moves the release to the aligned frame. A release whose ball is farther than `RELEASE_MATCH_M` from the action start makes the row unknown (`<NA>`), on both sides of the threshold. Precomputed `elastic_*` columns are read, not recomputed.
- **Order-insensitivity (ADR-065):** shuffled input in timestamp blocks gives the same re-typing, for every converter.
- **Bravery and causal:** each bravery output and the causal opportunity selection keep their meaning under the re-typing (6.2), atomic path included.
- **`cross_blocked` consumer:** the bravery block rate is the same before and after on a fixture where native crosses are re-typed.
- **Columns:** `cross_native` tests for each provider, and SkillCorner `<NA>`.
- **Enricher:** `cross_attempt` / `cross_height` tests for every row class of the 6.3 table, a blocked cross, a block contact on both sides of `BLOCK_WINDOW_S` (r7), a path without a contact (cap), partial ball coverage on both sides of `MIN_BALL_COVERAGE`, the `z_source == "estimated"` honest NaN, and orientation invariance.
- **Calibration (r7):** each Section 7.4 rule on a toy corpus, including that the release match ignores the GS and SkillCorner tails, that the release-gap sample is the matched rows, that a late block contact leaves the block sample, and the fixed-point check.
- **SkillCorner kick point (r8, 6.6 A):** a possession whose start and end differ puts the native row at `time_end` and `x_end`/`y_end`; a `start_type` interception and a keeper save stay at the start; a missing `time_end` gives a NaN time, never `time_start`; the carry from the reception to the kick becomes a synthetic dribble; the delivery scene still types correctly when the kick point, not the reception, is in the cross zone (both sides: a reception in the zone with the kick outside it is not a cross).
- **GS event time (r8, 6.6 B):** `time_seconds` equals `event_time − kickoff start_time` and is period-relative in every period, extra time included (it restarts near 0 after each kickoff; ADR-017's convention lock tests extend to GS); a period-1–4 action without its kickoff raises; a period-5 shootout row is anchored at the first period-5 `start_time`; a period with no converted action (the period-0 `G` marker) needs no anchor; an inversion pair (two events whose `startGameClock` order and `event_time` order disagree) comes out in `event_time` order with a chronological `action_id`, so `_assert_chronological_action_id` passes; a FOUL row with a null `startGameClock` gets its time from `event_time`; a converted row with a NaN `event_time` raises; a supplied input `time_seconds` does not change the output; order-insensitivity (ADR-065) still holds under a shuffled input. The pining loader's GS frames are period-relative (`periodElapsedTime`), and `validate_time_base` passes on a loaded GS match.
- **SkillCorner end coordinates (r8, 6.6 A, review B CD-SB-15; r9, CD-SB-23):** a pass without a targeted receiver and a clearance end at the next possession's reception point, not at the next kick, and the receiver's carry from that reception to their kick is a synthetic dribble (both sides: a targeted pass still ends at its targeted reception); at the last possession of a period the end falls back to `_derive_end_coordinates`; a shot, a `foul_suffered`, a `possession_loss` and an `unknown` row end at their start; the time is period-relative after the re-base in every period.
- **Native-release trigger (r8, 7.4):** the report's per-provider native passer-touch statistics, and the trigger on both sides of each bound (median ±0.30 s, share 40 %).
- **Driver `"native"` release source (r8, 7.4, review B CD-SB-16):** `calibrate` puts native rows into the release-gap sample and the fail share, and keeps IDSSE's `release_match_m` on its ELASTIC rows; the alignment table holds a `"native"` class and still partitions the rows. Both sides: a toy where the native rows would change a value if they were dropped.
- **`time_source` (r8, 6.6 C):** the column, its dtype and its vocabulary for every converter (`"tracking"` for GS and SkillCorner native rows, `"event"` elsewhere and on synthetic dribbles); atomic opt-in carry; the reflection declaration; `_finalize_output` keeps it.
- **Release order (r8, 6.3 step 0):** a `"tracking"` row uses `time_seconds` even when a confident `elastic_frame_id` points elsewhere (both sides: the same row as `"event"` uses ELASTIC); a missing `time_source` column means `"event"`; the release-match check still applies to native rows.
- **ELASTIC episodes (r8, 6.6 D):** frames with `stoppage_state` split episodes at `"dead"` and `"unknown"`; frames without it use `ball_state` as today; a GS-shaped case where a restart event lies inside its own `"unknown"` span (`tracking/_stoppages.py:14`) is assigned to the nearest live episode (`_assign_segments`, `tracking/_elastic_sync.py:718-735`), asserted on both sides of the span midpoint (review A CD-SPEC-14); the committed CC-BY oracle slice (`tests/tracking/test_elastic_sync_oracle.py`) keeps its floors.

### 7.2 Choosing `CROSS_START_X` (pre-declared)

- For each candidate in {52.5, 60, 65, 70, 75, 80}, compute the F1 between the rule and `cross_native` among open-play passes for each vendor with a full native cross label:
  - StatsBomb open data (`pass.cross`);
  - the GS matches of the xshot187 corpus.
- Choose the candidate with the highest mean F1, equal weight per vendor (decision 8). On a tie within 0.005, choose the larger value.
- IDSSE is left out of the mean: its native label covers about 4 crosses per match. Its F1 is reported.
- Record every vendor's F1 table in `docs/research/cross-definition/`. A disagreement between vendors is reported, not tuned away.
- **r7 (owner ruling 2026-11-14):** 75 and 80 join the candidates. The r6 grid ended at 70, and the first DGX run picked 70 with F1 still rising at every step (StatsBomb 0.596 → 0.627, GS 0.435 → 0.449). A maximum at the edge of the grid does not show that the grid holds the optimum. 80 is still 8.5 m short of the penalty area (88.5).

### 7.3 Corpus evidence (DGX, full xshot187 corpus)

- The median geometric crosses per match must agree within a ratio of 1.5 (max over min across GS, IDSSE and SkillCorner). The audit measured 21 / 16.5 = 1.27.
- Report the `cross_attempt` counts and the blocked share for each provider.
- Report `corner_crossed` and `freekick_crossed` medians per provider, before and after.
- Report, per provider, the restart-ended deliveries from the cross zone, and how many of them the tracking layer types as attempts. This is the event-only undercount (decision 7).
- Report, per provider and set-piece kind, the restart-ended set pieces, the event rule's recall against `cross_attempt`, and the `corner_kind_source` split (decision 10).
- Report, per provider and row class (open play in the cross zone, corner, free kick), the shares that are aligned by ELASTIC, unaligned but matched, misaligned (`<NA>`), and unanswerable for coverage. An xCross label or a `corner_kind` decided by tracking is only as good as these shares.
- The audits in Section 5 and decision 10 used approximate path ends (the next action, or 4 s). The 7.3 numbers use the implemented path (first non-passer contact). ADR-133 and the CHANGELOG cite the 7.3 numbers.

### 7.4 Calibrating `CrossDeliveryParams` (pre-declared, r7)

**Why r7 changes the rules.** The plan's r3 rules took 95th percentiles over every ELASTIC-confident row (`elastic_confidence >= 0.5`), pooled with equal weight per provider. Run 3 on commit 1 `4490016c` (line 70, provisional parameters) gave `release_match_m` 15.23 m and `block_radius_m` 20.43 m. Both measure alignment and detection error, not football:

| Sample (run 3, line 70) | Median | 95th percentile |
|---|---|---|
| ELASTIC release-to-start distance, IDSSE | 1.6 m | 5.0 m |
| the same, Gradient Sports | 2.0 m | 21.0 m |
| the same, SkillCorner | 2.5 m | 15.9 m |
| GS blocked crosses, opponent first contact (n = 49): distance from the release | 4.05 m | 20.43 m |
| the same: time after the release | 0.63 s | 3.30 s |

Of those 49 contacts, 32 come 0.17–0.90 s after the release; the next comes at 1.37 s.

With the r3 values, a row whose ball is 15 m from the event start counts as matched, and an opponent touch 20 m away counts as a block. Run 2 used the r3 values of run 1 (14.60 m, 20.43 m, 0.2002 s) at line 70: there the IDSSE blocked share of attempts is 22 %, against 8 % in run 3 with the provisional parameters. xCross label v2 reads `cross_attempt`, so these values would flow into training labels. The owner ruled (2026-11-14) to replace the rules, not to adopt the values.

**The rules.** Every sample is the open-play and set-piece rows to which `cross_attempt` applies, at the chosen start line.

| Field | Rule |
|---|---|
| `release_match_m` | the 95th percentile of `release_dist_m` over the IDSSE ELASTIC-confident rows. IDSSE is the clean-sync reference by measurement: the three medians are close (1.6, 2.0 and 2.5 m), but only IDSSE has no long tail (95th percentile 5.0 m, against 21 m for GS and 16 m for SkillCorner). The GS and SkillCorner 95th percentiles are reported, not used. |
| `release_gap_s` | the 95th percentile of `passer_contact_dt`, pooled with equal weight per provider, over the ELASTIC-confident rows whose `release_dist_m` is at most `release_match_m`. Floor: one frame at the highest provider frame rate. |
| `block_radius_m` | the 95th percentile of `first_contact_dist_m` over the GS native crosses with `cross_blocked == True` whose first contact is an opponent within `BLOCK_WINDOW_S` of the release. Only GS carries a known `cross_blocked` in the corpus; the memo says so. |
| `direction_window_s` | two frames at the lowest provider frame rate (unchanged). |
| `min_ball_coverage` | 0.8, unless more than 20 % of a provider's rows fail it; then the owner decides (unchanged). |
| `block_window_s`, `path_cap_s` | fixed at 1.0 s and 4.0 s, reported only. |

**Owner triggers (r7, review B R7-B-02).** The report flags `release_match_owner_decision_needed`, and the owner decides before the runs go on, when either holds:
- the IDSSE sample has fewer than 200 ELASTIC-confident rows with a finite release distance, which is the percentile's own sample (run 3: 766 rows, all 766 finite);
- the release match turns more than 20 % of a provider's ELASTIC-confident rows with a finite release distance into `<NA>` (the same bar as `min_ball_coverage`). A row without a finite distance is `<NA>` whatever the match is, so it is not counted. The report gives this share per provider. On the run-3 rows, the IDSSE match (5.02 m) turns 25.3 % of GS and 29.5 % of SkillCorner finite-distance rows into `<NA>` (IDSSE 5.1 %), so the trigger is expected to fire in run 2 (review A C11).

**The runs.** Each value depends only on what an earlier run fixed. All runs are on one clean commit, with the full corpus.
1. Provisional parameters: the start line (Section 7.2).
2. `--cross-start-x`: `release_match_m` and `release_gap_s`. No tracking parameter changes their samples.
3. Plus those two: `block_radius_m`. Its sample exists only for rows that pass the final release match, and the contact search starts at the final release gap. The block window is measured from the release.
4. Plus `--block-radius-m`: the 7.3 evidence. The report's `fixed_point` entry must be true for every calibrated field and for `cross_start_x` (the run used the selected start line; review A C9). If it is not, the run stops and the result goes to the owner. The `min_ball_coverage` owner trigger is read on this run, because the path end depends on the final release match and release gap.

Each override is passed exactly as the earlier report prints it (full float precision). `fixed_point` compares with a relative tolerance of 1e-9, so a rounded override gives a false stop (review B R7-B-03).

The ratio bar of Section 7.3 applies to run 4.

**r8: the runs move to the commit that carries Section 6.6, on a re-staged corpus.** The staged xshot187 `_actions` came from the old converters, so the SkillCorner and GS rows carry the old start points and times. The driver re-types staged actions (Section 7.2 header); it cannot recover the kick point or the event time. The `_actions` are therefore re-staged on the DGX from the pining cache with the r8 converters before run 1, and the runs read the re-staged corpus. The rules of this section keep their meaning, with one extension for the native release: the diagnostics' `release_source` gains the value `"native"` (step 0 of 6.3), and every rule that reads "ELASTIC-confident rows" reads "rows with a native or an ELASTIC-confident release" (`release_source` in `{"native", "elastic"}`). `release_match_m` still comes from IDSSE, whose rows are all `"event"`-timed and so ELASTIC-aligned; the owner triggers still apply; the release-gap sample and the fail share include the native rows. The 7.3 alignment shares (`alignment_table` in the driver) gain the class `"native"` (answered; release at the native event time).

**r8 owner trigger `native_release_owner_decision_needed` (review B CD-SB-13).** The report gives, per provider, the native rows' passer-touch share and the 5th/50th/95th percentiles of `passer_contact_dt`. The trigger fires when a provider's native rows have a median `passer_contact_dt` outside [−0.30 s, +0.30 s] or a passer-touch share below 40 %. Disclosure: the author chose both bounds after seeing the preview (GS −0.23 s and 50.4 %; SkillCorner −0.20 s and 78.4 %); they test that the re-staged corpus keeps that relation, and do not tune it. The trigger is read on run 2 and on run 4, like the release-match trigger. Section 5.1 predicts that the release-match trigger no longer fires: in the native-time preview, 1.2 % of GS rows and 4.3 % of SkillCorner rows lie beyond 4.67 m. The plan states this as a prediction; run 2 measures it.

## 8. Non-goals and anti-goals

- **Non-goal:** the xCross refit, the other owed retrains, and their adoption. Part Deux owns them, after this release and a restage of the xshot187 `_actions` (decision 6).
- **Anti-goal:** tuning the rule until one vendor's labels match. The vendor labels are a benchmark, never the target. The one fitted value, the start line, comes from a pre-declared mean over vendors (7.2).
- **Anti-goal:** a provider-specific branch inside the shared rule.
- **Non-goal (r8):** changing ELASTIC's score. The spike measured a location term and rejected it (Section 5.1): it lowered IDSSE ground-truth accuracy at every weight. The owner ruled on 2026-11-14 on the option that excluded it (review A C19). 6.6 D changes only how ELASTIC splits episodes.
- **Anti-goal (r8):** a provider name inside the enricher or ELASTIC. The release order reads the row's `time_source`; ELASTIC reads the frames' `stoppage_state`. The converters, which own the provider facts, set both.

## 9. Constraints

- **Converters stay pure** and order-insensitive (ADR-065). The rule runs after sorting and is deterministic.
- **The new `add_*` enricher follows the house rules:** pure (`PURITY_ENTRIES`), NaN-safe (ADR-003), one call shape (`test_call_convention_registry`), `metric_contracts` registration (ADR-098), the `feature_glossary` entry, an SB360 freeze-frame verdict (ADR-053), and aggregator liveness. No `*_xfns`: the enricher reads the delivery's own outcome (ADR-030/047).
- **Attribution and registries:**
  - `NOTICE` and the function's References docstring cite Vidal-Codina et al. 2022 (ADR-005).
  - The reflection registry (`reflection.py`, `tests/test_reflection.py`, ADR-045) declares `cross_native`, `cross_attempt` and `cross_height` as non-geometric.
  - The enricher gets a TF id from `TODO.md`, a row in the AGENTS.md TF table, and its place in the C4 model (ADR-048).
- **Dtype-safe id comparisons** (`id_compat`) for the team comparison in the next-action lookup.
- **Geometry constants come from `spadlconfig`** (ADR-050).
- **A public seam changes, so the plan enumerates every caller with evidence on both sides.** Four seams change. The lists below are the floor (`git grep` at `0cc07e6c`, and review B). The plan's sweep covers each seam in full.
  1. **The open-play `cross` type.**
     - SPADL core: `spadl/base.py:18`, `spadl/config.py:53`, `atomic/spadl/base.py:99`.
     - Valuation: `vaep/features/specialty.py:47,130,140`, `atomic/vaep/features.py:404`, `xthreat/_grid.py:132`, `xthreat/_model.py:97`, `xtgk/_moves.py:31`, `calibration/_carrier_objective.py:33`, `causal/opportunities.py:96-103`.
     - Tracking: `tracking/_xcross_attempt.py`, `_run_values.py:419`, `_packing.py:33`, `_elastic_sync.py:380`, `_line_breaking.py:37`, `features.py:5787`, `_model_eval.py:728`, `defensive_credit/_rules.py:26,52,456,478`, `defensive_credit/_params.py:54` (`ANCHOR_CROSS`), `defensive_credit/_bravery.py:17`, `atomic/tracking/features.py:266,305,618,661`.
     - Scripts: `scripts/_rq_corpus.py:16`, `build_sb360_coverage.py:56`, `validate_sb360_licensed_corpus.py:52`, `_toy_corpora.py:607`, `train_receiver_model.py:39`, `_receiver_validation.py:31`, `measure_covariate_invariance.py:231`, `probe_sb_cross_blocked.py:37`.
     - Readers of `pass` ∪ `cross` are recorded as unaffected only with evidence that they read the union.
     - **Readers of `pass` alone** (rows move both ways between `pass` and `cross`):
       - Value-changing: `expected_passing/_model.py:26,133` (`PassCompletionModel`, bundled `weights/`, so a retrain is owed), `scripts/train_pass_completion.py:48`, `team_metrics/_pressing.py:93,141`, `team_metrics/_buildup.py:101`, `team_metrics/_progression.py:159`, `territory/_compute.py:139`, `defensive_credit/_line_break_signal.py:61`, `scripts/validate_territory_counterfactual.py:413,587`, `scripts/hf_jobs/headtohead.py:503`.
       - Unaffected only with evidence: `_gk_resolve.py:89`, `causal/_confounders.py:93`, `xsuccess`.
  2. **The set-piece types** (`corner_*`, `freekick_*`): `corners/_identify.py`, `team_metrics/_possession.py`, `coordination/_windows.py`, `tracking/_stoppages.py`, `tracking/_ghost_gk.py`, `tracking/_gk_geometry.py`, `vaep/formula.py`, `spadl/utils.py`, `atomic/spadl/utils.py`, `atomic/spadl/base.py`, `tracking/_elastic_sync.py`, `tracking/_packing.py`, `tracking/defensive_credit/_bravery.py:18`, `scripts/build_skillcorner_s1_event_validity.py`, `scripts/_receiver_validation.py`.
  3. **The new column `cross_native`:** `spadl/schema.py` (`SPADL_COLUMNS` and the per-provider dicts), `_finalize_output`, `convert_to_atomic(preserve_native=…)`, `atomic/tracking/features.py:330,357,433,471,477` (atomic `compute_bravery`).
  4. **`cross_blocked` applicability and the bravery cross reads:** `defensive_credit/_bravery.py:64,65,68,113-114` (6.2), `scripts/probe_sb_cross_blocked.py`.
  5. **SkillCorner action start and time (r8, 6.6 A).** Every reader of SkillCorner SPADL `start_x`/`start_y`/`end_x`/`end_y`/`time_seconds` and every time-based frame link on SkillCorner actions (`link_actions_to_frames`, defined in `tracking/utils.py`, and its 26 calling modules). The new required inputs `time_end`, `x_end`, `y_end`. The lakehouse imports the private `skillcorner._PERIOD_START_SECONDS` in 4 places (review B). `docs/PRIVATE_CONSUMERS.md` has no entry for it today; the plan adds one (review B CD-SB-17). The constant does not change. The plan sweeps the readers and records which are value-changing.
  6. **GS `time_seconds` (r8, 6.6 B).** The same sweep for GS: every time-based frame link, every time delta (VAEP time features, possession gaps; GS does not call `_add_dribbles`), the pining loader's events-only path and its frame clock. The new required inputs `event_time`, `start_time` and the kickoff rows; the input `time_seconds` stops being required. The GS `action_id` order. The ADR-130 GS stoppage spans (`event_stoppage_spans`) and every reader of `stoppage_state` on GS frames (`training_frame_mask` for xShot and xCross). The lakehouse's GS adapter subtracts the period offsets from `startGameClock` before the converter (`spadl_adapter.py:646, 716-720`); under 6.6 B that normalization no longer reaches the output, and the lakehouse must pass `event_time`, `start_time` and the kickoff rows.
  7. **The new column `time_source` (r8, 6.6 C):** the same registries as `cross_native` (seam 3).
  8. **ELASTIC episodes (r8, 6.6 D):** `align_events_to_frames`, `add_elastic_sync`, `elastic_sync_xfns`, `link_actions_to_frames_elastic`, and every caller that passes frames with `stoppage_state`.
- **r8 re-stage and commits.** The xshot187 `_actions` are re-staged on the DGX from the pining cache with the r8 converters (Section 7.4). Publishing the re-staged dataset (a Hub upload or `datasets record`) needs the owner's go; the calibration runs read the DGX copy. The commits become four, under the provenance exception: commit 3 holds the r8 code, tests and docs; the four runs read the clean commit 3; commit 4 holds the values, the memos and the release. Each commit and each push has its own owner-approval gate in the plan, immediately before it (review B C3).
- **Trained artifacts.** The plan lists every bundled or Hub artifact whose training data contains SPADL `type_id`, for example VAEP models and xCross. For each one, it states whether the value change makes a retrain owed. Owed retrains are recorded in the CHANGELOG and handed to Part Deux (decision 6).
- **Release.** The change alters values, so it is a minor version bump. The CHANGELOG carries a Hyrum notice that lists every change:
  - The SPADL output schema gains `cross_native` for every converter. Atomic SPADL carries it opt-in through `preserve_native`.
  - Open-play `pass` counts change too, so `PassCompletionModel` and other `pass` readers shift (Section 9, seam 1).
  - Open-play `cross` and `pass` counts change for every provider; rows move between the two types.
  - `corner_crossed` / `corner_short` and `freekick_crossed` / `freekick_short` change for every provider. This supersedes the ADR-127 discriminator. SkillCorner's event-only corner typing drops from 62 % to about 22 % `corner_crossed`; on tracking, `cross_attempt` and TF-46 `corner_kind` decide instead (decision 10).
  - TF-46 output gains `corner_kind_source`. The `corner_kind` domain is unchanged; with tracking, more SkillCorner corners become `"direct"` and enter routine building.
  - `cross_blocked` applies where `cross_native` is True, not where the type is `cross`.
  - The xCross label is version 2, and the retrains are owed.
  - r8: SkillCorner native rows move from the reception to the kick (start point and time); shots, fouls and non-actions end at their new start; SkillCorner gains synthetic carries; `time_end`, `x_end` and `y_end` become required inputs. GS `time_seconds` moves from the whole-second start clock to the possession event's own time, period-relative (a median of 1.59 s later); `event_time`, `start_time` and the kickoff rows become required inputs, and a caller's own `startGameClock` normalization (the lakehouse's) no longer reaches the output. The pining loader's GS frames move from the match clock to the period-relative clock. GS and SkillCorner `action_id` renumber (GS: the new order; SkillCorner: the new order and the new carries), an ADR-065 cross-table migration for every stored table keyed by `action_id` (review A CD-SPEC-15 a). GS `stoppage_state` comes from the action times (`event_stoppage_spans`), so the GS dead-ball spans and the xShot/xCross `training_frame_mask` on GS move too (CD-SPEC-15 b). Both move every time-based frame link for those providers. SPADL gains `time_source`. ELASTIC episodes change for frames that carry `stoppage_state`. Every artifact trained on GS or SkillCorner SPADL or on frames linked to their actions becomes owed a retrain; the plan lists them with evidence.
  - Known downstream consumer: the luxury-lakehouse reads `type_id` and `cross_blocked` (`src/ingestion/bravery_writer.py`) and keeps per-column shift bands for adoptions (`src/tests/action_context/expected_shift_oracle.py`). The notice names the columns that shift.

## 10. The author's calls (to check while reading)

- In the converters, the "next on-ball action" excludes `non_action` and `foul`. Synthetic dribbles do not exist at the run position.
- `cross_native` is `<NA>` for SkillCorner, because its old rule was ours.
- `cross_blocked` keeps its meaning and is keyed to `cross_native`.
- `BLOCK_RADIUS_M`, `DIRECTION_WINDOW_S`, `RELEASE_GAP_S`, `MIN_BALL_COVERAGE` and `RELEASE_MATCH_M` are set by the Section 7.4 rules (r7). The 4 s path cap and the 1.0 s `BLOCK_WINDOW_S` are fixed.
- r7: IDSSE alone calibrates the release match, because it is the one provider whose event-to-ball distance has no long tail (run 3). This is a measurement, not a claim about how each provider builds its events: SkillCorner's events are derived from its tracking, and they still show a 16 m tail. The IDSSE sample is smaller (7 matches; 766 rows).
- The release is aligned through ELASTIC (TF-43), with a release-match check on every row. r8: a row whose `time_source` is `"tracking"` skips ELASTIC.
- r8: `time_source` is a per-row SPADL column, not a converter-level flag or an enricher keyword. The data carries its own provenance (ADR-054), every frame-aware consumer can read it, and the enricher stays free of provider names. The cost is one more SPADL column and its registry entries.
- r8: SkillCorner's start-type derived rows (interceptions, recoveries) and keeper saves stay at the possession start, because they happen there. Their 0.01 s offset before the possession's start time (`spadl/_skillcorner_inference.py:104,168`) stays. It cannot invert across possessions: in 106,346 possessions of 98 matches no `time_end` equals or passes the next possession's `time_start` (`r8_checks.py`; 0.1 s resolution). Inside a zero-length possession (24 % of them) the derived row still comes 0.01 s before the native row (review A C16).
- r8: a GS period 1–4 that holds a converted action but no kickoff event raises. None of the 64 cached matches has one, so a silent fallback would only hide a data defect. A period-5 shootout is anchored at its first event, and a period without converted actions needs no anchor (6.6 B).
- r8: `time_source` has dtype `object` with a closed vocabulary, like `result_source` and `action_provenance`, not a pandas category (review A C15): SPADL holds no category column today, and the two existing provenance columns set the house form.
- r8: Metrica gets `"tracking"` only if the plan measures it on the committed sample; the default is `"event"`.
- r8: SkillCorner restart rows start at the taker's `x_end`, the provider's own kick position. Snapping corners to the corner spot would be a provider branch that 5.1 does not justify (corner rows beyond 4.67 m: 7.5 %).
- The corpus agreement bar is a ratio of 1.5.
- Causal opportunities select by `xcross_label_version`.

## 11. References

- Anzer, G. et al. (2025). The Common Data Format (CDF). arXiv:2505.15820.
- Vidal-Codina, F., Evans, N., El Fakir, B., Billingham, J. (2022). Automatic event detection in football using tracking data. arXiv:2202.00804.
- Wyscout data glossary, "Cross": https://dataglossary.wyscout.com/cross/
- Opta event definitions: https://www.statsperform.com/opta-event-definitions/
