# One cross definition for every provider — design

- **Status:** design approved by the owner and by independent reviews A and B (2026-11-13 to 2026-11-15). This text restates that design in its final state (owner ruling 2026-11-15); the decision trail is in ADR-133, ADR-134 and the review reports `D:\Development\_reviews\2026-11-1[345]-cross-definition-*`.
- **Branch:** `feat/cross-definition`, off `main` `0cc07e6c`.
- **ADRs:** ADR-133 (this design), ADR-134 (the ball-contact detector v2; its own spec `2026-11-15-tf79-contact-detector-v2-design.md`).

## 1. Intent

The SPADL `cross` type meant different things for different providers. On the staged xshot187 corpus (187 matches), the open-play crosses per match were:

| Provider | Native rule | Median crosses per match |
|---|---|---|
| SkillCorner | our zone rule `_is_cross`: every wide pass in the final third | 101 |
| Gradient Sports (GS) | the provider's annotation (`OTB`/`CR`) | 26.5 |
| IDSSE (Sportec) | `play_height == "cross"` or `play_flat_cross` | 4 |

Shots per match agree (20–25), so the matches are comparable; only the cross label differs, by about 25x. Because of this, the xshot187 xCross refit failed the ADR-132 per-provider calibration gates. The goal: one definition of a cross, used the same way for every provider, that does not depend on how a vendor labels its events.

## 2. What a cross is (owner's coaching definition)

- It is aimed at the penalty area. "Early service" counts: a ball from further back, played at the front of the box.
- It usually comes from wide, outside the width of the penalty area.
- It is in the air. A ground cutback is not a cross; a lifted cutback for a header is one.
- Its intent is an immediate chance, not keeping the ball, building up or switching play.

## 3. Prior art

- No vendor-independent standard exists; the Common Data Format paper (Anzer et al., arXiv 2505.15820) uses the cross as its example of providers disagreeing.
- Wyscout: "a ball played from the offensive flanks aimed towards a teammate in the area in front of the opponent's goal" (high, low and blocked). Opta: a cross is separate from a pass; a pulled-back pass inside the box is a pass.
- Vidal-Codina et al. 2022 (arXiv 2202.00804) detect a cross from tracking: a release in a cross zone, the next player in control in the penalty area, at least one attacker in the box.

## 4. Decisions (owner)

1. **Two layers.** The converters use one geometric rule for every provider (6.1). The tracking layer adds the cross attempt and its height (6.3). The provider's own label stays as `cross_native` (6.2).
2. **The xCross attempt label counts all heights.** Height is an attribute, not a filter.
3. **A cross cut out by an opponent near the release is unknown, not an attempt.** The tracking layer cannot tell a block from an opponent's touch near the crosser (5.2), so it claims neither outcome (6.3 case 2).
4. **`cross_height` is an honest `<NA>` where the ball height is only estimated** (SkillCorner `z_source == "estimated"`).
5. **Set-piece deliveries use the same delivery rule** (6.1b).
6. **Retrains are owed, not done in this cycle.** Part Deux runs them after the release (Section 9).
7. **A restart-ended delivery is a `pass` in the converters** (6.1 condition 4); the tracking layer decides whether it was an attempt.
8. **The start line comes from the mean F1 over vendors with a full native cross label** (StatsBomb and GS, equal weight; 7.2).
9. **`cross_native` also keeps the provider's label on set pieces; the bravery cross reads move to `cross_native == True`; an unknown xCross label is excluded from training; `cross_height` reuses the ADR-125 aerial threshold (1.8 m); the tracking path ends at the first detected ball contact.**
10. **Tracking decides set pieces.** `cross_attempt` also covers corner and free-kick rows; TF-46 corners and the xCross set-piece runs read it when frames exist. The converters keep the one event rule; each provider's event-only recall is recorded as a limit (6.4).
11. **Native event timing.** Where a provider times its events from its own tracking clock (GS, SkillCorner), the SPADL row carries that time, and the release uses it instead of ELASTIC (5.1, 6.6).
12. **The ball-contact detector is redesigned in this cycle** (ADR-134): kinematic candidates, a reach gate without fallback, the actor skip, thresholds chosen on a GS development half and scored once on the held-out half against a pre-declared bar. After the detector is tuned, the block call is re-measured against PFF code B and reported to the owner, who may revisit decision 3 on that evidence.

## 5. Evidence

The audit scripts are in `D:\Temp\cross-def\`; their outputs are on the DGX in `~/cross-def/`. All runs are read-only.

**The geometric rule (audit 2, start line 65 m; the end is the next on-ball action's start):** median crosses per match GS 16.5, IDSSE 19, SkillCorner 21 (ratio 1.27). Taking the next on-ball action's start, not the provider's end, removes a provider effect: SkillCorner has no ball end for an untargeted pass. Moving the start line from 52.5 m to 70 m changes the medians by 1.5–3. Ball height over the path (1.5 m threshold): GS 52 % above, median peak 1.62 m; IDSSE 62 %, 3.13 m; SkillCorner (estimated z) 43 %, 0.75 m, which supports decision 4.

### 5.1 Release timing

The release must be the kick. Two provider defects put it elsewhere:
- **SkillCorner placed each action at the possession start** (the reception). The kick is at the possession end (`time_end`, `x_end`/`y_end`). On 98 matches, ELASTIC's release lies a median 0.85 m from the raw kick point; the gap to the SPADL start was the carry (a median 8.1 m on the misaligned rows).
- **GS took a whole-second clock at the start of the on-ball sequence** (`startGameClock`), while each possession event carries its own `eventTime`. Against PFF's own frame synchronisation (86,067 events, 62 matches), `eventTime` minus the period's kickoff `startTime` is within 0.067 s (99th percentile); the old `time_seconds` was a median 1.59 s early.

ELASTIC adds error for these two providers. GS and SkillCorner frames are "alive" on every frame, so each half is one ELASTIC episode; against PFF truth, ELASTIC lands within 0.5 s for 41.6 % of 71,623 GS pass and cross rows. Splitting episodes on `stoppage_state` cuts its gross mismatches (more than 3 s off) by 29 %, but not its accuracy. A location term in ELASTIC's score lowered its accuracy on the CC-BY IDSSE benchmark at every weight, so it was rejected. IDSSE events are hand-timed, and ELASTIC stays the right tool for them. The ball's own kick onset matches PFF's release time (median 0.000 s on 54,534 GS releases).

### 5.2 Block detection

Joined to PFF's `crossOutcomeType` on 847 GS open-play crosses (167 blocked B, 177 complete C, 503 defended D), the v1 tracking block call had precision 0.14 against a base rate of 0.20 and recall 0.20. An early opponent contact (within 1 s, about 4 m from the release) appeared almost as often on completed crosses as on blocked ones. No radius or contact-strength gate gave the call signal (best: radius 1.5 m, precision 0.45, recall 0.10). The cause was the v1 detector (ADR-134): it credited the nearest player within 5 m when nobody was within 2 m, and path-shape vertices of a ball in flight became contacts. PFF times the block itself a median 0.10 s after the cross (59 of 167 before 0.08 s).

## 6. Design

### 6.1 Converter layer: the geometric `cross`

One shared function, `classify_deliveries` in `silly_kicks/spadl/base.py`, re-types the open-play pass-class rows of every converter. A row is a `cross` when all of these hold:

1. **Open play.** Its type is `pass` or `cross` before re-typing.
2. **Wide start.** `start_y` lies outside `W/2 ± penalty_area_half_width` (`spadlconfig`, 20.16 m).
3. **Advanced start.** `start_x >= CROSS_START_X` (chosen by 7.2 from {52.5, 60, 65, 70, 75, 80}).
4. **Delivery into the box.** The start of the next on-ball action in the same period lies in the penalty area, in the passer's frame:
   - the next on-ball action is the next row in the converter's chronological order that is not `non_action` or `foul` (ties keep the converter's own stable order, ADR-065);
   - when it is a restart (a type in `_SET_PIECE_RESTART_TYPE_NAMES`, `shot_penalty` or `shot_freekick`), the event data has no delivery end, so the row is a `pass` (decision 7);
   - when its team differs from the passer's, it is mirrored into the passer's LTR frame, `(L − x, W − y)`;
   - the penalty area is `x >= field_length − penalty_area_depth` and `|y − W/2| <= penalty_area_half_width`.

A row that meets 1 but not 2–4 becomes a `pass`. The rule reads only SPADL columns, so it is the same for every provider.

### 6.1b Set-piece deliveries

The same function re-types corners and free kicks with condition 4 alone: `corner_crossed` / `freekick_crossed` when the next on-ball action starts in the penalty area, `_short` otherwise. A restart-ended set piece is `_short`, the event-only limit of decision 10. Penalties, throw-ins, goal kicks and `shot_freekick` do not change. This supersedes the Sportec `play_height` corner discriminator (ADR-127).

**Where it runs.** After `to_spadl_ltr` (canonical LTR), in final chronological order, before `_add_dribbles`, in all eight converters. kloppy and SkillCorner move `to_spadl_ltr` before `_add_dribbles`; `_add_dribbles` reads only same-team pairs, so the move changes nothing but the re-typing (a regression test proved it on the existing fixtures). Each converter's native rule stops deciding the type and moves to `cross_native`; SkillCorner's `_is_cross` is deleted, because it was our rule.

### 6.2 `cross_native` and `cross_blocked`

`cross_native` is a nullable `boolean` SPADL column: the provider's own "crossed" label for every delivery 6.1 or 6.1b re-types (open play: crossed or not; set pieces: crossed or short), `<NA>` elsewhere and for every row of a provider without a native label (all SkillCorner rows). It joins `SPADL_COLUMNS` and the per-provider schema dicts; atomic SPADL carries it opt-in through `convert_to_atomic(preserve_native=…)`.

`cross_blocked` (ADR-046) keeps its meaning, keyed to the provider's cross: applicable where `cross_native` is True on an open-play row, `<NA>` elsewhere. Bravery (`defensive_credit/_bravery.py`, atomic `compute_bravery`) reads `cross_native == True` for every cross count, so its outputs keep their denominators.

### 6.3 Tracking layer: `cross_attempt` and `cross_height`

`add_cross_deliveries(actions, frames, *, params)` (TF-86, `tracking/_cross_delivery.py`) adds two columns. It is pure, NaN-safe and frame-consuming (Section 9), and runs on finalized actions, one match per call.

**The release.**
0. A row whose `time_source` is `"tracking"` (6.6 C) is released at its own `time_seconds`; ELASTIC is not consulted.
1. Otherwise the confident ELASTIC frame (`elastic_confidence >= 0.5`; precomputed `elastic_*` columns are read, else `align_events_to_frames` runs once).
2. Otherwise `time_seconds`.
3. For every row, the ball at the nearest frame (within half a frame period) must lie within `release_match_m` of the action start; otherwise the path is unanswerable (`<NA>`).

**The path** runs from the release to the first resolved contact (TF-79, ADR-134) by a player other than the passer, at or after `release + release_gap_s`, capped at `release + path_cap_s` (4 s). `release_gap_s` guards the release frame: at the kick, the ball can be in reach of a defender closing the crosser down, and a contact there would end the path at the release. The passer's own contacts are skipped by player, not by the gap. The path is answerable when finite ball positions cover at least `min_ball_coverage` of the frames the ball's frame rate implies.

**`cross_attempt`** (nullable `boolean`) by row class:

| Row class | Value |
|---|---|
| open-play `pass`/`cross` released in the cross zone (6.1 conditions 1–3), or a corner / free-kick row; path answerable | `True`, `False` or `<NA>` (cut out) by the test below |
| the same rows, path unanswerable | `<NA>` (unknown) |
| open-play `pass`/`cross` whose start is known to lie outside the cross zone | `False` |
| every other row | `<NA>` (not applicable) |

The test, in order:
1. `True` when the path enters the penalty area.
2. `<NA>` (cut out) when the path does not enter it, the first contact is an opponent's within `cut_out_window_s` (1.0 s) of the release, and the initial direction points into the penalty area or cannot be resolved. The initial direction is the ball's movement over the frames within `direction_window_s` after the release, strictly before the first contact when that comes earlier; fewer than two finite frames leave it unresolved. The ball was heading into the box and an opponent ended the path before it got there; where it would have gone is unknown, and the detector cannot tell a block from a touch near the crosser (5.2).
3. `False` otherwise. An opponent's first contact after `cut_out_window_s` stays `False`: on GS most are real interceptions (a median 2.0 s and 27 m from the release).

All geometry is in the action-LTR frame (ADR-028). The diagnostics (`_cross_delivery_diagnostics`) expose `cut_out`, `direction_cut` and `direction_unresolved`.

**`cross_height`** (category `"aerial"`/`"ground"`), set only where `cross_attempt` is True: `"aerial"` when the peak ball z over the path is above `BallContactParams.aerial_z_m` (1.8 m); `<NA>` unless the ball `z_source` is `"measured"`.

### 6.4 The event-only limit and TF-46

Without tracking, a blocked cross and a restart-ended delivery are `pass` (a restart-ended set piece `_short`), and SkillCorner's events miss defenders' touches in the box (its event rule finds 43 % of corner deliveries, GS 94 %). `classify_deliveries` and each converter docstring state these limits and the provider's measured recall. The tracking layer covers the restart-ended deliveries and the in-box touches; a cross cut out near the release stays `<NA>` there too.

**TF-46 corners.** `identify_corners` sets `corner_kind` (domain `{"direct", "short", "unclear"}`, unchanged) from `cross_attempt` when the actions carry it (True → `"direct"`, False → `"short"`); where it is `<NA>` or absent, the SPADL type decides. The GI refinement then runs and may set `"unclear"`. The new column `corner_kind_source` (`"tracking"`, `"event"`, `"gi"`; closed vocabulary `CORNER_KIND_SOURCE_VALUES`) records what decided (ADR-054). The first-contact winner skips the taker (`taker_player_id`, ADR-134).

### 6.5 The xCross label and causal opportunities

`build_xcross_labels` labels a frame 1 when a same-team row of a selected kind with `cross_attempt == True` falls in the horizon (`"cross"`: open-play pass-class rows; `"corner_crossed"`, `"freekick_crossed"`: the set-piece kinds). A frame whose horizon holds no attempt but an unknown row of a selected kind (unanswerable or cut out) is unknown, excluded from training and evaluation and counted per match (`n_unknown_label_frames`); a not-applicable `<NA>` never makes a frame unknown. Missing `cross_attempt` raises `ValueError` naming `add_cross_deliveries`. Model metadata records `xcross_label_version` (2; absent = 1).

Causal opportunities select by `xcross_label_version`: version 2 marks the treatment by `cross_attempt == True`. A spell whose exposure window holds no attempt but an unknown row of a selected kind has an unknown treatment: it is excluded from the opportunity table and counted per match (`attrs["n_unknown_treatment_spells"]`), never `Z = 0`.

### 6.6 Native event timing (decision 11)

Each change is provider-neutral in the shared code; the provider facts live in the converter that owns them.

**A. SkillCorner: the action at the kick.** Every native `player_possession` row takes its `time_seconds` from `time_end` and its start from `x_end`/`y_end` (same parse, period re-base and transform as before). Rows that happen at the possession start keep `time_start` and `x_start`/`y_start`: the `start_type` interceptions and recoveries, and the keeper saves. `_add_dribbles` then adds the reception-to-kick carry. `time_end`, `x_end` and `y_end` are required inputs; a missing `time_end` gives a NaN time, never `time_start`, and one `UserWarning` names the count. End coordinates: a pass-class row ends at its targeted reception where present, else at the next possession's reception in the period (the last possession of a period falls back to `_derive_end_coordinates`); a shot, foul or non-action ends at its start.

**B. Gradient Sports: the possession event's own time.** `time_seconds = event_time − kickoff_start_time[period]`, period-relative (ADR-017). The kickoff anchors (`FIRSTKICKOFF` … `FOURTHKICKOFF`) are read before the exclusion filter. A period 1–4 action without its kickoff raises; a caller's period-5 shootout is anchored at its first event; a period without converted actions needs no anchor; an event before its kickoff raises. The converter sorts by the new time with the tiebreak `("start_time", "possession_event_id", "event_id")` (ADR-065), so GS `action_id` renumbers. The Option D imputation of a null `startGameClock` is removed (its reason no longer reaches any output); the input `time_seconds` is ignored, and a NaN `event_time` raises. The pining loader's GS frames use `periodElapsedTime` (period-relative).

**C. `time_source`.** Every converter emits `time_source` (dtype `object`, vocabulary `TIME_SOURCE_VALUES = ("tracking", "event")`, validated in `_finalize_output`): `"tracking"` on GS and SkillCorner native rows, `"event"` elsewhere (Metrica only if measured; kloppy always `"event"`); derived rows inherit; synthetic dribbles are `"event"`. It joins `SPADL_COLUMNS`, is non-geometric in the reflection registry, and is carried by `convert_to_atomic(preserve_native=…)`. A missing column means `"event"`.

**D. ELASTIC episodes from `stoppage_state`.** `_inplay_segments` reads `stoppage_state` when present (`"live"` in play; `"dead"`/`"unknown"` not), else `ball_state`. ELASTIC's score and the time-based linker are unchanged.

## 7. Validation

### 7.1 Tests

Each band is tested from both sides:
- **The rule and set pieces:** start line, box and wide edges, the mirrored next action, the period boundary, restarts; one test per converter (a native cross with a short delivery becomes `pass`, a native pass into the box from wide becomes `cross`); direction and order invariance; the kloppy/SkillCorner reorder regression.
- **Columns and consumers:** `cross_native` per provider; `cross_blocked` and every bravery output keep their meaning; TF-46 `corner_kind`/`corner_kind_source` with and without `cross_attempt`, GI winning, the taker skip; xCross label row classes, the unknown exclusion and the missing-column refusal; the causal unknown treatment (excluded inside the window, unchanged just outside).
- **The enricher:** every row class; the release order (native, ELASTIC, event time; per row); the release match; the passer exclusion; the cap; coverage; the cut-out (near and far contact, both sides of `cut_out_window_s`, away direction `False`, teammate `False`, box entry first `True`, no height, orientation invariance, the corner fallback, the unknown xCross label); the unresolved direction (one and two frames); the direction window cut at the contact, the contact's own frame excluded at 10 Hz; honest-NaN height.
- **Native timing:** SkillCorner kick point, start-type rows, missing `time_end`, the carry, the end coordinates; GS event time per period, the anchor rules, the inversion pair, the ignored input time; `time_source` per converter; ELASTIC episodes from `stoppage_state`.
- **Calibration driver:** each 7.4 rule on a toy corpus, the triggers on both sides, the alignment classes as a partition, the cut-out share and the block-call report.

### 7.2 Choosing `CROSS_START_X` (pre-declared)

For each candidate in {52.5, 60, 65, 70, 75, 80}, the F1 between the rule and `cross_native` among open-play passes, per vendor with a full native label (StatsBomb open data; the GS matches of the corpus). The candidate with the highest equal-weight mean F1 wins; on a tie within 0.005, the larger. IDSSE's F1 is reported, not averaged (about 4 native crosses per match).

### 7.3 Corpus evidence (DGX, full xshot187 corpus)

- The median geometric crosses per match agree within a ratio of 1.5 (max over min across GS, IDSSE and SkillCorner).
- Per provider: `cross_attempt` counts and the cut-out share (cut-out rows over the applicable rows whose attempt is decided or cut out); `corner_crossed`/`freekick_crossed` medians before and after; restart-ended deliveries and their tracking attempt share; set-piece event recall against `cross_attempt` and the `corner_kind_source` split; per row class, the alignment shares (native, ELASTIC, matched at the event time, unknown start, no ball at release, misaligned, coverage unanswerable, cut out).
- Run 3 also reports the cut-out rows whose direction window the contact cut, the unresolved-direction rows, and the GS block call against PFF code B (precision, recall, and the code-B crosses whose block falls inside `release_gap_s`), for the owner (decision 12).

### 7.4 Calibrating `CrossDeliveryParams` (pre-declared)

Every sample is the rows to which `cross_attempt` applies, at the chosen start line; "aligned rows" are those with a native or confident ELASTIC release.

| Field | Rule |
|---|---|
| `release_match_m` | the 95th percentile of `release_dist_m` over IDSSE's ELASTIC rows: IDSSE is the only provider without a long alignment tail (95th percentiles 5.0 m, against 21 m GS and 16 m SkillCorner under ELASTIC) |
| `release_gap_s` | the 95th percentile of `passer_contact_dt`, equal weight per provider, over aligned rows within `release_match_m`; floor one frame at the highest frame rate |
| `direction_window_s` | two frames at the lowest provider frame rate |
| `min_ball_coverage` | 0.8 |
| `cut_out_window_s`, `path_cap_s` | fixed at 1.0 s and 4.0 s |

**Owner triggers** (each stops the runs for an owner decision):
- `release_match_owner_decision_needed`: fewer than 200 finite IDSSE samples, or the match turns more than 20 % of a provider's aligned rows with a finite distance into `<NA>`;
- `native_release_owner_decision_needed`: a provider's native rows have a median `passer_contact_dt` outside [−0.30 s, +0.30 s] or a passer-touch share below 40 %;
- `coverage_owner_decision_needed`: more than 20 % of a provider's rows below `min_ball_coverage`;
- `cut_out_owner_decision_needed`: the cut-out rule makes more than 20 % of a provider's answerable rows `<NA>`.

**The runs**, on the commit-3 re-stage of the xshot187 `_actions` (the r8 converters; `~/cross-def/xshot187-r8/out`), all reading the clean code commit (commit 4) with every earlier value passed as a full-precision override, including the tuned detector values (`--contact-radius-m`, `--min-velocity-change-ms`):
1. provisional parameters → the start line;
2. `--cross-start-x` → `release_match_m`, `release_gap_s` (release-match and native-release triggers read);
3. plus those two → the 7.3 evidence; `fixed_point` must hold for every calibrated field and `cross_start_x` (relative tolerance 1e-9), and the coverage, native-release and cut-out triggers are read.

## 8. Non-goals and anti-goals

- **Non-goal:** the xCross refit and the other owed retrains (Part Deux, after the release).
- **Anti-goal:** tuning the rule until one vendor's labels match; the vendor labels are a benchmark.
- **Anti-goal:** a provider branch inside the shared rule, the enricher or ELASTIC; the converters own the provider facts (`time_source`, `stoppage_state`).
- **Non-goal:** changing ELASTIC's score.

## 9. Constraints

- **House rules** for the enricher: pure (`PURITY_ENTRIES`), NaN-safe (ADR-003), one call shape, `metric_contracts`, `feature_glossary`, an SB360 freeze-frame verdict (ADR-053), liveness; no `*_xfns` (it reads the delivery's own outcome). `NOTICE` cites Vidal-Codina et al. 2022; the reflection registry declares `cross_native`, `time_source`, `cross_attempt` and `cross_height` non-geometric; `id_compat` for every id compare; geometry from `spadlconfig`.
- **Public seams** (every caller enumerated with evidence on both sides in `docs/research/cross-definition/caller-sweep.md`): the open-play `cross` type and `pass` readers; the set-piece types; `cross_native`; `cross_blocked` applicability; SkillCorner action start and time; GS `time_seconds`, inputs and `action_id`; `time_source`; ELASTIC episodes; the TF-79 detector (ADR-134).
- **Commits** (one branch; provenance needs a committed tree for every run, ADR-037): commits 1–3 (`4490016c`, `b12c0fd9`, `dd41724d`) are pushed; commit 4 holds the remaining code, tests and these final-state docs; the detector runs and the three cross runs read the clean commit 4; commit 5 holds the tuned constants, the memos (stamped with commit 4's sha) and the release. Each commit and push has its own owner gate.
- **Release.** A minor version. The CHANGELOG carries a Hyrum notice of every change:
  - SPADL gains `cross_native` and `time_source` (atomic opt-in); open-play `cross` and `pass` counts change for every provider, so `pass` readers such as `PassCompletionModel` shift; set-piece `_crossed`/`_short` change (SkillCorner event-only corners drop from 62 % to about 22 % `corner_crossed`; tracking decides instead); `cross_blocked` applies where `cross_native` is True.
  - TF-86 `add_cross_deliveries` (new); TF-46 gains `corner_kind_source` and `taker_player_id`, and `add_corner_windows` refuses a corners table without `taker_player_id` (rebuild it with `identify_corners`); the xCross label is version 2 (a cut-out cross is unknown); causal xCross opportunities exclude unknown treatments.
  - SkillCorner rows move from the reception to the kick (new required inputs `time_end`, `x_end`, `y_end`; synthetic carries appear); GS `time_seconds` moves to the event's own time (new required inputs `event_time`, `start_time`, the kickoff rows; a caller's own `startGameClock` normalization no longer reaches the output); the pining loader's GS frames are period-relative; GS and SkillCorner `action_id` renumber (an ADR-065 cross-table migration); GS stoppage spans and every time-based frame link for both providers move; ELASTIC episodes change where frames carry `stoppage_state`.
  - The TF-79 detector v2 (ADR-134): every contact changes; `BallContactParams` is keyword-only, four fields are removed, `kinematic_window_s` is new and `min_velocity_change_ms` becomes the candidate threshold; `contact_source` emits only `"velocity_break"`/`"unresolved"`; `first_contact_in_window` gains `exclude_player_id`; `add_ball_contacts` skips the actor. The credit step is the contact scorer (ADR-134 amendments 1-2): a gradient-boosted scorer with post-peak control features and a first-touch decoding; `BallContactParams` gains `contact_scorer` and `min_contact_probability`, `min_velocity_change_ms` defaults to 0.0, `confidence` / `first_contact_confidence` become the scorer's probability, and `contact_scorer=None` is the untuned reach gate; in scorer mode a credit reads the frames within about 0.7 s of its peak, so callers pass whole matches; the bar holds on the track-supported pairs, and the memo states the all-pairs figures; new bundled weights `tracking/_contact_scorer_weights/default`.
  - Owed retrains (evidence in `caller-sweep.md`): xCross, PassCompletionModel, XSuccessModel, the TF-46 corner models, xShot, the ghost-GK and ghost-outfield models, and both xT-GK retention variants. Known downstream consumer: the luxury-lakehouse (`bravery_writer.py`, its GS adapter, `skillcorner._PERIOD_START_SECONDS`).

## 10. The author's calls

- The converters' "next on-ball action" skips `non_action` and `foul`.
- `cross_native` is `<NA>` for SkillCorner, because its old rule was ours.
- IDSSE alone calibrates the release match (the only provider without a long alignment tail).
- The cut-out keeps the initial-direction check: a ball heading away from the box was not aimed at it, whoever touches it next.
- The cut-out owner trigger uses the house 20 % bar.
- The causal treatment follows the label v2 unknown rule.
- `time_source` is a per-row column (the data carries its provenance, ADR-054), dtype `object` like the other provenance columns.
- SkillCorner restart rows start at the taker's `x_end`; snapping corners to the spot would be a provider branch.
- The detector's design calls are in its own spec (section 11).

## 11. References

- Anzer, G. et al. (2025). The Common Data Format (CDF). arXiv:2505.15820.
- Vidal-Codina, F., Evans, N., El Fakir, B., Billingham, J. (2022). Automatic event detection in football using tracking data. arXiv:2202.00804.
- Wyscout data glossary, "Cross": https://dataglossary.wyscout.com/cross/
- Opta event definitions: https://www.statsperform.com/opta-event-definitions/
