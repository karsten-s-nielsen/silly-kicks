# One cross definition for every provider — design

- **Status:** r6 (2026-11-13). The owner approved r1 and decisions 7–10. r2 answered review B round 1; r3 answered review A round 1; r4 answered CD-SB-08, CD-SPEC-08 and CD-SPEC-09; r5 answered CD-SB-09 and CD-SB-10; r6 answers CD-SPEC-10.
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

## 6. Design

### 6.1 Converter layer: the geometric `cross`

One shared function in `silly_kicks/spadl/base.py` re-types the open-play pass-class rows of every converter. A row is a `cross` when all of these hold:

1. **Open play.** Its type is `pass` or `cross` before re-typing. Corners and free kicks follow 6.1b; `throw_in` and `goalkick` do not change.
2. **Wide start.** `start_y < W/2 − penalty_area_half_width` or `start_y > W/2 + penalty_area_half_width`. This uses `spadlconfig.penalty_area_half_width` (20.16 m) and `field_width`.
3. **Advanced start.** `start_x >= CROSS_START_X`. The value is chosen once by the rule in Section 7.2, from {52.5, 60, 65, 70}.
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

**The release.** Event clocks and tracking clocks are not aligned for every provider. Review A measured that the ball is more than 5 m from the action start at the event time for 4041 of 6461 GS rows and 2463 of 4596 IDSSE rows; audit 5 could judge only the aligned rows. The release is therefore aligned through the TF-43 seam (`tracking/_elastic_sync.py`), not read from `time_seconds`:
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
- the first contact is by an opponent (a block), within `BLOCK_RADIUS_M` of the release, and the initial ball direction points into the penalty area. The initial direction is the mean ball velocity over the first `DIRECTION_WINDOW_S` after the release.

All geometry is in the action-LTR frame (ADR-028), with the direction from the GoalMap (ADR-055).

**`cross_height`** (category `"aerial"` / `"ground"`). On rows where `cross_attempt` is `True`:
- It is `"aerial"` when the peak ball z over the path is above `BallContactParams.aerial_z_m` (1.8 m, ADR-125), and `"ground"` otherwise. The threshold is the one constant of the contact detector, not a second one.
- It is `<NA>` where the ball `z_source` is not `"measured"` (decision 4). This is the same rule as the detector's `first_contact_aerial_source`.
- It is `<NA>` on every other row.

### 6.4 The event-only limit

Without tracking, two kinds of delivery cannot be seen as a cross:
- a blocked cross, because the block ends the delivery outside the box;
- a restart-ended delivery, because the event data has no end (decision 7).

6.1 types both as `pass`. A restart-ended set piece is `_short` (6.1b). SkillCorner's event data also misses defenders' touches in the box, so the event rule finds only 43 % of its corner deliveries (decision 10). ADR-133 and the converter docstrings state these limits with the measured recall for each provider. The tracking layer covers all of them.

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
- **Enricher:** `cross_attempt` / `cross_height` tests for every row class of the 6.3 table, a blocked cross, a path without a contact (cap), partial ball coverage on both sides of `MIN_BALL_COVERAGE`, the `z_source == "estimated"` honest NaN, and orientation invariance.

### 7.2 Choosing `CROSS_START_X` (pre-declared)

- For each candidate in {52.5, 60, 65, 70}, compute the F1 between the rule and `cross_native` among open-play passes for each vendor with a full native cross label:
  - StatsBomb open data (`pass.cross`);
  - the GS matches of the xshot187 corpus.
- Choose the candidate with the highest mean F1, equal weight per vendor (decision 8). On a tie within 0.005, choose the larger value.
- IDSSE is left out of the mean: its native label covers about 4 crosses per match. Its F1 is reported.
- Record every vendor's F1 table in `docs/research/cross-definition/`. A disagreement between vendors is reported, not tuned away.

### 7.3 Corpus evidence (DGX, full xshot187 corpus)

- The median geometric crosses per match must agree within a ratio of 1.5 (max over min across GS, IDSSE and SkillCorner). The audit measured 21 / 16.5 = 1.27.
- Report the `cross_attempt` counts and the blocked share for each provider.
- Report `corner_crossed` and `freekick_crossed` medians per provider, before and after.
- Report, per provider, the restart-ended deliveries from the cross zone, and how many of them the tracking layer types as attempts. This is the event-only undercount (decision 7).
- Report, per provider and set-piece kind, the restart-ended set pieces, the event rule's recall against `cross_attempt`, and the `corner_kind_source` split (decision 10).
- Report, per provider and row class (open play in the cross zone, corner, free kick), the shares that are aligned by ELASTIC, unaligned but matched, misaligned (`<NA>`), and unanswerable for coverage. An xCross label or a `corner_kind` decided by tracking is only as good as these shares.
- The audits in Section 5 and decision 10 used approximate path ends (the next action, or 4 s). The 7.3 numbers use the implemented path (first non-passer contact). ADR-133 and the CHANGELOG cite the 7.3 numbers.

## 8. Non-goals and anti-goals

- **Non-goal:** the xCross refit, the other owed retrains, and their adoption. Part Deux owns them, after this release and a restage of the xshot187 `_actions` (decision 6).
- **Anti-goal:** tuning the rule until one vendor's labels match. The vendor labels are a benchmark, never the target. The one fitted value, the start line, comes from a pre-declared mean over vendors (7.2).
- **Anti-goal:** a provider-specific branch inside the shared rule.

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
- **Trained artifacts.** The plan lists every bundled or Hub artifact whose training data contains SPADL `type_id`, for example VAEP models and xCross. For each one, it states whether the value change makes a retrain owed. Owed retrains are recorded in the CHANGELOG and handed to Part Deux (decision 6).
- **Release.** The change alters values, so it is a minor version bump. The CHANGELOG carries a Hyrum notice that lists every change:
  - The SPADL output schema gains `cross_native` for every converter. Atomic SPADL carries it opt-in through `preserve_native`.
  - Open-play `pass` counts change too, so `PassCompletionModel` and other `pass` readers shift (Section 9, seam 1).
  - Open-play `cross` and `pass` counts change for every provider; rows move between the two types.
  - `corner_crossed` / `corner_short` and `freekick_crossed` / `freekick_short` change for every provider. This supersedes the ADR-127 discriminator. SkillCorner's event-only corner typing drops from 62 % to about 22 % `corner_crossed`; on tracking, `cross_attempt` and TF-46 `corner_kind` decide instead (decision 10).
  - TF-46 output gains `corner_kind_source`. The `corner_kind` domain is unchanged; with tracking, more SkillCorner corners become `"direct"` and enter routine building.
  - `cross_blocked` applies where `cross_native` is True, not where the type is `cross`.
  - The xCross label is version 2, and the retrains are owed.
  - Known downstream consumer: the luxury-lakehouse reads `type_id` and `cross_blocked` (`src/ingestion/bravery_writer.py`) and keeps per-column shift bands for adoptions (`src/tests/action_context/expected_shift_oracle.py`). The notice names the columns that shift.

## 10. The author's calls (to check while reading)

- In the converters, the "next on-ball action" excludes `non_action` and `foul`. Synthetic dribbles do not exist at the run position.
- `cross_native` is `<NA>` for SkillCorner, because its old rule was ours.
- `cross_blocked` keeps its meaning and is keyed to `cross_native`.
- `BLOCK_RADIUS_M`, `DIRECTION_WINDOW_S`, `RELEASE_GAP_S`, `MIN_BALL_COVERAGE`, `RELEASE_MATCH_M` and the 4 s path cap are set in the plan from the audit data.
- The release is aligned through ELASTIC (TF-43), with a release-match check on every row.
- The corpus agreement bar is a ratio of 1.5.
- Causal opportunities select by `xcross_label_version`.

## 11. References

- Anzer, G. et al. (2025). The Common Data Format (CDF). arXiv:2505.15820.
- Vidal-Codina, F., Evans, N., El Fakir, B., Billingham, J. (2022). Automatic event detection in football using tracking data. arXiv:2202.00804.
- Wyscout data glossary, "Cross": https://dataglossary.wyscout.com/cross/
- Opta event definitions: https://www.statsperform.com/opta-event-definitions/
