# Cross definition (ADR-133): caller sweep

Five seams change: the open-play `cross` type (seam 1), the new column `cross_native` (seam 3), `cross_blocked`
applicability (seam 4), and the native event timing of SkillCorner (seam 5) and Gradient Sports (seam 6). The
set-piece types do not change (seam 2 of the original sweep has no reader to classify). This memo classifies every reader in `silly_kicks/` and `scripts/` at `main`
`0cc07e6c` (`git grep -n -e '"cross"' -e "'cross'" -e '"pass"' -e cross_native -e cross_blocked -- silly_kicks
scripts`).

Classes:
- **intended shift**: the value moves because the open-play type now means the geometric delivery. The release
  notice lists it.
- **unaffected (union)**: the reader takes `pass` ∪ `cross`, so a row that moves between the two does not change it.
- **changed in this cycle**: the reader moved to the new meaning, with tests.
- **retrain owed**: a trained artifact whose training rows, label or features read the type.

## Seam 1: the open-play `cross` type

| Reader | Reads | Class | Evidence |
|---|---|---|---|
| `spadl/base.py:18` `_DERIVE_END_TYPE_IDS` | pass, cross, set pieces (end derivation) | unaffected (union) | the set holds both types |
| `spadl/config.py:53` | the action-type list | unaffected | the id list does not change |
| `atomic/spadl/base.py:99` | atomic type mapping | unaffected (union) | pass and cross map to their atomic twins one to one |
| `atomic/tracking/features.py:266,305,632,675` | defensive-credit domain, pass-like mask, packing | unaffected (union) | each lists `("pass", "cross")` |
| `atomic/vaep/features.py:404` | move names | unaffected (union) | `("pass", "dribble", "cross")` |
| `vaep/features/specialty.py:47,130,140` | `cross_zone` / cross-type features | intended shift | `is_cross` reads the type; a VAEP model fitted before ADR-133 sees a new cross population |
| `xthreat/_grid.py:132`, `xthreat/_model.py:97` | xT move counts | unaffected (union) | pass, cross and dribble are all moves |
| `xtgk/_moves.py:29,31` | xT-GK moves | unaffected (union) | pass and cross both listed |
| `calibration/_carrier_objective.py:33` | carrier action types | unaffected (union) | `{"pass", "cross", "shot", "dribble"}` |
| `causal/opportunities.py:96-103` | treatment rows | intended shift | the treatment is a SPADL `cross` |
| `tracking/_xcross_attempt.py` | the xCross label | intended shift; **retrain owed** | the label reads the SPADL `cross` type |
| `tracking/_run_values.py:419`, `tracking/_packing.py:33-38`, `tracking/_elastic_sync.py:380-390`, `tracking/_line_breaking.py:37`, `tracking/features.py:5789` | pass-like ids | unaffected (union) | each lists both types |
| `tracking/defensive_credit/_rules.py:52,437,456,478`, `_params.py:54` (`ANCHOR_CROSS`) | the cross-defence credit rule | intended shift | the rule fires on a successful geometric `cross` |
| `tracking/defensive_credit/_bravery.py` | crosses faced | changed in this cycle | reads `cross_native` |
| `scripts/_rq_corpus.py:15,16` | crosses excluded from the pass-only headline | intended shift | the excluded rows are the geometric crosses |
| `scripts/build_sb360_coverage.py:56`, `validate_sb360_licensed_corpus.py:52` | GK domain types | intended shift | the GK domain counts geometric crosses |
| `scripts/train_receiver_model.py:39`, `scripts/_receiver_validation.py:31` | completed passes | unaffected (union) | `_PASS_TYPES = {pass, cross}` |

### Readers of `pass` alone (rows move both ways between `pass` and `cross`)

| Reader | Reads | Class | Evidence |
|---|---|---|---|
| `expected_passing/_model.py:26,133`, `scripts/train_pass_completion.py:47` | PassCompletionModel training rows | **retrain owed** | trains on `type_id == pass`; bundled `expected_passing/weights` |
| `team_metrics/_pressing.py:22`, `_buildup.py:32`, `_progression.py:24` | pass counts (PPDA, build-up, progression) | intended shift | geometric crosses leave the pass count and SkillCorner zone "crosses" join it |
| `territory/_compute.py:62`, `tracking/defensive_credit/_line_break_signal.py:20`, `_rules.py:53` | pass-only measures | intended shift | pass-only |
| `tracking/_gk_resolve.py:35,89` | GK distribution `is_open` | unaffected in practice | a geometric cross starts at or beyond `CROSS_START_X`, so a keeper distribution stays a `pass` |
| `xsuccess/_features.py:19,48` | the action-type one-hot | **retrain owed** | the feature set is every action type; bundled `xsuccess/weights` |
| `causal/_confounders.py:93`, `xsuccess/_features.py:71`, `_model.py:49`, `scripts/publish_xsuccess.py:41`, `tracking/_stoppages.py:123`, `scripts/hf_jobs/headtohead.py:304,336` | built rows and examples | unaffected | the rows are built, not read |
| `scripts/hf_jobs/headtohead.py:503`, `scripts/validate_territory_counterfactual.py:413,587`, `scripts/measure_f1b_feature_delta.py:240` | pass-only research measures | intended shift | pass-only |

## Seam 3: the new column `cross_native`

`spadl/schema.py` (the schema dicts), `_finalize_output`, `convert_to_atomic(preserve_native=…)`, `reflection.py`,
`atomic/tracking/features.py` (atomic bravery) and `tracking/defensive_credit/_bravery.py`: changed in this cycle.
Every converter emits the column.

## Seam 4: `cross_blocked` applicability

`tracking/defensive_credit/_bravery.py` and `atomic/tracking/features.py` read `cross_blocked` on rows with
`cross_native == True`: changed in this cycle. `scripts/probe_sb_cross_blocked.py` is a one-off probe of the
StatsBomb native block join, unaffected (it reads raw events).

## Seams 5 and 6: native event timing (SkillCorner, Gradient Sports)

The evidence for the moves is `native-timing.md` (section 2, the frame-link shift). Classes as above, plus **link
moves**: the reader links GS or SkillCorner actions to frames by time, so it now reads the frame at the action's own
moment (the kick), not the reception (SkillCorner, a median 1.1 s earlier) or the whole-second sequence start (GS, a
median 0.5 s earlier at the link).

| Seam | Reader | Class | Evidence |
|---|---|---|---|
| 5, 6 | `tracking/utils.py` `link_actions_to_frames` and its callers (`git grep -l link_actions_to_frames`) | link moves (GS, SkillCorner) | it reads `time_seconds`; every frame-aware `add_*` reads the frame at the action's own moment |
| 5 | every reader of SkillCorner `start_x`/`start_y` (VAEP features, xT moves, pressure at the action, packing, `classify_deliveries`) | intended shift | the start moves from the reception to the kick |
| 5 | SkillCorner `end_x`/`end_y` of shots, fouls and non-actions | intended shift | end = start (the GS convention) |
| 5 | `_add_dribbles` on SkillCorner | intended shift | carries from the reception to the kick appear |
| 6 | GS `time_seconds` readers: VAEP time features, possession gaps, `event_stoppage_spans` (ADR-130 GS dead-ball spans, `training_frame_mask`) | intended shift | the possession event's own time, period-relative |
| 6 | `scripts/_loader_pining.py` `_build_gradientsports` frames, `scripts/regenerate_gs_et_native_gk.py` frames | changed in this cycle | frames take `periodElapsedTime`, the clock the GS actions use |
| 6 | the GS converter's input `time_seconds` (`startGameClock`) | changed in this cycle | no longer read; a caller's period-offset normalization of it no longer reaches the output |
| 6 | `docs/examples/gradientsports_wc2022_walkthrough.py` (builds the converter input) | changed in this cycle | passes `start_time` and `event_time` |
| 5, 6 | stored tables keyed by `action_id` (GS, SkillCorner) | intended shift | `action_id` renumbers (GS: 24 native converted actions move in 21 of 64 matches; SkillCorner: new order and carries); an ADR-065 cross-table migration |

## Trained artifacts

| Artifact | Reads the open-play type? | Trained on GS or SkillCorner action times or links? | Retrain |
|---|---|---|---|
| `tracking/_xcross_weights` (xCross) | the label | yes (IDSSE + SkillCorner) | **owed** |
| `expected_passing/weights` (PassCompletionModel) | trains on `pass` rows | no (StatsBomb open data) | **owed** |
| `xsuccess/weights` (XSuccessModel) | action-type one-hot feature | no (StatsBomb open data) | **owed** |
| `tracking/_xshot_weights` | shot types only | yes (IDSSE + SkillCorner): shot times move to the kick | **owed** |
| `tracking/_ghost_gk_weights` (4 variants) | shots and the set-piece union | yes (GS + SkillCorner + Sportec); the extraction reads the staged actions per game | **owed** |
| `tracking/_ghost_outfield_weights` (2 variants) | shots and the set-piece union | yes (GS + SkillCorner + Sportec); goal times set `score_diff` | **owed** |
| `xtgk/_retention_weights/default` | the move union | yes (GS, 64 games): the release pressure is a time-based frame link at the action | **owed** |
| `xtgk/_retention_weights/skillcorner` | the move union | yes (SkillCorner, 108 games): goal-kick and distribution starts move to the kick | **owed** |
| `corners/_routine_weights`, `corners/_run_zone_weights` | set pieces (unchanged types) | yes (SkillCorner + IDSSE + SB360): SkillCorner corner start and time move to the kick | **owed** |
| `tracking/_receiver_weights` | completed `pass` ∪ `cross` | no (StatsBomb) | not owed |
| `match_outcome/weights`, `win_probability/weights` | shot types only | no (StatsBomb open data) | not owed |
| VAEP | no bundled weights | — | none (users refit) |
