# Action-time repair: reader sweep (bar A7)

Tree: `6eee84ce` (the branch point; the six commands give the same counts at the old C1 `3ec4b79a`). Spec: `docs/superpowers/specs/2026-11-18-action-time-repair-design.md`.

Decision key:
- **C**: changed by the repair.
- **T**: changed only to emit `time_source`.
- **I**: not changed; it reads the restart types or the time, and the repaired rows are correct input for it.
- **P**: not changed; a provider other than SkillCorner, or StatsBomb/SB360 only.
- **S**: not changed; a probe, profile, fixture or test-corpus script outside the artifact path.
- **R**: changed by owner ruling R54 (spec §4.13).

## Command 1: the readers of the six restart types and of the SkillCorner interruption markers

```bash
git grep -cE 'corner_crossed|corner_short|freekick_crossed|freekick_short|throw_in|goalkick|game_interruption_(before|after)|_SET_PIECE|_RESTART_TYPE|_CORNER_(TYPE|ID)|_FREEKICK_ID|_OUT_TRIGGER' -- 'silly_kicks/*.py' 'scripts/*.py'
```

Output: 45 files (match count in brackets).

| File | Key | Decision and reason |
|---|---|---|
| `silly_kicks/spadl/skillcorner.py` (24) | C | The derived restart rows. The first possession takes the type of its own action. The docstring is corrected. |
| `silly_kicks/spadl/_skillcorner_inference.py` (3) | C | New `infer_restarts`, next to `infer_defensive_actions` / `infer_keeper_saves` |
| `silly_kicks/spadl/base.py` (17) | C | `time_source` joins the finalized columns and is validated. `_add_dribbles` writes `"derived"`. `classify_deliveries` is not changed: restarts never deliver. |
| `silly_kicks/spadl/config.py` (6) | I | type ids only |
| `silly_kicks/spadl/utils.py` (9) | I | Possession carve-out after a foul. The derived row is a restart row, as the receiver row was. |
| `silly_kicks/spadl/gradientsports.py` (7) | T | `"tracking"`; synthetic rows inherit the value |
| `silly_kicks/spadl/sportec.py` (28) | T | `"event"` (the converter; the parser event block `providers/sportec/parse.py` is not touched, bar A9, except the R63 change of bar A25: the outcome under `Penalty > ShotAtGoal`) |
| `silly_kicks/spadl/statsbomb.py`, `opta.py`, `wyscout.py`, `_wyscout_events.py`, `_wyscout_mappings.py`, `metrica.py`, `kloppy.py` | T | `"event"` |
| `silly_kicks/corners/_identify.py` (11) | C | `t_kick` is NaN on a `"placeholder"` or `"order_only"` row; the row is kept. The GI refinement joins on the derived-row key `(game_id, original_event_id)` with the corner kinds, where the id is that of `b`, the possession after the gap (spec §4.18), not on `t_kick`. The `t_kick` description is corrected. |
| `silly_kicks/tracking/_ghost_gk.py` (7) | C | `_build_phase_lookup` (`:627-695`) reads the row time as the start of the 10 s set-piece window. It drops the `"order_only"` rows before it builds the lookup (R50, spec §5.2). The throw-in rule (`:634`) stays. |
| `silly_kicks/tracking/_stoppages.py` (16) | C | Docstring: `b` is the receiver, not the taker. The spans follow R44 (spec §2.1). |
| `silly_kicks/tracking/_gk_geometry.py` (29) | C | Rule points are reused by the derived rows (single source). `resolve_restart_geometry` reads `time_source` on a derived restart row: `"restart_prior"`/`"unresolved"` for a `"placeholder"` or `"order_only"` row, `"tracking_ball"` for a kick (spec §4.7). |
| `silly_kicks/tracking/_gk_resolve.py` (6) | I | A `goalkick` row is a keeper distribution, actor-independent. The derived goal-kick row has `player_id` `<NA>`. Before, the receiver (often an outfield player) carried it. |
| `silly_kicks/keeper_identity.py` (8) | I | It needs `player_id.notna()` (`:885`). The derived goal-kick row has none, so the receiver is no longer a keeper candidate; the rule for the repaired rows is unchanged. `:876-885` changes for another cause (R72, owner 2026-11-19, not the repair): the goal-kick type is read from `type_id` when the table has no `type_name`. |
| `silly_kicks/tracking/_elastic_sync.py` (8) | I | `_OUT_TRIGGER_TYPES` adds the virtual "out" before a restart; it is used by the new step on `"event"` rows. |
| `silly_kicks/tracking/_packing.py` (6) | I | restart types excluded from packing |
| `silly_kicks/tracking/_xcross_attempt.py` (2) | I | The label is open-play `cross` rows (spec §5.2) |
| `silly_kicks/tracking/defensive_credit/_bravery.py` (4) | I | Set-piece crosses faced. The count now keys to the real restart rows. |
| `silly_kicks/coordination/_windows.py` (6) | I | restart types split windows |
| `silly_kicks/team_metrics/_buildup.py`, `_possession.py`, `_progression.py` | I | open play vs set piece by first type |
| `silly_kicks/vaep/features/specialty.py` (7), `vaep/formula.py` (1) | I | set-piece assist / corner flags |
| `silly_kicks/xtgk/_moves.py` (3), `_retention_features.py` (6) | I | Move set and features. Retention bundles are certification `"blocked"` (not refit). |
| `silly_kicks/atomic/spadl/base.py` (15), `atomic/spadl/utils.py` (14), `atomic/tracking/features.py` (4) | I | Collapsed restart types. `time_source` is carried by `convert_to_atomic(preserve_native=...)` (spec §4.1). |
| `silly_kicks/providers/sportec/parse.py` (4), `providers/statsbomb/parse.py` (1) | P | bronze parsers; no-touch (A9), except the R63 change in `sportec/parse.py` (bar A25): the outcome under `Penalty > ShotAtGoal` |
| `scripts/train_gk_retention.py` (1) | I | retention: not refit |
| `scripts/build_skillcorner_s1_event_validity.py` (3) | S | S1 event-validity probe (set-piece / open-play split) |
| `scripts/_receiver_validation.py` (2), `build_sb360_coverage.py`, `validate_sb360_licensed_corpus.py`, `probe_sb_cross_blocked.py` | P | SB360 / StatsBomb |
| `scripts/measure_f1b_feature_delta.py`, `validate_gk_decision.py` | S | probes |

`scripts/validate_corners.py` and `scripts/build_corner_routines.py` call `identify_corners` and build the GI frame through it, unchanged for the GI join. Their own change (C) is the no-window rule (spec §5.2): `_snapshot_rows` and `_divergence_rows` give no row for a corner with no window or no `corner_side`. Command 1 does not list them because they hold no type literal.

## Command 2: code that compares an action time with a frame time

```bash
git grep -lE 'time_seconds' -- 'silly_kicks/*.py' 'scripts/*.py' | xargs grep -lE '\bframes\b'
```

Output: 70 files. This is a superset at file level. Every action-time-to-frame-time seam is in a file that names both.

The rule for every file: a retimed row (`"elastic"`, `"kick"`) is correct input. A `"placeholder"` or `"order_only"` row has `player_id` `<NA>`, so every actor-keyed feature is NaN on it. A frame feature that needs no actor would describe the frame at the row time: R45 and R50: the producer drops the pointer of such a row (spec §2.1). The frame linker (`tracking/utils.py`, `tracking/_link_identity.py`) is not touched (A9).

`silly_kicks/tracking/_run_features.py` is not in the output (it names no `frames` and `time_seconds` together). It changes for the R45 pointer drop and the blank step (C; spec §5.3).

| File | Key | Decision and reason |
|---|---|---|
| `scripts/_loader_pining.py` | C | The staging hook calls `sync_action_times` for providers with frames. The SkillCorner bronze carries the ball `is_detected`. |
| `silly_kicks/tracking/skillcorner.py` | C | Ball `visibility` comes from the ball `is_detected` (today `None`, `:268`) |
| `silly_kicks/corners/_windows.py` | C | No window on a NaN `t_kick`; the row is counted, not dropped |
| `silly_kicks/spadl/skillcorner.py`, `spadl/base.py`, `spadl/kloppy.py`, `spadl/utils.py` | C / T | see command 1 |
| `silly_kicks/reflection.py` | T | `time_source` is non-geometric |
| `silly_kicks/feature_glossary.py` | T | the `time_source` entry |
| `silly_kicks/tracking/schema.py` | I | frame schema; `visibility` already declared |
| `silly_kicks/corners/_encode.py` | C | reads `t_init`/`t_target`/`corner_side`; `_nearest_frame_xy` gives no frame for a missing time and `_encode_one` no encoding without a side (spec §5.2) |
| `silly_kicks/corners/_effectiveness.py` | C | reads `t_kick`/`t_outcome`; a corner with no kick time has `<NA>` counts, and the aggregate counts it in `n_corners_without_window` (spec §5.2) |
| `silly_kicks/corners/_reflect.py` | C | reads `corner_side`; a missing side is not reflected (spec §5.2) |
| `silly_kicks/tracking/_occurrence_labels.py`, `_xshot_occurrence.py`, `_xcross_attempt.py` | I | Labels from shot / cross row times (spec §5.2). The idsse shot times move by design (R32). |
| `silly_kicks/tracking/_ghost_gk.py`, `_ghost_outfield.py` | C / I | phase: no window at an `"order_only"` row (spec §5.2); `_ghost_outfield.py` imports it unchanged |
| `silly_kicks/tracking/_stoppages.py` | C | docstring (command 1) |
| `silly_kicks/tracking/_elastic_sync.py`, `_ball_contact.py` | I | engines used by the new step; not changed |
| `silly_kicks/tracking/utils.py` | I | no-touch (A9); the linker reads the repaired `time_seconds` |
| `silly_kicks/tracking/_kernels.py`, `_snapshot.py`, `_off_ball_runs.py`, `_run_values.py`, `_press_commitment.py`, `_shot_goalmouth.py`, `_packing.py`, `features.py` | I | time-linked features; correct input |
| `silly_kicks/tracking/_gk_geometry.py`, `_gk_resolve.py` | I | command 1 |
| `silly_kicks/tracking/gradientsports.py`, `kloppy.py`, `metrica.py`, `sportec.py` | I | frame adapters; not changed |
| `silly_kicks/tracking/preprocess/_butterworth.py`, `_interpolation.py` | I | frame-only |
| `silly_kicks/atomic/spadl/utils.py`, `atomic/tracking/features.py` | I | command 1 |
| `silly_kicks/causal/_confounders.py`, `causal/opportunities.py` | I | time-linked; correct input |
| `silly_kicks/coordination/_signals.py`, `coordination/_windows.py` | I | stoppage windows from frames / restart rows |
| `silly_kicks/keeper_identity.py`, `providers/statsbomb/appearances.py` | I / P | command 1; StatsBomb |
| `silly_kicks/positioning/_compute.py`, `restdefense/_windows.py`, `vaep/features/core.py` | I | time-linked / ordered by time; correct input |
| `silly_kicks/providers/sportec/parse.py` | P | no-touch (A9), except the R63 change (bar A25): the outcome under `Penalty > ShotAtGoal` |
| `scripts/train_ghost_gk.py`, `train_xshot_occurrence.py`, `train_xcross_attempt.py` | I | read the staged actions; retrained (spec §6) |
| `scripts/train_win_probability.py`, `train_xsuccess.py` | P | StatsBomb open data |
| `scripts/hf_jobs/headtohead.py` | I | compares staged outputs |
| `scripts/_loader_databricks.py` | S | legacy loader (Databricks is not an approved source); not staged in this cycle |
| `scripts/_coordination_hypotheses.py`, `_coordination_occlusion.py`, `_integrity_counters.py`, `_toy_corpora.py`, `build_synthetic_gk_fixtures.py`, `build_tf56_positioning_validity.py`, `calibrate_xt_bandwidth.py`, `derive_coordination_params.py`, `make_xshot_directional_fixture.py`, `measure_ball_contact_validation.py`, `measure_covariate_invariance.py`, `probe_action_context_baselines.py`, `regenerate_gs_et_native_gk.py`, `validate_das_native_parity.py`, `validate_shot_goalmouth_sb.py` | S | probes, fixtures, calibrations and validators outside the refit path |

## Command 3: the readers of frame visibility and detection

```bash
git grep -cE 'visibility|detected_mask|is_detected' -- 'silly_kicks/*.py' 'scripts/*.py'
```

Output: 64 files. The change is that SkillCorner ball rows get a real `visibility` (today `None`, `tracking/skillcorner.py:268`). Additional key: **V**, not changed; "visibility" here means the public/private corpus visibility of a match, which is a different meaning.

| File | Key | Decision and reason |
|---|---|---|
| `silly_kicks/tracking/skillcorner.py` (2) | C | Ball `visibility` comes from the ball `is_detected` |
| `scripts/_loader_pining.py` (8) | C | `_skillcorner_bronze` (`:856`) carries the feed's ball `is_detected`, on both input paths of `build_skillcorner_frames` (`:922`) |
| `silly_kicks/tracking/_provider_visibility.py` (21) | I | `detected_mask` is reused by the kick rule. SkillCorner stays detection-aware. |
| `silly_kicks/coordination/_signals.py` (11) | I | The mask is built over all rows (`:784-785`), but it is used on player rows only (`:336`, `:434`). No value changes. |
| `silly_kicks/coordination/_windows.py` (1), `tracking/_stoppages.py` (1) | I | `dead_ball_observed(provider)`; no row visibility |
| `silly_kicks/tracking/_ghost_gk.py` (14), `_ghost_outfield.py` (1) | I | keeper rows (`keeper_detection_mask`, `gk_visibility`); `region_observed_fraction` is the FOV meaning |
| `silly_kicks/gkdv/_engine.py` (3) | I | keeper rows (`:311-317`) |
| `silly_kicks/tracking/_gk_geometry.py` (4) | I | keeper rows (`:159`, `:214`: `is_goalkeeper & ~is_ball`) |
| `silly_kicks/tracking/utils.py` (8) | I | defending keeper rows (`:1386-1389`); the FOV area. No-touch (A9). |
| `silly_kicks/tracking/_snapshot.py` (2) | I | writes NaN visibility on snapshot rows |
| `silly_kicks/tracking/_visibility.py` (1), `_fov_registry.py` (2), `_polygon.py` (1), `defensive_credit/_orchestration.py` (1), `features.py` (5), `_run_values.py` (4), `vaep/features/core.py` (1) | I | The FOV / visible-area meaning, or text. Not the row `visibility` of the ball. |
| `silly_kicks/tracking/schema.py` (2), `__init__.py` (4), `_warnings.py` (1), `reflection.py` (1), `feature_glossary.py` (1) | I | declarations and exports |
| `silly_kicks/tracking/gradientsports.py`, `kloppy.py`, `metrica.py`, `sportec.py`, `preprocess/_smoothing.py` | I | other providers' adapters; smoothing passes the column through |
| `silly_kicks/providers/sportec/parse.py` (1), `providers/statsbomb/parse.py` (2) | P | bronze parsers; no-touch (A9), except the R63 change in `sportec/parse.py` (bar A25): the outcome under `Penalty > ShotAtGoal` |
| `scripts/materialize_tc3_frames.py` (12), `train_ghost_gk.py` (15) | I | `materialize_tc3_frames.py:223-230` refuses a frame set whose `visibility` is missing or entirely null; the ball rows only gain values. `train_ghost_gk.py` reads keeper rows (`gk_visibility`, `:461`, `:480`). |
| `scripts/train_receiver_model.py` (8) | V | `--corpus-visibility` (`:401`, `:507`) |
| `scripts/_coordination_corpus.py`, `_coordination_occlusion.py`, `calibrate_coordination.py`, `derive_coordination_params.py`, `validate_coordination_numerics.py`, `validate_team_coordination.py` | I | coordination: player rows (as `_signals.py`) |
| `scripts/_cache.py`, `_corpus.py`, `_corpus_identity.py`, `_corpus_visibility.py`, `_driver_sources.py`, `_offpitch.py`, `_partition.py`, `_partition_pining.py`, `hf_jobs/arm_policy.py`, `hf_jobs/campaign.py`, `hf_jobs/datasets.py`, `hf_jobs/dgx_e2e.py`, `hf_jobs/manifest.py`, `hf_jobs/registry.py`, `build_sb360_coverage.py`, `render_sb360_licensed_coverage.py`, `render_sb360_matrix.py` | V | public/private corpus visibility |
| `scripts/_toy_corpora.py`, `build_synthetic_gk_fixtures.py`, `make_xshot_directional_fixture.py`, `measure_covariate_invariance.py`, `validate_cover_shadow_rq1.py`, `validate_pass_risk_calibration.py`, `validate_shot_goalmouth_sb.py` | S | fixtures and probes |

Result: no current reader takes the detection of a ball row. No SkillCorner feature value or model input changes from the adapter change. The only new reader is the corner kick rule.

## Command 4: every reader that opens a time window (R50, bar A16)

```bash
git grep -lE 'window_s(econds)?\b|horizon_s(econds)?\b|decay_s(econds)?\b|_DECAY_SECONDS|_WINDOW_S\b|max_spell_seconds|pre_seconds|post_seconds' -- 'silly_kicks/*.py'
```

Output: 37 files. The question for each: does it open a window at the time of an action row that can be an `"order_only"` restart row? Spec §5.3 lists the ways a reader takes no window there. Two more files come from reading the callers: `silly_kicks/team_metrics/_possession.py` (spell `start_time`, `:73-98`) and the packing window of `silly_kicks/tracking/_packing.py` (from a reception row).

| File | Key | Decision and reason |
|---|---|---|
| `silly_kicks/tracking/_ghost_gk.py` | C | the phase window (spec §5.2) |
| `silly_kicks/tracking/_kernels.py`, `features.py`, `atomic/tracking/features.py` | I (blanked by the producer) | actor pre-window (`_actor_pre_window_kernel`, `:761-787`): the actor's rows only; `player_id` is `<NA>` on a derived row. The output on such a row is made missing by the producer's blank step (spec §5.3), not argued from the code. |
| `silly_kicks/tracking/_off_ball_runs.py`, `_run_values.py` | I (blanked by the producer) | `slice_around_event` by action time, then `_prepare_run_candidates`: no candidate on a derived row (`<NA>` actor, `id_compat.py:366-368`). But `_off_ball_runs_kernel` writes 0 runners for an action with frames in its slice (`:185-194`), so a derived row gets 0, not a blank (supervisor, `na_actor_runs_probe.py`: 100 of 100 rows). The producer's blank step makes the columns missing on the producer path (spec §2.1 R45, §5.3); a direct call is not covered. |
| `silly_kicks/tracking/_press_commitment.py` | I | the frame pointer (`:115-123`); R45 and R50 on the producer path |
| `silly_kicks/tracking/_shot_goalmouth.py` | I | shot rows only (`:772`) |
| `silly_kicks/tracking/_xshot_occurrence.py`, `_xcross_attempt.py` | I | shot and open-play cross rows only (spec §5.2) |
| `silly_kicks/corners/_windows.py`, `corners/_config.py` | C / I | `t_kick` `<NA>`: no snapshot and no outcome window (spec §5.2) |
| `silly_kicks/tracking/_gk_geometry.py` | C | the keeper-detection window (`:184-235`) is a tracking tier, skipped on a derived row without a kick (spec §4.7) |
| `silly_kicks/duels/_extract.py` | I | tackle and take-on rows only (`:154`) |
| `silly_kicks/causal/opportunities.py` | I | spells from frames; the outcome window reads shot and cross rows (`:474`) |
| `silly_kicks/tracking/_stoppages.py` | I | the `training_frame_mask` horizon is on frames; the spans come from the GI markers (spec §4.9) |
| `silly_kicks/tracking/_elastic_sync.py`, `_ball_contact.py` | I | engines on frames (slope window, dedup window) |
| `silly_kicks/tracking/utils.py` | I | `slice_around_event` itself; no-touch (A9); its callers are the rows above |
| `silly_kicks/tracking/preprocess/_config_dataclass.py`, `_provider_defaults_generated.py`, `_smoothing.py`, `_velocity.py` | I | frame-only windows |
| `silly_kicks/calibration/_features.py`, `_spaces.py`, `_vaep_brier_objective.py` | I | the `pre_seconds` parameter of off-ball runs (above) |
| `silly_kicks/positioning/_probe.py`, `feature_glossary.py`, `tracking/_artifact_integrity.py` | I | a horizon on frames, or text |
| `silly_kicks/spadl/utils.py`, `atomic/spadl/utils.py` | R | `brief_window_seconds` (`:941-963`): the gap between two actions, for the possession carve-out. An `"order_only"` row sits 0.01-0.03 s before the next reception. It goes with the team-metrics spells under R54 (spec §4.13). |
| `silly_kicks/team_metrics/_compute.py`, `_config.py` (with `_possession.py`) | R | post-recovery and counterpress windows from a spell's `start_time`; a spell that an `"order_only"` row starts counts as a recovery (`_possession.py:97`) |
| `silly_kicks/tracking/_packing.py` | R | the secured window `(t_r, t_r + 3 s]` from the reception row (`:320-330`); the next touch after a pass can be an `"order_only"` row |
| `silly_kicks/vaep/labels.py` | R | `window="time"` (`:423`, `:496`): a 15 s window from each row; `bool` labels |
| `silly_kicks/xtgk/_retention_labels.py` | R | `retains` (`:71`): a 10 s window from each row; NaN is a label value (`:93`) |
| `silly_kicks/xtgk/_turnover.py` | R | a finite `window_seconds` (`:249-263`) from a turnover row |

Result: every frame-window reader takes no window at an `"order_only"` row. The action-window readers marked R take no window there by owner ruling R54 (spec §4.13).

## Command 5: time lookups, for the period key (R51)

```bash
git grep -nE '(searchsorted|merge_asof)\(|sort_values\(\[?"game_id", "time_seconds"' -- 'silly_kicks/*.py' 'scripts/*.py'
```

Output: 43 lines. Result:
- **No period (repaired by R51, spec §4.11):** `silly_kicks/tracking/_ghost_gk.py:594`, `:619`, `:674`, `:687`.
- **Grouped by `(game_id, period_id)` or by `period_id`:** `team_metrics/_compute.py:142`; `team_metrics/_pressing.py:147`, `:162`; `tracking/_ball_contact.py:280-281` (`tt` per period, `:384-391`), `:591`; `tracking/_elastic_sync.py:872-873`; `tracking/_occurrence_labels.py:41`; `tracking/_stoppages.py:203`, `:225`, `:231`, `:233`, `:262`, `:369`; `tracking/preprocess/_butterworth.py:250-251`, `:406` (`_RESAMPLE_ENTITY` holds `period_id`); `tracking/utils.py:436`, `:618-619`, `:1532-1533`; `vaep/labels.py:423`, `:496`; `xtgk/_retention_labels.py:71`; `coordination/_signals.py:292-309` (one `(game, period)` grid, `:110`); `scripts/measure_ball_contact_validation.py:64`; `scripts/_coordination_hypotheses.py:385`.
- **Not a time lookup:** `coordination/_compute.py:576`, `:674`, `:777` (row indices); `coordination/_kernels/_spectral.py:48`; `coordination/_kernels/_numba.py:8` (a comment); `scripts/derive_coordination_params.py:814`.
- **Per shot, in its period:** `tracking/_shot_goalmouth.py:238` (inferred from the call path: the goal ends are read per `(game_id, period_id)`, `:786-787`).

The supervisor ran the command again (the same 43 lines) and searched for the forms that it cannot see (a sort or mask on `time_seconds`, a callback on a time, a loop over `groupby("game_id")`): every hit in `silly_kicks` has the period (`tracking/utils.py:1513-1533`, `tracking/features.py:6054-6096`, `tracking/_press_commitment.py:167-172`, `tracking/_ball_contact.py:521-527`, `keeper_identity.py:888-895`, `corners/_effectiveness.py:62`) or compares two neighbour rows (`vaep/formula.py:62`, `duels/_extract.py:154`).

## Command 6: the readers of the shot types (R58, bar A24, point 6)

```bash
git grep -lE 'shot_penalty|_SHOT_TYPE|SHOT_TYPES|"shot"' -- 'silly_kicks/*.py'
```

Output: 60 files. The new case is a derived SkillCorner `shot_penalty` row with `is_time_unmeasured` (spec §4.17, §5.4). A penalty kick lies at least 51.5 s after the previous possession's end (`sc_penalty_check.parquet`, `wait_s`), so no window of 15 s or less from an earlier row reaches it.

| File | Key | Decision and reason |
|---|---|---|
| `silly_kicks/spadl/_skillcorner_inference.py`, `spadl/skillcorner.py` | C | the penalty rows (spec §4.17); the `preserve_native` list (spec §4.15) |
| `silly_kicks/spadl/sportec.py` | C | the idsse penalty (spec §4.16) |
| `silly_kicks/tracking/_xshot_occurrence.py` | C | the label skips a shot row with an unmeasured time (spec §5.4) |
| `silly_kicks/tracking/features.py`, `atomic/tracking/features.py` | C | pre-shot keeper position and angle: NaN on such a row |
| `silly_kicks/tracking/_shot_goalmouth.py` | C | a post-shot frame window: NaN on such a row |
| `silly_kicks/causal/opportunities.py` | C | treatment and outcome windows: such a row is neither |
| `silly_kicks/coordination/_windows.py` | C | the R54 seventh reader and the goal interval (spec §4.13) |
| `silly_kicks/tracking/_gk_geometry.py` | C | the enrichment of a derived penalty row (spec §4.7) |
| `silly_kicks/tracking/_ghost_gk.py`, `vaep/labels.py`, `win_probability/_state.py` | C | the goal predicate (R52) and the score paths (R51, R56); a penalty goal row counts from its time |
| `silly_kicks/tracking/_packing.py` | C | a reception on such a row: `packing_secured` `<NA>` (spec §5.4) |
| `silly_kicks/tracking/_elastic_sync.py` | I | ELASTIC runs only on a match with `"event"` rows; SkillCorner has none; the idsse penalty row is `"event"` and is aligned as any shot |
| `silly_kicks/tracking/_stoppages.py` | I | `event_stoppage_spans` is the gradientsports path |
| `silly_kicks/tracking/_model_eval.py` | I | `"shot"` is an arm name |
| `silly_kicks/calibration/_carrier_objective.py` | I | carrier against the actor; the derived row has `player_id` `<NA>` |
| `silly_kicks/match_outcome/_compute.py`, `shot_stopping/_compute.py`, `corners/_effectiveness.py`, `team_metrics/_compute.py`, `_buildup.py`, `_possession.py`, `_progression.py` | I | counts, sums or order by type and result; the corner outcome window is 6 s and the team-metrics windows 10 s, which never reach a penalty kick |
| `silly_kicks/vaep/formula.py`, `vaep/features/context.py`, `vaep/features/specialty.py`, `atomic/vaep/labels.py`, `atomic/spadl/base.py`, `atomic/spadl/utils.py` | I | order and type; the atomic labels have no time mode |
| `silly_kicks/win_probability/_model.py`, `xsuccess/_features.py`, `xsuccess/_model.py` | I | type lists; `win_probability` excludes period 5 |
| `silly_kicks/tracking/defensive_credit/_bravery.py`, `_chaining.py`, `_orchestration.py`, `_params.py`, `_rules.py` | I | type `shot` only, or rule names; `shot_penalty` is not read |
| `silly_kicks/xtgk/_empirical.py`, `_moves.py`, `_retention_labels.py`, `_turnover.py`, `_validate.py`, `_xg_reward.py`, `xthreat/_grid.py`, `xthreat/_model.py` | I | type `shot` only; `shot_penalty` is not read |
| `silly_kicks/keeper_identity.py`, `feature_glossary.py` | I | a docstring example; text |
| `silly_kicks/spadl/base.py`, `spadl/config.py`, `spadl/utils.py` | I | type lists (the carve-out is R54) |
| `silly_kicks/spadl/gradientsports.py`, `kloppy.py`, `metrica.py`, `opta.py`, `statsbomb.py`, `_wyscout_events.py`, `_wyscout_mappings.py` | P | other providers' converters |
| `silly_kicks/providers/sportec/parse.py` | P | bronze parser; no-touch (A9), except the R63 change (bar A25): it now flattens the outcome of a `Penalty`'s nested shot into `shot_outcome_type` / `shot_outcome_*`, which the converter reads for `shot_penalty` (R57) |

Result: the readers of a shot row's time skip a row with an unmeasured time (spec §5.4). The other readers read the type, the result or the order.
