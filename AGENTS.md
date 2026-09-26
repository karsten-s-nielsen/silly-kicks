# silly-kicks

Maintained fork of socceraction — SPADL event conversion + VAEP action valuation for soccer analytics.

> **Maintaining this file.** `AGENTS.md` is the always-loaded, class-1 surface: CURRENT terse rules + pointers ONLY. Rationale / history / measurement lives on-demand in `docs/context/<domain>.md` (read it when the rule bites). A Key-conventions bullet is ≤ 600 chars and carries an `ADR-NNN` ref or a `docs/context/` pointer. Enforced by `tests/test_agents_md_budget.py` (byte target/ceiling + per-bullet char cap + a per-invariant completeness check against `tests/fixtures/agents_md_invariant_inventory.json`, snapshotted from `CLAUDE.md`@77286f4). Canonical filename is `AGENTS.md`; `CLAUDE.md` is a one-line `@AGENTS.md` importer for Claude Code. Full design: `docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md`.

## Architecture

Hexagonal: all core functions are pure (pandas in, pandas out), zero I/O, zero global state mutation. One-line map per subsystem below — the WHY/history is in the linked `docs/context/` file.

- **Converters** (`spadl/`): per provider; return `tuple[pd.DataFrame, ConversionReport]` with guaranteed columns/dtypes via `_finalize_output()`. Live contracts (`shotOutcomeType`/`owngoal`/`is_synthetic`/`nonEvent`, `_resolve_team_ids` null-actor NaN, `_derive_end_coordinates(extra_type_ids=)`, `ballCarryOutcome`, `_to_period_relative`, `result_source`/`pass_outcome`, `_transform_coords`/`_scale_to_spadl`, `_extract_play_eval`/`play_evaluation`, metrica `BALL LOST`) → `docs/context/providers.md`.
- **Restart enrichment** (`spadl.add_restart_coordinates`, engine `resolve_restart_geometry`): additive `enriched_start_x`/… provenance columns, never mutates canonical (ADR-025; `docs/context/providers.md`).
- **Tracking** (`tracking/`): 19-col `TRACKING_FRAMES_COLUMNS`; `link_actions_to_frames`/`slice_around_event` (ADR-004); `_frame_aware` xfns compose into VAEP (ADR-005); `validate_time_base`/`per_period_link_rate` period-relative (ADR-017); `is_goalkeeper_source`/`derive_goalkeepers` (ADR-007); metrica bronze y bottom-to-top; preprocess `smooth_frames`/`derive_velocities`/`PreprocessConfig`. Feature-family narrative → `docs/context/tracking-features.md`. TF public surface:

| TF | Module | Public surface |
|---|---|---|
| -- | `features.py` | `add_action_context` / `tracking_default_xfns`; `add_pre_shot_gk_position`/`add_pre_shot_gk_angle` / `pre_shot_gk_*_default_xfns` |
| TF-2 | `_pressure.py` | `add_pressure_on_actor` (`andrienko_oval`/`link_zones`/`bekkers_pi`); `pressure_on_target` |
| TF-3 | `_kernels.py` | `add_actor_pre_window` |
| TF-4 | `_off_ball_runs.py` | `add_off_ball_runs`/`add_line_break`/`add_off_ball_context` |
| TF-5 | `_ball_carrier.py` | `infer_ball_carrier`/`ball_carrier_at_action`/`derive_team_in_possession` |
| TF-6 | `utils.py` | `sync_score`/`add_sync_score` |
| TF-7 | `pitch_control/` | `add_pitch_control`/`pitch_control_at_target`/`PitchControlCache`/`compute_pitch_control_batch` |
| TF-13 | `_gk_resolve.py` | `defending_gk_from_frames`/`acting_gk_from_frames`/`gk_distribution_mask`/`resolve_defended_goals`/`GoalMap`/`GoalEndUnresolvedError`/`action_ltr_goal_map` |
| TF-14 | `_defensive_line.py` | `compute_defensive_line`/`add_defensive_line` |
| TF-15 | `_gk_influence.py` | `compute_gk_influence`/`add_gk_influence` |
| TF-16 | `_xshot_occurrence.py` | `XShotOccurrenceModel`/`add_xshot_occurrence`/`compute_xshot_occurrence` |
| TF-17 | `_xcross_attempt.py` | `XCrossAttemptModel`/`add_xcross_attempt` |
| TF-18 | `_ghost_gk.py` | `GhostGkModel`/`compute_ghost_gk`/`add_ghost_gk`/`serve_ghost_gk_positions` |
| TF-28 | `_das.py` | `get_das`/`get_individual_das`/`add_das` |
| TF-30 | `_cover_shadows.py` | `add_cover_shadows` |
| TF-31/44 | `_team_shape.py` | `compute_team_shape`/`add_team_shape` |
| TF-32 | `_line_breaking.py` | `detect_line_breaking`/`add_line_break(method="ward")` |
| TF-33/36 | `_player_influence.py` | `compute_player_influence`/`add_player_influence` |
| TF-35 | `_run_values.py` | `add_off_ball_run_values` / `off_ball_run_value_xfns` |
| TF-39 | `_shape_graph.py` | `compute_shape_graph`/`add_shape_graph` |
| TF-40 | `_obso.py` | `compute_obso_surface`/`add_obso` |
| TF-41 | `_space_creation.py` | `compute_space_created`/`add_space_creation` |
| TF-42 | `_pausa.py` | `compute_pausa`/`add_pausa` |
| TF-43 | `_elastic_sync.py` | `align_events_to_frames`/`add_elastic_sync` |
| TF-45 | `_structural_pass.py` | `compute_structural_pass_metrics`/`add_structural_pass` |
| TF-48 | `_shot_goalmouth.py` | `compute_shot_goalmouth`/`add_shot_goalmouth` — no xfns (post-contact leakage) |
| TF-49 | `_packing.py` | `compute_packing_metrics`/`add_packing`/`packing_xfns` (no default list) |
| TF-51 | `defensive_credit/` | `compute_defensive_credits`/`add_defensive_credit`/`compute_bravery`/`add_press_commitment` — no xfns |
| -- | `_visibility.py` | `point_observed`/`region_observed_fraction`/`add_visible_area_coverage` — no xfns |
| -- | orchestrator | `run_tracking_features` (provider-agnostic producer; not an aggregator) |

- **VAEP** (`vaep/`): GBM classifiers; `HybridVAEP` removes a0 result leakage; `xg_column` labels; own goals counted by RESULT (`_is_owngoal`, ADR-018); `rate_adjusted` (ADR-095), `rate_ximpact` (ADR-101). → `docs/context/vaep.md`.
- **xT** (`xthreat/`): `ExpectedThreat(method="singh_counts"|"kde_smoothed")` (`value_iteration`, `fit_from_counts`, `to_dict`/`from_dict`/`save`/`load`); y-inverted store, `physical_grid`/`values_at_points`/`require_fitted_xt`/`compute_threat_pc` (ADR-021/022/041/100/102). → `docs/context/xt.md`.
- **xT-GK v1** (`tracking/_xt_gk.py`): `compute_xt_gk`/`XtGkParams`/`GkCompletionModel`/`native_origin_is_trusted` (frozen, ADR-024). **v2** (`xtgk/`): `compute_xt_gk_v2`/`MarkovPossessionValue`/`GkRetentionModel`/`EmpiricalTurnoverValue`/`flat_zones` (ADR-036; not construct-validated). → `docs/context/xt-gk.md`.
- **GKDV** (`gkdv/`): `build_ghost_frames`/`delta_das`/`delta_threat_suppression`/`GkdvParams` (spearman-only, no `pitch_control_cache`); probes `layer0_instrument_verdict`/`PHYSICS_ARM_PROBE_RATIO` (ADR-043/075/082). → `docs/context/gkdv.md`.
- **Restdefense** (`restdefense/`): `compute_rest_defense`/`count_goalside`/`rd_danger_behind_line`; `build_restdefense_ghost_frames` (ADR-080/081/089); GK-ghost `sweeper`/`GhostGridSpec`/`DEFAULT_GHOST_GRID`/`predict_density` (ADR-083); `GhostOutfieldModel`/`serve_ghost_outfield_positions`/`GHOST_OUTFIELD_SOURCE_VALUES` (ADR-087). → `docs/context/rest-defense.md`.
- **Event metrics** (`shot_stopping/`,`territory/`,`duels/`,`match_outcome/`,`win_probability/`,`xsuccess/`,`expected_passing/`): `compute_shot_stopping`/`goals_prevented`/`psxg_column`; `compute_territorial_dominance`/`trim_fraction`/`TERRITORY_METHODS`; `compute_duel_ratings`/`update_glicko`/`duel_winner_source`; `compute_match_outcome`/`goal_count_pmf`/`same_possession`/`dixon_coles`/`DependenceModel`; `compute_win_probability`/`goal_leverage`/`certify_coherence`/`WinProbabilityModel`; `XSuccessModel`/`adjusted_value`/END-BLIND; `PassCompletionModel`/`destination_profiles`. → `docs/context/event-metrics.md`, `gk-metrics.md`.
- **GK-decision** (`gk_decision/`): `compute_gk_decision_value`/`decision_pct`/`reachability_min_xpass`/`parse_passing_options` (ADR-092). → `docs/context/gk-metrics.md`.
- **Tracking metrics** (`territorial_defense/`,`positioning/`): `compute_territorial_defense`/`a_threat_suppressed`/`b_attribution_slippage`/`frame_convention`/`build_trimmed_hull` (ADR-090, experimental); `optimise_positions`/`compute_positioning_gap`/`ThreatObjective`/`ReachabilityConstraint` (ADR-104). → `docs/context/tracking-metrics.md`.
- **Causal** (`causal/`): `att_power_curve`/`placebo_shift`/`OpportunityConfig` (ADR-015). **Atomic-SPADL** (`atomic/`): `convert_to_atomic`/`ATOMIC_SPADL_COLUMNS`. **Calibration** (`calibration/`): `CarrierAccuracyObjective`/`AugmentedVaepBrierObjective`/`FrozenXt`/`select_recommended_point` (ADR-009/060). **Providers** (`providers/`): parse ports `parse_dfl_events`/`shape_events_to_native`/`snapshot_to_tracking_frames` (ADR-031/054). → `docs/context/vaep.md`, `providers.md`.

## Where the history lives

`AGENTS.md` holds durable contracts only. Grep these instead of re-adding narrative:

| Question | Source |
|---|---|
| What shipped / broke per release | `CHANGELOG.md` (keyed by `PR-Snnn`) |
| Why a design is the way it is | `docs/superpowers/adrs/` |
| Deeper per-module narrative | `docs/context/<domain>.md` |
| What is planned/blocked | `TODO.md` |
| Measurement behind a claim | `docs/research/<topic>/` |
| Specs and plans | `docs/superpowers/specs/`, `plans/` |
| Downstream pins on private modules | `docs/PRIVATE_CONSUMERS.md` |

## Key conventions

- **Version single-sourced** in `silly_kicks/_version.py` (ADR-079); `pyproject` `dynamic`, `hatch.version` reads it. Bump that one file. (`docs/context/conventions-core.md`)
- **No pandera** — schemas are plain dicts (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`). Config dicts cached with `functools.cache` (`actiontypes_df`). Converters vectorize via `np.select`, never `apply(axis=1)`. All `warnings.warn(..., stacklevel=2)`. ML uppercase `X`/`Y`/`Pscores` allowed in `vaep/`+`xthreat/` (ruff `per-file-ignores`). (`docs/context/conventions-core.md`)
- **Metric-family output contracts** single-sourced in `metric_contracts` (`METRIC_CONTRACTS`, `*_METRIC_COLUMNS` + grain keys); a new family registers or `tests/test_metric_contracts.py` fails (ADR-098; `docs/context/conventions-core.md`).
- **No rescan-in-loop** — use `silly_kicks._frame_index.group_rows` (ADR-068); item-looping primitives carry a `assert_subquadratic_growth` guard with a scoped `call_counter`/`rows_scanned_counter` registered in `SCALE_GUARDED` (ADR-073; `docs/context/conventions-core.md`).
- **Converter identifiers are sacred** — never override caller `team_id`/`player_id` from qualifiers; qualifier facts surface as dedicated columns (`tackle_winner`/`tackle_loser` on sportec). (ADR-001; `docs/context/providers.md`)
- **Converters are order-insensitive; `action_id` chronological** — sort via `sort_actions_chronologically` before positional derivation; `_finalize_output` raise-guard `_assert_chronological_action_id`; GS gains a required `start_time` input column. `action_id` renumber = atomic cross-table migration. (ADR-065; `docs/context/corpus-drivers.md`)
- **Block-detection columns** `shot_blocked`/`cross_blocked` (nullable `boolean`, shared `_blocked_flag`, 3-valued). (ADR-046; `docs/context/conventions-core.md`)
- **`add_*` enrichers tolerate NaN ids** (`nan_safe_enrichment`/`_nan_safety`, ADR-003) and are **PURE** (input-mutation CI-gated via `PURITY_ENTRIES`; note `pitch_control_at_target` rename, ADR-033). (`docs/context/conventions-core.md`)
- **Dtype-safe id comparisons repo-wide** — never raw `==`/`!=` on ids, never merge id keys unaligned, never `astype(str)` an id used as dict-key/join-token. Use `silly_kicks.id_compat` (`ids_equal`/`ids_differ`/`ids_match`/`same_id`/`align_join_keys`/`restore_id_dtype`/`canonical_id[_series]`); loud guard `validate_id_dtypes`; gated by `PUBLIC_ID_SCALAR_ENTRIES` + the add_* dtype-invariance gate. The AST lint is DELETED — do not reinstate. (ADR-019/043; `docs/context/id-compat.md`)
- **Tracking-frame id dtypes are NULLABLE** (`Int64`, ball row NA by construction); `snapshot_to_tracking_frames` per-column `_cast_to_declared_schema` (the `IntCastingNaNError` trap); `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` is an alias. (ADR-058; `docs/context/tracking-features.md`)
- **Per-action geometry emitted in the SPADL action-LTR frame** — `_action_orientation`/`acting_team_attacks_rtl`/`reproject_to_action_ltr`; gated by `test_mirror_registry` Gate A/B. (ADR-028; `docs/context/orientation.md`)
- **Frame-LTR orientation single-sourced** — `orient_frames_to_ltr` (unlabeled) / `play_left_to_right` (labeled) via `compute_attacking_direction` (ADR-029); native adapters self-correct a wrong/absent `home_team_start_left_extratime` via `finalize_orientation`/`orient_frames_to_ltr_by_geometry` (the fail-loud ET-direction contract, ADR-010/035). Kloppy tracking pins the CS via `_kloppy_coordinates`/`socceraction_coordinate_system` (CS-only `transform`, never `to_pitch_dimensions`; ADR-031). (`docs/context/orientation.md`)
- **The defended goal end has ONE implementation** `resolve_defended_goals -> GoalMap` (build once per match, thread in; keys canonical strings; `attacked_goal` is a real lookup; `GoalEndUnresolvedError` refuses at the edge). Detection is Gate C (`gate_c_must_move`/`test_goal_map_consumers`). (ADR-055; `docs/context/orientation.md`)
- **Direction NEVER from team identity; "unresolved" is a `<NA>` VALUE** — `acting_team_attacks_rtl` returns `boolean`; `_unresolvable_direction_mask` deleted; guarded by `test_d3_direction_invariance`. The goal-relative transform is a 180° POINT reflection (`to_goal_relative_x`/`to_goal_relative_y`; bump `GEOMETRY_VERSION` on numeric change). (ADR-051; `docs/context/orientation.md`)
- **Reflection goes through `silly_kicks/reflection.py`** — `ReflectionKind` per column, `reflect`/`reflect_columns`, `on_unknown="warn"` (`UndeclaredGeometricColumnWarning`); guards per-row not aggregate. (ADR-045; `docs/context/orientation.md`)
- **`ExpectedThreat.xT` stored y-INVERTED** — `physical_grid`/`values_at_points` neutralize it, `require_fitted_xt` guards; `compute_threat_pc` REFUSES an unfitted xT (was 0.0). (ADR-041; `docs/context/xt.md`)
- **Trained artifacts are parameters-only, pickle-free, fail-closed** — npz/booster-JSON + JSON metadata + `SHA256SUMS`; `load()` runs `verify_chirality` + `_feature_contract` (declared constants compare first, `IntegrityError`); `load_xgb_booster_base_score_safe` strips the bracketed base_score. (ADR-011/016/040/044/050; `docs/context/trained-models.md`)
- **Ghost-GK leaf walk numba-accelerated bit-identically** — `_vectorized_leaf_values`/`_FlatTrees` numpy port, `SILLY_KICKS_GHOST_FORCE_NUMPY`, default `kde_backend="auto"` exact. (ADR-076; `docs/context/trained-models.md`)
- **Every model publish uses `publish_model_with_card`** (requires `--model-card`, `create_repo(exist_ok=True)`, card→README, allowlist) — never `upload_model_only` directly. (ADR-088; `docs/context/trained-models.md`)
- **Velocity-keyed model auto-select** serves `position_only` variants on freeze-frames (`variant_key_for_velocity`, `_resolve_ghost_model_for_frames`, `feature_set`); missing → NaN never `default`. (ADR-067; `docs/context/trained-models.md`)
- **Detection-aware provider visibility fails loud** — `_provider_visibility`/`_DETECTION_AWARE_PROVIDERS`/`assert_detection_aware_visibility`; consume-time `validate_corpus_visibility`. (ADR-069; `docs/context/trained-models.md`)
- **Geometry constants canonical + CI-enumerated** — `spadlconfig.penalty_area_half_width`, `in_penalty_area_absolute`/`in_penalty_area_goal_relative` (frame-explicit). (ADR-050; `docs/context/conventions-core.md`)
- **"Nothing there" ≠ "nothing VISIBLE there"** — `point_observed`/`region_observed_fraction`/`add_visible_area_coverage` (NaN on non-observed, never 1.0/0.0). (ADR-055; `docs/context/conventions-core.md`)
- **SB360 visibility companions are OPT-IN + additive** — `classify_region_observation`/`REGION_OBSERVATION_SOURCE_VALUES` (ADR-062). FOV single-sourced through `_fov_registry` (`validate_fov`/`FovDiagnosis`/`append_observability_companions`/`ObservabilityEntry`; observed-AREA of a convex region, ADR-077). (`docs/context/velocity-fov.md`)
- **Velocity availability diagnostic**: `validate_velocity_regime`/`VelocityRegimeDiagnosis`; a provenance COLUMN only where the VALUE changes (`ghost_gk_source`) — the ghost refuses at `_serve_positions_core`. (ADR-054; `docs/context/velocity-fov.md`)
- **Velocity-less providers get the zero-velocity model** via `zero_velocity_if_unavailable`/`_velocity_availability` (Tier-1 lift / Tier-2 NaN / Tier-3 honest-NaN); `compute_xshot_occurrence` guarded by `test_velocity_feature_contract`; `space_opponent_source` softening; `speed_source` ∈ `TRACKING_CATEGORICAL_DOMAINS` (3 tokens). (ADR-063; `docs/context/velocity-fov.md`)
- **GS keeper position clamped at 27.5 m** — `validate_gk_position_clamp`/`GkClampDiagnosis`/`GoalkeeperClampWarning`; GS GK-depth analysis invalid past the clamp. (ADR-083; `docs/context/velocity-fov.md`)
- **`DasUnscoreableError` is the ONLY degradable DAS exception** — `add_das` emits `das_source` over `DAS_SOURCE_VALUES` (ADR-043; `docs/context/tracking-features.md`).
- **Atomic defensive-credit is a FAITHFUL mirror** (`_defensive_credit_atomic_adapter` + the shared `_rollup_defending_aggregate`); the atomic `interception` dedup made `len(atomicconfig.actiontypes)` = 32 → **AtomicVAEP retrain + atomic re-materialize owed**. (ADR-096; `docs/context/tracking-features.md`)
- **Keeper identity has ONE resolver** `resolve_keeper_identities` (native delegates; `KeeperIdentity`, `apply_keeper_identities_to_frames` bridge, `add_defending_gk_player_id`, `KeeperAppearances` port) — public `silly_kicks.keeper_identity` (ADR-078/084/085; `docs/context/gk-metrics.md`). GK-domain keeper id uses resolved `player_key` from `fct_action_context`, never raw `player_id` (`docs/context/gk-metrics.md`).
- **Frame-consuming `add_*` have ONE call shape** (`frames` positional-or-keyword; optional kwargs keyword-only) — `test_call_convention_registry`; producer `run_tracking_features` injects models via `FAMILY_MODEL_REQUIREMENTS`. (ADR-078; `docs/context/tracking-features.md`)
- **Expected-receiver model uses PRE-PASS state ONLY** — `ReceiverModel`, `labeling_strategy_for_provider`, `NoReleaseDirectionError`, `pooling_gate` (ADR-066; `docs/context/tracking-features.md`).
- **Frame-aware `*_xfns` resolve frames POSITIONALLY** via `resolve_frame_ids_by_position`, never by `action_id` (`test_frame_aware_xfns_dup_action_id`; ADR-020; `docs/context/tracking-features.md`).
- **A feature reading its own `result_id`/post-contact outcome ships NO `*_xfns`** or stays out of every default list (`packing_xfns`, `off_ball_run_value_xfns`; ADR-030/047/049; `docs/context/conventions-core.md`).
- **Aggregator liveness CI-gated** (non-NaN + non-constant; `STRUCTURAL_CONSTANTS` empty; `test_aggregator_column_liveness`). Linkage-provenance idempotent (`test_provenance_skip_guard`, `link_quality_score`); pre-linking via `links` kwarg (`_pre_index_frames`, `infer_ball_carrier`); shared surface via `pitch_control_cache`/`PitchControlCache` (ADR-008). Frame-memory `category` for static low-card only (`ball_state`), bounded LRU, `compute_pitch_control_batch` (ADR-103). (`docs/context/tracking-features.md`)
- **Vectorized spearman batch is byte-identical via concat-and-slice** — `compute_spearman_batch`/`compute_pitch_control_batch` share the single-sourced `_spearman_combine`; the precondition is preserved per-frame row order (influence `.sum(axis=0)` is order-dependent). `compute_threat_pc_batch` scores the keeper-removed counterfactual over EXPLICIT slices, NEVER the frame-keyed cache (the ADR-043 landmine). rest_defense layer-2 + off_ball chunk via an invariance-gated `batch_size`; DAS cost guardrail `estimate_das_cost`/`DasCostWarning`. (ADR-105; `docs/context/tracking-features.md`)
- **Warning categories are separate, never one umbrella** (`SyntheticEPVWarning`/`IgnoredSurfaceInputsWarning`/`OrientationUnresolvedWarning`, none subclasses another). Policy at the edge, engine pure (`predict_mean`/`predict_proba` stay clean scorers). (`docs/context/conventions-core.md`)
- **Every band tested from BOTH sides; every counterfactual asserts non-vacuity** it moved something (`test_ghost_gk_mirror_invariant` off-centre probe). (`docs/context/conventions-core.md`)
- **A liveness gate's FIXTURE needs its own precondition test** (in-domain, no constant feature; the `PR-S118` degeneracy). (ADR-032; `docs/context/conventions-core.md`)
- **Private modules can have downstream consumers** — check `docs/PRIVATE_CONSUMERS.md` before renaming a `_*.py` (path pins fail silently). (`docs/context/conventions-core.md`)
- **Library ships RAW primitives** — composites/archetypes/rankings stay consumer-side (`structural_lbs`/`structural_sgm`/`structural_sdi`; ADR-009). `detect_input_convention` requires DISCRIMINATING evidence and DEFERS otherwise (`_a_team_spans_periods`, `POSSESSION_PERSPECTIVE`; ADR-059). (`docs/context/conventions-core.md`)
- **Academic attribution** — every published-methodology feature gets a `NOTICE` entry (ADR-005). Feature columns documented in `feature_glossary` (`FeatureColumn`; `describe_level` in reporting.py); the C4 model (`docs/c4`) is pinned to the code. (ADR-048; `docs/context/conventions-core.md`)
- **A coverage denominator must not masquerade as a signal** — `n_valued_disruptive_runs` never `n_disruptive_runs`; keep the NaN row (`RunValueCoverageWarning`, `peak_speed_source`). (ADR-042; `docs/context/conventions-core.md`)
- **Artifact-writing drivers REFUSE a dirty tree** — `require_clean_tree(git_provenance())`, stamp `run_tree_dirty` (`_provenance`; ADR-037). Corpus visibility fail-closed (`is_public_row`/`assert_public_corpus`/`artifact_label`; ADR-038). A validation harness makes the observed outcome unrepresentable (`InjectionSpec`, `att_power_curve`, `n_degenerate_by_size`). (`docs/context/corpus-drivers.md`)
- **Every corpus driver adopts `scripts/_driver.py`** — `for_each(items, key=, work=, shard_root=, token_inputs=)`, `assert_conservation` + `_require_injective`, `reconcile`; expensive passes are their own shardable drivers (`write_table_atomically`/`providers_for_slice`). (ADR-052; `docs/context/corpus-drivers.md`)
- **Every corpus driver RESUMES BEFORE LOAD** — `for_each(load=)` over cheap refs (`list_match_refs`), `.excluded.json`/`ItemExcluded`, xT fit is an events-only `xt_count_pass`, edge admission `events_only_loader`, `_KEY_EXCEPTIONS`; 4 AST rules. (ADR-052; `docs/context/corpus-drivers.md`)
- **A registry gate DERIVES its population and asserts it EXACTLY** — `ARTIFACT_DRIVERS`, `_UNDERIVABLE` asserted empty, single-sourced `_script_population`, content predicates read `string_literals` not docstrings (ADR-056). An artifact `declare_inputs` (`_input_contract`) declares WHICH symbols its numbers depend on. A fixture generator (`_generate_synthetic_match`) reproduces its committed fixture byte-for-byte first. (`docs/context/corpus-drivers.md`)
- **A spec changing a public seam enumerates every caller** of every changed function with evidence both sides — the sweep is the FLOOR (`_dominant_region_area`, `derive_opengoal_range` were the misses). (`docs/context/corpus-drivers.md`)
- **The pining `statsbomb` loader single-sources `_sb_raw`** (`flatten_events`, `shape_snapshots` resolves the `teammate` flag to real team ids). (ADR-062; `docs/context/corpus-drivers.md`)

## Testing

```bash
python -m pytest tests/ -m "not e2e" -v --tb=short
```

- Lint at CI scope, never `.` — `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright` bare. → `docs/context/ci.md`.
- CI duration-shards via `pytest-split` (`--splits`, `shard-reconcile`, committed `.test_durations`, `test_ci_shard_wiring`; ADR-074). Pandas-major span DECLARED (`test_ci_pandas_span_wired`; `.to_numpy()` needs `copy=True` under CoW; ADR-057). Lint pins (`test_ci_lint_pins_wired`, `pandas-stubs`). Build backend bounded + publisher pinned (`test_ci_publish_guard_wired`, `gh-action-pypi-publish`, `Metadata-Version`). Doctests on public surface (`--doctest-modules`, `test_ci_doctest_wired`). `@pytest.mark.slow` = expensive + invariant, run once in a dedicated `slow` job — not inline (`test_ci_slow_gating_wired` + conservation `test_ci_slow_reconcile_wired`; ADR-023). → `docs/context/ci.md`.
- **Every `add_*` carries an SB360 freeze-frame verdict** — observation locked, adjudication reviewable; `verdict_provenance`, `UNAUDITABLE_BOUNDARY` empty; regenerable via `_adjudicate` (ADR-053). Detection lands BEFORE the fix; every registry gate has an anti-rot meta-assertion (ADR-051). → `docs/context/ci.md`.
- **Claims about a gate's behaviour quote the assertion body, not its registration** (measure at the scale you assert at). → `docs/context/ci.md`.

## Open Items

See [TODO.md](TODO.md).

## Dependencies

Runtime: pandas, numpy, scipy, scikit-learn (no pandera). Optional: kloppy (tracking parsers), xgboost, catboost, lightgbm, `accessible-space` (DAS), `ruthless-efficiency[optuna]>=0.4.0` + xgboost>=2.0,<4.0 (`[calibration]`/`[train]`), numba (`[numba]`), `statsbombpy` (importorskip-guarded e2e only). numpy>=2.0 compatible. → `docs/context/conventions-core.md`.
