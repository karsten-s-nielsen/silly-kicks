# One Cross Definition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One provider-neutral geometric `cross` / set-piece delivery rule in every SPADL converter, plus a tracking enricher that gives `cross_attempt` and `cross_height`, and the consumers that read them.

**Architecture:**
- Layer 1 is a pure function in `spadl/base.py`. Each converter calls it at one precondition-defined position, and each native label moves to a new `cross_native` column.
- Layer 2 is a frame-consuming `add_*` enricher, TF-86. It aligns the release through ELASTIC (TF-43), follows the ball to the first non-passer contact (ADR-124/125), and labels the delivery.
- The consumers that read the layers: xCross label v2, causal opportunities, TF-46 `corner_kind`, and bravery.

**Tech Stack:** Python 3.10+, pandas 2/3 (CoW), numpy, pytest. DGX for tests and data runs.

**Spec:** `docs/superpowers/specs/2026-11-13-cross-definition-design.md` (r6; approved by the owner and by reviewers A and B). Read it with this plan. Section numbers below (§) refer to it.

## Global Constraints

- **Branch:** `feat/cross-definition` off `main` `0cc07e6c`. No worktree, no subagents.
- **Two commits for the cycle, by owner ruling (2026-11-13), under the provenance exception of the commit policy.** There are no per-task commits: every "Commit" step below is replaced by "ledger the task". The ledger is `.superpowers/sdd/2026-11-13-cross-definition-plan/progress.md`. The order is fixed:
  1. Tasks 1–8, and the code and tests of Tasks 9–10 (Steps 1–3), on the uncommitted tree.
  2. `/final-review`, then the A/B implementation reviews **on the uncommitted tree**. Both reports are answered.
  3. **Gate 1: commit 1 and push** (owner yes). Commit 1 holds the code, the tests and the drivers, with provisional constants: `CROSS_START_X = 52.5` (the most permissive candidate) and the `CrossDeliveryParams` defaults.
  4. The Task 9 and Task 10 DGX runs, on the clean commit 1 (`require_clean_tree`, ADR-037). Each driver computes every candidate value itself, so its numbers are exactly those of the final constants.
  5. Commit 2's content, on the uncommitted tree: the constants, the memos (stamped with commit 1's sha), Task 11's number-dependent docs, and a test that pins the defaults to the memo values. The number-independent docs (C4, AGENTS, docs/context, ADR-133's decision) already went in commit 1. A/B re-review this delta, and both reports are answered.
  6. **Gate 2: commit 2 and push** (owner yes).
  7. **Gate 3: the draft PR** (owner yes). Watch CI until it is green.
  8. The merge, the tag and the release each need their own yes later. Merge with `--merge`, never squash: the memos cite commit 1's sha, and a squash would orphan it.
- **The `compute_bravery` loud break is accepted by the owner** (2026-11-13). The release notice names it, including the lakehouse `bravery_writer.py`.
- **Tests run on the DGX checkout `~/sk-fsg`** (`karsten@192.168.68.73`). The AI-PC runs nothing heavy.
  - **Step 0, before Task 1:**
    - Create `D:\Temp\cross-def\sync.sh` as a copy of `D:\Temp\widening\sync.sh` with `BASE=0cc07e6c`.
    - Check that no DGX job uses `~/sk-fsg`, then run it once.
    - Expected: `~/sk-fsg` at `0cc07e6c` plus the working-tree files.
  - For the Step 4 runs after Gate 1, reset `~/sk-fsg` to commit 1 exactly (`git fetch && git reset --hard <commit-1-sha> && git clean -fd`), so the tree is clean.
  - Test command: `cd ~/sk-fsg && .venv/bin/python -m pytest <paths> -m "not e2e and not slow" -p no:randomly -q --basetemp ~/cross-def/pt`.
  - Lint: `.venv-lint/bin/ruff check silly_kicks/ tests/ scripts/`, `.venv-lint/bin/ruff format --check silly_kicks/ tests/ scripts/`, `.venv-lint/bin/pyright --pythonpath .venv/bin/python`.
  - The full suites run in CI on a draft PR (owner ruling).
- **Temp files** go under `D:\Temp\cross-def\` and DGX `~/cross-def/`, never C:.
- **Geometry constants** come from `spadlconfig`: `field_length` 105.0, `field_width` 68.0, `penalty_area_depth` 16.5, `penalty_area_half_width` 20.16.
- **Ids** are compared with `silly_kicks.id_compat` (`ids_equal` / `ids_differ`), never with raw `==`.
- **Every `warnings.warn`** uses `stacklevel=2`.
- **TF id:** TF-86. TF-85 is held by the campaign-throughput cycle (cycle 1b, the controller-step read fix). Its TODO row is a pending patch for that cycle's final commit, so it is not in `TODO.md` at `0cc07e6c`. Do not use TF-85.
- **The version is single-sourced** in `silly_kicks/_version.py`. Bump the minor at commit-prep only (4.143.0 → 4.144.0), unless another release lands first.
- **Data comes from pining only.** Databricks is not a source. No HF job or dataset upload is part of this cycle.

## Review Focus

1. **No next action:** a delivery that is the last on-ball row of a period (or of the match) has no next action. It is typed `pass` / `_short` and raises nothing. Test owner: Task 1 (`test_last_row_in_period_is_not_a_delivery`).
2. **NaN coordinates:** a pass-class row or its next action has a NaN `start_x` / `start_y`. The row is not a delivery, and nothing raises. GS has NaN ends on 0.8 % of open passes. Test owner: Task 1 (`test_nan_coordinates_never_deliver`).
3. **Mixed id dtypes:** `team_id` is `int64` on one row type and `object` / `category` on another (GS synthetic rows, SkillCorner). The same team must compare equal. Test owner: Task 1 (`test_team_id_dtype_invariance`).
4. **Extra time and shoot-outs:** periods 3 and 4 behave like periods 1 and 2. Period 5 has only `shot_penalty` rows, so nothing is re-typed. Test owner: Task 1 (`test_extra_time_and_shootout_periods`).
5. **Frames that cover only part of a match:** one period has no ball rows, or all ball z values are NaN. `cross_attempt` is `<NA>` (unknown) for that period, `cross_height` is `<NA>`, and nothing raises. Test owner: Task 5 (`test_period_without_ball_frames_is_unknown`).

---

### Task 1: The shared delivery rule

**Files:**
- Modify: `silly_kicks/spadl/base.py`, adding constants and `classify_deliveries`
- Test: `tests/spadl/test_classify_deliveries.py`, a new file

**Interfaces:**
- Produces: `classify_deliveries(actions: pd.DataFrame, *, cross_start_x: float = CROSS_START_X) -> pd.DataFrame` in `silly_kicks/spadl/base.py`.
  - It returns a copy with `type_id` re-typed for the rows of §6.1 conditions 1–4 and §6.1b. Every other column is unchanged.
  - It requires `period_id`, `team_id`, `type_id`, `start_x`, `start_y` in canonical SPADL LTR, in final chronological order, with no synthetic dribbles. It does not sort.
- Produces: the module constants
  - `CROSS_START_X: float = 52.5`, provisional and the most permissive candidate, in commit 1. Task 9 fixes it in commit 2. A cross under 52.5 that starts below a higher line X is exactly a `pass` under X, so the drivers can evaluate every candidate from one conversion by filtering on `start_x`.
  - `_DELIVERY_SKIP_TYPE_IDS`: `non_action`, `foul`.
  - `_RESTART_TYPE_IDS`: the ids of `_SET_PIECE_RESTART_TYPE_NAMES` plus `shot_penalty` and `shot_freekick`.

- [ ] **Step 1: Write the failing tests.** Build each fixture with a small `_actions(rows)` helper in the test file. Every band is tested from both sides (house rule):
  - `test_wide_advanced_delivery_into_box_is_cross`: start (80, 5); next same-team action at (95, 34) → `cross`.
  - `test_start_line_both_sides`: `start_x` at `CROSS_START_X − 0.01` → `pass`; at `CROSS_START_X` → `cross`.
  - `test_wide_edge_both_sides`: `start_y` at 13.84 − 0.01 → `cross`; at 13.84 → `pass`. The same at the 54.16 edge.
  - `test_box_edges_both_sides`: next at `x` 88.5 → `cross`, at 88.49 → `pass`. Next at `|y − 34|` 20.16 → `cross`, at 20.17 → `pass`.
  - `test_opponent_next_action_is_mirrored`: the opponent's next action at (10, 34) is in the passer's box → `cross`. At (95, 34) (the passer's own half) → `pass`.
  - `test_skips_foul_and_non_action`: a `foul` and a `non_action` between the delivery and an in-box action still give `cross`.
  - `test_native_dribble_counts_as_next_action`: a native `dribble` starting outside the box → `pass`.
  - `test_restart_next_action_is_not_a_delivery`: next is `goalkick` / `corner_crossed` / `throw_in` → `pass`. A `foul` followed by `shot_penalty` → `pass`.
  - `test_native_cross_with_short_delivery_becomes_pass`: a `cross` row whose next action is outside the box → `pass`.
  - `test_corner_delivery_both_sides` / `test_freekick_delivery_both_sides`: next in the box → `corner_crossed` / `freekick_crossed`; outside → `_short`. A restart next → `_short`. A central free kick (y = 34) into the box → `freekick_crossed`.
  - `test_unchanged_types`: `throw_in`, `goalkick`, `shot_freekick`, `shot_penalty`, `shot` and `dribble` rows keep their `type_id`.
  - `test_last_row_in_period_is_not_a_delivery`: Review Focus 1.
  - `test_nan_coordinates_never_deliver`: Review Focus 2.
  - `test_team_id_dtype_invariance`: Review Focus 3. `int64` vs `object` vs `category` `team_id` gives identical output.
  - `test_extra_time_and_shootout_periods`: Review Focus 4.
  - `test_period_boundary`: the next row is in the next period → `pass`.
  - `test_pure`: the input frame is unchanged (`pd.testing.assert_frame_equal` against a deep copy).
- [ ] **Step 2: Run the tests.** Expected: every test FAILS with `ImportError: cannot import name 'classify_deliveries'`.
- [ ] **Step 3: Implement `classify_deliveries` and the constants** in `spadl/base.py`. Approach:
  - Build a mask of on-ball rows (not in `_DELIVERY_SKIP_TYPE_IDS`).
  - Take the next on-ball row per `period_id` with a positional shift over the on-ball subset.
  - Compare teams with `id_compat.ids_differ`.
  - Mirror a different-team next start as `(field_length − x, field_width − y)`.
  - Test the box with `spadlconfig` constants.
  - Re-type with `np.select`. No `apply(axis=1)`, no loop over rows.
- [ ] **Step 4: Run the tests.** Expected: every test PASSES.
- [ ] **Step 5: Ledger the task.**

### Task 2: The `cross_native` column

**Files:**
- Modify: `silly_kicks/spadl/schema.py`: `SPADL_COLUMNS` and the per-provider dicts at lines 47, 54, 67 and 100.
- Modify: `silly_kicks/spadl/utils.py` `_finalize_output`, but only if the nullable `boolean` cast needs a case.
- Modify: `silly_kicks/atomic/spadl/base.py` `convert_to_atomic`, to carry `cross_native` under `preserve_native` the same way as `cross_blocked`.
- Test: `tests/spadl/test_cross_native_schema.py` (new) and `tests/atomic/` (the existing preserve-native test file; extend it).

**Interfaces:**
- Produces: the SPADL column `cross_native` of dtype `"boolean"`, placed after `cross_blocked` in every schema dict.
- Consumes: none.

- [ ] **Step 1: Write the failing tests.**
  - `test_cross_native_in_every_schema`: `"cross_native"` is in `SPADL_COLUMNS` and in each per-provider dict, with dtype `"boolean"`.
  - `test_finalize_keeps_nullable_cross_native`: a frame with `[True, False, pd.NA]` round-trips through `_finalize_output` unchanged.
  - `test_atomic_preserves_cross_native_opt_in`: `convert_to_atomic(actions, preserve_native=["cross_native"])` keeps the column on the rows derived from each action. Without the opt-in it is absent.
- [ ] **Step 2: Run the tests.** Expected: FAIL (`KeyError` / assertion).
- [ ] **Step 3: Implement.** Until Task 3 lands, converters that do not set the column get all `<NA>` from `_finalize_output`.
- [ ] **Step 4: Run the new tests, `tests/spadl/` and `tests/atomic/`.** Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 3: Wire the eight converters

**Files:**
- Modify:
  - `silly_kicks/spadl/statsbomb.py`, `opta.py`, `wyscout.py` (with `_wyscout_mappings.py`), `metrica.py`, `sportec.py`, `gradientsports.py`, `kloppy.py`, `skillcorner.py`.
  - `docs/PRIVATE_CONSUMERS.md` if it pins `skillcorner._is_cross`. Check it first.
- Test:
  - `tests/spadl/test_converter_reorder_regression.py` (new): the kloppy and SkillCorner reorder.
  - `tests/spadl/test_delivery_wiring.py` (new): per converter, direction, order-insensitivity and `cross_native`.
  - Update the existing per-converter tests that assert the old cross / set-piece typing. List each one in the ledger with its reason.

**Interfaces:**
- Consumes: `classify_deliveries` (Task 1) and the `cross_native` schema column (Task 2).
- Produces: every converter's output carries `cross_native` per §6.2, and types as §6.1 / §6.1b.

- [ ] **Step 1: The reorder regression, before any other change.**
  - Write `test_kloppy_reorder_is_byte_identical` and `test_skillcorner_reorder_is_byte_identical`.
  - Each converts the existing fixtures with the current code (golden frames captured on `0cc07e6c` into `tests/spadl/fixtures/reorder_golden/*.parquet`) and asserts `assert_frame_equal` against the new output, with `type_id` and `cross_native` excluded.
  - Move `to_spadl_ltr` before `_add_dribbles` in `kloppy.py` (239 → before 231) and `skillcorner.py` (557 → before 550), with no other change.
  - Prove the test can fail: perturb one `start_x` in one golden parquet by 0.5 m and run the test. Expected: FAIL. Then restore the golden.
  - Run the tests. Expected: PASS. For SkillCorner the move is a POSSESSION_PERSPECTIVE no-op (review B round 2).
- [ ] **Step 2: Write the failing wiring tests** in `tests/spadl/test_delivery_wiring.py`. Parametrize over the 8 converters with the fixtures in `tests/spadl/_converter_cases.py`:
  - `test_native_cross_short_delivery_becomes_pass[conv]` and `test_native_pass_into_box_from_wide_becomes_cross[conv]`. Use a synthetic event pair per provider; build it with the provider's own raw schema from the fixtures.
  - `test_direction_invariance[conv]`: the same delivery mirrored (team attacking right-to-left) and in period 2 gets the same `type_id`.
  - `test_order_insensitive[conv]`: shuffle the raw input within equal-timestamp blocks; the output is the same (ADR-065).
  - `test_cross_native_values[conv]`: the provider's native open-play cross → `True`, native pass → `False`, native corner/free-kick crossed/short → `True` / `False`, other rows → `<NA>`. For SkillCorner, every row is `<NA>`.
  - `test_restart_ended_set_piece_is_short[conv]`: a corner, then a goal kick → `corner_short`.
  - `test_cross_blocked_unchanged[conv]`, for every converter that emits a non-`<NA>` `cross_blocked` (StatsBomb, GS, Wyscout; confirm the set with `git grep -n cross_blocked silly_kicks/spadl`): `cross_blocked` equals the value on `0cc07e6c` for every row where `cross_native` is True.
- [ ] **Step 3: Run the tests.** Expected: FAIL. The types follow the old native rules, and `cross_native` is all `<NA>`.
- [ ] **Step 4: Implement each converter.** Per converter, three changes:
  - (a) Compute `cross_native` from the native rule that decides the type today.
  - (b) Let that rule type the row as `pass` / the set-piece base type only for the order in which rows are built. The final type comes from (c).
  - (c) Call `classify_deliveries` at the §6.1 position table row.
  - Converter specifics:
    - StatsBomb: keep `_open_play_cross_mask` as the source of `cross_native` and `cross_blocked`.
    - SkillCorner: delete `_is_cross` and its call (338).
    - Sportec: move `play_height` / `play_flat_cross` and the ADR-127 corner discriminator into `cross_native`.
    - GS: call after `to_spadl_ltr` (856), before `_finalize_output`.
    - kloppy: `PassType.CROSS` feeds `cross_native`.
  - **Docstrings (§6.4):** each converter's `convert_to_actions` docstring states the event-only limits:
    - a blocked cross and a restart-ended delivery are `pass`;
    - a restart-ended set piece is `_short`;
    - for SkillCorner, the missing in-box defensive touches;
    - the provider's measured recall against `cross_attempt`, written as a placeholder `<recall: Task 10>` that Task 11 fills from the Task 10 numbers.
- [ ] **Step 5: Run `tests/spadl/` and `tests/atomic/`.** Expected: PASS. Every changed existing assertion is ledgered with its reason.
- [ ] **Step 6: Ledger the task.**

### Task 4: `cross_blocked` consumers and bravery

**Files:**
- Modify: `silly_kicks/tracking/defensive_credit/_bravery.py`, lines 17, 18, 64, 65, 68 and 113–114.
- Modify: `silly_kicks/atomic/tracking/features.py`, lines 330, 357, 433, 471 and 477 (atomic `compute_bravery`).
- Test: `tests/tracking/test_bravery.py`, extended.

**Interfaces:**
- Consumes: the `cross_native` column (Tasks 2 and 3).
- Produces: `compute_bravery` reads `cross_native == True` rows for `n_open_play_crosses_faced`, `b_open`, `_known_domain` and the set-piece `n_set`.
  - Its output columns and meanings do not change.
  - When `cross_native` is absent, it raises `ValueError` and names the column. It never falls back silently.
  - This affects callers that pass SPADL produced before this release, for example materialized lakehouse actions. Those callers must re-convert first. The Task 11 Hyrum list names this break.

- [ ] **Step 1: Write the failing tests.**
  - `test_bravery_unchanged_under_retyping`: take a fixture whose native crosses partly become `pass` and whose native passes partly become `cross`. The outputs equal those of the same fixture before re-typing.
  - `test_bravery_atomic_matches_standard`: the atomic path gives the same outputs.
  - `test_bravery_requires_cross_native`: missing column → `ValueError`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.** Thread `cross_native` through the atomic path exactly as `cross_blocked` is threaded.
- [ ] **Step 4: Run `tests/tracking/test_bravery.py`, `tests/tracking/test_defensive_credit_*.py` and `tests/atomic/`.** Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 5: TF-86 `add_cross_deliveries`

**Files:**
- Create: `silly_kicks/tracking/_cross_delivery.py`.
- Modify:
  - `silly_kicks/tracking/__init__.py` (export) and `silly_kicks/tracking/features.py`, if public `add_*` functions are re-exported there; follow the `add_ball_contacts` pattern.
  - `silly_kicks/metric_contracts.py`: a `cross_delivery` contract with `CROSS_DELIVERY_METRIC_COLUMNS = ("cross_attempt", "cross_height")` and the action grain keys.
  - `silly_kicks/feature_glossary.py`: two `FeatureColumn` entries with interpretation text.
  - `silly_kicks/reflection.py`: `cross_attempt`, `cross_height` and `cross_native` declared non-geometric.
  - `tests/test_add_star_purity.py` (`PURITY_ENTRIES`), `tests/tracking/test_call_convention_registry.py`, `tests/sb360/test_registry_surface.py` with an SB360 verdict (ADR-053; regenerate through `_adjudicate`).
  - `tests/tracking/test_aggregator_column_liveness.py`. Registration is mandatory: `test_meta_surface_complete` (:795-797) pins every `add_*` in `tracking.__all__`. It comes with a fixture precondition test (ADR-056): the fixture holds an attempt and a non-attempt, and an aerial and a ground delivery.
  - The public-id scalar surface in `tests/invariants/conftest_id_scalar.py`, only if `add_cross_deliveries` falls under ADR-019's public-id rules. Check its registry comment.
  - `NOTICE`: Vidal-Codina et al. 2022, arXiv:2202.00804.
- Test: `tests/tracking/test_cross_delivery.py` (new).

**Interfaces:**
- Consumes:
  - `detect_ball_contacts(frames, *, params) -> DataFrame` and `first_contact_in_window(contacts, period_id, t0, t1)` from `tracking/_ball_contact.py`;
  - `BallContactParams.aerial_z_m`;
  - `align_events_to_frames(actions, frames, *, params) -> DataFrame` and `ElasticSyncParams.min_confidence` from `tracking/_elastic_sync.py`;
  - the GoalMap and action-LTR helpers (ADR-028/055).
- Produces: `add_cross_deliveries(actions: pd.DataFrame, frames: pd.DataFrame, *, params: CrossDeliveryParams = CrossDeliveryParams()) -> pd.DataFrame`.
  - It is decorated `@nan_safe_enrichment`.
  - It adds `cross_attempt` (`"boolean"`) and `cross_height` (`"category"` over `CROSS_HEIGHT_VALUES = ("aerial", "ground")`).
  - If the actions carry `elastic_frame_id` / `elastic_confidence`, it reads them. Otherwise it calls `align_events_to_frames` once.
- Produces: `CrossDeliveryParams`, a frozen dataclass.
  - Provisional values: `block_radius_m=5.0`, `direction_window_s=0.2`, `release_gap_s=0.12`, `min_ball_coverage=0.8`, `release_match_m=3.0`, `path_cap_s=4.0`.
  - It also has `cross_start_x: float = CROSS_START_X`, the zone line of the §6.3 table. A driver can then pass any candidate line explicitly.
  - Task 10 fixes the values in commit 2.

- [ ] **Step 1: Write the failing tests.** Use synthetic frames built with the ball-contact test helpers in `tests/tracking/test_ball_contact.py`:
  - `test_row_class_table`: every row of the §6.3 table gets its stated value:
    - cross-zone pass, answerable → True/False;
    - unanswerable → `<NA>`;
    - outside-zone pass → False;
    - corner / free kick → True/False;
    - shot, `shot_freekick`, `shot_penalty`, throw-in, goal kick and dribble → `<NA>`.
  - `test_path_enters_box_is_attempt` and `test_path_stays_wide_is_not`.
  - `test_blocked_cross_heading_into_box`: the first contact is an opponent within `block_radius_m`, and the initial direction points into the box → True. With the direction pointing away → False. Just outside `block_radius_m` → decided by the path only.
  - `test_passer_excluded`: the event time is 0.3 s before the kick, and the passer's touch must not end the path.
  - `test_path_capped_without_contact`.
  - `test_restart_ended_delivery_typed_from_path`: in the actions, a cross-zone `pass` (typed `pass` by the converter because the next action is a `goalkick`) and a `corner_short` (next action a `throw_in`). In the frames, both ball paths enter the box. Both rows → `cross_attempt` True.
  - `test_coverage_both_sides`: coverage just below and at `min_ball_coverage`.
  - `test_release_alignment`: a confident ELASTIC match moves the release. Precomputed `elastic_*` columns are read; patch `align_events_to_frames` to raise, to prove it is not called.
  - `test_misaligned_release_is_unknown`: the ball at the release is `release_match_m + 0.01` from the action start → `<NA>`; at `release_match_m` → decided.
  - `test_height_aerial_ground_and_estimated`: peak z 1.81 → `aerial`, 1.79 → `ground` (threshold from `BallContactParams.aerial_z_m`). `z_source == "estimated"` → `<NA>`.
  - `test_orientation_invariance`: a mirrored match gives identical columns.
  - `test_period_without_ball_frames_is_unknown`: Review Focus 5.
  - `test_pure`: the inputs are unchanged.
- [ ] **Step 2: Run the tests.** Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `_cross_delivery.py`.
  - Release: §6.3 steps 1–3.
  - Contacts: computed once per call, then a lookup per row through `first_contact_in_window` with `t0 = release + release_gap_s`. The non-passer filter uses `id_compat.ids_differ` on `player_id`.
  - Group frames with `silly_kicks._frame_index.group_rows`, not a rescan per row (ADR-068). Register a `SCALE_GUARDED` entry if the per-row loop is item-looping.
- [ ] **Step 4: Register in the house gates** listed under Files. Run:
  - `tests/tracking/test_cross_delivery.py`
  - `tests/test_add_star_purity.py`
  - `tests/tracking/test_call_convention_registry.py`
  - `tests/test_metric_contracts.py`
  - `tests/test_feature_glossary_*.py`
  - `tests/test_reflection.py`
  - `tests/tracking/test_aggregator_column_liveness.py`
  - `tests/sb360/`
  - `tests/test_notice_md_format.py`
  - `tests/test_enrichment_nan_safety.py`

  Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 6: xCross label v2 and causal opportunities

**Files:**
- Modify:
  - `silly_kicks/tracking/_xcross_attempt.py`: `build_xcross_labels` (353), `prepare_xcross_training_data` (~426–560), and the metadata writer (733) and reader (802).
  - `silly_kicks/causal/opportunities.py`, lines 96–103.
- Test: `tests/tracking/test_xcross_attempt.py` (extend the Task 4 section at line 157) and `tests/causal/test_opportunities.py`.

**Interfaces:**
- Consumes: `cross_attempt` (Task 5).
- Produces:
  - `build_xcross_labels(...)` returns a float Series: 1.0 for a positive, 0.0 for a negative, NaN for unknown (§6.5).
  - `prepare_xcross_training_data` drops NaN labels and reports `n_unknown_label_frames` per match in its returned meta.
  - Metadata key `xcross_label_version: 2`. A missing key reads as 1.
  - `xcross_config(model_metadata)` selects `cross_attempt == True` rows for version 2, and the SPADL type for version 1.

- [ ] **Step 1: Write the failing tests.**
  - `test_label_row_classes`: the §6.5 table. `"cross"` reads `cross_attempt`; `"corner_crossed"` reads corner rows with `cross_attempt == True`.
  - `test_unknown_label_only_from_selected_kind`: a horizon holding only a non-zone pass or a shot keeps a known 0. A horizon holding a cross-zone row with `<NA>` gets NaN.
  - `test_missing_cross_attempt_raises`: any `cross_types` → `ValueError` whose message contains `add_cross_deliveries`.
  - `test_unknown_frames_excluded_and_counted`.
  - `test_label_version_written_and_defaulted`: written as 2; an old metadata dict without the key reads as 1.
  - `test_causal_selects_by_label_version`: v1 → type rows, v2 → `cross_attempt` rows.
  - `test_xshot_labels_unchanged` is a non-regression guard. It PASSES before and after this task: `_build_occurrence_labels` (shared with xShot, `_xshot_occurrence.py:33,340`) gives identical xShot labels on the existing xShot fixture. Pin the expected labels from `0cc07e6c`.
- [ ] **Step 2: Run the tests.** Expected: every test FAILS except `test_xshot_labels_unchanged`, which PASSES.
- [ ] **Step 3: Implement.** The bundled `_xcross_weights/*/metadata.json` stay as they are. They are v1 artifacts, and the owed refit replaces them (decision 6).
  - Every caller that builds xCross labels must now call `add_cross_deliveries` first, or pass actions that carry `cross_attempt`. Update each caller and its test, and ledger each one:
    - `scripts/train_xcross_attempt.py`, the refit's HF trainer;
    - `scripts/hf_jobs/headtohead.py`;
    - `scripts/measure_box_constant_delta.py`;
    - `scripts/measure_f1b_feature_delta.py`;
    - `tests/test_public_api_examples.py`;
    - `tests/tracking/test_position_only_extractors.py`;
    - `tests/tracking/test_xcross_attempt_e2e.py`, `_integration.py` and `_real_data.py`;
    - `tests/tracking/test_xcross_return_meta.py`;
    - `tests/scripts/test_measure_f1b_feature_delta.py`;
    - `tests/invariants/conftest_id_scalar.py`.
  - Find any further caller with `git grep -nE "build_xcross_labels|prepare_xcross_training_data"`. That list is the floor.
  - `train_xcross_attempt.py` is in the HF in-job import closure. Its change must keep `tests/scripts/test_compute_contract.py` and the dataset read audit green.
- [ ] **Step 4: Run** these tests. Expected: PASS.
  - `tests/tracking/test_xcross_*.py`, `tests/tracking/test_xshot_*.py`, `tests/tracking/test_position_only_extractors.py`;
  - `tests/causal/`, `tests/test_bundled_*`, `tests/test_public_api_examples.py`;
  - `tests/scripts/test_measure_f1b_feature_delta.py`, `tests/scripts/test_compute_contract.py`, `tests/scripts/hf_jobs/`;
  - `tests/invariants/`.
- [ ] **Step 5: Ledger the task.**

### Task 7: TF-46 `corner_kind` from `cross_attempt`

**Files:**
- Modify:
  - `silly_kicks/corners/_identify.py`, lines 95–117;
  - `silly_kicks/corners/_columns.py`: add `CORNER_KIND_SOURCE = "corner_kind_source"` and `CORNER_KIND_SOURCE_VALUES = frozenset({"tracking", "event", "gi"})`, and the column in `CORNER_IDENTIFY_COLUMNS`;
  - `feature_glossary.py`: a `corner_kind_source` entry.
- Test:
  - `tests/corners/test_identify.py` and `tests/corners/test_columns.py`;
  - `tests/scripts/` for `build_corner_routines` selection, wherever its existing test lives (find it with `git grep -l build_corner_routines tests`).

**Interfaces:**
- Consumes: `cross_attempt` (Task 5).
- Produces: `corner_kind` keeps the domain `{"direct", "short", "unclear"}`, and `corner_kind_source` takes values in `CORNER_KIND_SOURCE_VALUES`.

- [ ] **Step 1: Write the failing tests.**
  - `test_kind_from_tracking`: True → `direct`/`tracking`; False → `short`/`tracking`.
  - `test_kind_fallback_to_type`: `<NA>` → the type-based value, with source `event`.
  - `test_kind_without_column`: column absent → today's values, with source `event`.
  - `test_gi_wins`: the GI refinement → `unclear`/`gi` over a tracking `direct`.
  - `test_source_vocabulary_closed`.
  - `test_routines_select_tracking_direct`: `build_corner_routines` keeps a tracking-`direct` corner and skips a tracking-`short` one.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** per §6.4 "TF-46 corners".
- [ ] **Step 4: Run `tests/corners/` and the routine test.** Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 8: The caller sweep (evidence on both sides)

**Files:**
- Create: `docs/research/cross-definition/caller-sweep.md`.
- Modify: only code the sweep classifies as "must change to keep its meaning" and that Tasks 3–7 did not already change. Each such change gets a failing test first, in that module's test file.

**Interfaces:**
- Consumes: the four seams of §9, and every hit of a fresh `git grep` per seam:
  - `"cross"` / `actiontype_id["cross"]`;
  - `"pass"` alone;
  - `corner_` / `freekick_`;
  - `cross_native` / `cross_blocked`.
- Produces: a table with one row per hit. The columns are: file:line, seam, what it reads, the classification, the evidence, and the test or reason.
  - Classifications: `value shifts as intended` / `reads the union, unaffected` / `must change` / `trained artifact: retrain owed`.

- [ ] **Step 1: Run the four greps** over `silly_kicks/` and `scripts/`, and save the raw hit lists in the memo's appendix. The §9 floor lists must be a subset of the hits; if one is missing, investigate before continuing.
- [ ] **Step 2: Classify every hit.**
  - Read each hit's code.
  - "Unaffected" needs the line that shows the union, or an independence argument.
  - Value-changing pass readers: `expected_passing/_model.py:26,133` (PassCompletionModel, bundled `weights/`: retrain owed), `team_metrics/_pressing.py`, `_buildup.py`, `_progression.py`, `territory/_compute.py`, `defensive_credit/_line_break_signal.py`.
- [ ] **Step 3: List the owed retrains.** Cover every bundled or Hub artifact whose training data contains SPADL `type_id`. Classify each from evidence, not in advance.
  - Candidates to check: xCross (`_xcross_weights`), PassCompletionModel (`expected_passing` `weights/`), the TF-46 corner weights (`corners/_routine_weights`, `_run_zone_weights`, which read `corner_kind`), the VAEP models if any are bundled, and the receiver model (`train_receiver_model.py:39`).
  - The receiver model reads `pass` ∪ `cross`, so it may be unaffected. Decide from the code.
  - Name each artifact's file and the evidence.
- [ ] **Step 4: Test-first fixes for any "must change" hit.** Run each module's tests. Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 9: The start line (§7.2, DGX)

**Files:**
- Create:
  - `scripts/measure_cross_start_line.py`, a read-only measurement driver on the `scripts/_driver.py` seam (ADR-052), with `require_clean_tree` (ADR-037);
  - `docs/research/cross-definition/start-line.md` and its `metrics.json`.
- Modify: `silly_kicks/spadl/base.py` `CROSS_START_X` (the final value).
- Test: `tests/scripts/test_measure_cross_start_line.py`. It covers the F1 and tie rule on a toy corpus, and passes `test_compute_contract`.

**Interfaces:**
- Consumes: `classify_deliveries` with `cross_start_x=` each of {52.5, 60, 65, 70}, and `cross_native`.
- Produces: `CROSS_START_X` = the candidate with the highest mean F1 over StatsBomb open data and the GS xshot187 matches (equal weight). On a tie within 0.005, the larger value. IDSSE is reported, not averaged.

- [ ] **Step 1: Write the failing test** `test_selection_rule`: given per-vendor F1 tables, the rule returns the expected candidate, including the tie case.
- [ ] **Step 2: Run the test.** Expected: FAIL.
- [ ] **Step 3: Implement the driver.**
  - It declares `PIPELINE` (ADR-128; required for every corpus walker). Start from the scaffolder: `python -m scripts.hf_jobs scaffold driver measure_cross_start_line`.
  - StatsBomb events come through the pining `statsbomb` loader. GS raw events come through pining.
  - It converts once with the committed `CROSS_START_X = 52.5`, then evaluates each candidate X by retyping open-play `cross` rows with `start_x < X` to `pass` (the monotonic filter of Task 1).
  - **First check pining coverage.** List the pining keys for StatsBomb open data and the 62 GS xshot187 matches. Any gap goes to the owner before the run.
  - Run the tests. Expected: PASS. This step belongs to commit 1.
- [ ] **Step 4 (after Gate 1): the DGX run** on the clean commit 1, with the full corpus and parallel workers.
  - Write the memo, stamped with commit 1's sha.
  - Set `CROSS_START_X` to the chosen value. Add `test_cross_start_x_matches_memo`, which pins it to `metrics.json`.
  - Re-run `tests/spadl/` with the final value. Expected: PASS. This step belongs to commit 2.
- [ ] **Step 5: Ledger the task**, with the chosen value and each vendor's F1.

### Task 10: Corpus evidence (§7.3, DGX) and the enricher constants

**Files:**
- Create:
  - `scripts/measure_cross_delivery_corpus.py`, a read-only driver as in Task 9;
  - `docs/research/cross-definition/corpus-evidence.md` and its `metrics.json`.
- Modify: the `CrossDeliveryParams` defaults (Task 5), with the values from this run.
- Test: `tests/scripts/test_measure_cross_delivery_corpus.py`, for the ratio bar and the report shape on a toy corpus.

**Interfaces:**
- Consumes: the re-converted xshot187 corpus (raw events and tracking via pining, the Task 3 converters), `add_cross_deliveries` and `align_events_to_frames`.
- Produces the §7.3 report:
  - per-provider median crosses per match and the max/min ratio, with a bar of ≤ 1.5;
  - `cross_attempt` counts and the blocked share;
  - set-piece medians before and after;
  - restart-ended deliveries and the tracking-typed share;
  - set-piece event recall against `cross_attempt`, and the `corner_kind_source` split;
  - the aligned / unaligned-matched / misaligned / coverage-unanswerable shares per provider and row class.

- [ ] **Step 1: Write the failing test** `test_ratio_bar`: medians {16.5, 19, 21} pass, and {10, 19, 21} fail with the provider named.
- [ ] **Step 2: Run the test.** Expected: FAIL.
- [ ] **Step 3: Implement the driver.**
  - It declares `PIPELINE` (ADR-128).
  - It takes the chosen start line and every `CrossDeliveryParams` value as explicit overrides. The run then reports the numbers of the final constants, not of the committed provisional defaults.
  - Run the tests. Expected: PASS. This step belongs to commit 1.
- [ ] **Step 4 (after Gate 1 and Task 9 Step 4): calibrate on the DGX run** on the clean commit 1. Pass the Task 9 start line as an override.
  - Set each `CrossDeliveryParams` value from the run's distributions.
  - Pre-declared rules, one per `CrossDeliveryParams` field:
    - `release_match_m`: the 95th percentile of the ball-to-start distance among ELASTIC-confident rows, pooled over the three providers with equal weight per provider. Unweighted pooling would let SkillCorner's 16,479 rows decide.
    - `block_radius_m`: the 95th percentile of the release-to-opponent-contact distance among `cross_native` crosses with `cross_blocked == True`. In the xshot187 corpus only GS carries a non-`<NA>` `cross_blocked`, so this is GS only, and the memo says so.
    - `min_ball_coverage`: 0.8, unless more than 20 % of a provider's rows fail it. In that case, report it and the owner decides.
    - `release_gap_s`: the 95th percentile of the time from the aligned release to the passer's own detected contact, among ELASTIC-confident rows, pooled with equal weight per provider. Floor: one frame at the highest provider frame rate (0.04 s at 25 Hz).
    - `direction_window_s`: two frames at the lowest provider frame rate in the corpus. At SkillCorner's 10 Hz this is 0.2 s. A one-frame difference is too noisy for a direction.
    - `path_cap_s`: 4.0, fixed by spec §6.3. It is not calibrated, and it is reported only.
  - Re-run the driver with the final values.
  - Expected: the ratio is ≤ 1.5. A failed bar stops the cycle, and the result goes to the owner.
- [ ] **Step 5 (commit 2):** set the `CrossDeliveryParams` defaults to the memo values. Add `test_cross_delivery_defaults_match_memo`, which pins them to `metrics.json`. Re-run `tests/tracking/test_cross_delivery.py` and the new test. Expected: PASS. Ledger the task.

### Task 11: Docs, ADR, release prep and handoff

**Files.** Docs travel with the code they describe. Only the number-dependent docs wait for commit 2.

- **Commit 1 docs** (number-independent; required for a green commit 1). `tests/test_c4_aggregator_count.py` and `tests/test_c4_feature_column_count.py` derive their counts from `tracking.__all__` and `FEATURE_GLOSSARY`, so commit 1 fails without the C4 update.
  - Create: `docs/superpowers/adrs/ADR-133-one-cross-definition.md`, with the decision, the design and the event-only limits. Its "Evidence" section says that commit 2 adds the measured numbers.
  - Modify:
    - `docs/superpowers/adrs/ADR-127-*.md`: a status note, "superseded in part by ADR-133";
    - the C4 model under `docs/c4` (the DSL and the rendered `architecture.html`; pinned to the code, ADR-048);
    - `AGENTS.md`: a TF-86 table row, plus a Key-conventions bullet ≤ 600 chars that points to ADR-133;
    - `docs/context/providers.md` (the event-only limits), `docs/context/trained-models.md` (`xcross_label_version`), `docs/context/tracking-features.md` (TF-86) and `docs/context/corners.md` (`corner_kind_source`);
    - `docs/PRIVATE_CONSUMERS.md`, if Task 3 found a pin.
- **Commit 2 docs** (number-dependent):
  - the ADR-133 "Evidence" section, with the Task 9 and Task 10 numbers, not the spec's audits;
  - the recall numbers in `docs/context/providers.md` and in the converter docstrings (`<recall: Task 10>`);
  - `CHANGELOG.md`: a new minor version, PR-S209 (or the next free PR-S number at commit-prep), with the full §9 Hyrum list and the measured shifts;
  - `silly_kicks/_version.py` (minor bump; the release follows commit 2);
  - `TODO.md`: replace the Current summary; no new backlog row without the owner's approval.
- **Test:**
  - `tests/test_agents_md_budget.py`, `tests/test_todo_md_format.py`, `tests/test_version_single_source.py`;
  - `tests/test_c4_*.py`, `tests/test_howto_links_wired.py`, `tests/test_notice_md_format.py`.

Task 11 has two parts, one per commit (Global Constraints, order).

- [ ] **Step 1 (before commit 1).**
  - Write the commit 1 docs. Run `tests/test_c4_*.py`, `tests/test_agents_md_budget.py`, `tests/test_howto_links_wired.py` and `tests/test_notice_md_format.py`. Expected: PASS.
  - Run the full lint trio and the CI-faithful pytest scope on the DGX. Expected: exit 0, with the real exit codes shown. Commit 1 must be green on its own (provenance exception).
  - Run `/mad-scientist-skills:final-review` and fix its findings.
  - Send A and B the implementation review requests for the uncommitted tree. Answer both reports, with fixes test-first.
  - **Gate 1:** show the owner the file list and diff summary. On the explicit yes, make commit 1 and push.
- [ ] **Step 2 (commit 2 content, after the Task 9 and 10 runs).**
  - Write the commit 2 docs listed under Files.
  - Fill the converter docstring placeholders `<recall: Task 10>` from the Task 10 numbers. Check that `git grep -n "<recall: Task 10>"` returns nothing.
  - Run the doc gates above, the lint trio and the CI-faithful pytest scope. Expected: PASS.
- [ ] **Step 3: the commit 2 review and gate.**
  - Send A and B a re-review of the commit 2 delta: constants, memos, docs and the pin tests. Answer both.
  - **Gate 2:** on the owner's explicit yes, make commit 2 and push.
- [ ] **Step 4: Gate 3, the draft PR.** Ask for a separate yes, then open the draft PR and watch CI until it is green. The merge, the tag and the release each need their own yes later.
- [ ] **Step 5: After the merge and release,** send Part Deux the owed-retrain list from Task 8 and the restage note (the xshot187 `_actions` must be re-converted).
