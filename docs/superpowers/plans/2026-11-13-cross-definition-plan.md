# One Cross Definition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One provider-neutral geometric `cross` / set-piece delivery rule in every SPADL converter, plus a tracking enricher that gives `cross_attempt` and `cross_height`, and the consumers that read them.

**Architecture:**
- Layer 1 is a pure function in `spadl/base.py`. Each converter calls it at one precondition-defined position, and each native label moves to a new `cross_native` column.
- Layer 2 is a frame-consuming `add_*` enricher, TF-86. It aligns the release through ELASTIC (TF-43), follows the ball to the first non-passer contact (ADR-124/125), and labels the delivery.
- The consumers that read the layers: xCross label v2, causal opportunities, TF-46 `corner_kind`, and bravery.

**Tech Stack:** Python 3.10+, pandas 2/3 (CoW), numpy, pytest. DGX for tests and data runs.

**Spec:** `docs/superpowers/specs/2026-11-13-cross-definition-design.md` (r9; r6 was approved by the owner and by reviewers A and B; r7 is the §7.4 calibration amendment of 2026-11-14; r8 is the native-timing amendment of 2026-11-14, approved by A round 2 and B round 2; r9 is the SkillCorner end-rule fix of 2026-11-14, owner-ruled on review B CD-SB-23). Read it with this plan. Section numbers below (§) refer to it.

**Plan r5 (2026-11-14), the native-timing amendment.** Run 2 on commit 2 (`b12c0fd9`) fired the release-match owner trigger. The owner paused, asked for the root cause and a spike, and ruled: fix the timing in this cycle, at the gold standard (spec decision 11, §5.1, §6.6). r5 adds Tasks 12–19 and a fourth commit. Tasks 9 and 10 Step 4 (the four runs) now read the clean commit 3 and the re-staged corpus; their memos are stamped with commit 3's sha; their constants, memos and number-dependent docs form commit 4. Where Tasks 9–11 say "commit 2" or "commit 3" for the numbers commit, read commit 4. The gates are listed once, with what each authorizes, in Global Constraints 5b ("The live gates").

**Plan r4 (2026-11-14), the calibration amendment.** The first DGX runs on commit 1 `4490016c` showed that the r3 calibration rules measure alignment and detection error (spec §7.4). The owner ruled: replace the rules (spec §7.4), add `block_window_s`, widen the start-line candidates to 80, and add one commit for provenance. r4 changes the Global Constraints order, Task 9 (candidates) and Task 10 Step 4 (rules and runs). The rest is unchanged.

## Global Constraints

- **Branch:** `feat/cross-definition` off `main` `0cc07e6c`. No worktree, no subagents.
- **Two commits for the cycle, by owner ruling (2026-11-13), under the provenance exception of the commit policy.** There are no per-task commits: every "Commit" step below is replaced by "ledger the task". The ledger is `.superpowers/sdd/2026-11-13-cross-definition-plan/progress.md`. The order is fixed:
  1. Tasks 1–8, and the code and tests of Tasks 9–10 (Steps 1–3), on the uncommitted tree.
  2. `/final-review`, then the A/B implementation reviews **on the uncommitted tree**. Both reports are answered.
  3. **Gate 1: commit 1 and push** (owner yes). Commit 1 holds the code, the tests and the drivers, with provisional constants: `CROSS_START_X = 52.5` (the most permissive candidate) and the `CrossDeliveryParams` defaults.
  4. The Task 9 and Task 10 DGX runs, on the clean commit 1 (`require_clean_tree`, ADR-037). Each driver computes every candidate value itself, so its numbers are exactly those of the final constants.
  5. **r4 (owner ruling 2026-11-14): three commits.** The first runs on commit 1 led to the §7.4 amendment. A run must use a committed driver (ADR-037), so the amended driver is its own commit:
     - Commit 2: the spec r7 and plan r4 amendment, the `calibrate()` rules, `CANDIDATE_LINES` to 80, `CrossDeliveryParams.block_window_s` and the enricher's block window, with their tests. A/B review this delta, and both reports are answered. **Gate 2: commit 2 and push** (owner yes).
     - The four §7.4 runs on the clean commit 2.
     - Commit 3's content, on the uncommitted tree: the constants, the memos (stamped with commit 2's sha), Task 11's number-dependent docs, and a test that pins the defaults to the memo values. The number-independent docs (C4, AGENTS, docs/context, ADR-133's decision) already went in commit 1. A/B re-review this delta, and both reports are answered. **Gate 3: commit 3 and push** (owner yes).
  5b. **r5 (owner ruling 2026-11-14): four commits.** Commit 2 (`b12c0fd9`) is pushed and CI is green. Runs 1–2 on it fired the release-match trigger; the native-timing fix is its own commit, because the runs must read a committed tree:
     - Commit 3: spec r8, plan r5 and Tasks 12–18 (code, tests, number-independent docs). `/final-review`, then the A/B implementation reviews of this delta on the uncommitted tree; both reports are answered. **Gate 3: commit 3 and push** (owner yes, immediately before). Watch CI until it is green.
     - Task 19, the re-stage, on the clean commit 3.
     - The four §7.4 runs (Tasks 9 and 10 Step 4) on the clean commit 3 and the re-staged corpus.
     - Commit 4's content, on the uncommitted tree: the constants, the memos (stamped with commit 3's sha), the number-dependent docs and the pin tests. A/B re-review; both reports are answered. **Gate 4: commit 4 and push** (owner yes, immediately before).
     - **The live gates (r5; review A CD-PLAN-06).** Gates 1 and 2 are done (commits `4490016c`, `b12c0fd9`). Only two gates remain, and each authorizes exactly one thing:
       - **Gate 3** authorizes commit 3 (the r8 code, tests and docs) and its push. It is asked in Task 18 Step 4.
       - **Gate 4** authorizes commit 4 (the constants, memos, number-dependent docs, CHANGELOG, version and TODO) and its push. It is asked in Task 11 Step 3.
       - Every other "Gate" in Tasks 9–11 is superseded by these two. The draft PR (#306) exists, so no PR gate remains. The merge, the tag and the release each need their own yes later.
  6. The draft PR (#306) was opened after commit 1 with the owner's yes. Watch CI on each push until it is green.
  7. The merge, the tag and the release each need their own yes later. Merge with `--merge`, never squash: the memos cite commit 3's sha (r5), and a squash would orphan it.
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
- **Data comes from pining only.** Databricks is not a source. No HF job or dataset upload is part of this cycle. r5: the re-stage (Task 19) writes a new DGX directory, `~/cross-def/xshot187-r8/out`; it never overwrites `~/xshot187/out`, which other work reads. Publishing the re-staged corpus (a Hub upload or `datasets record`) needs the owner's go and is not in this plan.

## Review Focus

1. **No next action:** a delivery that is the last on-ball row of a period (or of the match) has no next action. It is typed `pass` / `_short` and raises nothing. Test owner: Task 1 (`test_last_row_in_period_is_not_a_delivery`).
2. **NaN coordinates:** a pass-class row or its next action has a NaN `start_x` / `start_y`. The row is not a delivery, and nothing raises. GS has NaN ends on 0.8 % of open passes. Test owner: Task 1 (`test_nan_coordinates_never_deliver`).
3. **Mixed id dtypes:** `team_id` is `int64` on one row type and `object` / `category` on another (GS synthetic rows, SkillCorner). The same team must compare equal. Test owner: Task 1 (`test_team_id_dtype_invariance`).
4. **Extra time and shoot-outs:** periods 3 and 4 behave like periods 1 and 2. Period 5 has only `shot_penalty` rows, so nothing is re-typed. Test owner: Task 1 (`test_extra_time_and_shootout_periods`).
5. **Frames that cover only part of a match:** one period has no ball rows, or all ball z values are NaN. `cross_attempt` is `<NA>` (unknown) for that period, `cross_height` is `<NA>`, and nothing raises. Test owner: Task 5 (`test_period_without_ball_frames_is_unknown`).
6. **r5: a GS event before its period's kickoff.** None exists in the 64 cached matches (measured 2026-11-14, `D:\Temp\cross-def\gs_prekick_clock.py`: 0), and SPADL documents `time_seconds` as non-negative. Expected: the converter raises a `ValueError` that names the event and the period (spec §6.6 B, review B CD-SB-18). Test owner: Task 14 (`test_event_before_kickoff_raises`, both sides: at the kickoff and just before it).
7. **r5: a SkillCorner possession with no `time_end`.** Expected: the native row gets a NaN time (never `time_start`), its start stays `x_end`/`y_end`, one `UserWarning` names the count, nothing raises, and the row sorts last in its period (spec §6.6 A, review B CD-SB-21). Test owner: Task 13 (`test_missing_time_end_is_nan_not_time_start`).
8. **r5: mixed `time_source` in one match.** GS native rows are `"tracking"`, synthetic GS rows take their source row's value, SkillCorner synthetic dribbles are `"event"`. Expected: the enricher picks the release per row, never per match. Test owner: Task 16 (`test_release_order_is_per_row`).

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
  - `test_scene_typing_and_direction_invariance[conv]`: a native cross delivered short → `pass`, a native pass delivered into the box from wide → `cross`, and a corner followed by a goal kick → `corner_short`. The same delivery mirrored (team attacking right-to-left) and in period 2 gets the same `type_id`. Build the scene per provider in its own raw schema (`tests/spadl/_delivery_scenes.py`).
  - `test_order_insensitive[conv]`: shuffle the raw input within equal-timestamp blocks; the output is the same (ADR-065).
  - `test_cross_native_values[conv]`: the provider's native open-play cross → `True`, native pass → `False`, native corner/free-kick crossed/short → `True` / `False`, other rows → `<NA>`. For SkillCorner, every row is `<NA>`.
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
  - **Docstrings (§6.4):** the `classify_deliveries` docstring states the event-only limits once (a blocked cross and a restart-ended delivery are `pass`; a restart-ended set piece is `_short`). Each converter's `convert_to_actions` docstring points to it (owner, 2026-11-14 size audit) and states:
    - the provider's own label in `cross_native`;
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
  - `test_path_stays_wide_is_not`. The row-class table covers a path that enters the box.
  - `test_blocked_cross_by_direction_and_orientation`: the first contact is an opponent within `block_radius_m`, and the initial direction points into the box → True. With the direction pointing away → False. Both hold when the team attacks right-to-left. `test_block_radius_both_sides`: just outside `block_radius_m` → decided by the path only.
  - `test_passer_excluded`: the event time is 0.3 s before the kick, and the passer's touch must not end the path.
  - `test_path_capped_without_contact`.
  - `test_restart_ended_delivery_typed_from_path`: in the actions, a cross-zone `pass` (typed `pass` by the converter because the next action is a `goalkick`) and a `corner_short` (next action a `throw_in`). In the frames, both ball paths enter the box. Both rows → `cross_attempt` True.
  - `test_coverage_both_sides`: coverage just below and at `min_ball_coverage`.
  - `test_release_alignment`: a confident ELASTIC match moves the release. Precomputed `elastic_*` columns are read; patch `align_events_to_frames` to raise, to prove it is not called.
  - `test_misaligned_release_is_unknown`: the ball at the release is `release_match_m + 0.01` from the action start → `<NA>`; at `release_match_m` → decided.
  - `test_height_aerial_ground_and_estimated`: peak z 1.81 → `aerial`, 1.79 → `ground` (threshold from `BallContactParams.aerial_z_m`). `z_source == "estimated"` → `<NA>`.
  - `test_orientation_invariance`: a mirrored match gives identical columns.
  - `test_period_without_ball_frames_is_unknown`: Review Focus 5.
  - Purity: register `add_cross_deliveries` in `PURITY_ENTRIES` (`tests/test_add_star_purity.py`); the gate proves the inputs are unchanged.
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
- Consumes: `classify_deliveries` with `cross_start_x=` each of {52.5, 60, 65, 70, 75, 80} (r4; spec r7 §7.2), and `cross_native`.
- Produces: `CROSS_START_X` = the candidate with the highest mean F1 over StatsBomb open data and the GS xshot187 matches (equal weight). On a tie within 0.005, the larger value. IDSSE is reported, not averaged.

- [ ] **Step 1: Write the failing test** `test_selection_rule`: given per-vendor F1 tables, the rule returns the expected candidate, including the tie case.
- [ ] **Step 2: Run the test.** Expected: FAIL.
- [ ] **Step 3: Implement the driver.**
  - It declares `PIPELINE` (ADR-128; required for every corpus walker). Start from the scaffolder: `python -m scripts.hf_jobs scaffold driver measure_cross_start_line`.
  - StatsBomb events come through the pining `statsbomb` loader. GS raw events come through pining.
  - It converts once with the committed `CROSS_START_X = 52.5`, then evaluates each candidate X by retyping open-play `cross` rows with `start_x < X` to `pass` (the monotonic filter of Task 1).
  - **First check pining coverage.** List the pining keys for StatsBomb open data and the 62 GS xshot187 matches. Any gap goes to the owner before the run.
  - Run the tests. Expected: PASS. This step belongs to commit 1.
- [ ] **Step 4 (r4; after Gate 2): the DGX run** on the clean commit 2, with the full corpus and parallel workers. It is run 1 of the four spec §7.4 runs.
  - Write the memo, stamped with commit 2's sha.
  - Set `CROSS_START_X` to the chosen value. Add `test_cross_start_x_matches_memo`, which pins it to `metrics.json`.
  - Re-run `tests/spadl/` with the final value. Expected: PASS. This step belongs to commit 3.
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
- [ ] **Step 4 (r4; after Gate 2): calibrate on the DGX** on the clean commit 2, with the full corpus and parallel workers.
  - The rules are spec §7.4 (r7), one per `CrossDeliveryParams` field. They replace the r3 rules, which pooled every ELASTIC-confident row (record: the ledger's run 1-3 lines and spec §7.4).
  - Run the four §7.4 runs in order. Each run passes, as overrides, every value that an earlier run fixed:
    1. provisional parameters → the start line;
    2. `--cross-start-x` → `release_match_m`, `release_gap_s`;
    3. plus `--release-match-m`, `--release-gap-s` → `block_radius_m`;
    4. plus `--block-radius-m` → the §7.3 evidence.
  - Pass each override exactly as the earlier `cross_definition.json` prints it (full float precision; `fixed_point` uses rtol 1e-9).
  - After run 2: if `calibration.release_match_owner_decision_needed` is true (spec §7.4 owner triggers), stop and take the result to the owner before run 3.
  - r5: the runs read the clean commit 3 and the re-staged corpus (Task 19). After run 2 and after run 4, also read `native_release_owner_decision_needed` (spec §7.4 r8); if it is true, stop and take the result to the owner.
  - Expected: run 4's `fixed_point.all` is true (the four calibrated fields and `cross_start_x`), its `coverage_owner_decision_needed` is false, and the ratio is ≤ 1.5. A false fixed point or a failed bar stops the cycle, and the result goes to the owner.
  - `min_ball_coverage` stays 0.8 unless more than 20 % of a provider's rows fail it; then the owner decides.
- [ ] **Step 5 (commit 3, r4):** set the `CrossDeliveryParams` defaults to the memo values. Add `test_cross_delivery_defaults_match_memo`, which pins them to `metrics.json`. Re-run `tests/tracking/test_cross_delivery.py` and the new test. Expected: PASS. Ledger the task.

### Task 11: Docs, ADR, release prep and handoff

**Files.** Docs travel with the code they describe. Only the number-dependent docs wait for the numbers commit. **r4:** that is commit 3; commit 2 is the §7.4 amendment. Below, "commit 2 docs", "Step 2", "Step 3" and "Gate 2" mean the commit 3 content, its review and **Gate 3**. Step 4's draft PR exists already (#306).

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
- [ ] **Step 3 (r5: the commit 4 review and Gate 4).**
  - Send A and B a re-review of the commit 4 delta: constants, memos, docs and the pin tests. Answer both.
  - **Gate 4:** on the owner's explicit yes, make commit 4 and push. Watch CI until it is green.
- [x] **Step 4: superseded (r5).** The draft PR (#306) was opened after commit 1 with the owner's yes. No PR gate remains (Global Constraints 5b, live gates).
- [ ] **Step 5: After the merge and release,** send Part Deux the owed-retrain list from Task 8 and the restage note (the xshot187 `_actions` must be re-converted).

---

## r5 tasks (native timing, spec r8; commit 3 unless a step says otherwise)

The order is Task 12 (measurements), then Tasks 13–18 test-first, then commit 3's review and gate (Task 18 Step 4), then Task 19 (re-stage), then Tasks 9–10 Step 4 (the runs) and commit 4. Each task ledgers its result; no per-task commits.

### Task 12: The r8 measurements (before any code)

**Files:**
- Create: `docs/research/cross-definition/native-timing.md` and its `metrics.json`. They hold the §5.1 evidence and the numbers below, with the script paths (`D:\Temp\cross-def\*.py`) and the DGX output paths.
- Scripts stay off-repo under `D:\Temp\cross-def\` (read-only measurements, like §5.1's).

- [ ] **Step 1: Metrica (spec §6.6 C; review A CD-PLAN-07).** The bronze parquets in `tests/datasets/metrica/` carry no frame. The frame-bearing events are `tests/datasets/kloppy/metrica_events.json` (each event has `"start": {"frame": …, "time": …}`; `"time"` is the value the Metrica bronze carries as `start_time_s`), and the tracking is under `tests/datasets/tracking/metrica`. First check that the two are the same game (team ids and the first frames' times); ledger the result. Then compare every event's `start.time` with the tracking time of its `start.frame`. If the games differ, compare `start.time` with `start.frame` divided by the event file's frame rate, and say so in the ledger. Expected: the share within half a frame period. If every event is within it, Metrica emits `"tracking"`; otherwise `"event"`. Ledger the ruling with the number.
- [x] **Step 2: GS events before the kickoff and the clock constant (done 2026-11-14, before the plan review closed; review B CD-SB-18 and C2).** `gs_prekick_clock.py` over the 64 cached matches: 0 converted events before their kickoff; `periodGameClockTime − periodElapsedTime` equals 0/2700/5400/6300 s in every match and period within 0.43 frames. The memo records both.
- [ ] **Step 3: The frame-link shift (spec §6.6 "What stays").** For GS (62) and SkillCorner (98 cached) matches, link every action to frames with `link_actions_to_frames`, once with today's times and once with the r8 times (GS: `event_time − kickoff start_time` plus the period's constant 0/2700/5400/6300 s, because the staged frames are still on the match clock; SkillCorner: `time_end` after the period re-base) (review B C3). Report the median and 90th-percentile shift of the linked frame time, and the ball-to-action-start distance at the linked frame before and after.
- [ ] **Step 4: The consumer list.** List every module that calls `link_actions_to_frames` (26 callers plus the definer in `tracking/utils.py`) and every frame-aware `add_*` that links by time. This list feeds Task 18's sweep.
- [ ] **Step 5: Ledger the task.**

### Task 13: SkillCorner, the action at the kick (spec §6.6 A)

**Files:**
- Modify: `silly_kicks/spadl/skillcorner.py` (native rows from `time_end` and `x_end`/`y_end`; the inference inputs keep `time_start` and `x_start`/`y_start`; `time_end`, `x_end`, `y_end` required); `silly_kicks/spadl/_skillcorner_inference.py` only if its time input must be named explicitly (the keeper save reads `next_row["time_seconds"]`, line 168, so it must receive the start time).
- Modify fixtures: the SkillCorner test inputs that lack `time_end`. First find each fixture's generator. Where one exists, reproduce the committed fixture byte-for-byte before regenerating it (AGENTS.md registry rule). Where none exists, add `time_end` by a rule stated in the fixture's README and ledgered.
- Modify: `tests/spadl/_delivery_scenes.py` (`_skillcorner` writes `time_end`, `x_end`, `y_end` for every event).
- Retire the SkillCorner case of `tests/spadl/test_converter_reorder_regression.py` (review A CD-PLAN-08). The case proves one thing: moving `to_spadl_ltr` before `_add_dribbles` changed nothing but the typing, measured against goldens from the pre-move converter (`0cc07e6c`). Commit 1 (`4490016c`) proved it, and git history keeps that proof. §6.6 A changes every SkillCorner row on purpose, so a golden regenerated from the r8 converter would only snapshot the code under test. Delete `tests/spadl/fixtures/reorder_golden/skillcorner.parquet`, limit the parametrization to `kloppy`, and update the module docstring: say why the SkillCorner case retired and where its proof lives. Ledger the ruling.
- Test: `tests/spadl/test_skillcorner.py`, `tests/spadl/test_converter_deliveries.py`.

**Interfaces:**
- Consumes: today's `_parse_time_start`, `_to_period_relative`, `_transform_coords`, `infer_defensive_actions`, `infer_keeper_saves`.
- Produces: SkillCorner SPADL whose native rows sit at the kick (§6.6 A), with the end-coordinate rule per type.

- [ ] **Step 1: Write the failing tests** (spec §7.1, SkillCorner bullets):
  - `test_native_row_at_possession_end`: a possession with distinct start and end gives `time_seconds` from `time_end` (period re-based) and the start from `x_end`/`y_end`.
  - `test_start_type_rows_stay_at_start`: an interception and a keeper save keep the start time (−0.01 s) and position. Plan r6 (review B CD-SB-24): the keeper save is built and asserted, with `time_end` ≠ `time_start`, so feeding `infer_keeper_saves` the end time fails the test.
  - Plan r6 (spec r9, review B CD-SB-23) `test_untargeted_pass_and_clearance_end_at_the_next_reception`: an untargeted pass and a clearance end at the next possession's reception point, not at its kick, and the receiver's reception-to-kick carry is a synthetic dribble; a targeted pass still ends at its targeted reception; the last possession of a period falls back to `_derive_end_coordinates`. It replaces the r5 `test_pass_without_target_ends_at_the_next_start`, which pinned the next kick. `test_next_reception_never_crosses_a_period` (review A CD-IMPL-13): a period's last untargeted pass, followed by a possession in the next period, keeps end = start; a period-blind `shift(-1)` fails it.
  - `test_missing_time_end_is_nan_not_time_start` (Review Focus 7): NaN time, start from `x_end`/`y_end`, and `pytest.warns(UserWarning)` with the count in the message.
  - `test_carry_becomes_a_dribble`: the reception-to-kick carry of 5 m becomes a synthetic dribble.
  - `test_end_coordinates_per_type`: a pass without a targeted receiver ends at the next possession's reception within its period (plan r6, spec r9; review A C36); a shot, a `foul_suffered`, a `possession_loss` and an `unknown` row end at their start.
  - `test_period_relative_after_rebase`: the time restarts near 0 in periods 2–4.
  - `test_kick_point_decides_the_cross_zone`: the reception is in the cross zone and the kick is outside it, so the row is not a `cross`; the reverse case is a `cross`.
- [ ] **Step 2: Run the tests.** Expected: FAIL (today's rows sit at `time_start`/`x_start`).
- [ ] **Step 3: Implement** per §6.6 A.
- [ ] **Step 4: Run** `tests/spadl/`, `tests/atomic/` and `tests/providers/skillcorner/`. Expected: PASS. Every changed existing assertion is ledgered with its reason.
- [ ] **Step 5: Ledger the task.**

### Task 14: Gradient Sports, the possession event's own time (spec §6.6 B)

**Files:**
- Modify: `silly_kicks/spadl/gradientsports.py`:
  - the new `time_seconds` (`event_time − kickoff start_time`), computed before the top sort;
  - the anchors read before the exclusion filter;
  - the per-period anchor rule (periods 1–4 raise; period 5 anchors at its first `start_time`; a period without converted actions needs none);
  - the sort tiebreak `("start_time", "possession_event_id", "event_id")`;
  - remove the Option D imputation (`_impute_time_seconds_by_start_time`) and its call;
  - `time_seconds` leaves the required input columns; a NaN `event_time` on a converted row raises.
- Modify: `scripts/_loader_pining.py` `_build_gradientsports`: the frame `time_seconds` from `periodElapsedTime`.
- Modify fixtures (review A C27): the GS synthetic match already carries `eventTime`, `startTime` and the four kickoffs, with `eventTime == startTime` (`tests/datasets/gradientsports/_generate_synthetic_match.py:234-237`). The new cases need events whose `eventTime` and `startTime` differ, and an inversion pair. Change the generator, but first reproduce the committed `synthetic_match.json` byte-for-byte with it (the AGENTS.md generator rule); then regenerate and ledger the diff summary. `tests/spadl/_delivery_scenes.py::_gradientsports` gains `event_time`, `start_time` and kickoff rows.
- The Option D record (review A C28): its Chesterton's fence is spec `2026-08-20-chronological-action-id-invariant-design` §2b and ADR-065. The Option D tests in `tests/spadl/test_gradientsports.py` (about lines 1207, 1232 and 1334; check 1780) change or retire with it; each is ledgered with its reason.
- Modify: `tests/spadl/test_time_seconds_convention.py`. Add a GS lock case (period-relative in every period). Rewrite its docstring, whose scope note (lines 9–10, "Opta + StatsBomb only; GradientSports time_seconds is a verbatim pass-through") 6.6 B makes false (review B CD-SB-19).
- Test: `tests/spadl/test_gradientsports.py`, `tests/scripts/test_loader_pining*.py`, `tests/spadl/test_time_seconds_convention.py`. If no GS loader fixture exists for the `validate_time_base` test, build the smallest raw GS tracking + events fixture under `tests/datasets/gradientsports/` and ledger its source.

**Interfaces:**
- Consumes: `sort_actions_chronologically` (`spadl/base.py:34`).
- Produces: period-relative GS `time_seconds` on the possession event's own time; period-relative GS frames from the pining loader.

- [ ] **Step 1: Write the failing tests** (spec §7.1, GS bullet):
  - `test_time_is_event_time_minus_kickoff` in every period 1–4, and period-relative;
  - `test_missing_kickoff_raises` (periods 1–4, both sides: present and missing);
  - `test_shootout_period_5_anchors_at_first_event`;
  - `test_period_without_actions_needs_no_anchor` (a period-0 `G` marker);
  - `test_inversion_pair_sorts_by_event_time`: the `action_id` order follows the event time, and `_assert_chronological_action_id` passes;
  - `test_foul_without_start_clock_gets_event_time`;
  - `test_nan_event_time_raises`;
  - `test_input_time_seconds_is_ignored`;
  - `test_event_before_kickoff_raises` (Review Focus 6; both sides);
  - `test_order_insensitive_under_shuffle`;
  - `test_loader_frames_are_period_relative` and `validate_time_base` passing on a loaded GS fixture match.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement** per §6.6 B. Chesterton's fence for Option D: cite its spec (§2b) and ADR in the ledger, with the measured reason it no longer reaches any output (144,541 of 144,541 events carry `eventTime`).
- [ ] **Step 4: Run** `tests/spadl/`, `tests/atomic/`, `tests/scripts/` (loader tests) and `tests/tracking/test_stoppages.py`. Expected: PASS. Every changed existing assertion is ledgered with its reason.
- [ ] **Step 5: Ledger the task.**

### Task 15: The `time_source` column (spec §6.6 C)

**Files:**
- Modify: `silly_kicks/spadl/schema.py` (`SPADL_COLUMNS` and the per-provider dicts, after `cross_native`, dtype `object`; `TIME_SOURCE_VALUES`); `silly_kicks/spadl/utils.py` `_finalize_output` (default `"event"`, vocabulary check); every converter (GS and SkillCorner native rows `"tracking"`; Metrica per Task 12; others `"event"`; derived rows inherit; synthetic dribbles `"event"`); `silly_kicks/reflection.py` (non-geometric); `silly_kicks/atomic/spadl/base.py` (`preserve_native`); `docs/context/providers.md`.
- Test: `tests/spadl/test_time_source.py` (new), `tests/spadl/test_output_contract.py`, `tests/test_reflection.py`, `tests/spadl/test_schema.py`.
- Modify: `tests/spadl/test_converter_reorder_regression.py` `_EXCLUDED` gains `"time_source"`, as it holds `"cross_native"`. The kloppy golden keeps its pre-move provenance (`0cc07e6c`); regenerating it would repeat CD-PLAN-08 (review A C30). Ledger the ruling.

- [ ] **Step 1: Write the failing tests** (spec §7.1, `time_source` bullet): the column, its dtype and vocabulary for every converter; the synthetic-dribble and derived-row values; the atomic opt-in carry; the reflection declaration; `_finalize_output` keeps it, and rejects a value outside the vocabulary.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/spadl/`, `tests/atomic/`, `tests/test_reflection.py`. Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 16: The enricher's release order and the driver (spec §6.3 step 0, §7.4 r8)

**Files:**
- Modify: `silly_kicks/tracking/_cross_delivery.py` (`_release_times`: step 0 for `"tracking"` rows; `release_source` gains `"native"`; a missing `time_source` means `"event"`; the module and function docstrings).
- Modify: `scripts/build_cross_definition.py` (`calibrate`: native rows join the release-gap sample and the fail share, `release_match_m` stays on IDSSE ELASTIC rows; `alignment_class`/`ALIGNMENT_CLASSES` gain `"native"`; the report's per-provider native passer-touch statistics and `native_release_owner_decision_needed`, bounds ±0.30 s and 40 %).
- Test: `tests/tracking/test_cross_delivery.py`, `tests/scripts/test_build_cross_definition.py`.

- [ ] **Step 1: Write the failing tests** (spec §7.1, release-order, driver and trigger bullets):
  - `test_tracking_row_uses_native_time`: a `"tracking"` row uses `time_seconds` although a confident `elastic_frame_id` points elsewhere; the same row as `"event"` uses ELASTIC (both sides).
  - `test_missing_time_source_means_event`.
  - `test_release_match_applies_to_native_rows` (both sides of `release_match_m`).
  - `test_release_order_is_per_row` (Review Focus 8).
  - `test_calibrate_includes_native_rows`: a toy where dropping the native rows changes `release_gap_s` and the fail share, while `release_match_m` stays the IDSSE value.
  - `test_alignment_classes_include_native`, still a partition.
  - `test_native_release_trigger`: both sides of −0.30/+0.30 s and of 40 %.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/tracking/test_cross_delivery.py`, `tests/scripts/test_build_cross_definition.py`, `tests/test_add_star_purity.py`. Expected: PASS.
- [ ] **Step 5: Ledger the task.**

### Task 17: ELASTIC episodes from `stoppage_state` (spec §6.6 D)

**Files:**
- Modify: `silly_kicks/tracking/_elastic_sync.py` (`_inplay_segments` reads `stoppage_state` when present: `"live"` in play, `"dead"`/`"unknown"` not; else `ball_state`); `docs/context/tracking-features.md` (TF-43 note).
- Test: `tests/tracking/test_elastic_inplay_episodes.py`, `tests/tracking/test_elastic_sync_oracle.py` (floors unchanged).

- [ ] **Step 1: Write the failing tests** (spec §7.1, ELASTIC bullet): episodes split at `"dead"` and `"unknown"`; frames without `stoppage_state` keep today's episodes; a restart inside its own `"unknown"` span is assigned to the nearest live episode on both sides of the span midpoint.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/tracking/test_elastic*.py`. Expected: PASS, with the oracle floors met.
- [ ] **Step 5: Ledger the task.**

### Task 18: The r8 caller sweep, the docs and the commit-3 gate

**Files:**
- Modify: `docs/research/cross-definition/caller-sweep.md` (seams 5–8 with evidence on both sides, from Task 12 Step 4); `docs/PRIVATE_CONSUMERS.md`: the lakehouse `skillcorner._PERIOD_START_SECONDS` entry (review B CD-SB-17), and rows in the form of its line-31 table for each r8 change a downstream consumer sees (review B CD-SB-20): the GS time base and its new required inputs; the SkillCorner kick point and its new required inputs; the GS and SkillCorner `action_id` renumber (ADR-065); `time_source`; the ELASTIC episodes and the GS stoppage spans; `docs/superpowers/adrs/ADR-017-*.md`: an amendment note that GS `time_seconds` is now computed by the converter, period-relative, not passed through (review B CD-SB-19); `docs/superpowers/adrs/ADR-065-*.md`: an amendment note that its GS `start_time` justification (line 38, the Option D imputation) no longer applies, and that `start_time` stays as a sort tiebreak (review A C26); `docs/superpowers/adrs/ADR-133-one-cross-definition.md` (a native-timing decision); `docs/context/providers.md` (SkillCorner kick point, GS event time, `time_source`); `AGENTS.md` only if a Key-conventions bullet changes (≤ 600 chars).
- Owed retrains: extend Task 8's list with every artifact trained on GS or SkillCorner SPADL or on frames linked to their actions, classified from evidence.

- [ ] **Step 1: The sweep.** Classify every seam 5–8 hit as value-changing or unaffected, with evidence. Test-first fixes for any "must change" hit.
- [ ] **Step 2: The docs** listed under Files. Run the doc gates of Task 11.
- [ ] **Step 3: Verify.** The lint trio, the CI-faithful pytest scope, the slow suite and the CI doctest command on the DGX (synced tree). Expected: exit 0, with the real exit codes shown (the known `test_dgx_e2e` environment failure excepted and named).
- [ ] **Step 4: Review and Gate 3.** Run `/final-review`, then send A and B the implementation review of the commit-3 delta. Answer both reports, with fixes test-first. **Gate 3:** show the owner the file list, the diff summary, the verification and both verdicts; on the explicit yes, make commit 3 and push. Watch CI until it is green.
- [ ] **Step 5: Ledger the task.**

### Task 19: The re-stage (DGX, after Gate 3)

- [ ] **Step 1: Check the tree and the target.** A clean checkout of commit 3 on the DGX (`require_clean_tree`; never `--allow-dirty`). `~/cross-def/xshot187-r8/out` must be absent or empty at the start; the driver creates directories with `exist_ok=True`, so the check is explicit (review B C1).
- [ ] **Step 2: Re-stage.** Run `scripts.materialize_tc3_frames` with the xshot187 roster (`~/xshot187/roster.json`), the pining cache and the three providers, into `~/cross-def/xshot187-r8/out`, with K disjoint workers into one generation and one final pass (the `~/xshot187/stage_parallel.sh` pattern, pointed at the new directory). Never write to `~/xshot187/out`.
- [ ] **Step 3: Check the re-stage.**
  - On every GS match and period, check that `periodGameClockTime − periodElapsedTime` is the period's constant (0/2700/5400/6300 s) within one frame. Stop on a deviation and take it to the owner.
  - Check the manifest: 187 matches, 0 failed, `run_commit` = commit 3, `run_tree_dirty` false.
  - Compare the action counts per provider with the old corpus; SkillCorner grows by the new carries.
- [ ] **Step 4: Point the runs at it.** Update `~/cross-def/dgx_run.sh` (`SHA` = commit 3, the data symlink to `~/cross-def/xshot187-r8/out`) and `D:\Temp\cross-def\sync.sh` (`BASE` = commit 3). Then run Tasks 9 and 10 Step 4 (the four §7.4 runs, both owner triggers read as stated there).
- [ ] **Step 5: Ledger the task.**
