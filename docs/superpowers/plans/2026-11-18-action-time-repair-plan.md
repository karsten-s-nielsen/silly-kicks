# Action-time repair: implementation plan (silly-kicks 4.144.0, PR-S209, second part)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline; the owner bans subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved action-time repair, restage on it, and run the refit and the release of the first directive.

**Architecture:** One provenance column (`time_source`), one step that sets times from the tracking (`sync_action_times`), derived SkillCorner restart and penalty rows, and the reader rules of the spec. The code replaces C1 by an amend; the stage, the campaign and the adoption follow.

**Tech Stack:** Python >=3.10 (the DGX venv runs 3.12), pandas, numpy, pytest, ruff, pyright; HF Jobs campaign tooling (`scripts/hf_jobs`); the DGX for every test run and every stage build.

**Spec:** `docs/superpowers/specs/2026-11-18-action-time-repair-design.md`, r15, 622 lines, sha256 `df320784ffe77f479e19fef821d4cc1abe0746e5388b4c63b9da2fb568b7d2ef`. Sweep: `docs/research/action-time-repair/reader-sweep.md`, sha256 `8dc2295f934497d5cbcf4f14e96d7c090b7dedfb16c94f4f7125053ef9c07d12`. Directive: `D:\Development\_reviews\2026-11-18-action-time-repair-supervisor-directive-C.md` (rulings R29-R60, bars A1-A24). The spec holds the facts; this plan does not repeat them.

## Global Constraints

- Branch `feat/release-refit-144`. No worktree. No subagent.
- Commits: exactly three (bar A15 as amended by R42). Commit 1 replaces C1 `3ec4b79a` by `git commit --amend`; commit 2 is the campaign file, the lock and the stage provenance; commit 3 is the adoption. No commit per task.
- Before each commit: the scoped gate and the lint trio on the exact tree, `/final-review`, and an A/B review of that commit's delta (the owner's standing rule: A/B review of every implementation), repeated after every fix until both approve. Then stop, show the file list, `git diff --stat` and the `git write-tree` id that passed, and wait for the owner's yes on that exact commit. Each push needs his yes too; commit 1 pushes with `git push --force-with-lease`.
- Keep the old C1 on a local ref until the cycle closes: `git branch c1-3ec4b79a-keep 3ec4b79a`. Delete nothing on the Hub; the run keys `tf46_refit-3ec4b79-*` stay.
- No committed file names `3ec4b79a` as its base (Task 15). Commit 1 carries the spec, the sweep and this plan. This plan names `3ec4b79a` only as the old C1 (the commit that the amend replaces, the keep ref and its run keys), never as a base; Task 15 ends with a `git grep` check of the staged tree.
- Tests first: each test is written and seen to fail before its code (superpowers:test-driven-development).
- NEVER run the full test suite. The local gate is the §8 command plus the lint trio (Task 15), on the DGX, on the exact tree that is committed.
- Temp files: `D:\Temp\xshot187\` (local), `~/refit144/` (DGX). Do not touch `~/cross-def`, `~/xshot187/{out,gatecheck,staging}`, `~/dl-review-xdef`, `~/refit144/{staging,adopt,launch7}`, the logs or the bucket.
- No-touch (bar A9): the event block of `silly_kicks/providers/sportec/parse.py` (except the R63 change of bar A25: the outcome tag under `Penalty > ShotAtGoal` in the nested-child loop of `_build_event_row`), `silly_kicks/tracking/utils.py`, `silly_kicks/tracking/_link_identity.py`. The bronze stays as the provider gives it, except the R63 values on `Penalty` rows.
- Hub: every create, upload, `datasets record` and launch needs the owner's go. New dataset repos are private. The launch uses the classic write token (R48); no token code.
- Eyestone NDA Rule #1; Databricks is not a source; course datasets give no number or file (bar A13).
- The two HF tooling findings (the executor's workers on a signal, `executor.py:525-526`; the `datasets fetch` `nargs=argparse.REMAINDER`, `datasets.py:462`) are true on this tree and out of scope. Change nothing there without the owner's word.
- **Stop rule.** A failed gate, a count that is not 0 where the spec says 0, a need to touch an area of bar A9, or a measured number that misses a bar of A6 is a STOP: report to the supervisor and the owner, change nothing more.

## The interim tree (commits 1 and 2)

Commits 1 and 2 hold the repaired code with the bundles of launch 7 (fitted before the repair). Commit 3 replaces the bundles.
- **Served values.** A test that pins a served value of a bundled ghost-GK, ghost-outfield or xCross model can move, because `score_diff` and `phase` change (R51, R52, R56). Task 15 lists each such test with the reason. At commit 1 its expected value is the repaired-feature value of the pre-repair bundle, with the reason in the commit message; at commit 3 it is the value of the new bundle. No test is skipped or loosened.
- **Declared constants.** `tests/tracking/test_declared_constant_values.py` compares the declared values of its canonical keys with `spadlconfig` and reads the bundled `metadata.json` of ghost, xShot and xCross (`:27-29`). `match_context_revision` is not a geometry constant and not in `spadlconfig`: it lives in `_feature_contract.py`. The test gains its canonical value 1.0 from there. At commit 1 the ghost and xCross bundles lack the key, so the bundle check of that one key is marked "absent on a pre-repair bundle" with the reason; commit 3 removes the mark.
- **The warning.** At commits 1 and 2 each load of a pre-repair ghost-GK, ghost-outfield or xCross bundle warns `PredatesDeclaredConstantWarning`; the class is not in `filterwarnings` (`pyproject.toml:384-388`), so the loads pass. At commit 3 the class joins the `error::` entries of `filterwarnings`, and a new test `test_no_bundled_artifact_predates_a_declared_constant` asserts that every bundle under `silly_kicks/tracking/_*_weights/` stores every constant that its library block declares. That test guards the wheel after the adoption.

## Review Focus

1. A SkillCorner match with no tracking (20 stage matches have no cached raw tracking): the step runs, every derived row keeps `"placeholder"` or `"order_only"`, nothing raises. Test in Task 6.
2. A stored 4.144.0 table without `time_source`: `sync_action_times` refuses it; every reader treats it as `"event"` and gives today's values. Tests in Tasks 1 and 6.
3. Team ids of mixed dtype (`363`, `363.0`, `"363"`) in the running-score re-siding and the penalty team rule: `ids_match` gives one side. Test in Task 10.
4. A running score with `<NA>` on some native rows of a match: the per-side check treats the match as not consistent and uses the goal rows. Test in Task 10.
5. Frames whose ball `visibility` is `None` (a provider adapter that does not carry detection): the kick rule finds no detected frame and keeps the unmeasured source; it does not use undetected frames. Test in Task 6.

---

### Task 0: Set-up

- [ ] Run `git branch c1-3ec4b79a-keep 3ec4b79a` and `git status --short` (expected: the spec, the sweep and this plan untracked).
- [ ] Create the ledger `.superpowers/sdd/2026-11-18-action-time-repair-plan/progress.md` with first line `# SDD ledger — plan: docs/superpowers/plans/2026-11-18-action-time-repair-plan.md`.
- [ ] On the DGX, check that `~/refit144/sk` is clean at the old C1 (`git -C ~/refit144/sk status --short` empty; its HEAD equals the local keep ref). Each test run applies the local working tree there: `git diff --binary c1-3ec4b79a-keep | ssh karsten@192.168.68.73 'cd ~/refit144/sk && git checkout -- . && git clean -fdq -e .venv && git apply'` (untracked new files go with `git add -N` first).

### Task 1: The `time_source` column and its predicates (spec §4.1, §4.18)

**Files:** Modify `silly_kicks/spadl/schema.py`, `silly_kicks/spadl/base.py` (`_finalize_output`, `_add_dribbles`), `silly_kicks/spadl/utils.py` (`_finalize_output` validation), `silly_kicks/reflection.py`, `silly_kicks/feature_glossary.py`, `silly_kicks/atomic/spadl/base.py`, `silly_kicks/atomic/spadl/schema.py`. Test: `tests/spadl/test_time_source.py` (new).

**Produces:**
- `TIME_SOURCE_VALUES = ("event", "tracking", "elastic", "kick", "placeholder", "order_only", "derived")`; `time_source` in `SPADL_COLUMNS` (dtype `object`), non-geometric in the reflection registry, carried by `convert_to_atomic(preserve_native=...)`.
- `is_order_only(actions) -> np.ndarray`, `is_time_unmeasured(actions) -> np.ndarray` (`"placeholder"` or `"order_only"`), `restart_kind(type_id) -> str` (§4.18 map), all in `spadl/schema.py`; all `False`/type name without the column.

- [ ] Tests (fail first): `test_vocabulary_is_exact`, `test_finalize_refuses_unknown_value` (`ValueError`), `test_is_order_only_absent_column_all_false`, `test_is_time_unmeasured_both_values`, `test_restart_kind_map` (`corner_crossed`/`corner_short` → `corner`, `freekick_*` → `free_kick`, `shot_penalty` → `penalty`, `pass` → `pass`), `test_dribble_is_derived`, `test_atomic_carries_time_source`, `test_reflection_declares_time_source`.
- [ ] Implement; run `pytest tests/spadl/test_time_source.py -q` on the DGX. Expected: PASS.

### Task 2: Every converter emits `time_source` (spec §4.1, §4.6)

**Files:** Modify `silly_kicks/spadl/{sportec,statsbomb,opta,wyscout,_wyscout_mappings,metrica,kloppy,gradientsports,skillcorner}.py`. Test: `tests/spadl/test_time_source.py`.

- [ ] Tests (fail first): `test_time_source_per_converter` (parametrized: sportec/statsbomb/opta/wyscout/metrica/kloppy `"event"`; gradientsports and SkillCorner native `"tracking"`; GS synthetic rows take their source row's value; dribbles `"derived"`).
- [ ] Implement (one line per converter, before `_finalize_output`); run the test. Expected: PASS.

### Task 3: One goal predicate (R52, spec §4.12)

**Files:** Create `silly_kicks/spadl/_goals.py`. Modify `silly_kicks/vaep/labels.py:18-33`, `silly_kicks/win_probability/_state.py:18-20`, `:58-59`. Test: `tests/spadl/test_goals.py` (new).

**Produces:** `goal_mask(actions) -> pd.Series`, `owngoal_mask(actions) -> pd.Series` (read `type_id`, else `type_name`).

- [ ] Tests (fail first): `test_goal_mask_shot_types`, `test_owngoal_any_type`, `test_type_id_and_type_name_agree`. The existing `tests/vaep/test_labels*.py` and `tests/win_probability/test_state.py` stay green unchanged.
- [ ] Implement; delete the old private copies; run the three test groups. Expected: PASS.

### Task 4: SkillCorner derived restart and penalty rows (spec §4.2, §4.15 carrier, §4.17, §4.18)

**Files:** Modify `silly_kicks/spadl/_skillcorner_inference.py`, `silly_kicks/spadl/skillcorner.py` (remove the `game_interruption_before` dispatch `:379-382`; `preserve_native: bool | list[str] = False`; docstring `:242`). Tests: `tests/spadl/test_skillcorner_restarts.py`, `tests/spadl/test_skillcorner_penalties.py` (new).

**Produces:** `infer_restarts(pp) -> pd.DataFrame`, `infer_penalties(pp, match_metadata) -> pd.DataFrame`; each derived row has `original_event_id` = `b` (period-end penalty: `a`), `action_provenance` `"derived"`, `player_id` `<NA>`, `time_source` `"placeholder"` or `"order_only"`.

- [ ] Tests (fail first), with the spec's values:
  - count rule: paired, "before"-only, "after"-only; a period-end restart is not counted; a period-end penalty is counted;
  - team, including a disagreeing pair (the "before" marker wins);
  - subtype (15 m rule; nearer flag for an unknown side; "after"-only → crossed);
  - start: rule points from `tracking/_gk_geometry.py`; side of the previous possession's end; NaN start for a restart after an "after"-only restart (corner: `_CORNER_X`, `start_y` NaN; goal kick: `_RULE_POINT`);
  - end; result (paired, "after"-only with a goal of either team, another restart);
  - times: paired = reception − 0.02 s `"placeholder"`; "after"-only = next reception − 0.03 s `"order_only"`; period-end penalty = previous row + 0.01 s;
  - penalty result: "after"-only score rise → `success`; paired → `fail` with a rebound goal at `b`; goal-marker fallback; last period with no later event (metadata final score); metadata without a score key → `fail`;
  - key: the shared-possession case gives two keys (`(p1, corner)`, `(p2, penalty)`); `preserve_native=["team_score", "opponent_team_score"]` carries both columns; `True` keeps today's output.
- [ ] Implement; run both test files and `tests/spadl/test_skillcorner*.py`. Expected: PASS. A test that pins the old first-possession restart type is updated with the reason "R30: the first possession takes its own type" in the ledger.

### Task 5: The idsse penalty row (R57, spec §4.16)

**Files:** Modify `silly_kicks/spadl/sportec.py` (`_MAPPED_EVENT_TYPES`, the shot mask `:643`, `:947`, the empty `Penalty` player from `shot_player`); R63 (bar A25): `silly_kicks/providers/sportec/parse.py`, the nested-child loop of `_build_event_row` only. Tests: `tests/spadl/test_sportec_penalty.py` (new), `tests/providers/sportec/test_parse_penalty_outcome.py` (new: a scored and a saved nested shot, each failing on the pre-R63 parser). R64 (bar A26): `scripts/build_idsse_per_period_fixture.py` (new), `tests/datasets/idsse/per_period_match.parquet` (rebuilt) and its README section, the `NOTICE` entry, `scripts/extract_provider_fixtures.py` (idsse `per_period` refused); tests `tests/scripts/test_build_idsse_per_period_fixture.py` (new: the generator on the committed slice; the refusal) and `test_committed_fixture_penalty_through_native_shaping_is_the_home_teams_goal`.

- [ ] Tests (fail first): `test_scored_penalty_is_shot_penalty_success`, `test_missed_penalty_is_shot_penalty_fail`, `test_penalty_without_nested_shot_is_excluded_by_name`, `test_penalty_player_is_nested_shooter`; R63: `test_penalty_nested_shot_outcome_is_read[scored|saved]`, `test_penalty_nested_outcome_attributes_take_the_shot_outcome_prefix`; bar A23 point 7: `test_penalty_with_a_nested_shot_but_no_outcome_warns_and_names_the_row`, `test_penalty_without_nested_shot_does_not_warn`.
- [ ] Implement; run. Expected: PASS. Then check the 3 stage penalties (J03WMX, J03WOH, J03WR9) against the raw `ShotAtGoal Player` on the DGX; ledger the three ids. (The STOP of the first build, R63: the parser did not flatten the nested outcome; the owner ruled option a.) Tell the supervisor when the parser change is in the tree, for its bronze check of bar A25 point 4; then bar A23: `shot_penalty` `success` with the nested shooter on the 3 matches, 19 goal rows on the idsse stage.

### Task 6: The time step (spec §4.3, §4.4, §4.17 kick rule)

**Files:** Create `silly_kicks/tracking/_action_time.py`; export `sync_action_times`. Modify `silly_kicks/tracking/skillcorner.py:268` (ball `visibility` from `is_detected`), `scripts/_loader_pining.py:856` (`_skillcorner_bronze` keeps the ball `is_detected`, both input paths of `:922`). Test: `tests/tracking/test_action_time.py` (new).

**Produces:**
- `restart_gaps(actions, frames) -> pd.DataFrame` (key of §4.18, `gap_start`, `gap_end`; read before any rule).
- `sync_action_times(actions, frames, *, params=None) -> pd.DataFrame` (rules 1-4).

- [ ] Tests (fail first):
  - refusal of a table without `time_source`; ELASTIC skipped with no `"event"` row; an aligned `"event"` row → `"elastic"`, an unaligned one stays `"event"`;
  - corner kick: a seen rest and release → `"kick"`, time = last rest frame, start = rest point; an extrapolated (undetected) ball only → no kick; frames with `visibility` `None` → no kick;
  - the gap: a flag rest inside `b` gives no kick (corner and paired penalty); period-start gap from the period's first frame; period-end penalty gap to the period's last frame;
  - order-only corner: first-run rule with a later run; none when the flag run is the last run and the follower is a throw-in; a kick when the follower is a goal kick;
  - retype only a `"placeholder"` corner (a flipping subtype); an `"order_only"` corner with a kick and a goal of its team stays `corner_crossed`; end = rest point for an `"order_only"` row and a penalty row; a first-possession corner keeps its end;
  - penalty kick at the mark (rest within 1.5 m of `_PENALTY_SPOT`, a retake → last run);
  - dribble midpoint; inversion re-sorted by `sort_actions_chronologically` and renumbered;
  - a SkillCorner match with no tracking: no raise, all derived rows keep their unmeasured source;
  - the step keeps the §4.18 key of every row;
  - `sync_action_times` joins `PURITY_ENTRIES`, the call-convention registry and `feature_glossary` (and `metric_contracts` if its gate asks).
- [ ] Implement; run `pytest tests/tracking/test_action_time.py tests/tracking/test_skillcorner*.py -q`. Expected: PASS.

### Task 7: Stoppage spans with the kick mapping (R44, spec §4.9)

**Files:** Modify `silly_kicks/tracking/_stoppages.py:69-110` (`skillcorner_stoppage_spans(..., kicks: Mapping[tuple[str, str], tuple[int, str]] | None = None)`, docstring `:69-75`), `scripts/_loader_pining.py:632-651` (`_build_match` order for SkillCorner: build → `sync_action_times` → kick mapping with the `restart_gaps` check → spans → `with_stoppage_state`; for idsse (today `:629-631` returns `with_stoppage_state` right after `_build_idsse`) and gradientsports: build → `sync_action_times` → their existing spans path; the SkillCorner `preserve_native` list of §4.15; the `_build_match_actions` docstring of §4.10). Test: `tests/tracking/test_stoppages.py`, `tests/scripts/test_loader_pining_kicks.py` (new), `tests/scripts/test_loader_pining_events_only.py:10-11`.

**Size:** about 25 lines in `_stoppages.py`, about 20 in the loader; 10 span tests + the loader check test.

- [ ] Tests (fail first): the events-only property of §4.10 (events-only = full build before the step; full build = the step applied; keyed by §4.18, with a shared possession and two keeper saves in a row); one per row of the 8-row table; the shared possession; two kicks in one gap; a reversed pair (the `"after"` kick dropped and counted); the period-end side; a paired penalty whose `b` ends its period keeps `live`; the loader's check fires (an injected kick after `b.frame_start` → count 1); the idsse branch of `_build_match` returns `"elastic"` rows on a fixture with frames (R32), and the gradientsports branch returns its rows unchanged.
- [ ] Implement; run. Expected: PASS.

### Task 8: Restart enrichment (spec §4.7)

**Files:** Modify `silly_kicks/tracking/_gk_geometry.py:317-454` (`resolve_restart_geometry` reads `time_source`). Test: `tests/spadl/test_add_restart_coordinates.py`.

- [ ] Tests (fail first): placeholder/order-only corner, goal kick, throw-in (own start, `"restart_prior"`, `_PRIOR_CONF`); free kick (`"unresolved"`, NaN); NaN start (`"unresolved"`); derived penalty (`_PENALTY_SPOT`, `"restart_prior"`); kick row (`"tracking_ball"`); tracking tiers and the keeper window skipped.
- [ ] Implement; run. Expected: PASS.

### Task 9: Corners (spec §5.2 corners row, §4.18 GI join)

**Files:** Modify `silly_kicks/corners/_identify.py:76-97`, `corners/_windows.py:81-84`, `corners/_columns.py` (`t_kick` text); the window readers `corners/_encode.py`, `corners/_effectiveness.py`, `corners/_reflect.py`, `scripts/build_corner_routines.py` (`_snapshot_rows`), `scripts/validate_corners.py` (`_divergence_rows`) by the no-window rule (supervisor C, after the impl round 2). Test: `tests/corners/test_identify.py`, `tests/corners/test_windows.py`, `tests/corners/test_encode.py`, `tests/corners/test_effectiveness.py`, `tests/corners/test_reflect.py`, `tests/scripts/test_build_corner_routines.py`, `tests/scripts/test_validate_corners.py`.

- [ ] Tests (fail first): NaN `t_kick` and `<NA>` side on placeholder and order-only rows, no window, row kept; the GI join on the §4.18 key with the shared-possession case; a kick corner takes `t_kick`, side and kind from the stepped row.
- [ ] Implement; run. Expected: PASS.

### Task 10: Score and phase lookups (R50, R51, R52, R56; spec §4.11, §4.12, §4.15, §5.2)

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py:531-695`, `:1094`, `:1101`; `_ghost_outfield.py:252-254`; `_xcross_attempt.py:498`, `:980`; `causal/opportunities.py:338`. Test: `tests/tracking/test_ghost_gk.py:860-938` (call form), `tests/tracking/test_match_context_lookups.py` (new).

**Produces:** callbacks `(game_id, period_id, time_seconds)`.

- [ ] Tests (fail first): two periods and extra time for each lookup; the shoot-out row is not a goal; a penalty goal, a direct free-kick goal and an own goal on `bad_touch` move `score_diff`; order-only rows open and end no window (throw-in, goal, penalty, restart followers); a table without `time_source` gives today's phase; running score: side by `ids_match` with mixed id dtypes, derived row not read, effect rule, a decrease → goal rows, partial `<NA>` → goal rows, no columns → goal rows; one test per caller that the period arrives.
- [ ] Implement; update the 10 lookup tests to the three-argument call (values unchanged; reason in the ledger); run. Expected: PASS.

### Task 11: Action-window readers (R54, R60; spec §4.13)

**Files:** Modify `silly_kicks/vaep/labels.py` (`window="time"`), `xtgk/_retention_labels.py`, `xtgk/_turnover.py`, `tracking/_packing.py`, `team_metrics/{_possession,_compute,_pressing}.py`, `spadl/utils.py:941-975` and the atomic twin, `coordination/_windows.py:212-300`, `:474-498`. Tests: the existing test file of each module plus `tests/test_order_only_readers.py` (new).

- [ ] Tests (fail first): each reader on a table with an `"order_only"` row gives the value of §4.13; each reader on a table with none equals today's output (VAEP `"time"`: same values, nullable `boolean`).
- [ ] Implement; run. Expected: PASS.

### Task 12: Shot-time readers and the producer (R45, R50, R58; spec §5.3, §5.4)

**Files:** Modify `tracking/_xshot_occurrence.py:293-340`, `tracking/features.py:638-887`, `atomic/tracking/features.py`, `tracking/_shot_goalmouth.py:772`, `causal/opportunities.py:104-191`, `:474`, `tracking/_packing.py` (reception on such a row), `tracking/_run_features.py:284` (pointer drop + blank step). Tests: each module's test file plus `tests/tracking/test_run_features_blank.py` (new).

- [ ] Tests (fail first): each reader skips a shot row with `is_time_unmeasured`; the producer blanks every family-added column on placeholder and order-only rows (off-ball runner counts included); other rows unchanged.
- [ ] Implement; run. Expected: PASS. A family column whose dtype has no missing value: STOP.

### Task 13: The pre-repair bundle warning (R55, spec §4.14)

**Files:** Modify `tracking/_feature_contract.py:66-107`, `:198-260`; `tracking/_warnings.py`; the contract blocks `_ghost_gk.py:1809-1840`, `_ghost_outfield.py:549`, `_xcross_attempt.py:589`. Test: `tests/tracking/test_feature_contract.py`, `tests/tracking/test_declared_constant_values.py`.

- [ ] Tests (fail first): old artifact (no `match_context_revision`) warns `PredatesDeclaredConstantWarning` with the meaning and loads; new artifact loads silently; another value raises. The message's version literal is written at commit prep (Task 15).
- [ ] Implement; run. Expected: PASS.

### Task 14: Campaign, trainers and adoption (R38, R49, spec §4.8)

**Files:** Modify `scripts/hf_jobs/campaign.py:143-161`, `:240-263`; `scripts/hf_jobs/registry.py`; `scripts/train_xshot_occurrence.py:483-530`, `:883-930`; `scripts/train_xcross_attempt.py:1085-1097`; `scripts/hf_jobs/adopt.py:1135-1144`; `scripts/hf_jobs/campaigns/tf46_refit.toml`. Tests: `tests/scripts/hf_jobs/test_campaign_arms.py`, the xShot trainer test file that holds its `_select_work` tests, `tests/scripts/hf_jobs/test_adopt.py`.

- [ ] Tests (fail first): the 4 campaign tests of §4.8; xShot override and the refusal without the two-arm corpus; `rule_verdict` / `rule_why` in both trainers; the adoption-draft line. `adopt` reads both fields from the fresh `metrics.json` of the instance (`adopt.py:842`, the file in which each trainer writes `candidates`); the test pins that path. If a trainer writes `candidates` elsewhere, the test pins that file and the ledger names it.
- [ ] Implement. In `tf46_refit.toml`: `ship_variant = "public"` on both wheel jobs (`:55-61`, `:71-78`), comments cite R49 and R38; and `[campaign.units_per_job_by_stage] "xshot_occurrence/study" = 2` (sizing below). Run. Expected: PASS.

**xShot study sizing (2 h job limit).** The timeout logs (`~/refit144/xshot_timeout/summary.out`) show that a study job runs its units mostly one after another: the units done in a timed-out job sum to about the limit (6,762 to 7,214 s, each job with one unit open; the sums above 7,200 s show a small overlap). Unit times: `public` 613-1,524 s, `sc_extended` 1,078-2,780 s, `single` 1,112-1,818 s. Today `units_per_job = 20` puts 8-11 units in one job. With 2 units per job the worst job is 2 × 2,780 = 5,560 s, 77 % of 7,200 s; the repaired stage adds 2.80 % of the SkillCorner training frames (R44). A paired variant has 22 units → 11 jobs (≤ `max_shards` 12); a Hub variant 6 units → 3 jobs. The job start (install and data fetch) is outside the unit sums, and the logs do not bound it exactly. The two units of a job take at most 5,560 s even with no overlap, which leaves 1,640 s for the start before the limit. The two failed starts (`ReadTimeoutError`, exit code 2) are covered by `attempts = 3`.

### Task 15: Records, the base, the gate, the review and commit 1

**Files:** the new ADR (number at commit prep: the first free number after the highest ADR on `origin/main` and `origin/feat/cross-definition`); ADR-133 and ADR-132 amendments; `CHANGELOG.md` (4.144.0 lines of spec §7); `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (`:26`, `:162`, `:186-196`, `:337-396`); `docs/research/cross-definition/caller-sweep.md:93`; `docs/context/providers.md`; the ADR-133 bullet of `AGENTS.md`; the docstrings of spec §7; the spec header and the sweep "Tree" line; the acceptance scripts moved into `docs/research/action-time-repair/`.

- [ ] R65-R67 (owner, 2026-11-18): the `sample_match.parquet` licence lines (bar A27, text only); the idsse fixture tests read the two files through `tests/_idsse_fixtures.py` (`shape_events_to_native`) with `tests/test_idsse_fixture_path.py` as the pin (bar A28); the revision-tag gate `tests/tracking/test_match_context_revision_gate.py` on the probe `tests/tracking/_match_context_probe.py`, with mutation teeth (bar A29). R68 (owner, 2026-11-18): `extract_candidate_rows` takes `game_id` and `action_id` from the actions columns by position, and `scripts/_receiver_validation.trajectory_weak_labels` takes `action_id` the same way (bar A30, both scripts, no library code); tests `test_candidate_rows_keep_the_actions_id_dtype_on_an_all_numeric_table` (both strategies) and `test_trajectory_weak_labels_keep_the_actions_id_dtype_on_an_all_numeric_table` fail first; the object-id workaround in `test_sb_feature_builds_ignore_time_source.py` goes. Results to the supervisor before the A/B request.
- [ ] The StatsBomb-only artifacts (the `sb_open` and `sb360` jobs of `tf46_refit.toml`): a test `tests/scripts/test_sb_feature_builds_ignore_time_source.py` builds each one's features on a fixture with and without `time_source` and asserts equal frames (spec §6).
- [ ] Base: the spec header and the sweep "Tree" line name `6eee84ce` (the branch point, reachable after the amend). Run the six sweep commands and the §8 gate command at `6eee84ce` plus the working tree; if a count differs from the spec, correct the number and ledger it.
- [ ] Write the records listed above. 4.144.0 is released without the repair, so the wheel changes and the version moves by the version rule (ADR-079), set at commit prep, not before. The R55 message and the bar A23 point 7 warning are both written relative to 4.144.0, the last release without the repair, so neither takes the new version. Check: 4.144.0 is the last release before this one, else both texts ("in every release after silly-kicks 4.144.0 … fitted with silly-kicks 4.144.0 or earlier"; "a parser of silly-kicks 4.144.0 or earlier") change to the last release without the repair.
- [ ] Gate on the DGX (exact tree): `python -m pytest $(git grep -lE '<§8 terms>' -- 'tests/*.py') -q --basetemp ~/refit144/pytest_tmp` with the pining env of the DGX (`~/.pining_owner.env`), with no marker filter, so the e2e tests of the gate files run on real data (36 of the 668 files hold e2e tests, for example `test_skillcorner_e2e`, `test_xshot_occurrence_integration`). A skipped e2e test is listed with its skip reason in the ledger; a skip for missing data is a STOP unless the owner approves it, then `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `pyright`. Expected: all exit 0; ledger the counts. The moved pins (spec §8): the candidates are the edits to test files that exist at the old C1, from `git diff --diff-filter=M c1-3ec4b79a-keep -- tests/` (new files excluded; the keep ref isolates the repair from C1's own edits). Each expected value that this change edits there is named with its old value, its new value and the reason, in the ledger and in the commit message.
- [ ] The warning reach (spec §4.14): load every bundled artifact of the wheel and every manifest Hub entry on the repaired code; only the pre-repair ghost-GK, ghost-outfield and xCross bundles may warn `PredatesDeclaredConstantWarning`. Any other warning is a STOP. Repeated after the adoption (Task 17).
- [ ] Trial stage (not the stage of record): on the DGX, build a sample from the gated tree (all 7 idsse matches; gradientsports 10504, 10505, 10506; SkillCorner 1906819, the one period-end-penalty match of the xshot187 roster, 1307205, a paired penalty whose `b` ends its period, and 1021404. The second period-end penalty of the cache is in 2037382, outside the roster, so not in the trial; the supervisor's events-only build of 2037382 with the fix gives 3 derived penalty rows in period 1, each `"order_only"`, the third with the id of the period's last native row. Built with `materialize_tc3_frames --allow-dirty`, not the stage of record) into `~/refit144/trial-<tree7>/`. Run each spec §8 check that must count 0, and A6 items 1-2 on the 7 idsse matches (the 155 `CalculatedFrame` shots). A count that is not 0, or an A6 number below its bar, is a STOP before the commit.
- [ ] `/final-review` (mad-scientist-skills:final-review).
- [ ] A/B implementation review of the delta against the C1-approved tree `cc4fbdcb` (`git diff cc4fbdcb <staged tree>`); reports `D:\Development\_reviews\2026-11-18-action-time-repair-impl-{A,B}.md`. Fix every finding; any A/B disagreement goes to the owner.
- [ ] After the last fix: the gate and the lint trio again on the exact tree, the trial again if code changed, and an A/B re-review until both approve. Ledger the `git write-tree` id that passed.
- [ ] Base check: `git grep -n 3ec4b79a $(git write-tree)` lists only this plan's old-C1 references (the keep ref, the owner-gate diff). Any other hit is fixed first. The spec §8 line "668 of 1,227 at `3ec4b79a`" names `6eee84ce` with its count from the base step.
- [ ] Owner gate: show the file list, `git diff --stat c1-3ec4b79a-keep` and the tree id that passed. On his yes: `git commit --amend` (C1 message updated to name the repair; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Then his yes again for `git push --force-with-lease origin feat/release-refit-144`.
- [ ] CI: the full suite on PR #308 runs green before Task 16 (`gh run watch` on the run). A red run is a STOP.

### Task 16: Restage and acceptance (bars A6, A11; spec §8)

- [ ] On the DGX, from commit 1: build `xshot187`, `tc3` and `xcross_probe` by the staging procedure of `docs/howto/hf-jobs-training.md` ("Staged datasets"), with K disjoint workers into one generation (never serial). Output under `~/refit144/stage-<commit7>/`.
- [ ] Report the total share of SkillCorner frames that become `live` because `b` is `live` for every marked pair (spec §4.9), by restart type and in total.
- [ ] Run the acceptance scripts of spec §8 (A6 items 1-6, including the idsse ELASTIC share and accuracy against the investigation values) and ledger each output. Each count the spec sets to 0 must be 0 (else STOP): phase from an order-only row; `live` frames after an `"after"` kick; kick rows outside their gap; key duplicates; non-missing family columns on unmeasured rows.
- [ ] Bar A11: the local gate check of xShot, xCross, ghost-GK, ghost-outfield and corners on the repaired stage; logs to the ledger.
- [ ] Owner gate: Hub create (private) and upload of the three datasets; then `datasets record`. Each needs his go.

### Task 17: Commit 2 and the refit

- [ ] Update `scripts/hf_jobs/datasets.toml`, the lock and `stage_provenance.json` for the new stage. Scoped gate: `tests/scripts/hf_jobs/` and every test file that reads `datasets.toml`, the lock or `stage_provenance.json` (`git grep -lE 'datasets\.toml|datasets\.lock|stage_provenance' -- 'tests/*.py'`), plus the lint trio; `/final-review`; A/B review of the commit-2 delta. Owner gate on commit 2 (with the tree id) and on its push. CI green before the launch; red is a STOP.
- [ ] The refit by the first directive and `docs/howto/hf-jobs-training.md` ("The command sequence"): `preflight`, launch by the owner (classic token, R48), `status`, `retrieve`, `verify`, `compare` with two baselines (the run keys from the run ledger): the shipped launch 7 (the first directive's item E6), which shows what changes for a user, and the C1 launch on the f5fc2e2b stage (`tf46_refit-3ec4b79-*`), which shows what the repair changes. `adopt` (stops at the commit gate). Check that `adopt._expectation_edit` (`adopt.py:937`) takes the `public` values of the 4 wheel entries (spec §4.8); if not, STOP. The wheel xShot and xCross ship `public`; a failed gate of that arm is a STOP (bar A17).
- [ ] The corner run on the DGX (first directive item E8; the STOP of the last run): `validate_corners.py` and `build_corner_routines.py` on the repaired stage; the ledger takes the p values of 4.139.0, 4.144.0 and the repaired run. The corner memo headline and table follow it, with the cause (R41).
- [ ] The census by the rebaseline rule (R39, R25: `_rebaseline` of C1) on the new run; the ledger takes its output.
- [ ] Commit 3 (the adoption): the interim-tree changes come back to the strict state (served-value pins of the new bundles, `PredatesDeclaredConstantWarning` as `error::` in `pyproject.toml:384-388`, `test_no_bundled_artifact_predates_a_declared_constant`). Gate: the §8 command (with e2e) plus `tests/scripts/hf_jobs/test_adopt*.py` and the bundle tests, and the lint trio, on the exact tree; `/final-review`; A/B review of the commit-3 delta, again after every fix. Owner gate on commit 3 (with the tree id) and on its push. CI green before the merge; red is a STOP. Then parity, merge (owner), tag (owner, after green CI on `main`), publish (owner).

## Acceptance items and who runs them

| Item | What | Run by | When | Where (host: path) |
|---|---|---|---|---|
| V1 | the source table, each measure again | supervisor | after Task 16 | DGX: `~/dl-review-xdef/` |
| V2 | the six sweep commands | author, then supervisor | Task 15 base step | local: `docs/research/action-time-repair/reader-sweep.md` |
| V3 | the six A6 items on the repaired stage | author, then supervisor | Task 16 | DGX: `~/refit144/stage-<commit7>/acceptance/`; then local: `docs/research/action-time-repair/` |
| V4 | the A11 gate check of each family | author | Task 16 | DGX: `~/refit144/stage-<commit7>/gatecheck/` |
| V5 | the corrected records | supervisor reads | Task 15 | local: the commit-1 tree |
| V6 | no edit in the A9 areas (`git diff --stat`) | supervisor | Task 15 owner gate | local: `git diff --stat` |
| V7 | the size of the change | supervisor | Task 15 owner gate | local: `git diff --stat` |
| V8 | A12 numbers and ruling | closed (§4.5) | — | — |
| V9 | 0 frames with a phase from an order-only row | supervisor | Task 16 | DGX: `~/dl-review-xdef/` |
| V10 | the gate result of the public wheel xShot and its manifest entries | supervisor | Task 17 after `verify` | DGX: `~/refit144/launch-<commit7>/verify/` |
| V11 | his two period checks on the repaired code; the two-period tests | supervisor | after Task 10 (code) and Task 16 (stage) | DGX: `~/dl-review-xdef/` |
| V12 | the GS goals of the lookup = the goal rows of the full rule outside period 5; no frame after a shoot-out row | supervisor | Task 16 | DGX: `~/dl-review-xdef/` |
| V13 | each R54 reader on a table with no order-only row gives today's values | author test (Task 11), supervisor reads | Task 11, Task 16 | DGX: `~/refit144/pytest_tmp/` and the ledger |
| V14 | the 15 adopted bundles load with no warning; an old one warns | supervisor | Task 17 after `adopt` | DGX: `~/dl-review-xdef/` |
| V15 | the SkillCorner end-of-match score against the metadata; matches per path; early rises | supervisor | Task 16 | DGX: `~/dl-review-xdef/` |
| V16 | 19 idsse goal rows; goals per match = final score | supervisor | Task 16 | DGX: `~/dl-review-xdef/` |
| V17 | the penalty rows by class, result and source; the kick share; 0 `live` after a different-kind kick; xShot positives from penalty rows | author, then supervisor | Task 16 | DGX: `~/refit144/stage-<commit7>/acceptance/` |
