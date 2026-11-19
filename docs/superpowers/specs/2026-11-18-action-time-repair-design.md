# Action-time repair: design (silly-kicks 4.144.0, PR-S209, second part)

- Status: r15 with the rulings of the table in section 2.1 (the others are in the directive), no owner question open; date 2026-11-18, rulings to 2026-11-19.
- Branch: `feat/release-refit-144` (base `6eee84ce`, the branch point; draft PR #308). The sweep and the gate counts hold at `6eee84ce` (Task 15 base step).
- Directive (scope, rulings R29-R60, bars A1-A24): `D:\Development\_reviews\2026-11-18-action-time-repair-supervisor-directive-C.md`.
- Facts: the directive §3 (with its correction of fact 3.5) and the investigation report `~/refit144/investigation_r30.md` (DGX). This spec does not repeat them.
- Reader sweep (bar A7): `docs/research/action-time-repair/reader-sweep.md`.
- Measurement scripts: DGX `~/refit144/`, each with its `.out`. The stage `~/cross-def/xshot187-f5fc2e2b/out` is read only.
  - `elastic_check.py`, `release_check.py`, `retime_order_check.py`, `shift_check.py`.
  - `corner_kick_check.py`, `corner_kick_nodet.py`, `restart_census.py`, `taker_check.py`.
  - `spec_r2_checks.py` (points 1, 3, 6, 8 of the supervisor's read).
  - `order_only_overlap.py` (the order-only rows and the phase windows, §5.2).
  - `order_only_corner_check.py` (the rest runs in the brackets of the 690 order-only corners, §4.4).
  - `sc_score_lag.py` (the running-score effect rule and its lag, §4.15).
  - `derived_key_check.py` (the key of a derived row, §4.18).
  - `penalty_wait_check.py` (the wait before a penalty kick, §4.17), on the output of `sc_penalty_check.py`.
  - The repository keeps the cited scripts in `docs/research/action-time-repair/scripts/`, each with its inputs as arguments (R84).
  - The supervisor's checks: `~/dl-review-xdef/spec_checks_sc.py`, `spec_checks_stage.py`, `phase_checks.py`, `phase_checks_sc.py`, `phase_checks_frames.py`, `score_goals_check.py`, `sc_goal_sources.py`, `na_actor_runs_probe.py`, `sc_goal_rows_direct.py`, `sc_goal_schema.py`, `sc_score_columns.py`, `sc_score_columns2.py`, `sc_penalty_check.py`, `sc_penalty_post.py` and `gs_penalty_check.py`.
  - The plan moves the ones that the acceptance check needs into `docs/research/action-time-repair/`.

## 1. Goal

Each on-ball action time in the staged corpus is the moment of the action, or the row says that it is not. The refit then trains on that stage (R36). Scope, non-goals and limits are those of the directive §2.

## 2. Decisions for the owner

### 2.1 Ruled (2026-11-18, relayed by the supervisor)

The owner's words: "D1,D2,D4,D5 approved as recommended", then "1-6 as recommended, 7:keep using the classic token for now.", then (D8, D9) "1:approved." and "2:fix properly if possible/feasible", then (D10) "1:repair in this cycle", then (D11) "1:repair it." and (D12) "2:looking to repair it now, investigate so we can make an informed decision", then (D13, D14) "2+3 as recommended.", then (the form of D12) "a:yes. b:yes. c:c1. d:limit", then (the size audit, 2026-11-19, R83-R85) "F4: This cycle: F3:Make them take their inputs if gold standard. Other findings fine for adoption commit.". On D9 he added: "throw-ins are very much on my radar with tracking data going forward (similar to looking for man/zonal, blocking on corners, ultimately looking to analyze/classify movement patterns on throw-ins inspired by Thomas Gronnemark's course materials)".

| Ruling | Decision | Ruled |
|---|---|---|
| R38 (D1) | Wheel xCross bundles against the serve contract (`tests/tracking/test_from_variant_serve_identity.py:43-46`, ADR-044) | The wheel xCross bundles ship the `public` arm, by the serve contract. The Hub entries ship the arm that the paired rule picks. This amends ruling r3 of the corpus-widening cycle for the wheel. Mechanism: §4.8. |
| R39 (D2) | P15: the census of the new run | Accepted by the committed rebaseline rule (R25) when the repaired run exists. |
| R40 (D4) | A taker from the tracking for a derived restart row | No taker in this cycle. `player_id` stays `<NA>`. The evidence stays recorded: TF-79 names the provider's taker for 49 of 49 idsse corners and 0.842 of 114 gradientsports corners (`taker_check.py`). |
| R41 (D5) | The corner memo headline | It follows the new run: the 4.139.0, 4.144.0 and repaired p values, and the cause. |
| R43 | xShot in this release | xShot comes back into the refit. This replaces R28 and the supervisor's bar B10 (§6). |
| R44 (D6) | The SkillCorner stoppage spans (`tracking/_stoppages.py:69-75`) for the xShot/xCross training frames | The first possession after a SkillCorner restart is `live` for the xShot and xCross gates, and the dead span ends at a measured kick. Evidence: 98 stage matches, 4,364,743 frames, 69.2 % live. The first possessions are 108,420 frames (2.48 %), and the flight after a measured kick is 14,130 frames (0.32 %), so 2.80 % of the frames become training frames (`spec_r2_checks.py` 8). The supervisor checked that the ball is in play when `b` starts (`first_possession_live.py`: a resting ball at the start in 3.8-7.0 % of the rows, short, so a trapped ball). |
| R45 (D7) | Frame features at a `"placeholder"` row | The producer `run_tracking_features` (`tracking/_run_features.py:284`) drops the frame pointer of every `"placeholder"` row before the families run, so all its frame features are NaN on the producer path. A direct `add_*` call is not covered, and its docstring says so. The frame linker is not touched (bar A9). R50 applies the same drop to an `"order_only"` row (§4.1). **The pointer drop alone does not meet the ruling** (supervisor, `na_actor_runs_probe.py`): `add_off_ball_runs`, a default family (`_run_features.py:81`, `:114-120`), slices the frames by the action time, not by the pointer, and writes 0 runners for an action that has frames in its slice and no candidate (`_off_ball_runs.py:185-194`). With a `<NA>` actor, both runner counts are 0 on 100 of 100 probe rows. So the producer also blanks: after the families run, it writes the missing value of the column's dtype into each column that a family added, on each `"placeholder"` or `"order_only"` row (§5.3). |
| R49 (D8) | The wheel xShot bundles against the serve contract (`tests/tracking/test_from_variant_serve_identity.py:43-47` checks both wheel roots) | The wheel xShot bundles ship the `public` arm, by the serve contract, as the wheel xCross bundles do (R38). The Hub entries ship the arm that the paired rule picks. Bar A17: the mechanism of §4.8; the `public` arm passes its own gates on the repaired stage; a failed gate fails the run, with no fallback to the other arm, and is a STOP for the owner; the adoption record reports the verdict that was set aside. |
| R50 (D9) | The ghost `phase` at an "after"-only restart row | Not a limit. Bar A16: the defect class is each window that an "after"-only restart row opens over play that is not its own (1,333 rows: a throw-in follows 1,263, a goal 68, a penalty 2; supervisor, `phase_checks_sc.py`). A reader tells an order-only time from `time_source` alone; no reader opens a time window at it; every other row keeps today's phase rule on every provider, and the throw-in rule of `tracking/_ghost_gk.py:634` stays; the 599 "after"-only throw-in rows carry the same mark; a script counts 0 frames that take a phase from an order-only row. The end rule ("a throw-in row ends an open window") is not part of the repair. Mechanism: §4.1, §4.2, §5.2, §5.3. |
| R51 (D10) | The ghost score and phase lookups key on `game_id` and `time_seconds` with no period (`tracking/_ghost_gk.py:531-624`, `:627-695`); `time_seconds` starts again in each period (ADR-017) | Repaired in this cycle. Bar A18: a set piece opens a phase window in its own period only; `score_diff` at a frame is the score at that moment of the match (a goal of an earlier period counts, a goal of the same period counts from its time, a goal of a later period does not); every caller gives the period, at the fit and at the serve; reuse of `win_probability/_state.py:48-51` or the reason why not; tests that fail on today's code (two periods for each lookup, and extra time); the supervisor's one-clock checks show no difference on the repaired code; the records name the change and what a pre-repair bundle gets at the serve; every test that pins a value of today's lookups is named. The goals that the score lookup does not count are not part of R51 (they are R52 and R53-R59). Mechanism: §4.11. |
| R52 (D11) | The goal rule of the score lookup | Repaired. The score lookup counts each goal of the library's goal rule: a successful row of a shot type (`shot`, `shot_penalty`, `shot_freekick`), and an own-goal result on any type. Bar A19: one goal predicate for the library, read by `_build_score_lookup`, `vaep/labels.py` and `win_probability/_state.py` with no copy; the spec says what a shoot-out row is for the score; tests that fail on today's code (a penalty goal, a direct free-kick goal, an own goal on a type that is not a shot); the supervisor checks on the gradientsports stage that the lookup's goals equal the goal rows of the full rule (today 145 of 194). Mechanism: §4.12. |
| R53 (D12) | SkillCorner goals with no goal row | Repaired in this cycle; the form is R56-R59. The facts: 2,651 final goals in 889 matches, 2,334 goal rows (88.0 %), 317 goals (12.0 %) with no goal row in 252 matches (28.3 %; 58 with 2 or more; supervisor, `sc_goal_rows_direct.py`). |
| R54 (D13) | The readers that open a window at an action time with no frame | The one rule: each of the five action-window readers takes no window at an `"order_only"` row, through `is_order_only`. The sweeps found seven such readers; the rule covers each (§4.13). R60: the owner's word "yes" puts the coordination windows, found by the later sweep, under this rule. Bar A21: the spec says what each reader gives for such a row; the VAEP `"time"` mode label dtype changes, and the records name the change and its reach (that mode only); on a table with no `"order_only"` row each reader gives today's values, shown by a check; tests for each reader on a table with an `"order_only"` row. Mechanism: §4.13. |
| R55 (D14) | What a bundle fitted before the score and phase repairs gets at the serve | It loads with a warning, not a refusal and not silence. Bar A22: a ghost-GK, ghost-outfield or xCross bundle fitted before this release warns that its score and phase inputs are now computed differently from its fit; the mechanism is the ADR-050 feature contract, and the spec says what the contract does when the library declares a constant that the artifact lacks; a bundle fitted in this cycle loads with no warning, shown on the 15 adopted bundles; one warning covers each repair of these inputs in this release (R51, R52 and the ruled form of D12); tests for an old and a new artifact. Mechanism: §4.14. |
| R56 (D12 a) | The SkillCorner score | From the provider's own running score (`team_score`, `opponent_team_score`), carried by the list form of `preserve_native`. Named limits: the old score stays for the first live play after the kick-off reception (median 1.5 s); for 4 of 2,358 goals the new score shows early; a match whose running score decreases on a side (3 of 889) uses the goal rows. Bar A20. Mechanism: §4.15. |
| R57 (D12 b) | The idsse penalty | A `shot_penalty` row from a DFL `Penalty` event that carries a nested shot, in `spadl/sportec.py`. Bar A23. Mechanism: §4.16. |
| R58 (D12 c) | The SkillCorner penalty | A derived `shot_penalty` row for each penalty marker, built in this cycle. Bar A24. Mechanism: §4.17. |
| R59 (D12 d) | The other SkillCorner goals with no goal row | A named limit (§9). With R56 they do not change the score. |
| R60 | The coordination windows (`coordination/_windows.py`) under R54 | "yes": they follow the one rule, as the other readers (§4.13). |
| R83 (size audit S4, 2026-11-19) | The goal rule of the public `add_game_state` (`spadl/utils.py`) counts only a successful `shot` row | Repaired in the amended commit 2 (R95): it reads `goal_mask` and `owngoal_mask` of `spadl/_goals.py`, and an own goal counts for the other team. A shoot-out kick does not change the game state. Tests that fail on today's code: a penalty goal, a direct free-kick goal, an own goal on a `bad_touch` row, a shoot-out row. Bar A32. |
| R84 (size audit S3, 2026-11-19) | The research scripts of `docs/research/action-time-repair/scripts/` | Each script takes each input root and output folder as an argument. No local path, home folder or user name is in a script, a docstring or the README. Done before the acceptance run of Task 16, which uses these scripts. Bar A33. |
| R85 (size audit S1, S2, S7, S8, 2026-11-19) | The cleanups of the adoption commit | In the amended commit 2 (R95): the raw search outputs of the pandas-floor sweep leave the repository; no file outside this spec, the plan and ADR-135 names a bar, a ruling or a reviewer finding (the lines this release adds), except the 19 CHANGELOG lines (R97) and one older defect name that is a false hit (`tests/tracking/test_bekkers_home_away_asymmetry_e2e.py:62`, "the D1 defect"). In commit 4 (R95, R97): the release notes say what changed and what a user must do; the acceptance record is one table of numbers with commands. Bar A34. |
| R62 | `preserve_native=True` | It adds no column (§4.15). |
| R65 | The `sample_match.parquet` licence lines | Corrected, text only. Bar A27. |
| R66 | The idsse fixture tests | They go through `shape_events_to_native` by one helper. Bar A28. |
| R67 | The revision tag | A gate for the revision tag. Bar A29. |
| R68 | The receiver trainer ids | `game_id` and `action_id` come from the action columns, not the `iterrows` row, so an all-numeric table keeps integer ids; the same in `scripts/_receiver_validation.py`. Bar A30, no library code. |
| R69 (2026-11-19) | The test runs | "Do not waste an hour running tests": the broad gate stops; named test files only. |
| R70 | The library kick rule | "Measure it": the library kick rule against the measured rule; §4.4 cites the library's 0.755. |
| R71 | The span of a period-end penalty | "Code to the spec": it ends at its period's last frame (`period_last_frame`, §4.9). |
| R72 | Four failures outside this spec | "Fix all 4 existing failures": the causal `test_run_with_fake_loader` fake moves to the ADR-052 seam; restdefense `test_e2e_sb360_open_method` by a library fix (`keeper_identity` reads `type_id` without `type_name`); the bekkers mirror tests (the loader registered in `sys.modules`, the float32 coordinates upcast before the reflection). |
| R73 | Commit 1 | "Amend, force push, update pr as needed, then watch ci until green". |
| R74 | The statsbombpy leak and the pandas floor | "Fold both of these into this cycle as well": an autouse guard in `tests/conftest.py` restores `statsbombpy.config.OPEN_DATA_PATHS` after every test; the pandas floor is in this cycle. |
| R75 | The pandas floor | "Fix the code, keep 2.1.1": `resolve_frame_ids_by_position` reads the `Int64` `frame_id` with `na_value=np.nan`; the CI job `pandas-floor` runs the matrix selection on pandas 2.1.1; CI, not a local run, runs it first. |
| R79 | The first `pandas-floor` run | "A": its 57 failures (five causes) are repaired in this cycle. Bar A31: `silly_kicks._pandas_compat` holds `PANDAS_COMPARES_READ_ONLY_OBJECT` (pandas >= 2.2.1, measured) and `masked_to_numpy`; `id_compat` copies a read-only object input before pandas 2.2.1 only; the `add_possessions` `team_id` read uses `masked_to_numpy`; the coordination script writes out numpy 2's weighted quantile; the proven test reads take `na_value`. |
| R80 | The compare sites of the floor search | "2" plus "check this as well": the 34 literal-subject sites are read (30 safe, 2 proven, 4 in `add_pre_shot_gk_context` repaired with a red test each); the 51 non-literal sites and the 122 variable float reads are a named limit (§9); the two raw id compares are checked, no change. |
| R82 | Bar A11 | "The HF campaign is the check (Recommended)": the campaign is the gate check of xShot, xCross, ghost-GK and ghost-outfield; its skill gates fail closed and `adopt` refuses a failed gate before any commit; no local run of those instances; the corner models keep their run on the DGX; a failed gate fails the run, swaps no arm and is a STOP for the owner (bar A17 stays). |
| R86 | The blank check of the unmeasured rows | "All 118, background (Recommended)": all 118 SkillCorner matches, at most 6 workers, no gate waits for it. |
| R87 | The Hub gate of the stage of commit 1 | "Go: new private repos (Recommended)": new private repositories with the suffix `-26a2c33`, upload, `datasets record`. |
| R88 | The other places with a written-out goal rule | "Decisions 1-3 approved as recommended" (1): each place that decides "this row is a goal or an own goal" reads `goal_mask` and `owngoal_mask` (§4.12). `goalscore` counts an own goal for the other team and the two VAEP formula functions take a previous own goal as a goal; the other places change no output. Bar A35. |
| R89 | The gradientsports shoot-out | (2): the converter gives the shoot-out rows period 5 (§4.12). Bar A36. |
| R90 | The local-path gate | (3): `tests/test_research_artifacts_carry_no_ids.py` reads the `.py` files under `docs/research/`. Bar A37. |
| R91 | The cut list | "1:cut as recommended": `r67_mutations.py` leaves the repository; the `acc_*.py` scripts stay; no test is merged for size. Added to bar A34. |
| R92 | `goalscore` at the shoot-out | "2:yes": a period-5 row counts no goal, in `goalscore` and its atomic twin. Bar A35 point 8. |
| R93 | The failed guard of bar A36 | "This needs to be fixed properly, gold standard. Ask the supervisor if need be, no quick hacks": a period-5 row of the gradientsports converter has no location (§4.12); the stage is built again in full from the commit that holds the final library. |
| R94 | The campaign on the stage of commit 1, and the commits | "Yes to both (Recommended)": the campaign on the stage of commit 1 is cancelled; five commits in place of three (bar A15): 1 the repair, 2 the stage of commit 1 (superseded), 3 every known library, script and test change, 4 the stage of commit 3 (`datasets.toml`, the lock, the stage provenance, the pin test), 5 the adoption. |
| R95 | The commits | "no commit-3, amend and force push commit-2" and "I will not have a wasted trail of micro-commits, bare minimum commits needed to preserve provenance, that is it": four commits (bar A15): 1 the repair; 2 the stage record of commit 1, amended with every known library, script and test change; 3 the stage of the amended commit 2 (`datasets.toml`, the lock, the stage provenance, the pin test); 4 the adoption. |
| R96 | The team-id compares of `add_game_state` | "Fix it (Recommended)": the function compares team ids through `silly_kicks.id_compat` (ADR-019), each team mask once. The output changes only where a team id differs only in dtype (`1` and `"1"` are one team now); a red test pins it, and a guard test covers a goal row with a missing team id. A missing result or period is no goal. |
| R97 | The process ids of the CHANGELOG | "At the adoption commit (Recommended)": the 19 CHANGELOG lines with ruling or bar ids leave with the release notes in user form, in commit 4. |
| R98 | The goal termination of ELASTIC | "Include it (Recommended)": `tracking/_elastic_sync.py` reads `goal_mask` / `owngoal_mask`. Its old check read `result_name`, which converter output does not carry, so on the real pipeline no goal got the virtual goal event; now every goal and own goal gets it (owner: "Keep the fix (Recommended)"). On the 7 idsse matches no action time changes (trial build of the working tree: A6 item 1 0.955, bar at least 93 %); `add_elastic_sync` now fills the receive-frame columns of a goal shot. The rebuilt stage measures A6 again; below the bar is a STOP for the owner. Added to bar A35. |

D3 (bar A12, the gradientsports corner) is closed by the supervisor: no defect is shown (§4.5). The GS corner rows keep `"tracking"`. R50 replaces D9's earlier options (a limit, the end rule). R42 (the repair amends C1), R46 (the limits of §9), R47 (the uncommitted adoption is discarded) and R48 (the classic write token) bind the plan and the records, not this design.

### 2.2 Open

None. Every decision of this design is ruled (§2.1).

**The owner's question on R35 ("is there a long term better way?").** Yes, as a data upgrade only. The same step can later measure the free-kick, goal-kick and throw-in kicks: a rest-then-release rule for free kicks and goal kicks, and a throw-in release (the TF-46 design §6.4 has the forms). The derived rows and `time_source` stay as they are; only the value changes from `"placeholder"` or `"order_only"` to `"kick"`. The supervisor's reach measure bounds it: the ball is seen to leave for 45.1 % of free kicks, 42.5 % of throw-ins and 26.4 % of goal kicks.

## 3. What this design takes from the parked designs (bar A1)

- **`time_source`** (`origin/feat/cross-definition`, spec §6.6 C):
  - Taken: one per-row provenance column; dtype `object`; validated in `_finalize_output`; part of `SPADL_COLUMNS`; non-geometric in the reflection registry; carried by `convert_to_atomic(preserve_native=...)`.
  - Not taken: its two-value vocabulary. It cannot name a synced, measured or placeholder time.
  - Not taken: its premise that a GS/SkillCorner native time is the moment of the action. This premise is false for SkillCorner restarts (directive §3.4).
- **`kick_time_source`** (`origin/tf-46-arm1-clean`, spec §6):
  - Taken: the SkillCorner bracket (the stoppage to the first possession; GI frame ids equal the tracking frame ids, verified again: raw frame 4236 = stage 422.6 s = GI `time_start` 07:02.6).
  - Taken: the rest condition (within 0.5 m for at least 0.5 s), then a release.
  - Not taken: the arbitration states, the gate table, retakes, the 12 columns and the taker gate. Their size is out of proportion to R34 (bar A14). D4 holds the taker.
- Nothing is taken from either branch's code (R33).

## 4. Design

### 4.1 One column: `time_source`

Every converter emits `time_source`. The values:

| Value | Meaning | Written by |
|---|---|---|
| `"event"` | The provider's event clock, not checked against the tracking | sportec, statsbomb, opta, wyscout, metrica, kloppy native rows; an idsse row that ELASTIC does not align |
| `"tracking"` | The provider's time on its own tracking clock | gradientsports and SkillCorner native rows; SkillCorner start-of-possession rows (interceptions, keeper saves, at the reception); GS synthetic rows (their source row's value) |
| `"elastic"` | Set from the tracking by ELASTIC (`align_events_to_frames`) | the time step (§4.3) |
| `"kick"` | A corner kick or a penalty kick measured from the tracking; the row's start is the measured rest point | the time step (§4.4, §4.17) |
| `"placeholder"` | A restart with a first possession, with no measured kick. The time is the first reception minus 0.02 s: an upper bound of the kick, at or after it. The place is a rule point. | SkillCorner derived restart rows with a first possession (§4.2); SkillCorner derived penalty rows with a first possession (§4.17) |
| `"order_only"` | A restart with no first possession (an "after"-only marker). The time keeps the row order only: it lies after the next stoppage, so it is not a moment of this restart and bounds nothing. The place is a rule point. | SkillCorner derived "after"-only restart rows (3,395 in 889 matches: corner 690, free kick 1,168, goal kick 938, throw-in 599; §4.2); SkillCorner derived penalty rows with no first possession (§4.17) |
| `"derived"` | Computed from the neighbouring rows | synthetic dribbles (`_add_dribbles`) |

- The vocabulary is `TIME_SOURCE_VALUES` in `spadl/schema.py`. `_finalize_output` refuses any other value.
- **The unmeasured-time predicate (R58).** `is_time_unmeasured(actions) -> np.ndarray` in `spadl/schema.py` is `True` where `time_source` is `"placeholder"` or `"order_only"` (all `False` without the column). The readers of a shot row's time use it (§5.4).
- **The order-only rule (R50, bar A16).** A reader that takes a row time as a moment, or as the bound of a time window, takes no window at an `"order_only"` row. The predicate is `is_order_only(actions) -> np.ndarray` in `spadl/schema.py`: `True` where `time_source == "order_only"`, all `False` when the column is absent. It reads `time_source` only, not the offset, the row order or the provenance (bar A16, point 3a). Every reader that opens a window reads it through this predicate (§5.2, §5.3).
- A reader that meets a table without the column treats each row as `"event"`. The time step refuses such a table (§4.3).
- A `time_seconds` value is never written with a source it does not have (bar A3).

### 4.2 SkillCorner: the restart is its own row (R30, R35)

The function is `infer_restarts(pp)` in `spadl/_skillcorner_inference.py`. It reuses the derived-row pattern of that module (bar A5).

- **The count rule.** Every restart marker `<type>_for` or `<type>_against` (type in corner, free_kick, goal_kick, throw_in) on a `player_possession` is one restart. The marker can be the possession's `game_interruption_before`, or the previous possession's `game_interruption_after` in the same period. An "after" marker on possession `a` and a "before" marker of the same type on the next possession `b` are one restart.
  - **A period end is not a restart.** An "after" marker with no possession after it in the same period is not counted: corner 23, free kick 100, goal kick 93, throw-in 41. Reason: the open-data triage of 2026-10-19 (supervisor) found that period-end markers are not restarts, because the period ends before the restart is taken.
  - Count on the 889 cached matches (`spec_r2_checks.py` 1, minus the period-end markers): 8,528 corners, 21,206 free kicks, 12,878 goal kicks and 28,685 throw-ins.
  - Of these, 690 / 1,168 / 938 / 599 have an "after" marker only. These are restarts that another interruption follows with no possession between: another restart, a goal (corner 22, free kick 46, throw-in 3) or a penalty (2).
- **The limit.** A restart with no marker on either possession is not in the data. The data has no taker event (directive §3.4).
- **The team.** On a "before" marker: `_for` is the marked possession's team, `_against` is the other team. On an "after"-only marker, the same rule applies relative to `a`'s team.
  - The paired markers agree on the team: corner 1.0000, free kick 0.9793, goal kick 0.9991, throw-in 0.9884.
  - Where they disagree, the "before" marker counts, because its possession is the receiver's.
- **`player_id`:** `<NA>` (D4). **`bodypart_id`:** `other` for a throw-in, `foot` otherwise.
- **The type.** The corner subtype is `corner_short` when the first possession is the restart team's and its reception lies within 15 m of the start, else `corner_crossed`. Free kicks use the same rule. The rule is new: it takes the 15 m threshold of the converter's `is_short` rule (`skillcorner.py:370-374`), but it measures from the restart's start to the first possession's reception, not between two possession starts. An "after"-only restart has no first possession, so it takes the crossed subtype. A corner with no known side measures to the nearer of the two flags at `_CORNER_X`. A free kick with no known spot takes the crossed subtype, as an "after"-only one does.
- **The start (a rule point; not measured, as `"placeholder"` and `"order_only"` say).**
  - Corner: (`_CORNER_X`, the touchline on the side of the previous possession's end). That side is the true side for 0.823 of 4,660 corners with the ball seen at a flag. The reception's side is right for only 0.634 (supervisor, `spec_checks_sc.py` 3).
  - Goal kick: `_RULE_POINT`.
  - Throw-in: the previous possession's end x, on the touchline of its side.
  - Free kick: the previous possession's end (where play stopped), in the restart team's frame. Not measured.
  - **A restart that follows an "after"-only restart** (its previous possession `a` carries the "after" marker of another restart kind; 3,322 rows in 889 matches: throw-in 1,263, free kick 1,054, corner 231, goal kick 774; supervisor, from `phase_checks_sc.out`). The end of `a` belongs to the earlier stoppage, so it gives this row no place. A goal kick keeps `_RULE_POINT` (fixed). A throw-in and a free kick get `start_x` and `start_y` NaN. A corner gets `start_x` `_CORNER_X` and `start_y` NaN (the side is not known). A measured kick (§4.4) gives a corner its rest point, as for any corner.
  - **How a reader tells it:** a derived restart row (`action_provenance` `"derived"`) with a NaN start coordinate has no known place. NaN starts already occur on the stage (gradientsports 4,732, idsse 160, SkillCorner 20 rows of `xshot187-f5fc2e2b`). The enrichment writes `"unresolved"` for it (§4.7).
  - The rule points are single-sourced from `tracking/_gk_geometry.py` (no copy).
- **The end.** The first possession's reception (`x_start`, `y_start`) in the restart team's frame. An "after"-only restart ends at its start.
- **The result.** `success` when the first possession is the restart team's, else `fail`. `result_source` is `"inferred"`. An "after"-only restart has no first possession:
  - When a goal marker follows it, the result is `success` if the scorer is the restart team, else `fail`. A goal marker names the side of its own possession's team: `goal_for` means that team scored. All 2,189 goal pairs in the cache agree on the scorer (`goal_marker_check.py`).
  - Otherwise (another restart or a penalty follows) the result is `fail`.
- **The time.**
  - A restart with a first possession (`"placeholder"`): the first possession's reception time minus 0.02 s. This keeps the row before a start-of-possession interception (minus 0.01 s, `_skillcorner_inference.py:104`). It is at or after the kick.
  - An "after"-only restart (`"order_only"`): the next possession's reception time minus 0.03 s, so it comes before the next restart row. It lies after the next stoppage, after a goal where a goal follows (§9).
  - So both sit immediately before the next possession's reception. Only the `"placeholder"` time bounds this restart's kick.
- **The first possession** takes its type from its own `end_type`. The `game_interruption_before` dispatch (`skillcorner.py:379-382`) is removed from the native rows.
- **Provenance.** `action_provenance` is `"derived"`. `original_event_id` is the event id of `b`, the possession after the row's gap: the first possession of a paired or "before"-only restart, and the next possession of an "after"-only restart. The key that this gives is §4.18.
- With no frames, the row exists with its `"placeholder"` or `"order_only"` time and its rule-point place (bar A5).

### 4.3 One step sets the time from the tracking: `sync_action_times`

`sync_action_times(actions, frames, *, params=None) -> pd.DataFrame` is a new public function in a new module `silly_kicks/tracking/_action_time.py`.
- It is pure. It refuses an `actions` table without a `time_source` column (`ValueError`). Stored 4.144.0 GS or SkillCorner actions would otherwise read as `"event"` and be synced by ELASTIC, which is poor there.
- It applies these rules in order (bar A2):

1. **ELASTIC.** If any row has `time_source` `"event"`, ELASTIC runs once on all rows of the match (they are its context) (`align_events_to_frames`, default `ElasticSyncParams`; bar A1). A matched `"event"` row takes the time of its `elastic_frame_id`, and its `time_source` becomes `"elastic"`. A match with no `"event"` row skips ELASTIC, which saves 30-63 s per gradientsports match.
2. **The corner kick (§4.4) and the penalty kick (§4.17).** A row of a corner type with `"placeholder"` or `"order_only"` and a kick found in its bracket by the rule of its class takes the kick time and the rest point as its start, and `time_source` becomes `"kick"`. A row whose end is defined as its start (an `"order_only"` restart, §4.2; a penalty row, §4.17) takes the rest point as its end too, so its end never stays at the old rule point (for an `"order_only"` corner whose prior side is wrong, that would be the other flag, 68 m away). A row with a first possession keeps its end, the reception. The scope of R34 is kept: every derived corner is tried. With the measured start the step sets the subtype again only for a corner with a first possession, that is, a row whose `time_source` was `"placeholder"` before rule 2 (the step reads it before it writes `"kick"`). An `"order_only"` corner has no first possession, so it keeps `corner_crossed` (§4.2), even with a kick and a goal of its team after it. For a corner with a first possession the rule of §4.2 applies: `corner_short` when the row's result is `success` (the first possession is the restart team's) and its end (the first possession's reception) lies within 15 m of the measured start, else `corner_crossed`. The same function as the converter's decides it, so the two cannot differ. The key of §4.18 holds the restart kind (`corner`), not `type_id`, so it is the same before and after the step, and the loader's gap check (§4.9 step 3) and the kick mapping find every row.
3. **Dribbles.** A `"derived"` dribble takes the midpoint of its neighbours, the `_add_dribbles` rule (`spadl/base.py:292`).
4. **Order.** The rows are sorted by `sort_actions_chronologically(..., by=("game_id", "period_id", "time_seconds"), tiebreak=("action_id",))` (`spadl/base.py:34`, the ADR-065 sorter) on the old `action_id`, and `action_id` is renumbered. At staging this is a fresh write, not a migration of stored tables.
   - Evidence for the sort (`spec_r2_checks.py` 6, 7 idsse matches): the retime leaves 16 inversions outside dribble pairs. The tracking supports the time order for 7 (both rows have an ELASTIC or release-yardstick time), the provider order for 0, and 9 have no tracking time on one row.
   - So the rule follows the tracking where it speaks. The 9 rows with no evidence are reported, not hidden.

**Staging.** `scripts/_loader_pining.py` calls the step once for each match with frames, after the frames are built and before the stoppage spans (§4.9). An events-only load does not call it (§4.10).

**Measured on idsse:**
- Aligned share: outgoing 0.987, minor 0.993, incoming 1.000.
- Shift: median 0.97 s; 8.0 % of rows move more than 3 s.
- Those large shifts are real (`shift_check.py`). Rows shifted by more than 3 s agree with the release yardstick for 0.741, and with `CalculatedFrame` for 0.897 (29 rows). For the same rows, the provider time agrees with the yardstick for 0.056.

**Other providers with frames.** The `"elastic"` accuracy is measured on idsse only.
- Staging calls the step for idsse, gradientsports and SkillCorner. Only idsse has `"event"` rows there.
- Metrica or kloppy actions with frames would also get `"elastic"`, with no measurement in this cycle. The docstring states this limit, and the records name it.

### 4.4 The SkillCorner corner kick (R34)

**The rule** (ported, bar A1; for a `"placeholder"` corner; an `"order_only"` corner has its own rule, below):
- The bracket is the row's gap: from the time of the previous non-derived row (`a`'s row; a SkillCorner row time is the possession's end, so this is `a.frame_end`) to the row's own `"placeholder"` or `"order_only"` time, which lies 0.02-0.03 s before `b`'s reception (§4.2). So it ends at `b.frame_start`, not at `b`'s row time (the end of `b`, `skillcorner.py:298`), and a rest inside `b`'s possession is never a kick. A period-end penalty's bracket ends at the period's last frame (§4.17). The gap is defined once, from a source that the step does not change: `restart_gaps(actions, frames) -> pd.DataFrame` in `tracking/_action_time.py`. It reads the table BEFORE any rule of the step changes a time, and the frames, and returns one row per derived restart or penalty row with an unmeasured time: the key of §4.18, `gap_start` and `gap_end`.
  - `gap_start`: the time of the previous non-derived row of the same period (`a.frame_end`). When the row's `b` is the first possession of its period, the period's first frame time from `frames`, so the gap never crosses a period break (258 "before" markers sit on a period's first possession on the cache, `derived_key_check.out`).
  - `gap_end`: the row's own unmeasured time plus its offset (0.02 or 0.03 s), which is `b`'s reception, read before the step. For a period-end penalty, the period's last frame time from `frames`.
  - The step computes it once at its start and takes each bracket from it. The loader computes it on the same pre-step table and frames, and checks each `"kick"` row of the stepped table against it by key (§4.9 step 3). Because the input is the pre-step table, the check can fail: a kick placed after `b.frame_start` is caught.
  - The measurement used the same gap for rows inside a period (`corner_kick_check.py:94-95`; `sc_penalty_check.py:139-140`). It skipped brackets that cross a period break (`corner_kick_check.py:98-100`, "unmapped"); the period-start rule above is new and has no measured share.
- **At rest:** the last run of at least 0.5 s of detected ball frames that stay within 0.5 m of the run's first point, with that point within 2 m of a corner flag.
- **Leaving:** within 1 s after the run, a detected ball frame is more than 2 m from the rest point.
- **The result:** the kick time is the time of the run's last frame. The start is the rest point, in the restart team's frame, and with it the side.
- No kick found: the row keeps `"placeholder"` or `"order_only"`.

**An `"order_only"` corner (change 2 of the supervisor's r7 read; bar A3).** Its bracket, from `a`'s end to the next reception, also holds the restart that follows (median 73.6 s; `order_only_corner_check.py`). The last rest run at a flag can then be a following throw-in or free kick near the flag. So the rule for this class is different:
- The kick is the **first** rest-and-release run of the bracket, and only if that run starts within 2 m of a flag. The corner is the first restart after `a`'s end, so its rest comes before the follower's.
- That run must not be the following restart's. This holds when another rest-and-release run (any place) starts after the flag run's release and before the reception: the follower's own rest and release come after it, and the bracket ends at the follower's reception. It also holds when the follower cannot rest at a flag: a goal kick (the rule point is 25 m or more from a flag), a kick-off after a goal, or a penalty.
- A row that fails keeps `"order_only"`.
- Evidence on the 690 order-only corners (all have cached tracking): a flag run is seen in 278 (one in 218, two or more in 60). The rule gives a kick for 189: free kick follows 72, goal kick 76, throw-in 32, goal 8, penalty 1. 89 rows with a flag run fail it and keep `"order_only"`. In 242 of the 278, the first run of the bracket is the flag run.
- There is no reference time for this class, so its accuracy is not measured; the records say so. The paired rule's accuracy (0.800 within 0.5 s of the GS provider time) comes from paired corners only: `corner_kick_check.py` built its brackets from "before" markers.

**Where the four numbers come from (bar A13).**
- The rest (0.5 m, 0.5 s): the TF-46 design §6.4 on `origin/tf-46-arm1-clean`, and the supervisor's `restart_release_reach.py`.
- The release (2 m within 1 s): `restart_release_reach.py`.
- The flag radius (2 m): the author's choice, inside the TF-46 bound (a rest within 1.5 m of the 1 m arc).
- Nothing comes from the course datasets.

**Accuracy against a reference that is not the rule (bar A4).**
- The only reference with a corner time is the gradientsports provider time (D3: no defect shown there).
- On 559 GS corners the rule finds a kick for 0.268, within 0.5 s of the provider time for 0.800. 0.200 are more than 0.5 s off, and the median is -0.20 s (`spec_r2_checks.py` 3). The supervisor's run: 0.734. These are the research script's rule (`corner_kick_check.rest_release`: the last 0.5 s window at a flag). The library rule `_find_kick` (the last maximal rest run with a release at a flag), on the same 559 corners and brackets: a kick for 147 (0.263); within 0.5 s of the provider time for 0.755, 0.245 more than 0.5 s off, median -0.27 s. Where both rules find a kick (146), they agree within 0.5 s for all and within 0.1 s for 0.829 (`kick_rule_compare.py`, reviewer B RTR-IMPL-03, owner ruling 2026-11-19: measure).
- On the idsse stage the rule finds 1 of 57 corners, so idsse cannot check it.

**Why a ported rule and not a library engine (bar A1, against the same reference):**

| Source | Corners on which both it and the rule find a kick | Within 0.5 s of the provider time: the rule | Within 0.5 s: the engine |
|---|---|---|---|
| TF-79 contact | 127 | 0.811 | 0.543 |
| TF-79 resolved contact | 65 | 0.831 | 0.600 |
| ELASTIC candidate | 58 | 0.810 | 0.621 |

- Over all GS corners, the engines are within 0.5 s for 0.507-0.552 of the corners where they find a kick (medians -0.32 to -0.40 s).
- On SkillCorner, the engines stay as the check on the repaired stage.

**Detection.** The rule counts detected ball frames only, through the existing seam (`detected_mask`, ADR-109).
- Today `tracking/skillcorner.py:268` writes the ball `visibility` as `None`.
- The ball `is_detected` will be carried into the ball `visibility` by the SkillCorner frame adapter and by the bronze builder `_skillcorner_bronze` (`scripts/_loader_pining.py:856`), on both input paths of `build_skillcorner_frames` (`:922`). The feed already holds the flag. No other repository is touched.
- Measured (`corner_kick_nodet.py`): with undetected frames, a kick is "seen" for 49.3 % instead of 39.4 % of SkillCorner corners. The extra 94 rest on extrapolated ball positions.
- Effect on the other readers (command 3, `reader-sweep.md`): no current reader takes the detection of a ball row.
  - `coordination/_signals.py:784-785` builds the mask over all rows, but it uses it on player rows only (`:336`, `:434`).
  - The ghost models, `gkdv/_engine.py:311-317`, `tracking/_gk_geometry.py:159, 214` and `tracking/utils.py:1386-1389` read keeper rows.
  - So no SkillCorner feature value and no model input changes, except through the new kick rule.

**The taker:** D4.

### 4.5 The gradientsports corner (bar A12, R37)

No defect is shown.
- At the kick that the ball track shows, the taker's track is a median 8.3 m from the ball, and within 2 m for 14.8 % of 169 corners (supervisor, `spec_checks_stage.py`). So the release yardstick, which needs the ball within 2 m of the actor, cannot see that kick.
- The ball-only rule puts the provider time within 0.5 s of the kick for 0.734-0.800 (median -0.20 s).
- Nothing is repaired. The GS corner rows keep `"tracking"`. The open-play times are not touched. R37 is closed (§2.1).

### 4.6 The source of each time, with evidence (bar A4)

| Provider | Row class | `time_source` | Measured accuracy | Script |
|---|---|---|---|---|
| idsse | on-ball rows, aligned | `"elastic"` | 95.5 % of the 155 shots within 0.5 s of `CalculatedFrame` (92.9 % within 0.2 s); 0.94-1.00 agreement with the release yardstick per family | `elastic_check.py` |
| idsse | on-ball rows, not aligned (by family: corner 14.0 %, goal kick 10.4 %, free kick 7.6 %, throw-in 4.9 %, shot 1.2 %, cross 0.9 %, clearance 0.6 %, pass 0.5 %; `elastic_summary.out`); every row of a match without tracking | `"event"` | 19-31 % within 0.5 s of the release per family; 21.9 % of the shots within 0.5 s of `CalculatedFrame` | `release_check.py`, `elastic_check.py` |
| gradientsports | native rows | `"tracking"` | Pass, cross, shot: 0.83-0.89 within 0.5 s of the release; clearance 0.70; free kick 0.797; goal kick 0.760. Corner: 0.734-0.800 within 0.5 s of the ball-only kick (§4.5). Throw-in: not measured, because the release yardstick is not shown valid for a ball thrown from the hands. | `release_check.py`, `spec_r2_checks.py` |
| SkillCorner | native rows (the receiver's own action) | `"tracking"` | 0.84-0.96 within 0.5 s of the release, all families | `release_check.py` |
| SkillCorner | derived corner rows, kick seen (0.394 of the corners) | `"kick"` | The library rule against the GS provider time: 0.755 within 0.5 s (147 of 559 corners found), 0.245 more than 0.5 s off, median -0.27 s (the research script's rule: 0.800). On SkillCorner: the ELASTIC candidate agreement is reported on the repaired stage. | `corner_kick_check.py`, `spec_r2_checks.py`, `kick_rule_compare.py` |
| SkillCorner | derived restart rows with a first possession, no kick | `"placeholder"` | An upper bound of the kick. It lies a median 2.0-2.2 s after the release where the release is seen. | supervisor's `restart_release_reach.py` |
| SkillCorner | derived "after"-only restart rows, no kick | `"order_only"` | Not a moment of the restart: it lies after the next stoppage. The nearest earlier set-piece row is at least 16.99 s before it (median 136.8 s; 3,274 rows). | `order_only_overlap.py` |
| all | synthetic dribbles | `"derived"` | the midpoint of the neighbours, by construction | — |
| statsbomb, opta, wyscout, metrica, kloppy | all rows | `"event"` | Not measured: no stage frames in this cycle. No reader in this design relies on their accuracy. | — |

### 4.7 The restart-coordinate enrichment on a derived row

`add_restart_coordinates` (`spadl/utils.py:772`) wraps `resolve_restart_geometry` (`tracking/_gk_geometry.py:339`). Its origin tiers start with `"native"` when the start is finite. On a derived restart row the engine reads `time_source`, so a rule point never reads as native and a measured rest point never reads as a prior:
- **`"placeholder"` or `"order_only"`, corner, goal kick or throw-in:** `enriched_start_x/y` is the row's own start (the converter's rule point of §4.2), with origin source `"restart_prior"` and the existing confidence of its type (`_PRIOR_CONF`). The tier helpers `_side_y` and `_throwin_x` (`:317-336`) are not used for a derived row: they take the side from the reception (`end_y`, right for 0.634) and the throw-in x from the next event.
- **A restart with a NaN start (§4.2: it follows an "after"-only restart):** origin source `"unresolved"`, `enriched_start_x/y` NaN, whatever its type. No tier puts a place on it.
- **`"placeholder"` or `"order_only"`, free kick:** origin source `"unresolved"`, `enriched_start_x/y` NaN. Its start (the previous possession's end) is not a spot that the enrichment vouches for.
- The tracking tiers are skipped for a `"placeholder"` or `"order_only"` row, because its linked frame is a reception frame, not the spot. This also skips the keeper-detection window of `:184-235` (±1 s around the row time).
- **A derived `shot_penalty` row with `"placeholder"` or `"order_only"`:** `enriched_start_x/y` is its own start (`_PENALTY_SPOT`, `_gk_geometry.py:33`), with origin source `"restart_prior"` and `_PRIOR_CONF` of the penalty (`:40`).
- **`"kick"`:** `enriched_start_x/y` is the row's own start (the rest point), with origin source `"tracking_ball"` and `_CONF_TRACKING_BALL`.
- **The destination** (the first possession's reception) keeps the existing tiers. Every other row is unchanged.

### 4.8 The wheel xCross and xShot ship the public arm (R38, R49)

- **Today.** The xCross trainer has the switch: `--ship-variant` (`scripts/train_xcross_attempt.py:523-530`). It needs the two-arm corpus (`:818-822`) and overrides the paired verdict, which it keeps only in the `why` text (`:1085-1088`). The xShot trainer has no switch (`scripts/train_xshot_occurrence.py:912-914`); its `_select_work` (`:883-930`) is the same shape as the xCross one. `scripts/hf_jobs/campaign.py` passes no trainer option from a campaign file: `load_campaign` (`:143-161`) and `instances` (`:240-263`).
- **The change.**
  - `CampaignJob` gains `ship_variant: str | None = None`, read from a `[[jobs]]` entry by `load_campaign`.
  - The value is refused unless the job's trainer takes `--ship-variant` (a registry flag on the `xcross_attempt` and `xshot_occurrence` `TrainerSpec`s) and the value is one of the job's `arms`.
  - `instances` appends `("--ship-variant", value)` after `--arms-json`.
  - The xShot trainer gets the xCross switch: the `ship_variant` `Param` (as `train_xcross_attempt.py:523-530`), the refusal without the two-arm corpus (as `:818-822`), the value in the config and in the prep token inputs (as `:845`, `:1361`), and the override in `_select_work` (as `:1085-1088`).
  - Both trainers record the set-aside verdict as a field: `candidates["paired"]["rule_verdict"]` is the paired rule's arm and `candidates["paired"]["rule_why"]` its reason; `shipped` and `why` stay as today. `adopt` reads both from the staged training report and writes them into the per-entry line of its adoption draft (`_changelog_draft`, `scripts/hf_jobs/adopt.py:1135-1144`) for each entry where `rule_verdict` differs from `shipped` (bar A17). The plan names the report path that `adopt` reads.
  - The two wheel jobs of `scripts/hf_jobs/campaigns/tf46_refit.toml` (xShot `:55-61`, xCross `:71-78`) get `ship_variant = "public"`, and their comments (`:60`, `:77`) cite R49 and R38. The Hub jobs (`:63-69`, `:80-87`) are unchanged.
- **Size.** About 15 lines in `campaign.py` and `registry.py`, two lines in the campaign file, about 20 lines in `train_xshot_occurrence.py`, 2 lines in `train_xcross_attempt.py`, and about 10 lines in `adopt.py`. Tests: 4 in `tests/scripts/hf_jobs/test_campaign_arms.py` (parse, refuse a trainer without the option, refuse a value outside `arms`, the argument in the instance); for xShot, the override and the refusal without the two-arm corpus, in the xShot trainer test file that holds the `_select_work` tests (as the xCross ones); the `rule_verdict` field in both trainers; the adoption-record line.
- **Gates (bar A17).** The forced arm is the shipped arm, so the ADR-131/132 gates apply to it unchanged on the repaired stage (corpus-widening spec §5.2, §5.7): each provider's held-out skill and the size gate. This holds for xShot and xCross alike.
  - `verify` judges it as today (`refit_fresh_acceptance`).
  - A failed gate fails the run, with no fallback to the other arm. It is a STOP for the owner.
  - The paired verdict that the override set aside is reported in the adoption record.
- **Manifest.** The 2 wheel xCross and the 2 wheel xShot entries get the 7.1 values of the `public` arm at the expectation edit: `shipped_arm` `public`, certification `public`, corpus `public_ids`, and the `expect` keys as the run records them. The plan checks that `adopt._expectation_edit` (`scripts/hf_jobs/adopt.py:937`) accepts that edit. If it does not, the plan stops and reports.

### 4.9 The SkillCorner stoppage spans (R44)

The kick reaches the spans as data from the step. The step stays the only kick source (bar A2).
- **The call order in `_build_match`** (`scripts/_loader_pining.py:632-635`):
  1. `_build_skillcorner` builds the actions and the frames.
  2. `sync_action_times(actions, frames)` sets the kick rows.
  3. The loader maps each `"kick"` row to its gap and its side, with the key of §4.18: `kicks: Mapping[tuple[str, str], tuple[int, str]]`, keyed by `(canonical_id(original_event_id), kind)`, with the value `(kick frame, side)`. The loader passes the side, so the spans never infer it. The row's gap is (the previous possession of `b`, `b`), or (`a`, the period end) for a period-end penalty. The side is `"before"` when `b` carries a "before" marker of the row's kind (paired or "before"-only), `"period_end"` for a period-end penalty, else `"after"` (an "after"-only restart). A penalty kick (§4.17) is mapped the same way. The value is the stage frame whose `(period_id, time_seconds)` equals the kick time; the kick time is a frame time by construction (the rest run's last frame).
     - Two rows never share a key (§4.18), so one possession can be `b` of one gap and `a` of the next and keep both kicks. Example: a corner whose first possession `p1` ends in a foul and a penalty is the corner `(p1, corner)` in gap (`p0`, `p1`), and the penalty `(p2, penalty)` in gap (`p1`, `p2`).
     - The step finds a kick only inside the row's gap. The loader checks each kick against `restart_gaps` of the pre-step table (§4.4). A kick outside its gap is a defect, not a case: the loader asserts it and counts it, and the §8 count must be 0.
     - A period-end penalty's gap is `[a.frame_end, the period's last frame]`. Its kick is keyed on `a`'s id (§4.18) and carries the side `"period_end"` from the loader: `unknown` from `k` to the period's last frame. The spans take the side from the mapping and do not infer it from the possessions, so a paired or "before"-only penalty whose `b` is the last possession of its period keeps `live` from `k` (A ATR-SPEC-14). The next period's first possession is the `b` of a period-end penalty only for the result rule of §4.17.
  4. `skillcorner_stoppage_spans(events, kicks=kicks)` builds the spans.
  5. `with_stoppage_state(frames, spans=spans)` runs as today.
- idsse calls the step before `with_stoppage_state(frames, spans=None)`; the order does not matter there. Gradient Sports has no `"event"` row, so the step changes nothing, and `event_stoppage_spans(actions)` runs as today.
- **The span state of each class**, for the pair of possessions `a`, `b` around a marker (`k` = the kick frame):

One rule per marked pair, with `k_after` and `k_before` the kicks of the pair (each optional):
- Before the first kick: `dead` from `a.frame_end` when both possessions carry a marker, `unknown` when one does (today's two cases, `_stoppages.py:97-104`).
- From an `"after"` kick: `unknown` until the next kick of the pair or `b.frame_start`. The span holds the restart's flight, the next stoppage and the next restart, so no frame in it is `live` (ADR-130).
- From a `"before"` kick: `live` (the flight and `b`).
- With no `"before"` kick: `b` is `live` from `b.frame_start`.
- Two kicks in one pair need `k_after < k_before`; otherwise the `"after"` kick is dropped and counted.

| Pair | Today | R44 |
|---|---|---|
| Both markers, no kick | `dead` `[a.frame_end, b.frame_start)`; `unknown` `[b.frame_start, b.frame_end]` | `dead` `[a.frame_end, b.frame_start)`; `b` `live` |
| Both markers, a `"before"` kick | as above | `dead` `[a.frame_end, k)`; `live` from `k` |
| Both markers, an `"after"` kick | as above (`_stoppages.py:95-99` reads the presence of a marker, not its kind) | `dead` `[a.frame_end, k)`; `unknown` `[k, b.frame_start)`; `b` `live` |
| Both markers, both kicks | as above | `dead` `[a.frame_end, k_after)`; `unknown` `[k_after, k_before)`; `live` from `k_before` |
| One marker, no kick | `unknown` `[a.frame_end, b.frame_end]` | `unknown` `[a.frame_end, b.frame_start)`; `b` `live` |
| One marker, a `"before"` kick (a "before"-only restart or penalty) | as above | `unknown` `[a.frame_end, k)`; `live` from `k` |
| One marker, an `"after"` kick (an "after"-only restart whose `b` has no marker) | as above | `unknown` `[a.frame_end, b.frame_start)`; `b` `live` |
| Lost marker (rule 3, `_lost_marker_spans`) | as today | unchanged |

- The ball-out and period-edge rules are unchanged. They can still mark frames inside a `live` span `dead`.
- `b` is `live` for every marked pair, not only for the four restart types: a goal's kick-off possession is in play too. The 2.48 % of R44 counts the four restart types; the plan reports the total share.
- **Size.** `skillcorner_stoppage_spans` gains `kicks: Mapping[tuple[str, str], tuple[int, str]] | None = None` (about 25 lines). The loader call order and the pair mapping take about 20 lines. There are 10 tests: one per row of the table, the shared possession (a `"before"` kick and an `"after"` kick on one possession, in two pairs), two kicks in one pair, a reversed pair of kicks, and a kick outside its gap. The no-kick case of an "after"-only restart stays as "both markers, no kick": `dead` holds the first restart's play too, which drops those frames from training and adds no wrong `live` frame. The `skillcorner_stoppage_spans` docstring says so.

### 4.10 The events-only actions

`_build_match_actions` states that "the events-only actions are the full build's actions by construction (and test-pinned byte-identical)" (`scripts/_loader_pining.py:646-651`; `tests/scripts/test_loader_pining_events_only.py:10-11`). With the step this holds before the step only.
- **The new property:** the events-only actions equal the full build's actions before `sync_action_times`. The full build's actions equal the step applied to them. A test pins both.
- **What differs after the step:** idsse `time_seconds`, `time_source` and `action_id` (rules 1, 3 and 4); the SkillCorner `"kick"` corner rows (time, start and order).
- **The consumers of the events-only path** (`events_only_loader`, `scripts/_events_admission.py:144`): the xT fits (`_xt_corpus.py`), `_pass_completion_pining.py`, the TF-56 and TF-60 drivers, the calibrations, and `build_skillcorner_s1_event_validity.py`. None of them feeds this release: the campaign jobs train on the staged datasets, and `pass_completion` trains on `sb_open`. Their rows carry `time_source`, so a reader can tell a provider-clock time.
- The docstring and the test text are corrected in the repair commit (§7). The docstring names the key of §4.18 as the stable key between the two builds: `(game_id, original_event_id, action_provenance, kind)`. A row with no `original_event_id` (a keeper save or a synthetic dribble) is identified by the key of the previous keyed row and its offset after it (§4.18).

### 4.11 The period in the score and phase lookups (R51)

- **The callback.** Both lookups return `Callable[[Any, int, float], ...]`: `(game_id, period_id, time_seconds)`. No form takes the clock time alone, so a caller that gives no period fails at the call (bar A18, point 2c).
- **`_build_phase_lookup`.** The set-piece rows are grouped by `(game_id, period_id)`. A query reads its own period's rows only, so a window never crosses a period end. The 10 s decay, the types, the throw-in rule and the R50 order-only drop are unchanged.
- **`_build_score_lookup`.** The goals are sorted by `(period_id, time_seconds)`, the match order of `win_probability/_state.py:48`. The value at `(p, t)` counts each goal with `period_id < p`, or with `period_id == p` and `time_seconds <= t`. The goal rule (which rows are goals) is R52 (§4.12); the own-goal flip is unchanged.
- **The callers** pass the frame's (or the spell's) `period_id`:
  - ghost-GK fit and serve: `_ghost_gk.py:1094`, `:1101` (the per-frame loop, grouped by `(game_id, period_id, frame_id)`);
  - ghost-outfield: `_ghost_outfield.py:252-254`;
  - xCross fit and serve: `_xcross_attempt.py:498`, `:980` (the group's `pid`);
  - causal: `causal/opportunities.py:338` (the spell's `per`).
  - The builders at `_ghost_gk.py:1270-1271`, `:2749-2750`, `_ghost_outfield.py:343-348`, `_xcross_attempt.py:448`, `:958` and `causal/opportunities.py:257` are unchanged; `scripts/train_ghost_gk.py:446-457` passes the whole game's actions as today.
- **Reuse (bar A1, A18 point 2d).** The score lookup takes the match order of `derive_match_states` (`win_probability/_state.py:48-51`), but not the function. Reasons:
  - It returns the PRE-action state of each action row. The ghost and xCross extractors need the value at a frame time, between action rows.
  - Its goal rule is now the shared predicate of R52 (§4.12), so the goal rule is reused through `spadl/_goals.py`, not through this function.
  - It needs a home-team map and `WinProbabilityParams`, and it also derives cards. None of them is an input of these features.
- **Extra time.** Periods 3 and 4 follow the same rule: a goal of periods 1-3 counts at period 4. A period-5 (shoot-out) row is not a goal for the score (R52, §4.12).
- **What a bundle fitted before the repair gets at the serve.** It loads with a warning (R55, §4.14) and gets the repaired `phase` and `score_diff`. All 15 artifacts that read the lookups are retrained in this cycle (§6).
- **The change is not byte-identical.** `phase` and `score_diff` change on every provider with more than one period in a table. The supervisor's measure on the 187 stage matches: `phase` differs for 8.25 % (idsse), 8.37 % (gradientsports) and 5.30 % (SkillCorner) of the match seconds; `score_diff` for 37.0 %, 37.2 % and 39.9 %.
- **Size.** About 30 lines in `_ghost_gk.py`; one argument at each of the five call sites.

### 4.12 The goal rule of the score lookup (R52)

- **One predicate.** `goal_mask(actions) -> pd.Series` and `owngoal_mask(actions) -> pd.Series` move from `vaep/labels.py:18-33` (`_is_goal`, `_is_owngoal`) into one module, `silly_kicks/spadl/_goals.py`, with the shot-type set (`shot`, `shot_penalty`, `shot_freekick`). A goal is a successful row of a shot type; an own goal is a row with the `owngoal` result, on any type.
  - They read `type_id` when the column is present, else `type_name`. Both give the same rows on a table that has both.
  - Readers: `vaep/labels.py` (its 8 label functions, unchanged in result), `win_probability/_state.py:18-20`, `:58-59` (`_SHOT_TYPE_IDS`, `_SUCCESS`, `_OWNGOAL` give way to the two masks), and `_build_score_lookup` (`_ghost_gk.py:557-569`). No module keeps its own copy.
- **`_build_score_lookup`.** The goals are the rows of `goal_mask | owngoal_mask` (the goal-row path; on SkillCorner, R56 reads the running score first, §4.15). The scoring team of an own goal is the other team, as today (`:578-592`). The R51 period rule applies (§4.11).
- **The shoot-out (period 5).** A row in period 5 is not a goal for the score: a shoot-out kick does not change the match score, and `win_probability` already excludes period 5 (`_state.py:25`, "excluded upstream"). A frame in period 5 takes the score at the end of period 4.
  - **The gradientsports shoot-out is period 5.** The feed stores it in period 4, after the `END` game event of period 4: 41 `shot_penalty` rows in 5 matches (10506, 10508, 10510, 10511, 10517), with no tracking (10517 also has one in-match penalty in period 4 at 725 s, before the `END`). The converter gives the rows after that `END` period 5 when a penalty kick follows it, with the time from the first shoot-out event and no location (R89, R93): the feed gives no position or one stopped ball position for these rows. A caller's own period 5 also gets no location. The stage of commit 1 holds these rows in period 4; the stage of the amended commit 2 is built again (R94, R95).
- **The change is not byte-identical.** On the gradientsports stage the lookup counts 145 of 194 goal rows today; after R52 it counts all goal rows of the rule outside period 5 (supervisor's check). idsse and SkillCorner have no such rows on the stage. The 15 artifacts of R51 are retrained anyway (§6).
- **Size.** One new module of about 25 lines; about 10 lines less in `vaep/labels.py` and `_state.py`; about 10 lines in `_ghost_gk.py`.

### 4.13 The action-window readers and an `"order_only"` row (R54)

Each reader reads `is_order_only` (§4.1). On a table with no `"order_only"` row, or without `time_source`, each gives today's values (bar A21, point 4). Command 4 found six; the R58 sweep found a seventh, the coordination windows, which command 4 did not list because they name no window parameter.

| Reader | What it gives for an `"order_only"` row |
|---|---|
| VAEP `scores` / `concedes`, `window="time"` (`vaep/labels.py:377-452`, `:454-500`) | The row's label is `<NA>`: it opens no window. In `"time"` mode the label column is always nullable `boolean` (with `xg_column`: `float`, NaN for the row), so the dtype does not depend on the data. The row is no goal and no own goal, so it changes no other row's label. The `"action"` and `"possession"` modes are unchanged. |
| `retains` (`xtgk/_retention_labels.py:43-95`) | NaN, the existing value for an unknown label (`:93`). |
| `EmpiricalTurnoverValue` with a finite `window_seconds` (`xtgk/_turnover.py:249-275`) | The row is not a turnover anchor, so it adds nothing to the fit's support. The possession-bound scope (`window_seconds=None`) opens no time window and is unchanged. |
| TF-49 packing, the secured window (`tracking/_packing.py:317-373`) | `packing_secured` is `<NA>` for a pass whose reception row is an `"order_only"` row. The column is already nullable `boolean`. |
| Team metrics: spells and recovery windows (`team_metrics/_possession.py:73-98`, `_compute.py:121-160`, `_pressing.py:126-170`) | A spell's `start_time` and `end_time` skip its `"order_only"` rows. A spell of `"order_only"` rows only has `start_time`, `end_time` and `duration_s` NaN and is not a recovery, so it anchors no post-recovery or counterpress window. |
| The possession carve-out `brief_window_seconds` (`spadl/utils.py:941-975`, and the atomic twin) | No time test reads an `"order_only"` row at either end (`i` or `i + k`): such a pair is never a brief opposing action, so its boundary stays. |
| Coordination windows from actions (`coordination/_windows.py:212-300`, `_event_intervals` `:474-498`) | Found by the R58 sweep (command 6), under the R54 rule. A possession window does not start or end at an `"order_only"` row: it takes the time of its next or previous other row; a possession of `"order_only"` rows only opens no window. `_event_intervals` reads no `"order_only"` row as an interval end; the restart interval runs from the previous other row to the next other row. A goal interval of a shot row with `is_time_unmeasured` is not opened. |

- **The VAEP dtype change** (bar A21, point 3): `scores` and `concedes` in `"time"` mode return nullable `boolean`, not `bool`, on every table. The values on a table with no `"order_only"` row are today's. No caller in the repository or in the lakehouse uses that mode (supervisor's search: the lakehouse uses `EmpiricalTurnoverValue` only, `src/ingestion/xt_gk_v2_writer.py:221`, `scripts/train_xt_gk_v2_hf.py:189`). The records name the change and its reach.
- **Size.** About 5-15 lines in each of the seven modules.

### 4.14 The warning for a bundle fitted before the score and phase repairs (R55)

- **Today** (`tracking/_feature_contract.py:198-260`): `verify_feature_contract` compares the declared constants first. A constant on both sides that differs raises (`:245-253`, unless `legacy_override`). A constant that the artifact has and the library lacks warns `UnverifiableFeatureContractWarning` (`:228-235`). A constant that the library declares and the artifact lacks gives nothing.
- **The change.**
  - The ghost-GK, ghost-outfield and xCross contract blocks (`_ghost_gk.py:1809-1840`, `_ghost_outfield.py:549`, `_xcross_attempt.py:589`) declare one new constant, `match_context_revision = 1.0`. It names the form of the score and phase inputs: per period (R51), the full goal rule (R52), the SkillCorner running score (R56) and the penalty rows (R57, R58). One constant covers all of them (bar A22, point 4).
  - `verify_feature_contract` gets the missing branch: a constant that the library declares and the artifact lacks warns a new category, `PredatesDeclaredConstantWarning` (in `tracking/_warnings.py`; separate, not a subclass of another). The message names the constant and its meaning from a small registry in `_feature_contract.py` (for `match_context_revision`: "the score and phase inputs are computed per period, with the full goal rule, the SkillCorner running score and the penalty rows, in every release after silly-kicks 4.144.0; this artifact was fitted with silly-kicks 4.144.0 or earlier"). The text is relative to 4.144.0, the last release without the repair, so it needs no number of the release that ships it and no placeholder can ship; the commit-prep check of plan line 192 confirms that 4.144.0 is the last release before this one. It does not raise.
  - A bundle fitted in this cycle stores the constant, so it loads with no warning. A later change of the inputs bumps the value; a stored value that differs then raises by the existing rule, which is the ADR-050 behaviour for a changed declared constant.
  - The declared-constant registries take the new key: a third registry, `REVISION_CONTRACT_KEYS` in `_feature_contract.py` (hand-bumped version tags mapped to their value; the three contract blocks stamp from it), and the value pin of `tests/tracking/test_declared_constant_values.py` (each live block, the ghost-outfield block included, and each stamped artifact that declares the key has the registry's value). Not the geometry module-constant map (`_feature_contract.py:66-107`): its gate (`tests/tracking/test_geometry_constant_enumeration.py`) requires probe-movable geometry constants of the extractor modules, and a revision tag is neither; the two geometry registries and their gates stay as they are. `_unaccounted_keys` takes the union of the three registries. The per-model key-set pins keep their geometry sets net of the revision keys, and assert the revision keys per model (xShot: none). Two honesty tests: each revision key is stamped by a model; no key is in two registries.
  - The gate (R67, bar A29): `tests/tracking/test_match_context_revision_gate.py` runs `_build_score_lookup` and `_build_phase_lookup` (the builders of every score and phase input of the three models) on a fixed probe that exercises each rule the tag names (the period key; the goal rule with a penalty, a direct free kick, an own goal and a shoot-out row; the running score with its effect rule, a derived row and its fallback; the phase codes, the 10 s window and an `"order_only"` row) and fails when the outputs differ from the pin of the current revision; its message says to raise the tag, add the pin and refit. Mutation teeth: one code change per rule turns it red. The limit that stays: a change that the probe does not exercise is not detected.
- **Reach of the new branch.** It is generic: any artifact that lacks any constant that the library declares now warns. The plan checks every bundled artifact of the wheel and every manifest Hub entry on the repaired code. Only the pre-repair ghost-GK, ghost-outfield and xCross bundles may warn; any other warning is a STOP.
- **Size.** About 40 lines in `_feature_contract.py` (the branch, the meanings and the revision registry), 20 in `_warnings.py`, one line in each of the three contract blocks, and the registry gates in two test files.

### 4.15 The SkillCorner score from the running score (R56)

- **The carrier (bar A20, point 1).** `spadl/skillcorner.py` `convert_to_actions` takes `preserve_native: bool | list[str] = False`. `True` adds no column, because `original_event_id` is a schema column (R62, owner 2026-11-18: before, `True` raised the extra-columns overlap `ValueError`). A list keeps `original_event_id` and the listed event fields, validated by `_validate_preserve_native` (`spadl/utils.py:1772-1821`) and kept by `_finalize_output(extra_columns=...)` (`:1672-1706`). The loader's SkillCorner build asks for `["team_score", "opponent_team_score"]`. No `SPADL_COLUMNS` change, no new staged table. The reflection registry treats both as `invariant` with no warning (not geometry-shaped, `reflection.py:269`), and `convert_to_atomic(preserve_native=...)` carries them.
- **The lookup (point 2).** Where both columns are present, `_build_score_lookup` reads them per match:
  - native rows only (`action_provenance == "native"`), where the row team is the event team; a derived row is not read;
  - each value to the home and away side by `ids_match` (ADR-019);
  - per side, the score never decreases in the match. A match that fails, and a table without the columns, use the goal rows (R52). The two sources are never added.
- **The effect rule (point 3).** The new score takes effect at the row time of the native row after the first native row whose score is higher. A SkillCorner row time is the possession's end. Measured on 2,358 goals with a goal marker (`sc_score_lag.py`): the higher score first shows at the scoring possession for 1,325, at the kick-off possession for 1,029, and two possessions early for 4. The effect lies 64.0 / 83.9 / 127.2 s (quantiles 0.1 / 0.5 / 0.9) after the scoring possession's end, and 0.0 / 1.5 / 6.2 s after the kick-off reception. The docstring states the rule and the lag.
- **The R51 period rule and the R52 shoot-out rule apply** to both paths. The R55 constant `match_context_revision` covers this repair (§4.14).
- **Size.** About 15 lines in the converter, 1 in the loader, 30-40 in the lookup.

### 4.16 The idsse penalty (R57)

- **Today.** The DFL `Penalty` event is in neither `_MAPPED_EVENT_TYPES` (`spadl/sportec.py:150-165`) nor `_EXCLUDED_EVENT_TYPES` (`:176-191`), so it is dropped with the "unrecognized event types" warning (`:703-713`). The bronze row already holds the nested shot: the parser flattens a nested `ShotAtGoal` into `shot_*` columns (`providers/sportec/parse.py:241-252`). The 7 stage matches hold 3 penalties (J03WMX, J03WOH, J03WR9), each scored, at the mark (`X-Position="94.00" Y-Position="34.00"`).
- **The change (bar A23, points 1-2).** In `spadl/sportec.py`: `"Penalty"` joins `_MAPPED_EVENT_TYPES`; the shot mask (`:643`, `:947`) also takes a `"Penalty"` row that carries a nested shot (`shot_outcome_type` present), and the existing penalty path (`:950-955`) types it `shot_penalty`; a `"Penalty"` row with no nested shot is excluded by name.
- **The parser (R63, bar A25).** The build found that the parser does not flatten the outcome of the nested shot: it reads the children of `Penalty` (so `ShotAtGoal` gives the `shot_*` columns), not the grandchild outcome tag `Penalty > ShotAtGoal > SuccessfulShot`. So `shot_outcome_type` is empty on every parsed `Penalty` row. The owner's ruling R63 (option a) lifts bar A9 for one change: in the nested-child loop of `_build_event_row` (`providers/sportec/parse.py:866-875`), the outcome tag under `Penalty > ShotAtGoal` gets the rule that the loop has one level up for a `ShotAtGoal` event. It fills `shot_outcome_type` (`_SHOT_OUTCOME_NAMES`) and the outcome tag's attributes as `shot_outcome_*` (the `_NESTED_PREFIX_MAP` prefix of the outcome tags). No other line of the event block changes; its time rule is not touched.
  - This changes the bronze (a parser output) on `Penalty` rows only: the 3 stage penalties gain `shot_outcome_type` and their `shot_outcome_*` attributes. The supervisor checks, on a snapshot, that the bronze of the 7 idsse stage matches is equal in every other cell (bar A25, point 4).
  - The collision: the `ragnarok` working tree rewrites the same loop on a snapshot of the nested tags; its rebase must carry this rule. The records (the new ADR and the CHANGELOG) name both.
  - Who else takes the parser's output: every caller of `parse_dfl_events`. The luxury-lakehouse builds `bronze.idsse_events` from it (`src/ingestion/idsse.py:783-786`, delete-and-depend, ADR-031 T3), so the `Penalty` rows that it parses after its next upgrade of the library gain the outcome; its stored rows stay as they are (the next bullet). ADR-031 gets an amendment line: the port now differs from the lakehouse parser at `0efac60` on a `Penalty` with a nested shot.
  - Two side effects on a match with a `Penalty`: the column order of the bronze frame changes (the order follows the first row that has a key, so it depends on the data today too; the column set is unchanged), and `shot_outcome_assist_fouled_player` becomes `object` where it was an all-missing `string`. No reader in the repository reads the bronze by position or pins that dtype (search recorded in the ledger, V18).
- **Bronze from before R63 (bar A23 point 7).** A `Penalty` row with a nested shot (`shot_player` or `shot_team`) and no `shot_outcome_type` cannot be typed: it comes from a parser of silly-kicks 4.144.0 or earlier (the released 4.144.0 does not read the outcome), or from a provider file with no outcome tag. The warning names that literal, which needs no release number of its own. The converter excludes it, as a `Penalty` with no outcome, but never silently: a `UserWarning` names the event ids, both causes and the repair (parse the raw file again with `parse_dfl_events`). A `Penalty` with no nested shot is excluded by name with no warning. The CHANGELOG names the rule for callers that convert stored bronze (the lakehouse converts `bronze.idsse_events`, `src/ingestion/spadl_conversion.py:1287`).
- **The idsse fixture (R64, bar A26).** `tests/datasets/idsse/per_period_match.parquet` was the lakehouse production bronze pulled through Databricks (2026-04-29), so its `Penalty` row had no outcome. The owner's ruling R64 (option a) rebuilds it from the raw public DFL files of J03WMX (figshare doi:10.6084/m9.figshare.28196177, CC BY 4.0) by a new generator, `scripts/build_idsse_per_period_fixture.py` (`parse_dfl_match_info` + `parse_dfl_events`; no step after the parse, so `match_id` is the parser's `J03WMX`). `scripts/extract_provider_fixtures.py` refuses its idsse `per_period` path at the argument check. The README of `tests/datasets/idsse/` records the generator, the public file names with their size, md5 and sha256, the fetch path, and the sha256 of the built file; `NOTICE` cites the published source. The difference to the old file is named cell by cell (bar A26 point 4): `match_id` on every row, `_ingested_at`, the row order, 25 dtypes, 27 events moved from period 2 (negative times) to period 1, and the 5 outcome cells of the penalty. A test takes the fixture through `shape_events_to_native` and asserts the penalty action (`shot_penalty`, `success`, the home team, DFL-OBJ-J01BGM, the attacked end). `sample_match.parquet` and `paired_tracking.parquet` are not touched.
- **The player (point 3).** The shooter of the nested shot. The parsed `Penalty` row has an empty `player_id` (the native player resolution reads `play_player` and `foul_fouler` only), so the converter fills an empty `player_id` of a `Penalty` row from `shot_player`, the nested shot's `Player`. A non-empty `player_id` is never overridden (ADR-001).
- **The time (point 4).** As each idsse on-ball row: `"event"`, then `"elastic"` where ELASTIC aligns it (R32).
- **Size.** About 15-25 lines.

### 4.17 The SkillCorner penalty rows (R58)

The function is `infer_penalties(pp)` in `spadl/_skillcorner_inference.py`, next to `infer_restarts` (§4.2), with the same derived-row pattern.
- **The count rule (bar A24, point 1).** Every marker `penalty_for` or `penalty_against` on a `player_possession` is one penalty: the possession's `game_interruption_before`, or the previous possession's `game_interruption_after` in the same period. An "after" marker on `a` and a "before" marker of the same kind on `b` are one penalty. Unlike the four restarts, an "after" marker at a period end counts: a penalty is taken even when time is up (Law 14). Count on the 889 cached matches: 221 in 190 matches (219 with an "after" marker, 2 "before" only; supervisor, `sc_penalty_check.py`).
- **The row (point 2).**
  - Type `shot_penalty`; `bodypart_id` `foot`; `player_id` `<NA>` (R40: the data has no possession of the taker).
  - The team by the marker rule: `_for` is the marked possession's team, `_against` the other team. It is right for 173 of 173 score rises (against 8 of 8, for 165 of 165; `sc_penalty_check.out`), and the ball rests at the end that this team attacks for 161 of 161 kicks.
  - The start is the penalty mark of the attacked end, `_PENALTY_SPOT` (`tracking/_gk_geometry.py:33`), in the team's frame. The end is the start.
  - The result (`result_source` `"inferred"`):
    - A paired penalty (the next possession carries the penalty marker, so the ball stayed in play) is `fail`. The score at `b` can hold `b`'s own goal (§4.15: the higher score shows at the scoring possession for 1,325 of 2,358 goals); the cache has 6 paired penalties with a higher score at `b`, and in all 6 `b` ends in a shot by the taking team 1.7-2.5 s after the kick, a rebound goal (supervisor, `r9_checks.py`). The rebound row is the goal row; the penalty row is not a second one.
    - An "after"-only penalty is `success` when the running score of that team is higher at the next possession than at `a`, with no possession between; without the running-score fields, when the next possession carries a goal marker of that team. Else `fail`. For a penalty at a period end, the next possession is the first of the next period.
    - A penalty at the end of the last period with no later event (0 on the cache) is `success` when the team's final score in the metadata is higher than its running score at `a`, else `fail`.
  - `original_event_id` is that of `b`, the possession after the gap (as §4.2); a period-end penalty, with no `b` in its period, takes that of `a` (2 on the cache). `action_provenance` `"derived"`.
- **The time (point 3).**
  - `"kick"` where the tracking shows the rest and the release at the mark: the last run of at least 0.5 s of detected ball frames within 0.5 m of the run's first point, that point within 1.5 m of the mark, and a detected ball frame more than 2 m away within 1 s after the run. The kick time is the run's last frame; the start is the rest point. The last run is safe here because no other restart rests at the mark; a retaken penalty gives two runs, and the last is the one that counts.
  - The bracket is the row's gap, from the same `restart_gaps` as the corner (§4.4): from `a`'s row time (`a.frame_end`) to `b`'s reception; for a period-end penalty, to the period's last frame.
  - The numbers and their sources (bar A13): the rest (0.5 m, 0.5 s) and the release (2 m, 1 s) as for the corner (§4.4); the mark radius 1.5 m from the supervisor's `sc_penalty_check.py` (the rest lies a median 0.55 m from the mark).
  - No kick: `"placeholder"` (paired: the next reception minus 0.02 s) or `"order_only"` ("after" only: the next reception minus 0.03 s; at a period end: the previous row's time plus 0.01 s).
- **Accuracy (point 4).** Reference: the gradientsports penalties of periods 1 and 2, with the provider time. The rule finds 21 of 22; rule time minus provider time: median -0.23 s, 81.8 % within 0.5 s, 90.9 % within 1.0 s (supervisor, `gs_penalty_check.py`). The period-4 shoot-out kicks there (41 rows; §4.12) are no test, because one bracket holds many kicks; the one in-match penalty in period 4 (10517, 725 s) is not in the 22. On the SkillCorner cache the rule finds a kick for 161 of 221 (72.9 %); the ball then moves toward the goal for 99.4 %, and the defending keeper is within 1.5 m of the line for 156 of 161. The share on the repaired stage is reported (§8).
- **The spans (point 5).** A penalty kick joins the `kicks` mapping with the key of §4.9 step 3: a paired or "before"-only penalty has the side `"before"`, an "after"-only one the side `"after"`. The possession after a penalty always has its own marker (217 of 217 in the same period: goal 160, penalty 35, another restart 22), so every penalty pair is "both markers": a paired penalty with a kick is `dead` until `k` and `live` from `k`; an "after"-only penalty with a kick is `unknown` from `k` to `b`; a "before"-only penalty (2 on the cache) has one marker and is `live` from `k`. A check counts 0 `live` frames between such a kick and the next reception.
- **The xShot label (point 7).** The label (`_occurrence_labels.py:18-45`) sets a positive on a frame at `t` for a shot of its team in `[t, t + 1 s]`; the training mask (`_stoppages.py:242-265`) keeps `live` frames with no `unknown` frame in that horizon.
  - A penalty row with `"placeholder"` or `"order_only"`: the label skips it (§5.4), and its time lies in the `dead` gap before `b` anyway: 0 positives.
  - A `"kick"` row of a paired penalty: only the frame at `k` can take the positive, and that label is right (the shot is at `k`).
  - A `"kick"` row of an "after"-only penalty (side `"after"`): the kick frame is `unknown`, so it is not a training row: 0 positives. A "before"-only penalty is `live` from `k` like a paired one.
  - The acceptance counts the frames that take a positive from a penalty row on the repaired stage (§8).
- **Time-window readers of other rows.** The kick lies at least 51.5 s after the previous possession's end (median 127.3 s; `penalty_wait_check.py` on the output of `sc_penalty_check.py`), so no window of 15 s or less from an earlier row reaches a penalty row, at its true or its unmeasured time.
- **Size.** About 60-90 lines in `_skillcorner_inference.py` and `skillcorner.py`, and the kick rule's penalty point in `_action_time.py` (about 15 lines, sharing the corner rule's code).

### 4.18 The key of a derived row (A-09, B-09, widened by the supervisor)

**The defect in r9.** A derived restart row took the id of its first possession, and an "after"-only one the id of `a`. One possession can be both the first possession of a restart and the `a` of an "after"-only restart that follows: 315 possessions in 259 of 889 matches, 24 of them with two corner rows, 31 with two rows that can have a kick (supervisor, `derived_key_collision.py`). Every use of that id as a key could join or overwrite the wrong row.

**The rule.** A derived restart or penalty row takes the id of `b`, the possession after its gap (§4.2, §4.17). The key of every SkillCorner row is `(game_id, original_event_id, action_provenance, kind)`, with `original_event_id` as a `canonical_id` string. `kind = restart_kind(type_id)` (in `spadl/schema.py`): `corner` for `corner_crossed` and `corner_short`, `free_kick` for `freekick_crossed` and `freekick_short`, `goal_kick`, `throw_in`, `penalty` for `shot_penalty`, and the type name for every other type. A step that re-sets a subtype (§4.3 rule 2) leaves the kind, and so the key, as it was. It is unique:
- Native rows: one per possession (125,350 rows, 0 duplicates on `(game_id, original_event_id)`, `a63_key_check2.py`).
- Derived restart and penalty rows, by construction: distinct gaps have distinct `b`, and a gap holds at most one derived row of a kind, because an "after" and a "before" marker of one kind pair into one row (0 gaps with two rows of one kind on the cache, `derived_key_check.py`). Their types are restart types or `shot_penalty`, which no native row has.
- The period-end penalty (no `b`) takes `a`. It would share a key with a penalty row of any class in the gap before `a`, because that row also takes `a`'s id (`a` is that gap's `b`): a paired or "before"-only penalty (a penalty "before" marker on `a`), or an "after"-only penalty (a penalty "after" marker on the previous possession). Each class is 0 on the cache (`derived_key_check.py`; supervisor, `derived_key_r11.py`). The §8 key check covers it on the stage.
- The inferred defensive rows: 0 duplicates in 16,895 rows with an id on the f5fc2e2b stage (`derived_key_check.py`); their types are never restart types.
- Keeper saves (`skillcorner.py:566` writes `original_event_id` `<NA>`) and synthetic dribbles have no id; they keep `<NA>` and are not keyed. Where a comparison needs them (§4.10), a keyless row is identified by the key of the previous keyed row and its offset after it (1, 2, ...).

**The three uses, each with this key:**
- **The kick mapping** (§4.9 step 3): `(original_event_id, kind)` per match.
- **The GI refinement of the corner rows** (`corners/_identify.py`; the drivers `validate_corners.py` and `build_corner_routines.py` call it unchanged, their own change is the no-window rule of §5.2): the GI frame is built per gap and keyed by `(game_id, b event id, corner kind)`; it joins the corner rows on `(game_id, original_event_id)` with the type in the corner kinds. Not on the first possession alone.
- **The stable key between the two builds** (§4.10).
- The search for more uses: `git grep -n "original_event_id" -- 'silly_kicks/*.py' 'scripts/*.py'` (21 files). The other uses are other providers (StatsBomb, Wyscout, Opta, Metrica, Sportec, gradientsports, kloppy), the atomic carry (`atomic/spadl/base.py:135`, `:200`, `:228`, which copies the id and keys nothing), the synthetic-row test `spadl/base.py:162-167` (Metrica/Sportec `_synth_`), and the schema and reflection declarations. None keys a SkillCorner derived row.

## 5. Readers (bars A7, A8)

### 5.1 The sweep

There are six commands, with their full output and a decision for each file, in `docs/research/action-time-repair/reader-sweep.md`:
- the restart types and the GI markers (45 files);
- the code that compares an action time with a frame time (70 files);
- `visibility`, `detected_mask` and `is_detected` (64 files);
- every reader that opens a time window (37 files; R50, §5.3);
- the time lookups, for the period key (43 lines; R51). The supervisor's second search (a sort or mask on `time_seconds`, a callback on a time, a loop over `groupby("game_id")`) found no other lookup of this kind.
- the readers of the shot types (60 files; R58, §5.4).

These files change:
- `spadl/skillcorner.py` (with the `preserve_native` list, §4.15), `spadl/_skillcorner_inference.py` (with the penalty rows, §4.17), `spadl/base.py` and `spadl/schema.py` (with `is_order_only` and `is_time_unmeasured`, §4.1).
- `spadl/sportec.py` (R57, §4.16).
- `coordination/_windows.py` (R54, §4.13).
- `tracking/_xshot_occurrence.py`, `tracking/features.py`, `atomic/tracking/features.py`, `tracking/_shot_goalmouth.py` and `causal/opportunities.py` (R58, §5.4).
- Every converter, to emit `time_source`.
- `reflection.py` and `feature_glossary.py`.
- `corners/_identify.py` and `corners/_windows.py`; the readers of a corner window `corners/_encode.py`, `corners/_effectiveness.py`, `corners/_reflect.py` and the drivers `validate_corners.py` (`_divergence_rows`) and `build_corner_routines.py` (`_snapshot_rows`), by the no-window rule of §5.2.
- `tracking/skillcorner.py`, the new `tracking/_action_time.py`, and `tracking/_run_features.py` (R45).
- `tracking/_stoppages.py` (R44).
- `tracking/_gk_geometry.py` (the coordinate sources of derived rows, §4.7).
- `tracking/_ghost_gk.py` (`_build_phase_lookup` takes no window at an `"order_only"` row, §5.2; both lookups take the period, §4.11).
- `tracking/_ghost_outfield.py`, `tracking/_xcross_attempt.py` and `causal/opportunities.py` (the period at the call, §4.11).
- The new `spadl/_goals.py`, with `vaep/labels.py` and `win_probability/_state.py` (R52, §4.12).
- `vaep/labels.py`, `xtgk/_retention_labels.py`, `xtgk/_turnover.py`, `tracking/_packing.py`, `team_metrics/_possession.py`, `team_metrics/_compute.py`, `team_metrics/_pressing.py`, and `spadl/utils.py` with `atomic/spadl/utils.py` (R54, §4.13).
- `tracking/_feature_contract.py`, `tracking/_warnings.py`, and the contract blocks of `_ghost_gk.py`, `_ghost_outfield.py` and `_xcross_attempt.py` (R55, §4.14).
- `scripts/_loader_pining.py` (with the SkillCorner `preserve_native` list, §4.15).
- `scripts/hf_jobs/campaign.py`, `scripts/hf_jobs/registry.py`, `scripts/hf_jobs/campaigns/tf46_refit.toml`, `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` and `scripts/hf_jobs/adopt.py` (R38, R49, §4.8).

### 5.2 The rule of each reader that has a time label (bar A8), and every place that took the receiver for the taker

| Reader | `"event"` / `"tracking"` / `"elastic"` / `"kick"` | `"placeholder"` | `"order_only"` |
|---|---|---|---|
| xShot label (`_occurrence_labels.py`, `_xshot_occurrence.py:300-339`) | the shot row time; idsse shots move by R32 | Not read: a restart row is never a shot. | as `"placeholder"` |
| xCross label (`_xcross_attempt.py:358-367`) | the open-play `cross` row time | Not read: a restart row is never a `cross`. A SkillCorner first possession can now be a `cross` by `classify_deliveries`, the intended better model (R30). | as `"placeholder"` |
| Ghost-GK / ghost-outfield `phase` (`_ghost_gk.py:627-695`; `_ghost_outfield.py` imports it) | starts the 10 s set-piece window at the row time; the next set-piece row ends it | As the other values: the window starts at the first reception minus 0.02 s, a median 2.0-2.2 s after the kick (§4.2), over the restart's own first possession. | **R50:** `_build_phase_lookup` drops the `"order_only"` rows (`is_order_only`) before it builds the lookup, so the row opens no window and ends none. No frame takes a phase from it, whatever follows it (throw-in 1,263, goal 68, penalty 2, another restart). Today's rule would label 13,332.6 s on the 889 matches (`order_only_overlap.py`). The drop changes no other row's window: the nearest earlier set-piece row is at least 16.99 s before an order-only row (0 of 3,274 within 10 s), and a later one starts its own window as today. The throw-in rule (`:634`) stays. The other providers have no `"order_only"` row, so their phase is unchanged; a table without `time_source` is unchanged. The period key is repaired by R51 (§4.11). |
| Corners (`corners/_identify.py:76-97`, `_windows.py:81-84`, `_encode.py` (`_nearest_frame_xy`, `_encode_one`), `_effectiveness.py`, `_reflect.py`, `scripts/build_corner_routines._snapshot_rows`, `scripts/validate_corners._divergence_rows`) | `t_kick` = the row time; `kick_x`/`kick_y`/`corner_side` from the row's start; the windows as today | `t_kick`, `kick_x`, `kick_y` and `corner_side` are `<NA>`. There is no snapshot window and no outcome window (`corners/_config.py:37`). The corner stays in the table with its kind, and the count is reported (bar A6.5). No reader takes an unmeasured side for the side of the kick. No reader takes a value for it (supervisor C, bar A3): `_nearest_frame_xy` gives no frame for a missing time, so the run encoding, the routine features and the two drivers' snapshot rows are missing (the dominant routine is the -1 bucket); the chance-quality counts are `<NA>` and `sum_xg` NaN (the `Int64` columns hold it), and `aggregate_chance_quality_by_feature` takes no such corner into a mean and counts them in `n_corners_without_window`; a corner with no `corner_side` is not encoded, gives no snapshot row and is not reflected (never the text "None"). | as `"placeholder"` |
| `tracking/_stoppages.py:69-75` | — | R44: the first possession is `live`, and the dead span ends at a measured kick. The docstring names `b` the receiver. | The spans come from the GI markers, not from the row time (§4.9). |
| `tracking/_gk_resolve.py:88-102`, `keeper_identity.py:876-886` | — | The goal-kick row is the derived row with `player_id` `<NA>`. The receiver is no longer a keeper candidate; the rule for the repaired rows is unchanged. `keeper_identity.py:876-885` changes for another cause (R72, owner 2026-11-19, not the repair): the roster path reads `type_id` when the table has no `type_name` (it raised `KeyError 'type_name'` on converter output). | as `"placeholder"` |
| Frame features (the linker is not touched, bar A9) | correct input | R45: the producer drops the pointer; all frame features are NaN on the producer path | as `"placeholder"` (R45, R50); §5.3 |

### 5.3 Every reader that opens a time window (R50, bar A16, point 3b)

Command 4 (`reader-sweep.md`) lists the 37 files that open a time window. A reader takes no window at an `"order_only"` row in one of these ways:
- **The producer's blank step (R45, R50).** The window families that slice frames by the action time (`slice_around_event`: `_kernels._actor_pre_window_kernel`, `_off_ball_runs`, `_run_values.detect_off_ball_runs`, and the atomic and `features.py` pre-window wrappers) do not read the pointer. Their selection is empty on a derived row (`player_id` `<NA>`; `ids_differ` is not "differ" on `<NA>`, `id_compat.py:366-368`), but an empty selection is not a blank output: `_off_ball_runs_kernel` writes 0 runners (`:185-194`). So `run_tracking_features` blanks the output: after the families run, each column that a family added gets the missing value of its dtype on each `"placeholder"` or `"order_only"` row. A direct `add_*` call is not covered (R45), and the docstrings of these families say so.
  - **The pointer columns.** The dropped pointer gives the four link-provenance columns their values for a row that takes no frame (the link contract, `tracking/utils.py:381-387`): `frame_id` `<NA>` (nullable `Int64`), `time_offset_seconds` NaN, `link_quality_score` NaN, `n_candidate_frames` 0. The blank step leaves them as the dropped pointer made them; `n_candidate_frames` stays `int64` and is the count 0, not a missing value. No model reads these columns.
  - **The keeper identity.** `defending_gk_player_id` and `defending_gk_team_id` are stamped by the producer's resolver step before the families run; they are no frame feature and keep their value on these rows.
  - **The refusal.** A family column whose dtype has no missing value (numpy integer or bool) makes `run_tracking_features` raise `TypeError` when such a row exists, instead of writing a value that is not missing (the stricter form of the STOP).
  - **Proof by a run, not by the code (§8):** on the repaired stage, every default family of the producer runs, and each column it adds is checked blank on these rows. A column whose dtype has no missing value is a STOP for the plan.
- **The frame pointer.** The pointer families take no frame for the row on the producer path (R45, R50).
- **The type.** Readers of shot, cross, tackle or take-on rows only (`_shot_goalmouth.py`, `_occurrence_labels.py`, `duels/_extract.py`, `causal/opportunities.py`) never read a restart row.
- **The time source.** `_build_phase_lookup` (§5.2), the corner windows (`t_kick` `<NA>`), and the keeper-detection window of `_gk_geometry.py` (tracking tiers skipped, §4.7).
- **Not a window on an action row.** Frame-only windows (smoothing, resampling, `training_frame_mask`, ELASTIC, `_ball_contact.py`), and parameters, defaults or text (`calibration/`, `positioning/_probe.py`, `feature_glossary.py`, `_artifact_integrity.py`).

The seven action-window readers take no window at an `"order_only"` row by R54 (§4.13).

### 5.4 The readers of a shot row's time (R58, bar A24, point 6)

Command 6 (`reader-sweep.md`) lists the 60 files that read the shot types, with a decision for each. A derived SkillCorner penalty row with `is_time_unmeasured` (§4.1) is the new case. These readers take the time of a shot row, or the frames at it, and skip such a row:
- the xShot label (`_xshot_occurrence.py:293`, `:340`, `:867`): the row is not a labelled shot;
- the pre-shot keeper position and angle (`tracking/features.py:107`, `:638-887`) and the atomic twins (`atomic/tracking/features.py`): NaN for the row;
- the shot goal-mouth (`_shot_goalmouth.py:47`, `:772`): NaN for the row;
- the causal treatment and outcome windows (`causal/opportunities.py:104-191`, `:474`): the row is neither;
- TF-49 packing: a pass whose reception row is such a shot row gets `packing_secured` `<NA>` (as §4.13);
- the coordination goal interval (§4.13).
The producer's blank step (§5.3) covers these rows on the producer path too. The other readers read the type, the result or the order, not the time (command 6).

## 6. Retrain list (bar A8)

Every artifact fitted on the staged GS, SkillCorner or idsse actions is in the refit of the first directive. The repaired stage changes their inputs as follows:

| Artifacts | What changes |
|---|---|
| `xshot_*`, `hub_xshot_*` (4) | idsse shot times (R32); the idsse penalty rows (R57); the SkillCorner penalty rows with a measured kick (R58); the SkillCorner training frames (R44); the wheel bundles ship `public` (R49, §4.8) |
| `xcross_*`, `hub_xcross_*` (4) | `score_differential` per period, with the full goal rule and the SkillCorner running score (R51, R52, R56); idsse cross times; SkillCorner first possessions that become `cross`; the SkillCorner training frames (R44); the wheel bundles ship `public` (R38, §4.8) |
| `ghost_*`, `hub_ghost_*`, `ghost_full` (7); `outfield_*`, `hub_outfield_*` (4) | `phase` from the real restart rows, with no window at an `"order_only"` row (R50); `phase` and `score_diff` per period (R51); the full goal rule (R52); the SkillCorner running score (R56); idsse times |
| `corners_run_zone`, `corners_routine` (2) | the corner table, `t_kick`, the kick point and side |
| `receiver_model` (1) | No input change: it pools the tc3 gradientsports units (`tf46_refit.toml:110-117`), whose rows the step leaves unchanged except the new column. It runs in the campaign as before (`cv_parity`). |

The retention bundles read `goalkick`/`throw_in` rows, but their certification is `"blocked"`, so they are not refit. The StatsBomb-only artifacts get the new column (`"event"`) and no other change; the plan checks that their feature builds select columns by name.

**The datasets restaged from the repaired code:** `xshot187`, `xcross_probe` and `tc3`. `sb360` and `sb_open` are not restaged.

**xShot** is in the retrain list and in the campaign (R43). Its wheel bundles ship the `public` arm (R49, §4.8). The plan examines the cause of the 2 h timeouts of the xShot study stages before the launch.

## 7. Records (bar A10)

These are written or corrected in the repair commit:
- **A new ADR** for the four decisions: the `time_source` column (with the order-only rule), the `sync_action_times` step, the SkillCorner restart rows, and the period in the score and phase lookups (R51: not byte-identical, two features change on every provider, the 15 artifacts are retrained, and what a pre-repair bundle gets at the serve). The number is assigned at commit prep; ADR-134 is already used on `origin/feat/cross-definition`.
- **ADR-133:** a short amendment that corrects its SkillCorner statement and points to the new ADR.
- The 4.144.0 CHANGELOG line on SkillCorner rows.
- `docs/research/cross-definition/caller-sweep.md:93`.
- The SkillCorner converter docstring (`skillcorner.py:242`).
- The `t_kick` description (`corners/_identify.py`, `corners/_columns.py`).
- `tracking/_stoppages.py:69-75`.
- The events-only property: the `_build_match_actions` docstring (`scripts/_loader_pining.py:646-651`) and `tests/scripts/test_loader_pining_events_only.py:10-11` (§4.10).
- The `_build_phase_lookup` docstring: an `"order_only"` row opens no window (§5.2). Both lookup docstrings: the `(game_id, period_id, time_seconds)` callback (§4.11).
- The CHANGELOG line for R51 (the two features, every provider, not byte-identical) and for R52 (the score counts every goal of the library's rule; a period-5 shoot-out does not count; the gradientsports converter puts the shoot-out in period 5, with no location).
- The new ADR names the single goal predicate (R52), the SkillCorner running score and its effect rule (R56), and the penalty rows (R57, R58).
- The CHANGELOG lines for R56 (the SkillCorner `preserve_native` list; the score from the running score), R57 (the idsse `shot_penalty` row) and R58 (the SkillCorner `shot_penalty` rows).
- The docstrings: `skillcorner.convert_to_actions` (the list form), `_build_score_lookup` (the two paths, the effect rule and its lag), `infer_penalties` (the count rule, the result, the time), and the readers of §5.4.
- The docstrings of the seven R54 readers (§4.13), the coordination windows among them, and the CHANGELOG line for the VAEP `"time"` mode dtype (nullable `boolean`; that mode only).
- The CHANGELOG line and the ADR item for R55 (the `match_context_revision` constant and the new contract branch; a pre-repair bundle warns).
- The corner memo headline and table (R41).
- The wheel ship rule (R38, R49): ADR-132 item 6 (`:56-57`, "The wheel jobs get both arms and ship the rule's choice") gets an amendment. In `docs/superpowers/specs/2026-11-12-corpus-widening-design.md`, the ruling r3 line (`:26`), §5.2 "Outcome" (`:162`), the §5.5 wheel/Hub job rules (`:186-196`) and the §7.1 table (`sc_extended` column, `:337-396`) get the same amendment. The `tf46_refit.toml` comments (`:60`, `:77`) cite R49 and R38.
- `docs/context/providers.md`, and the AGENTS.md key-convention bullet of ADR-133.
- The limits of §9.

## 8. Acceptance, checks and tests

- **Acceptance:** bar A6 items 1-6, on the repaired stage, with the scripts of §4.6 run again. Item 3, defined on the f5fc2e2b stage:
  - SkillCorner: every native row (`action_provenance == "native"`: the first possessions are in, the derived rows are out), keyed on `(game_id, original_event_id)`. The key is unique: 125,350 rows, 0 duplicates (`a63_key_check2.py`). `time_seconds` must be equal.
  - Gradient Sports: no rule of the step touches its rows, so the whole actions table must be equal row by row in `action_id` order, every column except `time_source`. Its id is not unique: 13,097 of 85,780 non-synthetic rows share one with another row.
- **The phase count (bar A16, point 3f).** A script in `docs/research/action-time-repair/` runs `_build_phase_lookup` on the repaired stage as the trainers call it and counts the frames that take a phase from an `"order_only"` row: the count must be 0, for each class of follower. It also counts, for every other row, the frames whose phase differs from the lookup built on the same table with the order-only rows kept; the count must be 0 outside the windows that order-only rows open.
- **The period (bar A18, point 2f).** The supervisor runs `~/dl-review-xdef/phase_checks.py` and `phase_checks_frames.py` on the repaired code: no difference between the lookups and the one-clock reference.
- **The readers with no `"order_only"` row (bar A21, point 4).** On the repaired stage with the order-only rows removed, and on the f5fc2e2b stage, each R54 reader gives today's values.
- **The warning (bar A22, point 3).** The 15 adopted bundles load with no warning; the pre-repair bundles of the same families warn; no other bundle or Hub entry warns.
- **The running score (bar A20, point 5).** On the repaired stage: the matches on each path (running score, goal rows), the early rises, and the lookup's score at the end of each match against the final score of the metadata.
- **The idsse penalty (bar A23, point 6).** The supervisor's run: 19 goal rows on idsse (16 today), and the goals of each match equal its final score.
- **The SkillCorner penalty rows (bar A24, points 4, 5, 7).** The count by the count rule (221 on the cache); the rows with `success` (167 on the cache); for each SkillCorner match, the goal rows by the R52 rule are not more than its final score; the share with a kick; 0 `live` frames between an `"after"` penalty kick and the next reception; the frames that take an xShot positive from a penalty row.
- **The goal rule (bar A19, point 4).** The supervisor checks on the repaired code that the score lookup's goals on the gradientsports stage equal the goal rows of the full rule outside period 5, and that no staged frame lies after a shoot-out row.
- **The order-only corners.** The count of `"order_only"` corners that get `"kick"` by the rule of §4.4, by follower kind (r8 measure: 189 of 690).
- **The blank frame features (R45, R50).** A script runs `run_tracking_features` with its default families on the repaired stage (each provider) and counts, per added column, the `"placeholder"` and `"order_only"` rows whose value is not missing. The count must be 0 for every family column. The four pointer columns are checked for their values instead (`frame_id` `<NA>`, `time_offset_seconds` NaN, `link_quality_score` NaN, `n_candidate_frames` 0), and the two keeper-identity columns keep their value (§5.3). It also lists each column's dtype.
- **The spans after an order-only kick.** A count on the repaired stage: 0 `live` frames between an `"after"` kick and the next reception. Also counted: the possessions that are `b` of one gap and `a` of a kicked gap, the gaps with two kicks, and the `"kick"` rows whose kick frame lies outside their `restart_gaps` gap of the pre-step table (must be 0), and the reversed pairs (dropped).
- **The key (§4.18).** On the repaired stage: 0 duplicates of `(game_id, original_event_id, action_provenance, kind)` among the SkillCorner rows with an id; the `"kick"` corners whose measured start changes the subtype, and in which direction; the count of possessions that give their id to a native row and to a derived row, and of gaps with an "after"-only and a following restart.
- **The NaN starts.** The count of derived restart rows with a NaN start, by type (expected: throw-in 1,263, free kick 1,054, corner 231, minus the corners with a measured kick).
- **Gate check:** bar A11, a local check of xShot, xCross, ghost-GK, ghost-outfield and corners before the launch. The logs go to the ledger.
- **The local test gate** (on the DGX, before the commit): every test file that names a changed module, from this command, plus the lint trio. The command selects 668 of 1,227 test files at `6eee84ce` (the same at the old C1 `3ec4b79a`) (552 before r10; the r10 terms add R54, R55, R56-R58 and the four modules `spadl/utils.py`, `atomic/spadl/utils.py`, `tracking/features.py`, `atomic/tracking/features.py`). It holds a term for each module of §5.1. This is not the full suite; CI stays the full-suite gate.

  ```bash
  git grep -lE 'spadl\.(skillcorner|sportec|statsbomb|opta|wyscout|metrica|kloppy|gradientsports|base|schema)|spadl import|_skillcorner_inference|SPADL_COLUMNS|_finalize_output|corners\._(identify|windows|columns)|identify_corners|add_corner_windows|tracking\.skillcorner|tracking import skillcorner|_loader_pining|_stoppages|_action_time|sync_action_times|_run_features|run_tracking_features|reflection|feature_glossary|convert_to_atomic|validate_corners|build_corner_routines|hf_jobs\.campaign|hf_jobs import campaign|load_campaign|_gk_geometry|add_restart_coordinates|resolve_restart_geometry|hf_jobs\.registry|hf_jobs import registry|_ghost_gk|_build_phase_lookup|_build_score_lookup|vaep\.labels|vaep import labels|win_probability|_goals|spadl\.sportec|coordination\._windows|coordination import|_shot_goalmouth|add_pre_shot_gk|_xshot_occurrence|is_time_unmeasured|_ghost_outfield|_xcross_attempt|causal\.opportunities|causal import opportunities|is_order_only|train_xshot_occurrence|train_xcross_attempt|hf_jobs\.adopt|hf_jobs import adopt|_events_admission|events_only|vaep\.labels|_retention_labels|_turnover|_packing|team_metrics|_feature_contract|_warnings|test_declared_constant_values|spadl\.utils|spadl import utils|tracking\.features|tracking import features|atomic\.spadl|atomic\.tracking|atomic import|coordination\.(_windows|windows)|_xshot_occurrence|_shot_goalmouth|causal\.opportunities|causal import|_ghost_outfield|_xcross_attempt' -- 'tests/*.py'
  ```
- **Tests** (TDD; each written failing first):
  - `tests/spadl/test_time_source.py`: vocabulary, dtype and value per converter and row class; `is_order_only` (present, absent column).
  - `tests/spadl/test_skillcorner_restarts.py`:
    - the count rule (paired, "before"-only and "after"-only markers);
    - the team, including a disagreeing pair;
    - no first-possession player on a restart row;
    - the subtype, the rule points and the side of the previous possession's end;
    - the `"placeholder"` and `"order_only"` times and order, and the events-only behaviour;
    - every "after"-only class gets `"order_only"` (a throw-in, a goal and another restart follow), every paired one `"placeholder"`.
  - `tests/tracking/test_action_time.py`:
    - rules 1-4 on a synthetic match: an ELASTIC-aligned row, an unaligned row, a corner with a seen rest and release, one with an extrapolated ball only, a dribble, and an inversion;
    - the refusal of a table without `time_source`;
    - the skip of ELASTIC with no `"event"` row.
  - `tests/spadl/test_skillcorner_restarts.py` also: the period-end exclusion; the result of an "after"-only restart (a goal marker of the restart team, a goal marker of the other team, and another restart).
  - Corners: a NaN `t_kick` and `<NA>` side on a `"placeholder"` and an `"order_only"` row; the GI refinement joined on the key of §4.18, with the shared-possession case (an "after"-only corner on `p1` and the corner whose first possession is `p1`: each row takes its own gap's GI data).
  - The key (§4.18): each derived restart and penalty class takes the id of `b`; a period-end penalty takes `a`; the shared-possession case for the kick mapping, the GI join and the events-only key.
  - `tests/spadl/test_add_restart_coordinates.py`: §4.7 for each class (`"placeholder"` and `"order_only"` corner, goal kick, throw-in and free kick; kick row): source, confidence and coordinate, and the tracking tiers skipped.
  - `convert_to_atomic(preserve_native=...)` carries `time_source`.
  - The frame adapter: the ball `visibility`. The producer: the R45 pointer drop, for `"placeholder"` and `"order_only"` rows.
  - The stoppage spans: the 10 tests of §4.9 (each row of the table, the shared possession, two kicks in one gap, a reversed pair, a kick outside its gap).
  - The gap at a period start: a "before"-only restart on the first possession of period 2 has a gap from the period's first frame, not from the last row of period 1.
  - An `"order_only"` corner with a kick and a goal of its team after it stays `corner_crossed` (no first possession).
  - An `"order_only"` corner whose measured side differs from its prior side takes the rest point as start and end; a penalty row with a kick likewise; a corner with a first possession keeps its end at the reception.
  - The step re-sets the subtype: a corner with a first possession whose measured start changes its subtype by the 15 m rule gets the measured subtype, and the loader's gap check and the kick mapping still find it by the §4.18 key (the kind is unchanged).
  - The loader's kick check can fire: a stepped table with one `"kick"` row whose time lies after `b.frame_start` (injected) makes the check fail, and the count is 1.
  - The period-end penalty with a kick: `unknown` from `k` to the period's last frame; the first possession of the next period is not read as its `b` by the spans. A paired penalty whose `b` is the last possession of its period keeps `live` from `k` (the side comes from the mapping).
  - The gap (RTR-SPEC-12): a flag rest inside `b`'s possession (after `b.frame_start`) gives no `"kick"` for a corner and none for a paired penalty; a rest in the gap does.
  - The period-end penalty: its gap runs to the period's last frame; a kick gives `unknown` from `k` to the period end.
  - The order-only corner rule: a kick at the first flag run with a later run; none when the flag run is the last run and the follower is a throw-in; a kick when the follower is a goal kick.
  - The restart after an "after"-only restart: a NaN start for a throw-in and a free kick; `_CORNER_X` and a NaN `start_y` for a corner; `_RULE_POINT` for a goal kick; `"unresolved"` in the enrichment.
  - The producer's blank step: a family column on a `"placeholder"` and an `"order_only"` row is missing, including the off-ball runner counts; other rows are unchanged.
  - The ghost phase: an `"order_only"` row opens no window and ends none, for a throw-in, a goal and another restart as follower; a `"placeholder"` row opens a window as today; a table without `time_source` gives today's phase; the throw-in rule is unchanged.
  - The lookups with the period (fail on today's code): for each lookup, two periods (a first-half set piece or goal does not reach a second-half query at a smaller clock time; a second-half goal does not reach a first-half query; a first-half goal counts in the second half) and extra time (periods 3 and 4). The callers: one test per caller that the period reaches the lookup (ghost-GK fit and serve, ghost-outfield, xCross fit and serve, causal).
  - The tests that pin today's lookups (bar A18, point 2h). The lookup tests `tests/tracking/test_ghost_gk.py:860-938` (10 tests) change their call to the three-argument form; their values do not change, because every fixture row has `period_id` 1 (`:556`, `:579`, `:602`). A search of `tests/` for `_build_score_lookup`, `_build_phase_lookup`, `score_diff`, `score_differential` and `"phase"` finds 41 files. A search of these for a literal second period (`period_id` 2) finds only `tests/scripts/test_das_native_parity_driver.py` and `tests/spadl/test_wyscout.py`, which reach no lookup. Committed real-match fixtures are not seen by that search. The plan runs the gate files on the repaired code and names each test whose pinned value moves, with the reason, before the commit.
  - The goal rule (fail on today's code): a penalty goal, a direct free-kick goal, and an own goal on `bad_touch` each move `score_diff`; a period-5 goal does not; `vaep/labels.py` and `win_probability/_state.py` give the same results as before on their existing tests.
  - R56 (fail on today's code): the side of the values (`ids_match`), a derived row not read, the effect rule, a match with a decrease falls back to the goal rows, a table without the columns, and the atomic carry.
  - R57: a scored penalty, a missed penalty, and a `Penalty` with no nested shot (excluded by name).
  - R58: the count rule (paired, "after" only, "before" only, a period end); the team, the start, the result (an "after"-only score rise, no rise, the goal-marker fallback, a paired penalty with a rebound goal at `b` is `fail` and the rebound row is the goal row, a period-end penalty, the last period with no later event); the kick rule at the mark (a seen rest and release, a retake, no kick); the `"placeholder"` and `"order_only"` times; each reader of §5.4 on a table with such a row.
  - R54 seventh reader: the coordination windows and intervals on a table with an `"order_only"` row, and today's output on one without.
  - xShot `--ship-variant`: the override, the refusal without the two-arm corpus, and `rule_verdict` / `rule_why` in both trainers; the adoption-record line.
  - R54, each reader on a table with an `"order_only"` row: VAEP `"time"` labels `<NA>` (and NaN with `xg_column`), `retains` NaN, no turnover anchor, `packing_secured` `<NA>`, the spell start and recovery rule, the carve-out keeps the boundary. Each reader on a table with no `"order_only"` row equals today's output (VAEP `"time"` mode: the same values in nullable `boolean`).
  - R55: an artifact without `match_context_revision` warns `PredatesDeclaredConstantWarning` with the meaning in its message and loads; one with it loads with no warning; one with another value raises.
  - The events-only property of §4.10: events-only = full build before the step; full build = the step applied; the comparison keyed by §4.18, with a shared possession and two keeper saves in a row in the fixture.
  - The registries the new public function joins: `PURITY_ENTRIES`, the call-convention registry, `feature_glossary`, and `metric_contracts` if it applies.

## 9. Limits and no-touch areas

**No-touch (A9):**
- The event block of `providers/sportec/parse.py`, except the one change of R63 (bar A25) in the nested-child loop of `_build_event_row` (§4.16).
- `tracking/utils.py` and `tracking/_link_identity.py`, with the pointer columns.
- The bronze stays as the provider gives it: `_skillcorner_bronze` only stops dropping the feed's ball `is_detected`.
- The idsse sync runs on the converted actions, so none of these is needed. If the plan finds one is needed, it stops and reports.

**Limits named in the records** (accepted by the owner, R46):
- SkillCorner kicks are measured for corners and penalties only, and only where the tracking shows the rest and the release (corners: 0.394 of the paired corners, and 189 of the 690 order-only corners by the rule of §4.4; penalties: 72.9 % on the cache).
- The place of a `"placeholder"` or `"order_only"` restart row is a rule point. A restart that follows an "after"-only restart has no known place (NaN start), except a goal kick.
- An `"order_only"` corner gets a kick only where the rule of §4.4 shows that the rest is not the following restart's (189 of 690); its kick time has no reference measure. The corner side without a measured kick is a prior: the side of the previous possession's end, right for 0.823 of 4,660 corners with the ball seen at a flag. For a free kick, the previous possession's end is not measured as the spot.
- The SkillCorner restart rows have no taker (R40).
- **A restart that a goal follows with no possession between** (corner 22, free kick 46, throw-in 3 in 889 matches; 2 more are followed by a penalty).
  - The goal has no shot row, because the data has no possession that ends in it.
  - The restart row's result follows the goal marker (§4.2).
  - The row sits after the goal (its `"order_only"` time is before the kick-off reception). It opens no phase window (R50).
  - The corner outcome counts goals from shot rows, so it does not count these 22.
- **Direct free-kick shots.** 619 first possessions after a SkillCorner free kick end in a shot. Only 1.0 % of them start within 3 m of where play stopped, so these are the receivers' shots (supervisor, `spec_checks_sc.py` 4). A direct shot by the taker has no row in the data. Its derived row is `freekick_crossed` or `freekick_short`, never `shot_freekick`.
- The gradientsports throw-in time is not measured (§4.6).
- `"elastic"` is measured on idsse only (§4.3). idsse matches without tracking keep `"event"`.
- 20 SkillCorner stage matches have no cached raw tracking for the measures in this spec. The step itself runs on every staged match.
- **The SkillCorner running score (R56).** The old score stays for the first live play after the kick-off reception (median 1.5 s, quantile 0.9 6.2 s). For 4 of 2,358 goals the provider's score rises two possessions early, so the new score shows before the goal; the acceptance counts them. A match whose running score decreases on a side (3 of 889; supervisor, `sc_score_columns2.out`) uses the goal rows. The last goal of a match with no event after it (19 in 886 matches) takes no effect, as no row follows it.
- **The SkillCorner penalty rows (R58).** No taker (R40). The penalties with no measured kick (27.1 % on the cache) keep `"placeholder"` or `"order_only"`. The shoot-out is not a test of the kick rule.
- **Other SkillCorner goals with no goal row (R59).** 124 goals whose scoring touch is not in the data: 78 straight from a restart, 40 after the assist pass, 6 with an own goal or no scorer possession. 6 shot possessions with no goal marker stay failed shots. With R56 they do not change the score.
- **The declared pandas floor (R79, R80).** The floor (`pandas>=2.1.1`, `numpy>=1.26.0`) holds for every path the tests run and for the 34 literal-subject compare sites that were read. Two searched lists were not read, by the owner's ruling: the 51 compare sites with a non-literal subject (provider columns inside the converters) and the 122 float reads with a variable subject. A floor failure on one of those paths is possible and is not excluded. The searches and the table are in `docs/research/action-time-repair/pandas-floor/`.
