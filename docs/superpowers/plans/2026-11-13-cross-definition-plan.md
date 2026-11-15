# One Cross Definition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One provider-neutral geometric `cross` / set-piece delivery rule in every SPADL converter, a tracking enricher that gives `cross_attempt` and `cross_height`, the TF-79 detector v2 it reads, and the consumers of both.

**Architecture:**
- Layer 1 is `classify_deliveries` in `spadl/base.py`, called by every converter; each native label moves to `cross_native`.
- Layer 2 is TF-86 `add_cross_deliveries`: the release (native time, else ELASTIC), the ball path to the first non-passer contact (TF-79 v2), the cut-out rule.
- Consumers: the xCross label v2, causal opportunities, TF-46 `corner_kind`, bravery.

**Tech stack:** Python 3.10+, pandas 2/3 (CoW), numpy, numba, pytest. DGX for tests and data runs.

**Specs:** `docs/superpowers/specs/2026-11-13-cross-definition-design.md` (§ below) and `docs/superpowers/specs/2026-11-15-tf79-contact-detector-v2-design.md` (D§ below). Both state the final design; ADR-133 and ADR-134 hold the decisions.

**State.** Commits 1–3 are pushed (`4490016c`, `b12c0fd9`, `dd41724d`; draft PR #306) and CI is green on them. The commit-4 content is on the uncommitted tree: the cut-out rule, the detector v2, the truth harness, the consumers, the tests and the final-state docs. The ledger is `.superpowers/sdd/2026-11-13-cross-definition-plan/progress.md`; it records every completed task, ruling and review.

## Global Constraints

- **Branch:** `feat/cross-definition` off `main` `0cc07e6c`. No worktree, no subagents.
- **Five commits in total** (owner ruling 2026-11-15; every run must read a committed tree, ADR-037). Two remain:
  - **Commit 4:** the remaining code, tests and final-state docs, with the provisional detector defaults (`contact_radius_m` 1.2, `min_velocity_change_ms` 3.0). Every run below reads the clean commit 4.
  - **Commit 5:** the tuned constants, the memos (stamped with commit 4's sha), the pin tests, CHANGELOG, `_version.py` and TODO.
  - Each commit and its push need the owner's explicit yes, immediately before (Gate 4, Gate 5). The merge, the tag and the release each need their own yes. Merge with `--merge`, never squash: the memos cite commit 4's sha.
- **Tests run on the DGX checkout `~/sk-fsg`** (`karsten@192.168.68.73`), synced with `D:\Temp\cross-def\sync.sh`. Run only named test files: `.venv/bin/python -m pytest <files> -m "not e2e and not slow" -p no:randomly -q --basetemp ~/cross-def/pt`. Never the full suite or a whole test directory; CI is the full-suite gate.
- **Lint trio** (CI-faithful): `.venv-lint/bin/ruff check silly_kicks/ tests/ scripts/`, `.venv-lint/bin/ruff format --check silly_kicks/ tests/ scripts/`, `.venv-lint/bin/pyright --pythonpath .venv-lint/bin/python`. Show the real exit codes.
- **Runs:** `~/sk-fsg` reset to the clean commit (`git fetch && git reset --hard <sha> && git clean -fd`); the corpus is the commit-3 re-stage `~/cross-def/xshot187-r8/out`; parallel workers into one generation. `~/xshot187/out` is never overwritten.
- **Held-out discipline:** nothing is computed split-wise on the held-out GS half (`sha256(str(int(game_id))) % 2 == 1`) before the single held-out run (D§6).
- **Temp files** go under `D:\Temp\cross-def\` and DGX `~/cross-def/`, never C:.
- **Data comes from pining only.** Databricks is not a source. No HF job, dataset upload or `datasets record` without the owner's go.
- **Ids** are compared through `silly_kicks.id_compat`; geometry constants come from `spadlconfig`; every `warnings.warn` uses `stacklevel=2`.
- **TF id:** TF-86 (TF-85 belongs to another cycle).
- **Version:** single-sourced in `silly_kicks/_version.py`; the minor bump (4.143.0 → 4.144.0) goes in commit 5, unless another release lands first.
- **Scope:** nothing is deferred, dropped or added to TODO without the owner's yes.

## Review Focus

1. **A kick from an exactly static ball** (IDSSE freezes the dead ball): a rest longer than the velocity window is kept, so the kick is a candidate at 29.97, 25 and 10 Hz; a one-frame stale repeat in flight is still collapsed (D§4 item 1). Tests: `test_frame_rate_floor`, `test_kick_from_rest_passes_the_speed_gate`, `test_stale_repeat_in_flight_is_collapsed`, `test_rest_run_longer_than_rest_s_is_kept_whole`.
2. **An opponent contact exactly at `cut_out_window_s`, and a cut-out corner:** a contact at exactly 1.0 s is inside the window; a cut-out corner gives `<NA>` and TF-46 falls back to `corner_kind_source == "event"` (§6.3, §6.4). Tests: `test_cut_out_window_both_sides`, `test_cut_out_corner_falls_back_to_event`.
3. **A low frame rate:** at 10 Hz the windows floor at two frame intervals, and the cut direction window excludes the contact's own frame (D§4, §6.3). Tests: `test_window_spans_two_frame_intervals`, `test_cut_window_excludes_the_contact_frame_at_10hz`.
4. **A frame without player rows, or a ball with NaN z:** unresolved, or the height check skipped; nothing raises (D§4 item 3). Tests: `test_frame_without_players_is_unresolved`, `test_ball_above_reach_is_unresolved`.
5. **Mixed `time_source` in one match:** the enricher picks the release per row (§6.3). Test: `test_release_order_is_per_row`.

---

### Task 26: Commit 4 (first form) — done as `9b9e8838`

Done: commit 4 is pushed as `9b9e8838` (CI green), and the rule-mode detector development run on it found no feasible point (ledger). The owner approved the contact scorer (`docs/superpowers/specs/2026-11-15-tf79-contact-scorer-design.md`, S§ below). Commit 4 is amended with the scorer work (Tasks 31–38).

### Task 31: The shared truth module (S§3, S§4)

**Files:** create `scripts/_gs_truth.py` (move `split_of`, `otb_sequences`, `score_next_touch`, `_shares`, `_meets_bar`, `select_point`, the bar and grid constants from `scripts/measure_ball_contact_validation.py`); the harness imports them; its rule-mode held-out path (`run_gs_truth` with `split="holdout"`, `point_from_dev_report`) is removed. Tests: `tests/scripts/test_gs_truth.py` (the moved tests), `tests/scripts/test_measure_ball_contact_validation.py` (the harness keeps the rule dev grid and the other-provider reports).

- [ ] **Step 1:** Move the tests first; run them. Expected: FAIL (module missing).
- [ ] **Step 2:** Move the code; the harness imports it. Run both files. Expected: PASS.

### Task 32: The feature-contract probe keyword (S§2.3)

**Files:** `silly_kicks/tracking/_feature_contract.py` (`feature_contract(..., *, constants, probe=None)`); test `tests/tracking/test_feature_contract.py`.

- [ ] **Step 1:** Tests: a given probe changes `probe_sha256`; the default keeps the existing models' contracts byte-identical. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run the file plus `tests/tracking/test_geometry_constant_enumeration.py` and the bundled-model contract tests. Expected: PASS.

### Task 33: The scorer and the credit step (S§2)

**Files:** create `silly_kicks/tracking/_contact_scorer.py` (`contact_scorer_probe_frames`, the S§2.2 feature extractor, `ContactScorer` with `fit` / `predict_proba` / `to_dict` / `from_dict` / `save` / `load`, `ContactScorerIntegrityError`, the contract with `K_OPTIONS`, `GAP_CAP_M`, `kinematic_window_s`, `min_candidate_speed_ms`); `silly_kicks/tracking/_ball_contact.py` (`BallContactParams.contact_scorer: str | None = None`, `min_contact_probability`, the S§2.1 order, the scorer-mode outputs, the lazy cached load); tests `tests/tracking/test_contact_scorer.py` and additions to `tests/tracking/test_ball_contact.py`.

- [ ] **Step 1:** Write the S§7 tests for features, credit, outputs, params, model and artifact (with a fixture scorer fitted in the test). Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run both files and `tests/test_add_star_purity.py`, `tests/tracking/test_id_dtype_invariance.py`, `tests/test_public_api_examples.py`, `tests/test_scale_guard_registry.py`, `tests/test_scale_guards.py`. Expected: PASS.

### Task 34: The truth dataset tools (S§4)

**Files:** create `scripts/materialize_gs_contact_truth.py` (per GS match: `frames.parquet`, `actions.parquet`, `sequences.parquet` under `gradientsports-<game_id>/`; `run_commit` stamped) and `scripts/check_gs_contact_truth.py` (staged = served on every unit, the S§4 columns, exact); `_NOT_A_PIPELINE` entries in `tests/scripts/test_compute_contract.py` for both, with reasons. No `datasets.toml` declaration in this form. Tests: `tests/scripts/test_materialize_gs_contact_truth.py`, `tests/scripts/test_check_gs_contact_truth.py`.

- [ ] **Step 1:** Tests on toy data (the layout, the columns, the two-sided comparison). Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run the two files, `tests/scripts/test_compute_contract.py`, `tests/scripts/test_corpus_driver_resilience.py`. Expected: PASS.

### Task 35: The scorer driver (S§3, S§4)

**Files:** create `scripts/build_contact_scorer.py` (from the scaffolder's driver template: `PIPELINE`; `--split dev` (6 folds, the 19-point grid, the selection, the refit, the frozen scorer, the report) and `--split holdout --dev-report` (the S§3 guards); ids via `canonical_match_id`; the scorer sha256 in the tokens); the `DRIVERS` entry `build_contact_scorer` (`certification="rebaseline"`, `output_rel="docs/research/contact_scorer"`); tests `tests/scripts/test_build_contact_scorer.py`, `tests/scripts/test_parity_contact_scorer.py`.

- [ ] **Step 1:** Tests: the fold assignment, the selection, the held-out guards, the token change on a new `model.json`, a toy end-to-end run on a fake staged dataset. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run the two files, `tests/scripts/test_compute_contract.py`, `tests/scripts/hf_jobs/test_visibility_blind_gate.py`, `tests/scripts/hf_jobs/test_registry.py`, `tests/scripts/test_dataset_read_audit.py`, `tests/scripts/test_corpus_driver_resilience.py`. Expected: PASS.

### Task 36: The cross driver's scorer overrides (S§4)

**Files:** `scripts/build_cross_definition.py` (`--contact-scorer` and `--min-contact-probability` replace the two rule overrides; the scorer sha256 in the token); test `tests/scripts/test_build_cross_definition.py`.

- [ ] **Step 1:** Test both overrides and the token. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run. Expected: PASS.

### Task 37: Docs, verification, reviews and the first-form gate

- [ ] **Step 1: Docs:** the ADR-134 amendment (the credit step, the protocol); `docs/context/tracking-features.md` TF-79; `feature_glossary` (`confidence`); the cross spec release notice; the caller sweep.
- [ ] **Step 2: Verify** on the synced DGX: every changed test file, the repo-wide gates by name (`test_compute_contract`, `test_corpus_driver_resilience`, `test_scale_guard_registry`, `test_visibility_blind_gate`, `test_registry`, `test_headtohead`, `test_floor`, `test_dataset_read_audit`, `test_datasets_lock`, `test_datasets_fields`, `test_geometry_constant_enumeration`), the slow SB360 files, then the lint trio. Expected: PASS, exit 0.
- [ ] **Step 3:** `/final-review`; the A/B implementation reviews; answer both.
- [ ] **Step 4: Gate (first form).** On the owner's yes: commit locally (no push).

### Task 38: The dataset and commit 4 final form (DGX)

- [ ] **Step 1:** `git bundle` the first form to the DGX; check it out clean; run the materializer (8 workers).
- [ ] **Step 2:** With the owner's go: add the declaration uncommitted; `stage`; `datasets record gs_contact_truth`; `datasets id-kinds gs_contact_truth`.
- [ ] **Step 3:** Amend the declaration and the lock into commit 4; run `test_datasets_lock` and `test_datasets_fields`; on the owner's yes, force-push with lease over `9b9e8838`; CI green. Ledger the first-form sha.

### Task 39: The runs (DGX, on the final commit 4)

- [ ] **Step 1:** `check_gs_contact_truth` on all 64 units. A mismatch stops the cycle.
- [ ] **Step 2:** The harness's rule-mode development grid (the baseline) and the commit-3 v1 IDSSE/SkillCorner reference.
- [ ] **Step 3:** `build_contact_scorer --split dev`. No feasible point, or a point on the grid's edge: to the owner before any held-out run.
- [ ] **Step 4:** `build_contact_scorer --split holdout --dev-report`, once. A missed bar stops the cycle.
- [ ] **Step 5:** The harness's IDSSE/SkillCorner reports with the frozen scorer and `--reference`; read the trigger.
- [ ] **Step 6:** The three cross runs (§7.4) with `--contact-scorer` and `--min-contact-probability`; report run 3 to the owner (§7.3).

Result on `cb456270` (ledger): Steps 1–3 done; Step 3 found no feasible point (best right 0.335), so Steps 4–6 did not run. The owner ruled the diagnosis and the scorer v2 (S§ r7, approved 2026-11-16). Tasks 40–46 implement r7; the A/B implementation reviews stand in for a plan review (owner, 2026-11-16). Task 46 replaces Steps 1–6 on the new commit 4.

### Task 40: The two populations in the truth module (S§3)

**Files:** `scripts/_gs_truth.py` (`SUPPORT_RADIUS_M`, `track_supported`, the `sup_` counts of `score_next_touch` when `seq` has `supported`, the `sup_` keys and `supported_share` of `shares`, the required keyword `population` of `meets_bar` and `select_point`); `scripts/measure_ball_contact_validation.py` (the rule-mode GS grid adds the `supported` column from its frames, selects with `population="supported"`, reports both populations); tests `tests/scripts/test_gs_truth.py`, `tests/scripts/test_measure_ball_contact_validation.py`.

- [ ] **Step 1:** The S§7 support-flag tests (1.49/1.51 m, t2 ± τ inside and just beyond, no or non-finite S2 rows, a non-finite ball row skipped, two periods with the first row of period 2 flagged and never read), the `shares` keys with and without the counts, the two populations picking different points, the `KeyError`; the harness test that its grid carries both populations and selects on the supported rows. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run both files. Expected: PASS.

### Task 41: The control features (S§2.2)

**Files:** `silly_kicks/tracking/_contact_scorer.py` (`CONTROL_WINDOW_S = 0.4`; the seven control features computed in `option_table` from the deduped ball samples per period and the option rows; `FEATURE_NAMES` gains them, 16 in total); `silly_kicks/tracking/_ball_contact.py` only if the per-period deduped samples must be exposed to the option table; tests `tests/tracking/test_contact_scorer.py`.

- [ ] **Step 1:** The S§7 control-feature tests (a receiver who stops the ball, a ball passing a static player, both window bounds, a player with no rows after `k`, a non-finite ball sample in the after window, a held duplicate changes nothing) and the probe test (6 merged peaks, no missing probe feature). Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run the file. Expected: PASS.

### Task 42: The decoding (S§2.1)

**Files:** `silly_kicks/tracking/_contact_scorer.py` (`DECODE_WINDOW_S = 0.3`; `decide_options(..., *, decode_window_s)` and `credit_options(..., *, decode_window_s)` with the step-7 rule); `silly_kicks/tracking/_ball_contact.py` (`_finalize_contacts` passes the loaded scorer's `decode_window_s`; the frame-slice note in the `detect_ball_contacts` and `add_ball_contacts` docstrings); tests `tests/tracking/test_contact_scorer.py`, `tests/tracking/test_ball_contact.py`.

- [ ] **Step 1:** The S§7 decoding tests (0.30/0.31 s, equal probabilities, a suppressed peak unresolved, an uncredited peak still suppresses, a peak without options takes no part, window 0 decodes nothing, detector equals driver). Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run both files. Expected: PASS.

### Task 43: The v2 model and its artifact (S§2.3)

**Files:** `silly_kicks/tracking/_contact_scorer.py` (`ContactScorer.fit` with the S§2.3 `HistGradientBoostingClassifier` settings, the export to tree arrays without `count`/`gain`, the fit-time parity assert, the numpy tree walk, `decode_window_s` in `model.json`, the scikit-learn version for `metadata.json`, the five contract constants, the `load` refusals); tests `tests/tracking/test_contact_scorer.py`.

- [ ] **Step 1:** The S§7 model and artifact tests (parity 1e-12 with missing values, the fit-time assert on a corrupted export, two fits equal, round-trip, only declared arrays, no `count`/`gain`, the refusals incl. `decode_window_s`, the five constants each changing the fingerprint). Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run the file, `tests/tracking/test_ball_contact.py`, `tests/tracking/test_feature_contract.py`, `tests/tracking/test_geometry_constant_enumeration.py`. Expected: PASS.

### Task 44: The scorer driver on v2 (S§3, S§4)

**Files:** `scripts/build_contact_scorer.py` (the sequences stage loads `frames.parquet` and adds `supported`; the folds and the refit fit v2; `_score` passes `decode_window_s`; the reports carry both populations and `supported_share`; the selection uses `population="supported"`); tests `tests/scripts/test_build_contact_scorer.py`, `tests/scripts/test_parity_contact_scorer.py`.

- [ ] **Step 1:** Tests: the sequences stage output has `supported`; the dev report's grid rows carry both populations; the selection reads the supported rows; the held-out report likewise; the W1N1-vs-W2N2 parity on v2. Run. Expected: FAIL.
- [ ] **Step 2:** Implement. Run both files and `tests/scripts/test_compute_contract.py`, `tests/scripts/hf_jobs/test_visibility_blind_gate.py`, `tests/scripts/test_dataset_read_audit.py`, `tests/scripts/test_corpus_driver_resilience.py`, `tests/scripts/test_build_cross_definition.py`. Expected: PASS.

### Task 45: Docs, verification, reviews and the second-form gate

- [ ] **Step 1: Docs:** the second ADR-134 amendment (the v2 credit step, the decoding, the measured population); `docs/context/tracking-features.md` TF-79; `feature_glossary`; the cross spec release notice; the caller sweep (`decide_options`, `credit_options`, `meets_bar`, `select_point`).
- [ ] **Step 2: Verify** on the synced DGX: every changed test file and the named gates of Task 37 Step 2, the slow SB360 files, then the lint trio. Expected: PASS, exit 0.
- [ ] **Step 3:** `/final-review`; the A/B implementation reviews; answer both.
- [ ] **Step 4: Gate.** On the owner's yes: amend into commit 4 (the message per S§5) and force-push with lease over `cb456270` (bundled at `D:\Temp\cross-def\c4-cb456270.bundle`); CI green.

### Task 46: The runs (DGX, on the new commit 4)

- [ ] **Step 1:** `check_gs_contact_truth` on all 64 units. A mismatch stops the cycle.
- [ ] **Step 2:** The harness's rule-mode development grid (the baseline, both populations). The v1 IDSSE/SkillCorner reference (`~/cross-def/v1-ref`, `dd41724d`) is reused.
- [ ] **Step 3:** `build_contact_scorer --split dev`. No feasible point, or a point on the grid's edge: to the owner before any held-out run.
- [ ] **Step 4:** `build_contact_scorer --split holdout --dev-report`, once. A missed bar stops the cycle.
- [ ] **Step 5:** The harness's IDSSE/SkillCorner reports with the frozen scorer and `--reference`; read the trigger.
- [ ] **Step 6:** The three cross runs (§7.4) with `--contact-scorer` and `--min-contact-probability`; report run 3 to the owner (§7.3).

### Task 29: Commit 5 — weights, constants, memos, release prep and Gate 5

**Files:** the driver output (`docs/research/contact_scorer/`); the bundled weights `silly_kicks/tracking/_contact_scorer_weights/default/` (a byte copy); the `BallContactParams` defaults (S§2.5); `CROSS_START_X`; the `CrossDeliveryParams` defaults; the manifest root and entry (S§6); the SB360 re-adjudication for the default flip; the cross memos; the ADR-133/134 Evidence; the `<recall: Task 10>` placeholders; `CHANGELOG.md`, `_version.py`, `TODO.md`; the pin tests (`model.json` byte-equal to the output; the defaults equal the memos); `.gitattributes` marks `silly_kicks/tracking/_contact_scorer_weights/**` and `docs/research/contact_scorer/**` binary, like the other bundles, so `autocrlf` never changes the checksummed bytes.

- [ ] **Step 1:** The pin tests first. Expected: FAIL.
- [ ] **Step 2:** The constants, weights, memos and docs.
- [ ] **Step 3:** Verify: the named gates of Task 37 Step 2, the S§6 bundled-artifact gates and the doc gates; the lint trio. Expected: PASS, exit 0.
- [ ] **Step 4:** `/final-review`; the A/B re-review.
- [ ] **Step 5: Gate 5.** On the owner's yes: commit and push; CI green.

### Task 30: Release and handoff

- [ ] **Step 1:** The merge (`--merge`), the tag and the release, each on its own owner yes, after CI is green.
- [ ] **Step 2:** Send Part Deux the owed-retrain list (`docs/research/cross-definition/caller-sweep.md`) and the re-stage note (the xshot187 `_actions` must be re-converted with the new converters).
