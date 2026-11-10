# HFU reduce reads, job exit semantics, match-outcome pipeline — Implementation Plan

Spec: `docs/superpowers/specs/2026-11-04-hfu-reduce-reads-and-job-exit-design.md`. Executed inline; the ledger is
`.superpowers/sdd/2026-11-05-hfu-reduce-reads-and-job-exit-plan/progress.md` (every RED/GREEN result and
`Ruling:` line). The review rounds and the full task text of Tasks 1–27 are in git history.

## Global Constraints

- Branch `feat/minimal-hf-driver`, PR #299; one amend per change; the version stays `4.139.1` until the adoption
  commit, which releases `4.140.0` (owner 2026-11-10). Exception: the
  launch-7 adoption (Change 11) is a NEW commit on top of `a1a3b52f`, never an amend (spec §18.4).
- Lint pins (`ci.yml`): `ruff==0.15.7`, `pyright==1.1.409`, `pandas-stubs==2.3.3.260113`; lint at CI scope.
- Local runs: lint, targeted tests and `tests/scripts/hf_jobs` on its own; the heavy suites, the full pyright and
  the parity tests run in CI (owner rule). No edit of the working tree while a campaign driver runs (the
  dirty-tree code identity moves every stage generation).
- Failure-record `error` strings keep their exact format (Hyrum).
- Every HF launch, the amend, the force-push and the retrain need their own explicit owner yes.

## Done (Changes 1–7)

| Task | Spec | What |
|---|---|---|
| 1–2 | §3 | listed reads, the single-pass reduce, the reduce sites |
| 3 | §4.1–4.2 | `max_failures` is a reduce-time rule (`exceeds_tolerance`) |
| 4 | §4.3 | failure locus for both unit kinds |
| 4A | §11 | job liveness: `[alive]`, `children=`, the staggered deadline dump |
| 5 | §5 | `match_outcome_dependence` pipeline |
| 6–9 | §6–§9 | docs, gates, the HF measurement campaign, amend and retrain |
| 10–12 | §13.2 | numba core, `use_numba` at every site, `numba=` and `check_numba` |
| 13 | §13.3 | a FileSet read once per job; `[read]` lines |
| 14 | §13.4 | Hub timeouts; the `--plan` deadline |
| 15–16 | §13.5–13.6 | the 2-timeout STUCK rule; the log trail |
| 17 | §13.9 | docs, gates, resume |
| 18 | §14.10 | investigations I1, I2 |
| 19–21a | §14.2–14.4, §14.10a | recipes, corpus identity, public certification, optuna 5.0.0 |
| 22, 22a | §14.5, §14.10b | `DependenceModel.load(str)`; the stable back-line tie-break |
| 23–26 | §14.7–14.9, §14.13 | the open-data snapshot; progress, retrieve and verify lines; cards and docs |
| 27 | §14.12 | gates, snapshot staging, the I5 probe, amend (`22e0b65d`, CI green), launch 4 |

Task 27 Step 6 remains for after launch 5 (the launch on the Change 8 commit):

- [ ] **Step 6.** `retrieve` and `verify`; report 21/21, or each difference under spec §14.1/§14.6 (I3:
  `pool_coverage` 51,440 / 50,719 and `keep_pool` false; I4: the census rebaseline; F19 GO). Then the parameter
  check (spec §14.10a): save the block below as `<staging>/check_params_ulp.py` and run
  `python -I <staging>/check_params_ulp.py <staging>`. Expected: seven `ok` lines, exit 0. Paste the output into
  spec §14.10a and the ledger; any `FAIL` stops for the owner. `adopt` stops at its commit gate.

```python
"""The seven optuna-5.0.0 models: fresh best params within 3 ULP of the committed ones (spec 14.10a)."""
import json, pathlib, struct, sys
root = pathlib.Path(sys.argv[1]).resolve(); run = root.name
SEVEN = {"xshot_default": "xshot_occurrence-public/default", "xshot_position_only": "xshot_occurrence-public/position_only",
         "hub_xshot_sc_extended": "xshot_occurrence-sc_extended/default",
         "hub_xshot_sc_extended_position_only": "xshot_occurrence-sc_extended/position_only",
         "xcross_position_only": "xcross_attempt-public/position_only",
         "hub_xcross_sc_extended": "xcross_attempt-sc_extended/default",
         "hub_xcross_sc_extended_position_only": "xcross_attempt-sc_extended/position_only"}
bits = lambda v: struct.unpack("<q", struct.pack("<d", float(v)))[0]
def best(p):
    d = json.loads(p.read_text()); return d["candidates"][d["shipped_variant"]]["params"]
bad = 0
for aid, rel in SEVEN.items():
    inst, var = rel.split("/")
    c = best(root / "_shipped" / f"{aid}.metrics.json"); f = best(root / f"{run}-{inst}" / var / "metrics.json")
    ulp = max(abs(bits(c[k]) - bits(f[k])) for k in c) if c.keys() == f.keys() else -1
    ok = 0 <= ulp <= 3; bad += not ok
    print(f"{aid}: max_ulp={ulp} {'ok' if ok else 'FAIL'}")
sys.exit(1 if bad else 0)
```

## Change 8 — the launch 4 findings (spec §15)

Status: APPROVED (A r18, B r18); implemented. Each task is test-first; behaviour beyond the spec is out.

- **Task 28 (§15.2):** `launch.job_env` sets the exact §15.2 `PYTHONWARNINGS` string. RED:
  `test_job_env_hides_the_known_converter_warnings` (the string is present; in a job-mode subprocess the three
  raised from `scripts._sb_open_data` print nothing, while a `StatsBomb360 …` warning from another module and one
  unrelated warning still print).
- **Task 29 (§15.3):** done in the tree: `TrialPrinter.optuna_callback` and the xsuccess callback (9 tests). Add a
  `fold k/n` line to the win_probability calibration loop and the AST test
  `test_every_optuna_study_has_a_progress_callback` (non-vacuity: it finds the xsuccess call).
- **Task 30 (§15.4):** `driver._guarded_put(store, rel, data, *, sleep=time.sleep) -> bool` at the three writes. RED:
  one parametrized test over the three writes with an always-429 store: no exception, `contention=True`, the launch
  loop ends; shard keys: that launch is skipped; manifest: the line prints and the job is never charged;
  `_save_cache`: the step still launches with the cache unsaved (today it launches nothing). Plus a
  retry-then-succeed case; a non-429 error still raises.
- **Task 31 (§15.5):** `done_results` reads over 8 threads in sorted key order, with the read progress line; the
  final bootstrap line. RED: a 250-key read equals the serial read (order and values) with one listing and one
  read per file, prints throttled lines and one final line under a patched clock; a short bootstrap prints exactly
  one line.
- **Task 32 (§15.5a, §15.7):** xsuccess and pass_completion native pipelines on the win_probability pattern (the
  `compute-pipelines` skill).
  - `XSuccessObjective.evaluate_patch(..., fold_map=None)`, the serial path as the same composition; a per-fold
    result carries "skipped" and "no PR-AUC"; the fold children read `X`/`y` and the fold plan from job-local
    `.npy`/`.npz` files (the `_study_fold_task` pattern). `Stage.timeout_s` enters the `plan()` stage record
    (`_stages.py:1056`) and `_job_spec` reads it. Escalation is unchanged.
  - The `fit` stages of both print the §15.3 `fold k/n` lines.
  - Retirement edits: remove both trainers from `_LEGACY_UNMIGRATED` (`tests/scripts/test_compute_contract.py`),
    drop the `TODO.md` ADR-128 Cycle 2 row, and delete `scripts/hf_jobs/adapter.py`, its stage and its tests (owner
    2026-11-08); the registry `adapter` field and every reference go with it, including the adapter runner
    (`liveness._run_module`, its `"adapter"` dump offset and docstring, and its tests in `test_job_liveness.py`
    and `test_write_confirm.py`) and the spec lines that describe it (§11.1, §11.4, §12.2–12.3).
  - RED: fold-parallel equals serial bit for bit, including a skipped fold; native vs legacy model bytes for both;
    W1N1 = W2N2; the real xsuccess `PIPELINE` plan through `_job_spec` gives 14,400 s (a hand-built stage record
    is not enough) and a stage without it 7,200 s;
    `test_native_fit_stages_print_fold_lines` (both trainers); `test_campaign_recipes`.
- **Task 33 (§15.6):** ADR-128 items, the howto, `CHANGELOG.md`; spec §11 drops the `adapter` liveness role
  (deleted with the adapter in 8e).
- **Task 34:** local gates; A/B implementation review; `/final-review`; the owner gates (amend, force-push); CI
  green; owner go: preflight and `campaign … launch` on the new commit (all 21 instances and every stage rerun
  under new run_keys; about 5 h); then Task 27 Step 6.

## Change 9 — the launch 5 findings (spec §16)

Status: APPROVED (A r18, B r17; CONSIDER notes applied). Each task is test-first; behaviour beyond the spec is out. Amended into the cycle commit.

- **Task 35 (§16.2):** `train_xsuccess._fit_task(rows_path, kind, tr, te, family, params)` (module-level, per-process
  row cache) and `_fit_map(pool, pooled, splits, family, params)`; the `fit` stage `inner_parallel=True`,
  `inner_width=3` (§16.7), `timeout_s=14400`, the final fit submitted first. RED: a pool that completes in reverse gives the serial out-of-fold
  vector, metrics and model bytes; the real plan record of `fit` carries 14,400 s and width 3; the pool path prints
  n_splits + 1 `fit k/n` lines; the slow parity test asserts `inner_out_of_order` on `fit`.
- **Task 36 (§16.3):** one reader for `done_results` and `reduce_stage` in the caller's order. RED: a 250-unit
  reduce with a `reduce_order` writes the same `_output` bytes as today, reads over the pool and prints the `[read]`
  lines under a patched clock.
- **Task 37 (§16.4):** (A) the run loop passes cached plans of reduced instances to the step; (B) one recursive
  listing per `BucketStore`; (C) the read memo with write-through. RED: a reduced instance is planned once over
  two steps; a counting fake API lists once per plan with an unchanged plan JSON; `None` memoized; an upload is
  visible; an empty-prefix store lists one directory. After GREEN: rerun `count_plan.py` on the 4 instances and
  record the figures; none may exceed today's (47, 62, 65, 25), else stop for the owner.
- **Task 38 (§16.5):** `compute_arms(units)`; `campaign.py:185` passes the units; correct the `arm_policy.py:10-13`
  docstring and the `test_arm_policy.py` fixture stem `sportec__DFL-MAT-J03WMX` (no lock has a `sportec__` stem).
  RED: every unit of every locked visibility-gated dataset keys equal to the trainers' ref keys; the public corpus
  arm has 17 matches.
- **Task 39 (§16.6, §16.8):** the xsuccess `[[jobs]]` block first in `cycle1.toml` (RED: the first planned launch is
  xsuccess `refs`); ADR, howto, CHANGELOG.
- **Task 40:** local gates; A/B implementation review; `/final-review`; owner gates (amend, force-push); CI green;
  owner go: preflight and launch 6 of `cycle1.toml`; retrieve, verify, adopt; Task 27 Step 6.

## Change 10 — the launch 6 verify findings (spec §17)

Status: APPROVED (A r21, B r20); implemented and launched (launch 7). Each task is test-first; behaviour beyond the spec is out. Amended into the cycle commit.

- **Task 41 (§17.2):** both trainers `all_public=False`. RED: the AST gate over `scripts/train_*.py` (it finds the 2
  calls and fails today); on launch-6-shaped digest metrics (raw `pergame` pairs hashed, n 17, providers `[idsse,
  skillcorner]`) verify's identity check passes and `rehydrate_public_corpus_ids` returns the canonical pinned pairs.
  Invert `test_public_run_records_ids_and_public_reproducibility` (a public run records the digest, no ids), the
  `test_expect_variant_wiring.py` docstring and both trainer comments.
- **Task 42 (§17.3):** `VariantBuild.in_sample_gate`; xCross `default` `False`; `floor.certify(in_sample_gate=)`;
  `adopt._certify` passes it. RED: the launch-6 xCross numbers certify with the gate off, fail with it on; the
  registry test pins xCross `default` as the only build off.
- **Task 43 (§17.4):** `DriverSpec.tolerance` (census) and the tolerance comparison in `adopt._rebaseline`. RED: the
  REAL committed `census.json`/`ranking.parquet` against the launch-6 fresh census pass with `within_tolerance`
  (its pass counters differ and are not compared); `lo` +1e-4, `n_boot_effective` +2, a changed ranking row, a changed
  verdict `reason` and a fresh `n_failed` 1 each fail.
- **Task 44 (§17.5):** threaded `retrieve`/`verify` (4) with per-instance open-data caches; `launch_batch = 16`. RED:
  with a per-instance hook that completes in reverse, the returned verify report and the retrieve result keep instance
  order; two statsbomb_open instances get different cache folders; the cycle1 test pins `launch_batch` 16.
- **Task 45 (§17.7):** ADR, the rehydrate docstring, howto, CHANGELOG.
- **Task 46:** local gates; A/B implementation review; `/final-review`; owner gates (amend, force-push); CI green;
  owner go: preflight and launch 7; retrieve, verify, adopt; Task 27 Step 6.

## Change 11 — adopt the launch 7 artifacts (spec §18)

Status: Tasks 47-50 APPROVED (A r25, B r24); Tasks 51-53 APPROVED (A r27, B r26); Task 54 APPROVED (A r29, B r28). Each task is test-first; behaviour beyond the
spec is out. A NEW commit on top of `a1a3b52f`.

- **Task 47 (§18.2):** in `adopt`, call `_set_manifest_commit` only when `a.commit_field` is set. RED: the toy repo
  gains an artifact without `commit_field` and `training_commit` (`toy_nocommit`, wheel destination); `adopt` adopts
  it, its manifest entry is unchanged byte for byte, and `observed.json` has its entry. Guard: a test over
  `docs/artifacts/manifest.toml` asserts `training_commit` is present exactly when `commit_field` is.
- **Task 48 (§18.3):** the root-clean entry check and the rollback, on the real toy git repo. RED: (a), (b), (d), (e)
  and the idempotence change; regression pins (they pass today and must keep passing): (c), (f).
  (a) a `_bundle` monkeypatched to write one file, delete one tracked file and then raise leaves `git status` empty
  (the deleted file is back) and re-raises the same error;
  (b) the toy tree gets a real uncommitted edit and an untracked file, `git` injected clean: `adopt` refuses at entry
  ("the root tree is not clean") and both survive byte for byte;
  (c) the same edit and untracked file, `git` injected dirty: the existing entry refusal; both survive byte for byte;
  (d) a restore that fails raises `AdoptionRollbackError` naming the path, chained from the original error;
  (e) a `KeyboardInterrupt` from `_bundle` is rolled back and propagates;
  (f) a `KeyboardInterrupt` from `_local_gate` (`gate=True`) propagates and leaves the adoption's writes in place;
  the existing `test_adopt_fails_on_a_path_outside_the_allowlist` also leaves `git status` empty;
  idempotence (`test_adopt_writes_only_allowlisted_paths_and_is_idempotent`, today its second `adopt` call at
  lines 279-281 runs on the uncommitted adoption): the test commits the first adoption, then the second run over the
  same staging dir returns `written == []` (the write phase is a fixed point); a run over the uncommitted adoption refuses ("the root tree is not clean")
  and writes nothing.
- **Task 49 (§18.5):** ADR-128 amendment, howto step 8, CHANGELOG, the `adopt.py` docstring. The `manifest.py`
  docstring and the `manifest.toml` header comment are edited after the Task 50 adoption run (Task 50 step 8): `adopt`
  imports `manifest.py` and writes `manifest.toml`.
- **Task 50 (§18.4):** in order:
  1. Local gates: ruff 0.15.7 check and format at CI scope; pyright on the changed files; the touched tests.
  2. A/B implementation review of Tasks 47-49.
  3. Record `sha256sum scripts/hf_jobs/adopt.py` in the ledger; confirm `git diff --stat a1a3b52f` lists no module
     that `adopt` imports other than `adopt.py`.
  4. `mkdir -p D:/Temp/hfu-change11 && cp scripts/hf_jobs/adopt.py D:/Temp/hfu-change11/adopt_fixed.py` (owner
     rule: temp files under `D:\Temp`);
     its sha256 equals the recorded one.
  5. `git stash push -u -m change11`; then `git status --porcelain` is empty and `git rev-parse HEAD` is
     `a1a3b52fba07be3c21131c11df765cf7105524b3`.
  6. From the repo root (cwd = repo root; `git_provenance` reads the cwd), with `.venv-hfjobs`:
     load `D:/Temp/hfu-change11/adopt_fixed.py` with `importlib.util.spec_from_file_location("scripts.hf_jobs.adopt", ...)`,
     register it in `sys.modules["scripts.hf_jobs.adopt"]` before any other `scripts.hf_jobs` import, execute it, and
     call `adopt(load_campaign(Path("scripts/hf_jobs/campaigns/cycle1.toml")), <HEAD sha>,
     staging=Path("D:/Development/_staging/cycle1-a1a3b52"), gate=False)`; keep its JSON report.
  7. Check the adoption: every changed path is allowlisted; each adopted artifact with a `commit_field` has
     `training_commit = a1a3b52f...`; `match_outcome` and `expected_passing` entries are unchanged.
  8. `git stash pop` (the stashed change touches no path the adoption writes); then the `manifest.toml` header and
     the `manifest.py` docstring edits (§18.5), and an A/B delta check of those two edits.
  9. `/final-review`; owner commit gate (one new commit); owner push gate (no force); CI green; Task 27 Step 6.
- **Task 51 (§18.6):** `canonical_match_id` and the trainer. RED: `("gradientsports", "gradientsports-10502")` ->
  `"10502"`; `("sportec", "idsse-DFL-MAT-J03WMX")` -> `"DFL-MAT-J03WMX"`; `("idsse", "DFL-MAT-J03WMX")` and
  `("gradientsports", "10502")` unchanged; for every `xcross_probe` lock unit, `canonical_match_id(source_provider,
  game)` has no `f"{source_provider}-"` prefix; `_cohort_files` over a stubbed staged match `gradientsports-10502`
  records `matches == [["gradientsports", "10502"]]` and `match_groups` keyed `"10502"` (`_load_match` and
  `_write_probe_sample` stubbed).
- **Task 52 (§18.6):** `canonical_probe_ids`, the finalized copy, `_bundle`, `observed` and `publish`. RED:
  `canonical_probe_ids` maps the launch-7 per-game form to the committed form (pairs and fold keys), leaves the
  committed form and a dashed canonical id unchanged, leaves a metrics without the fields unchanged, and does not
  mutate its input. The toy `adopt` (a wheel and a Hub artifact whose staging `metrics.json` has per-game probe ids)
  writes canonical ids into the wheel `metrics.json` and into the Hub artifact's `observed.json` `metrics_summary`,
  and the observed `metrics.json` hash equals the hash of `<staging>/_final/<instance>/metrics.json`; a toy `publish`
  (a stub publisher capturing the uploaded dir) uploads that finalized dir, canonical ids included.
- **Task 53 (§18.6):** docs (ADR-128 item 36, howto step 8, CHANGELOG); local gates; A/B implementation review; then:
  1. discard the first run's output: each path in the `written` list of `D:/Temp/hfu-change11/adopt_run.log`
     (tracked: `git checkout HEAD -- <path>`; new: removed); `git status` then lists exactly the Change 11 paths;
  2. Task 50 steps 3-8 with two modules: record the sha256 of `scripts/hf_jobs/adopt.py` and
     `scripts/_corpus_identity.py`; `git diff --name-only a1a3b52f -- '*.py'` lists no other module that `adopt`
     imports; copy both to `D:/Temp/hfu-change11/` and check both sha256; `git stash push -u`; status empty and HEAD
     `a1a3b52f...`; the runner registers `scripts._corpus_identity` and then `scripts.hf_jobs.adopt` from the copies
     (`importlib.util.spec_from_file_location`) before any other `scripts` import; `gate=False`;
  3. check: `tests/test_bundled_probe_ids_are_committed_ids.py` passes on the adopted bundles, and the Hub xCross
     `observed.json` entries carry canonical ids; `git stash pop`; the §18.5 edits; the A/B delta check.
- **Task 54 (§18.7):** the two tests rewritten as §18.7 states (RED: on the HEAD bundles, without `metrics.json`,
  `test_a_bundle_metrics_is_its_committed_cv_record` fails; it passes on the adopted bundles); after the re-adoption
  run (Task 53 step 3), with the §18.5 edits: the three manifest entries lose `cv_reference` and its comment, and
  `docs/artifacts/cv_reference/xsuccess.json` is removed; `tests/test_artifact_manifest.py` and
  `tests/scripts/hf_jobs/test_adopt.py` pass on the adopted tree; the step 8 A/B delta check covers these edits.
  `test_rebaseline_tolerance.py` takes relative `n_boot_effective` deltas (RED: it failed over the adopted census);
  with the post-run manifest edits, `match_outcome` and `expected_passing` gain `commit_field` +
  `training_commit = a1a3b52f...`, and `tests/test_artifact_manifest.py::test_wheel_artifact_matches_its_expectations`
  passes for both (RED: with the entries but an old commit value it fails); the pairing guard's docstring says
  "declares". Pin (B IMPL-27): `_finalize` over an existing `_final/<instance>` with a stray file removes it.
