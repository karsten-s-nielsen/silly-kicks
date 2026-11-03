# HFU reduce reads, job exit semantics, match-outcome pipeline — Implementation Plan

Spec: `docs/superpowers/specs/2026-11-04-hfu-reduce-reads-and-job-exit-design.md`. Executed inline; the ledger is
`.superpowers/sdd/2026-11-05-hfu-reduce-reads-and-job-exit-plan/progress.md` (every RED/GREEN result and
`Ruling:` line). The review rounds and the full task text of Tasks 1–27 are in git history.

## Global Constraints

- Branch `feat/minimal-hf-driver`, PR #299; one amend per change; the version stays `4.139.1`.
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

Status: DRAFT r21. Each task is test-first; behaviour beyond the spec is out. Amended into the cycle commit.

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
