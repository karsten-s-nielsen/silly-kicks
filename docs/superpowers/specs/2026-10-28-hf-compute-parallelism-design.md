# HF compute parallelism — Cycle 1: multi-core fits + dispatch concurrency 25

**Status:** design v6 — **APPROVED by A and B (r6, 2026-10-28)**. Reviews:
`D:\Development\_reviews\2026-10-28-hf-compute-parallelism-spec-r6-{A,B}.md`. After approval, A's r6 OPTIONAL
considers 20 (reject is within-step) and 21 (floor-trade wording) were folded into §D — wording only, no design
change (so the live sha differs from the approved-at-vote sha `9cb06fc4`); no re-review triggered. A later
plan-r2 fold tightened §A's lead sentence to match its own per-type-lever bullets (the blanket-override wording
could be read as setting OMP for the xgb worker too) and named the `SILLY_KICKS_RUNTIME_THREADS` transport —
wording/consistency only, the per-type lever was already the design; no re-review triggered.

**Split (owner-approved).** This cycle = **thread core + dispatch concurrency 25** only. Cycle 2 (separate
spec) = image bake (per-job install → ADR-128 P1) → re-profiled `startup_s` → slot-filling sharding.

## Intent

The refit fits run on **1 of 8 vCPUs** because `_thread_pin.deterministic_env` pins
`OMP/OPENBLAS/MKL/VECLIB/NUMBA = "1"` for every unit (ADR-128 INV-2 — byte-identity AND anti-oversubscription).
A solo `HistGradientBoostingRegressor` fit (ghost `fit-assemble`) overran the ~133 min job wall for this
reason. The byte-identity half is **moot for the fits** (proven), so a fit can use its full vCPU share — same
artifact, shorter wall, lower cost.

## Determinism evidence (committed as the opt-in gate)
Measured 2026-10-28 on x86 (the HF `cpu-upgrade` production arch; the committed gate runs on x86 in CI):
- `HistGradientBoostingRegressor` (ghost): `predict` byte-identical across `OMP_NUM_THREADS` 1 vs 8. Lever =
  `OMP_NUM_THREADS` ONLY (the BLAS vars `OPENBLAS/MKL/VECLIB` were NOT varied — they stay `"1"`; the ghost
  assemble runs BLAS `cho_factor`/`einsum` at `_ghost_gk.py:1596,1638` whose multi-threaded reduction order this
  evidence does not cover — r5 B38).
- `xgboost(tree_method="hist")`: byte-identical across `n_jobs` 1 vs 8. Lever = the runtime `n_jobs` arg.

## Goals
1. **Full vCPU per job, oversubscription-safe, opt-in.** A `thread_safe` stage's worker launches with
   `threads = max(1, vcpu // (workers × inner_workers))` (so `threads × units_in_flight ≤ vcpu`). A solo fit
   (`workers=inner=1`) gets all cores → ghost `fit-assemble` ~133 min → ~20–30 min. Non-`thread_safe` stages
   unchanged (pinned 1).
2. **Dispatch concurrency → 25**, shard ceiling unchanged (r3 A/HCP-SPEC-12, B17). `concurrency_cap` today feeds
   BOTH the dispatch budget (`campaign.py:647`) and the planner shard ceiling (`planner.py:112,123`). Decouple:
   raise only the **dispatch budget** to 25 (probe the HF ceiling); keep the planner **shard ceiling at 20** so
   no stage re-shards (sharding is Cycle 2). Handle `run_job` rejection past the account limit without halting
   the instance; record the observed max concurrency in the campaign summary.
3. Byte-identity (ADR-128 P3, same-commit, every file) preserved and proven across all reachable thread counts
   on **the production host**. The fits run on **HF `cpu-upgrade` (x86/amd64)** — the one prod flavor (ADR-128
   INV-4) — so the identity gate runs on x86 in CI (`ci.yml:49-55`, same arch as HF). No aarch64/DGX leg: the
   DGX is a possible **backup** (it must work, but is never a byte-identity benchmark), NOT the
   production/benchmark host. If a fit is ever run on a different arch, that is a new host, proven then (P15:
   cross-host diffs are adopted environment deltas; cross-host byte-equality is neither required nor achievable).
   CI proves thread-independence on the CI runner's x86 microarch; the HF CPU may dispatch different SIMD
   kernels, so the on-HF-CPU P3 confirmation is the first HF refit's committed (W,N)+thread parity compare (same
   arch family, same commit) — r5 B40. The DGX-backup "must work" is a run-correctness intent, NOT a committed
   byte-identity check (no plan step, by design — r5 B41).

## Non-goals / anti-goals
- **No sharding, no image bake** (Cycle 2). **No job-wall/timeout change.**
- No model algorithm/hyperparameter change. xgb `metadata.json` `n_jobs` stays serialized as the **constant 1**
  (r3 B19) → artifact byte-identical, no rebaseline diff. The runtime thread budget reaches the fit via a
  **backward-compatible, defaulted keyword** on the model `fit()` (default → serialized `1`; a documented,
  additive public-API change — r4 B20), NOT an env-read inside `fit` (which would be a Hyrum hidden dependency).
- **No owner-tier raw data committed.** Identity-test fixtures are synthetic at real shape/scale or public-tier,
  with provenance recorded (r3 B16; redistribution policy; CHANGELOG:1428).

## Design

### A. Opt-in thread budget, injected ONLY at a thread_safe worker launch
Default/shared/reduce(`executor.py:274`)/adapter/default-pool(`executor.py:371`) env stays
`deterministic_env()` (all thread vars `"1"`). The executor, **only** when launching a worker for a
`thread_safe` stage, applies the Goal-1 budget to that subprocess **per the stage's declared lever** (below) —
NOT a blanket override of every thread var:
- `thread_lever="omp"` (HistGB) stages set `OMP_NUM_THREADS` to the budget ONLY; `OPENBLAS/MKL/VECLIB` stay
  `"1"` (HistGB parallelism is OpenMP `prange`, governed by `OMP_NUM_THREADS`; the ghost worker also runs BLAS
  `cho_factor`/`einsum` (`_ghost_gk.py:1596,1638`) whose multi-threaded reduction order the HistGB evidence — OMP
  varied only — never covered, so keep BLAS serial for P3 — r5 B38). `NUMBA_NUM_THREADS` stays `"1"` (numba opts
  in only via its own gate; DAS parallelism is its explicit `n_threads` arg, not the env).
- `thread_lever="n_jobs"` (xgb) stages keep `OMP/OPENBLAS/MKL/VECLIB` at `"1"` (so the xgb worker's numpy/BLAS
  prep — an axis the xgb determinism evidence never covered — never multi-threads); the budget is exported as
  `SILLY_KICKS_RUNTIME_THREADS` and read trainer-side (NOT inside `fit`) into the runtime `n_jobs` at fit time
  (serialized value stays 1).
- `vcpu` source = the job's `_cpus()` cgroup view (`executor.py:229`), reconciled with the planner's `vcpu`.
- Record the applied per-worker thread count in the **unit ledger** (`.ledger.json`, `_stages.py:742` — the D3
  parity-excluded, per-worker location), NOT in `env_provenance` (which is not stripped and is built before the
  budget exists, `_stages.py:1077`) — r3 B18, A/HCP-SPEC-10.

### B. Fail-closed thread-safety (opt-in)
`thread_safe` defaults **False** (pinned 1). A stage opts in ONLY with a committed test proving its artifact is
byte-for-byte identical across thread counts `1 .. vcpu//(w·i)` at a **non-vacuous** fixture (below). Ghost
HistGB fit + xgb fit opt in; everything else stays pinned.

### C. xgb serialization invariance (r3 B13 closed, B19)
Serialize xgb `n_jobs` as the constant `1` (unchanged schema, no rebaseline diff); set the runtime thread
budget via the backward-compatible defaulted `fit()` keyword (see Non-goals). `metadata.json` is then identical
across W/host → P3 holds. The identity test compares `metadata.json` + the booster bytes across thread counts.

### D. Dispatch concurrency 25 (r3 B09, r4 B09)
Two overflow modes past the account limit, both must NOT halt the instance (Goal 2):
- **Reject (within-step):** `run_job` raises → the unit is retried within budget; no stateless re-derivation
  is needed, because the dispatch budget `= dispatch_cap − running` (`campaign.py:647`) already self-throttles
  as slots free. (A reject leaves no JobInfo, so only the never-started-timeout path below uses the stateless
  re-derivation — r6 consider 20.)
- **Queue:** the exemption applies ONLY to a job whose **container never started** (`JobInfo.started_at` unset,
  `ledger.py:58`) and then timed out — a true dispatch overflow: reduce the live dispatch budget, reschedule
  the unit, NOT counted toward `error_resumes`/attempts. A job whose container **STARTED** (`started_at` set)
  and then timed out — INCLUDING a pre-unit hang in git-fetch/uv-sync/dataset-fetch (`launch.py:193-210`) that
  wrote no unit `.ledger.json` — is a **running-timeout** and COUNTS toward `error_resumes` (exactly its case,
  `planner.py:31-32`), keeping today's bounded behavior (r5 B22). So the signal is container `started_at`, NOT
  unit-ledger presence. **Fail-safe:** `started_at`'s meaning (does HF set it at container start? does queued
  time count against `timeout_s`?) is verified at implementation; on any ambiguity treat the job as **started**
  (counts) — never exempt on doubt.
- **Bounded + stateless (r5 HCP-SPEC-16, B22):** the dispatch back-off has a floor `dispatch_floor`
  (≥ `shard_cap`, so a stage is never throttled below its own parallelism — stages run in waves across
  stateless steps, so the floor protects a stage's wave, not a literal single job; r6 consider 21). Trade
  (**per the B24 owner ruling, RECORDED 2026-10-28: halt + surface**, not degrade-to-waves; r5 B39): if HF's
  true ceiling is below `shard_cap`, the instance halts after `M` rather than running in slow waves — surfaced,
  not silently stuck. The controller is stateless (`campaign.py:632`),
  so the
  back-off is **re-derived each step** from the observable job history (count of consecutive never-started
  timeouts for the campaign), never held in memory. After the floor is reached, `M` consecutive never-started
  timeouts **surface/halt** the instance with the unit named — the replacement for the `error_resumes` guard
  the exemption bypasses (ADR-128:147). No unbounded silent reschedule.
- **Probe-first:** the controller discovers the real ceiling by backing off on the first overflow (reject or
  queued-timeout); the observed max accepted concurrency is written into the campaign summary.
Validation tests both paths: a reject degrades; a queued-timeout backs off without halting or consuming an
attempt.

The two TOML keys (r4 B21): `dispatch_cap` (the live dispatch budget; `cycle1.toml` → 25) and `shard_cap` (the
planner per-stage shard ceiling; default = `dispatch_cap`). `Constraints` carries both; the planner
`min(n, shard_cap, …)`, dispatch uses `dispatch_cap`. **`cycle1.toml` MUST set `shard_cap = 20` explicitly**
(the default `= dispatch_cap = 25` would re-shard — HCP-SPEC-12); a gate asserts the committed `cycle1.toml`
pins `shard_cap=20` with `dispatch_cap=25` (r5 consider 18).

### E. Cost measured at the real thread budget (r3 B08)
`_profile` runs the stage at the thread budget it will use in production, so the recorded per-unit
cost/efficiency is not thread-confounded (stale 1-thread cost mis-sizes `workers`/`timeout`). A Validation test
asserts the profile uses the budgeted thread count.

## ADR-128 amendment (r3 B01 — full sweep)
Amend INV-2: "every unit is single-threaded UNLESS its stage is `thread_safe` with a committed
cross-thread-count byte-identity test; a `thread_safe` worker runs at `vcpu//(workers·inner_workers)` threads
via its declared lever ONLY — the omp lever raises `OMP_NUM_THREADS`, the n_jobs lever a runtime arg;
`OPENBLAS/MKL/VECLIB` and `NUMBA` stay `"1"` on every path (B38); default/shared/reduce/adapter env stays
single-threaded." Update every surface: `ADR-128` design+ADR;
`AGENTS.md:133`; `docs/howto/compute-pipelines.md`; the compute-pipelines skill; `scripts/_stages.py:6/10/117`;
`scripts/hf_jobs/executor.py:9/85`; `scripts/hf_jobs/adapter.py:8`; `scripts/_thread_pin.py:5/24`;
`scripts/hf_jobs/scaffold.py:30`. Keep `test_executor.py:129-130`, `test_thread_pin.py:8`, `test_det_env.py`,
`test_adapter.py:32` asserting `"1"` for the DEFAULT env; add budget assertions only on the thread_safe path.

## Validation
- Unit: the budget (`vcpu//(w·i)`, ≥1, never oversubscribing); injection only on the thread_safe path (all four
  default-env tests still see `"1"`); the dispatch-vs-shard-cap decoupling (shard ceiling stays 20 at
  dispatch 25).
- Identity tests on a **synthetic fixture at real shape/scale** (never owner-tier raw; provenance recorded),
  with a **non-vacuity precondition test** (AGENTS.md:126) proving the fixture is large enough that the
  multi-thread path genuinely engages (else the identity test is vacuous): HistGB artifact (npz+metadata)
  byte-identical across thread counts; xgb booster+metadata byte-identical across the runtime `n_jobs`. The
  gate runs on x86 in CI (ubuntu/windows, `ci.yml:49-55`) — the HF `cpu-upgrade` production arch; no aarch64/DGX
  leg (DGX is a possible backup, never the benchmark — P15).
- The ADR-128 (W,N) parity gate gains a THREAD axis for thread_safe stages.
- Tests for Goal 2: rejection degrades; shard ceiling stays 20 at dispatch 25 (asserted against the committed
  `cycle1.toml`); a **never-started** timeout backs off without counting; a **started** job with no unit record
  that times out **COUNTS** toward `error_resumes` (r5 B22); back-off is re-derived from job history across
  stateless steps and surfaces/halts after `M` at the floor. And §E (profile at the budgeted thread count).

## Left to the builder
- Which stages opt in after their identity test passes (empirical).
- The synthetic-fixture generator at real shape/scale (reproduces the parallel path without real positions).
- The exact worker-launch env-override mechanism, keeping `deterministic_env()` intact.

## Interaction / follow-on
Branches off `feat/controller-step-reads` (`8ff73ee`, PR #296 unmerged). Unblocks the ghost_outfield + `full`
refit. **Cycle 2** (separate spec): image bake (+ provenance fail-closed if installed env ≠ `uv.lock`, ADR-128
P1) → re-profile `startup_s` → slot-filling sharding with the `(n_jobs−1)·startup_s` cost bound.
