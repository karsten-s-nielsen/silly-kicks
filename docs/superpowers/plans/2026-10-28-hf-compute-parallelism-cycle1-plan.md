# HF compute parallelism — Cycle 1 Implementation Plan (v4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (owner runs inline; no subagents
> per CLAUDE.md). Steps use `- [ ]`. TDD each step.

**Goal:** Let opted-in, thread-count-deterministic fit stages use their full vCPU share (byte-identical
artifacts), and raise dispatch concurrency to 25 without re-sharding — removing the ghost_outfield fit wall
overrun and cutting wall/cost.

**Architecture:** per-job thread budget injected only at a `thread_safe` stage's worker launch
(`threads = vcpu // (workers·inner_workers)`); `concurrency_cap` split into `dispatch_cap`(25)/`shard_cap`(20);
xgb runtime threads via a defaulted `fit()` kwarg (serialized `n_jobs` stays constant 1); fail-closed opt-in
behind committed per-host cross-thread-count byte-identity tests on synthetic fixtures.

**Spec:** `docs/superpowers/specs/2026-10-28-hf-compute-parallelism-design.md` (v6, A+B APPROVED).
**Branch:** `feat/hf-compute-parallelism` (off `8ff73ee`).

## Global Constraints
- **ONE commit per cycle (CLAUDE.md).** Tasks build on one working tree; there are **no per-task commits**.
  The single commit happens once at the cycle gate, after the whole-branch A/B impl review AND explicit owner
  approval. (r1 HCP-PLAN-01.)
- ADR-128 byte-identity (P3, same-commit, per host) holds on every path. No job-wall/timeout change. No
  sharding/image bake (Cycle 2). No owner-tier raw data committed (synthetic fixtures only).
- Default/shared/reduce/adapter worker env stays `deterministic_env()` (all thread vars `"1"`); budget override
  only on the `thread_safe` worker-launch path. The omp lever raises `OMP_NUM_THREADS` ONLY; `OPENBLAS/MKL/VECLIB`
  + `NUMBA_NUM_THREADS` stay `"1"` on every path (BLAS serial for P3 — B38). Fail-closed default
  `thread_safe=False`.
- `vcpu` has ONE source: the job's `_cpus()` cgroup view (`executor.py:229`); the planner's `vcpu` reconciles
  to it.
- Task order (deps): T1, then T2 → T3 → T4 → T5; T6, T7, T8 independent of T4/T5 (T8 ← T2). T4 ← T2+T3; T5 ← T4.

## Verification (how to run, per step)
- Unit/local: `.venv/Scripts/python.exe -m pytest <path> -q`. hf_jobs tests need `.venv-hfjobs`
  (`hf_hub>=2.2`). Lint `uvx ruff@0.15.7 check silly_kicks/ tests/ scripts/` + `--format`; `pyright`.
- Identity tests run on x86 in CI (`ci.yml:49-55`) — the HF `cpu-upgrade` production arch. No aarch64/DGX leg
  (DGX is a possible backup, never the benchmark — P15).

## Review Focus (input classes the tests must exercise)
- Container STARTED then timed out, no unit ledger (pre-unit hang) → COUNTS toward `error_resumes`. [T6]
- `JobInfo` MISSING the `started_at` attribute vs set-but-None → must NOT exempt (treat as started). [T6]
- A `thread_safe` stage with `inner_workers>1` (workers=1, `executor.py:317`) → `threads×inner ≤ vcpu`, ≥1. [T2]
- A reject past the limit → degrades, backs off, no halt. [T6]
- A synthetic fixture with `n_features < budget` OR `n_samples < budget` (a parallel loop with idle threads) →
  the non-vacuity precondition fails loudly (observes work distribution, not OMP-config thread count). [T4]
- `cycle1.toml` missing an explicit `shard_cap` → the caps gate fails (default would re-shard). [T1]

---

## Task 1: Decouple dispatch_cap / shard_cap
**Files:** `scripts/hf_jobs/planner.py` (Constraints), `scripts/hf_jobs/campaign.py:56,63,122,647`,
`scripts/hf_jobs/campaigns/cycle1.toml`, `docs/howto/hf-jobs-training.md:50`, tests `test_planner.py:34-36`,
`test_campaign.py:57/65/76/294/584`, `tests/scripts/hf_jobs/_step_world.py:108` (constructors), new
`test_cycle1_caps.py` (B32).
**Produces:** `Constraints.dispatch_cap:int`, `Constraints.shard_cap:int` (default `=dispatch_cap`); planner
`min(n, shard_cap, …)` (:112/:123); dispatch budget uses `dispatch_cap` (:647); `load_campaign` reads both.
- [ ] S1: failing test — `plan_stage(shard_cap=20, dispatch_cap=25)` caps `n_jobs` at 20; dispatch budget sees 25.
- [ ] S2: run → fails (single `concurrency_cap`).
- [ ] S3: split the field; thread both through `Constraints`/`load_campaign`.
- [ ] S4: `cycle1.toml` → `dispatch_cap=25`, `shard_cap=20`; gate (`test_cycle1_caps.py`) asserts the committed
  toml pins both (missing `shard_cap` → fail). Run → green.

## Task 2: Per-job thread budget, opt-in injection
**Files:** `scripts/_thread_pin.py`, `scripts/hf_jobs/executor.py` (:274/:307/:317/:371), `scripts/_stages.py`
(`thread_safe` on `Stage` AND `ItemStage`, readable by T5's gate; ledger write :742), tests `test_thread_pin.py`,
`test_executor.py`, `test_det_env.py`.
**Produces:** `thread_safe:bool=False` + `thread_lever:Literal["omp","n_jobs"]` on both stage types;
`thread_budget(vcpu, workers, inner)->int` = `max(1, vcpu//(workers*inner))`. The executor, ONLY when launching
a `thread_safe` worker, injects the budget PER THE STAGE'S LEVER (spec §A — not a blanket OMP set):
`thread_lever="omp"` (HistGB) → set `OMP_NUM_THREADS`=budget ONLY; `OPENBLAS/MKL/VECLIB` + `NUMBA` stay `"1"`
(HistGB is OpenMP `prange`; the ghost worker's BLAS `cho_factor`/`einsum` at `_ghost_gk.py:1596,1638` are an
axis the HistGB evidence — OMP varied only — never covered, so keep BLAS serial for P3 — r5 B38);
`thread_lever="n_jobs"` (xgb) → the worker env stays `deterministic_env()` (OMP/BLAS/MKL/VECLIB all `"1"`, so
xgb's numpy/BLAS prep — the axis the xgb evidence never covered — never multi-threads) and the budget is
exported as `SILLY_KICKS_RUNTIME_THREADS=<budget>` for the trainer to read trainer-side. Applied budget stamped
to the unit ledger in both cases.
- [ ] S1: failing unit test for `thread_budget` — `vcpu=8`: `(w1,i1)→8`, `(w8,i1)→1`, `(w3,i1)→2`,
  **`(w1,i4)→2`** (inner-parallel), never 0, never `>vcpu`.
- [ ] S2: failing tests — default worker env still asserts all thread vars `"1"` (`test_executor.py:129-130`,
  `test_thread_pin.py:8`, `test_det_env.py`, `test_adapter.py:32` unchanged); an `omp`-lever `thread_safe` worker
  asserts `OMP_NUM_THREADS`=budget AND `OPENBLAS/MKL/VECLIB`+`NUMBA`=`"1"` (BLAS pinned, B38); an `n_jobs`-lever
  `thread_safe` worker asserts OMP/BLAS/MKL/VECLIB still `"1"` AND `SILLY_KICKS_RUNTIME_THREADS=budget`; the
  applied budget in the unit ledger both.
- [ ] S3: run → fail. S4: implement flags (`thread_safe`+`thread_lever`) + budget + per-lever injection + ledger
  stamp. Run → green.

## Task 3: xgb runtime-threads fit kwarg (serialized n_jobs constant)
**Files:** `silly_kicks/tracking/_xshot_occurrence.py` (fit :458/:472/:497/:544; save :426),
`silly_kicks/tracking/_xcross_attempt.py` (:627/:780), tests `test_xshot_occurrence.py`, `test_xcross_attempt.py`.
**Produces:** `XShotOccurrenceModel.fit(..., runtime_n_jobs:int=1)` / same on `XCrossAttemptModel` (additive,
defaulted); estimator runtime `n_jobs=runtime_n_jobs`; serialized metadata `n_jobs` stays constant `1`; no
env-read in `fit`.
- [ ] S1: failing test — `fit(runtime_n_jobs=8)` vs `=1` → byte-identical `metadata.json` + booster bytes;
  metadata `n_jobs==1` both. S2: run→fail. S3: add the kwarg. Run → green.

## Task 4: Synthetic fixtures + non-vacuity + identity tests  (← T2, T3)
**Files:** new `tests/scripts/hf_jobs/fixtures/` generator, `tests/tracking/test_thread_identity.py`.
**Produces:** `make_synthetic_fit_matrix(shape, seed)` at real shape/scale (never owner-tier; provenance string
in the fixture); identity tests over thread counts `1..8` **at the production env** — HistGB varies
`OMP_NUM_THREADS` with `OPENBLAS/MKL/VECLIB`=`"1"`; xgb varies runtime `n_jobs` with OMP/BLAS=`"1"` (B38) —
comparing the FULL artifact byte-for-byte — HistGB
(ghost): `npz` + `metadata.json`; xgb (via T3 kwarg): booster bytes + `metadata.json` (B35); a **non-vacuity
precondition test** that asserts BOTH thread-parallel axes actually DISTRIBUTE work across every `budget` thread
at the fixture — NOT the effective thread count (`_openmp_effective_n_threads()`==`omp_get_max_threads()`, OMP
config, fixture-independent — B29). In pinned sklearn 1.9.0 (`.venv-hfjobs`, verified) the parallelism is TWO
axes, and the artifact is thread-independent BECAUSE of how each runs:
- **Feature axis** (histogram SUM + split search): `prange(n_allowed_features)` at `histogram.pyx:170,270` and
  `splitting.pyx:519` — each thread sums ONE feature's histogram in sample order, so no cross-thread reduction.
  Engages every thread only when `n_features ≥ budget`.
- **Row axis** (data movement, not a reduction): the gather `histogram.pyx:160,164` over `n_samples`, and the
  split-apply `splitting.pyx:353,395` that partitions `n_samples` across threads and writes back in thread
  order. Engages every thread only when `n_samples ≥ budget`.

The precondition asserts `n_features ≥ budget` AND `n_samples ≥ budget` at real shape; the per-axis citation is
recorded in the test docstring. (DROP the earlier "row-partitioned reduction" claim for `:160,164` — it is a
gather — B r4.) xgb `tree_method="hist"`: the builder CITES the pinned xgboost 3.4.1 **sum-splitting** loop and
asserts the fixture clears THAT axis (which need NOT be `n_samples` — A r4 consider 12).
- [ ] S1: failing identity + non-vacuity tests, with a NEGATIVE case PER AXIS, each RED by shrinking the fixture
  ALONE: HistGB `n_features < budget` → the feature loops (`histogram.pyx:170,270`; `splitting.pyx:519`) leave
  threads idle; HistGB `n_samples < budget` → the row loops (`histogram.pyx:160,164`; `splitting.pyx:353,395`)
  leave threads idle; xgb below the builder-cited axis → idle. Proves the test observes work distribution, not
  the OMP-config thread count (B29; A r4 consider 12).
- [ ] S2: run→fail. S3: generator + tests. Run → green.
- [ ] S4: extend the ADR-128 (W,N) parity gate (`tests/scripts/_parity.py`) with a THREAD axis for
  `thread_safe` stages (B28). Runs on x86 in CI — the HF production arch; no aarch64/DGX leg.

## Task 5: Opt-in the ghost + xgb fit stages  (← T4)
**Files:** `scripts/train_ghost_gk.py`, `scripts/train_ghost_outfield.py`, `scripts/train_xshot_occurrence.py`,
`scripts/train_xcross_attempt.py`.
**Produces:** `thread_safe=True` on each fit/fit-assemble stage, behind its T4 test; a registry gate — a
`thread_safe` stage without a committed identity test fails. **The trainers WIRE the injected thread budget
into the fit** (B27): the xgb trainer reads `SILLY_KICKS_RUNTIME_THREADS` trainer-side (NOT inside the lib
`fit` — no Hyrum env-read) and calls `fit(runtime_n_jobs=<that>)`; its stage declares `thread_lever="n_jobs"`.
The HistGB (ghost) stages declare `thread_lever="omp"` and inherit the OMP env automatically. Without this
wiring xgb stays 1-thread and T4's kwarg test passes trivially while production never parallelizes.
- [ ] S1: failing gate test (a `thread_safe` stage lacking an identity test fails) + a test that the xgb
  trainer reads `SILLY_KICKS_RUNTIME_THREADS` and passes it into `fit(runtime_n_jobs)` (not the constant 1),
  with OMP/BLAS left `"1"` for that worker. S2: run→fail. S3: declare `thread_safe`+`thread_lever`, wire the
  gate, wire the budget → `fit`. Run → green.

## Task 6: Dispatch overflow handling (§D)
**Files:** `scripts/hf_jobs/campaign.py` (:515-547,632,647,708-712), `scripts/hf_jobs/ledger.py` (:54-60),
`scripts/hf_jobs/planner.py` (`dispatch_floor`), test `test_campaign.py`.
**Produces:** classifier using `JobInfo.started_at` — **require the attribute present** (missing ⇒ treated as
started, never exempt — B23) + a never-ran status; `dispatch_floor` (≥`shard_cap`); back-off re-derived each
stateless step from `jobs_for`/`_error_history`; `M` consecutive never-started timeouts at the floor →
surface/halt naming the unit; observed max concurrency written to the campaign summary. **The launch loop
(`campaign.py:708-712`, `launcher.launch` `launch.py:243-257`) catches a capacity reject and degrades (backs
off) instead of crashing the step mid-launch (the B09 defect); a NON-capacity error still fails loudly; an
AMBIGUOUS reject (cannot be classified as a capacity limit) fails loud, never degrades — mirroring §D's
`started_at` fail-safe (A r2 consider 09)** (B26).
- [ ] S1: failing tests — (a) started job, no unit ledger, timeout → COUNTS toward `error_resumes`;
  (b) never-started timeout → backs off, not counted; (c) `M` never-started timeouts at floor → halt;
  (d) `JobInfo` without `started_at` attr → treated as started; **(e) reject past the limit → degrades, backs
  off, no halt (r1 HCP-PLAN-02)**; **(f) observed max concurrency recorded in the summary (r1 HCP-PLAN-03)**;
  **(g) a NON-capacity `run_job` error still raises/fails loud — not swallowed by the reject classifier (B26 g);
  (h) an AMBIGUOUS reject fails loud, never degrades (A r2 consider 09)**.
- [ ] S2: run→fail. S3: implement. Run → green.
- **Owner ruling (B24) — RECORDED 2026-10-28:** halt-after-`M` when HF's true ceiling < `shard_cap` →
  **halt + surface the real ceiling** (name the stuck unit + observed max concurrency), NOT degrade-to-waves.
  Probe-cycle intent confirmed by the owner.

## Task 7: ADR-128 amendment + surface sweep
**Files:** `ADR-128`, `AGENTS.md:133`, `docs/howto/compute-pipelines.md`, compute-pipelines skill, comments in
`_stages.py:6/10/117`, `executor.py:9/85`, `adapter.py:8`, `_thread_pin.py:5/24`, `scaffold.py:30`.
- [ ] Amend INV-2 to the opt-in wording; update every surface; keep the four default-env assertions at `"1"`.
  Doc/contract only.
- [ ] **Document the new executor↔trainer contract** `SILLY_KICKS_RUNTIME_THREADS` (absent at `8ff73ee`): the
  `n_jobs`-lever transport — set by the executor at an `n_jobs`-lever `thread_safe` worker launch, read
  trainer-side into `fit(runtime_n_jobs)`. Add it to the ADR amendment, `compute-pipelines.md`, the skill, and
  `scaffold.py` env notes (B37).

## Task 8: Cost measured at the real thread budget (§E)  (← T2)
**Files:** `scripts/hf_jobs/executor.py` (`_profile` :510-540), test `test_executor.py`.
- [ ] S1: failing test — `_profile` of a `thread_safe` stage runs at its production thread budget (not W=1 at
  full threads). S2: run→fail. S3: implement. Run → green.

## Task 9: Release prep (at the commit gate)
- [ ] CHANGELOG entry (additive `fit()` kwarg; INV-2 amendment; dispatch_cap/shard_cap). Version bump because
  `silly_kicks/` `fit()` signatures change (wheel). TODO.md.

## Pre-commit prerequisites (B30 — recorded before the commit gate)
- The B24 owner ruling is **RECORDED (2026-10-28): halt + surface** the real ceiling, not degrade-to-waves (see
  T6). No prerequisite now blocks the commit gate. (No aarch64/DGX prerequisite: HF x86 is the benchmark host,
  proven in CI; the DGX is a possible backup that must work but is never a byte-identity benchmark — P15.)

## Execution handoff
All tasks on one working tree, no intermediate commits. After the last task is green + the prerequisites above:
run the FULL CI-faithful suite once (`pytest -m "not e2e"` + the `.venv-hfjobs` legs + the lint trio) and record
the result (B31) → whole-branch A/B impl review → fix findings (TDD) → **then** present the single commit at the
owner's gate for explicit approval. Then relaunch cycle1 to refit ghost_outfield + `full` on multi-core; that first HF refit's
(W,N)+thread parity compare is the on-HF-CPU P3 confirmation (CI proves it on the CI microarch — B40).
