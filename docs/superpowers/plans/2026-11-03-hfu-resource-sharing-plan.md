# HFU driver — resource-sharing overhaul — implementation plan

Status: DRAFT for independent A/B review (plan).
Spec: `docs/superpowers/specs/2026-11-03-hfu-resource-sharing-design.md` (A+B APPROVE, round 2, 0 blocking).
Lands as the **amend** to `acdf2cb491479708803387129692e50910cb33de`; one commit; force-push PR #299.

## Global constraints

- Driver-only: `scripts/hf_jobs/driver.py` + `scripts/hf_jobs/campaign.py` (`Campaign` config fields) +
  `scripts/hf_jobs/campaigns/cycle1.toml`. Engine (`launch`/`executor`/`_stages`/trainers/wheel) UNCHANGED.
- No planner/cost-model/profile. No full-utilization machinery. No `silly_kicks/` wheel change; version 4.139.1.
- Minimal: target low hundreds of changed lines incl. tests. TDD throughout (watch each test fail first).
- CI-faithful before done: ruff 0.15.7 trio, pyright bare, `tests/scripts/hf_jobs/` incl. `test_executor.py`.
  Mutation-check each new test kills its fix.

## Starting working-tree state

Branch at `acdf2cb` (driver + cap 25). Uncommitted already on the tree: the `classify` mount-401→RETRY fix
(`failures.py` + `test_failures.py`, written + green this session) and this spec/plan. All of it amends into the
one commit.

## Deferred decisions, now resolved (were "settle in the plan")

- **D1 — fix-3 state path + atomicity:** the cache lives at **`_partitions/_coverage/state.json`** — a path
  `BucketStore.upload_bytes` already permits (its allowlist is `_campaign/`/`_partitions/`, store.py:59), and the
  store prefix is the `run_key`, so it is isolated with NO store.py change (B-PLAN-01). Write the WHOLE object
  each step via `upload_bytes` → `batch_bucket_files(add=...)`, a single whole-object PUT (store.py:61): a full
  replace, no torn object, no write-then-rename. On read, a JSON parse error ⇒ treat as absent ⇒ one O(N) rebuild.
- **D2 — fix-3 reason storage:** store the `classify` `Verdict.reason` (short string), not the raw message. The
  amend changes the commit ⇒ a fresh `run_key` ⇒ a fresh state the first run, so every job is classified by the
  landed `classify`; there is no stale-reason-across-classify-versions case to defend. (Message+re-classify is
  the fallback only if cross-version robustness is ever needed; out of scope.)
- **D3 — fix-2 / fix-1 / fix-4 / fix-5 default values:** set in `cycle1.toml`, owner-tunable:
  `units_per_job = 20`, `max_shards = 12`; `mount_cap` default 25, `tc3 = 6`; `mount_fault_budget` default 12;
  per-stage `peak_rss_bytes` for the memory-heavy stages only (ghost_gk `extract`, `cache`); `launch_batch = 8`;
  backoff ×2 to a 300 s max, reset on a clean step. Per-worker cap = usable_RAM / W − headroom (no new knob).

## Task overview (dependency order, one commit)

1. Config surface (Campaign fields + cycle1.toml) — foundational for 2–5.
2. Fix 1 — bounded job-sizing (`_run_launches`/`_round_robin_launches`).
3. Fix 2 — per-dataset in-flight mount cap (decide).
4. Fix 3 — coverage recompute + by-locus attribution (the core).
5. Fix 4 — memory cap + sharded escalation.
6. Fix 5 — launch pace + contention backoff.

## Task 1 — config surface

Add to `Campaign` (frozen dataclass) + `load_campaign` + `cycle1.toml`: `units_per_job`, `max_shards`,
`mount_cap` (scalar default + optional per-dataset map), `mount_fault_budget`, `launch_batch`,
`backoff_max_s`, and a per-stage `peak_rss_bytes` map. All optional with the D3 defaults so `load_campaign`
stays back-compatible. Test (node `tests/scripts/hf_jobs/test_campaign.py::test_load_campaign_reads_the_driver_settings`):
`load_campaign(cycle1)` reads each field; defaults apply when absent. Update that one model assertion.

## Task 2 — fix 1, bounded job-sizing

Seam: `_run_launches` (driver.py:782, N=`len(keys)`), `_round_robin_launches` (:772, N=`len(units)`).
Change: `n = max(1, min(max_shards, ceil(N / units_per_job)))`. Thread `units_per_job`/`max_shards` from the
campaign through `decide` (it already passes `shard_cap`; replace with the two values or pass the campaign-
derived sizing). Tests (node `tests/scripts/hf_jobs/test_hf_driver.py::test_job_sizing_*`): shard count =
`ceil(N/upj)` capped at `max_shards` on both paths; disjoint cover preserved; `_round_robin_launches` covered by
a unit test (latent — 0 download datasets), stated latent.

## Task 3 — fix 2, per-dataset in-flight mount cap

Seam: `decide` stage loop (driver.py:436-483). Build `mounted = {dataset: count}` from in-flight jobs
(`_stage_in_flight`-style label match → instance via `(run_key,variant)` → `inst.datasets`, `access=="mount"`
only), and a same-pass `chosen` counter incremented as launches are appended. `_mount_datasets(inst)` returns
the SET of that instance's `access=="mount"` datasets (an `xcross_attempt` instance mounts TWO — its
`dataset` AND its `probe_dataset` `xcross_probe`, both `mount` in `datasets.toml`; A-PLAN-02). Skip launching a
stage if ANY of its mount datasets would exceed `mount_cap(d)` counting in-flight + chosen-this-pass. Tests
(node `tests/scripts/hf_jobs/test_hf_driver.py::test_mount_cap_*`): N ready instances on one mount dataset,
`mount_cap=k` ⇒ ≤ k launches in one `decide`; a dataset shared by several instances (`xcross_probe`, 4
instances) is capped across them; open/download never throttled; other-dataset instance still launches.

## Task 4 — fix 3, coverage recompute + by-locus (core)

Seam: `_coverage`, `_reduce_errors`, `_manifest_for`, `step`. Replace the per-step O(N) re-read with:
- A `_CoverageCache` loaded once per `step` from `_partitions/_coverage/state.json` (`BucketStore`), mapping
  `job_id → {stage, units, created_at, outcome, reason}` + `folded` set.
- `step` folds `new = terminal − folded`: read each new manifest ONCE (or use the at-launch mapping), add to
  the cache, persist (D1: whole-object PUT).
- `_coverage` RECOMPUTES in `created_at` order from the whole cache in memory, PRESERVING the committed
  `_coverage` semantics (A-PLAN-01): a COMPLETED job credits its units; a crash RETRY charges only its
  **not-yet-spent** units, **recorded failures first** (HFUD-IMPL-10, current driver.py:683,689-692). The ONLY
  change from committed `_coverage` is by-locus routing by `reason` — per-unit `cover` for
  `unit_crash`/`oom`/`disk_evict`/worker-death; per-dataset `mount_faults(d)` for `mount_auth_flake`; `timeout`
  ignored. Stage-prune reduced stages.
- `decide` reports a dataset-outage when `mount_faults(d) > mount_fault_budget` (a loud report, not a per-unit
  STUCK). `_reduce_errors` folds from the same cache.
Tests (node `tests/scripts/hf_jobs/test_hf_driver.py::test_coverage_*`): the equivalence ORACLE is the
**committed `_coverage`** with `mount_auth_flake` jobs removed from the per-unit tally — NOT a fresh re-impl, so
the test cannot pass vacuously (A-PLAN-01); recompute == that oracle INCLUDING an out-of-order-termination case
(an early-`created_at` job folded after a later one); cache-delete rebuild identity; `folded` monotonic; a
`mount_auth_flake` job leaves per-unit `cover` unchanged and bumps `mount_faults(d)`; over-budget ⇒
dataset-outage report. Keep the cache behind `store_for` so `decide`/tests stay mockless.

## Task 5 — fix 4, memory cap + sharded escalation

Seam: `_job_spec` (cap_bytes/peak_rss_bytes=None at :563-564), `decide`. Pass `cap_bytes` = fair share and the
static per-stage `peak_rss_bytes` from config; leave `mem_backend` unset (executor `detect_backend`→`rss`).
`decide`: when a stage's plan reports `memory_units`, launch escalate job(s) (`escalate=True`, W=1) sharded by
`units_per_job` over the deferred set. The escalate job's `cap_bytes` is the **W=1** share (usable_RAM −
headroom, i.e. the whole box), NOT the W=8 fair-share that deferred the unit (A-PLAN-03). Tests (node
`tests/scripts/hf_jobs/test_hf_driver.py::test_escalate_*`): a plan with `memory_units` ⇒ escalate launch(es)
sized by `units_per_job`, each `escalate=True` + `workers=1` + the W=1 cap; `cap_bytes` and the static peak
reach the argv (assert via `launch._run_argv`); engine files unchanged (diff check).

## Task 6 — fix 5, launch pace + contention backoff

Seam: `step` (budget burst at :112), `run` (fixed `poll_s` at :180). `step` launches ≤ `launch_batch`. `run`
carries a backoff delay: on a contention signal — a 429 raised by `live()`/`do_launch`, or `mount_faults`
rising this step — multiply the next `poll_s` (×2, ≤ `backoff_max_s`); a clean step resets it. Tests (node
`tests/scripts/hf_jobs/test_hf_driver.py::test_pace_backoff_*`): budget 25 with `launch_batch=8` ⇒ ≤ 8 launches;
an injected 429 lengthens the next poll; a clean step resets; backoff is bounded (no deadlock of an
idle-but-not-done campaign).

## Docs (in the amend)

ADR-128 amendment (driver now sizes/shares; by-locus attribution; the 5 knobs), `hf-jobs-training.md`
(the knobs + command notes), `compute-pipelines.md` (sizing), CHANGELOG `[Unreleased]`, C4 `campaignCtl` line.
No new ADR number.

## Commit and review gates

Implementation A/B review before any commit. On both-clear + explicit owner commit yes: amend `acdf2cb`,
force-push `feat/minimal-hf-driver`, watch CI to green. One commit; no micro-commits; no commit without the
owner's yes.
