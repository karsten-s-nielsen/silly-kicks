# HFU driver — resource-sharing overhaul — design

Status: round 2 — all round-1 A+B findings answered below.
Branch: `feat/minimal-hf-driver`. Lands as an **amend** to `acdf2cb491479708803387129692e50910cb33de` (one commit).
Scope owner: Karsten. Reviewers: A and B (independent).

## Round-1 review answers

- **A-SPEC-01 / B-SPEC-01 (fix 3 ordering) — BLOCKING (B), fixed.** The fold is now `created_at`-exact by
  construction: `folded_job_ids` is only a read-once guard; the cache stores each job's
  `(stage, units, created_at, outcome)`; coverage is RECOMPUTED in `created_at` order from the cache each step
  (in memory, no bucket re-reads). Contract states the invariant; the test list adds the out-of-order-termination
  case. See fix 3.
- **A-SPEC-04 (mount-flake accounting) — owner ruling 2026-11-03: attribute by failure locus.** A
  `mount_auth_flake` never counts toward the unit's `attempts` (the unit never executed); it increments a
  PER-DATASET mount-fault tally with its own budget; over budget ⇒ a dataset-level mount-outage report, not a
  per-unit STUCK. Unit-execution failures (`unit_crash`/`oom`/worker-death) keep counting per unit; timeouts
  ride. Folded into fix 3.
- **B-SPEC-02 (fix 2 same-pass) — fixed.** The mount cap counts in-flight PLUS jobs chosen earlier in the same
  `decide` pass; filtered to `access == "mount"`; named distinct from the in-job `read_concurrency`.
- **A-SPEC-02 / B-SPEC-06 (fix 4) — fixed.** `cap_bytes` + auto-detected `rss` backend gives correctness
  (defer → escalate); a static per-stage `peak_rss_bytes` hint (no profiler) sizes W to avoid kill-then-escalate
  waste; the escalate is SHARDED (W=1 solo, sized by fix 1) so a large deferred set never serializes into the
  2 h timeout.
- **B-SPEC-07 (fix 5) — fixed.** The contention signal is named (a 429 from `list_jobs`/`launch`, or the recent
  `mount_auth_flake` rate); `run` gains the cross-step backoff state (added to the component list).
- **Fix-1 precision (A prose, B-SPEC-05) — fixed.** Per-path N (`len(units)` for `_round_robin_launches`,
  `len(keys)` for `_run_launches`); the round-robin path is latent (0 `download` datasets) with a unit test.
- **B-SPEC-04 (wording) — fixed.** The total-in-flight truncation is in `step` (driver.py:112), not `decide`.
- **B-SPEC-03 (consider) — addressed:** `mount_cap` (controller-side concurrent-mount count) is named distinct
  from `read_concurrency` (in-job whole-file read gate, `scripts._dataset_reads`). They do not interact.

Round 2 — both reviewers APPROVE, 0 blocking; three new CONSIDER folded here:
- **A-SPEC-05 — fixed:** the fix-3 cache stores the classify `reason` (or message, re-classified), not just the
  `outcome` enum, so `mount_auth_flake` (RETRY) and `unit_crash` (RETRY) route differently.
- **B-SPEC-08 — fixed:** `mount_fault_budget` named in the fix-2 config bullet (`cycle1.toml`).
- **B-SPEC-09 — fixed:** the by-locus list routes `disk_evict` (and any other unit-execution resource fault)
  per-unit, with `oom`.

## Executive summary

The D1 minimal HF driver (`scripts/hf_jobs/driver.py`, ADR-128 amendment 2026-11-02) ran the cycle1 retrain on
2026-11-02/03. It did not complete: ~1836 jobs, 0/21 instances fully reduced, the owner tier crawling. Root
cause is one thing — **too many small jobs fired without sharing rate-limited resources** ("machine-gunning") —
surfacing as: private-dataset mount-sidecar 401 flakes, HTTP 429 throttling of the bucket API, OOM churn on
`ghost_gk` extract, and O(N)-per-step coverage reads. This cycle fixes exactly five things, all under one
discipline: **share and pace rate-limited resources; size jobs; never re-scan the world each step.**

The engine is unchanged (`launch`, `executor`, `_stages`, trainers, wheel). All five changes are controller-side
(`driver.py` + `campaign.py` config + `cycle1.toml`), plus the already-written `classify` mount-401→RETRY fix
riding in the amend — after the by-locus ruling it is a first-class routed signal (fix 3) and the fix-5
contention signal, not a mere safety net. No planner/cost-model resurrection, no full-utilization
machinery, no wheel change, no new instances.

## Goals

1. Cut total job count several-fold by sizing few/big jobs instead of sharding every stage to the cap.
2. Bound concurrent mounts of a given dataset so the CSI mount sidecar is not stormed.
3. Make per-step coverage O(active work), not O(all terminal jobs) — eliminate the bucket 429s.
4. Size workers to memory and escalate an over-cap unit to a solo whole-box job — eliminate OOM churn.
5. Pace launches and back off when a shared resource signals contention (429 / mount flake).

## Non-goals (explicitly out of scope)

- No cost-based planner, profile stage, or wall-target sizing (that was the cut bloat). Sizing is a fixed
  `units_per_job` floor, nothing adaptive.
- No full-utilization machinery (canary, mem_gate, tiers, dispatch-overflow, measurement jobs).
- No `silly_kicks/` wheel change; the version stays 4.139.1.
- No new campaign instances, no new datasets, no engine (`launch`/`executor`/`_stages`) behaviour change.
- No speculative abstraction. Each change is the minimum to meet its contract, with tests scoped to it. Target:
  low hundreds of changed lines total, not thousands.

## Locked decisions (owner, 2026-11-03)

- The five fixes below, in this priority order.
- Land as an **amend + force-push** to `acdf2cb`; nothing already committed is a standalone-worthy commit.
- `concurrency_cap` stays **25** (account max), but mounts are shared UNDER it (fix 2) — cap is the ceiling,
  not a per-resource quota.
- Minimal, clean, gold-standard. A/B review of spec, then plan, then implementation.

## Architecture — the five fixes (priority order)

### Fix 1 — bounded job-sizing (the root)

- **Broken:** `_run_launches` (driver.py:782, shards `len(keys)`) and `_round_robin_launches` (:772, shards
  `len(units)`) use `n = max(1, min(shard_cap, N))` with `shard_cap = concurrency_cap = 25`, so a 179-unit stage
  becomes 25 jobs and 21 instances ~1733 jobs.
- **Fix:** `n = clamp(ceil(N / units_per_job), 1, max_shards)`, a fixed `units_per_job` floor from the campaign
  (no cost model). N is per-path: `len(keys)` for `_run_launches`, `len(units)` for `_round_robin_launches`.
  Contiguous near-even split unchanged.
- **Contract:** a stage of N units launches `ceil(N / units_per_job)` jobs, capped at `max_shards`
  (≤ a value below `concurrency_cap`). Fewer, bigger jobs; outputs unchanged (placement only, INV-1).
- **Config:** `cycle1.toml` `units_per_job`, `max_shards` (campaign-level).
- **Test:** shard count vs N on both paths for a real plan; `max_shards` ceiling; still a disjoint cover. The
  `_round_robin_launches` path is LATENT (cycle1 has 0 `download` datasets — `datasets.toml`: 5 mount, 1 open);
  cover it with a unit test and state latent-by-design.

### Fix 2 — per-dataset in-flight mount cap (share the sidecar)

- **Broken:** the only bound on load is the total-in-flight truncation in `step` (driver.py:112,
  `budget = concurrency_cap − in_flight`), which is dataset-blind; `decide` walks all instances in one pass
  (:436-483) with only the per-stage `_stage_in_flight` guard. Nothing limits how many in-flight jobs mount one
  dataset, so on step 1 all 7 `tc3` instances (ghost_gk ×5 + ghost_outfield ×2) launch at once → the mount
  storm → 401 flakes.
- **Fix:** a decide-time, per-dataset constraint. Map each in-flight job → its instance (by `(run_key, variant)`
  labels) → its mounted dataset(s), counting ONLY `access == "mount"` datasets (`open`/`download` are not
  CSI-mounted). `decide` does not launch a stage whose mount dataset is already at `mount_cap(d)` — counting
  in-flight PLUS jobs it has chosen earlier in the SAME pass (not just pre-existing in-flight). Cap defaults
  small for private mounts.
- **Contract:** at the end of any `decide` pass, for every dataset `d`,
  `in_flight(d) + chosen_this_pass(d) ≤ mount_cap(d)`.
- **Config:** a per-dataset (or default) `mount_cap`, and the `mount_fault_budget` used by fix 3's
  dataset-outage report (both `cycle1.toml`, per-dataset or default). Distinct from the existing in-job
  `read_concurrency`
  (`datasets.toml`, e.g. tc3=4; a whole-file read gate inside a job via `scripts._dataset_reads`) — `mount_cap`
  bounds the COUNT of concurrently-mounting jobs; they do not interact.
- **Test:** N ready instances on one `mount` dataset with `mount_cap=k` ⇒ ≤ k launches in ONE `decide` pass
  (the same-pass case, not just pre-existing in-flight); an `open`/`download` instance is never throttled; a
  different-dataset instance still launches.

### Fix 3 — coverage set+push, not re-read-all (share the bucket)

- **Broken:** `_coverage` / `_reduce_errors` call `_manifest_for → read_bytes` for every terminal
  COMPLETED/RETRY job every step → O(N) bucket reads → HTTP 429 throttling that worsens as N grows. The charge
  rule is also ORDER-DEPENDENT (`_coverage` folds in `created_at` order, driver.py:679; a RETRY charges only
  not-yet-spent units, recorded-first, :689-692), so a naive "fold only new ids into a persisted tally" is NOT
  commutative under out-of-order termination (a slow early-`created_at` ghost_gk job finishing after fast later
  jobs) — it would silently diverge from a full rescan and reintroduce the stall.
- **Fix (gold standard):** a per-`run_key` persisted CACHE `_coverage/state.json` mapping
  `job_id → {stage, units, created_at, outcome, reason}`, plus a `folded_job_ids` set used ONLY as the read-once
  guard. The `reason` (the `classify` `Verdict.reason`, or the raw status message re-classified on recompute —
  the plan picks; message is the more robust single-source choice) is REQUIRED: `mount_auth_flake` and
  `unit_crash` are both `JobOutcome.RETRY` and the by-locus routing below can only separate them by reason
  (A-SPEC-05).
  Each step: one `list_jobs`; for `new = terminal − folded`, read each manifest ONCE (or use the `job_id→units`
  recorded at launch), add to the cache; then RECOMPUTE coverage and the per-dataset mount-fault tally in
  `created_at` order from the WHOLE cache in memory (no bucket re-reads). Stage-prune: skip cached jobs whose
  stage is already `reduced` (plan flag × `stage` label). The cache is reconstructable — absent/corrupt ⇒ one
  O(N) rebuild; it is never an authoritative tally.
- **Attribution by failure locus (owner ruling 2026-11-03).** On the recompute, route each cached job by
  `classify` reason: `unit_crash`/`oom`/`disk_evict`/worker-death-no-record — and any other unit-execution
  resource fault — → the PER-UNIT `cover` (counts toward `attempts`); `mount_auth_flake` → a PER-DATASET
  `mount_faults(d)` tally (NEVER per-unit); `timeout` → ignored (rides).
  A dataset whose `mount_faults(d) > mount_fault_budget` is reported as a mount OUTAGE (dataset-level), not a
  per-unit STUCK. So a flaky mount never condemns a unit, and a real outage still fails loud.
- **Contract:** recomputed coverage == a full `created_at`-ordered rescan, REGARDLESS of termination order; each
  manifest is read ≤ once per controller process; deleting the cache yields identical results; `mount_auth_flake`
  contributes 0 to any unit's `cover`.
- **Test:** equivalence on a real plan with mixed COMPLETED/RETRY/empty-manifest jobs INCLUDING an
  out-of-order-termination case (an early-`created_at` job folded after a later one); cache-delete rebuild
  identity; `folded_job_ids` monotonic; a `mount_auth_flake` job leaves per-unit `cover` unchanged and bumps
  `mount_faults(d)`; `mount_faults(d) > budget` ⇒ dataset-outage report.

### Fix 4 — memory-aware sizing + escalation (share RAM)

- **Broken:** `_job_spec` passes `cap_bytes=None, peak_rss_bytes=None`; the executor already supports a
  per-worker cap, `size_workers`, and `--escalate` (whole-box solo for `deferred_memory_units`), and `JobSpec`
  already carries the fields with argv plumbed (launch.py:76,77,85,175-182) — so fix 4 is DRIVER-ONLY. But the
  executor sizes W DOWN only when `peak_rss_bytes` is set (executor.py:318-319; `size_workers` requires it,
  `_parallel_launch.py:23-28`), and the only prior producer was the profile pass — a non-goal. `ghost_gk`
  extract OOMs at 8 workers / 32 GB (83 OOMs one run), masked only by retry.
- **Fix:** (a) correctness — pass a per-worker `cap_bytes` (fair share) and leave `mem_backend` unset so
  `detect_backend` auto-selects the `rss` watchdog (NEVER the in-container-broken `rlimit`/`RLIMIT_AS`); an
  over-cap unit is then recorded memory-deferred and `decide` launches an escalate job. (b) efficiency — a
  STATIC per-stage `peak_rss_bytes` hint from config (no profiler) so W is sized down and the unit is not
  run-then-killed-then-escalated. (c) the escalate is SHARDED by fix 1 (W=1 solo, `units_per_job` over the
  deferred set) so a large deferred set never serializes into the 2 h timeout.
- **Contract:** a unit over cap is deferred, not failed; the driver launches escalate job(s) for a stage with
  `memory_units`, sharded to ≤ `units_per_job` deferred units each; no unit OOM-churns to STUCK; `cap_bytes`
  and the static `peak_rss_bytes` reach the `JobSpec`/argv.
- **Config:** `cycle1.toml` per-stage (or per-job) `peak_rss_bytes` hint; the per-worker cap is the box
  fair-share (32 GB / W − headroom), no new knob.
- **Test:** a plan with `memory_units` triggers escalate launch(es) sized by `units_per_job`; `cap_bytes` and
  the static peak reach the argv; `mem_backend` resolves to `rss`.

### Fix 5 — pace launches + back off on contention (share, don't hammer)

- **Broken:** `step` launches the whole `budget = concurrency_cap − in_flight` in one burst; `run`'s `poll_s`
  is a fixed float (driver.py:180) with no cross-step state; a 429/flake signal does not slow the controller.
- **Fix:** (a) a per-step launch pace `launch_batch ≤ budget`; (b) exponential backoff of the next poll on a
  named CONTENTION SIGNAL — a 429 raised by `list_jobs`/`launch`, or the recent `mount_auth_flake` rate over a
  window. `run` gains the cross-step backoff state (current delay + a reset-on-clean-step), so this is a small
  `run` change, not just `step` (added to the component list).
- **Contract:** a step launches ≤ `launch_batch`; a contention signal multiplies the next poll (bounded max);
  a clean step resets it; backoff never deadlocks a legitimately idle-but-not-done campaign (bounded + reset).
- **Test:** budget 25 with `launch_batch=8` launches ≤ 8; an injected 429 signal lengthens the next `poll_s`
  and a subsequent clean step resets it.

## Testing

Each fix carries a focused unit test (above), all on real `_stages.plan()` fixtures and real captured HF
`JobInfo` (no hand-built doubles). The existing driver suite stays green. CI-faithful: ruff 0.15.7 trio +
pyright + `tests/scripts/hf_jobs/` (incl. `test_executor.py`). Mutation-check each new test kills its fix.

## Risks and open items

- Fix 3 cache schema is now specified (`job_id → {stage, units, created_at, outcome, reason}`; recompute-in-order).
  `run_key` isolation holds: `run_key = {campaign}-{commit7}-{job}-{tier}` (campaign.py:93-96), so the
  `a761710` session (different commit7) cannot collide on the state path. Two controllers on the SAME commit
  share it, but the cache is reconstructable — a lost update is a rebuild, not corruption. The plan settles the
  atomic-write mechanism and the write-then-rename vs store-native-atomic choice.
- Fix 2 in-flight→instance join is by `(run_key, variant)` labels (present on every launched job) → `inst`
  → `inst.datasets`. The plan confirms the labels carry both on real `JobInfo` fixtures.
- Fix 5 backoff is bounded (max delay) and resets on a clean step, so it never deadlocks an idle-but-not-done
  campaign (in the fix-5 contract).
- Caps stay INDEPENDENT ceilings: fix 1 (fewer jobs), fix 2 (per-dataset mounts), fix 5 (launch pace) each bound
  a different axis; do not tune one to compensate for another.

## Docs and C4

Update the ADR-128 amendment (the driver now sizes/shares), `docs/howto/hf-jobs-training.md` sizing +
`concurrency_cap`/`units_per_job`/`mount_cap` description, `docs/howto/compute-pipelines.md` (sizing),
CHANGELOG `[Unreleased]`, and the C4 `campaignCtl` line. No new ADR number (same amendment).

## Commit plan (approval-gated; amend only)

One amend to `acdf2cb` after implementation A/B review passes and the owner gives an explicit commit yes. Then
force-push `feat/minimal-hf-driver` (updates PR #299), watch CI to green. No new commit; no micro-commits.
