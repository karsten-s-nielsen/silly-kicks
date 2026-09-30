# Design: memory-aware parallel launcher + shared-corpus study-parallelism

**Date:** 2026-09-29
**Status:** Draft for review
**Branch:** `feat/combined-provenance-dgx` (this cycle; single additional commit, owner-gated)
**Repo:** silly-kicks (`scripts/`)

---

## 1. Context and motivation

The DGX (aarch64, 20 cores, 119 GiB) is **reliable but under-utilised** for this class of work. The
current F1b retrain runs 4 driver processes at **load ~4 — 16 cores idle — for ~24 h**. The binding
constraint is **RAM, not cores**: measured per-worker RSS is 6.6–10.7 GiB **for the F1b retrain workers** (DAS match-workers have a different footprint — see §9), so naively running 20 workers
(≈180 GiB) would OOM — the failure mode the owner has hit before by running too many things at once.

Every corpus driver already adopts `scripts/_driver.py::for_each` (ADR-052): a serial, streamed,
resumable, shard-per-item loop. The two drivers this cycle cares about both use it:

- `validate_das_native_parity.py` — its **entire** workload is `for_each` over matches.
- `train_xshot_occurrence.py` / `train_xcross_attempt.py` — `for_each` for the match/feature pass, then a
  **separate serial** `_paired_nested` loop over ~15 `(candidate × fold)` studies (the F1b bottleneck).

This design adds **memory-safe parallelism** so the idle cores can be used without OOM, and so a single
study/task uses less RAM (shared corpus). It is **output-preserving**: every unit runs unchanged and
deterministic; only *how many run concurrently* and *their memory bounds* change. It is designed to
**generalise across machine architectures** (Linux DGX today; macOS/Windows degrade gracefully).

## 2. Goals / non-goals

**Goals.**
- Use the DGX's spare cores without OOM: bounded, RAM-aware concurrency with a fail-safe per-worker cap.
- Cut per-worker RAM (share the loaded corpus) so more workers fit.
- Fully output-preserving — parallel results byte-identical to serial.
- Portable: full safety on Linux, graceful degradation elsewhere; no hard dependency that crashes off-Linux.

**Non-goals (YAGNI).**
- **No within-study HPO parallelism (5b).** Parallel TPE changes the trial trajectory → a different model.
  Out of scope; studies stay serial internally. Only *quality-neutral* parallelism ships.
- No multi-machine / distributed execution (single box).
- No changes to `for_each`'s internals (every corpus driver depends on that seam) — the launcher wraps invocations.
- No GPU offload (a separate lever).
- Driver-agnostic launcher, but only DAS parity + F1b are wired as consumers now.

## 3. Scope

| Item | Component | Summary |
|---|---|---|
| 1 RSS-cap concurrency | A (launcher) | N sized from measured peak-RSS + free RAM, not core count |
| 2 per-worker memory cap | A + §5 | pluggable backend; Linux cgroup hard-cap, best-effort/backpressure elsewhere |
| 3 RAM backpressure | A | start a worker only while `MemAvailable > peak + margin`; add more as RAM frees |
| 4 thread pin | A | per-worker thread env via a shared pin helper: `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`=1 (aarch64 DGX numpy uses OpenBLAS) + `VECLIB_MAXIMUM_THREADS`=1 (macOS Accelerate); no core oversubscription |
| 5a shared corpus | B | build the design matrix once, workers `mmap` it read-only |
| 5c F1b across-study | B | run the ~15 `(candidate × fold)` studies concurrently (each serial, fixed-seed → deterministic). 5b (within-study parallel HPO) is out of scope, §2 |

## 4. Component A — the launcher (`scripts/_parallel_launch.py`)

A driver-agnostic orchestrator (importable helpers + a CLI `main`). Approach #1 (approved): it wraps the
**existing resumable driver invocations**, it does not reimplement `for_each`.

- **Work split.** The corpus is partitioned into N subsets and each worker is invoked over its subset via
  the existing `--match-ids-json` mechanism against a **shared `shard_root`**. Resume-before-load makes a
  relaunch skip finished shards, so a subset is a hint, not a hard partition — a dead worker's remaining
  items can be re-handed.
- **Worker model.** N worker processes, each running the driver over its subset. Each is launched through
  the **memory-cap backend** (§5) and with threads pinned. A worker holds one item at a time (`for_each`
  is streamed), so worker RSS ≈ one item's footprint.
- **Sizing.** `N = floor((MemAvailable − headroom) / peak_RSS)`, `peak_RSS` measured (§9), capped at
  `nproc`, and **clamped to N≥1** — if a single worker's `peak_RSS` exceeds usable RAM, run one worker and warn (never N=0 / no-progress); refuse only if even one worker cannot fit. Headroom default ~20 GiB (OS + page cache). **`peak_RSS` is per-workload**: sized from the DAS match-worker's or the F1b study-worker's *own* measurement (§9), never one shared number.
- **Backpressure.** Launch up to the RAM budget; poll `MemAvailable`; start the next worker only when a
  worker-slot of RAM is free. Never "launch N at once."
- **Failure handling.** `max_consecutive_failures` stays **per-worker** (each worker keeps its own count
  over its subset), so the systematic-bug abort still fires. A worker killed by its memory cap → its
  shards persist → the launcher detects the non-zero exit and relaunches its remaining subset.
  **Individual death, never the box.**
- **Completion.** On all-workers-done, run the existing `reconcile` / `assert_conservation` over the shard
  root (unchanged), producing the same artifact a serial pass would.
- **Provenance.** Runs the driver from the committed SHA, clean-tree required, `run_commit` stamped —
  unchanged from the current discipline. The launcher writes no artifact of its own.

## 5. Pluggable memory-cap backend (portability)

A small `MemoryCap` strategy selected by auto-detection, so the launcher runs anywhere:

| Backend | Platform | Mechanism | Guarantee |
|---|---|---|---|
| `cgroup` | Linux + systemd | `systemd-run --scope -p MemoryMax=<cap>` wrapping the worker | **Hard** per-worker cap; OOM-kills the worker **and its children** never the box. **Dependency:** the child case matters only if/when the parity-reference cycle's out-of-process pandas-2 reference leg lands; it is NOT on this branch, where `validate_das_native_parity.py` uses an **in-process** `_reference_leg`, so there is no child to cover yet |
| `rlimit` | POSIX best-effort | `setrlimit(RLIMIT_AS, cap)` in the worker pre-exec | Soft/partial (unreliable on macOS; helps on Linux without systemd) |
| `none` | any | no per-worker cap | Relies on backpressure + conservative N only |

- **Detection order:** `cgroup` if `systemd-run` present and cgroup v2 writable; else `rlimit` where it is
  honoured; else `none`. Overridable by a `--mem-backend` flag.
- **Backpressure runs in every backend** — it is the portable floor of safety; the cap is the belt on top.
- **Rationale for graceful degrade:** macOS has no cgroups and unreliable `RLIMIT_AS`, but its memory
  pressure compresses/swaps *before* killing, so over-subscription degrades to **slowdown, not crash** —
  backpressure-only is materially survivable there. Windows similar (working-set trimming + pagefile).
- The launcher **must not import or hard-require `systemd`/Linux-only modules at module load** — the
  backend is chosen at runtime so the CLI imports and unit-tests on any OS.

## 6. Component B — shared corpus + F1b across-study parallel

- **5a shared corpus.** Build `(X, y, groups)` **once** and persist to a memory-mappable artifact
  (file-backed `np.memmap` / `.npy`), so the workers **`mmap` it read-only** instead of each rebuilding
  it (~9 GiB × 4 → ~1× + small per-worker deltas). File-backed mmap is chosen deliberately: it is the
  correct shared-memory primitive under `spawn`-based Python (macOS/Windows), not just `fork` (Linux).
- **F1b across-study parallel.** The serial `_paired_nested` loop over the ~15 `(candidate × fold)`
  studies runs in the launcher's bounded pool. Each study is **serial internally and fixed-seed**, so its
  result is a deterministic function of its data slice + seed — **running studies concurrently yields
  byte-identical weights to serial.** No 5b.

## 7. Data flow

- **DAS (A's test case):** launcher → N match-subset workers → `validate_das_native_parity` per subset →
  shards → `reconcile` → parity artifact, **identical to serial**.
- **F1b (B's test case):** build + persist shared `X,y` once → launcher → N study-workers reading the
  `mmap` → per-study params → assemble → weights, **identical to serial**.

## 8. Error handling / OOM safety

- Per-worker cap (§5) → individual OOM-kill (Linux) / soft-limit / backpressure elsewhere.
- Backpressure prevents the "launch everything, then OOM" failure.
- Resume-before-load → a killed worker's completed items persist; relaunch resumes with no lost work.
- Per-worker consecutive-failure abort preserved.
- Thread-pin → no core oversubscription (which is slowdown, not crash — bounded separately from RAM).

## 9. Testing and validation

**Output-preservation is the load-bearing guarantee; it is proved, not asserted.**

- **Launcher unit tests (stdlib, cross-OS, CI):** N-sizing math; subset split; backend selection +
  command construction (assert the `systemd-run` argv on a Linux fixture; assert `rlimit`/`none`
  fallbacks); backpressure decision function; **resume-after-simulated-OOM** (a fake driver that exits
  non-zero mid-subset → relaunch resumes from shards). The cgroup-command test asserts argv **without
  executing** it, so it runs on any OS; an execute-it integration test is Linux-gated (skip elsewhere).
- **A on DAS (owner-run, DGX):** run parity through the launcher on a **small match subset first**
  (confirm no OOM under load + speedup + shards identical to a serial subset), then the full run →
  **assert the parity artifact is byte-identical to a serial run** (each match deterministic).
- **B on F1b (owner-run, small set):** **byte-identity gate** — parallel-study weights == serial-study
  weights on a small `(candidate × fold)` set (each study deterministic per seed); plus a
  shared-`mmap` `X,y` == per-process-built `X,y` byte-check. **On any mismatch the parallel path is blocked and the run falls back to serial — a non-identical result never ships.** NOT run against the in-flight F1b job.
- **Peak-RSS measurement:** capture the true high-watermark **per worker workload** (the DAS match-worker
  incl. any child it spawns; the F1b study-worker over one `sc_extended` study), and size each run's `N`
  from its own number (§4), never one shared value; sized from data, not a mid-run snapshot.
- **CI scope:** launcher units only (no DGX). DGX integration is owner-run, like the other DGX artifacts.

## 10. Portability matrix

| Capability | Linux (DGX) | macOS (Mac Studio) | Windows |
|---|---|---|---|
| RSS-sized concurrency, backpressure, thread-pin | ✅ | ✅ (+`VECLIB_MAXIMUM_THREADS`) | ✅ |
| Shared file-backed `mmap` corpus | ✅ | ✅ (correct for `spawn`) | ✅ |
| Resume / shards | ✅ | ✅ | ✅ |
| **Hard per-worker memory cap** | ✅ cgroup | ❌ (backpressure + soft only) | ❌ (backpressure only) |
| Failure mode when over-subscribed | abrupt OOM-kill (why the cap matters) | compress/swap → slowdown | working-set trim → slowdown |
| Cross-machine **byte-identical weights** | ❌ | ❌ | ❌ |

> Cross-machine weight identity is not achievable (NEON vs GB10 vs x86 FMA) and is a training-determinism concern, **not** the launcher's; the launcher is output-neutral on a given machine.

## 11. Provenance and governance

- Output-preserving orchestration → **no artifact value changes** → no version bump on its own.
- Clean-tree + `run_commit` discipline unchanged (the wrapped driver enforces it).
- Lands as a **single commit** on `feat/combined-provenance-dgx`, owner-gated (per-commit approval).
- `scripts/_parallel_launch.py` is `_`-prefixed → skipped by the driver ASCII gate and the
  `ARTIFACT_DRIVERS` enrollment (it writes no artifact); it still carries an argparse CLI (never a
  parser-less script).

## 12. Open questions / risks

- **cgroup v2 availability on the DGX** — verify `systemd-run --user`/`--scope` + `MemoryMax` is writable
  in the run environment before relying on the hard cap (else it degrades to `rlimit`/backpressure). A
  first-run preflight check reports which backend is active.
- **Subset re-hand vs static split** — the simplest first cut is a static N-way split with per-worker
  resume; a shared work-queue (workers pull the next unclaimed key) is a later refinement, not needed for
  the DAS test case. Flagged for the plan to pick the simplest that passes the resume-after-kill test.
- **F1b consumer wiring** — B requires the trainer to expose a single `(candidate, fold)` study as a unit
  reading the shared `mmap`; the plan scopes that refactor. It is validated on a small set, never on the
  in-flight run.
