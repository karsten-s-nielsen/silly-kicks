# ADR-074: CI wall-clock < 10 min via a duration-sharded test matrix

| Field | Value |
|---|---|
| **Date** | 2026-08-26 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

> Ships in silly-kicks **4.96.0** (the CI commit of the scale-guard branch, on top of 4.95.0 / ADR-073);
> number re-verified against merged `origin/main` at commit-prep. Tooling-only — no `silly_kicks`
> runtime change, no retrain, C4-free.

## Context

CI took **22–25 min** wall-clock and runs twice per release (PR + post-merge). Billing was never the
constraint (the account has never exceeded its monthly minutes); **latency was.** Measured on main run
`33000702218`: the bottleneck is a single **serial** `pytest` process per leg (primary `not e2e`
**20:19**; windows `not slow` **17:48**), not install/benchmark/doctest. `pytest-xdist -n auto` inside
one job had already been reverted — it memory-killed py3.12 on the 4-core/7 GB runners (4× heavy ML
imports + 4× numba compiles + the ghost-GK suite). So intra-job parallelism is closed on these runners.

A `--durations=0` profile put a real floor under the design: the **longest single test is 39 s**
(train-script smokes; then DAS/gkdv 9–32 s) — **no multi-minute indivisible test**, so sharding is not
floor-limited and the *long-term* lever (fixing slow tests) is not required. The binding constraint is
the **Windows install (1:49, undivided)**, which sharding cannot divide.

## Decision

**Shard the suite across parallel jobs with `pytest-split`.** The `test` job becomes an
`os × python × shard[1..N]` matrix (N=3); each job runs `pytest … --splits N --group ${{ matrix.shard }}`
on its **own** runner, so the xdist memory-kill cannot recur. The split is **duration-balanced** from a
**committed `.test_durations`** (count-balancing is useless — a few `@slow`/ghost-GK tests dominate).

Supporting decisions:
- **`.test_durations` is committed in the SAME commit as `ci.yml`.** Without it, pytest-split's
  count-mode split is **non-deterministic** — measured: shard sizes drift run-to-run (2708/2708/2708 one
  run, 2706/2706/2705 the next) and can under-cover. With the file the split is deterministic and
  complete. It is regenerated (`pytest -m "not e2e" --store-durations`) on a material suite shift.
- **`-p no:randomly` on every sharded command** pins collection order (the partition is only valid if
  every shard collects identically); a static guard bans any shuffle plugin from `[test]`.
- **ADR-023 slow-gating preserved:** primary shards run `-m "not e2e"` (incl `@slow`), non-primary run
  `-m "not e2e and not slow"`; pytest-split re-balances within each leg's selection.
- **Benchmark is a STANDALONE parallel job** (not folded onto a shard) — latency is the SLA and a
  standalone job hides fully behind the ~7-min shards, contributing 0 to wall-clock. doctest +
  pandas-major run on each leg's **shard 1 only** (once per leg; the pandas artifact name has no shard
  component, so N shards would collide and `upload-artifact@v4` hard-fails).
- **Lever C — numba disk cache** (`NUMBA_CACHE_DIR` + `actions/cache`, no source change; the kernels
  already declare `@njit(cache=_NUMBA_CACHE)`) + **pip caching**, prioritized on the install-bound
  Windows leg.
- **Two anti-silent-drop guards.** Static `tests/test_ci_shard_wiring.py` (contiguous `1..N`,
  `--splits == N`, `-p no:randomly`, benchmark-standalone, reconcile-job-present, and — via AST, not a
  regex — the numba-cache key covers every `@njit` file incl. `_turnover.py`'s call form). Runtime
  `shard-reconcile` job proving the node-ID sets **partition** each leg's suite (`union == full ∧
  pairwise-disjoint`), the only guard that catches cross-runner collection divergence.

At N=3: **16 distinct jobs, ~14 peak-concurrent** (< the Free-plan account-wide 20 cap). Coverage is
preserved in full (decision A); trimming Windows to the platform-/version-sensitive subset (lever
B-Windows) is the recorded next lever if latency/cost later warrant it.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Larger runners + re-enable `xdist -n auto` | Larger runners are a paid org feature (may be off); the memory-kill history makes it the riskier bet; and it scales worse than sharding. Kept as a fallback only. |
| Count-balanced split (no `.test_durations`) | Non-deterministic and can under-cover (measured); a few tests dominate runtime so count balance is poor. |
| Fold benchmark onto primary shard 1 | Optimizes job-count/minutes (non-constraints) against wall-clock (the SLA) — the standalone job is strictly better for latency with job headroom to spare. |
| Reduce coverage (trim Windows) now | Owner decision A: preserve all coverage; sharding hits < 10 min without it. Deferred as lever B-Windows. |
| Count-sum reconcile | A duplicate in one shard cancels a drop in another; node-ID **set** union/disjoint is the enumerate-and-conserve guard this repo uses everywhere. |

## Consequences

### Positive
- CI wall-clock target **< 10 min** (the slowest leg), validated on a real run; all coverage preserved.
- Two independent partition guards (static pre-flight + runtime node-ID reconcile) make a silent test
  drop CI-visible, matching the repo's `pandas-span` / ADR-052 conservation discipline.

### Negative / maintenance
- A committed `.test_durations` to regenerate periodically (balance-only; never affects coverage).
- `N` lives in two literals (`shard: [1..N]` + `--splits N`) kept consistent by the static guard;
  bumping it is a two-line edit bounded by the account-wide 20-job concurrency cap.

### Neutral
- Tests + workflow + docs only — **no `silly_kicks` runtime change, no retrain, C4-free.**

### Known limits (stated, not discovered)
- **Windows is install/numba-bound**; sharding cannot divide the 1:49 install, so the SLA there rests on
  the numba/pip caches (cold-cache first run may sit higher — a steady-state claim).
- The `.test_durations` balance is tuned for the ubuntu primary leg; other legs may run hotter shards.
- Concurrency headroom is against an **account-wide** cap shared with other automation; a future N-bump
  or 5th matrix leg is bounded by it.

## Attribution

Internal (optimization follow-up). Builds on ADR-023 (slow-gating), ADR-057 (pandas span), and the
`test_ci_*_wired` structural-guard idiom.
