# CI wall-clock < 10 min — measure the suite, then parallelize + fix the tail

**Date:** 2026-08-26
**Status:** Draft (brainstormed; revised after review 1; awaiting spec review → plan)
**Owner:** Karsten S. Nielsen
**Cycle:** Follow-on to the scale-guard harness (same feature branch, second commit)

## 1. Problem & goal

silly-kicks CI takes **22–25 min** wall-clock and runs twice per release (PR + post-merge main).
Billing is **not** the constraint (the account has never exceeded its monthly minutes); **latency is.**

**Deliverable (SLA, not a mechanism):** a real CI run completes in **< 10 min** GitHub wall-clock,
with **zero coverage loss** and the CI-integrity contracts (ADR-023 slow-gating, ADR-057 pandas span,
the `test_ci_*_wired` guards) preserved. *How* we get there — shard count, which slow tests to
fix/cache, numba caching — is chosen by the plan from measured data (§2), not asserted here.

### Measured breakdown (main run `33000702218`, 22:07 total)

| Job | Wall | Dominant step |
|---|---|---|
| **test ubuntu-3.12 (primary)** | **21:57** | bulk `pytest -m "not e2e"` (incl `@slow`) = **20:19**; install 49s; benchmark-only 37s; doctest 3s |
| **test windows-3.12** | **20:04** | bulk `pytest -m "not e2e and not slow"` = **17:48**; install 1:49 |
| test ubuntu-3.10 | 12:57 | bulk `not slow` ~12 min |
| test ubuntu-3.11 | 12:39 | bulk `not slow` ~12 min |
| lint | 2:07 | ruff + pyright |
| pandas-span | 4s | waits on all test jobs |

**Root cause:** a single **serial** `pytest` process per leg (~18–20 min). `pytest-xdist -n auto`
inside one job was tried and **reverted** — it memory-killed py3.12 on the 4-core/7 GB runners (4×
heavy ML imports + 4× numba compiles + the ghost-GK suite). Intra-job parallelism is closed on these
runners; the lever is **splitting the suite across parallel jobs**, each on its own runner.

## 2. Approach — measure first, then apply the cheapest mix of three levers

Wall-clock is bounded below by **the longest thing a single job must do that cannot be divided**:

```
per-shard floor  ≈  max(longest single test)  +  install  +  numba cold-compile  +  collection
```

Sharding drives down the *sum-of-tests* term (`total / N`) but **cannot** push a shard below its
longest indivisible test. The primary leg carries the `@slow` set (train-script smokes, calibration
cache-equivalence) — exactly where a multi-minute *individual* test could sit. So `total / N` is a
ceiling on the win, not the win.

**Plan Task 0 (measurement, gates everything):** profile the top ~20 + single-longest test with
`--durations=0`, and — critically — **budget BOTH the ubuntu-primary AND the windows leg** (review 2,
N2). The two legs bind for *different* reasons and the SLA holds only if *both* clear 10 min:

- **ubuntu-primary** (`-m "not e2e"`, install 49 s): test-bound. Floor `F_ubuntu =
  longest_single_test + 49 s + numba_warmup + collection`.
- **windows-3.12** (`-m "not e2e and not slow"`, install **1:49**): **install/numba-bound.** Sharding
  divides only the *test* term; the 1:49 install is a per-job fixed cost **paid N× in parallel, never
  divided**. Rough floor `F_win ≈ 17:48/N + 1:49 + numba_warmup ≈ 8:15 at N=3` — margin only ~1:45, and
  **the binding leg for "< 10 min" is plausibly Windows, not primary.** So N must be tuned to Windows,
  and **Levers C (numba cache) + pip cache matter MOST on Windows and least on primary** — the plan
  prioritizes caching there, not as a uniform win. (Cold-cache first run may sit higher; the SLA is a
  steady-state claim — record that caveat rather than hide it.)

Then **derive**, not assume: the per-shard floor per leg; the shard count `N` such that `total/N` is
comfortably above the floor on **both** legs and under the SLA with margin; and **which tail tests to
fix or cache first (Lever B / C)** — every minute off the *longest single test* both raises the floor
sharding can reach and lets `N` drop (less matrix, less `.test_durations` churn, less concurrency
pressure).

**Preliminary profile (this brainstorm, local Python 3.14, `-m "not e2e" --durations=30`, 7930 tests
in 19:27):** the **longest single test is 39 s** (`test_train_script_smoke_position_only`; the two
xcross train-smokes 39/38 s, calibration cache-equivalence 37 s, then a cluster of DAS/gkdv tests
9–32 s). **No multi-minute indivisible test exists** — so B1's worst case does not occur: the ubuntu
per-shard floor is ≈ 39 s + 49 s install + numba warmup ≈ **2–3 min**, and the N=3 ubuntu shard budget
(~6.8 min) sits **far above** it → **N=3 is not floor-limited on ubuntu, and Lever B is NOT required to
hit the SLA** (sharding A + numba cache C suffice; B stays optional/long-term). The binding constraint
is therefore **Windows install (1:49, undivided)**, exactly N2 — so plan Task 0's job is less "find a
scary long test" (there is none) and more "confirm the Windows leg clears 10 min with caching," on a
real CI runner across both legs.

The plan chooses a mix of three levers from that data:

- **Lever A — matrix-sharding via `pytest-split`.** Each job runs `pytest … --splits N --group
  ${{ matrix.shard }}`, partitioned into N **duration-balanced** groups from a committed
  `.test_durations` (count-balancing is useless: a few `@slow`/ghost-GK tests dominate). Each shard =
  its own runner → the `xdist` memory-kill cannot recur.
- **Lever B — make the top-K slow tests faster WITHOUT verifying less** (review 2, N3). This is the
  *long-term* half of the ask: a 20-min serial suite for a pure pandas-in/pandas-out library is itself
  a smell, and sharding alone *hides* it behind more machines. **Admissible B-moves** exercise the same
  code path for less wall-clock: memoize an expensive artifact *within a run* across tests that share
  it, shrink a fixture while still running the full path, or lean on Lever C's numba cache.
  **INADMISSIBLE in this cycle — a coverage decision, not a speedup:** replacing a re-fit / e2e /
  train-smoke test with a **frozen-output assertion** (a re-fit test proves *fitting is still
  reproducible*; a stored-vector assert proves only *inference matches a vector* — the fit path stops
  being exercised). A latency cycle must not smuggle a coverage trim; "a guard that never runs is not a
  guard" (ADR-023 ethos). Anything requiring a retrain, a fixture redesign, or that would verify *less*
  is **recorded, not forced** — surfaced to the owner as a separate coverage decision.
- **Lever C — cache numba's compiled kernels.** The reverted xdist was killed partly by "4× numba
  compiles"; sharding removes the *memory* contention but each shard still cold-compiles the `@njit`
  pitch-control + ghost-GK KDE kernels. Set `NUMBA_CACHE_DIR` + `actions/cache` on it so compiled
  kernels restore across runs — directly lowering the per-shard **floor**, on every leg.

**Working hypothesis (owner-set, to be validated by Task 0):** **N = 3** with Levers A + C, Lever B
applied only to whatever the profile flags as clearly-cheap. N=3 → `4 legs × 3 = 12` test jobs; if
Task 0 shows a single test within ~2 min of the N=3 budget, the plan either fixes that test (B) or
raises N — and, since N was an owner decision, surfaces the choice rather than silently changing it.

### Constraints

- **GitHub Free → 20 concurrent jobs, ACCOUNT-WIDE** across every repo/workflow (this owner also runs
  HF publish + other automation). The cap limits **peak-concurrent** jobs, which at N=3 is **~14**
  (lint + 12 shards + standalone benchmark; the two `needs: test` aggregators run downstream, not
  concurrently) — *when nothing else is in flight*. An overlapping PR + main push (different refs → not
  covered by `cancel-in-progress`) momentarily demands **~28**, over the cap → GitHub queues (a latency
  blip, not a failure). So **N is bounded by the concurrency cap, not just by balance** — a future
  "just bump N" (or a 5th matrix leg) is not free.
- **Coverage decision (A, owner-set):** preserve **all** current coverage — every leg runs its whole
  suite, just sharded. Lever **(B-Windows)** — trim Windows to only the platform-/version-sensitive
  subset — is **explicitly deferred** to a later cycle if minutes/cost/principle warrant it (§9).
- No paid runners; no new **runtime** dependency (`pytest-split` is dev/test-only in `[test]`).

## 3. Job graph (target at the working N=3; N is a Task-0 output)

```
lint                              unchanged (~2 min)
test   os × python × shard[1..N]  N·4 jobs; each runs one duration-balanced shard
                                  each leg's shard 1 ALSO runs doctest + records pandas-major (~12 s)
benchmark                         STANDALONE parallel job, primary env, `-m "not e2e" --benchmark-only`
pandas-span      needs: test      asserts the pandas-major span (unchanged intent)
shard-reconcile  needs: test      asserts the shards PARTITION each leg's suite by NODE-ID SET (§6b)
```

**Benchmark is a standalone parallel job, NOT folded onto a shard (reversed after review 2 — N1).**
The deliverable is wall-clock; folding benchmark(37 s)+doctest+pandas onto primary shard 1 would make
it the systematically heaviest shard, and `.test_durations` cannot rebalance a non-test load away from
it — adding ~40 s directly to the critical path on the critical leg. Standalone, benchmark's
install(~50 s)+run(37 s) ≈ 1.5 min runs **fully hidden** behind the ~7-min shards and contributes **0**
to wall-clock. The only cost is one job + billed minutes — both explicitly non-constraints. doctest +
pandas-record stay on each leg's shard 1 (they are per-leg-once and version-sensitive; ~12 s, which
Task 0 budgets into that shard).

Job count at N=3: **16 distinct** (lint + 12 test + benchmark + pandas-span + shard-reconcile), but the
two `needs: test` aggregators run *downstream*, so **peak-concurrent ≈ 14** (lint + 12 shards +
benchmark). That is the number the 20-job cap limits. Whether N=3 lands < 10 min is confirmed by Task 0
(both legs — §2) + a real run, never by `20:19 / 3`.

## 4. Sharding mechanics (Lever A)

- **Command per leg** (marker selection unchanged; sharding + determinism flags appended):
  - non-primary: `pytest -m "not e2e and not slow" --splits N --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25`
  - primary:     `pytest -m "not e2e" --splits N --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25`
  `pytest-split` re-computes the split **within each leg's marker selection**, so a leg's shards always
  cover exactly that leg's set (primary includes `@slow`; others exclude it — ADR-023 preserved).
- **`--durations=25` is KEPT per shard** (reversing the first draft): when a shard drifts toward
  budget (the §2 floor / cross-env imbalance failure mode), the per-shard durations tail is how you see
  *which* test did it without re-profiling. It is free.
- **Collection-order determinism is load-bearing and now PINNED.** `pytest-split` yields a valid
  partition only if every shard collects in the same order. Today the `[test]` extra has no
  `pytest-randomly`/`pytest-xdist` and `conftest.py` has no ordering hook, so order is deterministic —
  but a future (even transitive) `pytest-randomly` auto-activates and silently breaks the partition.
  Two defenses: `-p no:randomly` in every sharded command (works even if the plugin is present), and a
  static assertion in the §6a guard that no collection-shuffling plugin is active.
- **N is two literals kept consistent by the §6a guard:** the matrix `shard: [1..N]` list and the
  `--splits N` value (GitHub cannot compute a matrix length into a flag). Changing N later is a
  two-line edit the guard protects.
- **`.test_durations`** (committed at repo root, `pytest-split`'s default): affects **balance only,
  never coverage** — an unknown-duration (new/renamed) test gets the plugin's fallback and is still in
  exactly one group. **Balance is tuned for the ubuntu primary leg**; windows (~2×) and other
  py-versions may run hotter shards — acceptable, because the reconcile guard (§6b) still proves
  completeness and the worst case is one slower shard, never a dropped test. Generate the file from a
  **CI primary-runner** profile where feasible (closer to what actually runs than the 16-core dev box).
  **Regen trigger (documented, manual for v1):** regenerate on a material suite shift or when a shard
  drifts toward budget; a `scripts/` one-liner + a `CLAUDE.md` note records it.

## 5. Install & caching (honest accounting)

- **`setup-python` `cache: pip` caches the wheel DOWNLOAD cache, not the built/installed env.**
  `pip install -e ".[kloppy,xgboost,das,test]"` still re-runs the editable build + install every job;
  we save *download*, not *install*. And it does **not** help same-commit parallel jobs — the cache is
  written at a job's END, so N shards launched together on one commit all miss and race to populate;
  only *subsequent commits* restore. Net: a real but modest win, and it does **not** shrink the
  per-shard install line in the first run's budget. Added to the `test` jobs only; the lint job's exact
  typing pins (ruff/pyright/pandas-stubs/numpy) stay untouched (ADR-057 coverage).
- **`NUMBA_CACHE_DIR` + `actions/cache` (Lever C)** is the higher-leverage caching win — it lowers the
  per-shard *floor* directly and helps every leg. See §2 Lever C.

## 6. Anti-silent-drop guards (both, per owner decision (b))

Sharding must **never** silently drop a test — this repo enumerates-and-conserves over actual keys
everywhere (ADR-052 conservation, the mirror/purity/id registries, `pandas-span`). Two guards with a
**distinct** division of labour (not redundant):

- **(a) Static wiring guard** — `tests/test_ci_shard_wiring.py` parses `ci.yml` and asserts, *before
  any run*: the `shard` axis is `[1..N]` contiguous; every `test` pytest command carries
  `--splits <N> --group ${{ matrix.shard }}` with the **same** `N` as the shard-list length; every
  sharded command carries `-p no:randomly`; and the `shard == 1` conditionals gate exactly the doctest
  + pandas-major steps (simple `matrix.shard == '1'` predicates — benchmark being a standalone job per
  N1 means no compound `matrix.primary && shard` predicate to normalize, keeping this within the
  existing `test_ci_*_wired` `matrix.primary`/`!matrix.primary` idiom). Catches the **wiring typo**
  (matrix `[1,2]` while `--splits 3` → group 3 never runs) at parse time.
- **(b) Runtime reconcile job** — `shard-reconcile` (`needs: test`). Each `test` job uploads its
  shard's **collected node-ID list** (`pytest -m "<markers>" --splits N --group <shard> --co -q`);
  each leg's shard 1 also uploads that leg's **full** collected node-ID list (`--co -q`, no splits). The
  reconcile job asserts, **per leg**: `union(shard node-ID sets) == full set` **AND** the shard sets
  are **pairwise-disjoint**. This is strictly stronger than a count sum (which passes when one shard
  *duplicates* A while another *drops* B — the miscounts cancel) and it is the only guard that can catch
  the failure a YAML parse cannot: **cross-runner collection divergence**, where a module-top
  `importorskip` resolves differently on one of the 3–4 machines (cache/network flake) and silently
  changes that runner's collected set. Fails loud with the leg + the offending node IDs. Modeled on
  `pandas-span`; node-ID sets are the same enumeration idiom the repo uses elsewhere.

## 7. Risks & mitigations

- **The shardability trap is SPECIFIC, so Task 1 probes for it — not a generic "run 3 shards, confirm
  green."** The core library is pure; the *test suite* is not automatically. The concrete hazard is a
  test that **writes a file another test reads** — e.g. a script-invoking `@slow` test that regenerates
  a golden / `SHA256SUMS`, asserted by a separate test; serial-in-one-process hides the dependency,
  split-across-machines surfaces it, and this repo *does* have script-driven golden regeneration. Task 1
  (a) greps for tests that write into `silly_kicks/**` or `tests/datasets/**` and pairs them with their
  readers, and (b) runs each of the N shards in a **fresh** process and confirms green. A real
  cross-shard dependency is a bug to fix or isolate, not a reason to abandon sharding.
- **Per-shard floor / cross-env imbalance** (§2, §4) — bounded by Task 0; `--durations=25` surfaces the
  culprit; Lever B/C lower it; reconcile still proves completeness.
- **Concurrency cap (Free = 20, account-wide)** — §2. ~14 peak-concurrent leaves headroom;
  `cancel-in-progress` collapses same-ref reruns; an overlapping PR+push (~28) queues (latency blip),
  not a failure. N is bounded by this, stated so a future bump is not mistaken for free.
- **Windows still 2× hardware** — after sharding, windows shards are within budget; its 2× billing is
  irrelevant (billing not a constraint); full-suite coverage preserved per (A). Deferred trim = §9.
- **`.test_durations` diff churn** — data, not code; repo-root, outside ruff/pyright globs.

## 8. Success criteria

- A real CI run (PR or main) completes in **< 10 min** wall-clock, measured via `gh run view` on the
  actual sharded run (not `total/N` arithmetic).
- **Zero coverage loss:** the reconcile job (§6b) proves — by node-ID set union/disjointness, per leg —
  the shards partition each leg's suite; every OS×Python leg runs its full selection;
  `@slow`/benchmark/doctest/pandas-span all still run.
- Task 0's `--durations=0` profile is recorded (in the plan/PR), and the chosen N is **derived** from
  the longest-single-test floor on **both** the ubuntu-primary and the windows leg (the install-bound
  one), not asserted — the SLA holds only if the slowest leg clears 10 min.
- The updated `test_ci_*_wired` suite (incl. the new shard-wiring + determinism assertions) is green.
- No new runtime dependency; `pytest-split` is dev/test-only.

## 9. Rollout & non-goals

- **Commits on the one `scale-guard-harness` branch, no squash:** (1) the scale-guard harness (ready);
  (2) the CI **sharding infrastructure** (`ci.yml`, the wiring/reconcile guards, `pytest-split` in
  `[test]`, caching) **with `.test_durations` committed in the SAME commit** — if the durations file is
  absent on the first sharded run, `pytest-split` silently falls back to *count*-balancing (valid
  partition, bad balance) exactly on the debut run (review 2, nit 3). **(3) IF Lever B produces any
  test-body edits**, they land as their **own** commit, so a Lever-B regression never forces reverting
  the sharding and vice-versa (review 2, nit 4; owner set the 2-commit structure — this 3rd commit
  materializes only if Task 0 finds a clearly-cheap, coverage-preserving tail fix). One PR. Each commit
  carries its own version bump; exact numbers assigned at commit-prep from merged `origin/main`, the
  final tagged.
- **Deferred (explicit, owner-decided):** lever **(B-Windows)** — trim Windows to the platform-/
  version-sensitive subset — recorded in `TODO.md` on ship as the next latency/cost lever; not built
  here. Any tail-test fix (Lever B) that would need a retrain or a fixture redesign is likewise recorded,
  not forced into this cycle.
- **Non-goals / YAGNI:** no larger/paid runners; no scheduled durations-regen workflow (manual regen
  documented); no `uv`/venv-cache rework (pip + numba cache suffice for the SLA); no change to *which*
  tests exist or their markers beyond appending `--splits/--group/-p no:randomly`.
