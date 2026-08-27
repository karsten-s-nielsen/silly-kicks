# ADR-073: A deterministic sub-quadratic-growth test harness (scale blind-spot closure)

| Field | Value |
|---|---|
| **Date** | 2026-08-26 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

> Ships in silly-kicks **4.95.0 (PR-S166)**; ADR-073 confirmed next-free against merged `origin/main` (highest committed ADR-072, CHANGELOG top 4.94.0 / PR-S165).

## Context

silly-kicks guards performance with two layers and **both are blind to a scale-only O(n²) regression**. The structural guards (`call_counter` / `row_iteration_counter`, the `*_perf_budget.py` set, `TestCoverShadowComplexityGuard`) are all **scale-invariant**: they fix the fixture size and assert a call-count invariant, so a `call_counter == 1` passes whether the one call is internally O(n) or O(n²). pytest-benchmark only **measures**, never asserts. There was no growth/complexity assertion anywhere. The 4.92.0 turnover bug (ADR-068 / PR-S163) was exactly this shape — one `EmpiricalTurnoverValue.fit` call, internally O(n²), invisible to `call_counter`, only measured — and it reached a downstream lakehouse report instead of CI. This ADR closes that gap and pays down two follow-ups the ADR-068 review deferred (per-site mismatched-dtype tests).

Constraints: **deterministic** (no flaky wall-clock), runs on every CI leg, **no production behaviour change** (tests + docs + convention only; no retrain; C4-free), no new runtime dependency.

## Decision

Add a general, reusable **`tests/_perf_structural.assert_subquadratic_growth(measure_work, *, sizes=(256,1024,4096), max_exponent=1.5)`** that asserts the empirical **operation-count growth exponent** `log(work_hi/work_lo)/log(size_hi/size_lo) <= max_exponent`. Because the counts are **integers**, the exponent is exact and the guard **never flakes**; fixed overhead only biases the exponent *down* (toward "more linear"). Reference exponents at these sizes: linear 1.0, n·log n 1.16, n^1.5 1.50 (boundary), quadratic 2.0 — a **quadratic-ish detector** by design. Because the assertion is on the *ratio*, fixtures stay small (256–4096) and fast.

Each adopter supplies a **scoped** counter that isolates its super-linear-suspect operation (`rows_scanned_counter` — boolean-mask `__getitem__` + `.groupby` construction + axis-0 `.take` — is the rescan proxy; compiled kernels count their pure-Python fallback via a counting-array; a constant-work primitive like the databricks loader uses an equality guard). A **`SCALE_GUARDED` registry** + meta-assertions force coverage: every `group_rows` caller (AST-derived) must register a guard, entries must resolve, and a degenerate-by-design entry must carry a discriminating companion. Applied to **12 adopters** (the 8 `group_rows`-calling functions + `_opp_first_shot_scan`, `_possession_labels`, and both `add_possessions`). Ships alongside the mismatched-dtype characterization tests (2 seam-level + one end-to-end per library `group_rows` consumer, 7; the databricks query-builder is covered by the constant-query guard instead).

**A rescan is O(groups × table), so a growth fixture MUST scale the *group* (loop-iteration) dimension, not a within-group dimension** — otherwise the loop count is constant, the regressed rescan stays linear, and the guard passes on the buggy code (a guard that cannot guard). Three fixtures shipped that way in the first draft and were caught only by running the discrimination proof: `_off_ball_runs_kernel`/`detect_off_ball_runs` scaled frames within ONE game and `derive_goalkeepers` scaled frames within a FIXED 2×2 `(game,team)` set — all corrected to scale the number of games (the realistic batched-corpus axis). The lesson: **`assert_subquadratic_growth` passing is necessary but not sufficient; every adopter needs the regression-goes-quadratic proof, and a fixture that scales the wrong axis is silently vacuous.**

## Alternatives considered

| Option | Why rejected |
|---|---|
| Absolute work-bound `work <= C·n` | Each `C` is a hand-tuned magic number that drifts — the flaky-budget failure mode moved from time to a constant; can't distinguish n·log n from n². |
| Generic always-on pandas hook (count for all code) | Over-counts unrelated framework ops, misses pure-Python/numba loops, un-scopable. The chosen counters patch a few **public** method entry points and install only for the measured call. |
| **AST rescan-pattern lint** (`df[df[k]==v]` in a loop) | **Declined.** ADR-019 deleted its id-compat AST lint because a safe and an unsafe compare are the identical AST, which bred false-positive exemptions; the rescan pattern is more distinctive but an intentional small-inner-collection filter would still false-positive. Recorded here as considered-and-declined. |

## Consequences

### Positive
- CI now catches a scale-only super-linear regression in any guarded primitive; the 12 highest-risk primitives (all ADR-068 sites + the two real 4.92.0 quadratics) are guarded, and a new `group_rows` caller must register a guard or CI fails.
- The harness is deterministic (integer counts) and small-fixture — never flakes, runs on every leg.
- **Each `group_rows` guard was proven DISCRIMINATING** — a rescan shim (the pre-ADR-068 raw-`==` defect) monkeypatched into each adopter drives every regressed exponent to **~1.85–2.0** (all clearly above the 1.5 boundary) while the shipped code stays exactly **1.0**. turnover and possession carry their own dedicated discrimination proofs (`test_turnover_fixture_breaks_actually_bind`, `test_possession_labels_ref_loop_is_superlinear`); the two `add_possessions` guards are non-vacuous linear-scaling guards. Fixture sizes are kept small (top of 128–256 per adopter) so the whole growth-guard suite adds ~11s to a CI leg — the exponent is exact for the deterministic integer counts, so a small span discriminates just as cleanly as a large one.

### Negative
- One new test-only seam plus a registry a future reader must know about. A quadratic in an *un-counted* part of an adopter is invisible (each counter is chosen to be the dominant suspect op, documented per adopter; e.g. `compute_defensive_credits` pre-links to isolate the group_rows site from the pre-existing linking cost).

### Neutral
- Tests + docs + one CLAUDE.md bullet only — **no production behaviour change, no retrain, C4-free.** Ships as silly-kicks 4.95.0 / PR-S166 for traceability (`silly_kicks` byte-identical apart from the version string).

### Known limits (stated, not discovered)
- **Sub-n^1.5 regressions are out of scope by design** — n^1.5 sits on the default boundary; a milder super-linear passes.
- **A quadratic sharing its counter with a LARGE linear co-term can be masked at small n** (measured: `n²+10000n` → exp 1.05 at 256/1024). Mitigated by counter-isolation + the third size; a `≥10⁴·n` co-term is not caught at default sizes.
- **A brand-new rescan that neither routes through `group_rows` nor rebuilds a `groupby` in the loop is not force-caught** — no reliable AST signal for "has a dominant loop"; genuinely new loops rely on review + the convention.

## Attribution

Internal (optimization-audit follow-up). Builds on ADR-068 (rescan-in-loop remediation) and ADR-019 (`id_compat`, and the deleted-lint lesson).
