# Design: CI-time reduction via slow-test gating (measurement-first)

**Date:** 2026-06-08
**Status:** Approved (brainstorming) → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** The recurring "~20-min CI" first-class problem (see `feedback_ci_test_perf_investigate_not_patch`),
surfaced again on the SK-xT-2 (#113) run.

## Context

CI's `test` job is a 4-leg matrix — `ubuntu-latest` × {3.10, 3.11, 3.12} + `windows-latest` × 3.12 — and
each leg runs the **full** non-e2e suite twice: a bulk step (`--benchmark-skip`) and a benchmark step
(`--benchmark-only`). Measured on the SK-xT-2 CI run:

| Leg | Wall-clock |
|---|---|
| lint (ubuntu 3.12) | ~1.5 min |
| ubuntu 3.10 / 3.11 / 3.12 | 8.5 / 10 / 12.4 min |
| **windows 3.12** | **~16–20 min (long pole)** |

Heavy integration / training-subprocess / heavy-numeric tests (xS/xCross/ghost-GK train smokes,
ghost-GK KDE parity, calibration objective cache-equivalence) run on **all 4 legs**, and the slow Windows
runner runs everything. A local `--durations` profile is **not** a faithful proxy for CI cost: the CI
test-extra `[kloppy,xgboost,test]` does **not** install `pyright`, so `test_pyright_clean_tracking_namespace`
(~8.7 s locally) **skips on CI** — but it DOES install `ruthless-efficiency[optuna]` + `xgboost`, so the
calibration objective tests (≈38 s locally) **do run on CI** (an earlier draft of this spec wrongly
assumed they skip — corrected). Beyond that, the same test can cost very differently on the Windows
runner than locally. So the `slow` set must be chosen from **real CI (Windows-leg) timings**, not local
ones — the measure-first rule stands regardless of the exact skip accounting.

The `slow` pytest marker is already registered (`pyproject.toml [tool.pytest.ini_options].markers`) and
applied to 3 tests, but it currently has **no gating effect** — every CI leg runs `-m "not e2e"`, which
does not exclude `slow`.

## Decision

Gate heavy tests + the benchmark step to a single **primary leg (`ubuntu-latest` 3.12)**, keeping the
full unit/contract suite on every leg, **driven by real CI durations** measured first. No xdist.

## Scope

### In scope
1. **Phase 0 — CI measurement (diagnostic).** For the one-time analysis push, add **`--durations=0`** to
   the CI bulk-test step (full distribution — so a heavy test at rank #31 isn't invisible). Push and read
   the **actual `windows-latest` 3.12 per-test timings**, classifying on **setup + call + teardown summed**
   (fixture-borne ghost-GK/xT fits show a cheap `call` but expensive `setup`; `--durations` reports the
   phases separately — sum them). This is the ground truth for the `slow` list. Phase 1 settles the
   *permanent* flag at `--durations=25` (a CI-log trend signal — not durable history; logs expire on the
   repo's retention window, and a durations artifact upload is out of scope).
2. **Phase 1 — gating, informed by Phase 0 data.**
   - **Mark `@pytest.mark.slow`** the tests that satisfy the (three-clause) selection criterion below,
     confirmed against the Phase-0 CI timings + per-test invariance classification.
   - **CI YAML — single source of truth for the primary leg via a matrix `include` flag.** Add
     `include: [{os: ubuntu-latest, python-version: "3.12", primary: true}]` — GitHub *merges* this into
     the existing ubuntu-3.12 combination (matching `os`+`python-version` augments rather than spawns a
     leg), so `matrix.primary` is `true` on exactly that leg and null/falsy elsewhere. Every step then
     keys off the **one** truth: non-primary bulk `if: ${{ !matrix.primary }}` runs `-m "not e2e and not
     slow"`; primary bulk `if: ${{ matrix.primary }}` runs `-m "not e2e"` (everything, incl. `slow`) in a
     single process. **This eliminates the hand-copied `os==… && py==…` predicate** — its drift failure
     mode is catastrophic (retarget the primary leg, update 2 of 3 copies → a leg matches *neither* bulk
     step → runs **zero** bulk tests → green-but-empty, the repo's cardinal silently-skipping sin). With
     the flag, the partition is structural and drift-proof.
   - **Benchmark step** (`--benchmark-only`): `if: ${{ matrix.primary }}` (non-asserting trend data after
     the structural-guard conversion; Windows / older-py numbers are noisy). The primary leg's only
     *additional* process.
   - **Gating tripwire (ships in-scope, red-first):** a meta-test that **parses `ci.yml`** (structured
     YAML, not substring grep) and asserts: (i) **exactly one** matrix entry has `primary: true`; (ii) the
     bulk steps key off `matrix.primary` (both `!matrix.primary` and `matrix.primary` branches present) so
     the 4 legs partition into exactly one bulk step each; (iii) the `slow`-marked collection is non-empty.
     Written **red-first** (fails on the pre-change `ci.yml`, passes after) to prove it discriminates. Plus
     a one-line criterion rule in `CLAUDE.md`. The full auto-marker-lint (enforcing new heavy tests carry
     `@slow`) stays deferred.
3. **Phase 2 — CI validation before merge.** Push the gated branch; confirm on a **real CI run** that the
   Windows long-pole dropped, the primary leg still runs the slow + benchmark steps green, and all legs
   pass. Only then merge. (Hard lesson: perf/parallelism tuned locally does not transfer to CI — validate
   on CI, per `feedback_ci_test_perf_investigate_not_patch`.)

### Selection criterion for `@pytest.mark.slow`
A test is marked `slow` iff **all three** hold:
- (a) it is genuinely expensive on the CI Windows leg (Phase-0 call-time at or above ~5 s, or a
  subprocess training-script smoke regardless of the exact second-count), **and**
- (b) it is an **integration / training-subprocess / heavy-numeric** test — NOT a cheap
  behavioral-contract guard, **and**
- (c) it is **platform- AND interpreter-INVARIANT**: its correctness does not depend on the OS or the
  Python/numpy version. "Does-it-run" subprocess train smokes and same-run internal-consistency checks
  (e.g. numba-`@njit`-vs-numpy parity computed in one process) qualify — the property holds identically
  per platform, so running it once on the primary leg loses nothing.

**`slow` therefore means "expensive AND safe to run once on the primary leg."** Two distinct categories
are deliberately **NOT** marked `slow` and stay on **every** leg:
- **Cheap behavioral-contract guards** — parity/golden, the dup-`action_id` gate, id-dtype-invariance
  (ADR-019), provider orientation/roster. They are the cross-version/platform regressions we most want
  caught, and they are cheap.
- **Expensive *version-sensitive* tests** — any heavy test that asserts an absolute / golden-hash /
  numeric-snapshot value. These are slow **and** exactly the tests history shows fail
  platform-/version-specifically (HGBR binning differing across py3.10/3.11/3.12; numpy micro-version
  hash drift). Gating them to one leg would discard the OS axis (`windows-3.12`) *and*, if leg-restricted
  at all, the interpreter axis (`3.10/3.11`). So they keep full all-leg coverage despite being slow.

**Why per-test invariance, not a blanket "slow on both 3.12 legs":** a blanket two-leg rule would keep
the platform-invariant train-smokes on Windows (no long-pole win) yet still drop version-sensitive tests
from `3.10/3.11` (interpreter-axis coverage loss). Routing by invariance instead removes the
platform-invariant heavy set from Windows entirely (the win) while keeping version-sensitive tests on all
legs (full OS *and* interpreter coverage). It dominates the blanket rule on both axes.

The primary movers expected in the `slow` set are the **train-script smokes** (xS / xCross / ghost-GK CLI
+ integration — "does-it-run", platform-invariant) and any **same-run internal-consistency** numeric
tests. The 3 already-`slow`-marked tests (hybrid-VAEP-with-tracking ×2, atomic-with-tracking) keep their
mark (and now actually gate) **iff** they meet clause (c) on the Phase-0 read.

**The plan enumerates each candidate's classification — absolute/golden/version-sensitive vs.
internal-consistency/does-it-run — read from the Phase-0 CI timings + the test body.** That single
distinction decides each test's routing and is the load-bearing output of Phase 0.

### Out of scope (explicit non-goals)
- **xdist / parallelization.** Reverted before — it regressed the 4-core/7-GB runners (py3.12 pass→OOM
  kill). Source-level gating only.
- **Session-scoped fixture caching** of repeated ghost-GK/xT fits. A real lever, but deferred: gating
  removes the platform-invariant heavy tests from the non-primary legs first; revisit only if Phase-2
  shows the **primary** leg is still the bottleneck (YAGNI until measured). If it is, the first lever is
  the single-bulk-process design already in this spec, *then* caching.
- **Full auto-marker-lint** that enforces every new heavy test carries `@slow`. Deferred — but a cheaper
  tripwire (the CI-wiring meta-test + CLAUDE.md rule) ships *in scope* now, so the gating cannot silently
  un-wire and the convention is documented.
- **`test_pyright_clean_tracking_namespace`.** Skips on CI (pyright absent from `[test]`); not a CI cost.
  Left untouched.
- **Version bump / publish.** This is CI + test-infra only — `ci.yml` is not shipped, the `@slow`
  decorators are not in the wheel, no library code changes. Recommend **no version bump and no tag**
  (also avoids colliding with the other session's pending `v4.19.1` tag). Final call at merge — see
  Coordination.

## Architecture / mechanism

The primary leg is identified by a **single matrix `include` flag** (`matrix.primary`), not a hand-copied
`os`/`py` predicate — so the truth lives in one place and the matrix partition is structural. Sketch
(`.github/workflows/ci.yml`, `test` job):

```yaml
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
        exclude:                                    # unchanged
          - { os: windows-latest, python-version: "3.10" }
          - { os: windows-latest, python-version: "3.11" }
        include:
          - { os: ubuntu-latest, python-version: "3.12", primary: true }  # merges into the existing leg
    steps:
      - run: pip install -e ".[kloppy,xgboost,test]"
      # Non-primary legs: full fast/contract suite, platform-invariant slow set excluded. ONE process.
      - if: ${{ !matrix.primary }}
        run: pytest tests/ -m "not e2e and not slow" --benchmark-skip --tb=short --durations=25
      # Primary leg: EVERYTHING incl. slow, in a single bulk process.
      - if: ${{ matrix.primary }}
        run: pytest tests/ -m "not e2e" --benchmark-skip --tb=short --durations=25
      # Primary leg only: benchmark measurements (non-asserting trend data).
      - if: ${{ matrix.primary }}
        run: pytest tests/ -m "not e2e" --benchmark-only --tb=short
```

(Phase 0 is just `--durations=0` on the *existing single* `-m "not e2e"` bulk step — the measurement push.
The matrix `include` flag, the `!matrix.primary` / `matrix.primary` split, the `slow` marks, and the
benchmark guard land in Phase 1 once the timings are read; the permanent flag settles at `--durations=25`.)

## Coordination / isolation

1. **Active parallel session.** Another session is merging to `main` (just landed 4.19.1, tag pending).
   This change is **CI-config + test-marker only** — low conflict surface with their tracking/spadl work.
   Rebase/FF onto the latest `main` before the Phase-2 validation push and again before merge.
2. **No version interaction.** Recommend no version bump (rationale above), so no collision with their
   pending `v4.19.1` tag. If the maintainer prefers the per-PR-bump convention, it would be a patch
   (next free after their release lands) with **no publish** — decided at merge.
3. **C4-free, NOTICE-free, ADR recommended.** No architecture/dependency/published-method change, so
   C4/NOTICE are untouched. But the **invariance-routing rule is a non-obvious convention with downstream
   consumers** — a future maintainer will challenge "why does the heavy suite run only on ubuntu-3.12 —
   isn't that a coverage hole?". It clears the final-review Phase-2.5 ADR bar ("introduces a convention
   with downstream consumers"). CLAUDE.md states the *what*; a short **ADR (next free number, ~023)**
   preserves the *why* — per-test invariance dominates a blanket two-leg rule, version-sensitive tests
   stay on all legs, measure-first, xdist stays reverted. Write it in this PR.
4. **Branch:** `ci/slow-test-gating` (already created, off `main`). One feature branch; commits per phase
   are acceptable here because Phase 0 must be pushed to CI before Phase 1 can be authored — but the
   whole thing is one PR. No commit/push without explicit approval.

## Success criteria
- Phase-0 `--durations` output visible in a real CI run; the `slow` list chosen from those numbers, with
  each candidate classified invariant (→ `slow`) vs. version-sensitive (→ stays on all legs).
- Windows 3.12 leg wall-clock **materially reduced** by removing the platform-invariant heavy set (target
  quantified from Phase-2); the primary leg does not become the new long pole (one bulk process + the
  benchmark process).
- **No coverage lost where it matters:** version-sensitive heavy tests + all contract guards still run on
  **every** leg (OS *and* interpreter axes); platform-invariant slow tests run once on the primary leg.
  No test deleted or permanently skipped.
- Gating tripwire green: the structured-YAML meta-test asserts exactly one `primary: true` matrix entry +
  both `matrix.primary` branches present (the 4 legs partition into exactly one bulk step each) + a
  non-empty `slow` set — so no leg can silently run zero bulk tests and the gate can't silently un-wire.
  CLAUDE.md criterion rule present; ADR (~023) records the why.
- A real CI run (Phase 2) green before merge; branch FF'd onto the latest `main` (already at 4.19.1
  `d23a352`; its `v4.19.1` tag may not exist yet — do not assume it).
