# ADR-023: CI slow-test gating — invariant heavy tests on a single primary leg

| Field | Value |
|---|---|
| **Date** | 2026-06-08 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen; main-worktree session (4 review rounds) |

## Context

CI's `test` job is a 4-leg matrix — `ubuntu-latest` × {3.10, 3.11, 3.12} + `windows-latest` × 3.12 — and
each leg ran the **full** non-e2e suite (bulk + `--benchmark-only`). Measured on a real run, the
`windows-latest` 3.12 leg was the **~16–20 min long pole** (ubuntu legs 8.5–12.4 min, lint ~1.5 min)
because the slow Windows runner ran *everything*, including expensive integration/training/numeric tests
that also ran redundantly on the other three legs.

A measurement push (`--durations=0`) gave the ground-truth Windows-leg per-test cost — necessary because
**local profiling is not a faithful proxy** for CI (the `[test]` extra omits `pyright`, so
`test_pyright_clean_tracking_namespace` skips on CI; and Windows runs ~2–3× slower than local, e.g. the
xCross train-smoke is 57.6 s on the Windows leg vs ~24 s locally). The top cost is concentrated in
train-script smokes, ghost-GK KDE parity, and calibration cache-equivalence tests.

The forcing tension: cut the Windows long pole **without** discarding the cross-platform / cross-version
coverage that heavy *numeric* tests most need (repo history shows HGBR binning and numpy-hash behaviour
differing across OS / interpreter versions).

## Decision

Mark the **platform- and interpreter-INVARIANT** heavy tests `@pytest.mark.slow` and run them — plus the
`--benchmark-only` step — **once on a primary leg (`ubuntu-latest` 3.12)**; every other leg runs the full
fast/contract suite with `-m "not e2e and not slow"`. The primary leg is identified by a single matrix
`include` flag (`primary: true`, merged into the existing ubuntu-3.12 combination); steps key off
`${{ matrix.primary }}` / `${{ !matrix.primary }}` so the 4 legs partition into exactly one bulk pytest
process each. The `slow` set is chosen from real CI (Windows-leg) durations, classified per-test by
invariance.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep the full suite on all 4 legs (status quo) | simplest; max redundancy | ~16–20 min Windows long pole; the recurring "20-min CI" complaint | the problem being solved |
| B. `pytest-xdist -n auto` | parallel speedup locally | OOM-killed the 4-core/7-GB CI runners (py3.12 pass→kill); 4× heavy imports + 4× numba compiles | reverted before — documented in `ci.yml`; not re-attempted |
| C. Blanket "slow on both 3.12 legs" | keeps OS axis for slow tests | keeps the invariant train-smokes on Windows (no long-pole win) **and** drops version-sensitive tests from 3.10/3.11 (interpreter-axis loss) | dominated by per-test invariance routing |
| D. (chosen) per-test invariance routing to one primary leg | removes invariant heavy set from Windows (the win); version-sensitive tests keep full OS + interpreter coverage | requires classifying each heavy test; a marker-decay risk | — |
| E. Hand-copied `os==… && py==…` predicate per step | no matrix change | drift → a leg matching neither bulk step runs **zero** bulk tests yet goes green (silent-skip) | replaced by the single `matrix.primary` flag + a structural tripwire |

## Consequences

### Positive
- The Windows long pole drops by the invariant heavy set (~6.5 min of bulk on the measured run) without
  deleting or permanently skipping any test — `slow` tests still run once per CI run.
- The primary leg runs everything in **one** bulk process (mutually-exclusive `if:`s), so it does not
  become the new long pole via redundant cold imports.

### Negative
- The invariant heavy tests lose Windows / 3.10 / 3.11 coverage. Accepted **only** for tests proven
  platform/interpreter-invariant (does-it-run smokes, same-run internal-consistency/parity); the property
  holds identically per platform, so running once loses nothing real.
- A marker-decay risk: a future heavy test added without `@slow` runs on all 4 legs unnoticed. Mitigated
  by a CLAUDE.md rule; a full auto-marker-lint is deferred.

### Neutral
- **Version-sensitive heavy tests** (golden-hash / snapshot / absolute-numeric) are deliberately **not**
  marked `slow` — they stay on **all** legs (the genuinely golden tests, e.g. `test_golden_*`, are cheap
  and unaffected). Cheap behavioral-contract guards (dup-`action_id` ADR-020, id-dtype-invariance
  ADR-019, provenance/orientation/roster) also stay on all legs even when moderately slow — they are the
  cross-version/platform regressions we most want caught.
- The partition is guarded structurally by `tests/test_ci_slow_gating_wired.py` (parses `ci.yml`: exactly
  one `primary: true`, both bulk branches present, the `not slow` gating effect, non-empty `slow` set).
- `pyyaml` promoted to a direct `[test]` dep (the tripwire's parser) rather than riding the transitive
  `huggingface_hub → pyyaml` edge.

## Related
- **Specs:** `docs/superpowers/specs/2026-06-08-ci-slow-test-gating-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-08-ci-slow-test-gating.md`
- **Files:** `.github/workflows/ci.yml`, `tests/test_ci_slow_gating_wired.py`, `pyproject.toml` (`[test]`).
- **ADRs:** contract guards kept on all legs relate to ADR-019 (id-dtype) and ADR-020 (frame-aware xfns).
- **No version bump / no PyPI publish** — CI + test-infra only; `tests/` and `.github/` are not in the
  wheel/sdist, and `publish.yml` fires only on `v*` tags.
