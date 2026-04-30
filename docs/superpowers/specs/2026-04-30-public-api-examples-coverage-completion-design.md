# Public-API Examples coverage completion + atomic `coverage_metrics` (PR-S14)

**Status:** Approved (design)
**Target release:** silly-kicks 2.2.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.1.1 (`docs/superpowers/specs/2026-04-30-docstring-examples-design.md`)

---

## 1. Problem

PR-S13 (silly-kicks 2.1.1) closed TODO D-8 by adding `Examples` sections to ~50 public surfaces and shipped a CI guardrail (`tests/test_public_api_examples.py`). The guardrail's `_PUBLIC_MODULE_FILES` covers 14 module files. **Five public-API module files were missed**, with 25 public functions across them lacking Examples:

| File | Public surfaces |
|---|---|
| `silly_kicks/vaep/labels.py` | `scores`, `concedes`, `goal_from_shot`, `save_from_shot`, `claim_from_cross` (5) |
| `silly_kicks/vaep/formula.py` | `offensive_value`, `defensive_value`, `value` (3) |
| `silly_kicks/atomic/vaep/features.py` | `actiontype`, `feature_column_names`, `play_left_to_right`, `actiontype_onehot`, `location`, `polar`, `movement_polar`, `direction`, `goalscore` (9) |
| `silly_kicks/atomic/vaep/labels.py` | `scores`, `concedes`, `goal_from_shot`, `save_from_shot`, `claim_from_cross` (5) |
| `silly_kicks/atomic/vaep/formula.py` | `offensive_value`, `defensive_value`, `value` (3) |

These are real public functions consumers can call — `formula.value()` is the pure VAEP-value computation (callable directly, not just via `VAEP.rate()`); `labels.scores`/`concedes` are the default label functions consumers can override; `atomic/vaep/features.py` mirrors the standard feature-extractor module. Without Examples, the gold-standard documentation claim from PR-S13 is incomplete.

Separately, **TODO C-1 tracks a parity gap from silly-kicks 1.10.0**: the standard `silly_kicks.spadl.coverage_metrics` utility (added in 1.10.0) has no atomic counterpart. The note says "defer until a concrete consumer ask," but bundling with PR-S14 closes a 1.10.0 follow-up at marginal cost. Atomic-SPADL has its own action-type vocabulary (33 types vs standard's 23), so a direct copy isn't appropriate — but the API shape is identical.

## 2. Goals

1. **Close the PR-S13 coverage gap.** Add Examples to the 25 missing surfaces. Extend `_PUBLIC_MODULE_FILES` from 14 to 19 entries. CI gate now mechanically enforces Examples coverage across the entire public API.
2. **Add `silly_kicks.atomic.spadl.coverage_metrics` (closes C-1).** Mirror `silly_kicks.spadl.coverage_metrics` API shape; validate against atomic action-type vocabulary; re-export from `silly_kicks.atomic.spadl.__init__`. Patch precedent — same-shape API as the standard counterpart.
3. **Minor release 2.2.0.** Per pragmatic semver: PR-S14's docstring additions are patch-worthy on their own (2.1.2), but bundling C-1 (a new public utility) crosses into minor territory.

## 3. Non-goals

1. **No doctest verification.** Style matches PR-S13's illustrative pattern.
2. **No examples for private (underscore-prefixed) symbols.**
3. **No examples for `learners.py`** — every top-level function in `silly_kicks/vaep/learners.py` is underscore-prefixed (verified). Module is fully private.
4. **No new tests beyond the gate-extension** + the standard atomic `coverage_metrics` tests (mirror the existing `tests/spadl/test_coverage_metrics.py` shape on the atomic side).
5. **No CLAUDE.md amendment, no ADR.** Same precedent as PR-S13 — CI gate is self-documenting.

## 4. Architecture

### 4.1 File structure

| Path | Action | Surfaces |
|---|---|---|
| `tests/test_public_api_examples.py` | Modify | Extend `_PUBLIC_MODULE_FILES` from 14 → 19 entries (no other changes) |
| `silly_kicks/vaep/labels.py` | Modify | 5 functions get Examples |
| `silly_kicks/vaep/formula.py` | Modify | 3 functions get Examples |
| `silly_kicks/atomic/vaep/features.py` | Modify | 9 functions get Examples |
| `silly_kicks/atomic/vaep/labels.py` | Modify | 5 functions get Examples |
| `silly_kicks/atomic/vaep/formula.py` | Modify | 3 functions get Examples |
| `silly_kicks/atomic/spadl/utils.py` | Modify | New `coverage_metrics` function (reuses `silly_kicks.spadl.utils.CoverageMetrics` — single source of truth) |
| `silly_kicks/atomic/spadl/__init__.py` | Modify | Re-export `coverage_metrics`; also re-export the standard `CoverageMetrics` for atomic-side discoverability |
| `tests/atomic/test_atomic_coverage_metrics.py` | Create | Mirror tests/spadl/test_coverage_metrics.py (~10 unit tests) |
| `pyproject.toml` | Modify | Version 2.1.1 → 2.2.0 |
| `CHANGELOG.md` | Modify | New `[2.2.0]` entry |
| `TODO.md` | Modify | Close C-1 entry (move out of "Tech Debt") |

Total: 1 new test file + 8 modified source files + 3 admin files. ~250 lines added.

### 4.2 Style

Identical to PR-S13. Illustrative, 3-7 lines, context-free. Variable-naming conventions:
- `actions` — SPADL DataFrame
- `atomic` — atomic-SPADL DataFrame
- `states` — gamestates output
- `feats` — feature DataFrame
- `p_scores`, `p_concedes` — probability Series from VAEP._estimate_probabilities (used in formula examples)
- `actions_with_names` — actions DataFrame after `add_names()` (formula functions consume this)

### 4.3 Per-group templates

**Group A — `vaep/labels.py` and `atomic/vaep/labels.py` (5+5 = 10 fns).**

Label functions take `actions: pd.DataFrame` and return `pd.DataFrame` with binary label columns. Template:

```python
Examples
--------
Compute scoring labels for VAEP training::

    from silly_kicks.vaep.labels import scores

    y_scores = scores(actions, nr_actions=10)
    # y_scores has one column 'scores' with True/False per action.
```

Same template applies to `concedes`. The `goal_from_shot` / `save_from_shot` / `claim_from_cross` functions are alternative label generators with similar shape.

**Group B — `vaep/formula.py` and `atomic/vaep/formula.py` (3+3 = 6 fns).**

Formula functions take action DataFrame + probability Series. Template:

```python
Examples
--------
Compute the offensive value component of VAEP::

    from silly_kicks.vaep.formula import offensive_value

    ov = offensive_value(actions_with_names, p_scores, p_concedes)
    # Returns a Series of offensive values, one per action.
```

Same shape for `defensive_value` and `value` (the latter combines both).

**Group C — `atomic/vaep/features.py` (9 fns).**

Atomic feature transformers parallel `silly_kicks/vaep/features.py` helpers. Same fixture motif (`states = gamestates(atomic, nb_prev_actions=3); feats = X(states)`). Atomic-specific functions (`location`, `polar`, `movement_polar`, `direction`) replace the standard `startlocation` / `startpolar` / `endlocation` / `endpolar` since atomic has single (x, y) per action plus (dx, dy) displacement.

```python
Examples
--------
Extract atomic location features per gamestate slot::

    from silly_kicks.atomic.vaep.features import location

    feats = location(states)
```

### 4.4 Atomic `coverage_metrics` design (closes C-1)

`silly_kicks.atomic.spadl.coverage_metrics(*, actions, expected_action_types) -> CoverageMetrics`. Direct mirror of `silly_kicks.spadl.coverage_metrics`:

- **Reuses the standard TypedDict.** `silly_kicks.spadl.utils.CoverageMetrics` is the single source of truth (shape `{"counts": dict[str, int], "missing": list[str], "total_actions": int}`). `silly_kicks/atomic/spadl/utils.py` imports it for its return annotation; `silly_kicks/atomic/spadl/__init__.py` re-exports it for atomic-side discoverability. The gate's `_SKIP_SYMBOLS` already excludes `CoverageMetrics` (TypedDict — fields are the documentation), so the re-export adds no Examples-section work. Atomic already cross-imports from standard (`atomic/spadl/config.py` does this), so the dependency direction is established precedent.
- **Atomic action-type vocabulary** for resolution: `silly_kicks.atomic.spadl.config.actiontypes` (33 types vs standard's 23). Includes atomic-only types and standard types that collapse in atomic (`corner_short`/`corner_crossed` → `corner`; `freekick_*` → `freekick`). Authoritative list at fit time is whatever `atomic.spadl.config.actiontypes` enumerates — no hard-coded subset in the implementation.
- **Same validation logic:** `expected_action_types` validated against the atomic vocabulary; unknown names raise `ValueError`.
- **Re-export:** add `coverage_metrics` (function) and the imported `CoverageMetrics` (type) to `silly_kicks/atomic/spadl/__init__.py::__all__` in alphabetical position, mirroring the standard `silly_kicks/spadl/__init__.py` pattern.
- **Tests** (`tests/atomic/test_atomic_coverage_metrics.py`): mirror the standard `tests/spadl/test_coverage_metrics.py` shape — three test classes (`TestAtomicCoverageMetricsContract`, `TestAtomicCoverageMetricsCorrectness`, `TestAtomicCoverageMetricsDegenerate`); ~10 tests. Atomic-only assertions cover atomic-vocabulary acceptance (e.g., `receival`, post-collapse `corner` / `freekick`) and rejection of standard-only names absent from atomic (e.g., `corner_short`, `freekick_short`).

### 4.5 Updated `_PUBLIC_MODULE_FILES` in the gate test

Add 5 entries (the 5 files from §1):

```python
_PUBLIC_MODULE_FILES = (
    # ... existing 14 entries ...
    "silly_kicks/vaep/labels.py",
    "silly_kicks/vaep/formula.py",
    "silly_kicks/atomic/vaep/features.py",
    "silly_kicks/atomic/vaep/labels.py",
    "silly_kicks/atomic/vaep/formula.py",
)
```

**Verified `silly_kicks/vaep/learners.py` not added** — every top-level function is underscore-prefixed (private).

## 5. Implementation order (TDD-first, mirrors PR-S13)

1. **T1 — Extend the failing CI gate.** Add 5 new entries to `_PUBLIC_MODULE_FILES`. Run the gate locally; expect 5 of 19 cases failing with 25 missing surfaces enumerated.
2. **T2 — Group A: vaep/labels.py + atomic/vaep/labels.py** (10 examples).
3. **T3 — Group B: vaep/formula.py + atomic/vaep/formula.py** (6 examples).
4. **T4 — Group C: atomic/vaep/features.py** (9 examples).
5. **T5 — C-1: atomic `coverage_metrics`.** Write failing tests for `tests/atomic/test_atomic_coverage_metrics.py`, then implement, then re-export, then verify.
6. **T6 — Verification gates** (ruff/format/pyright/pytest).
7. **T7 — /final-review.**
8. **T8 — CHANGELOG, TODO close C-1, version bump 2.1.1 → 2.2.0.**
9. **T9 — Single commit gate (user approval).**
10. **T10 — Push + PR + merge + tag.**

## 6. Verification gates (before commit)

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113

uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/

# Spot-check a representative new example per group
uv run python -c "from silly_kicks.vaep.formula import value; help(value)" | head -30

# Full pytest suite
uv run pytest tests/ --tb=short
```

Expected after this PR: **803 tests passing** (788 baseline + 5 new gate cases + 10 atomic coverage_metrics tests + ~0 other; 4 skipped unchanged). Zero ruff errors. Zero pyright errors.

## 7. Commit cycle

Same as PR-S13. Single commit. Branch: `feat/public-api-examples-coverage-completion`. Version bump 2.1.1 → 2.2.0 (minor — new public API via atomic.coverage_metrics).

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Examples for `formula.value` / `offensive_value` etc. require realistic-looking probability inputs | Use abstract Series motif (`p_scores`, `p_concedes`) — readers infer source. Same approach as PR-S13's `gamestates`-based examples. |
| Atomic `coverage_metrics` accidentally diverges from standard API shape | Mirror tests verify same return shape + same error messages. |
| Atomic vocabulary differs subtly from standard (e.g., `keeper_save_take` only exists in atomic) | Test cases for atomic-only types lock the contract. |
| `atomic/vaep/features.py` parallel helpers have nuanced differences from standard | Examples are illustrative — describe atomic-specific behavior in 1-line comments where relevant (e.g., `location` works on `(x, y)` not `(start_x, start_y)`). |
| 5 new gate cases might surface other style inconsistencies | Spot-check `help(...)` rendering on one example per group during T6. |

## 9. Acceptance criteria

1. `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` has 19 entries (was 14).
2. All 25 previously-missing public surfaces have `Examples` sections.
3. `silly_kicks.atomic.spadl.coverage_metrics` is importable from `silly_kicks.atomic.spadl` (re-exported), returns a `CoverageMetrics` TypedDict, validates against atomic vocabulary, and is covered by ~10 unit tests in `tests/atomic/test_atomic_coverage_metrics.py`.
4. `TODO.md` C-1 entry closed.
5. `CHANGELOG.md` has a `[2.2.0]` entry: Added (atomic.coverage_metrics + Examples completion) + Changed (gate now covers 19 files instead of 14).
6. `pyproject.toml` version is `2.2.0`.
7. Verification gates (ruff, pyright with exact pins, full pytest suite) all pass before commit. Test count: ~803 passing, 4 skipped (was 788 + 4).
8. `/final-review` clean.
9. Tag `v2.2.0` pushed → PyPI auto-publish workflow fires successfully.
10. Style consistent with PR-S13 — no rewriting of pre-existing examples in scope.
