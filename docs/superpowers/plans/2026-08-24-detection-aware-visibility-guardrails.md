# Detection-aware provider visibility guardrails — implementation plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use `- [ ]`.
> Rev 2 incorporates the first spec/plan review (B1–B3, S1–S3, M1–M4, the incident-reconciliation
> question). Spec: `docs/superpowers/specs/2026-08-24-detection-aware-visibility-guardrails-design.md`.

**Goal:** A detection-aware provider (SkillCorner) with all-null `visibility` fails **loud** at BOTH
the build seam (`materialize_tc3_frames`) and the consume seam (`train_ghost_gk` pre-flight), *before*
any corpus pass — never per-game-3-in via the `for_each` abort. Covers systematic AND mixed corpora.

**Architecture:** One shared rule `assert_detection_aware_visibility`, reused by `keeper_detection_mask`
(refactor), a module-level materializer seam `_guard_provider_frames` (Layer 1), and a
`train_ghost_gk` pre-flight `validate_corpus_visibility` using parquet `null_count` metadata (Layer 2).

**Tech stack:** pandas/numpy/pyarrow; pytest TDD; the `_DETECTION_AWARE_PROVIDERS` / `validate_provider`
single-source idiom; the `_load` + fit-spy trainer-entrypoint test idiom
(`tests/scripts/test_trainer_cache_and_providers.py`).

## Global Constraints

- `keeper_detection_mask`: RAISE (type + trigger, incl. empty→raise) and MASK output preserved; the
  **message is unified** (provider-generic) across the three call sites — NOT byte-equivalent, and the
  plan says so. No `len()` guard (empty detection-aware series still raises, matching the original).
- Fail-loud (raise) with the `tracking.skillcorner` remedy; never silent-exclude.
- Fully-observed providers unaffected; single source for the "detection-aware + all-null → raise" rule.
- Layer 2 resolves provider from `source_provider` (ONE walk, captured as `provider_by_path`) — never a
  `{provider}/` path segment (the generation dir is flat). Mechanism: parquet `null_count` metadata
  over EVERY detection-aware shard (catches mixed). Verified: `visibility` nulls are real parquet nulls
  (object/bool), not float-NaN.
- No model change, no retrain trigger. ADR/PR-S/version numbers not written until commit time.
- Verification: run `.venv` (pandas 2) + `.venv312` (pandas 3) interpreters **directly** (not `uv run`);
  lint/type at CI scope; capture ALL `FAILED` lines (no `tail`).

## Module placement (review M3 — DECIDED: B; review-2 MEDIUM — TRUE clean break via module alias)

The taxonomy + rule move to a new neutral **private** module `tracking/_provider_visibility.py`
(`_DETECTION_AWARE_PROVIDERS`, `_FULLY_OBSERVED_PROVIDERS`, `validate_provider`,
`assert_detection_aware_visibility`). `keeper_detection_mask` STAYS in `_ghost_gk.py` and delegates.

**The clean break must be via a module alias, NOT bare imports.** `keeper_detection_mask` consumes
`validate_provider` + `_FULLY_OBSERVED_PROVIDERS`, so a bare `from ._provider_visibility import
validate_provider, _FULLY_OBSERVED_PROVIDERS` in `_ghost_gk` would re-export both as `_ghost_gk`
attributes (a transitive re-export — the shim we claim to avoid), while `_DETECTION_AWARE_PROVIDERS`
would break: an asymmetric half-clean break. So `_ghost_gk` does `from . import _provider_visibility
as _pv` and references `_pv.validate_provider` / `_pv._FULLY_OBSERVED_PROVIDERS` /
`_pv.assert_detection_aware_visibility` — no moved name enters `_ghost_gk`'s namespace, so
`_ghost_gk.validate_provider` genuinely fails (pinned by a negative test, Task 1). Migrate the in-repo
sites: `train_ghost_gk.py:141`, `test_validate_provider_is_shared_not_duplicated`
(`test_trainer_cache_and_providers.py:122`), and the new materializer importer. Private not public/
top-level (YAGNI; every consumer is tracking-adjacent). Rationale + blast-radius verification in the
spec's "Module placement" block.

---

## File structure

- Create: `silly_kicks/tracking/_provider_visibility.py` (neutral home for the taxonomy + shared rule).
- Modify: `silly_kicks/tracking/_ghost_gk.py` (DELETE the moved symbols; `from . import _provider_visibility as _pv`; `keeper_detection_mask` references `_pv.*` and delegates — no bare re-export).
- Modify: `scripts/materialize_tc3_frames.py` (module-level `_guard_provider_frames`, called from `_work`; imports from the neutral module).
- Modify: `scripts/train_ghost_gk.py` (`:141` `validate_provider` import → `_provider_visibility`; `provider_by_path` capture + `validate_corpus_visibility`).
- Create: `tests/tracking/test_detection_aware_visibility.py` (shared rule + `keeper_detection_mask` + Layer-1 seam + clean-break negative test).
- Modify: `tests/scripts/test_trainer_cache_and_providers.py` (Layer-2 fires-before-fit test; MIGRATE `test_validate_provider_is_shared_not_duplicated:122` import → `_provider_visibility`).
- Create: `docs/superpowers/adrs/ADR-069-detection-aware-visibility-guardrails.md` (number at commit).
- Modify: `CHANGELOG.md` (`[Unreleased]`), `CLAUDE.md`, `docs/PRIVATE_CONSUMERS.md` (one-liner in the **in-repo first-party** table for `tracking/_provider_visibility.py`; the `_ghost_gk.py` drift-guard path pin is unaffected).
- **No change:** `pyproject.toml` — `packages = ["silly_kicks"]` (`:154-155`) auto-includes the new module; the only `exclude`s are the `full/` weight dirs (`:160`).

---

### Task 1: Shared rule + `keeper_detection_mask` refactor

- [ ] **Step 1: Failing test** (`tests/tracking/test_detection_aware_visibility.py`):

```python
import numpy as np, pandas as pd, pytest
from silly_kicks.tracking._provider_visibility import assert_detection_aware_visibility
from silly_kicks.tracking._ghost_gk import keeper_detection_mask

def test_all_null_detection_aware_raises_with_remedy():
    with pytest.raises(ValueError, match="tracking.skillcorner"):
        assert_detection_aware_visibility(pd.Series([None, None], dtype="object"), provider="skillcorner")

def test_empty_detection_aware_raises():  # preserves original keeper_detection_mask semantics (B2)
    with pytest.raises(ValueError):
        assert_detection_aware_visibility(pd.Series([], dtype="object"), provider="skillcorner")

def test_non_null_detection_aware_ok():
    assert assert_detection_aware_visibility(pd.Series([True, None, True]), provider="skillcorner") is None

def test_fully_observed_all_null_noop():
    assert assert_detection_aware_visibility(pd.Series([None, None]), provider="gradientsports") is None

def test_keeper_detection_mask_mask_output_unchanged():  # MASK preserved (B2)
    out = keeper_detection_mask(pd.Series([True, None, True]), provider="skillcorner")
    np.testing.assert_array_equal(out, np.array([True, False, True]))

def test_keeper_detection_mask_still_raises_on_all_null():
    with pytest.raises(ValueError, match="tracking.skillcorner"):
        keeper_detection_mask(pd.Series([None, None], dtype="object"), provider="skillcorner")

def test_moved_symbols_not_reexported_from_ghost_gk():  # TRUE clean break (review-2 MEDIUM)
    import silly_kicks.tracking._ghost_gk as gg
    for name in ("validate_provider", "assert_detection_aware_visibility",
                 "_DETECTION_AWARE_PROVIDERS", "_FULLY_OBSERVED_PROVIDERS"):
        assert not hasattr(gg, name), f"{name} must live only in _provider_visibility, not re-export via _ghost_gk"
```

- [ ] **Step 2: Run → FAIL** (`assert_detection_aware_visibility` undefined; `test_moved_symbols_not_reexported_from_ghost_gk` also fails — the names still live in `_ghost_gk`).
- [ ] **Step 3: Implement.** Create `silly_kicks/tracking/_provider_visibility.py` and MOVE
  `_DETECTION_AWARE_PROVIDERS`, `_FULLY_OBSERVED_PROVIDERS`, `validate_provider` there (DELETE from
  `_ghost_gk.py`); add `assert_detection_aware_visibility` with a provider-generic message naming
  `tracking.skillcorner`, NO `len()` guard. In `_ghost_gk.py` use a **module alias** —
  `from . import _provider_visibility as _pv` (NOT `from ._provider_visibility import validate_provider,
  _FULLY_OBSERVED_PROVIDERS`, which would transitively re-export those names and fail the negative
  test). `keeper_detection_mask` STAYS in `_ghost_gk.py`, calls `_pv.validate_provider(...)`, keeps the
  `_pv._FULLY_OBSERVED_PROVIDERS` all-True return, delegates its all-null branch to
  `_pv.assert_detection_aware_visibility(...)`, and keeps the `fillna(False).astype(bool)` mask exactly.
- [ ] **Step 4: Run → PASS** (incl. `test_moved_symbols_not_reexported_from_ghost_gk`).
- [ ] **Step 5: Migrate the two existing consumers.** `scripts/train_ghost_gk.py:141`
  `from ..._ghost_gk import validate_provider` → `from ..._provider_visibility import validate_provider`;
  `tests/scripts/test_trainer_cache_and_providers.py:122` `test_validate_provider_is_shared_not_duplicated`
  → import `validate_provider` from `_provider_visibility` (so the "single source" test pins the new
  home). Leave `test_keeper_detection_mask_still_rejects_an_unknown_provider:135` unchanged.
- [ ] **Step 6: Run the migrated file** (`tests/scripts/test_trainer_cache_and_providers.py`) → PASS.

### Task 2: Layer 1 — materializer build-time guard (testable seam)

- [ ] **Step 1: Failing test** (`tests/tracking/test_detection_aware_visibility.py`): import
  `_guard_provider_frames` from `scripts.materialize_tc3_frames`; a detection-aware frame with all-null
  `visibility` → raises; with the `visibility` column **absent** → raises (M2); native (non-null) →
  passes; fully-observed all-null → passes. Plus a wiring assertion (AST-parse `materialize_tc3_frames`,
  assert `_work` contains a call to `_guard_provider_frames`).
- [ ] **Step 2: Run → FAIL** (not module-level / not wired).
- [ ] **Step 3: Implement** `_guard_provider_frames(frames, provider)` at MODULE level (fixes B1): for a
  detection-aware provider, raise if `"visibility" not in frames.columns` (M2), else call the shared
  rule. Call it from `_work` before `return frames` (`:255`).
- [ ] **Step 4: Run → PASS.**

### Task 3: Layer 2 — consume-time pre-flight (fires before fit)

- [ ] **Step 1: Failing test** (`tests/scripts/test_trainer_cache_and_providers.py`, beside
  `test_unclassified_provider_fails_BEFORE_any_fitting`): build a flat data-dir with a SkillCorner shard
  written all-null `visibility` (+ `source_provider="skillcorner"`); monkeypatch `GhostGkModel.fit`
  (and the extractor) to a call-count spy; drive the trainer entrypoint via `_load`; assert it RAISES
  (`match="tracking.skillcorner"`) with `spy calls == 0` (B3 — proves it fires BEFORE the corpus pass).
  Plus: a native / fully-observed-only data-dir passes the pre-flight.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** In the discovery loop (`train_ghost_gk.py:337-346`) capture
  `provider_by_path: dict[Path,str]` from `source_provider` (S1). Add `validate_corpus_visibility(provider_by_path)`:
  for each path whose provider ∈ `_DETECTION_AWARE_PROVIDERS`, open the parquet, sum the per-row-group
  `visibility` `null_count` from metadata; raise (shared message) if `null_count == num_rows` or if the
  `visibility` column is absent (M2). Call it in the pre-flight after `validate_corpus_providers`, before
  the corpus pass.
- [ ] **Step 4: Run → PASS.**

### Task 4: ADR + docs + full verification

- [ ] **Step 1:** ADR (contract + two layers; cross-ref ADR-038 + spec 4.3; record the M3=B module
  decision — taxonomy + rule in the neutral `tracking/_provider_visibility.py`, clean break). `ADR-069`
  until commit.
- [ ] **Step 2:** `CHANGELOG [Unreleased]` (additive guardrail; no retrain) + CLAUDE.md durable bullet
  (the visibility-usability contract + its single source in `_provider_visibility.py`) +
  `PRIVATE_CONSUMERS.md` entry recording the new `tracking/_provider_visibility.py` module (note the
  `_ghost_gk.py` drift-guard path pin is unaffected).
- [ ] **Step 3:** Full net — `ruff check`/`ruff format --check` (CI scope), whole-branch `pyright`,
  full suite on `.venv` and `.venv312` (interpreters directly). Confirm green; capture ALL `FAILED`.
