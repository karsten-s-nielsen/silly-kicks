# CI-time reduction via slow-test gating — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the ~16–20 min Windows CI long-pole by running the expensive **platform-invariant** tests (train-script smokes, same-run numba parity) + the benchmark step **once on a primary leg (ubuntu-3.12)**, while keeping the full fast/contract suite and all **version-sensitive** tests on every leg.

**Architecture:** A matrix `include` flag (`primary: true`, merged into the existing ubuntu-3.12 leg) is the single source of truth; bulk steps key off `${{ matrix.primary }}` / `${{ !matrix.primary }}` so the 4 legs partition into exactly one bulk pytest process each. The `slow` set is chosen from **real CI durations** (measure-first: local ≠ CI), classified per-test by invariance. A structured-YAML meta-test guards the partition; an ADR records the why. No xdist. No version bump (CI/test-infra only — nothing ships).

**Tech Stack:** GitHub Actions matrix, pytest markers, PyYAML (tripwire). Spec: `docs/superpowers/specs/2026-06-08-ci-slow-test-gating-design.md`.

---

## File Structure

- **Modify** `.github/workflows/ci.yml` — Phase 0: `--durations=0` on the bulk step; Phase 1: matrix `include` flag + mutually-exclusive bulk steps + benchmark `if:` guard + permanent `--durations=25`.
- **Modify** test files (TBD-from-Phase-0) — add `@pytest.mark.slow` to the classified-invariant heavy tests.
- **Create** `tests/test_ci_slow_gating_wired.py` — structured-YAML tripwire (bulk-step partition + `-m` gating effect + non-empty slow set), red-first.
- **Modify** `pyproject.toml` (`[test]` extra) + `uv.lock` — declare `pyyaml` directly (the tripwire's parser; not a transitive ride).
- **Create** `docs/superpowers/adrs/ADR-0NN-ci-slow-test-gating.md` — records the invariance-routing CI policy (number confirmed free at implementation — expected `023`).
- **Modify** `CLAUDE.md` — one-line `slow`-marker criterion rule.

**Commit / push policy (gating-critical):** CI only runs via a PR against `main` (`ci.yml` triggers on `push: [main]` / `pull_request: [main]`), so a branch push alone does **not** run CI — a **draft PR** is opened in Phase 0 to get real timings, iterated, then readied. Every commit / push / PR action is **gated on explicit user approval** (present diff + HOLD; no sentinel without a per-commit "yes"). Intra-branch commits squash to one commit at merge. **No version bump, no tag.** Use `.venv/Scripts/python -m pytest` and `.venv/Scripts/ruff`/`pyright` locally.

---

### Task 1: Phase 0 — CI measurement (the diagnostic)

**Files:**
- Modify: `.github/workflows/ci.yml:63`

- [ ] **Step 1: Add full-distribution durations to the bulk step**

Change line 63 from:
```yaml
      - run: pytest tests/ -m "not e2e" --benchmark-skip --tb=short
```
to:
```yaml
      - run: pytest tests/ -m "not e2e" --benchmark-skip --tb=short --durations=0
```
(`--durations=0` = full distribution for the one-time analysis; Task 5 settles it to `25`.)

- [ ] **Step 2: Local sanity — ci.yml still parses**

Run: `.venv/Scripts/python -c "import yaml,pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text(encoding='utf-8')); print('ci.yml OK')"`
Expected: `ci.yml OK`.

- [ ] **Step 3: Present diff + HOLD for approval to commit + push + open draft PR**

Present `git diff .github/workflows/ci.yml` and the proposed commit (`ci: add --durations to surface CI test timings (measurement for slow-test gating)`). On approval: commit (via `git commit -F`), push `ci/slow-test-gating`, and `gh pr create --draft --base main --title "ci: slow-test gating (4-leg → primary-leg heavy tests)" --body-file <file>`. The draft PR triggers CI.

- [ ] **Step 4: Wait for the PR's CI run, then read the windows-3.12 durations (scoped to that job)**

Run: `gh run list --branch ci/slow-test-gating --limit 1 --json databaseId --jq '.[0].databaseId'` → `$RUN`.
Wait for completion (background `gh run watch $RUN`). **Resolve the windows-3.12 job id and scope the log to it** — `gh run view $RUN --log` dumps all 4 legs interleaved, so an unscoped grep blends leg timings (the exact mistake the "measure on the real Windows leg" rule exists to avoid):
```bash
JOB=$(gh run view $RUN --json jobs --jq '.jobs[] | select(.name|test("windows.*3.12")) | .databaseId')
gh run view $RUN --log --job "$JOB" 2>&1 | grep -iE '[0-9]+\.[0-9]+s +(call|setup|teardown)' | sort -rn | head -60
```
Record, **per test, the summed `setup + call + teardown`** (fixture-borne ghost-GK/xT fits show a cheap `call` but expensive `setup`).

- [ ] **Step 5: Checkpoint** — Phase-0 timings captured. No further commit yet.

---

### Task 2: Classify candidates (analysis — no code change)

**Files:** none (produces the `slow` list used by Task 4).

- [ ] **Step 1: Apply the three-clause criterion to each test ≥ ~5 s (summed) on the windows-3.12 log**

For each heavy test, decide:
- **→ `slow` (gate to primary leg)** iff it is (a) expensive AND (b) integration / training-subprocess / heavy-numeric AND (c) **platform- & interpreter-INVARIANT**: a "does-it-run" train-script smoke, or a same-run internal-consistency check (e.g. numba-`@njit`-vs-numpy parity computed in one process). The property holds identically per platform.
- **→ leave UNMARKED (stays on all legs)** iff it is a **version-sensitive** heavy test — asserts an absolute / golden-hash / numeric-snapshot value (HGBR-binning / numpy-hash class). These keep full OS + interpreter coverage.
- **→ leave UNMARKED** iff it is a cheap behavioral-contract guard (parity/golden, dup-`action_id`, id-dtype-invariance ADR-019, orientation/roster) — even if moderately slow.
- **→ leave UNMARKED — it is not a CI cost at all** iff the Phase-0 windows log shows it at **~0 s (skipped)**: a test needing an extra the CI job doesn't install (`[kloppy,xgboost,test]` — e.g. a `[train]`-only path) skips on CI and fails clause (a). Decide off the **CI** numbers, never the local profile, so a CI-skipped test isn't marked `slow` for a cost that doesn't exist on CI. (Note: `[test]` DOES include `ruthless-efficiency[optuna]` + `xgboost`, so the calibration objective tests run on CI and are genuine candidates; only the pyright-as-a-test skips.)

- [ ] **Step 2: Starting hypothesis from the local profile (confirm/adjust against the CI log)**

Likely **`slow`** (invariant integration/smoke): `tests/tracking/test_xcross_attempt_integration.py::test_train_script_smoke` + `::test_train_script_fail_closed_writes_no_artifact`; `tests/tracking/test_xshot_occurrence_integration.py::test_train_script_smoke` + `::test_train_script_fail_closed_writes_no_artifact`; `tests/tracking/test_train_ghost_gk_cli.py::*`; `tests/tracking/test_ghost_gk_integration.py::TestTrainScriptSmoke::test_smoke`. Inspect each ghost-GK KDE test (`tests/tracking/test_ghost_gk_kde_vectorized.py::test_model_traveling_parity`, `::test_fft_cic_raw_grid_tighter_than_ngp`, `::test_numba_loop_matches_numpy_closed_form`, `::test_vectorized_chunking_invariant`): if it asserts **internal parity/consistency in one run** → `slow`; if it asserts an **absolute/golden value** → leave unmarked (version-sensitive). The 3 already-`slow` tests (`tests/vaep/test_hybrid_with_tracking.py` ×2, `tests/atomic/vaep/test_atomic_with_tracking.py`) keep `slow` iff they meet clause (c) (integration lifecycle — confirm no golden-value assertion).

- [ ] **Step 3: Write the final classified list into Task 4's marking step** (replace its placeholder list with the confirmed paths). Checkpoint.

---

### Task 3: Declare pyyaml + write the gating tripwire (red-first)

**Files:**
- Modify: `pyproject.toml` (`[test]` extra), `uv.lock`
- Create: `tests/test_ci_slow_gating_wired.py`

- [ ] **Step 1: Declare pyyaml directly in the `[test]` extra**

The tripwire `import yaml`; today `pyyaml` is only available transitively (`huggingface_hub → pyyaml`). A
permanent CI guard must not ride a transitive edge a future bump could drop (else it errors red with an
opaque `ModuleNotFoundError: yaml`). In `pyproject.toml`, append to the `test = [...]` extra (after
`"xgboost>=2.0,<4.0",`):
```toml
    # pyyaml backs the CI-gating tripwire (tests/test_ci_slow_gating_wired.py), which parses
    # .github/workflows/ci.yml. Declared directly so the guard never rides a transitive edge.
    "pyyaml>=6.0",
```
Then re-lock: `uv lock` (expect `uv.lock` updated; pyyaml already resolved transitively, so this just
promotes it to a direct dep — no version churn).

- [ ] **Step 2: Write the tripwire test**

```python
"""Structural guard: the CI matrix partitions into exactly one bulk pytest process per leg.

Prevents the cardinal silently-skipping sin -- if the primary-leg predicate drifts (or a bulk step
is dropped), a leg can run zero bulk tests yet go green, and the slow tests run nowhere. We assert
the SEMANTIC partition on the BULK steps -- which step activates per resolved leg and its -m gating
-- not mere string presence (three steps carry matrix.primary expressions; presence proves nothing).
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"


def _guard(expr: object) -> str:
    """Normalize a step `if:` to its inner expression: whitespace-stripped, ${{ }} unwrapped
    (GitHub accepts both `${{ matrix.primary }}` and brace-less `matrix.primary`)."""
    s = "".join(str(expr).split())
    if s.startswith("${{") and s.endswith("}}"):
        s = s[3:-2]
    return s


def test_ci_bulk_steps_partition_with_slow_gating() -> None:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    test_job = ci["jobs"]["test"]

    # exactly one primary leg, from a single source of truth (the matrix include flag)
    include = test_job["strategy"]["matrix"].get("include", [])
    primaries = [e for e in include if e.get("primary") is True]
    assert len(primaries) == 1, f"expected exactly one matrix include with primary: true, got {primaries}"

    # the two BULK (non-benchmark) pytest steps must partition the matrix on matrix.primary
    bulk = [
        s
        for s in test_job["steps"]
        if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]
    ]
    assert len(bulk) == 2, f"expected exactly two bulk steps, got {len(bulk)}: {[s.get('run') for s in bulk]}"
    guards = {_guard(s.get("if", "")): s for s in bulk}
    assert set(guards) == {"matrix.primary", "!matrix.primary"}, f"bulk steps not complementary: {set(guards)}"

    # the gating EFFECT: non-primary excludes slow; primary runs everything (incl slow)
    assert "not slow" in guards["!matrix.primary"]["run"], "non-primary bulk step must exclude slow"
    assert "not slow" not in guards["matrix.primary"]["run"], "primary bulk step must run slow (no 'not slow')"


def test_slow_marker_set_is_non_empty() -> None:
    hits = sum(
        1 for p in (_REPO / "tests").rglob("*.py") if "pytest.mark.slow" in p.read_text(encoding="utf-8")
    )
    assert hits >= 1, "no tests carry @pytest.mark.slow; the gating would be a no-op"
```

- [ ] **Step 3: Run it to confirm RED (partition not yet wired)**

Run: `.venv/Scripts/python -m pytest tests/test_ci_slow_gating_wired.py -v`
Expected: `test_ci_bulk_steps_partition_with_slow_gating` **FAILS** — current `ci.yml` has no `include`
(→ `len(primaries) == 0`) and a single bulk step (→ `len(bulk) == 1`). `test_slow_marker_set_is_non_empty`
PASSES (3 existing `@slow` tests). The red proves the guard discriminates on the real property.

- [ ] **Step 4: Checkpoint** — red confirmed. No commit yet (bundled).

---

### Task 4: Mark the classified-invariant tests `@pytest.mark.slow`

**Files:**
- Modify: each test file from Task 2's confirmed list (paths finalized in Task 2 Step 3).

- [ ] **Step 1: Add the marker to each confirmed-invariant test**

For a function-style test, add the decorator (import `pytest` is already present in these files; verify):
```python
@pytest.mark.slow
def test_train_script_smoke(...):
    ...
```
For a class-method test (`test_ghost_gk_integration.py::TestTrainScriptSmoke::test_smoke`), mark the method or the class:
```python
@pytest.mark.slow
class TestTrainScriptSmoke:
    ...
```
Apply to **only** the Task-2-confirmed invariant set. Do **not** mark version-sensitive (golden/absolute) tests.

- [ ] **Step 2: Verify the marks collect**

Run: `.venv/Scripts/python -m pytest tests/ -m "slow and not e2e" --collect-only -q 2>&1 | tail -5`
Expected: the collected count equals the number of tests you marked (+ the 3 pre-existing). Sanity-check the names.

- [ ] **Step 3: Verify the fast suite still collects everything else**

Run: `.venv/Scripts/python -m pytest tests/ -m "not e2e and not slow" --collect-only -q 2>&1 | tail -3`
Expected: a large count (full suite minus the slow set), no collection errors.

- [ ] **Step 4: Checkpoint** — marks applied. No commit yet.

---

### Task 5: CI YAML gating (matrix flag + mutually-exclusive steps)

**Files:**
- Modify: `.github/workflows/ci.yml` (matrix block + the `test`-job run steps)

- [ ] **Step 1: Add the `primary` flag to the matrix**

In the `test` job's `strategy.matrix`, add an `include` (after the existing `exclude`):
```yaml
        include:
          - { os: ubuntu-latest, python-version: "3.12", primary: true }
```
(GitHub merges this into the existing ubuntu-3.12 leg — it does not create a new leg.)

- [ ] **Step 2: Replace the two run steps (lines ~63 + ~66) with the gated form**

```yaml
      # Non-primary legs: full fast/contract suite, platform-invariant slow set excluded. ONE process.
      - if: ${{ !matrix.primary }}
        run: pytest tests/ -m "not e2e and not slow" --benchmark-skip --tb=short --durations=25
      # Primary leg (ubuntu 3.12): EVERYTHING incl. slow, in a single bulk process.
      - if: ${{ matrix.primary }}
        run: pytest tests/ -m "not e2e" --benchmark-skip --tb=short --durations=25
      # Primary leg only: benchmark measurements (non-asserting trend data).
      - if: ${{ matrix.primary }}
        run: pytest tests/ -m "not e2e" --benchmark-only --tb=short
```

- [ ] **Step 3: ci.yml parses + tripwire goes GREEN**

Run: `.venv/Scripts/python -m pytest tests/test_ci_slow_gating_wired.py -v`
Expected: both tests **PASS** (matrix now has exactly one `primary: true`; both `matrix.primary` branches present).

- [ ] **Step 4: Checkpoint** — gating wired + guarded. No commit yet.

---

### Task 6: ADR-023 + CLAUDE.md criterion rule

**Files:**
- Create: `docs/superpowers/adrs/ADR-023-ci-slow-test-gating.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Pick the next free ADR number, then write the ADR** (follow `docs/superpowers/adrs/ADR-TEMPLATE.md`)

Run `ls docs/superpowers/adrs/ | grep -oE 'ADR-[0-9]+' | sort -t- -k2 -n | tail -1`. `main` shipped ADR-022 (4.19.0) and 4.19.1 (TF-27) referenced existing ADR-007, so `023` is expected free — use it now. (Task 7 Step 1's FF onto latest `main` re-checks for a collision and renames the file if the other session has since landed a new ADR-023.)

Record: **Decision** — gate platform-invariant heavy tests + the benchmark step to a single primary leg (ubuntu-3.12) via a matrix `primary` flag, chosen from real CI durations. **Alternatives** — (a) keep full suite on all legs (the ~20-min status quo); (b) xdist (rejected — reverted before: OOM-killed the 4-core/7-GB runners); (c) blanket "slow on both 3.12 legs" (rejected — keeps invariant smokes on Windows = no win, *and* drops version-sensitive tests from 3.10/3.11). **Consequences** — invariant heavy tests run once; **version-sensitive (golden/snapshot/absolute) tests + contract guards stay on all legs**; partition is structural (matrix flag + tripwire); measure-first because local ≠ CI (the pyright-as-a-test skips on CI — `[test]` omits pyright — while calibration tests run; and per-test Windows cost ≠ local cost). **Status:** Accepted.

- [ ] **Step 2: Add the CLAUDE.md rule** under the `## Testing` section:

```markdown
- **`@pytest.mark.slow` = expensive AND platform/interpreter-invariant** (does-it-run train-script
  smokes, same-run numba-vs-numpy parity). These run once on the CI primary leg (ubuntu-3.12) only;
  every other leg runs `-m "not e2e and not slow"`. Do NOT mark version-sensitive tests (golden-hash /
  snapshot / absolute-numeric) `slow` — they must stay on all legs (OS + interpreter axes). The
  partition is guarded by `tests/test_ci_slow_gating_wired.py`. Decision: ADR-023.
```

- [ ] **Step 3: Lint the docs/markers locally**

Run: `.venv/Scripts/ruff check tests/ silly_kicks/` and `.venv/Scripts/ruff format --check tests/`
Expected: clean (the new tripwire test + marker additions pass).

- [ ] **Step 4: Checkpoint.**

---

### Task 7: Phase 2 — validate on real CI

**Files:** none (validation gate)

- [ ] **Step 1: FF onto latest main (re-check ADR number), present diff + HOLD for approval to commit + push**

Run: `git fetch origin main` then `git merge --ff-only origin/main` (pick up any new main; resolve only if needed). **Re-run** `ls docs/superpowers/adrs/` — if the other session has landed an `ADR-023` since Task 6, rename this PR's ADR to the next free number and update its cross-references (CLAUDE.md rule, CHANGELOG if any). Present the full `git diff --stat` + proposed commit message. On approval, commit (`git commit -F`) + push to the existing PR branch.

- [ ] **Step 2: Read the new CI run; confirm the win + correctness**

Wait for the PR's CI run. Confirm from `gh run view <id> --json jobs`:
- **windows-latest 3.12** wall-clock **materially lower** than the ~16–20 min baseline (it now runs `-m "not e2e and not slow"`).
- **ubuntu-latest 3.12** (primary) ran the full `-m "not e2e"` bulk + the `--benchmark-only` step, green, and is **not** the new long pole.
- ubuntu 3.10/3.11 green.
- The tripwire test passed on every leg.

Run: `gh run view <id> --json jobs --jq '.jobs[] | {name, started:.startedAt, completed:.completedAt, conclusion}'`

- [ ] **Step 3: If the primary leg is the new long pole**, the first lever is already in the design (single bulk process). If still slow, STOP and report — do not add xdist; session-fixture caching is the next (separate) lever. Otherwise proceed.

- [ ] **Step 4: Checkpoint** — CI green, Windows long-pole reduced, primary leg healthy.

---

### Task 8: Finalize (mark PR ready, merge — gated)

**Files:** none

- [ ] **Step 1: Confirm no version bump is needed**

This PR touches only `.github/`, `tests/`, `docs/`, `CLAUDE.md` — none shipped in the wheel/sdist (`[tool.hatch.build.targets.{wheel,sdist}]` packages `silly_kicks` only). `publish.yml` fires only on `v*` tags. So **no version bump, no tag** — confirm none of `pyproject.toml` / `silly_kicks/__init__.py` version fields changed.

- [ ] **Step 2: Mark PR ready + present merge command, HOLD for approval**

`gh pr ready <#>`; present the squash-merge command (`gh pr merge <#> --squash --admin --delete-branch`). On explicit approval, merge. **No tag push** (this ships nothing to PyPI).

- [ ] **Step 3: Checkpoint** — merged to main; CI-time win landed; no release.

---

## Self-Review

**Spec coverage:** Phase 0 measurement w/ `--durations=0` + setup+call+teardown (Task 1) ✓; three-clause invariance classification (Task 2) ✓; `slow` marks on invariant set, version-sensitive stay unmarked (Task 4) ✓; matrix `primary` flag + mutually-exclusive single-process bulk steps + benchmark guard + `--durations=25` (Task 5) ✓; structured-YAML tripwire red-first (Task 3) ✓; ADR-023 + CLAUDE.md rule (Task 6) ✓; Phase-2 CI validation, no-new-long-pole check, no-xdist (Task 7) ✓; no version bump (Task 8) ✓; FF onto 4.19.1 (Task 7 Step 1) ✓.

**Placeholder scan:** The Task-2 `slow` list is *intentionally* finalized from Phase-0 CI data (measure-first); Task 2 gives the explicit criterion + a concrete starting hypothesis + the confirm-from-CI rule (not a vague "TBD"). All code/command steps are complete.

**Type/identifier consistency:** `matrix.primary` flag, `${{ matrix.primary }}` / `${{ !matrix.primary }}` step guards, and the tripwire's `_guard`-normalized (brace-stripped) `matrix.primary` / `!matrix.primary` keys all match; the tripwire keys on the two `--benchmark-skip` bulk steps and asserts the `not slow` gating effect (Task 3) against the Task-5 YAML. `--durations=0` (Phase 0) → `--durations=25` (Phase 1 permanent) consistent across Task 1 / Task 5 / spec. `pyyaml` declared (Task 3) before the tripwire it backs.
