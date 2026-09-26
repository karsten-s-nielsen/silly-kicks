# CI runtime: regenerate durations + decouple the slow suite — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]` checkboxes. Subagents are banned in this repo — execute inline.

**Goal:** Get binding CI job wall-clock under ~10 min by regenerating the stale `.test_durations` (A) and moving the 37%-of-runtime `@pytest.mark.slow` tail out of the primary leg into a dedicated ×2-sharded job (C), keeping the shard matrix symmetric and the coverage proof strictly extended.

**Architecture:** GitHub Actions matrix + `pytest-split`. The `test` matrix runs `not e2e and not slow --splits 3` on every leg; a new `slow` job runs `slow and not e2e --splits 2` once on ubuntu-3.12; `shard-reconcile` proves both partitions plus a `non-slow ⊎ slow == not-e2e` conservation cross-check on the primary interpreter. Three `ci.yml`-reading guard tests pin the topology.

**Tech Stack:** GitHub Actions YAML, `pytest`, `pytest-split`, `pyyaml` (guard tests), Python 3.10–3.12.

**Spec:** `docs/superpowers/specs/2026-09-25-ci-runtime-slow-decouple-design.md` (r2 APPROVE).

## Global Constraints

- Docs/CI-infra only — **no package change, no version bump, no PyPI**. `silly_kicks/` source untouched.
- Single feature branch `ci/regen-test-durations` (already checked out, @ `5bc80ed`). Minimal commits, squashed at merge. **Every commit and the merge are separately owner-gated** — do not commit without explicit approval for that commit.
- `.test_durations` MUST be CI-measured (already regenerated + applied in the working tree; do not re-capture locally).
- TDD: guard assertions land RED against the current `ci.yml`, verified red, before the `ci.yml` edit makes them green.
- Load-bearing CI infra → after implementation, FREEZE the tree and hand off for the owner's independent `/review-impl`. Do not run `/final-review` concurrently with that.
- Lint at CI scope only: `python -m ruff check silly_kicks/ tests/ scripts/` + `ruff format --check …`; `pyright` bare.

---

### Task 1: Part A — regenerate durations + remove the temporary capture job

**Status: already applied in the working tree** (uncommitted/unstaged). This task documents it and re-verifies; no new edit unless a check fails.

**Files:**
- Modify: `.test_durations` (regenerated from the CI-measured `test-durations-ci` artifact of run 36170683006)
- Modify: `.github/workflows/ci.yml` (remove the `durations-capture` job added in `5bc80ed`)

- [ ] **Step 1: Verify the regenerated durations cover the current suite.**
```bash
python -c "import json; d=json.load(open('.test_durations',encoding='utf-8')); print('entries',len(d),'total_min',round(sum(d.values())/60,1))"
```
Expected: `entries 9823 total_min 36.5` (superset of the 9772 collected).

- [ ] **Step 2: Verify the capture job is gone and `ci.yml` is net-zero vs `main` for it.**
```bash
grep -c "durations-capture" .github/workflows/ci.yml   # -> 0
git diff --quiet origin/main -- .github/workflows/ci.yml && echo "ci.yml == origin/main (net zero from A)"
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8')); print('valid YAML')"
```
Expected: `0`, the net-zero line, `valid YAML`. (Net-zero holds only until Task 3 edits `ci.yml`; that is expected.)

---

### Task 2: Rewrite the two existing guards + add the new reconcile guard (RED)

Assertions first. All three must be RED against the current `ci.yml` (which still has the inline-slow topology). This is the whole TDD proof — a guard that is green before the `ci.yml` edit is asserting nothing new.

**Files:**
- Modify: `tests/test_ci_slow_gating_wired.py`
- Modify: `tests/test_ci_shard_wiring.py`
- Create: `tests/test_ci_slow_reconcile_wired.py`

- [ ] **Step 1: Rewrite `tests/test_ci_slow_gating_wired.py`.** Replace `test_ci_bulk_steps_partition_with_slow_gating` (keep `_guard` helper unused-removal + keep `test_slow_marker_set_is_non_empty` verbatim):
```python
def test_slow_is_a_dedicated_job_not_inline() -> None:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    test_job = ci["jobs"]["test"]

    # (a) matrix.primary is gone -- slow no longer runs inline on a special leg
    include = test_job["strategy"]["matrix"].get("include", [])
    assert not any(e.get("primary") for e in include), f"matrix.primary must be removed, got {include}"
    assert "matrix.primary" not in str(test_job["steps"]), "no matrix.primary token anywhere in test steps"

    # (b) exactly one UNCONDITIONAL bulk step, excluding slow
    bulk = [
        s for s in test_job["steps"]
        if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]
    ]
    assert len(bulk) == 1, f"expected one unconditional bulk step, got {len(bulk)}: {[s.get('run') for s in bulk]}"
    assert "if" not in bulk[0], "matrix bulk step must be unconditional (no per-leg gate)"
    assert "not e2e and not slow" in bulk[0]["run"], "matrix bulk step must exclude slow"

    # (c) a dedicated `slow` job runs the invariant tail once on the primary interpreter
    slow = ci["jobs"]["slow"]
    assert str(slow["runs-on"]).startswith("ubuntu"), f"slow job must run on ubuntu, got {slow['runs-on']}"
    setup = [s for s in slow["steps"] if "setup-python" in str(s.get("uses", ""))]
    assert setup and str(setup[0]["with"]["python-version"]) == "3.12", "slow job must use python 3.12 (ADR-023 invariance)"
    sbulk = [
        s for s in slow["steps"]
        if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]
    ]
    assert len(sbulk) == 1, f"expected one slow bulk step, got {len(sbulk)}"
    assert "slow and not e2e" in sbulk[0]["run"], "slow job must select 'slow and not e2e'"
```
Delete the now-unused `_guard` helper.

- [ ] **Step 2: Run it — expect RED.**
```bash
python -m pytest tests/test_ci_slow_gating_wired.py -q
```
Expected: `test_slow_is_a_dedicated_job_not_inline` FAILS — current `ci.yml` has `matrix.primary` + two bulk steps + no `slow` job (`KeyError: 'slow'` or the primary assertion). `test_slow_marker_set_is_non_empty` PASSES.

- [ ] **Step 3: Extend `tests/test_ci_shard_wiring.py`.** Generalize `_sharded_cmds` and add the slow axis:
```python
def _sharded_cmds() -> list[str]:
    jobs = [_CI["jobs"]["test"]]
    if "slow" in _CI["jobs"]:
        jobs.append(_CI["jobs"]["slow"])
    return [
        s["run"]
        for job in jobs
        for s in job["steps"]
        if "run" in s and "--splits" in s["run"] and "pytest tests/" in s["run"]
    ]


def test_shard_axis_is_contiguous_1_to_N() -> None:
    shards = _CI["jobs"]["test"]["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1)), f"test shard axis must be 1..N contiguous, got {shards}"
    slow_shards = _CI["jobs"]["slow"]["strategy"]["matrix"]["slow-shard"]
    assert slow_shards == list(range(1, len(slow_shards) + 1)), f"slow-shard axis must be 1..N, got {slow_shards}"


def test_splits_value_matches_shard_count() -> None:
    test_n = len(_CI["jobs"]["test"]["strategy"]["matrix"]["shard"])
    slow_n = len(_CI["jobs"]["slow"]["strategy"]["matrix"]["slow-shard"])
    checks = [
        (_CI["jobs"]["test"], test_n, "matrix.shard"),
        (_CI["jobs"]["slow"], slow_n, "matrix.slow-shard"),
    ]
    for job, n, group_var in checks:
        cmds = [s["run"] for s in job["steps"] if "run" in s and "--splits" in s["run"] and "pytest tests/" in s["run"]]
        assert cmds, f"no sharded pytest commands in job with group var {group_var}"
        for cmd in cmds:
            m = re.search(r"--splits\s+(\d+)", cmd)
            assert m and int(m.group(1)) == n, f"--splits must equal shard count {n}: {cmd}"
            assert f"--group ${{{{ {group_var} }}}}" in cmd, f"missing per-shard --group {group_var}: {cmd}"
```
Update `test_shard_reconcile_job_exists_and_needs_test`:
```python
def test_shard_reconcile_job_exists_and_needs_test_and_slow() -> None:
    job = _CI["jobs"].get("shard-reconcile")
    assert job is not None, "shard-reconcile job must exist"
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    assert "test" in needs and "slow" in needs, f"shard-reconcile must need both test and slow, got {needs}"
```
Update `test_numba_cache_key_covers_all_njit_files` to check BOTH jobs' cache steps:
```python
    for job_name in ("test", "slow"):
        cache = [s for s in _CI["jobs"][job_name]["steps"] if "actions/cache" in str(s.get("uses", ""))]
        assert cache, f"no numba actions/cache step in the {job_name} job"
        patterns = re.findall(r"'([^']+)'", str(cache[0]["with"]["key"]))
        covered: set[str] = set()
        for pat in patterns:
            covered |= {str(p.relative_to(_REPO)).replace("\\", "/") for p in _REPO.glob(pat)}
        missing = set(njit_files) - covered
        assert not missing, f"@njit files not covered by the {job_name} numba cache key: {sorted(missing)}"
```
(`_every_sharded_command_pins_collection_order` and the shuffle/benchmark tests need no change — `_sharded_cmds` now spans both jobs, so `-p no:randomly` is checked on the slow command too.)

- [ ] **Step 4: Run it — expect RED.**
```bash
python -m pytest tests/test_ci_shard_wiring.py -q
```
Expected: the four touched tests FAIL (`KeyError: 'slow'`). The unchanged ones (`no_collection_shuffling`, `benchmark_standalone`) PASS.

- [ ] **Step 5: Create `tests/test_ci_slow_reconcile_wired.py`.** New anti-rot guard (spec §4.4, CI-SPEC-01):
```python
"""Structural guard: the slow-decouple conservation invariant lives ONLY in CI runtime
(the shard-reconcile body + the slow job's node-ID uploads), so a future ci.yml edit could
drop it with every other guard staying green. This pins it the pandas-span way: parsed-YAML
topology + exactly one text-grep of the reconcile run scalar for a required sentinel.

See docs/superpowers/specs/2026-09-25-ci-runtime-slow-decouple-design.md (CI-SPEC-01)."""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _upload_names(job: dict) -> set[str]:
    return {
        str(s["with"]["name"])
        for s in job["steps"]
        if "upload-artifact" in str(s.get("uses", "")) and isinstance(s.get("with"), dict) and "name" in s["with"]
    }


def test_slow_job_uploads_reconcile_artifacts() -> None:
    slow = _CI["jobs"]["slow"]
    names = _upload_names(slow)
    # names carry a ${{ matrix.slow-shard }} suffix on the per-shard set; match by prefix
    assert any(n.startswith("slow-shard-nodeids") for n in names), f"slow job must upload slow-shard-nodeids, got {names}"
    assert "slow-full-nodeids" in names, f"slow job must upload slow-full-nodeids, got {names}"
    assert "combined-full-nodeids" in names, f"slow job must upload combined-full-nodeids, got {names}"


def test_reconcile_consumes_slow_and_combined() -> None:
    job = _CI["jobs"]["shard-reconcile"]
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    assert "slow" in needs, f"shard-reconcile must need slow, got {needs}"
    patterns = {
        str(s["with"]["pattern"])
        for s in job["steps"]
        if "download-artifact" in str(s.get("uses", "")) and isinstance(s.get("with"), dict) and "pattern" in s["with"]
    }
    # slow shards carry a -<shard> suffix (glob); the two single artifacts are matched by their
    # exact name as a literal glob pattern (a `-*` suffix would NOT match a suffix-less name).
    for required in ("slow-shard-nodeids-*", "slow-full-nodeids", "combined-full-nodeids"):
        assert required in patterns, f"reconcile must download {required} (proof #2/#3 inputs), got {patterns}"


def test_reconcile_body_has_conservation_sentinel() -> None:
    job = _CI["jobs"]["shard-reconcile"]
    bodies = "\n".join(s["run"] for s in job["steps"] if "run" in s)
    # the single text check: a load-bearing sentinel the conservation proof must carry, so this
    # guard is non-vacuous (numba-cache idiom). Removing proof #3 removes the marker -> RED.
    assert "# CONSERVATION:" in bodies, "reconcile body must carry the '# CONSERVATION:' proof (non-slow union slow == not-e2e)"
```

- [ ] **Step 6: Run it — expect RED.**
```bash
python -m pytest tests/test_ci_slow_reconcile_wired.py -q
```
Expected: all three FAIL (`KeyError: 'slow'` / no `combined-full-nodeids` / no sentinel).

- [ ] **Step 7: Lint the touched test files.**
```bash
python -m ruff check tests/test_ci_slow_gating_wired.py tests/test_ci_shard_wiring.py tests/test_ci_slow_reconcile_wired.py
python -m ruff format --check tests/test_ci_slow_gating_wired.py tests/test_ci_shard_wiring.py tests/test_ci_slow_reconcile_wired.py
```
Expected: clean.

---

### Task 3: Edit `ci.yml` topology — make all three guards GREEN

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Symmetric matrix.** In `jobs.test`:
  - Remove the `include:` block (the `primary: true` ubuntu-3.12 entry, `ci.yml` lines 61-67).
  - Replace the two `if: ${{ !matrix.primary }}` / `if: ${{ matrix.primary }}` bulk steps (lines 123-126) with ONE unconditional step:
```yaml
      - run: pytest tests/ -m "not e2e and not slow" --splits 3 --group ${{ matrix.shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
```
  - De-ternary the two `--co` steps (lines 159, 166): replace `-m "${{ matrix.primary && 'not e2e' || 'not e2e and not slow' }}"` with `-m "not e2e and not slow"` in both.

- [ ] **Step 2: Add the `slow` job** (after `jobs.test`, before `pandas-span`):
```yaml
  # ADR-023: the platform-/interpreter-INVARIANT heavy tail (@pytest.mark.slow, 249 tests / ~13.6 min)
  # runs ONCE here, sharded x2, on the primary interpreter -- decoupled from the symmetric `test` matrix
  # so no matrix leg is structurally heavier. Balanced by the SAME committed .test_durations (pytest-split
  # filters to the -m selection). Wiring pinned by test_ci_slow_gating_wired / _shard_wiring / _slow_reconcile_wired.
  slow:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        slow-shard: [1, 2]
    env:
      SILLY_KICKS_ASSERT_INVARIANTS: "1"
      NUMBA_CACHE_DIR: ${{ github.workspace }}/.numba_cache
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
          cache: pip
      - uses: actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0
        with:
          path: ${{ github.workspace }}/.numba_cache
          key: numba-ubuntu-latest-3.12-${{ hashFiles('silly_kicks/tracking/**/*_numba*.py', 'silly_kicks/xtgk/_turnover.py') }}
      - run: pip install -e ".[kloppy,xgboost,das,test]"
      - run: pytest tests/ -m "slow and not e2e" --splits 2 --group ${{ matrix.slow-shard }} -p no:randomly --benchmark-skip --tb=short --durations=25
      # reconcile artifacts (proofs #2/#3): each slow-shard uploads its node-ID set; slow-shard 1 also
      # uploads the slow FULL set and the combined `not e2e` (incl slow) set for the conservation check.
      - shell: bash
        run: pytest tests/ -m "slow and not e2e" --splits 2 --group ${{ matrix.slow-shard }} -p no:randomly --co -q | grep -E "^tests/.*::" | sort > slow-shard-nodeids.txt
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: slow-shard-nodeids-${{ matrix.slow-shard }}
          path: slow-shard-nodeids.txt
      - if: ${{ matrix.slow-shard == 1 }}
        shell: bash
        run: pytest tests/ -m "slow and not e2e" -p no:randomly --co -q | grep -E "^tests/.*::" | sort > slow-full-nodeids.txt
      - if: ${{ matrix.slow-shard == 1 }}
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: slow-full-nodeids
          path: slow-full-nodeids.txt
      - if: ${{ matrix.slow-shard == 1 }}
        shell: bash
        run: pytest tests/ -m "not e2e" -p no:randomly --co -q | grep -E "^tests/.*::" | sort > combined-full-nodeids.txt
      - if: ${{ matrix.slow-shard == 1 }}
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: combined-full-nodeids
          path: combined-full-nodeids.txt
```

- [ ] **Step 3: Extend `shard-reconcile`.** Change `needs: test` → `needs: [test, slow]`; add three `download-artifact` steps (the two single artifacts use their exact name as the literal `pattern` — a `-*` suffix would not match a suffix-less name):
```yaml
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          pattern: slow-shard-nodeids-*
          path: slow-shards/
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          pattern: slow-full-nodeids
          path: slow-full/
      - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          pattern: combined-full-nodeids
          path: combined/
```
Then append to the Python body, after the existing per-leg loop (before the `if errs:` exit block), the two new proofs. The combined proof carries the required sentinel comment:
```python
          # --- proof #2: the slow job's 2 shards partition the `slow and not e2e` set ---
          slow_shard_sets = [load(f) for f in pathlib.Path("slow-shards").rglob("slow-shard-nodeids.txt")]
          slow_full_files = list(pathlib.Path("slow-full").rglob("slow-full-nodeids.txt"))
          combined_files  = list(pathlib.Path("combined").rglob("combined-full-nodeids.txt"))
          assert slow_shard_sets, "::error::no slow-shard-nodeids -- slow partition would pass vacuously"
          assert slow_full_files, "::error::no slow-full-nodeids -- slow partition would pass vacuously"
          assert combined_files,  "::error::no combined-full-nodeids -- conservation would pass vacuously"
          slow_full = load(slow_full_files[0])
          combined  = load(combined_files[0])
          seen = set()
          for g in slow_shard_sets:
              dup = seen & g
              if dup: errs.append(f"slow: {len(dup)} node(s) in >1 slow-shard e.g. {sorted(dup)[:3]}")
              seen |= g
          if seen != slow_full:
              errs.append(f"slow shards != slow full: missing {len(slow_full - seen)}, extra {len(seen - slow_full)}")

          # CONSERVATION: non-slow(ubuntu-latest-3.12) disjoint-union slow == combined `not e2e`. This is
          # the proof that decoupling dropped NOTHING -- the tests removed from the matrix reappear exactly
          # once in the slow job. Do not remove; test_ci_slow_reconcile_wired.py pins this marker.
          nonslow_primary = fulls.get("ubuntu-latest-3.12")
          if nonslow_primary is None:
              errs.append("conservation: no ubuntu-latest-3.12 non-slow full to cross-check")
          else:
              overlap = nonslow_primary & slow_full
              if overlap: errs.append(f"conservation: {len(overlap)} node(s) in BOTH non-slow and slow e.g. {sorted(overlap)[:3]}")
              union = nonslow_primary | slow_full
              if union != combined:
                  errs.append(f"conservation: non-slow U slow != not-e2e: missing {len(combined - union)}, extra {len(union - combined)}")
```
(Place these before the existing `if errs:` exit block so all failures report together.)

- [ ] **Step 4: Validate YAML.**
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8')); print('valid YAML')"
```

- [ ] **Step 5: Run the three guards — expect GREEN.**
```bash
python -m pytest tests/test_ci_slow_gating_wired.py tests/test_ci_shard_wiring.py tests/test_ci_slow_reconcile_wired.py -q
```
Expected: all pass.

---

### Task 4: Local verification + freeze for independent review

**Files:** none (verification only).

- [ ] **Step 1: Full local suite (not e2e).**
```bash
python -m pytest tests/ -m "not e2e" -p no:randomly --benchmark-skip -q
```
Expected: green. (Local timing/balance is not the check — CI is; this proves nothing broke.)

- [ ] **Step 2: Lint + types at CI scope.**
```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
pyright
```
Expected: clean (only `tests/` touched + `ci.yml`; no `silly_kicks/` change).

- [ ] **Step 3: Confirm what the runtime proofs will show, then STOP.** The matrix partition, slow partition, and conservation cross-check are observable only on a real CI run — they cannot be reproduced locally. Freeze the tree. Hand off for the owner's independent `/review-impl`. Do NOT commit, push, or run `/final-review` until that review returns.

---

### Task 5: Commit + ship (each step separately owner-gated)

Do nothing here without explicit per-action approval.

- [ ] **Step 1 (owner-gated): commit.** A single A+C commit on `ci/regen-test-durations` (PLAN-02: Task 3 layers C onto A's `ci.yml` before any commit, so the two are not cleanly carvable, and the merge squashes to one regardless — one commit is the most-minimal form the owner asked for). Contents: `.test_durations`, `ci.yml` (capture-job removal + the C topology), the three guard files, the ADR-023 + ADR-074 amendment notes (Phase 2.5 — delivery-mechanism change), the §4.5 downstream doc-consistency edits (`AGENTS.md` slow bullet, `docs/context/ci.md` slow paragraph, `tests/fixtures/agents_md_invariant_inventory.json` `slow-gating` token swap), the spec + this plan (docs travel with the change). `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; NO `Claude-Session:` trailer.
- [ ] **Step 2 (owner-gated): push + update PR #256** (retitle from "DO NOT MERGE …" to the A+C cycle; body = spec summary + the wall-clock table).
- [ ] **Step 3: watch CI to GREEN** — verify the `slow` job runs, `shard-reconcile` prints its OK line (both partitions + conservation), and the binding job wall-clock lands ~10 min. Never propose merge before green.
- [ ] **Step 4 (owner-gated): squash-merge.** Prefer `gh pr merge --squash --delete-branch` **on the checks** (PLAN-03 — the matrix job names are preserved, C only ADDS the `slow`/reconcile checks, so nothing stale needs routing around; branch protection is `required_status_checks:null` so a plain merge passes). Use `--admin` only if a genuinely stale required check blocks. Then verify `main` CI green. No tag (no release).
- [ ] **Step 5:** save memory + note F (staleness guard) as the owner-sequenced next consideration.

## Self-review

- **Spec coverage:** A (Task 1), symmetric matrix + slow job (Task 3.1/3.2), extended reconcile with conservation (Task 3.3), the two guard rewrites (Task 2.1/2.3) + the new CI-SPEC-01 guard (Task 2.5), F cited-not-built (Global Constraints + Task 5.5), no version bump (Global Constraints). All spec sections map to a task.
- **Placeholder scan:** none — every guard body and `ci.yml` block is concrete.
- **Type/name consistency:** artifact names (`slow-shard-nodeids`, `slow-full-nodeids`, `combined-full-nodeids`) match between the slow job (Task 3.2 uploads), the reconcile downloads (Task 3.3), and the guard (Task 2.5). The `# CONSERVATION:` sentinel matches between Task 3.3 (body) and Task 2.5 (grep). `matrix.slow-shard` group var matches between Task 3.2 and the `test_splits_value_matches_shard_count` check (Task 2.3).
- **TDD order:** Task 2 lands all guards RED (verified per file) before Task 3's `ci.yml` edit turns them GREEN.
