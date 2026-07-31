# Corpus-Driver Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every `scripts/` driver that walks a corpus a shared seam that persists each item, resumes after a crash, reports progress, and cannot silently reuse stale results.

**Architecture:** One new module `scripts/_driver.py` owns the per-item loop (`for_each`) and exposes seven primitives for loops that cannot invert, plus one operator utility (`prune_stale_generations`) that `for_each` deliberately never calls. It consumes the existing `_partition.py` (partitioning, manifest aggregation) and `_cache.py` (fingerprint, fail-closed metadata) rather than replacing them. A staleness token names a *generation directory*, so a changed input yields a different path and stale shards are unreachable rather than merely guarded. A rewritten CI gate checks helper **adoption**, not source keywords.

**Tech Stack:** Python 3.10, pandas, pytest, ruff, pyright. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-07-29-corpus-driver-resilience-design.md` (874 lines, three review rounds applied).

---

## Before you start

**Environment.** `.venv/Scripts/python.exe` is Python 3.10 and is the default for all commands below. `.venv312` exists for CI reproduction (pandas 3.x) and is used only where a task says so. Never `pip install` into `.venv`.

**CI's quality gate, run exactly as CI runs it.** Note that ruff is **path-scoped** and pyright is
**config-driven** — they are not the same shape, and getting it backwards produces a wall of noise:

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -rs --tb=short -q
```

Verbatim from `.github/workflows/ci.yml:25-30`. Three traps, each measured rather than assumed:

- **Bare `ruff check` is NOT what CI runs.** Without the paths it lints the whole repo, including
  `docs/research/**/harness/*.py` — throwaway research scripts that were never meant to pass. Measured
  on a commit whose CI run was green: **232 errors, 17 files "would be reformatted."** Running
  `ruff format` on that output would produce a large, wrong diff justified by a green local gate. An
  earlier revision of this plan told you to run it bare; that instruction was wrong.
- **`pyright` takes no PATH arguments** — its scope is `[tool.pyright] include`, covering
  `silly_kicks`, `tests` *and* `scripts`. A scoped `pyright scripts/` is a different check and has
  produced a false green before. **But it DOES need `--pythonpath`.** MEASURED with `--verbose`:
  bare `.venv/Scripts/python.exe -m pyright` resolves against
  `AppData\Roaming\Python\Python314\site-packages` — a system Python 3.14 holding *different,
  older* distributions (ruthless 0.2.1 there vs 0.4.0 in `.venv`), which manufactured a bogus
  `"fingerprint" is unknown import symbol`. Always pass
  `--pythonpath .venv/Scripts/python.exe`. Do NOT add `venvPath`/`venv` to `pyproject.toml`: CI
  installs into its ambient interpreter and runs bare, so CI is self-consistent and a `.venv` pin
  would break it. The mismatch is purely local.
- **pyright is NOT expected to be zero.** Against `.venv` the baseline is **36 errors, all
  pre-existing in `tests/`** (13 in `test_xt_gk.py`, 4 in `test_enrichment_nan_safety.py`, the rest
  spread over tracking/invariants/conftest). None in `silly_kicks/`, `scripts/` or `tests/scripts/`.
  Prove YOUR contribution is zero with
  `grep "error:" <output> | grep -E "scripts\\|scripts/"` and require it empty — cheaper and
  safer than a stash-based baseline (see the warning below).
- **NEVER chain `git stash` … long command … `git stash pop` in one shell invocation.** A 28 s tool
  timeout killed exactly that compound mid-flight during this cycle and left every uncommitted
  change sitting in `stash@{0}` behind a working tree that looked reverted. Recovery was a clean
  `git stash pop`, but run the three steps separately.
- **The tool versions are pinned in CI and unbounded in `pyproject.toml`.** CI installs
  `ruff==0.15.7`; the `dev` extra says `ruff>=0.8.0`, so a fresh sync gets whatever is newest (0.15.12
  here). Checked: both versions produce identical output on this tree, so the gap is currently benign —
  but it is unguarded, and it is the same shape as the `xgboost <3.0`-vs-`<4.0` doc/build drift found
  in this repo on 2026-07-30. Worth tightening; not this cycle's job.

**Use `-rs` on every suite run.** A pass→skip transition is silent in a bare total: only the reasons
make it attributable. This cycle spent a run chasing an unexplained `-1 passed / +1 skipped` that a
captured skip list would have named immediately.

**Measured baseline on this branch's base (`89dd9af`, 4.71.0), with `SILLY_KICKS_ASSERT_INVARIANTS=1`:**

| Command | Result | Wall clock |
|---|---|---|
| `pytest tests/ -m "not e2e"` | **6346 passed, 57 skipped, 184 deselected, 10 xfailed** | **18m27s** |
| `pytest tests/scripts/` | 195 passed, 36 skipped | 43s |
| `ruff check` + `format --check` (scoped) | clean / 791 files formatted | seconds |
| `pyright --pythonpath .venv/Scripts/python.exe` | **36 errors, all pre-existing in `tests/`**; zero in `scripts/` | ~1m |

Two things follow. **Compare against these numbers, not against zero** — 57 skips and 10 xfails are
the healthy state, and the 10 xfails are ADR-051's strict markers, which must not change in this
cycle. And **the full suite takes eighteen minutes**, so it belongs at the three commit boundaries,
not after every task; per-task verification is the scoped `tests/scripts/` run, which is why the plan
calls it 44 times and the full suite 5. A backgrounded full run whose output file is still empty is
buffering, not hanging.

**Test-file conventions.** `tests/scripts/` deliberately has **no** `__init__.py` — adding one shadows the top-level `scripts` namespace package and breaks collection across the suite. Script tests use a bare `import scripts.<name> as mod`. Never add a module-level `sys.path.insert`.

**Commit policy — THREE commits, not squashed.** The owner revised this from "a commit per script,
squashed on merge": one feature branch, **three** commits, merged without squashing.

| # | Contents | Tasks |
|---|---|---|
| 1 | The seam — `scripts/_driver.py`, `_partition.py`'s generation handling, the resume oracle, the adoption gate landing red | 0, 1, **1b**, 2–6, **6b**, 7–10 |
| 2 | The migrations — all 21 drivers onto the seam, the cohort cache, Databricks auth precedence | 11–16 |
| 3 | Correctness fixes, ADR, version — behavioural bundling guard, the cover-shadow RC1 repair, three-state provenance, gate closed, docs + bump | **14b**, **14c**, **16b**, 17–19 |

Tasks 14b and 14c are correctness changes — a defective guard and a live ADR-028 RC1 defect — not migrations, so they land in commit 3 even though Task 14 touches both files in commit 2. A geometry repair should be legible in its own diff, not buried in a resilience refactor. Task 19 (kill-and-resume) is an acceptance run, not a code change — it produces no commit of its own.

**The per-task `git commit` blocks below are CHECKPOINTS, not commits.** Reaching one means: run the
four gate commands, confirm green, `git add` the listed paths — and stop there. The three real
commits happen at the boundaries above, each proposed to the owner and awaiting confirmation. Do not
push or open a PR until the whole cycle is done.

Why three and not twenty: a commit per script reads well in a log that is about to be squashed away,
and this branch will not be. Three commits that each leave the tree green and each answer one
question ("what is the seam", "who adopts it", "what guards it") are more useful to `git bisect` and
to a reviewer than twenty that only make sense in sequence.

**Branch.** All work happens on `feat/corpus-driver-resilience`, created from `main` in Task 0.

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/_driver.py` | **New.** `for_each` + the seven primitives it composes from + `prune_stale_generations` (operator-only, never called by the loop) + the cohort cache. The only file that owns a corpus loop. |
| `scripts/_partition.py` | **One behaviour change:** `aggregate_manifests` gains named handling for `generation` (Task 6b), mirroring `run_commit` — without it the token `manifest_fields` records is silently dropped from the corpus artifact, measured. Plus one docstring correction (Task 18) naming which cycle its read-only fence bound. |
| `scripts/_cache.py` | **Unchanged.** Consumed by `_driver.py` for fail-closed cache metadata. |
| `tests/scripts/test_driver.py` | **New.** Unit tests for every primitive and for `for_each`. |
| `tests/scripts/test_corpus_driver_resilience.py` | **Rewrite.** Adoption gate, replacing the drafted keyword version. |
| `tests/scripts/test_driver_resume_oracle.py` | **New.** Double-invocation oracle for the three resumable drivers. |
| `scripts/build_layer2_spells.py` etc. | **Modified** — 21 drivers, one checkpoint each, all inside commit 2. |
| `scripts/_provenance.py` | **Modified** (Task 16b) — adds `tree_state` beside the unchanged `dirty` boolean. |
| `scripts/train_gk_completion.py` | **Modified** (Task 14b) — `--mode`/`--reason` and a served-prediction bundling guard, beyond its migration. |
| `scripts/measure_cover_shadow_argmax_agreement.py` | **Modified** (Task 14c) — the live ADR-028 RC1 passer reprojection, beyond its migration. |
| `scripts/_loader_databricks.py` | **Modified** (Task 16) — auth precedence only, in `_connect`. |
| `docs/superpowers/adrs/ADR-0NN-corpus-driver-contract.md` | **New** (Task 18). |

---

## Task 0: Branch

- [ ] **Step 1: Confirm a clean tree and current main**

Run:
```bash
git -C D:/Development/karstenskyt__silly-kicks_part-deux status --porcelain
```
Expected, exactly four entries:

```
 M uv.lock
?? docs/superpowers/plans/2026-07-29-corpus-driver-resilience.md
?? docs/superpowers/specs/2026-07-29-corpus-driver-resilience-design.md
?? tests/scripts/test_corpus_driver_resilience.py
```

Anything else — stop and ask.

**`uv.lock` is a MODIFIED tracked file and it belongs to this cycle.** It is the
`ruthless-efficiency 0.2.1 → 0.4.0` resolve (3 lines: version, sdist, wheel), made when the token
delegation in Task 1 was adopted, and it pairs with the declared-floor bump in Task 1 Step 0 — the
lock resolves 0.4.0 while `pyproject.toml` still permits 0.2.1, which is the gap that step closes.
Carry it into **commit 1**. An earlier revision of this step listed only the three untracked files
and would have stopped the executor on its own first command.

- [ ] **Step 2: Create the branch**

```bash
git checkout -b feat/corpus-driver-resilience
```

---

## Task 1: `generation_dir` — the staleness token as a directory

**Files:**
- Create: `scripts/_driver.py`
- Create: `tests/scripts/test_driver.py`

- [ ] **Step 0: Raise the `ruthless-efficiency` floor to the version that has `fingerprint`**

`pyproject.toml` declares `ruthless-efficiency[optuna]>=0.2.1` in **three** places (`[calibration]`
`:70`, `[test]` `:105`, `[train]` `:116`). `fingerprint` became public in **0.4.0**, so the declared
floor currently permits a version where `from ruthless import fingerprint` raises `ImportError`.
`uv.lock` happens to resolve 0.4.0 today, which hides it — a fresh resolve against the declared
floor would not. This repo already treats these floors as load-bearing (`:69` records that `>=0.2.1`
itself fixes a `warm_start` off-by-one).

Edit all three to `ruthless-efficiency[optuna]>=0.4.0`, then:

```bash
uv lock && git diff --stat uv.lock
```
Expected: `uv.lock` unchanged or trivially re-stamped — 0.4.0 is already the resolved version.

Verify the symbol is actually there rather than trusting the release note:

```bash
.venv/Scripts/python.exe -c "import ruthless; print(ruthless.__version__, callable(ruthless.fingerprint))"
```
Expected: `0.4.0 True`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_driver.py`:

```python
"""Unit tests for the shared corpus-driver seam.

The digest itself is NOT tested here — it is `ruthless.fingerprint`, which carries its own golden
table of 44 pinned literals and a stated stability contract (ruthless 0.4.0). These tests pin what
`_driver` adds on top: that a token names a DIRECTORY, that an empty declaration needs a reason, and
that path inputs are normalised before they reach the digest.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

import pytest

import scripts._driver as mod  # bare import: tests/scripts/ has NO __init__.py


def test_same_inputs_give_the_same_generation(tmp_path):
    a = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    b = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    assert a == b
    assert a.is_dir()
    assert a.parent == tmp_path


def test_a_changed_input_gives_a_DIFFERENT_generation(tmp_path):
    """The whole point: a changed declared input must make the old shards unreachable, not merely
    invalid. Different directory => the resume glob cannot see them."""
    a = mod.generation_dir(tmp_path, token_inputs={"box": 40.32})
    b = mod.generation_dir(tmp_path, token_inputs={"box": 40.30})
    assert a != b


def test_key_ORDER_does_not_change_the_generation(tmp_path):
    """Declared inputs are a set of facts, not a sequence. Re-ordering a declaration must not
    invalidate a corpus that took hours to build. (Inherited from `ruthless.fingerprint`, which sorts
    mapping keys; asserted here because `_driver`'s contract promises it.)"""
    a = mod.generation_dir(tmp_path, token_inputs={"provider": "gs", "box": 40.32})
    b = mod.generation_dir(tmp_path, token_inputs={"box": 40.32, "provider": "gs"})
    assert a == b


def test_int_and_str_inputs_are_DISTINGUISHED(tmp_path):
    """Type-tagging, inherited from ruthless. 5 and "5" are different declarations."""
    assert mod.generation_dir(tmp_path, token_inputs={"v": 5}) != mod.generation_dir(
        tmp_path, token_inputs={"v": "5"}
    )


def test_an_UNSUPPORTED_type_is_REFUSED_not_silently_digested(tmp_path):
    """Fail-closed, inherited from ruthless and load-bearing here.

    The hand-rolled token this replaced used `repr()`, so an object without `__repr__` digested its
    MEMORY ADDRESS: a different token every process, meaning the driver never matched its own
    generation and silently full-recomputed every run. Refusing is the only safe answer for a
    cache key."""

    class Unsupported:
        pass

    with pytest.raises(TypeError):
        mod.generation_dir(tmp_path, token_inputs={"cfg": Unsupported()})


def test_PATH_inputs_are_normalised_so_the_token_is_platform_STABLE(tmp_path):
    """ruthless guarantees the same LOGICAL value digests identically everywhere; CONSTRUCTING the
    value is our job. `Path(str)` parses per-platform — a backslash is a separator on Windows and an
    ordinary character on POSIX — and this repo spans a Windows dev box, a Linux DGX and both in CI.
    An un-normalised path input would orphan a generation on the other platform with no version
    having changed."""
    win = mod.generation_dir(tmp_path, token_inputs={"data_dir": PureWindowsPath("a/b")})
    posix = mod.generation_dir(tmp_path, token_inputs={"data_dir": PurePosixPath("a/b")})
    assert win == posix


def test_an_EMPTY_declaration_REQUIRES_a_reason(tmp_path):
    """`()` is legal and means "no staleness risk" -- but a silent omission and a considered
    decision must not look identical in the source."""
    with pytest.raises(ValueError, match="token_reason"):
        mod.generation_dir(tmp_path, token_inputs={})


def test_an_EMPTY_declaration_WITH_a_reason_is_accepted(tmp_path):
    d = mod.generation_dir(tmp_path, token_inputs={}, token_reason="pure re-read of a pinned table")
    assert d.is_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'scripts._driver'`.

- [ ] **Step 3: Create `scripts/_driver.py` with `generation_dir`**

```python
"""The shared seam for maintainer drivers that walk a corpus.

WHY THIS EXISTS. Twenty-one drivers in this directory do expensive per-item work; three survived a
crash. The convention was not absent -- four partial mechanisms already existed (`_partition.py`,
`_cache.py`, `train_ghost_gk`'s `_feature_cache`, `calibrate_xt_bandwidth`'s `--corpus-cache`) --
but none of them owned the LOOP, so resume and progress had nowhere to live except in each author's
memory. `validate_xs_probe.py` then ran for 14 hours, held everything in RAM, wrote once at the end,
and printed nothing.

This module owns the loop. `for_each` is the default shape; the individual primitives are the
escape hatch for a driver whose loops genuinely cannot invert, and such a driver MUST still call
`assert_conservation` (see the spec, section 4.1).

Relationship to its neighbours: `_partition.py` keeps partitioning and manifest aggregation (it
reads `manifest_*.json` only -- shard reconciliation was always driver-local), and `_cache.py`
supplies the fail-closed cache metadata the cohort cache reuses.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from pathlib import PurePath, PurePosixPath

# NOTE: `ruthless` and `pandas` are imported INSIDE the functions that need them, not here.
# `ruthless-efficiency` ships only in the [calibration] / [test] / [train] extras, so a
# module-level import would make `shard_path`, `progress` and `already_done` -- pure stdlib
# helpers -- unreachable without one. `reconcile` imports pandas the same way.

#: Joins the components of a composite shard key. Rejected in any component -- see `shard_path`.
KEY_SEPARATOR = "__"


def _normalise(value: object) -> object:
    """Make a declared input's digest platform-independent before it reaches `fingerprint`.

    ruthless guarantees the same LOGICAL value digests identically on every platform; CONSTRUCTING
    that value is the caller's responsibility, and `Path(str)` parses per-platform -- a backslash is
    a separator on Windows and an ordinary character on POSIX. This repo spans a Windows dev box, a
    Linux DGX and both OSes in CI, so an un-normalised path input would orphan a generation on the
    other platform with no version having changed.
    """
    if isinstance(value, PurePath):
        return PurePosixPath(value.as_posix())
    if isinstance(value, Mapping):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalise(v) for v in value)
    return value


def _token(token_inputs: Mapping[str, object], token_reason: str | None) -> str:
    """Digest of the DECLARED inputs, delegated to `ruthless.fingerprint`.

    NOT hand-rolled, and that is a correctness decision rather than a convenience one. An earlier
    draft hashed `sorted(f"{type(v).__name__}:{v!r}")`, which was defective in two measured ways: an
    object with no `__repr__` digested its MEMORY ADDRESS (a different token every process, so the
    driver never matched its own generation and silently full-recomputed), and a numpy major changed
    the digest of an unchanged declared value. `ruthless.fingerprint` is structural, type-tagged in
    both the value and the key position, and FAIL-CLOSED -- it raises on a type it does not
    understand rather than inventing an unstable digest for a cache key.

    ruthless states digest stability as a compatibility contract with no carve-out, backed by a
    golden table of 44 pinned literals (0.4.0). That contract is what makes it safe to key resume on
    these bytes; see the spec, section 4.2.
    """
    from ruthless import fingerprint  # public since 0.4.0; extra-only, so imported here not at module scope

    payload = dict(token_inputs)
    if not payload:
        if not token_reason:
            raise ValueError(
                "token_inputs={} means 'this pass has no staleness risk' and REQUIRES token_reason. "
                "A silent omission and a considered decision must not look identical in the source."
            )
        return fingerprint({"empty_reason": token_reason})
    return fingerprint({k: _normalise(v) for k, v in payload.items()})


def generation_dir(shard_root, *, token_inputs, token_reason: str | None = None) -> pathlib.Path:
    """Resolve (and create) the generation directory for this set of declared inputs.

    The token names a DIRECTORY, not a filename suffix. Reconciliation across this codebase is a
    bare ``glob("*.parquet")``, so a suffix would make a combined table concatenate the old-token
    and new-token shard for the same item -- with different values -- the first time a declared
    input changed. A directory keeps every existing glob correct once scoped to it, and leaves
    stale generations visible on disk rather than silently mixed into the output.
    """
    root = pathlib.Path(shard_root)
    out = root / _token(token_inputs, token_reason)
    out.mkdir(parents=True, exist_ok=True)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Run the lint and type gate**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
```
Expected: all clean. If `ruff format --check` complains, run `.venv/Scripts/python.exe -m ruff format` and re-run.

- [ ] **Step 6: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.generation_dir -- staleness token as a directory"
```

---

## Task 1b: `prune_stale_generations` — the spec's `--prune-stale`, built

The spec asserts `--prune-stale` twice (§4.1 "`--prune-stale` becomes a directory removal", §11's
risk table "`--prune-stale` removes a generation") as the accepted mitigation for the one cost the
directory form carries: stale generations accumulate on disk. Nothing built it. A risk table whose
mitigation does not exist is not a mitigation.

**It is never automatic.** A generation directory is the only evidence that a pass at a given set of
declared inputs ever ran; deleting it on the way past would make an accidental token change
unrecoverable *and* unnoticeable. Pruning is an explicit operator action, and it keeps the CURRENT
generation unconditionally.

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def test_prune_removes_stale_generations_and_KEEPS_the_current_one(tmp_path):
    root = tmp_path / "shards"
    old = mod.generation_dir(root, token_inputs={"box": 40.30})
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    mod.write_shard(mod.shard_path(old, "m1"), pd.DataFrame({"a": [1]}), tag="all")
    mod.write_shard(mod.shard_path(cur, "m1"), pd.DataFrame({"a": [2]}), tag="all")

    removed = mod.prune_stale_generations(root, keep=cur)

    assert removed == [old.name]
    assert not old.exists()
    assert cur.exists(), "the current generation is never pruned"
    assert mod.already_done(cur, "m1"), "and its shards survive"


def test_prune_with_only_the_current_generation_removes_NOTHING(tmp_path):
    """The other side of the band. Without it, `shutil.rmtree(root)` passes the test above."""
    root = tmp_path / "shards"
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    assert mod.prune_stale_generations(root, keep=cur) == []
    assert cur.exists()


def test_prune_REFUSES_a_directory_that_is_not_a_generation(tmp_path):
    """`shard_root` is caller-supplied and may be a directory holding other things. Only a
    16-hex-character token directory -- the shape `ruthless.fingerprint` produces -- is eligible.
    Anything else is left alone and named, rather than deleted on the operator's behalf."""
    root = tmp_path / "shards"
    cur = mod.generation_dir(root, token_inputs={"box": 40.32})
    (root / "notes").mkdir()
    (root / "notes" / "readme.txt").write_text("keep me", encoding="utf-8")

    removed = mod.prune_stale_generations(root, keep=cur)

    assert removed == []
    assert (root / "notes" / "readme.txt").is_file()


def test_prune_on_a_MISSING_root_is_a_no_op(tmp_path):
    assert mod.prune_stale_generations(tmp_path / "never-ran", keep=tmp_path / "never-ran" / "abc") == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q -k prune`
Expected: 4 failures on missing `prune_stale_generations`.

- [ ] **Step 3: Implement**

Append to `scripts/_driver.py`:

```python
#: A generation directory's name is a `ruthless.fingerprint` digest: 16 lowercase hex characters.
_GENERATION_NAME = re.compile(r"^[0-9a-f]{16}$")


def prune_stale_generations(shard_root, *, keep) -> list[str]:
    """Delete every generation directory under ``shard_root`` EXCEPT ``keep``. Returns what it removed.

    Explicit operator action only -- never called on the way past. A generation directory is the sole
    evidence that a pass at a given set of declared inputs ever ran, so pruning automatically would
    make an accidental token change both unrecoverable and unnoticeable; the whole point of the
    directory form is that stale generations stay VISIBLE.

    Only names matching a fingerprint digest are eligible. ``shard_root`` is caller-supplied and may
    hold other things, and a prune helper that recursed over whatever it found would be one typo away
    from deleting a corpus.
    """
    import shutil

    root, keep = pathlib.Path(shard_root), pathlib.Path(keep)
    if not root.is_dir():
        return []
    removed = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == keep.name or not _GENERATION_NAME.match(child.name):
            continue
        shutil.rmtree(child)
        removed.append(child.name)
    return removed
```

Add `import re` to the module imports.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: all pass.

- [ ] **Step 5: Surface it in the driver template**

`for_each` does NOT call this. Drivers that adopt `for_each` add the flag and call it themselves
after resolving the generation, so the deletion is visible at the call site rather than buried in a
shared loop:

```python
    ap.add_argument(
        "--prune-stale",
        action="store_true",
        help="Delete every shard generation under --out except the one this run resolves to. "
        "Stale generations are kept by default: a generation directory is the only record that a "
        "pass at those declared inputs ever ran.",
    )
```

and, immediately after `generation_dir(...)` in the driver:

```python
    if args.prune_stale:
        for name in prune_stale_generations(shard_root, keep=generation):
            print(f"pruned stale generation {name}", flush=True)
```

This lands with each driver's own migration commit (Tasks 11-14), not as a separate pass.

- [ ] **Step 6: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.prune_stale_generations -- the spec's --prune-stale"
```

---

## Task 2: `shard_path` — a required, separator-validated key

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def test_a_string_key_becomes_one_parquet_file(tmp_path):
    p = mod.shard_path(tmp_path, "gradientsports__10502")
    assert p.name == "gradientsports__10502.parquet"
    assert p.parent == tmp_path


def test_a_tuple_key_is_joined_with_the_separator(tmp_path):
    """Existing shards across this codebase are named `{provider}__{match_id}.parquet`. The tuple
    form preserves that byte-for-byte, so a migration is a path-prefix change and nothing more."""
    assert mod.shard_path(tmp_path, ("gradientsports", "10502")).name == "gradientsports__10502.parquet"


def test_a_component_containing_the_SEPARATOR_is_rejected_LOUDLY(tmp_path):
    """`validate_xshot_causal.py:266-268` already warns that a provider containing `__` "would
    silently mis-split". Silently is the problem: two providers sharing a game_id would overwrite
    each other's shard while the resume check reported a hit."""
    with pytest.raises(ValueError, match="__"):
        mod.shard_path(tmp_path, ("grad__sports", "10502"))


def test_an_EMPTY_component_is_rejected(tmp_path):
    """An empty component collapses two distinct keys onto one path."""
    with pytest.raises(ValueError):
        mod.shard_path(tmp_path, ("gradientsports", ""))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 4 failures, `AttributeError: module 'scripts._driver' has no attribute 'shard_path'`.

- [ ] **Step 3: Add `shard_path` to `scripts/_driver.py`**

Append after `generation_dir`:

```python
def join_key(key) -> str:
    """Validate and join a shard key.

    A bare item id is NOT enough in this corpus: providers demonstrably share ``game_id``s (see
    ``test_validate_xshot_causal_shards.py::test_cluster_key_distinguishes_providers_sharing_a_game_id``),
    so ``match_id`` alone would let two providers overwrite each other's shard while the resume
    check reported a hit. Components are validated rather than trusted -- the failure this prevents
    is silent.
    """
    parts = [key] if isinstance(key, str) else [str(k) for k in key]
    for part in parts:
        if not part:
            raise ValueError(f"empty key component in {key!r}: two distinct keys would share a path")
        if KEY_SEPARATOR in part:
            raise ValueError(
                f"key component {part!r} contains the separator {KEY_SEPARATOR!r} and would "
                f"mis-split on read; rename the component or change the key"
            )
    return KEY_SEPARATOR.join(parts)


def shard_path(generation, key) -> pathlib.Path:
    """The parquet path for ``key`` inside a generation directory."""
    return pathlib.Path(generation) / f"{join_key(key)}.parquet"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.shard_path with a validated composite key"
```

---

## Task 3: `write_shard` and `already_done` — an empty result STILL writes a shard

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

This is the load-bearing invariant of the whole cycle. Read the three citations in the test docstring before implementing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
import pandas as pd


def test_a_NONE_result_STILL_writes_a_shard(tmp_path):
    """The invariant this cycle must not break.

    `build_layer2_spells.py:131-132`: "Written even when EMPTY: an absent shard means 'not yet
    run', a present empty one means 'run, produced no spell'. Conflating them would make a resume
    silently recompute." The same rule is in `validate_xshot_causal.py:230-232` and is pinned by
    `test_validate_xshot_causal_shards.py::test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one`.

    None from `work` means ZERO ROWS, never NO SHARD.
    """
    p = mod.shard_path(tmp_path, "m1")
    mod.write_shard(p, None, tag="all")
    assert p.is_file()
    assert pd.read_parquet(p).empty


def test_an_EMPTY_frame_STILL_writes_a_shard(tmp_path):
    p = mod.shard_path(tmp_path, "m2")
    mod.write_shard(p, pd.DataFrame(), tag="all")
    assert p.is_file()


def test_already_done_distinguishes_ABSENT_from_EMPTY(tmp_path):
    """Absent means "not yet run"; present-and-empty means "ran, produced nothing". If these were
    conflated, every barren item would be recomputed on every resume, forever."""
    assert not mod.already_done(tmp_path, "m3")
    mod.write_shard(mod.shard_path(tmp_path, "m3"), None, tag="all")
    assert mod.already_done(tmp_path, "m3")


def test_write_shard_leaves_no_temp_file(tmp_path):
    """N workers rebuild from a shared directory, so a torn write is reachable."""
    mod.write_shard(mod.shard_path(tmp_path, "m4"), pd.DataFrame({"a": [1]}), tag="w0")
    assert not list(tmp_path.glob("**/*.tmp*"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 4 failures on the missing `write_shard` / `already_done` attributes.

- [ ] **Step 3: Add the two functions to `scripts/_driver.py`**

Append:

```python
def write_shard(path, frame, *, tag: str) -> None:
    """Write one item's result, atomically. A ``None`` or empty frame STILL writes a file.

    An absent shard means "not yet run"; a present empty one means "ran, produced nothing".
    Conflating them makes every barren item recompute on every resume, forever -- which is exactly
    the trap the 14-hour driver this module exists for would fall into, since it has barren items
    (``validate_xs_probe:133`` counts them). The distinction is the resume check's entire input.
    """
    import pandas as pd

    from scripts._partition import write_table_atomically

    write_table_atomically(pd.DataFrame() if frame is None else frame, pathlib.Path(path), tag=tag)


def already_done(generation, key) -> bool:
    """True when this item's shard exists in this generation -- empty shards included."""
    return shard_path(generation, key).is_file()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.write_shard/already_done -- empty result still writes"
```

---

## Task 4: `progress` — the unbuffered per-item line

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_driver.py`:

```python
def test_progress_reports_index_total_and_label(capsys):
    mod.progress("match", 2, 64, elapsed_s=12.5, note="1234 rows")
    out = capsys.readouterr().out
    assert "2/64" in out
    assert "match" in out
    assert "1234 rows" in out


def test_progress_renders_an_UNKNOWN_total_as_a_question_mark(capsys):
    """The ONLY branch `for_each` takes. It streams, so it has no total and passes `n=None` at all
    three call sites (skip, failure, success) -- while the `2/64` case above is exercised only by the
    primitives path. Without this, a typo rendering `[3/None]` ships green."""
    mod.progress("match", 3, None, elapsed_s=1.0)
    out = capsys.readouterr().out
    assert "[3/?]" in out, f"unknown total must render as '?', got: {out!r}"
    assert "None" not in out


def test_progress_is_flushed(monkeypatch):
    """A detached DGX run that buffers its output is indistinguishable from a hung one. This is the
    trick `train_ghost_gk.py` already applies locally, with the comment "so background tasks show
    progress immediately" -- centralised here so every adopter inherits it."""
    seen = {}
    monkeypatch.setattr("builtins.print", lambda *a, **kw: seen.update(kw))
    mod.progress("match", 1, 2, elapsed_s=0.0)
    assert seen.get("flush") is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 2 failures, missing attribute `progress`.

- [ ] **Step 3: Add `progress`**

Append to `scripts/_driver.py`:

```python
def progress(label: str, i: int, n: int | None, *, elapsed_s: float, note: str = "") -> None:
    """One line per item, FLUSHED.

    An unflushed detached run is indistinguishable from a hung one, which is how a 14-hour pass
    became unobservable. Flushing per item costs nothing at corpus scale.
    """
    tail = f"  {note}" if note else ""
    total = n if n is not None else "?"   # a streamed corpus has no length; see `for_each`
    print(f"  [{i}/{total}] {label}  {elapsed_s:6.1f}s{tail}", flush=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.progress -- flushed per-item output"
```

---

## Task 5: `assert_conservation` — `shards == attempted − failed`

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def test_conservation_holds_when_every_item_wrote_a_shard(tmp_path):
    keys = ["m1", "m2", "m3"]
    for k in keys:
        mod.write_shard(mod.shard_path(tmp_path, k), None, tag="all")
    mod.assert_conservation(tmp_path, keys=keys, failed=0)


def test_conservation_accounts_for_a_FAILED_item(tmp_path):
    """A failed item is the ONLY thing that can be missing, because a completed item always writes
    a shard even when empty (see write_shard). So the relation needs no third category."""
    for k in ("m1", "m2"):
        mod.write_shard(mod.shard_path(tmp_path, k), None, tag="all")
    mod.assert_conservation(tmp_path, keys=["m1", "m2", "m3"], failed=1)


def test_conservation_FAILS_when_an_attempted_item_wrote_NO_shard(tmp_path):
    """What this genuinely proves: every item this pass attempted either wrote a shard or is
    counted as failed. It catches a completed item that silently skipped its write, and off-by-one
    counting. It does NOT prove the driver has no other loop -- see the docstring on the function."""
    mod.write_shard(mod.shard_path(tmp_path, "m1"), None, tag="all")
    with pytest.raises(AssertionError, match="conservation"):
        mod.assert_conservation(tmp_path, keys=["m1", "m2", "m3"], failed=0)


def test_conservation_counts_only_THIS_generation(tmp_path):
    """A stale generation sitting beside this one must not be counted. That is the F1 double-count
    in miniature, and this assertion detects it independently of the directory layout."""
    gen_a = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    gen_b = mod.generation_dir(tmp_path, token_inputs={"v": "v2"})
    mod.write_shard(mod.shard_path(gen_a, "m1"), None, tag="all")
    mod.write_shard(mod.shard_path(gen_b, "m1"), None, tag="all")
    mod.assert_conservation(gen_b, keys=["m1"], failed=0)


def test_conservation_IGNORES_shards_outside_this_pass_slice(tmp_path):
    """THE partitioned-run case, and the reason this counts keys rather than globbing.

    N workers share one `--out` and therefore ONE generation directory: the token derives from
    `token_inputs`, which is identical across workers, and `tag` names the manifest file, not the
    directory. A worker owning a 10-match slice would otherwise glob a directory holding every
    other worker's shards too, compare that to 10, and raise -- non-deterministically, AFTER its
    multi-hour loop, and before its manifest was written, so `aggregate_manifests` would never see
    that partition. On re-run every item is skipped, the counts are unchanged, and it fires again:
    the driver could never complete.
    """
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    for k in ("mine1", "mine2"):
        mod.write_shard(mod.shard_path(gen, k), None, tag="w0")
    for k in ("theirs1", "theirs2", "theirs3"):
        mod.write_shard(mod.shard_path(gen, k), None, tag="w1")

    mod.assert_conservation(gen, keys=["mine1", "mine2"], failed=0)


def test_conservation_accepts_a_GENERATOR_of_keys(tmp_path):
    """`keys` is counted and measured, so a naive implementation consumes it twice: the count
    exhausts a generator, the length then reads 0, and a healthy pass raises AFTER its expensive
    loop. `assert_conservation` is THE documented primitive for the escape-hatch path, where
    `keys=(join_key(k) for k in items)` is the natural thing to write."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), None, tag="all")

    mod.assert_conservation(gen, keys=(k for k in ["m1"]), failed=0)
```

The matching half of this asymmetry — that `reconcile` must keep reading the **whole** generation — is pinned in Task 6, where `reconcile` exists.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 6 failures, missing attribute `assert_conservation`.

- [ ] **Step 3: Add `assert_conservation`**

Append to `scripts/_driver.py`:

```python
def assert_conservation(generation, *, keys, failed: int) -> None:
    """Every item THIS PASS attempted either wrote a shard or is counted as failed.

    Counts only the pass's OWN keys, never a directory-wide glob. N workers share one ``--out`` and
    therefore one generation directory -- the token derives from ``token_inputs``, identical across
    workers, while ``tag`` names the manifest file rather than the directory -- so a glob would
    compare one worker's slice against every worker's shards. That fires non-deterministically,
    after the expensive loop, and before the manifest is written, so the partition vanishes from
    ``aggregate_manifests``: the very "64-match artifact reported ``n_matches: 8``" defect
    ``_partition.py`` was extracted to prevent. It is also unrecoverable, because a resume skips
    everything and reaches the same comparison.

    Race-free because ``providers_for_slice`` guarantees disjoint slices, so only this worker writes
    its own keys.

    WHAT THIS DOES NOT PROVE. It does not prove the driver has no OTHER loop. A driver that calls
    ``for_each`` over something trivial and separately accumulates over the real corpus writes no
    shards for that second loop and lists none of its items here, so this passes. Catching that
    needs a fan-in check at reconcile time -- the union of all manifests' key sets against the
    directory contents -- which is a different property and a recorded follow-up. What this does
    catch: a completed item that silently skipped its write, off-by-one counting, and
    stale-generation contamination.

    Getting it exactly right before it ships matters more than shipping it early: an invariant that
    fires on healthy runs is weakened or deleted by the first person it inconveniences.
    """
    # Materialise FIRST: the counting pass and the length would otherwise consume `keys` twice, so
    # a generator -- the natural thing to write at the escape-hatch call site -- would be exhausted
    # by the count, report a length of 0, and raise on a perfectly healthy pass. After the
    # expensive loop, which is precisely the failure this docstring's last line warns about.
    keys = list(keys)
    gen = pathlib.Path(generation)
    present = sum(1 for k in keys if (gen / f"{k}.parquet").is_file())
    expected = len(keys) - failed
    if present != expected:
        raise AssertionError(
            f"conservation violated in {gen}: {present} of this pass's {len(keys)} keys have "
            f"shards, but keys-failed={len(keys)}-{failed}={expected}. A completed item did "
            f"not write its shard, or the failure count is wrong."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.assert_conservation -- shards == attempted - failed"
```

---

## Task 6: `reconcile` — combined table at `dest/`, generation recorded

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def test_reconcile_writes_the_combined_table_at_DEST_not_inside_the_generation(tmp_path):
    """`run_signoff_power --spells/--arm-values` reads `dest/<name>.parquet` today. Moving the
    combined table into the generation directory would be a Hyrum break on a documented CLI."""
    dest = tmp_path / "out"
    dest.mkdir()
    gen = mod.generation_dir(dest / "shards", token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), pd.DataFrame({"a": [1]}), tag="all")
    mod.write_shard(mod.shard_path(gen, "m2"), pd.DataFrame({"a": [2]}), tag="all")

    out = mod.reconcile(gen, dest / "combined.parquet", tag="all")

    assert (dest / "combined.parquet").is_file()
    assert len(out) == 2


def test_reconcile_skips_EMPTY_shards_without_dropping_them_from_disk(tmp_path):
    """An empty shard contributes no rows but must remain on disk -- it is the resume check's
    record that the item ran."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "m1"), pd.DataFrame({"a": [1]}), tag="all")
    mod.write_shard(mod.shard_path(gen, "barren"), None, tag="all")

    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="all")

    assert len(out) == 1
    assert mod.already_done(gen, "barren"), "the empty shard must survive reconciliation"


def test_reconcile_of_an_empty_generation_writes_no_table(tmp_path):
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="all")
    assert out.empty
    assert not (tmp_path / "combined.parquet").exists()


def test_reconcile_still_reads_the_WHOLE_generation(tmp_path):
    """The deliberate asymmetry against Task 5: conservation is PER-PASS, reconciliation is
    CORPUS-WIDE. Every worker rebuilds the combined table from ALL shards -- today's behaviour,
    which must not change. If both were scoped the same way, a partitioned run's combined table
    would hold only the last worker's slice. Pinned so nobody "fixes" the two to match."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    mod.write_shard(mod.shard_path(gen, "mine"), pd.DataFrame({"a": [1]}), tag="w0")
    mod.write_shard(mod.shard_path(gen, "theirs"), pd.DataFrame({"a": [2]}), tag="w1")

    out = mod.reconcile(gen, tmp_path / "combined.parquet", tag="w0")

    assert len(out) == 2, "reconcile must see every worker's shards, not just this pass's"


def test_manifest_fields_carry_the_generation_token(tmp_path):
    """Two generations write the same combined-table path, so the file is whichever token ran last.
    write_table_atomically makes that atomic, not ATTRIBUTABLE. Recording the token in the manifest
    makes the ambiguity visible rather than structural -- the way `commits_seen` already surfaces a
    multi-commit corpus."""
    gen = mod.generation_dir(tmp_path, token_inputs={"v": "v1"})
    fields = mod.manifest_fields(gen, attempted=3, failed=1)
    assert fields["generation"] == gen.name
    assert fields["n_attempted"] == 3
    assert fields["n_failed"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 5 failures on missing `reconcile` / `manifest_fields`.

- [ ] **Step 3: Add both functions**

Append to `scripts/_driver.py`:

```python
def reconcile(generation, combined_path, *, tag: str):
    """Combine this generation's shards and write the table to ``combined_path``.

    ``combined_path`` is deliberately OUTSIDE the generation directory: existing consumers read
    ``dest/<name>.parquet`` and moving it would break a documented CLI contract. The cost is that
    two generations write the same path, so `manifest_fields` records which one produced it.
    """
    import pandas as pd

    from scripts._partition import write_table_atomically

    shards = sorted(pathlib.Path(generation).glob("*.parquet"))
    frames = [pd.read_parquet(s) for s in shards]
    non_empty = [f for f in frames if len(f)]
    combined = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if len(combined):
        write_table_atomically(combined, pathlib.Path(combined_path), tag=tag)
    return combined


def manifest_fields(generation, *, attempted: int, failed: int) -> dict:
    """The fields every adopting driver merges into its ``manifest_<tag>.json``.

    ``generation`` is here so a reader can tell whether the combined table beside the shard root
    corresponds to the generation directory beside it.

    NOTE: `aggregate_manifests` does NOT collect arbitrary string fields -- an earlier draft of this
    docstring claimed it did, and it was measured false. Its field loop handles `run_commit` and
    `run_tree_dirty` BY NAME, sums ints, merges dicts, and a `str` matches no branch and is DROPPED.
    So `generation` reaches `manifest_<tag>.json` but would never reach the corpus artifact a reader
    actually consults. Task 6b adds the named handling that makes it surface.
    """
    return {
        "generation": pathlib.Path(generation).name,
        "n_attempted": int(attempted),
        "n_failed": int(failed),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Verify `aggregate_manifests` does not CRASH on the new string field**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q -k "partition or manifest or arm_values"
```
Expected: all pass. `_partition.aggregate_manifests` sums ints and merges dicts; a `str` matches neither branch and is **silently dropped**. That is tolerance, not support — the field survives into `manifest_<tag>.json` but never reaches the corpus artifact a reader actually consults. Task 6b makes it surface. If anything fails here, stop — the manifest contract has changed and Task 6 needs revisiting before any driver adopts it.

- [ ] **Step 6: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.reconcile/manifest_fields -- combined table stays at dest/"
```

---

## Task 6b: make `aggregate_manifests` surface the generation

**Files:**
- Modify: `scripts/_partition.py:78-160`
- Modify: `tests/scripts/test_partition.py`

Task 6 records the generation token in each worker's manifest. It stops one step short of
useful: the helper that turns N worker manifests into the ONE corpus-wide dict drops it.

**Both facts here are measured, not reasoned.** A probe wrote real manifests into a temp
directory and ran them through the real `aggregate_manifests`:

```
two workers, DIFFERENT generations  ->  'generation' in aggregate: False
today  (analysis manifest, n_matches: 0)   ->  commit_consistent: True
after  (analysis manifest, n_attempted: 64) ->  commit_consistent: False   <- false alarm re-armed
```

The first is B2: mixed-generation corpora are undetectable from the artifact. The second is B3
and it is a **regression this cycle would otherwise introduce**: the `contributed` rule at
`_partition.py:128` demotes a manifest only when it positively declares it built nothing. A
full-resume pass declares `n_matches: 0` today and abstains. If `manifest_fields` were called
with `attempted=res.attempted + res.skipped`, that same pass would declare `n_attempted: 64`,
regain its vote, and reproduce the exact `commit_consistent: false` false alarm the docstring
at `_partition.py:91-101` was written to kill. Hence `attempted=res.attempted` at every call
site — true attempts, excluding skips. Two quantities, two names; the conservation relation
belongs in `assert_conservation` (Task 5), not in the manifest.

**Honest ceiling, stated here so nobody claims more later.** Named handling buys **detection**
that a mixed-generation corpus exists. It does not buy **attribution** of the combined table:
`dest/<name>.parquet` is still whichever generation finished last, and `write_table_atomically`
makes that atomic, not attributable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_partition.py`. The file already has `import json`, a
`_write(dest, name, payload)` helper and the `mod.` prefix convention — reuse all three:

```python
def test_a_MIXED_generation_corpus_is_visible_in_the_aggregate(tmp_path):
    """MEASURED before the fix: 'generation' in aggregate -> False. A `str` matches neither the
    int-sum nor the dict-merge branch, so the field was silently dropped and two workers running
    against DIFFERENT staleness tokens produced an artifact that looked single-generation."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    _write(tmp_path, "w1", {"generation": "bbb", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["generations_seen"] == ["aaa", "bbb"]
    assert got["generation_consistent"] is False


def test_a_single_generation_corpus_reports_consistent(tmp_path):
    """The other side of the band. Without it, an implementation hard-coding False would pass."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    _write(tmp_path, "w1", {"generation": "aaa", "n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["generations_seen"] == ["aaa"]
    assert got["generation_consistent"] is True


def test_manifests_WITHOUT_a_generation_still_aggregate(tmp_path):
    """Every pre-cycle manifest on disk lacks the field. Absent must not read as inconsistent."""
    _write(tmp_path, "w0", {"n_matches": 4, "run_commit": "c1", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["generations_seen"] == []
    assert got["generation_consistent"] is True


def test_an_UNNAMED_string_field_is_REPORTED_not_silently_dropped(tmp_path):
    """The trap this seam keeps springing. A `str` matches neither the int-sum nor the dict-merge
    branch, so an unnamed string field vanishes between the per-worker manifest and the corpus
    artifact. It has now caught this cycle THREE times -- `generation` (this task),
    `run_tree_state` (Task 16b), and it would catch the next one too.

    Dropping stays the behaviour: a named case carries per-field SEMANTICS (`run_commit` is
    contributor-gated, `run_tree_dirty` is OR-ed, `generation` is a set-plus-consistency-flag), and
    a generic collector would give all of them one wrong semantic. What changes is that the drop is
    now VISIBLE in the output rather than silent."""
    _write(tmp_path, "w0", {"n_matches": 4, "some_new_field": "v1", "run_commit": "c1"})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_matches",))
    assert got["dropped_fields"] == ["some_new_field"]
    assert "some_new_field" not in got, "reported, but still not aggregated -- semantics are per-field"


def test_named_fields_are_NOT_reported_as_dropped(tmp_path):
    """Non-vacuity: `dropped_fields` must name the unhandled, not everything."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_matches": 4, "run_commit": "c1", "run_tree_dirty": False, "partition": "w0"})
    assert mod.aggregate_manifests(tmp_path, defaults=("n_matches",))["dropped_fields"] == []


def test_a_FULL_RESUME_pass_does_not_re_arm_the_commit_false_alarm(tmp_path):
    """B3, and the regression this cycle would otherwise introduce.

    MEASURED: with `n_attempted: 64` on a pass that skipped all 64, `commit_consistent` flips to
    False -- reproducing the section 3.3 entanglement false alarm that `_partition.py:91-101` exists
    to prevent. `manifest_fields` is therefore called with `attempted=res.attempted` (true attempts),
    never `res.attempted + res.skipped`. The non-contributor still appears in `commits_seen`, so its
    lineage is recorded rather than erased."""
    _write(tmp_path, "w0", {"generation": "aaa", "n_attempted": 8, "n_failed": 0, "run_commit": "AAA", "run_tree_dirty": False})
    # every item skipped -- this pass built nothing and must not vote
    _write(tmp_path, "resume", {"generation": "aaa", "n_attempted": 0, "n_failed": 0, "run_commit": "BBB", "run_tree_dirty": False})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["commit_consistent"] is True, "a pass that built nothing must not vote"
    assert got["commits_seen"] == ["AAA", "BBB"], "but its commit is still recorded"
    assert got["run_commit"] == "AAA"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_partition.py -q -k "generation or full_resume"`
Expected: the three `generation` tests fail with `KeyError: 'generations_seen'`. The
`full_resume` test **passes already** — it pins behaviour that exists today and must survive
the migration; it is a regression guard, not a red test. Note that distinction rather than
treating a green test as a failed red step.

- [ ] **Step 3: Add the named handling**

In `scripts/_partition.py`, mirror `run_commit` exactly. Add to the accumulator block:

```python
    generations: set[str] = set()
```

Extend `_meta` so the field is not mistaken for countable data (a no-op today, since a `str`
already fails both `isinstance` checks — made explicit so a future edit that widens `countable`
does not accidentally let a token vote on contribution):

```python
        _meta = ("run_commit", "run_tree_dirty", "partition", "generation")
```

Add the branch, adjacent to `run_commit`:

```python
            elif k == "generation":
                generations.add(str(v))
```

Then close the trap itself. The field loop's final `elif isinstance(v, dict)` has no `else`, so a
value matching no branch is discarded without trace — which is how `generation` vanished, and how
`run_tree_state` would vanish one task later. Add the accumulator:

```python
    dropped: set[str] = set()
```

and the `else` at the end of the field loop:

```python
            else:
                # Matched no branch: not meta, not an int to sum, not a dict to merge. Dropping is
                # CORRECT -- a named case carries per-field semantics (`run_commit` is
                # contributor-gated, `run_tree_dirty` is OR-ed, `generation` is a set plus a
                # consistency flag) and one generic collector would give all of them one wrong
                # semantic. Reporting it is what was missing: a field that needs to reach the corpus
                # artifact must be given a named case, and until then it is now VISIBLE rather than
                # absent. This seam has swallowed a field twice in one cycle.
                dropped.add(k)
```

And the two output fields, adjacent to `commits_seen`:

```python
        # The staleness token each worker ran under. The combined table at `dest/` is whichever
        # generation finished LAST -- `write_table_atomically` makes that atomic, not attributable.
        # Surfacing the set buys DETECTION of a mixed-generation corpus, which is what a reader
        # needs to know before trusting the table. Absent (every pre-cycle manifest) reads as
        # consistent, not as a violation.
        "generations_seen": sorted(generations),
        "generation_consistent": len(generations) <= 1,
        # Manifest keys that reached no accumulating branch. Empty is the healthy case; a name here
        # means a driver is writing a field that never reaches the corpus artifact.
        "dropped_fields": sorted(dropped),
```

Update the docstring's opening line — it currently says "Integer fields SUM, dict fields merge
as counters, and `partition` names are collected", which omits that a bare `str` is dropped:

```
    Integer fields SUM and dict fields merge as counters. ``partition``, ``run_commit``,
    ``run_tree_dirty`` and ``generation`` are handled BY NAME. Any other string is DROPPED --
    it matches neither branch — so a field that must reach the corpus artifact needs a named
    case here, not merely a place in the per-worker manifest.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_partition.py -q`
Expected: all pass.

- [ ] **Step 5: Verify the existing consumers still aggregate**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q
```
Expected: all pass. `build_gkdv_arm_values`, `build_layer2_spells` and `validate_xshot_causal`
write manifests with no `generation` field until Task 11/12 migrates them; Step 1's third test
pins that this stays benign.

- [ ] **Step 6: Propose the commit**

```bash
git add scripts/_partition.py tests/scripts/test_partition.py
git commit -m "feat(scripts): surface the generation token in aggregate_manifests"
```

---

## Task 7: `for_each` — the loop, with failure accounting

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def _items(n):
    return [(f"m{i}", i) for i in range(n)]


def test_for_each_writes_one_shard_per_item_and_returns_counts(tmp_path):
    res = mod.for_each(
        _items(3),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.attempted == 3
    assert res.failed == 0
    assert len(list(res.shard_dir.glob("*.parquet"))) == 3


def test_for_each_SKIPS_an_item_whose_shard_already_exists(tmp_path):
    """Resume. The second run must not re-enter `work` for an item already on disk."""
    kw = dict(key=lambda it: it[0], shard_root=tmp_path, token_inputs={"v": "v1"})
    mod.for_each(_items(2), work=lambda it: pd.DataFrame({"v": [it[1]]}), **kw)

    entered = []
    res = mod.for_each(_items(2), work=lambda it: entered.append(it) or pd.DataFrame(), **kw)

    assert entered == [], "work was re-entered for an item that already had a shard"
    assert res.skipped == 2


def test_for_each_recomputes_when_a_DECLARED_INPUT_changes(tmp_path):
    """The other side of resume: a changed token is a different generation, so nothing is reused."""
    mod.for_each(_items(2), key=lambda it: it[0], work=lambda it: pd.DataFrame({"v": [it[1]]}),
                 shard_root=tmp_path, token_inputs={"v": "v1"})
    entered = []
    mod.for_each(_items(2), key=lambda it: it[0],
                 work=lambda it: (entered.append(it), pd.DataFrame({"v": [it[1]]}))[1],
                 shard_root=tmp_path, token_inputs={"v": "v2"})
    assert len(entered) == 2


def test_one_FAILING_item_does_not_lose_the_others(tmp_path):
    """One bad item must not cost fourteen hours. The failure is recorded and the pass continues."""
    def work(it):
        if it[0] == "m1":
            raise ValueError("bad item")
        return pd.DataFrame({"v": [it[1]]})

    res = mod.for_each(_items(4), key=lambda it: it[0], work=work, shard_root=tmp_path,
                       token_inputs={"v": "v1"})

    assert res.attempted == 4
    assert res.failed == 1
    assert "m1" in res.failures
    assert len(list(res.shard_dir.glob("*.parquet"))) == 3


def test_consecutive_failures_ABORT(tmp_path):
    """Tolerating per-item failure must not turn a systematic bug into a short, clean-looking
    table. A run of consecutive failures is a systematic bug, not bad luck."""
    with pytest.raises(RuntimeError, match="consecutive"):
        mod.for_each(_items(10), key=lambda it: it[0],
                     work=lambda it: (_ for _ in ()).throw(ValueError("always")),
                     shard_root=tmp_path, token_inputs={"v": "v1"}, max_consecutive_failures=3)


def test_a_NONE_result_from_work_still_writes_its_shard(tmp_path):
    """End-to-end restatement of the invariant, through the loop rather than the primitive."""
    res = mod.for_each([("barren", 0)], key=lambda it: it[0], work=lambda it: None,
                       shard_root=tmp_path, token_inputs={"v": "v1"})
    assert mod.already_done(res.shard_dir, "barren")
    assert res.failed == 0


def test_a_NON_INJECTIVE_key_is_refused_BEFORE_any_work(tmp_path):
    """Two items mapping to one shard path is silent data loss that the conservation check would
    CERTIFY as healthy: item B finds A's shard, is counted as skipped, and is never processed --
    while `present` counts the same file once per duplicate key, so present == len(keys).

    Measured against the pre-fix implementation: 2 items in, 1 processed, conservation (2, 2), pass.

    Refused before the loop because this is a driver bug, and the point of this module is that such
    a pass fails in the first second rather than the last.
    """
    entered = []
    items = [("gradientsports", "m1"), ("skillcorner", "m1")]  # same match_id, different providers

    with pytest.raises(ValueError, match="not injective"):
        mod.for_each(
            items,
            key=lambda it: it[1],  # match_id ONLY -- the natural mistake
            work=lambda it: entered.append(it) or pd.DataFrame(),
            shard_root=tmp_path,
            token_inputs={"v": "v1"},
        )

    # STREAMING contract: the collision is caught at the SECOND item, so exactly one item's work has
    # run. Not zero -- `for_each` consumes a generator that loads a match per iteration, so it cannot
    # inspect every key before starting without materialising the corpus (see the body comment).
    # What matters is preserved: item 2 is never silently counted as `skipped` and lost.
    assert len(entered) == 1, "the collision must be caught at the colliding item, not later"


def test_the_SAME_items_are_accepted_with_a_distinguishing_key(tmp_path):
    """Non-vacuity for the guard above: it must reject the key, not the corpus. Without this half,
    a guard that rejected everything would look identical."""
    items = [("gradientsports", "m1"), ("skillcorner", "m1")]
    res = mod.for_each(
        items,
        key=lambda it: (it[0], it[1]),
        work=lambda it: pd.DataFrame({"v": [1]}),
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.attempted == 2
    assert len(list(res.shard_dir.glob("*.parquet"))) == 2


def test_for_each_does_NOT_materialise_the_corpus(tmp_path):
    """The blocker this streaming design exists for.

    `load_matches` yields (provider, match_id, actions, frames, home_team_id) and loads each match
    INSIDE the loop; its docstring says `max_per_provider` "bounds total memory ... loading all
    matches at full depth can OOM". A `list(items)` in `for_each` would hold every match's tracking
    frames alive at once and defeat resume, since nothing is skipped until everything is loaded.

    Asserted structurally: the generator must not be drained before the first `work` call."""
    produced: list[int] = []
    consumed: list[int] = []

    def source():
        for i in range(5):
            produced.append(i)
            yield (f"m{i}", i)

    def work(item):
        consumed.append(item[1])
        # At the first item, a materialising implementation would already have produced ALL five.
        assert len(produced) == len(consumed), (
            f"corpus was materialised: {len(produced)} items produced but only {len(consumed)} "
            f"consumed -- for_each must stream"
        )
        return pd.DataFrame({"v": [item[1]]})

    res = mod.for_each(source(), key=lambda it: it[0], work=work,
                       shard_root=tmp_path, token_inputs={"v": "v1"})

    # Streaming produces exactly one more item per call: [1, 2, 3, 4, 5].
    # `list(items)` produces all five before the first call: [5, 5, 5, 5, 5].
    assert produced_at_each_call == [1, 2, 3, 4, 5], (
        f"for_each must stream; items produced at each work() call was {produced_at_each_call}"
    )
    assert res.attempted == 5
    assert res.attempted == 5


def test_for_each_collects_per_item_counters(tmp_path):
    res = mod.for_each(
        _items(2),
        key=lambda it: it[0],
        work=lambda it: pd.DataFrame({"v": [it[1]]}),
        counters=lambda it, frame: {"n_rows": len(frame)},
        shard_root=tmp_path,
        token_inputs={"v": "v1"},
    )
    assert res.counters["n_rows"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 9 failures, missing attribute `for_each`.

- [ ] **Step 3: Add `CorpusPassResult` and `for_each`**

Append to `scripts/_driver.py`:

```python
import dataclasses
import time


def _require_injective(keys) -> None:
    """Refuse a key function that maps two items to the same shard path.

    `join_key` validates that a key does not MIS-SPLIT on read; it cannot see that two different
    items produce the same key. That collision is silent data loss AND it defeats the conservation
    check, which is the worse half: item B finds A's shard, `already_done` returns True, B is
    counted as `skipped` and never processed -- and then `present` counts the same file once per
    duplicate key, so `present == len(keys)` and conservation CERTIFIES the run. A guard that
    reports a run with a dropped item as healthy is worse than no guard.

    It closes a second-order break in the same relation: `failures` is a dict keyed by the shard
    key, so two failures on colliding keys would collapse to one entry and `len(failures)` would
    UNDER-count -- making conservation raise spuriously. Enforcing injectivity up front makes both
    directions unreachable rather than guarding each.

    Raised BEFORE any work: this is a programming error in the driver, and the entire point of this
    module is that a 14-hour pass fails in the first second rather than the last.
    """
    import collections

    dupes = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
    if dupes:
        raise ValueError(
            f"key() is not injective over this corpus: {dupes}. Two items map to the same shard "
            f"path, so the second is skipped as 'already done' and silently lost. Include the "
            f"provider (or another distinguishing component) in the key -- providers in this "
            f"corpus demonstrably share match ids."
        )


@dataclasses.dataclass(frozen=True)
class CorpusPassResult:
    """What a pass did. ``shard_dir`` is the generation directory callers glob and reconcile."""

    shard_dir: pathlib.Path
    attempted: int
    skipped: int
    failed: int
    failures: dict
    counters: dict

    def manifest(self) -> dict:
        return manifest_fields(self.shard_dir, attempted=self.attempted, failed=self.failed)


def for_each(
    items,
    *,
    key,
    work,
    shard_root,
    token_inputs: Mapping[str, object],
    token_reason: str | None = None,
    counters=None,
    tag: str = "all",
    label: str = "item",
    max_consecutive_failures: int = 3,
) -> CorpusPassResult:
    """Walk ``items``, persisting each result so a crash resumes instead of restarting.

    ``work(item)`` returns ONE long-form DataFrame, or ``None`` meaning zero rows -- which still
    writes a shard. Per-item scalars go through ``counters(item, frame)`` into the manifest, where
    ``aggregate_manifests`` already sums ints and merges dict counters; that is why the contract is
    a tidy frame plus counters rather than a dict of frames, which no manifest could absorb.

    A failing item is recorded and skipped, because one bad item must not cost a whole corpus pass.
    ``max_consecutive_failures`` in a row aborts, because that is a systematic bug rather than bad
    luck, and a short clean-looking table is worse than a crash.
    """
    generation = generation_dir(shard_root, token_inputs=token_inputs, token_reason=token_reason)
    # STREAMED, never `list(items)`. `load_matches` is an Iterator that downloads and parses a match
    # -- actions AND a full tracking DataFrame -- inside the loop before yielding, and its own
    # docstring says `max_per_provider` "bounds total memory ... loading all matches at full depth
    # can OOM". Materialising the corpus would hold ~80 matches' frames alive at once, defeat resume
    # (nothing is skipped until everything has been downloaded), and invert this cycle's own thesis:
    # it indicts 14 drivers for holding every RESULT in memory, and inputs are far larger.
    own_keys: list[str] = []  # accumulated as we go -- 80 strings, not 80 tracking frames
    seen: set[str] = set()
    attempted = skipped = 0
    failures: dict = {}
    totals: dict = {}
    run = 0

    for i, item in enumerate(items, start=1):
        k = join_key(key(item))
        if k in seen:
            raise ValueError(
                f"key() is not injective over this corpus: {k!r} appeared twice. Two items map to "
                f"the same shard path, so the second would be skipped as 'already done' and silently "
                f"lost. Include the provider (or another distinguishing component) in the key -- "
                f"providers in this corpus demonstrably share match ids."
            )
        seen.add(k)
        own_keys.append(k)

        if already_done(generation, k):
            skipped += 1
            progress(f"{label} {k}", i, None, elapsed_s=0.0, note="skip (shard exists)")
            continue

        attempted += 1
        t0 = time.perf_counter()
        try:
            frame = work(item)
        except Exception as exc:  # noqa: BLE001 -- recorded and counted, never swallowed silently
            failures[k] = f"{type(exc).__name__}: {exc}"
            run += 1
            progress(f"{label} {k}", i, None, elapsed_s=time.perf_counter() - t0, note=f"FAILED {exc}")
            if run >= max_consecutive_failures:
                raise RuntimeError(
                    f"aborting after {run} consecutive failures (last: {k}). This is a systematic "
                    f"problem, not a bad item; fix it rather than resuming past it."
                ) from exc
            continue

        run = 0
        write_shard(shard_path(generation, k), frame, tag=tag)
        if counters is not None:
            import pandas as pd

            for ck, cv in counters(item, pd.DataFrame() if frame is None else frame).items():
                totals[ck] = totals.get(ck, 0) + cv
        progress(f"{label} {k}", i, None, elapsed_s=time.perf_counter() - t0,
                 note=f"{0 if frame is None else len(frame)} rows")

    assert_conservation(generation, keys=own_keys, failed=len(failures))
    return CorpusPassResult(generation, attempted, skipped, len(failures), failures, totals)
```

`own_keys` accumulates as the stream is consumed, so the relation is still `present == len(own_keys) - failed` — a skipped item counts as present, because its shard is on disk from an earlier run. Passing keys rather than a count is what makes a partitioned run work: the check reads only the slice this worker owns, inside a directory every worker shares.

**What streaming costs, stated rather than glossed.** The injectivity check now fires at the COLLIDING item rather than before any work, so a bad `key` costs one item's compute instead of failing in the first second. What it still prevents is the failure that matters: the second item is never silently counted as `skipped`, so conservation cannot certify a run that dropped data. `_require_injective`'s pre-loop form is retained for the primitives path and **called there** (Task 13's snippet and Task 14's template both invoke it), where the caller enumerates keys into a list anyway and can afford the up-front check. It is not dead code: without that call the escape hatch would have no injectivity protection at all, which is worse than `for_each`'s position, not better -- there, a collision is silent AND self-certifying, because `assert_conservation` counts one shared file once per duplicate key and reports the run healthy.

**And `total` is now unknown**, because a generator has no length. `progress` takes `n: int | None` and renders `[3/?]` when it is `None`; a driver that genuinely knows its total (the primitives path, or a materialised list) passes it. Counting the corpus just to render a denominator would reintroduce exactly the materialisation this avoids.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Run the full gate**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
```
Expected: clean. `dataclasses` and `time` are imported mid-file by these steps — move both to the top import block alongside `hashlib`/`json`/`pathlib` so `ruff` (E402) passes.

- [ ] **Step 6: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.for_each -- the corpus loop with failure accounting"
```

---

## Task 8: The cohort cache

**Files:**
- Modify: `scripts/_driver.py`
- Modify: `tests/scripts/test_driver.py`

Built here, at step 1 of the rollout, because Task 12 migrates `calibrate_xt_bandwidth` onto it. A helper first shipping as a side effect of a driver's commit would invert this plan's build-against-a-reference ordering.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_driver.py`:

```python
def test_cohort_cache_is_a_pure_passthrough_when_no_path_is_given(tmp_path):
    """Absent the flag, behaviour must be byte-identical to today."""
    calls = []
    got = mod.cohort_cache(None, build=lambda: (calls.append(1), pd.DataFrame({"a": [1]}))[1])
    assert len(got) == 1
    assert calls == [1]


def test_cohort_cache_builds_once_then_reuses(tmp_path):
    calls = []
    path = tmp_path / "cohort.parquet"

    def build():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})

    a = mod.cohort_cache(path, build=build)
    b = mod.cohort_cache(path, build=build)
    assert calls == [1], "the cohort was re-fetched despite a populated cache"
    assert len(a) == len(b) == 2


def test_cohort_cache_fails_FAST_when_no_parquet_engine_is_available(tmp_path, monkeypatch):
    """Kept from `calibrate_xt_bandwidth.py:225-239`: pandas only surfaces "Unable to find a usable
    engine" at write time, i.e. AFTER the multi-minute cohort load. Check before paying for it."""
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="parquet engine"):
        mod.cohort_cache(tmp_path / "c.parquet", build=lambda: pd.DataFrame())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: 3 failures, missing attribute `cohort_cache`.

- [ ] **Step 3: Add `cohort_cache`**

Add `import importlib.util` to the top import block, then append:

```python
def cohort_cache(path, *, build):
    """Fetch-once / reuse for a whole-cohort query, opt-in via an explicitly named path.

    Deliberately NOT automatic and deliberately NOT inside the loader. A query result has no token
    this module can compute without running the query, and the marts behind these cohorts
    re-materialize regularly -- so an automatic cache would silently serve a stale cohort, which is
    a plausible number from a computation that did not happen. A path the operator names cannot be
    reused by accident.

    ``path=None`` is a pure passthrough, so a caller that does not opt in behaves exactly as before.
    """
    import pandas as pd

    if path is None:
        return build()
    path = pathlib.Path(path)
    if not path.exists() and not (
        importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
    ):
        # Before the multi-minute load, not after it: pandas only raises at to_parquet time.
        raise ValueError("--cohort-cache requires a parquet engine: pip install pyarrow")
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver.py -q`
Expected: **every test in the file passes, and the N tests this task just added are among them**. Deliberately not an absolute count: `tests/scripts/test_driver.py` is appended to by eight tasks, three of which were inserted after the totals were first written, so a fixed number goes stale on every insertion and an executor cannot tell a stale expectation from a real regression. Run with `-rs` and compare against the previous task's run.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_driver.py tests/scripts/test_driver.py
git commit -m "feat(scripts): add _driver.cohort_cache -- opt-in, explicitly named"
```

---

## Task 9: Rewrite the adoption gate

**Files:**
- Rewrite: `tests/scripts/test_corpus_driver_resilience.py`

The drafted version of this file scores capability **tokens** (`"shard" in src`, `"flush=True" in src`) and certifies exactly five drivers, three of which are accumulate-then-write. Delete it rather than keep it alongside — a gate wrong in both directions is misleading, not merely weak.

- [ ] **Step 1: Replace the file wholesale**

```python
"""Every driver that walks a corpus must adopt the shared `_driver` seam.

WHY THIS EXISTS. `scripts/validate_xs_probe.py` walked ~80 matches for 14 hours, held every result
in memory, wrote once at the end, and printed nothing. A crash at hour 13 lost the run. That was not
a missing convention: four partial mechanisms already existed, and three fully-resumable drivers
predate that script by weeks. Prose plus exemplars was not enough -- the same finding as
`test_provenance_wiring.py`.

WHAT THIS CHECKS. The population is DERIVED, so a new corpus driver is enrolled the moment it is
written. The verdict is ADOPTION: does the driver call `for_each`, or a registered primitive from
`scripts/_driver.py`?

WHAT IT CANNOT CHECK. Adoption is not correctness -- a driver can call `for_each` for something
trivial and still accumulate over the real corpus in a second loop. **That evasion is NOT caught**,
by this gate or by the runtime invariant: the second loop writes no shards and lists no keys, so
`_driver.assert_conservation` never sees it. An earlier draft claimed otherwise; the claim was
wrong. What `assert_conservation` does prove is narrower and still worth having -- every item a
pass ATTEMPTED either wrote a shard or is counted as failed. Covering the second-loop evasion needs
a fan-in check (the union of all manifests' key sets against the directory contents) and is a
recorded follow-up, not something this gate does today.

A PREVIOUS VERSION OF THIS GATE WAS WRONG IN BOTH DIRECTIONS. It scored capability tokens
(`"shard" in src`, `".is_file()" in src`, `"flush=True" in src`) and certified five drivers, three
of which are accumulate-then-write, while pinning `build_gkdv_arm_values` -- genuinely resumable --
as debt for lacking `flush=True`. Substring and keyword tests over source are not evidence of
behaviour. Do not reintroduce one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

#: Calling any of these means the driver pulls a corpus.
_CORPUS_CALLS = {"load_matches", "select_match_ids", "load_xtgk_cohort", "load_retention_cohort"}
#: A corpus-shaped CLI surface. Paired with a per-item loop it means the same thing.
_CORPUS_ARGS = {"--data-dir", "--match-ids-json", "--max-per-provider", "--providers"}
#: The public surface of `scripts/_driver.py`. Calling any of them is adoption.
_DRIVER_API = {
    "for_each",
    "generation_dir",
    "shard_path",
    "join_key",
    "write_shard",
    "already_done",
    "progress",
    "assert_conservation",
    "reconcile",
    "manifest_fields",
    "cohort_cache",
}
#: The one primitive an escape-hatch driver must call. `for_each` calls it internally.
_CONSERVATION = "assert_conservation"
_INJECTIVE = "_require_injective"


def _called_names(tree: ast.AST) -> set[str]:
    return {
        (getattr(n.func, "id", "") or getattr(n.func, "attr", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("--")
    }


def _accumulates(node: ast.AST) -> bool:
    """Append/extend/update/add, or a subscript assignment.

    The `.append`-only version of this predicate misclassified two drivers as having no per-item
    state -- `.extend` and `out[match_id] = value` are the same defect.
    """
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Subscript) for t in targets):
                return True
        if isinstance(n, ast.Call):
            if (getattr(n.func, "id", "") or getattr(n.func, "attr", "")) in {
                "append", "extend", "update", "add"
            }:
                return True
    return False


def _has_per_item_loop(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if not isinstance(n, ast.For):
            continue
        if isinstance(n.target, ast.Tuple) and len(n.target.elts) >= 3:
            return True
        if any(k in ast.dump(n.iter) for k in ("match", "game", "provider", "cohort")):
            return True
    return False


def _is_corpus_driver(tree: ast.AST) -> bool:
    if _called_names(tree) & _CORPUS_CALLS:
        return True
    return bool(_string_literals(tree) & _CORPUS_ARGS) and _has_per_item_loop(tree)


def _adopts(tree: ast.AST) -> bool:
    return bool(_called_names(tree) & _DRIVER_API)


def _uses_for_each(tree: ast.AST) -> bool:
    return "for_each" in _called_names(tree)


def _population() -> dict[str, ast.AST]:
    out = {}
    for p in sorted(_SCRIPTS.glob("*.py")):
        if p.name.startswith("_"):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if _is_corpus_driver(tree):
            out[p.stem] = tree
    return out


#: Drivers not yet migrated, each with the reason. Asserted EXACTLY, both ways (see the test below):
#: a new offender cannot join silently and a migrated one must be removed. Emptied by Task 16.
_NOT_YET_MIGRATED: dict[str, str] = {}


@pytest.mark.parametrize("name", sorted(_population()))
def test_corpus_driver_adopts_the_shared_seam(name):
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    assert _adopts(_population()[name]), (
        f"{name}.py walks a corpus but calls nothing from scripts/_driver.py. A driver delegated to "
        f"a remote box must persist each item so a crash resumes, skip work already done, and print "
        f"progress. Use `for_each`; if its loops genuinely cannot invert, use the primitives and "
        f"record why."
    )


@pytest.mark.parametrize("name", sorted(_population()))
def test_an_ESCAPE_HATCH_driver_still_asserts_conservation(name):
    """`for_each` calls `assert_conservation` internally. A driver on the primitives path is
    otherwise gated statically only -- and step 4 of the rollout deliberately puts the hardest
    multi-loop driver on exactly that path, so this is where it would go unchecked."""
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    tree = _population()[name]
    if _uses_for_each(tree) or not _adopts(tree):
        pytest.skip("uses for_each (which asserts internally) or is covered by the adoption test")
    assert _CONSERVATION in _called_names(tree), (
        f"{name}.py uses the primitives directly but never calls {_CONSERVATION}. Then neither the "
        f"static gate nor the runtime invariant covers it."
    )


@pytest.mark.parametrize("name", sorted(_population()))
def test_an_ESCAPE_HATCH_driver_still_checks_key_injectivity(name):
    """`assert_conservation` alone is SATISFIABLE BY A LOSSY RUN, so the conservation gate above is
    not sufficient on this path.

    A colliding key makes `already_done` return True for the duplicate, so the second item is
    skipped and lost -- and `present` counts the single shared shard once per duplicate key, giving
    `present == len(own_keys)`. Conservation then certifies the run as healthy. `for_each` closes
    this with an inline `seen` check it grew when it went streaming; the primitives path has to call
    `_require_injective` itself, and `items` is materialised there so the up-front form is cheap."""
    if name in _NOT_YET_MIGRATED:
        pytest.skip(f"pending migration: {_NOT_YET_MIGRATED[name]}")
    tree = _population()[name]
    if _uses_for_each(tree) or not _adopts(tree):
        pytest.skip("uses for_each (which checks inline) or is covered by the adoption test")
    assert _INJECTIVE in _called_names(tree), (
        f"{name}.py uses the primitives directly but never calls {_INJECTIVE}. A non-injective key "
        f"would then be silent AND self-certifying: the duplicate is skipped as 'already done' and "
        f"{_CONSERVATION} still passes."
    )


def test_the_pending_list_is_EXACT():
    """Fails BOTH ways -- the only thing that stops a debt list becoming a dumping ground."""
    actual = {n for n, tree in _population().items() if not _adopts(tree)}
    assert actual == set(_NOT_YET_MIGRATED), (
        f"newly unmigrated: {sorted(actual - set(_NOT_YET_MIGRATED))}; "
        f"now migrated, remove from the list: {sorted(set(_NOT_YET_MIGRATED) - actual)}"
    )


def test_the_population_is_not_silently_empty():
    """A derived population that resolved to nothing makes every case above vacuous."""
    pop = _population()
    assert len(pop) >= 20, f"detection collapsed: only {len(pop)} drivers found"
    for expected in ("validate_xs_probe", "build_layer2_spells", "train_ghost_gk"):
        assert expected in pop, f"{expected} no longer detected as a corpus driver"


def test_detection_catches_a_planted_UNMIGRATED_driver():
    """Non-vacuity for the detector: a naive corpus loop must be in-population and NOT adopting."""
    planted = ast.parse(
        "def main():\n"
        "    for provider, match_id, actions, frames, home in load_matches(providers=['x']):\n"
        "        results.append(expensive(frames))\n"
        "    write(results)\n"
    )
    assert _is_corpus_driver(planted)
    assert not _adopts(planted)


def test_detection_catches_a_planted_MIGRATED_driver():
    """The other side: adoption must be recognised, or the gate can never go green."""
    planted = ast.parse(
        "from scripts._driver import for_each\n"
        "def main():\n"
        "    for_each(load_matches(providers=['x']), key=k, work=w, shard_root=d, token_inputs={'v': 1})\n"
    )
    assert _is_corpus_driver(planted)
    assert _adopts(planted)


def test_detection_is_not_indiscriminate():
    trivial = ast.parse("import json\n\ndef main():\n    print(json.dumps({'a': 1}))\n")
    assert not _is_corpus_driver(trivial)


def test_the_accumulation_predicate_sees_more_than_append():
    """Pins the correction that found two misclassified drivers."""
    assert _accumulates(ast.parse("for m in matches:\n    out.append(m)"))
    assert _accumulates(ast.parse("for m in matches:\n    out.extend(m)"))
    assert _accumulates(ast.parse("for m in matches:\n    out[m] = 1"))
    assert not _accumulates(ast.parse("for m in matches:\n    print(m)"))
```

- [ ] **Step 2: Run the gate and confirm it is RED across the population**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_corpus_driver_resilience.py -q`

Expected: `test_the_pending_list_is_EXACT` FAILS, listing ~21 newly unmigrated drivers, and ~21 parametrized `test_corpus_driver_adopts_the_shared_seam` cases FAIL. That red list is the work of Tasks 11–16.

- [ ] **Step 3: Populate `_NOT_YET_MIGRATED` from the failure output**

Copy the `newly unmigrated:` list from Step 2 into `_NOT_YET_MIGRATED`, one entry per driver, each with a one-line reason naming its rollout step. For example:

```python
_NOT_YET_MIGRATED: dict[str, str] = {
    "build_gkdv_arm_values": "step 3 -- resumable, migrates onto for_each",
    "build_layer2_spells": "step 3 -- resumable, the reference migration",
    "calibrate_tracking_defaults": "step 7 -- 3 corpus loops, primitives path",
    "calibrate_xt_bandwidth": "step 4 -- 4 corpus loops, the primitives-path proof",
    "derive_opengoal_range": "step 7 -- accumulate-then-write",
    "measure_cover_shadow_argmax_agreement": "step 7 -- another session's file (last touched 4.67.0), goes last",
    "run_signoff_power": "step 7 -- accumulate-then-write",
    "train_ghost_gk": "step 5 -- own _feature_cache + cache_token, folds into token_inputs",
    "train_gk_completion": "step 7 -- accumulate-then-write",
    "train_gk_retention": "step 8 -- Shape B, cohort cache only",
    "train_xcross_attempt": "step 5 -- uses _cache.py, folds into token_inputs",
    "train_xshot_occurrence": "step 5 -- uses _cache.py, folds into token_inputs",
    "tune_structural_pass_sigma": "step 7 -- accumulate-then-write",
    "validate_shot_goalmouth_sb": "step 7 -- accumulate-then-write",
    "validate_xcross_causal": "step 7 -- accumulate-then-write",
    "validate_xs_probe": "step 6 -- THE motivating driver: 14h, unresumable, silent",
    "validate_xshot_causal": "step 3 -- resumable, migrates onto for_each",
    "validate_xtgk_possession_value": "step 7 -- accumulate-then-write",
    "validate_xtgk_v2": "step 8 -- Shape B, cohort cache only",
    "xtgk_v2_kappa_sweep": "step 8 -- Shape B, cohort cache only",
    "xtgk_v2_keeper_discrimination": "step 8 -- Shape B, cohort cache only",
}
```

If Step 2's list differs from this one, **use Step 2's** — the derived population is authoritative and the repo may have moved.

- [ ] **Step 4: Run the gate again**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_corpus_driver_resilience.py -q`
Expected: all pass, with ~21 skips. Every skip is a task below; the list empties as they land.

- [ ] **Step 5: Propose the commit**

```bash
git add tests/scripts/test_corpus_driver_resilience.py
git commit -m "test(scripts): replace the keyword resilience gate with an adoption gate"
```

---

## Task 10: The resume oracle that does not exist yet

**Files:**
- Create: `tests/scripts/test_driver_resume_oracle.py`

No existing test re-runs any driver, so nothing today would notice a migration silently converting a resumable driver into a full-recompute one. Build the oracle **before** touching the drivers.

- [ ] **Step 1: Write the oracle, red-first**

Create `tests/scripts/test_driver_resume_oracle.py`:

```python
"""The double-invocation oracle: run a driver twice, prove the second run does no work.

WHY THIS EXISTS. Before this file, no test on any of the three resumable drivers exercised the
resume branch. `test_build_layer2_spells.py:78-82` runs `main()` once and asserts `shard.is_file()`,
never re-running to reach `if shard.is_file(): continue`; `test_build_gkdv_arm_values.py`'s 15 tests
never touch the shard loop; and `test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one` tests the
resume check's PRECONDITION rather than the skip. So the safety net covered writes and aggregation
and was blind to resume -- while the migration changes shard paths.

The empty-shard round trip is included deliberately. Losing it is invisible except as a slow resume,
and it is currently pinned for only one of the three drivers.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import scripts.build_layer2_spells as layer2


@pytest.fixture()
def stub_layer2(monkeypatch):
    """Stub the corpus loader and the expensive builders; count entries into the real work."""
    entered: list[str] = []

    import scripts._loader_pining as loader
    import silly_kicks.causal as causal
    import silly_kicks.causal._confounders as conf

    def _load(**_kw):
        return iter(
            [
                ("gradientsports", "m1", object(), object(), "5"),
                ("gradientsports", "barren", object(), object(), "5"),
            ]
        )

    def _build(frames, actions, **_kw):
        entered.append("build")
        # "barren" is the second item; return an EMPTY frame for it.
        return pd.DataFrame() if len(entered) == 2 else pd.DataFrame({"Z": [0, 1], "r": [1.0, 2.0]})

    monkeypatch.setattr(loader, "load_matches", _load)
    monkeypatch.setattr(causal, "build_opportunities", _build)
    monkeypatch.setattr(causal, "layer2_config", lambda *a, **k: object())
    monkeypatch.setattr(conf, "join_layer2_confounders", lambda sp, **k: sp)
    return entered


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_layer2_spells.py", "--out", str(tmp_path), "--allow-dirty"])
    layer2.main()


def test_layer2_second_run_does_NO_work(tmp_path, monkeypatch, stub_layer2):
    _run(tmp_path, monkeypatch)
    first = len(stub_layer2)
    assert first == 2, "the first run should have built both matches"

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) == first, "the second run re-entered the expensive builder"


def test_layer2_second_run_produces_an_IDENTICAL_table(tmp_path, monkeypatch, stub_layer2):
    _run(tmp_path, monkeypatch)
    before = (tmp_path / "layer2_spells.parquet").read_bytes()
    _run(tmp_path, monkeypatch)
    assert (tmp_path / "layer2_spells.parquet").read_bytes() == before


def test_layer2_a_BARREN_match_is_not_recomputed(tmp_path, monkeypatch, stub_layer2):
    """The empty-shard round trip. A match producing zero rows must leave a shard behind and be
    skipped on re-run -- otherwise every barren match recomputes forever, which is the exact cost
    this cycle exists to remove."""
    _run(tmp_path, monkeypatch)
    shards = list(tmp_path.rglob("*barren*.parquet"))
    assert shards, "the barren match left no shard, so a resume will recompute it"
    assert pd.read_parquet(shards[0]).empty

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) == 2, "the barren match was recomputed on resume"


def test_the_oracle_would_CATCH_a_broken_resume(tmp_path, monkeypatch, stub_layer2):
    """Non-vacuity: with the resume check disabled, the oracle must fail. Without this, a green
    oracle is indistinguishable from one that never exercised resume at all."""
    _run(tmp_path, monkeypatch)
    first = len(stub_layer2)
    for shard in tmp_path.rglob("*.parquet"):
        if shard.parent != tmp_path:
            shard.unlink()  # simulate a migration that lost resume

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) > first, "the oracle cannot detect a lost resume -- it is vacuous"
```

- [ ] **Step 2: Run the oracle against the UNMIGRATED driver**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver_resume_oracle.py -q`

Expected: **all five pass** against the current `build_layer2_spells.py`, which already resumes correctly. That is the point — the oracle characterises today's correct behaviour so Task 11's migration must preserve it. If any fail now, stop: the pre-migration behaviour is not what the spec assumed.

- [ ] **Step 3: Propose the commit**

```bash
git add tests/scripts/test_driver_resume_oracle.py
git commit -m "test(scripts): add the double-invocation resume oracle (incl. empty-shard round trip)"
```

---

## Task 11: Migrate `build_layer2_spells` — the reference migration

**Files:**
- Modify: `scripts/build_layer2_spells.py:100-153`
- Test: `tests/scripts/test_driver_resume_oracle.py` (must stay green, unchanged)

This is the smallest resumable driver and the template for Task 12. Do it first and read the diff carefully.

- [ ] **Step 1: Migrate the loop**

In `scripts/build_layer2_spells.py`, replace the block from `tag = worker_tag(...)` (`:101`) through the `manifest_{tag}.json` write (`:148`) with:

```python
    from scripts._driver import for_each, manifest_fields, reconcile
    from scripts._partition import worker_tag

    tag = worker_tag(args.match_ids_json)
    dest = Path(args.out)

    def _work(item):
        _provider, _match_id, actions, frames, home_team_id = item
        sp = build_opportunities(
            frames, actions, home_team_id=home_team_id, model_metadata={}, config=layer2_config({})
        )
        if len(sp):
            sp = join_layer2_confounders(sp, frames=frames, actions=actions, home_team_id=home_team_id)
            sp = sp.copy()
            sp["provider"] = str(_provider)
            sp["match_id"] = str(_match_id)
        return sp

    def _counters(_item, frame):
        return {
            "n_matches": 1,
            "n_spells": len(frame),
            "n_treated": int(frame["Z"].sum()) if len(frame) else 0,
        }

    res = for_each(
        load_matches(
            providers=providers,
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=_counters,
        shard_root=dest / "shards",
        # Layer 2 spells are produced by the opportunity builder and its config. `matching.py` is
        # NOT declared: it runs in the downstream analysis, which re-reads these shards on every
        # invocation. Declare what determines the CONTENT, not what consumes it.
        token_inputs={
            "layer2_config": "v1",
            "build_opportunities": "v1",
            "join_layer2_confounders": "v1",
        },
        tag=tag,
        label="match",
    )

    combined = reconcile(res.shard_dir, dest / "layer2_spells.parquet", tag=tag)
    (dest / f"manifest_{tag}.json").write_text(
        json.dumps(
            {
                **res.counters,
                **manifest_fields(res.shard_dir, attempted=res.attempted, failed=res.failed),
                "run_commit": prov["commit"],
                "run_tree_dirty": prov["dirty"],
                "partition": tag,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
```

Then delete the now-unused `write_table_atomically` and `pd` imports if nothing else in the file uses them (`pd` is still used by nothing after `reconcile` takes over — check with `ruff check`, which reports F401).

Note `_counters` returns `n_matches: 1` per item, so `aggregate_manifests` sums it exactly as the hand-rolled `totals` dict did.

- [ ] **Step 2: Run the oracle**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver_resume_oracle.py -q`
Expected: `5 passed`. If `test_layer2_second_run_does_NO_work` fails, resume is broken — do not proceed.

- [ ] **Step 3: Run the driver's own tests**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_build_layer2_spells.py -q`
Expected: **one failure**, in `test_main_walks_a_match_end_to_end_and_writes_its_shard` — it asserts `tmp_path/"shards"/"gradientsports__m1.parquet"`, and that path is now **inside a generation directory**. That failure is expected and correct; it is the migration being visible. Update the assertion to:

```python
    shards = list((tmp_path / "shards").rglob("gradientsports__m1.parquet"))
    assert shards, "the per-match shard was never written"
    assert (tmp_path / "layer2_spells.parquet").is_file()
    assert not list(tmp_path.glob("**/*.tmp*")), "atomic temp file left behind"
```

`rglob` rather than a hard-coded token: the token is derived from the declared inputs, and pinning its value in a test would make every future declaration change a test edit.

Re-run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_build_layer2_spells.py -q`
Expected: `5 passed`.

- [ ] **Step 4: Confirm the adoption gate now passes for this driver**

Remove `"build_layer2_spells"` from `_NOT_YET_MIGRATED` in `tests/scripts/test_corpus_driver_resilience.py`, then run:

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/test_corpus_driver_resilience.py -q
```
Expected: passes with one fewer skip.

- [ ] **Step 5: Run the full gate and propose the commit**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --tb=short -q
```

```bash
git add scripts/build_layer2_spells.py tests/scripts/test_build_layer2_spells.py tests/scripts/test_corpus_driver_resilience.py
git commit -m "refactor(scripts): migrate build_layer2_spells onto _driver.for_each"
```

---

## Task 12: Migrate `build_gkdv_arm_values` and `validate_xshot_causal`

**Files:**
- Modify: `scripts/build_gkdv_arm_values.py:160-260`
- Modify: `scripts/validate_xshot_causal.py:236-335`
- Modify: `tests/scripts/test_driver_resume_oracle.py` (extend to both)

- [ ] **Step 1: Extend the oracle to both drivers before migrating**

Append to `tests/scripts/test_driver_resume_oracle.py` a fixture and three tests per driver mirroring the layer2 set exactly — `second_run_does_NO_work`, `second_run_produces_an_IDENTICAL_table`, `a_BARREN_item_is_not_recomputed`. Stub each driver's loader the same way (`monkeypatch.setattr(loader, "load_matches", ...)`) and its expensive per-item call so the fixture can count entries. For `validate_xshot_causal`, drive it through `mod.run(tmp_path, ["gradientsports"], 0.6, 0, provenance={"commit": "abc", "dirty": False}, build_only=True)` — the shape its own test already uses at `tests/scripts/test_validate_xshot_causal_shards.py:169` — rather than `main()`, so the analysis step is not exercised.

- [ ] **Step 2: Run the extended oracle against the UNMIGRATED drivers**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_driver_resume_oracle.py -q`
Expected: all pass. Both already resume; the oracle is characterising that before the change.

- [ ] **Step 3: Migrate `build_gkdv_arm_values`**

Apply the Task 11 pattern. Its per-item body starts at `:167` with the `shard`/`if shard.is_file()` guard — that guard and the manual `write_table_atomically` both disappear into `for_each`. Its `token_inputs` declare what determines an arm value: the ghost model variant, the pitch-control method, and the carrier parameters. Its `key` is `lambda item: (str(item[0]), str(item[1]))`, matching the existing `f"{_provider}__{match_id}"`.

- [ ] **Step 4: Migrate `validate_xshot_causal`**

Same pattern in `build_shards`. Its `token_inputs` are `causal/opportunities.py` + `shot_arm_config` + the model metadata — **not** `causal/matching.py`, which runs in `_entanglement_analysis` at `:127` and re-reads the shards every invocation. `analyze_shards` at `:288` and the `n_shards` count at `:331` must now glob the generation directory returned by `for_each` rather than `Path(out) / "shards"`; thread `res.shard_dir` through.

- [ ] **Step 5: Run the oracle, both drivers' tests, and the gate**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q
```
Expected: all pass. Update any test asserting a flat shard path to `rglob`, as in Task 11 Step 3.

- [ ] **Step 6: Remove both from `_NOT_YET_MIGRATED` and propose two commits**

```bash
git add scripts/build_gkdv_arm_values.py tests/scripts/
git commit -m "refactor(scripts): migrate build_gkdv_arm_values onto _driver.for_each"
git add scripts/validate_xshot_causal.py tests/scripts/
git commit -m "refactor(scripts): migrate validate_xshot_causal onto _driver.for_each"
```

---

## Task 13: `calibrate_xt_bandwidth` — the primitives-path proof

**Files:**
- Modify: `scripts/calibrate_xt_bandwidth.py:225-239` and its four corpus loops

At four corpus loops this is the largest "`for_each` cannot express this" risk in the population, deliberately met early. **If the primitives path turns out inadequate, stop and report it as a design finding about the architecture** — that is this task's purpose, and it is a success condition, not a failure.

- [ ] **Step 1: Migrate `--corpus-cache` onto the shared helper**

Replace the body at `:225-239` with:

```python
    from scripts._driver import cohort_cache

    df = cohort_cache(getattr(args, "corpus_cache", None), build=lambda: _assemble_corpus(args))
```

The fail-fast parquet-engine pre-check moves into `cohort_cache` (Task 8) and is deleted here.

- [ ] **Step 2: Decide, per loop, `for_each` or primitives**

For each of the four corpus loops, apply this rule: a loop that walks items and produces one persistable result per item uses `for_each`. A loop that is a sweep over *parameters* rather than corpus items is not a corpus pass and stays as it is — the gate's population test only requires that the driver adopt the seam somewhere, not that every loop invert.

- [ ] **Step 3: If any genuine per-item loop cannot use `for_each`, use the primitives and call `assert_conservation`**

```python
    from scripts._driver import (
        already_done, assert_conservation, generation_dir, join_key, shard_path, write_shard,
    )

    gen = generation_dir(dest / "shards", token_inputs={...})
    # THIS pass's keys, not a directory glob: N workers share one generation directory, so a glob
    # would compare this slice against every worker's shards. See assert_conservation's docstring.
    own_keys = [join_key((str(item.provider), str(item.match_id))) for item in items]
    # REQUIRED on the primitives path. `for_each` grew an inline `seen` check when it went
    # streaming; this path got nothing, and it is the one both gates are blindest to (the static
    # gate proves a call exists, not that it is passed the right keys). A colliding key here is
    # SILENT and self-certifying: `already_done` returns True for the duplicate, the item is skipped
    # and lost, and `present` counts the one shared file once per duplicate key -- so
    # `present == len(own_keys)` and `assert_conservation` reports the lossy run as healthy.
    # Affordable up front here precisely because `items` is materialised on this path.
    _require_injective(own_keys)
    failed = 0
    for item, k in zip(items, own_keys, strict=True):
        if already_done(gen, k):
            continue
        write_shard(shard_path(gen, k), compute(item), tag=tag)
    assert_conservation(gen, keys=own_keys, failed=failed)
```

`items` must be a materialised sequence here, not a generator — `own_keys` and the loop both consume it. If the driver's source is a generator, wrap it in `list(...)` first.

**The ordering half of the provenance gate does not cover this driver.**
`test_the_guard_precedes_the_corpus_walk_within_main` detects the corpus walk as
`_calls_in(main_fn, "load_matches")` and `pytest.skip`s when it finds none — so a driver whose corpus
call is `load_xtgk_cohort`, `load_retention_cohort` or a delegated walk is silently exempt, and the
suite reports a skip rather than a gap. Measured on today's registry: **5 checked, 2 skipped**
(`validate_xs_probe`, `validate_xshot_causal`). Ordering here rests on
`test_the_ENTRY_POINT_enforces_the_clean_tree` plus `preflight` running first inside `main`, not on
that check. Do not read its green as coverage.

Record the reason for not using `for_each` in a comment at the call site. The gate enforces **two** calls on this path, not one: `test_an_ESCAPE_HATCH_driver_still_asserts_conservation` (`assert_conservation`) and `test_an_ESCAPE_HATCH_driver_still_checks_key_injectivity` (`_require_injective`). The second exists because the first is satisfiable by a run that silently dropped items — a duplicate key is skipped as 'already done', and the one shared shard is counted once per duplicate, so conservation balances.

- [ ] **Step 3.5: Wire fail-closed run provenance (spec §4.5)**

This driver writes `<report_out>.json` + `.md` — an audit record by its own docstring — and today has
**no** provenance wiring (measured: `_provenance` imports 0, `--allow-dirty` 0, absent from
`ARTIFACT_DRIVERS`). Add, in `main()` and **before** any corpus work:

```python
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
```

with the flag on the parser:

```python
    ap.add_argument("--allow-dirty", action="store_true",
                    help="permit a dirty tree (dev only; the report is marked dirty)")
```

and the two fields into `build_manifest`'s output so the artifact records the truth either way:

```python
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
```

Then add `"calibrate_xt_bandwidth"` to `ARTIFACT_DRIVERS` in `tests/scripts/test_provenance_wiring.py`.
That gate checks the guard is called **from `main()`** and precedes the corpus walk, so placing it in a
helper will fail it — which is the intent (a `run()` that refuses to execute on a dirty checkout cannot
be tested without mocking git; the CLI refuses, `run()` records).

- [ ] **Step 4: Run the gate and this driver's tests**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ tests/calibration/ -q
```
Expected: all pass, including `test_provenance_wiring.py`'s **five** parametrised checks for the
newly registered driver (`@pytest.mark.parametrize("name", ARTIFACT_DRIVERS)` at `:41`, `:49`, `:71`,
`:106`, `:124` — counted, not remembered).

**One of the five may SKIP, and that is worth reading rather than scrolling past.**
`test_the_guard_precedes_the_corpus_walk_within_main` checks ordering only when both
`require_clean_tree` and `load_matches` appear inside `main()`; when the walk is delegated out it
skips with "the entry-point gate covers it". Measured on today's registry: **5 checked, 2 skipped —
`validate_xs_probe` and `validate_xshot_causal`.** That the cycle's own motivating driver is one of
the two is the point: after migration its walk moves into `for_each`, so the ordering check will
keep skipping, and the guarantee that the guard runs first rests entirely on
`test_the_ENTRY_POINT_enforces_the_clean_tree` plus `for_each`'s own pre-flight. Do not read a green
run as evidence that ordering was verified for every driver. `test_cli_smoke.py::test_build_manifest_has_data_and_version_identity` calls
`build_manifest` directly — if the two new fields are added as **required** positional/keyword args it
will fail, so give them a default or pass them from `main()` only.

- [ ] **Step 5: Remove from `_NOT_YET_MIGRATED` and propose the commit**

```bash
git add scripts/calibrate_xt_bandwidth.py tests/scripts/test_corpus_driver_resilience.py
git commit -m "refactor(scripts): migrate calibrate_xt_bandwidth -- primitives path + shared cohort cache"
```

---

## Task 14: Migration template for the remaining Shape-A drivers

**Applies to** (one commit each, in this order): `train_ghost_gk`, `train_xshot_occurrence`, `train_xcross_attempt`, `validate_xs_probe`, `derive_opengoal_range`, `train_gk_completion`, `tune_structural_pass_sigma`, `validate_shot_goalmouth_sb`, `validate_xcross_causal`, `validate_xtgk_possession_value`, `run_signoff_power`, `calibrate_tracking_defaults`, `measure_cover_shadow_argmax_agreement` (**last**, after a rebase check — another session owns it;
`git log -1` on the file reads `748cd1d` / 4.67.0).

**RETRACTED: its measured numbers do NOT stand, and the reason is not a code change.** An earlier
revision of this line said they did, on the evidence that `_cover_shadows.py` is byte-unchanged since
`748cd1d`. That module is unchanged — but it was the wrong thing to check, and checking the right
thing inverts the conclusion.

What was actually verified, quoted rather than summarised:

- The driver does **not** go through `features.py`. It imports `_compute_cover_shadow_dict` directly
  (`:56-58`) and calls it at `:121`/`:124`. So RC1's `features.py` hunks are genuinely not on its
  call path, and the reviewer's version of this concern does not apply either.
- `_cover_shadows.py` contains **zero** occurrences of `acting_team_attacks_rtl` or `reproject` —
  the module does not reproject the passer. That is why ADR-051 RC1 fixed the *callers*.
- The fixed library caller now does: `_flip = acting_team_attacks_rtl(actions, frames)`, build
  `passer_xy` from `start_x`/`start_y`, then point-reflect it `(FIELD_LENGTH - x, FIELD_WIDTH - y)`
  when flipped (`features.py:3681-3711`).
- The driver, at `:114`, does the first half and **not** the reflection:
  `passer_xy = (float(row["start_x"]), float(row["start_y"]))`, passed straight in beside a
  frame-LTR `frame_data`. It has no home-only filter.

So the driver is a **third RC1 site that RC1 missed** — the same convention mixing, live on `main`
today. Its headline is the argmax of the *cheap* path's `max_single_defender_player_id`, and RC1
measured that cheap-path column changing on **90.7% / 100% of away rows**, so roughly the away half
of its 970-action sample was scored with the passer at the wrong end of the pitch.

**This is out of scope and must not be silently fixed here.** It is another session's file, it is a
correctness fix in ADR-051's defect class rather than a resilience change, and the ADR-051 cycle has
PRs 4 and 5 still to land. What this task owes is a hand-off, not a repair:

- migrate the driver for resilience only, changing no geometry;
- state in the commit body that `docs/research/cover_shadow_identity/` needs re-measuring after the
  passer is reprojected, and that the recorded `0.1992` is a pre-RC1 number;
- raise it with the ADR-051 session as a candidate for their PR 4, since they own the defect class
  and the registry of sites.

The general lesson is the reviewer's, and it applies to the retracted claim itself: clearing a
*module* does not clear a *call path*. Quote the import and the call site, not the neighbouring file.

---

## Task 14b: `train_gk_completion`'s bundling guard - key it on behaviour, not parameters

Lands in `train_gk_completion`'s own commit from Task 14. Separated here because it is a correctness
change to an existing guard plus one new control, not a migration.

**What is actually in THIS tree - verified by reading `scripts/train_gk_completion.py`, not
relayed.** An earlier draft of this task instructed "replace the max-over-arrays comparison" and
"wire it into the mode dispatch". Neither exists here:

| Claimed | Verified |
|---|---|
| a `mode` argument | `main()`'s parser has `--providers`, `--max-per-provider`, `--tracking-limit`, `--variant`, `--cache-features`. No `mode`. |
| a must-move retrain guard | There is no retrain path at all. **Both** variants re-bundle: fit fresh, assert the fresh fit matches the committed weights, then save the **committed** model back with fresh gate metadata. `:228-231` and `:356-357` say so explicitly - "Re-fit is NEVER persisted on a re-bundle." |
| a max-over-arrays comparison | `_CORPUS_IDENTITY_ATOL = 0.05` is applied **per array** by four separate `np.testing.assert_allclose` calls (`:239-242`, `:365-368`), in the must-**NOT**-move direction. |
| a test file | `tests/scripts/test_train_gk_completion.py` does not exist. |

So the ADR-051 session's machinery has to be **built** here, not amended. Writing an implementation
step against code nobody opened is the exact defect this cycle indicts elsewhere; recorded rather
than quietly corrected.

**Their design, which stays.** A required `mode` with no default so neither re-bundle nor retrain is
reachable by accident, a mandatory reason string, and a record of the superseded coefficients.
Retrain then asserts the weights **moved** - a retrain reproducing the old weights means the input
change never reached the model, and shipping it as *"retrained on X"* would be a false claim.

**The defect they self-reported.** A must-move guard keyed on parameter deltas takes the maximum
across coefficients, intercept, mean and standard deviation - and mean/std are **raw-feature
statistics in metres**. A translation-class change, exactly what a coordinate or geometry correction
produces, moves those by metres while the coefficients move by ~3e-17. The guard passes and stamps
the artifact as changed.

**Their proposed fix was "key on coefficients alone." Do not do that.** It keeps testing *parameters*
as a proxy for *behaviour*, and this repo solved that twice already: `_chirality.py` fingerprints
model **output** on a fixed probe frame, `_feature_contract.py` fingerprints the **feature vector** on
one. A coefficients-only rule also fails in the other direction - a change confined to
standardisation genuinely alters served probabilities while the coefficients sit still.

**MEASURED, and it changes the API.** The single-probe signature drafted first
(`predictions_moved(old, new, probe)`) **cannot express the property**. Run:

```
translation, `assert not predictions_moved(old, shifted, probe + 5.0)` -> got True   (test FAILS)
translation, same probe for both                                      -> got True
```

Serving both models on one array asks "do these two functions agree on this input", but the question
is "does the model, applied to the data **it** will see, behave as the committed model did on the
data **it** saw". Under a translation each model sees its own coordinates, so two probes are
required:

```
two-probe form:
  translation      (old on old coords, new on new coords) -> moved=False   (want False)
  real retrain     (identical coords)                     -> moved=True    (want True)
  standardisation-only change, coef untouched             -> moved=True    (want True)
```

**And the two-probe signature has a call-site consequence, handled in Step 5.** The committed
weights directory does not persist a design matrix, so `probe_old` is not recoverable at retrain
time; passing `X_all` for both would silently reinstate the single-probe form this task just
measured as inadequate. Step 5 makes the operator declare whether the feature space moved and
refuses the case the artifact format cannot answer, rather than defaulting to the wrong question.

**Surfaced, NOT fixed in this cycle: the existing corpus-identity check has the mirror defect.** Same
probe, same run:

```
max|coef_fresh - coef_committed| = 0    -> assert_allclose(atol=0.05) passes
max|mean_fresh - mean_committed| = 5    -> assert_allclose(atol=0.05) RAISES
max|std_fresh  - std_committed|  = 0    -> assert_allclose(atol=0.05) passes
```

A geometry correction upstream translates the raw features, so `_mean` moves by metres and the
re-bundle **aborts** - reporting a corpus-identity failure for a change that provably leaves every
served probability identical. Same parameters-as-proxy error, pointing the other way. It is **not**
repaired here: `_CORPUS_IDENTITY_ATOL` was deliberately relaxed 1e-9 -> 0.05 in 4.21.4 to tolerate
unrecorded-`tracking_limit` density float noise, so re-keying it is a decision about a released
calibration gate and belongs to whoever owns that gate, with the owner's sign-off. Raise it in the
commit body alongside the relay.

- [ ] **Step 0: Build the `mode` control (it does not exist yet)**

In `scripts/train_gk_completion.py`'s `main()`, after the `--variant` argument:

```python
    ap.add_argument(
        "--mode",
        required=True,
        choices=["rebundle", "retrain"],
        help="REQUIRED, no default. `rebundle` re-attaches fresh gate metadata to the COMMITTED "
        "weights and asserts the fresh fit still reproduces them. `retrain` ships the fresh fit and "
        "asserts the SERVED PREDICTIONS moved -- a retrain that reproduces the old behaviour means "
        "the input change never reached the model, and shipping it as 'retrained on X' is a false "
        "claim. Neither is reachable by accident.",
    )
    ap.add_argument(
        "--reason",
        required=True,
        help="REQUIRED. Why this run is bundling -- recorded verbatim in metrics.json. A weights "
        "change with no stated cause is unreviewable six months later.",
    )
```

`--mode` needs no threading: `_train_skillcorner(args)` already takes the whole namespace.

Record the superseded coefficients on every bundling run, next to the existing `metrics` dict at
`:378` (and its skillcorner twin at `:260`):

```python
    metrics["mode"] = args.mode
    metrics["reason"] = args.reason
    metrics["superseded_coef"] = (
        dict(zip(feats, served_before._coef.tolist(), strict=True)) if served_before is not None else None
    )
```

where `served_before` is the committed model loaded before the save, and `None` on a first-ever
bundle. The existing `try: served = GkCompletionModel.load(...) / except FileNotFoundError` blocks
already produce exactly that value - bind it there rather than loading a second time.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_train_gk_completion.py`:

```python
"""The bundling guard is keyed on SERVED PREDICTIONS, not on parameter deltas."""

import numpy as np
import pytest

import scripts.train_gk_completion as mod


def _weights():
    rng = np.random.default_rng(0)
    return (
        {"coef": rng.normal(size=4), "intercept": 0.1, "mean": np.zeros(4), "std": np.ones(4)},
        rng.normal(size=(64, 4)),
    )


def test_a_pure_translation_is_NOT_a_retrain():
    """The measured defect. A geometry correction translates the raw features, so `mean` moves by
    metres while the coefficients move by ~3e-17. Standardisation absorbs it exactly, so
    `(x - mean) / std` -- and every served probability -- is unchanged. A guard keyed on ANY array
    moving calls that a retrain and stamps the artifact `retrained on X`."""
    old, probe = _weights()
    shifted = {**old, "mean": old["mean"] + 5.0}
    assert not mod.predictions_moved(old, shifted, probe_old=probe, probe_new=probe + 5.0)


def test_a_real_retrain_IS_detected():
    """Non-vacuity: the guard must reject a translation, not reject everything. Without this half,
    `return False` passes the test above."""
    old, probe = _weights()
    new = {**old, "coef": old["coef"] + 0.5}
    assert mod.predictions_moved(old, new, probe_old=probe, probe_new=probe)


def test_a_STANDARDISATION_ONLY_change_IS_detected():
    """The direction the ADR-051 session's proposed coefficients-only fix would get wrong: the
    coefficients are byte-identical and the served probabilities still move."""
    old, probe = _weights()
    new = {**old, "std": old["std"] * 2.0}
    assert mod.predictions_moved(old, new, probe_old=probe, probe_new=probe)


def test_a_SINGLE_probe_cannot_express_the_property():
    """Pins WHY the signature takes two probes -- the one-probe form was drafted first and measured
    to fail. Serving both models on one array asks whether two functions agree on an input; the
    question is whether each model behaves the same on the coordinates IT sees."""
    old, probe = _weights()
    shifted = {**old, "mean": old["mean"] + 5.0}
    # The one-probe form: the same array to both. It reports movement where there is none.
    assert mod.predictions_moved(old, shifted, probe_old=probe, probe_new=probe)


def test_mode_and_reason_are_REQUIRED(monkeypatch):
    """No default on either: a weights change must never be reachable by accident, and must never
    ship without a stated cause.

    Three-argument `setattr`, not the two-argument string form: pytest 9.1 removed string targets,
    and the repo already has two two-arg uses that will break on that bump. New code does not add a
    third."""
    import sys

    monkeypatch.setattr(sys, "argv", ["train_gk_completion.py"])
    with pytest.raises(SystemExit):
        mod.main()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_gk_completion.py -q`
Expected: FAIL - `AttributeError: module ... has no attribute 'predictions_moved'` on four, and
`test_mode_and_reason_are_REQUIRED` fails because `main()` currently parses successfully with no
arguments.

- [ ] **Step 3: Implement the behavioural guard**

Add to `scripts/train_gk_completion.py`:

```python
def predictions_moved(old: dict, new: dict, *, probe_old, probe_new, atol: float = 1e-6) -> bool:
    """Did the SERVED probabilities change -- each model evaluated on the coordinates IT sees?

    Keyed on behaviour, not on parameter deltas, following the house pattern: `_chirality.py`
    fingerprints model OUTPUT on a fixed probe frame, `_feature_contract.py` fingerprints the
    feature vector on one. A parameter-delta rule is wrong in BOTH directions -- a pure translation
    moves `mean` by metres while leaving every served probability identical (reads as a retrain when
    nothing changed), and a change confined to standardisation alters served probabilities while the
    coefficients sit still (reads as no-change when the model moved). Both are measured in
    `tests/scripts/test_train_gk_completion.py`.

    TWO probes, not one. `probe_old` is the design matrix the committed model was fit on and
    `probe_new` the one the fresh fit was; they are the SAME array whenever the feature space did
    not move, which is the ordinary case. A single shared probe was drafted first and MEASURED
    unable to express the property: it asks whether two functions agree on one input, when the
    question is whether the model behaves the same on the data each version actually sees.
    """

    def _serve(w: dict, p) -> np.ndarray:
        z = (np.asarray(p) - w["mean"]) / w["std"]
        return 1.0 / (1.0 + np.exp(-(z @ w["coef"] + w["intercept"])))

    return not np.allclose(_serve(old, probe_old), _serve(new, probe_new), atol=atol, rtol=0.0)
```

The probe is the model's own design matrix (`X_all`), never synthetic noise - a probe that does not
exercise the region a retrain changed would report "no movement" for a real retrain.

- [ ] **Step 4: Run to verify all five pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_train_gk_completion.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Wire it in — and make the un-answerable case REFUSE rather than default**

A first draft of this step passed `probe_old=X_all, probe_new=X_all`, which is precisely the
single-probe form Step 1 measures as unable to express the property. That is not a slip that a
comment fixes; it is forced, because **the committed weights directory does not persist a design
matrix** (it holds `coef`/`intercept`/`mean`/`std`), so the old feature space is genuinely not
recoverable at retrain time.

**The correct instrument depends on a fact only the operator knows, and the two defaults are wrong
in opposite cases:**

| Feature space | Right question | `probe_old = probe_new = X_all`? |
|---|---|---|
| unchanged (more data, new hyperparameters) | do the two models disagree on the same inputs? | **Correct.** Both models legitimately see the same coordinates. |
| moved (a geometry or coordinate correction) | does each model behave the same on the coordinates *it* sees? | **Wrong.** The old model is fed coordinates it never saw, disagrees for that reason alone, and the guard passes a retrain that changed nothing. |

Note what the second row rules out: comparing coefficients in *standardised* space would answer the
moved case, and it is exactly the coefficients-only rule this task rejects for the unchanged case
(where a standardisation-only change moves served probabilities while `coef` sits still). Neither
instrument is universally right, so the choice cannot be defaulted — it has to be declared.

Add the declaration, required whenever `--mode retrain`:

```python
    ap.add_argument(
        "--feature-space",
        choices=["unchanged", "moved"],
        default=None,
        help="REQUIRED with --mode retrain. `unchanged` = more data or new hyperparameters, so both "
        "models are validly served on the same design matrix. `moved` = a geometry/coordinate "
        "correction changed the raw features, so the committed model must be served on the "
        "PRE-change matrix or the comparison is meaningless. There is no safe default: the two "
        "choices need opposite instruments.",
    )
    ap.add_argument("--probe-old", default=None, help="parquet of the pre-change design matrix; required for --feature-space moved")
```

and the wiring:

```python
    if args.mode == "retrain":
        if args.feature_space is None:
            raise SystemExit("--mode retrain requires --feature-space {unchanged,moved}; see --help.")
        if args.feature_space == "moved" and args.probe_old is None:
            raise SystemExit(
                "--feature-space moved requires --probe-old: the committed weights directory stores "
                "coef/intercept/mean/std but NOT a design matrix, so the pre-change feature space "
                "cannot be reconstructed from the artifact. Without it this guard would serve the "
                "committed model on coordinates it never saw, report a difference caused by the "
                "coordinate change alone, and stamp the artifact `retrained` when nothing "
                "behavioural moved -- the exact defect it exists to catch. Persist a probe sample "
                "beside the weights (see the ADR) or re-run with the pre-change extractor."
            )
        probe_old = pd.read_parquet(args.probe_old)[feats] if args.probe_old else X_all

        def _as_weights(m):
            return {"coef": m._coef, "intercept": m._intercept, "mean": m._mean, "std": m._std}

        if served_before is not None and not predictions_moved(
            _as_weights(served_before), _as_weights(model), probe_old=probe_old, probe_new=X_all
        ):
            raise SystemExit(
                "RETRAIN produced the committed model's served predictions unchanged -- the input "
                "change never reached the model. Shipping this as a retrain would be a false claim. "
                f"(reason given: {args.reason!r})"
            )
```

Gate the existing corpus-identity assertions on `args.mode == "rebundle"` — they ask the re-bundle
question (did the fresh fit reproduce the committed weights) and a parameter comparison is the right
instrument for that one, subject to the mirror defect surfaced above.

**What this does and does not buy, stated plainly.** With no probe persisted anywhere today,
`--feature-space moved` currently always refuses. That is the point: a loud refusal naming why the
comparison is impossible is strictly better than a silent `True`, and it is the same principle §7
applies to `assert_conservation` — a stated gap beats a guard believed to cover a case it cannot.
Persisting a fixed probe sample beside the weights is the real end state and belongs in ADR-011's
artifact format, not in a `scripts/` resilience cycle; record it in the ADR's consequences as the
follow-up that would let `moved` actually run.

Add the two refusal tests:

```python
def test_retrain_REFUSES_without_a_declared_feature_space(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["train_gk_completion.py", "--mode", "retrain", "--reason", "x"])
    with pytest.raises(SystemExit, match="feature-space"):
        mod.main()


def test_retrain_REFUSES_a_moved_feature_space_with_no_probe(monkeypatch):
    """The motivating case, and the one the artifact format cannot currently serve. Refusing names
    why; defaulting to `probe_old = X_all` would silently answer the wrong question."""
    import sys

    monkeypatch.setattr(
        sys, "argv",
        ["train_gk_completion.py", "--mode", "retrain", "--reason", "x", "--feature-space", "moved"],
    )
    with pytest.raises(SystemExit, match="probe-old"):
        mod.main()
```

- [ ] **Step 6: Run the driver's smoke and the full script suite**

Run:
```bash
.venv/Scripts/python.exe scripts/train_gk_completion.py --help
.venv/Scripts/python.exe -m pytest tests/scripts/ -q
```
Expected: `--help` lists `--mode` and `--reason` as required; suite passes.

- [ ] **Step 7: Relay the correction**

Note in the commit body that (a) the guard was built here rather than amended, since none of the
relayed machinery was in this tree, (b) it is keyed on served predictions with a two-probe signature,
and why, and (c) the corpus-identity check carries the mirror defect and was deliberately left alone
pending the owner of that calibration gate. The ADR-051 session needs all three when rebasing.

---

## Task 14c: fix the cover-shadow driver's live ADR-028 RC1 defect

**Files:**
- Modify: `scripts/measure_cover_shadow_argmax_agreement.py:53-59,105,114`
- Modify: `docs/research/cover_shadow_identity/` (a provenance note, not a re-measurement)

A correctness fix, not a migration — it lands in commit 3 beside Task 14b, so the geometry change is
legible in its own diff rather than buried in a resilience refactor.

**The defect, verified line by line.** ADR-051 RC1 (4.70.0) fixed `add_cover_shadows` and
`cover_shadow_xfns` in `features.py`, which built `passer_xy` from raw **action-LTR**
`start_x`/`start_y` and differenced it against **frame-LTR** defenders, receivers and the ball. This
driver was never a registered site because it bypasses `features.py` entirely — it imports
`_compute_cover_shadow_dict` directly (`:56-58`) and calls it at `:121`/`:124`. So the same defect is
still live here:

- `_cover_shadows.py` contains **zero** occurrences of `acting_team_attacks_rtl` — the module never
  reprojects the passer; that is why RC1 fixed the callers.
- The fixed library caller computes `_flip = acting_team_attacks_rtl(actions, frames)` once, then
  point-reflects `passer_xy` to `(FIELD_LENGTH - x, FIELD_WIDTH - y)` when flipped
  (`features.py:3681-3711`).
- This driver, at `:114`, does neither: `passer_xy = (float(row["start_x"]), float(row["start_y"]))`,
  passed straight in beside a frame-LTR `frame_data`, with no home-only filter.

**Why it matters more here than in an ordinary consumer.** The script's whole purpose is to compare
the cheap and exact paths, and **the cheap path consumes the passer while the exact path does not**.
So RC1 degrades exactly one arm of the comparison — it does not cancel.

**The recorded verdict survives, by arithmetic that needs no re-run.** `docs/research/cover_shadow_identity/`
reports agreement **0.157 on 970 qualifying actions** against a pre-registered **0.90** floor. Home
rows are byte-identical under this fix by construction (`flip` is `False` for a home action, so
`passer_xy` is untouched), so only away rows can move. Reaching 0.90 needs `0.90 x 970 = 873`
agreements; there are `0.157 x 970 = 152`; so it would take **721 further agreements, 74.3% of the
sample**, from a population of away rows that is roughly half a match. Even if *every* away row
flipped from disagree to agree the ceiling is about `152 + 485 = 637`, i.e. **0.657 < 0.90**. The
`detailed=True` gate on `max_single_defender_player_id` therefore stands on its own without waiting
for a corpus pass.

That bound is derived here from the two published figures, not measured. The *point estimate* is
still wrong and is corrected by re-running, which is owner-gated (real WC2022) — so this task fixes
the code and records the status, and does not silently leave a stale number looking fresh.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_measure_cover_shadow_orientation.py`:

```python
"""ADR-028 RC1: the driver must reproject the passer into frame coords, like the library caller."""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "measure_cover_shadow_argmax_agreement.py"


def _calls(tree) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


def test_the_driver_reprojects_the_passer_to_frame_coords():
    """`_compute_cover_shadow_dict` differences `passer_xy` against FRAME-LTR positions, and the
    module never reprojects (zero `acting_team_attacks_rtl` references in `_cover_shadows.py`) --
    the caller must. `features.py:3681-3711` is the reference implementation. Without this the away
    half of every measurement puts the passer at the wrong end of the pitch, and because the CHEAP
    path consumes the passer while the EXACT path does not, the error does not cancel between the
    two arms this script exists to compare."""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    assert "acting_team_attacks_rtl" in _calls(tree), (
        "the driver builds passer_xy from raw action-LTR start_x/start_y and never reprojects it; "
        "this is the ADR-051 RC1 defect, unregistered here because the driver bypasses features.py"
    )


def test_the_reflection_uses_BOTH_axes():
    """ADR-028 is a 180-degree POINT reflection, not an x-only mirror. An x-only flip is exact only
    for a y-symmetric configuration, which is precisely what hid the incomplete repair in PR-S119."""
    src = _SRC.read_text(encoding="utf-8")
    assert "FIELD_LENGTH -" in src and "FIELD_WIDTH -" in src, (
        "reflection must negate both axes; an x-only mirror silently passes y-symmetric fixtures"
    )
```

- [ ] **Step 2: Run to verify both fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_measure_cover_shadow_orientation.py -q`
Expected: 2 failures — the driver calls neither.

- [ ] **Step 3: Apply the fix**

Add to the imports (`:55`, beside the existing `silly_kicks.tracking` import):

```python
from silly_kicks.tracking._action_orientation import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    acting_team_attacks_rtl,
)
```

In `measure_match`, after `frame_groups = ...`:

```python
    # ADR-028 RC1. This driver bypasses `features.py` -- it calls `_compute_cover_shadow_dict`
    # directly -- so RC1 (4.70.0) never reached it and the defect stayed live in the research
    # harness. `_cover_shadows.py` never reprojects the passer (zero `acting_team_attacks_rtl`
    # references); the CALLER must, exactly as `features.py:3681-3711` does. Computed ONCE.
    #
    # It does not cancel between the two arms: the CHEAP path consumes the passer, the EXACT path
    # does not, so an away-row error degrades precisely the comparison this script measures.
    flip = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
```

Change the loop header to carry a positional index:

```python
    for j, (_idx, row) in enumerate(actions.iterrows()):
```

And reflect the passer at `:114`:

```python
        passer_xy = (float(row["start_x"]), float(row["start_y"]))
        if flip[j]:  # action-LTR -> frame coords; BOTH axes (ADR-028 is a point reflection)
            passer_xy = (FIELD_LENGTH - passer_xy[0], FIELD_WIDTH - passer_xy[1])
```

- [ ] **Step 4: Run to verify both pass, plus the whole script suite**

```bash
.venv/Scripts/python.exe -m pytest tests/scripts/ -q -rs
```
Expected: all pass; the two new tests included.

- [ ] **Step 5: Record the status of the recorded numbers**

Add a note at the top of `docs/research/cover_shadow_identity/` stating that the point estimate
(agreement 0.157, and the 0.1992 harm figure) was produced **before** this fix and is pending an
owner re-run, **and** that the gating decision it supports is unaffected — with the arithmetic
above, so a reader can check the claim rather than trust it. Do not delete the numbers: a stale
figure labelled stale is useful; a deleted one loses the comparison point for the re-run.

- [ ] **Step 6: Checkpoint**

```bash
git add scripts/measure_cover_shadow_argmax_agreement.py tests/scripts/test_measure_cover_shadow_orientation.py docs/research/cover_shadow_identity/
```

Accumulates into **commit 3**.

---

## Task 15: Shape B — the four loop-free drivers

**Applies to:** `train_gk_retention`, `validate_xtgk_v2`, `xtgk_v2_kappa_sweep`, `xtgk_v2_keeper_discrimination`.

These have no per-item loop; the expensive thing is one uncached Databricks query. They get the cohort cache only.

- [ ] **Step 1: Add the flag to each driver's argument parser**

```python
    ap.add_argument(
        "--cohort-cache",
        default=None,
        help="parquet path; fetch the cohort once and reuse it. Absent = fetch every run (today's "
             "behaviour). Explicitly named because a mart re-materializes and a cached cohort has "
             "no token this can verify -- so reuse must be the operator's decision, never automatic.",
    )
```

- [ ] **Step 2: Route the cohort load through the helper**

```python
    from scripts._driver import cohort_cache

    df = cohort_cache(args.cohort_cache, build=lambda: load_xtgk_cohort(provider=args.provider))
```

(For `train_gk_retention`, the build call is `load_retention_cohort(...)`.)

- [ ] **Step 3: Verify the no-flag path is unchanged**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/ tests/xtgk/ -q`
Expected: all pass. With `--cohort-cache` absent, `cohort_cache` is a pure passthrough (pinned by `test_cohort_cache_is_a_pure_passthrough_when_no_path_is_given`).

- [ ] **Step 4: Remove all four from `_NOT_YET_MIGRATED` and propose one commit**

```bash
git add scripts/train_gk_retention.py scripts/validate_xtgk_v2.py scripts/xtgk_v2_kappa_sweep.py scripts/xtgk_v2_keeper_discrimination.py tests/scripts/test_corpus_driver_resilience.py
git commit -m "feat(scripts): add --cohort-cache to the four loop-free xT-GK drivers"
```

---

## Task 16: Databricks auth precedence

**Files:**
- Modify: `scripts/_loader_databricks.py:44-53`
- Modify: `tests/scripts/test_loader_databricks_connect.py`

`_connect()` takes the PAT branch on **any** non-empty `DATABRICKS_TOKEN`, so a stale credential pre-empts the working OAuth fallback below it. Measured on this machine: `len=36 prefix=dapi`, `DATABRICKS_CONFIG_PROFILE` unset.

- [ ] **Step 1: Harden the four existing tests BEFORE adding the variable**

In `tests/scripts/test_loader_databricks_connect.py`, add to all four tests (`:61`, `:72`, `:84`, `:92`):

```python
    monkeypatch.delenv("DATABRICKS_AUTH", raising=False)
```

Without this, a developer who exports `DATABRICKS_AUTH=oauth` — exactly the person this feature is for — sees `test_pat_path_uses_access_token` fail locally while CI stays green. An environment-dependent test is how a suite loses its authority.

- [ ] **Step 2: Add the two new failing tests**

```python
def test_DATABRICKS_AUTH_oauth_overrides_a_present_token(monkeypatch, fake_dbsql, fake_sdk):
    """The whole point: a stale PAT sitting in the environment must not pre-empt working OAuth."""
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-stale")
    monkeypatch.setenv("DATABRICKS_AUTH", "oauth")
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    ldb._connect()
    assert fake_dbsql.kwargs is not None
    assert fake_dbsql.kwargs.get("credentials_provider") is not None
    assert "access_token" not in fake_dbsql.kwargs


def test_DATABRICKS_AUTH_pat_without_a_token_RAISES(monkeypatch, fake_dbsql, fake_sdk):
    """Explicitly asking for PAT with no PAT present is an error, not a silent fallback to OAuth --
    a silent fallback would make the flag look honoured when it was ignored."""
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setenv("DATABRICKS_AUTH", "pat")
    with pytest.raises(RuntimeError, match="DATABRICKS_AUTH=pat"):
        ldb._connect()
```

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_databricks_connect.py -q`
Expected: the two new tests FAIL; the four hardened ones pass.

- [ ] **Step 3: Implement the precedence override**

Replace `scripts/_loader_databricks.py:44-53` with:

```python
    # Auth precedence. `DATABRICKS_AUTH` selects the branch explicitly; UNSET preserves the historic
    # behaviour (a non-empty token wins), so CI and legacy setups are untouched. The override exists
    # because a non-empty token pre-empts the OAuth fallback below even when that token is unusable,
    # and the resulting error names the workspace rather than the environment variable.
    auth = os.environ.get("DATABRICKS_AUTH", "").strip().lower()
    if auth not in ("", "pat", "oauth"):
        raise RuntimeError(f"DATABRICKS_AUTH must be 'pat', 'oauth' or unset; got {auth!r}")
    token = os.environ.get("DATABRICKS_TOKEN")
    if auth == "pat" and not token:
        raise RuntimeError("DATABRICKS_AUTH=pat but DATABRICKS_TOKEN is unset or empty")
    if token and auth != "oauth":
        try:
            return dbsql.connect(
                server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
                http_path=http_path,
                access_token=token,
            )
        except Exception as exc:
            # Name the precedence AND both of its causes. This branch fails for a stale PAT and for
            # an expired short-lived bearer alike -- the lakehouse deliberately puts a ~299 s minted
            # OAuth bearer in this same variable -- so a message naming only one mis-diagnoses half
            # the cases.
            raise RuntimeError(
                "Databricks PAT auth failed. A non-empty DATABRICKS_TOKEN took priority over the "
                "OAuth profile; the token may be a STALE PAT or an EXPIRED short-lived bearer. "
                "Unset DATABRICKS_TOKEN, re-mint it, or set DATABRICKS_AUTH=oauth."
            ) from exc
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_loader_databricks_connect.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Propose the commit**

```bash
git add scripts/_loader_databricks.py tests/scripts/test_loader_databricks_connect.py
git commit -m "fix(scripts): DATABRICKS_AUTH precedence override + a diagnosis naming both causes"
```

---

## Task 16b: three-state provenance — `clean` / `dirty` / `unknown`

`git_provenance` collapses "git is unavailable" into `dirty: True` (`_provenance.py:42,46`). The
fail-closed *behaviour* is right and does not change here. The *record* is not: `dirty: true` is a
positive claim that uncommitted modifications exist, and on a tarball checkout or a box without git
that claim is simply false. An artifact asserting something false about its own provenance is the
defect this module was written to prevent, one level down.

ruthless-efficiency arrived at the same three-state vocabulary independently for its own run
metadata, which is a mild corroboration rather than a reason.

**Additive, deliberately.** `dirty` keeps its exact current meaning and value, including `True` for
unknown. Consumers read it (`_partition.aggregate_manifests` ORs `run_tree_dirty` across workers;
every artifact on disk carries the boolean), and widening a published field to a tri-state string
would make `bool(v)` read `"clean"` as **truthy** — silently inverting the aggregate. The new field
sits beside it.

**One policy fork, surfaced not decided.** With `unknown` still folded into `dirty`, a driver run on
a git-less box is refused unless `--allow-dirty`. Letting `unknown` pass unaided would be a
loosening of a fail-closed control and is the owner's call, not this cycle's. Recorded in the ADR's
consequences.

**Files:**
- Modify: `scripts/_provenance.py`
- Modify: `tests/scripts/test_provenance.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_provenance.py`:

```python
def test_a_clean_tree_reports_state_clean(monkeypatch):
    monkeypatch.setattr(mod, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else "")
    prov = mod.git_provenance()
    assert prov["tree_state"] == "clean"
    assert prov["dirty"] is False


def test_a_dirty_tree_reports_state_dirty(monkeypatch):
    monkeypatch.setattr(mod, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else " M scripts/x.py")
    prov = mod.git_provenance()
    assert prov["tree_state"] == "dirty"
    assert prov["dirty"] is True


def test_git_UNAVAILABLE_reports_unknown_but_STILL_refuses(monkeypatch):
    """The honest record and the fail-closed behaviour are independent. `dirty` stays True so the
    refusal and every existing consumer are byte-unchanged; `tree_state` stops asserting that
    modifications exist when nothing was ever inspected."""
    def _boom(*a):
        raise OSError("git not found")

    monkeypatch.setattr(mod, "_git", _boom)
    prov = mod.git_provenance()
    assert prov["tree_state"] == "unknown"
    assert prov["dirty"] is True, "fail-closed: unknown provenance is never treated as clean"
    with pytest.raises(SystemExit, match="unknown"):
        mod.require_clean_tree(prov, allow_dirty=False)


def test_the_boolean_is_UNCHANGED_for_every_state(monkeypatch):
    """Hyrum: `run_tree_dirty` is read by `_partition.aggregate_manifests` (which ORs it) and sits in
    every artifact already written. Widening it to the tri-state string would make `bool("clean")`
    truthy and silently invert the aggregate. Pinned so nobody 'tidies' the two into one field."""
    for porcelain, expected in ((("", False)), ((" M x.py", True))):
        monkeypatch.setattr(mod, "_git", lambda *a, _p=porcelain: "abc123" if a[0] == "rev-parse" else _p)
        assert mod.git_provenance()["dirty"] is expected
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_provenance.py -q`
Expected: 3 failures on `KeyError: 'tree_state'`; `test_the_boolean_is_UNCHANGED_for_every_state`
passes already — it pins existing behaviour and is a regression guard, not a red test.

- [ ] **Step 3: Implement**

In `scripts/_provenance.py`, add the field at all three return sites and document it:

```python
def git_provenance() -> dict:
    """``{"commit", "tree_state", "dirty", "dirty_files"}`` for the current tree.

    ``tree_state`` is ``"clean"``, ``"dirty"`` or ``"unknown"``. ``dirty`` is the ORIGINAL boolean,
    unchanged: ``True`` for BOTH ``dirty`` and ``unknown``, because unknown provenance is treated as
    untrustworthy and never as clean. Two fields rather than one widened field -- ``run_tree_dirty``
    is already published, is OR-ed across workers by `_partition.aggregate_manifests`, and
    ``bool("clean")`` is truthy, so a tri-state string in the boolean's place would silently invert
    every aggregate.

    The distinction is not cosmetic. ``dirty: true`` asserts that uncommitted modifications EXIST; on
    a tarball checkout or a box without git that assertion is false, and an artifact making a false
    claim about its own provenance is the exact failure this module exists to prevent.
    """
    try:
        commit = _git("rev-parse", "HEAD")
    except _GIT_FAILURES:
        return {"commit": "unknown", "tree_state": "unknown", "dirty": True, "dirty_files": []}
    try:
        porcelain = _git("status", "--porcelain")
    except _GIT_FAILURES:
        return {"commit": commit, "tree_state": "unknown", "dirty": True, "dirty_files": []}
    files = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {
        "commit": commit,
        "tree_state": "dirty" if files else "clean",
        "dirty": bool(files),
        "dirty_files": files,
    }
```

and differentiate the refusal message in `require_clean_tree`:

```python
    if prov["dirty"] and not allow_dirty:
        if prov.get("tree_state") == "unknown":
            raise SystemExit(
                "refusing to write a registered artifact with UNKNOWN provenance: git is "
                "unavailable, so the tree could not be inspected. Nothing here claims the tree is "
                "modified -- it claims nothing is known about it, which is equally unusable as a "
                "provenance record. Run from a git checkout, or pass --allow-dirty for a dev run."
            )
        listed = ", ".join(prov["dirty_files"][:5])
        raise SystemExit(
            f"refusing to write a registered artifact from a DIRTY tree (HEAD={prov['commit'][:12]}): "
            f"{listed}. The recorded commit would not describe the code that ran. "
            "Commit first, or pass --allow-dirty for a dev run (the artifact will be marked dirty)."
        )
```

Note the removed `or "(git unavailable)"` fallback on `listed`: it existed only because the
unknown case fell through the dirty branch, and it now has its own message. Leaving it would be a
second, weaker description of a case that is handled above it.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_provenance.py -q`
Expected: all pass.

- [ ] **Step 5: Give `run_tree_state` a named case in `aggregate_manifests`**

Without this the field is born dead at the corpus level. `manifest_<tag>.json` is one of the output
blocks it gets stamped into, and Task 6b's own docstring states the rule that kills it: *"Any other
string is DROPPED ... a field that must reach the corpus artifact needs a named case here, not
merely a place in the per-worker manifest."* The aggregate would keep only the OR-ed boolean, so a
reader still could not tell *dirty* from *unknown* — the entire distinction this task creates.

Task 6b's `dropped_fields` makes the omission visible rather than silent; it does not make the field
arrive. Both are needed. In `scripts/_partition.py`:

```python
    tree_states: set[str] = set()
```

```python
            elif k == "run_tree_state":
                tree_states.add(str(v))
```

```python
        # A SET, not a worst-of reduction. A corpus where one worker ran clean and another could not
        # be inspected is exactly the signal worth seeing, and reducing it to a single worst value
        # would hide it -- the same argument as `commits_seen`.
        "tree_states_seen": sorted(tree_states),
```

Add `"run_tree_state"` to `_meta`, and a test beside Task 6b's:

```python
def test_MIXED_tree_states_are_all_visible_in_the_aggregate(tmp_path):
    """One clean worker and one that could not be inspected is a real corpus state, and a
    worst-of reduction would render it identical to two dirty workers."""
    _write(tmp_path, "w0", {"n_attempted": 4, "run_commit": "c1", "run_tree_dirty": False, "run_tree_state": "clean"})
    _write(tmp_path, "w1", {"n_attempted": 4, "run_commit": "c1", "run_tree_dirty": True, "run_tree_state": "unknown"})
    got = mod.aggregate_manifests(tmp_path, defaults=("n_attempted",))
    assert got["tree_states_seen"] == ["clean", "unknown"]
    assert got["run_tree_dirty"] is True, "the OR-ed boolean is unchanged"
    assert got["dropped_fields"] == [], "and the field is no longer among the dropped"
```

- [ ] **Step 6: Stamp it, and check nothing reads the old shape positionally**

Every artifact driver stamps the provenance dict; add `run_tree_state` beside `run_tree_dirty` in
the drivers' output blocks. Then:

```bash
grep -rn "tree_dirty\|dirty_files" scripts/ tests/ | grep -v "_provenance"
.venv/Scripts/python.exe -m pytest tests/scripts/ -q
```
Expected: every hit is a keyed read, never an unpack or an index; suite passes.

- [ ] **Step 7: Propose the commit**

```bash
git add scripts/_provenance.py scripts/_partition.py tests/scripts/test_provenance.py tests/scripts/test_partition.py scripts/*.py
git commit -m "feat(scripts): record tree_state clean/dirty/unknown beside the dirty boolean"
```

This is `_partition.py`'s **second** edit in the cycle (Task 6b was the first, in commit 1). That is
deliberate: each edit is motivated by the task that needs it, rather than bundled into a speculative
one.

---

## Task 17: Close the gate

- [ ] **Step 1: Confirm `_NOT_YET_MIGRATED` is empty**

`tests/scripts/test_corpus_driver_resilience.py` should now have `_NOT_YET_MIGRATED: dict[str, str] = {}`.

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_corpus_driver_resilience.py -q`
Expected: all pass, **zero skips**.

- [ ] **Step 2: Run the whole suite on both interpreters**

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --tb=short -q
.venv312/Scripts/python.exe -m pytest tests/ -m "not e2e" --tb=short -q
```
Expected: both green. The 3.12 leg runs pandas 3.x and catches dtype behaviour the 3.10 leg does not.

---

## Task 18: ADR, docs, and the version bump

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-corpus-driver-contract.md`
- Modify: `scripts/_partition.py:13,56` (docstring only)
- Modify: `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `pyproject.toml:7`, `silly_kicks/__init__.py:7`, `uv.lock`

- [ ] **Step 1: Pick the ADR number**

Run: `ls docs/superpowers/adrs/ | tail -5` and take the next free number. Do not assume one — another session may have landed an ADR on this branch's base.

- [ ] **Step 2: Write the ADR**

Follow `docs/superpowers/adrs/ADR-TEMPLATE.md`. Its Consequences section must record **three** limits, taken verbatim from the spec's §7 — an ADR that claims the guard is complete is worse than no ADR:

1. **Token completeness is unverifiable.** An author can declare the wrong inputs and get a fingerprint that never changes. Requiring a reason for the empty declaration closes *silent omission*, not *mis-declaration*.
2. **Neither gate catches a second, uncounted loop.** The static gate proves adoption; `assert_conservation` proves *per-pass* conservation over the pass's own keys. A driver that calls `for_each` trivially and accumulates the real corpus elsewhere writes no shards for that loop and lists none of its keys, so nothing sees it. Record the fan-in check (union of all manifests' key sets against the generation's contents) as the follow-up that would cover it. **Do not describe this as closed.**
3. **Exception-path drivers get even the per-pass guarantee only by contract.** `for_each` calls `assert_conservation` internally; a primitives-path driver must call it itself, and the gate can check the call exists but not that it is passed the right keys.

It supersedes the CLAUDE.md prose rule, which failed twice: `validate_xshot_causal.py` wrote an artifact with no provenance, and `validate_xs_probe.py` stamped a bare `git rev-parse HEAD`.

- [ ] **Step 3: Correct the `_partition.py` fence**

At `:13` and `:56`, the text reads "`scripts/_loader_*` is READ-ONLY from here … which this cycle may not modify". Amend it to name the TF-19 partition cycle it bound, so a future reader does not read it as permanent.

- [ ] **Step 4: Update CLAUDE.md**

Replace the prose resilience rule with a pointer to the new ADR and one sentence naming `scripts/_driver.py` as the seam. Keep the measured evidence (the 8.7h loss) — it is why the rule exists.

- [ ] **Step 5: Bump the version at all five sites**

Target **4.72.0** — `pyproject.toml:7` reads `4.71.0` on this branch's base (`89dd9af`). Confirm that
first rather than trusting this line: another session may have released in between, and no session
owns the next number.

```bash
git fetch origin && git log --oneline origin/main -1 && grep -n "^version" pyproject.toml
```

`pyproject.toml:7`, `silly_kicks/__init__.py:7`, `uv.lock`, the `TODO.md` header, and a new
`CHANGELOG.md` entry. This is a `scripts/`-only cycle, so record that the **wheel is unchanged from its predecessor** (`pyproject.toml:131` packages `silly_kicks` only).

Verify:
```bash
grep -n "^version" pyproject.toml; grep -n "__version__" silly_kicks/__init__.py; head -6 TODO.md
```
All must agree.

- [ ] **Step 6: Full gate, then propose the final commit**

```bash
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m pyright --pythonpath .venv/Scripts/python.exe
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --tb=short -q
```

```bash
git add docs/ CLAUDE.md TODO.md CHANGELOG.md pyproject.toml silly_kicks/__init__.py uv.lock scripts/_partition.py
git commit -m "docs: ADR for the corpus-driver contract + version bump"
```

---

## Task 19: The acceptance criterion — kill and resume

> **Every command in this task runs with `--allow-dirty`.** The acceptance run happens on the
> working branch before the final commit, so `require_clean_tree` refuses it otherwise — the guard
> doing exactly its job. The artifacts it produces are throwaway and correctly stamped
> `dirty: true` / `tree_state: "dirty"` (Task 16b); nothing here is a registered deliverable.

Tests passing is not the acceptance criterion for a resilience cycle.

- [ ] **Step 1: Prepare the existing shards**

The `build_gkdv_arm_values` shards from the TF-19 corpus pass sit flat in their shard directory; the migrated driver looks in `shard_root/<token>/`. Run the driver once on a two-match slice to learn the generated token name (it is derived, not chosen), then `mkdir` that directory and move the existing shards into it. Filenames are unchanged, so this is a path prefix only.

- [ ] **Step 2: Kill mid-run**

Launch the driver on a small slice, wait for two or three per-item progress lines, then interrupt it.

- [ ] **Step 3: Restart and verify**

Re-launch with the identical command. Confirm from the progress output that it skips exactly the items already sharded, resumes at the right one, and that the final manifest's totals describe the whole corpus rather than one partition.

- [ ] **Step 4: Verify the token actually invalidates**

Change one declared `token_inputs` value, re-run, and confirm a **new** generation directory appears and the pass recomputes. This is the `cache_token` scenario finally exercised rather than reasoned about.

- [ ] **Step 5: Report the result**

Record what was killed, what resumed, and the two generation directory names in the PR description. If resume did not work, that is a blocker regardless of a green suite.

---

## Self-review against the spec

**Spec coverage.** §4.1's `for_each`, seven primitives, generation directory, validated key, tidy-frame contract, empty-shard invariant, conservation invariant, and combined-table placement → Tasks 1–7, plus Task 6b for the manifest field `aggregate_manifests` was measured to drop. §4.2 token contract → Tasks 1 and 14 Step 3. §4.3 cohort cache → Tasks 8 and 15. §4.4 auth → Task 16. §5 adoption gate → Task 9. §6 rollout steps 1–9 → Tasks 8–18. §7 ceilings → Task 18 Step 2. §8 kill-and-resume → Task 19. §10 ruthless-efficiency → adopted, not deferred: `_token` delegates to `ruthless.fingerprint` (Task 1), and Task 1 Step 0 raises the declared floor to the version that actually exports it. §11 risks → mitigations are embedded (Task 10 before Task 11; Task 13 early), **including `--prune-stale`, which the risk table named as the mitigation for disk accumulation and which nothing built until Task 1b.**

**Three spec claims that had no task, now closed.** Each was an assertion in the spec that no code
would have satisfied: `--prune-stale` (§4.1, §11 → Task 1b), `aggregate_manifests` surfacing the
generation (§4.1 → Task 6b), and the `ruthless` floor implied by delegating the token (§4.2 → Task 1
Step 0). A fourth, `train_gk_completion`'s guard, was written against machinery that does not exist
in this tree at all — Task 14b now builds it and says so.

**Known gap, stated rather than hidden.** Task 14 is a template covering thirteen drivers rather than thirteen fully-expanded tasks. The template contains complete code and a per-driver `token_inputs` rule, but the *specific* declared inputs for ten of those thirteen are not enumerated — they cannot be without reading each driver, which is the first step of each migration.

**One deliberate non-fix, surfaced not decided.** `train_gk_completion`'s corpus-identity check
carries the mirror of the defect Task 14b repairs: a pure translation moves `_mean` by metres and
makes `assert_allclose(atol=0.05)` **raise** on a change that provably leaves every served
probability identical (measured). It is left alone because `_CORPUS_IDENTITY_ATOL` was deliberately
relaxed to 0.05 in 4.21.4 for a recorded reason, so re-keying it is a decision about a released
calibration gate and belongs to its owner.

**Type consistency.** `CorpusPassResult` fields (`shard_dir`, `attempted`, `skipped`, `failed`, `failures`, `counters`) are used consistently in Tasks 7, 11, 12 and 14. `for_each`'s keyword names match between the signature in Task 7 and every call site. `manifest_fields(generation, *, attempted, failed)` is called with `attempted=res.attempted` at every site — true attempts, deliberately EXCLUDING skips (Task 6b explains why the conservation relation is the wrong quantity here).
