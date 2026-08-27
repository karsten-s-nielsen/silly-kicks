# Scale-guard harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, general sub-quadratic-growth test harness so CI catches a scale-only O(n²) regression (the class ADR-068 fixed but CI could not see), apply it to the 12 highest-risk primitives, and add the 6 mismatched-dtype characterization tests the ADR-068 review deferred.

**Architecture:** A new `assert_subquadratic_growth(measure_work, ...)` in `tests/_perf_structural.py` asserts the empirical growth exponent of a deterministic operation count stays ≤ 1.5. A `rows_scanned_counter()` context manager counts three public-pandas seams (boolean `__getitem__`, `.groupby`, `.take`) as the rescan proxy. A `SCALE_GUARDED` registry + meta-assertions force coverage of `group_rows` callers and forbid vacuous guards. Tests + docs only — no production change.

**Tech Stack:** Python, pytest, pandas, numpy. Test-only; no new runtime dependency. `math.log` for the exponent. `unittest.mock` / context-manager monkeypatching for counters.

**Spec:** `docs/superpowers/specs/2026-08-25-scale-guard-harness-design.md` (reviews #1 and #2 incorporated).

## Global Constraints

- **Deterministic only** — integer op-counts, never wall-clock. No `assert ms < budget` anywhere.
- **No production behaviour change** — tests + docs + one CLAUDE.md bullet + one ADR. No retrain, C4-free.
- **No new runtime dependency.**
- **The harness is a quadratic-ish detector** — default `max_exponent=1.5`; sub-n^1.5 regressions are out of scope by design.
- **Every counter isolates the super-linear-suspect op** (spec decision 4) — never mix a large linear co-term into the counted work.
- **Every guard proves non-degeneracy** (`work[max] >= work_floor`) — a registered-but-vacuous guard must fail.
- **One commit on one feature branch.** The task order below is review/execution ordering ONLY — never commit boundaries. **No task contains a `git commit` step.**
- **CI-faithful gate** before ready-to-commit: full `pytest -m "not e2e"` (no `--benchmark-skip`), `ruff check` + `ruff format --check` on `silly_kicks/ tests/ scripts/`, bare `python -m pyright`; then `/final-review`.

---

## File Structure

- **Modify** `tests/_perf_structural.py` — add `assert_subquadratic_growth`, `rows_scanned_counter`, `_is_boolean_key`.
- **Create** `tests/test_perf_structural.py` — the harness self-test (planted O(n)/O(n·log n)/O(n²)/mixed-term/in-loop-rebuild/key-discrimination/non-degeneracy).
- **Create** `tests/_scale_guarded.py` — the `SCALE_GUARDED` registry.
- **Create** `tests/test_scale_guard_registry.py` — the three meta-assertions.
- **Create** `tests/test_scale_guards.py` — the 11 growth/constant adopter tests (rows 1–7, 9–12) that don't have a natural home file; reuse existing per-site fixtures where they exist.
- **Modify** the 6 site test files for Batch-1 dtype tests (or one new file — Task 1 decides).
- **Create** `docs/superpowers/adrs/ADR-0NN-subquadratic-growth-guard.md` (number at commit-prep).
- **Modify** `CLAUDE.md` — one "Key conventions" bullet.

---

## Task 1: Batch 1 — mismatched-dtype characterization tests (6 group_rows raw-`==`→canonical sites)

**Files:**
- Create: `tests/test_group_rows_consumer_dtype.py`

**Interfaces:**
- Consumes: `silly_kicks._frame_index.group_rows` (already shipped); the 6 consumer functions.
- Produces: nothing consumed downstream (leaf tests).

**Rationale:** each site replaced a raw `==` with canonical `group_rows`. A test that passes a **mismatched-dtype** lookup key (int column vs `str` key, and reverse) and asserts a **non-empty / correct** result would FAIL under the old raw `==` (which returned empty) and PASS under canonical matching — so it characterizes the intended ADR-019 behaviour and is discriminating by construction.

- [ ] **Step 1: Write the seam-level dtype characterization tests** (the seam is the shared mechanism; per-site fixtures are heavy, so pin the behaviour at the `group_rows` seam that all 6 sites route through, plus one representative end-to-end site).

```python
"""ADR-068 review (agent 2): the 6 group_rows sites that replaced a raw `==` match
canonically (ADR-019). A mismatched-dtype key must still match -- the old raw `==`
returned empty. Discriminating by construction: assert non-empty on a cross-dtype key."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks._frame_index import group_rows


@pytest.mark.parametrize(
    "col_dtype, key",
    [
        ("int64", "1"),        # int column, str key
        ("string", 1),         # str column, int key
        ("Int64", "1"),        # nullable-int column, str key
    ],
)
def test_group_rows_matches_across_dtype_at_every_consumer_key(col_dtype, key):
    # A single-key lookup and a multi-key lookup -- the two shapes the 6 sites use.
    df = pd.DataFrame({"k": pd.array([1, 1, 2], dtype=col_dtype), "v": [10, 20, 30]})
    got = group_rows(df, "k").get(key)
    assert not got.empty, f"canonical match must survive {col_dtype} col vs {type(key).__name__} key"
    assert got["v"].tolist() == [10, 20]
```

- [ ] **Step 2: Add a multi-key cross-dtype case** (mirrors `_gk_identification`/`_confounders`, which key on tuples).

```python
def test_group_rows_multikey_matches_across_dtype():
    df = pd.DataFrame({"g": [1, 1, 2], "t": [5, 5, 6], "v": [1, 2, 3]})
    got = group_rows(df, ("g", "t")).get("1", "5")   # str keys vs int columns
    assert got["v"].tolist() == [1, 2]
```

- [ ] **Step 3: Add SIX per-site END-TO-END mismatched-dtype tests — one per `group_rows` site (spec §4.5; owner-restored 2026-08-25).** Each builds the consumer's input from its existing fixture, makes the `group_rows` **join key** differ in dtype from the grouped column, runs the consumer, and asserts a **non-degenerate** result (matched rows, not the silent-empty the old raw `==` produced). Template (the `_off_ball_runs` case):

```python
def test_off_ball_runs_survives_mismatched_game_id_dtype():
    from tests.tracking.test_off_ball_run_perf import _two_game_fixture
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    actions, frames = _two_game_fixture()
    actions = actions.copy()
    actions["game_id"] = actions["game_id"].astype(str)  # int frames vs str actions -> raw `==` misses
    out = _off_ball_runs_kernel(actions, frames, home_team_id=1)
    assert out is not None and len(out) == len(actions)  # matched, not silently empty
```

Write the other five to this shape. The dtype mismatch is **natural** where the key is cross-source (action↔frame) and **constructed** where the key is same-source (the implementer reads each loop to pick the right construction; a same-source site's test still characterizes that `group_rows` canonicalizes — it may be a documented no-op there, which is itself worth pinning):

| site | consumer | fixture source | mismatch (natural = cross-source) | assert non-degenerate |
|---|---|---|---|---|
| `opportunities` | `build_opportunities` | `tests/causal/test_opportunities_perf.py` (`frames(...)`, `actions([])`) | frames `frame_id` / the iterated key dtype (read the loop; same-source ⇒ construct a str-vs-int key) | opportunity rows produced (not empty) |
| `_skillcorner_inference` | `infer_defensive_actions` | the skillcorner-inference fixture | `(period, team_id)` key dtype vs column | inferred actions present |
| `_confounders` | `_pressure_at_entry` | `tests/causal/test_confounders_perf.py::_frames_and_spells` | spells `game_id`→str (frames int) — cross-source, natural | carriers resolved (not all `NA`) |
| `_gk_identification` | `derive_goalkeepers` | `tests/tracking/test_gk_identification_perf.py` frames | `(game_id, team_id)` key dtype vs frames (same-source ⇒ construct) | GK picks non-empty |
| `defensive_credit/_resolution` | `compute_defensive_credits` | `tests/tracking/test_defensive_credit_perf.py::_shot_scene` | actions `frame_id`→str (frames int) — cross-source, natural | credits produced (non-empty) |

- [ ] **Step 4: Run** `python -m pytest tests/test_group_rows_consumer_dtype.py -q` → all pass (canonical behaviour already ships; these characterize it per-site). If a consumer **raises** on the dtype tweak instead of matching, that is a real ADR-019 gap in that consumer — investigate and fix before proceeding (it means the consumer mangles the key before the seam, which is exactly what per-site coverage exists to catch).

---

## Task 2: Batch 2a — the growth harness `assert_subquadratic_growth`

**Files:**
- Modify: `tests/_perf_structural.py`
- Create: `tests/test_perf_structural.py`

**Interfaces:**
- Produces: `assert_subquadratic_growth(measure_work, *, sizes=(256,1024,4096), max_exponent=1.5, work_floor=1, degenerate_ok=False, label="") -> float` (returns the exponent on pass; raises `AssertionError` on fail/degeneracy).

- [ ] **Step 1: Write the self-test** (`tests/test_perf_structural.py`) — the planted witnesses, both sides, per spec §4.6.

```python
import math
import pytest
from tests._perf_structural import assert_subquadratic_growth

SIZES = (256, 1024, 4096)

def test_passes_linear():
    assert_subquadratic_growth(lambda n: n, sizes=SIZES)

def test_passes_nlogn():
    assert_subquadratic_growth(lambda n: n * max(int(math.log2(n)), 1), sizes=SIZES)

def test_catches_pure_quadratic():
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(lambda n: n * n, sizes=SIZES)

def test_red_witness_is_robust_mixed_term_quadratic():
    # RED gate is n^2 + 100n (exp ~1.89), asserted with a >=1.6 margin so a later
    # `sizes` change cannot silently defang the harness's own catch-proof (R1).
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(lambda n: n * n + 100 * n, sizes=SIZES)
    counts = [n * n + 100 * n for n in SIZES]
    exp = math.log(counts[-1] / counts[0]) / math.log(SIZES[-1] / SIZES[0])
    assert exp >= 1.6, f"RED witness margin eroded: exp={exp:.3f}"

def test_reference_boundary_values_not_a_gate():
    # Documented monotonicity/reference (R1) -- NOT the pass/fail gate.
    for fn, expected in [(lambda n: n * n + 1000 * n, 1.51), (lambda n: n * n + 10000 * n, 1.11)]:
        counts = [fn(n) for n in SIZES]
        exp = math.log(counts[-1] / counts[0]) / math.log(SIZES[-1] / SIZES[0])
        assert abs(exp - expected) < 0.02, f"reference drift: {exp:.3f} vs {expected}"

def test_non_degeneracy_enforced():
    with pytest.raises(AssertionError, match="work_floor"):
        assert_subquadratic_growth(lambda n: 0, sizes=SIZES)

def test_degenerate_ok_opt_in_passes():
    assert_subquadratic_growth(lambda n: 0, sizes=SIZES, degenerate_ok=True) is None

def test_two_sizes_accepted():
    assert_subquadratic_growth(lambda n: n, sizes=(1500, 10000))
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_perf_structural.py -q` → FAIL (`assert_subquadratic_growth` not defined / ImportError).

- [ ] **Step 3: Implement `assert_subquadratic_growth`** in `tests/_perf_structural.py`.

```python
import math

def assert_subquadratic_growth(measure_work, *, sizes=(256, 1024, 4096),
                               max_exponent=1.5, work_floor=1,
                               degenerate_ok=False, label=""):
    """Assert a primitive's deterministic work-count grows sub-quadratically.

    measure_work(n) -> int builds a size-n input, runs the primitive with a work
    counter installed, and returns the observed integer op-count. Asserts the
    extreme-pair growth exponent log(work_hi/work_lo)/log(size_hi/size_lo) <=
    max_exponent. Requires work[max] >= work_floor unless degenerate_ok. Returns
    the exponent on pass. Reference exponents at (256,1024,4096): linear 1.0,
    n*log n 1.16, n^1.5 1.50, quadratic 2.0. See the design spec.
    """
    if len(sizes) < 2:
        raise ValueError("assert_subquadratic_growth needs >= 2 sizes")
    counts = [int(measure_work(n)) for n in sizes]
    lo, hi = counts[0], counts[-1]
    n_lo, n_hi = sizes[0], sizes[-1]
    if hi < work_floor or hi == 0:   # a 0-count is ALWAYS degenerate (avoids math.log(0)) -- M2
        if degenerate_ok:
            return None
        raise AssertionError(
            f"{label or 'assert_subquadratic_growth'}: work_floor not met "
            f"(work[{n_hi}]={hi} < {work_floor}) -- the counter never fired, so this is a "
            f"mis-wired guard, not a passing one. Counts {dict(zip(sizes, counts))}. "
            f"Pass degenerate_ok=True with a reason for a genuinely zero-work primitive."
        )
    exponent = math.log(hi / max(lo, 1)) / math.log(n_hi / n_lo)
    assert exponent <= max_exponent, (
        f"{label or 'assert_subquadratic_growth'}: growth exponent {exponent:.3f} > "
        f"{max_exponent} -- super-linear scaling. Counts {dict(zip(sizes, counts))}."
    )
    return exponent
```

- [ ] **Step 4: Run to verify it passes** — `python -m pytest tests/test_perf_structural.py -q` → all pass. (`test_two_sizes_accepted`, `test_degenerate_ok_opt_in_passes`, the RED-margin, and the reference-value tests confirm the API contract.)

- [ ] **Step 5: Prove RED-before-GREEN discrimination** — temporarily set `max_exponent=3.0` in the function, re-run: `test_catches_pure_quadratic` and `test_red_witness...` must now FAIL (the harness stops catching). Restore `max_exponent=1.5`. This confirms the self-test has teeth, not just green.

---

## Task 3: Batch 2b — `rows_scanned_counter`

**Files:**
- Modify: `tests/_perf_structural.py`
- Modify: `tests/test_perf_structural.py`

**Interfaces:**
- Produces: `rows_scanned_counter()` (context manager yielding `{"n": int}`); `_is_boolean_key(key) -> bool`.

- [ ] **Step 1: Write the counter self-test** (append to `tests/test_perf_structural.py`).

```python
import numpy as np
import pandas as pd
from tests._perf_structural import rows_scanned_counter

def test_boolean_mask_counts_label_select_does_not():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    with rows_scanned_counter() as c:
        _ = df["a"]                 # label -> 0
        _ = df[["a", "b"]]          # label list -> 0
        _ = df[df["a"] > 50]        # boolean mask -> +100
        _ = df[np.array([True] * 50 + [False] * 50)]  # boolean ndarray -> +100
    assert c["n"] == 200

def test_int_array_getitem_is_label_select_not_scan():
    # r2/M1: df[int_list] is COLUMN selection by label -- only valid on an int-column frame,
    # and it is NOT a row rescan -> counts 0. (A string-column frame raises KeyError on df[[0,1]],
    # which is exactly why M1's original one-liner on {"a","b"} crashed.)
    df = pd.DataFrame({0: range(100), 1: range(100), 2: range(100)})
    with rows_scanned_counter() as c:
        _ = df[[0, 1]]              # int labels -> column select -> 0
    assert c["n"] == 0

def test_groupby_and_take_count():
    df = pd.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3]})
    with rows_scanned_counter() as c:
        df.groupby("g")            # +3
        df.take([0, 1])            # +2
    assert c["n"] == 5

def _rebuild_in_loop(n):           # the S4 regression: m ~ n items, groupby rebuilt each
    df = pd.DataFrame({"g": np.arange(n) % 10, "x": np.arange(n)})
    with rows_scanned_counter() as c:
        for _ in range(n):
            df.groupby("g")
    return c["n"]

def _build_once(n):                # the fixed pattern: groupby once, O(1) lookups
    df = pd.DataFrame({"g": np.arange(n) % 10, "x": np.arange(n)})
    with rows_scanned_counter() as c:
        g = df.groupby("g").indices
        for _ in range(n):
            _ = g  # O(1)
    return c["n"]

def test_in_loop_groupby_rebuild_is_caught():
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(_rebuild_in_loop, sizes=(64, 128, 256), label="in-loop-rebuild")

def test_build_once_passes():
    assert_subquadratic_growth(_build_once, sizes=(64, 128, 256), label="build-once")

def test_relative_growth_is_the_robust_gate():
    # PRIMARY, pandas-version-independent (rev2): a boolean-mask rescan-in-loop GROWS (exp ~2); a
    # column-select-in-loop does NOT. The absolute == 200/0/5 above are SECONDARY and pinned to the
    # observed pandas -- uv.lock spans pandas 2.3.3 (py<3.11) and 3.0.2, whose internal .take routing
    # can differ, so the growth property (not a magic number) is the durable assertion.
    def _mask_in_loop(n):
        df = pd.DataFrame({"k": np.arange(n) % 10, "x": np.arange(n)})
        with rows_scanned_counter() as c:
            for v in range(n):
                _ = df[df["k"] == (v % 10)]
        return c["n"]

    def _colselect_in_loop(n):
        df = pd.DataFrame({"a": np.arange(n), "b": np.arange(n)})
        with rows_scanned_counter() as c:
            for _ in range(n):
                _ = df[["a", "b"]]
        return c["n"]

    with pytest.raises(AssertionError):
        assert_subquadratic_growth(_mask_in_loop, sizes=(64, 128, 256))          # rescan -> exp ~2
    assert_subquadratic_growth(_colselect_in_loop, sizes=(64, 128, 256), degenerate_ok=True)  # 0
```
(Note: the S4 self-test uses smaller sizes `(64,128,256)` — the O(n²) `_rebuild_in_loop` at 4096 is 16.7M groupbys, too slow; exp≈2 is already unambiguous at 256.)

- [ ] **Step 2: Run to verify it fails** — the two counter tests + the S4 pair FAIL (`rows_scanned_counter` undefined).

- [ ] **Step 3: Implement `_is_boolean_key` + `rows_scanned_counter`** in `tests/_perf_structural.py`.

```python
from contextlib import contextmanager

def _is_boolean_key(key) -> bool:
    """True only for a boolean mask/array row filter -- NOT a label/column or int-array select."""
    if isinstance(key, pd.Series):
        return key.dtype == bool
    if isinstance(key, np.ndarray):
        return key.dtype == bool
    if isinstance(key, list) and key:
        return all(isinstance(x, (bool, np.bool_)) for x in key)
    return False

@contextmanager
def rows_scanned_counter():
    """Count rows touched by boolean-mask __getitem__ / .loc[mask], .groupby construction,
    and .take -- the rescan proxy. Installed only for the with-block (restored in finally)."""
    from pandas.core.indexing import _LocIndexer
    counts = {"n": 0}
    real_getitem, real_take, real_groupby = (
        pd.DataFrame.__getitem__, pd.DataFrame.take, pd.DataFrame.groupby)
    real_loc = _LocIndexer.__getitem__

    depth = {"mask": 0}   # re-entrancy guard: pandas routes df[mask] through an internal .take (rev2)

    def _getitem(self, key):
        if _is_boolean_key(key):
            counts["n"] += len(self)
            depth["mask"] += 1
            try:
                return real_getitem(self, key)
            finally:
                depth["mask"] -= 1
        return real_getitem(self, key)

    def _take(self, indices, *a, **k):
        axis = k.get("axis", a[0] if a else 0)
        # Count ONLY a direct ROW take (axis 0, NOT inside a mask-getitem): this skips both the
        # re-entrant mask-take double-count and every axis=1 column-selection take. The
        # group_rows .get()->df.take(pos) path is a direct axis-0 take -> still counts (rev2).
        if depth["mask"] == 0 and axis in (0, "index"):
            counts["n"] += len(indices)
        return real_take(self, indices, *a, **k)

    def _groupby(self, *a, **k):
        counts["n"] += len(self)
        return real_groupby(self, *a, **k)

    def _loc(self, key):
        k0 = key[0] if isinstance(key, tuple) else key
        if _is_boolean_key(k0):
            counts["n"] += len(self.obj)
            depth["mask"] += 1
            try:
                return real_loc(self, key)
            finally:
                depth["mask"] -= 1
        return real_loc(self, key)

    pd.DataFrame.__getitem__, pd.DataFrame.take, pd.DataFrame.groupby = _getitem, _take, _groupby
    _LocIndexer.__getitem__ = _loc
    try:
        yield counts
    finally:
        pd.DataFrame.__getitem__, pd.DataFrame.take, pd.DataFrame.groupby = (
            real_getitem, real_take, real_groupby)
        _LocIndexer.__getitem__ = real_loc
```

- [ ] **Step 4: Run to verify it passes** — `python -m pytest tests/test_perf_structural.py -q` → all pass. If `_LocIndexer.__getitem__` patching interacts badly (pandas may route `.loc` through a cached property), fall back to counting only the three `DataFrame` seams and drop the `.loc[mask]` seam — document the reduction; the group_rows sites use `.get`→`.take` and boolean `__getitem__`, not `.loc[mask]`, so coverage is unaffected. Confirm `ruff`/`pyright` clean on the file.

---

## Task 4: Batch 3 — the 9 `rows_scanned_counter` growth adopters (rows 1–7, 11, 12)

**Files:**
- Create: `tests/test_scale_guards.py`

**Interfaces:**
- Consumes: `assert_subquadratic_growth`, `rows_scanned_counter`; each adopter function; existing per-site fixture builders (extended to take `n`).

**Template** — every one of the 9 is this shape; only `build_input(n)` + the call differ:

```python
def measure_<name>(n):
    inp = _build_<name>_input(n)              # size-n input (extend the existing fixture helper)
    with rows_scanned_counter() as c:
        <call the adopter on inp>
    return c["n"]

def test_<name>_is_subquadratic():
    assert_subquadratic_growth(measure_<name>, label="<name>")
```

- [ ] **Step 1: Implement the 9 adopter measures + tests** using this table. Where a size-parametrized fixture builder does not yet exist, add a small `_build_<name>_input(n)` (scale the existing single-scene fixture to `n` rows/frames/actions).

| # | adopter | input builder (source) | call |
|---|---|---|---|
| 1 | `_pressure_at_entry` | extend `tests/causal/test_confounders_perf.py::_frames_and_spells(n)` | `C._pressure_at_entry(spells, frames, _stub_add_pressure)` |
| 2 | `build_opportunities` | extend `tests/causal/test_opportunities_perf.py` frames helper to n frames | `O.build_opportunities(frames, actions([]), home_team_id=5, model_metadata=META)` |
| 3 | `compute_defensive_credits` | scale `tests/tracking/test_defensive_credit_perf.py::_shot_scene()` to n actions | `compute_defensive_credits(actions, frames, xg_column="xg", xt=fitted_xt)` |
| 4 | `infer_defensive_actions` | scale the skillcorner-inference fixture to n rows | `infer_defensive_actions(...)` |
| 5 | `_off_ball_runs_kernel` | scale `_two_game_fixture()` frames to n | `_off_ball_runs_kernel(actions, frames, home_team_id=1)` |
| 6 | `derive_goalkeepers` | scale `tests/tracking/test_gk_identification_perf.py` frames to n | `derive_goalkeepers(frames)` |
| 7 | `detect_off_ball_runs` | reuse #5's scaled frames | `detect_off_ball_runs(actions, frames)` |
| 11 | `add_possessions` | `tests/test_benchmark.py::_make_spadl_actions(n)` (exists) | `spu.add_possessions(actions)`; sizes `(1500, 10000)` |
| 12 | `atomic add_possessions` (n10) | atomic actions builder scaled to n | `atomic_spu.add_possessions(actions)` |

Expected: every one exponent ≤ 1.5 (all are O(n) after ADR-068). `max_exponent` override only if a genuine n·log n sort dominates — with a stated reason.

**Fixture-reach notes (s3 — verify exact signatures before running the snippets):** `#1` `_frames_and_spells(n_frames)` (the kwarg is `n_frames`, not `n`); `#3` `compute_defensive_credits` needs a `fitted_xt` (wire the `fitted_xt` fixture/source and the `compute_defensive_credits`-shaped shot scene — there are two `_shot_scene` helpers in the tree, pick the one that supplies `xg`/`xt`); `#6` `derive_goalkeepers(frames)`; `#4` `infer_defensive_actions` has **no named existing fixture** in the perf tests (unlike the other eight) — budget a fresh `_build_skillcorner_inference_input(n)` scaling a minimal `obe_regains`/events frame to n rows; the Task-6 `_ref` companion signature is `_ref(actions, xg_column, *, col, same_is_goal)` (pass the keyword args). Name each helper correctly so the embedded snippets run first try.

- [ ] **Step 2: Prove each is discriminating (RED once)** — for ONE representative (`#5 _off_ball_runs_kernel`), temporarily replace the `group_rows(frames, "game_id")` call in the source with the old `frames[frames["game_id"] == game_id]`-in-loop, run `measure` at the sizes, confirm exponent → ~2 (FAIL). Restore. (Documents that the counter+harness catch the real regression on a real adopter, not just the synthetic self-test.)

- [ ] **Step 3: Run** `python -m pytest tests/test_scale_guards.py -q` → all pass.

---

## Task 5: Batch 3 — turnover adopter #9 (`_opp_first_shot_scan`)

**Files:**
- Modify: `tests/test_scale_guards.py`

**Interfaces:**
- Consumes: `silly_kicks.xtgk._turnover._opp_first_shot_scan` (pure-Python), `_equality_codes`.

- [ ] **Step 1: Build a break-binding, discriminating fixture + counting-array measure.**

```python
import numpy as np

class _CountingArray:
    """1-D array wrapper counting element reads -- proxy for inner-scan work (spec §4.2)."""
    def __init__(self, arr, counts):
        self._a = np.asarray(arr); self._c = counts
    def __getitem__(self, i):
        self._c["n"] += 1
        return self._a[i]
    def __len__(self):
        return len(self._a)

def _turnover_fixture(n, *, window):
    # n events across MANY matches with FINITE window so the inner breaks bind (k bounded).
    # ~50 events/match, a turnover every ~10 events, times spaced so the window cuts the scan.
    import pandas as pd
    from silly_kicks.xtgk._turnover import _equality_codes
    game = np.repeat(np.arange(n // 50 + 1), 50)[:n]
    t = (np.arange(n) % 50).astype(float)          # resets each match -> window binds
    poss = np.arange(n) // 3
    team = np.where((np.arange(n) // 5) % 2 == 0, 0, 1)
    typ = np.zeros(n, dtype=np.int64); typ[np.arange(n) % 17 == 0] = 1  # some shots
    xg = np.zeros(n); turn = (np.arange(n) % 10 == 0)
    win = np.inf if window is None else float(window)
    return dict(turn=turn.astype(bool),
                game=_equality_codes(pd.Series(game)), poss=_equality_codes(pd.Series(poss)),
                team=_equality_codes(pd.Series(team)), typ=typ, xg=xg, t=t, shot=1, window=win)

def measure_turnover_scan(n):
    from silly_kicks.xtgk._turnover import _opp_first_shot_scan
    f = _turnover_fixture(n, window=5.0)           # FINITE window -> O(n)
    counts = {"n": 0}
    # Wrap the array read on EVERY inner iteration (spec decision 4: inner-j-dominant work). The
    # kernel's first inner check is `game[j] != game[i]`, so game_c is read every inner step; the
    # `if not turn[i]: continue` rows read no game. This is inner-j dominant with a <=0.1n outer
    # co-term (game_c[i] re-read per step; coefficient ~0.1, harmless -- exp stays ~1 on a
    # break-binding fixture; s2). The implementer VERIFIES which array is read first in the kernel.
    _opp_first_shot_scan(f["turn"], _CountingArray(f["game"], counts), f["poss"], f["team"],
                         f["typ"], f["xg"], f["t"], f["shot"], f["window"])
    return counts["n"]

def test_turnover_scan_is_subquadratic():
    assert_subquadratic_growth(measure_turnover_scan, sizes=(256, 1024, 4096), label="turnover_scan")
```

- [ ] **Step 2: Prove the fixture's breaks actually BIND (S2b — T2 fix).** A bare triangular loop is O(n²) for *any* input, so a fixture-ignoring "no-break" test proves nothing about THIS fixture. Instead assert the REAL kernel's counted work at the max size is far below the triangular bound `n²/2` — i.e. the finite window / match boundaries genuinely cut the scan. Without this, a fixture too weak to make the kernel O(n) would fail Step-1 on the *correct* code while a tautological discriminating test still passed — the exact blind spot S2b targets.

```python
def test_turnover_fixture_breaks_actually_bind():
    n = 4096
    work = measure_turnover_scan(n)   # the REAL pure-Python kernel on the finite-window fixture
    assert work < 0.1 * n * n, (
        f"breaks not binding: counted {work} vs triangular {n * n // 2} -- the fixture's window/"
        "match boundaries must cut the scan, else Step-1 would fail on the CORRECT code (S2b)."
    )
```
(Optional belt-and-suspenders: a faithful break-STRIPPED copy of the real kernel run on the same fixture dict via the same `_CountingArray`, asserted `exp > 1.5`. The `< 0.1·n²` bound is the minimum proof that the breaks bind.)

- [ ] **Step 3: Run** `python -m pytest tests/test_scale_guards.py -k turnover -q` → both pass (production scan sub-quadratic AND the fixture's breaks proven to bind, so the O(n) verdict is about the algorithm on realistic data, not a too-small fixture).

---

## Task 6: Batch 3 — possession-labels adopter #10 (`_possession_labels`)

**Files:**
- Modify: `tests/test_scale_guards.py`

**Interfaces:**
- Consumes: `silly_kicks.vaep.labels._scores_possession`; `pandas.core.indexing._LocIndexer.__getitem__` via `call_counter`.

- [ ] **Step 1: Write the growth test** (reuse the `_single_possession(k)` builder in `tests/vaep/test_labels_possession_perf.py`, which the existing scale-independence test already uses).

```python
def measure_possession_labels(n):
    import pandas.core.indexing as _idx
    from silly_kicks.vaep.labels import _scores_possession
    from tests.vaep.test_labels_possession_perf import _single_possession
    from tests._perf_structural import call_counter
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    calls = call_counter(mp, _idx._LocIndexer, "__getitem__")
    try:
        _scores_possession(_single_possession(n), "xg")
    finally:
        mp.undo()
    return calls["n"]

def test_possession_labels_loc_is_subquadratic():
    # Vectorized path issues ZERO `.loc` in the hot path (verified: labels.py has no scalar .loc) ->
    # measure is 0 at every size, so this is degenerate-BY-DESIGN. Use work_floor=1 (default) +
    # degenerate_ok=True (M2: work_floor=0 would crash on math.log(0)); a constant/zero .loc IS the
    # guarantee. Registry qualname is `_possession_labels` (the guarded primitive); the test drives
    # it through the `_scores_possession` public wrapper (labels.py:357, delegates).
    assert_subquadratic_growth(measure_possession_labels, sizes=(64, 256, 1024),
                               max_exponent=1.5, degenerate_ok=True, label="possession_labels_loc")
```
Because #10 is `degenerate_ok`, its Step-2 discriminating companion below is a **HARD requirement**, not optional (T1): a zero-work guard MUST be paired with a companion proving the counter distinguishes — else "counts 0" is indistinguishable from a mis-wired counter. The Task-8 non-degeneracy meta-assertion exempts `degenerate_ok` entries **only if** they carry such a companion.

- [ ] **Step 2: Add the MANDATORY discriminating companion** (`test_possession_labels_ref_loop_is_superlinear` — the exact nodeid `DEGENERATE_OK` points at, T1). It measures the verbatim pre-ADR-068 `_ref` nested loop, whose growing scalar `.loc` proves the counter DISTINGUISHES the vectorized path (0 `.loc`) from the O(k²) one.

```python
def test_possession_labels_ref_loop_is_superlinear():
    import pandas.core.indexing as _idx
    import pytest as _pt
    from tests._perf_structural import assert_subquadratic_growth, call_counter
    from tests.vaep.test_labels_possession_perf import _ref, _single_possession

    def measure_ref(n):
        mp = _pt.MonkeyPatch()
        calls = call_counter(mp, _idx._LocIndexer, "__getitem__")
        try:
            _ref(_single_possession(n), "xg", col="scores", same_is_goal=True)  # old O(k^2) .loc loop
        finally:
            mp.undo()
        return calls["n"]

    with _pt.raises(AssertionError):   # O(k^2) .loc -> exp ~2 -> harness must reject it
        assert_subquadratic_growth(measure_ref, sizes=(32, 64, 128), label="possession_ref_loc")
```

- [ ] **Step 3: Run** `python -m pytest tests/test_scale_guards.py -k possession -q` → both pass (the guard is degenerate-by-design green; the companion proves the counter has teeth).

---

## Task 7: Batch 3 — databricks constant-query guard #8 (`load_matches`)

**Files:**
- Modify: `tests/test_scale_guards.py`

**Interfaces:**
- Consumes: `scripts._loader_databricks` (its `_query_param` seam, already spied in `tests/scripts/test_loader_databricks_batch.py`).

- [ ] **Step 1: Write the equality guard** (constant query count regardless of match count — NOT the exponent harness, per s8). Reuse the fake-cursor scaffolding from `tests/scripts/test_loader_databricks_batch.py`.

```python
def test_load_matches_query_count_is_constant_in_match_count(monkeypatch):
    import pandas as pd
    import scripts._loader_databricks as ld
    from tests.scripts.test_loader_databricks_batch import _FakeConn  # module-level, EXISTS (M3)

    def _count_queries(n_matches):
        seen = []
        all_frames = pd.DataFrame({"match_id": list(range(n_matches)), "frame_id": 0, "x": 1.0})
        all_events = pd.DataFrame({"match_id": list(range(n_matches)), "ev": 0})

        def _fake_query(cur, sql, params=None):     # inline fake -- no phantom import (M3)
            seen.append(sql)
            return all_frames.copy() if "T_TRACK" in sql else all_events.copy()

        monkeypatch.setattr(ld, "_connect", lambda: _FakeConn())
        monkeypatch.setattr(ld, "_table", lambda p, kind: {"tracking": "T_TRACK", "events": "T_EVT"}[kind])
        monkeypatch.setattr(ld, "_convert", lambda p, e, f: (e, f, "home"))
        monkeypatch.setattr(ld, "_query_param", _fake_query)
        list(ld.load_matches(providers=["skillcorner"],
                             match_ids={"skillcorner": [str(i) for i in range(n_matches)]},
                             tracking_limit=None))
        return len(seen)

    assert _count_queries(2) == _count_queries(8) == 2  # one IN-list query per table, always
```
(The inline fake mirrors `tests/scripts/test_loader_databricks_batch.py`'s own `_fake_query`; only `_FakeConn` is imported — it is module-level and real. Verify the exact `_convert`/`_table` monkeypatch set against the current `load_matches` body.)

- [ ] **Step 2: Run** `python -m pytest tests/test_scale_guards.py -k load_matches -q` → pass.

---

## Task 8: Batch 4 — `SCALE_GUARDED` registry + meta-assertions

**Files:**
- Create: `tests/_scale_guarded.py`
- Create: `tests/test_scale_guard_registry.py`

**Interfaces:**
- Produces: `SCALE_GUARDED: dict[str, str]` (guarded qualname → test nodeid); `group_rows_callers() -> set[str]`.

- [ ] **Step 1: Write `tests/_scale_guarded.py`** — the registry (all 12 adopters) + the AST discovery of `group_rows` callers.

```python
"""Registry of scale-guarded primitives + AST discovery of group_rows callers (spec §4.3)."""
import ast, pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: guarded primitive qualname -> the test that guards it (growth or constant). ENUMERATE ALL 12
#: from the Task 4-7 adopter table (rows 1-12); each qualname is the primitive, each value the
#: test nodeid. Example first + last shown; the implementer fills the middle 10 verbatim.
SCALE_GUARDED: dict[str, str] = {
    "silly_kicks.causal._confounders._pressure_at_entry": "tests/test_scale_guards.py::test__pressure_at_entry_is_subquadratic",
    # rows 2-11 (build_opportunities, compute_defensive_credits, infer_defensive_actions,
    # _off_ball_runs_kernel, derive_goalkeepers, detect_off_ball_runs, _opp_first_shot_scan,
    # _possession_labels, add_possessions x2) -> their test nodeids ...
    "scripts._loader_databricks.load_matches": "tests/test_scale_guards.py::test_load_matches_query_count_is_constant_in_match_count",
}

#: Entries that are degenerate-by-design (zero counted work IS the guarantee) -> their MANDATORY
#: discriminating companion test (T1). #10 is the only one this cycle.
DEGENERATE_OK: dict[str, str] = {
    "silly_kicks.vaep.labels._possession_labels": "tests/test_scale_guards.py::test_possession_labels_ref_loop_is_superlinear",
}

def group_rows_callers() -> set[str]:
    """Every function that CALLS group_rows in silly_kicks/ + scripts/ (AST; excludes the def site)."""
    out: set[str] = set()
    for base in ("silly_kicks", "scripts"):
        for py in (_ROOT / base).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            mod = str(py.relative_to(_ROOT).with_suffix("")).replace("/", ".").replace("\\", ".")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for sub in ast.walk(node):
                        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                                and sub.func.id == "group_rows"):
                            if not (mod.endswith("_frame_index") and node.name == "group_rows"):
                                out.add(f"{mod}.{node.name}")
    return out
```

- [ ] **Step 2: Write the meta-assertions** (`tests/test_scale_guard_registry.py`).

```python
import importlib
from tests._scale_guarded import SCALE_GUARDED, DEGENERATE_OK, group_rows_callers

def _has_test(nodeid: str) -> bool:
    parts = nodeid.split("::")           # tolerate a future ::Class::method entry (minor rev2)
    modpath, testname = parts[0], parts[-1]
    mod = importlib.import_module(modpath.removesuffix(".py").replace("/", ".").replace("\\", "."))
    return hasattr(mod, testname)

def test_registry_is_superset_of_group_rows_callers():
    missing = group_rows_callers() - set(SCALE_GUARDED)
    assert not missing, f"group_rows callers with no scale guard: {sorted(missing)}"

def test_registry_entries_resolve_to_collected_tests():
    # s1: import + getattr (no subprocess, exact-name match, self-burning-down).
    for qual, nodeid in SCALE_GUARDED.items():
        assert _has_test(nodeid), f"stale registry entry: {qual} -> {nodeid}"

def test_degenerate_entries_carry_a_discriminating_companion():
    # T1: non-degeneracy for ordinary entries is enforced by the harness `work_floor` at TEST time
    # (a vacuous guard's own test fails -> CI red), so no meta-test re-runs measures. A
    # degenerate-BY-DESIGN entry (zero counted work, e.g. #10) is exempt from work_floor ONLY if
    # paired with a companion proving the counter distinguishes -- enforce that pairing here.
    for qual, companion in DEGENERATE_OK.items():
        assert qual in SCALE_GUARDED, f"{qual} in DEGENERATE_OK but not registered"
        assert _has_test(companion), f"{qual} is degenerate_ok but companion {companion} is missing"
```

- [ ] **Step 3: Run to verify it fails first** — with an intentionally-incomplete `SCALE_GUARDED` (drop one group_rows caller), `test_registry_is_superset...` FAILS naming the missing caller. Restore the full registry.

- [ ] **Step 4: Run** `python -m pytest tests/test_scale_guard_registry.py -q` → all pass.

---

## Task 9: Batch 4 — ADR + CLAUDE.md convention

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-subquadratic-growth-guard.md` (number assigned at commit-prep from merged origin/main)
- Modify: `CLAUDE.md` (one "Key conventions" bullet)

- [ ] **Step 1: Write the ADR** from `docs/superpowers/adrs/ADR-TEMPLATE.md` — Context (the scale blind-spot; the 4.92.0 turnover bug reached a lakehouse report not CI), Decision (operation-count growth harness, scoped counters, `group_rows`-caller meta-assertion, non-degeneracy), Alternatives (absolute work-bound; generic always-on hook; the **declined AST rescan-lint** with the ADR-019 id-compat precedent), Consequences (tests+docs only, no retrain, C4-free) + the stated **known limits** (sub-n^1.5 out of scope; large-linear-co-term masking; new non-group_rows rescan not force-caught).

- [ ] **Step 2: Add the CLAUDE.md bullet** under "Key conventions":

```
- **Item/row-looping primitives carry a sub-quadratic-growth guard (ADR-0NN).** `tests/_perf_structural.assert_subquadratic_growth(measure_work, ...)` asserts the empirical operation-count growth exponent stays <= 1.5 (deterministic integer counts at n=256/1024/4096 -- never wall-clock). The counter isolates the super-linear-suspect op; `rows_scanned_counter` is the rescan proxy (boolean-mask `__getitem__` + `.groupby` + `.take`). A NEW `group_rows` caller MUST register a guard in `tests/_scale_guarded.SCALE_GUARDED` (meta-assertion enforces the superset + non-degeneracy). Known limit: a new rescan that neither routes through `group_rows` nor rebuilds a `groupby` in-loop is not force-caught. The AST rescan-lint was considered and declined (ADR-019 lint lesson).
```

- [ ] **Step 3: Run** `python -m pytest tests/ -m "not e2e" -k "c4 or adr or claude" -q` and any ADR/CLAUDE structure gates → pass. Verify the ADR count/registry gates (if any) accept the new ADR.

---

## Task 10: Final CI-faithful gate + /final-review

**Files:** none (verification only)

- [ ] **Step 1: Ruff** — `python -m ruff format --check silly_kicks/ tests/ scripts/` then `python -m ruff check silly_kicks/ tests/ scripts/` → clean (fix any format/lint on the new test files).
- [ ] **Step 2: Pyright** — `python -m pyright` → 0 errors (capture the real summary line, not a piped exit code).
- [ ] **Step 3: Full suite** — `python -m pytest tests/ -m "not e2e" -q` (no `--benchmark-skip`) → 0 failed. Read the actual summary line.
- [ ] **Step 4: `/final-review`** — the mandatory pre-commit review (code quality + ADR + docs drift + C4). C4 should be unchanged (no new aggregator/subpackage); confirm the aggregator-count + feature-column-count gates stay green.
- [ ] **Step 5: Present** the full `git status --short` + `git diff --stat` + proposed single-commit message and STOP for explicit commit approval. **Do NOT run `git commit`.** Version/PR/ADR numbers assigned at commit-prep from merged `origin/main`.
