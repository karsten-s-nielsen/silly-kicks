# Ghost-GK re-fit onto the canonical box constant + TF-24 refresh — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** Unify `_ghost_gk` onto the canonical penalty-area constant behind one vectorized predicate,
re-fit and re-publish both ghost variants, and refresh TF-24's recommendations on corrected geometry.

**Architecture:** A vectorized sibling of the existing scalar box predicate lands first and absorbs
both vectorized call sites; xCross's migration is provably value-identical and is the proof that the
helper is faithful. Ghost then flips constant + predicate + *declaration* together (they cannot be
split — the contract raises otherwise), re-fits from one extraction, and re-stamps on x86. TF-24
follows on the same materialized corpus.

**Tech Stack:** Python 3.10–3.14, numpy, pandas, pytest, scikit-learn (HGBR), Optuna (TF-24),
`huggingface_hub`. DGX Spark (aarch64) for training; local x86 for stamping and validation.

**Spec:** `docs/superpowers/specs/2026-08-11-ghost-refit-and-tf24-refresh-design.md`

## Global Constraints

* **NEVER run `git commit`.** No task ends in a commit. The owner commits, once, on explicit
  approval. Ordering below is review ordering, not commit granularity.
* **ADR number and version are assigned at COMMIT-PREP off `main`, never pre-claimed.** Registers at
  writing: 4.79.0 / PR-S149 / ADR-059. Re-read `main` at commit-prep.
* **`grep -q add_argument <script>` BEFORE invoking any `scripts/*.py`.** No match ⇒ do not run it.
  `--help` is *ignored* by a parser-less script and `main()` runs. **`stamp_feature_contracts.py` has
  NO parser and stamps all three models** — there is no way to scope it.
* **DGX = training/tuning only** (`ssh karsten@192.168.68.73`, aarch64). Data access, validation and
  stamping run **locally on x86**.
* **Lint at CI scope, never `.`:** `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright` (bare). Neither
  tool is on PATH — always `python -m`.
* **`scripts/` is ASCII-only** (ruff RUF001/2/3 + the cp1252 gate). No `—`, `·`, `≥`, `→`.
* **A driver cannot run in the change that introduces it** — `scripts/_provenance.py` counts untracked
  files as dirty. Land driver code, then run it.
* Run pytest to a **unique** log path with its own exit marker; never share a log between runs.
  `| tail` masks the exit code — always `echo "PYTEST_EXIT=$?"` into the log.

---

## File Structure

| file | responsibility |
|---|---|
| `silly_kicks/tracking/_geometry.py` | **modify.** Add `in_penalty_area_goal_relative_array`, the vectorized sibling. Owns the rule. |
| `tests/tracking/test_geometry_box_predicate_parity.py` | **new.** Durable array-vs-scalar property test. Outlives the cycle. |
| `silly_kicks/tracking/_xcross_attempt.py:251` | **modify.** Migrate to the helper; keep `& ~is_ball` at the call site. |
| `silly_kicks/tracking/_ghost_gk.py` | **modify.** `:237-239` constants deleted, `:671` predicate migrated, `:1550-1554` docstring rewritten, `:1578-1579` declaration migrated. |
| `silly_kicks/tracking/_feature_contract.py:53-59` | **modify.** Prune ghost's three registry entries. |
| `tests/tracking/test_declared_constant_values.py` | **new.** General declared-value test across all three extractors. |
| `scripts/materialize_tc3_frames.py` | **new.** D7 writer + parity assertion. |
| `scripts/measure_box_constant_delta.py` | **new.** D2 driver. |
| `scripts/publish_ghost_gk.py` | **new.** HF upload, mirroring `publish_xcross_attempt.py`. |
| `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` | **modify.** Remove the false claim. |

---

## Task 1: The vectorized predicate and its durable parity test

**Files:**
- Modify: `silly_kicks/tracking/_geometry.py` (after `in_penalty_area_goal_relative`, ~`:130`)
- Test: `tests/tracking/test_geometry_box_predicate_parity.py` (new)

**Interfaces:**
- Produces: `in_penalty_area_goal_relative_array(gr_x: np.ndarray, y: np.ndarray) -> np.ndarray`
  (bool array). Consumed by Tasks 2 and 4.

- [x] **Step 1: Write the failing parity test**

Create `tests/tracking/test_geometry_box_predicate_parity.py`:

```python
"""The array predicate must agree with the scalar one, everywhere.

This is the DURABLE artifact of the migration. The one-off grid sweep in Task 2 is a
characterization test against an expression being deleted -- once xCross migrates, the thing it
compared against is gone. This test is what permanently pins the two forms together.
"""

import numpy as np
import pytest

from silly_kicks.tracking._geometry import (
    GOAL_Y,
    in_penalty_area_goal_relative,
    in_penalty_area_goal_relative_array,
)
from silly_kicks.spadl import config as spadlconfig


def _ulp_neighbourhood(x: float, n: int = 50) -> list[float]:
    """The `2n+1` doubles centred on `x`. Bound EQUALITY does not imply predicate equality (spec 1.1).

    MUST WALK. The obvious comprehension --
    `[np.nextafter(x, inf if d > 0 else -inf) for d in range(-n, n+1)]` -- returns the SAME two
    neighbours over and over: measured 101 entries collapsing to **3 distinct doubles** at n=50, and
    `test_the_grid_is_not_vacuous` (any/not-all) cannot see it. That would gut the one dimension this
    grid exists to cover.
    """
    out, lo, hi = [x], x, x
    for _ in range(n):
        lo = np.nextafter(lo, -np.inf)
        out.append(float(lo))
        hi = np.nextafter(hi, np.inf)
        out.append(float(hi))
    return out


def _grid() -> tuple[np.ndarray, np.ndarray]:
    half = spadlconfig.penalty_area_half_width
    depth = spadlconfig.penalty_area_depth
    ys: list[float] = [0.0, GOAL_Y, 68.0, 13.85, 13.84, 54.16, 54.15]
    for c in (GOAL_Y - half, GOAL_Y + half):
        ys.extend(_ulp_neighbourhood(c))
    # The depth boundary needs the SAME treatment: the `<` -> `<=` flip lives exactly at
    # `gr_x == depth`, so ULP-walking only the y bounds would leave the boundary contributor
    # (spec 1.1 item 2) covered by three hand-picked points.
    xs: list[float] = [-5.0, -0.001, 0.0, 5.0, 16.49, 16.51, 120.0]
    xs.extend(_ulp_neighbourhood(depth))
    gx, gy = np.meshgrid(np.array(xs, dtype=float), np.array(ys, dtype=float))
    return gx.ravel(), gy.ravel()


def test_the_ulp_neighbourhood_actually_walks():
    """Non-vacuity of the GRID ITSELF, not just of its outcomes.

    The distinguishing dimension of this whole test is the doubles either side of a bound. A
    neighbourhood that collapses to 3 values still yields a grid that is neither all-True nor
    all-False, so the outcome-based non-vacuity check below passes while covering nothing.
    """
    assert len(set(_ulp_neighbourhood(13.84, n=50))) == 101


def test_array_form_agrees_with_scalar_everywhere():
    gr_x, y = _grid()
    got = in_penalty_area_goal_relative_array(gr_x, y)
    want = np.array([in_penalty_area_goal_relative(float(a), float(b)) for a, b in zip(gr_x, y)])
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"{bad.size} disagreements, first at gr_x={gr_x[bad[0]]!r} y={y[bad[0]]!r}: "
        f"array={got[bad[0]]} scalar={want[bad[0]]}"
    )


def test_the_grid_is_not_vacuous():
    """A parity test over a grid that is all-True or all-False proves nothing."""
    gr_x, y = _grid()
    got = in_penalty_area_goal_relative_array(gr_x, y)
    assert got.any() and not got.all(), f"grid is degenerate: {got.sum()}/{got.size} True"


@pytest.mark.parametrize("gr_x,y", [(float("nan"), 34.0), (5.0, float("nan")),
                                    (float("nan"), float("nan"))])
def test_nan_is_False_on_both_forms(gr_x, y):
    """SPECIFIED contract, not incidental behaviour: NaN on either argument -> False.

    The scalar form yields this because `NaN <= depth` is False. Pinning it here stops a future
    array implementation (e.g. one using np.abs with a mask) from silently returning True.
    """
    assert in_penalty_area_goal_relative(gr_x, y) is False
    out = in_penalty_area_goal_relative_array(np.array([gr_x]), np.array([y]))
    assert out.dtype == np.bool_ and not bool(out[0])
```

- [x] **Step 2: Run it and verify it fails on the missing symbol**

Run: `python -m pytest tests/tracking/test_geometry_box_predicate_parity.py -q`
Expected: FAIL — `ImportError: cannot import name 'in_penalty_area_goal_relative_array'`

- [x] **Step 3: Implement the helper**

In `silly_kicks/tracking/_geometry.py`, directly below `in_penalty_area_goal_relative`:

```python
def in_penalty_area_goal_relative_array(gr_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized `in_penalty_area_goal_relative`. Same rule, same boundaries, array in/out.

    Exists because the two consuming extractors (`_ghost_gk`, `_xcross_attempt`) operate on numpy
    arrays per frame, and a scalar call per player per frame is a real cost on a 179-match
    extraction. ADR-050 §6 rebound the CONSTANTS at those sites and left the EXPRESSION duplicated;
    it never evaluated a vectorized canonical predicate, which is what this is.

    NaN on either argument yields False, matching the scalar form (`NaN <= depth` is False).
    Pinned by `tests/tracking/test_geometry_box_predicate_parity.py`.
    """
    return (gr_x <= _spadlconfig.penalty_area_depth) & (
        np.abs(y - GOAL_Y) <= _spadlconfig.penalty_area_half_width
    )
```

Verify `numpy` is imported in `_geometry.py`; add `import numpy as np` at the top if absent.

- [x] **Step 4: Run the parity test**

Run: `python -m pytest tests/tracking/test_geometry_box_predicate_parity.py -q`
Expected: PASS (3 parametrized NaN cases + 2 others = 5 passed)

- [x] **Step 5: Lint**

Run: `python -m ruff check silly_kicks/ tests/ && python -m ruff format --check silly_kicks/ tests/`

---

## Task 2: Migrate xCross, and prove it value-identical

**Files:**
- Modify: `silly_kicks/tracking/_xcross_attempt.py:251`
- Test: `tests/tracking/test_xcross_box_migration_identity.py` (new, temporary characterization)

**Interfaces:**
- Consumes: `in_penalty_area_goal_relative_array` from Task 1.
- Produces: nothing. This task exists to prove the helper faithful before ghost depends on it.

- [x] **Step 1: Capture the pre-migration expression as a characterization test**

Create `tests/tracking/test_xcross_box_migration_identity.py`:

```python
"""One-off: the migrated xCross predicate is VALUE-IDENTICAL to the expression it replaces.

Characterization only. It compares against a literal copy of the pre-migration expression, which
ceases to exist in the source after this task -- so it is deliberately temporary. The DURABLE
guarantee lives in test_geometry_box_predicate_parity.py.

If this shows any delta, the spec's premise is wrong and the cycle STOPS (spec D1).
"""

import numpy as np

from silly_kicks.spadl import config as _spc
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._geometry import in_penalty_area_goal_relative_array

_BOX_DEPTH_M = _spc.penalty_area_depth
_BOX_HALF_WIDTH_M = _spc.penalty_area_half_width


def _legacy(gr_x, y):
    """Verbatim copy of _xcross_attempt.py:251's predicate, minus the `& ~is_ball` term."""
    return (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M)


def test_migration_is_byte_identical_over_a_dense_grid():
    xs = np.linspace(-10.0, 30.0, 401)
    ys = np.linspace(0.0, 68.0, 681)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    assert np.array_equal(_legacy(gx, gy), in_penalty_area_goal_relative_array(gx, gy))


def test_grid_covers_both_outcomes():
    xs = np.linspace(-10.0, 30.0, 401)
    ys = np.linspace(0.0, 68.0, 681)
    gx, gy = np.meshgrid(xs, ys)
    out = _legacy(gx.ravel(), gy.ravel())
    assert out.any() and not out.all()
```

- [x] **Step 2: Run it — it must PASS before the edit**

Run: `python -m pytest tests/tracking/test_xcross_box_migration_identity.py -q`
Expected: PASS. This establishes the helper reproduces the current expression **before** anything is
changed. A failure here means STOP — the premise is wrong.

- [x] **Step 3: Migrate the call site**

In `silly_kicks/tracking/_xcross_attempt.py`, replace line 251:

```python
    in_box = (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M) & ~is_ball
```

with:

```python
    in_box = _geo.in_penalty_area_goal_relative_array(gr_x, y) & ~is_ball
```

The helper has no ball concept, so `& ~is_ball` **composes at the call site** — this is not a
one-line symbol swap and attempting one yields a shape error.

Leave `_BOX_DEPTH_M` / `_BOX_HALF_WIDTH_M` (`:79-80`) in place: they are still read by the feature
contract block at `:464-465`, so the "a constant exists iff something reads it" rule still holds.

- [x] **Step 4: Verify xCross is unchanged**

Run:
```
python -m pytest tests/tracking/test_xcross_box_migration_identity.py tests/tracking/ -k "xcross" -q \
  > /tmp/t2_xcross.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/t2_xcross.log; tail -5 /tmp/t2_xcross.log
```
Expected: all pass, `PYTEST_EXIT=0`.

- [x] **Step 5: Verify the stamped contract did NOT move**

Run:
```
git diff --stat silly_kicks/tracking/_xcross_weights/
```
Expected: **empty**. The fingerprint hashes values + constants, not source (spec D1), so a
value-identical migration must leave `metadata.json` untouched. **Any diff here contradicts the
spec's central claim — stop and re-derive.**

---

## Task 3: The general declared-value test (lands RED, before ghost changes)

**Files:**
- Test: `tests/tracking/test_declared_constant_values.py` (new)

**Interfaces:**
- Consumes: nothing. Produces: the guard Task 4 must satisfy.

**Why RED first:** this test would have caught the 20.15/20.16 divergence from the day it was
introduced. Landing it before Task 4 proves it can fail; landing it after would be a gate never
observed failing (spec D4).

- [x] **Step 1: Write the test**

Create `tests/tracking/test_declared_constant_values.py`:

```python
"""Every stamped geometry constant must EQUAL the canonical one.

ADR-050's enumeration gate asserts on constant NAMES, never VALUES
(`test_geometry_constant_enumeration.py` compares key sets at :120 and :158-160). So an extractor
could migrate its PREDICATE to the canonical constant while its DECLARATION still derived from a
local one, and stamp an artifact that lies about the geometry it was fit on -- with every gate
green. That is exactly what ghost did before this cycle (declared 20.15, canonical 20.16).

Keyed on the CANONICAL NAME rather than on where a constant happens to live, so it survives
constants being relocated out of extractor modules -- which is precisely what this cycle does to
ghost, and which would otherwise silently narrow the enumeration gate to xCross alone.
"""

import json
import pathlib

import pytest

from silly_kicks.spadl import config as spadlconfig

_WEIGHTS = {
    "ghost": "silly_kicks/tracking/_ghost_gk_weights/default/metadata.json",
    "xshot": "silly_kicks/tracking/_xshot_weights/default/metadata.json",
    "xcross": "silly_kicks/tracking/_xcross_weights/default/metadata.json",
}

#: Canonical source for each declared contract key.
_CANONICAL = {
    "penalty_area_half_width": lambda: spadlconfig.penalty_area_half_width,
    "penalty_area_depth": lambda: spadlconfig.penalty_area_depth,
}


def _stamped(model: str) -> dict:
    p = pathlib.Path(_WEIGHTS[model])
    meta = json.loads(p.read_text(encoding="utf-8"))
    return meta.get("feature_contract", {}).get("constants", {})


#: Live contract builders -- the CODE-level view, distinct from what is stamped on disk.
def _built(model: str) -> dict:
    from silly_kicks.tracking import _ghost_gk, _xcross_attempt, _xshot_occurrence

    return {
        "ghost": _ghost_gk._feature_contract_block,
        "xshot": _xshot_occurrence._feature_contract_block,
        "xcross": _xcross_attempt._feature_contract_block,
    }[model]()["constants"]


@pytest.mark.parametrize("model", sorted(_WEIGHTS))
def test_declared_values_equal_the_canonical_values(model):
    """ARTIFACT-level: what ships must not lie about the geometry it was fit on."""
    for key, value in _stamped(model).items():
        if key not in _CANONICAL:
            continue  # goal_width etc. have their own canonical source
        assert value == _CANONICAL[key](), (
            f"{model} declares {key}={value} but the canonical value is {_CANONICAL[key]()}. "
            f"An artifact that declares a constant it was not fit on is exactly what ADR-050's "
            f"contract exists to prevent."
        )


@pytest.mark.parametrize("model", sorted(_WEIGHTS))
def test_built_values_equal_the_canonical_values(model):
    """CODE-level, and it carries a DIFFERENT meaning from its artifact-level sibling.

    `_stamped()` reads `metadata.json`, which is only refreshed at stamp time. Between the code
    migration and the re-stamp -- roughly an entire corpus pass -- there would otherwise be no signal
    that the declaration and the predicate now agree, and one red would conflate "the code is wrong"
    with "the artifact is stale". This one goes green the moment the code is fixed; the artifact one
    stays red until the re-stamp. It also fires in the PR that INTRODUCES a future divergence rather
    than in the one that stamps it.
    """
    for key, value in _built(model).items():
        if key not in _CANONICAL:
            continue
        assert value == _CANONICAL[key](), (
            f"{model}'s live contract block declares {key}={value}, canonical is {_CANONICAL[key]()}"
        )


def test_the_check_is_not_vacuous():
    """At least one model must actually declare a key this test covers, or it asserts nothing."""
    covered = [m for m in _WEIGHTS if set(_stamped(m)) & set(_CANONICAL)]
    assert covered, f"no model declares any of {sorted(_CANONICAL)}; this gate is inert"
```

- [x] **Step 2: Run it and RECORD the red**

Run: `python -m pytest tests/tracking/test_declared_constant_values.py -q`

Expected: **BOTH ghost cases FAIL** — `test_declared_values_equal_the_canonical_values[ghost]`
(artifact: stamped 20.15) and `test_built_values_equal_the_canonical_values[ghost]` (code: the live
block derives 20.15 from `(_PENALTY_AREA_Y_MAX - _PENALTY_AREA_Y_MIN)/2`). xshot and xcross pass on
both. Record both failures — they are the acceptance evidence that the guard sees what the key-name
gates cannot.

They then clear at **different times**, which is the point: the code-level one goes green at Task 4,
the artifact-level one not until the Task 8 re-stamp.

---

## Task 4: Ghost — constant, predicate, declaration and registry, together

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:237-239` (delete), `:671` (predicate), `:1550-1554`
  (the contract-block docstring), `:1578-1579` (declaration)
- Modify: `silly_kicks/tracking/_feature_contract.py:53-59` (prune three entries)
- Modify: `tests/tracking/test_geometry_constant_enumeration.py` (stale docstring)
- **Modify: `scripts/train_ghost_gk.py:46-48` — `cache_token()`. FOUND DURING EXECUTION.**
- **Modify: `tests/tracking/test_feature_contract.py` (2 monkeypatch sites) and
  `tests/scripts/test_trainer_cache_and_providers.py` (1). FOUND DURING EXECUTION.**

**Five files, not three — and the two extra were found by running, not by reading.** A `grep` for the
constant names is what surfaces them; the plan's original list was assembled without one, which is
the "complete by heuristic" failure this cycle's own ADR argues against. Run
`grep -rn "_PENALTY_AREA" silly_kicks/ tests/ scripts/ --include=*.py` before starting, and expect
the list to be longer than any file-by-file reading suggests.

**Interfaces:**
- Consumes: `in_penalty_area_goal_relative_array` (Task 1); the RED test from Task 3.
- Produces: a ghost extractor whose predicate and declaration share one source.

**Indivisible.** The contract raises on an unaccompanied flip, so there is no green intermediate
state between these edits.

- [x] **Step 1: Delete the local constants**

In `silly_kicks/tracking/_ghost_gk.py`, delete lines 238-239:

```python
_PENALTY_AREA_Y_MIN = (_FIELD_WIDTH - 40.3) / 2.0
_PENALTY_AREA_Y_MAX = (_FIELD_WIDTH + 40.3) / 2.0
```

and delete `_PENALTY_AREA_X = 16.5` (`:237`).

- [x] **Step 2: Migrate the predicate**

At `:671`, replace:

```python
        in_box = (atk_xs < _PENALTY_AREA_X) & (atk_ys >= _PENALTY_AREA_Y_MIN) & (atk_ys <= _PENALTY_AREA_Y_MAX)
```

with:

```python
        in_box = _geo.in_penalty_area_goal_relative_array(atk_xs, atk_ys)
```

`atk_xs` is already goal-relative (produced by `to_gr_x`), which is what the helper expects.

**The alias is `_geo`, not `_geometry`.** `_ghost_gk.py:35` already reads
`from . import _geometry as _geo`. Writing `_geometry.…` yields a `NameError`; adding a second
import to make it resolve is a duplicate that ruff rejects. Do not add an import.

- [x] **Step 3a: Rewrite the contract-block docstring — it asserts the OPPOSITE of what ships**

`_feature_contract_block`'s docstring at `:1550-1554` sits **25 lines above** the declaration Step 3b
edits, so the worker scrolls straight past it. Every sentence is falsified by this task, it is the
primary in-code explanation of the exact thing the cycle changes, and it lives in a module CI runs
under `--doctest-modules`. Replace:

```
    Ghost's declared half-width evaluates to **20.15**, not the canonical 20.16 -- deliberately.
    Its bundled weights were trained on the 40.3 m box, so unifying the constant without a re-fit
    would skew ``attackers_in_box``, a real trained feature. Recording the divergence is what turns
    the "do not unify before the re-fit" instruction from a comment into a mechanism: after this
    artifact is stamped, flipping the constant makes ``load()`` raise.
```

with (keeping the history, which is the part worth keeping):

```
    Ghost's declared half-width is the canonical 20.16, and its predicate reads the same source --
    ADR-050 §6 closed. It previously declared **20.15** against a 40.3 m box because the bundled
    weights were fit on it; recording that divergence is what made ``load()`` raise on an
    unaccompanied flip, and the re-fit discharged it. VALUES, not merely key names, are pinned by
    ``tests/tracking/test_declared_constant_values.py`` -- the enumeration gate only ever compared
    names, which is how a 20.15 declaration survived against a 20.16 extractor.
```

- [x] **Step 3b: Migrate the declaration to the same source**

At `:1578-1579`, replace:

```python
        constants={
            "penalty_area_half_width": (_PENALTY_AREA_Y_MAX - _PENALTY_AREA_Y_MIN) / 2.0,
            "penalty_area_depth": float(_PENALTY_AREA_X),
        },
```

with:

```python
        constants={
            # Declared from the SAME source the predicate consumes. Deriving these independently is
            # how an artifact comes to declare a constant it was not fit on -- ghost declared 20.15
            # against a 20.16 extractor, and every key-name gate stayed green.
            "penalty_area_half_width": float(spadlconfig.penalty_area_half_width),
            "penalty_area_depth": float(spadlconfig.penalty_area_depth),
        },
```

- [x] **Step 4: Prune the constant registry**

In `silly_kicks/tracking/_feature_contract.py:53-59`, delete these three entries:

```python
    "_PENALTY_AREA_X": "penalty_area_depth",
    "_PENALTY_AREA_Y_MIN": "penalty_area_half_width",
    "_PENALTY_AREA_Y_MAX": "penalty_area_half_width",
```

Nothing reads them after Steps 1-3b, and `test_no_dead_entries_in_either_list` (`:96-100`) asserts
`set(DECLARED_CONSTANT_SOURCES) - found` is empty — leaving them fails that test.

**A coupling this creates, recorded here so the next reader does not re-derive it.** After the prune,
`penalty_area_half_width` and `penalty_area_depth` survive in the registry **only** through xCross's
`_BOX_HALF_WIDTH_M` / `_BOX_DEPTH_M` (`:55-56`). `test_the_registry_and_the_built_contracts_agree`
asserts `all_keys <= registry_keys`, and models still STAMP both keys, so the registry must keep at
least one module constant mapping to each. Two consequences: **Task 2 leaving xCross's aliases in
place is now load-bearing, not incidental**; and a future cycle that applies D4's "generalises beyond
ghost" reasoning to xCross would empty the registry and fail that same assertion from the other
direction. The value test (Task 3) is what should absorb that responsibility if it ever happens.

- [x] **Step 4b: Migrate `cache_token()` — it reads all three deleted constants**

`scripts/train_ghost_gk.py:48` was
`f"v3-box{gg._PENALTY_AREA_Y_MIN:.4f}-{gg._PENALTY_AREA_Y_MAX:.4f}-{gg._PENALTY_AREA_X:.4f}"`, so
after Step 1 the trainer raises `AttributeError` on import. **This is not incidental plumbing:** it
is the guard ADR-050 7 built so the re-fit sequence (*extract, flip the constant, re-run*) cannot
silently reuse the first run's 40.3 m features while stamping a 20.16 m contract — load-bearing for
exactly the re-fit this cycle performs. Point it at the canonical source, preserving the band shape:

```python
    import silly_kicks.spadl.config as _spc
    from silly_kicks.tracking import _geometry as _geo

    lo = _geo.GOAL_Y - _spc.penalty_area_half_width
    hi = _geo.GOAL_Y + _spc.penalty_area_half_width
    return f"v3-box{lo:.4f}-{hi:.4f}-{_spc.penalty_area_depth:.4f}"
```

Verify the token actually MOVED — that movement is the invalidation:
`v3-box13.8500-54.1500-16.5000` → `v3-box13.8400-54.1600-16.5000`.

- [x] **Step 4c: Repoint the three monkeypatch sites**

`tests/tracking/test_feature_contract.py` (2) and `tests/scripts/test_trainer_cache_and_providers.py`
(1) all patch the deleted constants to simulate the 40.3 → 40.32 flip — i.e. they simulate the
migration this task makes permanent. Repoint them at `spadlconfig.penalty_area_half_width`.

**One trap, hit during execution:** the pin test
(`test_ghost_pin_is_enforced_by_a_raise_not_by_prose`) patches to a value that must DIFFER from
whatever the bundled artifact declares. Patching to `20.15` while the artifact still declares 20.15
MATCHES, `load()` succeeds, and the test fails `DID NOT RAISE` — it silently stops detecting the
thing it exists for. Use a value **no artifact has ever been stamped with** (e.g. `19.0`), which
diverges both before and after the re-stamp.

- [x] **Step 5: Fix the now-stale docstring**

In `tests/tracking/test_geometry_constant_enumeration.py`, the pin at `:158-160` documents "ghost's
pair is the 40.3-derived one its weights were fit on". Replace that sentence with:

```
    ghost's pair is now the canonical one (ADR-050 §6 closed): its predicate and its declaration both
    read `spadlconfig`, and the VALUES are pinned by test_declared_constant_values.py -- this gate
    only ever compared key NAMES.
```

- [x] **Step 6: Run the geometry, contract and ghost suites**

Run:
```
python -m pytest tests/tracking/test_geometry_constant_enumeration.py \
  tests/tracking/test_declared_constant_values.py \
  tests/tracking/test_geometry_box_predicate_parity.py \
  tests/tracking/ -k "ghost or contract" -q > /tmp/t4_ghost.log 2>&1
echo "PYTEST_EXIT=$?" >> /tmp/t4_ghost.log; tail -12 /tmp/t4_ghost.log
```

Expected, and the split matters:

- `test_built_values_equal_the_canonical_values[ghost]` — **now PASSES.** The code-level divergence is
  closed the moment the declaration migrates.
- `test_declared_values_equal_the_canonical_values[ghost]` — **still FAILS.** The bundled
  `metadata.json` holds 20.15 until Task 8 re-stamps.
- Ghost `load()` tests — expected to RAISE on the contract mismatch, for the same reason.
- Enumeration + parity — PASS.

**All four are correct at this point.** Record the exact failure list so Task 8 Step 7 can assert it
shrinks to zero.

- [x] **Step 7: Lint and types**

Run: `python -m ruff check silly_kicks/ tests/ && python -m ruff format --check silly_kicks/ tests/ && python -m pyright`

---

## Task 5: The TC3 materializer and its parity assertion (D7)

**Files:**
- Create: `scripts/materialize_tc3_frames.py`
- Test: `tests/scripts/test_materialize_tc3_frames.py` (new)

**Interfaces:**
- Produces: `--out/shards/<generation-token>/<provider>__<match_id>.parquet`, via ADR-052's
  `for_each`, plus `--out/manifest_all.json`.

**REVISED DURING EXECUTION — it must adopt `scripts/_driver.py`, and ADR-052's gate enforces it.**
The first draft was a naive `for … : frames.to_parquet(...)` loop. `test_corpus_driver_resilience`
refused it: *"a driver delegated to a remote box must persist each item so a crash resumes, skip
work already done, and print progress."* Materializing 179 matches and losing them to a crash at
match 150 is precisely the 8.7-hour failure that seam exists to prevent.

**Consequence the naive design hid: the output layout changes.** `for_each` writes
`shard_root/<token>/<key>.parquet`, NOT `{provider}/{id}/frames.parquet`. That is fine for both
consumers — `train_ghost_gk.py:291` has a flat `*.parquet` fallback and a generation directory holds
only shards — but every downstream `--data-dir` must point at the generation directory (trainer,
TF-24) or at `--out` (the delta driver, which searches both). Tasks 7, 8 and 10 reflect this.

**It also writes a manifest**, as every other `for_each` adopter does. That is what makes the corpus
cache checkable: a consumer training on these frames can read which commit produced them and whether
the pass conserved. Use `res.manifest()`, never a hand-written `manifest_fields(...)` — only the
method threads `counters_unrecorded`, so a hand-rolled call defaults it to 0 and a resumed pass
reports a complete corpus it never walked.

**Why this exists:** `_loader_pining.load_matches(cache_dir=...)` persists **raw downloaded provider
artifacts** at `cache_dir/{provider}/{match_id}/` (`:292-294`); it has no `to_parquet` anywhere.
`--data-dir <pining cache>` therefore finds nothing and the trainer `sys.exit(1)`s. The directory
*shape* coincides; the contents do not.

- [x] **Step 1: Write the parity test**

Create `tests/scripts/test_materialize_tc3_frames.py`:

```python
"""The materialized frames must match what the trainer's established input contains.

The trainer's existing input comes from a different pipeline. If the pining parse yields a different
schema, dtype set, or row filtering, the trainer silently fits on DIFFERENT DATA -- the same
train/serve-skew family this cycle exists to close, arriving through the fix for it, and landing
underneath the D2 measurement built to detect trouble.
"""

import pandas as pd
import pytest

from scripts.materialize_tc3_frames import assert_frames_parity


def _frame(**over):
    base = pd.DataFrame(
        {"game_id": [1, 1], "period_id": [1, 1], "frame_id": [1, 2],
         "player_id": ["a", "b"], "x": [1.0, 2.0], "y": [3.0, 4.0], "team_id": ["H", "A"]}
    )
    for k, v in over.items():
        base[k] = v
    return base


def test_identical_frames_pass():
    assert_frames_parity(_frame(), _frame(), match_id="m1")


def test_missing_column_is_rejected():
    with pytest.raises(AssertionError, match="column"):
        assert_frames_parity(_frame().drop(columns=["team_id"]), _frame(), match_id="m1")


def test_row_count_mismatch_is_rejected():
    with pytest.raises(AssertionError, match="row count"):
        assert_frames_parity(_frame().iloc[:1], _frame(), match_id="m1")


def test_dtype_drift_is_rejected():
    with pytest.raises(AssertionError, match="dtype"):
        assert_frames_parity(_frame(x=[1, 2]), _frame(), match_id="m1")


def test_value_drift_is_rejected():
    """The check must compare CONTENT, not just shape -- a schema-equal frame with different
    coordinates is exactly the silent-skew case."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(x=[1.5, 2.0]), _frame(), match_id="m1")


def test_NON_KEY_value_drift_is_also_rejected():
    """Measured defect in the first draft: hashing only the identity columns let `vx` drift from
    0.5 to 99.0 undetected. Ghost's extractor consumes velocity and so does `infer_ball_carrier`, so
    a positions-right / velocities-wrong parse is precisely the silent skew this gate names."""
    ref = _frame()
    ref["vx"] = [0.5, 0.6]
    bad = ref.copy()
    bad["vx"] = [99.0, 0.6]
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(bad, ref, match_id="m1")


def test_duplicate_identity_rows_are_order_insensitive():
    """Two rows tying on every identity column but differing on `vx` must hash the same regardless
    of order. They did not, until the sort key was widened to all columns -- and GS duplicate frames
    make this reachable, at the same spurious-STOP cost as negative zero."""
    dup = _frame()
    dup["frame_id"] = [1, 1]
    dup["player_id"] = ["a", "a"]
    dup["x"] = [1.0, 1.0]
    dup["y"] = [3.0, 3.0]
    dup["team_id"] = ["H", "H"]
    dup["vx"] = [0.5, 9.9]
    assert_frames_parity(dup, dup.iloc[::-1].reset_index(drop=True), match_id="m1")


def test_negative_zero_does_not_trip_a_spurious_failure():
    """`-0.0` and `0.0` hash differently unless normalised. Negative zero is reachable (`105 - x`,
    `68 - y`), and Task 7 treats a parity failure as STOP -- so this would cost a corpus pass."""
    neg, pos = _frame(), _frame()
    neg["x"] = [-0.0, 2.0]
    pos["x"] = [0.0, 2.0]
    assert_frames_parity(neg, pos, match_id="m1")
```

- [x] **Step 2: Run it and verify it fails on the missing module**

Run: `python -m pytest tests/scripts/test_materialize_tc3_frames.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.materialize_tc3_frames'`

- [x] **Step 3: Implement the materializer**

Create `scripts/materialize_tc3_frames.py` (ASCII-only; must have argparse):

```python
"""Materialize pining-loaded tracking frames into the TC3 layout the ghost trainer reads.

`_loader_pining.load_matches(cache_dir=...)` caches RAW provider artifacts under
`cache_dir/{provider}/{match_id}/` and parses frames in memory; it never writes them. The ghost
trainer globs `**/frames.parquet`. This bridges the two so ONE download serves both this cycle's
workstreams, and leaves a reusable corpus cache.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import pandas as pd

from scripts._provenance import git_provenance, require_clean_tree

_KEY_COLUMNS = ("game_id", "period_id", "frame_id", "player_id", "x", "y", "team_id")


def _checksum(df: pd.DataFrame) -> str:
    """Hash EVERY column, ordered deterministically by the identity columns.

    Keying the SORT on `_KEY_COLUMNS` is right; restricting the HASH to them loses the content this
    gate exists to protect. Measured: with only the key columns hashed, a parse that gets positions
    right and drifts `vx` from 0.5 to 99.0 passes -- and ghost's extractor consumes velocity
    (`to_gr_vx`, `_VELOCITY_WINDOW_S`) while `infer_ball_carrier` consumes `vx`/`vy` for its `beta`
    term. One omission here would undermine both Task 8's fits and Task 10's argmax check.

    `+ 0.0` normalises `-0.0` to `0.0`: they hash DIFFERENTLY otherwise. **The reachable source is
    NEGATION, not subtraction** -- `-vx` where `vx == 0.0` yields `-0.0`, whereas `68.0 - 68.0` is
    `+0.0` under round-to-nearest, so the reflection's POSITIONS never produce it. Task 10's velocity
    negation does, on every stationary player. Task 7 treats a parity failure as STOP, so a spurious
    one costs a corpus pass to diagnose.

    Sorting on ALL columns, not just `_KEY_COLUMNS`: measured, two rows tying on every identity
    column but differing on `vx` hash DIFFERENTLY when reordered, because `sort_values` leaves ties
    in input order. That is reachable here -- this repo has a documented GS duplicate-frames issue --
    and it is the same spurious-STOP cost as the negative zero. With every column in the sort key,
    remaining ties are fully identical rows, whose order cannot affect the hash.
    """
    ordered = df.reindex(sorted(df.columns), axis=1)
    float_cols = ordered.select_dtypes(include="floating").columns
    if len(float_cols):
        ordered[float_cols] = ordered[float_cols] + 0.0
    ordered = ordered.sort_values(list(ordered.columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values.tobytes()).hexdigest()


def assert_frames_parity(produced: pd.DataFrame, reference: pd.DataFrame, *, match_id: str) -> None:
    """Raise unless `produced` matches `reference` in schema, dtypes, row count and content."""
    missing = set(reference.columns) - set(produced.columns)
    assert not missing, f"{match_id}: produced frames missing column(s) {sorted(missing)}"
    assert len(produced) == len(reference), (
        f"{match_id}: row count {len(produced)} != reference {len(reference)}"
    )
    for col in reference.columns:
        assert produced[col].dtype == reference[col].dtype, (
            f"{match_id}: dtype drift on {col}: {produced[col].dtype} != {reference[col].dtype}"
        )
    assert _checksum(produced) == _checksum(reference), (
        f"{match_id}: content checksum differs despite matching schema -- the trainer would fit on "
        f"different data than the established pipeline produces"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--reference-parquet", type=pathlib.Path, default=None,
                    help="An existing TC3 frames.parquet to assert parity against before the run.")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    import scripts._loader_pining as pining

    n = 0
    for provider, match_id, _actions, frames, _home in pining.load_matches(
        providers=args.providers, cache_dir=args.cache_dir
    ):
        dest = args.out / provider / str(match_id)
        dest.mkdir(parents=True, exist_ok=True)
        if args.reference_parquet is not None and n == 0:
            assert_frames_parity(frames, pd.read_parquet(args.reference_parquet), match_id=str(match_id))
            print(f"parity OK against {args.reference_parquet}")
        frames.to_parquet(dest / "frames.parquet", index=False)
        n += 1
        print(f"  [{n}] {provider}/{match_id}: {len(frames)} rows", flush=True)
    print(f"materialized {n} matches to {args.out}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the parity tests**

Run: `python -m pytest tests/scripts/test_materialize_tc3_frames.py -q`
Expected: 5 passed.

- [x] **Step 5: Confirm the script satisfies the ADR-052 / provenance gates**

Run: `python -m pytest tests/scripts/ -q > /tmp/t5_scripts.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/t5_scripts.log; tail -5 /tmp/t5_scripts.log`

Expected: 0 failed. **Both new scripts must be enrolled in `ARTIFACT_DRIVERS`** — they consume the
pining corpus (external data) and write documents whose numbers are cited. Enrol, never exempt: an
artifact driver with no guard is the exact defect ADR-056's registry exists to surface.

**Order matters, and it bit during execution.** Enrol *after* adopting `for_each`, not before. The
derivation rule is `(--*out* flag) AND (a call in _WRITE_CALLS)`; once `for_each` owns the writing,
the driver no longer calls `to_parquet` itself and becomes **underivable** — the gate then reports
*"enrolled but no longer derivable"*. The fix is NOT `_UNDERIVABLE` (which
`test_UNDERIVABLE_is_empty` asserts empty) but to write the manifest, which every other adopter does
and which the driver should have anyway. That restores derivability through `write_text` and gives
the cache its provenance in one move.

---

## Task 6: The D2 delta driver

**Files:**
- Create: `scripts/measure_box_constant_delta.py`
- Test: `tests/scripts/test_measure_box_constant_delta.py` (new)

**Interfaces:**
- Consumes: TC3 frames from Task 5.
- Produces: `docs/research/box_constant_delta/metrics.json` with keys `n_rows`, `n_flipped`,
  `n_flipped_band_only`, `n_flipped_boundary_only`, `n_flipped_both`, `n_behind_line`, plus
  `run_commit` / `run_tree_dirty`.

- [x] **Step 1: Write the attribution test**

Create `tests/scripts/test_measure_box_constant_delta.py`:

```python
"""`n_flipped` must be ATTRIBUTABLE, not merely reported.

Spec 1.1: the migration has exactly two contributors -- the 1 cm band (20.15 -> 20.16) and the depth
boundary (`<` -> `<=`). A flip count that cannot be decomposed is a number that cannot be reasoned
about next cycle.
"""

import numpy as np

from scripts.measure_box_constant_delta import classify_flips


_ZERO = {"n_flipped": 0, "n_flipped_band_only": 0, "n_flipped_boundary_only": 0, "n_flipped_both": 0}


def test_band_only():
    # y in the 1 cm strip, x comfortably inside: inside under 20.16, outside under 20.15
    out = classify_flips(np.array([5.0]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_band_only": 1}


def test_boundary_only():
    # exactly on the depth line: `<` excludes, `<=` includes
    out = classify_flips(np.array([16.5]), np.array([34.0]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_boundary_only": 1}


def test_both_causes_is_its_own_bucket():
    """Both changes individually NECESSARY -- neither pure bucket may claim it."""
    out = classify_flips(np.array([16.5]), np.array([34.0 + 20.155]))
    assert out == {**_ZERO, "n_flipped": 1, "n_flipped_both": 1}


def test_unaffected_point_flips_nothing():
    assert classify_flips(np.array([5.0]), np.array([34.0])) == _ZERO


def test_y_in_strip_but_x_outside_does_not_flip():
    """Negative case: the band change is irrelevant when depth already excludes the point."""
    assert classify_flips(np.array([40.0]), np.array([34.0 + 20.155])) == _ZERO


def test_the_shipped_legacy_band_form_is_modelled_not_the_abs_form():
    """THE regression that matters. `y = 13.85` is the ONLY value separating the two legacy forms
    (spec 1.1 item 3): the shipped band says outside, the abs form says inside, canonical says
    inside. So it IS a flip, and a driver modelling legacy with the abs form reports 0 -- an
    undercount at the exact boundary this driver measures."""
    out = classify_flips(np.array([5.0]), np.array([13.85]))
    assert out["n_flipped"] == 1, "the shipped band form was replaced by the abs form"
    assert out["n_flipped_band_only"] == 1
```

- [x] **Step 2: Run it, verify it fails on the missing module**

Run: `python -m pytest tests/scripts/test_measure_box_constant_delta.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [x] **Step 3: Implement the driver**

Create `scripts/measure_box_constant_delta.py` (ASCII-only, argparse, provenance-wired):

```python
"""Measure what the ghost box-constant unification actually changes, before re-fitting.

Spec D2: the re-fit happens either way (the contract raises on an unaccompanied flip), but the SHIP
CLAIM depends on this count. Zero-versus-nonzero is the whole question, so no threshold is needed --
but the count must be ATTRIBUTABLE to the band vs the depth boundary.

Also counts the behind-the-line population (gr_x < 0) while the rows are in hand: ADR-050 parks
"should a behind-the-line point count as in-box", and answering it later would otherwise cost a
second corpus pass.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from scripts._provenance import git_provenance, require_clean_tree
from silly_kicks.spadl import config as spadlconfig

_LEGACY_HALF = 40.3 / 2.0
_LEGACY_DEPTH = 16.5


def _legacy_y_in_band(y: np.ndarray) -> np.ndarray:
    """The MIN/MAX BAND form ghost actually ships (`_ghost_gk.py:671`) -- NOT the abs form.

    Do not "simplify" this to `np.abs(y - 34.0) <= 40.3/2`. Spec 1.1 item 3 proves the two forms
    equivalent at the CANONICAL constant and explicitly records that they DISAGREE at the LEGACY one,
    at exactly `y = 13.85`: the double sits fractionally below `(68-40.3)/2`, so the band says
    outside while the abs form says inside. Modelling legacy with the abs form makes that row a
    no-flip when it is a real flip -- an UNDERCOUNT at precisely the boundary this driver exists to
    measure.
    """
    return (y >= (68.0 - 40.3) / 2.0) & (y <= (68.0 + 40.3) / 2.0)


def classify_flips(gr_x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """Split the legacy-vs-canonical disagreement by CAUSE, three ways.

    A three-way split states a fact; a two-way split forces a convention. A row where both changes
    are individually NECESSARY (y in the 1 cm strip AND x exactly on the depth line) belongs to
    neither pure bucket, and folding it into one makes the other a systematic undercount.
    """
    legacy_y = _legacy_y_in_band(y)
    canon_y = np.abs(y - 34.0) <= spadlconfig.penalty_area_half_width
    legacy_x = gr_x < _LEGACY_DEPTH
    canon_x = gr_x <= spadlconfig.penalty_area_depth

    legacy = legacy_x & legacy_y
    canonical = canon_x & canon_y
    flipped = legacy != canonical

    y_agrees = legacy_y == canon_y
    x_agrees = legacy_x == canon_x
    return {
        "n_flipped": int(flipped.sum()),
        "n_flipped_band_only": int((flipped & ~y_agrees & x_agrees).sum()),
        "n_flipped_boundary_only": int((flipped & y_agrees & ~x_agrees).sum()),
        "n_flipped_both": int((flipped & ~y_agrees & ~x_agrees).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = git_provenance()
    require_clean_tree(prov, allow_dirty=args.allow_dirty)

    xs_all, ys_all = [], []
    for pq in sorted(args.data_dir.glob("**/frames.parquet")):
        df = pd.read_parquet(pq, columns=["x", "y"])
        xs_all.append(df["x"].to_numpy(dtype=float))
        ys_all.append(df["y"].to_numpy(dtype=float))
    gr_x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)

    out = classify_flips(gr_x, y)
    out["n_rows"] = int(gr_x.size)
    out["n_behind_line"] = int((gr_x < 0).sum())
    out["run_commit"] = prov.commit
    out["run_tree_dirty"] = prov.dirty

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the tests**

Run: `python -m pytest tests/scripts/test_measure_box_constant_delta.py -q`
Expected: 4 passed.

---

## Task 7: Run the corpus pass on the DGX (materialize, then measure)

**Files:** none modified. This task produces `docs/research/box_constant_delta/metrics.json`.

**Interfaces:** consumes Tasks 5 and 6; produces the artifact Task 8's ship claim cites.

**Prerequisite:** Tasks 1-6 must be tracked in git before this runs — `scripts/_provenance.py`
counts untracked files as dirty and `require_clean_tree` refuses. **Ask the owner to commit first.**

**DGX mechanics that are NOT optional** (all documented from prior runs; ignoring any one of them
fails on first contact):

* **`ssh box 'cmd'` runs with a STRIPPED PATH.** `uv` is not found and `which python` gives only
  `/usr/bin/python3`, which has **no pandas**. Use ABSOLUTE interpreter paths. Do not conclude the
  box is unprovisioned.
* **`pyarrow` is NOT a base dependency.** Task 5 writes `.to_parquet` and Task 6 reads it — both fail
  without it.
* **The pining token lives in `~/.pining_env` (600)** and a non-interactive script does NOT inherit
  `.bashrc`. Every runner must `source ~/.pining_env`.
* **PowerShell strips inner double quotes** from remote command strings, silently corrupting anything
  quoted. Ship a **script file** (`scp`, binary-safe) and run `bash /tmp/x.sh`. Do not inline
  non-trivial quoted commands.
* **Never mix a tar/archive pipe with a backgrounded `nohup &` in one ssh call** — the stream
  truncates. Ship, then launch separately.
* **A failed `ssh` is NOT "the process ended."** Poll with an explicit token:
  `st=$(ssh box 'pgrep -f "[m]aterialize_tc3" >/dev/null && echo LIVE || echo DEAD')` and treat only
  `DEAD` as finished — empty means the ssh failed, so retry. Re-check the process ended **before**
  `scp`-ing results; a live run serves the previous run's file.

- [ ] **Step 1: Ship the code and provision the venv**

```bash
ssh karsten@192.168.68.73 'mkdir -p ~/silly-kicks-refit'
git archive --format=tar HEAD | ssh karsten@192.168.68.73 'tar -x -C ~/silly-kicks-refit'
```

Then, as a separate call (never combined with the pipe above):

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && /home/karsten/.local/bin/uv venv .venv && \
  /home/karsten/.local/bin/uv pip install -e . pyarrow "pandas==2.3.3" "kloppy>=3.18" scipy scikit-learn'
```

`pandas==2.3.3` is pinned deliberately — the box otherwise resolves pandas 3.0.x.

- [ ] **Step 2: Confirm the token and the interpreter before spending hours**

```bash
ssh karsten@192.168.68.73 'test -f ~/.pining_env && echo TOKEN_PRESENT || echo TOKEN_MISSING'
ssh karsten@192.168.68.73 '~/silly-kicks-refit/.venv/bin/python -c "import pandas, pyarrow, silly_kicks; print(pandas.__version__, silly_kicks.__version__)"'
```

If `TOKEN_MISSING`, write it without exposing the value in argv:

```bash
printf 'export PINING_FOR_THE_DATA_TOKEN=%q\n' "$PINING_FOR_THE_DATA_TOKEN" \
  | ssh karsten@192.168.68.73 'umask 077; cat > ~/.pining_env'
```

- [ ] **Step 3: Materialize the corpus via a script file (one download, both workstreams)**

Write `/tmp/materialize.sh` locally (LF endings), `scp` it, then run it:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.pining_env
cd ~/silly-kicks-refit
mkdir -p ~/logs
nohup ./.venv/bin/python scripts/materialize_tc3_frames.py \
  --cache-dir ~/pining-cache --out ~/tc3-cache \
  --providers skillcorner idsse gradientsports \
  --reference-parquet ~/tc3-reference/frames.parquet \
  > ~/logs/materialize.log 2>&1 &
echo "PID $!"
```

```bash
scp /tmp/materialize.sh karsten@192.168.68.73:/tmp/materialize.sh
ssh karsten@192.168.68.73 'bash /tmp/materialize.sh'
```

**Do not proceed until the log prints `parity OK`.** If the assertion raises, STOP — the trainer
would fit on different data than the established pipeline produces (spec D7).

**PRODUCE the reference; do not ship without it and do not go looking for one.** No
`~/tc3-reference/frames.parquet` exists on the box (verified 2026-08-12: `find ~/ -maxdepth 4 -name
frames.parquet` returns one hit, `~/sk_gs_et_out/frames.parquet`, which is the TF-23b GS
extra-time output, not a TC3 corpus reference). The established producer is already in the repo —
`scripts/_loader_pining_to_cache.py` (PR-S81), which wrote the input the currently-bundled weights
were fit on. One match is enough:

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  ./.venv/bin/python scripts/_loader_pining_to_cache.py \
    --providers skillcorner --max-per-provider 1 --out ~/tc3-reference-src'
REF=$(ssh karsten@192.168.68.73 'ls ~/tc3-reference-src/skillcorner/*/frames.parquet | head -1')
```

Pass `$REF` as `--reference-parquet`. **State its power honestly in the write-up:** both pipelines
consume the SAME `load_matches` generator and write the yielded frames unchanged, so this cannot
detect a divergent PARSE — there is only one parse. It checks the WRITE path: schema, row count,
dtypes and a full-content checksum surviving the parquet round-trip. Spec D7's phrase "the trainer's
established input comes from a different pipeline" is wrong, and the assertion is worth running
anyway.

- [ ] **Step 3b: ONE-MATCH END-TO-END SMOKE — materialize, then TRAIN, before the corpus pass**

**Non-negotiable, and cheap.** Task 7 costs hours and Task 8 costs hours; the plan previously ran
them back to back with nothing in between, so a trainer-side input error surfaced only after both
were paid for. Two such errors were found statically on 2026-08-12 (the missing home-team map and
the dropped actions), and neither needed real data to find — which is the argument for a smoke that
needs almost none.

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  ./.venv/bin/python scripts/materialize_tc3_frames.py \
    --cache-dir ~/pining-cache --out ~/tc3-smoke \
    --providers skillcorner --max-per-provider 1'
SGEN=$(ssh karsten@192.168.68.73 'ls -d ~/tc3-smoke/shards/*/ | head -1')
ssh karsten@192.168.68.73 "cd ~/silly-kicks-refit && ./.venv/bin/python scripts/train_ghost_gk.py \
  --data-dir $SGEN --output-dir ~/ghost-smoke \
  --home-teams ~/tc3-smoke/home_teams.json --actions-dir ~/tc3-smoke/_actions \
  --variant default --subsample-cap 2000 --training-platform dgx-spark-aarch64"
```

Required in the smoke log, all four:
`Home team mapping: 1 games` · `Loaded actions for 1 games` · no `SKIP game` line · a written
artifact. Then **delete `~/tc3-smoke`** so its one-match generation cannot be mistaken for the
corpus later.

- [ ] **Step 4: Measure the delta**

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  ./.venv/bin/python scripts/measure_box_constant_delta.py --data-dir ~/tc3-cache --out ~/box_delta'
mkdir -p docs/research/box_constant_delta
scp karsten@192.168.68.73:~/box_delta/metrics.json docs/research/box_constant_delta/metrics.json
```

**`--data-dir` is `~/tc3-cache`, the materializer's `--out` — not a `frames.parquet` tree.**
`for_each` writes `shard_root/<generation-token>/<key>.parquet`, so the driver searches
`**/frames.parquet` first and falls back to `**/*.parquet`, which finds the shards beneath
`~/tc3-cache/shards/<token>/`. Verified during execution: the original `**/frames.parquet`-only glob
found nothing under a `for_each` output and exited reporting an empty corpus, which on a remote box
reads as "no data" rather than "wrong directory".

- [ ] **Step 5: Record which ship claim applies**

Read `n_flipped`:
- `0` → the PR ships as **"unification, measured no-op"**, citing `0 / n_rows`.
- `> 0` → the PR carries the band/boundary split and a before/after weights comparison.

Write a short `docs/research/box_constant_delta/README.md` stating the count, the split, the
behind-the-line population, and which claim was selected.

- [ ] **Step 6: DECIDE the behind-the-line question on the number, and record it as a DECISION**

`in_penalty_area_goal_relative*` has no `0 <= gr_x` guard, so points beyond the goal line count as
in-box. ADR-050 parked this; parking it again without looking at `n_behind_line` is how a question
becomes permanent. The driver emits `n_behind_line` and `n_rows` for exactly this.

**Read `n_behind_line / n_rows`, then record ONE of these in the README and in the ADR — never
another "not this cycle's" comment:**

- **Retain unbounded** (the expected outcome): state the measured fraction and the revisit trigger.
- **Clamp**: this is a SCOPE EXPANSION and needs owner approval before any of Task 8 runs.

**Three facts the decision needs, all verified 2026-08-12 — do not re-derive them under time
pressure:**

1. **Clamping pulls xCross in.** The two array call sites take a SIGNED `gr_x`, so a guard moves
   ghost's `attackers_in_box` AND xCross feature #6. Both are trained. xCross is otherwise NOT
   being re-fit this cycle, so clamping turns a one-model re-fit into a two-model re-fit and
   republish.
2. **`in_penalty_area_absolute` is unaffected** — it folds with `abs()` before calling in, so its
   `gr_x` is never negative and `defensive_credit` sees no change. Exposure differs by entry point.
3. **The feature contract will NOT catch it.** Measured: `_ghost_gk._feature_contract_block()` is
   byte-identical with and without the clamp, because a lower bound declares no new constant and
   the probe frame carries no behind-the-line player. ADR-050 catches a CONSTANT change, not a
   predicate-SHAPE change. **If the decision is to clamp, the re-fit is enforced by discipline
   alone** — nothing in CI will fail to remind you.

**Do not fold a clamp into this cycle silently.** Task 8 re-fits ghost; a predicate change riding
along would be indistinguishable, in the resulting weights, from the constant unification this
cycle exists to measure.

---

## Task 8: Re-fit both variants, re-stamp on x86

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk_weights/default/` (regenerated)

**Interfaces:** consumes Task 7's materialized frames. Produces the artifacts Task 9 publishes.

**Resolve the shard generation directory FIRST — the trainer cannot read `~/tc3-cache` directly.**
`train_ghost_gk.py:289` globs `**/frames.parquet` (finds nothing under a `for_each` output) and
`:291` falls back to a flat `*.parquet` — which requires pointing at the generation directory
itself, not its parent:

```bash
GEN=$(ssh karsten@192.168.68.73 'ls -d ~/tc3-cache/shards/*/ | head -1')
echo "generation dir: $GEN"    # e.g. ~/tc3-cache/shards/<16-hex-digest>/
```

Use `$GEN` as `--data-dir` below. A generation directory contains only per-item `.parquet` shards
plus `.json` sidecars, so the flat glob picks up exactly the frames and nothing else.

> **`--data-dir` ALONE IS NOT ENOUGH, and the failure is late and total.** The trainer resolves the
> home-team map from a `meta.json` BESIDE each parquet (`train_ghost_gk.py:379-387`) — present in
> the TC3 tree layout, absent from a flat generation. With no map it `sys.exit(1)`s at `:388`,
> *after* Task 7's corpus pass has been paid for. It also takes per-game actions via
> `--actions-dir` (`:359`) and feeds them to `prepare_ghost_gk_training_data(actions=...)` (`:534`),
> recording `with_actions` in the artifact metadata — so omitting them trains on different inputs
> than the established pipeline used, silently. `materialize_tc3_frames.py` therefore emits
> `home_teams.json` and `_actions/` under its `--out`; **both flags below are mandatory.**

- [ ] **Step 1: Fit the bundled `default` variant on the DGX**

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  nohup ./.venv/bin/python scripts/train_ghost_gk.py \
    --data-dir "$GEN" --output-dir ~/ghost-default \
    --home-teams ~/tc3-cache/home_teams.json \
    --actions-dir ~/tc3-cache/_actions \
    --variant default --subsample-cap 36000 \
    --training-platform dgx-spark-aarch64 \
    > ~/ghost_default.log 2>&1 &'
```

Confirm from the log before letting it run: `Home team mapping: N games` with N equal to the
materialized match count, and `Loaded actions for N games`. A `SKIP game <id>: no home_team_id`
line means the map is short and the fit is running on a truncated corpus.

- [ ] **Step 2: Fit the `full` variant from the SAME extraction**

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  nohup ./.venv/bin/python scripts/train_ghost_gk.py \
    --data-dir "$GEN" --output-dir ~/ghost-full \
    --home-teams ~/tc3-cache/home_teams.json \
    --actions-dir ~/tc3-cache/_actions \
    --variant full --training-platform dgx-spark-aarch64 \
    --skip-permutation-importance \
    > ~/ghost_full.log 2>&1 &'
```

`--variant` is only a metadata label; `--subsample-cap` is the real axis and is applied AFTER
extraction (`train_ghost_gk.py:648-656`), so both fits reuse one extraction.

- [ ] **Step 3: Bring both artifacts back to x86**

```bash
scp -r karsten@192.168.68.73:~/ghost-default/* silly_kicks/tracking/_ghost_gk_weights/default/
scp -r karsten@192.168.68.73:~/ghost-full/ /tmp/ghost-full/
```

`full` is excluded from wheel and sdist (`pyproject.toml:147,154`), so it stays out of the package
tree and is published from `/tmp` in Task 9.

- [ ] **Step 4: Re-stamp the feature contracts — ON X86**

**`stamp_feature_contracts.py` has NO argparse and stamps ALL THREE models.** It cannot be scoped.

```bash
grep -q add_argument scripts/stamp_feature_contracts.py && echo "HAS PARSER" || echo "NO PARSER - expected"
python scripts/stamp_feature_contracts.py
```

- [ ] **Step 5: Assert the blast radius — positively AND negatively**

```bash
git diff --name-only silly_kicks/tracking/
git diff --quiet silly_kicks/tracking/_ghost_gk_weights/default/metadata.json \
  && echo "FAIL: ghost metadata did NOT change" || echo "OK: ghost metadata changed"
git diff --quiet silly_kicks/tracking/_xshot_weights/ silly_kicks/tracking/_xcross_weights/ \
  && echo "OK: xshot/xcross byte-unchanged" || echo "FAIL: xshot/xcross moved"
```

`_xshot_weights/` and `_xcross_weights/` were re-stamped too (the script cannot be scoped) but their
values did not move, so they must be byte-identical — any diff means Task 2's value-identity claim
was wrong. **The positive assertion is not ceremony:** `git diff --name-only` reports only changed
*tracked* paths, so if these artifacts were ever gitignored or LFS-managed it would print nothing and
pass vacuously. Assert that ghost's metadata **did** change.

- [ ] **Step 6: Stamp-parity for the `full` variant — the artifact this step would otherwise miss**

**`save()` writes the contract itself** (`_ghost_gk.py:1977`:
`"feature_contract": _feature_contract_block()`), so an artifact is stamped **wherever it is
trained**. `stamp_feature_contracts.py`'s `TARGETS` are hard-coded to the three
`<model>_weights/default` directories (`:36-51`) and it has no parser to point elsewhere. Therefore:

| artifact | contract written | re-stamped on x86 | published |
|---|---|---|---|
| `default` | DGX (aarch64) | **yes**, Step 4 | Task 9 |
| `full` | DGX (aarch64) | **no** — `/tmp/ghost-full` is not a TARGET | Task 9 |

D6 says *"x86 stamps"*. That holds for `default` and is **silently violated for `full`** — the
artifact that reaches the public through `from_hub`. Step 5 cannot catch it: `/tmp/ghost-full` is
outside the tree, so the diff passes while blind to half the deliverable.

Rather than extend `TARGETS` (adding a parser to a script whose parser-less-ness is now a documented
global constraint is its own reviewable change), **assert the platform inheritance instead**:

```bash
python - <<'PY'
import json, pathlib
from silly_kicks.tracking import _ghost_gk
local = _ghost_gk._feature_contract_block()                     # recomputed HERE, on x86
carried = json.loads(pathlib.Path("/tmp/ghost-full/metadata.json").read_text())["feature_contract"]
assert local["constants"] == carried["constants"], (local["constants"], carried["constants"])
assert local["probe_sha256"] == carried["probe_sha256"]
assert local["fingerprint"] == carried["fingerprint"], "aarch64 and x86 disagree on the probe vector"
print("full: aarch64-written contract == x86 recompute (platform inheritance DEMONSTRATED)")
PY
```

This converts an **inherited** assumption into this cycle's measurement. The 4.74.0 baseline
(`max_abs_delta = 0.0`) is inherited and its own caveat says the legs confound architecture with
interpreter. Bit-identical → demonstrated for this cycle, and §8's platform-provenance gap gets its
first real datum. **Different → stop**: that is exactly the condition the gap was registered for, and
it would otherwise ship to the Hub unnoticed.

- [ ] **Step 7: The Task 3 and Task 4 reds must now clear**

Run:
```
python -m pytest tests/tracking/test_declared_constant_values.py tests/tracking/ -k "ghost" -q \
  > /tmp/t8_ghost.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/t8_ghost.log; tail -8 /tmp/t8_ghost.log
```
Expected: **0 failed** — `test_declared_constant_values[ghost]` now passes (stamped 20.16), and the
ghost `load()` chirality + contract checks pass against the re-fit artifact.

---

## Task 9: Publish both variants and correct the model card

**Files:**
- Create: `scripts/publish_ghost_gk.py`
- Modify: `docs/huggingface/model-cards/ghost-gk-v1-model-card.md:84`

**Interfaces:** consumes Task 8's artifacts. Mirrors `scripts/publish_xcross_attempt.py`.

**Note:** there is no ghost publish script today — only `publish_xcross_attempt.py` and
`publish_xshot_occurrence.py`. `from_hub` is currently broken: the hosted artifact predates ADR-040's
chirality block, so `load()` fail-closes on it. This task discharges that.

- [ ] **Step 1: Correct the false model-card claim**

At `docs/huggingface/model-cards/ghost-gk-v1-model-card.md:84`, the card claims the published
artifact contains "leaf-aggregated GK positions". There are none in the file. Replace with:

```
Only the learned model parameters are published -- tree structure, split thresholds, leaf values and
the boosting baselines. No per-sample training data and no raw provider tracking data are
redistributed. (Artifacts have been parameters-only since v4.54.0 / ADR-044; `predict_density`
requires a locally `fit()` model and is unavailable on a distributed artifact.)
```

- [ ] **Step 2: Write the publisher, mirroring the xCross one**

Create `scripts/publish_ghost_gk.py`:

```python
"""Publish a ghost-GK artifact to the Hub, with a round-trip verification.

Mirrors scripts/publish_xcross_attempt.py: verify locally, upload the folder, re-download via
from_hub and assert the served positions are identical. `--verify-only` stops before upload.
"""

from __future__ import annotations

import argparse
import pathlib

from silly_kicks.tracking._ghost_gk import GhostGkModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/ghost-gk-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    art = pathlib.Path(args.artifact_dir)
    local = GhostGkModel.load(art)
    # NAMED DELIBERATELY. There is no probe-frame METHOD on GhostGkModel; there are two
    # module-level frames and they answer different questions:
    #   _chirality.canonical_probe_frame()      -- y-asymmetric, designed so a mirrored model
    #                                              differs; `load()` already validates against it
    #   _feature_contract.contract_probe_frame() -- what the CONTRACT is fingerprinted on
    # A round-trip check wants a frame that is DISCRIMINATING about the served output, so this uses
    # the chirality frame. Using the other would prove something adjacent to what it claims.
    from silly_kicks.tracking._chirality import canonical_probe_frame

    probe = canonical_probe_frame()
    before = local.predict_mean(probe)
    print(f"local load OK; probe prediction {before!r}")

    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")
    back = GhostGkModel.from_hub(args.repo_id)
    after = back.predict_mean(probe)
    assert (before == after).all(), "round-trip prediction mismatch after upload"
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
```

`predict_mean(self, features: pd.DataFrame) -> np.ndarray` is verified correct (`_ghost_gk.py:1738`).

- [ ] **Step 2b: Update `from_variant`'s docstring — it is PUBLIC and this cycle falsifies it**

`GhostGkModel.from_variant` (`:2160-2168`) documents `"full"` as *"Hub-hosted and **pre-contract** —
it cannot be re-uploaded under the standing owner hold — so it emits `MissingFeatureContractWarning`"*,
and quotes `~12 MB / 36 k` and `~170 MB / 887 k`. This cycle uploads a **contract-bearing** `full` and
re-fits both, so that paragraph becomes false and the documented consumer behaviour changes — a
consumer escalating `MissingFeatureContractWarning` is told to expect a warning that will no longer
fire. It lives in a module CI runs under `--doctest-modules`, and unlike the model card and the pinned
test docstring it was not otherwise scheduled. Replace the stale sentence and re-check the two sample
counts and sizes against the artifacts Task 8 actually produced.

- [ ] **Step 3: Verify locally without uploading**

```bash
python scripts/publish_ghost_gk.py --artifact-dir silly_kicks/tracking/_ghost_gk_weights/default --verify-only
```
Expected: prints the probe prediction, no network access.

- [ ] **Step 4: Upload both variants**

```bash
python scripts/publish_ghost_gk.py --artifact-dir silly_kicks/tracking/_ghost_gk_weights/default
python scripts/publish_ghost_gk.py --artifact-dir /tmp/ghost-full
```

- [ ] **Step 5: Confirm `from_hub` is un-broken**

```bash
python -c "
from silly_kicks.tracking._ghost_gk import GhostGkModel
m = GhostGkModel.from_hub()
print('from_hub OK:', type(m).__name__)
"
```
Expected: loads without raising. This is the recorded follow-up being discharged; before this it
fail-closed on the missing chirality block.

---

## Task 10: TF-24 — the Stage-1 argmax check, then Stage 2

**Files:**
- Create: `scripts/check_stage1_argmax.py`
- Test: `tests/scripts/test_check_stage1_argmax.py` (new)

**Interfaces:** consumes Task 7's TC3 frames and the prior Optuna store.

- [ ] **Step 1: Write the gate test**

Create `tests/scripts/test_check_stage1_argmax.py`:

```python
"""The Stage-1 gate is pre-registered: invariance >= 99.9%, and the argmax must not move."""

import pytest

from scripts.check_stage1_argmax import invariance_verdict


def test_at_threshold_passes():
    assert invariance_verdict(same=9990, total=10000) == "stands"


def test_below_threshold_requires_sweep():
    assert invariance_verdict(same=9989, total=10000) == "sweep"


def test_perfect_invariance_passes():
    assert invariance_verdict(same=10, total=10) == "stands"


def test_zero_rows_is_an_error_not_a_pass():
    """An empty comparison must never read as 'stands' -- that is a silent no-op gate."""
    with pytest.raises(ValueError):
        invariance_verdict(same=0, total=0)


def test_the_reflection_negates_velocity():
    """Prong 1 is STRUCTURALLY BLIND to this (beta=0 kills the velocity term), and prong 2 -- whose
    neighbours have beta != 0 -- is what gets corrupted. So it must be tested directly."""
    import pandas as pd

    from scripts.check_stage1_argmax import reflect_frames

    src = pd.DataFrame({"game_id": [1, 1], "period_id": [1, 1], "frame_id": [1, 1],
                        "player_id": ["a", "b"], "team_id": ["H", "A"],
                        "x": [10.0, 95.0], "y": [20.0, 48.0],
                        "vx": [1.5, -2.0], "vy": [-0.5, 3.0]})
    out = reflect_frames(src)
    assert list(out["x"]) == [95.0, 10.0]
    assert list(out["y"]) == [48.0, 20.0]
    assert list(out["vx"]) == [-1.5, 2.0], "velocities were not negated -- not a reflection"
    assert list(out["vy"]) == [0.5, -3.0]


def test_missing_velocity_columns_raise_rather_than_zero_silently():
    """`_ball_carrier.py:53,131` substitutes pvx=0.0 when vx/vy are absent, which makes beta inert
    and every neighbour score identically -- an argmax that 'cannot move' for the wrong reason."""
    import pandas as pd

    from scripts.check_stage1_argmax import require_velocity

    with pytest.raises(ValueError, match="vx"):
        require_velocity(pd.DataFrame({"x": [1.0], "y": [2.0]}))
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/scripts/test_check_stage1_argmax.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the checker**

Create `scripts/check_stage1_argmax.py` (ASCII-only, argparse, provenance-wired). It must:

1. **Assert `vx`/`vy` are present on the loaded frames before scoring anything.**
   `_ball_carrier.py:53` sets `has_velocity = "vx" in frames.columns and "vy" in frames.columns` and
   **silently substitutes `pvx = 0.0`** otherwise (`:131`). With velocity zeroed, `beta` becomes
   inert and every neighbour scores identically to the optimum — the argmax "cannot move" for a
   reason that has nothing to do with geometry, and prong 2 reports a clean pass having tested
   nothing. Fail loudly instead.
2. Load TC3 frames, run `infer_ball_carrier` at `beta=0.0, gamma=0.25, tolerance_m=3.0`.
3. Re-run on an exact point reflection and compare assignments. **The reflection MUST negate
   velocities as well as mirror positions:**

   ```python
   ref = frames.copy()
   ref["x"] = 105.0 - frames["x"]
   ref["y"] = 68.0 - frames["y"]
   ref["vx"] = -frames["vx"]      # NOT optional
   ref["vy"] = -frames["vy"]
   ```

   Mirroring positions while leaving velocities pointing the original way is not a reflection — it is
   a physically inconsistent frame. `infer_ball_carrier` scores
   `cand_dists[ci] - beta * v_toward` (`_ball_carrier.py:255`), so the velocity term is live.
   **The trap is that prong 1 cannot see this**: the recorded optimum has `beta = 0.0`, the term
   vanishes, and the invariance fraction is unaffected. Prong 2 is what gets corrupted, because its
   neighbours have `beta != 0` by construction — the exact scores the argmax comparison depends on.
   Pin the negation in a unit test on a two-player fixture.
4. **Pin "immediate neighbours" before the run**, as the threshold was: the rule is the K nearest
   trials in normalised parameter space around the recorded optimum, K stated in the artifact. Note
   `beta = 0.0` sits on a boundary, so neighbours exist on one side only — record how many were found
   rather than assuming symmetry.
5. **State the no-carrier rule.** Frames where inference returns no carrier must be counted
   explicitly: treating `None == None` as agreement inflates the invariance fraction by however many
   dead-ball frames the corpus holds; excluding them changes the denominator. Either is defensible;
   silence is not, in a pre-registered gate. Record `n_no_carrier` alongside the fraction and state
   which convention was used.
6. Re-score the recorded optimum **and those neighbours** on corrected frames only, and report
   whether the argmax remains at the recorded point.
7. Write `docs/research/tf24_stage1_confirmation/metrics.json` with `n_frames`, `n_same`,
   `n_no_carrier`, `no_carrier_convention`, `invariance_fraction`, `verdict`, `k_neighbours`,
   the neighbour scores, `argmax_moved`, `run_commit`, `run_tree_dirty`.

```python
_INVARIANCE_THRESHOLD = 0.999  # pre-registered before any data was seen (spec D5)

_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0


def invariance_verdict(*, same: int, total: int) -> str:
    if total <= 0:
        raise ValueError("no frames compared; an empty comparison cannot be a pass")
    return "stands" if (same / total) >= _INVARIANCE_THRESHOLD else "sweep"


def require_velocity(frames: pd.DataFrame) -> None:
    """Fail loudly when vx/vy are absent.

    `_ball_carrier.py:53` sets `has_velocity` from column presence and `:131` substitutes
    `pvx = 0.0` when it is False. Silently zeroed velocity makes `beta` inert, so every neighbour
    scores identically to the optimum and the argmax "cannot move" for a reason unrelated to
    geometry -- a green prong 2 that tested nothing.
    """
    missing = [c for c in ("vx", "vy") if c not in frames.columns]
    if missing:
        raise ValueError(
            f"frames lack {missing}; carrier inference would silently zero velocity and make the "
            f"beta term inert. Derive velocities before running this gate."
        )


def reflect_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Exact point reflection: mirror POSITIONS and NEGATE VELOCITIES.

    Mirroring positions while leaving velocities pointing the original way is not a reflection, it
    is a physically inconsistent frame. `infer_ball_carrier` scores `dist - beta * v_toward`
    (`_ball_carrier.py:255`), so the velocity term is live for any `beta != 0` -- which is every
    neighbour prong 2 scores, even though the recorded optimum (`beta = 0.0`) is blind to it.
    """
    require_velocity(frames)
    out = frames.copy()
    out["x"] = _PITCH_LENGTH - frames["x"]
    out["y"] = _PITCH_WIDTH - frames["y"]
    out["vx"] = -frames["vx"]
    out["vy"] = -frames["vy"]
    return out
```

**Do NOT compare accuracy to the value the prior Stage 1 recorded.** That baseline was computed on
pre-ADR-028 geometry, so the difference measures how much the correction moved *accuracy*, not
whether the *argmax* moved (spec D5). Any scalar drift is reported as a sensitivity measurement, not
a gate.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/scripts/test_check_stage1_argmax.py -q`
Expected: 4 passed.

- [ ] **Step 5: Run the confirmation on the DGX, then Stage 2**

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  ./.venv/bin/python scripts/check_stage1_argmax.py \
    --data-dir "$GEN" --store ~/tf24-stage1.db --out ~/tf24_stage1'
```

If `verdict == "sweep"` **or** `argmax_moved` is true, run the full Stage-1 sweep before Stage 2.
Otherwise proceed:

```bash
ssh karsten@192.168.68.73 'cd ~/silly-kicks-refit && source ~/.pining_env && \
  nohup ./.venv/bin/python scripts/calibrate_tracking_defaults.py \
  --stage 2 --source pining --providers skillcorner idsse gradientsports \
  --carrier-best ~/tf24_stage1/carrier_best.json --store ~/tf24-stage2.db \
  --report-out ~/tf24_stage2/report.json --cache-dir ~/pining-cache \
  > ~/tf24_stage2.log 2>&1 &'
```

- [ ] **Step 6: Land the artifacts and the ADR-009 note**

Copy both metrics files to `docs/research/tf24_stage1_confirmation/` and
`docs/research/tf24_stage2_refresh/`. Append a dated note to
`docs/superpowers/adrs/ADR-009-*.md` recording that TF-24's recommendations were recomputed on
post-ADR-028 geometry, with the artifact paths. **No library default changes** — ADR-009's standing
rule is that TF-24 recommends only; adopting a recommendation is a separate PR.

---

## Task 11: Full gates and final review

- [ ] **Step 1: Assert the tool versions match `ci.yml`'s pins BEFORE running anything**

```bash
python -m ruff --version      # must be 0.15.7   (ci.yml:25)
python -m pyright --version   # must be 1.1.409  (ci.yml:25)
```

**Run pyright per TASK, not only here.** During execution it caught a genuine runtime bug that ruff
and the unit tests structurally could not: `measure_box_constant_delta` wrote `prov.commit`, but
`git_provenance()` returns a **dict** — so it would have raised `AttributeError` *after the entire
corpus pass*, in the cheap write step, which is this repo's signature way to lose hours of work. No
test exercises a driver's `main()`, so types are the only check that sees it. Six of the seven other
errors were pandas-stub nits in the same two new files; all seven were mine, and all were in code
written that hour.

`ci.yml:25` pins `ruff==0.15.7 pyright==1.1.409 pandas-stubs==2.3.3.260113`. The global constraint
fixes the INVOCATION (`python -m`, since neither is on PATH) and says nothing about the VERSION — but
a ruff or pyright minor can add or drop a rule, so a green local run and a red CI run are both
reachable from identical code. Task 11's output is the final evidence presented to the owner, so the
difference is between "the gates passed" and "the gates CI will actually run passed". *(Verified
2026-08-11: this machine has ruff 0.15.7 and pyright 1.1.409 — both match. pyright prints an upgrade
notice to 1.1.411; do NOT take it.)*

- [ ] **Step 2: Run the six CI-parity commands, quoted from `ci.yml`**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e" --benchmark-skip --tb=short -q > /tmp/final_bulk.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/final_bulk.log
python -m pytest tests/ -m "not e2e" --benchmark-only --tb=short -q > /tmp/final_bench.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/final_bench.log
python -m pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" --tb=short -q > /tmp/final_doctest.log 2>&1; echo "PYTEST_EXIT=$?" >> /tmp/final_doctest.log
```

`ci.yml` has **three jobs** (`lint` :18, `test` :32, `pandas-span` :120); `--benchmark-only` (:92)
and `--doctest-modules` (:100) are steps inside `test`. `:88` is the superset leg — run that one.
The doctest `--ignore-glob` is load-bearing: dropping it widens scope to private modules.

**The third job has no local equivalent, and that is a statement, not an omission.** `pandas-span`
consumes the `pandas-major-*` artifacts the matrix legs upload, so it cannot run on one machine; it
is checked on the PR. This bullet exists because a from-memory enumeration was wrong once, so a
reader must be able to tell "covered", "deliberately not reproducible" and "forgotten" apart.

- [ ] **Step 3: `/final-review`, including the C4 regeneration**

Not optional, and not satisfied by the C4 gates passing — they pin the DSL; nothing reads
`architecture.html`. Full pipeline: `structurizr.war export` -> `plantuml.jar -tsvg` ->
`c4_assemble.py docs/c4 --svg-dir <tmp>`.

- [ ] **Step 4: Assemble the evidence and STOP**

Present to the owner: the D2 counts and selected ship claim, the Task 8 blast-radius diff, the
`from_hub` verification, the TF-24 verdict, and all six gate results. **Do not commit.** Numbers are
read off `main` at this moment.

---

## Self-Review

**Spec coverage:** D1 → Tasks 1-2. D2 → Tasks 6-7. D3 → Task 8 (both variants, one extraction).
D4 → Tasks 3-4 (RED first, then the migration + registry prune). D5 → Task 10. D6 → Tasks 7-8
(DGX fits, x86 stamping) and Task 11. D7 → Tasks 5, 7. §6's durable property test → Task 1. §6's
declared-value test → Task 3. HF + model card → Task 9.

**Placeholders:** none.

**Type consistency:** `in_penalty_area_goal_relative_array(gr_x, y) -> np.ndarray` is defined in
Task 1 and consumed under that exact name in Tasks 2 and 4. `assert_frames_parity(produced,
reference, *, match_id)` defined in Task 5, used in its own tests. `classify_flips(gr_x, y) -> dict`
defined in Task 6, used in its tests. `invariance_verdict(*, same, total) -> str` defined in Task 10,
used in its tests. `_feature_contract_block()` is module-level on all three extractors (verified) and
is consumed by Task 3's code-level test and Task 8 Step 6.

**Signatures verified against source (the earlier draft's stated gap, now closed):**
`predict_mean(self, features) -> np.ndarray` at `_ghost_gk.py:1738` — correct.
**`GhostGkModel.canonical_probe_frame` DOES NOT EXIST** — there is no probe-frame method on the model.
Two module-level frames do: `_chirality.canonical_probe_frame()` (`:21`) and
`_feature_contract.contract_probe_frame()` (`:75`), and they answer different questions. Task 9 Step 2
now imports the chirality frame explicitly and says why.

**External review (round 5, partial — Task 4, the round-4 repairs, Task 11):** the four round-4
repairs were independently re-verified as correct (band form, three-way partition over 301k points,
widened checksum on all four properties, reflection involution). Adopted from it: **T4-a**, the
`_feature_contract_block` docstring at `:1550-1554` — the primary in-code explanation of exactly what
this cycle changes, every sentence falsified by it, 25 lines above the edit and unscheduled;
**T4-b**, the import alias is `_geo` not `_geometry`, so the literal instruction would `NameError`;
**T4-c**, line numbers off by one (`:671` predicate, `:1578-1579` declaration) — carried unchecked
since the spec; **T4-d**, the registry coupling that makes Task 2's aliases load-bearing; **T11-a**,
`pandas-span` declared non-reproducible rather than silently absent; **T11-b**, assert the tool
versions against `ci.yml`'s pins. Also a mechanism correction: `-0.0` comes from the velocity
NEGATION, not from `68 - y` (subtraction of equal operands gives `+0.0`).

**External review (round 4, partial — Tasks 5/6/10) findings adopted, all four verified by
execution:**

* **The D2 driver mis-modelled the SHIPPED legacy predicate** — it used the abs form where ghost
  ships the min/max band. Measured at `y = 13.85`: a TRUE flip that the driver reported as no-flip.
  I had proved the two forms equivalent at the CANONICAL value in spec 1.1 and then applied that
  equivalence at the LEGACY value, where **my own spec records it is false**. Fixed via
  `_legacy_y_in_band`, with that exact point as a named regression test.
* **`classify_flips` forced a convention** on rows where both causes are individually necessary.
  Now a three-way split (`band_only` / `boundary_only` / `both`), which states the fact instead.
* **`test_counts_are_consistent` was tautological** — any partition of `flipped` satisfies it, so it
  constrained nothing. Replaced by per-case attribution with hand-derived answers, including the
  negative case (y in strip, x outside).
* **`_checksum` hashed only the identity columns**, so `vx` drifting 0.5 -> 99.0 passed the gate whose
  stated purpose is catching a divergent parse — while ghost's extractor and `infer_ball_carrier`
  both consume velocity. Now hashes every column, sorted by the identity columns, with `-0.0`
  normalised (it hashed differently and Task 7 treats a parity failure as STOP).
* **The Task 10 reflection omitted velocity negation.** `infer_ball_carrier` scores
  `dist - beta * v_toward`; prong 1 is blind because `beta = 0.0`, while prong 2's neighbours have
  `beta != 0` — so the corrupted inputs would sit under the one comparison prong 2 was rewritten to
  make. Plus a `require_velocity` guard, since absent `vx`/`vy` are silently zeroed.

**External review (round 3) findings adopted:** P1 (the `full` variant is trained-and-stamped on
aarch64, never re-stamped on x86, and Step 5's tree-scoped diff is blind to it) → Task 8 Step 6.
P2 (the non-existent probe method) → Task 9 Step 2. P3 (`_ulp_neighbourhood` collapsed 101 entries to
**3 distinct doubles**, gutting the exact dimension the durable test exists to cover) → Task 1, with
a self-check assertion so a future simplification cannot re-collapse it. P4 (artifact-level guard
gives no code-level signal for the length of a corpus pass) → Task 3's second test. Plus the depth
boundary's ULP treatment, the positive blast-radius assertion, the exact `==` in place of
`pytest.approx(..., abs=0.0)`, and the stale public `from_variant` docstring.
