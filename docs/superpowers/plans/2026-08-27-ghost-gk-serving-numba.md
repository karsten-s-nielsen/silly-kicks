# Ghost-GK serving numba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate the ghost-GK boosted-tree leaf traversal with an exact `@njit` kernel pair (numpy fallback), giving ~12× on the production `predict_mean` plus the research KDE leaf-match and `fit`, all **bit-identical**; and make `predict_density`'s default resolve to the fastest **exact** backend (`cpu-numba`).

**Architecture:** Hexagonal — `_vectorized_leaf_values` / `_vectorized_leaf_indices` are the PORT (numpy reference + fallback); two serial `@njit` kernels in `tracking/_ghost_gk_numba.py` are the ADAPTER; a `FlatTrees` value object (per-tree LOCAL left/right + offsets + field arrays) is passed EXPLICITLY. Numba is auto-used when installed, numpy otherwise (ADR-013/ADR-008 pattern). No retrain, no re-materialize.

**Tech Stack:** numba (optional `[numba]` extra), numpy, sklearn HGBR node arrays.

**Spec:** `docs/superpowers/specs/2026-08-27-ghost-gk-serving-numba-design.md` (APPROVED, two lakehouse reviews). Read it first — this plan implements it.

## Global Constraints

- **Single feature branch, single commit, single PR.** No `git commit` in any step.
- **TDD, red-first.** Every kernel/dispatch change lands a failing test first (or, where an import-error can't be red, a perturb-to-fail discrimination proof).
- **Hexagonal.** `FlatTrees` passed as an explicit arg to the port functions; the `@njit` adapter is swappable and the numpy body stays the reference + fallback.
- **e2e backstop.** The real-seam tests (`predict_mean`/`predict_density`/`fit` through chirality/feature-contract/serve-mean/KDE golden) must pass **unchanged** — they are the end-to-end gate, not just the kernel units.
- **Bit-identical** for the leaf traversal (`np.array_equal`), incl. the ASYMMETRIC failure mode: `_vectorized_leaf_values`/`_leaf_values_numba` RAISE on >depth-cap; `_vectorized_leaf_indices`/`_leaf_indices_numba` do NOT (return the non-converged LOCAL index).
- **numba optional + numpy fallback** — no new hard dependency. `SILLY_KICKS_GHOST_FORCE_NUMPY=1` forces the numpy path (both-paths coverage on every leg).
- **NOBODY CLAIMS VERSION NUMBERS UNTIL COMMIT-PREP.** During execution: write the ADR with a PLACEHOLDER (`ADR-NNN`), do NOT touch `pyproject.toml`/`silly_kicks/__init__.py`/`uv.lock`/CHANGELOG/TODO. Only at commit-prep — after `git fetch && git merge origin/main` — take the real NEXT-FREE and fill all six places in one pass. `4.98.0 / PR-S169 / ADR-076` is a NON-BINDING note (another PR may take it).
- Lint at CI scope (`ruff check/format silly_kicks/ tests/ scripts/`); bare `pyright`; full `-m "not e2e"`. Tools via `python -m`.

## File Structure

- **Modify** `silly_kicks/tracking/_ghost_gk_numba.py` — add `_leaf_values_numba` (guarded) + `_leaf_indices_numba` (unguarded, returns local `cur`).
- **Modify** `silly_kicks/tracking/_ghost_gk.py` — add `_flatten_trees` + `FlatTrees`; `_vectorized_leaf_values`/`_vectorized_leaf_indices` gain `*, flat=None` dispatch; `GhostGkModel` caches `_flat_trees`/`_flat_trees_y` at `fit`/`load`; `predict_mean`/`predict_density`/`fit` pass `flat=`; `predict_density` default `kde_backend="auto"` + resolver.
- **Create** `tests/tracking/test_ghost_gk_leaf_numba.py` — kernel bit-identity (+ NaN/single-node/depth/>cap-asymmetric), both-paths fixture, flat-tree round-trip, dispatch guard.
- **Create** `tests/tracking/test_ghost_gk_kde_auto_default.py` — `"auto"` resolution + default-reliance sweep note.
- **Modify** `CLAUDE.md` (ghost-GK note). **Create** `docs/superpowers/adrs/ADR-NNN-...md` (placeholder number). CHANGELOG/version — commit-prep only.

---

## Task 1: `FlatTrees` + `_flatten_trees`

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py`; Test `tests/tracking/test_ghost_gk_leaf_numba.py` (create).

**Interfaces:**
- Produces: `FlatTrees` (NamedTuple: `left, right, feat, thr, miss, val, offsets` — all 1-D np arrays) and `_flatten_trees(nodes_list: list[np.ndarray]) -> FlatTrees`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_ghost_gk_leaf_numba.py (create)
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit(n_estimators=30, n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    m = GhostGkModel(n_estimators=n_estimators)
    m.fit(X, labels)
    return m, X


def test_flatten_trees_preserves_local_indices_and_dtypes():
    from silly_kicks.tracking._ghost_gk import _flatten_trees

    m, _ = _fit()
    flat = _flatten_trees(m._tree_nodes)
    # offsets: one per tree + 1, monotonic, last == total nodes
    assert flat.offsets[0] == 0
    assert flat.offsets[-1] == sum(len(t) for t in m._tree_nodes)
    assert np.all(np.diff(flat.offsets) >= 0)
    # dtypes pinned
    assert flat.left.dtype == np.int64 and flat.val.dtype == np.float64
    # LOCAL left/right: a flattened tree's slice equals the source tree's RAW left/right
    for i, t in enumerate(m._tree_nodes):
        lo, hi = flat.offsets[i], flat.offsets[i + 1]
        assert np.array_equal(flat.left[lo:hi], t["left"].astype(np.int64)), "left must stay per-tree LOCAL (not offset-shifted)"
        assert np.array_equal(flat.right[lo:hi], t["right"].astype(np.int64))
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name '_flatten_trees'`).

Run: `python -m pytest tests/tracking/test_ghost_gk_leaf_numba.py::test_flatten_trees_preserves_local_indices_and_dtypes -v`

- [ ] **Step 3: Implement**

```python
# silly_kicks/tracking/_ghost_gk.py (near _vectorized_leaf_values)
from typing import NamedTuple


class FlatTrees(NamedTuple):
    """Trees flattened for the numba kernels: per-tree LOCAL left/right + offsets.

    ``offsets[t]..offsets[t+1]`` is tree ``t``'s node block; ``left``/``right`` stay LOCAL
    (child index within the tree), so the kernel walks ``cur = left[base+cur]`` and accesses
    ``left[base + cur]`` -- remapping to global would double-add ``base``.
    """

    left: np.ndarray
    right: np.ndarray
    feat: np.ndarray
    thr: np.ndarray
    miss: np.ndarray
    val: np.ndarray
    offsets: np.ndarray


def _flatten_trees(nodes_list: list[np.ndarray]) -> FlatTrees:
    offsets = np.zeros(len(nodes_list) + 1, dtype=np.int64)
    for i, t in enumerate(nodes_list):
        offsets[i + 1] = offsets[i] + len(t)
    cat = lambda field, dt: (  # noqa: E731
        np.concatenate([t[field] for t in nodes_list]).astype(dt)
        if nodes_list
        else np.zeros(0, dtype=dt)
    )
    return FlatTrees(
        left=cat("left", np.int64),
        right=cat("right", np.int64),
        feat=cat("feature_idx", np.int64),
        thr=cat("num_threshold", np.float64),
        miss=cat("missing_go_to_left", np.int64),
        val=cat("value", np.float64),
        offsets=offsets,
    )
```

- [ ] **Step 4: Run — expect PASS.**

---

## Task 2: The `@njit` kernels (guarded values, unguarded indices)

**Files:** Modify `silly_kicks/tracking/_ghost_gk_numba.py`; Test `tests/tracking/test_ghost_gk_leaf_numba.py`.

**Interfaces:**
- Produces: `_leaf_values_numba(left,right,feat,thr,miss,val,offsets,X) -> (n,)` (RAISES on non-convergence); `_leaf_indices_numba(...) -> (n, T)` LOCAL `cur` (no raise).

- [ ] **Step 1: Write the failing tests** — bit-identity + NaN + single-node + depth + the ASYMMETRIC >cap failure mode

```python
# tests/tracking/test_ghost_gk_leaf_numba.py (append)
pytestmark_numba = pytest.importorskip("numba")  # kernel tests need the [numba] extra


def test_leaf_kernels_are_bit_identical_to_numpy_incl_nan():
    from silly_kicks.tracking._ghost_gk import _flatten_trees, _vectorized_leaf_indices, _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    m, X = _fit()
    Xa = X.copy()
    Xa.iloc[0, :3] = np.nan  # exercise BOTH missing_go_to_left branches
    Xn = Xa[m._feature_names() if hasattr(m, "_feature_names") else GHOST_GK_FEATURE_NAMES].to_numpy(np.float64)
    flat = _flatten_trees(m._tree_nodes)

    ref_v = _vectorized_leaf_values(m._tree_nodes, Xn)
    got_v = _leaf_values_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)
    assert np.array_equal(ref_v, got_v), "values kernel must be BIT-identical (incl NaN rows)"

    ref_i = _vectorized_leaf_indices(m._tree_nodes, Xn)
    got_i = _leaf_indices_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)
    assert np.array_equal(ref_i, got_i), "indices kernel must return the LOCAL current index, bit-identical"


def _one_tree(nodes):  # helper: build FlatTrees for a single hand-built node array
    from silly_kicks.tracking._ghost_gk import _flatten_trees

    return _flatten_trees([nodes])


def test_single_node_leaf_root_tree():
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_values_numba

    dt = np.dtype([("left", "i8"), ("right", "i8"), ("feature_idx", "i8"),
                   ("num_threshold", "f8"), ("missing_go_to_left", "i8"), ("value", "f8")])
    nodes = np.array([(0, 0, 0, 0.0, 0, 3.5)], dtype=dt)  # root IS a leaf (left==0)
    f = _one_tree(nodes)
    X = np.zeros((4, 26))
    assert np.array_equal(_vectorized_leaf_values([nodes], X), _leaf_values_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X))


def test_over_depth_cap_failure_mode_is_ASYMMETRIC():
    """values: BOTH raise; indices: NEITHER raises and both return the same non-converged index."""
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_indices, _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    dt = np.dtype([("left", "i8"), ("right", "i8"), ("feature_idx", "i8"),
                   ("num_threshold", "f8"), ("missing_go_to_left", "i8"), ("value", "f8")])
    # A degenerate chain 0->1->...->199 (never a leaf within the 100-step cap; terminal leaf at 199):
    # each node i<199 has left=i+1, so the walk reaches local node 100 and stops non-converged.
    n = 200
    nodes = np.zeros(n, dtype=dt)
    for i in range(n - 1):
        nodes[i] = (i + 1, i + 1, 0, 1e18, 1, 0.0)  # always go_left to i+1 (feat 0 <= 1e18)
    nodes[n - 1] = (0, 0, 0, 0.0, 0, 1.0)  # terminal leaf, but unreachable within 100 steps
    f = _one_tree(nodes)
    X = np.zeros((2, 26))

    with pytest.raises(RuntimeError, match="did not converge"):
        _vectorized_leaf_values([nodes], X)
    with pytest.raises(RuntimeError, match="did not converge"):
        _leaf_values_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)

    # indices: NEITHER raises; both return the SAME non-converged LOCAL index.
    ri = _vectorized_leaf_indices([nodes], X)
    gi = _leaf_indices_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)
    assert np.array_equal(ri, gi), "indices path must not raise and must return the identical non-converged index"
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError` on the kernels).

- [ ] **Step 3: Implement the kernels** (in `_ghost_gk_numba.py`, beside `_kde_numba_loop`)

```python
@njit(cache=_NUMBA_CACHE)
def _leaf_values_numba(left, right, feat, thr, miss, val, offsets, X):
    n = X.shape[0]
    T = offsets.shape[0] - 1
    out = np.zeros(n)
    for s in range(n):
        acc = 0.0
        for t in range(T):
            base = offsets[t]
            cur = 0
            for _ in range(100):
                gi = base + cur
                if left[gi] == 0:
                    break
                fv = X[s, feat[gi]]
                go_left = (miss[gi] != 0) if np.isnan(fv) else (fv <= thr[gi])
                cur = left[gi] if go_left else right[gi]
            # Convergence guard -- matches numpy _vectorized_leaf_values' RuntimeError.
            if left[base + cur] != 0:
                raise RuntimeError("leaf traversal did not converge within depth cap")
            acc += val[base + cur]
        out[s] = acc
    return out


@njit(cache=_NUMBA_CACHE)
def _leaf_indices_numba(left, right, feat, thr, miss, val, offsets, X):
    n = X.shape[0]
    T = offsets.shape[0] - 1
    out = np.zeros((n, T), dtype=np.int64)
    for s in range(n):
        for t in range(T):
            base = offsets[t]
            cur = 0
            for _ in range(100):
                gi = base + cur
                if left[gi] == 0:
                    break
                fv = X[s, feat[gi]]
                go_left = (miss[gi] != 0) if np.isnan(fv) else (fv <= thr[gi])
                cur = left[gi] if go_left else right[gi]
            out[s, t] = cur  # LOCAL index, matching numpy's `current`. NO guard (numpy doesn't raise here).
    return out
```

> `val` is passed to `_leaf_indices_numba` only for signature symmetry (unused) — **keep it** so both kernels share one flat-array marshaling and the dispatch passes the identical 8-arg tuple to both. If ruff `ARG001` flags the unused arg, suppress it with `# noqa: ARG001` on the `def` line (or a harmless `_ = val` numba compiles away) — do NOT drop the arg, which would desync the two kernels' signatures and force a divergent call site (review #3).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Discrimination proofs** (each gate must be able to fail). Temporarily flip `go_left`'s inequality in the values kernel → `test_leaf_kernels_are_bit_identical` fails; remove the values-kernel guard → `test_over_depth_cap...` fails on the "values raises" leg. Revert. (Document in the test module docstring; do not commit the perturbations.)

---

## Task 3: Dispatch + model-cached `FlatTrees` (the port)

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py`; Test `tests/tracking/test_ghost_gk_leaf_numba.py`.

**Interfaces:**
- `_vectorized_leaf_values(nodes_list, X, *, flat: FlatTrees | None = None)` and `_vectorized_leaf_indices(..., flat=None)` — numba kernel iff `flat is not None and _use_ghost_numba()`, else the existing numpy body (unchanged).
- `GhostGkModel._flat_trees` / `._flat_trees_y` cached at `fit`/`load`.

- [ ] **Step 1: Write the failing test** — both paths identical; force-numpy override works

```python
# tests/tracking/test_ghost_gk_leaf_numba.py (append)
def test_dispatch_both_paths_identical(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    Xn = X.to_numpy(np.float64)
    flat = G._flatten_trees(m._tree_nodes)

    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")
    numpy_path = G._vectorized_leaf_values(m._tree_nodes, Xn, flat=flat)
    monkeypatch.delenv("SILLY_KICKS_GHOST_FORCE_NUMPY", raising=False)
    numba_path = G._vectorized_leaf_values(m._tree_nodes, Xn, flat=flat)
    assert np.array_equal(numpy_path, numba_path)


def test_model_caches_flat_trees_for_both_x_and_y():
    m, _ = _fit()
    assert m._flat_trees is not None and m._flat_trees_y is not None
    assert m._flat_trees.offsets[-1] == sum(len(t) for t in m._tree_nodes)
    assert m._flat_trees_y.offsets[-1] == sum(len(t) for t in m._tree_nodes_y)
```

- [ ] **Step 2: Run — expect FAIL** (`flat` kwarg / `_flat_trees` attr missing).

- [ ] **Step 3: Implement dispatch + caching**

```python
# _ghost_gk.py
import os

_HAS_GHOST_NUMBA: bool
try:
    from ._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba
    _HAS_GHOST_NUMBA = True
except ImportError:  # no [numba] extra
    _HAS_GHOST_NUMBA = False


def _use_ghost_numba() -> bool:
    return _HAS_GHOST_NUMBA and os.environ.get("SILLY_KICKS_GHOST_FORCE_NUMPY", "") != "1"


# head of _vectorized_leaf_values(nodes_list, X, *, flat=None):
    if flat is not None and _use_ghost_numba():
        return _leaf_values_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, X)
    # ... existing numpy body unchanged ...

# head of _vectorized_leaf_indices(nodes_list, X, *, flat=None):
    if flat is not None and _use_ghost_numba():
        return _leaf_indices_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, X)
    # ... existing numpy body unchanged ...
```

In `GhostGkModel.__init__` add `self._flat_trees = None; self._flat_trees_y = None`. In `fit` after `self._tree_nodes`/`self._tree_nodes_y` are set (`~:1768`) and in `load` after `model._tree_nodes = tree_nodes` (`~:2158`), build them:

```python
        self._flat_trees = _flatten_trees(self._tree_nodes)
        self._flat_trees_y = _flatten_trees(self._tree_nodes_y)
```

Then thread `flat=` into the call sites:
- `predict_mean` (`~:1838-1839`): `_vectorized_leaf_values(self._tree_nodes, X, flat=self._flat_trees)` and `..._y, X, flat=self._flat_trees_y)`.
- `fit`'s training leaf-match (`~:1777`) and `predict_density`'s query leaf-match: `_vectorized_leaf_indices(self._tree_nodes, X, flat=self._flat_trees)`. (In `fit`, build `_flat_trees` BEFORE the training leaf-match line, or leave that one on the numpy path — either is bit-identical; prefer building first so `fit` is accelerated too.)

- [ ] **Step 4: Run — expect PASS.**

---

## Task 4: `predict_mean` bit-identity + e2e seams unchanged

**Files:** Test `tests/tracking/test_ghost_gk_leaf_numba.py`; run existing ghost suites.

- [ ] **Step 1: Write the test**

```python
def test_predict_mean_numba_equals_numpy_bit_identical(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")
    ref = m.predict_mean(X)
    monkeypatch.delenv("SILLY_KICKS_GHOST_FORCE_NUMPY", raising=False)
    got = m.predict_mean(X)
    assert np.array_equal(ref, got), "predict_mean must be bit-identical numba-vs-numpy"
```

- [ ] **Step 2: Run — expect PASS** (dispatch already in place; this pins predict_mean).

- [ ] **Step 3: e2e backstop — the real seams must pass UNCHANGED.**

Run: `python -m pytest tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_serve_mean.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_ghost_gk_kde_vectorized.py -q`
Expected: all PASS unchanged (chirality fingerprint, feature-contract, serve-mean parity, KDE golden). Any failure here means the dispatch changed served output — STOP and root-cause (bit-identity is the contract).

---

## Task 5: KDE default → `"auto"` (fastest EXACT backend) + default-reliance sweep

**Files:** Modify `silly_kicks/tracking/_ghost_gk.py`; Test `tests/tracking/test_ghost_gk_kde_auto_default.py` (create).

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_ghost_gk_kde_auto_default.py (create)
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit(n=400):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    m = GhostGkModel(n_estimators=30)
    m.fit(X, pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)}))
    return m, X


def test_auto_resolves_to_cpu_numba_when_numba_present(monkeypatch):
    pytest.importorskip("numba")
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    seen = []
    real = G._kde_density_numba

    def spy(*a, **k):
        seen.append("cpu-numba")
        return real(*a, **k)

    monkeypatch.setattr(G, "_kde_density_numba", spy)
    m.predict_density(X.iloc[:2])  # DEFAULT (no kde_backend) -> must hit cpu-numba
    assert seen == ["cpu-numba", "cpu-numba"]


def test_auto_falls_back_to_vectorized_without_numba(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    monkeypatch.setattr(G, "_HAS_GHOST_NUMBA", False)  # simulate no [numba]
    seen = []
    real = G._kde_density_vectorized
    monkeypatch.setattr(G, "_kde_density_vectorized", lambda *a, **k: (seen.append("vec"), real(*a, **k))[1])
    m.predict_density(X.iloc[:2])
    assert seen == ["vec", "vec"]


def test_explicit_backends_still_selectable():
    m, X = _fit()
    for backend in ("vectorized", "scipy"):
        out = m.predict_density(X.iloc[:1], kde_backend=backend)
        assert out[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run — expect FAIL** (default is still `"vectorized"`).

- [ ] **Step 3: Implement** — default `kde_backend="auto"` + resolve

```python
# predict_density signature: kde_backend: str = "auto"
# early in the body:
        if kde_backend == "auto":
            kde_backend = "cpu-numba" if _use_ghost_numba() else "vectorized"
```

Extend the docstring's `kde_backend` value list to include `"auto"` (the default; "fastest EXACT backend: cpu-numba if the [numba] extra is present, else vectorized -- fft stays an explicit opt-in for a binned raw grid"). Add the ADR-057 1e-9 note.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: DEFAULT-RELIANCE SWEEP (the 4.97.0 lesson).**

Run: `grep -rn "predict_density(" tests/ silly_kicks/ scripts/ | grep -v "kde_backend"` — every call using the IMPLICIT default now resolves to `cpu-numba`. For each hit that asserts exact (`<1e-9`) equality to a `vectorized`/`scipy` reference, pin `kde_backend="vectorized"` explicitly (raw-grid oracle) or adopt the golden's cpu-numba tolerance. Then run the FULL KDE suite: `python -m pytest tests/tracking/ -k "ghost_gk and kde" -q` → all PASS.

---

## Task 6: Structural dispatch guard

**Files:** Test `tests/tracking/test_ghost_gk_leaf_numba.py`.

- [ ] **Step 1: Write + run**

```python
def test_predict_mean_uses_the_numba_kernel_on_the_default_path(monkeypatch):
    """Regression guard: a silent revert to the numpy path (e.g. dropping flat=) is caught."""
    pytest.importorskip("numba")
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    calls = {"n": 0}
    real = G._leaf_values_numba
    monkeypatch.setattr(G, "_leaf_values_numba", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
    m.predict_mean(X.iloc[:4])
    assert calls["n"] == 2, "predict_mean must dispatch to the numba kernel (x-tree + y-tree) when numba is available"
```
> Import name note: `predict_mean` calls `_vectorized_leaf_values`, which references the module-level `_leaf_values_numba`; patch `G._leaf_values_numba` (the name the dispatch resolves). Confirm the dispatch reads the module attribute at call time (it does via the top-level import) so the patch lands.

Run: `python -m pytest tests/tracking/test_ghost_gk_leaf_numba.py -q` → PASS.

---

## Task 7: Docs, ADR (placeholder), C4, gates, /final-review

- [ ] **Step 1: Docstrings** — `predict_mean` (note the numba dispatch, bit-identical, numpy fallback), `predict_density` (`"auto"` default + 1e-9 note), the two kernels, `_flatten_trees`/`FlatTrees`. Keep `predict_mean`'s "Cheap — pure leaf traversal" line accurate (add "numba-accelerated when installed").

- [ ] **Step 2: ADR (PLACEHOLDER number).** Create `docs/superpowers/adrs/ADR-NNN-ghost-gk-serving-numba.md` from the template: the numba-default-serving auto-selection + the exact KDE default (`cpu-numba`, `fft` opt-in) + the asymmetric convergence-guard fidelity. Leave the number as `ADR-NNN` — renamed at commit-prep.

- [ ] **Step 3: CLAUDE.md** — extend the ghost-GK bullet: the leaf traversal is numba-accelerated (bit-identical, numpy fallback via `SILLY_KICKS_GHOST_FORCE_NUMPY`), `FlatTrees` cached on the model; `predict_density` default is `"auto"` (cpu-numba exact, fft opt-in); no retrain.

- [ ] **Step 4: NOTICE** — unchanged (no new methodology). **C4** — no new aggregator; verify the completeness gate green (`python -m pytest tests/ -k c4 -q`); re-render only if flagged.

- [ ] **Step 5: DO NOT bump versions or write CHANGELOG.** These happen at commit-prep only (Global Constraints). Leave `pyproject.toml`/`__init__`/`uv.lock`/CHANGELOG/TODO untouched during execution.

- [ ] **Step 6: Full CI-faithful gate**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e" -v --tb=short
```
All clean. The `[numba]` extra is installed here, so the kernels run; a no-`[numba]` leg exercises the `importorskip` + the `_HAS_GHOST_NUMBA=False` fallback.

- [ ] **Step 7: /final-review + /c4**, then STOP and report. Do NOT commit. At commit-prep (owner-approved): `git fetch && git merge origin/main`, take the real NEXT-FREE version/PR-S/ADR, fill the 5 version strings + rename the ADR, add the CHANGELOG entry, single commit, single PR.

---

## Self-review notes (author)

- **Spec coverage:** §4.1 kernels → Task 2 (asymmetric guard included); §4.2 FlatTrees/local-index → Task 1; §4.3 dispatch/force-numpy/cache → Task 3; §4.4 auto default → Task 5; §5.1–5.7 gates → Tasks 2/3/4/5/6; §7 ordering ← task order; §9 explicit-FlatTrees ← Task 3.
- **Type consistency:** `FlatTrees`, `_flatten_trees`, `_use_ghost_numba`, `_HAS_GHOST_NUMBA`, `_leaf_values_numba`, `_leaf_indices_numba`, `_flat_trees`/`_flat_trees_y` named identically across tasks.
- **No placeholders** except the deliberate `ADR-NNN` + the commit-prep version bump (both mandated by the no-early-numbers rule).
- **Not a commit boundary anywhere** — single commit at the end, owner-approved.
- **Hexagonal:** port = `_vectorized_leaf_*` (numpy ref + fallback); adapter = `@njit` kernels; `FlatTrees` explicit; force-numpy exercises both adapters. **e2e:** Task 4 Step 3 is the real-seam backstop.
- **Open implementer note:** if `_kde_density_numba` / `_kde_density_vectorized` are not the exact module-level names the resolver dispatches, adjust the Task-5 spies to the real dispatch names (grep `predict_density`'s backend branch, `_ghost_gk.py:~1908-1918`). (Review #3 verified they ARE `_kde_density_numba:1426` / `_kde_density_vectorized:1386`.)

## Review log
- **Round-3 (plan) review — APPROVED, incorporated (2026-08-27).** All anchors verified against the tree (round-2 local-`cur` fix, the asymmetric convergence guard, KDE spy targets, `_HAS_GHOST_NUMBA` non-redundant, dispatch-guard patch point, `_flatten_trees` field names). One cosmetic note folded in: keep the symmetric `val` arg on `_leaf_indices_numba` via `# noqa: ARG001` rather than dropping it (dropping desyncs the two kernels' signatures + the shared call site). TDD/hexagonal/e2e/version-discipline confirmed genuinely encoded, not just claimed.
- **Round-4 (revised plan) review — APPROVED, ready for implementation (2026-08-27).** A fresh-eyes pass over the plan's TEST CODE verified every anchor against the tree (leaf sentinel `left==0`, `range(100)` cap, the raise-message substring on both legs, the 200-node non-convergence chain, the two-row `["cpu-numba","cpu-numba"]` spy, `predict_mean`'s twice-dispatch = x+y trees, all line anchors, the 4 e2e files exist, current default `"vectorized"`). One trivial comment off-by-one fixed (chain is 0..199, terminal leaf at 199). No blocking items.
