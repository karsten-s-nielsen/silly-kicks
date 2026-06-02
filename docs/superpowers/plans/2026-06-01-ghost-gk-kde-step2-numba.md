# Ghost-GK KDE — Step 2 (cpu-numba backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cpu-numba` KDE backend — a serial `@njit` fully-fused closed-form hot loop (no `(kb,m)` temporaries) — selectable via `kde_backend="cpu-numba"`, validated against the numpy closed-form, delivering the kill-gate-measured ~10× single-thread speedup of `predict_density`'s hot path.

**Architecture:** Keep the numpy **setup** (weighted Scott covariance + `cho_factor` PD-branch + `log_det` + `det`) — extracted into a shared `_kde_setup` so numpy and numba use byte-identical setup (Leg-B integrity) and the singular→uniform boundary stays `cho_factor` (== 4.2.0). JIT **only** the per-sample `exp`+weighted-reduction loop (the dominant single-thread cost), in a separate lazily-imported `_ghost_gk_numba.py` mirroring `pitch_control/_numba_kernels.py`. Thread `kde_backend` through `add_ghost_gk → compute_ghost_gk → predict_density`.

**Tech Stack:** Python 3.10, numpy 2.x, scipy 1.15 (`cho_factor` in setup), **numba ≥0.59** (already a silly-kicks `[numba]` + `[test]` dep), pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md` (§3 Step 1 done/held; §4 Step 2). **Kill-gate PASSED** (~10.3× single-thread, parity 1e-9, prototype-validated).

**Bundling & commit policy:** This bundles with the **held Step-1 closed-form** (already on branch `feat/ghost-gk-kde-closed-form`, uncommitted). **Single commit at the end** (Task 6) covers Step-1 + Step-2, version **4.3.0** (new `cpu-numba` backend = minor bump).

---

## File structure

**Modified:**
- `silly_kicks/tracking/_ghost_gk.py` — extract `_kde_setup`; add `_kde_density_numba`; add `kde_backend == "cpu-numba"` dispatch in `predict_density`.
- `silly_kicks/tracking/features.py` — thread `kde_backend` through `add_ghost_gk` → `compute_ghost_gk`.

**Created:**
- `silly_kicks/tracking/_ghost_gk_numba.py` — the `@njit(cache=_NUMBA_CACHE)` serial fused `_kde_numba_loop` (mirrors `pitch_control/_numba_kernels.py`; lazily imported only on the `cpu-numba` path).
- tests in `tests/tracking/test_ghost_gk_kde_vectorized.py` (kernel parity, dispatch parametrization, lazy-import guard) + `tests/tracking/test_ghost_gk_integration.py` (public-API e2e).

**Current integration points (verified):**
- `predict_density(self, features, *, kde_backend: str = "vectorized")` — dispatch: `if "scipy" → _kde_density_scipy; elif "vectorized" → _kde_density_vectorized; else ValueError`.
- `compute_ghost_gk(frames, *, model=None, home_team_id, actions=None, link_frame_ids=None)` → calls `resolved.predict_density(batch_features)`.
- `add_ghost_gk(actions, frames, *, model=None, links=None, home_team_id, actions_for_context=None)` → calls `compute_ghost_gk(...)`.
- Numba guard pattern (mirror): `pitch_control/_numba_kernels.py` — `try: from numba import njit except ImportError: raise ImportError("...[numba]")`; `_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE","0")=="1" or bool(os.environ.get("NUMBA_CACHE_DIR"))`; `@njit(cache=_NUMBA_CACHE)`.

---

## Task 1: Extract `_kde_setup` (shared numpy+numba setup; Leg-B integrity)

Pull the cov/`cho_factor`/`det`/`log_det`/`norm` setup out of `_kde_density_vectorized` into one helper, so numpy and numba consume identical setup (and the PD-branch stays `cho_factor`, == 4.2.0). Pure refactor — golden must stay green.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py`

- [ ] **Step 1: Add `_kde_setup` above `_kde_density_vectorized`**

```python
def _kde_setup(gk_x_w, gk_y_w, w):
    """Shared closed-form KDE setup for the numpy + numba kernels (Leg-B: identical setup).

    Weighted Scott-bandwidth covariance + Cholesky PD-branch + self-consistent det/norm.
    cho_factor raises np.linalg.LinAlgError on a non-PD covariance -> predict_density's
    `except np.linalg.LinAlgError` degrades to the uniform grid (boundary == 4.2.0). det is
    derived from the same factor (det = (L00*L11)^2) for consistency with log_det/norm.

    Returns (data, w, h11, h12, h22, det, norm): data (2,k) float64, w normalized.
    """
    from scipy.linalg import cho_factor

    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    data = np.vstack([np.asarray(gk_x_w, np.float64), np.asarray(gk_y_w, np.float64)])  # (2, k)
    d = 2
    neff = 1.0 / np.sum(w**2)
    factor = neff ** (-1.0 / (d + 4))
    covariance = np.atleast_2d(np.cov(data, rowvar=True, bias=False, aweights=w)) * factor**2
    chol = cho_factor(covariance, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
    norm = np.exp(-0.5 * (log_det + d * np.log(2.0 * np.pi)))
    det = (chol[0][0, 0] * chol[0][1, 1]) ** 2
    return data, w, covariance[0, 0], covariance[0, 1], covariance[1, 1], det, norm
```

- [ ] **Step 2: Refactor `_kde_density_vectorized` to call `_kde_setup`**

Replace the body of `_kde_density_vectorized` from the `from scipy.linalg import cho_factor` line through the `norm = ...`/`det = ...` block (the setup) with a single call, keeping the chunked loop. The function becomes:

```python
def _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points, *, train_block=1024):
    """Vectorized weighted Gaussian KDE on the fixed grid (closed-form 2x2 whitening).

    Reuses _kde_setup (shared with the numba backend); streams the training subset in blocks
    of train_block to bound memory. Returns the UNnormalized-then-norm-scaled density grid.
    """
    data, w, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    inv_det = 1.0 / det
    gx = grid_points[0]
    gy = grid_points[1]
    m = grid_points.shape[1]
    out = np.zeros(m, dtype=np.float64)
    k = data.shape[1]
    for start in range(0, k, train_block):
        sl = slice(start, min(start + train_block, k))
        dx = gx[None, :] - data[0, sl][:, None]  # (kb, m)
        dy = gy[None, :] - data[1, sl][:, None]  # (kb, m)
        energy = 0.5 * inv_det * (h22 * dx * dx - 2.0 * h12 * dx * dy + h11 * dy * dy)  # (kb, m)
        out += np.einsum("k,km->m", w[sl], np.exp(-energy))
    out *= norm
    return out.reshape(GRID_NX, GRID_NY)
```

(The original docstring's detail now lives in `_kde_setup`; keep this one concise.)

- [ ] **Step 3: Run the closed-form gates — pure refactor, must stay green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -q --no-header -p no:warnings -k "near_singular or no_cho_solve or vectorized_kernel_matches_scipy or chunk or golden or degenerate or singular or lt2"`
Expected: **all PASS** (the setup is byte-identical, just relocated; `cho_factor` still the branch; `det` still `(L00·L11)²`).

(No commit — single commit at Task 6.)

---

## Task 2: `_ghost_gk_numba.py` — serial `@njit` fused kernel (red-first)

**Files:**
- Create: `silly_kicks/tracking/_ghost_gk_numba.py`
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the failing kernel-parity test**

Append to `tests/tracking/test_ghost_gk_kde_vectorized.py`:

```python
def test_numba_loop_matches_numpy_closed_form():
    """The @njit fused loop == the numpy closed-form kernel (Leg-B, same setup), across the regimes
    that matter for parity divergence:
      - LARGE-k (k~36000): the PRODUCTION regime (lakehouse: real candidate clouds are ~36k leaf
        positions). numba's j-outer/i-inner SEQUENTIAL accumulation vs numpy einsum's PAIRWISE
        reduction diverges most when many terms are summed — this is the real-world gap, not
        near-singular. (lakehouse fixture finding)
      - NEAR-SINGULAR: a conservative 1/det numerical-edge guard. Real ghost-GK candidate clouds
        measure cond <= ~5.3 (lakehouse, n=204), so this is a THEORETICAL edge, not a production
        scenario — but worth keeping as a cheap robustness guard. (lakehouse review #1)
    """
    from silly_kicks.tracking._ghost_gk import (
        GRID_NX,
        GRID_NY,
        _GRID_X,
        _GRID_Y,
        _kde_density_vectorized,
        _kde_setup,
    )
    from silly_kicks.tracking._ghost_gk_numba import _kde_numba_loop

    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])

    def _numba_grid(gk_x_w, gk_y_w, w):
        data, wn, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
        return _kde_numba_loop(gp[0], gp[1], data[0], data[1], wn, h11, h12, h22, 1.0 / det, norm).reshape(
            GRID_NX, GRID_NY
        )

    rng = np.random.default_rng(11)
    # Well-conditioned, incl. the production-scale k~36000 (the real accumulation-order regime).
    for k in (3, 50, 2000, 36000):
        x = rng.uniform(0, 30, k)
        y = rng.uniform(18, 50, k)
        w = rng.uniform(0.1, 1.0, k)
        w = w / w.sum()
        np.testing.assert_allclose(_numba_grid(x, y, w), _kde_density_vectorized(x, y, w, gp), rtol=1e-9, atol=1e-12)

    # Conservative ill-conditioned guard (cond ~1e6). Real ghost-GK never reaches this (cond <= ~5.3,
    # lakehouse n=204) — a theoretical edge, looser Leg-B tol where 1/det amplifies rounding.
    x, y, w = _near_singular_inputs()  # the Step-1 helper
    np.testing.assert_allclose(_numba_grid(x, y, w), _kde_density_vectorized(x, y, w, gp), rtol=1e-7, atol=1e-12)
```

(If the **large-k** case exceeds `rtol=1e-9`, the sequential-vs-pairwise summation gap is the cause —
the terms are all positive so naive error is bounded ~`(k−1)·eps ≈ 8e-12`, comfortably under 1e-9;
a failure means a real divergence to investigate, e.g. pairwise/Kahan accumulation in the loop, NOT
a blind loosen. The near-singular case at 1e-7 is the conservative-edge tolerance.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_numba_loop_matches_numpy_closed_form -v --no-header`
Expected: FAIL — `No module named 'silly_kicks.tracking._ghost_gk_numba'`.

- [ ] **Step 3: Create `silly_kicks/tracking/_ghost_gk_numba.py`**

```python
"""Optional numba-accelerated ghost-GK KDE hot loop.

Serial @njit fully-fused closed-form weighted-Gaussian KDE over the fixed grid — no (kb,m)
temporaries (the numpy closed-form kernel is memory-bound; this keeps it in registers for a
~10x single-thread speedup, kill-gate-validated). Lazily imported only on the cpu-numba path:

    try:
        from ._ghost_gk_numba import _kde_numba_loop
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False

Setup (weighted covariance + Cholesky PD-branch + det/norm) stays in numpy (_kde_setup);
numba does ONLY the exp+reduction loop, so the singular->uniform boundary stays cho_factor's
(== 4.2.0). Serial (NOT parallel=True): in Databricks serverless applyInPandas, Spark already
saturates cores across groups; an in-group prange would oversubscribe.

See docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover - exercised only without the [numba] extra
    raise ImportError(
        "numba is required for the cpu-numba ghost-GK backend. Install with: pip install silly-kicks[numba]"
    ) from e

# On-disk cache OFF by default (serverless read-only paths) — same rationale as
# pitch_control/_numba_kernels.py. Full native JIT speed retained; only cross-process
# persistence is dropped (one-time per-process recompile).
_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))


@njit(cache=_NUMBA_CACHE)
def _kde_numba_loop(gx, gy, xs, ys, w, h11, h12, h22, inv_det, norm):
    """Fused weighted-Gaussian KDE over the grid. gx,gy: (m,); xs,ys,w: (k,) -> (m,).

    energy = 0.5/det * (h22*dx^2 - 2*h12*dx*dy + h11*dy^2); density = norm * sum_j w_j exp(-energy).
    No (k,m) temporaries — scalar accumulate in registers.
    """
    m = gx.shape[0]
    k = w.shape[0]
    out = np.zeros(m)
    half = 0.5 * inv_det
    for j in range(k):
        wj = w[j]
        xj = xs[j]
        yj = ys[j]
        for i in range(m):
            ddx = gx[i] - xj
            ddy = gy[i] - yj
            e = half * (h22 * ddx * ddx - 2.0 * h12 * ddx * ddy + h11 * ddy * ddy)
            out[i] += wj * np.exp(-e)
    return out * norm
```

- [ ] **Step 4: Run the kernel-parity test**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_numba_loop_matches_numpy_closed_form -v --no-header`
Expected: PASS for `k=3,50,2000,36000` + near-singular (first run includes JIT compile; the k=36000 case runs ~138M loop iterations — a couple seconds compiled; parity at rtol 1e-9 well-conditioned / 1e-7 near-singular).

(No commit.)

---

## Task 3: `_kde_density_numba` + `cpu-numba` dispatch (red-first)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py`
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Parametrize the existing parity/golden/degenerate tests over `cpu-numba`**

Add `"cpu-numba"` to the `@pytest.mark.parametrize("kde_backend", [...])` lists of:
`test_golden_continuous` (`["vectorized"]` → `["vectorized", "cpu-numba"]`),
`test_predict_density_backend_parity_small_model`, `test_degenerate_paths_produce_identical_uniform_fallback`
(`["scipy", "vectorized"]` → `["scipy", "vectorized", "cpu-numba"]`).

(Near-singular numba parity is already covered at the kernel level by Task 2's
`test_numba_loop_matches_numpy_closed_form` — lakehouse review #1 — so the kernel-level
`test_kernel_near_singular_parity` need not be re-parametrized.)

**`test_golden_discrete_mode` — mode-tie robustness for cpu-numba (lakehouse review #4).**
numba's j-outer/i-inner accumulation order can shift a *near-tie* argmax by ≤1 grid cell vs
numpy's vectorized reduction. Exact-argmax equality holds for `"vectorized"` (it matched the
frozen golden in 4.2.0) but can flake on `"cpu-numba"`. Make the assertion backend-aware
(the *density field* is the primary check via `test_golden_continuous`; the mode is derived):

```python
@pytest.mark.parametrize("kde_backend", ["vectorized", "cpu-numba"])
def test_golden_discrete_mode(default_model_features, golden, kde_backend):
    """mode_x/y vs the frozen golden. Exact for vectorized; for cpu-numba allow a <=1-cell shift
    on near-tie samples (different accumulation order) — guarded by GRID_RESOLUTION."""
    from silly_kicks.tracking._ghost_gk import GRID_RESOLUTION

    model, X = default_model_features
    densities = model.predict_density(X, kde_backend=kde_backend)
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    gmx = golden["mode_x"][: len(mode_x)]
    gmy = golden["mode_y"][: len(mode_y)]
    if kde_backend == "vectorized":
        np.testing.assert_array_equal(mode_x, gmx)
        np.testing.assert_array_equal(mode_y, gmy)
    else:  # cpu-numba: <=1 grid cell (near-tie argmax can flip on a different reduction order)
        assert np.all(np.abs(mode_x - gmx) <= GRID_RESOLUTION + 1e-9)
        assert np.all(np.abs(mode_y - gmy) <= GRID_RESOLUTION + 1e-9)
```

(Match the existing `test_golden_discrete_mode`'s actual fixture/param names — `default_model_features`/`golden`
are illustrative; keep whatever it already uses, just add the backend param + the branch.)

- [ ] **Step 2: Run to verify the new `cpu-numba` params fail**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -v --no-header -k "cpu-numba"`
Expected: FAIL — `ValueError: Unknown kde_backend: 'cpu-numba'`.

- [ ] **Step 3: Add `_kde_density_numba` + the dispatch branch**

In `_ghost_gk.py`, add after `_kde_density_vectorized`:

```python
def _kde_density_numba(gk_x_w, gk_y_w, w, grid_points):
    """cpu-numba KDE: numpy _kde_setup (cho_factor branch + det/norm) + the @njit fused loop.

    No train_block chunking needed — the numba loop has no (k,m) temporaries. Lazy import so
    `import silly_kicks.tracking._ghost_gk` stays numba-free.
    """
    from ._ghost_gk_numba import _kde_numba_loop

    data, w, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    out = _kde_numba_loop(
        grid_points[0], grid_points[1], data[0], data[1], w, h11, h12, h22, 1.0 / det, norm
    )
    return out.reshape(GRID_NX, GRID_NY)
```

In `predict_density`'s dispatch, add the `cpu-numba` branch:

```python
                    if kde_backend == "scipy":
                        probs = _kde_density_scipy(gk_x_w, gk_y_w, w, grid_points)
                    elif kde_backend == "vectorized":
                        probs = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)
                    elif kde_backend == "cpu-numba":
                        probs = _kde_density_numba(gk_x_w, gk_y_w, w, grid_points)
                    else:
                        msg = f"Unknown kde_backend: {kde_backend!r}"
                        raise ValueError(msg)
```

- [ ] **Step 4: Run the parametrized suite (cpu-numba now green)**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -q --no-header -p no:warnings -k "cpu-numba or golden or degenerate or near_singular or backend_parity"`
Expected: **all PASS** — cpu-numba matches the numpy closed-form / golden within tolerance; degenerate (`<2`, singular→uniform via the shared `cho_factor` branch) and near-singular all hold on cpu-numba.

(No commit.)

---

## Task 4: Thread `kde_backend` through `compute_ghost_gk` + `add_ghost_gk` + public-API e2e

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`compute_ghost_gk`)
- Modify: `silly_kicks/tracking/features.py` (`add_ghost_gk`)
- Test: `tests/tracking/test_ghost_gk_integration.py`

- [ ] **Step 1: Write the failing public-API e2e test**

Append to `tests/tracking/test_ghost_gk_integration.py` (reuse that file's frame/action fixtures — match its existing helper names; the test below assumes a `_frames`/`_actions`/`home_team_id` builder like the other tests there):

```python
def test_add_ghost_gk_cpu_numba_matches_default(ghost_gk_frames_actions):
    """add_ghost_gk(kde_backend="cpu-numba") == default (vectorized) within tolerance,
    guarding the kde_backend threading through compute_ghost_gk -> predict_density.
    """
    import numpy as np

    from silly_kicks.tracking.features import add_ghost_gk

    frames, actions, home = ghost_gk_frames_actions
    base = add_ghost_gk(actions, frames, home_team_id=home)
    nb = add_ghost_gk(actions, frames, home_team_id=home, kde_backend="cpu-numba")
    for col in ("ghost_gk_x", "ghost_gk_y", "ghost_gk_spread"):
        np.testing.assert_allclose(
            nb[col].to_numpy(), base[col].to_numpy(), rtol=1e-7, atol=1e-9, equal_nan=True
        )
```

(If `test_ghost_gk_integration.py` has no shared frames/actions fixture, add a minimal one mirroring `tests/tracking/test_ball_carrier.py::_tiny_poss_frames` + `synthesize_actions`, or reuse `tests/tracking/_provider_inputs.py`. Name it `ghost_gk_frames_actions`.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_integration.py::test_add_ghost_gk_cpu_numba_matches_default -v --no-header`
Expected: FAIL — `add_ghost_gk() got an unexpected keyword argument 'kde_backend'`.

- [ ] **Step 3: Thread `kde_backend` through both functions**

`compute_ghost_gk` — add the param + forward it:

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
    kde_backend: str = "vectorized",
) -> pd.DataFrame:
```

and change its predict call:

```python
    densities = resolved.predict_density(batch_features, kde_backend=kde_backend)
```

`add_ghost_gk` (`features.py`) — add the param to the signature:

```python
def add_ghost_gk(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model=None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str,
    actions_for_context: pd.DataFrame | None = None,
    kde_backend: str = "vectorized",
) -> pd.DataFrame:
```

and forward it at the `compute_ghost_gk(...)` call inside `add_ghost_gk` (locate the call — it passes `home_team_id=`, `actions=`, `link_frame_ids=`; add `kde_backend=kde_backend`). Add a one-line docstring entry for the param: `kde_backend : "vectorized" (default, cpu-numpy) | "scipy" | "cpu-numba"`.

- [ ] **Step 4: Run the e2e + the ghost-gk integration suite**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_integration.py -q --no-header -p no:warnings`
Expected: **all PASS** — the threading works end-to-end; cpu-numba matches the default within tolerance.

(No commit.)

---

## Task 5: Lazy-import guard + CI coverage

**Files:**
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the lazy-import guard test**

```python
def test_ghost_gk_does_not_eagerly_import_numba():
    """Importing _ghost_gk must NOT transitively import numba or _ghost_gk_numba
    (numba is loaded lazily only on the cpu-numba path; keeps bare imports light).
    """
    import importlib
    import sys

    for mod in ("numba", "silly_kicks.tracking._ghost_gk_numba"):
        sys.modules.pop(mod, None)
    sys.modules.pop("silly_kicks.tracking._ghost_gk", None)
    importlib.import_module("silly_kicks.tracking._ghost_gk")
    assert "numba" not in sys.modules, "import _ghost_gk eagerly imported numba"
    assert "silly_kicks.tracking._ghost_gk_numba" not in sys.modules
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_ghost_gk_does_not_eagerly_import_numba -v --no-header`
Expected: PASS — `_kde_density_numba` imports `_ghost_gk_numba` *inside* the function, so `import _ghost_gk` stays numba-free. (If it FAILS, an eager top-level `from ._ghost_gk_numba import ...` slipped in — move it into `_kde_density_numba`.)

Note: numba is in the `[test]` extra, so the cpu-numba parity/golden/e2e tests **run in CI** (hard gate, not skip-if-absent) — same as the pitch_control `@njit` parity. No marker needed.

(No commit.)

---

## Task 6: Bundle (Step-1 + Step-2), full verify, version 4.3.0, single commit

**Files:** (version sites + CHANGELOG)

- [ ] **Step 1: Full regression**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q -p no:warnings`
Expected: **all PASS** (Step-1 closed-form + Step-2 numba; the parametrized cpu-numba cases add to the ghost-gk file).

- [ ] **Step 2: CI-exact lint + type (full sequence)**

Run: `.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/`
Then: `.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/` (run `ruff format` on the touched files first if flagged)
Then: `.venv/Scripts/python.exe -m pyright silly_kicks/`
Expected: all clean. (`_ghost_gk.py` already has the `N803/N806` per-file-ignore; `_ghost_gk_numba.py` uses lowercase loop vars + `h11/h12/...` so no N-rule issue — confirm; add to per-file-ignores only if ruff flags it.)

- [ ] **Step 3: Version bump 4.2.0 → 4.3.0 (5 sites) + CHANGELOG**

A new selectable `cpu-numba` backend (additive) → **minor bump 4.3.0**. Edit `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `TODO.md` "Current release", add a CHANGELOG `[4.3.0]` entry, then `uv lock`. CHANGELOG entry:

```
## [4.3.0] — 2026-06-01

### Added — `cpu-numba` ghost-GK KDE backend (~10× the closed-form hot loop, single-thread)

`GhostGkModel.predict_density` / `compute_ghost_gk` / `add_ghost_gk` accept `kde_backend="cpu-numba"`
(default stays `"vectorized"` = cpu-numpy). It runs a serial `@njit` fully-fused closed-form KDE loop
(no per-block temporaries), validated parity-exact (rtol 1e-9, incl. the near-singular zone) against the
numpy kernel; **~10× on the hot loop measured numba-serial vs numpy with all thread env vars pinned to 1**
(`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS=1`) — single-thread-vs-single-thread, the Spark-`applyInPandas`
in-venue reality. numpy setup keeps `cho_factor` for the PD/singular branch + `log_det`, so the fallback
boundary is byte-identical to the numpy path. Requires the `[numba]` extra (lazily imported; `import
silly_kicks` stays numba-free). Opt-in — value-equivalent to the numpy default within golden tolerance.

### Changed — default ghost-GK KDE whitening is now closed-form (removes `cho_solve`)

**Heads-up for pinned consumers (Hyrum's Law): this shifts the DEFAULT `vectorized` backend's output, not
just the opt-in `cpu-numba` path.** `_kde_density_vectorized` (Step 1, bundled) computes the 2×2
Mahalanobis energy in closed form (`0.5/det·(h₂₂·dx² − 2·h₁₂·dx·dy + h₁₁·dy²)`) instead of `cho_solve`,
sharing `_kde_setup` with the numba backend. So **every** consumer's `ghost_gk_x`/`ghost_gk_y`/`ghost_gk_spread`
move by `~1e-12..1e-9` on a plain `4.2.0 → 4.3.0` upgrade, even without selecting a new backend. `cho_factor`
retained for the PD-branch + `log_det` (singular→uniform boundary == 4.2.0); value-equivalent within the
frozen golden's `rtol≈1e-7` (golden NOT regenerated). Single-thread the closed-form alone is ~1.0× (the win
is the numba loop above); it lands as the shared foundation.
```

Then consistency check: `grep -rn "4\.3\.0" pyproject.toml silly_kicks/__init__.py TODO.md uv.lock` shows the bump; `grep -rn "4\.2\.0" pyproject.toml silly_kicks/__init__.py TODO.md` returns nothing (historical CHANGELOG `[4.2.0]` is the only allowed hit). Re-check `main` for version contention at PR time.

- [ ] **Step 4: Single bundled commit**

```bash
git add -A && git commit -F - <<'EOF'
perf(ghost-gk): cpu-numba KDE backend (~10x) + closed-form whitening -- silly-kicks 4.3.0

Phase-1 of ghost-GK KDE acceleration (spec: ...ghost-gk-kde-numba-acceleration-design.md).
Bundles Step 1 (closed-form) + Step 2 (numba) — Step 1 alone is ~1.0x single-thread (the
cho_solve "24%" was a multi-thread-BLAS artifact); the win is the numba loop.

- _kde_setup: shared weighted-cov + cho_factor PD-branch + det/log_det/norm (numpy + numba use
  identical setup; boundary == 4.2.0).
- _kde_density_vectorized: closed-form 2x2 energy (no cho_solve / (2,kb,m) temporaries).
- _ghost_gk_numba._kde_numba_loop: serial @njit fully-fused exp+reduction loop, lazily imported;
  kde_backend="cpu-numba" on predict_density/compute_ghost_gk/add_ghost_gk. ~10x single-thread,
  parity-exact (rtol 1e-9) vs numpy; default stays cpu-numpy.
- Tests: kernel parity, near-singular, degenerate, golden (anchored, NOT regenerated) all
  parametrized over cpu-numba; public-API add_ghost_gk(kde_backend=) e2e; lazy-import guard
  (import silly_kicks stays numba-free); numba parity runs in CI ([test] extra).

Lakehouse (non-blocking): net-of-compile single-thread in-venue re-measure + run_work_unit ->
golden.parquet + marts value-change gate before production adoption.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

## Notes for the implementer
- **Serial `@njit`, never `parallel=True`** — Spark saturates cores across `applyInPandas` groups; in-group `prange` oversubscribes (spec §3.3). `NUMBA_NUM_THREADS=1` belt-and-suspenders is the lakehouse's bootstrap (a serial kernel doesn't spawn threads anyway).
- **No `train_block` in the numba path** — the fused loop has no `(k,m)` temporaries, so no chunking/memory bound is needed (it removed the reason for `train_block`).
- **Golden-anchor:** do NOT regenerate `ghost_gk_kde_golden.npz`; cpu-numba must match the frozen scipy-Cholesky values within `rtol≈1e-7`.
- **Kill-gate already passed** (~10.3× steady-state) — measured **numba-serial vs numpy with every thread env var pinned to 1** (`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS=1`), i.e. single-thread-vs-single-thread (lakehouse review #2). This is honest for the in-venue reality and conservative (a multi-thread numpy baseline would be faster → a *smaller* reported ratio). Do NOT later quote a number from an un-pinned numpy baseline. The lakehouse owns the **net-of-compile in-venue** confirmation (cache=OFF per-process recompile) + the mart value-change gate — non-blocking for merge, gates their production adoption.
- **The value-change gate covers the PLAIN `4.2.0 → 4.3.0` upgrade, not just `cpu-numba`** (lakehouse review #3): Step-1's closed-form shifts the *default* `vectorized` output `~1e-12..1e-9`. The lakehouse `run_work_unit → golden.parquet + dependent marts` gate must confirm no mart moves beyond contract tolerance for **both** the default upgrade and opting into `cpu-numba` before production adoption.
- **The real parity regime is large-k, not near-singular** (lakehouse fixture finding — they instrumented `predict_density`'s KDE inputs over a full real match, IDSSE J03WMX p1, default model, n=204 candidate clouds): real ghost-GK clouds are ~36k leaf positions and inherently **well-conditioned — cond ∈ [3.82, 5.33], median 4.42, p90 5.03, zero clouds > 100**. So there is NO real near-singular fixture to swap in; the synthetic `_near_singular_inputs()` (cond ~1e6) is a conservative theoretical guard, kept as-is with a clarifying comment. The genuine divergence risk is numba's **sequential** accumulation vs numpy's **pairwise** reduction at production scale → Task 2 adds a well-conditioned **k≈36000** parity case (rtol 1e-9), no binary fixture needed.
- **GPU still deferred** (spec §8 / Phase-0 §4 / ADR-012) — only if numba in-venue proves insufficient + volume + a venue re-arch.
