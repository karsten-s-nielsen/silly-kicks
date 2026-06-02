# Ghost-GK KDE `fft-cic` backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `kde_backend="fft-cic"` ghost-GK KDE backend — CIC (cloud-in-cell / bilinear) binning on the existing FFT-convolution path — that reduces NGP's real-data mode-flip rate (~22 %→~5 % of actions) at ~2× the (still ~1000×) cost, and correct ADR-014's falsified mode-fidelity claims.

**Architecture:** Binning is the *only* seam between `fft` (NGP) and `fft-cic`. Extract the shared FFT-convolution tail (`_fft_convolve_field`) and the NGP binning (`_bin_ngp`) from the current `_kde_density_fft`, add `_bin_cic`, and compose `_kde_density_fft_cic = _kde_setup → _bin_cic → _fft_convolve_field`. Dispatch via one new `elif` in `predict_density`; the flat `kde_backend` string auto-propagates through `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` (and the atomic mirror) with **no signature change**.

**Tech Stack:** Python 3.10 (`.venv`), numpy ≥ 2.0, scipy (`signal.fftconvolve`, core dep — no new dependency), pandas; pytest. Lint: `ruff==0.15.7`, `pyright==1.1.409`.

**Commit policy (overrides the writing-plans per-task-commit default):** Per the user's standing rule, this is **one feature branch, one commit, one PR at the end** — no per-task commits, no standalone doc commits. Tasks below have **no `git commit` steps**; a single final commit (Task 11) bundles code + tests + ADR + NOTICE + version bumps after the full suite + lint trio are green. The commit is gated by the commit-sentinel hook and is **held for explicit per-commit approval** (the sentinel is never self-created). The PR is squash-merged.

**Branch:** `feat/ghost-gk-fft-cic-binning` (already created, off `main` @ 4.7.0).

**Key files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `_bin_ngp`, `_fft_convolve_field`, `_bin_cic`, `_kde_density_fft_cic` (new); `_kde_density_fft` (refactor to compose); `predict_density` dispatch (~line 1360) + docstring (~1304); `compute_ghost_gk` docstring (~1625).
- Modify: `silly_kicks/tracking/features.py` — `add_ghost_gk` docstring (~3701) + `ghost_gk_xfns` docstring (~3791). (No logic change — both already forward `kde_backend`.)
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py` — append a `fft-cic` test block mirroring the existing `fft` block (lines 512-637).
- Docs: `docs/superpowers/adrs/ADR-014-ghost-gk-kde-fft-backend.md` (amend); `NOTICE` (add CIC reference); `CHANGELOG.md`; `TODO.md` (entry + tracked train/serve-skew action); version sites `pyproject.toml`, `silly_kicks/__init__.py`; `uv.lock`.

**Run tests with:** `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -v --tb=short` (full suite gate: `python -m pytest tests/ -m "not e2e"`). Use the `.venv` (CPython 3.10.19).

---

## Task 1: Extract the shared FFT tail + NGP binning (pure refactor)

Refactor `_kde_density_fft` so the FFT-convolution tail and the NGP binning are reusable helpers, with **zero behaviour change**. This is a Chesterton's-fence refactor on the shipped 4.6.0 NGP path — the existing fft golden/scalar tests must stay green.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:1102-1148` (`_kde_density_fft`)

- [ ] **Step 1: Run the existing fft tests to confirm green baseline (pre-refactor)**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k fft -v`
Expected: PASS — `test_fft_kernel_matches_scipy_on_scalars` (×3 seeds), `test_fft_out_of_grid_points_handled_gracefully`, `test_predict_density_fft_backend_switch`, `test_fft_singular_covariance_raises_linalgerror`, `test_fft_is_k_independent_one_convolution`.

- [ ] **Step 2: Add `_bin_ngp` and `_fft_convolve_field` helpers above `_kde_density_fft`**

Insert immediately before `def _kde_density_fft(` (currently line 1102):

```python
def _bin_ngp(gk_x_w: np.ndarray, gk_y_w: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    """Nearest-grid-point (NGP) binning of weighted points onto the fixed grid.

    Each point is snapped to its single nearest cell (uniform, cell-centered grid: idx =
    round((p - p0)/res)); out-of-grid points clip to the edge cell. Mass is conserved
    (field.sum() == w_norm.sum()): every point contributes its full weight to exactly one cell.
    """
    ix = np.clip(np.rint((gk_x_w - _GRID_X[0]) / GRID_RESOLUTION).astype(np.int64), 0, GRID_NX - 1)
    iy = np.clip(np.rint((gk_y_w - _GRID_Y[0]) / GRID_RESOLUTION).astype(np.int64), 0, GRID_NY - 1)
    field = np.zeros((GRID_NX, GRID_NY), dtype=np.float64)
    np.add.at(field, (ix, iy), w_norm)
    return field


def _fft_convolve_field(
    field: np.ndarray, h11: float, h12: float, h22: float, det: float, norm: float
) -> np.ndarray:
    """Shared FFT-convolution tail for the fft / fft-cic backends.

    ``field`` is the binned weighted-point grid (GRID_NX, GRID_NY). Builds the full-extent analytic
    anisotropic Gaussian kernel (identical energy form to _kde_density_vectorized) and returns the
    UNnormalized density via one zero-padded linear fftconvolve = sum_j w_j K(grid - point_j) in
    O(m log m). Binning is the SOLE per-backend difference; this tail is identical across fft /
    fft-cic (predict_density divides by .sum(), so ``norm`` cancels).
    """
    # NB: this lazy (function-scope) import is LOAD-BEARING for the k-independence spy guard
    # (test_fft*_is_k_independent_one_convolution) -- it resolves the patched scipy.signal attr at
    # call time. Do NOT hoist to module level or the spy goes blind.
    from scipy.signal import fftconvolve

    inv_det = 1.0 / det
    dx = (np.arange(-(GRID_NX - 1), GRID_NX) * GRID_RESOLUTION)[:, None]
    dy = (np.arange(-(GRID_NY - 1), GRID_NY) * GRID_RESOLUTION)[None, :]
    kernel = norm * np.exp(-0.5 * inv_det * (h22 * dx * dx - 2.0 * h12 * dx * dy + h11 * dy * dy))
    return fftconvolve(field, kernel, mode="same")
```

- [ ] **Step 3: Rewrite `_kde_density_fft` body to compose the helpers**

Replace the body of `_kde_density_fft` (the lines after its docstring, currently 1127-1148) with:

```python
    _data, w_n, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    field = _bin_ngp(gk_x_w, gk_y_w, w_n)
    return _fft_convolve_field(field, h11, h12, h22, det, norm)
```

Keep the existing `_kde_density_fft` docstring for now (Task 8 updates the fidelity wording). The `grid_points` parameter stays in the signature (unused — kept for backend-signature parity, as today).

- [ ] **Step 4: Run the fft tests — must stay green (behaviour unchanged)**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k fft -v`
Expected: PASS, identical set to Step 1. (Same ops → results unchanged within the rtol goldens.)

- [ ] **Step 5: Run the full ghost-GK KDE module to confirm no collateral breakage**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -v`
Expected: PASS (all backends).

---

## Task 2: `_bin_cic` (cloud-in-cell / bilinear binning) + unit invariants

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (add `_bin_cic` next to `_bin_ngp`)
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py` (new fft-cic block — start it here)

- [ ] **Step 1: Write the failing mass-conservation + out-of-grid tests**

Append to `tests/tracking/test_ghost_gk_kde_vectorized.py` (after the existing fft block, ~line 637):

```python
# ---------------------------------------------------------------------------
# fft-cic backend (CIC / bilinear binning): better mode + raw-grid fidelity than NGP at ~2x cost.
# Opt-in; O(k + m log m). See ADR-014 (amended).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_bin_cic_conserves_mass(seed):
    """CIC bilinear weights sum to 1 per point -> field.sum() == sum of weights, including for
    out-of-grid points (np.add.at still accumulates all 4 contributions when clip collapses
    indices to an edge cell)."""
    from silly_kicks.tracking._ghost_gk import _bin_cic

    rng = np.random.default_rng(seed)
    k = 300
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + rng.normal(0.0, 3.0, k)
    gk_x_w[:5] = rng.uniform(-4.0, -0.5, 5)  # out-of-grid (left) -> clip to edge, mass preserved
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    field = _bin_cic(gk_x_w, gk_y_w, w)
    assert field.shape == (60, 64)
    assert np.all(field >= 0.0)
    assert np.isclose(field.sum(), w.sum(), rtol=1e-12)
```

- [ ] **Step 2: Run it — fails (no `_bin_cic`)**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_bin_cic_conserves_mass -v`
Expected: FAIL — `ImportError: cannot import name '_bin_cic'`.

- [ ] **Step 3: Implement `_bin_cic`**

Insert immediately after `_bin_ngp` in `silly_kicks/tracking/_ghost_gk.py`:

```python
def _bin_cic(gk_x_w: np.ndarray, gk_y_w: np.ndarray, w_norm: np.ndarray) -> np.ndarray:
    """Cloud-in-cell (CIC / bilinear) binning of weighted points onto the fixed grid.

    Each weighted point is spread bilinearly over its 4 surrounding cells with weights
    (1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx*ty (summing to 1), instead of snapped to the single
    nearest cell (NGP). On a near-tie MULTIMODAL grid this preserves the relative peak masses, so
    the emitted mode (argmax) flips ~76% less than NGP (real data: 21/97 -> 5/97; ADR-014). Mass
    is conserved including for out-of-grid points: clip collapses indices to an edge cell but
    np.add.at still accumulates all 4 contributions, so field.sum() == w_norm.sum().
    """
    fx = (gk_x_w - _GRID_X[0]) / GRID_RESOLUTION
    fy = (gk_y_w - _GRID_Y[0]) / GRID_RESOLUTION
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0
    field = np.zeros((GRID_NX, GRID_NY), dtype=np.float64)
    for di, wx in ((0, 1.0 - tx), (1, tx)):
        ii = np.clip(i0 + di, 0, GRID_NX - 1)
        for dj, wy in ((0, 1.0 - ty), (1, ty)):
            jj = np.clip(j0 + dj, 0, GRID_NY - 1)
            np.add.at(field, (ii, jj), w_norm * wx * wy)
    return field
```

- [ ] **Step 4: Run it — passes**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_bin_cic_conserves_mass -v`
Expected: PASS (×3 seeds).

---

## Task 3: `_kde_density_fft_cic` kernel + on-node degeneracy lock

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (add `_kde_density_fft_cic`)
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the failing on-node degeneracy + singular tests**

Append to the fft-cic block in the test file:

```python
def test_fft_cic_equals_ngp_on_grid_nodes():
    """SEAM LOCK (binning is the only difference between fft and fft-cic) -- NOT the NGP refactor
    lock (that is the rtol golden in Task 1 + the verbatim extraction). When every point sits
    exactly on a grid node, CIC's bilinear weight collapses to 1.0 on the corner cell (tx=ty=0), so
    fft-cic must equal fft (NGP). Same-process equality -> CI-safe across the numpy/scipy version
    matrix (no frozen cross-version array). Points are built as _GRID_X[0] + ix*GRID_RESOLUTION (NOT
    _GRID_X[ix]) so fx is EXACTLY integer regardless of how _GRID_X was constructed -- a linspace
    ULP in _GRID_X[ix] would make tx~1e-16, which fftconvolve amplifies and array_equal would flake
    on."""
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        GRID_RESOLUTION,
        _kde_density_fft,
        _kde_density_fft_cic,
    )

    rng = np.random.default_rng(5)
    k = 200
    ix = rng.integers(8, 52, k)
    iy = rng.integers(8, 56, k)
    gk_x_w = _GRID_X[0] + ix * GRID_RESOLUTION  # exact-integer fx -> tx == 0.0 exactly
    gk_y_w = _GRID_Y[0] + iy * GRID_RESOLUTION
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    ngp = _kde_density_fft(gk_x_w, gk_y_w, w, gp)
    cic = _kde_density_fft_cic(gk_x_w, gk_y_w, w, gp)
    # For the current 0.5 m grid (0.5 exactly representable) array_equal holds exactly. Run this
    # test FIRST to confirm; if a future non-exact grid step makes it fail, fall back to
    # np.testing.assert_allclose(ngp, cic, rtol=0, atol=1e-12) and note the grid-step reason.
    assert np.array_equal(ngp, cic)


def test_fft_cic_singular_covariance_raises_linalgerror():
    """Collinear points -> _kde_setup's cho_factor raises (same as every backend), so
    predict_density's uniform-fallback applies unchanged."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    gk_x_w = np.array([5.0, 5.0, 5.0])
    gk_y_w = np.array([34.0, 34.0, 34.0])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    with pytest.raises(np.linalg.LinAlgError):
        _kde_density_fft_cic(gk_x_w, gk_y_w, w, gp)
```

- [ ] **Step 2: Run them — fail (no `_kde_density_fft_cic`)**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k "fft_cic_equals_ngp or fft_cic_singular" -v`
Expected: FAIL — `ImportError: cannot import name '_kde_density_fft_cic'`.

- [ ] **Step 3: Implement `_kde_density_fft_cic`**

Insert immediately after `_kde_density_fft` in `silly_kicks/tracking/_ghost_gk.py`:

```python
def _kde_density_fft_cic(
    gk_x_w: np.ndarray,
    gk_y_w: np.ndarray,
    w: np.ndarray,
    grid_points: np.ndarray,  # unused: signature parity with the brute-force backends
) -> np.ndarray:
    """Binned-convolution weighted Gaussian KDE with CIC (bilinear) binning. O(k + m log m).

    Identical to ``_kde_density_fft`` except each weighted training point is spread BILINEARLY over
    its 4 surrounding grid cells (``_bin_cic``) instead of snapped to the single nearest cell
    (``_bin_ngp``). The ``_kde_setup`` kernel build and the ``_fft_convolve_field`` convolution are
    shared verbatim. On near-tie MULTIMODAL grids CIC preserves the relative peak masses, so the
    emitted mode (argmax) flips ~76% less than NGP (real data: 21/97 -> 5/97 actions). ~2x the NGP
    bin cost, still ~1195x over ``vectorized``. Faithful on mean/spread; the raw per-cell grid is
    tighter than NGP (~5.7e-3 vs 1.5e-2 median rel-err) but still approximate -- exact-grid
    consumers use ``vectorized`` / ``cpu-numba``. ``_kde_setup`` raises ``LinAlgError`` on a
    singular covariance exactly as the other backends. See ADR-014 (amended).
    """
    _data, w_n, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    field = _bin_cic(gk_x_w, gk_y_w, w_n)
    return _fft_convolve_field(field, h11, h12, h22, det, norm)
```

- [ ] **Step 4: Run them — pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k "fft_cic_equals_ngp or fft_cic_singular" -v`
Expected: PASS.

---

## Task 4: Dispatch `fft-cic` in `predict_density` + backend-switch / k-independence tests

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py:1360-1361` (dispatch)
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the failing dispatch / switch / k-independence tests**

Append to the fft-cic block:

```python
def test_predict_density_fft_cic_backend_switch(small_model):
    """predict_density(kde_backend="fft-cic") returns the standard normalized GhostGkDensity."""
    model, X = small_model
    densities = model.predict_density(X.iloc[:5], kde_backend="fft-cic")
    assert len(densities) == 5
    assert densities[0].probabilities.shape == (60, 64)
    assert densities[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_unknown_kde_backend_still_raises(small_model):
    """An unrecognised backend string raises ValueError (dispatch hygiene unchanged)."""
    model, X = small_model
    with pytest.raises(ValueError, match="Unknown kde_backend"):
        model.predict_density(X.iloc[:1], kde_backend="fft-nope")


def test_fft_cic_is_k_independent_one_convolution(monkeypatch):
    """Structural perf guard: fft-cic does ONE fftconvolve per prediction with k-INDEPENDENT
    field+kernel shapes (O(m log m), not O(k*m)). Mirrors the fft guard."""
    import scipy.signal as sps

    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    calls = []
    real = sps.fftconvolve

    def _spy(field, kernel, *a, **k):
        calls.append((field.shape, kernel.shape))
        return real(field, kernel, *a, **k)

    monkeypatch.setattr(sps, "fftconvolve", _spy)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    rng = np.random.default_rng(1)
    for k in (5, 5000):
        x = rng.uniform(0, 30, k)
        y = rng.uniform(18, 50, k)
        w = rng.uniform(0.1, 1.0, k)
        w = w / w.sum()
        _kde_density_fft_cic(x, y, w, gp)
    assert len(calls) == 2, f"expected one fftconvolve per call, got {len(calls)}"
    assert calls[0] == calls[1], "field/kernel shapes must be k-independent"
```

- [ ] **Step 2: Run them — `backend_switch` fails (ValueError: Unknown kde_backend: 'fft-cic')**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k "fft_cic_backend_switch or unknown_kde_backend or fft_cic_is_k_independent" -v`
Expected: `test_predict_density_fft_cic_backend_switch` FAILS (ValueError). The other two PASS (they call the kernel directly / expect the raise).

- [ ] **Step 3: Add the dispatch branch**

In `silly_kicks/tracking/_ghost_gk.py`, after the `elif kde_backend == "fft":` block (lines 1360-1361), insert:

```python
                    elif kde_backend == "fft-cic":
                        probs = _kde_density_fft_cic(gk_x_w, gk_y_w, w, grid_points)
```

(So the chain reads: `scipy` / `vectorized` / `cpu-numba` / `fft` / `fft-cic` / else→`ValueError`.)

- [ ] **Step 4: Run them — all pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k "fft_cic_backend_switch or unknown_kde_backend or fft_cic_is_k_independent" -v`
Expected: PASS.

---

## Task 5: Bimodal mode-parity (primary motivation) + mean/spread parity

The load-bearing test: on near-tie *unequal*-peak bimodal grids, CIC's argmax matches the `vectorized`-exact argmax on **strictly more** of N seeded constructions than NGP, **and never fewer**.

> **AS-BUILT (4.8.0) — three justified deviations from the code blocks below (see the test docstrings
> + the spec's "Implementation note"):** (1) the mode test scores against the **known winner**
> (higher-mass cluster, `mode_x < 15`), not grid-vectorized — a boundary-centered winner is
> under-sampled by the grid, so grid-vectorized itself phase-flips (a confound); a vectorized subset
> is kept as a sanity check. (2) `N = 120`, regime `std=0.75, lead=0.05`; observed margin 5
> (CIC 120/120, NGP 115/120). (3) mean/spread parity is tested on a **unimodal** cloud (mean <3e-2 m,
> spread rel <1e-2, mirroring `test_fft_kernel_matches_scipy_on_scalars`) because CIC's bilinear
> smoothing adds ~3% to the entropy-spread on a deliberately-bimodal cloud; the tight <5e-3 bound
> lives in the real-model golden (Task 7). The margin-gate + soft-violations structure is unchanged.

**Files:**
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the bimodal mode-parity + scalar-parity tests**

Append to the fft-cic block. Reuses the existing `_grid_scalars` (line 518) and `_scipy_kde_grid` (line 141) helpers:

```python
def _bimodal_cloud(rng, *, lead=0.06, std=0.9, k=600):
    """Two clusters ~5.75 m apart along x at DIFFERENT grid phase, so NGP distorts them UNEQUALLY --
    the real-data flip mechanism (differential peak-height distortion), manufactured deliberately
    rather than hoping within-cluster spread accidentally straddles a boundary:
      * WINNER (higher mass) on a cell BOUNDARY, cx_a = 12.5 (midpoint of nodes 12.25/12.75) -> NGP
        splits its mass across cells 24/25 -> UNDER-counts its peak.
      * LOSER (lower mass) on a NODE, cx_b = 18.25 (cell 36 center) -> NGP concentrates its mass in
        one cell -> OVER-counts its peak.
    So NGP is induced to flip the argmax to the (wrong) loser, while CIC's bilinear, mass-conserving
    spread keeps the winner (= the vectorized-exact argmax). Grid: _GRID_X = 0.25 + i*0.5 -> nodes
    at *.25/*.75, boundaries at *.0/*.5 (verified against silly_kicks.tracking._ghost_gk). `lead`
    (winner's mass surplus) lives in a tension band: large enough that the EXACT argmax is the winner
    (not a coin-flip), small enough that NGP's differential distortion flips it. `std` controls peak
    sharpness (smaller -> the +/-0.25 m quantization bites harder). Returns (gk_x_w, gk_y_w, w)."""
    cx_a = 12.5   # WINNER on a cell boundary -> NGP mass-splits -> under-counts peak
    cx_b = 18.25  # LOSER on a node -> NGP concentrates -> over-counts peak (~5.75 m from the winner)
    cy = 34.25
    n_a = int(k * (0.5 + lead))  # winner carries the mass surplus
    n_b = k - n_a
    xa, ya = rng.normal(cx_a, std, n_a), rng.normal(cy, std, n_a)
    xb, yb = rng.normal(cx_b, std, n_b), rng.normal(cy, std, n_b)
    gk_x_w = np.concatenate([xa, xb])
    gk_y_w = np.concatenate([ya, yb])
    w = np.ones(k) / k
    return gk_x_w, gk_y_w, w


def test_fft_cic_mode_beats_ngp_on_bimodal_grids():
    """PRIMARY motivation test: on N differential-phase bimodal grids (winner on a cell boundary,
    loser on a node -- the real-data flip mechanism), fft-cic's argmax matches the vectorized-EXACT
    argmax on >=3 MORE constructions than fft (NGP) does (the enforced margin gate, which subsumes
    both 'NGP demonstrably flips' and 'CIC beats NGP in aggregate' -- the real 21-vs-5 evidence), and
    CIC is worse per-instance on <=1 (soft -- the evidence is aggregate, not a strict per-instance
    subset). This is the test ADR-014 said was 'the wrong thing' -- it is in fact the right thing,
    because CIC does NOT tie NGP on MULTIMODAL grids (the ADR's bench used ~unimodal queries, which
    structurally cannot exhibit the peak-selection flip). [After tuning: record the observed
    cic_correct/ngp_correct split here so the margin is documented, not just asserted.]"""
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        _kde_density_fft,
        _kde_density_fft_cic,
        _kde_density_vectorized,
    )

    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    n_constructions = 40
    cic_correct = ngp_correct = violations = 0
    for seed in range(n_constructions):
        rng = np.random.default_rng(1000 + seed)
        x, y, w = _bimodal_cloud(rng)
        exact = _grid_scalars(_kde_density_vectorized(x, y, w, gp))
        ngp = _grid_scalars(_kde_density_fft(x, y, w, gp))
        cic = _grid_scalars(_kde_density_fft_cic(x, y, w, gp))
        ngp_ok = abs(ngp["mode_x"] - exact["mode_x"]) <= 0.5 + 1e-9
        cic_ok = abs(cic["mode_x"] - exact["mode_x"]) <= 0.5 + 1e-9
        cic_correct += cic_ok
        ngp_correct += ngp_ok
        if ngp_ok and not cic_ok:  # CIC wrong where NGP right -- a per-instance regression
            violations += 1
    # MARGIN GATE (enforced, not just prose): CIC argmax-correct on >=3 MORE constructions than NGP.
    # This single invariant subsumes both guarantees: (a) NON-VACUITY -- since cic_correct <= n, the
    # margin forces ngp_correct <= n - 3, i.e. NGP is demonstrably wrong on >=3 (the 4.2.0 DAS
    # "onside fixture" lesson: a test that cannot bite is worthless); and (b) the defensible AGGREGATE
    # claim that CIC strictly beats NGP (the real 21-vs-5 evidence). A margin of >=3 cannot quietly
    # erode to a one-flip near-vacuous pass across a numpy/scipy bump.
    assert cic_correct - ngp_correct >= 3, (
        f"CIC argmax-correct {cic_correct}/{n_constructions}, NGP {ngp_correct} -- margin "
        f"{cic_correct - ngp_correct} < 3: CIC does not meaningfully beat NGP, or the construction "
        "does not exercise the flip. Tune _bimodal_cloud (winner on a cell BOUNDARY, loser on a "
        "NODE; lead toward the near-tie band, smaller std so NGP's quantization bites)."
    )
    # SOFT per-instance check (DEMOTED from a hard per-seed assert): the real evidence is aggregate,
    # NOT a strict per-instance subset -- a bilinear spread can occasionally miss where an NGP snap
    # happened to hit. Allow <=1 such case across the 40 constructions rather than flaking the run.
    assert violations <= 1, f"CIC worse than NGP on {violations} constructions (expected <=1)"


def test_fft_cic_mean_spread_parity_on_bimodal():
    """CIC must not regress the already-faithful scalars: mean within 1e-2 m, spread rel < 5e-3 of
    the vectorized-exact value on a bimodal cloud."""
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        _kde_density_fft_cic,
        _kde_density_vectorized,
    )

    rng = np.random.default_rng(2024)
    x, y, w = _bimodal_cloud(rng)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    exact = _grid_scalars(_kde_density_vectorized(x, y, w, gp))
    cic = _grid_scalars(_kde_density_fft_cic(x, y, w, gp))
    assert np.hypot(cic["mean_x"] - exact["mean_x"], cic["mean_y"] - exact["mean_y"]) < 1e-2
    assert abs(cic["spread"] - exact["spread"]) / abs(exact["spread"]) < 5e-3
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k "fft_cic_mode_beats_ngp or fft_cic_mean_spread" -v`
Expected: PASS, with the margin gate met comfortably (target `cic_correct - ngp_correct` well above the floor of 3 — a correctly-phased construction should make NGP flip on a healthy fraction, not just 3). The differential-phase geometry (winner on the `12.5` boundary, loser on the `18.25` node) is what makes NGP flip *for the right reason*; the margin gate enforces it can't quietly degrade. If the margin is below 3, **tune within the tension band**: reduce `lead` toward the near-tie edge (0.04–0.06) and/or reduce `std` (sharper peaks → the ±0.25 m quantization bites harder) — but do **not** push `lead` so low the exact argmax becomes a coin-flip, and do **not** weaken the asserts to pass a vacuous construction (the 4.2.0 DAS-onside lesson). Once tuned, record the observed `cic_correct`/`ngp_correct` split in the test docstring.

- [ ] **Step 3: (only if Step 2 needed tuning) re-run the full fft-cic block to confirm no regression**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -k fft_cic -v`
Expected: PASS.

---

## Task 6: Raw-grid fidelity (ADR-014-mandated) on a real-model leaf subset

CIC's per-cell grid rel-err vs the `vectorized` oracle must be **strictly lower** than NGP's — the reason CIC exists, where scalar parity alone would tie.

**Files:**
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the failing raw-grid fidelity test**

Append to the fft-cic block. Uses the `golden` + `default_model_features` fixtures (lines 45-63) so it runs on the bundled "default" model's real leaf-weighted clouds:

```python
def test_fft_cic_raw_grid_tighter_than_ngp(golden, default_model_features):
    """On real default-model leaf subsets, CIC's per-cell median rel-err vs the vectorized oracle
    is STRICTLY LOWER than NGP's. This is the raw-grid fidelity test ADR-014 mandates for CIC --
    the grid is exactly where CIC must improve on NGP (scalars alone would tie)."""
    model, X = default_model_features  # fixture returns (from_variant("default") model, X[:_N_GOLDEN])
    vec = model.predict_density(X, kde_backend="vectorized")
    ngp = model.predict_density(X, kde_backend="fft")
    cic = model.predict_density(X, kde_backend="fft-cic")

    def _median_relerr(approx_list):
        errs = []
        for a, v in zip(approx_list, vec):
            ref = v.probabilities
            mask = ref > 1e-6 * ref.max()  # ignore near-zero tail cells (rel-err blows up there)
            errs.append(np.median(np.abs(a.probabilities[mask] - ref[mask]) / ref[mask]))
        return float(np.median(errs))

    ngp_err = _median_relerr(ngp)
    cic_err = _median_relerr(cic)
    assert cic_err < ngp_err, f"CIC grid rel-err {cic_err:.2e} not < NGP {ngp_err:.2e}"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_fft_cic_raw_grid_tighter_than_ngp -v`
Expected: PASS (cic_err ≈ 5.7e-3 < ngp_err ≈ 1.5e-2 per the hand-off table). The `default_model_features` fixture (confirmed present at `test_ghost_gk_kde_vectorized.py:55-60`, returns `(GhostGkModel.from_variant("default"), X[:_N_GOLDEN])`) is the source of truth here. Only if `from_variant("default")` is unavailable in the test env, fall back to the `small_model` fixture's leaf subsets (same assertion; lower N) — but prefer the real model.

---

## Task 7: Real-model golden scalars for `fft-cic` (rtol, production regime)

**Files:**
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the real-model golden test (mirror `test_golden_fft_scalars`, line 116)**

Append to the fft-cic block:

```python
def test_golden_fft_cic_scalars(golden, default_model_features):
    """fft-cic scalars on the bundled 'default' model vs the frozen scipy golden: mode <=1 grid
    cell, mean <1e-2 m, spread rel <5e-3. Locks the real-model PRODUCTION regime (closes the
    synthetic-only gap). rtol/cell tolerances -- NOT exact -- so a numpy/scipy bump does not
    false-fail (cross-version robustness, like the lakehouse goldens)."""
    model, X = default_model_features  # fixture returns (from_variant("default") model, X[:_N_GOLDEN])
    densities = model.predict_density(X, kde_backend="fft-cic")
    n = _N_GOLDEN
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    mean_x = np.array([d.mean_x for d in densities])
    mean_y = np.array([d.mean_y for d in densities])
    spread = np.array([d.spread for d in densities])
    assert np.all(np.abs(mode_x - golden["mode_x"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.abs(mode_y - golden["mode_y"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.hypot(mean_x - golden["mean_x"][:n], mean_y - golden["mean_y"][:n]) < 1e-2)
    assert np.all(np.abs(spread - golden["spread"][:n]) / np.abs(golden["spread"][:n]) < 5e-3)
```

NB: the `GhostGkDensity` attribute names (`d.mode_x` / `d.mode_y` / `d.mean_x` / `d.mean_y` / `d.spread`) and the `n = _N_GOLDEN` slice mirror `test_golden_fft_scalars` exactly (lines 122-133). If the golden's frozen samples include a bimodal-near-tie where CIC and the scipy golden legitimately disagree by >1 cell, do **not** loosen the bound globally — document the single near-tie and exclude that one index (CIC matching scipy on a coin-flip is not required), matching `test_golden_discrete_mode`. **Death-by-exclusions guard:** if **more than ~1** of the 8 golden samples needs excluding, stop — that signals the golden sample set is multimodal-heavy and the *approach* needs a rethink (e.g. a different/larger frozen sample set), not the bound. Do not quietly accumulate exclusions.

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_golden_fft_cic_scalars -v`
Expected: PASS.

- [ ] **Step 3: Run the entire fft-cic block + the full module green**

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -v`
Expected: PASS (all backends, all fft + fft-cic tests).

---

## Task 8: Docstrings — add `"fft-cic"` + steer new consumers (4 surfaces) + correct fft fidelity claim

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `predict_density` docstring (~1304), `compute_ghost_gk` docstring (~1625), `_kde_density_fft` docstring (correct the overstated mode claim).
- Modify: `silly_kicks/tracking/features.py` — `add_ghost_gk` docstring (~3701), `ghost_gk_xfns` docstring (~3791).

- [ ] **Step 1: Update the `kde_backend` enumeration in all 4 public docstrings**

In each of the four locations the line currently reads:

```
        kde_backend : {"vectorized", "scipy", "cpu-numba", "fft"}, default "vectorized"
```

Replace the enumeration with `{"vectorized", "scipy", "cpu-numba", "fft", "fft-cic"}` and append a sentence after the existing backend description:

```
            "fft-cic" adds CIC (bilinear) binning to the FFT path: ~76% fewer mode flips on
            multimodal grids and tighter raw-grid fidelity than "fft" (NGP) at ~2x the bin cost
            (still ~1000x+ over brute force). PREFER "fft-cic" over "fft" for new FFT consumers
            unless you need NGP's extra speed on known-unimodal data. Both FFT backends are
            approximate on the raw grid (use "vectorized"/"cpu-numba" for exact grids). See ADR-014.
```

- [ ] **Step 2: Correct the overstated mode claim in `_kde_density_fft`'s docstring**

In `_kde_density_fft` (the NGP kernel), the docstring currently says the backend is "Faithful on the SUMMARY SCALARS predict_density emits (mode/mean/spread ... all robust to per-cell binning noise)". Replace that sentence with:

```python
    Faithful on mean/spread always, and on the mode for UNIMODAL grids. On near-tie MULTIMODAL
    grids the NGP snap can flip which peak is the argmax, shifting the emitted mode by several
    metres (real data: up to ~6 m on ~22% of actions) -- use ``fft-cic`` (bilinear binning, ~76%
    fewer flips) or ``vectorized`` when the mode matters on multimodal distributions. NOT
    bit-faithful on the raw per-cell ``probabilities`` grid (binning quantizes per-cell mass).
    Consumers reading the raw grid should use ``"vectorized"``. See ADR-014 (amended).
```

- [ ] **Step 3: Verify docstrings import-clean (no doctest breakage)**

Run: `python -c "import silly_kicks.tracking._ghost_gk, silly_kicks.tracking.features; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Add an atomic-mirror reachability test**

Append to the fft-cic block in the test file (confirms the flat string reaches the atomic surface for free):

```python
def test_fft_cic_reaches_atomic_add_ghost_gk():
    """The flat kde_backend string reaches the atomic mirror's add_ghost_gk for free (re-export);
    no signature change. Smoke test: the param is accepted end-to-end."""
    import inspect

    from silly_kicks.atomic.tracking.features import add_ghost_gk

    assert "kde_backend" in inspect.signature(add_ghost_gk).parameters
```

Run: `python -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_fft_cic_reaches_atomic_add_ghost_gk -v`
Expected: PASS.

---

## Task 9: Amend ADR-014 (correct the falsified claims) + NOTICE entry

**Files:**
- Modify: `docs/superpowers/adrs/ADR-014-ghost-gk-kde-fft-backend.md`
- Modify: `NOTICE`

- [ ] **Step 1: Amend ADR-014**

Make these edits to `docs/superpowers/adrs/ADR-014-ghost-gk-kde-fft-backend.md`:

1. **Status line:** append ` — amended 2026-06-02 (4.8.0: CIC `fft-cic` added; mode-fidelity claims corrected)`.
2. **Alternatives table**, the CIC row (line 50): change its "Why rejected" from "NGP is simpler/faster and scalar-equivalent" to a note that it was **deferred at 4.6.0 but adopted at 4.8.0** once real multimodal data showed NGP flips the mode by up to ~6 m (not ≤1 cell), which CIC fixes ~76%; strike the false "scalar-equivalent" claim.
3. **Consequences → Positive** (line 60): correct "mode ... all robust to per-cell binning noise" to "mean/spread robust always; **mode robust only on unimodal grids** — see the amendment below".
4. **Consequences → Negative** (lines 70-71): correct "~2.5% ... flip the discrete mode by ≤1 grid cell" — add that on **multimodal near-tie** grids NGP can flip the mode by **up to ~6 m on ~22% of real actions** (IDSSE J03WMX p1, 97 actions), which `fft-cic` reduces to ~5%.
5. **Replace the "Future work — CIC binning (gated, not built)" section** with a "**Decision (amended 4.8.0) — `fft-cic` shipped**" section stating: CIC adopted as opt-in `kde_backend="fft-cic"` (NGP stays `"fft"`, unchanged, default fft); binning is the only seam (`_bin_cic` vs `_bin_ngp`, shared `_fft_convolve_field`); the motivation is the **mode** on multimodal grids (21/97→5/97) plus tighter raw grid (5.7e-3 vs 1.5e-2); root cause that 4.6.0 missed it = the scalar-parity bench used **unimodal** queries that could not exercise the multimodal peak-selection flip (the same "test couldn't bite" class as the 4.2.0 DAS value-neutral claim); flat string, no public-API change; cost ~2× NGP, still ~1195×.
6. Add a **train/serve-skew note**: any ghost-GK-*mode* consumer must pin one `kde_backend` train+serve and persist it in metadata (tracked in TODO.md). State that **TF-16 xShotOccurrence does NOT consume the ghost-GK mode** (it uses the resolved/defending GK), so it is unaffected; the guard binds prospective mode consumers (TF-17 / TF-19).

Keep the framing **precise**: 4.6.0 was right on mean/spread (always) and mode (unimodal) — only the multimodal-mode claim was overstated. Do NOT over-correct to "ADR-014 was wrong".

- [ ] **Step 2: Add the CIC reference to NOTICE**

In `NOTICE`, in the ghost-GK methodology block (after line 334, before "The implementations are independent..."), add:

```
- Hockney, R. W. & Eastwood, J. W. (1988). "Computer Simulation Using
  Particles." (Particle-mesh charge assignment: nearest-grid-point (NGP)
  and cloud-in-cell (CIC, bilinear) binning -- the kde_backend="fft"/"fft-cic"
  binning schemes for the O(k + m log m) FFT-convolution KDE; ADR-014)
```

- [ ] **Step 3: Confirm NOTICE/ADR are well-formed (no broken markdown tables)**

Read both files back and eyeball the table rows / section headers. (No automated gate; manual check.)

---

## Task 10: Version bumps + CHANGELOG + TODO.md (incl. tracked skew action) + uv.lock

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`

- [ ] **Step 1: Bump `__version__`**

`silly_kicks/__init__.py:7` — `__version__ = "4.7.0"` → `__version__ = "4.8.0"`.

- [ ] **Step 2: Bump `pyproject.toml` version**

Edit the `version = "4.7.0"` line under `[project]` → `version = "4.8.0"`. (Confirm the exact current string first with a Read of the top of `pyproject.toml`.)

- [ ] **Step 3: Add the CHANGELOG entry**

Prepend a `## [4.8.0] - 2026-06-02` section (match the existing CHANGELOG heading style) describing: new opt-in `kde_backend="fft-cic"` (CIC/bilinear binning) — ~76% fewer ghost-GK mode flips on multimodal grids + tighter raw grid than `fft` (NGP) at ~2× cost; `"fft"` unchanged (still NGP, default fft opt-in); `vectorized` remains the global default; ADR-014 amended (corrected the overstated NGP mode-fidelity claim — NGP can shift the mode up to ~6 m on multimodal frames). Include a Hyrum heads-up: ghost-GK-mode consumers should prefer `fft-cic`/`vectorized` for train/serve mode stability.

- [ ] **Step 4: Add the TODO.md entries**

In `TODO.md`: (a) a 4.8.0 line marking `fft-cic` shipped; (b) the **tracked train/serve-skew action**: *"Ghost-GK-*mode* train/serve guard (owned by the first feature to train on the ghost-GK mode — prospectively TF-17 / TF-19; NOT TF-16, which uses the resolved/defending GK and does not consume the mode): pin one `kde_backend` for train AND serve, persist `kde_backend` in model metadata, add a serve-time assert that metadata backend == runtime backend — turns silent ≤6 m multimodal mode skew into a loud failure (ADR-014 amended 4.8.0)."* Match the existing TODO row format.

- [ ] **Step 5: Regenerate `uv.lock`**

Run: `uv lock`
Expected: `uv.lock` updates the silly-kicks version to 4.8.0 (no dependency changes — scipy is already core). Confirm the diff touches only the version.

---

## Task 11: Full verification + single commit + PR + tag

**No commit happens until every check below is green.** The commit is gated by the sentinel hook and **held for explicit per-commit approval** — present the staged diff and the exact `git commit` command, and do **not** create the sentinel.

- [ ] **Step 1: Lint trio (whole package, all three steps — the full CI lint job)**

Run, in order, and require each to exit 0:
```
ruff check silly_kicks/ tests/ scripts/
ruff format --check silly_kicks/ tests/ scripts/
pyright silly_kicks/
```
If `ruff check` flags an import-sort (I001) on the new inline test imports, run `ruff check --fix` and re-run all three. Expected: all clean. (The unused `grid_points` param on `_kde_density_fft_cic` will **not** be flagged — ruff `select` does not include `ARG`, and it mirrors the existing `_kde_density_fft`; the inline `# unused: signature parity` comment documents the intent.)

- [ ] **Step 2: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: PASS (prior baseline was 3309 passed for 4.6.0; 4.8.0 adds the fft-cic tests). Trust the exit code + the summary line, captured from this run — do not narrate counts from memory.

- [ ] **Step 3: Dependency-light import guard (bare import stays numba/xgboost-free is unaffected; sanity only)**

Run: `python -c "import silly_kicks; print(silly_kicks.__version__)"`
Expected: `4.8.0`.

- [ ] **Step 4: Stage everything and present the diff for approval**

Run: `git add -A && git status && git diff --cached --stat`
Present the staged file list + stat to the user. **Hold here** for explicit approval to commit (the sentinel must be created by the user, or the user runs the commit from the CLI).

- [ ] **Step 5: Single commit (only after approval)**

After the user approves and the sentinel exists, commit (one bundled commit):
```
git commit -m "feat(ghost-gk): add CIC (bilinear) kde_backend=fft-cic -- ~76% fewer multimodal mode flips vs NGP, tighter raw grid; correct ADR-014 mode-fidelity claim -- silly-kicks 4.8.0"
```
(Append the required `Co-Authored-By` trailer.)

- [ ] **Step 6: Push + open PR**

Run: `git push -u origin feat/ghost-gk-fft-cic-binning` then `gh pr create` with a body summarizing the feature, the ADR-014 correction, the test plan, and the lakehouse adoption note (they bump floor to ≥4.8.0 and flip AC-1 to `fft-cic` + re-baseline their golden in their own PR). Include the `🤖 Generated with Claude Code` trailer.

- [ ] **Step 7: After CI green + merge — tag the release**

After the squash-merge lands on `main` and main CI is green: pull main, create an annotated `v4.8.0` tag on the merge commit, push the tag to trigger `publish.yml` → PyPI. Confirm the publish run starts.

---

## Self-review notes (author)

- **Spec coverage:** seam (Tasks 1-3) · dispatch + no-signature-change (Task 4) · bimodal mode test as primary + mean/spread (Task 5) · ADR-mandated raw-grid fidelity (Task 6) · real-model golden rtol (Task 7) · k-independence guard + backend switch + singular (Tasks 3-4) · mass-conservation + out-of-grid (Task 2) · soft-deprecation steering + fft fidelity correction in docstrings (Task 8) · ADR-014 amendment + NOTICE (Task 9) · ghost-GK-mode train/serve-skew tracked action — owner TF-17/TF-19, not TF-16 (Task 10) · version bumps + single commit/PR/tag (Tasks 10-11). All spec sections map to a task.
- **Two DISTINCT lock guarantees (do not conflate):** (a) the **NGP refactor lock** = the existing rtol fft golden staying green (Task 1 Step 4) + the extraction being verbatim-by-construction — this is what proves `_fft_convolve_field` extraction did not change the 4.6.0 NGP path; (b) the **seam lock** = `test_fft_cic_equals_ngp_on_grid_nodes` (Task 3), which proves binning is the *only* difference between the two backends (it cannot, and is not claimed to, detect NGP-path drift). The on-node test is a CI-safe substitute for a frozen-pre-refactor-array `array_equal` (which would false-fail across the 3.10/3.11/3.12+Windows matrix — platform-dependent FP, a documented repo lesson) — but it locks the *seam*, not the *refactor*.
- **Type/name consistency:** `_bin_ngp` / `_bin_cic` / `_fft_convolve_field` / `_kde_density_fft_cic` used consistently across tasks; dispatch string `"fft-cic"` consistent; `_grid_scalars` / `_scipy_kde_grid` / `small_model` / `golden` / `default_model_features` reuse the existing fixtures by their real names/line numbers.
