# Ghost-GK KDE — Step 1 (numpy closed-form 2×2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `cho_solve` cost (~24% of the ghost-GK KDE kernel) by computing the 2×2 Mahalanobis energy in closed form, while keeping the `cho_factor` PD-branch + `log_det` byte-identical to 4.2.0 — a dependency-free ~1.3–1.4× kernel win, value-equivalent within the existing golden tolerance.

**Architecture:** In `_ghost_gk._kde_density_vectorized`, keep `cho_factor(covariance)` (the cheap 2×2 factorization that decides the singular→uniform branch and supplies `log_det` — *unchanged so the fallback set stays exactly 4.2.0's*), and replace only the expensive `cho_solve`-over-`(2, kb·m)` whitening with the closed-form energy `0.5/det·(h₂₂·dx² − 2·h₁₂·dx·dy + h₁₁·dy²)`. This also drops the `(2,kb,m)` `diff`/`tdiff` temporaries.

**Tech Stack:** Python 3.10, numpy 2.x, scipy 1.15 (`scipy.linalg.cho_factor` retained; `cho_solve` removed from this kernel), pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md` §3 (PLAN-READY, lakehouse rounds 1–3).

**Commit policy:** Per project workflow (one feature branch → single commit → PR; squash-merge), this plan uses a **single commit at the end** (Task 3), not per-task commits.

---

## File structure

**Modified:**
- `silly_kicks/tracking/_ghost_gk.py` — `_kde_density_vectorized` loop body: `cho_solve` whitening → closed-form energy; drop the `cho_solve` import (keep `cho_factor`).

**Created/extended (tests):**
- `tests/tracking/test_ghost_gk_kde_vectorized.py` — add a near-singular characterization parity test + a structural "no `cho_solve`" guard.

**Unchanged (must keep passing — golden-anchor, do NOT regenerate):**
- `tests/tracking/fixtures/ghost_gk_kde_golden.npz` (frozen scipy-Cholesky values) and the existing `test_golden_*`, `test_vectorized_kernel_matches_scipy`, degenerate, model-traveling tests.

**Current kernel (for reference — `silly_kicks/tracking/_ghost_gk.py`):** `_kde_density_vectorized` computes `covariance` (2×2), `chol = cho_factor(covariance, lower=True)`, `log_det = 2·Σlog(diag(chol[0]))`, `norm = exp(-0.5·(log_det + d·log2π))`, then loops train-blocks: `diff = grid_points[:,None,:] - data[:,sl,None]` → `tdiff = cho_solve(chol, flat)` → `energy = 0.5·Σ(flat·tdiff)` → `out += einsum("k,km->m", w[sl], exp(-energy))`.

---

## Task 1: Near-singular characterization test (lock the ill-conditioned zone)

The closed-form `1/det` is less numerically stable than Cholesky as `det→0` (near-collinear players → near-degenerate-but-PD covariance). The existing golden uses random standard-normal features that rarely produce ill-conditioned covariances, so add an explicit near-singular case **before** the change, to catch any precision loss. It must pass on the current `cho_solve` kernel (characterization).

**Files:**
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the near-singular parity test (sharp Leg-B anchor + scipy backstop)**

Append to `tests/tracking/test_ghost_gk_kde_vectorized.py`. `_cho_solve_kde_grid` is a
**version-independent** reproduction of the 4.2.0 `cho_solve` whitening — so after the Step-2
change it isolates exactly the closed-form-vs-Cholesky drift (Leg-B), not scipy-vs-both noise.
`_scipy_kde_grid` (already in this file, line 107) is the Leg-A backstop.

```python
def _cho_solve_kde_grid(gk_x_w, gk_y_w, w, grid_points):
    """4.2.0's cho_solve whitening, reproduced version-independently (Leg-B reference)."""
    from scipy.linalg import cho_factor, cho_solve

    from silly_kicks.tracking._ghost_gk import GRID_NX, GRID_NY

    w = np.asarray(w, np.float64)
    w = w / w.sum()
    data = np.vstack([np.asarray(gk_x_w, np.float64), np.asarray(gk_y_w, np.float64)])
    neff = 1.0 / np.sum(w**2)
    factor = neff ** (-1.0 / 6.0)  # Scott, d=2 -> -1/(d+4)
    cov = np.atleast_2d(np.cov(data, rowvar=True, bias=False, aweights=w)) * factor**2
    chol = cho_factor(cov, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
    norm = np.exp(-0.5 * (log_det + 2.0 * np.log(2.0 * np.pi)))
    diff = grid_points[:, :, None] - data[:, None, :]  # (2, m, k)
    tdiff = cho_solve(chol, diff.reshape(2, -1))
    energy = 0.5 * np.sum(diff.reshape(2, -1) * tdiff, axis=0).reshape(grid_points.shape[1], data.shape[1])
    out = (np.exp(-energy) @ w) * norm
    return out.reshape(GRID_NX, GRID_NY)


def _near_singular_inputs():
    # 30 points almost on a line y = 34 + 1e-3*(x-15): tiny off-axis spread -> high 2x2
    # covariance condition number (near-singular but positive-definite).
    rng = np.random.default_rng(123)
    gk_x_w = rng.uniform(0, 30, 30)
    gk_y_w = 34.0 + 1e-3 * (gk_x_w - 15.0) + rng.normal(0, 1e-4, 30)
    w = rng.uniform(0.1, 1.0, 30)
    w = w / w.sum()
    return gk_x_w, gk_y_w, w


def test_kernel_near_singular_parity():
    """Near-singular-but-PD covariance: vectorized kernel matches BOTH the cho_solve
    reference (Leg-B, sharp) and scipy (Leg-A, backstop) within tolerance.
    """
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    gk_x_w, gk_y_w, w = _near_singular_inputs()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    got = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)
    np.testing.assert_allclose(got, _cho_solve_kde_grid(gk_x_w, gk_y_w, w, grid_points), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got, _scipy_kde_grid(gk_x_w, gk_y_w, w, grid_points), rtol=1e-7, atol=1e-12)
```

- [ ] **Step 2: Run it on the current (cho_solve) kernel — expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_kernel_near_singular_parity -v --no-header`
Expected: **PASS** (current kernel == cho_solve reference essentially exactly, and == scipy). If it
FAILS, the geometry is too extreme for the contract — relax the off-axis scale (`1e-3`/`1e-4`) until
the *current* kernel passes; that calibrates the realistic near-singular bar. (The lakehouse will
later supply a real near-singular covariance from IDSSE/skillcorner frames to replace the synthetic
line; the synthetic one is a fine starting calibration.)

(No commit — single commit at Task 3.)

---

## Task 2: Replace `cho_solve` with the closed-form energy (red-first structural guard)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`_kde_density_vectorized`)
- Test: `tests/tracking/test_ghost_gk_kde_vectorized.py`

- [ ] **Step 1: Write the structural "no cho_solve" guard (RED on the current kernel)**

Append:

```python
def test_vectorized_kernel_uses_no_cho_solve(small_model, monkeypatch):
    """Step-1 optimization: the vectorized kernel must NOT call cho_solve (closed-form energy
    replaces it). cho_factor is still used for the PD-branch + log_det.

    Patch scipy.linalg.cho_solve (the MODULE attribute), NOT
    silly_kicks.tracking._ghost_gk.cho_solve: the kernel imports cho_solve *function-scope*
    (`from scipy.linalg import ...` inside _kde_density_vectorized), re-binding it from
    scipy.linalg on every call -> patching scipy.linalg DOES intercept it (verified: 3 calls on
    the pre-change kernel). There is NO module-level _ghost_gk.cho_solve, so a
    `monkeypatch.setattr(_ghost_gk, "cho_solve", ...)` / `hasattr(...)` form would be a no-op
    (always green). Do not "fix" it to that.
    """
    import scipy.linalg as sla

    calls = {"solve": 0}
    real_solve = sla.cho_solve
    monkeypatch.setattr(
        sla, "cho_solve",
        lambda *a, **k: (calls.__setitem__("solve", calls["solve"] + 1), real_solve(*a, **k))[1],
    )
    model, X = small_model
    model.predict_density(X.iloc[:3], kde_backend="vectorized")
    assert calls["solve"] == 0, "vectorized kernel still calls cho_solve"
```

- [ ] **Step 2: Run to verify it FAILS on the current kernel**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_vectorized_kernel_uses_no_cho_solve -v --no-header`
Expected: **FAIL** — `vectorized kernel still calls cho_solve` (the current loop calls `cho_solve` 3×,
one per sample; verified the `scipy.linalg`-patch intercepts the function-scope import).

- [ ] **Step 3: Implement the closed-form energy in `_kde_density_vectorized`**

In `silly_kicks/tracking/_ghost_gk.py`, change the kernel's import line:

```python
    from scipy.linalg import cho_factor  # cho_solve removed: closed-form 2x2 energy below
```

Keep the covariance / `cho_factor` / `log_det` / `norm` block exactly as-is (it anchors the PD-branch
and `log_det` to 4.2.0). Replace the loop block — from `m = grid_points.shape[1]` through
`return out.reshape(GRID_NX, GRID_NY)` — with:

```python
    # Closed-form 2x2 whitening (replaces cho_solve): H = covariance is 2x2 PD (cho_factor
    # succeeded above), so H^-1 = (1/det)[[h22,-h12],[-h12,h11]] and
    #   energy = 0.5 * diff^T H^-1 diff = 0.5/det * (h22*dx^2 - 2*h12*dx*dy + h11*dy^2).
    # Computed directly from dx,dy -> no (2,kb,m) diff/tdiff temporaries. det>0 (PD).
    h11 = covariance[0, 0]
    h12 = covariance[0, 1]
    h22 = covariance[1, 1]
    # Derive det from the SAME factor as log_det/norm: det(H) = det(L)^2 = (L00*L11)^2 (chol[0]
    # holds L on its diagonal). Keeps the whitening (inv_det) and the normalization (norm via
    # log_det) self-consistent in the near-singular zone, vs an independently-rounded h11*h22-h12^2.
    det = (chol[0][0, 0] * chol[0][1, 1]) ** 2
    inv_det = 1.0 / det
    gx = grid_points[0]  # (m,)
    gy = grid_points[1]  # (m,)
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

(Note: `chol`/`log_det`/`norm` and `d`, `factor`, `neff`, `data`, `covariance` are all unchanged above
this block. `chol` remains used for the PD-branch + `log_det`; only `cho_solve` is gone.)

- [ ] **Step 4: Run the structural guard + near-singular + kernel-parity + golden + degenerate**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -v --no-header -k "no_cho_solve or near_singular or vectorized_kernel_matches_scipy or golden or degenerate or singular or lt2 or chunk"`
Expected: **all PASS** — the structural guard now green; the closed-form is value-equivalent to scipy within `rtol=1e-7` (golden anchored, NOT regenerated); the singular/`<2` fallbacks unchanged; chunking-invariance holds.

(If `test_golden_*` or near-singular fails the `1e-7` tolerance: closed-form precision genuinely degraded in the ill-conditioned zone — do NOT loosen the golden and do NOT widen the singular branch. **Expected shape of the fix:** a `well-conditioned → closed-form / near-singular → triangular-solve via the retained chol` branch (partially reintroducing the cho_solve-equivalent for that subset only — correctness > the last few %). The chol-derived `det` (Step 3, #2) makes this far less likely, and for the synthetic `1e-3`/`1e-4` geometry it won't trigger; named here so it's not a surprise.)

(No commit — single commit at Task 3.)

---

## Task 3: Full verification, re-measure, single commit

**Files:** (none modified)

- [ ] **Step 1: Full ghost-GK suite (regression) + the full non-e2e tracking DAS/ghost suites**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_ghost_gk_frame_restriction.py -q --no-header -p no:warnings`
Expected: **all PASS** (the closed-form change is value-equivalent; `compute_ghost_gk`/`add_ghost_gk` outputs unchanged within tolerance — the frozen golden is the anchor).

- [ ] **Step 2: CI-exact lint + type (the full sequence — learned lesson)**

Run: `.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/`
Then: `.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/`
Then: `.venv/Scripts/python.exe -m pyright silly_kicks/`
Expected: all clean (run `ruff format silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_kde_vectorized.py` first if `format --check` flags them).

- [ ] **Step 3: Re-measure the single-thread baseline (informational; lakehouse owns the authoritative in-venue number)**

Run: `.venv/Scripts/python.exe scripts/profile_ac1_hotpaths.py > _phase1_step1_bench.txt 2>&1` (no contention; persist before cleanup).
Record the new `vectorized` ms/sample (closed-form) — this is the **new numpy single-thread baseline** that the numba kill-gate (spec §4/§6) must beat by ≥1.5× net-of-compile in-venue. Note it in the spec §6 / the PR body. Then `rm _phase1_step1_bench.txt`. (Expected: faster than 4.2.0's `vectorized` — the spec projects ~1.3–1.4× on the kernel; the structural no-`cho_solve` guard is the hard CI gate, the wall-clock number is informational per the repo's flaky-wall-clock lesson.)

- [ ] **Step 4: Version bump (5 sites) + CHANGELOG**

This is a value-equivalent perf change → **patch bump 4.2.0 → 4.2.1**. silly-kicks has **no bump
script** — the canonical version lives in **5 sites** (manual is the established convention; it worked
for 4.2.0): `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `TODO.md`
"Current release", a CHANGELOG `[4.2.1]` entry ("Changed — ghost-GK KDE computes the 2×2 whitening in
closed form (removes `cho_solve`); ~1.3–1.4× kernel, value-equivalent within golden tolerance;
`cho_factor` PD-branch + `log_det` unchanged"), then `uv lock` (updates the editable self-package in
`uv.lock`). **After editing, run the consistency check before commit:**
(a) `grep -rn "4\.2\.1" pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md uv.lock` shows the
new version in each; (b) **explicit negative grep** — `grep -rn "4\.2\.0" pyproject.toml
silly_kicks/__init__.py TODO.md` must **return nothing** (the historical CHANGELOG `[4.2.0]` entry is
the only allowed `4.2.0` hit; pyproject/`__init__`/TODO must be clean).
(Re-check `main` for version contention at PR time — bump higher if 4.2.1 is taken.)

- [ ] **Step 5: Single bundled commit**

```bash
git add -A && git commit -F - <<'EOF'
perf(ghost-gk): closed-form 2x2 KDE whitening (remove cho_solve) -- silly-kicks 4.2.1

Phase-1 Step 1 of ghost-GK KDE acceleration (spec: ...ghost-gk-kde-numba-acceleration-design.md §3).
Replace _kde_density_vectorized's cho_solve-over-(2,kb*m) whitening with the closed-form 2x2
Mahalanobis energy (0.5/det*(h22*dx^2 - 2*h12*dx*dy + h11*dy^2)); KEEP cho_factor for the PD-branch
+ log_det so the singular->uniform fallback set + log_det stay byte-identical to 4.2.0 (no model
change). Drops the (2,kb,m) diff/tdiff temporaries. Value-equivalent within the frozen golden's
rtol=1e-7 (golden NOT regenerated); near-singular parity + structural no-cho_solve guards added.
~1.3-1.4x kernel (dep-free); sets the new numpy single-thread baseline for the numba kill-gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
```

---

## Notes for the implementer
- **No new dependency, no second kernel** — this is a one-function in-place math change.
- **Golden-anchor (spec §5 / lakehouse §3):** the frozen `ghost_gk_kde_golden.npz` stays the scipy-Cholesky reference. Do NOT regenerate it to the closed-form output — the existing golden passing within `rtol=1e-7` *is* the validation.
- **Model boundary (spec §3 / lakehouse §1):** `cho_factor` stays as the singular branch — do NOT add or widen a `det`-based fallback threshold; that would move the modeling boundary (more uniforms than 4.2.0).
- **Value-change governance:** this changes ghost_gk values at ~1e-12..1e-9 vs 4.2.0; the lakehouse runs `run_work_unit → golden.parquet + marts` to confirm no mart moves beyond contract tolerance before production adoption (their gate; not blocking this PR's merge).
- **Numba (Step 2) is NOT in this plan** — it's gated on the §3-step re-measured baseline + the net-of-compile in-venue kill-gate (spec §4/§6).
