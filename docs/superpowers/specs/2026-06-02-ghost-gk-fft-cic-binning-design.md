# Ghost-GK KDE `fft-cic` backend — CIC (bilinear) binning

| Field | Value |
|---|---|
| **Date** | 2026-06-02 |
| **Status** | Design (pending review) |
| **Author** | Karsten Nielsen (maintainer) |
| **Requested by** | luxury-lakehouse AC-1 session (hand-off 2026-06-02) |
| **Target release** | silly-kicks 4.8.0 |
| **Builds on** | ADR-014 (`fft` NGP backend, 4.6.0); ADR-012/013 (AC-1 hot path) |

## Context

silly-kicks 4.6.0 shipped `kde_backend="fft"` — a binned-convolution ghost-GK KDE that
NGP-bins (nearest-grid-point) the weighted training points onto the fixed grid, then runs a
single `scipy.signal.fftconvolve` against the analytic Gaussian. It is **O(k + m·log m)** vs the
brute-force **O(k·m)** of the `scipy` / `vectorized` / `cpu-numba` backends. NGP measured **~2259×**
on the reproducible lakehouse harness (`tmp/fft_kde_parity.py`: 4195 → 1.86 ms/sample, k ≈ 35.8k
training points on every prediction; ~2355× on the maintainer box). It ships **opt-in**; the default
stays `vectorized`.

The lakehouse runs AC-1 on `cpu-numba` (exact) and wants to switch to the FFT path for the ~2000×
win, but **NGP is too lossy to adopt as the AC-1 default**. The new real-data finding (IDSSE
J03WMX p1, 97 actions, NGP/CIC vs exact scipy truth):

| binning | actions with shifted mode | mean shift | max shift | #≥2 m | #≥4 m |
|---|---|---|---|---|---|
| NGP (current `fft`) | 21 / 97 (22 %) | 0.343 m | **6.0 m** | 7 | 1 |
| CIC (bilinear) | 5 / 97 (5 %) | 0.097 m | 4.0 m | 2 | 1 |

CIC removes **~76 %** of the spurious mode flips (21 → 5), mean shift 3.5× tighter. The residual 5
(one at 4 m) are genuine near-ties — unfixable by any binning (either peak is defensible). Some of
those 5 are true bimodal-equal grids where the **exact** argmax is itself arbitrary, so CIC's real
accuracy is a touch better than "5/97 misses vs exact" implies (it is not penalised for disagreeing
on a coin-flip the exact backend also can't resolve).

### Why this contradicts ADR-014 (and why that matters)

ADR-014 asserts the emitted **scalars (mode/mean/spread) are robust to per-cell binning noise** and
that **CIC's only benefit is raw-grid fidelity** — "CIC ... does NOT fix the single near-tie mode
flip ... equivalent to NGP on the 3 emitted scalars" (Alternatives table), mode flips are "≤1 grid
cell" (Consequences), and "CIC's reason to exist is the raw grid ... it MUST ship with a
raw-grid-fidelity test — not just the scalar-parity suite, where CIC ties NGP and would test the
wrong thing" (Future work).

**Real multimodal data falsifies all three.** ghost-GK density grids are frequently bimodal (GK
plausibly at near *or* far post, two ~equal peaks ~6 m apart). NGP snaps each training point to its
nearest cell (±0.25 m on the 0.5 m grid); on a near-tie bimodal grid that quantization flips **which
peak is highest**, so the emitted `ghost_gk_x/y` (the argmax mode) jumps up to **6 m** — not the
"≤1 cell" ADR-014 claims, and CIC *does* fix ~76 % of these.

The root cause is a measurement gap: ADR-014's scalar-parity bench used **~unimodal** synthetic
queries, which structurally cannot exhibit the multimodal peak-selection flip. **A test that cannot
exercise the failure path proves nothing** — the same trap as the 4.2.0 "value-neutral" DAS claim
(a fixture placed onside so the offside-exemption it was meant to validate never fired). So 4.6.0's
"scalars are robust" is *overstated*, not wrong-everywhere: it holds on unimodal grids and on
mean/spread always, but the **mode is NOT binning-robust on multimodal grids**.

This spec therefore **corrects ADR-014's falsified claims** alongside adding the backend. It is not a
purely additive extension.

## Goals / non-goals

**Goals**
- Add `kde_backend="fft-cic"` — CIC (cloud-in-cell / bilinear) binning, same FFT kernel path.
- Reachable from `add_ghost_gk(..., kde_backend="fft-cic")` (lakehouse's only integration point) with
  **zero public-API signature change**, propagating through `compute_ghost_gk` / `ghost_gk_xfns`.
- Correct the overstated mode-fidelity claims in the `_kde_density_fft` docstring, `predict_density`
  docstring, and ADR-014.
- Ship as 4.8.0: one feature branch, one commit, PR, tag, PyPI (mirrors the 4.6.0 fft ship).

**Non-goals (YAGNI)**
- **Not** changing the `fft` default or repurposing the `"fft"` string — `"fft"` keeps its 4.6.0
  meaning (NGP). Repurposing it would silently change every existing `"fft"` caller's results.
- **Not** making `fft-cic` the global default — `vectorized` stays the default; both FFT backends are
  opt-in (approximate; consumers re-baseline goldens on adoption).
- **Not** a `binning=` kwarg. Decision (confirmed with the consumer): a flat `kde_backend` string.
  A `binning` param would be valid only when `kde_backend="fft"` — false orthogonality across 4
  public surfaces, an invalid-combination matrix to validate, and permanent signature API (Hyrum).
- **No** `fft-cic` wall-clock CI budget (shared-CI wall-clock is flaky — TF-16 lesson); covered by a
  structural k-independence guard instead.
- **No** new dependency (`scipy.signal` is core, as for `fft`).

## Design

### The seam

The only difference between `fft` and `fft-cic` is the binning of weighted points onto the grid.
`_kde_setup` (weighted Scott covariance + `cho_factor` PD-branch + `det`/`norm`), the analytic
anisotropic-Gaussian kernel build, and `fftconvolve(field, kernel, mode="same")` are
**backend-invariant**. To keep them provably shared (and avoid a second copy drifting), extract the
shared tail into a private helper and have both kernels own only their binning step:

```python
def _fft_convolve_field(field, h11, h12, h22, det, norm):
    """Shared FFT-convolution tail for the fft / fft-cic backends.

    `field` is the binned weighted-point grid (GRID_NX, GRID_NY). Builds the full-extent analytic
    anisotropic Gaussian kernel (identical to _kde_density_vectorized's energy form) and returns the
    UNnormalized density via one zero-padded linear fftconvolve. Binning is the SOLE per-backend
    difference; this tail is byte-identical across fft / fft-cic.
    """
    from scipy.signal import fftconvolve
    inv_det = 1.0 / det
    dx = (np.arange(-(GRID_NX - 1), GRID_NX) * GRID_RESOLUTION)[:, None]
    dy = (np.arange(-(GRID_NY - 1), GRID_NY) * GRID_RESOLUTION)[None, :]
    kernel = norm * np.exp(-0.5 * inv_det * (h22 * dx * dx - 2.0 * h12 * dx * dy + h11 * dy * dy))
    return fftconvolve(field, kernel, mode="same")
```

`_kde_density_fft` (NGP) becomes: `_kde_setup` → NGP bin (`np.rint` + `np.add.at`, unchanged) →
`_fft_convolve_field(...)`. Behaviour byte-identical to 4.6.0 (refactor only; locked by the existing
golden test). `_kde_density_fft_cic` is identical except the binning step:

```python
def _kde_density_fft_cic(gk_x_w, gk_y_w, w, grid_points):  # grid_points unused (module grid), parity sig
    """Binned-convolution KDE with CIC (cloud-in-cell / bilinear) binning. O(k + m log m).

    Identical to _kde_density_fft except each weighted training point is spread bilinearly over its
    4 surrounding grid cells (weights (1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx*ty) instead of snapped to
    the single nearest cell (NGP). On a near-tie MULTIMODAL grid this preserves the relative peak
    masses, so the emitted mode (argmax) flips ~76% less often than NGP (real data: 21/97 -> 5/97).
    ~2x the NGP bin cost, still ~1195x over `vectorized`. See ADR-014.
    """
    _data, w_n, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
    field = _bin_cic(gk_x_w, gk_y_w, w_n)
    return _fft_convolve_field(field, h11, h12, h22, det, norm)


def _bin_cic(gk_x_w, gk_y_w, w_norm):
    """Cloud-in-cell (bilinear) binning of weighted points onto the fixed grid."""
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

`_kde_setup` raises `LinAlgError` on a singular covariance exactly as the other backends, so
`predict_density`'s uniform-fallback applies unchanged. Out-of-grid points clip to the edge (the
`np.clip` on `i0+di` / `j0+dj`); negligible in practice and the full-extent kernel carries tail mass
inward, matching the NGP edge behaviour. **Mass is conserved**: the 4 bilinear weights sum to 1 per
point and `np.add.at` accumulates all 4 contributions even when clip collapses indices to the same
edge cell, so `field.sum() == w_norm.sum()` (tested invariant — see Testing #3/#5).

### Dispatch

One `elif` in `predict_density` (after the existing `fft` branch):

```python
elif kde_backend == "fft":
    probs = _kde_density_fft(gk_x_w, gk_y_w, w, grid_points)
elif kde_backend == "fft-cic":
    probs = _kde_density_fft_cic(gk_x_w, gk_y_w, w, grid_points)
```

`compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` already forward `kde_backend` verbatim, so
`fft-cic` is reachable end-to-end with no signature change. The atomic mirror re-exports, so it is
free there too. Docstring backend enumerations (`predict_density`, `compute_ghost_gk`,
`add_ghost_gk`) gain `"fft-cic"`.

### Fidelity contract (corrected)

| backend | mean/spread | mode (unimodal) | mode (multimodal near-tie) | raw grid |
|---|---|---|---|---|
| `fft` (NGP) | faithful | faithful (±1 cell) | **can flip up to ~6 m** (21/97 real) | quantized (~1.5 % typ) |
| `fft-cic` | faithful | faithful (±1 cell) | flips ~76 % less (5/97 real) | tighter (~5.7e-3 vs 1.5e-2 median) |

Both remain approximate on the raw grid; the residual `fft-cic` mode flips are genuine near-ties.
**`vectorized` / `cpu-numba` remain the only exact-grid backends.**

**Doc recommendation (soft-deprecate NGP for new consumers).** `fft-cic` dominates `fft` on *both*
mode and raw-grid fidelity; NGP's only edge is ~2× speed (and both are ~1000×+ over brute force).
`"fft"` stays for back-compat (Hyrum — never repurpose or remove it), but the `predict_density` /
`compute_ghost_gk` / `add_ghost_gk` docstrings will **steer new FFT consumers to `"fft-cic"`** unless
they specifically need the extra speed on known-unimodal distributions. This is a documentation
recommendation only — no behaviour or default change.

## Testing (TDD, all non-e2e unless noted)

The IDSSE fixture behind the 21→5 table is **not** committed (gated), so the real-data table is a
lakehouse-side / e2e validation — it cannot be a CI gate here. CI proves the *mechanism* on
constructions that are committed:

1. **Bimodal mode parity (the primary motivation test).** Build **N seeded** synthetic weighted point
   sets, each with two near-tie clusters at **different grid phase** to manufacture NGP's real-data
   flip mechanism (differential peak-height distortion): the higher-mass **winner on a cell boundary**
   (`12.5` — NGP splits its mass → under-counts its peak) and the lower-mass **loser on a node**
   (`18.25` — NGP concentrates → over-counts), inducing NGP to flip the argmax to the wrong (loser)
   peak while CIC's bilinear, mass-conserving spread keeps the winner. (The grid is
   `_GRID_X = 0.25 + i·0.5`: nodes at `*.25/*.75`, boundaries at `*.0/*.5` — `12.25`/`18.25` are
   *both nodes*, NGP's best case, so centering there would not bite; the phase split is essential.)
   **Two assertions:** (i) **enforced margin gate** — `_kde_density_fft_cic`'s argmax matches the
   `vectorized`-exact argmax on **≥ 3 more** of the N constructions than `_kde_density_fft` does. This
   single gate subsumes both *non-vacuity* (since `cic_correct ≤ N`, the margin forces
   `ngp_correct ≤ N−3` → NGP demonstrably flips on ≥ 3; the 4.2.0 DAS-onside lesson) and the
   defensible *aggregate* claim that CIC beats NGP (the real 21-vs-5 evidence), and cannot erode to a
   one-flip near-vacuous pass across a lib bump; (ii) **soft per-instance** — CIC is worse than NGP on
   **≤ 1** construction (demoted from a hard per-seed assert: the evidence is aggregate, **not** a
   strict per-instance subset — a bilinear spread can occasionally miss where an NGP snap hit — so a
   hard per-seed assert would flake). The `lead` mass-imbalance lives in a tension band: large enough
   that the exact argmax is the winner (not a coin-flip), small enough that NGP flips. This is the
   test ADR-014 said was "the wrong thing" — it is in fact the right thing, because CIC does **not**
   tie NGP on multimodal grids. Deterministic (fixed seeds), no model/fixture needed.
2. **Mean / spread parity.** On the same bimodal constructions, `|mean_cic - mean_exact| < 1e-2 m` and
   `|spread_cic/spread_exact - 1| < 5e-3` (CIC must not regress the already-faithful scalars).
3. **`_bin_cic` mass conservation (unit invariant).** `field.sum()` ≈ `w_norm.sum()` (`np.isclose`,
   rtol 1e-12): the 4 bilinear weights sum to 1 per point, and `np.add.at` still accumulates all 4
   contributions when edge-clip collapses indices, so total mass is conserved including for clipped
   points. Cheap, strong, catches binning bugs the scalar tests miss.
4. **Raw-grid fidelity (ADR-014-mandated).** On a real "default"-model leaf subset, CIC's per-cell
   median rel-err vs the `vectorized` oracle is **strictly lower** than NGP's. Proves CIC does what it
   exists for on the grid, where scalar parity alone would tie.
5. **Backend invariants + out-of-grid.** `fft-cic` returns the same `GhostGkDensity` shape;
   `probabilities.sum() == 1`; singular covariance → uniform grid (LinAlgError fallback). **Add an
   out-of-grid point** (beyond a grid edge): assert no crash and `_bin_cic` mass conserved (the clip
   contract), matching the NGP edge behaviour.
6. **Structural k-independence guard.** Spy that `fft-cic` issues exactly **one** `fftconvolve` and
   that the binning array shapes are independent of k (mirrors the existing `fft` guard). Replaces a
   wall-clock budget.
7. **Two distinct locks (NGP refactor + binning seam).** (a) **NGP refactor lock:** the existing `fft`
   rtol golden/scalar tests stay green — this, plus the extraction being verbatim-by-construction, is
   what proves `_fft_convolve_field` extraction did not change the 4.6.0 NGP path (Chesterton's fence).
   A frozen-pre-refactor-array `np.array_equal` was considered and **rejected** — an exact float array
   false-fails across the 3.10/3.11/3.12+Windows CI matrix (platform-dependent FFT/BLAS rounding, a
   documented repo lesson). (b) **Seam lock:** an **exact** `np.array_equal` that `fft-cic` equals
   `fft` when all points sit on grid nodes (CIC's bilinear weight collapses to 1.0 on the corner cell;
   points built as `_GRID_X[0] + i*GRID_RESOLUTION` for exact-integer `fx`). Same-process → CI-safe.
   This locks "binning is the only seam"; it does **not** (and is not claimed to) detect NGP-path drift.
8. **Real-model golden (non-e2e), rtol.** `fft-cic` scalars on the bundled "default" model vs a frozen
   golden, mirroring `test_golden_fft_scalars`, compared with **`rtol`** (not exact) so a numpy/scipy
   bump does not false-fail — the same cross-version robustness the lakehouse goldens use. Closes the
   synthetic-only gap (the production regime).
9. **Dispatch / lt2.** Unknown-backend `ValueError` unchanged; `< 2` weight short-circuit unchanged;
   atomic mirror reachable.

Tolerance summary: **#7 seam lock exact (`array_equal`)** — on-grid-node degeneracy, same-process so
CI-safe (the NGP refactor itself is locked by the rtol golden, not an exact frozen array); **#8 rtol**
— cross-version golden; #3 `rtol 1e-12` — fp mass sum; #1 enforced margin gate `cic_correct −
ngp_correct ≥ 3` (subsumes non-vacuity) + soft per-instance `violations ≤ 1`; #2 the metre/relative
tolerances above.

**Implementation note (as-built, 4.8.0 — three justified deviations from the design above):**
- **#1 reference = the KNOWN winner, not grid-vectorized.** The construction puts the higher-mass
  winner on a cell *boundary* (12.5), which grid-vectorized itself under-samples → grid-vectorized
  phase-flips to the loser on some seeds (a confound). The higher-mass cluster is the unambiguous
  continuous mode, so the test scores CIC/NGP against that known winner (`mode_x < 15`), with a
  vectorized subset as a sanity check that the winner *is* the grid mode. `N = 120`, regime
  `std=0.75, lead=0.05`; observed margin 5 (CIC 120/120, NGP 115/120) — comfortably over the gate.
- **#2 scalar parity tested on a UNIMODAL cloud** (mirrors `test_fft_kernel_matches_scipy_on_scalars`:
  mean <3e-2 m, spread rel <1e-2). On a deliberately-bimodal cloud CIC's bilinear smoothing adds ~3%
  to the entropy-spread (a known tradeoff for the mode-flip reduction), so the tight <5e-3 spread
  bound was unrealistic there; that production-regime bound lives in the real-model golden (#8).

## Downstream / Hyrum's-Law notes

- **`fft` is unchanged and stays the fft default** — existing `"fft"` callers are unaffected.
- The corrected contract has a consequence for any **trained-model consumer of the ghost-GK *mode***
  (the emitted `ghost_gk_x/y` argmax): under `fft`, the mode can shift ~6 m on multimodal frames, so
  it is **not** unconditionally "fft-safe" as previously assumed. This is a real **train/serve skew**
  risk — training a feature on one backend's mode and serving on another silently differs by up to
  6 m on multimodal frames, and **no unit test surfaces it** (the same "test can't bite" class this
  whole PR corrects). A documented caveat alone would get lost, so this PR **elevates it to a tracked
  action** (a `TODO.md` row, not just prose): *any ghost-GK-mode consumer MUST pin one `kde_backend`
  for train AND serve, persist it in model metadata, and add a serve-time assert that the metadata
  backend matches the runtime backend — turning silent skew into a loud failure.* No runtime change in
  this PR. **Scope check (verified):** the current GKDV Layer-2 model, **TF-16 xShotOccurrence, does
  NOT consume the ghost-GK mode** — it uses the resolved/defending GK from `_gk_resolve`, so it is
  unaffected and needs no `kde_backend` metadata. The guard binds whichever feature *first* trains on
  the ghost-GK mode — prospectively **TF-17 / TF-19** — so the tracked action is owned by that
  consumer, not by the TF-16 weights run.
- **TF-19 (GKDV Layer 3)**, if it goes density-integrated over the raw grid, should use `fft-cic`
  (or `vectorized`); `fft-cic` is the better-fidelity FFT option this unblocks.

## Release mechanics (4.7.0 → 4.8.0)

Single feature branch (`feat/ghost-gk-fft-cic-binning`), single commit, PR at the end when fully
green. Bundle:
- code (`_ghost_gk.py`: `_fft_convolve_field`, `_kde_density_fft_cic`, `_bin_cic`, dispatch +
  docstrings) + tests;
- **ADR-014 amendment** (correct the falsified mode/CIC claims with the real-data evidence; move
  `fft-cic` from "Future work — gated" to "Decision — shipped 4.8.0");
- **NOTICE** one-line reference for CIC / cloud-in-cell (particle-mesh bilinear assignment);
- 5 version-bump sites: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG`, `TODO.md`,
  `uv.lock` (via `uv lock`);
- **`TODO.md` tracked-action row** for the ghost-GK-mode train/serve-skew guard (see Downstream
  notes) — pin one `kde_backend` train+serve, persist in metadata, serve-time backend assert; owned
  by whichever feature first trains on the ghost-GK *mode* (prospectively TF-17 / TF-19 — **not** the
  TF-16 weights run, which does not consume the mode);
- C4: **no regen** — internal kernel, container description unchanged (per prior fft note).

Pre-push: full lint trio (`ruff check silly_kicks/ tests/ scripts/` + `ruff format --check ...` +
`pyright silly_kicks/`) + full non-e2e suite green, verified from exit codes / JUnit (never narrated
from memory). After merge: annotated `v4.8.0` tag → publish.yml → PyPI. The commit is **not** created
until the work is fully tested, and the sentinel is **never** created without explicit per-commit
approval.

## Reproduce (lakehouse harnesses, owner box)

`uv run python tmp/fft_kde_parity.py` (NGP/CIC/oracle speedup + grid rel-err) ·
`uv run python tmp/test_cic_real.py` (21→5 real-fixture mode shift). Seed 20260602.
Refs: ADR-014, ADR-035 (lakehouse), `project_ac1_ghost_gk_gpu_venue_roi`.
