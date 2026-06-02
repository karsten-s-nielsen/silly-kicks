# ADR-014: Ghost-GK KDE `fft` backend — binned-convolution (O(k + m·log m))

| Field | Value |
|---|---|
| **Date** | 2026-06-02 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (maintainer), luxury-lakehouse AC-1 session |

## Context

ghost-GK `predict_density` is ~91% of the lakehouse AC-1 tracking-enrichment chain and the reason a
single Metrica game exceeds the 1800 s serverless iteration budget. All three existing KDE backends
(`scipy`, `vectorized`, `cpu-numba`) are **brute-force point×grid: O(k·m)**. The measured cost (real
4.4.0 internals, real bundled "default" model, 40 realistic queries — harness reproduced on the
maintainer box):

- `_leaf_match_weights` returns nonzero weight for **all 35 816 training points on every prediction**
  (measured k = 35 816, min == max), so each prediction is the full `35 816 × 3 840 ≈ 137 M`
  Gaussian evals → **`vectorized` oracle ≈ 4 247 ms/prediction**.
- `cpu-numba` (ADR-013) is only a ~10× constant-factor win and measured ~nil on Databricks serverless
  (per-executor recompile + thread saturation). It does not change the O(k·m) order.

The algorithmic lever is to stop evaluating the kernel per point: **bin the weighted points onto the
fixed grid once, then convolve with the analytic Gaussian via FFT** — O(k + m·log m), independent of
k for the convolution.

## Decision

Add a 4th **opt-in** `kde_backend="fft"` to `predict_density` (threaded through `compute_ghost_gk` /
`add_ghost_gk` / `ghost_gk_xfns`, like the other backends). `_kde_density_fft`:

1. Reuses the **exact `_kde_setup` contract** — same weighted Scott covariance + `cho_factor`
   PD-branch + `det`/`norm`; the convolution kernel is the identical anisotropic Gaussian.
   `LinAlgError` on a singular covariance propagates exactly as today → existing uniform-fallback.
2. **NGP-bins** each weighted training point to its nearest grid cell (the grid is uniform +
   cell-centered, verified), then runs **one** `scipy.signal.fftconvolve(field, kernel, mode="same")`
   at full-grid kernel extent (zero-padded linear conv → captures the untruncated-Gaussian tails, no
   wraparound). `scipy.signal` is already a core runtime dep — **no new dependency** (unlike numba).

**Default stays `"vectorized"`.** `fft` is opt-in until a consumer re-baselines (it is not
bit-identical — see Consequences). The lakehouse flips AC-1 to `fft` in a dedicated PR that
re-baselines the AC-1 golden under a tolerance'd CI gate.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| Keep brute force + numba (ADR-013) | already shipped, exact grid | O(k·m); ~10× / ~nil on serverless | doesn't move the order; ghost-GK stays 91% of the chain |
| GPU backend | large raw speedup | no serverless GPU venue + ~84 matches (ADR-013 §8) | deferred; FFT gets the win on the existing CPU venue |
| CIC (bilinear) binning instead of NGP | halves cell-level grid error | 2× slower; does NOT fix the single near-tie mode flip (flat-ridge argmax); equivalent to NGP on the 3 emitted scalars | NGP is simpler/faster and scalar-equivalent |
| (chosen) NGP binned-convolution, opt-in | ~2000×, no new dep, reuses `_kde_setup` | not bit-faithful on the raw grid (binning) | the emitted scalars are faithful; raw-grid consumers keep `vectorized` |

## Consequences

### Positive
- **~2355× measured** (4247 → 1.80 ms/prediction; lakehouse 2259×) on the full-k production regime;
  O(k + m·log m). No new dependency (core scipy).
- Faithful on the three scalars `predict_density` emits — **mode** (39/40 exact, max 1 cell),
  **mean** (≤5.5 mm), **spread** (≤0.16% rel) — because mean is a grid integral, spread an entropy,
  mode the argmax peak, all robust to per-cell binning noise.
- Reuses `_kde_setup`, so the kernel + singular-cov boundary are identical to the other backends.

### Negative (Hyrum's Law)
- **NOT bit-faithful on the raw per-cell `probabilities` grid** (NGP quantizes per-cell mass: ~1.5%
  typical, up to ~65% on near-zero tail cells). `GhostGkDensity.probabilities` is a public field; a
  consumer reading the raw grid (not just the 3 scalars) must use `vectorized`. **No silly-kicks
  consumer reads the raw grid** (verified: only the `GhostGkDensity` dataclass stores it; the
  action-coupled features emit only mode/spread), so this affects only external grid consumers.
- ~2.5% of predictions flip the discrete mode by ≤1 grid cell (a genuine flat-ridge near-tie, also
  seen across `vectorized`/`cpu-numba`). **Consumers freezing a golden on `ghost_gk_x/y` must
  re-baseline when adopting `fft`** (default stays `vectorized`, so this is opt-in/non-breaking).
- Near-singular covariances have an ill-defined (flat-ridge) mode for *all* backends; `fft` is not
  asserted to match there (not a production regime — real clouds are cond ≤ 5.3).

### Neutral
- Four KDE backends now coexist (`scipy` oracle / `vectorized` default / `cpu-numba` / `fft`), all
  sharing `_kde_setup`. Parity is locked by the scipy-oracle scalar test + a structural
  k-independence guard (one `fftconvolve`, k-independent shapes).

## Future work — CIC binning (gated, not built)

If a **raw-grid** consumer ever needs both speed and tighter per-cell fidelity — concretely, **if
TF-19 (GKDV Layer 3) adopts a density-integrated counterfactual** (`Σ_cells P(action|GK=cell)·
ghost_density(cell)`) rather than a mode-point counterfactual — add **CIC (cloud-in-cell / bilinear)
binning** as a new `kde_backend="fft-cic"` (NGP stays `"fft"`). It is a clean, additive extension:
binning is the **only** seam (the kernel + `fftconvolve` + `_kde_setup` contract are
backend-invariant), so the work is extract `_bin_to_grid(..., scheme)` + one dispatch `elif` (~15
LOC; a validated reference `_bin_cic` exists in the lakehouse harness) with **no public-API change**
(flat `kde_backend` string, not a new kwarg). CIC's reason to exist is the raw grid (≈5.7e-3 vs
1.5e-2 median rel-err), so it MUST ship with a **raw-grid-fidelity test** vs the scipy oracle — not
just the scalar-parity suite, where CIC ties NGP and would test the wrong thing. **Do not build it
until a real grid consumer needs it** (YAGNI; gated like the numba→GPU ladder). CIC does NOT fix the
≤1-cell mode flip (a genuine flat-ridge near-tie).

## Related
- **Continues:** ADR-012 (AC-1 hot-path; DAS ~1% / ghost-GK ~91%) and ADR-013 (cpu-numba backend, GPU
  deferred). `fft` is the algorithmic lever ADR-013 anticipated.
- **Source/harness:** lakehouse `tmp/fft_kde_parity.py` (drives the real internals; reproduced on the
  maintainer box: NGP 2355×, mode 39/40 exact, mean ≤5.5 mm, spread ≤0.16%).
- **External references:** binned-KDE / particle-mesh NGP (nearest-grid-point) assignment; FFT
  convolution. scipy `signal.fftconvolve`.
