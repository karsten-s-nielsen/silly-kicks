# ADR-014: Ghost-GK KDE `fft` backend — binned-convolution (O(k + m·log m))

| Field | Value |
|---|---|
| **Date** | 2026-06-02 |
| **Status** | Accepted — amended 2026-06-02 (4.8.0: CIC `fft-cic` added; mode-fidelity claims corrected) |
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
| CIC (bilinear) binning instead of NGP | halves cell-level grid error; **also fixes the multimodal mode flip** | 2× slower | **Deferred at 4.6.0, ADOPTED at 4.8.0** as opt-in `fft-cic` — see the amended Decision below. (The original "scalar-equivalent / doesn't fix the mode flip" rejection was based on a *unimodal* bench; real multimodal data showed NGP flips the mode by up to ~6 m on ~22% of actions, which CIC fixes ~76%.) |
| (chosen) NGP binned-convolution, opt-in | ~2000×, no new dep, reuses `_kde_setup` | not bit-faithful on the raw grid (binning) | the emitted scalars are faithful; raw-grid consumers keep `vectorized` |

## Consequences

### Positive
- **~2355× measured** (4247 → 1.80 ms/prediction; lakehouse 2259×) on the full-k production regime;
  O(k + m·log m). No new dependency (core scipy).
- Faithful on **mean** (≤5.5 mm) and **spread** (≤0.16% rel) — grid integral / entropy, robust to
  binning — always, and on the **mode** for **unimodal** grids (39/40 exact, max 1 cell on the
  unimodal bench). **CORRECTION (amended 4.8.0):** the mode is NOT robust on near-tie *multimodal*
  grids — see the corrected Negative item and the amended Decision below.
- Reuses `_kde_setup`, so the kernel + singular-cov boundary are identical to the other backends.

### Negative (Hyrum's Law)
- **NOT bit-faithful on the raw per-cell `probabilities` grid** (NGP quantizes per-cell mass: ~1.5%
  typical, up to ~65% on near-zero tail cells). `GhostGkDensity.probabilities` is a public field; a
  consumer reading the raw grid (not just the 3 scalars) must use `vectorized`. **No silly-kicks
  consumer reads the raw grid** (verified: only the `GhostGkDensity` dataclass stores it; the
  action-coupled features emit only mode/spread), so this affects only external grid consumers.
- **CORRECTION (amended 4.8.0):** the original claim that mode flips are "~2.5% of predictions, ≤1
  grid cell" held only for the *unimodal* bench. On near-tie **multimodal** grids (GK plausibly at
  near *or* far post — two ~equal peaks ~6 m apart), NGP's per-point ±0.25 m snap can flip *which*
  peak is the argmax, shifting the emitted mode by **up to ~6 m on ~22% of real actions** (IDSSE
  J03WMX p1, 97 actions, NGP vs exact). `fft-cic` (CIC bilinear binning) reduces this to ~5%
  (21/97 → 5/97; the residual are genuine near-ties unfixable by any binning). The unimodal bench
  could not surface this — it structurally cannot exhibit a peak-selection flip (the same
  "test couldn't bite" class as the 4.2.0 DAS value-neutral claim). **Consumers freezing a golden on
  `ghost_gk_x/y` must re-baseline when adopting `fft`/`fft-cic`** (default stays `vectorized`, so
  this is opt-in/non-breaking).
- Near-singular covariances have an ill-defined (flat-ridge) mode for *all* backends; `fft` is not
  asserted to match there (not a production regime — real clouds are cond ≤ 5.3).

### Neutral
- Four KDE backends now coexist (`scipy` oracle / `vectorized` default / `cpu-numba` / `fft`), all
  sharing `_kde_setup`. Parity is locked by the scipy-oracle scalar test + a structural
  k-independence guard (one `fftconvolve`, k-independent shapes).

## Decision (amended 4.8.0) — `fft-cic` shipped

CIC (cloud-in-cell / bilinear) binning is now a **fourth opt-in backend `kde_backend="fft-cic"`**
(NGP stays `"fft"`, unchanged, still the fft-default — never repurposed, so no existing `"fft"`
caller's results change). Binning is the **only** seam: `_kde_density_fft` and `_kde_density_fft_cic`
share `_kde_setup` + `_fft_convolve_field` verbatim and differ only in `_bin_ngp` vs `_bin_cic`. Flat
`kde_backend` string, **no public-API signature change** — it auto-propagates through
`compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns` and the atomic mirror.

**Motivation (what changed since 4.6.0's "gated, YAGNI" stance):** the lakehouse measured, on real
multimodal data, that NGP shifts the *emitted mode* (the `ghost_gk_x/y` argmax) by up to ~6 m on
~22% of actions; CIC cuts that ~76% (21/97 → 5/97). So CIC's value is **not** only the raw grid (as
4.6.0 assumed) — it is the **mode on multimodal grids**, the scalar consumers actually read. It also
tightens the raw grid (≈5.7e-3 vs 1.5e-2 median rel-err). Cost ~2× NGP, still ~1195× over
`vectorized`. **Default stays `vectorized`**; both FFT backends are approximate (raw-grid consumers
needing exactness use `vectorized`/`cpu-numba`).

**Root cause that 4.6.0 missed it:** the scalar-parity bench used ~**unimodal** queries, which
structurally cannot exhibit a multimodal peak-selection flip — so the "scalars are robust / CIC ties
NGP on scalars" claim was validated on a fixture that couldn't bite (the 4.2.0 DAS value-neutral
lesson). The 4.8.0 test suite therefore makes the **bimodal mode-parity test the primary motivation
gate** (CIC lands on the true-winner side on ≥3 more of N differential-phase constructions than NGP)
**plus** the raw-grid-fidelity test (CIC median rel-err strictly < NGP on the real model) **plus** a
real-model golden — see `tests/tracking/test_ghost_gk_kde_vectorized.py`.

**Tests / locks:** on-grid-node seam lock (`fft-cic == fft` exactly when points sit on nodes, so
binning is provably the only seam); the NGP path is locked behaviour-identical across the refactor by
the existing rtol fft golden; mass-conservation + out-of-grid for `_bin_cic`; k-independence spy;
real-model golden (rtol — CI-version-robust).

**Train/serve-skew note:** any **trained-model consumer of the ghost-GK mode** must pin one
`kde_backend` for train AND serve and persist it in model metadata (a serve-time backend assert turns
the silent ≤6 m multimodal mode skew into a loud failure). Verified scope: **TF-16 xShotOccurrence
does NOT consume the ghost-GK mode** (it uses the resolved/defending GK), so it is unaffected; the
guard binds whichever feature first trains on the mode — prospectively TF-17 / TF-19 (tracked in
TODO.md).

**Future — `fft-cic-cic`/CIC-grid escalation (still gated, not built):** if a *raw-grid* consumer
later needs per-cell fidelity beyond CIC's, the same binning seam admits a higher-order scheme; YAGNI
until a real grid consumer needs it.

## Related
- **Continues:** ADR-012 (AC-1 hot-path; DAS ~1% / ghost-GK ~91%) and ADR-013 (cpu-numba backend, GPU
  deferred). `fft` is the algorithmic lever ADR-013 anticipated.
- **Source/harness:** lakehouse `tmp/fft_kde_parity.py` (drives the real internals; reproduced on the
  maintainer box: NGP 2355×, mode 39/40 exact, mean ≤5.5 mm, spread ≤0.16%).
- **External references:** binned-KDE / particle-mesh NGP (nearest-grid-point) assignment; FFT
  convolution. scipy `signal.fftconvolve`.
