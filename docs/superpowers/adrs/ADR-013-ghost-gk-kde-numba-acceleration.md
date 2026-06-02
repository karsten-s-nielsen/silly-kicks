# ADR-013: Ghost-GK KDE acceleration — closed-form whitening + `cpu-numba` backend

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (maintainer), lakehouse review session |

## Context

After ADR-012 shipped the vectorized ghost-GK KDE (silly-kicks 4.2.0), the lakehouse serverless
re-profile (IDSSE J03WMX p1, clean A/B vs 4.1.1, same box) confirmed a 4.2.0 whole-chain win
(1135 → 915 s, 1.24×) but **ghost-GK `predict_density` is still ~91 % of the chain** — the
vectorization removed scipy's per-call object overhead but not the per-sample weighted-KDE
evaluation over ~36k leaf positions × a 3840-cell grid.

Two findings reshaped the GPU-first draft into a numba-first ladder:

1. **No GPU venue, thin volume.** AC-1 runs on CPU-only serverless `applyInPandas` (no GPU
   runtime) over only ~84 tracking matches. A GPU backend would be silly-kicks' *first* (the
   existing `pitch_control` numba kernels are CPU-only per ADR-008; the JAX kernel lives in the
   lakehouse repo, not here) and would have no production venue + thin ROI.
2. **The `cho_solve` "~24 %" was a multi-thread-BLAS artifact.** A closed-form 2×2 Mahalanobis
   energy replacing `cho_solve` is a *numpy* optimisation; measured **single-thread** it is only
   ~1.0× (cho_solve's LAPACK ran across cores; the equivalent scalar math costs the same serially).
   The real single-thread cost is the `exp` + weighted reduction over (k, m), which numpy
   materializes as `(kb, m)` temporaries.

The forcing function is single-thread in-venue throughput (Spark saturates cores across
`(period, frame_batch)` groups, so each task is effectively single-threaded). A serial `@njit`
fully-fused kernel — no `(kb, m)` temporaries — passed an early kill-gate at **~10.3×** over the
numpy closed form (k = 4k/12k/36k, parity 1e-9, all thread env vars pinned to 1).

## Decision

Bundle two laddered, measurement-gated steps into one release (silly-kicks 4.3.0):

- **Step 1 (closed-form, default path).** Replace `cho_solve` in `_kde_density_vectorized` with the
  closed-form 2×2 energy `0.5/det·(h₂₂·dx² − 2·h₁₂·dx·dy + h₁₁·dy²)`, extracted into a shared
  `_kde_setup` (weighted Scott covariance + `cho_factor` PD-branch + `log_det` + `det = (L₀₀·L₁₁)²`).
  `cho_factor` is **retained** so the singular→uniform fallback boundary is byte-identical to 4.2.0.
  Single-thread this is ~1.0× — it ships as the *shared foundation* for numba, not for its own speed.
- **Step 2 (`cpu-numba` backend).** Add `kde_backend="cpu-numba"` — a **serial** `@njit` fully-fused
  closed-form loop in a lazily-imported `_ghost_gk_numba.py` (mirrors `pitch_control/_numba_kernels.py`:
  `try import njit`, env-gated `_NUMBA_CACHE`). numba does **only** the `exp` + reduction loop; the
  numpy `_kde_setup` keeps the PD-branch, so numba needs no Cholesky/boundary replication. Thread
  `kde_backend` through `add_ghost_gk → compute_ghost_gk → predict_density` (and the VAEP factory
  `ghost_gk_xfns`, for API symmetry). Default stays `"vectorized"`; `"scipy"` remains the reference
  oracle.

**Defer GPU** (ADR-012 §4 / spec §8) — revisit only if numba in-venue proves insufficient *and*
volume grows past a few hundred matches *and* a GPU venue exists.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. GPU backend now (JAX/CuPy) | Largest raw speedup | No serverless GPU venue; ~84 matches; first GPU dep in silly-kicks | No production venue + thin ROI; gate it instead |
| B. `parallel=True` + `NUMBA_NUM_THREADS=1` | Looks parallel | Worst case: Spark already saturates cores across groups → in-group `prange` oversubscribes | Serial `@njit` is the correct in-venue model |
| C. Closed-form numpy only (no numba) | No new runtime path | ~1.0× single-thread (the win is loop fusion, which numpy can't express without temporaries) | Insufficient on its own; ships as numba's foundation |
| D. Keep `cho_solve` | Simplest | Blocks the fused-loop numba kernel; the (2,kb,m) temporaries remain | Closed form is the prerequisite for the numba win |
| E. (chosen) Closed-form default + serial `cpu-numba` opt-in; GPU deferred | ~10× hot loop single-thread; no eager numba import; golden-anchored | Default output shifts ~1e-12..1e-9; per-process JIT compile tax in-venue | — |

## Consequences

### Positive

- `cpu-numba` is ~10× the hot loop single-thread (kill-gate; net-of-compile in-venue confirmation is
  the lakehouse's adoption gate — 10× has large headroom over the ≥1.5× ship bar).
- `_kde_setup` is shared by the numpy and numba kernels, so both consume **byte-identical** setup
  (Leg-B parity integrity) and the singular boundary stays `cho_factor`'s (== 4.2.0).
- numba is lazily imported only on the `cpu-numba` path — `import silly_kicks` stays numba-free;
  numba is a `[test]` dep so cpu-numba parity (incl. production-scale k≈36000 and near-singular) is
  a hard CI gate, not skip-if-absent.

### Negative

- **Default-output shift (Hyrum's Law).** Step 1 changes the *default* `vectorized` backend, so every
  consumer's `ghost_gk_x/y/spread` move by ~1e-12..1e-9 on a plain 4.2.0 → 4.3.0 upgrade, even
  without selecting a new backend. Value-equivalent within the frozen golden's `rtol≈1e-7` (golden
  **not** regenerated). The lakehouse `run_work_unit → golden.parquet + marts` value-change gate
  covers both the default upgrade and the cpu-numba opt-in before production adoption.
- numba's serial accumulation (j-outer/i-inner) differs from numpy's pairwise reduction → a near-tie
  argmax (`mode_x/y` → `ghost_gk_x/y`) can shift ≤1 grid cell. Mode parity tests are therefore exact
  for `vectorized`, ≤`GRID_RESOLUTION` for `cpu-numba`; the density field is the primary check.
- A per-process JIT compile tax (`cache=OFF` by default for read-only serverless paths) is hidden by
  warm local benches → the authoritative gate is the lakehouse's net-of-compile single-thread in-venue
  re-measure.

### Neutral

- Three KDE kernels now coexist (`scipy` oracle / `vectorized` default / `cpu-numba`), kept in parity
  by the shared `_kde_setup` + the Leg-B golden/parity tests.
- Real ghost-GK candidate clouds are well-conditioned (lakehouse instrumentation, n=204:
  cond ∈ [3.82, 5.33]); the near-singular parity case is a conservative theoretical guard, while the
  k≈36000 case exercises the regime that actually occurs.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-01-ghost-gk-kde-step1-closed-form.md`,
  `docs/superpowers/plans/2026-06-01-ghost-gk-kde-step2-numba.md`
- **ADRs:** extends ADR-008 (pitch-control numba precedent: numpy default + `@njit` mirror + golden
  parity + numba-in-CI); continues ADR-012 (AC-1 hot-path acceleration, GPU gate).
- **External references:** scipy `stats.gaussian_kde` (Scott bandwidth / weighted covariance); numba `@njit`.
