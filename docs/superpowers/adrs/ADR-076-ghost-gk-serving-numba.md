# ADR-076: Ghost-GK serving — numba-accelerate the shared leaf traversal + exact KDE default

| Field | Value |
|---|---|
| **Date** | 2026-08-27 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

The ghost-GK boosted-tree **leaf traversal** is a shared hot loop. Two numpy functions implement
it: `_vectorized_leaf_values` backs the production `predict_mean` (the served
`ghost_gk_x`/`ghost_gk_y` estimate — the exact HGBR boosted prediction, reconstructed pickle-free
from the stored tree node arrays), and `_vectorized_leaf_indices` backs the KDE query leaf-match
(`predict_density`) **and** the `fit()` training leaf-match. Both are "vectorized over samples,
looped over trees × depth" with per-step fancy-indexing / `np.where` temporaries.

Measured on a 500-tree model over 3000 rows (a Metrica game): one `_vectorized_leaf_values`
traversal is **922 ms**; `predict_mean` runs it twice (x-tree + y-tree) ≈ 1.9 s/game. A fused
`@njit` per-sample tree walk collapses the temporaries to **78 ms (11.8×), BIT-IDENTICAL**
(`np.array_equal`, max |Δ| = 0.000e+00). The prior optimization-backlog framing ("ghost-GK KDE /
1800 s Metrica budget", ADR-014) was fact-checked **stale** — 4.14.0 made the served value the HGBR
mean and ADR-044 retired the KDE from every aggregator, so the production path is `predict_mean`
(already ~900× under the per-unit watchdog) and `predict_density` survives only on a locally-`fit()`
model. This cycle is therefore deliberate gold-standard polish of the paths that ARE live, not an
urgent bottleneck fix ("how you do anything is how you do everything").

Separately, `predict_density`'s default `kde_backend="vectorized"` is the *slowest* backend. Two
faster ones exist but are opt-in: `cpu-numba` (exact within ~1e-9; the KDE golden already runs it)
and `fft` (~2000×, but **approximate** on the raw grid — a deliberate binning trade-off).

## Decision

Add ONE exact `@njit` leaf-traversal kernel pair (`_leaf_values_numba`, `_leaf_indices_numba`) in
`tracking/_ghost_gk_numba.py`; dispatch the two numpy traversals to it when the `[numba]` extra is
installed (numpy fallback otherwise), passing an explicit `_FlatTrees` value object cached on the
model at `fit`/`load`. The traversal stays **bit-identical** — every golden / chirality /
feature-contract test passes unchanged, no retrain, no re-materialize, no artifact change. And make
`predict_density`'s default `kde_backend="auto"`, resolving to the fastest **exact** backend
(`cpu-numba` if numba is usable, else `vectorized`); `fft`/`fft-cic` stay an explicit opt-in.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Leave the numpy traversal as-is | Zero work | Production serving stays 12× slower than free; `fit()` + the research KDE leaf-match stay slow | The speedup is free and bit-identical; declining it fails the gold-standard bar |
| B. Make numba a hard dependency | Simpler dispatch (no fallback branch) | Adds a heavy runtime dep; breaks the "bare `import _ghost_gk` is dependency-light" contract | numba stays optional (ADR-013 / ADR-008 precedent); numpy fallback preserved |
| C. Default the KDE to `fft` (~2000×) | Fastest | `fft` is **approximate** on the raw grid (NGP/CIC binning) — a numerical research tool would silently go approximate by default | Rejected on principle: the default must stay EXACT; `fft` is a deliberate opt-in |
| D. Parallel (`prange`) / GPU numba | More throughput | Spark already saturates cores across `applyInPandas` groups (ADR-013); an in-group `prange` oversubscribes | Serial `@njit` only |
| E. (chosen) Exact serial `@njit` kernel pair + numpy fallback + exact KDE default | 11.8× bit-identical serving; exact KDE default; one kernel serves 3 paths | Per-process JIT warm-up (one-time) | — |

## Consequences

### Positive

- `predict_mean` ~1.9 s → ~0.16 s/game (11.8×), **bit-identical** — no retrain, no re-baseline.
- One kernel also accelerates `fit()`'s training leaf-match and the research KDE query leaf-match.
- `predict_density`'s default is now the fastest **exact** backend instead of the slowest.

### Negative

- Per-process JIT warm-up (~0.1–1 s) on the first numba call; the on-disk cache is off by default
  on serverless (ADR-013). Not per-prediction; a single-prediction caller sees no benefit and the
  numpy fallback has no warm-up.
- The KDE default shifts a caller who relied on the implicit `vectorized` default onto `cpu-numba`,
  which differs by ≤1e-9 (numba `exp` vs numpy `exp`). Exact-arithmetic, golden-covered. A raw-grid
  exactness consumer pins `kde_backend="vectorized"` explicitly.
- A small flat-tree cache per model (O(total nodes), a few MB for 500 trees).

### Neutral

- The dispatch is hexagonal: `_vectorized_leaf_*` are the numpy PORT (reference + fallback), the
  `@njit` kernels are the ADAPTER, `_FlatTrees` is passed explicitly.
  `SILLY_KICKS_GHOST_FORCE_NUMPY=1` forces the numpy path so both adapters run on every CI leg.
- **The convergence-guard fidelity is ASYMMETRIC by design.** `_vectorized_leaf_values` RAISES on a
  >depth-cap tree (it reads `value` = garbage), so `_leaf_values_numba` carries the same
  `RuntimeError`. `_vectorized_leaf_indices` never reads `value` and returns the non-converged LOCAL
  index silently, so `_leaf_indices_numba` carries NO guard — a guard there would itself break
  bit-identity in the >depth-cap case. Each kernel matches its numpy sibling's ACTUAL behaviour.
- The numba leaf import is LAZY (bound on first use, never at module load) so a bare
  `import _ghost_gk` stays numba-free — the contract pinned by
  `test_ghost_gk_does_not_eagerly_import_numba`, mirroring the KDE `cpu-numba` path.

## Related

- **Specs:** `docs/superpowers/specs/2026-08-27-ghost-gk-serving-numba-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-27-ghost-gk-serving-numba.md`
- **ADRs:** builds on ADR-008 (pitch-control `@njit` + fallback), ADR-013 (ghost-GK KDE `cpu-numba`),
  ADR-014 (KDE backends), ADR-044 (ghost-GK parameters-only artifacts + KDE retired from the
  aggregators — the flat-tree cache is likewise derived state, never serialized), ADR-011
  (trained-model feature lifecycle — the bundle `GhostGkModel` belongs to).

## Notes

- Bit-identity measured on a 500-tree model, 3000 rows (Metrica game): `_vectorized_leaf_values`
  922 ms → `_leaf_values_numba` 78 ms, `np.array_equal` True (max |Δ| 0.000e+00).
- The two discrimination proofs (flip the `go_left` inequality → the bit-identity gate fails; remove
  the values-kernel guard → the >depth-cap "values raises" leg fails) were run and reverted; they
  are documented in the test module docstring, not committed as perturbations.
