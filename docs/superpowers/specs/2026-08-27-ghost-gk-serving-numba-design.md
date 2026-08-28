# Ghost-GK serving performance — numba-accelerate the shared leaf traversal + exact KDE default — design

**Status:** APPROVED — lakehouse reviews #1 + #2 incorporated (2026-08-27). Ready for implementation planning.
**Origin:** perf survey after 4.97.0 (gkdv batching). The #13 "ghost-GK KDE / 1800 s budget" note was
fact-checked as **stale** (ADR-044 retired the KDE from production; the serving path is `predict_mean`).
This cycle instead speeds the paths that ARE live, done to a gold standard.
**Retrain / re-materialize:** NONE. Part 1 is **bit-identical**; Part 2's default shift is exact within
1e-9 and already golden-validated. No production consumer of the KDE exists.

The ghost-GK boosted-tree **leaf traversal** is a shared hot loop: `_vectorized_leaf_values` backs the
production `predict_mean` (serving), and `_vectorized_leaf_indices` backs the KDE query leaf-match
(`predict_density`) AND the `fit()` training leaf-match. Both are numpy "vectorized over samples,
looped over trees × depth" with fancy-indexing/`np.where` temporaries per step. A fused `@njit`
per-sample tree walk collapses them. Measured (500-tree model, 3000 rows = a Metrica game):
`_vectorized_leaf_values` **922 ms → 78 ms (11.8×), BIT-IDENTICAL** (max |Δ| = 0.000e+00). `predict_mean`
(two traversals) therefore drops ~1.9 s → ~0.16 s/game.

This cycle adds ONE exact numba leaf-traversal kernel pair, dispatches the two numpy traversals to it
(numpy fallback when numba is absent), and makes the KDE default resolve to the fastest **exact**
backend (`cpu-numba`) instead of the slow `vectorized` — keeping `fft` as the explicit
approximation opt-in.

---

## 1. Problem

- **`predict_mean` (production serving) is 12× slower than it needs to be.** The numpy leaf traversal
  allocates per-tree, per-depth temporaries. Measured 922 ms for one traversal over 3000 rows / 500
  trees; `predict_mean` runs it twice (x-tree + y-tree) ≈ 1.9 s/game. It is well under the 1800 s
  per-unit budget today, but it is the live production path and 12× is free and bit-identical.
- **The KDE (`predict_density`) default is the slowest backend.** Default `kde_backend="vectorized"`
  is the O(k·m) brute-force numpy oracle (~261 ms/pred synthetic; ~4247 ms/pred on the real corpus per
  ADR-014). Two faster backends exist but are opt-in: `cpu-numba` (exact within 1e-9, ~10×) and `fft`
  (~2000×, binning-approximate on the raw grid). The KDE is research/dev-only (ADR-044 retired it from
  every aggregator), so its default should be the fastest **exact** path, not the slowest.
- **The leaf-match is shared.** `_vectorized_leaf_indices` runs in `predict_density` (query leaves,
  `_ghost_gk.py:45`), `fit()` (training leaves, `:1777`); `_vectorized_leaf_values` in `predict_mean`
  (`:1811`). One kernel accelerates all three.

## 2. Goals / non-goals

**Goals**
- A fused `@njit` leaf-traversal kernel pair (`_leaf_values_numba`, `_leaf_indices_numba`) in the
  existing `tracking/_ghost_gk_numba.py`, **bit-identical** to the numpy traversals.
- `_vectorized_leaf_values` / `_vectorized_leaf_indices` **dispatch** to numba when available, numpy
  fallback otherwise (the ADR-013 / ADR-008 pattern). No new hard dependency.
- `predict_density`'s default resolves to `cpu-numba` when numba is present, else `vectorized`
  (fastest **exact** backend; `fft` stays explicit opt-in).
- Flatten each model's trees **once** (cached at `fit`/`load`), not per prediction.

**ROI, stated plainly (review #1):** this is deliberate gold-standard polish of a **non-bottleneck** —
`predict_mean` is already ~900× under the per-unit watchdog, and 4.97.0's DAS batching already fixed
the gkdv-scale cost. The bundle is defensible because ONE exact kernel speeds the production serving,
`fit()`, AND the research KDE leaf-match, and because the intent is a reference-quality example
(owner's "how you do anything is how you do everything"). Production urgency is low; correctness bar is
not.

**Non-goals**
- `fft` as a default (kept opt-in — it is approximate on the raw grid; gold standard = exact default).
- GPU / `parallel=True` numba (serial only — Spark saturates cores across `applyInPandas` groups,
  ADR-013 rationale).
- Any change to the trained weights, the served geometry, the feature set, or the model artifacts.

## 3. Settled decisions (with rationale)

1. **Bit-identical, not tolerance, for the leaf traversal.** Leaf INDICES are integer walks and the
   summed leaf VALUES use the same operations in the same order → exactly equal (measured
   0.000e+00). So the dispatch is transparent: every golden / chirality / feature-contract test passes
   unchanged, and there is no retrain and no re-baseline. This is the whole reason Part 1 is safe.
2. **Numba is auto-used, never required.** `_HAS_NUMBA` (lazy import) gates the dispatch; without the
   `[numba]` extra the numpy path runs, byte-identical output. Matches pitch-control's `@njit` +
   fallback (ADR-008) and the KDE `cpu-numba` (ADR-013).
3. **Exact default for the KDE (`cpu-numba`), `fft` opt-in.** The default must not silently make a
   numerical research tool approximate. `cpu-numba` == `vectorized` == `scipy` at 1e-9 (the golden
   already runs `cpu-numba`), so the default shift is exact-arithmetic; `fft` (binning) stays a
   deliberate opt-in for the "2000× and I accept a binned raw grid" case (scalars stay faithful).
4. **Flat trees cached on the model.** The numba kernels take flat `left/right/feat/thr/miss/val` +
   per-tree `offsets` (numba cannot take a list of structured arrays). Building them once at
   `fit`/`load` (from `_tree_nodes`) avoids re-concatenating 500 trees per prediction; they are derived
   state, NOT serialized (rebuilt on `load`).
5. **Serial `@njit`, shared cache flag.** `cache=_NUMBA_CACHE` (the module's existing
   `SILLY_KICKS_NUMBA_CACHE`/`NUMBA_CACHE_DIR` flag); no `prange`.

## 4. Architecture

### §4.1 The kernels (`tracking/_ghost_gk_numba.py`)

Two serial `@njit` kernels beside `_kde_numba_loop`, each a per-sample scalar tree walk over flat
arrays:

```python
@njit(cache=_NUMBA_CACHE)
def _leaf_values_numba(left, right, feat, thr, miss, val, offsets, X):
    n, T = X.shape[0], offsets.shape[0] - 1
    out = np.zeros(n)
    for s in range(n):
        acc = 0.0
        for t in range(T):
            base = offsets[t]
            cur = 0                         # local index within tree; root=0, leaf iff left==0
            for _ in range(100):            # depth bound, matches the numpy path
                gi = base + cur
                if left[gi] == 0:
                    break
                fv = X[s, feat[gi]]
                go_left = (miss[gi] != 0) if np.isnan(fv) else (fv <= thr[gi])
                cur = left[gi] if go_left else right[gi]
            # Convergence guard -- MUST match the numpy path's RuntimeError (Chesterton's fence,
            # review #1): a tree deeper than the cap would read an INTERNAL node's value = garbage.
            # numba nopython supports a constant-message raise.
            if left[base + cur] != 0:
                raise RuntimeError("leaf traversal did not converge within depth cap")
            acc += val[base + cur]
        out[s] = acc
    return out
```

`_leaf_indices_numba` is identical **except it carries NO convergence guard** — and that is
deliberate, because the two numpy functions genuinely DIFFER in their failure mode:
`_vectorized_leaf_values` RAISES on non-convergence (it reads `value`), while
`_vectorized_leaf_indices` does NOT (`_ghost_gk.py:1210-1244` returns the non-converged `current`
index silently — it never reads `value`). So the indices kernel returns the reached `(n, T)`
**LOCAL** leaf index — `cur`, matching numpy's per-tree `current` (`leaves[:, t_idx] = current`) — per
(sample, tree), WITHOUT raising. `base + cur` is used ONLY to ACCESS the flat arrays (`left[gi]`,
`val[gi]`); it is NEVER the returned value (review #2: returning global `base + cur` would break the
§5.1 `np.array_equal` against numpy's local `current`, and is a latent Hyrum surface —
`_leaf_match_weights` compares per-tree columns where `base` cancels, so end-to-end is correct either
way, but the gate AND the contract require local). Adding a guard here would itself BREAK bit-identity
in the >depth-cap case (numpy returns, numba would raise). Matching each function's ACTUAL behaviour is
the point (review #1: "same for indices" is one step too far). NaN → `missing_go_to_left`, the
`left==0` leaf test, and the depth bound mirror the numpy path exactly.

### §4.2 Flat-tree marshaling + caching

A pure helper `_flatten_trees(nodes_list) -> FlatTrees` (offsets + the six field arrays, dtypes pinned:
`int64` for left/right/feat/miss, `float64` for thr/val). **`left`/`right` stay per-tree LOCAL indices,
NOT remapped to global** (review #1 note): the kernel walks `cur = left[gi]` (local child) then
`gi = base + cur`, so remapping to global would double-add `base`. The §7.1 round-trip test pins this
by asserting `left`/`right` in a flattened slice equal the source tree's raw `left`/`right` (not
offset-shifted). `GhostGkModel` caches `FlatTrees` lazily — computed on first traversal and after
`fit`/`load`, invalidated if `_tree_nodes` is (re)assigned. Derived state, never serialized; `load()`
rebuilds it from the deserialized `_tree_nodes`.

### §4.3 Dispatch (`_vectorized_leaf_values` / `_vectorized_leaf_indices`)

Each numpy function gains a thin head: if numba is importable AND the caller passes the cached
`FlatTrees` (or the model dispatches with it), call the kernel; else run the existing numpy body
verbatim (kept as the reference + fallback). A module `_HAS_NUMBA` flag plus a **test-only env override**
(`SILLY_KICKS_GHOST_FORCE_NUMPY`) lets CI exercise BOTH paths on every leg (the ADR-013 discipline).
`predict_mean` / `predict_density` / `fit` pass the model's cached `FlatTrees`.

### §4.4 KDE default → fastest exact backend

`predict_density(..., kde_backend="auto")` becomes the default (was `"vectorized"`). `"auto"` resolves
to `"cpu-numba"` when numba is importable, else `"vectorized"`. The explicit values
(`vectorized`/`scipy`/`cpu-numba`/`fft`/`fft-cic`) are unchanged and still selectable. The density loop
(`_kde_numba_loop`) and the leaf-match (`_leaf_indices_numba`) are then both numba on the default path.

### §4.5 The 1e-9 default shift (documented, not hidden)

Callers who relied on the implicit `vectorized` default now get `cpu-numba`, which differs by ≤1e-9
(numba `exp` vs numpy `exp`; exact-arithmetic parity, the golden's own tolerance). `predict_mean` has
**no** shift (bit-identical). Both are stated in the docstrings + CHANGELOG. Raw-grid consumers who need
the exact numpy grid pass `kde_backend="vectorized"` explicitly (unchanged).

## 5. Testing strategy (gold standard)

1. **Bit-identity (the core gate).** `_leaf_values_numba` and `_leaf_indices_numba` == their numpy
   siblings **exactly** (`np.array_equal`), on: a fitted multi-tree model, single-node (leaf-root)
   trees, deep trees *at* the depth bound, and **NaN feature rows** (both `missing_go_to_left`
   branches). The discrimination proof (equality can't be red-first when the kernel doesn't import
   yet): perturb the kernel (flip the `go_left` inequality) → the equality fails.
   **Failure-mode parity (review #1), and it is ASYMMETRIC.** A hand-built synthetic tree that
   *exceeds* the 100-depth cap asserts: the **values** path raises `RuntimeError` on BOTH the numpy
   `_vectorized_leaf_values` and `_leaf_values_numba` (same message); the **indices** path raises on
   NEITHER and returns the identical non-converged index on both (`np.array_equal`). This pins that
   the kernel matched each numpy sibling's real behaviour, not a blanket "both raise" — the exact trap
   the reviewer flagged one step past.
2. **Both paths on every leg.** A parametrized fixture runs the traversal with numba forced ON and
   forced OFF (`SILLY_KICKS_GHOST_FORCE_NUMPY`), asserting identical output — so the numpy fallback is
   never silently broken and the no-`[numba]` leg is covered.
3. **`predict_mean` / `predict_density` / `fit` end-to-end unchanged.** The existing chirality
   fingerprint, feature-contract, serve-mean, KDE golden, and parameters-only tests pass **unchanged**
   (Part 1 bit-identical; the KDE golden already runs `cpu-numba`). Add an explicit
   `predict_mean(numba) == predict_mean(numpy)` bit-identity test.
4. **KDE default resolution.** `predict_density()` with no `kde_backend` resolves to `cpu-numba` when
   numba present (assert via a spy on the backend dispatch) and `vectorized` when forced numpy;
   `"auto"` is in the documented value set; explicit backends still selectable; `fft` still opt-in.
5. **Structural perf guard (no wall-clock).** A call-count / dispatch spy asserting the numba kernel IS
   invoked on the default path when numba is available (so a regression that silently reverts to numpy
   is caught) — the `tests/_perf_structural` discipline, deterministic, not timing.
6. **Flat-tree cache.** Built once (spy the flatten call count across repeated `predict_mean`),
   rebuilt after `load`, and a fresh object per model (no cross-model leakage). Purity: the kernels do
   not mutate `X` or the node arrays.
7. **Cross-version.** The numba parity + fallback run under the CI pandas-major span (ADR-057); the
   kernel is dtype-pinned so `int64`/`Int64` node arrays behave identically.

## 6. Constraints

- **numba optional + numpy fallback** — no new runtime dependency; `[numba]` extra unchanged (already
  in `[test]`, so CI exercises the kernels).
- **Serial `@njit`**, `cache=_NUMBA_CACHE` — matches `_kde_numba_loop`.
- **No retrain, no re-materialize, no artifact change.** The bundled weights + geometry are untouched.
- **ADR-011 / chirality / feature-contract** load guards are unaffected (served output identical for
  `predict_mean`; the density is not in an artifact contract).
- Lint at CI scope; bare pyright; full `-m "not e2e"` suite.
- **Single feature branch, single commit, single PR** (owner-restated).
- **Nobody claims version numbers until commit-prep — ENFORCE (review #2).** During execution, write the
  ADR with a PLACEHOLDER number (`ADR-NNN`) and do NOT bump `pyproject.toml`/`__init__`/`uv.lock`/
  CHANGELOG/TODO. ONLY at commit-prep — after `git fetch && git merge origin/main` — take the real
  NEXT-FREE version/PR-S/ADR and fill all six places (5 version strings + the ADR rename) in one pass.
  The `4.98.0 / PR-S169 / ADR-076` written anywhere here is a NON-BINDING note; another PR may take it
  first. (4.97.0 lesson: that cycle assigned the numbers pre-commit — do not repeat.)
- **No commit without explicit owner approval.**

## 7. Execution ordering (review-tractable; NOT commit boundaries)

1. `_flatten_trees` + `FlatTrees` + its unit tests (dtypes, offsets, round-trip vs `_tree_nodes`).
2. `_leaf_values_numba` + `_leaf_indices_numba` + the bit-identity gates (§5.1) incl. NaN + single-node
   + depth-bound, with the perturb-to-fail discrimination proof.
3. Dispatch in `_vectorized_leaf_values` / `_vectorized_leaf_indices` + the `SILLY_KICKS_GHOST_FORCE_NUMPY`
   both-paths fixture (§5.2); cache the `FlatTrees` on the model at `fit`/`load` (§5.6).
4. `predict_mean` bit-identity test + confirm chirality/feature-contract/serve-mean unchanged (§5.3).
5. KDE default → `"auto"` (cpu-numba-if-numba-else-vectorized) + resolution tests (§5.4); confirm the
   KDE golden + parity unchanged; the 1e-9 docstring/CHANGELOG note. **Sweep for any test that calls
   `predict_density()` on the implicit default and asserts exact (`<1e-9`) equality to a
   `vectorized`/`scipy` reference** — such a test now sees `cpu-numba` and must either pin
   `kde_backend="vectorized"` explicitly (if it is a raw-grid exactness oracle) or adopt the golden's
   `cpu-numba` tolerance. The full `-m "not e2e"` suite is the backstop; this sweep is the deliberate
   pre-check (the SB360-boundary lesson from 4.97.0 — a default change can move a gate the plan's own
   tests don't list).
6. Structural dispatch guard (§5.5).
7. Docs: docstrings (`predict_mean`/`predict_density`/the kernels), CHANGELOG, CLAUDE.md ghost-GK note,
   ADR (numba-default exact serving + KDE exact-default), NOTICE unchanged. C4 verify (no new
   aggregator — count unchanged). Full CI-faithful gate + /final-review + /c4.

## 8. Known limits (stated, not discovered)

- **JIT warm-up.** The first numba call per process pays compile time (~0.1–1 s); the on-disk cache is
  off by default on serverless (ADR-013). This is a per-process one-time cost, not per-prediction —
  and the numpy fallback has no warm-up, so a single-prediction caller sees no benefit. Documented.
- **The 1e-9 KDE default shift** (§4.5) — exact-arithmetic, golden-covered, but it IS an output change
  for a caller that relied on the implicit `vectorized` default and reads the raw grid at <1e-9. They
  pass `kde_backend="vectorized"` to pin the exact numpy grid.
- **Flat-tree cache memory** is O(total nodes) per model — a few MB for 500 trees; negligible.
- **Scope is the leaf traversal + KDE default.** The KDE density-loop `fft` (2000×) is deliberately not
  the default (§3.3); the `predict_mean` traversal is the only production speedup and it is complete.

## 9. Resolved (both reviews)

- **Dispatch call convention → explicit `FlatTrees` arg** (review #2 / hexagonal): `_vectorized_leaf_*`
  is the PORT (numpy reference + fallback), the `@njit` kernel is the ADAPTER, and `FlatTrees` is passed
  explicitly — keeping the port pure/testable and the adapter swappable; the `SILLY_KICKS_GHOST_FORCE_NUMPY`
  override exercises both adapters on every leg. No hidden model state on the traversal functions.
- **`"auto"` stays on `predict_density` only** — ADR-044 retired `kde_backend` from
  `compute_ghost_gk`/`add_ghost_gk`/`ghost_gk_xfns`; this cycle does not re-add it (the density path is
  not an aggregator concern).
