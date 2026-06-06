# Ghost-GK Option A (exact boosted `predict_mean`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline — no subagents per
> project policy). One commit at the very end after /final-review; NO per-task commits.

**Goal:** Serve the exact sklearn HGBR boosted `predict_mean` (expected ~1.1 m — **NOT established; Task-12
re-measures on the same split, ship gate §3.6**), reconstructed pickle-free via leaf `value` traversal,
replacing the rejected Option-B conditional mean (7.0 m) and the status-quo mode (4.65 m).

**Spec:** `docs/superpowers/specs/2026-06-04-pr-s83-ghost-gk-option-a-design.md` — read it first.
**Prior context:** memory `project_pr_s83_ghost_gk_option_a`. The Option-B code is in the working tree
(uncommitted) — this plan MORPHS it. Branch `pr-s83-ghost-gk-serve-mean` on main 915099b (4.12.0).

## ⚠️ Policy
ONE commit at the end after `/final-review` + explicit approval (sentinel-gated). Per-task = verification
gate only: `pytest tests/tracking/test_ghost_gk*.py -m "not e2e" -q` + `ruff check` + `ruff format --check`
+ `pyright silly_kicks/`. Session start: `pip install -e ".[test]"`.

---

## Task 0: Pre-flight
- [ ] Confirm current `fit()` structure (`categorical_features`, `regressor._predictors`,
      `regressor._baseline_prediction`), `save()`/`load()` npz layout, and that `_tree_nodes` carries the
      `value` field. Confirm the working tree still has the B changes (`_central_estimate`, `central_x/y`,
      `SERVED_ESTIMATOR`, predict_mean=B, fit-drops-regressors).
- [ ] `grep -n "categorical_features\|_baseline_prediction\|_predictors\|def fit\|def save\|def load\|def predict_mean\|def predict_density\|central_x\|_central_estimate" silly_kicks/tracking/_ghost_gk.py`
- [ ] Confirm **`phase` is one of the 26 `GHOST_GK_FEATURE_NAMES`** (the tests do `X["phase"]=...` assuming
      it's a model feature, and `phase_idx` is computed from it) — `python -c "from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES as F; print(len(F), 'phase' in F, F.index('phase'))"`.
- [ ] `pip install -e ".[test]"`; baseline `pytest tests/tracking/test_ghost_gk.py -m "not e2e" -q` (expect
      the B tests; some will be rewritten).

## Task 1: `_vectorized_leaf_values` + the parity gate (the correctness spine)

**Files:** `silly_kicks/tracking/_ghost_gk.py`; test `tests/tracking/test_ghost_gk_serve_mean.py`.

> **This test stays RED until Task 7** — it depends on phase-numeric (T2) + the new predict_mean (T3) +
> save/load restoring trees_y+baselines (T7). Land T1→T2→T3→T7, then it goes GREEN. A worker should NOT read
> the lingering RED as a failure; it is the spine asserting the whole chain.
>
> **Pre-validated (2026-06-04):** a throwaway prototype of this exact reconstruction (`categorical_features=
> None`; `baseline + Σ leaf_values` vs sklearn `.predict()`) hit **1.155e-14** parity (8.88e-15 on NaN rows),
> 0 categorical nodes, max_depth 8 — so the kernel + phase-numeric + NaN routing are known-correct; this task
> is implementing a proven approach, not exploring one.

- [ ] **Step 1 — failing test** (parity vs sklearn, **including NaN rows** for the missing-value branch;
      also asserts phase-numeric removed categoricals):

```python
def test_boosted_predict_mean_matches_sklearn():
    """predict_mean (numpy leaf-value reconstruction) == sklearn .predict() to 1e-6, incl. NaN routing."""
    from silly_kicks.tracking import _ghost_gk as gg
    import numpy as np, pandas as pd
    rng = np.random.default_rng(3); n = 300
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)  # exercise phase values
    # NaN in real features (#6: derive from the schema, never hardcode names) so HGBR learns
    # missing_go_to_left and the reconstruction's missing branch is parity-checked (#4):
    nan_cols = [c for c in gg.GHOST_GK_FEATURE_NAMES if c != "phase"][:4]
    for col in nan_cols:
        X.loc[rng.choice(n, 30, replace=False), col] = np.nan
    y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    m = gg.GhostGkModel(n_estimators=30).fit(X, y)
    # No categorical split nodes after phase-numeric (guards exact reconstruction):
    assert sum(int(t["is_categorical"].sum()) for t in m._tree_nodes) == 0
    # Parity vs the live sklearn regressors kept transiently after fit() (canonical col order):
    Xv = X[gg.GHOST_GK_FEATURE_NAMES].values
    sk = np.column_stack([m._sk_reg_x.predict(Xv), m._sk_reg_y.predict(Xv)])
    np.testing.assert_allclose(m.predict_mean(X), sk, atol=1e-6)
```

- [ ] **Step 2 — run, expect RED** (`_vectorized_leaf_values`/`_sk_reg_x` absent; predict_mean is still B).
- [ ] **Step 3 — implement** `_vectorized_leaf_values` (sibling of `_vectorized_leaf_indices` — same
      traversal, but accumulate `node_data["value"]` at the reached leaf per tree):

```python
def _vectorized_leaf_values(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Sum of reached-leaf `value` across all trees (HGBR raw additive prediction).

    Numeric thresholds only (valid because fit() trains with no categorical features —
    spec §3.2). Same NaN / missing_go_to_left handling as _vectorized_leaf_indices.
    """
    n_samples = X.shape[0]
    total = np.zeros(n_samples, dtype=np.float64)
    for nodes in nodes_list:
        current = np.zeros(n_samples, dtype=np.intp)
        for _ in range(100):
            node = nodes[current]
            is_leaf = node["left"] == 0
            if np.all(is_leaf):
                break
            feat = X[np.arange(n_samples), node["feature_idx"]]
            go_left = np.where(np.isnan(feat), node["missing_go_to_left"].astype(bool),
                               feat <= node["num_threshold"])
            nxt = np.where(go_left, node["left"], node["right"])
            current = np.where(is_leaf, current, nxt)
        if not np.all(nodes[current]["left"] == 0):  # (#7) convergence guard — raise, not silent garbage
            raise RuntimeError("leaf traversal did not converge within depth cap")
        total += nodes[current]["value"]
    return total
```
   (predict_mean wiring is Task 3; `_sk_reg_x/y` + phase-numeric are Task 2 — so Step 4 lands GREEN after
   Tasks 2–3. Order Tasks 1→2→3 then re-run this test.) Uses `left == 0` for leaf detection to stay a
   verbatim sibling of `_vectorized_leaf_indices`; the dtype also has an `is_leaf` field (equivalent) — the
   parity test guards whichever is used.

## Task 2: `fit()` — phase-numeric + re-add gk_y + store trees/baselines

- [ ] **Step 1 — failing test** (`tests/tracking/test_ghost_gk_serve_mean.py`):

```python
def test_fit_stores_both_ensembles_and_baselines_no_categoricals():
    from silly_kicks.tracking import _ghost_gk as gg
    import numpy as np, pandas as pd
    rng = np.random.default_rng(1); n = 120
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=gg.GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    y = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    m = gg.GhostGkModel(n_estimators=10).fit(X, y)
    assert m._tree_nodes_y is not None and len(m._tree_nodes_y) > 0
    assert isinstance(m._baseline_x, float) and isinstance(m._baseline_y, float)
    assert sum(int(t["is_categorical"].sum()) for t in m._tree_nodes) == 0  # phase numeric
```

- [ ] **Step 2 — RED.**
- [ ] **Step 3 — implement** in `fit()` (read it first; morph the B version):
  - `__init__`: add `self._tree_nodes_y = None`, `self._baseline_x = None`, `self._baseline_y = None`.
    Keep transient `self._sk_reg_x/y` for the parity test (NOT serialized — like the old regressors).
  - Remove `categorical_features=...` → pass **`categorical_features=None`** to both regressors (all numeric).
  - Train gk_x regressor (existing) AND re-add the gk_y regressor (same hyperparams).
  - **Fail-fast (#8 — `raise`, NOT `assert`; `python -O` strips asserts, #5):**
    `if not hasattr(regressor, "_predictors") or regressor._baseline_prediction.size != 1: raise
    RuntimeError("sklearn HGBR private API changed — reconstruction needs review")`.
  - Extract `_tree_nodes` (gk_x, existing) AND `_tree_nodes_y` from `regressor_y._predictors`.
  - **numpy-2 safe (#6):** `self._baseline_x = float(regressor._baseline_prediction.item())`;
    `self._baseline_y = float(regressor_y._baseline_prediction.item())` (shape (1,1); bare `float(ndarray)`
    warns/raises under numpy≥2).
  - Keep `training_leaves` (gk_x) + `training_gk_x/y` (KDE needs them).
  - `self._sk_reg_x = regressor; self._sk_reg_y = regressor_y` (transient, for parity test only).
- [ ] **Step 4 — run Task-2 test GREEN.**

## Task 3: `predict_mean` — boosted reconstruction (load-safe)

- [ ] **Step 1 — failing tests:** load-safe + fit==load parity:

```python
def test_predict_mean_boosted_load_safe_and_parity():
    from silly_kicks.tracking import _ghost_gk as gg
    import numpy as np, pandas as pd, tempfile
    from pathlib import Path
    rng = np.random.default_rng(2); n = 100
    X = pd.DataFrame(rng.standard_normal((n,26)), columns=gg.GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0,3,n).astype(float)
    y = pd.DataFrame({"gk_x": rng.uniform(2,20,n), "gk_y": rng.uniform(25,45,n)})
    m = gg.GhostGkModel(n_estimators=20).fit(X, y)
    before = m.predict_mean(X[:8])
    with tempfile.TemporaryDirectory() as t:
        p = Path(t)/"m"; m.save(p); after = gg.GhostGkModel.load(p).predict_mean(X[:8])
    np.testing.assert_array_equal(before, after)  # bit-identical pre/post load
```

- [ ] **Step 2 — RED** (predict_mean is still B; load doesn't restore trees_y/baselines until Task 7).
- [ ] **Step 3 — implement** `predict_mean` (replace the B body):

```python
    def predict_mean(self, features: pd.DataFrame) -> np.ndarray:
        """Served estimate: exact sklearn HGBR boosted prediction, pickle-free + load-safe.
        predict = baseline + sum of reached-leaf values (squared-error, identity link)."""
        if self._tree_nodes is None or self._tree_nodes_y is None or self._baseline_x is None or self._baseline_y is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
        # Reindex to canonical fit-time column order (#3 Hyrum): reconstruction indexes
        # X[:, feature_idx] positionally; a reordered DataFrame would silently mis-predict.
        X = features[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)
        out = np.empty((len(X), 2), dtype=np.float64)
        out[:, 0] = self._baseline_x + _vectorized_leaf_values(self._tree_nodes, X)
        out[:, 1] = self._baseline_y + _vectorized_leaf_values(self._tree_nodes_y, X)
        return out
```
   Apply the same `features[GHOST_GK_FEATURE_NAMES]` reindex to the `predict_density` KDE `features.values`
   path (same latent positional assumption).
   (Requires Task 7 save/load to restore trees_y+baselines for the load-safe test; sequence Task 7 before
   re-running this + the Task-1 parity test, or land 1→2→3→7 then run all four.)

## Task 4: `predict()` serves the boosted mean
- [ ] Confirm `predict()` returns `self.predict_mean(features)` (already from B). Keep. Mode via
      `predict_density(...).mode_x/y`. Tests: `predict()==predict_mean`; mode reachable.

## Task 5: `predict_density` + `GhostGkDensity` — drop the B central fields
- [ ] Remove `central_x/central_y` from `GhostGkDensity` (and its docstring/`__post_init__` untouched).
- [ ] Remove the `all_central = _central_estimate(...)` line + the `central_x=/central_y=` kwargs in
      `predict_density`'s `GhostGkDensity(...)`. Remove `_central_estimate` (no longer used) and its B tests.
- [ ] Update the 2 `GhostGkDensity(...)` construction sites in `test_ghost_gk.py` (remove central_x/y).
- [ ] KDE golden + density tests still pass *structurally* (values shift in Task 10 after re-fit; for the
      synthetic `_fitted_model` they recompute live, so just ensure no central_* references remain).

## Task 6: `compute_ghost_gk` — boosted position + density spread (+ #8 rename)
- [ ] `ghost_gk_x/y` from `predict_mean(batch_features)`; **`ghost_gk_density_spread`** (renamed, #8) from
      `predict_density(batch_features).spread`. (compute now calls both — predict_mean cheap.)
- [ ] **RENAME SWEEP `ghost_gk_spread` → `ghost_gk_density_spread`** (owner-decided #8; emitted column only —
      the internal `GhostGkDensity.spread` *attribute* keeps its name): `compute_ghost_gk`, `add_ghost_gk`,
      the `ghost_gk_xfns` column list (`silly_kicks/tracking/features.py`), the atomic mirror, the backcompat
      golden columns (Task 10), and **all tests** referencing `ghost_gk_spread`
      (`grep -rn ghost_gk_spread tests/ silly_kicks/`). The **lakehouse column** is cross-repo → the
      no separate lakehouse review this cycle — CHANGELOG/ADR flag the breaking rename loudly so the lakehouse
      side updates on consume.
- [ ] **Docstring (#7 Hyrum):** note `ghost_gk_density_spread` is the conditional-**density** dispersion, NOT
      the standard error of the served boosted point (`ghost_gk_x/y`) — they no longer come from the same
      point (position = boosted mean, spread = density around the mode/cloud). The rename makes this
      structural; the docstring reinforces it. Also note density is now the only cost driver here (position
      is ~free).
- [ ] **Discriminating test (module-pollution-robust):** build model+compute from the CURRENT `gg` module
      (fresh, no `_fitted_model` cache); assert served `ghost_gk_x` == `model.predict_mean(feats)` for the
      kept frames and `!= predict_density(feats).mode_x`. (Reuse the B test's robust pattern; swap central→
      predict_mean.)

## Task 7: `save()`/`load()` — serialize gk_y trees + baselines + serve_estimator
- [ ] **Failing test:** save→reload→`predict_mean` parity (Task 3's test depends on this).
- [ ] `save()`: add `tree_nodes_y_{i}` (+ dtype) arrays, `n_trees_y`, `baseline_x`, `baseline_y` to the npz;
      `metadata["serve_estimator"] = SERVED_ESTIMATOR`; bump `metadata["version"] = "1.2.0"`; SHA256SUMS.
- [ ] `load()`: reconstruct `_tree_nodes_y`, `_baseline_x/_baseline_y`; keep the R3 `serve_estimator`
      fail-closed check (absent→default, conflict→IntegrityError). **Fail-closed on pre-Option-A artifacts
      (#3):** if `tree_nodes_y_*`/baselines are missing, raise a clear "artifact predates Option A — re-fit
      required" error (not a cryptic KeyError). Add the **definite** `sklearn_version`-mismatch warning
      (informational, not a correctness guard — §3.5).
- [ ] `SERVED_ESTIMATOR = "boosted_mean"`.

> **SEQUENCING (format change, like PR-S81):** the moment `load()` requires `_tree_nodes_y`, every test that
> loads the bundled `default` (`from_variant`) goes RED — the committed bundled weights are still old-format
> until **Task 12** re-fits + re-bundles. Develop Tasks 1–11 against **fresh-model** tests; the bundled-default
> suite + the §3.x smoke test go GREEN after Task 12. The per-task verification gate for Tasks 1–11 is
> "fresh-model tests + ruff/pyright green"; the full bundled-default suite is gated at Task 12.

## Task 8: metadata fail-loud test
- [ ] Keep/adapt the B metadata tests (record, absent→ok, conflict→raise) with the new `"boosted_mean"`.

## Task 9: Re-run the full new test file + Task-1 parity GREEN
- [ ] `pytest tests/tracking/test_ghost_gk_serve_mean.py -q` — parity ≤1e-6, no-categoricals, load-safe,
      predict==served, compute-serves-boosted, metadata. All GREEN (fresh-model tests).
- [ ] **Write the bundled-artifact smoke test (#3)** in the new test file: load `GhostGkModel.from_variant
      ("default")`, assert `predict_mean(small_fixture)` returns finite coords within pitch bounds + `_tree_nodes_y`
      present + baselines restored. (The only test that proves the *re-published weights* serialize+reconstruct;
      synthetic-model tests can't.) **RED until Task 12 re-bundles the new-format default** — same window as
      the bundled-default `from_variant` tests; do not treat its RED as a failure during Tasks 1–11.
- [ ] `pytest tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_integration.py -m "not e2e" -q`
      (bundled-default-loading cases RED until Task 12).

## Task 10: Regenerate goldens (phase-numeric shifts the partition)
- [ ] Backcompat fixture `ghost_gk_backward_compat.parquet` — regenerate to the A output (the B recipe in
      `test_backward_compat`, run against the new code). **Review the diff** — the served x/y are now the
      boosted mean; spread shifts (phase-numeric). Load-bearing.
- [ ] KDE golden `ghost_gk_kde_golden.npz` — regenerate (phase-numeric removes the 24 categorical nodes →
      density shifts). Note in the gen-script run that this is the categorical-fix consequence.

## Task 11: Measurement script for A (fast — no chunking) — SAME SPLIT as the B table
- [ ] Adapt `scripts/measure_ghost_gk_estimators.py`: A's `predict_mean` (boosted) is the primary candidate.
      It's CHEAP (leaf-value traversal, no leaf-match, no KDE) so no chunking for it. Keep **mode**
      (predict_density fft-cic, chunked) as the comparison. **Measure boosted-mean on the IDENTICAL held-out
      split as the B-rejection table** (same StratifiedGroupKFold(provider_labels) seed=42 split + same
      8000-eval subsample seeds + fft-cic for the mode) so **mode 4.65 / B 7.0 / boosted ?** are directly
      comparable (blocking #1). Keep the B/geom-median arms as the same-split reference. Report MAE + RMSE.
- [ ] **Stratify by multimodality (#1 — the path that bites):** compute a per-frame multimodal flag from the
      density (mode↔grid-mean gap > 4 m, the same proxy as the §1 table) and report boosted-mean MAE
      **pooled + multimodal-subset + unimodal-subset** (alongside mode + B on the same strata). The boosted
      mean is also a conditional mean → a good pooled MAE could hide a ~7 m miss on the high-leverage
      multimodal ~50% (wide crosses, set-pieces). The §3.6 gate consumes the multimodal-subset number, not
      only the pooled MAE. (Confirms the §3.5 "B's-bad-density, not real multimodality" mechanism.)

## Task 12: DGX re-fit + measure + re-publish (maintainer)
- [ ] scp `_ghost_gk.py` + `train_ghost_gk.py` + measure script to the box (box repo refspec broken — use
      `git fetch origin main && git checkout -f FETCH_HEAD`, then scp).
- [ ] Re-fit BOTH variants from the feature cache (default = `--subsample-cap`; full = all 887k) → new-format
      artifacts. **BLOCKING parity-on-fresh-fit gate (#4):** before publishing, on a fresh box fit assert
      `predict_mean(X)` == the live `_sk_reg_x/y.predict(X)` to ≤1e-6 (regressors are transient post-load, so
      this runs on the fresh fit, not the loaded artifact). Then `load()` the saved artifact + serve a sample
      (finite, in-bounds). Do NOT publish if parity fails.
- [ ] Run the A measurement on the same split as the B table, **stratified by the multimodal flag**.
      **Pre-registered decision procedure (spec §3.6 — NO arbitrary absolute bar; do NOT assume ~1.1 m):**
      - **HARD FAIL → STOP/debug:** boosted-mean MAE **≥ the mode** on *either* pooled OR multimodal subset
        (if pooled, suspect a reconstruction bug → re-check parity).
      - **CLEAR PASS → ship:** boosted beats the mode by a clear margin on **both** pooled AND multimodal
        subset, AND the multimodal-subset MAE is not pathological (well below B's ~7 m).
      - **Anything between → owner checkpoint:** present the full stratified table (mode/B/boosted ×
        pooled/multimodal/unimodal, MAE+RMSE) to the owner; the absolute "good enough" bar is the
        VAEP-consumer requirement (owner-set with the data, by how the feature is consumed), not pre-guessed.
        Record decision + numbers in
        ADR-016 / PR.
      Record the measured boosted-mean MAE; the chosen number feeds the CHANGELOG + model card (Task 13).
- [ ] Bundle `default` in the wheel (≤ wheel limit); publish `full` to Hub `silly-kicks/ghost-gk-v1` (see
      `project_pr_s81_ghost_gk_refit` recipe). Check BOTH sdist + wheel < 100 MB
      (`feedback_hatch_sdist_exclude_separate_from_wheel`). **Note:** A does NOT shrink the artifact — it
      still stores `training_leaves` + `training_gk_x/y` (the 887k-sample bulk) for `predict_density`
      (spread/mode); the added gk_y tree arrays are negligible.

## Task 13: Version + ADR + docs
- [ ] 4.14.0 in pyproject + `__init__` + CHANGELOG + TODO (hard gate; re-confirm free). Rename/replace the
      branch's ADR-016 + spec/plan to the A versions (this spec/plan supersede the B ones — `git rm` or
      overwrite the B spec/plan; verify ADR-016 still free).
- [ ] **ADR-016** must record (spec §§3.2/3.4/3.5): the integrity gap; the B-rejection measurement; why A
      beats B and the mode; the **alternatives analysis — skops / treelite / ONNX considered and why the
      numpy reconstruction was chosen** (consistency with the existing `_vectorized_leaf_indices`, no new
      runtime dep, sklearn-free + deterministic inference, parity-guaranteed; acknowledged tax = re-validate
      parity per sklearn bump + the phase-numeric concession); the phase-numeric decision framed as a
      **capability gap closed** (NOT an asserted active served-data bug, unless a characterization test
      proves it); the position/spread coherence note; the Hyrum flags (every served `ghost_gk_*` value
      changes + `predict()` public-API change + lakehouse re-materialize + density/spread shift).
- [ ] CHANGELOG 4.14.0: served value mode→boosted-mean (4.65 → **`<Task-12 measured>`** m, measured delta);
      `predict()` public-API semantic change; phase-numeric (+ KDE categorical capability-gap closed) +
      density/spread shift; lakehouse re-materialize; re-published weights. ADR-016.
- [ ] Model card: three-number table; backfill the **Task-12 measured** boosted-mean MAE (not the old 1.1 m).

## Task 14: Final review + single commit
- [ ] Full non-e2e suite + ruff + ruff format --check + pyright `silly_kicks/` green.
- [ ] `/final-review`; address findings.
- [ ] Present diff + the Task-12 measured number; on approval, ONE sentinel-gated commit; push; PR; wait
      CI green (don't poll); tag after main CI green.

## Self-review (spec → tasks)
§3.1 reconstruction → T1,T3; §3.2 phase-numeric → T2 (+no-categorical tests T1/T2); §3.3 serialize+re-publish
→ T7,T12; §3.4 served surface + drop central → T4,T5,T6; §7 tests → T1,T3,T6,T8,T10; §8 docs → T13. Parity
(≤1e-6) is the spine — landed in T1, depends on T2 (phase-numeric) + T3 (predict_mean) + T7 (load). Re-run
T1's test after T2/T3/T7.
