# Ghost-GK Served Estimator — Option A (exact boosted `predict_mean`, pickle-free) — Design

**Date:** 2026-06-04
**Status:** Approved direction (owner: Option A) — design pending review
**Supersedes:** `2026-06-03-ghost-gk-serve-mean-design.md` (Option B — empirically rejected)
**Feature track:** TF-18 (Ghost-GK) serve-correctness
**Release:** silly-kicks **4.14.0** (4.12.0 shipped by part-deux PR #100). ADR-016.

---

## 1. Why this supersedes Option B

The integrity gap: `compute_ghost_gk` serves the KDE **mode** (~4.65 m held-out euclidean MAE) while the
model card reported **~1.1 m** for the sklearn `predict_mean` that `save()`/`load()` discard (it raised
`RuntimeError` after `load()` — **never served**).

Option B (serve the leaf-weighted **conditional mean**, pickle-free from stored leaves+labels, no
re-publish) was built and then **empirically rejected** by the DGX gate (fold 0, 8000 eval, fft-cic,
887k-sample model):

| estimator | MAE | RMSE |
|---|---|---|
| **mode** (status quo) | **4.65 m** | 5.87 |
| mean / B | 7.00 m | 7.80 |
| grid-mean / C | 7.01 | 7.80 |
| geom-median | 6.71 | 7.49 |

multimodal frac (gap > 4 m) = **50 %**. All central tendencies (~6.7–7.0 m) **lose to the mode** — the
conditional GK-position density given game-state is broad + multimodal, so the mean sits in low-density
valleys. The validated 1.1 m is the **boosted sklearn HGBR** `predict_mean` — a structurally different,
far stronger estimator (gradient boosting directly minimizes squared error). **Serving B would degrade
quality (4.65 → 7.0 m).** Owner decision: **Option A now** (long-term quality over a fast ship).

## 2. Goal / Non-goals

**Goal:** `predict_mean` (and therefore `compute_ghost_gk` / `predict()`) serves the **exact sklearn HGBR
boosted prediction**, reconstructed **pickle-free and load-safe** from serialized tree node arrays — closing
the integrity gap with a genuine quality improvement over the served mode.

**On the "~1.1 m":** that figure is the *old phase-categorical* model card's `predict_mean` number — it is
**EXPECTED, not established** for this PR. It was NOT measured for the phase-numeric re-fit (§3.2) and was
NOT produced on the same held-out split as the §1 B-rejection table (mode 4.65 / B 7.0). The shippable
number is the **Task-12 re-measurement of boosted-mean on the *identical* split as the B table** (so mode /
B / boosted are all comparable). Every "1.1 m" in this spec is "expected, pending Task-12"; the CHANGELOG +
model-card number is **copied from the Task-12 measurement, from nowhere else**. Ship gate: §3.5.

**Non-goals:** building TF-19; changing the KDE math beyond the phase-numeric consequence; the geom-median
(measured-only reference, not served).

## 3. Decision

### 3.1 Reconstruct the boosted prediction pickle-free

sklearn `HistGradientBoostingRegressor` (squared-error loss, identity link):

```
predict(X) = _baseline_prediction + Σ_trees  leaf_value(tree, X)
```

The learning-rate is **baked into** the stored leaf `value`s. The node arrays (already serialized for the
gk_x KDE partition; `value` field confirmed present) carry everything except the per-regressor
`_baseline_prediction`. So:
- Add **`_vectorized_leaf_values(nodes_list, X) -> (n_samples,)`** — a sibling of `_vectorized_leaf_indices`
  that, instead of returning the reached leaf *index*, **sums the reached leaf's `value`** across all trees.
- `predict_mean(X)` returns `[baseline_x + Σ gk_x-leaf-values, baseline_y + Σ gk_y-leaf-values]`.
- **Deterministic, pickle-free, load-safe** (stored state only), and **cheap** — pure leaf traversal, **no
  leaf-match, no grid KDE** (so the A measurement is fast; no chunking needed, unlike B/predict_density).

### 3.2 Train `phase` as **numeric** (eliminates categorical splits) — load-bearing

`fit()` currently passes `categorical_features=[phase_idx]`; the bundled model has **24 categorical split
nodes** whose routing bitsets (`raw_left_cat_bitsets`) are **not serialized** (only `.nodes` is). Numeric
leaf traversal cannot route them, so §3.1's reconstruction could not match sklearn through those nodes.

**Decision: drop `categorical_features` — train `phase` (and all features) numerically.** Then every split
is a `num_threshold` comparison and `_vectorized_leaf_values` matches sklearn `.predict()` **exactly** (the
parity gate, §7). Benefits:
- No bitset serialization, no categorical routing code.
- **Removes a correctness *capability* gap (stated conservatively, not as a proven active bug):** the
  numeric `_vectorized_leaf_indices` **cannot** route categorical-split nodes (it lacks the bitsets), so for
  the current bundled model the KDE leaf-match is **not guaranteed correct** on frames routed through those
  24 nodes. Whether any *served* frames actually mis-routed through them is **unproven** (would need a
  characterization test on the current model). phase-numeric eliminates the class entirely, so the new
  model's KDE is guaranteed correct by construction. **The ADR/CHANGELOG must state this as a capability gap
  closed**, not as "the ≤4.12 served outputs were wrong" (unless a characterization test proves the latter).
- `phase` has 3 ordered-ish values ({0 open, 1 set_piece, 2 goal_kick}); ordinal splits are nearly as
  expressive (two thresholds isolate any single level). Negligible modeling change.

**Cost:** the re-fit gk_x trees → the KDE partition → the density/`spread` (and mode) shift slightly vs the
current bundled model. We re-fit + re-publish anyway, and the served value is now the boosted mean (not the
mode), so the mode shift is moot. Documented in the CHANGELOG.

*(Rejected alternative A2: keep `phase` categorical + serialize `raw_left_cat_bitsets` + implement bitset
routing in both `_vectorized_leaf_values` and `_vectorized_leaf_indices`. More code + a larger artifact, no
quality benefit — phase-numeric is strictly simpler and fixes the latent KDE bug.)*

### 3.3 Artifact format change → re-fit + re-publish

`fit()` re-adds the **gk_y** HGBR (dropped during B) and stores, in addition to today's gk_x trees +
`training_leaves` + `training_gk_x/y` (kept — the KDE/density still needs them):
- `gk_y` tree node arrays (with `value`),
- `baseline_x` (gk_x regressor `_baseline_prediction`), `baseline_y`.

This is an artifact-format change, so **both bundled `default` (wheel) and Hub `full` weights must be
re-fit, re-saved, and re-published**. `metadata.json` gains `serve_estimator = "boosted_mean"` (R3: `load()`
fails closed on mismatch; absent → default for any legacy artifact) and a model-version bump (1.2.0).

### 3.4 Served surface

- `compute_ghost_gk` / `predict()` serve `predict_mean` (the boosted mean). `ghost_gk_x/y` carry it.
- `predict_density` is **retained** for `ghost_gk_spread` and the **mode** (still reachable via
  `predict_density(...).mode_x/mode_y`); it is now categorical-bug-free (§3.2). `compute_ghost_gk` calls
  `predict_density` (spread) + `predict_mean` (position) — two passes, but `predict_mean` is cheap.
- **Drop** `GhostGkDensity.central_x/central_y` (the Option-B leaf-weighted-mean fields) — A's served
  estimate is the boosted regressor, **not** a density read-out, so the single-pass central fields no
  longer apply.
- **Position/spread coherence note (Hyrum).** Through ≤4.12, `ghost_gk_x/y` (mode) and `ghost_gk_spread`
  came from the *same* density, so spread described dispersion *around the served point*. Now the served
  point is the **boosted mean** (a different location than the mode), while `ghost_gk_spread` remains the
  **conditional-density dispersion** — it is NOT the standard error of the boosted point estimate. Document
  this in the `compute_ghost_gk` docstring + ADR so consumers don't read spread as the served point's
  uncertainty.
  - **RENAME (owner-decided #8): `ghost_gk_spread` → `ghost_gk_density_spread`.** Makes "density dispersion,
    not the served point's SE" *structural* (un-misreadable) rather than a docstring people skip — at this
    PR's uniquely-low marginal cost (already breaking every value + re-materializing the lakehouse).
    **Scope of the rename (emitted column only):** `compute_ghost_gk`, `add_ghost_gk`, the `ghost_gk_xfns`
    column list, the atomic mirror, the backcompat golden's columns, all tests referencing `ghost_gk_spread`,
    and the **lakehouse column** (cross-repo — no separate lakehouse review this cycle; the CHANGELOG/ADR
    flag the breaking rename loudly so the lakehouse side updates on consume). The internal
    `GhostGkDensity.spread` *attribute* keeps its name (it is unambiguous in context). CHANGELOG + ADR flag
    the column rename as a (deliberate) breaking change.
- **Cost-model inversion (note, not a change).** `predict_mean` is now ~free (leaf-value traversal); the
  `predict_density` fft-cic pass for `spread` is the *only* cost driver in `compute_ghost_gk`. A future
  optimization could skip the density when `spread` isn't requested — recorded, not done here.

### 3.5 Why the numpy reconstruction over skops / treelite / ONNX (long-term lens)

The driving constraint is **pickle-free, secure, deterministic serving**. Standard alternatives to a
hand-rolled traversal were considered:
- **skops** (the sklearn team's secure-sharing format — allowlist-based load, no arbitrary-code execution,
  version-aware). Serves the *exact* model including categorical routing (no phase-numeric concession) and
  needs no hand-rolled traversal to re-validate per sklearn bump. **Rejected because** it reintroduces
  **sklearn at load/inference** — the ghost-GK inference path is deliberately **sklearn-free + numpy-only +
  numba-free-bare-import** (the existing `_vectorized_leaf_indices` KDE traversal already commits to this).
  skops would also still carry sklearn-version coupling.
- **treelite / ONNX** (compile the ensemble to a stable inference artifact). Rejected: a new heavyweight
  build/runtime dependency + a compiled-artifact lineage foreign to the repo's npz+JSON+SHA256 convention,
  for a small two-ensemble regressor.

**Chosen: numpy leaf-value reconstruction**, because it (a) **extends the already-proven
`_vectorized_leaf_indices` pattern** (same traversal, accumulate `value` instead of leaf index — the KDE
path already relies on raw-X numeric routing matching sklearn), (b) adds **no new runtime dependency** and
keeps inference sklearn-free + deterministic, (c) stays in the established serialization lineage, and (d) is
**correctness-guaranteed by the parity test** (§7) — an independent sklearn oracle.

**Inference is sklearn-version-INDEPENDENT (corrects the "tax" framing, #4).** The reconstruction reads
stored structured arrays (`node["value"]`/`node["num_threshold"]`/…) whose dtype + field names are
serialized in the npz and preserved by `np.load` — inference imports no sklearn and is unaffected by the
runtime sklearn version. sklearn is coupled **only at fit/extract time** (`regressor._predictors`). So there
is **no per-release runtime mis-predict risk** — only a *re-fit-under-a-new-sklearn* risk, which the parity
test on the fresh fit catches (Task 12, **blocking** before publish). The `load()` `sklearn_version` note is
**informational provenance, not a correctness guard**. This version-independence is a point in favour of the
numpy reconstruction, not against it. The only standing tax is re-validating parity *when we re-fit* + the
phase-numeric concession (§3.2). *(This whole analysis must appear in ADR-016.)*

**Why the boosted mean should escape B's valley pathology (#2 — confirm empirically via §3.6 stratification).**
The boosted HGBR mean is *also* a squared-error-optimal conditional mean, so the natural worry is that it too
predicts between-modes points on multimodal `P(gk|state)`. The likely reason it does **not** fail like B: the
§1 "50% multimodal" fraction is a property of **B's coarse leaf-co-occurrence density** (pooling labels over a
crude leaf partition that mixes distinct game-states), **not** necessarily the true conditional. The boosted
regressor conditions on the **full 26-feature interaction** → a much sharper conditional → closer to unimodal
→ its mean lands near a real mode. If so, B's 7.0 m reflects B's bad *density*, not irreducible
multimodality. **§3.6's stratified measurement is what confirms this** — if the boosted mean is also poor on
the multimodal subset, this mechanism is wrong and we learn it *before* shipping.

### 3.6 Ship gate / abort rule (Task 12)

**We pre-register the decision *procedure* + the *relative* criteria — NOT an arbitrary absolute number**
(owner decision #9, resolved). Rationale: boosted-mean's distribution is unknown until Task 12, so locking a
magic absolute bar now is false precision; and a single *pooled* scalar is exactly the aggregate that hid
B's failure. But the criteria are locked *before* measuring (anti-goalpost-moving). Measure on the same
fold/8000-eval/fft-cic split as the §1 B table AND **stratify by the per-frame multimodal flag** (#1) —
report **boosted-mean MAE pooled + multimodal-subset (gap > 4 m) + unimodal-subset**, alongside mode + B on
the same strata (MAE + RMSE).

**Pre-locked rails (data-independent, no judgment):**
- **HARD FAIL → STOP/debug:** boosted-mean MAE **≥ the mode** on *either* the pooled OR the multimodal
  subset. It must beat the status quo on the frames that matter; if it fails pooled, suspect a
  reconstruction bug (re-check parity).
- **CLEAR PASS → ship:** boosted beats the mode by a clear margin on **both** pooled AND the multimodal
  subset, AND the multimodal-subset MAE is **not pathological** (well below B's ~7 m).

**Owner checkpoint (everything in between):** present the full stratified table to the owner, who decides
ship / iterate / escalate **with the data in hand**. Pre-committed so it cannot be cherry-picked post-hoc:
(a) the **multimodal-subset** number is a first-class input — a good pooled MAE does NOT by itself pass; (b)
the absolute "is X m good enough" bar is a **VAEP-consumer requirement** (what MAE makes the
ghost-GK-distance feature useful) — **owner-set at the checkpoint, informed by how the ghost-GK-distance
feature is consumed downstream**, not a number silly-kicks pre-guesses. The decision + numbers are recorded
in ADR-016 / the PR.

## 4. What carries over from the Option-B work (already in the uncommitted tree)

Reuse: `predict()` realigned to serve `predict_mean` (M1); `compute_ghost_gk` serving `predict_mean` (not
mode); `serve_estimator` metadata record + fail-closed `load()` (R3); the `scripts/publish_ghost_gk.py`
ruff-ignore cleanup; the model-card three-number framing; the chunked measurement script
(`scripts/measure_ghost_gk_estimators.py`); the module-pollution-robust compute test pattern.

Replace: `_central_estimate` → `_vectorized_leaf_values` + boosted `predict_mean`; `fit()` re-adds gk_y +
drops `categorical_features` + serializes trees/baselines; `save()/load()` gain gk_y trees + baselines; drop
`GhostGkDensity.central_*`; rewrite the B spec/plan; regenerate the backcompat golden to the A output.

## 5. Architecture / Components (`silly_kicks/tracking/_ghost_gk.py`)

1. **`_vectorized_leaf_values(nodes_list, X)`** — mirror `_vectorized_leaf_indices` but accumulate the
   reached leaf's `value` per tree → `(n_samples,)`. Same NaN/`missing_go_to_left` handling; numeric
   thresholds only (valid after §3.2). **Post-loop convergence guard (#7, `raise` not silent):** after the
   depth-bounded loop, `if not np.all(nodes[current]["left"] == 0): raise RuntimeError("leaf traversal did
   not converge")` — a tree deeper than the cap would otherwise read an internal node's `value` (garbage)
   silently. (HGBR `max_leaf_nodes=31` → depth ≤ ~30, can't hit a 100-cap; cheap insurance in a load-bearing
   kernel. Confirm the bundled model's max depth < 100.)
2. **`fit()`** — pass **`categorical_features=None`** (not "from_dtype" — explicit, all numeric); train
   gk_x + gk_y HGBR; store `_tree_nodes` (gk_x), `_tree_nodes_y`, `_baseline_x`, `_baseline_y`; keep
   `training_leaves` (gk_x) + `training_gk_x/y`.
   - **numpy-2 safe baseline (#6):** `_baseline_x = float(regressor._baseline_prediction.item())` (it is
     shape (1,1); bare `float(ndarray)` warns/raises under numpy≥2).
   - **fail-fast on sklearn private attrs (#8/#5 — `raise`, NOT `assert`):** after `fit`,
     `if not hasattr(regressor, "_predictors") or regressor._baseline_prediction.size != 1: raise
     RuntimeError(...)` (a future sklearn rename fails loud — **`assert` is stripped by `python -O`**, which
     would silently reintroduce the risk the guard prevents). The parity test (§7) is the real correctness
     guard. `load()` gets a **definite (not conditional)** `sklearn_version`-mismatch warning — but it is
     **informational provenance, not a correctness guard** (inference is sklearn-version-independent, §3.5).
3. **`predict_mean(X)`** — **reindex to the canonical feature order first** (Hyrum guard, #3):
   `Xv = features[GHOST_GK_FEATURE_NAMES].values.astype(np.float64)` (or assert `list(features.columns) ==
   GHOST_GK_FEATURE_NAMES`), because the reconstruction indexes `X[:, feature_idx]` by fit-time column
   position — a reordered DataFrame would silently mis-predict. Then `[_baseline_x +
   _vectorized_leaf_values(_tree_nodes, Xv), _baseline_y + _vectorized_leaf_values(_tree_nodes_y, Xv)]`.
   Load-safe. (The same positional assumption exists in the `predict_density` / KDE `features.values` path —
   apply the same reindex/guard there.)
4. **`predict()`** — returns `predict_mean` (unchanged from B).
5. **`predict_density`** — drop the `central_*` population (and the `GhostGkDensity` fields); otherwise
   unchanged (KDE on gk_x leaf-match + labels; now categorical-bug-free).
6. **`compute_ghost_gk`** — `ghost_gk_x/y` from `predict_mean`; `ghost_gk_spread` from `predict_density`.
7. **`save()/load()`** — serialize `tree_nodes_y_*` + `baseline_x` + `baseline_y`; `serve_estimator`;
   version 1.2.0; SHA256SUMS. `load()` reconstructs both ensembles + baselines.
8. **`SERVED_ESTIMATOR = "boosted_mean"`**.

Atomic mirror unaffected (re-exports propagate).

## 6. Re-fit + re-publish (DGX box)

Box has the feature cache (`~/Development/ghost_gk_refit/measure_run/ghost_gk_v1/_feature_cache`, 887k
samples, 4.7.0 carrier, fps 1.0) + venv. After the code lands: scp `_ghost_gk.py` + `train_ghost_gk.py`,
run the trainer to **re-fit both variants** (default = subsample-cap; full = all 887k) → new-format
artifacts. Verify load + parity on box. Publish Hub `full`; bundle `default` in the wheel. See
`project_pr_s81_ghost_gk_refit` for the Hub recipe; `feedback_hatch_sdist_exclude_separate_from_wheel`
(full → Hub not git; check sdist size).

## 7. Testing

- **Parity (THE gate):** fit a small model on varied synthetic features (incl. `phase` values), assert
  `predict_mean(X)` == the live sklearn regressors' `.predict(X)` stacked, to **≤ 1e-6**. This proves the
  numpy reconstruction is exact (and that phase-numeric removed categorical splits). **Inject NaN (#4/#6)**
  into a few rows of columns **selected from `GHOST_GK_FEATURE_NAMES`** (e.g. the first 4 non-`phase`
  features — never hardcode names that might not exist → a 27th column / silent no-op), in **both train and
  test** X, so HGBR actually learns `missing_go_to_left` and the reconstruction's missing branch is
  parity-checked against sklearn, not asserted-by-construction.
- **Bundled-artifact smoke test (#3 — the real e2e):** load the **shipped bundled `default`** and assert
  `predict_mean` on a small fixture returns finite coords within pitch bounds (+ `_tree_nodes_y` present,
  baselines restored). The synthetic-model tests prove the *code*; only this proves the *re-published
  weights* serialized + reconstruct correctly. **RED until Task 12 re-bundles the new-format `default`**
  (like the parity test is RED until T7).
- **`load()` fail-closed on pre-Option-A artifacts:** an old-format artifact lacks `_tree_nodes_y` +
  baselines → `predict_mean` can't reconstruct. `load()` must raise a clear "artifact predates Option A —
  re-fit required" error (not a cryptic KeyError). **Consequence:** every test that loads the bundled
  `default` (`from_variant`) is RED from the code change until Task 12 re-bundles — develop Tasks 1–11 with
  **fresh-model** tests; the bundled-default suite + smoke test go GREEN after Task 12 (same window as a
  PR-S81-style format change).
- **Load-safe + fit==load parity:** `predict_mean` works after `load()` and is bit-identical pre/post save.
- **`predict()` == `ghost_gk_x/y`** (served column); mode reachable via `predict_density`.
- **`compute_ghost_gk` serves boosted mean, not mode** (module-identity-robust: build model+compute from
  the current `_ghost_gk` module, fresh, no `_fitted_model` cache — see the B pollution lesson).
- **No categorical nodes** in a freshly-fit model (guards §3.2): assert `is_categorical.sum() == 0`.
- **metadata serve_estimator** record + fail-loud (absent→default, conflict→raise).
- **Regenerate** `ghost_gk_backward_compat.parquet` to the A output (review the diff — load-bearing).
- KDE golden: will shift (phase-numeric changes the partition) → **regenerate** + note it’s the
  categorical-fix consequence (not a perturbation bug).
- Use varied/random features for any `.fit()` (HGBR constant-feature crash lesson).
- Full non-e2e suite + ruff + ruff format --check + pyright `silly_kicks/` green.

**What the goldens do and don't prove (e2e honesty).** The backcompat parquet (Task 10) is regenerated from
the new code → it is a **change-detector, not a correctness oracle** (golden = its own output; it cannot
catch a bug in A's output). Correctness rests on (1) the **parity-vs-sklearn test in CI** (independent
oracle) and (2) the **DGX MAE (Task 12)**. The **quality claim is inherently un-CI-able** (private DGX
cache) — quality evidence lives in the **ADR + model card from Task 12**, not in the test suite. State this
explicitly so no one mistakes the regenerated golden for validation.

## 8. Versioning / docs

- 4.14.0 across `pyproject.toml` + `silly_kicks/__init__.py` + `CHANGELOG.md` + `TODO.md` (hard gate;
  re-confirm free at tag time — part-deux shares the line).
- **ADR-016** (Ghost-GK served estimator = exact boosted HGBR mean): the integrity gap, the B-rejection
  measurement, why A (the boosted mean is the only estimator beating the mode), the phase-numeric decision
  + latent-KDE-bug fix, the re-fit/re-publish, the Hyrum flags (every served `ghost_gk_*` value changes +
  `predict()` public-API semantic change + lakehouse re-materialize + slight density/spread shift).
- **Model card:** three-number table — old card 1.1 m (phase-categorical, *never served*) / mode 4.65 m
  (served ≤4.12) / **boosted mean 4.14.0 = `<Task-12 measured>` m, served**. The 4.14.0 number is copied
  from the Task-12 re-measurement only (not the old card's 1.1 m). NOTICE: no new citation (existing
  RFCDE/HGBR).

## 9. Risks

- **Parity through categorical handling** — mitigated by §3.2 (phase-numeric removes all categorical
  splits); the parity test + the "no categorical nodes" test are the guards. If for some reason categoricals
  must stay, fall back to A2 (serialize bitsets) — but the spec assumes phase-numeric.
- **Re-publish discipline** — sdist size / Hub for `full` (see the hatch lesson).
- **Density/spread shift** from phase-numeric — expected + documented; the served value (boosted mean) is
  unaffected by the partition for its own trees (gk_x/gk_y boosted predictions are what's served).
- **Version/ADR collision** with part-deux — re-confirm 4.14.0 + ADR-016 free at finalize.
