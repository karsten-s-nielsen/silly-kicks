# ADR-016: Ghost-GK served estimator — exact boosted HGBR mean, reconstructed pickle-free

| Field | Value |
|---|---|
| **Date** | 2026-06-04 |
| **Status** | Accepted (silly-kicks 4.14.0, PR-S83); amended 4.22.1 (physical-pitch clamp at the serving seam — see Amendment) |
| **Deciders** | Karsten Nielsen (maintainer), silly-kicks part-deux review sessions |

(ADR-015 is reserved by TF-17 PR-C for the `silly_kicks/_causal/` matching port — see the ADR-011
update + the TF-17 spec. ADR-017 is the time-base contract (PR #100). This ADR is 016.)

## Context

`compute_ghost_gk` served the **joint 2D mode** of the KDE density (`GhostGkDensity.mode_x/mode_y`,
grid argmax) into `ghost_gk_x/ghost_gk_y`, with a held-out euclidean MAE of **~4.65 m**. But the model
card / CV reported **~1.1 m**, measured on a *different* estimator — `predict_mean`, two sklearn
`HistGradientBoostingRegressor`s fit during `fit()`. Those regressors are transient: `save()` does not
serialize them (pickle-free policy), so after `load()` `predict_mean` raised `RuntimeError`. Production
always uses `load()`. **The published number and the served value never met** — the card was ~4× better
than what was actually served. A pre-existing integrity gap (predates PR-S81), surfaced by the PR-S81
re-fit gate.

## The rejected first attempt (Option B) and what the gate found

The first design (the 2026-06-03 spec, built and then rejected) served the **leaf-weighted conditional
mean** of the model's own density (`Σⱼ wⱼ·labelⱼ / Σⱼ wⱼ` from the leaf-match weights the KDE already
computes). It was attractive because the artifact already stores `training_leaves` + `training_gk_x/y`,
so it needed **no re-train and no re-publish** — a different read-out of state already on disk.

The DGX empirical gate (fold 0, 8000 eval, fft-cic backend, 887k-sample model) **refuted the core
hypothesis**:

| estimator | MAE | RMSE |
|---|---|---|
| **mode** (status quo served) | **4.65 m** | 5.87 |
| mean / B (`_central_estimate`) | 7.00 m | 7.80 |
| grid-mean / C | 7.01 | 7.80 |
| geom-median | 6.71 | 7.49 |

multimodal frac (mode↔mean gap > 4 m) = **50 %**. **All central tendencies (~6.7–7.0 m) LOSE to the
mode (4.65 m)** because the conditional GK-position density given game-state is broad + multimodal — the
mean sits in low-density valleys between plausible GK positions (exactly reviewer H1's risk). Serving B
would have **degraded** served quality (4.65 → 7.0 m). The validated 1.1 m was the **boosted sklearn HGBR
`predict_mean`** — a structurally different, far stronger estimator (gradient boosting directly minimizes
squared error against the full 26-feature interaction; B is a single-pass forest-kernel average over a
coarse leaf partition). The gate did its job pre-ship.

## Decision

Serve the **exact sklearn HGBR boosted `predict_mean`**, reconstructed **pickle-free and load-safe**.

### Reconstruction (pickle-free)

sklearn `HistGradientBoostingRegressor` (squared-error loss, identity link):
`predict(X) = _baseline_prediction + Σ_trees leaf_value(tree, X)` (the learning rate is baked into the
stored leaf `value`s). `_vectorized_leaf_values` is added as a sibling of the existing
`_vectorized_leaf_indices` KDE-traversal kernel: same numeric traversal, but it accumulates the reached
leaf's `value` per tree. `predict_mean(X) = [baseline_x + Σ gk_x-leaf-values, baseline_y + Σ
gk_y-leaf-values]`. `fit()` re-adds the **gk_y** regressor (dropped during B), stores both ensembles'
node arrays + each regressor's baseline, and keeps `training_leaves` + `training_gk_x/y` (the KDE/density
still needs them for `ghost_gk_density_spread` and the mode). `save()/load()` serialize the gk_y tree
arrays + baselines; `metadata.json` records `serve_estimator = "boosted_mean"` (version 1.2.0) and `load()`
fails closed on a conflicting tag (R3) **and** on pre-Option-A artifacts (missing gk_y trees → clear
"re-fit required" error). This is an artifact-format change, so **both `default` (wheel) and `full` (Hub)
weights were re-fit + re-published.**

### phase-numeric (load-bearing, spec §3.2)

`fit()` previously passed `categorical_features=[phase_idx]`; the bundled model had **24 categorical
split nodes** whose routing bitsets (`raw_left_cat_bitsets`) are **not serialized** (only `.nodes` is).
A numeric leaf-value reconstruction cannot route through them. **Decision: train `phase` (and all
features) numerically** (`categorical_features=None`) → every split is a `num_threshold` comparison and
`_vectorized_leaf_values` matches sklearn `.predict()` **exactly** (parity gate ≤ 1e-6; pre-validated at
1.155e-14 in a throwaway prototype, 8.88e-15 on NaN rows).

This also **closes a latent KDE capability gap** (stated conservatively, *not* as a proven active bug):
the numeric `_vectorized_leaf_indices` cannot route categorical-split nodes either, so for the old
bundled model the KDE leaf-match was **not guaranteed correct** on frames routed through those 24 nodes.
Whether any *served* frame actually mis-routed is unproven; phase-numeric eliminates the class entirely,
so the new model's KDE is correct by construction. `phase` has 3 ordered-ish values ({0 open, 1
set_piece, 2 goal_kick}); ordinal splits are nearly as expressive — negligible modeling change.
*Rejected alternative A2* (keep `phase` categorical + serialize bitsets + implement bitset routing in
both traversal kernels): more code + larger artifact, no quality benefit.

### Why the numpy reconstruction over skops / treelite / ONNX

The driving constraint is **pickle-free, secure, deterministic, sklearn-free serving** (the existing
`_vectorized_leaf_indices` KDE path already commits to this). Standard alternatives were considered:
- **skops** (the sklearn team's secure-sharing format — allowlist load, no arbitrary-code execution).
  Serves the exact model *including* categorical routing (no phase-numeric concession). **Rejected:** it
  reintroduces **sklearn at load/inference** — the ghost-GK inference path is deliberately sklearn-free +
  numpy-only — and still carries sklearn-version coupling.
- **treelite / ONNX** (compile the ensemble to a stable inference artifact). **Rejected:** a new
  heavyweight build/runtime dependency + a compiled-artifact lineage foreign to the repo's
  npz+JSON+SHA256 convention, for a small two-ensemble regressor.

**Chosen: numpy leaf-value reconstruction**, because it (a) extends the already-proven
`_vectorized_leaf_indices` pattern (same traversal, accumulate `value` instead of leaf index), (b) adds
no new runtime dependency and keeps inference sklearn-free + deterministic, (c) stays in the established
serialization lineage, and (d) is correctness-guaranteed by an independent sklearn-oracle parity test.

**Inference is sklearn-version-INDEPENDENT.** The reconstruction reads stored structured arrays
(`node["value"]`/`node["num_threshold"]`/…) whose dtype + field names are serialized in the npz and
preserved by `np.load`; inference imports no sklearn. sklearn is coupled **only at fit/extract time**
(`regressor._predictors`). So there is **no per-release runtime mis-predict risk** — only a
*re-fit-under-a-new-sklearn* risk, which the parity test on the fresh fit catches (blocking before
publish). The `load()` `sklearn_version` note is **informational provenance, not a correctness guard**.
The only standing tax is re-validating parity *when we re-fit* + the phase-numeric concession.

### Why the boosted mean escapes B's valley pathology

The boosted HGBR mean is *also* a squared-error-optimal conditional mean, so the natural worry is that it
too predicts between-modes points on multimodal `P(gk|state)`. It does not, because the §"gate found"
**50 % multimodal fraction is a property of B's coarse leaf-co-occurrence density** (pooling labels over
a crude leaf partition that mixes distinct game-states), **not** the true conditional. The boosted
regressor conditions on the **full 26-feature interaction** → a much sharper conditional → closer to
unimodal → its mean lands near a real mode. The ship gate **stratifies the boosted-mean MAE by the
per-frame multimodal flag** to confirm this empirically before shipping (if it were also poor on the
multimodal subset, this mechanism would be wrong and we would learn it pre-ship).

### Ship gate (pre-registered procedure, not a magic number)

Measured on the same fold/8000-eval/fft-cic split as the B table, **stratified** by the per-frame
multimodal flag (mode↔grid-mean gap > 4 m): report boosted-mean MAE **pooled + multimodal + unimodal**,
alongside mode + B on the same strata (MAE + RMSE). Pre-locked rails (data-independent):
- **HARD FAIL → STOP/debug:** boosted-mean MAE **≥ the mode** on *either* pooled OR multimodal (if
  pooled, suspect a reconstruction bug → re-check parity).
- **CLEAR PASS → ship:** boosted beats the mode by a clear margin on **both** pooled AND multimodal, AND
  the multimodal-subset MAE is not pathological (well below B's ~7 m).
- **Owner checkpoint (everything between):** present the full stratified table; the absolute "good
  enough" bar is a **VAEP-consumer requirement** (owner-set with the data, by how the ghost-GK-distance
  feature is consumed downstream), not pre-guessed. Decision + numbers recorded here / in the PR.

## Consequences

- **Served quality** moves from the ~4.65 m mode to the boosted-mean MAE **1.07 m** (5-fold aggregate on
  the clean re-fit; per-fold pooled 1.11/1.08/1.06/1.08/~1.0, multimodal ~1.18 vs the mode's ~3.8 and
  b_central's ~8.3 — the latter empirically confirming the "boosted escapes B's valley" mechanism); the
  card now reports what is served (integrity gap closed).
- **`ghost_gk_spread` → `ghost_gk_density_spread` (breaking column rename).** Through ≤4.12, position
  (mode) and spread came from the *same* density, so spread described dispersion around the served point.
  Now the served point is the **boosted mean** (a different location than the mode), while the spread
  remains the **conditional-density dispersion** — it is **NOT** the standard error of the served point.
  The rename makes "density dispersion, not the served point's SE" structural rather than a docstring
  people skip, at this PR's uniquely-low marginal cost (already breaking every value + re-materializing
  the lakehouse). The internal `GhostGkDensity.spread` *attribute* keeps its name.
- **Hyrum's Law.** Every served `ghost_gk_x/y` value changes (deliberate value change, not an API break);
  `model.predict()` is a **public-API semantic change** (callers expecting the mode now get the boosted
  mean — same as B's intent, different value); the bundled + Hub weights are **re-published** (format
  change; old artifacts fail closed on load); the density/`*_density_spread` shifts slightly from
  phase-numeric; the lakehouse must **re-materialize** `ghost_gk_*` and **rename the column** on consume.
  All flagged in the CHANGELOG. (No separate lakehouse review this cycle — the CHANGELOG/ADR are the
  cross-repo handshake.)
- **Cost-model inversion (note).** `predict_mean` is now ~free (leaf-value traversal); the
  `predict_density` fft-cic pass for `*_density_spread` is the only cost driver in `compute_ghost_gk`. A
  future optimization could skip the density when spread isn't requested — recorded, not done here.
- **TF-19 forecloses nothing** — the full density (incl. the mode) remains reachable via
  `predict_density`.

See ADR-013 (numba), ADR-014 (fft backend), ADR-011 (trained-model lifecycle).

## Amendment (4.22.1, 2026-06-11): physical-pitch clamp at the serving seam

`compute_ghost_gk` now clamps the served `ghost_gk_x/y` (goal-relative) to the physical pitch
(x ∈ [0, 105], y ∈ [0, 68]) with a warning (lakehouse 4.22.0 production report item 2: a corrupted
upstream `is_goalkeeper` flag wrong-footed the goal-side flip and the boosted regressor extrapolated
to 5.7 m *behind* the goal line — training labels are filtered to goal-relative x ∈ [0, 30], so such
values are pure out-of-distribution output on garbage input). The clamp lives at the **serving seam
in `compute_ghost_gk`**, deliberately NOT in `GhostGkModel.predict_mean` — the model-level
exact-boosted parity contract (and its blocking parity gate) is untouched. Clamp target is the
**physical pitch, not the trained grid domain**: healthy slight extrapolation past the 30 m label
filter (sweeper rushes, observed up to ~31.8 m on clean providers) stays byte-unchanged, so the
clamp only ever fires on physically-impossible (corrupt-input) rows.
