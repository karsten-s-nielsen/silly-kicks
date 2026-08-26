---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - goalkeeper
  - conditional-density-estimation
  - tracking-data
  - ghosting
pipeline_tag: tabular-regression
library_name: silly-kicks
---

# Ghost-GK v1 &mdash; Conditional Density Estimation for Goalkeeper Positioning

Predicts where a league-average goalkeeper would position themselves given the current game state. Uses RFCDE (Random Forest Conditional Density Estimation) over HistGradientBoostingRegressor leaf assignments with weighted 2D kernel density estimation.

Part of the [silly-kicks](https://github.com/karsten-s-nielsen/silly-kicks) soccer analytics library (GKDV research program, TF-18, Layer 2).

## Model Description

Standard goalkeeper evaluation metrics (xGOT, save percentage, goals prevented) measure what happens *after* a shot is taken. Ghost-GK addresses the upstream question: *given the current game state, where should the goalkeeper be standing?*

The model learns a league-average positional density from thousands of match frames across multiple tracking data providers. For any given frame, it outputs a full 2D probability distribution over the goal-relative region, not just a point estimate. This enables downstream metrics like the GK Deterrent Value (GKDV) &mdash; comparing the actual GK position against the ghost position to quantify positioning-as-deterrent.

Key properties:

- **Density estimation, not regression**: Outputs a 60&times;64 probability grid (3,840 cells at 0.5m resolution), not a single (x, y) point. Captures multimodal positioning (e.g., split between near-post and central when the ball is wide).
- **No pickle**: Serialized as npz (NumPy arrays) + JSON (metadata) + SHA-256 integrity sidecar. No pickle anywhere in the load/save path.
- **Vectorized inference**: Tree traversal uses NumPy array operations (no sklearn at inference time). Batch prediction of 1,000 frames completes in under 1 second.
- **Two variants**: `"default"` (approx. 1.0 MB, 36k-frame subsample) ships bundled in the wheel; `"full"` (approx. 2.4 MB, 179-match corpus) downloads from this Hub repo on first use. Both are **parameters-only** artifacts &mdash; the two trained tree ensembles and their baselines, with no per-sample training data (silly-kicks 4.54.0; ADR-044).

## Architecture

The model implements RFCDE (Pospisil &amp; Lee 2018) adapted for goalkeeper positioning:

1. **Feature extraction**: 26 goal-relative features per frame (ball state, defensive geometry, game context). `phase` is trained **numerically** (not categorical) so the pickle-free numeric tree traversal matches sklearn exactly (4.14.0; ADR-016).
2. **Leaf assignment**: `HistGradientBoostingRegressor` (500 trees, max depth 8) trained on GK x-coordinate; leaf assignments partition the feature space
3. **Co-occurrence weighting**: Training frames sharing leaf assignments with the query frame receive higher weight (Dutta et al. 2024 NFL Ghosts approach)
4. **2D KDE**: Weighted Gaussian KDE over (x, y) positions of weighted training frames produces the density surface (`mode`, `mean`). This step needs the per-sample training positions, so it runs only on a **locally fit** model &mdash; the distributed artifact is parameters-only and does not carry them (silly-kicks 4.54.0; ADR-044).
5. **Served point estimate**: a second `HistGradientBoostingRegressor` is trained on GK y-coordinate; `ghost_gk_x/y` serve the exact boosted mean of both ensembles, reconstructed pickle-free as `baseline + Σ_trees leaf_value` (no sklearn at inference). 4.14.0; ADR-016.

### Features (26)

| Category | Features |
|----------|----------|
| Ball state | `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `ball_distance_to_goal`, `ball_to_goal_angle`, `ball_speed` |
| Defensive geometry | `defensive_line_x`, `defensive_line_depth`, `defensive_line_width`, `defensive_line_speed`, `defenders_behind_ball`, `deepest_defender_x`, `defending_team_compactness`, `defending_centroid_vx` |
| Attacking geometry | `attackers_in_box`, `nearest_attacker_to_goal_x`, `attacker_centroid_x`, `attacker_centroid_y`, `ball_to_nearest_attacker_dist` |
| Game context | `phase`, `team_in_possession`, `score_diff`, `time_seconds`, `period_id`, `ball_in_own_half` |

All coordinates are goal-relative: the defending goal is at x=0, pitch center at y=34.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | `HistGradientBoostingRegressor` |
| Number of trees | 500 |
| Max depth | 8 |
| Grid resolution | 0.5m (60&times;64 cells) |
| Grid coverage | x: [0, 30]m from goal line, y: [18, 50]m across pitch |

## Variants

| Variant | Training corpus | File size | Source |
|---------|----------------|-----------|--------|
| `default` | 36k-frame subsample | ~1.0 MB | Bundled in `pip install silly-kicks` |
| `full` | 179 matches / ~1.04M frames | ~2.4 MB | Downloaded from this HF repo via `pip install silly-kicks[ghost-gk]` |

Both variants are **parameters-only** artifacts (no per-sample training data is stored or redistributed). They serve identical point-estimate machinery (`ghost_gk_x/y`); the `full` variant is trained on the larger corpus. The KDE **density** read-out (`predict_density`) is available only on a **locally fit** model &mdash; see Usage.

## Training Data

Trained on licensed tracking data from professional football matches:

| Provider | Competitions | Notes |
|----------|-------------|-------|
| Sportec (DFL) | Bundesliga | Native GK identification |
| SkillCorner | Multiple leagues | Derived GK identification (ADR-007) |
| Gradient Sports | FIFA World Cup 2022 | Owner-tier source — only the trained model weights are distributed here; the underlying raw tracking data is **not** redistributed |

The `full` variant is trained on 179 matches / ~1.04M frames across all three providers above (the SkillCorner cohort was expanded to include owner-tier matches — silly-kicks 4.51.0 / TF-19 PR-2); the `default` variant is a lighter 36k-frame subsample. Only the learned model parameters are published: the two gradient-boosted tree ensembles (split thresholds, feature indices, leaf values) and their additive baselines. **No per-sample training data and no raw provider tracking data is redistributed** (parameters-only artifact, silly-kicks 4.54.0 — the per-sample density arrays were removed; see ADR-044).

Training targets are restricted to a fixed goal-relative box — x &isin; [0, 30] m from the defended goal line, y &isin; [18, 50] m — a **purely geometric filter with no action or possession condition**. A keeper that has rushed far upfield (a sweeper action) falls outside this box and is naturally excluded, so the model represents normal in-goal positioning; the exclusion is geometric, not action-based.

**Label domain**: GK (x, y) position in goal-relative coordinates, filtered to the grid region [0, 30] &times; [18, 50].

## Usage

```python
import silly_kicks.tracking as tracking

# Default variant (bundled, works offline)
densities = tracking.compute_ghost_gk(frames, model="default")

# Full variant (downloads from HF Hub on first use)
densities = tracking.compute_ghost_gk(frames, model="full")

# Action-coupled aggregator for VAEP integration
actions = tracking.add_ghost_gk(actions, frames, model="full")

# Direct model loading — served POSITIONS (works on the distributed parameters-only artifact)
model = tracking.GhostGkModel.from_variant("full")
positions = tracking.compute_ghost_gk(frames, model=model, home_team_id=1)  # ghost_gk_x / ghost_gk_y

# The KDE density read-out (predict_density / GhostGkDensity) requires a LOCALLY FIT model:
# distributed artifacts are parameters-only and do not carry the per-sample data the density
# needs (silly-kicks 4.54.0, ADR-044). On a loaded artifact predict_density raises.
local = tracking.GhostGkModel(n_estimators=500).fit(features, labels)
density = local.predict_density(feature_vector)
print(f"Mode: ({density.mode_x:.1f}, {density.mode_y:.1f})")
print(f"Spread: {density.spread:.2f}")
```

### Output

Each prediction returns a `GhostGkDensity` frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `mode_x` | float | Joint 2D mode x (argmax), goal-relative meters |
| `mode_y` | float | Joint 2D mode y (argmax), goal-relative meters |
| `mean_x` | float | Density-weighted (grid) mean x |
| `mean_y` | float | Density-weighted (grid) mean y |
| `spread` | float | Effective area (entropy-based **density** dispersion measure) |
| `probabilities` | ndarray (60, 64) | Full density grid |
| `grid_x` | ndarray (60,) | X-axis cell centers |
| `grid_y` | ndarray (64,) | Y-axis cell centers |

The **served** point estimate (`ghost_gk_x/y`, `model.predict()`) is the exact boosted HGBR
`predict_mean` (below), reconstructed pickle-free — it is **not** a field of `GhostGkDensity`.

### Served point estimate (v4.14.0)

`ghost_gk_x/y` and `model.predict()` serve the **exact sklearn `HistGradientBoostingRegressor` boosted
mean** — the same estimator the old card's ≈1.1 m number measured, but now reconstructed **pickle-free**
(`baseline + Σ_trees leaf_value`) so it survives `load()` and is actually served. This closes a
pre-existing integrity gap: the card reported ≈1.1 m for an estimator that `save()`/`load()` discarded,
while production served the KDE **mode** (≈4.65 m):

| Estimator | Held-out euclidean MAE | Served? |
|-----------|------------------------|---------|
| old card number (`predict_mean`, sklearn, phase-categorical) | ≈1.1 m | never served (unavailable after `load()`) |
| KDE mode (≤ v4.12) | ≈4.65 m | served through 4.12 |
| **boosted mean (reconstructed pickle-free)** | **1.13 m** (current `full` re-fit, 5-fold aggregate; per-provider GS 1.08 / SkillCorner 1.18 / Sportec 1.67) | **served now** |

The 4.14.0 number is re-measured at re-fit on the same held-out split as the mode (not copied from the
old ≈1.1 m card, which was a *different*, phase-categorical model). An intermediate design that served
the leaf-weighted *conditional mean* (no re-fit) was empirically rejected — it measured ≈7.0 m, worse
than the mode, because the conditional density is broad + multimodal. The boosted mean is a structurally
stronger estimator and is the only candidate that beats the mode. The mode remains available via
`predict_density(...).mode_x/mode_y`. See ADR-016.

**Weights re-fit + re-published (4.14.0).** This is an artifact-format change (the npz now carries the
gk_y tree ensemble + baselines for the reconstruction; `serve_estimator = "boosted_mean"`), and `fit()`
now trains `phase` numerically (closing a latent KDE categorical-routing capability gap). Both the
bundled `default` and this Hub `full` model are re-fit; old-format artifacts fail closed on load with a
clear "re-fit required" error.

> **Parameters-only (4.54.0; ADR-044).** The artifact format is now `metadata.version = 1.3.0`
> (`stores_training_data = false`): the per-sample density arrays (`training_gk_x/y`, `training_leaves`)
> are no longer stored or distributed. The served point estimate (`ghost_gk_x/y` via `predict_mean`) is
> byte-identical. The emitted `ghost_gk_density_spread` column (renamed from `ghost_gk_spread` in 4.14.0)
> is **retired** from `compute_ghost_gk` / `add_ghost_gk` / `ghost_gk_xfns`; the KDE density read-out now
> requires a locally fit model.

### Serialization Format

```
model_dir/
  rfcde_weights.npz    # NumPy arrays: gk_x + gk_y gradient-boosted tree nodes + additive baselines (parameters only; no per-sample data)
  metadata.json        # Feature names, grid spec, hyperparameters, version, corpus provenance, chirality fingerprint
  SHA256SUMS           # Integrity checksums (CRLF-normalized for cross-platform safety)
```

No pickle is used anywhere in the serialization or deserialization path.

## Coordinate System

Input frames must be in **LTR-normalized convention** (home team attacks right in all periods &mdash; the standard silly-kicks tracking output after `play_left_to_right` normalization).

Features are extracted in **goal-relative coordinates**:
- Origin: defending goal center (x=0, y=34)
- The defending goal is inferred per (game_id, period_id, team_id) from mean GK x position

## Limitations

- **League-average ghost**: The model predicts where an *average* goalkeeper would stand, not where a *specific* goalkeeper would stand. Stylistic differences (sweeper-keeper vs. line-keeper) are averaged out.
- **No shot-stopping ability**: Ghost-GK models positioning, not reactions. It does not predict save probability or diving reach.
- **Tracking data quality**: Predictions inherit noise from the underlying tracking system. SkillCorner broadcast-derived coordinates are noisier than optical systems (Sportec DFL).
- **LTR normalization required**: Input frames must be LTR-normalized. Feeding raw provider coordinates produces incorrect goal-relative features.
- **Static density**: Each frame produces an independent density estimate. Temporal smoothing is not built into the model (apply externally if needed).

## References

```bibtex
@inproceedings{le2017ghosting,
  title={Data-Driven Ghosting Using Deep Imitation Learning},
  author={Le, Hoang M. and Yue, Yisong and Carr, Peter and Lucey, Patrick},
  booktitle={MIT Sloan Sports Analytics Conference},
  year={2017}
}
```

```bibtex
@article{dutta2024nflghosts,
  title={NFL Ghosts: A framework for evaluating defender positioning
         with conditional density estimation},
  author={Dutta, Rishav and Yurko, Ronald and Ventura, Samuel},
  journal={arXiv preprint arXiv:2406.17220},
  year={2024}
}
```

```bibtex
@article{pospisil2018rfcde,
  title={RFCDE: Random Forests for Conditional Density Estimation},
  author={Pospisil, Taylor and Lee, Ann B.},
  journal={arXiv preprint arXiv:1804.05753},
  year={2018}
}
```

```bibtex
@software{nielsen2026ghostgk,
  title={Ghost-GK: Conditional Density Estimation for Goalkeeper Positioning},
  author={Nielsen, Karsten Skyt},
  year={2026},
  url={https://github.com/karsten-s-nielsen/silly-kicks}
}
```

## Model Files

| File | Size | Description |
|------|------|-------------|
| `rfcde_weights.npz` | ~2.4 MB | gk_x + gk_y gradient-boosted tree nodes + additive baselines (parameters only) |
| `metadata.json` | ~2 KB | Feature names, grid spec, hyperparameters, `serve_estimator`, version, corpus provenance, chirality fingerprint |
| `SHA256SUMS` | 164 B | Integrity checksums |

## More Information

- **License**: [MIT](https://opensource.org/licenses/MIT) (same as silly-kicks)
- **Library**: [silly-kicks](https://pypi.org/project/silly-kicks/) (v3.24.0+)
- **Documentation**: [silly-kicks GitHub](https://github.com/karsten-s-nielsen/silly-kicks)
- **Research program**: GKDV (GK Deterrent Value) &mdash; TF-15 through TF-19
