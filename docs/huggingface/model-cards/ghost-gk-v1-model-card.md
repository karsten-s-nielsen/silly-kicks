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
- **Two variants**: `"default"` (approx. 9 MB, 36k training frames) ships bundled in the wheel; `"full"` (approx. 91 MB, 537k frames) downloads from this Hub repo on first use.

## Architecture

The model implements RFCDE (Pospisil &amp; Lee 2018) adapted for goalkeeper positioning:

1. **Feature extraction**: 26 goal-relative features per frame (ball state, defensive geometry, game context)
2. **Leaf assignment**: `HistGradientBoostingRegressor` (500 trees, max depth 8) trained on GK x-coordinate; leaf assignments partition the feature space
3. **Co-occurrence weighting**: Training frames sharing leaf assignments with the query frame receive higher weight (Dutta et al. 2024 NFL Ghosts approach)
4. **2D KDE**: Weighted Gaussian KDE over (x, y) positions of weighted training frames produces the density surface

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

| Variant | Training frames | File size | Source |
|---------|----------------|-----------|--------|
| `default` | 36,000 | 9 MB | Bundled in `pip install silly-kicks` |
| `full` | 537,000 | 91 MB | Downloaded from this HF repo via `pip install silly-kicks[ghost-gk]` |

The `default` variant provides nearly identical point-estimate accuracy (mode x/y) with faster density estimation. The `full` variant produces smoother, more detailed density surfaces &mdash; recommended for research applications where the full density shape matters.

## Training Data

Trained on licensed tracking data from professional football matches:

| Provider | Competitions | Notes |
|----------|-------------|-------|
| Sportec (DFL) | Bundesliga | Native GK identification |
| SkillCorner | Multiple leagues | Derived GK identification (ADR-007) |

Training frames are filtered to remove sweeper-rush events (GK outside penalty area during active defensive actions) to ensure the ghost represents normal positioning behavior.

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

# Direct model loading
model = tracking.GhostGkModel.from_variant("full")
density = model.predict_density(feature_vector)
print(f"Mode: ({density.mode_x:.1f}, {density.mode_y:.1f})")
print(f"Spread: {density.spread:.2f}")
```

### Output

Each prediction returns a `GhostGkDensity` frozen dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `mode_x` | float | Joint 2D mode x (argmax), goal-relative meters |
| `mode_y` | float | Joint 2D mode y (argmax), goal-relative meters |
| `mean_x` | float | Density-weighted mean x |
| `mean_y` | float | Density-weighted mean y |
| `spread` | float | Effective area (entropy-based dispersion measure) |
| `probabilities` | ndarray (60, 64) | Full density grid |
| `grid_x` | ndarray (60,) | X-axis cell centers |
| `grid_y` | ndarray (64,) | Y-axis cell centers |

### Serialization Format

```
model_dir/
  rfcde_weights.npz    # NumPy arrays: leaf assignments, training positions, tree structure
  metadata.json        # Feature names, grid spec, hyperparameters, version
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
| `rfcde_weights.npz` | 91 MB | Tree structure, leaf assignments, training GK positions |
| `metadata.json` | 1 KB | Feature names, grid specification, hyperparameters |
| `SHA256SUMS` | 166 B | Integrity checksums |

## More Information

- **License**: [MIT](https://opensource.org/licenses/MIT) (same as silly-kicks)
- **Library**: [silly-kicks](https://pypi.org/project/silly-kicks/) (v3.24.0+)
- **Documentation**: [silly-kicks GitHub](https://github.com/karsten-s-nielsen/silly-kicks)
- **Research program**: GKDV (GK Deterrent Value) &mdash; TF-15 through TF-19
