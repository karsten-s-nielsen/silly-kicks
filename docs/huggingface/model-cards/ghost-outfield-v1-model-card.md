---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - tracking-data
  - ghosting
  - rest-defense
pipeline_tag: tabular-regression
library_name: silly-kicks
---

# Ghost-Outfield v1 &mdash; League-Average Rearguard Positioning (Rest-Defense)

Predicts where a league-average outfield **rearguard** defender would position themselves, per
tracking frame, team, and lateral slot. The outfield sibling of
[Ghost-GK v1](https://huggingface.co/silly-kicks/ghost-gk-v1): a point estimate (`ghost_gr_x/y`,
goal-relative) served as the exact pickle-free boosted `HistGradientBoostingRegressor` mean.

Part of the [silly-kicks](https://github.com/karsten-s-nielsen/silly-kicks) soccer analytics library
(TF-60 rest-defense; ADR-087). It is the league-average baseline the rest-defense outfield
counterfactual arm differences against ("how far does the actual rest-defense shape sit from average?").

## What it models

For an in-possession team's **rest defense** (the rearguard it keeps while attacking, to blunt the
counter after a loss), the model predicts the league-average position of each of the deepest-`n`
rearguard defenders, ranked left-to-right by a lateral `slot_index`. It is **possession-conditioned**:
one model serves both regimes via a live `team_in_possession` feature &mdash; the ball-carrier's
rest-defense rearguard *and* the defending line facing an attack.

The feature vector is **leakage-safe by construction**: no input encodes the modeled team's own
rearguard coordinates (the prediction target). It is ball state + opponent counter-threat geometry +
game context + the lateral slot rank.

## Variants

| Variant | Repo | Features | Use |
|---|---|---|---|
| `default` (this repo) | `silly-kicks/ghost-outfield-v1` | 20 (velocity-bearing) | Continuous tracking (Sportec, SkillCorner, Gradient Sports) |
| `position_only` | [`silly-kicks/ghost-outfield-position-only-v1`](https://huggingface.co/silly-kicks/ghost-outfield-position-only-v1) | 16 (4 velocity features dropped) | Velocity-less StatsBomb-360 freeze-frames |

The serve seam auto-selects the variant from the frames' declared velocity availability; a
velocity-less freeze-frame with no bundled `position_only` yields honest-NaN, never the (invalid)
velocity model.

## Model description

- **Point estimate only.** `ghost_gr_x/y` serve the exact boosted HGBR mean, reconstructed
  pickle-free (`baseline + &Sigma;_trees leaf_value`; no sklearn at inference). There is **no** KDE /
  density read-out (a point estimate is what the counterfactual arm differences).
- **Parameters-only, pickle-free:** npz (two boosted x/y ensembles + baselines) + JSON metadata +
  SHA-256 checksums. No per-sample training data, no raw provider tracking data (ADR-011/044).
- **Fail-closed load:** SHA-256 + a behavioural **chirality** fingerprint (a y-mirrored model is
  rejected) + a **feature-contract** fingerprint. The chirality frame hash is pandas-major-invariant
  (loads identically under pandas 2 and 3).
- **FOV honest-NaN (StatsBomb-360):** a served frame whose rearguard region is not sufficiently
  observed is returned NaN (`ghost_outfield_source="fov_cropped"`), never a fabricated ghost from the
  deepest-`n` *visible* players.
- **Hyperparameters:** `HistGradientBoostingRegressor`, 500 trees, max depth 8, 5-fold CV; trained at
  1 frame/second (25 fps tracking is highly autocorrelated &mdash; 1 fps is ~25&times; fewer,
  near-duplicate rows with no meaningful signal loss for a mean-positioning model).

## Metrics

| Metric | Value |
|---|---|
| Held-out CV euclidean MAE | **6.00 m** (per-provider: Gradient Sports 6.14 / SkillCorner 5.92 / Sportec 6.33) |
| Per-possession CV MAE | in-possession 6.97 m / out-of-possession 5.04 m |
| Per-slot CV MAE (slots 1&ndash;4) | 5.96 / 6.05 / 6.02 / 5.98 m |
| Rearguard coherence (slot ordering) | `ordering_fraction = 1.0` (the independently-predicted slots order as a line) |
| Boosted-reconstruction parity vs sklearn | exact (round-trip verified &mdash; safe to publish) |

## Training data

179 matches / ~4.17M per-(frame, team, slot) rows (at 1 fps) of licensed professional tracking
(Sportec/DFL Bundesliga, SkillCorner, Gradient Sports FIFA World Cup 2022). Only the learned
parameters are published; **no raw provider tracking data is redistributed** (parameters-only
artifact). Trained from a clean, CI-green commit.

## Usage

```python
import silly_kicks.tracking as tracking

model = tracking.GhostOutfieldModel.from_variant("default")
# One row per (frame, team, slot) with goal-relative ghost_gr_x / ghost_gr_y + a provenance token.
served = tracking.serve_ghost_outfield_positions(frames, model=model, home_team_id=1)
```

Input frames must be LTR-normalized (home attacks right). All coordinates are goal-relative to the
modeled team's defended goal.

## Limitations

- League-average positioning, not a specific player's or team's tactical style.
- Static per-frame estimate; inherits tracking-system noise.
- LTR normalization required.
- Unlike the goalkeeper model, this model is **not** affected by the Gradient Sports keeper clamp
  (outfielders are tracked over the full pitch).

## References

Le et al. 2017 (Data-Driven Ghosting, MIT Sloan). This model is the boosted-**mean** point-estimate
variant of the ghosting concept (no density estimation). See the silly-kicks `NOTICE` for the full
citation.

## More information

- **License:** [MIT](https://opensource.org/licenses/MIT) &middot; **Library:**
  [silly-kicks](https://pypi.org/project/silly-kicks/) (v4.109.0+) &middot;
  [GitHub](https://github.com/karsten-s-nielsen/silly-kicks) &middot; ADR-087.
