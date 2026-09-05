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

# Ghost-Outfield Position-Only v1 &mdash; League-Average Rearguard Positioning (Velocity-Less)

The **velocity-less** variant of [Ghost-Outfield v1](https://huggingface.co/silly-kicks/ghost-outfield-v1):
predicts a league-average outfield **rearguard** position per tracking frame, team, and lateral slot,
using a 16-feature vector with the 4 velocity features dropped. Built for **StatsBomb-360
freeze-frames**, which carry no per-player velocity. Point estimate (`ghost_gr_x/y`, goal-relative)
served as the exact pickle-free boosted `HistGradientBoostingRegressor` mean.

Part of the [silly-kicks](https://github.com/karsten-s-nielsen/silly-kicks) soccer analytics library
(TF-60 rest-defense; ADR-087).

## Why a separate variant

StatsBomb-360 freeze-frames have no temporal history, so the 4 situational-velocity features of the
`default` variant (ball velocity components, ball speed, opponent-forward centroid velocity) cannot be
computed. This variant is trained **without** them so it is valid on velocity-less input. The serve
seam auto-selects it from the frames' declared velocity availability; if it is not bundled, a
velocity-less frame yields honest-NaN &mdash; **never** the `default` velocity model, which is invalid
on velocity-less input.

## Variants

| Variant | Repo | Features | Use |
|---|---|---|---|
| `default` | [`silly-kicks/ghost-outfield-v1`](https://huggingface.co/silly-kicks/ghost-outfield-v1) | 20 (velocity-bearing) | Continuous tracking (Sportec, SkillCorner, Gradient Sports) |
| `position_only` (this repo) | `silly-kicks/ghost-outfield-position-only-v1` | 16 (4 velocity features dropped) | Velocity-less StatsBomb-360 freeze-frames |

## Model description

- **Point estimate only.** `ghost_gr_x/y` serve the exact boosted HGBR mean, reconstructed
  pickle-free (`baseline + &Sigma;_trees leaf_value`; no sklearn at inference). No KDE / density
  read-out.
- **Possession-conditioned & leakage-safe:** one model serves both the in-possession rest-defense
  rearguard and the defending line (a live `team_in_possession` feature); no input encodes the modeled
  team's own rearguard coordinates (the target).
- **Parameters-only, pickle-free:** npz + JSON metadata + SHA-256 checksums; no raw provider tracking
  data (ADR-011/044).
- **Fail-closed load:** SHA-256 + behavioural chirality + feature-contract; the chirality frame hash
  is pandas-major-invariant.
- **FOV honest-NaN (StatsBomb-360):** an insufficiently-observed rearguard region is returned NaN
  (`ghost_outfield_source="fov_cropped"`), never a fabricated ghost from the deepest-`n` visible
  players.
- **Hyperparameters:** `HistGradientBoostingRegressor`, 500 trees, max depth 8, 5-fold CV; trained at
  1 frame/second.

## Metrics

| Metric | Value |
|---|---|
| Held-out CV euclidean MAE | **6.07 m** (per-provider: Gradient Sports 6.20 / SkillCorner 5.99 / Sportec 6.40) |
| Per-possession CV MAE | in-possession 7.03 m / out-of-possession 5.10 m |
| Per-slot CV MAE (slots 1&ndash;4) | 6.03 / 6.11 / 6.09 / 6.04 m |
| Rearguard coherence (slot ordering) | `ordering_fraction = 1.0` |
| Boosted-reconstruction parity vs sklearn | exact (round-trip verified &mdash; safe to publish) |

Slightly higher MAE than the `default` variant, as expected: dropping the velocity features removes
information. Use `default` on velocity-bearing tracking; use this variant only on velocity-less input.

## Training data

179 matches / ~4.17M per-(frame, team, slot) rows (at 1 fps) of licensed professional tracking
(Sportec/DFL Bundesliga, SkillCorner, Gradient Sports FIFA World Cup 2022). Only the learned
parameters are published; **no raw provider tracking data is redistributed** (parameters-only
artifact). Trained from a clean, CI-green commit.

## Usage

```python
import silly_kicks.tracking as tracking

# Auto-selected on declared velocity-less freeze-frames when model=None; or explicitly:
model = tracking.GhostOutfieldModel.from_variant("position_only")
served = tracking.serve_ghost_outfield_positions(frames, model=model, home_team_id=1)
```

Input frames must be LTR-normalized (home attacks right). All coordinates are goal-relative to the
modeled team's defended goal.

## Limitations

- League-average positioning, not a specific player's or team's style.
- Velocity features dropped &mdash; use `default` where velocity is available.
- Static per-frame estimate; LTR normalization required.

## References

Le et al. 2017 (Data-Driven Ghosting, MIT Sloan). Boosted-**mean** point-estimate variant of the
ghosting concept (no density estimation). See the silly-kicks `NOTICE` for the full citation.

## More information

- **License:** [MIT](https://opensource.org/licenses/MIT) &middot; **Library:**
  [silly-kicks](https://pypi.org/project/silly-kicks/) (v4.109.0+) &middot;
  [GitHub](https://github.com/karsten-s-nielsen/silly-kicks) &middot; ADR-087.
