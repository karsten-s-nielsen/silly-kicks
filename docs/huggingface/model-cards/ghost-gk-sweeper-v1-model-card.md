---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - goalkeeper
  - tracking-data
  - ghosting
  - rest-defense
pipeline_tag: tabular-regression
library_name: silly-kicks
---

# Ghost-GK Sweeper v1 &mdash; Extended-Grid Goalkeeper Positioning (Rest-Defense)

Predicts where a league-average goalkeeper would position themselves, **including the in-possession
high-sweeper regime** that the standard [Ghost-GK v1](https://huggingface.co/silly-kicks/ghost-gk-v1)
model cannot represent. Point-estimate (`ghost_gk_x/y`) served as the exact pickle-free boosted
`HistGradientBoostingRegressor` mean.

Part of the [silly-kicks](https://github.com/karsten-s-nielsen/silly-kicks) soccer analytics library
(TF-60 rest-defense; ADR-083).

## Why a separate model

The bundled Ghost-GK `default`/`full` variants train on a fixed goal-relative box **x &isin; [0, 30] m**
from the defended goal: a keeper who has swept far upfield falls outside it and is dropped as a
"sweeper rush". That is correct for normal in-goal positioning, but it makes the model **hard-saturate
at 30 m** &mdash; it cannot place an *in-possession* keeper who has pushed up to 30&ndash;45 m to blunt
a counter-attack (the rest-defense regime). This `sweeper` variant **lifts the label ceiling to
x_max = 52.5 m** so it represents the high-sweeper regime, for TF-60 rest-defense / TF-19 GKDV
counterfactual arms.

The frozen `default`/`position_only`/`full` variants are **unchanged** &mdash; this is an additive
variant; consumers opt in via `from_variant("sweeper")`.

## Variants

| Variant | Repo | Features | Use |
|---|---|---|---|
| `sweeper` (this repo) | `silly-kicks/ghost-gk-sweeper-v1` | 26 (velocity-bearing) | Continuous tracking (Sportec, SkillCorner, Gradient Sports) |
| `sweeper_position_only` | [`silly-kicks/ghost-gk-sweeper-position-only-v1`](https://huggingface.co/silly-kicks/ghost-gk-sweeper-position-only-v1) | 21 (5 velocity features dropped) | Velocity-less StatsBomb-360 freeze-frames |

## Model description

- **Point estimate only.** `ghost_gk_x/y` serve the exact boosted HGBR mean, reconstructed pickle-free
  (`baseline + &Sigma;_trees leaf_value`; no sklearn at inference). The KDE **density** read-out
  (`predict_density`) is **not supported on this extended grid** and raises &mdash; the density path
  stays on the default 30 m grid (ADR-083). Use `compute_ghost_gk` / `serve_ghost_gk_positions`.
- **Grid:** first-class per-model `GhostGridSpec` &mdash; x &isin; [0, **52.5**] m, y &isin; [18, 50] m,
  0.5 m resolution (105 &times; 64 cells).
- **Parameters-only, pickle-free:** npz (tree ensembles + baselines) + JSON metadata + SHA-256
  checksums. No per-sample training data, no raw provider tracking data (ADR-044).
- **Hyperparameters:** `HistGradientBoostingRegressor`, 500 trees, max depth 8, 5-fold CV.

## Metrics

| Metric | Value |
|---|---|
| Held-out CV euclidean MAE | **1.142 m** (per-provider: Gradient Sports 1.078 / SkillCorner 1.167 / Sportec 1.734) |
| Boosted-reconstruction parity vs sklearn | 1.21e-13 (exact &mdash; safe to publish) |
| **> 30 m high-sweeper stratum MAE** | **~2.06 m** (the sweeper *places* high keepers where the default is blind) |

The `> 30 m` (high-sweeper) coverage of the training corpus is **IDSSE/Sportec-dominated (11.5 %)**;
SkillCorner 0.24 %; **Gradient Sports 0.0 %** &mdash; see the data caveat below.

## Training data

179 matches / ~1.05M frames of licensed professional tracking (Sportec/DFL Bundesliga, SkillCorner,
Gradient Sports FIFA World Cup 2022). Only the learned parameters are published; **no raw provider
tracking data is redistributed** (parameters-only artifact). Trained from a clean, CI-green commit.

> **&#9888; Gradient Sports goalkeeper clamp.** Gradient Sports' tracking clamps the goalkeeper's
> position to a hard **27.5 m from goal** (a source-data limitation, verified on the raw provider data;
> silly-kicks passes it through faithfully). Consequently GS contributes **no** high-sweeper training
> signal, and **any GS goalkeeper-depth analysis is invalid past 27.5 m**. The high-sweeper regime this
> model represents is learned from IDSSE/Sportec. silly-kicks flags this at conversion time via
> `validate_gk_position_clamp` / `GoalkeeperClampWarning`. See `docs/research/gs_keeper_clamp/`.

## Usage

```python
import silly_kicks.tracking as tracking

# The extended-grid sweeper (velocity-keyed within its family on velocity-less frames)
model = tracking.GhostGkModel.from_variant("sweeper")
out = tracking.compute_ghost_gk(frames, model=model, home_team_id=1)  # ghost_gk_x / ghost_gk_y

# predict_density is NOT supported on the extended grid (raises); use the mean/serve path above.
```

Input frames must be LTR-normalized (home attacks right). All coordinates are goal-relative.

## Limitations

- League-average positioning, not a specific keeper's style; no shot-stopping ability.
- **No KDE density** on the extended grid (mean/serve path only).
- Inherits tracking-system noise; **Gradient Sports keeper positions are unusable past 27.5 m** (above).
- LTR normalization required; static per-frame estimate.

## References

Le et al. 2017 (Data-Driven Ghosting, MIT Sloan); Dutta et al. 2024 (NFL Ghosts, arXiv:2406.17220);
Pospisil &amp; Lee 2018 (RFCDE, arXiv:1804.05753). See the silly-kicks `NOTICE` for full citations.

## More information

- **License:** [MIT](https://opensource.org/licenses/MIT) &middot; **Library:**
  [silly-kicks](https://pypi.org/project/silly-kicks/) (v4.105.0+) &middot;
  [GitHub](https://github.com/karsten-s-nielsen/silly-kicks) &middot; ADR-083.
