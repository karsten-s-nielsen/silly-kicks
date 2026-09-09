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

# Ghost-GK Sweeper (position-only) v1 &mdash; Extended-Grid GK Positioning for Velocity-less Frames

The **velocity-less** companion to
[`silly-kicks/ghost-gk-sweeper-v1`](https://huggingface.co/silly-kicks/ghost-gk-sweeper-v1): the same
extended-grid (x_max = 52.5 m) high-sweeper model, but with the **5 velocity features dropped** (21
features instead of 26) so it scores on **StatsBomb-360 freeze-frames**, which carry no per-player
temporal history.

Part of the [silly-kicks](https://github.com/karsten-s-nielsen/silly-kicks) soccer analytics library
(TF-60 rest-defense; ADR-067 velocity-keyed variants; ADR-083).

## Why a separate model

The bundled Ghost-GK variants train on a fixed **x &isin; [0, 30] m** goal-relative box and
**hard-saturate at 30 m**, so they cannot place an in-possession high sweeper (30&ndash;45 m). This
`sweeper_position_only` variant lifts the ceiling to **x_max = 52.5 m** AND drops the velocity features
so the model is valid on velocity-less freeze-frames (velocity features are **dropped, not NaN-filled**
&mdash; the feature contract raises on a non-finite input rather than imputing). It is auto-selected at
the serve seam for a `sweeper` request on declared velocity-less frames.

The frozen `default`/`position_only`/`full` variants are **unchanged** &mdash; additive; opt in via
`from_variant("sweeper")` (which resolves to this variant on velocity-less frames).

## Model description

- **Point estimate only** (`ghost_gk_x/y`, exact pickle-free boosted HGBR mean); `predict_density` is
  **not supported on the extended grid** and raises (ADR-083).
- **21 features** (the 5 velocity features &mdash; `ball_vx`, `ball_vy`, `ball_speed`,
  `defensive_line_speed`, `defending_centroid_vx` &mdash; are dropped, not imputed).
- **Grid:** x &isin; [0, **52.5**] m, y &isin; [18, 50] m, 0.5 m resolution.
- **Parameters-only, pickle-free** (npz + JSON + SHA-256); no per-sample or raw provider data (ADR-044).
- **Hyperparameters:** `HistGradientBoostingRegressor`, 500 trees, max depth 8, 5-fold CV.

## Metrics

| Metric | Value |
|---|---|
| Held-out CV euclidean MAE | **1.157 m** (per-provider: Gradient Sports 1.087 / SkillCorner 1.216 / Sportec 1.724) |
| Boosted-reconstruction parity vs sklearn | 1.78e-13 (exact) |
| **> 30 m high-sweeper stratum MAE** | **~1.98 m** |

`> 30 m` coverage is IDSSE/Sportec-dominated (11.5 %); SkillCorner 0.24 %; **Gradient Sports 0.0 %**
(see the data caveat).

> **Both-axes goal-relative re-fit (silly-kicks 4.111.0 / ADR-089; `training_commit=4bda048`).** These
> weights were re-fit after the ghost-GK feature extractor was corrected to apply the full **180°
> goal-relative point reflection (both axes)** — the previous convention flipped only x, leaving the
> signed-y features (the velocity-less variant keeps `ball_y`, `ball_to_goal_angle`,
> `attacker_centroid_y`) and the target `gk_y` in absolute frame coordinates, mislocating away-team
> keepers in y. The served `ghost_gk_x/y` change (chiefly for away-team keepers) and the
> feature-contract + chirality fingerprints change, so **pre-4.111.0 artifacts fail closed on load**.
> The aggregate MAE is essentially unchanged (the correction affects the away-team subset).

## Training data

Same 179-match licensed corpus as the faithful sweeper (Sportec/DFL, SkillCorner, Gradient Sports
WC2022); parameters-only, no raw tracking data redistributed; clean CI-green training commit.

> **&#9888; Gradient Sports goalkeeper clamp.** Gradient Sports clamps the goalkeeper's tracked
> position to a hard **27.5 m from goal** (source-data limitation, verified on the raw provider data).
> GS contributes no high-sweeper signal and **any GS goalkeeper-depth analysis is invalid past 27.5 m**;
> silly-kicks flags it via `validate_gk_position_clamp` / `GoalkeeperClampWarning`. See
> `docs/research/gs_keeper_clamp/`.

## Usage

```python
import silly_kicks.tracking as tracking

# from_variant("sweeper") auto-selects THIS variant on declared velocity-less (freeze-frame) input.
model = tracking.GhostGkModel.from_variant("sweeper_position_only")
out = tracking.compute_ghost_gk(frames, model=model, home_team_id=1)  # ghost_gk_x / ghost_gk_y
```

Input frames must be LTR-normalized; coordinates are goal-relative. `predict_density` raises on the
extended grid &mdash; use the mean/serve path.

## Limitations

- League-average positioning, not a specific keeper's style; no shot-stopping ability.
- Velocity-less by design (21 features) &mdash; use the 26-feature `ghost-gk-sweeper-v1` on
  velocity-bearing continuous tracking.
- **No KDE density** on the extended grid; **Gradient Sports keeper positions unusable past 27.5 m**.
- LTR normalization required; static per-frame estimate.

## References

Le et al. 2017 (MIT Sloan); Dutta et al. 2024 (arXiv:2406.17220); Pospisil &amp; Lee 2018
(arXiv:1804.05753). See the silly-kicks `NOTICE` for full citations.

## More information

- **License:** [MIT](https://opensource.org/licenses/MIT) &middot; **Library:**
  [silly-kicks](https://pypi.org/project/silly-kicks/) (**v4.111.0+** for these both-axes weights) &middot;
  [GitHub](https://github.com/karsten-s-nielsen/silly-kicks) &middot; ADR-067 / ADR-083 / ADR-089.
