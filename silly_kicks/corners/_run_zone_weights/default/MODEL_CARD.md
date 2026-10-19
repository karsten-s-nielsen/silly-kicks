---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - corner-kicks
  - tracking-data
library_name: silly-kicks
---

# Corner Run-Zone Model v1 &mdash; TF-46 Arm 1

`RunZoneModel` is a Gaussian-mixture zone encoder for corner-kick routines (Shaw & Gopaladesikan
playbook, TF-46 Arm 1). It is **bundled in the `silly-kicks` wheel** (params-only, redistributable);
there is no Hub repo.

## Model Description

Two fitted Gaussian mixtures over reflected attacking-player positions at a corner:

- **Initial zones** from the `t_kick - 2 s` snapshot (6 components).
- **Target zones** over the goal-centre-radius region (`goal_radius_m = 25.0`, box + top-of-box +
  second-ball edge), BIC-selected to 7 components.

`encode_runs` turns each corner into the `6 x 7 = 42`-column probabilistic run vector (initial zone
to target zone), the term matrix the `CornerRoutineModel` factors. sklearn `GaussianMixture` at **fit
only**; the serve path is pure numpy.

## Training corpus

Pooled multi-provider: SkillCorner + IDSSE/Sportec + StatsBomb SB360. 2702 direct corners, 26732
per-player snapshots. Only learned parameters (means, covariances, weights, `goal_radius_m`) are
stored &mdash; no raw provider tracking data.

## Provenance

**Provenance.** `training_commit=6ee7e62`. Fit on a clean tree, enforced by the fit driver's
`require_clean_tree` (ADR-037).

## Construct validity

The construct-validity memo is `docs/research/corner_routines/` (README + metrics.json). Headline:
corner shot generation (`n_shots`) separates teams beyond a permutation null and survives netting out
the opponent faced and the routine mix. This is an **instrument of corner shot creation, not a
team-quality ranking**; `n_goals`/`sum_xg` are underpowered and season-to-season transfer is not
established.

## Integrity and load-time guards

`load()` is fail-closed: (1) **SHA256SUMS** verified before parsing; (2) **chirality fingerprint** on a
fixed y-asymmetric probe; (3) **feature contract** (`goal_radius_m`) &mdash; a mismatch raises
`IntegrityError` rather than serving against geometry the model was not fit on.

## Usage

```python
from silly_kicks.corners import RunZoneModel

model = RunZoneModel.load("silly_kicks/corners/_run_zone_weights/default")
run_vectors = model.encode_runs(init_xy, target_xy)
```

## Limitations

- Descriptive / unsupervised: the zones are a mechanism encoding, carrying no accuracy claim.
- Reflected to the canonical corner side; serve on the same reflection convention.

## Model Files

| File | Purpose |
|---|---|
| `model.npz` | GMM means / covariances / weights (pickle-free) |
| `metadata.json` | zone counts, `goal_radius_m`, chirality fingerprint, feature contract, provenance |
| `SHA256SUMS` | integrity manifest, verified by `load()` |
| `MODEL_CARD.md` | this card |

## References

See the `NOTICE` file in the silly-kicks repository. Attribution: Shaw & Gopaladesikan, "Routine
Inspection" (42 Analytics). Decisions: ADR-011 (trained-model lifecycle), ADR-110 (raw primitives).

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
