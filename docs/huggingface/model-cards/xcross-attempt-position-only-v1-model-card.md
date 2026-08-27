---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - goalkeeper
  - cross-attempt
  - tracking-data
  - causal-inference
pipeline_tag: tabular-classification
library_name: silly-kicks
---

# xCrossAttempt v1 (`sc_extended_position_only`) &mdash; Cross Propensity for Velocity-less Frames

> **Read this first.** This repo serves the **`sc_extended_position_only`** (owner-tier,
> **position-only**) variant — the owner-tier xCross model with **velocity features dropped**, for
> scoring **velocity-less** snapshots such as StatsBomb-360 freeze-frames. It is **NOT bundled** with
> the wheel (restricted owner-tier training data — a **licensing** constraint). If you do not have
> owner-tier access, use the bundled `position_only` variant. Reachable ONLY via
> `from_variant("sc_extended_position_only")`; asking for `sc_extended` returns the **faithful**
> (velocity-bearing) model (ADR-070).

## Model Description

`XCrossAttemptModel` (position-only) estimates **P(the in-possession team attempts a cross within
~1 s of a tracking frame)** using **position-only** features — the velocity-derived features are
**dropped** (not NaN-filled), so the model scores frames that carry no per-player velocity at all.

It carries the STATE-anchored cross-propensity framing (Cao et al., arXiv:2505.11841) plus the
isolatable GK-position confounder block, minus the velocity terms.

- **15 features** (`position_only`), goal-relative coordinates via the shared `_geometry` helper
- Domain filter: alive-ball, **wide-area**; base rate 5.0% positive
- Fit on the owner-tier corpus (IDSSE + SkillCorner incl. owner-tier)

## When to use this vs the bundled `position_only`

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `position_only` (bundled) | public corpus | **bundled in the wheel** | **Default** for velocity-less frames without owner-tier access |
| `sc_extended_position_only` | IDSSE + SkillCorner incl. **98 owner-tier** SkillCorner matches | **this repo** (HF-only) | A **stronger** velocity-less model, if you have owner-tier access and can accept a Hub download |

Use this when your frames genuinely lack velocity. For velocity-bearing tracking, prefer the faithful
`sc_extended` model (`silly-kicks/xcross-attempt-v1`).

## Why this variant is HF-only

HF-only for **licensing**, not quality: trained on **restricted owner-tier SkillCorner data** that
cannot be redistributed in the wheel (ADR-038). Only learned parameters are published — **no raw
provider tracking data**. The Hub repo is the owner-tier **archive** (ADR-071). `training_commit`: `1ce63ef`.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.1297 (± 0.0087) | base rate 0.0500 |
| Brier | 0.0454 | base-rate Brier 0.0475 |
| Log loss | 0.1771 | &mdash; |

All four acceptance gates pass. Estimates are **CV, not the shipped fit**. Dropping velocity costs
discrimination vs the faithful model (PR-AUC 0.189 → 0.130) — the trade is that this model *runs at
all* on velocity-less frames.

## TF-19 GK-substitution probe

The frozen **GK-substitution probe** (`gk_substitution_probe` in `metrics.json`; 200 frames; ADR-037's
two-prong gate — ratio ≥ 2.0 × the nearest-defender control **and** an absolute floor ≥ 0.01):

| Metric | Value |
|---|---|
| `gk_median_abs_delta` | 0.01639 |
| `nearest_def_median_abs_delta` | 0.00664 |
| ratio (gk / control) | 2.47× — **clears** (≥ 2.0) |
| absolute floor | 0.01639 ≥ 0.01 — **clears** |
| **`tf19_ready`** | **true** |

This position-only variant **clears both prongs** of the frozen TF-19 gate — the only one of the four
xShot/xCross re-fits to do so. The GK-block ablation is consistent (removing the GK block drops
held-out PR-AUC by 0.0023). See ADR-037 for the gate definition.

## Usage

```python
from silly_kicks.tracking import XCrossAttemptModel

model = XCrossAttemptModel.from_variant("sc_extended_position_only")  # this repo, Hub download
```

Requires `pip install silly-kicks[xcross]` and **silly-kicks >= 4.94.0** (the release that introduced
this variant key — ADR-070). The `>= 4.74.0` geometry floor (`goal-relative-2`) also applies; `load()`
is fail-closed on the feature contract + chirality fingerprint (ADR-040). `from_hub()` takes no
`revision` argument yet — treat the library version as the pin.

## Integrity and load-time guards

`load()` is **fail-closed** on **SHA256SUMS** and the **chirality fingerprint** (ADR-040), plus a
`base_score` guard for xgboost 3.x serialization. The position-only feature contract raises on a
non-finite (NaN-filled) velocity feature — this model expects those columns **absent**, not imputed.

## Limitations

- **Not the bundled model** (restricted corpus). Redistribution limit, not a performance one.
- **Weaker than the faithful `sc_extended`** by construction (velocity dropped) — use only when frames
  lack velocity.
- `tf19_ready = true` (see the TF-19 section) — this variant clears the frozen gate, but the verdict is
  a routing signal, not a construct-validity claim; validate before building a headline TF-19 consumer.
- Trained on an owner-tier corpus that is **heavily one-club**; club/style confounding is real and
  unquantified.
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Cao et al. "Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer."
  arXiv:2505.11841 (2025).
- Decisions: ADR-011 (trained-model lifecycle), ADR-015 (causal-validation port), ADR-038 (corpus +
  visibility), ADR-040 (chirality enforcement), ADR-063 (velocity-less lift), ADR-067 (position-only
  variants + velocity auto-select), ADR-070 (position-only Hub variant), ADR-071 (owner-tier archive).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics, GK-substitution probe, ablation, permutation importance |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
