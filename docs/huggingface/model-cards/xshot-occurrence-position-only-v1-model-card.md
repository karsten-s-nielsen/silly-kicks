---
license: mit
language: en
tags:
  - sports-analytics
  - soccer
  - goalkeeper
  - shot-prediction
  - tracking-data
pipeline_tag: tabular-classification
library_name: silly-kicks
---

# xShotOccurrence v1 (`sc_extended_position_only`) &mdash; Shot Propensity for Velocity-less Frames

> **Read this first.** This repo serves the **`sc_extended_position_only`** (owner-tier,
> **position-only**) variant — the owner-tier xShot model with **velocity features dropped**, for
> scoring **velocity-less** snapshots such as StatsBomb-360 freeze-frames. It is **NOT bundled** with
> the wheel (restricted owner-tier training data — a **licensing** constraint). If you do not have
> owner-tier access, use the bundled `position_only` variant. Reachable ONLY via
> `from_variant("sc_extended_position_only")`; asking for `sc_extended` returns the **faithful**
> (velocity-bearing) model (ADR-070).

## Model Description

`XShotOccurrenceModel` (position-only) estimates **P(a shot is attempted by the in-possession team
within ~1 s of a tracking frame)** using **position-only** features — the 5 velocity-derived features
are **dropped** (not NaN-filled), so the model scores frames that carry no per-player velocity at all
(e.g. a single freeze-frame).

- **26 features** (`position_only`), goal-relative coordinates via the shared `_geometry` helper
- Domain filter: alive-ball, **attacking third**
- Trained on **964,263** rows / **182,517** positives (18.9% positive rate) — same owner-tier corpus
  as the faithful `sc_extended` variant

## When to use this vs the bundled `position_only`

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `position_only` (bundled) | public corpus | **bundled in the wheel** | **Default** for velocity-less frames without owner-tier access |
| `sc_extended_position_only` | **115 matches** — 7 IDSSE + 108 SkillCorner (incl. **98 owner-tier**) | **this repo** (HF-only) | A **stronger** velocity-less model, if you have owner-tier access and can accept a Hub download |

Use this when your frames genuinely lack velocity. For velocity-bearing tracking, prefer the faithful
`sc_extended` model (`silly-kicks/xshot-occurrence-v1`) — velocity features carry real signal.

## Why this variant is HF-only

HF-only for **licensing**, not quality: trained on **restricted owner-tier SkillCorner data** that
cannot be redistributed in the wheel (ADR-038). Only learned parameters are published — **no raw
provider tracking data**. Produced by a two-provider (IDSSE + SkillCorner) single-candidate run; the corpus **is** the
owner-tier tier. `training_commit`: `1ce63ef`.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.5276 (± 0.0345) | base rate 0.1893 |
| Brier | 0.1213 | base-rate Brier 0.1534 |
| Log loss | 0.3919 | &mdash; |

All four acceptance gates pass. Estimates are **CV, not the shipped fit**. As expected, dropping
velocity costs some discrimination vs the faithful model (PR-AUC 0.585 → 0.528) — the trade is that
this model *runs at all* on velocity-less frames.

## Usage

```python
from silly_kicks.tracking import XShotOccurrenceModel

model = XShotOccurrenceModel.from_variant("sc_extended_position_only")  # this repo, Hub download
```

Requires `pip install silly-kicks[xshot]` and **silly-kicks >= 4.94.0** (the release that introduced
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
- No GK measurement exists for the xS arm (the TF-19 xS probe is blocked; unmeasured, not validated).
- Trained on **115 matches, heavily one-club**; club/style confounding is real and unquantified.
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Attribution: arXiv:2512.00203.
- Decisions: ADR-011 (trained-model lifecycle), ADR-038 (corpus + visibility), ADR-040 (chirality
  enforcement), ADR-063 (velocity-less lift), ADR-067 (position-only variants + velocity auto-select),
  ADR-070 (position-only Hub variant), ADR-071 (owner-tier archive).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics and acceptance record |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
