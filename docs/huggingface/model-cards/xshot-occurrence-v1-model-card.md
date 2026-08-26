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

# xShotOccurrence v1 (`sc_extended`) &mdash; Shot-Occurrence Propensity from Tracking State

> **Read this first.** This repo serves the **`sc_extended`** (owner-tier) variant. It is **NOT
> bundled with the `silly-kicks` wheel** because it is trained on **restricted owner-tier data that
> cannot be redistributed** inside a PyPI package — a **licensing** constraint. Only the learned
> parameters are published here. If you do not have owner-tier access, use the bundled `default`
> variant instead — see [Which variant should I use?](#which-variant-should-i-use).

## Model Description

`XShotOccurrenceModel` is a deterministic-XGBoost classifier estimating **P(a shot is attempted by
the in-possession team within ~1 s of a tracking frame)** &mdash; the xS surface of the GKDV research
arc (TF-16).

Paper-faithful **27-feature** extractor in goal-relative coordinates via the shared `_geometry`
helper: ball r/θ/z/speed, an `openGoal` goal-mouth obstruction term, GK distance/bearing, and the 5
nearest defenders + 5 nearest attackers.

- Domain filter: alive-ball, **attacking third**
- Feature set: **`faithful`** (velocity-bearing, 27 features)
- Trained on **964,263** rows / **182,517** positives (18.9% positive rate)

## Which variant should I use?

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `default` (`public`) | 17 matches &mdash; SkillCorner + IDSSE, redistributable | **bundled in the wheel** | **Default choice** &mdash; offline, fully reproducible, no restricted data |
| `sc_extended` | **115 matches** &mdash; 7 IDSSE + 108 SkillCorner (incl. **98 owner-tier**) | **this repo** (HF-only) | Yes, if you have owner-tier access and can accept a Hub download + the corpus caveats below |
| `sc_extended_position_only` | same 115-match corpus, **velocity features dropped** (26-feature) | **separate repo** `silly-kicks/xshot-occurrence-position-only-v1` (HF-only) | Yes, if scoring **velocity-less** frames (StatsBomb-360 freeze frames) with owner-tier access — a stronger position-only model than the bundled `position_only`. Reachable ONLY via `from_variant("sc_extended_position_only")`; asking for `sc_extended` still returns this **faithful** model (ADR-070). |

## Why this variant is HF-only

`sc_extended` is HF-only for **licensing**, not quality: it is trained on **restricted owner-tier
SkillCorner data** that cannot be redistributed inside the PyPI wheel (ADR-038). Only learned
parameters are published here &mdash; **no raw provider tracking data** (only split thresholds, feature indices and leaf values are stored — no per-sample training data).

This artifact was produced by a **two-provider (IDSSE + SkillCorner) single-candidate** run: the
corpus **is** the owner-tier `sc_extended` tier, so there is no paired comparison against `public`
(that comparison governs the *wheel bundle* selection, not this archive; ADR-071). The bundled
`default`/`public` model remains the reproducible offline choice.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.5851 (± 0.0436) | base rate 0.1893 |
| Brier | 0.1131 | base-rate Brier 0.1534 |
| Log loss | 0.3671 | &mdash; |

All four acceptance gates pass (`enough_usable_folds`, `pr_auc_gt_base_rate`,
`brier_lt_base_rate_brier`, `log_loss_lt_uniform`). Estimates are **CV, not the shipped fit**.
`training_commit`: `1ce63ef`.

## TF-19 status: the shot arm has never been measured

Unlike its xCross sibling, this model carries **no GK-substitution probe** in `metrics.json`. That is
**blocked, not missing**: `xs_substitution_probe` consumes ghost-substituted targets from the GKDV
engine, and the registered xS probe has not been run against this arm. **A TF-19 spec must not assume
the shot arm is healthy** — it is *unmeasured*, not validated.

## Usage

```python
from silly_kicks.tracking import XShotOccurrenceModel

model = XShotOccurrenceModel.from_variant("default")       # recommended, bundled, offline
model = XShotOccurrenceModel.from_variant("sc_extended")   # this repo, downloads from the Hub
```

Requires `pip install silly-kicks[xshot]` and **silly-kicks >= 4.74.0** (the `sc_extended_position_only`
sibling repo requires **>= 4.94.0**, which introduced its variant key — ADR-070).

> **The `>= 4.74.0` floor is a hard requirement.** These weights are on the corrected goal-relative
> transform (`geometry_version: goal-relative-2`); ADR-051 found the previous transform was **chiral**
> (an x-only mirror at one goal end, identity at the other), so one physical scene scored differently
> depending which end the attacking team attacked. `load()`'s feature-contract prong is **fail-closed**,
> so an older silly-kicks refuses these weights with `IntegrityError` rather than serve them against the
> geometry they were not fit on. `from_hub()` takes no `revision` argument yet, so treat the library
> version as the pin; prior revisions are addressable by commit SHA in this repo's git history.

## Integrity and load-time guards

`load()` is **fail-closed** on two independent checks: (1) **SHA256SUMS** verified before anything is
parsed; (2) **chirality fingerprint** (ADR-040) — the model re-runs its own outputs on a fixed
y-asymmetric probe frame and compares to the recorded fingerprint, raising on a mismatch **and on a
missing one**. A `base_score` guard handles the xgboost 3.x bracketed-string serialization that 2.x
silently drops to `0.5`.

## Limitations

- **Not the bundled model** (restricted corpus). This is a redistribution limit, not a performance one.
- **No GK measurement exists for this arm at all** — see the TF-19 section.
- Trained on **115 matches, heavily one-club** — the 98 owner-tier additions are a single club, so
  club/style confounding is real and unquantified here.
- SkillCorner keepers are detected in only **~19.6%** of frames (~80% interpolated), which is why GKDV
  *measurement* is registered to Gradient Sports frames only (ADR-038 §5).
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Attribution: arXiv:2512.00203.
- Decisions: ADR-011 (trained-model lifecycle), ADR-037 (TF-19 re-gate), ADR-038 (corpus + visibility),
  ADR-040 (chirality enforcement), ADR-070 (position-only Hub variant), ADR-071 (owner-tier archive).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics and acceptance record |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
