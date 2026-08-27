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

# xCrossAttempt v1 (`sc_extended`) &mdash; Cross-Attempt Propensity from Tracking State

> **Read this first.** This repo serves the **`sc_extended`** (owner-tier) variant. It is **NOT
> bundled with the `silly-kicks` wheel** because it is trained on **restricted owner-tier data that
> cannot be redistributed** inside a PyPI package — a **licensing** constraint. Only the learned
> parameters are published here. If you do not have owner-tier access, use the bundled `default`
> variant instead — see [Which variant should I use?](#which-variant-should-i-use).

## Model Description

`XCrossAttemptModel` is a deterministic-XGBoost classifier estimating **P(the in-possession team
attempts a cross within ~1 s of a tracking frame)** &mdash; a **STATE-anchored** framing, reframed
from the sender-level event treatment in Cao et al. (arXiv:2505.11841).

It carries **7 of the paper's 8 confounders** (crosser-position, #7, is omitted &mdash; no faithful
tracking-only proxy) plus a **novel, isolatable GK-position confounder block**, which is the paper's
headline gap and the reason this model exists inside the GKDV research arc (TF-17 → TF-19).

- **16 features**, `faithful` (velocity-bearing), goal-relative coordinates via the shared `_geometry` helper
- Domain filter: alive-ball, **wide-area**
- The `sc_extended` model is fit on the owner-tier corpus (IDSSE + SkillCorner incl. owner-tier);
  base rate 5.0% positive.

## Which variant should I use?

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `default` (`public`) | 17 matches &mdash; SkillCorner + IDSSE, redistributable | **bundled in the wheel** | **Default choice** &mdash; offline, fully reproducible, no restricted data |
| `sc_extended` | IDSSE + SkillCorner incl. **98 owner-tier** SkillCorner matches | **this repo** (HF-only) | Yes, if you have owner-tier access and can accept a Hub download + the corpus caveats below |
| `sc_extended_position_only` | same owner-tier corpus, **velocity features dropped** (15-feature) | **separate repo** `silly-kicks/xcross-attempt-position-only-v1` (HF-only) | Yes, if scoring **velocity-less** frames (StatsBomb-360 freeze frames) with owner-tier access — a stronger position-only model than the bundled `position_only`. Reachable ONLY via `from_variant("sc_extended_position_only")`; asking for `sc_extended` still returns this **faithful** model (ADR-070). |

## Why this variant is HF-only

`sc_extended` is HF-only for **licensing**, not quality: it is trained on **restricted owner-tier
SkillCorner data** that cannot be redistributed inside the PyPI wheel (ADR-038). Only learned
parameters are published here &mdash; **no raw provider tracking data** (only split thresholds, feature indices and leaf values are stored — no per-sample training data).

The Hub `sc_extended` repo is the owner-tier **archive**: it holds the owner-tier model independent of
the wheel-bundle selection gate, which decides only what ships *in the wheel* (ADR-071). This artifact
was produced with that operator override (`--ship-variant sc_extended`); the gate's verdict and
per-fold deltas are recorded in `metrics.json` (`candidates.paired`). `training_commit`: `b658445`.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.1888 (± 0.0108) | base rate 0.0500 |
| Brier | 0.0436 | base-rate Brier 0.0475 |
| Log loss | 0.1658 | &mdash; |

All four acceptance gates pass (`enough_usable_folds`, `pr_auc_gt_base_rate`,
`brier_lt_base_rate_brier`, `log_loss_lt_uniform`). Estimates are **CV, not the shipped fit**.

## TF-19 GK-substitution probe

The frozen **GK-substitution probe** (`gk_substitution_probe` in `metrics.json`; 200 frames; ADR-037's
two-prong gate — ratio ≥ 2.0 × the nearest-defender control **and** an absolute floor ≥ 0.01):

| Metric | Value |
|---|---|
| `gk_median_abs_delta` | 0.00625 |
| `nearest_def_median_abs_delta` | 0.00329 |
| ratio (gk / control) | 1.90× — **misses** (needs ≥ 2.0) |
| absolute floor | 0.00625 < 0.01 — **misses** |
| **`tf19_ready`** | **false** |

The GK-block ablation shows the GK confounder block *does* carry signal (removing it drops held-out
PR-AUC by 0.0089), but the substitution probe does not clear the frozen gate. Per ADR-037 this is a
`gated_clean_fail` — TF-19 routes to **GK feature engineering**, explicitly *not* "no signal." Do not
build a TF-19 consumer on this surface. (The position-only sibling — `xcross-attempt-position-only-v1` —
*does* clear the gate; see its card.)

## Usage

```python
from silly_kicks.tracking import XCrossAttemptModel

model = XCrossAttemptModel.from_variant("default")       # recommended, bundled, offline
model = XCrossAttemptModel.from_variant("sc_extended")   # this repo, downloads from the Hub
```

Requires `pip install silly-kicks[xcross]` and **silly-kicks >= 4.74.0** (the
`sc_extended_position_only` sibling repo requires **>= 4.94.0**, which introduced its variant key —
ADR-070).

> **The `>= 4.74.0` floor is a hard requirement.** These weights are on the corrected goal-relative
> transform (`geometry_version: goal-relative-2`); ADR-051 found the previous transform was **chiral**
> (an x-only mirror at one goal end, identity at the other), so one physical scene scored differently
> depending which end the attacking team attacked. `load()`'s feature-contract prong is **fail-closed**,
> so an older silly-kicks refuses these weights with `IntegrityError`. `from_hub()` takes no `revision`
> argument yet, so treat the library version as the pin; prior revisions are addressable by commit SHA.

## Integrity and load-time guards

`load()` is **fail-closed** on two independent checks: (1) **SHA256SUMS** verified before anything is
parsed; (2) **chirality fingerprint** (ADR-040) — the model re-runs its own outputs on a fixed
y-asymmetric probe frame and compares to the recorded fingerprint, raising on a mismatch **and on a
missing one**. A `base_score` guard handles the xgboost 3.x bracketed-string serialization that 2.x
silently drops to `0.5`.

## Limitations

- **Not the bundled model** (restricted corpus). This is a redistribution limit, not a performance one.
- `tf19_ready = false` (see the TF-19 section) — do not build a TF-19 consumer on this surface.
- Trained on an owner-tier corpus that is **heavily one-club** (the 98 owner-tier additions are a
  single club), so club/style confounding is real and unquantified here.
- SkillCorner keepers are detected in only **~19.6%** of frames (~80% interpolated), which is why GKDV
  *measurement* is registered to Gradient Sports frames only (ADR-038 §5).
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Cao et al. "Framing Causal Questions in Sports Analytics: A Case Study of Crossing in Soccer."
  arXiv:2505.11841 (2025).
- Decisions: ADR-011 (trained-model lifecycle), ADR-015 (causal-validation port), ADR-037 (TF-19
  re-gate), ADR-038 (corpus + visibility), ADR-040 (chirality enforcement), ADR-070 (position-only Hub
  variant), ADR-071 (owner-tier archive).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics, GK-substitution probe, ablation, permutation importance |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
