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

> **Read this first.** This repo serves the **`sc_extended`** variant, which is **NOT the
> recommended model** and is **NOT the one bundled with the `silly-kicks` wheel**. It is
> published for reproducibility and independent scrutiny of a specific research result. The
> recommended model is the bundled `default` (`public` arm), which ships inside the package
> and needs no download. See [Which variant should I use?](#which-variant-should-i-use).

## Model Description

`XCrossAttemptModel` is a deterministic-XGBoost classifier estimating **P(the in-possession
team attempts a cross within ~1 s of a tracking frame)** &mdash; a **STATE-anchored** framing,
reframed from the sender-level event treatment in Cao et al. (arXiv:2505.11841).

It carries **7 of the paper's 8 confounders** (crosser-position, #7, is omitted &mdash; no
faithful tracking-only proxy) plus a **novel, isolatable GK-position confounder block**, which
is the paper's headline gap and the reason this model exists inside the GKDV research arc
(TF-17 → TF-19).

- **16 features**, goal-relative coordinates via the shared `_geometry` helper
- Domain filter: alive-ball, **wide-area**
- Trained on **1,209,333** rows / **39,766** positives (5.0% positive rate)

## Which variant should I use?

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `default` (`public`) | 17 matches &mdash; SkillCorner + IDSSE, redistributable | **bundled in the wheel** | **Yes** &mdash; this is the shipped model |
| `sc_extended` | + 98 owner-tier SkillCorner matches (179 total) | **this repo** (HF-only) | Only for reproducing the TF-19 probe result below |

`sc_extended` is HF-only because it is trained on **restricted** data and cannot be
redistributed inside the PyPI wheel. Only learned parameters are published here &mdash; **no raw
provider tracking data**, and no artifact from which raw rows can be reconstructed (every
booster leaf aggregates ≥ 14 samples by the binding `min_child_weight` floor).

## Why this variant did NOT ship

Under the **pre-registered fixed-sequence paired test** (`scripts/_paired.py`, spec 4.1), a
candidate ships only if its held-out deltas are positive in ≥ K−1 of K folds **and** the mean
is positive. `sc_extended` recorded deltas of **exactly 0.0 across all 5 folds** &mdash; no
demonstrated improvement &mdash; so the sequence stopped and `public` shipped:

```
"shipped": "public",
"why": "sc_extended failed the rule; the sequence stops (full cannot ship here)"
```

**Do not read "clears the ratio prong" (below) as "is the better model."** The paired test
measures held-out cross-prediction performance; the GK probe measures GK *sensitivity*. A
model can be more GK-sensitive without predicting crosses better. These are different
questions and this variant lost the one that governs shipping.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.1882 (± 0.0102) | base rate 0.0500 |
| Brier | 0.0437 | base-rate Brier 0.0475 |
| Log loss | 0.1659 | &mdash; |

All four acceptance gates pass (`enough_usable_folds`, `pr_auc_gt_base_rate`,
`brier_lt_base_rate_brier`, `log_loss_lt_uniform`). Estimates are **CV, not the shipped fit**.

## The TF-19 result this variant exists to support

The frozen **GK-substitution probe** (`TF19_PROBE_RATIO=2.0`, `TF19_PROBE_ABS_FLOOR=0.01`,
both frozen before the run; `tf19_ready` requires **both** prongs):

| Metric | bundled `public` | **this `sc_extended`** |
|---|---|---|
| `gk_median_abs_delta` | 0.002417 | **0.009697** |
| `nearest_def_median_abs_delta` | 0.001718 | 0.004380 |
| ratio (gk / nearest_def) | 1.41× &mdash; **misses** | **2.21× &mdash; clears** |
| abs floor ≥ 0.01 | 4.1× short | misses by ~10% relative |
| `tf19_ready` | **false** | **false** |

**Both models fail the gate.** They fail differently: the bundled model misses both prongs by
a wide margin; this one misses a single prong by ~10%. Per ADR-037 §4 the verdict is
`gated_clean_fail`, which routes TF-19 to **GK feature engineering** &mdash; explicitly *not*
"no signal."

Probe run out-of-fold on held-out Gradient Sports matches `10502` / `10503`
(`probe_sample_in_training_folds` records `false` for both).

## Usage

```python
from silly_kicks.tracking import XCrossAttemptModel

model = XCrossAttemptModel.from_variant("default")       # recommended, bundled, offline
model = XCrossAttemptModel.from_variant("sc_extended")   # this repo, downloads from the Hub
```

Requires `pip install silly-kicks[xcross]` and **silly-kicks ≥ 4.51.0** (earlier versions have
no `sc_extended` routing).

## Integrity and load-time guards

`load()` is **fail-closed** on two independent checks:

1. **SHA256SUMS** verified before anything is parsed.
2. **Chirality fingerprint** (ADR-040) &mdash; the model re-runs its own outputs on a fixed
   y-asymmetric probe frame and compares to the recorded fingerprint. It raises on a mismatch
   **and on a missing one**, because every pre-PR-2 artifact belongs to the y-mirrored
   mis-served class.

A `base_score` guard also handles the xgboost 3.x bracketed-string serialization that 2.x
silently drops to `0.5`.

## Limitations

- **Not the shipped model.** See above.
- `tf19_ready = false`. Do not build a TF-19 consumer on this surface.
- Trained on a corpus that is **179 matches, heavily Real Madrid** &mdash; the 98 owner-tier
  additions are one club, so club/style confounding is real and unquantified here.
- SkillCorner keepers are detected in only **~19.6%** of frames (~80% interpolated), which is
  why GKDV *measurement* is registered to Gradient Sports frames only (ADR-038 §5).
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Cao et al. "Framing Causal Questions in Sports Analytics: A Case Study of Crossing in
  Soccer." arXiv:2505.11841 (2025).
- Decisions: ADR-011 (trained-model lifecycle), ADR-015 (causal-validation port),
  ADR-037 (TF-19 re-gate), ADR-038 (corpus + visibility), ADR-040 (chirality enforcement).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics, GK-substitution probe, ablation, permutation importance |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
