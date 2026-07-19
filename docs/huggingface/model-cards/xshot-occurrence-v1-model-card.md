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

> **Read this first.** This repo serves the **`sc_extended`** variant, which is **NOT bundled
> with the `silly-kicks` wheel** — it is trained on restricted owner-tier data and cannot be
> redistributed there. That is a **licensing** constraint, not a quality one: `sc_extended`
> **cleared** its pre-registered paired test. Pick the variant that matches your situation —
> see [Which variant should I use?](#which-variant-should-i-use).

## Model Description

`XShotOccurrenceModel` is a deterministic-XGBoost classifier estimating **P(a shot is
attempted by the in-possession team within ~1 s of a tracking frame)** &mdash; the xS surface of
the GKDV research arc (TF-16).

Paper-faithful **27-feature** extractor in goal-relative coordinates via the shared
`_geometry` helper: ball r/θ/z/speed, an `openGoal` goal-mouth obstruction term, GK
distance/bearing, and the 5 nearest defenders + 5 nearest attackers.

- Domain filter: alive-ball, **attacking third**
- Trained on **2,024,373** rows / **366,834** positives (18.9% positive rate)
- `prepare_xshot_training_data` always returns the **faithful** class distribution; negative
  subsampling lives in a separate train-only helper.

## Which variant should I use?

| Variant | Corpus | Where it lives | Use it? |
|---|---|---|---|
| `default` (`public`) | 17 matches &mdash; SkillCorner + IDSSE, redistributable | **bundled in the wheel** | **Default choice** &mdash; offline, fully reproducible, no restricted data |
| `sc_extended` | + 98 owner-tier SkillCorner matches (179 total) | **this repo** (HF-only) | Yes, if you can accept a Hub download and the corpus caveats below — it beat `public` on held-out folds |

`sc_extended` is HF-only because it is trained on **restricted** data and cannot be
redistributed inside the PyPI wheel. Only learned parameters are published here &mdash; **no raw
provider tracking data**, and no artifact from which raw rows can be reconstructed (every
booster leaf aggregates ≥ 9 samples by the binding `min_child_weight` floor).

## Why this variant is HF-only (it did NOT fail its paired test)

**`sc_extended` CLEARED the pre-registered paired test.** It is HF-only because it is trained
on **restricted owner-tier data and cannot be redistributed inside the PyPI wheel** (ADR-038)
— a licensing constraint, not a quality verdict.

The Stage-B run's own record:

```
"shipped": "sc_extended",
"why": "sc_extended clears; full does not dominate it -- ties go to less data"
```

Stage-B deltas vs `public`, held out on public folds: `[-0.008, +0.009, +0.005, +0.029,
+0.023]` — positive in 4 of 5 folds, mean **+0.0117**. The registered rule
(`scripts/_paired.py::clears_rule`: positive in ≥ K−1 of K folds **and** a positive mean)
returns **True**.

### Do not misread the BUNDLED artifact's paired block

The bundled `default` artifact carries a *different* run's record — **Stage A**, whose corpus
contained **no owner-tier SkillCorner rows at all**. There,
`cand_masks["sc_extended"] = is_public | is_sc_private` with `is_sc_private` **empty**, so the
`sc_extended` candidate was the *same mask* as `public`: the same model scored against itself.
Its deltas are `[0.0, 0.0, 0.0, 0.0, 0.0]` **by construction** — a degenerate self-comparison,
not a measured null — and since the rule requires strictly positive values, zero fails and the
fixed sequence stops. That is why `public` is the bundled default.

Reading those zeros as "`sc_extended` failed" is a trap; an earlier revision of this card fell
into it. `docs/research/tf19_pr2/decision_table.md` has the Stage A / Stage B split correct.

The separate 4.9.0 verdict — that owner-tier Gradient Sports data degraded public
held-out PR-AUC in all 5 folds — concerned the `full` arm, not `sc_extended`.

## Held-out CV (5 folds, out-of-fold)

| Metric | Value | Baseline |
|---|---|---|
| PR-AUC | 0.5968 (± 0.0133) | base rate 0.1888 |
| Brier | 0.1110 | base-rate Brier 0.1532 |
| Log loss | 0.3613 | &mdash; |

All four acceptance gates pass. Estimates are **CV, not the shipped fit**.

## TF-19 status: the shot arm has never been measured

Unlike its xCross sibling, this model carries **no GK-substitution probe, no GK-block
ablation and no permutation importance** in `metrics.json`. That is **blocked, not missing**:

- The registered xS probe rule and its locked constants shipped in silly-kicks 4.47.0
  (`tracking/_model_eval.py::evaluate_xs_probe`, `PROBE_WRAPPERS["xs"]`).
- `xs_substitution_probe` consumes **ghost-substituted `targets`** produced by the
  `silly_kicks/gkdv/` engine, which is ADR-037's PR-3 and does not exist yet.

**A TF-19 spec must not assume the shot arm is healthy.** It is *unmeasured*, not validated.

## Usage

```python
from silly_kicks.tracking import XShotOccurrenceModel

model = XShotOccurrenceModel.from_variant("default")       # recommended, bundled, offline
model = XShotOccurrenceModel.from_variant("sc_extended")   # this repo, downloads from the Hub
```

Requires `pip install silly-kicks[xshot]` and **silly-kicks ≥ 4.51.0** (earlier versions have
no `sc_extended` routing).

## Integrity and load-time guards

`load()` is **fail-closed** on two independent checks:

1. **SHA256SUMS** verified before anything is parsed.
2. **Chirality fingerprint** (ADR-040) &mdash; the model re-runs its own outputs on a fixed
   y-asymmetric probe frame and compares to the recorded fingerprint. It raises on a mismatch
   **and on a missing one**, because every pre-PR-2 artifact belongs to the y-mirrored
   mis-served class. This enforcement is what caught the xgboost 3.x `base_score`
   serialization bug, which 2.x silently drops to `0.5`.

## Limitations

- **Not the bundled model** (restricted corpus, so it cannot ship in the wheel). This is a redistribution limit, not a performance one.
- **No GK measurement exists for this arm at all** &mdash; see the TF-19 section.
- Trained on a corpus that is **179 matches, heavily Real Madrid** &mdash; the 98 owner-tier
  additions are one club, so club/style confounding is real and unquantified here.
- SkillCorner keepers are detected in only **~19.6%** of frames (~80% interpolated), which is
  why GKDV *measurement* is registered to Gradient Sports frames only (ADR-038 §5).
- Estimates are cross-validated, not a held-out test of the shipped fit.

## References

See the `NOTICE` file in the silly-kicks repository for full bibliographic citations.

- Attribution: arXiv:2512.00203.
- Decisions: ADR-011 (trained-model lifecycle), ADR-037 (TF-19 re-gate),
  ADR-038 (corpus + visibility), ADR-040 (chirality enforcement).

## Model Files

| File | Purpose |
|---|---|
| `model.json` | XGBoost booster (pickle-free) |
| `metadata.json` | features, hyperparameters, chirality fingerprint, provenance |
| `metrics.json` | CV metrics and acceptance record |
| `SHA256SUMS` | integrity manifest, verified by `load()` |

## More Information

https://github.com/karsten-s-nielsen/silly-kicks
