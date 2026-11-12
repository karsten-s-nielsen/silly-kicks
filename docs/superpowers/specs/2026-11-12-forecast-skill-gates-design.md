# Forecast-skill acceptance gates for xShot and xCross — design

Status: r5 APPROVED by A and B (r5 re-reviews: A no findings; B one CONSIDER, FSG-B13, applied as wording only,
section 16). APPROVED by the owner on 2026-11-12 ("Approve, write plan"). r5 answers the r4 re-reviews (A: APPROVE, FSG-SPEC-16 to 18; B: APPROVE, FSG-B11, B12) and
carries the owner's confirmation of the retirement rule with the corrected evidence (header, section 6.1). r4 answered
the r3 re-reviews (A: APPROVE, FSG-SPEC-15; B: APPROVE, FSG-B09, B10) and carried two owner rulings made after the
section 6 diagnosis: the gate uses nested cross-validation (section 4.1), and the
refit data decides retirement, arm by arm, applied by Part Deux (section 6.1). r3 answered the r2 re-reviews (A: REQUEST
CHANGES, FSG-SPEC-12 to 14; B: REQUEST CHANGES, FSG-B07, B08) and carried the owner's ruling on the dropped-fold rule.
r2 answered the r1 reviews (A: APPROVE,
FSG-SPEC-01 to 11; B: APPROVE, FSG-B01 to B06) and carried four owner rulings (table at the end). Branch
`feat/forecast-skill-gates` on `origin/main` `05f3a692`.

Owner rulings of 2026-11-12:
- "Investigate further first".
- The gold standard for maximum long-term value; a refit of every arm is acceptable.
- "Fix both gates" (Brier and PR-AUC).
- "Measure first, then spec".
- The gate rule "As designed" (section 4), confirmed as the gold standard. It replaces the log-loss gate too (r1
  section 4.4 named all three gates).
- r2: the PR-AUC gate uses the within-fold lift (section 4.2).
- r2: the integrity check uses the same rule as the trainer (section 4.5).
- r2, O1: the failing arm is investigated in this cycle (section 6).
- r2, O2: the four model cards change with the adoption of the refit artifacts, not in this cycle (section 4.5).
- r3: in a bootstrap draw, a fold with no usable weight or a one-class weighted rate is dropped and the other folds
  are re-weighted (section 4.3), "Drop the fold". The question stated the fact: under the fail-closed rule both
  xcross public arms fail `pr_auc_lift_lb95_gt_0` (section 3).
- r4: "Nested CV". The gate scores each fold with params tuned on that fold's training rows only (section 4.1).
  The question stated the cost (5 extra studies per arm) and the risk (a marginal arm may then fail).
- r4: "Retire it", refined by the owner the same day: "do make sure to do it based on data, i.e. consider if Part
  Deux should be the one to conclude things based on the refitting & retraining that it does". So retirement is a
  rule applied by Part Deux to its refit results, for every arm (section 6.1). This spec's measurement of
  `xcross_attempt-public` / `position_only` (section 6) is a prediction, not the decision. Growing the public xCross
  corpus is a separate scope for the owner to schedule.
- r5: "Data rule, refined". The r4 question said the committed wheel WEIGHTS fail the Brier bound. That was wrong:
  the measurement refit the committed PARAMS on each run 9a training fold; the weights were not scored and cannot be
  scored out of sample here (FSG-SPEC-16, FSG-B11). The owner confirmed the rule in a question that stated the
  correction. The rule retires only on a skill gate that fails with finite values; any other failure stops the arm
  for investigation (section 6.1).

## 1. Intent

The xShot and xCross trainers refuse to ship a model that does not beat a trivial forecast. Run 9a (`tf46_refit`,
commit `aadd4cf7`) failed the arm `xcross_attempt-public`, variant `position_only`, on `brier_lt_base_rate_brier`:
fold-mean Brier 0.037276 against the bar 0.036896. The other seven arms passed.

The investigation found that the gate itself is biased. The failure is therefore not evidence that the model is
worse than the base rate. This cycle replaces the Brier, PR-AUC and log-loss gates of both trainers with
forecast-skill gates. The new gates compare the model with an honest reference forecast on the same held-out rows,
and they require the gain to hold under match-level uncertainty. Part Deux then refits all 8 run 9a arms.

The TF-16 spec fixed the old gates before its run ("pre-registered, no tuning to observed", review M5). This record
states plainly that the defect was found because of a gate failure. The section 4 rule was designed before the
measurement of section 3 (the measurement script implements it). Two changes came after the measurement:

- **The PR-AUC reference (r2, section 4.2).** The r1 reviews found a bias against the model; `gold2.py` measured it.
  No arm failed the r1 PR-AUC gate, so this change was not made to pass an arm.
- **The dropped-fold rule (r3, section 4.3).** The within-fold lift made a fold without positives possible in a
  draw. `gold2.py` failed such a draw closed. Under that rule both xcross public arms fail `pr_auc_lift_lb95_gt_0`,
  and `default` passes every other gate. I then wrote the drop rule into r2 after seeing that result, without a
  ruling, and r2 wrongly said no change was made because an arm failed. The r2 reviews caught it. The owner ruled
  for the drop rule in a question that stated that fact (header). `gold3.py` records all three candidate rules.

The arm that fails, `xcross_attempt-public` / `position_only`, fails on the Brier bound. Neither change touches it.

A third change came from the section 6 diagnosis: the HPO and the gate used the same 5 folds, so the gate scored
params tuned on the folds it evaluates. The owner ruled for nested CV (r4). It changes every arm's numbers a little;
section 3 gives the nested values, and they give the same outcome as the non-nested ones.

## 2. The defect

Both trainers compute the same gate (`scripts/train_xshot_occurrence.py:207-215`,
`scripts/train_xcross_attempt.py:208-216`):

- `brier_lt_base_rate_brier`: the unweighted mean of the per-fold Brier scores must be below `base * (1 - base)`.
  `base` is the pooled positive rate of all rows of the arm.
- `pr_auc_gt_base_rate`: the unweighted mean of the per-fold PR-AUC must be above `base`.
- `log_loss_lt_uniform`: the mean per-fold log loss must be below `ln 2`.

**The Brier bar is biased low.** `base * (1 - base)` is the Brier score of a forecaster that knows the pooled rate of
the held-out rows in advance. A cross-validated model cannot know that rate. It learns the training folds' rate, and
the folds differ in rate. On each fold, the Brier score of a constant forecast at the training-fold rate is
`rate_te * (1 - rate_te) + (rate_tr - rate_te)^2`. The cause of the bias is the second term, the squared rate
mismatch, which is never negative. The first term alone has a row-weighted mean at or below `base * (1 - base)`
(concavity), and the unweighted fold mean can move it either way; on `xshot_occurrence-public` the row-weighted
constant Brier is 0.0176316 against the bar 0.0176289. A model with zero skill fails the gate when the mismatch term
dominates. A model with small real skill can fail it too.

Evidence on the failing arm (5 folds, 70,579 rows, 17 matches, read-only DGX measurement):

| Fold | Test rows | Test rate | Model Brier | Constant train-rate Brier |
|---|---|---|---|---|
| 0 | 12,991 | 0.04134 | 0.039140 | 0.039641 |
| 1 | 12,785 | 0.04028 | 0.038741 | 0.038664 |
| 2 | 15,787 | 0.03560 | 0.033719 | 0.034344 |
| 3 | 17,048 | 0.03074 | 0.029773 | 0.029893 |
| 4 | 11,968 | 0.04763 | 0.045006 | 0.045483 |

The constant train-rate forecast scores 0.037605 as a fold mean. That is above the bar of 0.036896, so a constant
forecast fails by 0.00071. The model beats the constant on 4 of 5 folds.

**The PR-AUC bar has the same mismatch, to a smaller degree.** The expected PR-AUC of a random ranking on a fold is
that fold's test rate. Their mean differs from the pooled rate by at most 0.00075 on the 8 run 9a arms. No arm is near
that bar today (the smallest margin is 0.0247), but the gate compares unlike quantities.

**The log-loss gate is vacuous.** `ln 2` is the log loss of a forecast of 0.5. At positive rates of 0.016 to 0.070,
a constant forecast at the base rate scores 0.08 to 0.25. The gate cannot fail a model that is worse than the base
rate.

## 3. Measurement (read-only, DGX, before the implementation)

Two scripts, run on the DGX (arm64) at commit `f8a7c47` against local copies of the run 9a inputs (from bucket
`silly-kicks/hf-refit-scratch`, now held under `~/tf46-diag/gold/` on the DGX):

- `D:\Temp\hfu-topup\diag\gold.py` (r1): it refits every fold exactly as the trainer's `cv` stage does (same fold
  plan, pinned params, `base_score` from the training fold, negative subsampling of the training fold only), keeps
  the out-of-fold predictions, and computes the r1 quantities. Report `gold_report.json`.
- `D:\Temp\hfu-topup\diag\gold2.py` (r2): the same refit. It adds the within-fold PR-AUC lift, the pooled PR-AUC of
  the zero-skill reference, the degenerate-draw counts and every no-gain fraction. It uses the call sequence that
  section 4.3 pins, but fails a draw closed when a fold has no positive. It saves each arm's out-of-fold arrays.
  Report `gold2_report.json`.
- `D:\Temp\hfu-topup\diag\gold3.py` (r3): no refit; it reads the arrays `gold2.py` saved. It implements sections 4.2
  and 4.3 as written, with the section 4.5 key names, and also reports the PR-AUC lower bound under the three
  candidate fold rules (drop, count as 0, fail closed). Report `gold3_report.json` (non-nested; the r1 to r3 table
  below). r4 adds the two dropped-fold counts.
- `D:\Temp\hfu-topup\diag\o1.py` (r3): the section 6 diagnosis of the failing arm. Report `o1_report.json`.
- `D:\Temp\hfu-topup\diag\nested.py` (r4): the nested measurement. For each arm and gate fold it runs the trainer's
  own `_hpo_once` (50 trials) on the fold's training rows, fits at those params as the `cv` stage fits a fold, and
  scores the fold. It saves `nested_oof_single.npz` and `nested_params.json` per arm (copies of the params:
  `D:\Temp\hfu-topup\diag\nested_params\<arm>__<variant>.json`). `gold3.py` on those arrays gives
  `nested_report.json`, the reference of section 8.

**Self-check.** `gold.py` fixed its check before the run: the fold-mean Brier must equal the run 9a `cv` brier within
1e-12. That check FAILED on all 8 arms. The differences are 9.1e-11 to 1.49e-9 (maximum relative 2.4e-8). The run 9a
values came from HF `cpu-upgrade` jobs (x86); the measurement ran on the DGX (arm64). The cause is inferred to be
floating-point differences between the two platforms; it is not verified. The size does not change any number in the
table below at its printed precision.

**Non-nested (r1 to r3; params from the run 9a full-rows study; `gold3_report.json`):**

| Arm | Variant | OOF rows | Matches | Brier skill | lb Brier gain | LL skill | lb LL gain | WF PR-AUC lift | lb WF lift |
|---|---|---|---|---|---|---|---|---|---|
| xshot_occurrence-public | default | 127,513 | 17 | +0.03970 | +0.0005098 | +0.16116 | +0.012481 | +0.07088 | +0.05978 |
| xshot_occurrence-public | position_only | 127,513 | 17 | +0.03875 | +0.0004953 | +0.15857 | +0.012238 | +0.06820 | +0.05727 |
| xshot_occurrence-sc_extended | default | 711,747 | 115 | +0.07540 | +0.0010822 | +0.22721 | +0.017988 | +0.12845 | +0.11775 |
| xshot_occurrence-sc_extended | position_only | 711,747 | 115 | +0.06997 | +0.0010109 | +0.21285 | +0.016809 | +0.12096 | +0.11158 |
| xcross_attempt-public | default | 70,579 | 17 | +0.02117 | +0.0000822 | +0.06287 | +0.004542 | +0.04841 | +0.02570 |
| xcross_attempt-public | position_only | 70,579 | 17 | +0.00889 | **−0.0000404** | +0.03211 | +0.001705 | +0.02446 | +0.01171 |
| xcross_attempt-sc_extended | default | 407,076 | 115 | +0.07752 | +0.0042351 | +0.12925 | +0.029518 | +0.13826 | +0.12720 |
| xcross_attempt-sc_extended | position_only | 407,076 | 115 | +0.03898 | +0.0021615 | +0.07322 | +0.016716 | +0.07987 | +0.07333 |

"lb" is the 5th percentile over 2,000 match-bootstrap draws (section 4.3). "WF" is the within-fold lift (section 4.2).
Every value is in `gold3_report.json` (`skill` key of each arm).

Further measured values (gold2 and gold3):

- The fraction of draws with no gain: Brier 0.034 (`xcross public default`) and 0.0695 (`xcross public
  position_only`), 0 elsewhere; log loss 0.005 (`xcross public position_only`), 0 elsewhere; within-fold lift 0 on all
  8 arms.
- One `xcross_attempt` match has no positive (one in the public arms, one in `sc_extended`). In 51 of the 2,000 draws
  of each xcross public arm, one fold holds only that match and so has no positive (`n_one_class_fold_draws` = 51).
  PR-AUC lower bound for `default` / `position_only` under the three rules (gold3): drop +0.0257 / +0.0117; count as
  0 +0.0254 / +0.0117; fail closed NaN / NaN (the gate fails). The other 6 arms have no such draw and the same bound
  under all three rules. No draw of any arm has no usable fold. The owner ruled for "drop" (header).
- The pooled PR-AUC of r1 is biased against the model. A zero-skill forecast at the training-fold rate gets a pooled
  lift of −0.00027 to −0.00357 instead of 0 (FSG-B01, FSG-SPEC-10). Its within-fold lift is exactly 0.

**Nested (r4; the gate as specified; `nested_report.json`, the reference of section 8):**

| Arm | Variant | OOF rows | Matches | Brier skill | lb Brier gain | LL skill | lb LL gain | WF PR-AUC lift | lb WF lift |
|---|---|---|---|---|---|---|---|---|---|
| xshot_occurrence-public | default | 127,513 | 17 | +0.03787 | +0.0004680 | +0.15787 | +0.012215 | +0.06930 | +0.05810 |
| xshot_occurrence-public | position_only | 127,513 | 17 | +0.03575 | +0.0004390 | +0.15324 | +0.011807 | +0.06577 | +0.05589 |
| xshot_occurrence-sc_extended | default | 711,747 | 115 | +0.07386 | +0.0010582 | +0.22466 | +0.017776 | +0.12578 | +0.11560 |
| xshot_occurrence-sc_extended | position_only | 711,747 | 115 | +0.06854 | +0.0009854 | +0.21014 | +0.016593 | +0.11904 | +0.10980 |
| xcross_attempt-public | default | 70,579 | 17 | +0.02148 | +0.0001611 | +0.06298 | +0.004724 | +0.04789 | +0.02534 |
| xcross_attempt-public | position_only | 70,579 | 17 | +0.00855 | **−0.0000139** | +0.03186 | +0.001923 | +0.02289 | +0.01102 |
| xcross_attempt-sc_extended | default | 407,076 | 115 | +0.07688 | +0.0041979 | +0.12859 | +0.029362 | +0.13733 | +0.12630 |
| xcross_attempt-sc_extended | position_only | 407,076 | 115 | +0.03796 | +0.0020963 | +0.07145 | +0.016301 | +0.07808 | +0.07168 |

Nested no-gain fractions: Brier 0.016 (`xcross public default`) and 0.057 (`xcross public position_only`), 0
elsewhere; log loss 0.0015 (`xcross public position_only`), 0 elsewhere; within-fold lift 0 on all 8 arms. Dropped
fold-draws (nested and non-nested alike): `n_empty_fold_draws` 274 (xshot public arms) and 282 (xcross public arms),
0 on the 115-match arms; `n_one_class_fold_draws` 51 on the xcross public arms, 0 elsewhere.

Nested CV moves the Brier skill by at most 0.0030 against the non-nested values (`xshot public position_only`,
+0.03875 to +0.03575). On the 115-match arms it lowers every value a little, as expected when tuning no longer sees the scored fold. On `xcross public default` it
raises the Brier bound (+0.0000822 to +0.0001611), because each fold gets params tuned for its own training rows.

Expected outcome under section 4 (nested): 7 arms pass all seven gates. `xcross_attempt-public` / `position_only`
fails one gate (`brier_gain_lb95_gt_0`, lower bound −0.0000139). It has positive skill on all three scores, but its
Brier gain is not shown over 17 matches (5.7% of the draws show no gain). This is a prediction for Part Deux's refit
(section 6.1); the refit decides.

## 4. The gates

### 4.1 Nested, pooled out-of-fold scoring

**Nested CV (owner ruling, r4).** The gate fold plan stays as today (`StratifiedGroupKFold`, `n_splits =
max(2, min(5, n_matches))`, `shuffle=True`, `random_state=42`, on the arm's rows). For gate fold `k`, the params come
from a study run on that fold's TRAINING rows only (the arm's rows outside fold `k`). The study is the trainer's
existing `_hpo_once`: the same objective, search space, trial count, sampler seed and inner fold plan (the
objective's own `cv_fold_plan` on those rows). Fold `k`'s held-out rows never enter the study that tunes the model
scoring them. The arm's full-rows study (today's `study` unit for the arm) still gives the params of the shipped fit.

Why: a gate that scores params tuned on its own folds overstates skill (Varma and Simon 2006; Cawley and Talbot
2010). Nested CV estimates the skill of the whole procedure, tuning included, which is what ships.

The `cv` stage refits the arm on each fold at that fold's nested params. Each usable fold yields, for every held-out
row:

- `p`: the model's predicted probability (as today);
- `r`: the reference forecast, the TRUE positive rate of that fold's training rows (before any negative
  subsampling);
- the fold index and the row's match label (the arm's `groups` value).

A fold that is skipped today (one class in its training rows, before or after subsampling) contributes no rows.
Every other row of the arm is scored exactly once. The pooled rows are in ascending arm row position.

### 4.2 Point quantities

With `y` the labels of the pooled rows and `clip(q) = min(max(q, 1e-15), 1 - 1e-15)`:

- **Brier.** `brier_model = mean((y - p)^2)`, `brier_ref = mean((y - r)^2)`, `brier_skill = 1 - brier_model /
  brier_ref`.
- **Log loss.** `log_loss_model = mean(-(y ln clip(p) + (1 - y) ln(1 - clip(p))))`, `log_loss_ref` the same with
  `r`, `log_loss_skill = 1 - log_loss_model / log_loss_ref`.
- **PR-AUC, within-fold lift.** For each fold `k` with both classes among its held-out rows:
  `lift_k = AP(y_k, p_k) - rate_k`. `AP` is scikit-learn's `average_precision_score`, `rate_k = mean(y_k)`. Then
  `pr_auc_lift = sum_k n_k lift_k / sum_k n_k` over those folds, with `n_k` the fold's held-out row count.

Why within-fold: each fold model's predictions carry its own offset (`base_score` = its training rate). On these arms
the training rate moves against the test rate, so a pooled ranking mixes offsets that rank the low-rate folds first.
Within one fold, a constant forecast ties every row, so `AP = rate_k` and its lift is exactly 0. The reference for
PR-AUC is thus the expected average precision of a random ranking inside each fold.

Also reported, not gated: `pr_auc_pooled = AP(y, p)` and `positive_rate_pooled = mean(y)`.

### 4.3 Match-cluster bootstrap

Rows within a match are not independent. The bootstrap resamples matches, not rows. The rule is pinned, so it is
pre-registered and leaves nothing to the implementer:

```python
labels, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)   # M = len(labels)
rng = np.random.default_rng(0)
for b in range(2000):                                   # in this order, one call per draw
    idx = rng.integers(0, M, M)
    w = np.bincount(idx, minlength=M)[inv].astype(np.float64)
    brier_gain[b]    = (w * (bref - bmodel)).sum() / w.sum()      # per-row Brier terms of r and p
    log_loss_gain[b] = (w * (lref - lmodel)).sum() / w.sum()      # per-row log-loss terms
    pr_auc_lift[b]   = within_fold_lift(y, p, fold, w)
lb = np.percentile(gain, 5)                             # default linear interpolation; NaN propagates
```

`within_fold_lift(y, p, fold, w)`: for each fold `k`, `s_k = sum(w_k)` and `rate_k = sum(w_k y_k) / s_k`. A fold with
`s_k = 0` or `rate_k` of 0 or 1 is not usable in that draw and is dropped. For the usable folds, `lift_k =
AP(y_k, p_k, sample_weight=w_k) - rate_k`, and the draw's lift is `sum s_k lift_k / sum s_k`. A draw with no usable
fold gives NaN; NaN propagates through `np.percentile` and fails the gate.

Two counts are reported, summed over all draws and folds:

- `n_empty_fold_draws`: folds with `s_k = 0` (no drawn match), dropped;
- `n_one_class_fold_draws`: folds with `s_k > 0` and `rate_k` of 0 or 1, dropped.

A fold of 3 of 17 matches gets no drawn match with probability `(14/17)^17`, about 0.037 per draw, so
`n_empty_fold_draws` is large on the 17-match arms (section 3). `n_one_class_fold_draws` counts the cases the
owner's drop rule was ruled on.

The weighted `AP` is computed from one descending sort of `p` per fold, reused for every draw, with scikit-learn's
tie handling (distinct thresholds). A test pins it to `average_precision_score(..., sample_weight=w)` within 1e-12.

Also reported: the fraction of draws with `brier_gain <= 0`, `log_loss_gain <= 0` and `pr_auc_lift <= 0`.

**What the bound means.** It is an approximate one-sided 95% bound. It holds the five fitted fold models fixed, so it
covers the sampling of held-out matches only, not refit or fold-assignment variance. A percentile bootstrap over 17
clusters tends to under-cover. The gate is a pre-registered decision rule; it is not an exact confidence statement.

### 4.4 The acceptance block

Each gate is strict and NaN-safe (a NaN value fails).

| Key | Passes when |
|---|---|
| `enough_usable_folds` | `n_usable_folds >= 2` (unchanged) |
| `brier_skill_gt_0` | `brier_skill > 0` |
| `log_loss_skill_gt_0` | `log_loss_skill > 0` |
| `pr_auc_lift_gt_0` | `pr_auc_lift > 0` |
| `brier_gain_lb95_gt_0` | the lower bound of `brier_gain > 0` |
| `log_loss_gain_lb95_gt_0` | the lower bound of `log_loss_gain > 0` |
| `pr_auc_lift_lb95_gt_0` | the lower bound of `pr_auc_lift > 0` |

The keys `pr_auc_gt_base_rate`, `brier_lt_base_rate_brier` and `log_loss_lt_uniform` leave the acceptance block.

Why all seven are required: Brier and log loss are proper scoring rules with different tail weight. PR-AUC measures
the ranking of the rare positive class. Requiring every lower bound is an intersection-union test. It adds no
multiplicity inflation, so the combined rule is at least as strict as each bound (approximate, section 4.3).

The three point gates alone could pass on float noise: a zero-skill XGBoost model predicts `base_score` through a
logit and a sigmoid. The lower-bound gates catch that case.

`B`, the seed, the level and the clip are named constants (section 5). No CLI flag, campaign field or config value
changes them.

### 4.5 The metrics shape and its consumers (Hyrum's law)

The existing per-candidate metric keys stay, with unchanged values: `pr_auc`, `brier`, `log_loss`, `pr_auc_std`,
`positive_rate`, `base_rate_brier`, `n_usable_folds`. The adopt `compare` step, the integrity reproduction tripwire
(`reproduces` reads `pr_auc` and `positive_rate`) and the committed research verdicts keep working.

Each candidate's metrics gain one additive key, `skill`:

```json
"skill": {
  "reference": "train_fold_true_rate",
  "n_oof_rows": 70579, "n_matches": 17,
  "brier_model": 0.0, "brier_ref": 0.0, "brier_skill": 0.0,
  "log_loss_model": 0.0, "log_loss_ref": 0.0, "log_loss_skill": 0.0,
  "pr_auc_lift": 0.0, "pr_auc_pooled": 0.0, "positive_rate_pooled": 0.0,
  "bootstrap": {
    "unit": "match", "draws": 2000, "seed": 0, "level": 0.95,
    "brier_gain_lb": 0.0, "log_loss_gain_lb": 0.0, "pr_auc_lift_lb": 0.0,
    "frac_brier_gain_le_0": 0.0, "frac_log_loss_gain_le_0": 0.0, "frac_pr_auc_lift_le_0": 0.0,
    "n_empty_fold_draws": 0, "n_one_class_fold_draws": 0
  },
  "nested": true
}
```

**The existing keys change value, not meaning.** With nested CV, `pr_auc`, `brier`, `log_loss` and `pr_auc_std` are
still the per-fold means of the gate's folds, now at each fold's nested params. Their values move a little
(section 3). Consumers that compare them across runs: `reproduces` (relative tolerance 0.05 on `pr_auc`) and the adopt
`compare` step. After the refit, every committed artifact carries nested values, so like compares with like. Before
the adoption, a rematerialize campaign would compare nested `pr_auc` with committed non-nested values. Measured shift
of the fold-mean `pr_auc` (non-nested to nested, run 9a arms): +0.03% to −3.32% relative (largest:
`xshot_occurrence-public` / `position_only`, 0.08790 to 0.08498), inside the 0.05 relative tolerance of `reproduces`
and `PUBLIC_CV_REL_TOL` (`scripts/hf_jobs/floor.py:57`). Source: `D:\Temp\hfu-topup\diag\prauc_shift.py`, report
`prauc_shift.json`. ADR-131 records the change and the shift. Each candidate's metrics also gain `fold_params`: the nested params of each gate fold, in fold
order, for audit.

The `acceptance` block changes its keys as in section 4.4. Consumers, enumerated with evidence:

- `scripts/hf_jobs/adopt.py:782-790` (refit certification): reads `all(acceptance.values())`, not key names. No
  change.
- `scripts/train_xshot_occurrence.py:878,937` and `scripts/train_xcross_attempt.py:1051,1190`: call `_gates`. They
  move to the shared gate function.
- `scripts/validate_trained_artifact_integrity.py:84-101` (`verdict`), required by `is_go` (`:110`); it runs in adopt's
  rematerialize mode (`adopt.py:796-798`). Owner ruling: the same rule as the trainer. `lift_ok` = `pr_auc_lift_gt_0`
  and `pr_auc_lift_lb95_gt_0`. `calibration_ok` = `brier_skill_gt_0`, `log_loss_skill_gt_0`, `brier_gain_lb95_gt_0`
  and `log_loss_gain_lb95_gt_0`. Both come from `skill_gates`, so the integrity verdict and the trainer's acceptance
  cannot disagree. The key names stay (Hyrum); ADR-131 records that `calibration_ok` now means proper-score skill
  (calibration and resolution), not calibration alone. A metrics dict without `skill` fails both (fail-closed).
  `enough_folds` (`_MIN_USABLE_FOLDS = 5`) is unchanged.
- `scripts/compare_position_only_variants.py`: flattens `acceptance` but compares only `pr_auc`, `brier`,
  `log_loss` (`:22-25`). No change.
- Tests that change with the gate: `tests/tracking/test_xshot_occurrence_integration.py:471-474`,
  `tests/scripts/hf_jobs/test_adopt.py:1201-1220`, `tests/tracking/test_xcross_attempt_integration.py:131,174,294`,
  `tests/scripts/test_validate_trained_artifact_integrity.py`. Section 7 states the rule for the toy corpora.
- The degenerate toy (`scripts/_toy_corpora.py:445`: "no model beats the base rate and the acceptance gates must
  fail") and its tests (`tests/tracking/test_xshot_occurrence_integration.py:589-598`,
  `tests/tracking/test_xcross_attempt_integration.py:184-191`, `only.metrics_FAILED.json`). They must still fail,
  now on a skill gate: the tests assert that at least one of the three lower-bound gates is False.
- Committed artifacts and records keep their old acceptance blocks until a refit replaces them:
  `silly_kicks/tracking/_xshot_weights/*/metrics.json`, `silly_kicks/tracking/_xcross_weights/*/metrics.json`,
  `docs/artifacts/observed.json`, `docs/research/trained_artifact_integrity/verdicts/*.json`. They are historical
  records of what ran. This cycle does not edit them.
- Model cards that name the four old gates: `docs/huggingface/model-cards/xshot-occurrence-v1-model-card.md`,
  `xcross-attempt-v1-model-card.md`, `xshot-occurrence-position-only-v1-model-card.md:65`,
  `xcross-attempt-position-only-v1-model-card.md:62`. Owner ruling (O2): they change with the adoption of the refit
  artifacts (Part Deux, PR #302), because they describe the published artifacts. The hand-off message to Part Deux
  names the four cards.

## 5. Where it lives

One implementation for both trainers and the integrity gate. It extends the existing seam
`silly_kicks/tracking/_artifact_integrity.py` (the construct-validity checks of trained-artifact metrics, ADR-126).
No new module.

- Constants: `SKILL_BOOTSTRAP_DRAWS = 2000`, `SKILL_BOOTSTRAP_SEED = 0`, `SKILL_LEVEL = 0.95`,
  `SKILL_LOG_CLIP = 1e-15`.
- `forecast_skill(y, p, r, fold, groups) -> dict`: the section 4.2 and 4.3 quantities, as the `skill` dict of
  section 4.5.
- `skill_gates(metrics) -> dict`: the section 4.4 acceptance block from one candidate's metrics dict.

The trainers' `_gates` functions are deleted. Their call sites call `skill_gates`.

**The nested studies.** The `prep` stage adds one study tag per cv arm and gate fold, `{arm}_gf{k}` for
`k < cv_arms[arm]` (`single_gf{k}` on the non-paired path), to `study_tags`. `_study_rows` (both trainers) resolves
such a tag to the arm's rows outside gate fold `k`, from the same gate fold plan the `cv` stage walks, in ascending
row order. The `study` stage runs them as it runs every tag today (ADR-128 units, InnerPool fold map, resume). The
`_gf` suffix does not collide with the paired path's `{name}_f{fold}` tags, which use the public folds.

**The fold seam.** The `cv` fold unit for arm `a`, fold `k` reads its params from study `{a}_gf{k}`, not from the
arm's full-rows study. `_cv_metrics_fold` (xShot) and `_xcross_eval._cv_score_fold` (xCross) keep their fold logic
and also return the held-out row positions, `p` and `r`. Each `cv` fold unit returns a `FileSet`, as `train_xsuccess`
already does for unit arrays:

- `fold.json`: today's JSON result plus the training-fold true rate;
- `te.npy` (int64): the held-out row positions within the arm;
- `p.npy` (float64): the predictions;
- `groups.npy` (fixed-width unicode, `allow_pickle=False`): the held-out rows' match labels as strings.

The `cv` reduce reads only the unit results, as today; it does not load the prep inputs or the design matrix. It
combines the folds exactly as today (the existing keys are byte-identical) and adds `skill` from `forecast_skill`,
with the rows in ascending `te` order.

The non-staged compositions `_cv_metrics` (both trainers) and `_xcross_eval._cv_score` stay the composition of plan,
fold and combine, and stay exact-equality tested against the staged path at the same per-fold params. `gk_block_ablation` reads only `pr_auc`
and `log_loss`, so it is not affected. The plan decides whether the ablation's two `_cv_score` calls skip the
bootstrap; the ablation's reported keys must not change.

**Cost.** Nested CV adds `cv_arms[arm]` studies per cv arm (5 on every run 9a arm, 40 in all). Measured on the DGX,
one study per process, single-threaded (`nested.py`): 64 to 259 s on the public arms, 511 to 1,250 s on
`xcross_attempt-sc_extended`, and 2,021 to 2,620 s on `xshot_occurrence-sc_extended`. On HF they are ordinary
`study` units and run in parallel with the full-rows studies. The DGX times size nothing on HF: the launch wall-time
estimate comes from Part Deux (40 more study units, which run before the `cv` stage can start). The fold units now return FileSets (four members each). The reduce runs 2,000 draws per arm, each with up
to 5 weighted AP evaluations, over up to 711,747 rows. Section 8 measures the reduce time and reports it.

The fold plan, `base_score`, negative subsampling, the final fit, the ship decision and the paired nested deltas do
not change. The HPO objective, search space and trial count do not change (section 6).

## 6. The failing arm (owner ruling O1: in this cycle)

Under section 4, `xcross_attempt-public` / `position_only` still fails `brier_gain_lb95_gt_0`. The owner ruled that
its investigation is part of this cycle, so that Part Deux refits all 8 arms once.

Lead, inferred and not verified: its run 9a study chose `n_estimators` 53, `max_depth` 4, `learning_rate` 0.029,
`min_child_weight` 14.4, `reg_lambda` 4.36. The forecast then moves little from `base_score`. The committed
`position_only` weights used about 258 trees at depth 2 (`silly_kicks/tracking/_xcross_weights/position_only/metrics.json`).

The investigation is read-only, on the DGX, on the local run 9a inputs:

1. Read the arm's study: the trial history and how the HPO objective scores a candidate.
2. Score the committed params and the study's best params on the run 9a folds under section 4.
3. Find why the study chose its params (objective, search space, trial count or the data).

Caution: the HPO objective and the gate use the same 5 folds (`_xcross_attempt_objective.cv_fold_plan` and
`_xcross_eval._cv_score_plan`, both `StratifiedGroupKFold(shuffle=True, random_state=42)`). Params chosen by their
scores on these folds are selected on the gate's own folds. Step 2 diagnoses whether the HPO found its own optimum;
it is not a param search for shipping. Script `D:\Temp\hfu-topup\diag\o1.py`, report `o1_report.json`.

**Finding (run 9a folds, non-nested, 74 param settings; `o1_report.json`).**

- The study did not quite reach its own optimum. Its best fold-mean log loss (the HPO objective) is 0.160146; the
  best grid setting reaches 0.159409 (0.46% lower). This fits 50 TPE trials over 5 parameters. It is not a defect.
- The settings best on the HPO objective still fail the Brier bound: lower bounds −0.0000031 and −0.0000012
  (`n_estimators` 200, `max_depth` 2, `learning_rate` 0.02), Brier skill +0.0105.
- 2 of 74 settings pass all three bounds (`n_estimators` 100, `max_depth` 2, `learning_rate` 0.02). They rank 7th
  and 8th on the HPO objective. To choose them for passing would select on the gate's own folds.
- The committed params (from `silly_kicks/tracking/_xcross_weights/position_only/metrics.json`), refit on each run 9a
  training fold like any grid setting, fail the Brier bound: lower bound −0.0001401, 14.4% of draws with no gain.
  The committed weights themselves were not scored. They cannot be scored out of sample on this corpus, because they
  were trained on these public matches.
- Nested CV (section 3) gives the same result: Brier lower bound −0.0000139.

Conclusion: the cause is the data, not the HPO. On 17 public matches, the arm's Brier gain over the training-fold
rate (about 1%) cannot be shown at the 95% level. The HPO objective, search space and trial count do not change.

### 6.1 Retirement rule, applied by Part Deux to the refit data (owner ruling, r4)

**The rule.** Part Deux refits all 8 arms at the commit of this cycle (each launch needs the owner's go). For each
arm, the refit's own acceptance block decides:

- **Every gate passes:** the refit is a candidate for adoption, as today (ADR-130 refit certification).
- **A skill gate fails on data** (the trainer writes `only.metrics_FAILED.json` and no artifact; at least one of the
  six skill gates of section 4.4 is False; every numeric value in the `skill` dict is finite; `enough_usable_folds`
  is True):
  the refit is not adopted, and the committed artifact that the arm would replace is retired at the adoption. The
  FAILED metrics file is the evidence.
- **Any other failure** (`enough_usable_folds` False, a NaN or infinite numeric `skill` value, a missing `skill`
  dict, or a job error without `only.metrics_FAILED.json`; a gate failure also exits non-zero,
  `AcceptanceGatesFailedError`, so the FAILED file is what tells the two apart): the arm stops. It is not adopted and nothing is retired. Part Deux reports it to the owner for
  investigation, because such a failure points to a defect or a corpus problem, not to the model's skill.

Part Deux concludes from its refit results, not from this spec's measurement. Section 6 predicts that only
`xcross_attempt-public` / `position_only` fails, and that its committed wheel artifact
`silly_kicks/tracking/_xcross_weights/position_only/` is then retired. If the refit outcome differs from the
prediction, the refit outcome rules. Part Deux reports the per-arm outcome (the seven gate values and the `skill`
dict of every arm) to the owner with the adoption PR, and the owner approves it as any merge.

Why the committed artifact goes too: the refit runs the shipping procedure (data, tuning, gate) on the current
corpus. If that procedure cannot show skill, an older artifact from an older corpus is not evidence of skill
either. One gate applies to everything shipped. (The committed weights cannot be scored out of sample on the current
corpus, which contains their training matches; the committed params, refit on the run 9a folds, fail the Brier bound,
section 6. The rule does not depend on that measurement.)

What serves after a removal: for `xcross_attempt` `position_only`, `XCrossAttemptModel.from_variant("position_only")`
and the velocity auto-select already handle a missing bundle (`silly_kicks/tracking/_xcross_attempt.py:881-947`).
They warn and return NaN for velocity-less frames, never the `default` model (ADR-067). No serving logic changes.
If the refit fails a different arm, Part Deux enumerates that artifact's consumers the same way before removal.

The checklist for the predicted retirement, enumerated with evidence (this cycle writes no code for it):

- Artifact records: `docs/artifacts/manifest.toml:132` (the `xcross_position_only` entry) and `:432` (`wheel_sibling` of
  `hub_xcross_sc_extended_position_only`); `docs/artifacts/observed.json`.
- Campaign registry: `scripts/hf_jobs/registry.py:283-289` (the `position_only` `VariantBuild` of `xcross_attempt`).
- Tests: `tests/tracking/test_position_only_bundled.py` (`:48,73,99,125,167,173,174`),
  `tests/tracking/test_velocity_variant_resolution.py` (`:190,192,199,373`), `tests/tracking/test_feature_contract.py:424`,
  `tests/tracking/test_declared_constant_values.py:29`, `tests/test_bundled_probe_ids_are_committed_ids.py:20`,
  `tests/tracking/test_xcross_attempt_velocity_contract.py:144`, `tests/scripts/hf_jobs/test_corpus_scope_gate.py:82,87`.
- Scripts that read the bundle by path: `scripts/validate_xcross_causal.py:270`.
- Docs: `docs/huggingface/model-cards/xcross-attempt-position-only-v1-model-card.md` (the Hub `sc_extended`
  artifact stays; its card loses the wheel sibling), `docs/context/trained-models.md`, `CHANGELOG.md`.
- Downstream (Hyrum): `docs/PRIVATE_CONSUMERS.md:28`. SB360 and other velocity-less frames flip from a real
  position-only xCross value back to NaN with a warning. The CHANGELOG entry states it as a behavior change.

The Hub artifact `silly-kicks/xcross-attempt-position-only-v1` (`sc_extended`, owner tier) is predicted to stay: its
arm passes every gate in section 3. Its refit decides.

## 7. Tests (TDD; each test is written and seen to fail first)

1. **The defect, first.** A constant forecast at the training-fold rate, on a synthetic grouped corpus with unequal
   fold rates, fails today's `brier_lt_base_rate_brier`. Under `skill_gates` its `brier_skill` and `log_loss_skill`
   are exactly 0, its `pr_auc_lift` is exactly 0, and the three point gates are False. The test shows the old bar's
   bias and the neutrality of all three references.
2. **Skill detects skill.** A forecast that uses an informative feature on a synthetic corpus passes all seven gates.
3. **Bootstrap bounds react to the sample size.** The same per-match effect with few matches gives a lower bound at
   or below 0; with many matches, above 0.
4. **Weighted AP.** The single-sort `AP` equals `average_precision_score(y, p, sample_weight=w)` within 1e-12 on
   random draws, with ties in `p` and zero weights.
5. **Pinned sequence.** On a fixed small input, `forecast_skill` returns the lower bounds computed by the section 4.3
   reference code, exactly.
   Owner ruling 2026-11-12 (implementation review FSG-IB01): the Brier and log-loss bounds compare exactly; the PR-AUC
   bound compares within 1e-12, the single-sort AP tolerance of Test 4.
6. **Determinism.** Equal inputs give byte-equal JSON. The `cv` reduce gives the same `skill` for any order of its
   unit results.
7. **NaN and dropped folds.** A NaN skill fails its gate. A draw with one fold without positives drops that fold and
   adds 1 to `n_one_class_fold_draws`; a fold with no drawn match is dropped and adds 1 to `n_empty_fold_draws`; a
   draw with no usable fold makes `pr_auc_lift_lb95_gt_0` False. `enough_usable_folds` is unchanged.
8. **Hyrum.** At the same per-fold params, the existing candidate metric keys from the staged path are byte-identical
   to the non-staged composition on the staged-against-legacy fixtures (`tests/tracking/test_objective_fold_split.py`).
9. **ADR-128 parity.** The W1N1 against W2N2 parity test of both trainers stays green with the new `cv` unit result.
10. **Integrity gate.** `verdict` computes `lift_ok` and `calibration_ok` from `skill_gates` with the section 4.5
    mapping, and fails closed without `skill`.
11. **Degenerate toy.** Both trainers still refuse the degenerate toy, and at least one lower-bound gate is False.
12. **Nested study tags.** `prep` emits `{arm}_gf{k}` for every cv arm and `k < cv_arms[arm]`; `_study_rows` resolves
    each to the arm's rows outside gate fold `k`, ascending, and these rows never include a row of fold `k`.
13. **Nested fold params.** The `cv` unit `{arm}__{k}` fits at the params of study `{arm}_gf{k}`, and the shipped fit
    still uses the arm's full-rows study. A test with two distinct fold studies shows each fold uses its own params.
14. **No collision.** On the paired path, the `_gf` tags and the `{name}_f{fold}` tags are disjoint and each resolves
    to its own rows.

**The toy corpora (FSG-B04, FSG-SPEC-04).** These tests assert that all gates pass:
`tests/tracking/test_xshot_occurrence_integration.py:471-474` (real data, `max_per_provider=3`) and `:536-550`
(`test_train_script_smoke`, `n_games=4`), `tests/tracking/test_xcross_attempt_integration.py:131,174`
(`n_games=4`), and the parity and wiring tests on the default toy (`XSHOT_TOY_PROVIDERS`, 8 matches,
`scripts/_toy_corpora.py:479`). Rule:

- The plan's first task runs each of these corpora under the new gates on the DGX and records the seven values.
- A test whose corpus passes keeps its assertion that all gates pass.
- A test whose corpus fails gets a stronger toy signal or more toy games, not a weaker assertion. The tests that pin
  that toy change with it, in the same task.
- If neither is feasible for a test, it goes to the owner before the plan is final. No assertion is weakened without
  an owner ruling.

## 8. Validation (before the commit, DGX, read-only)

Run both trainers' new nested `study` units (`{arm}_gf{k}`), `cv` fold units and `cv` reduce on the local run 9a
inputs of all 8 arms. Two checks:

1. **Nested params.** Each `{arm}_gf{k}` study gives the params of the matching fold in the arm's
   `nested_params.json` (written by `nested.py`, which calls the trainer's own `_hpo_once` on the same rows). The TPE
   sampler is seeded and independent of the study tag, so the params must be equal.
2. **Skill.** Every value of the new `skill` dict equals the same key of the arm's `skill` dict in
   `nested_report.json` (`gold3.py` on `nested.py`'s out-of-fold arrays) within 1e-9. Integer keys (`n_oof_rows`,
   `n_matches`, `n_empty_fold_draws`, `n_one_class_fold_draws`) and string keys must be equal. `gold3.py` uses the
   section 4.5 key names, so no mapping is needed. Its extra key `_rule_comparison_pr_auc_lift_lb` is a record only
   and has no counterpart; the new key `nested` has no counterpart in `gold3.py`.

Both runs are on the DGX, so this is a DGX-against-DGX comparison; the 1.5e-9 of section 3 is the DGX-against-HF
difference and does not apply. `gold3_report.json` (non-nested) stays as the record of the r1 to r3 measurement.
The acceptance blocks must give the section 3 outcome. Report the reduce time per arm. No HF job. The CI-faithful
pytest invocation and the lint trio run on the DGX copy of the branch; CI is authoritative.

## 9. Records

- ADR-131: the defect, the gold-standard gates, nested CV, the measurement (including the failed 1e-12 self-check),
  the order of events (section 1), the meaning of the bound (section 4.3), the integrity mapping (section 4.5), and
  the retirement ruling with its evidence (section 6). It amends ADR-126 (`lift_ok` and `calibration_ok`) and records
  that the existing CV keys are now nested values.
- `docs/context/trained-models.md`: the acceptance-gate paragraph.
- `CHANGELOG.md`, and the version bump if the wheel changes (`_artifact_integrity.py` and `_xcross_eval.py` are in
  the wheel). Release numbers are set at commit preparation.
- The TF-16 spec and older plans stay unedited (historical). ADR-131 is the pointer from the old gate to the new.

## 10. Not in this cycle

- Any change to features, labels, corpus or fold plan.
- Any change to the HPO objective, search space or trial count (section 6 finding).
- The model-card text (owner ruling O2: with the adoption).
- Any retirement and its checklist (owner ruling: Part Deux applies the section 6.1 rule to its refit results, at
  the adoption).
- Growing the public xCross corpus (owner ruling: a separate scope for the owner to schedule).

## 11. Process and hand-off

**Amendment (owner ruling 2026-11-12, after implementation): "Widen before refit".** The refit does not follow this commit directly. The 10 newer public SkillCorner matches (registered in `PUBLIC_CORPUS` since 2026-09-10) join the training corpora first, in their own cycle: public arm 17 to 27, owner arm `xshot115` to 125 (it supersedes combined-cycle ruling D4 for the next refit). Then ONE refit of all arms on the widened corpora under these gates. The hand-off below applies to that single refit. Section 3's numbers are run 9a's (17 / 115) and stay the record of this cycle's measurement.

Spec r4 re-review by A and B, then the plan, A/B review, TDD implementation, A/B review, `/final-review`, and the
owner's yes to commit (one commit for the cycle). Then I message Part Deux (`2#silly-kicks (Deux): HF Jobs`) with:

- the commit and "all 8 arms";
- the four model cards (owner ruling O2);
- the section 6.1 retirement rule, its prediction, and the checklist for the predicted case.

Each launch needs the owner's go. Part Deux applies the rule to its refit results: it adopts the arms that pass and
retires the committed artifact of each arm that fails, and it reports the per-arm outcome to the owner. Draft PR #302
carries both.

## 12. r1 findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-B01, FSG-SPEC-10 (pooled PR-AUC biased) | Measured (section 3): −0.00027 to −0.00357 for a zero-skill forecast. Owner ruling: within-fold lift (section 4.2), neutral by construction; Test 1 covers it. |
| FSG-B02 (bootstrap call sequence) | Pinned in section 4.3; Test 5. Superseded in r3 and r4: section 8 compares with `gold3.py` output (now `nested_report.json`), not `gold2.py`. |
| FSG-B03 (self-check, unmeasured parts) | Section 3 states the failed 1e-12 check, its values and the inferred cause. Degenerate draws and all no-gain fractions are measured (gold2). |
| FSG-B04, FSG-SPEC-04 (toy corpora) | Section 7 rule: measure first, strengthen the toy, never weaken an assertion without an owner ruling. |
| FSG-B05, FSG-SPEC-01 (reduce labels) | Each fold unit returns `groups.npy`; the reduce reads only unit results (section 5). |
| FSG-B06, FSG-SPEC-08 (level approximate) | Section 4.3 "What the bound means"; section 4.4 wording; ADR-131. |
| FSG-SPEC-02 (integrity mapping) | Owner ruling: the same rule as the trainer; mapping in section 4.5. |
| FSG-SPEC-03 (missed consumers) | All four cards (owner ruling O2) and the degenerate-toy tests listed in section 4.5; Test 11. |
| FSG-SPEC-05 (imprecise numbers) | 0.00075 and 0.0247 (section 2). |
| FSG-SPEC-06 (two reasons) | Section 2: the mismatch term is the cause; the concavity effect is stated. |
| FSG-SPEC-07 (two tolerances) | Section 8 names the two comparisons. |
| FSG-SPEC-09 (point gates on noise) | Section 4.4 states it; the lower bounds catch it. |
| FSG-SPEC-11 (cost) | Section 5 states the cost; section 8 measures the reduce time. |
| B scope note (log-loss gate) | The owner header lists the ruling that covers it. |

## 13. r2 findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-B07, FSG-SPEC-12 (dropped-fold rule decides an arm; no ruling; section 1 wrong) | Section 1 states the facts and corrects the r2 claim. The owner ruled "Drop the fold" in a question that stated the fail-closed outcome (header). Section 3 gives all three rules from `gold3_report.json`. |
| FSG-B08, FSG-SPEC-13 (section 8 reference holds NaN; keys differ; values without a file) | Section 8 compares with `gold3_report.json`, which implements section 4.3 as written with the section 4.5 key names. Every section 3 value is in that file. |
| FSG-B02 (PARTIAL in r2) | Closed by the same change: `gold3.py` runs the pinned sequence and the drop rule as written. |
| FSG-SPEC-14 (O1 caution) | Section 6 carries it. |

## 14. r3 findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-SPEC-15, FSG-B09 (`n_dropped_fold_draws` has two readings) | Two keys, `n_empty_fold_draws` and `n_one_class_fold_draws`, defined in section 4.3; `gold3.py` counts both; Test 7; section 8 compares both. |
| FSG-B10 (B02 row still cites `gold2.py`) | Marked superseded in section 12. |

## 15. r4 findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-SPEC-16, FSG-B11 (committed weights were not scored) | Sections 6 and 6.1 say "committed params, refit on the run 9a folds" and why the weights cannot be scored out of sample. The owner was shown the corrected statement and confirmed the rule (header, r5). |
| FSG-SPEC-17 (DGX times size nothing on HF) | Section 5 says so; Part Deux gives the launch estimate. |
| FSG-SPEC-18 (retire only on a data failure) | Section 6.1: retirement only on a skill gate that fails with finite values; any other failure stops the arm for investigation (owner ruling, r5). |
| FSG-B12 (rematerialize compares nested with non-nested) | Section 4.5 records the measured `pr_auc` shift (at most −3.32%), inside the 0.05 tolerance. |

## 16. r5 findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-B13 (wording: job error, numeric values) | Section 6.1: "a job error without `only.metrics_FAILED.json`" and "every numeric value". No change of meaning. |
