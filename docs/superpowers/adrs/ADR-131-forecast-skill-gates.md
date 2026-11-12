# ADR-131: Nested-CV forecast-skill acceptance gates for xShot and xCross

| Field | Value |
|---|---|
| **Date** | 2026-11-12 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (owner rulings 2026-11-12); reviewers A and B (spec r1 to r5, plan r1 to r4) |

## Context

The xShot and xCross trainers refuse to ship a model that does not beat a trivial forecast. Until this ADR they
gated on three bars, fixed by the TF-16 spec before its run: the unweighted mean of the per-fold Brier scores below
`base * (1 - base)`, the mean per-fold PR-AUC above `base`, and the mean log loss below `ln 2`. `base` was the pooled
positive rate of the arm.

Refit run 9a (`tf46_refit`, commit `aadd4cf7`) failed `xcross_attempt-public` / `position_only` on the Brier bar:
0.037276 against 0.036896. The investigation found that the bar is biased low. A cross-validated model can only learn
the training folds' rate, and the folds differ in rate. On each fold a constant forecast at the training rate scores
`rate_te * (1 - rate_te) + (rate_tr - rate_te)^2`; the squared mismatch term is never negative. On the failing arm,
that constant forecast scored 0.037605 as a fold mean, so a zero-skill forecast failed the gate. The PR-AUC bar
compared unlike quantities to a smaller degree (at most 0.00075 on the run 9a arms). The log-loss bar was vacuous: a
constant forecast at rates of 0.016 to 0.070 scores 0.08 to 0.25, far below `ln 2`.

This record states the order of events plainly. A gate failure exposed the defect. The replacement rule (section
Decision, items 1 to 4) was designed before the measurement and implemented by the measurement script. Three changes
came after it, each from a review or a diagnosis and each ruled by the owner with the facts stated: the within-fold
PR-AUC reference (the pooled reference was measured biased against the model, −0.00027 to −0.00357 for a zero-skill
forecast); the dropped-fold rule (under the first, fail-closed rule both xcross public arms failed the PR-AUC bound
because one match has no cross attempt); and nested CV (the HPO and the gate used the same 5 folds). No change was
made to pass the arm that fails: it fails on the Brier bound under every variant of the rule.

## Decision

1. **Reference forecast.** Each held-out row's reference is the TRUE positive rate of its training folds (before
   negative subsampling). All skill is measured against it, on the same held-out rows.
2. **Point quantities, pooled out of fold.** `brier_skill = 1 - B_model / B_ref`; `log_loss_skill = 1 - LL_model /
   LL_ref` (`p` and `r` clipped to `[1e-15, 1 - 1e-15]` in the log-loss terms only); `pr_auc_lift` = the row-weighted
   mean over folds of `AP_k - rate_k` (within-fold, so a constant forecast has a lift of exactly 0).
3. **Match-cluster bootstrap.** 2,000 draws, `numpy.random.default_rng(0)`; one `rng.integers(0, M, M)` per draw over
   the sorted match labels; row weights = match multiplicity; lower bound = `np.percentile(gain, 5)`. A fold with no
   drawn match or a one-class weighted rate is dropped from the PR-AUC lift of that draw (both counts are reported);
   a draw with no usable fold is NaN and fails. The bound is approximate: it holds the fold models fixed, so it covers
   the sampling of held-out matches only, and a percentile bootstrap over 17 clusters tends to under-cover.
4. **Seven gates, all required.** `enough_usable_folds` (unchanged), `brier_skill_gt_0`, `log_loss_skill_gt_0`,
   `pr_auc_lift_gt_0`, `brier_gain_lb95_gt_0`, `log_loss_gain_lb95_gt_0`, `pr_auc_lift_lb95_gt_0`. Requiring every
   bound is an intersection-union test, so the combined rule adds no multiplicity inflation. The draw count, seed,
   level and clip are constants; no flag or config changes them.
5. **Nested CV.** Gate fold `k` is scored at the params of a study run on that fold's training rows only
   (`{arm}_gf{k}` study tags). The arm's full-rows study still gives the shipped fit's params.
6. **One implementation.** `forecast_skill` and `skill_gates` in `silly_kicks/tracking/_artifact_integrity.py` serve
   both trainers and `scripts/validate_trained_artifact_integrity.py`; the trainers' `_gates` are deleted. The cv
   fold units return their held-out arrays (FileSet); the reduce pools them (`scripts/_study_shared.cv_skill`).
7. **The integrity check uses the same rule** (amends ADR-126): `lift_ok` = the PR-AUC lift gate and its bound;
   `calibration_ok` = the Brier and log-loss skill gates and their bounds. `calibration_ok` now means proper-score
   skill (calibration and resolution). A metrics dict without `skill` fails both.
8. **Metrics shape (Hyrum).** The existing keys `pr_auc`, `brier`, `log_loss`, `pr_auc_std`, `positive_rate`,
   `base_rate_brier`, `n_usable_folds` keep their names; their values are now nested-CV values (fold-mean `pr_auc`
   moved +0.03% to −3.32% on the run 9a arms, inside the 0.05 tolerance of `reproduces` and `PUBLIC_CV_REL_TOL`).
   Each candidate gains `skill` and `fold_params`. The acceptance keys `pr_auc_gt_base_rate`,
   `brier_lt_base_rate_brier` and `log_loss_lt_uniform` are removed.
9. **Retirement is a rule on the refit data.** The session that runs the refit applies it per arm: an arm whose refit
   fails a skill gate with every numeric `skill` value finite (and enough usable folds) is not adopted, and the
   committed artifact it would replace is retired at the adoption. Any other failure (too few folds, a NaN, a missing
   `skill`, a job error without `only.metrics_FAILED.json`) stops the arm for investigation; nothing is retired.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep the old bars, fix only `base * (1 - base)` to the CV-null bar | Small change | Still no uncertainty; PR-AUC and log-loss bars unchanged (the latter vacuous) | Not the gold standard the owner asked for |
| B. Point skill only, bounds reported | Cheaper; all 8 run 9a arms pass | A gain within noise ships; chosen after seeing the numbers | Rejected by the owner ("as designed") |
| C. Bootstrap bound on log loss only | Most sensitive proper score | Chosen after seeing the numbers | Rejected by the owner |
| D. Pooled PR-AUC lift | Simple | Mixes fold-model offsets; measured biased against the model | Replaced by the within-fold lift (owner ruling) |
| E. Fail a draw closed when a fold has no positive | Strict | Fails an arm because a match had no cross attempt, not because of the model's ranking | Owner ruled "Drop the fold" |
| F. Non-nested gate | No extra studies | Scores params tuned on the gate's own folds: optimistic | Owner ruled nested CV |
| G. This ADR | Honest reference, uncertainty, nested, one implementation | 5 extra studies per arm; a larger cv payload | — |

## Consequences

### Positive

- A zero-skill model ties its reference exactly; the gate no longer fails a model for the folds' rate mismatch.
- A model passes only when its gain over the reference holds across resampled matches, at honest (nested) estimates.
- The trainers and the integrity check share one rule and cannot disagree.

### Negative

- Each cv arm runs one extra study per gate fold (5 on every run 9a arm). On the DGX the slowest
  (`xshot_occurrence-sc_extended`) took 34 to 44 min each, single-threaded.
- The cv stage writes five files per fold unit (`fold.json`, `te.npy`, `p.npy`, `y.npy`, `groups.npy`).
- On 17 public matches a small real gain may not clear the bound: on run 9a, `xcross_attempt-public` /
  `position_only` has Brier skill +0.9% but a Brier lower bound of −0.0000139. Growing the public corpus is the fix.

### Neutral

- The measurement behind this ADR is in `D:\Temp\hfu-topup\diag\` (`gold.py` to `gold3.py`, `nested.py`, `o1.py`,
  `prauc_shift.py`, with their reports). The pre-set 1e-12 self-check of `gold.py` failed on all 8 arms with
  differences of 9.1e-11 to 1.49e-9 (DGX arm64 against HF x86; the cause is inferred, not verified).
- The committed `metrics.json` files, `docs/artifacts/observed.json` and the integrity verdicts keep their old
  acceptance blocks until a refit replaces them; they are records of what ran.

## Related

- **Specs:** `docs/superpowers/specs/2026-11-12-forecast-skill-gates-design.md`
- **Plans:** `docs/superpowers/plans/2026-11-12-forecast-skill-gates-plan.md`
- **ADRs:** amends ADR-126 (`lift_ok`, `calibration_ok`); builds on ADR-128 (stage contract) and ADR-130 (refit mode);
  replaces the acceptance bars of the TF-16 spec (`docs/superpowers/specs/2026-06-02-tf16-weights-run-design.md`).
- **External references:** Murphy (1973), Brier-score decomposition; Varma and Simon (2006) and Cawley and Talbot
  (2010), bias of CV estimates after model selection.
