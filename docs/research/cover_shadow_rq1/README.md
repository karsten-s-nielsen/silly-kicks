# Cover-shadow RQ1 -- real-data validation

Corpus: **full** (owner-tier Gradient Sports WC2022). Raw per-pass positions are NOT committed (see the gitignored `pass_scores.parquet`); only these aggregate rates are.

**This measures OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall) -- recall needs the failed-pass class, which is both leaked (outcome-selected end_xy target) and confounded, until the deferred Power-2017 expected-receiver model lands. The clean headline is not a full validation.**

## Headline (leakage-free): completed-pass false-positive rate, PASS-ONLY
- majority rule: 0.1545341467748623
- p_blocked>0.5 (center/mean/max): 0.08934036209994274 / 0.08912318110920255 / 0.13283579142727397

The pass-only cut leads because `lane_control` models GROUND-lane screening and crosses are aerial.

## Optimistic (read failed passes -> leakage-inflated; not the headline)
- AUC, discriminating score (n_blocked / mean margin): 0.6922685932188641 / 0.7644478154296962
- AUC, absolute p_blocked magnitude (center/mean/max, ~0.5 -- the WRONG score): {'center': 0.5109356239341754, 'mean': 0.5086981246492929, 'max': 0.5047173659653338}
  The model compares `p_blocked` to `p_received` per lane, so the discriminating quantity is the
  margin / `n_blocked` count the majority rule thresholds, not the absolute `p_blocked` intensity.
- reliability slope (on P(screened) = p_blocked_mean): 0.20437143496497784
- confusion (paper-comparable): {'tp': 5677, 'fp': 8044, 'tn': 42930, 'fn': 5293, 'precision': 0.41374535383718386, 'recall': 0.5175022789425706, 'specificity': 0.8421940597167183, 'balanced_accuracy': 0.6798481693296445}

## Paper reconciliation
our majority recall 0.518 vs the paper's 0.369 (recomputed from Cascioli Appendix B, not the handoff table).

## Limitations
- Selection bias (unfixable): only attempted passes are observed; precision is a lower bound.
- Failed-pass `end_xy` target is outcome-selected -> the failed-pass legs are optimistic bounds.
- Screening != failure: `p_blocked` is P(lane screened); the reliability curve is a mapping.
