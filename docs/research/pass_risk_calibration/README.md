# Pass-risk calibration -- real-data validation

Corpus: **full** (owner-tier Gradient Sports WC2022). Raw per-pass positions are NOT committed (see the gitignored `pass_scores.parquet`); only these aggregate rates are.

**This measures OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall) -- recall needs the failed-pass class, which is both leaked (outcome-selected end_xy target) and confounded, until the deferred Power-2017 expected-receiver model lands. The clean headline is not a full validation.**

## Headline (leakage-free): completed-pass false-alarm rate `P(control < tau | completed)`
{'0.1': 0.04179918218991367, '0.2': 0.05523181162712601, '0.3': 0.06799280959247772}

## Optimistic (reads failed passes -> leakage-inflated; not the headline)
- AUC(is_success, control): 0.6008519303851549
- ECE: 0.31643854082233475
- reliability slope: 0.3166979708500353

## Low-control COMPLETION band (CONTAMINATED, not the headline)
{'0.1': 0.6280795488275452, '0.2': 0.6334390575441776, '0.3': 0.6432442534105774}
P(success | control<tau) over ALL passes -- the 'technically complete, functionally lost' read. Failed passes cluster at low control via the end_xy selection, so this is CONTAMINATED and kept distinct from the clean false-alarm headline; never conflated.

## Limitations
- Selection bias (unfixable): only attempted passes are observed.
- Control != completion: pitch control is a positional model; the reliability curve is a mapping.
