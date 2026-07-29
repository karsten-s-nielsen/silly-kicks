# TF-19 §6.1 power curves — ICC and ATT

**Run:** 2026-07-28, `run_commit 6b242cf`, `run_tree_dirty: false`, `lock_commit 6b242cf`.
**Driver:** `scripts/run_signoff_power.py --spells … --arm-values …`.
**Inputs:** Layer 2 spells (64 GS matches, 37,086 spells, built at `6b242cf`) + the GKDV
arm-values table (64 matches, 123,430 scored frames, built at `93ac3ba`). Both upstream tables were
provenance-checked by the driver before any work: clean tree, commit-consistent across workers.

This discharges the obligation ADR-037 §6.1 registered and PR-3 shipped as a docstring promise no
code could keep — *"a power curve is reported at all three anchors"*, with the gate registered only
if detection at the anchor is ≥ 0.8.

## Result — the two legs SPLIT

### ICC leg (the §6.1 primary criterion): **precondition discharged**

| Anchor | Power | Mean observed ICC | Mean null ICC |
|---|---|---|---|
| 0.015 | **1.00** | 0.0154 | 0.0072 |
| 0.020 | **1.00** | 0.0206 | 0.0087 |
| 0.026 | **1.00** | 0.0255 | 0.0104 |

`mean_observed_icc_at_zero = −0.00034`. That number is what makes power 1.0 believable rather than
suspicious: with **no** injected effect the estimator returns ~zero, so it is detecting signal, not
manufacturing it. 41 keepers, of which 8 appear in a single match — for those the block permutation
is a pure relabelling, which the report surfaces rather than hides.

### ATT leg: `N_MIN_MATCHED` is **None**

| size | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| power (`Y_attempt`, 0.15 anchor) | 0.000 | 0.010 | 0.015 | 0.045 | **0.055** |
| degenerate replicates (of 200) | 62 | 25 | 3 | 0 | 0 |

Max power **0.055** against a required 0.80 — indistinguishable from the 0.05 false-positive rate.
No size reaches the threshold at any anchor, for either outcome, so `N_MIN_MATCHED` stays `None`.

**The degenerate counts are what make that readable.** At n=4000 and n=8000 *zero* replicates were
inestimable, so the near-zero power there is not an artifact of positivity failure: the design is
estimable and simply cannot detect the registered effect sizes at 151 treated units corpus-wide
(prevalence 0.0041). Without counting them — the behaviour added in 4.65.0 — this would have read
as a weak effect rather than an underpowered one.

## Why the split matters

ADR-037 finding **F3** separated two estimands the spec had conflated: an ICC variance share and a
spell-level ATT. They return **opposite** answers here. Had they stayed merged, either the ATT's
failure would have wrongly blocked a registrable ICC gate, or the ICC's success would have wrongly
licensed an `N_min` the data cannot support.

## Consequence

- The §6.1 **ICC gate may be registered**: its detection precondition is met at all three anchors.
- **`N_MIN_MATCHED` remains `None`.** §6.1's own rule applies — adjust floors/sampling first; do not
  register a row-5 threshold this corpus cannot support.
- The registered 16.5 m Layer 2 treatment threshold was **not** retuned to raise prevalence. It is
  Law-defined precisely so the decider stays untuned; changing it is a re-registration decision, not
  an implementation one.
