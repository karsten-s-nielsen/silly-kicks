# TF-19 §6.1 power curves — ICC and ATT

**Run:** 2026-10-05, `run_commit b62c1f2` (M, combined-cycle C2), `run_tree_dirty: false`,
`lock_commit 6b242cf` (the registered-threshold lock is unchanged; only the data was re-materialized).
**Driver:** `scripts/run_signoff_power.py --spells … --arm-values …`.
**Inputs:** Layer-2 spells (64 GS matches, **37,086 spells, 151 treated**, prevalence 0.0041) + the
GKDV `delta_das` arm-values table, both rebuilt on F1b float32 frames with the native DAS engine in the
same wave (`run_commit b62c1f2`). Both upstream tables were provenance-checked by the driver before any
work: clean tree, commit-consistent across workers (`upstream_provenance` in `metrics.json`).

This discharges the obligation ADR-037 §6.1 registered and PR-3 shipped as a docstring promise no
code could keep — *"a power curve is reported at all three anchors"*, with the gate registered only
if detection at the anchor is ≥ 0.8.

## Result — the two legs SPLIT

### ICC leg (the §6.1 primary criterion): **precondition discharged**

| Anchor | Power | Mean observed ICC | Mean null ICC |
|---|---|---|---|
| 0.015 | **0.995** | 0.0156 | 0.0078 |
| 0.020 | **1.00** | 0.0203 | 0.0093 |
| 0.026 | **1.00** | 0.0256 | 0.0110 |

`mean_observed_icc_at_zero = −0.00034`. That number is what makes power ≥ 0.995 believable rather than
suspicious: with **no** injected effect the estimator returns ~zero, so it is detecting signal, not
manufacturing it. 41 keepers, of which 8 appear in a single match — for those the block permutation
is a pure relabelling, which the report surfaces rather than hides.

### ATT leg: `N_MIN_MATCHED` is **None**

| size | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|
| power (`Y_attempt`, 0.15 anchor) | 0.015 | 0.005 | 0.020 | 0.035 | **0.050** |
| degenerate replicates (of 200) | 62 | 25 | 3 | 0 | 0 |

Max power **0.050** against a required 0.80 — indistinguishable from the 0.05 false-positive rate.
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

## Provenance refresh (combined-cycle C2, 4.128.0)

`metrics.json` here is the re-run at the release commit (`run_commit` in the file), over the F1b
float32 GKDV arm-values and Layer-2 spells rebuilt in the same wave. It **supersedes** the earlier
`6b242cf` run that the now-removed `invalidation.json` annotated; the sibling annotation is no longer
needed, so it is deleted and its `_UNPROVENANCED` registry entry removed in the same change.
