# TF-24 Stage-2 tracking-defaults refresh — recommendation (Phase C, ADR-009/ADR-060)

`calibration_report.json` / `.md` in this directory. Produced by
`scripts/calibrate_tracking_defaults.py --stage 2` at `run_commit aa34017`, `run_tree_dirty false`,
over the **full 179-match pining corpus** (skillcorner 108 + gradientsports 64 + idsse 7), 60 Optuna
trials minimizing the augmented-VAEP held-out Brier over `k3` / `pre_seconds` / `min_displacement_m`,
holding the ADR-060 Stage-1 carrier params (`docs/research/tf24_stage1_confirmation/carrier_selected.json`,
`run_commit 2cecd2b`, clean). Frozen exogenous xT: `~/tf24-store/calibration_xt.npz`, disjoint 16-match
corpus, sha256 `52d7a8…`, fail-closed. No DAS degradation, no excluded providers. silly_kicks 4.82.0,
xgboost 3.4.0, ruthless 0.4.0.

## The recommendation, and why it should NOT be adopted

| | k3 | pre_seconds | min_displacement_m | held-out Brier |
|---|---:|---:|---:|---:|
| **Incumbent** (trial 0 = current library defaults) | 1.0 | 1.5 | 3.0 | 0.009608 |
| **Recommendation** (best, trial 25) | 2.94 | 2.26 | 4.77 | **0.009553** |

The recommendation beats the incumbent by **0.000055** (0.57 % relative) — **well inside every
per-provider standard error**: `brier_se` gradientsports 0.00030, skillcorner 0.00042, idsse 0.0019.
The improvement is not statistically distinguishable from the incumbent; the augmented-VAEP Brier is
**essentially flat** across the swept tracking-default space (the trial spread's `max 0.048` is a
single degenerate parameter combination, not signal).

Per **ADR-009** the harness only *recommends* and never changes a library constant — and this result
**argues against adoption**: any change would be within noise. This is consistent with the TF-24
non-identifiability theme established for Stage 1 in **ADR-060** (the carrier params were weakly
identified; Stage 2 shows the tracking defaults are too). Adopting a default remains a separate PR
that this cycle does not open.
