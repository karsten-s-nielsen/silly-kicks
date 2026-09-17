# TF-53 match-outcome calibration study

**Driver:** `scripts/validate_match_outcome_calibration.py` (ADR-052 `for_each` shards, ADR-037
clean-tree provenance, ADR-056 input contract, reported-not-gated / ADR-009).
**Corpus:** full redistributable StatsBomb open-data — **3,961 matches** (7,922 team-rows per config),
`assert_statsbomb_open_data_mode` fail-closed public-only.
**Provenance:** `run_commit = dd133e0…` (the clean commit-1 tree), `run_tree_dirty = false`. Bundled ρ
(`silly_kicks/match_outcome/weights/`) fit on the same commit → **ρ = 0.0216** (`training_commit`
= dd133e0).

The study scores four method configs — `independent`, `collapse` (Rung 3b), `dixon_coles` (Rung 3a),
`both` — with a **per-fold cross-validated ρ** (grouped by `game_id`, evaluated held-out, NEVER the
bundled weights, so the study evaluates the METHOD, not the artifact on its own fit data).

## Per-config calibration (`metrics.json`)

| config | 3-way Brier | calibration slope | xPoints-bias |
|---|---|---|---|
| both (collapse + dixon_coles) | **0.4911** | 0.918 | −0.0080 |
| collapse | 0.4913 | 0.919 | −0.0097 |
| dixon_coles | 0.4918 | 0.910 | −0.0071 |
| independent | 0.4920 | 0.911 | −0.0088 |

- **Well-calibrated:** xPoints-bias ≈ −0.008 (xPoints ≈ realized points on average); slopes ~0.91
  (mild over-confidence, expected of a retrospective xG-only model).
- **CV ρ per fold:** `[0.033, 0.025, 0.022, 0.010, 0.017]` (mean ≈ 0.022 ≈ bundled 0.0216) — the fit is
  stable across folds, not overfit.

## Why the default is BOTH corrections ON (the paired test)

The aggregate Brier gaps are small (~0.001), which is misleading. A **paired per-match Brier test**
(pair by `(game_id, team_id)`, Wilcoxon signed-rank; positive Δ = the 2nd config has lower/better Brier)
is decisive:

| comparison | mean Δ | % pairs better | Wilcoxon p |
|---|---|---|---|
| collapse vs independent | +0.00065 | 45.4%* | 1.7e-4 |
| dixon_coles vs independent | +0.00021 | **77.3%** | **7e-125** |
| both vs collapse | +0.00022 | **77.3%** | 4e-124 |
| both vs independent | +0.00087 | **68.7%** | 3e-71 |
| dixon_coles vs collapse | −0.00044 | 70.6% | 7e-53 |

- **collapse and dixon_coles fix DISTINCT errors.** `collapse` (same-possession non-independence) is a
  no-op on most matches (`*`median 0 / better on <half) but wins big on the few multi-shot-possession
  matches (positive mean). `dixon_coles` (cross-team low-score dependence) helps a small amount on MANY
  matches (better on 77%). `both` captures both.
- **`both` dominates:** beats independent on 68.7% of matches, largest mean improvement (+0.00087),
  p≈3e-71.
- Decision (ADR-097): default = **both**. `collapse` on correctness (double-counting same-possession
  shots is simply wrong); `dixon_coles` on this overwhelming, held-out-stable paired evidence.
  `"independent"` remains available opt-in on either axis.

## Honest limits

- **Retrospective**, not predictive — xG is post-hoc; this is "how the match should have gone given the
  chances", not a pre-match forecast.
- Team independence (when `team_dependence="independent"`) is an assumption; `dixon_coles` corrects the
  low-score cells but is a phenomenological reweighting (τ was derived for Poisson marginals — its
  validity on Poisson-binomial marginals rests on this empirical fit, which the paired test confirms).
- Never surface G−xG year-to-year variation as "finishing skill" (it is near-zero-autocorrelation
  noise).
- **Cross-provider (Wyscout/Pappalardo) calibration is out of scope:** public Wyscout carries no xG,
  and the metric is xG-driven end-to-end.
