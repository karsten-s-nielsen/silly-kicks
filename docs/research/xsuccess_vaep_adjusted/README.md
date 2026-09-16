# xSuccess feature discovery + VAEP_adjusted validation (TF-61)

This folder records the TF-61 OpenEvolve Stage-A feature-representation discovery (ADR-093 discipline:
the evolve *process* is documented, the *decision* is committed). The construct-validity battery for the
bundled model + `VAEP.rate_adjusted` (calibration-in-the-large, per-type reliability, outcome-bias
reduction) lands here in the Commit-2 (weights + validation) commit.

## Stage A — OpenEvolve feature search (2026-09-15)

Two runs on the DGX-Spark via OpenRouter (Sonnet-5 w0.8 + Opus-4.8 w0.2, 100 iterations each,
diff-based, 3 islands). Fitness = held-out `StratifiedGroupKFold`-by-match log-loss on the **WC2022
discovery subsample** (64 matches, 138,945 on-ball actions, success 0.863), single-sourced through
`scripts/evolve_xsuccess_features.py::evaluate_candidate` with a hard in-fitness END-BLIND guard
(inject synthetic `end_x`/`end_y` → features must be byte-identical, else the candidate is rejected).
Full config, metrics, and spend in `provenance.json`.

| Fitness model | seed logloss | evolved logloss | Δ | verdict |
|---|---|---|---|---|
| **XGBoost** (primary, shipping) | 0.2278 | 0.2273 | 0.0006 (~0.2%) | **null** — the tree already extracts the geometry; representation search has a low ceiling by construction |
| **per_type_logistic** (fallback) | 0.2356 | 0.2286 | 0.0070 (~3%) | **real** — closes ~80% of the linear-vs-tree gap, near XGBoost's 0.2278 |

Both winners (`evolved_xgboost_winner.py`, `evolved_logistic_winner.py`) are END-BLIND (confirmed by
hand-review + the in-fitness guard) and interpretable: goal-mouth subtended angle (near/far post),
y-symmetry about the pitch centre, log/exp-decay location transforms, smooth sin/cos bearing for a GLM,
and systematic per-type/bodypart geometry interactions.

## Decision

**The seed ships** (the shared `silly_kicks/xsuccess/_features.py` is unchanged). Stage A is evaluated
against the shipping family — XGBoost is primary; `per_type_logistic` is a *gate-conditioned fallback*
(spec §5.4) that ships only if the calibrated XGBoost fails the per-type calibration bar (spec §9.1) at
Commit-2. The XGBoost gain is null under any reasonable margin (the spec's "pre-registered margin" was
never numerically pinned — a gap; the decision is robust to it, see `provenance.json`).

The logistic-fitness run (commissioned to fully exercise OpenEvolve where representation actually
matters) found real value **for the fallback family only**. Per owner decision (2026-09-15) it is
**deferred to the Commit-2 family gate**: if the fallback is selected there, adopt a family-specific
evolved builder for it then (`evolved_logistic_winner.py` is the candidate). It is *not* folded into the
shared builder now — that would bloat the XGBoost feature contract 35 → 152 columns for ~0 XGBoost gain.

## Commit 2 — bundled model + validation

**Bundled `default` model** (`silly_kicks/xsuccess/weights/`, `training_commit 12e5677`, clean tree):
calibrated XGBoost (xgboost family) trained on the **full redistributable StatsBomb open-data corpus —
3,961 matches / 7,974,436 on-ball actions** (80 comp:season pairs, enumerated in `metadata.json` /
`MODEL_CARD.md`), Optuna HPO 30 trials. GroupKFold-by-match out-of-fold: **AUC 0.895, Brier 0.084**
(base rate 0.835).

**§9.1 xSuccess calibration — family gate PASS, the calibrated XGBoost ships** (the `per_type_logistic`
fallback is not triggered; the Stage-A logistic finding stays moot):
- Calibration-in-the-large: sum(pred) / sum(obs) ratio **1.0000** (6,658,306 / 6,658,333).
- Per-type reliability: max |pred - obs| **0.006** (shot_penalty), mean **0.001** across 21 types (all < 0.6%).

**§9.2 VAEP_adjusted validation** (WC2022: 64 games, 138,945 actions, 1,384 shots with xG, 425 players;
StatsBomb xG injected -- silly-kicks ships none):
- **Outcome-bias reduction (the paper's headline):** corr(value, realized success) **0.154 -> 0.004**;
  success-minus-fail value gap **0.0159 -> 0.0003**. The adjusted value is decoupled from the realized outcome.
- **xG alignment:** per-player shot-value vs xG Pearson **0.423 -> 0.547** -- the adjusted value tracks
  EXPECTED quality (xG) *better* while decoupling from realized goals, so de-biasing does not break the
  value scale.

Reported-not-gated (applied results; ADR-009).
