# ADR-094: TF-61 — xSuccess (event-only action-completion) + VAEP_adjusted (outcome-bias-free VAEP)

| Field | Value |
|---|---|
| **Date** | 2026-09-15 |
| **Status** | Proposed — number PROVISIONAL (reconciled at commit-prep after `git fetch && git merge origin/main`; a concurrent silly-kicks session may claim 094 first) |
| **Deciders** | Karsten Nielsen |

Design doc: `docs/superpowers/specs/2026-09-14-tf61-xsuccess-vaep-adjusted-design.md` (rev 4). Plan: `docs/superpowers/plans/2026-09-14-tf61-xsuccess-vaep-adjusted.md`.

## Context

Standard VAEP has a known **outcome bias**: it values a game state *after* the action's realized result, so a lucky completion of a risky action is over-credited and a sound decision that fails is penalized (the VAEP authors note a made shot scores ≈ outcome − xG). Paul, Klemp & Memmert (2025, "Beyond Outcome Bias", MLSA 2025 paper 225) correct this by separating **completion probability** from **value** and re-weighting the score/concede deltas by completion probability.

silly-kicks already ships an event-only pass-completion seam (`expected_passing.PassCompletionModel`), but it is pass-only and serves a *supplied* target; it does not answer "will THIS on-ball action complete?" for all action types, which VAEP_adjusted needs for every action. There is no risk-adjusted VAEP variant.

Forcing function: TF-61, owner-routed as this session's cycle. Event-only, additive, no default retrain.

## Decision

Ship two event-only primitives:

1. **`silly_kicks.xsuccess.XSuccessModel`** — a bundled, calibrated action-completion model `P(success | pre-action context)` over ALL on-ball action types. **END-BLIND** (start-anchored geometry + type + bodypart + time; never the realized action end). Calibrated XGBoost served via the `[xgboost]` extra + pure-numpy isotonic; pickle-free booster-JSON + metadata + SHA256, fail-closed load (SHA → feature-contract → chirality; ADR-011/016/040/050). A **per-type-logistic fallback family** (`fit(family="per_type_logistic")`) is the ADR-009 calibration fallback. Feature representation discovered offline by OpenEvolve (leakage-safe, END-FREE allowlist; committed deterministic output + provenance); HPs by Optuna (`XSuccessObjective`, ruthless `CachedObjective`). Event-only import boundary (never `tracking`).

2. **`VAEP.rate_adjusted`** (+ `vaep.adjusted.adjusted_value`) — re-scores an already-fitted STANDARD VAEP on a **surgical result-feature counterfactual** (flip only the current action's `result_onehot`/`actiontype_result_onehot`; hold locations, goalscore, predecessors fixed), weights by `xSuccess`/`(1−xSuccess)`, and runs the existing `formula.value` delta (eqs 8–12). **No retrain.** Requires result-bearing features (raises on HybridVAEP — the flip is a no-op).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. End-USING xSuccess features (paper Table-1: realized end geometry) | matches paper; higher AUC; per-action pass length/direction | a failed action's end IS the outcome → **target leakage** (postdiction), re-injects the bias VAEP_adjusted removes | rejected — invalid ex-ante for both VAEP_adjusted and scouting; `PassCompletionModel` precedent doesn't transfer (it serves supplied targets) |
| B. Paper's whole-dataset result flip | literal fidelity | inflates goalscore (every shot→goal in all-success) + rewrites predecessor history — confounds the de-biasing | rejected — surgical single-variable flip is the correct ceteris-paribus counterfactual |
| C. Single logistic + one-hot type | one simple artifact | additive-linear → mis-calibrated geometry across types (calibration is the acceptance bar; we multiply these) | rejected as primary; kept per-type-logistic as the fallback family |
| D. **(chosen)** END-BLIND calibrated XGBoost + per-type-logistic fallback + surgical flip | leak-clean; calibrated; convention-consistent (xShot serve, PassCompletion discipline); no retrain | coarser event-only pass granularity than the paper; xgboost-at-inference for xSuccess | — |

## Consequences

### Positive
- An outcome-bias-free VAEP variant (`rate_adjusted`) reusing any caller-fitted standard VAEP — no retrain, additive.
- A reusable, calibrated, all-action-type completion model; the shot de-biasing (the paper's headline) is unaffected by end-blindness.
- Full ruthless-efficiency use (OpenEvolve representation search + Optuna HPO), deterministic committed output.

### Negative
- End-blind forgoes event-only per-action *pass* length/direction granularity (unrecoverable clean without tracking; the risky-pass illustration is milder than the paper's). The clean richer paths are `PassCompletionModel` (supplied target) and a future tracking-augmented xSuccess.
- xSuccess bundled *inference* needs the `[xgboost]` extra (no new dep for the VAEP_adjusted use, which already runs xgboost).
- The `_load_booster_base_score_safe` helper is duplicated (not imported) from `tracking/_xshot_occurrence.py` to keep xsuccess event-only.

### Neutral
- No default VAEP retrain (no `*_xfns`, in no default xfn list); no `feature_glossary` growth (a model + a rating method, not `add_*`). +1 C4 container (`xsuccess`); action-coupled aggregator count unchanged (33).
- **Stage-A OpenEvolve outcome (2026-09-15, `docs/research/xsuccess_vaep_adjusted/`):** the **seed shipped** — the XGBoost-primary evolve gain was null (held-out log-loss 0.2278 → 0.2273, ~0.2%; a gradient-boosted tree already extracts the geometry, so representation search has a low ceiling by construction). A supplementary logistic-fitness run (owner-commissioned) found a real ~3% gain **for the `per_type_logistic` fallback family only** (0.2356 → 0.2286); per owner decision it is **deferred to the Commit-2 family gate** — if the calibrated XGBoost fails the per-type calibration bar there and the fallback ships, adopt a family-specific evolved builder then (`evolved_logistic_winner.py` is the candidate). The spec §6.1 "pre-registered margin" was pinned retroactively at ≥ 0.005 abs log-loss (rev 5); the null result is robust to any sane margin.
- Delivered CODE-first; bundled weights + construct-validity report land in the release's second, clean-tree-provenance commit (`training_commit`; the TF-57/TF-62 2-phase pattern). Version/PR-S/ADR claimed at that commit.

Attribution (NOTICE, ADR-005): Paul/Klemp/Memmert 2025; Anzer–Bauer Expected Passes; von Neumann–Morgenstern / Bernoulli utility; XGBoost (Chen & Guestrin 2016); OpenEvolve/AlphaEvolve.
