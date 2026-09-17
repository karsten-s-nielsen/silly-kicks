# ADR-097: TF-53 -- match-outcome simulation (win probability / xPoints)

| Field | Value |
|---|---|
| **Date** | 2026-09-17 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

silly-kicks values individual actions (VAEP, xT) but had no MATCH-level outcome primitive: given the
shots a team took, how likely was a win, and how many league points did it deserve (xPoints)? The
Twelve match report / Soccermatics module 3 model answers this from per-shot xG. silly-kicks ships no
xG model, so xG is INJECTED (the port pattern used across `territory`/`shot_stopping`/`gk_decision`).

The naive public model (`Poisson(ΣxG)` Monte-Carlo) is wrong three ways, and the gap to gold-standard
is modelling honesty + validation rigor, not arithmetic: (1) `Poisson(ΣxG)` discards chance quality
(one 0.8 + eight 0.1 shots share ΣxG but give different win distributions); (2) same-possession shots
(save→rebound→goal) are not independent chances; (3) cross-team scorelines are not independent
(Dixon-Coles low-score dependence). Spec `2026-09-16-tf53-match-outcome-design.md`.

## Decision

Ship a new event-only `silly_kicks.match_outcome` package -- a hexagonal `compute_*` sibling (imports
`spadl` + `id_compat` + `_frame_index.group_rows` only, never `tracking`; AST import-allowlist both
directions; nothing imports it). Per `(game_id, team_id)`: injected per-shot xG → each team's **exact
Poisson-binomial** goal PMF (DP convolution over per-shot Bernoullis, NOT `Poisson(ΣxG)`) → win/draw/loss
simplex + `xpoints = 3·p_win + p_draw` + `expected_goals`.

- **Public surface:** `goal_count_pmf`, `match_outcome_probabilities`, `compute_match_outcome`,
  `MatchOutcomeParams` (frozen; `default`/`is_default`/`for_provider`), `MatchOutcomeReport`,
  `DependenceModel` + `MatchOutcomeIntegrityError` + `apply_dependence` + `dixon_coles_tau`.
- **Two orthogonal string-dispatched corrections** on `MatchOutcomeParams`: `same_possession`
  (`collapse` combines same-possession shots as `1 − ∏(1 − xg)` via `spadl.add_possessions`) and
  `team_dependence` (`dixon_coles` applies the Dixon & Coles 1997 low-score τ reweighting with a fitted
  ρ). Composable; `"independent"` on either axis recovers the naive exact Poisson-binomial.
- **Fitted ρ artifact:** pickle-free JSON (`rho`, `training_commit`, corpus, SHA256), fail-closed load
  (SHA-256 + `|ρ| < 1` range + `training_commit` present), `functools.cache`'d serve. Fit by scipy
  `minimize_scalar` MLE on the full redistributable StatsBomb open-data corpus (3,961 matches) →
  ρ = 0.0216. No Optuna (the likelihood is smooth + 1-D).
- **DEFAULT = both corrections ON.** `collapse` on CORRECTNESS (same-possession shots are not
  independent Bernoulli trials; the independent model double-counts). `dixon_coles` on MEASURED
  evidence: the full-corpus calibration study's PAIRED per-match Brier (not the aggregate) shows
  dixon_coles beats independent on 77.3% of matches (Wilcoxon p≈7e-125) and `both` beats independent on
  68.7% (p≈3e-71). This is a promotion FROM the spec's original `independent` default, taken on the
  measured study (the owner's "gold-standard best practice, measure the gating quantity" call).
- **Calibration study** (`scripts/validate_match_outcome_calibration.py`, ADR-052 `for_each` + ADR-037
  provenance + ADR-056 input contract, reported-not-gated): 3-way Brier + calibration slope +
  xPoints-vs-points per config, with a per-fold CV ρ (grouped by `game_id`, held-out, NEVER the bundled
  weights → clean 2-commit) → `docs/research/tf53_match_outcome_calibration/`.
- **Own goals** counted by RESULT (`result_id == owngoal`, ADR-018) for the Report's scoreline census,
  excluded from the xG model. Conservation census (ADR-042); honest-NaN (no-shots team → `P(0)=1`;
  NaN-xg shot excluded + counted; non-two-team game excluded + counted). Canonical-id grouped (ADR-019).

## Consequences

- Additive: no existing feature/model change, **no default VAEP retrain, no `*_xfns`** (a match-grain
  metric reading no gamestate features). +5 `feature_glossary` columns (446→451); +1 C4 container; the
  action-coupled `add_*` aggregator count is unchanged (33).
- **The default compute path now requires the bundled ρ** (fail-closed load) — acceptable: the ρ ships
  bundled in the release's second commit; a missing/tampered/out-of-range artifact RAISES rather than
  silently degrading. `"independent"` remains an artifact-free opt-in.
- **Pappalardo/Wyscout cross-provider calibration is out of scope** — public Wyscout carries no xG and
  the metric is xG-driven end-to-end; adding it would need an xG model silly-kicks does not ship.
- Honest limits: retrospective (post-hoc xG, not predictive); team independence (when not corrected) is
  an assumption; never surface G−xG year-to-year noise as finishing skill.
- 2-commit delivery (TF-57/TF-61 pattern): commit 1 = code (clean-tree provenance base); commit 2 =
  bundled ρ + calibration artifact stamping commit 1. `for_provider` ships EMPTY (ADR-009); a
  per-provider ρ tune is a separate gated PR.

## Attribution

Dixon & Coles (1997), "Modelling Association Football Scores"; the Poisson-binomial goal model;
Sumpter, Soccermatics (module 3) / the Twelve match report. Recorded in `NOTICE` (ADR-005).
