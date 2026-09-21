# In-game win-probability model -- `default` variant (TF-63, ADR-101)

**What it is.** A per-action in-game win-probability model: an interval-hazard logistic on
`(score_diff, minutes_remaining, base_strength, home, man_advantage)` feeds a forward Markov chain on
the score difference. `goal_leverage` reads the exact per-state dP(win | goal); `VAEP.rate_ximpact`
weights `VAEP_adjusted` by it. Logistic at fit, pure-numpy at serve (no runtime sklearn; the isotonic
recalibration layer is reserved and unused in this base-model v1).
Loaded via `silly_kicks.win_probability.WinProbabilityModel.bundled()`.

**Strength.** `base_strength(match, team)` = mean xG-supremacy over the team's STRICTLY-EARLIER matches
(leakage-free by date). At serve it is an injected pre-match prior (odds-derived supremacy); optional,
default even.

**Corpus + metrics.** 3961 public StatsBomb open-data matches (80 (competition, season)
releases). GroupKFold-by-match out-of-fold: ECE 0.017, reliability slope 0.991. Corpus
goals/match 2.958; model expected goals (0-0, full match) 2.465. See `metrics.json`.

**Gates.** `certify_coherence` (leverage>=0, monotone in score_diff) passes on these weights; the
expected-goals gate checks the model's expected goals match the corpus rate; calibration `ece<=0.10`
and `|slope-1|<=0.25`.

**Provenance.** `metrics.json` records `training_commit` (ae10f8a75c84ef7f8b55297c8ac1d96bbfae8a32) + tree state (clean). Pickle-free
JSON + SHA256 + feature-contract probe; `load()` is fail-closed (chirality N/A -- no geometric
features). Every bundled model carries a card (ADR-088). Attribution: Paul, Klemp & Memmert (2025);
Dixon & Robinson (1998); Robberechts, Van Haaren & Davis (2019) -- see NOTICE.
