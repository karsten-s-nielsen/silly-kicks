# ADR-100: xImpact — match-context-weighted action value via a self-contained in-game win-probability model

| Field | Value |
|---|---|
| **Date** | 2026-09-20 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

VAEP and its outcome-bias-free refinement `VAEP_adjusted` (TF-61) value an action by how much it changes
the probability of the acting team scoring or conceding next. They are match-context-blind: a 3 m pass is
valued identically at 0-0 in the 5th minute and at 0-3 in the 89th. Paul, Klemp & Memmert (2025)
"Expected Impact on Match Outcome" weights each action's value by how much a goal at the current game
state would swing the match result — `xImpact = VAEP_adjusted × ΔP(win | goal)`.

silly-kicks already ships the pre-match outcome model (`match_outcome`, TF-53: exact Poisson-binomial
goal PMFs → the W/D/L simplex) and `VAEP.rate_adjusted` (TF-61). The missing piece is an **in-game,
time-varying** win-probability model from which the per-state goal leverage `ΔP(win | goal)` is read.
The source paper's win-probability model is Bayesian (PyMC/ADVI); silly-kicks ships no PyMC and no NN, so
the model is reimplemented classically.

## Decision

Ship a new self-contained event-only package `silly_kicks/win_probability/` — an interval-hazard logistic
feeding a **forward Markov chain on `score_diff`** — plus `goal_leverage` and a `VAEP.rate_ximpact`
rating method (mirroring `rate_adjusted`). The library ships the full per-action stack; only the
season/player ranking aggregate stays consumer-side (ADR-009). The chain **replaces, not reuses**, the
TF-53 convolution/simplex (team independence is exactly the approximation the chain removes), so
`win_probability` imports no `match_outcome`, no `tracking`, no `vaep`; `vaep → win_probability` is the
only cross-edge and it is a lazy, function-local import.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Primitive only (win-prob model); leverage + xImpact consumer-side | thinnest surface | the signed-VAEP × non-negative-leverage alignment is a silent-error footgun re-derived by every consumer | orphans an obvious primitive |
| B. Primitive + `goal_leverage`; xImpact composite consumer-side | leverage is a pure model output | same footgun for the multiply; `rate_adjusted` precedent for an in-library rate variant unused | the perspective/alignment guard belongs centralized once |
| C. Full per-action stack in-library (chosen) | one tested/glossaried `xImpact`; `rate_adjusted` precedent; ADR-009 line at the ranking | one more public rate method | — |
| Win-prob core (a): interval-convolution + TF-53 reuse | reuses `goal_count_pmf`/simplex; exact leverage | freezes `score_diff` across the projection → drops the scoring-rate feedback that IS the game-state effect; must bound a two-goal-per-interval truncation | not the gold-standard treatment (owner directive) |
| Win-prob core (b): self-contained Markov chain on `score_diff` (chosen) | generatively-correct game-state dynamics; exact leverage (re-run from score+1); two-goal minutes are net-0 (no truncation) | drops the TF-53 reuse pillar | — |
| Direct multinomial W/D/L regression | direct calibration | `ΔP(win\|goal)` = difference of two black-box evals, no coherence guarantee | coherence + exact leverage not retrofittable |

Scope decision C and win-prob core (b) were both owner-approved in the 2026-09-20 TF-63 brainstorm (the
(b) swap superseded the round-2-reviewed interval-convolution core on the explicit gold-standard
directive).

## Consequences

### Positive

- A coherent, calibrated in-game win-probability trajectory + exact per-state goal leverage + a per-action
  `xImpact`, additive and opt-in. No VAEP/tracking retrain, no re-materialize.
- The Markov chain is generatively correct (chasing/protecting feedback is inside the process) and cheap
  (~15-20 `score_diff` states × ~95 intervals); leverage is an exact chain re-run.

### Negative

- Leverage non-negativity and `score_diff`-monotonicity are **gate-enforced, not structural**: a
  state-dependent (mean-reverting) kernel is not guaranteed stochastically monotone, so a fitted hazard is
  rejected at bundle time (`certify_coherence`) if it yields negative leverage — fix the model, never
  clamp. (Option chosen: unconstrained fit + the fail-closed certification gate.)
- `p_win`/`p_draw`/`p_loss` are base-name-shared with `match_outcome` (a Hyrum collision: same column
  name, two producers/mechanisms — pre-match Poisson-binomial vs in-game Markov). The glossary entries
  document both producers under the one key; the two are different tables at different grains.
- A third inline copy of the id-based goal predicate (mirrors `match_outcome/_compute.py`); consolidation
  into a shared leaf is noted-not-built (Chesterton's Fence on `match_outcome`'s inline). Guarded by a
  cross-check test that it ≡ `vaep.labels._is_goal` on a fixture.

### Neutral

- **Chirality (ADR-011) is N/A for this model** — it has no geometric/coordinate features (state =
  score/time/strength/home/man-advantage), so there is no mirror symmetry to verify. The behavioral
  fail-closed load guard is SHA-256 + the feature-contract probe (recorded hazard outputs on a fixed
  probe). This is a deliberate, documented deviation from the ADR-011 template's SHA→contract→chirality
  triple.
- `win_probability` exports `WIN_PROBABILITY_COLUMNS` (not `*_METRIC_COLUMNS`): its output is per-action
  feature-grain, not a per-(entity, match) mart family, so it is exempt from `METRIC_CONTRACTS`
  (SK-EXPORT/ADR-098) like `xsuccess`; `win_prob_leverage` is glossaried (ADR-048), `ximpact` is a VAEP
  rating-method output (glossary-exempt like `vaep_value`). `compute_*`, not `add_*` — the action-coupled
  aggregator count stays 33; +1 C4 container.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-20-tf63-ximpact-ingame-winprob-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-20-tf63-ximpact.md`
- **Builds on:** TF-53 (`match_outcome`), TF-61 (`VAEP.rate_adjusted`)
- **External references:** Paul, Klemp & Memmert (2025) MLSA26_paper_326; Dixon & Robinson (1998)
  "A Birth Process Model for Association Football Matches"; Robberechts, Van Haaren & Davis (2019)
  "Who Will Win It? An In-Game Win-Probability Model for Football" (arXiv:1906.05029).

## Notes

The bundled public-corpus weights + the construct-validity report (late-equalizer high / garbage-time ≈ 0)
land in the release's second clean-`training_commit` commit (the TF-53/TF-57/TF-61 two-phase pattern),
with a corpus expected-goals gate (fitted model vs the empirical goals/match rate) certifying the hazard
scale, alongside the calibration gate (`ece ≤ 0.10`, `|slope − 1| ≤ 0.25`).
