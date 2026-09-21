# TF-63 — xImpact: match-context-weighted action value via an in-game win-probability model

- **Status:** Design (round-3 changes applied 2026-09-20 — TF63-SPEC-09/10/11 resolved, SPEC-12 folded in as a perf note; (b) self-contained Markov core stamped; awaiting re-review or plan)
- **Date:** 2026-09-20
- **Feature:** TF-63 (On Deck)
- **Proposed decision record:** ADR-101 (to be written on approval)
- **Proposed release:** silly-kicks (next available minor from `main`; PR-S number assigned at ship)
- **Source:** Paul, Klemp & Memmert 2025 (MLSA 2026, paper MLSA26_paper_326 — "Expected Impact on Match Outcome")
- **Builds on:** TF-53 `match_outcome` (pre-match Poisson-binomial simplex, shipped), TF-61 `VAEP.rate_adjusted` (outcome-bias-free action value, shipped)

---

## Executive summary (for the reviewer)

TF-63 delivers **xImpact** — an action value weighted by how much a goal at the current game state
would swing the match result. A late equalizer scores high; a fourth goal in stoppage time of a 4-0
match scores ≈ 0.

The value identity is:

```
xImpact(action) = VAEP_adjusted(action) × ΔP(win | goal at the pre-action game state)
```

`VAEP_adjusted` already ships (TF-61). The **new, reusable library primitive** is an **in-game,
time-varying win-probability model** `P(win/draw/loss | game state)`, from which the leverage term
`ΔP(win | goal)` is derived exactly. This is the in-game, time-conditioned extension of TF-53's
pre-match, whole-match `match_outcome`.

Three scope decisions were taken during brainstorming and are load-bearing:

1. **Scope boundary (owner: C).** The library ships the full per-action stack — the win-probability
   model, the `goal_leverage` primitive, and a `VAEP.rate_ximpact` rating method (mirroring the
   already-shipped `VAEP.rate_adjusted`). Only the season/player **ranking aggregate** stays
   consumer-side (ADR-009). Rationale: the actual-vs-hypothetical-goal perspective/alignment step is a
   silent-error footgun that belongs centralized and tested once, and `rate_adjusted` sets the
   in-library precedent for a VAEP rating variant.

2. **Win-prob core (owner: approach (b), self-contained Markov chain — revised 2026-09-20).** The
   interval-hazard GLM is retained, but `P(win)` is computed by a **forward Markov chain on `score_diff`**,
   not by independent-interval convolution: per remaining minute the chain steps `+1` (home scores, away
   doesn't) / `−1` (away scores) / `0` (neither or both), with the step probabilities from the two teams'
   hazards evaluated at the *current, evolving* `score_diff` and time, propagated to the final
   `P(score_diff >0 / =0 / <0)` = win/draw/loss. This is the generatively-correct treatment of the
   game-state effect (the chasing/protecting feedback is *in* the process, not just at the leverage
   endpoints), handles two-goals-in-one-minute cleanly (net `0` on `score_diff`), and is cheap
   (~15-20 `score_diff` states × ~95 intervals). Leverage `ΔP(win | goal)` is exact: re-run the chain from
   the `score+1` initial state. **Consequence:** team-independence is exactly the approximation this
   removes, so the TF-53 reuse pillar is dropped — the chain replaces both `goal_count_pmf` and
   `match_outcome_probabilities`, and `win_probability` becomes **self-contained** (no `match_outcome`
   engine dependency). This revision supersedes the round-2-approved interval-convolution core and needs
   re-review. Coherence is partly structural, partly gate-enforced (§ 3.2, § 9.1): the exact
   goal-difference and dead-state → 0 ARE structural, but **leverage non-negativity and monotonicity in
   `score_diff` are gate-enforced, not structural** — a state-dependent (mean-reverting) kernel does not
   guarantee terminal-`P(win)` monotonicity in the initial state without a monotone-kernel constraint.
   Calibration is added with the codebase's existing discipline. Rejected: independent-interval
   convolution (approach (a)) and a direct multinomial regression (§ 12).

3. **Feature set + injected ports (owner-confirmed).** The hazard state is
   `(score_diff, time_remaining, base_strength, home, man_advantage)`. The only injected serve-time
   port is `base_strength` (the pre-match supremacy / betting-odds prior); per-shot xG is a
   **trainer-only** input (it builds the leakage-free strength rating). xImpact scores **completed
   matches retrospectively**, so `time_remaining` is observed and there is no added-time forecast.

Additive: no VAEP or tracking retrain, no re-materialize. `compute_*`, not `add_*` — the action-coupled
aggregator count stays 33. +1 C4 container (`win_probability`).

---

## 1. Motivation and the gap being closed

VAEP (and its outcome-bias-free refinement `VAEP_adjusted`, TF-61) value an action by how much it
changes the probability of the acting team scoring or conceding *next*. They are **match-context-blind**:
a 3 m progressive pass is valued identically at 0-0 in the 5th minute and at 0-3 in the 89th. The Paul
et al. 2025 argument is that an action's *impact on the match result* depends on the game state — the
same expected-goal change matters far more when it can flip the outcome.

silly-kicks already owns the two halves this needs except one:

- The **pre-match** outcome model (`match_outcome`, TF-53): exact Poisson-binomial goal PMFs per team →
  the W/D/L simplex. Static: no current-score or time conditioning.
- The **outcome-bias-free action value** (`VAEP.rate_adjusted`, TF-61).

The missing piece is an **in-game** win-probability model that conditions on the live game state, from
which the per-state goal leverage is read. That is TF-63's library contribution.

The paper's own win-probability model is Bayesian (PyMC/ADVI). silly-kicks ships no PyMC and no NN
(scipy/sklearn idiom only), so the model is reimplemented classically. This is a reimplementation of
the *methodology*, not a port of the *code*.

## 2. Non-goals

- **No player/season xImpact ranking in the library.** The ranking aggregate is a consumer concern
  (ADR-009), exactly as VAEP season tables are.
- **No live/real-time forecasting.** xImpact is a retrospective valuation of actions in completed
  matches; `time_remaining` is observed. A genuine live model (unknown remaining/added time) is a
  different consumer product and is out of scope. This is *not* a deferred item and reserves *no* hook.
- **No xG or betting-odds model shipped.** Both are injected (port pattern), consistent with the whole
  `xg_column` idiom. silly-kicks ships neither.

## 3. Scope decisions and their rationale

### 3.1 Scope boundary — full per-action stack in-library (decision C)

Options considered:

- **(A)** Primitive only (win-prob model); leverage and xImpact both consumer-side.
- **(B)** Primitive + `goal_leverage`; xImpact composite consumer-side.
- **(C, chosen)** Primitive + `goal_leverage` + `VAEP.rate_ximpact`; only the ranking aggregate stays
  consumer-side.

The deciding factor is that the xImpact multiply is not a trivial one-liner: `ΔP(win | goal)` is a
**non-negative state weight for the acting team**, while `VAEP_adjusted` is a **signed** quantity
(offensive − defensive, acting-team perspective). The multiply must weight the signed value by the
same-team leverage at the same pre-action state. Getting the team or the slot wrong produces a silently
wrong xImpact with no crash. Centralizing that alignment/perspective guard once (with a
perspective-invariance test) is worth more than the thin ADR-009 purity of pushing it out. The
in-library precedent is `VAEP.rate_adjusted` — the method on `VAEP` at `vaep/base.py:428` (its
`adjusted_value` helper lives in `vaep/adjusted.py`, imported lazily at `base.py:477`), a rate variant
that already ships — so a sibling `rate_ximpact` is consistent rather than novel.

**Owner approval (provenance for TF63-SPEC-01):** decision C — moving the composite in-library — was
approved by the owner in the TF-63 brainstorm on 2026-09-20 (this cycle's brainstorm session), which
supersedes the earlier B-shape phrasing in `TODO.md:25`; `TODO.md` is updated in the same change to
match. Only the season/player **ranking aggregate** stays consumer-side (ADR-009).

The layering is acyclic — see § 4.2.

### 3.2 Win-prob core — self-contained forward Markov chain on `score_diff` (approach (b))

`P(win | score_home, score_away, t)` is computed by a **forward Markov chain on `score_diff`**, using the
per-minute interval-hazard GLM (§ 5.1) as the transition engine:

1. Discretize the *remaining* match into per-minute intervals; the chain's state is the current
   `score_diff` (own − opponent), on a bounded integer lattice (`[−K, +K]`, `K` a small pad; mass beyond
   the pad is negligible and pinned by a validation gate).
2. Per interval, evaluate two hazards — `p_home = P(home scores this minute | score_diff, time, home
   strength/flags)` and `p_away = P(away scores this minute | −score_diff, time, away strength/flags)`.
   The per-interval `score_diff` transition is `+1` w.p. `p_home·(1−p_away)`, `−1` w.p.
   `p_away·(1−p_home)`, `0` otherwise (neither, or both — a net-zero step, which is why a two-goal minute
   needs no truncation handling).
3. Propagate the `score_diff` distribution across the remaining intervals (a vector–matrix recursion,
   the transition re-evaluated per interval because `time` and the current `score_diff` both change).
4. Read the outcome directly from the final `score_diff` distribution:
   `P(win) = P(final score_diff > 0)`, `P(draw) = P(= 0)`, `P(loss) = P(< 0)`. No simplex construction —
   the chain yields the outcome distribution.

Why the chain (approach (b)) over independent-interval convolution (approach (a)) and over a direct
regression:

- **Generatively correct game-state dynamics.** The scoring rate depends on the live `score_diff`
  (chasing/protecting), and `score_diff` evolves as goals are scored. The chain carries that feedback
  *inside* the process. Independent-interval convolution (a) freezes `score_diff` across the projection
  — a defensible simplification, but not the gold-standard treatment. Owner directive: gold standard
  (dated-brainstorm provenance stamped at the end of this section).
- **Coherence — partly structural, partly gate-enforced (TF63-SPEC-09).**
  `ΔP(win | goal) = P(win | score+1) − P(win | score)` is an **exact** goal difference (two runs of the
  same chain from initial states differing by one) and the **dead-state → 0** case is structural (with
  zero remaining intervals both terms are 1, so `ΔP = 0`). But **leverage non-negativity and
  monotonicity in `score_diff` are NOT structural** for a state-dependent kernel: the game-state effect
  makes the kernel mean-reverting (a leader's scoring hazard falls), and a mean-reverting birth–death
  chain is not guaranteed stochastically monotone — a monotone coupling can cross (from `d+1` a
  down-step against `d` an up-step), so terminal `P(win)` monotonicity in the initial `score_diff` is not
  automatic. Two options, decided at implementation (§ 13): **(i)** constrain the GLM `score_diff`
  coefficient to a monotone/sign-restricted form that makes the kernel stochastically monotone → genuinely
  structural; or **(ii)** leave the hazard unconstrained and rely on the § 9.1 hard, red-first
  leverage-≥0 + monotonicity gates, **rejecting at bundle time any fitted model that violates them** (a
  violation signals a mis-specified model, to fix — never to clamp). Either way the *bundled artifact* is
  monotonicity-certified; the spec no longer claims non-negativity is free. A direct regression has
  neither the exact difference nor a gate that can certify it cheaply.
- **Two-goal minutes need no truncation handling.** Both-score is a net-zero `score_diff` step, so the
  Bernoulli-per-interval truncation that approach (a) had to bound simply does not arise.
- **Calibration is retrofittable** via the codebase's existing discipline (isotonic on the derived
  win-prob against held-out outcomes; ECE / reliability-slope gates already used by `GkRetentionModel`
  and `xSuccess`).

**Consequence (the TF-53 reuse pillar is dropped).** Team-independence is exactly the approximation the
chain removes, so `goal_count_pmf` (the independent-Bernoulli convolution) and
`match_outcome_probabilities` (the independent outer-product simplex) are no longer used —
`win_probability` is **self-contained**. `match_outcome` remains its pre-match, independent-team sibling;
there is no code dependency between them (see § 4.2). This is the change that supersedes the
round-2-approved core and is the subject of the re-review.

**Owner approval (provenance for TF63-SPEC-11).** The core swap from approach (a) (interval-convolution
+ TF-53 reuse, round-2-approved) to approach (b) (the self-contained Markov chain) was approved by the
owner in the TF-63 planning session on 2026-09-20, on the explicit gold-standard directive, after being
shown that (b) is the generatively-correct treatment of the game-state effect and that it drops the
TF-53 reuse pillar. This carries the same dated-brainstorm provenance stamp as decision C (§ 3.1); it is
not a self-authored reversal of the reviewer-approved core.

The chain requires the hazard to depend only on `score_diff` (plus per-match-fixed strength/home/
man-advantage and time) so the evolving state is one-dimensional. That is exactly the § 3.3 feature set:
`strength`, `home`, `man_advantage` are fixed across a forward roll (future substitutions/cards are
unknown and not projected), and `score_diff` + `time` are the two evolving inputs. The interval hazard
**must** condition on `score_diff` and `time_remaining` (the game-state effect); a homogeneous Poisson
would miss exactly the effect that makes a late equalizer high-leverage.

### 3.3 Feature set and the single injected port

The per-interval hazard state is:

| Feature | Source | Notes |
|---|---|---|
| `time_remaining` | derived | Absolute match minute = period offset + `time_seconds`; `time_seconds` is **period-relative** (resets each period), so the trainer and the serve path both rebuild the absolute minute. `time_remaining = observed_final_minute − current_minute`. |
| `score_diff` | derived | Acting-team perspective; cumulative goals to date. |
| `base_strength` | **injected (port)** | Pre-match supremacy (expected goal difference). Optional; default 0.0 (even, an honest neutral prior). |
| `home` | derived | `team_id == home_team_id`. Explicit feature — not folded into `base_strength`, so a consumer injecting a neutral-venue rating does not lose home advantage. |
| `man_advantage` | derived | Cumulative red-card differential (direct red + second yellow), event-only. |

**The only injected serve-time port is `base_strength`.** This tightens the TODO's "injected xG + a
betting-odds prior" to one port, and the tightening is a consequence of two earlier decisions:

- Live-xg-so-far was excluded as a state variable (owner decision), so the hazard consumes no live xG.
- Leverage is the swing of a hypothetical **+1** goal, independent of any actual shot's xG, so there is
  no serve-time per-shot-xG path.

Per-shot xG therefore appears **only** in the trainer, to construct the leakage-free strength rating
(see § 5.2). `base_strength` at serve is the betting-odds prior (or any consumer rating); at training it
is the leakage-free rating. It is the same feature, sourced differently at the two times — the standard
train/serve port split.

`base_strength` is **optional** with an even-strength default: the model works out of the box on
`(score_diff, time_remaining, home, man_advantage)` and sharpens when a strength prior is injected.
Even-strength (supremacy 0.0) is an in-distribution value, not a fabrication.

### 3.4 Considered and excluded (not deferred, no reserved hook)

The following were evaluated and are **out of scope for this metric**, recorded here as decisions, not
as future work. They introduce no TODO row and reserve no extension hook; a future need is a future
feature's own brainstorm.

- **Possession / momentum features.** No measured requirement; adds state without a validated
  contribution to leverage.
- **Live accumulated xG as a state variable.** Explicitly excluded (owner): leverage is a
  hypothetical-goal swing, not a function of xG created so far.
- **Weather / lineup / fatigue features.** Out of scope for an event-only, corpus-portable primitive.
- **Live added-time forecasting.** See § 2 non-goals — retrospective valuation observes the final
  minute.

## 4. Architecture

### 4.1 Package layout

```
silly_kicks/win_probability/
    __init__.py          # public surface
    _model.py            # WinProbabilityModel: fit / save / load / serve (interval-hazard GLM)
    _chain.py            # forward Markov chain on score_diff (transition build + propagate + outcome read)
    _compute.py          # compute_win_probability, goal_leverage
    _state.py            # event-only game-state derivation (score, minute, man-advantage, home)
    _config.py           # WinProbabilityParams (frozen)
    _columns.py          # column-set constants
    _report.py           # WinProbabilityReport (conservation census)
    weights/             # bundled model artifact (JSON + SHA256SUMS + metadata.json)
```

`vaep/adjusted.py` (or a sibling `vaep/ximpact.py`) gains the `rate_ximpact` helper; `VAEP.rate_ximpact`
in `vaep/base.py` builds the counterfactual and calls it, mirroring `rate_adjusted`.

### 4.2 Dependency direction (hexagonal, acyclic — resolves TF63-SPEC-02)

```
win_probability  →  spadl, id_compat, numpy  (id-based state derivation + Markov chain; NO vaep, NO match_outcome engine dep)
vaep             →  win_probability  (rate_ximpact only, LAZY function-local import)
```

`win_probability` is **self-contained** — the (b) Markov chain replaced the `goal_count_pmf` /
`match_outcome_probabilities` reuse, so there is no `win_probability → match_outcome` code dependency.
`match_outcome` remains the independent-team pre-match sibling; a consumer may run both, but neither
imports the other. (An optional cross-check test may import `match_outcome` to compare the two models'
pre-match numbers, but it is a test-only import and the two are expected to *differ* — the chain couples
the teams through `score_diff`, `match_outcome` assumes independence — so it is a sanity band, not an
equality gate.)

The package graph is **acyclic**, and the earlier `vaep ⇄ win_probability` cycle the reviewer flagged
is removed at the source, not merely by laziness:

- **`win_probability` imports no `vaep`.** Current score is derived **id-based** — a scored shot is
  `type_id ∈ {shot, shot_penalty, shot_freekick}` with `result_id == success`; own goals are
  `result_id == owngoal` credited to the other team (ADR-018 own-goal-by-result). This mirrors the
  already-accepted id-based counting in `match_outcome/_compute.py` (`_OWNGOAL = result_id["owngoal"]`,
  shot `type_id`) and needs **no `add_names`** and no `vaep.labels`. The id-based and the name-based
  `vaep.labels._is_goal`/`_is_owngoal` encode the *same* ADR-018 rule (one on `type_id`/`result_id`,
  one on `type_name`/`result_name`); this is not a new divergence — `match_outcome` is already an
  event-only id-based site that does not route through `vaep.labels`.
- **`vaep → win_probability` is the ONLY cross-edge, and it is lazy.** `rate_ximpact` imports
  `win_probability` function-locally (the exact `rate_adjusted` precedent — `base.py:477` imports
  `adjusted_value` lazily), so `vaep/__init__` and `vaep/base.py` module-load do not import
  `win_probability`. No cycle even at module-load grain.

`win_probability` imports **no `tracking`** and **no `match_outcome`** (self-contained). Enforced by an
AST import-allowlist gate `tests/win_probability/test_import_allowlist.py`, checked in both directions:
the `win_probability` package may import `spadl` / `id_compat` / numpy / pandas only (NOT `tracking`,
NOT `vaep`, NOT `match_outcome`); nothing imports `win_probability` except `vaep`, and `vaep`'s import
must be function-local (a module-level `import win_probability` in `vaep` fails the gate). This is the
event-only sibling shape of `match_outcome` / `shot_stopping` / `territory` / `duels`.

A future consolidation of the id-based goal predicate into a shared leaf (below both `match_outcome`
and `win_probability`) is a possible cleanup, noted not built — the two inlined copies encode one rule
and Chesterton's Fence applies to `match_outcome`'s existing inline.

### 4.3 Public API

```python
from silly_kicks.win_probability import (
    WinProbabilityModel,
    compute_win_probability,
    goal_leverage,
    WinProbabilityParams,
    WinProbabilityReport,
    WIN_PROBABILITY_COLUMNS,
)

# Per-action win-probability trajectory + leverage.
samples, report = compute_win_probability(
    actions, model=WinProbabilityModel.bundled(), strength_column="home_supremacy",
)

# Leverage alone (ΔP(win | goal) at each pre-action state).
lev = goal_leverage(actions, model=WinProbabilityModel.bundled())

# The composite rate method (in vaep).
vaep_model = VAEP(...).fit(...)                     # a STANDARD, result-bearing VAEP
xi = vaep_model.rate_ximpact(actions, win_prob_model=WinProbabilityModel.bundled())
```

- `compute_win_probability` returns per-action rows aligned to `actions`, carrying
  `p_win` / `p_draw` / `p_loss` / `win_prob_leverage` plus provenance, over `WIN_PROBABILITY_COLUMNS`,
  and a `WinProbabilityReport`.
- `goal_leverage` returns an `actions`-aligned `pd.Series` of `ΔP(win | goal)` at each pre-action state.
- `VAEP.rate_ximpact` returns an `actions`-aligned `pd.Series` of `VAEP_adjusted × win_prob_leverage`.
  It **raises on `HybridVAEP`** (inherited from `rate_adjusted` — the result-feature counterfactual is a
  no-op there; non-vacuity guard). A NaN leverage or NaN `VAEP_adjusted` propagates to NaN xImpact
  (never fabricated).

## 5. The win-probability model

### 5.1 Hazard model class

The default hazard is a **calibrated GLM** (logistic on the per-interval, per-team goal indicator) with
a pure-numpy serve path, plus an isotonic recalibration layer. GLM is chosen over a gradient-boosted
model because:

- It is smooth and keeps monotonicity tractable (signed `score_diff` and `time_remaining` terms), which
  the coherence gates require.
- It is calibrated-by-construction (a small isotonic layer closes any residual ECE).
- Pure-numpy serve keeps inference sklearn-free (the `xSuccess` / `GkCompletionModel` serve idiom).

**v1 ships exactly one hazard model (the GLM) and NO dispatch scaffolding** — no `method=` selector, no
string-dispatch family, no reserved door (consistent with § 3.4's no-speculative-hooks stance). A
gradient-boosted or otherwise alternative hazard is a future feature's own brainstorm if a measured need
arises; it is not a hook built now.

### 5.2 Training and the leakage-free strength rule (correctness gate)

- **Corpus:** public StatsBomb open-data (3961 matches — the corpus the `DependenceModel` ρ and the
  re-bundled `PassCompletionModel` use). `assert_public_corpus`-gated; clean `training_commit`;
  driver refuses a dirty tree (`scripts/_provenance.py`).
- **Label:** for each (match, minute-interval, team), whether that team scored in that interval, derived
  **id-based** (the same `type_id`/`result_id` rule as § 7 — NO `vaep` import in the trainer either;
  ADR-018 own-goal-by-result, never a `type_name` shot-gate).
- **Strength label — the load-bearing correctness rule:** the per-state `base_strength` used in
  training must be built from **prior matches only** (a chronological xG-Elo / rating computed in the
  trainer), never from the match's own aggregate. Using the match's own shots to set its pre-match
  strength leaks the outcome. This is a spec-level correctness requirement with a dedicated leakage
  test (perturb a future match's result → a past match's training strength byte-identical).
- **Calibration:** held-out `ece ≤ 0.10` AND `|reliability_slope − 1| ≤ 0.25`, CI-certified against the
  artifact's recorded `metrics.json`.

### 5.3 Serve (the forward Markov chain)

At serve, the model needs only the state — no per-shot xG. The `_chain.py` engine (§ 4.1):

1. Initializes the `score_diff` distribution as a point mass at the current `own_score − opp_score`.
2. For each remaining minute, builds the per-interval transition on the `[−K, +K]` lattice from the two
   team hazards evaluated at the current `score_diff` and the interval's `time_remaining` — the GLM
   serve is a pure-numpy `sigmoid(X @ beta)` (§ 5.1) — and applies it (a vector–matrix step).
3. After the last interval, reads `P(win) = P(score_diff > 0)`, `P(draw) = P(= 0)`,
   `P(loss) = P(< 0)`, then applies the isotonic recalibration map (§ 5.1).

Two-goal minutes require no special handling: a both-score minute is a net-zero `score_diff` step, so the
truncation bound approach (a) needed does not arise. There is **no `team_dependence` / Dixon-Coles
correction** here — that was a `match_outcome_probabilities` (independent-team) feature; the chain models
the coupled `score_diff` process directly, so the correction has no place in this core (the earlier
open question about its default is therefore dissolved, § 13).

**Interval width is pinned at one minute** — fine-grained enough that at most one goal per team per
interval is the norm, cheap to propagate, and finer buys nothing measurable. The lattice pad `K` is
chosen so the tail mass beyond `±K` is negligible; a validation gate pins that the chain's total expected
goals match the empirical corpus rate (§ 9.1), catching a mis-set `K` or a mis-scaled hazard.

### 5.4 Bundled artifact

Pickle-free JSON weights + `metadata.json` + `SHA256SUMS`, fail-closed `load()` (SHA → feature-contract
→ chirality, ADR-011/016/040/050). `WinProbabilityModel.bundled()` serves the public-corpus default
(`functools.cache`'d). A missing / tampered / out-of-range artifact **raises** — never silently
degrades. `metadata.json` records `training_commit`, corpus taxonomy, and the calibration metrics.

## 6. Leverage and rate_ximpact

### 6.1 goal_leverage

For each action, the pre-action game state is derived (§ 7), and:

```
win_prob_leverage(action) = P(win | score + 1 for the acting team, state)
                          − P(win | score, state)
```

Both terms are the same forward chain (§ 5.3) evaluated from two initial `score_diff` states differing by
one, so the difference is **exact** (it is a true goal increment, not an approximation); its
**non-negativity is gate-enforced** (§ 9.1), not structural (TF63-SPEC-09). For a period-boundary or
otherwise unresolvable state, the value is honest-NaN.

**Performance (corpus scale, TF63-SPEC-12) — precompute, do not re-roll per action.** Rolling the chain
per action is O(actions × intervals × states). Instead, per match precompute **one** backward-induction
table `Pwin[score_diff, minute]` (~`(2K+1) × ~95` cells) by backward DP from the terminal
`score_diff` distribution using the same per-interval transitions, then every action's leverage is an
O(1) lookup `Pwin[d+1, m] − Pwin[d, m]`. This collapses the per-action cost to a table build per match
plus O(1) per action — the ADR-068/073 no-rescan-in-loop discipline, and the analogue of the
"expensive per-item work is precomputed once" pattern. The plan builds `_chain.py` with this table as the
primary API and the single-state roll as a thin special case.

### 6.2 VAEP.rate_ximpact

```
xImpact(action) = VAEP_adjusted(action) × win_prob_leverage(action)
```

`rate_ximpact` reuses `rate_adjusted` verbatim for `VAEP_adjusted`, computes `win_prob_leverage`, aligns
both on the action index and the **same acting-team perspective**, and multiplies element-wise. The
alignment/perspective guard is unit-tested with a perspective-invariance test: a defensive action's
xImpact must be weighted by the same-state leverage, and swapping the acting-team perspective must move
the number in the specified way (a non-vacuous counterfactual, per the codebase's both-sides-of-the-band
rule).

`rate_ximpact` reads no post-action outcome beyond what `rate_adjusted` already reads as a training
label — leverage is entirely state-based (score, time, strength, home, man-advantage) — so there is no
`*_xfns` factory and no default-xfn-list membership (a leaky factory would be a HybridVAEP-class break;
this feature has no leakage surface, so it ships no factory at all, consistent with the ADR-030 /
ADR-047 posture).

## 7. Event-only game-state derivation

All state is derived from the SPADL actions, event-only:

- **Cumulative score:** running goals derived **id-based**, mirroring `match_outcome/_compute.py` (NO
  `add_names`, NO `vaep` import — see § 4.2): a scored shot is `type_id ∈ {shot, shot_penalty,
  shot_freekick}` with `result_id == success`; own goals are `result_id == owngoal` credited to the
  *other* team (ADR-018 own-goal-by-result — never a `type_name` shot-gate). This encodes the same rule
  as `vaep.labels._is_goal`/`_is_owngoal`, on ids rather than names.
- **Absolute minute:** period offset + `time_seconds` (period-relative → rebuilt). Sorted on the robust
  `(game_id, period_id, time_seconds, action_id)` key (ADR-065 §3d — never `action_id` alone).
- **man_advantage:** cumulative red-card differential to the action's time (`result_id == red_card` on a
  `foul` action, plus a per-player second-yellow → red derivation).
- **home:** `id_compat.ids_match(team_id, home_team_id)` (ADR-019 — never raw `==`).
- **Pre-action vs at-action:** leverage and xImpact use the state **before** the action (the same slot
  VAEP values a0). The score increment counted for an action's own goal belongs to the *next* state, not
  its own leverage state.

## 8. Conventions compliance

- **ids:** canonical via `id_compat` (ADR-019); grouped on the canonical id, raw id emitted; `home` and
  team comparisons via `ids_match` / `ids_equal`, never raw `==`.
- **Conservation (ADR-042):** `WinProbabilityReport` conserves the match population — a game whose
  actions carry ≠ 2 distinct team ids is excluded-and-counted, mirroring `MatchOutcomeReport`. An
  unresolvable action-state is counted, not dropped-silent.
- **Honest-NaN (ADR-027 / ADR-055):** an unresolvable state → NaN across all outputs; never a fabricated
  0 or 0.5.
- **compute_*, not add_*:** the action-coupled aggregator count stays **33**; in NO default xfn list;
  no `*_xfns`.
- **feature_glossary (ADR-048):** the per-action columns (`p_win`, `p_draw`, `p_loss`,
  `win_prob_leverage`, and the `ximpact` column a consumer materializes from `rate_ximpact`) are
  documented; `rate_ximpact` is a VAEP rating method (the `xsuccess` exemption class — no mart column
  set). **metric_contracts (SK-EXPORT / ADR-098):** `win_probability`'s output is per-action
  (feature-grain), not a per-(entity, match) mart family, so it is expected to be **exempt** from
  `METRIC_CONTRACTS` (like `xsuccess`); this exemption is confirmed against
  `tests/test_metric_contracts.py` at implementation time (the package exports its column constants under
  a non-`*_METRIC_COLUMNS` name so the completeness gate does not enroll it, with the exemption recorded
  in the test's exempt set).
- **NOTICE (ADR-005):** three concrete citations, not a placeholder —
  **Paul, Klemp & Memmert 2025** (MLSA 2026, MLSA26_paper_326) for the xImpact match-context weighting;
  **Dixon & Robinson 1998**, "A Birth Process Model for Association Football Matches" (*The Statistician*
  47(3), 523-538) for the time- and state-dependent scoring-intensity match process the Markov chain
  reimplements classically; and **Robberechts, Van Haaren & Davis 2019**, "Who Will Win It? An In-Game
  Win-Probability Model for Football" (arXiv:1906.05029) for the in-game win-probability application.
  `match_outcome` keeps its own Dixon-Coles / Poisson-binomial citations (no longer reused here). Exact
  bibliographic fields are re-verified when the `NOTICE` entry is written (the ADR-005 discipline);
  per-feature docstrings cross-link.
- **C4:** +1 container (`win_probability`) → update `docs/c4/architecture.dsl` and regenerate
  `architecture.html`. `WinProbabilityParams` frozen; `for_provider` empty per ADR-009.
- **Warnings:** any `warnings.warn` carries `stacklevel=2`; a new warning category (e.g. for an
  unresolved-state notice) is its own class, not an umbrella (per the separate-categories convention).

## 9. Validation gates

### 9.1 Coherence (CI-gated, red-first)

- `win_prob_leverage ≥ 0` weakly, everywhere.
- `P(win)` monotone non-decreasing in `score_diff` (state otherwise fixed).
- **Dead-state:** a 4-0 (or symmetric) lead at 90' → `win_prob_leverage ≈ 0` and `P(win) ≈ 1` for the
  leader.
- **Kickoff anchor:** at `(t = 0, 0-0)`, model `P(win)` equals the injected pre-match prior (or the
  even-strength prior when none injected). A hard coherence check the direct-regression approach cannot
  offer.
- **Expected-goals identity:** the chain's total expected goals over a full match (from `(t = 0, 0-0)`)
  must match the empirical corpus goals-per-match rate within tolerance, catching a mis-set lattice pad
  `K` or a mis-scaled hazard (§ 5.3).
- **Mass conservation:** the `score_diff` distribution sums to 1.0 (± float tolerance) at every interval
  step — a total-probability check on the vector–matrix propagation.
- **Chain–recompute leverage identity:** `goal_leverage` equals `P(win | chain from score+1) −
  P(win | chain from score)` computed directly, proving the leverage is the exact chain re-run and not an
  approximation (the (b) analogue of the dropped simplex-reuse identity).

### 9.2 Calibration (CI-gated vs recorded metrics)

Held-out `ece ≤ 0.10` AND `|reliability_slope − 1| ≤ 0.25`, certified against the bundled artifact's
`metrics.json`.

### 9.3 Construct validity (reported, not gated)

A face-validity report (`docs/research/tf63_ximpact/`) on the public corpus: late equalizers score
high, garbage-time goals ≈ 0, and the distribution of xImpact vs VAEP shows the expected re-weighting.
Reported-not-gated; promotes no default.

## 10. Testing / TDD plan

Every gate lands red-first (test observed failing before the implementation exists).

- **Golden trajectory:** a fixed toy match with hand-derived state → a known `P(win)` trajectory and
  known leverage at chosen actions.
- **Leverage exactness:** `goal_leverage` equals the difference of two forward-chain evaluations
  (initial `score_diff` and `score_diff + 1`), proving it is the exact chain re-run.
- **Chain mass conservation:** the propagated `score_diff` distribution sums to 1.0 at every step, and
  the transition rows sum to 1.0.
- **Perspective invariance / non-vacuity:** `rate_ximpact` weights signed VAEP by same-team leverage;
  a mutation that should move xImpact out of an expected band does so (both sides of the band).
- **HybridVAEP raise:** `rate_ximpact` raises on a HybridVAEP (the `rate_adjusted` guard).
- **Leakage-free strength:** perturbing a future match's outcome leaves a past match's training strength
  byte-identical.
- **Import allowlist:** both directions (`win_probability` imports no `tracking`; nothing but `vaep`
  imports `win_probability`).
- **Purity:** `compute_win_probability` / `goal_leverage` / `rate_ximpact` never mutate `actions`
  (ADR-033 shape).
- **Fail-closed load:** SHA / feature-contract / chirality mismatches raise (ADR-011/040/050).
- **Conservation:** `WinProbabilityReport` conserves the match population (ADR-042).

Suite command: `python -m pytest tests/win_probability tests/vaep -m "not e2e" -v --tb=short`, plus the
full suite green before the human-approval commit gate. Lint at CI scope
(`python -m ruff check silly_kicks/ tests/ scripts/`) and bare `python -m pyright`.

CI collects the new `tests/win_probability/` directory automatically — the CI scope is `pytest tests/`
(duration-sharded, ADR-074), so no per-package registration is needed. The test directory **carries an
`__init__.py`**, matching the repo convention for metric test dirs (`tests/match_outcome/`,
`tests/territory/`, `tests/vaep/`): under pytest's default prepend import-mode, identical test-file
basenames across dirs (`test_compute.py` / `test_config.py` / `test_import_allowlist.py`, also under
`tests/restdefense/`) collide unless package-qualified. (The namespace-shadow trap applies only to
`tests/scripts/` mirroring the `scripts` NAMESPACE package; `win_probability` is a regular package, so
`tests.win_probability` shadows nothing. Corrected from the initial "no `__init__.py`" note — TF63-IMPL-01
— after a full-suite `import file mismatch` proved the collision.)

## 11. Release shape and the human-approval gate

Two commits, each a fully-tested coherent state (never micro-commits):

1. **Code + docs + C4 + gates** (this spec and the ADR land in this first commit — no standalone doc
   commit). No bundled weights yet; tests that need the model use a small fixture-fit model.
2. **Bundled weights + construct-validity report**, with a clean `training_commit` (the TF-53 / TF-57 /
   TF-61 two-phase clean-provenance pattern).

Both commits are additive: **no VAEP or tracking retrain, no re-materialize** (xImpact is a new,
opt-in per-action value; existing columns are byte-identical).

**Commit discipline (blocking):** neither commit is made without Karsten's explicit approval for that
specific commit. The full suite is green and the diff is shown *before* asking; a green suite is not
approval. The plan produced from this spec must carry an explicit human-approval gate immediately
before each commit and before any push/merge/tag — a plan that lands code without that gate is a review
block.

## 12. Rejected alternatives

- **Independent-interval convolution (approach (a), the round-2-approved core).** For a single `P(win)`
  it freezes `score_diff` across the remaining intervals and convolves independent Bernoullis
  (`goal_count_pmf`) + the independent simplex (`match_outcome_probabilities`), reusing TF-53. Rejected in
  favour of the Markov chain on the owner's gold-standard directive: freezing `score_diff` drops the
  scoring-rate feedback that *is* the game-state effect, and it must bound a Bernoulli-per-interval
  two-goal truncation the chain avoids entirely. (a) is a legitimate, cheaper simplification whose
  calibration gap vs (b) is likely second-order over short late-game horizons; it was rejected on the
  gold-standard bar, not on a measured failure.
- **Direct multinomial / ordered-logistic outcome regression** `P(W/D/L | score_diff, minutes,
  strength[, cards])`. Rejected: its `ΔP(win | goal)` is a difference of two independent classifier
  evaluations with no coherence guarantee (can go negative on noise), and it is a discriminative
  black-box rather than a generative state-dependent process. The chain's exact, non-negative leverage
  and structural coherence are not retrofittable onto it; the one property the regression has natively —
  direct calibration — is retrofittable onto the chain (isotonic recalibration).
- **Bayesian win-prob (PyMC/ADVI), as in the source paper.** Rejected: silly-kicks ships no PyMC and no
  NN; the methodology is reimplemented classically.
- **Two injected serve ports (xG and odds), per the TODO's original phrasing.** Rejected/tightened to
  one port (`base_strength`): live xG is not a state variable (owner decision) and leverage is
  xG-independent, so per-shot xG has no serve-time path — it is trainer-only.
- **Scope boundary B (leverage in-library, xImpact composite consumer-side).** Rejected in favour of C:
  the signed-VAEP × non-negative-leverage perspective alignment is a silent-error footgun best
  centralized, and `rate_adjusted` is the in-library precedent for a rate variant.

## 13. Open questions for the reviewer

- The exact home for `rate_ximpact` (`vaep/adjusted.py` extension vs a new `vaep/ximpact.py`). Either
  keeps the layering; the split is a file-organization call.
- The lattice pad `K` and the isotonic-recalibration placement (on the final win-prob vs on the hazard).
  Both are implementation-detail choices the calibration + mass-conservation gates constrain; called out
  so the reviewer can weigh in rather than discover them in the plan.
- Confirmation that the per-action output belongs in `feature_glossary` rather than `metric_contracts`
  (the exemption reasoning in § 8), against the current `tests/test_metric_contracts.py` enumeration.
