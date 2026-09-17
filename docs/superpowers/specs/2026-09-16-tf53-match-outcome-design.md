# TF-53 — match-outcome simulation (win probability / xPoints) design

**Status:** design (brainstormed 2026-09-16). **Sizing:** Wicked+ (Rung 1 core + Rung 2 validation +
Rung 3a fitted dependence + Rung 3b possession collapse). **Numbers:** assigned at commit-prep — do
NOT hardcode a version / PR-S / ADR here; derive the next-free set after `git fetch && git merge
origin/main` (provisional numbers get consumed by concurrent releases — 4.116.0/PR-S187/ADR-095 was
taken by TF-61 while this was in design). ADR is the next free ADR at that time.

**Scope ratification (owner, 2026-09-16 — closes spec-review SPEC-01).** The origin
future-work-plan §T2 sized TF-53 as a "small, independence-only pure module, no aggregator" — which is
Rung 1 exactly. Rungs 2 (calibration CV harness) + 3a (fitted-ρ Dixon-Coles artifact + scipy MLE +
training script) + 3b (possession collapse) are a deliberate **~4× expansion**, EXPLICITLY OWNER-RATIFIED
in one cycle under the "gold standard, best practice" directive + the "both 3a + 3b" Rung-3 choice.
The independent-Poisson-binomial canonical metric is unchanged from §T2; the corrections ship as
validated opt-ins. Delivery stays 2 commits (§10).

## 1. Goal

A new **event-only** package `silly_kicks/match_outcome/` that turns injected per-shot xG into a
match's **win/draw/loss probabilities + xPoints**, at the "gold standard, proven" bar. The core
arithmetic is exact and trivial; the value is in **modeling honesty** (correct chance-quality
handling, explicit/swappable dependence assumptions, honest own-goal handling) and **validation
rigor** (a held-out calibration study). Source: Twelve match report / Soccermatics module 3
(`docs/superpowers/specs/2026-07-16-soccermatics-pro-future-work-plan.md` §T2); Dixon & Coles 1997.

**The reframe (why no optimizer on the core):** the per-team goal distribution is an exact
Poisson-binomial over ~15 shots — an O(n²) DP convolution, nothing to tune. Optuna/evolutionary search
add nothing here (wrong shape). The only fitted parameter in the whole design is the 1-D Dixon-Coles
ρ, whose likelihood is smooth and one-dimensional ⇒ **scipy MLE**, not Optuna.

## 2. Module identity, grain, public surface

- **`silly_kicks/match_outcome/`** — a `compute_*` sibling of `territory` / `duels` / `shot_stopping`
  / `gk_decision`. Imports `spadl` + `id_compat` + `silly_kicks._frame_index.group_rows` (ADR-068
  build-once) only (+ `spadl.add_possessions` for Rung 3b; `scipy` imported ONLY inside the Rung-3a fit
  path, never at serve); **never** `tracking`. AST import-allowlist
  both directions (`tests/match_outcome/test_import_allowlist.py`); nothing imports `match_outcome`.
- **Grain: per-`(game_id, team_id)`** — two rows per match. Columns: `p_win`, `p_draw`, `p_loss`,
  `xpoints` (= 3·p_win + p_draw), `expected_goals` (ΣxG). Canonical-id grouped (ADR-019), raw id
  emitted; a non-two-team game is excluded-and-counted (ADR-042).
- **Public API:**
  - `compute_match_outcome(actions, *, xg_column, params=MatchOutcomeParams.default()) -> (samples, MatchOutcomeReport)`
  - `goal_count_pmf(shot_xgs) -> np.ndarray` — a team's exact goal-count PMF (the distribution is NOT
    discarded; consumers who want the full goal-difference PMF / scoreline simplex get it).
  - `match_outcome_probabilities(home_pmf, away_pmf, *, params) -> (p_home, p_draw, p_away)`.
  - `MatchOutcomeParams` (frozen; the sibling `default()`/`is_default()` idiom of ADR-086/094 — NOT
    ADR-009; `for_provider()` ships EMPTY until an ADR-009 apply-gate clears), `MatchOutcomeReport`
    (frozen), `MatchOutcomeIntegrityError` (fitted-artifact load).
- **`compute_*`, NOT an `add_*`** (C4 action-coupled aggregator count stays 33); in NO default xfn list
  (it reads no gamestate features — it is a match-grain metric).

## 3. Rung 1 — the rigorous core

- Filter to shot-class **`type_id`s** via `spadlconfig.actiontype_id` (`shot`, `shot_penalty`,
  `shot_freekick`) — real-SPADL `type_id`, consistent with the sibling `compute_*` packages
  (team_metrics/shot_stopping). NOT `type_name`: `vaep.labels._is_goal` gates on `type_name` ONLY to
  preserve the VAEP label-input contract (add_names'd actions) — that is a documented exception, not
  the axis for a `compute_*` metric. (A repo-wide shared `SHOT_TYPE_IDS` constant in `spadlconfig`,
  DRY-ing the per-package copies, is a reasonable follow-up but OUT OF SCOPE here — it touches multiple
  packages.) Own goals excluded from the xG model (§7). Per team: the per-shot xG list → **exact
  Poisson-binomial PMF** by DP convolution
  (per-shot Bernoullis; NOT `Poisson(ΣxG)`, which discards chance-quality — one 0.8 + eight 0.1 shots
  share ΣxG but give different win distributions).
- Joint (default independence): `P(home i, away j) = home_pmf[i] · away_pmf[j]` → sum the scoreline
  simplex → `p_win/draw/loss`; `xPoints = 3·p_win + p_draw`.
- Distribution surfaced via the two primitives (§2), so nothing is thrown away.

## 4. Rung 3 — two ORTHOGONAL corrections (composable)

> **AMENDED (ADR-097, commit 2):** the DEFAULT was promoted from `independent` to **both corrections ON**
> — `collapse` on correctness (same-possession shots are not independent trials) + `dixon_coles` on the
> measured full-corpus calibration (paired per-match Brier: dixon_coles beats independent on 77.3% of
> matches, Wilcoxon p≈1e-124; both beats independent on 68.7%, p≈1e-71). `"independent"` remains opt-in on
> either axis. This supersedes the "both opt-in / DEFAULT stays independent" statements in this §4 and §5
> below (which described the pre-promotion design); the promotion is the ADR-009-gated separate decision §5
> anticipated, taken on the study's evidence.

They act at different stages, so `MatchOutcomeParams` carries **two independent string-dispatched
axes** (house `method=`/frozen-params idiom, like `xthreat`/`gk_decision`), both defaulting to the
naive baseline so **v1 canonical = exact independent Poisson-binomial**:

- **`same_possession` (Rung 3b, within-team, pre-PMF):** `"independent"` (default) vs `"collapse"`.
  `collapse` groups a team's shots by `spadl.add_possessions` possession and combines same-possession
  shots into ONE opportunity — `P(≥1 goal in possession) = 1 − ∏_k (1 − xg_k)` — one Bernoulli per
  possession, so a save→rebound→goal is one chance, not several. Pure; no corpus.
- **`team_dependence` (Rung 3a, cross-team joint):** `"independent"` (default) vs `"dixon_coles"`.
  `dixon_coles` applies the Dixon-Coles low-score τ-reweighting to the joint's four low cells (0-0,
  0-1, 1-0, 1-1), using each team's PB marginal mean (ΣxG) as λ/μ, with ρ from a **bundled fitted
  artifact** (§5). τ was derived for Poisson marginals; the PB extension is the standard practitioner
  form and ρ is fit empirically to correct exactly that low-score joint (stated in docstring/NOTICE).

Composable (`collapse` + `dixon_coles`) because the stages don't interact — cleaner than one
4-value enum. **The canonical metric never moves under consumers**: corrections are offered + proven,
not silent defaults.

## 5. Rung 3a — the fitted ρ artifact

- **Fit:** 1-D ρ by **scipy MLE** — maximise `Σ log P(realized scoreline | τ(ρ)-corrected PB joint)`
  over the public corpus (`minimize_scalar`, bounded). No Optuna.
- **Corpus:** StatsBomb open-data (`statsbomb_xg` + realized scorelines; the TF-52 public corpus).
- **Artifact:** pickle-free JSON (`rho`, `training_commit`, corpus provenance, `SHA256`), **fail-closed
  load** (SHA + plausible-range sanity + `training_commit`), ADR-011/040/050 family. Scalar ⇒ no
  chirality/feature-contract machinery. `MatchOutcomeParams.bundled()`-style accessor serves the
  bundled ρ; `dixon_coles` without a loadable artifact RAISES (fail-closed), never silently degrades.
- **ADR-009:** the bundled ρ ships as the artifact behind the OPT-IN `dixon_coles` method; the DEFAULT
  stays `independent`. The calibration study (§6) reports whether the correction earns promotion; any
  default change is a separate gated PR.
- **Training script** `scripts/train_match_outcome_dependence.py`: `assert_public_corpus`-style guard
  (reuses `assert_statsbomb_open_data_mode`), clean-tree provenance (ADR-037), stamps `training_commit`.

## 6. Rung 2 — calibration validation harness

- `scripts/validate_match_outcome_calibration.py` — mirrors `validate_team_kpi_reliability`: ADR-052
  `for_each` shards, ADR-037 clean-tree provenance + `run_commit`, ADR-056 input contract,
  **reported-not-gated** (ADR-009), fail-closed public-only (`assert_statsbomb_open_data_mode` +
  `_PAPPALARDO_PUBLIC_COMPETITIONS`).
- **Metrics per method config** (`independent` / `collapse` / `dixon_coles` / `both`): 3-way-outcome
  **Brier score**, **calibration reliability/slope** (binned predicted-vs-observed), and
  **xPoints-vs-realized-points** calibration. The `dixon_coles` arm is **cross-validated** — ρ fit on
  train folds (grouped by `game_id`), evaluated held-out — so the study evaluates the METHOD, never the
  bundled artifact on the data it was fit on. This is what makes the correction's promotion honest, and
  what keeps the cycle a clean 2 commits (the study never reads the bundled weights).
- **Corpus:** StatsBomb open-data (primary) + optionally public Wyscout/Pappalardo (also carries
  realized scores) for cross-provider.
- **Output:** `docs/research/tf53_match_outcome_calibration/` (findings.md + metrics.json +
  provenance); corpus bound recorded IN the artifact.
- **Honest limits:** retrospective (post-hoc xG, not predictive); independence is an assumption;
  never surface G−xG year-to-year noise as finishing skill.

## 7. Own goals, Report, honest-NaN

- **Own goals — option A:** canonical xPoints stays PURE-xG (own goals aren't shot-xG events →
  excluded from the model); realized own-goal + goal counts surfaced in `MatchOutcomeReport` so a
  consumer can reconcile p_win/xPoints to the actual scoreline. (Own goals as `bad_touch`+`owngoal`
  per ADR-018, identified by result not a shot-gate.)
- **`MatchOutcomeReport`** (frozen): match census (`n_matches_in`, `n_scored`,
  `n_excluded_not_two_teams`; conserved) + shot-xG census (`n_shots`/`n_with_xg`/`n_null_xg`) +
  realized-outcome census (`n_own_goals`, realized goals).
- **Honest-NaN:** a team with no shots → degenerate-but-valid PMF `P(0)=1`; non-two-team game
  excluded+counted; NaN xG on a shot excluded from that team's PB with a counted census.

## 8. Repo-wide wiring (a new sibling `compute_*` package trips ~8 gates)

`feature_glossary` (+ emitted columns; documented + NOTICE-linked); C4 `match_outcome` container
(dot-rendered, count line updated); `_PUBLIC_MODULE_FILES` (public-API-examples gate); import-allowlist
gate; purity gate (if any `add_*` — none here); id-scalar registry (if any public id-scalar fn); ADR-052
`for_each` + ADR-037 provenance-wiring + ADR-056 input-contract enrollment for the two scripts; scale
guard IF any `group_rows` loop (the collapse groups per possession — build `group_rows` once).

## 9. Testing (gold-standard)

- **Poisson-binomial DP validated against brute-force 2ⁿ enumeration** on small n (exactness proof).
- **Face-validity anchor:** the course's Arsenal 33% / draw 25% / West Ham 41%, xPoints 1.25 (1-0 win
  on lower xG). Probs-sum-to-1; symmetric shot lists → symmetric probs; single-shot edges.
- **`collapse`** on a known multi-shot-possession fixture (save→rebound); **`dixon_coles` τ** against
  the DC reference formula on Poisson marginals; **scipy-MLE recovers a known ρ** on synthetic data.
- **Cross-row invariant** (per `(game_id, team_id)` pair): team A `p_win == ` team B `p_loss`, and the
  two rows share `p_draw` (and A `p_loss ==` B `p_win`); `xpoints_A + xpoints_B` consistent with the
  simplex. Named test `test_two_team_rows_are_mirror_consistent`.
- Import-allowlist (both directions), purity, id-dtype invariance, conservation census, order-insensitivity.
- Fitted-artifact: fail-closed load (bad SHA / out-of-range ρ raises), bundled `dixon_coles` end-to-end.

## 10. Delivery — 2 commits (clean provenance)

Branch `feat/tf53-match-outcome`. **Each commit requires the owner's explicit approval for that
specific commit** (show the diff/file-list, wait for an explicit yes — no commit on "ready"/"tests
green"/a plan step); no micro-commits (each commit is a fully-tested coherent state).
1. **Commit 1 — library code:** `match_outcome/` (Rung-1 core, Rung-3b collapse, Rung-3a serving with
   fail-closed ρ load, default `independent`) + the training script + the calibration driver + all
   tests + glossary/NOTICE/C4 wiring. CI-green.
2. **Commit 2 — bundled ρ weights + calibration artifact:** both produced on the clean commit-1 tree
   with outputs staged OUTSIDE the repo (ADR-037), then copied in; both stamp commit 1. The study fits
   its own per-fold ρ, so it doesn't depend on the bundled weights (keeps it a clean 2-commit).

Additive — no VAEP/tracking retrain, no re-materialize. **HONEST LIMIT (docstrings/glossary/NOTICE):**
this is a *retrospective* match-outcome model (post-hoc xG), not a predictive one; the independence
default is an explicit assumption; xPoints is a distributional summary, not a skill measure.

## 11. Reserved / out of scope

- Genuinely-predictive (pre-match) win probability (needs a team-strength prior / betting odds) — that
  is TF-63's in-game extension territory; TF-53 is the retrospective per-shot-xG base it builds on.
- A game-state (score-differential) dependence term beyond the static Dixon-Coles ρ — a future
  `team_dependence=` value, reserved-typed if we want the door.
