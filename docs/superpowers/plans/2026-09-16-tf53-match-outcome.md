# TF-53 match-outcome (win probability / xPoints) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an event-only `silly_kicks.match_outcome` package that turns injected per-shot xG into per-`(game_id, team_id)` win/draw/loss probabilities + xPoints, with an exact Poisson-binomial core, two opt-in honesty corrections (possession-collapse + fitted Dixon-Coles dependence), and a cross-validated public-corpus calibration study.

**Architecture:** A `compute_*` sibling of `territory`/`duels`/`shot_stopping`/`gk_decision` — pure, event-only, imports `spadl`+`id_compat` only (+`spadl.add_possessions` for collapse; `scipy` only in the fit script). Two orthogonal string-dispatched correction axes on a frozen `MatchOutcomeParams`. A bundled fitted-ρ artifact (pickle-free JSON, fail-closed load). A scripts-side reported-not-gated calibration driver.

**Tech Stack:** Python, numpy (DP convolution), pandas, scipy (MLE, fit-only), pytest. No sklearn, no Optuna.

**Spec:** `docs/superpowers/specs/2026-09-16-tf53-match-outcome-design.md` (read it alongside this plan).

## Global Constraints

- **Numbers:** NOT claimed until commit-prep — do NOT hardcode a version / PR-S / ADR. At commit-prep, `git fetch && git merge origin/main`, then take the next-free set (provisional numbers get consumed by concurrent releases — TF-61 took the set this was first penciled with).
- **Real SPADL `type_id`/`result_id` (ints)** via `from silly_kicks.spadl import config as spadlconfig` (`spadlconfig.actiontype_id[...]`/`result_id[...]`); NEVER `type_name`/`result_name`.
- **ADR-019:** canonical-id grouping via `id_compat` (`canonical_id_series`/`canonical_id`), raw id emitted; no raw `==` on ids.
- **ADR-042:** conservation census; honest-NaN, never a fabricated 0.
- **ADR-009:** `for_provider` ships EMPTY; the calibration study is reported-not-gated. **AMENDED (ADR-097, commit 2):** the DEFAULT was promoted from `independent` to **both corrections ON** on the measured full-corpus calibration (paired per-match Brier: dixon_coles beats independent on 77.3% of matches, p≈1e-124; both beats independent on 68.7%, p≈1e-71) + the correctness of `collapse` — the separate ADR-009-gated decision this plan anticipated. `"independent"` remains opt-in. The default compute path now requires the bundled ρ (fail-closed).
- **ADR-037/052/056:** the two scripts adopt `require_clean_tree` + `run_commit`, `for_each` shards, `declare_inputs`.
- **Commit discipline:** each commit needs the owner's explicit approval for that specific commit; no micro-commits; 2 commits total (§Delivery).
- **Event-only:** never import `tracking`; AST import-allowlist both directions; nothing imports `match_outcome`.
- **A new sibling `compute_*` package trips ~8 repo-wide gates** (feature_glossary coverage, C4 count + description cap, NOTICE linkage, `_PUBLIC_MODULE_FILES`, provenance/input-contract enrollment, import allowlist, scale-guard if any `group_rows` loop) — Task 7 wires them; several fail ONLY in the full suite.

---

## File Structure

- `silly_kicks/match_outcome/__init__.py` — public exports.
- `silly_kicks/match_outcome/_config.py` — `MatchOutcomeParams` (frozen) + `_PROVIDER_PARAMS` (empty).
- `silly_kicks/match_outcome/_report.py` — `MatchOutcomeReport` (frozen).
- `silly_kicks/match_outcome/_columns.py` — output column schema (single source).
- `silly_kicks/match_outcome/_pmf.py` — Poisson-binomial DP + joint + outcome probs (the primitives).
- `silly_kicks/match_outcome/_collapse.py` — Rung-3b same-possession collapse.
- `silly_kicks/match_outcome/_dependence.py` — Rung-3a Dixon-Coles τ + `DependenceModel` (fail-closed load).
- `silly_kicks/match_outcome/_compute.py` — `compute_match_outcome` orchestrator.
- `silly_kicks/match_outcome/weights/` — bundled ρ (commit 2).
- `scripts/train_match_outcome_dependence.py` — scipy-MLE ρ fit (commit-2 producer).
- `scripts/validate_match_outcome_calibration.py` — Rung-2 CV calibration driver (commit-2 producer).
- `tests/match_outcome/` — package tests (a package: `__init__.py`, `_helpers.py`).
- `tests/scripts/test_match_outcome_*.py` — kernel tests for the two scripts.

---

## Task 1: Package scaffold (Params / Report / columns / import-allowlist)

**Files:** Create `_config.py`, `_report.py`, `_columns.py`, `__init__.py`, `tests/match_outcome/__init__.py`, `tests/match_outcome/_helpers.py`, `tests/match_outcome/test_config.py`, `tests/match_outcome/test_import_allowlist.py`.

**Interfaces — Produces:**
- `MatchOutcomeParams(same_possession: Literal["independent","collapse"]="independent", team_dependence: Literal["independent","dixon_coles"]="independent", possession_max_gap_seconds=7.0, _is_universal_default=False)` with `default(*, force_universal=False)` / `for_provider(provider)` / `is_default()`. (`high_opportunity_xg` was in the draft interface but DROPPED, owner-approved 2026-09-16: no consumer in the outcome metric — a threshold nothing buckets on; a dead public knob is speculative-API debt. Big-chance thresholds live in `team_metrics`.)
- `MatchOutcomeReport(params, n_matches_in, n_matches_scored, n_matches_excluded_not_two_teams, n_shots, n_shots_with_xg, n_shots_null_xg, n_own_goals)` (frozen).
- `MATCH_OUTCOME_COLUMNS: dict[str,str]` (keys `game_id`/`team_id` object; `p_win`/`p_draw`/`p_loss`/`xpoints`/`expected_goals` float64).

- [ ] **Step 1:** Write `test_config.py`: `MatchOutcomeParams.default().is_default() is True`; `TeamKpi`-style `for_provider("statsbomb") == MatchOutcomeParams()`; `__post_init__` raises on an invalid `same_possession`/`team_dependence` string; frozen (`dataclasses.FrozenInstanceError` on assignment).
- [ ] **Step 2:** Run → FAIL. Implement `_config.py` (mirror `shot_stopping/_config.py`: frozen dataclass, `_is_universal_default` compare-excluded, empty `_PROVIDER_PARAMS`, `__post_init__` validating the two enums).
- [ ] **Step 3:** Implement `_report.py` (mirror `ShotStoppingReport`) + `_columns.py` (single-source schema). Write `test_config.py` cases for `MatchOutcomeReport` conservation identities (as pure asserts on a hand-built instance).
- [ ] **Step 4:** Implement `__init__.py` exporting `compute_match_outcome`, `goal_count_pmf`, `match_outcome_probabilities`, `MatchOutcomeParams`, `MatchOutcomeReport`, `MatchOutcomeIntegrityError`. (compute/pmf are stubs raising `NotImplementedError` until Task 2 — keep imports resolvable.)
- [ ] **Step 5:** Write `test_import_allowlist.py` (mirror `tests/gk_decision/test_import_allowlist.py`): AST-assert `match_outcome/*` imports never include `silly_kicks.tracking`; and nothing under `silly_kicks/` (except match_outcome itself) imports `match_outcome`. Add `_helpers.py` (`make_shots(records)` builder using real `type_id`/`result_id`, an `xg` column).
- [ ] **Step 6:** Run all Task-1 tests → PASS. Commit staged (NOT committed — owner-gated).

## Task 2: Rung 1 core — Poisson-binomial PMF + joint + xPoints

**Files:** Create `_pmf.py`; Modify `_compute.py` (create), `__init__.py`; Test `tests/match_outcome/test_pmf.py`, `tests/match_outcome/test_compute.py`.

**Interfaces — Produces:**
- `goal_count_pmf(shot_xgs: Sequence[float]) -> np.ndarray` — exact Poisson-binomial PMF (index k = P(exactly k goals)); empty → `array([1.0])`.
- `match_outcome_probabilities(home_pmf, away_pmf, *, params=MatchOutcomeParams.default()) -> tuple[float,float,float]` — `(p_home_win, p_draw, p_away_win)` (dependence applied here per `params.team_dependence`; Task 4 wires dixon_coles — here independence only).
- `compute_match_outcome(actions, *, xg_column, params=MatchOutcomeParams.default()) -> (samples, MatchOutcomeReport)`.

- [ ] **Step 1: failing test — PMF exactness.** `test_pmf.py::test_poisson_binomial_matches_brute_force`: for random xg lists of n≤12, `goal_count_pmf(xgs)` equals a brute-force 2ⁿ enumeration (sum over all subsets) to `atol=1e-12`. Plus: empty→`[1.0]`; single `[p]`→`[1-p,p]`; sums to 1.

```python
def _brute_force_pmf(xgs):
    from itertools import product
    n = len(xgs); pmf = np.zeros(n + 1)
    for combo in product([0, 1], repeat=n):
        p = 1.0
        for bit, xg in zip(combo, xgs, strict=True):
            p *= xg if bit else (1 - xg)
        pmf[sum(combo)] += p
    return pmf
```

- [ ] **Step 2:** Run → FAIL. Implement `goal_count_pmf` (DP convolution: start `pmf=[1.0]`, for each xg convolve with `[1-xg, xg]`; O(n²)).
- [ ] **Step 3: failing test — outcome probs + face validity.** `test_pmf.py`: `match_outcome_probabilities` sums to 1; symmetric pmfs → `p_home==p_away`, and `p_draw==Σ home[k]·away[k]`; single-shot edges. **Face-validity (PLAN-08 — do NOT tune-to-target):** the course publishes only ΣxG (1.57 Arsenal / 1.72 West Ham), and the spec §3 reframe proves ΣxG cannot pin the Poisson-binomial distribution — so a "reproduce 33/25/41" test would be vacuous. Instead assert the ROBUST facts: with West Ham's ΣxG > Arsenal's, `p_westham_win > p_arsenal_win`, both outcome probs and draw within a documented rough-magnitude band (e.g. each in [0.2, 0.5]), and xPoints ordered accordingly — `test_compute.py::test_face_validity_ordering_and_magnitude`. Optionally, an `@e2e` `test_face_validity_arsenal_real_shotmap` sources that match's PER-SHOT `statsbomb_xg` from SB open data and asserts the exact 33/25/41 ± tol (the only honest way to hit the exact figures).
- [ ] **Step 4:** Run → FAIL. Implement `match_outcome_probabilities` (independence: iterate i,j over the two pmfs, accumulate win/draw/loss) and `compute_match_outcome` (filter shot `type_id`s via `spadlconfig`; per-team `goal_count_pmf` over `xg_column`; per-`(game,team)` row with `p_win/p_draw/p_loss`, `xpoints=3*p_win+p_draw`, `expected_goals=ΣxG`; canonical-id group + raw-id emit; non-two-team game excluded+counted; shot-xG + own-goal census into `MatchOutcomeReport`). **Owngoal predicate (PLAN-09, ADR-018):** `n_own_goals` counts `result_id == spadlconfig.result_id["owngoal"]` — own goals are `bad_touch`, so NO shot-`type_id` gate (mirror `vaep.labels._is_owngoal`'s result-only rule); do NOT improvise. Own goals are excluded from the shot-xG model (they carry no xG).
- [ ] **Step 5: failing test — mirror consistency + census + honest-NaN.** `test_compute.py::test_two_team_rows_are_mirror_consistent` (A.p_win==B.p_loss, shared p_draw, simplex-consistent xpoints); `test_report_conserves` (n_scored + n_excluded == n_matches_in); no-shots team → PMF `[1.0]` → p_loss vs the other team's scoring; NaN-xg shot excluded + counted; no `xg_column`/all-NaN → honest handling.
- [ ] **Step 6:** Run → PASS. Order-insensitivity test (permute input rows → identical samples modulo sort). Purity test (input `actions` unmutated). id-dtype invariance (numeric vs string ids). Commit staged.

## Task 3: Rung 3b — same-possession collapse

**Files:** Create `_collapse.py`; Modify `_compute.py`; Test `tests/match_outcome/test_collapse.py`; register in `tests/_scale_guarded.py` + a growth test if a `group_rows` loop is introduced.

**Interfaces — Produces:** `collapse_possession_xgs(team_shots: pd.DataFrame, *, xg_column) -> np.ndarray` — one combined Bernoulli per possession: `P(≥1 goal) = 1 − ∏(1 − xg_k)`; shots grouped by `add_possessions` `possession_id`.

- [ ] **Step 1: failing test.** `test_collapse.py::test_same_possession_shots_collapse`: a fixture with a 2-shot possession (xg 0.3, 0.4) + a 1-shot possession (0.2) → collapsed xgs `[1-(0.7*0.6), 0.2] = [0.58, 0.2]`; a fixture with all-distinct possessions → collapse is a no-op (identical PMF to independent). Assert `compute_match_outcome(..., params=MatchOutcomeParams(same_possession="collapse"))` uses the collapsed xgs and DIFFERS from `independent` on the multi-shot-possession fixture, and is IDENTICAL on the all-distinct one.
- [ ] **Step 2:** Run → FAIL. Implement `_collapse.py`: call `spadl.add_possessions` once (on the scored actions), group each team's shots by `possession_id` via `group_rows` (ADR-068 — build once), combine per possession. Thread `same_possession` through `_compute.py` (per-team, before `goal_count_pmf`).
- [ ] **Step 3:** Run → PASS. If a `group_rows` loop was added, register `silly_kicks.match_outcome._collapse.<fn>` in `tests/_scale_guarded.SCALE_GUARDED` + add `test_collapse_is_subquadratic` (scale the GAME dimension; RED-GREEN prove a per-game rescan goes quadratic). Commit staged.

## Task 4: Rung 3a serving — Dixon-Coles dependence + fail-closed ρ load

**Files:** Create `_dependence.py`; Modify `_pmf.py` (`match_outcome_probabilities` dependence branch), `_compute.py`, `__init__.py`; Test `tests/match_outcome/test_dependence.py`.

**Interfaces — Produces:**
- `dixon_coles_tau(i, j, lam, mu, rho) -> float` — the DC low-score τ (1 outside the 2×2 low block; the four standard cells inside).
- `DependenceModel` — holds `rho`; `bundled()` classmethod (loads `weights/`); `apply(home_pmf, away_pmf) -> joint_2d`; fail-closed `load()` (SHA256 + `training_commit` + `rho` plausible-range); `MatchOutcomeIntegrityError`.

- [ ] **Step 1: failing test — τ reference.** `test_dependence.py::test_tau_matches_dixon_coles_reference`: `dixon_coles_tau` on the four low cells equals the published DC formula (τ(0,0)=1−λμρ, τ(0,1)=1+λρ, τ(1,0)=1+μρ, τ(1,1)=1−ρ; τ=1 elsewhere) for sample λ,μ,ρ; ρ=0 → all τ=1 → joint == independent product (to 1e-12).
- [ ] **Step 2:** Run → FAIL. Implement `dixon_coles_tau` + the joint construction (independent outer product, then multiply the four low cells by τ using each team's PMF mean as λ/μ; renormalize).
- [ ] **Step 3: failing test — fail-closed load.** `test_dependence.py`: a good bundled artifact loads; a tampered SHA raises `MatchOutcomeIntegrityError`; an out-of-range ρ raises; `team_dependence="dixon_coles"` with no loadable artifact raises (never silently independent).
- [ ] **Step 4:** Run → FAIL. Implement `DependenceModel` (JSON `{rho, training_commit, corpus, sha256}`, pure-numpy serve, no sklearn/pickle) + wire the `team_dependence="dixon_coles"` branch in `match_outcome_probabilities` (sum the τ-corrected joint) + `compute_match_outcome`. (No bundled weights yet — Task 5 produces them; tests use a synthetic in-memory model.)
- [ ] **Step 5:** Run → PASS. Composition test: `same_possession="collapse"` + `team_dependence="dixon_coles"` both apply, independently.
- [ ] **Step 6 (PLAN-06 — bundled-serve, skip-if-absent):** `test_dependence.py::test_bundled_dependence_serves_when_present` — `pytest.mark.skipif(not weights_dir.exists())`; when present, `DependenceModel.bundled()` loads AND `compute_match_outcome(fixture, xg_column="xg", params=MatchOutcomeParams(team_dependence="dixon_coles"))` runs end-to-end producing finite in-[0,1] simplex probs. SKIPS in commit 1 (no weights yet); RUNS in commit 2 / Task 8 Step 5 (weights present) — so a mis-serialized bundled artifact (bad `SHA256SUMS`/key/out-of-range ρ) is caught, not shipped. Commit staged.

## Task 5: Training script — scipy-MLE ρ fit (commit-2 producer)

**Files:** Create `scripts/train_match_outcome_dependence.py`; Test `tests/scripts/test_match_outcome_train.py`.

**Interfaces — Produces:** `fit_rho(matches) -> float` (pure: matches = list of `(home_xgs, away_xgs, home_goals, away_goals)`); `main()` (public corpus loader + `require_clean_tree` + stamp `training_commit` + write `weights/`).

- [ ] **Step 1: failing test — MLE recovers ρ.** `test_match_outcome_train.py::test_fit_recovers_known_rho`: simulate matches from a known ρ (sample scorelines from the τ-corrected joint), `fit_rho` recovers it within tolerance; ρ=0 data → fit ≈ 0.
- [ ] **Step 2:** Run → FAIL. Implement `fit_rho` (negative-log-likelihood of realized scorelines under the τ-corrected PB joint; `scipy.optimize.minimize_scalar`, bounded). `scipy` imported function-locally (fit-only).
- [ ] **Step 3:** Implement `main()`: reuse `scripts/_sb_open_data.load_open_data_matches` (full manifest, `assert_statsbomb_open_data_mode`), realized goals from the scorelines; `require_clean_tree` (ADR-037); write `weights/{model.json, SHA256SUMS}` with `training_commit`. `@e2e`/`@slow` does-it-run smoke (owner-run, needs statsbombpy). **PLAN-10:** enroll `train_match_outcome_dependence` in `ARTIFACT_DRIVERS` (it is a weight-trainer that stamps `training_commit` — the same class as `train_pass_completion`/`train_ghost_gk`/`train_xshot_occurrence`, all of which are enrolled), + `declare_inputs` (ADR-056).
- [ ] **Step 4:** Run → PASS (unit) + provenance-wiring gate green. Commit staged.

## Task 6: Calibration validation driver (Rung 2, CV; commit-2 producer)

**Files:** Create `scripts/validate_match_outcome_calibration.py`; Test `tests/scripts/test_match_outcome_calibration.py`.

**Interfaces — Produces:** pure kernels `three_way_brier(probs, outcomes)`, `calibration_slope(probs, outcomes)`, `reduce_calibration(shards)`, `cv_rho_by_fold(matches, folds)`; `main()` (for_each shards + provenance + input-contract).

- [ ] **Step 1: failing tests — kernels.** `three_way_brier` on a known toy (perfect prediction → 0; uniform → known value); `calibration_slope` recovers 1.0 on synthetic-calibrated data; `cv_rho_by_fold` fits per fold on train games, evaluates held-out (never on the fit data).
- [ ] **Step 2:** Run → FAIL. Implement the kernels.
- [ ] **Step 3:** Implement `main()`: for_each over public matches; per method config (`independent`/`collapse`/`dixon_coles`/`both`) emit per-match predicted simplex + realized outcome; reduce → per-config 3-way Brier + calibration slope + xPoints-vs-points; the `dixon_coles`/`both` arms use `cv_rho_by_fold` (grouped by `game_id`), never the bundled weights. ADR-052 `for_each` + ADR-037 provenance + ADR-056 `declare_inputs`; fail-closed public-only. Output `docs/research/tf53_match_outcome_calibration/`.
- [ ] **Step 4:** Run kernel tests → PASS; provenance/input-contract gates green. Register `validate_match_outcome_calibration` in `ARTIFACT_DRIVERS` + `declare_inputs`. (PLAN-10: BOTH scripts are enrolled — the train script in Task 5 Step 3, this driver here; the ADR-052/056 completeness gate asserts the population EXACTLY, so a missing enrollment fails CI.) Commit staged.

## Task 7: Repo-wide gate wiring

**Files:** Modify `silly_kicks/feature_glossary.py`, `NOTICE`, `docs/c4/architecture.dsl` (+ regenerate `architecture.html` via Graphviz `dot`), `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`), `tests/scripts/test_provenance_wiring.py` + `test_input_contracts.py` (enroll both scripts), the glossary emitted-columns leg.

- [ ] **Step 1:** Add `FeatureColumn` entries for `p_win`/`p_draw`/`p_loss`/`xpoints`/`expected_goals` (emitting_module `silly_kicks.match_outcome._compute`); add the `_A_MATCH_OUTCOME` attribution constant; wire the `match_outcome` leg in `tests/invariants/glossary_emitted_columns.py::emitted_columns`.
- [ ] **Step 2:** Add the `NOTICE` entry (Poisson-binomial; Dixon & Coles 1997; Twelve/Soccermatics module 3) with a verbatim attribution token matching `_A_MATCH_OUTCOME`.
- [ ] **Step 3:** Add the `match_outcome` C4 container to `architecture.dsl` (desc ≤200 chars), update the "N derived feature columns" count, regenerate `architecture.html` via the pinned Graphviz `dot` pipeline (structurizr.war → c4_assemble --inject-wrap-width → plantuml.jar -graphvizdot → c4_assemble --svg-dir).
- [ ] **Step 4:** Register the 3 public modules (`_config.py`/`_compute.py`/`_report.py` + any defining `_pmf.py`/`_dependence.py`) in `_PUBLIC_MODULE_FILES`; ensure every public symbol has a real Examples section (documented on creation, like the siblings).
- [ ] **Step 5:** Run the full glossary/C4/public-API/provenance/input-contract gate set → PASS.

## Task 8: Spec-coverage self-check + full CI-faithful suite (STOP at commit gate)

- [ ] **Step 1:** Re-read the spec; point every §2–§9 requirement at a task/test; fix gaps.
- [ ] **Step 2:** `python -m ruff check silly_kicks/ tests/ scripts/` + `ruff format --check` + `pyright` (bare) → clean.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e" -p no:randomly -q` → green (mirror ci.yml; a new sibling trips full-suite-only gates).
- [ ] **Step 4 (PLAN-07 — separate approval gates):** **STOP.** Present the commit-1 (code) diff + message and wait for an explicit **commit** yes. Then STOP again and wait for an explicit **push** yes. Then STOP again and wait for an explicit **PR** yes. Each of commit / push / PR is its own gate — never bundle them under one approval. Then watch CI.
- [ ] **Step 5 (commit 2, after commit 1 lands clean):** on the clean tree, run `train_match_outcome_dependence.py` (→ bundled ρ) + `validate_match_outcome_calibration.py` (→ calibration artifact) with `--out` staged OUTSIDE the repo; copy `weights/` + `docs/research/tf53_match_outcome_calibration/` in. **PLAN-06(b): with the weights now present, RE-RUN the full CI-faithful suite** (`-m "not e2e" -p no:randomly`) so the skip-if-absent bundled-serve test (Task 4 Step 6) actually EXECUTES and certifies the bundled artifact serializes/loads before the gate. Only then present commit 2 (separate commit/push approvals, same as Step 4).

---

## Delivery — 2 commits (per spec §10, owner-approval-gated)

1. **Commit 1 — library code** (Tasks 1–8 code + the two scripts + all tests + glossary/NOTICE/C4/`_PUBLIC_MODULE_FILES` wiring). Default `independent`; `dixon_coles` fail-closed with no weights yet.
2. **Commit 2 — bundled ρ + calibration artifact** (produced on the clean commit-1 tree, staged external, copied in; both stamp commit 1). The study fits per-fold ρ (never reads the bundled weights) — keeps it a clean 2-commit.

## Self-Review

- **Spec coverage:** §2 (Task 1/2), §3 Rung-1 (Task 2), §4/§5 Rung-3a (Task 4/5), §4 Rung-3b (Task 3), §6 Rung-2 (Task 6), §7 own-goals/Report (Task 1/2), §8 wiring (Task 7), §9 testing (each task's tests + Task 8), §10 delivery (Delivery). Covered.
- **Placeholders:** none — signatures + test sketches are concrete.
- **Type consistency:** `MatchOutcomeParams`/`MatchOutcomeReport`/`goal_count_pmf`/`match_outcome_probabilities`/`compute_match_outcome`/`DependenceModel` names consistent across tasks.
