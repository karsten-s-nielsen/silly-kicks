# TF-61 — xSuccess (event-only action-completion) + VAEP_adjusted (outcome-bias-free VAEP) — Design

- **Status:** Accepted — **rev 5** (2026-09-15): RELEASED as **4.116.0 / PR-S187 / ADR-095** — the concurrent TF-52 session released first and took the provisional 4.115.0 / PR-S186 / ADR-094; the rev-1..4 "provisional" number notes below are superseded. — **rev 4** (2026-09-15): the **version / PR-S / ADR number is unclaimed until Commit 2** (owner decision) — a concurrent silly-kicks session may release first, so `_version.py` stays at `4.114.0` through the code commit and the next-free minor is claimed at commit-prep (CHANGELOG carries a `[Unreleased]` heading meanwhile); the version bump moves from Commit 1 to Commit 2 in §11. — **rev 3** (2026-09-15): §11 commit structure amended to **CODE | ARTIFACTS** (owner-ratified; supersedes rev-2's feature-split) — clean-tree provenance (`training_commit`, ADR-052/056) forces code-before-weights, per plan finding TF61-PLAN-01 (the TF-57/TF-62 2-phase pattern). — **rev 2** (2026-09-14): folds in the independent spec review `D:\Development\_reviews\2026-09-14-tf61-xsuccess-vaep-adjusted-spec.md` (**APPROVE WITH FOLLOW-UPS**). **TF61-SPEC-01 resolved by making xSuccess END-BLIND** (start-anchored geometry only; no realized-end features) — the realized end is target leakage for failed actions (a postdiction), and the rev-1 "keep the end / PassCompletionModel precedent" call was wrong (PassCompletionModel serves on *supplied* targets, not realized outcomes); the leakage guard becomes end-LOCATION invariance, the ReceiverModel bar (also fixes SPEC-02). SPEC-03/04/05/06/07 folded (ADR-005 citation precision; `xs[i-1]` prev-weighting test; penalty/corner note; evolve-seed fallback; commit-structure note). — rev 1 (2026-09-14): initial draft. **Uncommitted.** Version / PR-S / ADR set at commit-prep after `git fetch && git merge origin/main`: next ADR ≈ **ADR-094**, next release ≈ **4.115.0** (current `main` is 4.114.0), PR ≈ **PR-S186** — provisional.
- **Date:** 2026-09-14
- **Feature:** TF-61 — two complementary event-only library primitives from Paul, Klemp & Memmert, *"Beyond Outcome Bias: Incorporating Action Completion Probability and Risk-Return into Soccer Evaluation Models"* (MLSA 2025, `MLSA25_paper_225`): **(1) xSuccess** — a bundled, calibrated action-completion model `P(success | context)` over *all* on-ball action types; **(2) VAEP_adjusted** — a risk-aware VAEP rating that weights the score/concede deltas by completion probability to remove **outcome bias**, reusing a caller-fitted standard VAEP with no retrain.
- **Delivery:** a single coherent feature branch `feat/tf61-xsuccess-vaep-adjusted`, **no worktrees**, **two commits** (xSuccess + bundled weights; then VAEP_adjusted), **one PR**. Each commit is a fully-tested coherent state (no micro-commits). Every commit, the HF publish, and the merge/tag are **separate explicit human go-aheads**; the `/review-impl` gate runs in an **independent session**, never the author's.
- **Execution:** all computation (OpenEvolve feature search, Optuna HPO, training, calibration, validation, artifact production) is **run by the implementing session** (DGX available), not owner-run. The only human gates are the go-aheads above and the independent review.
- **Attribution boundary:** the method is transcribed from the **open-access paper** (`dtai.cs.kuleuven.be/events/MLSA25/papers/MLSA25_paper_225.pdf`); there is no reference code to lift, so no license boundary as in TF-57/PathCRF. Training data is **StatsBomb Open Data** (redistributable by construction — trained only via the open-data loader `load_open_data_matches`, the `train_pass_completion` convention; see §8 / TF61-IMPL-01). OpenEvolve / AlphaEvolve is the feature-evolution *method*, attributed in NOTICE.
- **Related:** `expected_passing.PassCompletionModel` (the event-only completion precedent this generalizes — same bundled-artifact discipline); `tracking._xshot_occurrence.XShotOccurrenceModel` + `_xshot_occurrence_objective.XShotOccurrenceObjective` (the xgboost-served bundled-model + ruthless `CachedObjective` templates copied here); ADR-011/016/040/050 (trained-model fail-closed lifecycle: pickle-free JSON + SHA256 + chirality + feature-contract); ADR-005 (the tracking-aware-features ADR that also established the `See NOTICE …` academic-attribution convention); ADR-009 (ship raw primitives, not composites; frozen params `for_provider`; applied results reported-not-gated); ADR-052/056 (corpus-driver resilience + artifact provenance / dirty-tree refusal); ADR-088 (single card-required HF publish seam); ADR-093 (the TF-57 OpenEvolve provenance + held-out-validation discipline reused here). **C4:** one new `xsuccess` container; the action-coupled `add_*` aggregator count (33) is **unchanged**.

---

## 1. Executive summary (for the reviewer)

Standard VAEP has a known **outcome bias**: because it values a game state *after* the action's realized result, a lucky completion of a risky action is over-credited and a sound decision that fails is penalized — the VAEP authors themselves note a made shot scores ≈ (outcome − xG). This cycle ships the Paul/Klemp/Memmert (2025) correction as two event-only primitives:

1. **xSuccess** (`silly_kicks/xsuccess/`, new event-only package) — a calibrated XGBoost model estimating `P(action completes | pre-action context)` for **every** on-ball action type (not just passes; that gap is already filled by `PassCompletionModel`). Bundled, pickle-free, fail-closed, xgboost-served (the `XShotOccurrenceModel` precedent), trained by the implementing session on the full redistributable StatsBomb Open Data corpus.

2. **VAEP_adjusted** (`silly_kicks/vaep/adjusted.py` + a `VAEP.rate_adjusted` method) — for each action, re-score the *already-fitted* P_scores / P_concedes classifiers on a **surgical result-feature counterfactual** (the action's completion flipped to success, resp. fail, everything else held at actual), weight by `xSuccess` and `(1 − xSuccess)`, and run the *existing* `formula.value` delta machinery. **No retrain of any existing model.**

**What makes this a rigorous silly-kicks build, not a transcription:**
- xSuccess is **end-blind** — it uses only pre-action/start-anchored geometry, never the realized action end. For a *failed* action the SPADL end is the outcome (the death/interception point exists only *because* it failed), so an end-using completion model is a **postdiction** (target leakage) that would re-inject the very bias VAEP_adjusted removes. This meets the `ReceiverModel` bar (end-*location* invariance); it deviates from the paper's Table-1 feature set deliberately (§5.2). Shots are unaffected (start geometry is the clean xG-style signal); event-only pass length/direction granularity is forgone (unrecoverable clean without tracking).
- The completion model is **calibrated** (isotonic, validated per-type) because VAEP_adjusted *multiplies* these probabilities; per-type calibration is *necessary but not sufficient* (§9.1), with the end-location leakage guard as the crisp leak gate.
- The feature representation is **discovered by OpenEvolve** (the paper's own stated future-work lever — "tailored feature engineering") within a **leakage-safe, end-free input allowlist**, then the XGBoost hyperparameters are tuned by **Optuna** on that representation; both are dev/train-time (ruthless-efficiency), the runtime is deterministic committed code, and the result carries clean provenance + held-out validation (the ADR-093 discipline).
- The counterfactual is the **gold-standard ceteris-paribus form** (flip only the completion variable), which is *more* correct than the paper's literal "flip the whole dataset" (that inflates goalscore and rewrites predecessor history).

**Blast radius:** purely additive. New `xsuccess` package + bundled weights; new opt-in `vaep.rate_adjusted`. No existing feature/model changes, **no default-config VAEP retrain**, no `*_xfns` (so no leakage-gate factory) and no `feature_glossary` growth. One new C4 container.

---

## 2. Goals and non-goals

### Goals
1. **xSuccess:** a bundled, calibrated, event-only, **end-blind** `P(success | context)` model over all on-ball action types (start-anchored features only; no outcome-contaminating realized-end features), served pure of sklearn (xgboost booster serve, `[xgboost]` extra), fail-closed on load (SHA + chirality + feature-contract).
2. **VAEP_adjusted:** an opt-in rating that removes outcome bias by re-scoring a caller-fitted **standard** VAEP under a surgical completion counterfactual, weighted by xSuccess, with **no retrain**.
3. **Full ruthless-efficiency use:** OpenEvolve discovers the xSuccess feature representation; Optuna tunes the XGBoost hyperparameters; both dev/train-time with `assert_cache_equivalence` (Optuna) and committed-deterministic-output + provenance + held-out validation (evolve).
4. **Reproducible bundled weights** trained on the full redistributable StatsBomb Open Data corpus (sourced only via the open-data loader `load_open_data_matches` — redistributable by construction; clean `training_commit`), published via the ADR-088 card-required seam.
5. **Validation battery** reproducing the paper's construct-validity checks (xSuccess calibration-in-the-large + per-type reliability; VAEP_adjusted aggregate-vs-xG alignment + outcome-bias reduction), reported in `docs/research/`.

### Non-goals
- **Tracking.** Both primitives are event-only; `xsuccess/` must never import `tracking`. (A tracking-augmented xSuccess is the paper's *other* future-work note; out of scope.)
- **Shipping an xG model.** silly-kicks ships none; the aggregate-vs-xG validation uses StatsBomb's own xG column (injected, validation-only).
- **A new VAEP subclass or a different fit.** VAEP_adjusted is an *alternative rating of a standard-trained VAEP*, delivered as a method, not a subclass.
- **Composing with HybridVAEP.** Hybrid removes the current action's result feature, making the completion counterfactual a no-op; VAEP_adjusted **requires standard result-bearing features** and raises otherwise.
- **A `*_xfns` gamestate feature.** xSuccess is a per-action model consumed by the rating, not a VAEP feature transformer; it reads `result` only as a *training label*, never as a serve feature.
- **Per-provider tuning.** Params are intent-set / discovered-once; `for_provider` stays empty per ADR-009. Any later per-provider retune is a separate ADR-009 cycle.
- **The paper's literal "flip the whole test dataset" counterfactual.** Superseded by the surgical flip (§7.2); the literal form is computed *once* only as a validation cross-check.

---

## 3. Method provenance (from the paper)

Transcribed from `MLSA25_paper_225` (read in full during design). Notation matches the paper.

### 3.1 xSuccess (paper §3.2)
A binary classifier `P(success | X)` for a given action and contextual features `X`, labelled 1 iff the action's direct outcome is a success (provider definition), else 0 — for **all** on-ball action types. The paper's feature set (Table 1): `seconds`, `start_x`, `start_y`, `action_type`, `bodypart`, `action_distance` (length in m), `distance_to_goal` (from the action's **end** to goal centre), `shot_angle_centered`. Trained with XGBoost. Reported: ROC-AUC **0.95** (0.87 excluding structurally one-sided types), Brier **0.067** (vs 0.094 action-type-average baseline); calibration-in-the-large Σ predicted = 1,501,376 ≈ 1,500,934 observed; per-type calibration passes 79.14 %/79.08 %, shots 9.70 %/9.64 %.

### 3.2 VAEP_adjusted (paper §3.3, eqs 8–12)
Standard VAEP values action `aᵢ` as `ΔP_scores(aᵢ) − ΔP_concedes(aᵢ)`, the change in scoring/conceding probability between states `S_{i−1}` and `S_i`. The adjustment:

- Build the counterfactual probabilities: `P(scores | success)` (all actions treated as successful) and `P(concedes | fail)` (all actions treated as failed).
- Weight by completion probability:
  - (8) `P(scores|success)_adj(aᵢ) = xSuccess(aᵢ) · P(scores|success)(aᵢ)`
  - (9) `P(concedes|fail)_adj(aᵢ) = (1 − xSuccess(aᵢ)) · P(concedes|fail)(aᵢ)`
- Delta and combine:
  - (10) `ΔP(scores|success)_adj(aᵢ) = P(scores|success)_adj(Sᵢ) − P(scores|success)_adj(S_{i−1})`
  - (11) `ΔP(concedes|fail)_adj(aᵢ) = P(concedes|fail)_adj(Sᵢ) − P(concedes|fail)_adj(S_{i−1})`
  - (12) `VAEP_adj(aᵢ) = ΔP(scores|success)_adj(aᵢ) − ΔP(concedes|fail)_adj(aᵢ)`

Reported construct validity: Σ `P(scores|success)_adj / k` (k = 10) = **2,279.6** ≈ total StatsBomb xG **2,273.5**; outcome-bias visibly removed (adjusted values no longer separate by realized binary outcome at fixed context); rare-outcome value spikes suppressed. Case study (Dortmund–Hoffenheim): a low-completion forward pass gets a *negative* adjusted value; a lucky header's inflated standard VAEP 0.822 is pulled to 0.194, near its xG 0.115.

### 3.3 How eqs 8–12 map onto the existing code (load-bearing)
`silly_kicks/vaep/formula.py::value(actions, Pscores, Pconcedes)` already computes `offensive_value + defensive_value = (scores − prev_scores) − (concedes − prev_concedes)` with correct same-/other-team previous-state handling, penalty/corner fixed odds, and prev-goal zeroing. Feeding it `Pscores = p_scores_adj` and `Pconcedes = p_concedes_adj` yields exactly eqs 10–12 **per action**, because each action's adjusted probability is computed on its own gamestate and the delta machinery supplies the `S_{i−1}` term. No new delta logic is written.

---

## 4. Architecture & module layout

### 4.1 New package `silly_kicks/xsuccess/` (event-only)
```
silly_kicks/xsuccess/
  __init__.py        # public: XSuccessModel, XSuccessIntegrityError
  _features.py       # xsuccess_features(actions) -> (X, FEATURE_NAMES); feature_contract_block()
  _model.py          # XSuccessModel (fit / predict_success / save / load / bundled / to_dict / from_dict)
  weights/
    model.json       # xgboost booster (native JSON)
    metadata.json    # feature_names, one-hot vocab, feature_contract, chirality, isotonic params,
                     #   geometry constants, training_commit, corpus manifest
    SHA256SUMS
    MODEL_CARD.md
```
- **Imports:** `silly_kicks.spadl` (+ `spadl.config`), `silly_kicks.id_compat`, numpy, pandas — and `xgboost` **function-locally** (in `fit`; and at serve behind the `[xgboost]` extra). **Never** `silly_kicks.tracking`. Nothing imports `xsuccess` except `vaep`. Pinned by `tests/xsuccess/test_import_allowlist.py` (AST, mirrors `tests/expected_passing/test_import_allowlist.py`).
- **Serve dependency:** xgboost at inference (gated on `[xgboost]`), exactly like `XShotOccurrenceModel`. For the headline VAEP_adjusted use this adds nothing — the underlying VAEP already needs xgboost.

### 4.2 `silly_kicks/vaep/adjusted.py` + `VAEP.rate_adjusted`
- `adjusted.py::adjusted_value(actions, p_scores_success, p_concedes_fail, xsuccess) -> DataFrame` — the pure eq 8–12 combiner (weights + `formula.value`), independently testable.
- `VAEP.rate_adjusted(game, actions, xsuccess_model, *, frames=None, return_components=False) -> DataFrame` — the method that builds the surgical counterfactual feature copies, re-scores the fitted classifiers, and calls `adjusted_value`. Lives on `VAEP` because it needs the fitted `__models`; a standalone function would force exposing private state.
- No new package; this is inside the existing `vaep` container.

### 4.3 Optimization / training (dev-train-time; `[train]` extra + `scripts/`)
```
silly_kicks/xsuccess/_objective.py   # XSuccessObjective (ruthless CachedObjective) — Optuna HPO
scripts/evolve_xsuccess_features.py  # Stage A OpenEvolve driver (provenance-stamped)
scripts/train_xsuccess.py            # Stage B+C: HPO + calibrate + train final + save + provenance
scripts/publish_xsuccess.py          # HF publish via _hub_publish.publish_model_with_card
```
`_objective.py` is in the library (like `_xshot_occurrence_objective.py`) but **not imported by `__init__` or the inference path**; it needs `[train]` (ruthless-efficiency[optuna] + xgboost). The evolve/train/publish drivers are scripts-side.

### 4.4 What does NOT change
No edits to `formula.py`, `base.py`'s fit/rate, `labels.py`, `expected_passing/`, or any `tracking` module. `HybridVAEP` inherits `rate_adjusted` but it raises on use (§7.3). C4 gains one container; the `add_*` count is unchanged.

---

## 5. The xSuccess model

### 5.1 Label & domain
`y = (result_id == success_id)` → 1, else 0, over all real on-ball actions. Rows with `type_id == non_action` are excluded. Structurally one-sided types (e.g. `bad_touch` ≈ always fail, `clearance` ≈ often success-only) are **kept** (paper retained all); the model learns their base rate and per-type calibration is validated. `result_id` values other than `success` (fail / offside / owngoal / cards) are all label 0.

### 5.2 Geometry & orientation — END-BLIND (TF61-SPEC-01)
xSuccess is **end-blind**: it uses only start-anchored geometry (`distance_to_goal` from `(start_x, start_y)` to goal centre `(105, 34)`, `angle_to_goal` from the start) plus `start_x`/`start_y` — **never** `end_x`/`end_y` or anything derived from the realized end (`action_distance`/length, distance-to-goal-from-end, realized direction). SPADL actions are already canonical action-LTR (converter output), so geometry is computed directly — **no orientation/`goal_map` handling** (a tracking concern). Non-finite input coordinates → all-NaN feature row → NaN probability (never fabricated).

**Why end-blind — a correctness requirement, not a caveat.** For a *failed* action the SPADL end is the outcome: the interception/death point exists only *because* the action failed. A model reading that end is therefore a **postdiction** (target leakage) — it inflates apparent discrimination by peeking at the answer, is not a valid ex-ante "expected completion", and would re-inject outcome bias into the metric built to remove it. `PassCompletionModel` is **not** a counter-precedent: it is *served* on a caller-*supplied* candidate target (a clean input), whereas xSuccess scores *real* actions whose end is the realized outcome. The house bar for this leak class is `ReceiverModel`, which is end-blind with an end-*location*-invariance guard (§6.4). This deviates from the paper's Table-1 (which used the realized end, contamination unmeasured — the review's SPEC-01 point); the deviation is deliberate. The richer completion capabilities live elsewhere, cleanly: **supplied-target** completion is `PassCompletionModel` (already shipped); **clean per-action pass length/direction** needs the *intended* target (ball release velocity), a tracking feature — a future tracking-augmented xSuccess (the paper's other future-work note), out of scope here.

### 5.3 Feature representation
Seed = the paper's Table-1 **minus every end-derived feature** (§5.2): `seconds`, `period_id`, `start_x`, `start_y`, `distance_to_goal` (from start), `angle_to_goal` (from start), `action_type`, `bodypart`. Categoricals (`action_type`, `bodypart`) one-hot into a fixed, contract-lockable column set. `FEATURE_NAMES` and the one-hot vocabulary are pinned in the artifact's feature-contract. The **final** representation is the OpenEvolve-discovered `xsuccess_features()` (§6) over the end-free allowlist **OR the seed if no candidate clears the §6.1 margin** (seed-fallback), committed as deterministic code; `FEATURE_NAMES` reflects whatever ships. **Stage-A outcome (2026-09-15, `docs/research/xsuccess_vaep_adjusted/`): the seed shipped** — the XGBoost-primary evolve gain was null (0.0006; a tree already extracts the geometry), so this seed IS the final representation.

### 5.4 Learner & calibration
- **Model:** XGBoost classifier (the gold-standard tabular choice; the paper's choice), tuned by Optuna (§6.2). Trained with a proper score (log-loss); **no `scale_pos_weight`** — imbalance is handled by calibration, not recall-trading (the `XShotOccurrenceObjective` reasoning).
- **Calibration:** an isotonic calibrator (cross-validated) is fitted on top **iff** it improves held-out per-type reliability (measured; not applied blindly). Its breakpoints serialize as JSON and apply pure-numpy at serve.
- **Fallback (gate-conditioned):** if the calibrated XGBoost fails the per-type calibration gate (§9.1) on the public corpus, fall back to a **per-type logistic** (one pure-numpy logistic per `action_type` on the shared feature set — `PassCompletionModel` generalized), calibrated-by-construction, one-sided types serving the base rate. The decision is recorded in the validation report; the shipped artifact declares which family it is.

### 5.5 Artifact, serialization, serve (ADR-011/016/040/050; xShot pattern)
- Pickle-free: `model.json` (native xgboost booster JSON; load via `load_xgb_booster_base_score_safe` for the 2.x/3.x `base_score` skew) + `metadata.json` + `SHA256SUMS`.
- `metadata.json` carries: `feature_names`, one-hot vocab, `feature_contract` (names + geometry constants + fixed-probe feature vector), `chirality` (fixed probe input + recomputed prediction fingerprint), isotonic params, `training_commit`, corpus manifest, model family (`xgboost` | `per_type_logistic`).
- **Fail-closed `load`:** SHA → chirality (recompute served predictions on a fixed asymmetric probe, `atol=1e-6`, `equal_nan=True`) → feature-contract (names + declared geometry constants, `atol=1e-6`); tamper/mismatch → `XSuccessIntegrityError`; missing fingerprint/contract → warn (pre-contract artifacts undeclared, not known-bad).
- `predict_success(actions) -> np.ndarray` (∈ [0,1]): build features → `booster.predict` → isotonic → NaN-in/NaN-out; unfitted → `XSuccessIntegrityError`.
- `bundled()` loads `weights/`.

---

## 6. Optimization pipeline (ruthless-efficiency, full stack)

All stages are dev/train-time, run by the implementing session; the shipped runtime is deterministic committed code.

### 6.1 Stage A — OpenEvolve feature discovery
- **Search object:** the body of `xsuccess_features(actions) -> (X, FEATURE_NAMES)` — an LLM code-search (OpenEvolve) that may invent interaction terms, nonlinear geometry transforms, and per-type tailoring, seeded from the **end-blind** baseline (§5.3) and restricted to the end-free allowlist (§6.4).
- **Fitness:** held-out `StratifiedGroupKFold`-by-match on a stratified corpus **subsample** (for tractable per-candidate eval): calibrated log-loss + a calibration-in-the-large term + per-type reliability, with TF-57-style gain/regression penalties; seeded, islanded, early-stopped on a combined score.
- **Seed fallback (TF61-SPEC-06):** if no evolved candidate beats the end-blind seed representation on held-out fitness by a pre-registered margin — **pinned at ≥ 0.005 absolute held-out log-loss** (rev 5, 2026-09-15: the earlier revs said "a pre-registered margin" without a number; pinned here so any re-run is governed) — **ship the seed**. Evolution is an opportunistic improvement, not a gate on delivery. **Stage-A outcome (2026-09-15, `docs/research/xsuccess_vaep_adjusted/`):** the XGBoost-primary run's gain was 0.0006 (null under any sane margin) → **seed ships**. A supplementary logistic-fitness run (owner-commissioned, where representation matters more than for a tree) found a real 0.0070 gain **for the `per_type_logistic` fallback family only** — deferred to the Commit-2 family gate (§9.1), not folded into the shared builder now.
- **Discipline (ADR-093):** the winning function is **human-reviewed and committed as deterministic code**; `docs/research/xsuccess_vaep_adjusted/provenance.json` records the OpenEvolve config, `run_commit`, clean-tree flag, and metrics; a **held-out LOO/CV check** confirms it is not in-sample-inflated. The evolve *process* is documented, not required to be bit-reproducible; the *output* is.

### 6.2 Stage B — Optuna HPO
`silly_kicks/xsuccess/_objective.py::XSuccessObjective` = the `XShotOccurrenceObjective` template verbatim: a ruthless `CachedObjective` with `prepare()` (build trial-invariant `(X, y, groups)` on the evolved representation), `evaluate_patch()` (per-trial `StratifiedGroupKFold` CV → held-out log-loss + Brier/PR-AUC diagnostics), and `evaluate()` (independent recompute) so `assert_cache_equivalence` holds to 1e-9. Search space: `{n_estimators, max_depth, learning_rate, min_child_weight, reg_lambda, reg_alpha, subsample, colsample_bytree}`. Minimizes held-out log-loss on the **full** corpus.

### 6.3 Stage C — calibrate + finalize
Fit the isotonic calibrator (§5.4) if it helps; train the final booster on the full corpus with the tuned HPs; validate (§9); save the artifact with clean provenance (`require_clean_tree` + `declare_inputs`, ADR-052/056).

### 6.4 Leakage safety (both stages)
- **Input allowlist (end-free):** the evolve harness and the feature builder may read only `{start_x, start_y, type_id, bodypart_id, time_seconds, period_id}` and **start**-derived geometry — **never** `end_x`/`end_y` (target leakage, §5.2), and **never** `result_id`/`result_name` or any label-derived column.
- **CI backstop — end-LOCATION invariance (TF61-SPEC-01/02):** perturb `end_x`/`end_y` (and `result_id`) on the input and assert the produced feature matrix is byte-identical. This directly mirrors **`ReceiverModel`'s** end-location-invariance guard (`tests/tracking/test_receiver_leakage_guard.py`). The guard is **novel to xSuccess** — `expected_passing` has no such guard, and a `result_id`-only perturbation (rev 1) checked the wrong vector (it would pass an end-contaminated model); an implementer must copy the end-location form, not a result-only one.

---

## 7. VAEP_adjusted

### 7.1 Inputs & flow
Inputs: a caller-fitted **standard** `VAEP` (result-bearing xfns), an injected `XSuccessModel`, `game`, `actions` (canonical SPADL), optional `frames` (for frame-aware xfns). Flow (`rate_adjusted`):
1. Compute the real gamestate features `X` (as `rate` does), and two **surgical** copies (§7.2): `X_succ`, `X_fail`.
2. Re-score the *already-fitted* classifiers: `p_scores_success = P_scores(X_succ)`, `p_concedes_fail = P_concedes(X_fail)`.
3. `xs = xsuccess.predict_success(actions)`.
4. `p_scores_adj = xs · p_scores_success`; `p_concedes_adj = (1 − xs) · p_concedes_fail`.
5. `adjusted_value(actions, p_scores_success, p_concedes_fail, xs)` → applies the weights and calls `formula.value` → `offensive_value / defensive_value / vaep_value` (adjusted). `return_components=True` also returns `p_scores_success_adj` etc. for the validation harness.

### 7.2 The surgical counterfactual (gold-standard, ceteris paribus)
Build `X_succ` / `X_fail` by overriding **only the current action's result-encoding feature columns** (`result_onehot`, `actiontype_result_onehot`) to success / fail on the already-computed `X` — holding locations, `goalscore`, temporal features, and the predecessors' (a1/a2) real result features fixed. This isolates the completion variable exactly (eqs 8–12 intent) and avoids the two artifacts of the paper's literal "flip the whole dataset" form (goalscore inflates when every shot becomes a goal; predecessor history is counterfactually rewritten). **Honest boundary (documented):** even the surgical flip holds the *realized* location — a full "what if it had succeeded" would move the ball to target vs turnover; re-simulating trajectories is a different (tracking/simulation) method, out of scope. Flipping the result feature while holding realized geometry is the standard, tractable, event-only counterfactual, and it is what VAEP's state value is built to respond to.

### 7.3 Guards
- **Non-vacuity + standard-features:** assert `X_succ` differs from `X_fail` on ≥1 row; if not (HybridVAEP removes the current-action result feature → no-op), **raise** with that explanation (the "every counterfactual needs a non-vacuity assertion" rule).
- **NotFittedError** if the VAEP has no fitted classifiers.
- **NaN propagation:** a NaN `xSuccess` (non-finite features) → NaN adjusted value for that row (never fabricated).
- **Purity:** `rate_adjusted` mutates no caller-supplied DataFrame (targeted test; consistent with the ADR-033 spirit though it is not an `add_*`).
- **frames** threaded through both counterfactual feature builds when the VAEP has frame-aware xfns.

---

## 8. Corpus & data
- **Training corpus:** the full **redistributable** StatsBomb Open Data set (multiple men's/women's World Cups and Euros, FA WSL, the La Liga Barça seasons, Champions League finals, NWSL, etc.), loaded only via the open-data loader `load_open_data_matches` (open-data == redistributable by construction — the `train_pass_completion` convention; `assert_public_corpus` is a *pining-corpus visibility* check, circular for open data, so it is NOT used here — TF61-IMPL-01) so nothing non-redistributable enters a published artifact. The **exact set of competitions** used is enumerated in the training script + recorded in the corpus manifest (not assumed "all of it"). This is much larger than WC2022 alone — important for rare-type calibration and closer to the paper's 3.6M-action regime.
- **Validation xG:** StatsBomb's `shot_statsbomb_xg` (injected, validation-only) for the aggregate-vs-xG check (§9.2). silly-kicks ships no xG model.
- **CV:** `StratifiedGroupKFold` grouped by `game_id` (stringified for cross-competition dtype safety, per the xShot objective).

---

## 9. Validation & success criteria

Method/unit tests **gate CI**; the applied corpus numbers are **reported** (ADR-009), in `docs/research/xsuccess_vaep_adjusted/` with clean provenance.

### 9.1 xSuccess (acceptance battery)
- **End-location leakage guard (CI-gated):** the §6.4 end-invariance test is the decisive, crisp leak gate — a contaminated model cannot pass it. Necessary because per-type calibration alone cannot detect outcome-peeking (a contaminated model would *ace* it — the SPEC-01 point).
- **Outcome-conditional negative control (proves the guard bites):** train an end-*using* control model and show the shipped end-blind model does **not** gain the control's extra discrimination — demonstrating the end-blind model earns its signal ex-ante (the "every counterfactual/guard needs a non-vacuity assertion" rule applied to the leak guard).
- **Per-type reliability:** predicted vs observed success rate per `action_type`, reliability curves + per-type Brier. *Necessary but not sufficient* on its own (see the guard above); decisive for VAEP_adjusted's multiply once leakage is excluded.
- **Calibration-in-the-large:** Σ predicted xSuccess ≈ # observed successes.
- **Discrimination:** OOF ROC-AUC + Brier overall and excluding one-sided types (report both, as the paper does).
- **Held-out generalization** of the evolved representation (LOO/CV, not in-sample-inflated).
- **Family gate (train-time + human review, not CI):** if the calibrated XGBoost misses the per-type calibration bar, ship the per-type-logistic fallback (§5.4); record the decision.
- **Gating map (SPEC-01 reconciliation):** CI-gates the unit/method tests **+** the end-location leakage guard; the calibration *family choice* is a train-time gate + human review; the applied calibration/AUC *numbers* are **reported, not gated** (ADR-009). "Calibration is the acceptance bar" means the train-time family-gate + review, not a CI assertion.

### 9.2 VAEP_adjusted (construct validity)
- **Aggregate-vs-xG alignment:** Σ `P(scores|success)_adj / k` ≈ total StatsBomb xG (the paper's headline check). **Penalty/corner note (TF61-SPEC-05):** `formula.offensive_value` overrides the previous-state odds with fixed penalty (0.792) and corner (0.0465) rates (`formula.py:71-77`); those rows contribute fixed odds, not xSuccess-weighted terms, and the aggregate check accounts for them explicitly.
- **Outcome-bias reduction:** adjusted values no longer separate by realized binary outcome at fixed context; rare-outcome value spikes suppressed vs standard VAEP.
- **Case-study sanity:** reproduce the *shape* of the paper's Dortmund sequence. Expectation under end-blind (§5.2): the **shot** correction reproduces cleanly (a lucky shot pulled toward its xG); the **risky-pass** illustration is *milder* than the paper's (end-blind xSuccess has coarser per-action pass granularity) — the documented, intended cost, not a defect.
- **Cross-check:** compute the paper's literal whole-dataset-flip numbers **once** to quantify the artifact and confirm the surgical version is in the right ballpark.

---

## 10. Testing & CI registries
- `tests/xsuccess/test_import_allowlist.py` — AST import boundary (no `tracking`; only `vaep` imports `xsuccess`).
- `tests/xsuccess/test_model_fit_predict.py` / `test_model_serialization.py` — fit→predict shape/range∈[0,1]; save/load round-trip; fail-closed load (SHA tamper / chirality / contract → raise; missing → warn).
- `tests/xsuccess/test_leakage_guard.py` — **end-location invariance** (TF61-SPEC-01/02): perturb `end_x`/`end_y` (and `result_id`) → features byte-identical (the `ReceiverModel` bar, §6.4); plus the outcome-conditional negative control (the end-blind model does not gain an end-using control's spurious discrimination).
- `tests/vaep/test_adjusted.py` — eq 8–12 correctness on a synthetic fixture (hand-computed adjusted value); non-vacuity (flip changes X; HybridVAEP → raise); surgical-flip scope (goalscore/locations/predecessors unchanged); NaN xSuccess → NaN; purity; and a **prev-weighting pin (TF61-SPEC-04)** — a multi-action, team-switch fixture asserting the `S_{i-1}` term is weighted by `xSuccess[i-1]` (via `formula`'s `prev()`), never `xSuccess[i]`.
- **No `*_xfns`, no glossary entry:** xSuccess is a model (reads `result` only as a training label) and VAEP_adjusted is a rating method — neither is an `add_*`/`*_xfns`, so the leaky-`*_xfns`-absence guard stays satisfied and `feature_glossary` does not grow (like `VAEP.rate`).
- **NOTICE** (ADR-005): Paul/Klemp/Memmert 2025; Anzer–Bauer *Expected Passes* (the paper's pass precedent); von Neumann–Morgenstern / Bernoulli expected-utility (theoretical basis); OpenEvolve / AlphaEvolve (feature-evolution method).
- **C4:** new `xsuccess` container in `architecture.dsl` → regenerate `architecture.html` (Graphviz `dot`); `add_*` count unchanged.
- **Provenance:** `train_xsuccess.py` + `evolve_xsuccess_features.py` call `require_clean_tree` + `declare_inputs`, register in `ARTIFACT_DRIVERS`, stamp `run_commit`/`run_tree_dirty` (ADR-052/056).
- **`[train]` extra:** `XSuccessObjective` + trainer never imported by `__init__`/inference. `[xgboost]` required at inference.
- **Cross-version safety:** `base_score` 2.x/3.x guard so no golden breaks across xgboost/pandas majors; ruff per-file-ignore for `X` naming; doctests via literal blocks where real `actions` are needed (public-surface only).
- **Slow/shard gating (ADR-023/074):** a does-it-run train/evolve smoke marked `@slow` (primary leg); serialization/golden tests on all legs; the full training/evolve are scripts, not CI.

---

## 11. Delivery, commit structure, approval gates
- **Branch:** `feat/tf61-xsuccess-vaep-adjusted` off `main` (no worktrees). Rebase onto latest `main` at commit-prep; set version/PR-S/ADR then.
- **Commit structure is CODE | ARTIFACTS (rev 3, owner-ratified 2026-09-15), NOT feature-split — clean-tree provenance forces code-before-weights (TF-57/TF-62 2-phase pattern):**
  - **Commit 1 — all code:** the `xsuccess/` package (incl. the **committed evolved `_features.py`**) + `_objective.py`; `vaep/adjusted.py` + `VAEP.rate_adjusted`; `evolve_xsuccess_features.py` + `train_xsuccess.py` + `publish_xsuccess.py`; all unit tests; NOTICE + C4 + CHANGELOG (`[Unreleased]` heading). **No version bump** (claimed in Commit 2, owner decision so a concurrent session can release first). Weights-dependent tests (`bundled()`, applied validation) **skipped-pending-weights**. Full suite green (`pytest -m "not e2e"`, ruff, pyright at CI scope), clean tree — the final train stamps `training_commit` against THIS commit.
  - **Commit 2 — artifacts:** bundled `weights/` + `MODEL_CARD.md` + the §9.1/§9.2 validation reports + `provenance.json`; un-skip the weights-dependent tests; **the per-PR minor version bump in `silly_kicks/_version.py` + the CHANGELOG/TODO version heading + PR-S/ADR** (next-free at commit-prep). Clean provenance.
- **Two commits on the one feature branch, one PR, one per-PR minor version bump** in `silly_kicks/_version.py` (claimed in Commit 2, deferred so a concurrent session can release first). Each commit is a fully-tested coherent state (no micro-commits).
- **Human gates (each separate, explicit):** approval of each commit's exact diff before it is made; the **HF publish** (outward-facing, after Commit 2 is approved); the **merge**; the **tag**. The `/review-impl` gate runs in an **independent session**; the author never self-reviews. TODO + CHANGELOG updated on release. Docs (spec/plan) are committed in Commit 1 before any artifact-driver run (a dirty tree, incl. untracked docs, makes `require_clean_tree` refuse).

---

## 12. Risks & mitigations
- **Calibration on a smaller corpus than the paper's 3.6M.** Mitigation: full open-data corpus (not WC2022), isotonic calibration validated per-type, per-type-logistic fallback if the gate fails.
- **Evolved-feature overfitting / opacity.** Mitigation: leakage-safe input allowlist; held-out LOO/CV fitness + non-inflation check; human review of the committed feature function for interpretability; feature-contract + chirality lock.
- **XGBoost calibration drift across xgboost majors.** Mitigation: `load_xgb_booster_base_score_safe`; behaviour-not-literal tests; isotonic on top.
- **Misuse on HybridVAEP** (silent no-op). Mitigation: the non-vacuity raise.
- **OpenEvolve run cost on a large corpus.** Mitigation: fitness on a stratified subsample; final HPO/train on the full corpus.
- **Faithfulness-vs-rigor question on the counterfactual.** Mitigation: ship the surgical (rigorous) form; report the literal-flip cross-check so the deviation is quantified, not hidden.
- **End-blind loses per-action pass granularity** (vs the paper's end-using model). Mitigation: it is the only *valid* event-only choice (§5.2, TF61-SPEC-01); shots — the headline de-biasing case — are unaffected; the coarser pass sensitivity is documented (§9.2), and the clean richer paths (`PassCompletionModel` supplied-target; a future tracking-augmented xSuccess) are named rather than faked.

## 13. Rejected alternatives
- **Single logistic + one-hot type** (weakest per-type calibration) — rejected for the model this multiplies.
- **Ghost-GK numpy leaf-walk promotion** to serve xSuccess pure of xgboost — unnecessary; the xShot precedent (xgboost at inference, `[xgboost]` extra) is simpler and adds no dependency for the VAEP_adjusted use.
- **A `VAEP_adjusted` subclass** — falsely implies a different fit; a `rate_adjusted` method is honest.
- **End-using / realized-end xSuccess features** (the paper's Table-1) — target-leaky: a failed action's end exists only *because* it failed, so the model becomes a postdiction, invalid as an ex-ante completion for *both* VAEP_adjusted and the scouting baseline (TF61-SPEC-01). The clean supplied-target case is `PassCompletionModel`; clean pass-length sensitivity needs tracking (future tracking-augmented xSuccess).
- **Shipping both an end-blind and an end-using xSuccess variant** — rejected: the end-using one is not a valid second model, only the same leak wearing a "variant" hat; the valid richer path already exists (`PassCompletionModel`, supplied-target).
- **The paper's literal whole-dataset flip** — confounded (goalscore inflation, predecessor rewrite); kept only as a one-off validation cross-check.
- **WC2022-only corpus** — an unexamined echo of `PassCompletionModel`; the real constraint is redistributability, and more data is strictly better for rare-type calibration.
- **Optuna-only (no evolve)** — leaves the paper's explicit "tailored feature engineering" lever unused; the owner asked to exercise the evolve capability.

## 14. References & attribution
- Paul, Klemp, Memmert (2025). *Beyond Outcome Bias …* MLSA 2025, `MLSA25_paper_225`. — xSuccess + VAEP_adjusted.
- Decroos, Bransen, Van Haaren, Davis (2019). *Actions speak louder than goals* (VAEP). — the base model.
- Anzer, Bauer (2022). *Expected Passes.* — the pass-completion precedent the paper generalizes.
- von Neumann & Morgenstern (1953); Bernoulli (1738/1954). — expected-utility / risk-reward basis.
- OpenEvolve / AlphaEvolve — the feature-representation evolution method (dev-time).
All added to NOTICE per ADR-005; per-feature docstrings cross-link `See NOTICE for full bibliographic citations.`
