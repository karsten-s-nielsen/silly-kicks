# Forecast-skill acceptance gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task
> (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the biased Brier, PR-AUC and log-loss acceptance gates of the xShot and xCross trainers with seven
nested-CV forecast-skill gates (point skill and match-bootstrap lower bounds), single-sourced for both trainers and
the integrity check.

**Architecture:** A pure `forecast_skill` / `skill_gates` pair in `silly_kicks/tracking/_artifact_integrity.py`
(the ADR-126 seam). Both trainers gain nested gate-fold studies (`{arm}_gf{k}`); each `cv` fold unit fits at its own
fold's params and returns its out-of-fold arrays as a FileSet; the `cv` reduce adds the `skill` dict. The trainers'
`_gates` are deleted, and `validate_trained_artifact_integrity.verdict` maps its two keys onto `skill_gates`.

**Tech Stack:** Python 3.12, numpy, scikit-learn (`average_precision_score`, `StratifiedGroupKFold`), xgboost,
ruthless (Optuna TPE), the ADR-128 stage pipeline (`scripts/_stages.py`).

**Spec:** `docs/superpowers/specs/2026-11-12-forecast-skill-gates-design.md` (r5, approved by A, B and the owner).
Section numbers below ("spec 4.3") refer to it.

**Status:** r4 APPROVED by A and B (2026-11-12). r3 was approved by A (FSG-PLAN-06) and B (no new finding); r4 answers FSG-PLAN-06. r2 was approved by A (FSG-PLAN-05) and B (no findings); r3 answered FSG-PLAN-05 and recorded the owner ruling on the AGENTS.md bullet. r1 was approved by A (FSG-PLAN-01 to 04) and B (FSG-PB01 to PB06); r2 answers every
finding (table at the end).

## Global Constraints

- One feature branch `feat/forecast-skill-gates`; no worktree; no subagents; ONE commit for the cycle, only after
  `/final-review` and the owner's explicit yes (no per-task commits).
- No full test suite on the AI-PC (overheating). Run the tests on a DGX git checkout, `~/sk-fsg` (a new directory;
  `~/sk-tf46-wt` is not touched). Set it up once:
  `git clone https://github.com/karsten-s-nielsen/silly-kicks.git ~/sk-fsg && git -C ~/sk-fsg checkout -b
  feat/forecast-skill-gates 05f3a692`, then a fresh venv the way CI builds one:
  `cd ~/sk-fsg && uv venv .venv && uv pip install -e ".[kloppy,xgboost,test,train]"` (CI's extras plus `train`,
  which the trainers' `ruthless` import needs). Sync the working tree before each run with
  `rsync -a --delete --exclude .git/ --exclude '.venv*/' --exclude __pycache__/ --exclude .pytest_cache/
  --exclude .ruff_cache/ --exclude .superpowers/ ./ karsten@192.168.68.73:~/sk-fsg/` from the repo root (staged or
  not; never a commit). The checkout keeps its own `.git` at `05f3a692`, so `git` calls in tests and
  `git_provenance` see the branch's changes as a dirty tree, as on the AI-PC. Run with
  `cd ~/sk-fsg && .venv/bin/python -m pytest ...`. Single fast unit tests may run locally. CI is authoritative.
- Lint runs in a second venv that mirrors CI's lint job (`.github/workflows/ci.yml:30-46`), never in the test venv:
  `cd ~/sk-fsg && uv venv --python 3.12 .venv-lint && uv pip install --python .venv-lint ruff==0.15.7
  pyright==1.1.409 pandas-stubs==2.3.3.260113 && uv pip install --python .venv-lint -e ".[test]" "numpy==2.5.2"`
  (two installs in CI's order: the second pins numpy, which pyright reads for types). Run the trio with that venv
  active: `cd ~/sk-fsg && . .venv-lint/bin/activate && ruff check silly_kicks/ tests/ scripts/ && ruff format --check
  silly_kicks/ tests/ scripts/ && pyright`. On the first run, `pyright --verbose` must list `.venv-lint` in its search
  paths (not `.venv`). The rsync exclusion `'.venv*/'` keeps both venvs.
- Temp files under `D:\Temp\hfu-topup\` (AI-PC) and `~/tf46-diag/` (DGX). Nothing in the bucket is deleted. No HF job.
- Constants (spec 4.3/4.4): `SKILL_BOOTSTRAP_DRAWS = 2000`, `SKILL_BOOTSTRAP_SEED = 0`, `SKILL_LEVEL = 0.95`,
  `SKILL_LOG_CLIP = 1e-15`. No CLI flag, campaign field or config value changes them.
- The pinned bootstrap call sequence of spec 4.3 is reproduced exactly (one `rng.integers(0, M, M)` per draw,
  `np.bincount(idx, minlength=M)[inv].astype(np.float64)`, `labels, inv = np.unique(groups.astype(str),
  return_inverse=True)`, `np.percentile(gain, 5)` with NaN propagation).
- Acceptance keys, in this order (spec 4.4): `enough_usable_folds`, `brier_skill_gt_0`, `log_loss_skill_gt_0`,
  `pr_auc_lift_gt_0`, `brier_gain_lb95_gt_0`, `log_loss_gain_lb95_gt_0`, `pr_auc_lift_lb95_gt_0`.
- Existing candidate metric keys keep their names (`pr_auc`, `brier`, `log_loss`, `pr_auc_std`, `positive_rate`,
  `base_rate_brier`, `n_usable_folds`); the new keys are `skill` and `fold_params` (spec 4.5).
- STE prose in docs, comments and the commit message. Lint at CI scope: `ruff check silly_kicks/ tests/ scripts/`,
  `ruff format --check silly_kicks/ tests/ scripts/`, bare `pyright`.
- The retirement and model-card changes are NOT in this cycle (spec 10); this cycle writes no code for them.

## Plan rulings (deviations from the spec text, for the A/B review)

- **R1. `y.npy` in the cv fold FileSet.** Spec 5 lists `fold.json`, `te.npy`, `p.npy`, `groups.npy`. The reduce reads
  only unit results (FSG-SPEC-01), so it also needs the held-out labels. Add `y.npy` (int8). Cost if wrong: one
  extra small member per unit.
- **R2. The toy-corpus measurement runs in Task 6, not as the first task.** Spec 7 says "the plan's first task runs
  each of these corpora under the new gates". The seven values on a toy corpus exist only once the trainers are wired
  (Tasks 4 and 5). Task 6 runs the measurement BEFORE any toy or test edit, which keeps the spec's intent (measure
  first, never weaken without a ruling). Cost if wrong: none to the code; only the order.
- **R3. `_cv_score` (the GK-block ablation scorer) gets no `skill`.** Spec 5 leaves this to the plan. The ablation
  reports only `pr_auc`/`log_loss`, so it keeps its one-params signature and its exact keys.

## Review Focus

1. A `_gf` tag must resolve to its gate-fold rows. Today `single_gf0` and `public_gf1` contain no `_f`, so
   `rpartition("_f")` returns an empty separator and the tag falls through to `inp.arms[tag]` (`KeyError`); a future
   arm name that contains `_f` could also be parsed as a public-fold tag. `_study_rows` checks `_gf` first (Task 3,
   test `test_gf_tag_never_parsed_as_public_fold_tag`).
2. A skipped fold (one class in its training rows) contributes no rows; its `fold_params` entry stays (its `_gf`
   study ran); the pooled arrays stay aligned (Task 4, test `test_reduce_with_a_skipped_fold`).
3. Weighted AP with zero weights and tied scores must equal scikit-learn exactly (Task 1, test 4).
4. The reduce result must not depend on the order of its unit results or on W/N (Task 4 test, Task 5 parity).
5. A FAILED payload must carry the full `skill` dict so Part Deux can apply the finite-values rule of spec 6.1
   (Task 5, test `test_failed_payload_carries_skill`).

---

### Task 1: `forecast_skill` and `skill_gates` (the pure core)

**Files:**
- Modify: `silly_kicks/tracking/_artifact_integrity.py` (append; keep `physical_upper_bound`, `base_rate_plausible`)
- Test: `tests/tracking/test_forecast_skill.py` (new)

**Interfaces:**
- Produces:
  - constants `SKILL_BOOTSTRAP_DRAWS`, `SKILL_BOOTSTRAP_SEED`, `SKILL_LEVEL`, `SKILL_LOG_CLIP` (values above);
  - `weighted_average_precision(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float` — scikit-learn's
    `average_precision_score(y, p, sample_weight=w)`, computed from one sort; a `_PresortedFold` helper (private)
    holds the descending order and distinct-threshold index per fold so the bootstrap sorts once per fold;
  - `forecast_skill(y, p, r, fold, groups, *, nested: bool) -> dict` — the spec 4.5 `skill` dict, key for key
    (including `n_empty_fold_draws`, `n_one_class_fold_draws`, and `"nested": nested`);
  - `skill_gates(metrics: Mapping[str, Any]) -> dict[str, bool]` — the 7 keys of spec 4.4 from one candidate's
    metrics (`n_usable_folds` and `skill`); a missing `skill` gives False for the six skill keys.

- [ ] **Step 1: Write the failing tests** in `tests/tracking/test_forecast_skill.py`:
  - `test_constant_train_rate_forecast_fails_old_bar_and_is_neutral` (spec Test 1): synthetic grouped corpus, 10
    matches, 5 folds by `StratifiedGroupKFold(5, shuffle=True, random_state=42)` with unequal fold rates; `p = r` =
    training-fold rate. Assert the old bar fails: `mean_k Brier_k(r) >= base * (1 - base)`. Assert
    `s["brier_skill"] == 0.0`, `s["log_loss_skill"] == 0.0`, `s["pr_auc_lift"] == 0.0` exactly, and
    `skill_gates(...)` gives False for `brier_skill_gt_0`, `log_loss_skill_gt_0`, `pr_auc_lift_gt_0`.
  - `test_informative_forecast_passes_all_seven` (Test 2): 60 matches, `p` a calibrated monotone function of a
    feature that drives `y`; all 7 gates True.
  - `test_bounds_react_to_sample_size` (Test 3): the same per-match effect with 4 matches gives
    `bootstrap["brier_gain_lb"] <= 0`, with 200 matches `> 0`.
  - `test_weighted_ap_equals_sklearn` (Test 4): 200 random cases with ties in `p` (rounded to 2 decimals) and
    zero weights; `abs(weighted_average_precision(y, p, w) - average_precision_score(y, p, sample_weight=w)) <= 1e-12`.
  - `test_pinned_sequence_matches_reference` (Test 5): a fixed 12-match input; the test holds the spec 4.3 reference
    code verbatim (as `gold3.py` implements it, using `average_precision_score` directly) and asserts the three lower
    bounds and both counts equal `forecast_skill`'s exactly (`==`).
  - `test_byte_equal_json` (Test 6, first half): two calls give equal `json.dumps(..., sort_keys=True)`.
  - `test_nan_and_dropped_folds` (Test 7): a NaN in `skill["brier_skill"]` makes `brier_skill_gt_0` False; an input
    with one match without positives in a 3-match fold gives `n_one_class_fold_draws > 0` and a finite
    `pr_auc_lift_lb`; a 2-match input with one positive match gives draws with no usable fold, so
    `pr_auc_lift_lb` is NaN and `pr_auc_lift_lb95_gt_0` False; `enough_usable_folds` follows `n_usable_folds >= 2`.
  - `test_gate_keys_and_order`: `list(skill_gates(m))` equals the Global Constraints key list.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/tracking/test_forecast_skill.py -v` (local is fine; light).
  Expected: FAIL with `ImportError` (`forecast_skill` not defined).

- [ ] **Step 3: Implement** the four Produces items in `_artifact_integrity.py`. Brier and log-loss terms per row as
  spec 4.2: the Brier terms are unclipped; in the log-loss terms only, clip `p` and `r` to
  `[SKILL_LOG_CLIP, 1 - SKILL_LOG_CLIP]`; point `pr_auc_lift` per spec 4.2 over folds with both classes,
  weights `n_k`; bootstrap exactly as the Global Constraints sequence; per draw, for each fold compute `s_k`,
  `rate_k`, drop per spec 4.3, count `n_empty_fold_draws` (`s_k == 0`) and `n_one_class_fold_draws` (`s_k > 0` and
  rate 0 or 1). Fractions `<= 0` over draws (NaN draws count as not `<= 0`). Module docstring gets a paragraph
  pointing to ADR-131.

- [ ] **Step 4: Run** the same command. Expected: 8 passed.

### Task 2: the integrity verdict reads `skill_gates`

**Files:**
- Modify: `scripts/validate_trained_artifact_integrity.py:68-110` (`verdict`, module docstring lines 1-20)
- Test: `tests/scripts/test_validate_trained_artifact_integrity.py`

**Interfaces:**
- Consumes: `skill_gates` (Task 1).
- Produces: `verdict(...)` with `lift_ok = g["pr_auc_lift_gt_0"] and g["pr_auc_lift_lb95_gt_0"]` and
  `calibration_ok = g["brier_skill_gt_0"] and g["log_loss_skill_gt_0"] and g["brier_gain_lb95_gt_0"] and
  g["log_loss_gain_lb95_gt_0"]` (spec 4.5); keys and `is_go` unchanged.

- [ ] **Step 1: Write the failing tests** `test_verdict_maps_skill_gates` (a metrics dict whose `skill` passes all
  six gives both True; flipping only `bootstrap.brier_gain_lb` to −1e-6 gives `calibration_ok` False and `lift_ok`
  True; flipping only `bootstrap.pr_auc_lift_lb` gives the reverse) and `test_verdict_fails_closed_without_skill`
  (old-shape metrics with `brier < base_rate_brier` and `pr_auc > positive_rate` but no `skill` → both False,
  `is_go` False). Update `test_verdict_passes_corrected_fails_defect` fixtures to carry a `skill` dict.
- [ ] **Step 2: Run** `pytest tests/scripts/test_validate_trained_artifact_integrity.py -v`. Expected: the two new
  tests FAIL (old bars used).
- [ ] **Step 3: Implement** the mapping in `verdict`; docstring layer 2 text: "proper-score skill (ADR-131)".
- [ ] **Step 4: Run** again. Expected: all pass.

### Task 3: nested gate-fold study tags (both trainers)

**Files:**
- Modify: `scripts/train_xshot_occurrence.py` (`_prep_work` ~`:599-606`, `_study_rows` `:669-680`)
- Modify: `scripts/train_xcross_attempt.py` (`_prep_work` `:767-773`, `_study_rows` `:838-846`)
- Test: `tests/scripts/test_nested_gate_folds.py` (new)

**Interfaces:**
- Produces:
  - `prep` output `study_tags` = today's list + `[f"{arm}_gf{k}" for arm in sorted(cv_arms) for k in
    range(cv_arms[arm])]` (appended; `single_gf{k}` on the non-paired path);
  - `_study_rows(inp, tag)`: for `name, sep, k = tag.rpartition("_gf")` with `sep`, `k.isdigit()` and `name` equal
    to `"single"` or in `inp.arms`: `rows = _arm_rows(inp, name)`; `tr` = training indices of fold `int(k)` of the
    trainer's gate plan (`_cv_metrics_plan` for xShot, `_xcross_eval._cv_score_plan` for xCross) on
    `(inp.y[rows], inp.groups[rows])`; return `rows[tr]` (ascending). This branch is checked BEFORE the `_f`
    branch. `_arm_rows` already exists in both trainers.

- [ ] **Step 1: Write the failing tests** (parametrized over both trainer modules, on a small synthetic
  `StudyInputs`-like object with `X`, `y`, `groups`, `arms`, `is_public`):
  - `test_prep_emits_gate_fold_tags` (spec Test 12): build `cv_arms`; assert the `_gf` tags appear for every arm
    and `k < cv_arms[arm]`, after the existing tags. Exercise through the tag-list helper that `_prep_work` uses;
    extract that list construction into `_study_tags(arms, folds, cv_arms, run_paired) -> list[str]` in each trainer
    so the test does not need a dataset.
  - `test_gf_rows_exclude_their_fold` (Test 12): for each `k`, `_study_rows(inp, f"{arm}_gf{k}")` is ascending,
    is a subset of the arm's rows, and is disjoint from fold `k`'s held-out rows of the gate plan; the union with
    fold `k`'s rows is the arm's rows.
  - `test_gf_tag_never_parsed_as_public_fold_tag` (Test 14, Review Focus 1): on a paired input, `single_gf0`,
    `public_gf1` and `public_f1` resolve to three different row sets, and `public_f1` still equals today's result.
- [ ] **Step 2: Run** `pytest tests/scripts/test_nested_gate_folds.py -v`. Expected: FAIL (`_study_tags` missing,
  `_gf` tag raises `KeyError`).
- [ ] **Step 3: Implement** `_study_tags` and the `_gf` branch in both trainers.
- [ ] **Step 4: Run** again. Expected: all pass.

### Task 4: the fold seam, the `cv` units and the `cv` reduce (both trainers)

**Files:**
- Modify: `scripts/train_xshot_occurrence.py` (`_cv_metrics_fold` `:151-172`, `_cv_metrics_combine` unchanged,
  `_cv_metrics` `:193-204`, `_cv_work` `:776-787`, `_cv_reduce` `:790-799`)
- Modify: `silly_kicks/tracking/_xcross_eval.py` (`_cv_score_fold` `:43-75`; `_cv_score`, `_cv_score_combine`
  unchanged)
- Modify: `scripts/train_xcross_attempt.py` (`_cv_metrics` `:194-205`, `_cv_work` `:934-950`, `_cv_reduce`
  `:953-969`)
- Test: `tests/tracking/test_objective_fold_split.py`, `tests/scripts/test_nested_gate_folds.py`

**Interfaces:**
- Consumes: `forecast_skill` (Task 1); `{arm}_gf{k}` studies (Task 3).
- Produces:
  - `_cv_metrics_fold(..., *, return_oof: bool = False)` and `_cv_score_fold(..., return_oof: bool = False)`:
    with `return_oof=True` they return `(metrics_tuple, p: np.ndarray, train_rate: float)` or None, where
    `train_rate = float(y[tr].mean())` (the TRUE rate, before subsampling); with False, today's return, byte-equal;
  - `_cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42)` (both trainers): `params` is a dict
    (every fold) or a sequence of per-fold dicts; returns today's keys plus `skill` (`nested=not isinstance(params,
    Mapping)`) and `fold_params` (list, one entry per gate fold in fold order; a skipped fold keeps its entry,
    because its params exist; only its rows are absent);
  - `_cv_work` returns `FileSet({"fold.json", "te.npy", "p.npy", "y.npy", "groups.npy"})`: `fold.json` =
    `{"fold": <tuple or null>, "positive_rate": <arm rate>, "train_rate": <float or null>}`; arrays empty for a
    skipped fold; `groups.npy` is `np.asarray(groups[te]).astype(str)` saved with `allow_pickle=False`; params from
    `ctx.upstream("study")[f"{arm}_gf{k}"]["params"]`;
  - `_cv_reduce` returns, per arm, today's dict (from the `fold.json` tuples, fold order) plus `skill` (from the
    pooled arrays sorted by `te`, `nested=True`) and `fold_params` (from the `_gf` studies, fold order).

- [ ] **Step 1: Write the failing tests:**
  - in `test_objective_fold_split.py`: `test_return_oof_keeps_the_tuple` (for both fold functions, the first
    element with `return_oof=True` equals the plain return, and `p` has `len(te)`); extend
    `test_cv_metrics_split_equals_the_monolith` and `test_xcross_cv_score_split_equals_the_monolith` so the
    existing keys stay `==` the legacy oracle at one params dict (spec Test 8).
  - in `test_nested_gate_folds.py`: `test_cv_unit_uses_its_fold_study` (spec Test 13): a fake `ctx` whose
    `upstream("study")` holds two different `_gf` params (n_estimators 5 vs 50) for folds 0 and 1; the two fold
    units' `p` differ from a run with one shared params, and each equals a direct fit at its own params;
    `test_reduce_order_independent` (Test 6, Review Focus 4): shuffled unit results give `==` reduce output;
    `test_reduce_with_a_skipped_fold` (Review Focus 2): a fold with a one-class training set gives no rows, the
    pooled arrays stay aligned, and `n_usable_folds` drops by 1; `test_staged_equals_cv_metrics_composition`:
    the reduce output equals `_cv_metrics(X, y, groups, fold_params)` key for key.
  - The fake `ctx` patches the trainer's `_inputs` to return a local `StudyInputs` (`monkeypatch.setattr`).
- [ ] **Step 2: Run** `pytest tests/tracking/test_objective_fold_split.py tests/scripts/test_nested_gate_folds.py -v`.
  Expected: the new tests FAIL (`return_oof` unknown; `_cv_work` returns a dict).
- [ ] **Step 3: Implement** the Produces items. Read FileSet members with the `train_xsuccess` pattern
  (`np.load(io.BytesIO(b), allow_pickle=False)`); write with `np.save` into `io.BytesIO`. Keep the existing combine
  calls unchanged so the existing keys stay byte-identical.
- [ ] **Step 4: Run** again. Expected: all pass.

### Task 5: the acceptance gate in the trainers, and the degenerate toy

**Files:**
- Modify: `scripts/train_xshot_occurrence.py` (delete `_gates` `:207-215`; `:878`, `:937` call `skill_gates`)
- Modify: `scripts/train_xcross_attempt.py` (delete `_gates` `:208-216`; `:1051`, `:1190` call `skill_gates`)
- Test: `tests/tracking/test_xshot_occurrence_integration.py` (`:470-474`, `:589-598`),
  `tests/tracking/test_xcross_attempt_integration.py` (`:184-191`, `:294`), `tests/scripts/hf_jobs/test_adopt.py`
  (`:1201-1220`)

**Interfaces:**
- Consumes: `skill_gates` (Task 1); the reduce's `skill` (Task 4).
- Produces: `metrics.json["acceptance"]` and `only.metrics_FAILED.json["acceptance"]` with the 7 keys; the FAILED
  payload's `candidates[*]["metrics"]["skill"]` present.

- [ ] **Step 1: Write the failing tests:** `test_failed_payload_carries_skill` (Review Focus 5) and the degenerate-toy
  tests changed to assert `not all(acceptance.values())` AND at least one of the three `*_lb95_gt_0` keys False
  (spec Test 11); `:470-474` uses `skill_gates(m)` and keeps its assertion that all seven are True (Task 6 measures
  its corpus and applies the spec 7 rule; this task changes only the keys); `test_adopt.py` refit-mode fixtures use the 7 new keys (the
  all-True and one-False cases).
- [ ] **Step 2: Run** those tests on the DGX copy. Expected: FAIL (old keys).
- [ ] **Step 3: Implement**: delete both `_gates`, import `skill_gates`, and grep that no `_gates(` caller remains
  (`git grep -n "_gates(" scripts tests` shows only `skill_gates`).
- [ ] **Step 4: Run** again on the DGX. Expected: pass.

### Task 6: the toy corpora under the new gates (spec 7 rule)

**Files:**
- Test (measure, then maybe modify): `tests/tracking/test_xshot_occurrence_integration.py:471-474` and `:536-550`,
  `tests/tracking/test_xcross_attempt_integration.py:131,174`, `tests/scripts/test_parity_xshot_occurrence.py`,
  `tests/scripts/test_parity_xcross_attempt.py`, the wiring tests on `XSHOT_TOY_PROVIDERS`
  (`scripts/_toy_corpora.py:479`)

- [ ] **Step 1: Measure first.** On the DGX, run each test above without edits. Record for each corpus the seven
  values and the `skill` dict (print from the produced `metrics.json` or `only.metrics_FAILED.json`) in
  `D:\Temp\hfu-topup\toys.md`.
- [ ] **Step 2: Apply the rule** per test: passes → keep "all gates pass"; fails → strengthen the toy signal or add
  toy games in `scripts/_toy_corpora.py`, update every test that pins that toy in this task, and re-run. A test that
  cannot pass after that goes to the owner with the Step 1 numbers before any assertion changes.
- [ ] **Step 3: Run** the full toy set again on the DGX. Expected: all pass, with the owner's ruling recorded for any
  exception.

### Task 7: records

**Files:**
- Create: `docs/superpowers/adrs/ADR-131-forecast-skill-gates.md` (from `docs/superpowers/adrs/ADR-TEMPLATE.md`)
- Modify: `docs/context/trained-models.md` (new bullet), `AGENTS.md` (one Key-conventions bullet, ≤ 600 chars,
  `ADR-131` pointer; beyond spec 9's list; owner ruling 2026-11-12: "Add the bullet"), `CHANGELOG.md`, `silly_kicks/_version.py`, `docs/superpowers/adrs/ADR-126-*.md` (status note:
  amended by ADR-131)

- [ ] **Step 1: Write ADR-131** with the content spec 9 lists: the defect, the seven gates, nested CV, the
  measurement (the failed 1e-12 self-check, gold to gold3, nested), the order of events (spec 1), the meaning of the
  bound (spec 4.3), the integrity mapping (spec 4.5), the existing keys now nested (+0.03% to −3.32% `pr_auc`), and
  the retirement rule with its corrected evidence (spec 6, 6.1).
- [ ] **Step 2: Write** the `trained-models.md` bullet and the AGENTS.md bullet; run
  `pytest tests/test_agents_md_budget.py -v` locally. Expected: pass.
- [ ] **Step 3: CHANGELOG and version** at commit preparation (release numbers are set then, not earlier): the entry
  names the gate change, nested CV, the new `skill`/`fold_params` keys, the removed acceptance keys, and the
  integrity mapping. The version moves in `silly_kicks/_version.py` only (ADR-079); the wheel changes
  (`_artifact_integrity.py`, `_xcross_eval.py`).

### Task 8: DGX validation against the recorded measurement (spec 8)

**Files:**
- Create (off-repo): `D:\Temp\hfu-topup\diag\validate_r9a.py`, copied to `~/tf46-diag/`

- [ ] **Step 1: Write** `validate_r9a.py`; it runs from `~/sk-fsg` (the synced branch) with `~/sk-fsg/.venv/bin/python`. For each of the 8 run 9a arm-variants under `~/tf46-diag/gold/`: patch
  the trainer's `_inputs` to return the local `inputs/` (`load_study_inputs`); build the `_gf` tags with
  `_study_tags`; run the trainer's `_study_work(tag, ctx)` per tag with a stub `ctx` (`pool=None`, a temp store,
  `STUDY_RESUME` off); run `_cv_work` for every `{arm}__{k}` and `_cv_reduce` on the results. Time the reduce.
- [ ] **Step 2: Check 1.** Each `{arm}_gf{k}` params dict `==` the matching `f{k}` entry of
  `nested_params\<arm>__<variant>.json`.
- [ ] **Step 3: Check 2.** Each `skill` value within 1e-9 of `nested_report.json`; integer and string keys equal;
  `nested` True. The acceptance blocks give 7 passes and `xcross_attempt-public` / `position_only` failing only
  `brier_gain_lb95_gt_0`.
- [ ] **Step 4: Run** in the background on the DGX, one process per study as `nested.py` did:
  `concurrent.futures.ProcessPoolExecutor(max_workers=20)` over the 40 `(arm, tag)` study jobs (each process patches
  `_inputs` itself), then the cv units and reduces in the parent. Expected wall time, from `nested.py`'s measured
  times (public phase about 5 min; sc_extended phase bounded by the slowest xShot study, 44 min): about 60 min.
  Command: `setsid nohup timeout 10800 ...` (3 h). At the limit the run is killed; no partial result counts; report
  it to the owner with the log before any rerun. Expected: both checks pass; record the reduce times for the
  hand-off.

### Task 9: full verification, review and the commit gate

- [ ] **Step 1:** On the DGX copy: the lint trio in `.venv-lint` (Global Constraints). The full test suites run in
  CI, not on the DGX (owner ruling 2026-11-12: "a waste of over an hour when ci can check it in 10 minutes"). CI runs
  only on a pull request to `main` or a push to `main` (`.github/workflows/ci.yml:3-7`), so the Step 4 gate covers
  commit, push and a draft PR. The per-task targeted tests and the Task 6 toy run stay as run.
- [ ] **Step 2:** `/final-review`, before the A/B review (owner ruling 2026-11-12), so its fixes land before the
  reviewers read the diff.
- [ ] **Step 3:** Send the implementation to reviewers A and B (`/review-impl`, reports to `D:\Development\_reviews\`);
  fix or answer every finding.
- [ ] **Step 4:** Show the owner the staged diff and file list, and ask one question that names the three actions: "commit,
  push to `feat/forecast-skill-gates`, and open a draft PR so CI runs?". Act only on an explicit yes to that question:
  one commit (Conventional Commits, STE body, `Co-Authored-By` line), the push, the draft PR. A merge or tag each need
  their own yes. A red CI is fixed before the PR leaves draft.
- [ ] **Step 5 (amended by the owner ruling "Widen before refit", 2026-11-12):** the refit waits for the corpus-widening
  cycle. After this commit, message Part Deux that the gate change is on the branch; send the full hand-off below with
  the widening cycle's commit.
- [ ] **Step 5b:** Message Part Deux (`2#silly-kicks (Deux): HF Jobs`): the commit, "all 8 arms", the four model cards
  (O2), the spec 6.1 rule with its prediction and checklist, and the Task 8 reduce times. Each launch needs the
  owner's go.

## r1 review findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-PLAN-01, FSG-PB01 (Task 8 parallelism and timeout) | Task 8 Step 4: one process per study, 20 workers, about 60 min expected, `timeout 10800`; at the limit nothing counts and the owner gets the log. |
| FSG-PLAN-02 (CI markers) | Task 9 Step 1 uses CI's selections and flags (`ci.yml:127`, `:201`, `:138`), without sharding. |
| FSG-PLAN-03 (rsync) | Global Constraints: the exact rsync command into a new `~/sk-fsg`, exclusions named, `--delete`, the import check. |
| FSG-PLAN-04, FSG-PB03(b) (AGENTS.md bullet beyond spec 9) | Marked in Task 7 for the owner's ruling at plan approval. |
| FSG-PB02 (one yes for commit and push) | Task 9 Step 4: one question that names both actions. |
| FSG-PB03(a) (`:470-474` before Task 6) | Task 5 states the assertion stays "all seven True"; Task 6 applies the spec 7 rule. |
| FSG-PB04 (Review Focus 1 premise) | Rewritten: the `_gf` tags contain no `_f` and fall through to a `KeyError` today. |
| FSG-PB05 (clip wording) | Task 1 Step 3: Brier unclipped; `p` and `r` clipped in the log-loss terms only. |
| FSG-PB06 (skipped-fold `fold_params`) | One rule: one entry per gate fold, a skipped fold keeps its entry (Task 4 Interfaces, Review Focus 2). |

## r2 review findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-PLAN-05 (the copy has no `.git`) | Global Constraints: `~/sk-fsg` is a git clone at `05f3a692` with the working tree synced over it, and a fresh venv built with CI's extras plus `train`. |
| FSG-PLAN-04 (AGENTS.md bullet) | Owner ruling 2026-11-12: "Add the bullet". |

## r3 review findings and their resolution

| Finding | Resolution |
|---|---|
| FSG-PLAN-06 (lint env is not CI's) | Global Constraints: a separate `.venv-lint` built exactly as CI's lint job (pinned ruff, pyright, pandas-stubs; `.[test]` with `numpy==2.5.2`); Task 9 Step 1 lints there. |
