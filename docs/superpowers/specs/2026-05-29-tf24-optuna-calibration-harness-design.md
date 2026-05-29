# TF-24 PR-B — Optuna calibration harness — design

**Date:** 2026-05-29
**Status:** Design — pending user + lakehouse review
**Type:** New optional subpackage + extra + CLI. Minor release.
**Sequence:** PR-B, the calibration harness. Consumes PR-A (`add_gradientsports_player_ids`,
shipped 3.27.0) for GS data prep, and `ruthless-efficiency[optuna]` for the Optuna machinery.
**Scope:** harness ONLY — it produces recommended default values + a report; it does NOT change
the library default constants (that is a separate, data-dependent "apply" PR after the
maintainer's real run).

## 1. Goal

Replace three sets of **engineering-choice tracking defaults** in silly-kicks with
**Optuna-calibrated** values, validated against real multi-provider tracking/event data:

| Stage | Target | Params | Objective | Direction |
|-------|--------|--------|-----------|-----------|
| **1** | `infer_ball_carrier` | `tolerance_m`, `beta`, `gamma` | carrier accuracy `(inferred == action.player_id).mean()` over on-ball actions at linked frames | maximize |
| **2** | `LinkParams.k3` + off-ball-runs | `k3`, `pre_seconds`, `min_displacement_m` | augmented-VAEP held-out **Brier**, per-provider CV, equal-provider-weight mean | minimize |

Stage 1 runs first — its optimum (carrier params) feeds Stage 2's `derive_team_in_possession` →
DAS-context enrichment. **Optuna TPE, not evolve** (single/fixed-form scalars vs a downstream
metric is textbook Bayesian optimization; do not re-litigate — handoff rubric).

This is silly-kicks **TF-24** / lakehouse **TC-3** (same work from each repo's side). The prior
lakehouse implementation (a 2092-LOC monolith) reached Stage-1 done + Stage-2 12/30 trials before
dying; this reimplements it on `ruthless` Phase 2 primitives.

## 2. Architecture — pure core + I/O loaders

**Pure, provider-agnostic, CI-testable logic in the library; all I/O + orchestration in `scripts/`.**

### `silly_kicks/calibration/` (new subpackage; `[calibration]` extra = `ruthless-efficiency[optuna]` + `xgboost>=2.0,<3.0` (bounded — a future major can silently change `tree_method` defaults/determinism and break the 1e-9 cache gate; R4); lazy-import-guarded like `tracking/_das.py`; NOT imported by `silly_kicks/__init__`)
- `_features.py` — the enrichment functions + the `ALL_FEATURES` column list (§4a): `enrich_invariant(actions, frames, *, xt, home_team_id, carrier_params) -> (base_actions, links)` (the 14 trial-independent steps, with the 5 trial-varying cols as NaN placeholders) and `patch_trial_columns(base_actions, frames, links, *, home_team_id, k3, pre_seconds, min_displacement_m)` (the 2 trial-varying steps). Pure; provider-agnostic.
- `_xt.py` — `fit_frozen_xt(corpus_actions, *, exclude_match_ids) -> FrozenXt` (asserts zero overlap), `save_xt`/`load_xt` (npz + sha256), `FrozenXt` (wraps `ExpectedThreat` + provenance: source, corpus match-ID set, fit date, grid shape, sha256).
- `_carrier_objective.py` — `CarrierAccuracyObjective` (ruthless `Objective`).
- `_vaep_brier_objective.py` — `AugmentedVaepBrierObjective` (ruthless `CachedObjective`).
- `_cv.py` — match-stratified CV splitter, **single count-driven threshold**: GroupKFold(5) for >7 matches, leave-one-match-out for ≤7. The loader **logs each provider's actual match count + the scheme chosen** so the assignment self-adjusts and is auditable. Random-action splits forbidden (leak match structure).
- `_gates.py` — H1 degenerate-feature penalty + per-provider signal sanity gate.
- `_spaces.py` — the two `OptunaConfig` builders (param spaces, warm-start, SQLite store).
- `_diagnostics.py` — Phase-3 per-provider sensitivity + TF-25 gate.
- `__init__.py` — curated exports.
- The objectives consume `(frames, actions)` DataFrames + a `FrozenXt` only — provider/source agnostic.

### `scripts/calibrate_tracking_defaults.py` — thin CLI
`--stage 1|2|diagnostics`, `--store <sqlite path>`, `--n-trials`, `--seed`, `--providers`,
`--source <pining|databricks|auto>`, `--xt-artifact <path>` (fit-and-freeze on first use from
`bronze.spadl_actions` minus calibration matches, else load + sha256-verify; `--xt-bundled` falls
back to socceraction's grid). Owns I/O + study orchestration via two pluggable loaders,
each yielding a uniform `(frames, actions, home_team_id)` per `(provider, match_id)`:
- `_loader_pining.py` — provider-agnostic pining-for-the-data fetch (Bearer → **302 → presigned S3
  two-step**: GET `/{provider}/matches/{id}/{artifact}` with the bearer + auto-redirect DISABLED,
  read `Location`, then GET the presigned URL **without** the bearer; `PINING_FOR_THE_DATA_TOKEN`
  for owner-tier, `test-token-pining-for-the-data` public default; base URL via `PINING_API_URL`).
  **Serves all three providers** (verified 2026-05-29): **SkillCorner** (public, 10) + **IDSSE**
  (public, 7 — now live) + **Gradient Sports** (owner-tier, 64). Per-provider artifact formats
  differ, so the loader has a **provider dispatch table**: IDSSE = DFL/Sportec XML
  (`events.xml`/`metadata.xml`/`tracking.xml`, tracking ~419 MB → stream to temp file → kloppy
  Sportec events + native `tracking.sportec` frames); SkillCorner = native CSV/JSON/JSONL with
  **match-id-prefixed artifact keys** (`{id}_dynamic_events.csv` / `{id}_match.json` /
  `{id}_tracking_extrapolated.jsonl` → SkillCorner / kloppy gateway); GS = JSON + bz2-JSONL
  (`add_gradientsports_player_ids` → frames, per PR-A).
- `_loader_databricks.py` — `soccer_analytics.bronze.{provider}_{tracking,events}` via
  `databricks-sql-connector` (env `DATABRICKS_HOST`/`HTTP_PATH`/`TOKEN`/`SQL_WAREHOUSE_ID`;
  lazy-import + actionable install hint). **Now the operator-scale / fallback path + the
  `bronze.spadl_actions` xT-corpus source** (IDSSE no longer needs it — IDSSE is public on pining).
  Runs silly-kicks' own converters so calibration reflects *current* silly-kicks output.

**Reproducibility split (no committed local paths, nothing GS-derived/internal committed):**
- Committed reproducible surface = synthetic CI fixtures (anyone runs) + a pining-**public** e2e
  smoke — now **SkillCorner AND IDSSE** are fully reproducible via the public token (only GS is
  owner-tier). Default CI e2e stays SkillCorner (light); IDSSE's 419 MB tracking makes its e2e a
  heavier opt-in.
- Full internal sweep = pining for all three (SkillCorner + IDSSE public, GS owner) + Databricks
  for the xT corpus, maintainer-run; output = recommended params + `render_summary_md`/`render_json`
  report.

## 3. Stage 1 — `CarrierAccuracyObjective`

`Objective`, direction **maximize**, metric `"carrier_accuracy"`. Constructed with the loaded
fold (per-`(provider, match)` frames + actions). `evaluate(candidate)`:
1. per match, `infer_ball_carrier(frames, tolerance_m=…, beta=…, gamma=…)` (silly-kicks);
2. link on-ball actions (pass/cross/shot/dribble/ball-carry — actor == carrier by definition) to
   their frame via `link_actions_to_frames`;
3. accuracy = `mean(inferred_carrier_player_id == action.player_id)` per provider, then
   **equal-weight mean across providers** (prevents match-count imbalance dominance);
4. return `{"carrier_accuracy": …, per-provider attrs}`.

Search space (`_spaces.py`, from handoff): `tolerance_m FloatRange(1.0, 8.0)/3.0`,
`beta FloatRange(0.0, 2.0)/0.5`, `gamma FloatRange(0.0, 3.0)/1.0`. `warm_start` = current
defaults (enqueued as trial 0). `OptunaConfig(kind="optuna", metric="carrier_accuracy",
direction=MAXIMIZE, sampler="tpe", n_trials, param_space, warm_start, store=SQLite)`.

## 4. Stage 2 — `AugmentedVaepBrierObjective` (CachedObjective)

The model is **disposable** — Optuna's objective only, never a deliverable.

- `patch_params = frozenset({"k3", "pre_seconds", "min_displacement_m"})`. `OptunaStrategy` asserts
  `param_space ⊆ patch_params` at construction (a tuned non-patch param would reuse the stale
  invariant → silent wrong score).
- **`prepare()`** (the trial-independent invariant, once): per match — the trial-independent
  enrichment steps, links, `vaep.labels.scores(actions, 10)` + `concedes(actions, 10)` labels,
  and the CV fold map. The base feature matrix has the trial-varying columns
  (`pressure_on_actor__link_zones`, off-ball-**run** cols) as NaN placeholders; the line-break
  columns are **NOT model features** (verified against the prior monolith's proven feature set —
  see §4a), so the only trial-varying outputs are the link_zones pressure column + the 4 off-ball-RUN
  columns; M2/R6 reduce to "the patch uses `add_off_ball_runs`, not the umbrella," and line-break is
  not computed at all. **xT is a FROZEN EXOGENOUS artifact, fit once on a corpus DISJOINT from the
  calibration matches (C2, resolved Option 1):** the leak is removed not by per-fold refitting but by
  fitting `ExpectedThreat` on `bronze.spadl_actions` with the tracking-calibration `match_id`s
  EXCLUDED (zero-overlap asserted), freezing it as a versioned sha256'd artifact, and using that one
  grid everywhere. Rationale (decisive): **train–serve consistency** — at deployment the calibrated
  `k3`/`pre_seconds`/`min_displacement_m` defaults run against a *fixed* league-level xT, never a
  per-match refit; calibrate under the serving regime. xT is a fixed upstream feature *extractor*
  (like a frozen pretrained embedding), not part of the model under calibration, so fit-once-on-
  disjoint-data-and-freeze is the correct idiom — not per-fold refit. Consequence: xT is **trial- AND
  fold-independent**, so `add_gk_influence`/`add_cover_shadows` (the only 9 xT-consuming features)
  compute ONCE per match (the prior's simple per-match invariant structure, minus the leak) — no
  per-fold base matrices. Stage-1 carrier optimum is fixed input (feeds `derive_team_in_possession`).
  **Lazy-per-match + `del frames` is the DEFAULT (L6)** — an optional on-disk invariant cache dir
  bounds RAM further; the eager all-in-memory path is opt-in. (The prior run OOM'd at ~9 GB; the
  safe path is the default, not an option.) **The exact invariant-vs-patch step boundary is
  enumerated in the plan (L5)** — derived from the prior monolith's 16-step `_enrich_match` minus
  the 2 trial-varying steps; the enumeration is the core CachedObjective correctness claim and is
  auditable/kept-current there.
- **`evaluate_patch(invariant, candidate)`** (cheap, per trial): re-run ONLY the **two
  trial-varying steps** — `add_pressure_on_actor(methods=("link_zones",),
  params_per_method={"link_zones": LinkParams(k3=…)})` (the pressure column) and
  **`add_off_ball_runs(…, home_team_id, pre_seconds=…, min_displacement_m=…)`** (the off-ball-RUN
  columns only — NOT the `add_off_ball_context` umbrella; line-break is not a feature so it is never
  computed; M2/R6) — overwrite exactly those columns, then train
  **two** disposable XGBoost classifiers per CV fold — one for `scores`, one for `concedes` —
  held-out Brier = `mean(scores_brier, concedes_brier)` per provider. Return
  `{"brier": equal-weight provider mean, per-provider Brier AND per-provider CV SE attrs (M1)}`.
  (Prior run: ~7× speedup.) **XGBoost determinism is a first-class constraint (C1):**
  `random_state=<fixed>`, `n_jobs=1` (multi-thread histogram construction is non-deterministic),
  `tree_method="hist"` (named explicitly, not left to the version default — R4), no unseeded
  `subsample`/`colsample` — so identical features → identical Brier to 1e-9 (see §7).
- **`evaluate()`** (full recompute, the `Objective` port): an **INDEPENDENT monolithic recompute**
  (`enrich_full`) that runs ALL 16 steps with the trial params applied **inline at their natural
  positions** (4b/9 before steps 11-15), **no NaN placeholders, no cached base** (lakehouse H1).
  This is deliberately NOT `prepare()`+`evaluate_patch` reused — that would make the equivalence
  test tautological. Because `evaluate_patch` computes steps 11-15 (gk-influence/cover-shadows/
  team-shape/DAS) with the trial columns as NaN placeholders while `evaluate` computes them after
  the real trial values exist, `ruthless.testing.assert_cache_equivalence` (compares at
  `rel_tol=1e-9`, `abs_tol=1e-12`) now genuinely PROVES the invariant/patch decomposition is correct
  — it would CATCH a "trial-independent" step that secretly reads a trial-varying column (a
  silent-wrong-score bug). The deterministic XGBoost (C1) is what lets the two independent paths
  match to 1e-9.

CV (match-stratified, applying the single >7 threshold above to **bronze counts verified
2026-05-29**): Gradient Sports (**64** matches) and SkillCorner (**10** matches) →
`GroupKFold(5, groups=match_id)`; IDSSE (**7** matches, ≤7) → leave-one-match-out. (R2: this
resolves the earlier contradiction — SkillCorner at 10 matches is comfortably above the threshold
and gets GroupKFold(5), so only IDSSE is LOMO; the loader logs the count+scheme so the assignment
tracks the data if counts change.) Objective = `mean(idsse_brier, skillcorner_brier,
gradientsports_brier)` (equal provider weight, so match-count doesn't let GS dominate). **The
objective ALSO returns per-provider CV standard error (M1):** the LOMO provider (IDSSE) yields a
noisier estimate than the GroupKFold(5) providers (GS, SkillCorner), so surfacing per-provider SE
in the metrics (not just diagnostics) makes it visible when a small/noisy provider is driving the
TPE argmax. **Constant-source caveat (C2/C3):** xT is a frozen exogenous artifact (fit on a
disjoint corpus, §4 above) → **zero leak** into any fold, and it injects no fold-structure variance
into the metric (cleaner TPE signal than per-fold xT would give). The Stage-1 carrier params are
calibrated globally then held fixed for Stage 2 — this is **not** label leakage (Stage-1's objective
is carrier accuracy, independent of VAEP labels) and, being fixed across all Stage-2 trials, cannot
bias the param-selection argmax; it can only shift the *absolute* Brier level, not the ranking. The
report presents the held-out Brier as a clean number and the carrier-param fixity + frozen-xT
identity as documented, ranking-neutral assumptions.

### §4a. Feature set + xT artifact (grounded against the prior monolith)

The Stage-2 augmented feature matrix is the prior monolith's proven `ALL_FEATURES` =
`_SPADL_FEATURES` (base VAEP/SPADL columns) + `_TRACKING_FEATURES` (the ~45 tracking columns:
pressure×3, pitch-control×3, defensive-line, team-shape×12, DAS×3, gk-influence×4, cover-shadows×5,
sync×3, actor-pre-window×2, action-context×4, off-ball-run×4). The plan enumerates it verbatim.
**Line-break is NOT in this set** (neither threshold nor ward) — so it is never computed (resolves
M2/R6 trivially: the patch is just `add_pressure_on_actor(link_zones)` + `add_off_ball_runs`).

**xT artifact (C2 Option 1).** Of `ALL_FEATURES`, exactly **9** consume xT: `gk_influence` (4) +
`cover_shadows` (5) (verified — `off_ball_xt_*` / `add_gk_distribution_metrics` are NOT in the set).
xT is fit ONCE on `bronze.spadl_actions` with the calibration `match_id`s excluded (the fit **fails
CLOSED** if any calibration id is absent from the corpus — H2: an id-space mismatch, pining match_id
vs bronze game_id, would otherwise no-op the exclusion and silently leak; `n_excluded` must equal the
calibration-match count and is recorded in the manifest), frozen as a versioned npz + sha256,
and passed to `add_gk_influence`/`add_cover_shadows` everywhere. Because it is trial- AND
fold-independent, those 9 features compute **once per match** in `prepare()` — single base matrix
per match (the prior's structure), no per-fold variants. The `del frames` lazy-per-match default
(L6) keeps `prepare()` under the prior ~9 GB OOM. socceraction's bundled grid is the acceptable
fallback if the disjoint-corpus fit is unavailable, but is less transparent (un-recordable corpus).

Search space: `k3 FloatRange(0.1, 5.0, log=True)/1.0`, `pre_seconds FloatRange(0.5, 5.0)/1.5`,
`min_displacement_m FloatRange(1.0, 8.0)/3.0`. **Saturation note:** the prior run pinned
`pre_seconds≈4.7–5.0` and `min_displacement_m≈7.0` to the upper edge — if that recurs after a
clean run, widen the upper bounds and rerun. `warm_start` = current defaults; `store` = SQLite
(resumable across the multi-hour run); `direction=MINIMIZE`.

## 5. Gates

- **H1 degenerate-feature gate:** per trial, if any tuned feature's variance < 10% of its
  default-param variance, return a finite, deliberately-bad penalty Brier — `penalty_metrics(
  "brier", Direction.MINIMIZE, magnitude=…)` (NOT `1e9`, NOT `TrialPruned` — a finite bad value
  steers TPE). **Magnitude rationale (M3 + R1):** VAEP `scores`/`concedes` are rare events (~3–5%
  base rate), so achievable Brier is ~0.03–0.06 and the *informed* baseline (predict the base rate)
  is ~0.03 — a p=0.5 "random-guess" 0.25 is **not** a meaningful floor here, it merely happens to
  be ~5–8× any real trial. To stay robustly "bad" if the absolute Brier scale shifts (e.g. a
  library bump) **without making the objective stateful**, the penalty is anchored to the
  **default-param held-out Brier computed ONCE in `prepare()`**: `penalty = k × default_param_brier`
  (k≈5 → ~0.15–0.30, clearly ~5× any real trial). The anchor is computed with the **same
  equal-provider `mean(scores, concedes)` CV logic as a real trial** (M2) — not scores-only — so the
  "~5× any real trial" ratio actually holds (the two labels have different base rates). This is **trial-independent and stateless** — it
  persists implicitly with the cached invariant, is identical across resume (no per-instance
  "worst-observed-this-study" state that resets on restart and would score the same degenerate
  candidate differently before vs after a crash — R1), gives TPE a consistent signal for the
  degenerate region, and is path-comparable so `assert_cache_equivalence` can check it. The earlier
  `2 × worst-observed` form is **dropped** (stateful, order-dependent, untestable for equivalence).
  Default-param variances + default-param Brier are both computed once in `prepare()`.
- **Per-provider signal sanity gate:** a provider contributing ~0 matched carrier events (the old
  GS=0.0 failure mode — now fixed by PR-A, but guard regardless) is flagged + excluded with a
  loud warning, never silently averaged in. **This gate fires at LOAD / `prepare()` time, not
  per-trial (R7):** it is data-determined and stable across the whole study, so the equal-weight
  denominator (number of contributing providers) is fixed for the entire run and the objective
  scale never shifts mid-study.
- **TF-25 diagnostics gate (`--stage diagnostics`):** per-provider re-eval at the global optimum;
  k3 1-D sensitivity (20 log-spaced steps, only `add_pressure_on_actor` re-runs); the principled
  TF-25 trigger `gap = Brier(global k3) − Brier(provider-best k3) > that provider's CV SE` ⇒
  recommend provider-specific defaults (TF-25). **The SE is computed against the scheme each
  provider actually uses (M1 + R2)** — GroupKFold(5) SE for Gradient Sports **and SkillCorner**
  (both >7 matches), **leave-one-match-out SE for IDSSE** (the only ≤7 provider) — NOT a blanket
  "5-fold SE" (IDSSE has no 5-fold SE).
  Plus a geometry sensitivity scan (informational) + feature-importance optimum-vs-default (H1
  defense-in-depth).

## 6. ruthless integration (verified against ruthless 0.2.0)

Each stage = `OptunaStrategy(OptunaConfig(...), seed=…).run(objective, backend=InProcessBackend())`
→ `Result(best, history, diagnostics, provenance)`. Two studies, orchestrated by the CLI
(ruthless has no "stage" concept). `_spaces.py` uses `FloatRange(kind="float", lo, hi, log=…)`;
`OptunaConfig.warm_start` (forced trial 0), `.store=StoreConfig(path=…)` (SQLite resume).
`Direction` is imported from the **public** surface (`from ruthless import Direction` — it is in
`ruthless.__all__`), not a private path (L8). Report via `render_summary_md`/`render_json`.
**Report = data + version manifest, not just params (R3):** ruthless's `Result.provenance` carries
seed / storage / direction but **not data identity**, so the CLI augments the rendered report with
a calibration manifest — silly-kicks version, ruthless version, xgboost version, git SHA,
`n_trials`, seed, source (`pining`/`databricks`), the **per-provider match-ID list** actually used,
and the **frozen xT artifact identity** (grid source, corpus match-ID set with zero-overlap
assertion, fit date, grid shape, sha256). This is the trust anchor for the downstream "apply" PR's
claim ("Optuna-calibrated against &lt;fold&gt; on &lt;date&gt;") — the recommended defaults stay
re-attributable months later, xT surface included.
**Resume semantics (L7):** Optuna does NOT persist sampler RNG across resume (per ruthless's
`OptunaStrategy` docstring), so a resumed multi-hour sweep is no-lost/no-duplicate-trials +
convergent but **not trajectory-identical** — fine for calibration, flagged so resume isn't
surprising. The differential goldens `tc3_stage*.db` (lakehouse) are a **manual one-time**
reproduction check, NOT a CI gate (silly-kicks-dependent Brier drifts with library bumps).

## 7. Testing (TDD; CI = no real data)

- Synthetic multi-provider fixture (tiny matches per provider; reuse tracking + SPADL synth helpers).
- **`assert_cache_equivalence(objective, candidates)`** — the load-bearing Stage-2 correctness
  gate (full `evaluate` ≡ `prepare()`+`evaluate_patch`, compared at `rel_tol=1e-9`). CI-friendly.
  **Two requirements the fixture must meet:** (1) the candidate set must vary **all three** patch
  params (`k3`, `pre_seconds`, `min_displacement_m`) across ≥2 distinct values each — ruthless's
  `assert_cache_equivalence` *raises* otherwise (L4); (2) the equivalence holds only with
  **deterministic XGBoost** (`random_state` fixed, `n_jobs=1`, deterministic `tree_method`, no
  unseeded subsample/colsample — C1), so identical feature matrices give identical Brier to 1e-9.
  Tiny shallow-tree CI fixtures make single-thread+seed pass deterministically and cheaply.
- Unit: Stage-1 accuracy on a known-carrier fixture; equal-weight per-provider averaging; H1
  penalty fires on a degenerate feature **and returns the identical default-Brier-anchored value
  via both `evaluate` and `evaluate_patch`** (R1 — the stateless penalty is path-comparable; the
  dropped stateful form could not be); sanity gate excludes a 0-signal provider; CV splitter
  GroupKFold-vs-LOMO selection at the >7 boundary (7→LOMO, 8→GroupKFold) + no match leakage; config builders (`warm_start ⊆ param_space`;
  `OptunaStrategy` raises on `param_space ⊄ patch_params`); `n_trials=2` Optuna smoke returns a
  `Result` with `best`.
- Loaders: pining loader against GS/SkillCorner schemas (monkeypatched fetch); Databricks loader
  with a stubbed cursor; **optional pining-public e2e smoke** (SkillCorner, `@pytest.mark.e2e`,
  reproducible).
- `[calibration]` deps added to `[test]` so CI exercises `ruthless[optuna]` + `xgboost`.

## 8. Housekeeping
- **Version:** minor → **3.28.0** (collision-check at bump time; main is 3.27.0).
- **Docs:** CHANGELOG `### Added`; a `scripts/` calibration README/walkthrough; no README/CLAUDE
  drift beyond noting the optional extra.
- **No new ADR** — within ADR-004 (tracking namespace) + the established optional-dep pattern.
  ruthless usage + Optuna/TPE noted; NOTICE already carries the pressure/off-ball/VAEP citations.
- **Licence/reproducibility:** no GS-derived/internal data committed; committed e2e on pining-public.

## 9. Out of scope (→ follow-ups)
- Changing the three default constants (the "apply" PR, after the maintainer's real sweep).
- ~~IDSSE→pining~~ — **DONE** (IDSSE is public on pining, verified 2026-05-29; the pining loader serves it directly).
- TF-25 provider-specific defaults (only if the diagnostics gate fires).
- The actual multi-hour sweep (maintainer-run, not committed).
- evolve strategy / compute backends (TF-24 is Optuna, local, in-process).
