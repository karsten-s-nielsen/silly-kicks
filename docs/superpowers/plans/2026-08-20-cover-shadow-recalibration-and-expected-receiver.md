# Cover-shadow σ/λ discrimination re-tuning + expected-receiver model — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (or subagent-driven-development). Steps use `- [ ]` checkboxes.

**Goal:** Ship a public expected-receiver model that de-leaks the RQ1 failed-pass target, then re-tune cover-shadow σ/λ for discrimination against that de-leaked signal and apply it **per-provider** only if it clears pre-registered receiver-validity + bias + noise gates — exploratory, honest-null the likely landing.

**Architecture:** Receiver model (ADR-011 bundle, trained on SB360 open-data, pre-pass features only) + a trajectory-weak-labelled failed-pass validation set → de-leaked failure-mode-conditional RQ harness → a TF-24 σ/λ discrimination objective → a three-conjunct, per-provider `for_provider` apply. All corpus work behind `scripts/` drivers (ADR-052 `for_each`, ADR-037 provenance).

**Tech stack:** pandas/numpy/scipy/sklearn (+ xgboost for the scorer), pytest/ruff/pyright. Spec: `docs/superpowers/specs/2026-08-20-cover-shadow-recalibration-and-expected-receiver-design.md`.

## Global Constraints

- **Leakage:** the receiver feature extractor consumes ONLY pre-pass state + release kinematics — NEVER the end/loss location. Enforced by an output-invariance test (Task 2).
- **Exploratory / honest-null:** keeping the incumbent σ/λ (and shipping the geometric proxy) is a first-class, acceptable outcome. No step assumes σ/λ moves.
- **Pre-registered thresholds:** `MIN_COVERAGE=0.30`, `MIN_RECEIVER_MARGIN=0.05`, `MAX_BIAS_SHARE=0.50` + ADR-060 `MIN_EFFECT_SIZE`/`exceeds_noise_floor` — named constants in one module, referenced not literal; fixable only before the run, with a reason.
- **Per-provider:** σ/λ apply is via `CoverShadowParams.for_provider` (mirrors `PreprocessConfig.for_provider`); a GS re-tune never changes another provider's resolved σ/λ.
- **Bundle:** ADR-011 (numpy/JSON weights + `SHA256SUMS` + chirality probe + `_feature_contract`; pickle-free, sklearn-free inference). Public corpus only for the bundled default (ADR-038 fail-closed visibility).
- **Provenance:** every `scripts/` artifact driver calls `require_clean_tree` in `main()` + stamps `run_commit`/`run_tree_dirty`; corpus passes use `for_each`; weights carry `metrics.json` provenance → **non-squash merge**.
- **Lint at CI scope** (`python -m ruff check silly_kicks/ tests/ scripts/`), bare pyright; run `-m "not e2e"`.

---

### Task 1: Receiver feature extractor + geometric proxy (pure, pre-pass only)

**Files:**
- Create: `silly_kicks/tracking/_receiver.py`
- Test: `tests/tracking/test_receiver_features.py`

**Interfaces:**
- Produces: `receiver_candidate_features(action_row, frame, *, params, feature_set="public") -> pd.DataFrame` (one row per passing-team teammate in the release frame). **`feature_set="public"` (bundled default) = POSITIONS ONLY:** `ball_dist` (passer→candidate), `lane_pressure` (defenders in the passer→candidate corridor), `space` (candidate's nearest-defender distance) — all pre-pass positional facts, leakage-free on velocity-less SB360 freeze frames. **`feature_set="owner"` ADDS the velocity-derived features** `release_dir_align` (candidate bearing vs the **ball's release-velocity** direction from the frame ball `vx/vy` — the only leakage-free release direction; the pass-event angle is origin→end, banned) + `closing_speed` (candidate velocity · unit passer→candidate ray). The owner set reads `vx/vy` and RAISES if asked for on a frame set without them. `geometric_proxy_receiver(action_row, frame, *, params) -> candidate_id` (nearest teammate to the **ball-release-velocity ray**; requires ball velocity → GS-only; the must-beat baseline + fallback, NOT a feature). `ReceiverParams` frozen dataclass with the feature-set column lists.

- [ ] **Step 1: Write the failing test** — a synthetic release frame (passer + 3 teammates + 2 defenders + a ball row with a release velocity, known geometry): assert `feature_set="public"` returns exactly `{ball_dist, lane_pressure, space}` (**no velocity feature**), `lane_pressure` is higher for the corridor with the interposed defender, `space` is smallest for the marked teammate; assert `feature_set="owner"` additionally returns `{release_dir_align, closing_speed}` with `release_dir_align` maximal for the teammate on the ball-velocity ray; assert `geometric_proxy_receiver` returns the on-ray teammate and RAISES on a velocity-less frame.
- [ ] **Step 2: Run it, verify it fails** (`ImportError`).
- [ ] **Step 3: Implement** the extractor + proxy over `id_compat` for team membership. The `public` list is velocity-free **by construction** (not a runtime `vx/vy` degrade — a velocity feature must never silently become a constant on the corpus it trains on, H1); `owner` reads the ball row's `vx/vy` for the release direction + candidate `vx/vy` for closing speed, raising if absent.
- [ ] **Step 4: Run, verify pass.** Add a purity check (input frame unmutated).

### Task 2: Leakage guard (output-invariance) — the load-bearing constraint

**Files:** Test: `tests/tracking/test_receiver_leakage_guard.py`

- [ ] **Step 1: Write the failing test** — build an action+frame, compute `receiver_candidate_features`; then **perturb the action's `end_x`/`end_y`** to arbitrary values and recompute; assert the two feature frames are **byte-identical** (`pd.testing.assert_frame_equal`). This proves the extractor cannot read the outcome-selected end location. (Pin ONE behavior — invariance — not "raises or ignores".)
- [ ] **Step 2: Run** — it passes iff Task 1 truly ignores `end_*`; if it fails, the extractor is reading the end location — fix Task 1, not the test.
- [ ] **Step 3:** add the reverse control — perturbing a *pre-pass* input (a teammate position) DOES change the features (proves the test isn't vacuously invariant to everything).

### Task 3: Receiver scorer (per-candidate binary) + ADR-011 bundle

**Files:**
- Create: `silly_kicks/tracking/_receiver.py` (extend), `silly_kicks/tracking/_receiver_weights/` (bundle dir), `silly_kicks/tracking/_receiver_contract.py`
- Test: `tests/tracking/test_receiver_model.py`

**Interfaces:**
- Produces: `class ReceiverModel` with `predict_candidates(features) -> np.ndarray` (per-row P(intended)), `rank(action, frame) -> pd.Series` (candidate_id → prob, full distribution), `load(dir)`/`save(dir)`, `variant_key_for_provider(provider) -> str`. Inference is pure-numpy `sigmoid(Xb)` or booster-JSON; NO sklearn import at serve.

- [ ] **Step 1: Write the failing test** — fit on a tiny synthetic completed-pass set (positive = observed receiver, negatives = other teammates); assert `rank` puts the observed receiver first on held-out synthetic rows; assert `save`→`load` round-trips and the loaded model's `predict_candidates` matches bit-for-bit; assert `load` raises the model's `IntegrityError` on a chirality-fingerprint mismatch and on a `_feature_contract` declared-constant mismatch.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** the scorer (xgboost booster, `load_xgb_booster_base_score_safe`), the npz/JSON + `SHA256SUMS` bundle, `_chirality.verify_chirality` on `canonical_probe_frame`, and `_receiver_contract._feature_contract_block()`. The bundle records its `feature_set` and the contract declares **exactly the columns that variant uses** — the public bundle declares the four position/direction features and NOT `closing_speed` (H1: no declared-but-unused velocity feature). Serve path imports no sklearn (guard test).
- [ ] **Step 4: Run, verify pass.**

### Task 4: Trajectory-weak-labelled failed-pass validation set (H1/R1/R4)

**Files:**
- Create: `scripts/_receiver_validation.py`
- Test: `tests/scripts/test_receiver_validation.py`

**Interfaces:**
- Produces: `trajectory_weak_labels(actions, frames) -> pd.DataFrame` (one row per *intercepted* failure whose ball travelled ≥ `min_travel_m` toward a teammate before the cut-out; columns: `action_id`, `weak_receiver_id` (nearest teammate to the **forward-projected** meeting point = ball path × teammate run, R4-mitigated), `label_confidence`, plus a `covered` bool over all intercepted failures). `receiver_failed_pass_accuracy(model, actions, frames) -> dict` with `top1`, `top1_proxy`, `coverage`, `n_covered`, and the R1/R4 caveat strings.

- [ ] **Step 1: Write the failing test** — synthetic intercepted failures: one with a clear straight trajectory to a teammate (labelled, `covered=True`), one foot-blocked with no travel (`covered=False`), one leading pass where the receiver runs onto the ball (label uses the projected meeting point, not the endpoint). Assert coverage counts, that the clear case labels the correct teammate, and that `receiver_failed_pass_accuracy` reports `coverage` and both `top1` and `top1_proxy`.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement.** Failure = next SPADL action is an opponent interception/tackle (Task 8 helper). Weak label = nearest teammate to the forward-projected meeting point; `covered` requires `min_travel_m` and an unambiguous nearest teammate. The returned dict carries the fixed caveat text: (a) accuracy is an **upper bound on the easy tail** (R1); (b) the label mislabels through-balls (R4); (c) **this validates a DOUBLE transfer** — SB360-completed → GS-failed (M3) — and only on the easy tail, so a conjunct-1 pass licenses **neither** the completed→failed nor the SB360→GS transfer on the hard subset; (d) the SB360-train vs GS-serve **candidate-count gap** (M2, from Task 6/8 metrics) is a ranking-distribution shift the accuracy number does not correct for.
- [ ] **Step 4: Run, verify pass.**

### Task 5: Public receiver surface + wiring

**Files:** Modify `silly_kicks/tracking/__init__.py`, `silly_kicks/tracking/features.py`; Test: `tests/tracking/test_receiver_public_api.py`

**Interfaces:**
- Produces: `resolve_intended_receiver(actions, frames, *, model=None) -> pd.Series` (`player_id`; `model=None` → `geometric_proxy_receiver`), `intended_receiver_positions(actions, frames, *, model=None) -> pd.DataFrame` (`action_id`, `x`, `y`, `source ∈ {intended_receiver, geometric_proxy}`).

- [ ] **Step 1: Write the failing test** — `resolve_intended_receiver(..., model=None)` returns the proxy pick; with a fitted model returns its argmax; `intended_receiver_positions` returns frame coords + `source`. NaN-safe on missing ids (ADR-003) and pure (ADR-033) — register both.
- [ ] **Step 2–4:** fail → implement (`@nan_safe_enrichment` where applicable) → pass. Update `id_compat` scalar registry / purity registry / glossary as required by the meta-assertions.

### Task 6: SB360 training driver for the public receiver variant (D1)

**Files:**
- Create: `scripts/train_receiver_model.py`
- Test: `tests/scripts/test_train_receiver_model.py`

- [ ] **Step 1: Write the failing test** — monkeypatch `load_statsbomb_matches` to yield 2 tiny SB360-shaped matches (freeze frames w/ real team ids per ADR-062, no velocity); run `main()` with `--out` tmp + `--allow-dirty`; assert it writes a loadable `ReceiverModel` bundle + `metrics.json` with `run_commit`, held-out `top1`, and `corpus_visibility` from `artifact_label` (public); assert `--help` exits 0.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — `require_clean_tree` first; `for_each`-shard candidate rows per match (`scripts/_sb_raw` + `load_statsbomb_matches`, ADR-062); positives = `resolve_next_touch_receiver`; match-stratified `GroupKFold`; `feature_set="public"`; fit; save bundle; stamp provenance; ship-mask public.
- [ ] **Step 3b (M2): record the candidate-set shift** — write the per-frame **candidate-count distribution** (mean/percentiles of teammates present) to `metrics.json`, because SB360's visible area truncates the *negative* candidate set (the model ranks among visible teammates in training but among all teammates at GS serve — a distribution shift on its own output, orthogonal to the positive being visible). The GS serve-side candidate-count distribution is recorded by Task 8's driver; the RQ1 artifact (Task 13) carries the train-vs-serve gap as a caveat. Visible-area limitation also noted.
- [ ] **Step 4: Run, verify pass** (`-m slow` — train smoke, interpreter-invariant).

### Task 6b: GS velocity ablation (COMPLETED passes) + owner-variant deployment gate (FAILED subset)

**Files:** Modify `scripts/train_receiver_model.py` (`--feature-set {public,owner} --provider gradientsports`); Test: `tests/scripts/test_receiver_velocity_ablation.py`, `tests/scripts/test_receiver_deployment_gate.py`

**M-A resolution — two DISTINCT measurements on their correct populations** (the velocity question is measurable on completed passes; the deployment question is intrinsically a failed-pass question). Both are distinct from Task 11's lane-pressure/bias ablation. Requires TWO GS variants: GS-`positions-only` and GS-`positions+velocity` (`owner`), trained via `--feature-set`.

**(i) Velocity ablation — held-out COMPLETED passes (full ground truth, NOT the easy tail).** Isolates velocity cleanly and where it is actually measurable: on completed passes the observed receiver is ground truth for *every* pass, easy and hard alike, so velocity's value is not measured only where positions already resolve it (the R1 trap that a failed-easy-tail ablation would fall into).

- [ ] **Step 1a: Write the failing test** — `velocity_ablation_completed(pos_model, posvel_model, actions, frames)` reports held-out-completed top-1 for GS-`positions-only` vs GS-`positions+velocity` + `velocity_delta`; on synthetic completed passes where a teammate runs onto the ball (velocity-separable) `velocity_delta > 0`; on position-separable completions `velocity_delta ≈ 0`. Carries the caveat: measured on COMPLETED as the best proxy for velocity's failed-pass value (the completed→failed transfer, H1, is the whole model's caveat, not velocity's alone).
- [ ] **Step 1b–2:** fail → implement (train both GS variants via `--feature-set`; held-out-completed `GroupKFold` eval).

**(ii) Deployment gate — FAILED-pass validated subset (which variant to serve on GS-failed).** SB360-`public` vs the GS-`owner` (positions+velocity) variant on the Task 4 trajectory-validated failed subset — the bundling decision.

- [ ] **Step 3:** write + pass a test — `deployment_gate(public_model, gs_owner_model, actions, frames)` returns `decisive` iff GS-owner beats public by ≥ `MIN_RECEIVER_MARGIN` on the validated subset, with the R1 caveat that a NON-decisive result reads *"the GS variant's advantage is unmeasurable on the easy tail,"* NOT "GS-native/velocity adds nothing" — the velocity ablation (i) is the un-easy-tail-limited read of that.
- [ ] **Step 4 (bundle):** bundle the GS-owner variant iff the deployment gate is `decisive` (variant-keyed via `variant_key_for_provider`); else keep serving the public model (honest null). Record BOTH the velocity ablation (i) `velocity_delta` and the deployment verdict in `metrics.json` (+ TODO follow-up if non-decisive). GS-owner corpus is owner-tier (ship-mask; `stores_training_data:false`).
- [ ] **Step 5 (L-B — model-selection flow):** `--receiver-model` (Task 8/14) + the objective (Task 11) resolve the served model through `variant_key_for_provider("gradientsports")` — a **decisive** owner variant feeds the de-leak AND the σ/λ objective on GS; a **non-decisive** run keeps the public model (never a stale mismatch). A test pins the resolution.

### Task 7: `CoverShadowParams.for_provider` (H3 — per-provider σ/λ)

**Files:** Modify `silly_kicks/tracking/_cover_shadows.py`; Test: `tests/tracking/test_cover_shadow_for_provider.py`

**Interfaces:**
- Produces: `CoverShadowParams.for_provider(provider: str) -> CoverShadowParams` (returns per-provider σ/λ from a module-level `_PROVIDER_PARAMS` map, else the incumbent default). Additive — the default `CoverShadowParams()` is byte-identical.

- [ ] **Step 1: Write the failing test** — `for_provider("gradientsports")` and `for_provider("unknown")` both return the incumbent σ/λ today (empty map); assert `CoverShadowParams()` unchanged; assert the map is the ONLY mutation point (a later GS entry changes GS and nothing else). Mirror `PreprocessConfig.for_provider`.
- [ ] **Step 2–4:** fail → implement (frozen dataclass classmethod + `_PROVIDER_PARAMS` dict) → pass.

### Task 8: De-leaked, failure-mode-conditional RQ harness (M2) + shard bump

**Files:** Modify `scripts/_rq_corpus.py`, `scripts/build_rq_pass_scores.py`; Test: `tests/scripts/test_rq_corpus.py`, `tests/scripts/test_build_rq_pass_scores.py`

**Interfaces:**
- Produces: `classify_failure_mode(actions) -> pd.Series` (`{intercepted, out, other}` from the next SPADL action); `extract_played_passes` failed-pass leg: intercepted → `intended_receiver_positions(model=)`, out → trajectory-informed target, other → dropped-and-counted. `target_source` vocabulary `{receiver, intended_receiver, trajectory, geometric_proxy, end_xy_legacy}`. Shard schema `rq-scores-3` (+ `n_blocked` margin cols from 4.87.0 retained; + `failure_mode`, `target_source`).

- [ ] **Step 1: Write the failing test** — an intercepted failure routes to the receiver target; an out-of-play failure routes to the trajectory target; a completed pass keeps the observed receiver; `_EMITTED_SHARD_COLUMNS` matches the dict `score_match` actually builds (compare keys, NOT `pd.DataFrame(rows, columns=...)`); `_SHARD_SCHEMA_VERSION == "rq-scores-3"` and `token_inputs["schema"]` references the constant.
- [ ] **Step 2–4:** fail → implement (thread `model=` through; failure routing; bump schema) → pass. `classify_failure_mode` reads the next SPADL action after sorting via **`_sort_actions_chronological_or_action_id`** (4.89.0 — robust to a non-chronological mart). Record the **GS serve-side candidate-count distribution** in the shard/manifest (M2: pairs with Task 6's SB360-train distribution so the shift is measurable). Fix the 4.87.0 line-108 selection to a compare-actual-keys assertion while here (ADR-052).

### Task 9: GS failure-mode tagging reliability check (R6)

**Files:** Create `scripts/_gs_failure_mode_check.py`; Test: `tests/scripts/test_gs_failure_mode_check.py`

- [ ] **Step 1: Write the failing test** — on a fixture with clean next-action tags, `failure_mode_reliability(actions)` returns `ambiguous_rate` below `MAX_AMBIGUOUS_RATE`; on a fixture where failed passes lack a classifiable next action, it returns a high rate + `reliable=False`.
- [ ] **Step 2–4:** fail → implement (fraction of failed passes whose next action is neither a clean interception/tackle nor a clean restart) → pass. The build driver logs it and, if `reliable=False`, records the failure-mode split as unreliable (does NOT mix legs silently).

### Task 10: Pre-registered thresholds module (R5)

**Files:** Create `silly_kicks/calibration/_cover_shadow_thresholds.py` (or `scripts/`-side if calibration-extra-gated); Test: `tests/calibration/test_cover_shadow_thresholds.py`

- [ ] **Step 1: Write the failing test** — the module exposes `MIN_COVERAGE=0.30`, `MIN_RECEIVER_MARGIN=0.05`, `MAX_BIAS_SHARE=0.50`; a test asserts the apply gate (Task 12) *references* these names (AST: not literals), so the bar can't move silently.
- [ ] **Step 2–4:** fail → implement constants + the AST wiring assertion → pass.

### Task 11: σ/λ discrimination re-tuning objective (M1) + ablation

**Files:** Create `silly_kicks/calibration/_cover_shadow_objective.py`; Test: `tests/calibration/test_cover_shadow_objective.py`

**Interfaces:**
- Produces: `CoverShadowDiscriminationObjective(fold, *, receiver_model)` with `evaluate(trial) -> float` (maximize margin-AUC on the **trajectory-validated intercepted subset + completed passes only**; reject if the completed-pass FP rate exceeds incumbent) and `lane_pressure_ablation() -> float` (σ/λ argmax with vs without the lane-pressure feature; returns the shift share). **No `xt`** (Low-2: `lane_control` `_cover_shadows.py:545` takes no `xt`, so `p_blocked − p_received` is xt-free; adding it would be a frozen-model provenance surface for nothing). **Out-of-play failures are deliberately NOT in the objective** (Low-1: they are empty-space, low-`p_blocked` by construction — not a blocking phenomenon; σ/λ is a blocking model, so its discrimination class is completed vs *blocked* failures). Manifest carries M1 (shape-not-magnitude), R3 (residual bias), the out-failure exclusion, attempted-/model-conditional caveats.

- [ ] **Step 1: Write the failing test** — on a synthetic fold, the objective RESPONDS to σ/λ (non-vacuity: two σ/λ give different AUC); a σ/λ that raises the FP rate above incumbent is rejected (returns the sentinel/penalty); `lane_pressure_ablation` returns a value in [0,1]; the validated-subset restriction is honored (objective ignores uncovered intercepted failures for the primary, reports full-population as a sensitivity field).
- [ ] **Step 2–4:** fail → implement (reuse `_rq_metrics` AUC/FP; **no `xt`**, per the interface; deterministic; cache-equivalent if `CachedObjective`) → pass.

### Task 12: Three-conjunct, per-provider apply (b2) + outcome vocabulary

**Files:** Create `scripts/apply_cover_shadow_retune.py`; Test: `tests/scripts/test_apply_cover_shadow_retune.py`

**Interfaces:**
- Produces: `decide_apply(recommendation, validation, ablation, *, thresholds) -> ApplyOutcome` where `outcome ∈ {applied, null:unvalidatable, null:biased, null:within-noise}` — applies iff coverage≥`MIN_COVERAGE` AND receiver-margin≥`MIN_RECEIVER_MARGIN` AND ablation-share<`MAX_BIAS_SHARE` AND `exceeds_noise_floor`. `main()` writes GS's `for_provider` entry ONLY on `applied`.

- [ ] **Step 1: Write the failing test** — a clears-all case → `applied`; then FOUR cases each failing exactly one conjunct → the matching `null:*` reason (per the CLAUDE.md "gate from both sides" rule); the null cases leave `_PROVIDER_PARAMS` and `CoverShadowParams()` byte-identical; `applied` writes only GS's entry and no other provider's resolved σ/λ changes.
- [ ] **Step 2–4:** fail → implement (reads the Task 10 thresholds; ADR-060 `exceeds_noise_floor`) → pass. `main()` refuses a dirty tree; the constant edit is the only mutation.

### Task 13: Recall re-run + honesty (R1/M3) in the RQ1 artifact

**Files:** Modify `scripts/validate_cover_shadow_rq1.py`; Test: `tests/scripts/test_validate_cover_shadow_rq1.py`

- [ ] **Step 1: Write the failing test** — the metrics dict gains `failed_pass_validity` (top1, top1_proxy, coverage, and the **upper-bound** caveat string), a `robustness_band` (recall under model vs proxy) labelled a same-failure-mode agreement check NOT a validity claim, `residual_bias_note` (R3), a `candidate_count_shift` field (SB360-train vs GS-serve, M2), and a note that **recall spans intercepted + out failures while σ/λ was tuned on intercepted-validated only** (Low-1: out-failures are low-`p_blocked` empty-space, not a blocking phenomenon). Assert the headline FP rate (4.87.0, leakage-free) is unchanged by the additions.
- [ ] **Step 2–4:** fail → implement → pass.

### Task 14: Corpus run wiring + drivers

**Files:** Modify `scripts/build_rq_pass_scores.py` (thread `--receiver-model`); ensure `train_receiver_model.py`, `apply_cover_shadow_retune.py` are ADR-052/037 compliant; Test: `tests/scripts/test_provenance_wiring.py` (enroll the new drivers in `ARTIFACT_DRIVERS`), `tests/scripts/test_ci_*` as needed.

- [ ] **Step 1:** enroll `train_receiver_model`, `apply_cover_shadow_retune` (+ any new artifact driver) in `ARTIFACT_DRIVERS`; assert each imports `require_clean_tree`, offers `--allow-dirty`, never shells `rev-parse`, calls it in `main()`.
- [ ] **Step 2:** the ADR-056 registry-population gates (derive-and-assert-exactly) stay green with the new drivers.

### Task 15: ADR-066 + docs + registries + C4

**Files:** Create `docs/superpowers/adrs/ADR-066-*.md`; modify `CHANGELOG.md`, `TODO.md`, `docs/PRIVATE_CONSUMERS.md`, `NOTICE`, `feature_glossary.py`, C4 model.

- [ ] **Step 1:** write ADR-066 (this design; **amends ADR-009** for the gated in-cycle per-provider apply; records the exploratory/honest-null stance + the R1/R3 residuals). Cross-link ADR-009/011/060/052/037.
- [ ] **Step 2:** `NOTICE` entry (Power et al. 2017 receiver; Cascioli σ/λ); glossary rows for new columns; PRIVATE_CONSUMERS if any `_*` seam is consumed cross-package; C4 — add the receiver module container if it warrants one (mirror xShot/xCross), re-render with **Graphviz dot**.
- [ ] **Step 3:** version to all 5 places at commit-prep only (`pyproject.toml`, `__init__.py`, `uv.lock`, `CHANGELOG` heading, `TODO` Release) — NEXT-FREE **4.90.0 / PR-S160 / ADR-066** (4.89.0/PR-S159/ADR-065 taken by the other session), re-verified against `origin/main` first.

### Task 16: Full-suite + lint + owner corpus run

- [ ] **Step 1:** `python -m pytest tests/ -m "not e2e" --benchmark-skip -q` green; ruff check+format at CI scope; bare pyright 0.
- [ ] **Step 2:** `/final-review`.
- [ ] **Step 3 (owner-run, local pining):** train the public receiver model (SB360) → build RQ scores on GS with `--receiver-model` → run the σ/λ objective + `apply_cover_shadow_retune` → refresh the RQ1 artifact. All `--out` outside the repo; artifacts stamp the clean code SHA (amend code commit before the run if a fix is found, per the 4.87.0 pattern).

## Self-review notes
- **Spec coverage:** receiver model (T1–6), H1 validation (T4), leakage (T2), de-leak (T8), R6 (T9), σ/λ objective + ablation (T11), pre-registration (T10), per-provider apply (T7,T12), recall honesty (T13), corpus/provenance (T6,T14), ADRs/docs (T15). D2/H3/D1 resolved in the spec.
- **Plan review 1 incorporated:** H1 — public `feature_set` is positions+release-direction only, no dead `closing_speed`, contract declares only used columns (T1/T3); M1 — the GS **velocity** ablation is now Task 6b (distinct from T11's lane-pressure bias ablation), owner variant bundled only if decisive per D1; M2 — the SB360→GS candidate-count shift is recorded (T6/T8) and caveated (T4/T13); M3 — the double transfer (SB360-completed → GS-failed) is stated as validated only on the easy tail (T4); Low — `xt` dropped from the objective (T11), out-failure exclusion from σ/λ stated (T11/T13).
- **Plan review 2 incorporated:** L-A — stale `xt` reference removed from T11's implement step; L-B — the `variant_key_for_provider` model-selection flow made explicit (T6b Step 5); **M-A** — Task 6b split into a velocity ablation on **held-out completed passes** (isolates velocity where it's measurable, off the R1 easy tail) + a **deployment gate on the failed subset** (bundling decision, R1 caveat), per the owner's best-practice call.
- **4.89.0 merge:** ADR-065/PR-S159 taken → renumbered to ADR-066, NEXT-FREE 4.90.0/PR-S160; next-touch/failure-mode logic builds on the chronological-`action_id` guarantee + `_sort_actions_chronological_or_action_id` (T8); GS byte-identical (no retrain) but now needs a `start_time` input column (loader supplies).
- **Type consistency:** `ReceiverModel`/`resolve_intended_receiver`/`intended_receiver_positions`/`for_provider`/`decide_apply`/`ApplyOutcome` names are stable across tasks.
- **Honest-null reachable:** T12's four `null:*` outcomes + T13's upper-bound framing make "keep incumbent, ship proxy" a clean, well-labelled landing.
