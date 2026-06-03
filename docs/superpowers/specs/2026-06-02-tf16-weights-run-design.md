# TF-16 xShotOccurrence Weights Run + Ghost-GK 4.7.0 Re-fit — Design

**Date:** 2026-06-02
**Status:** Reviewed (part-deux session, 2026-06-02 — all HIGH/MEDIUM/LOW incorporated) + owner decisions on H1/L7 folded in. Ready for the PR-S80 implementation plan.
**PR:** PR-S80 (xS weights). Ghost-GK re-fit follows as PR-S81 off the same corpus.
**Layer:** GKDV Layer 2 (TF-16 weights follow-up) + TF-18 maintenance re-fit
**Predecessor spec:** `2026-05-31-tf16-xshot-occurrence-design.md` (the *code* spec; shipped 4.1.0 / PR-S75, untrained)
**Provisional version:** 4.8.0 **or 4.9.0 — must be coordinated with the part-deux session** (both branched off 4.7.0; see §11 / review H1). Minor either way — bundled weights + default-xfn wiring are additive behaviour changes.

---

## 1. Purpose & scope

PR-S75 shipped the **complete xS code path** untrained. This cycle executes the ADR-011
"bundled/Hub weights" stage: **train xShotOccurrence on a real multi-provider corpus, bundle it,
publish it, turn it on.** Because the same corpus is staged on the same box, we **opportunistically
re-fit Ghost-GK** against the 4.7.0 carrier defaults in the same session.

### In scope

1. **Train xS** on a pining-first multi-provider corpus, against the **4.7.0 carrier defaults**
   (`tolerance_m=3.0, beta=0.0, gamma=0.25`), via the existing ruthless HPO objective.
2. **Bring `scripts/train_xshot_occurrence.py` up to the Ghost-GK maturity bar:** per-match
   streaming from the loader generator, feature disk-cache, per-provider diagnostics, `metrics.json`,
   acceptance gates, permutation importance, round-trip verify.
3. **Train two candidates (`public` vs `full`) and decide ship-one-vs-two from evidence** (§5);
   bundle the shipped default in the wheel + publish to HuggingFace Hub; wire `from_variant` /
   `from_hub` to actually load.
4. **`scripts/publish_xshot_occurrence.py`** — verify + upload + download-verify (mirrors
   `publish_ghost_gk.py`).
5. **Flip the xS e2e acceptance gates** from skipped placeholders to real assertions (PR-AUC > base
   rate **and** Brier < base-rate Brier; cross-provider non-degradation).
6. **Wire `xshot_occurrence_xfns` into the default xfn list(s).**
7. **Sync xS's stale carrier defaults** to 4.7.0 (a latent code bug — see §4).
8. **Record a coordinate/units metadata template** (`pitch_length`/`pitch_width` + geometry version)
   in the xS artifact — the TF-38 insurance + the template Ghost-GK's R3 follow-up will adopt.
9. **Ghost-GK re-fit** against 4.7.0 defaults, **acceptance-gated / non-regression** (§7) — executed
   on the **same staged corpus** but **shipped as a separate follow-up PR-S81** (release-granularity
   decision L7/§11), so this PR (PR-S80) is xS-weights-only at the commit/version boundary.

### Out of scope

- `extended` feature variant (still `NotImplementedError`; TF-19-time).
- TF-17 / TF-19.
- Ghost-GK's R3 record+consume-carrier-params structural fix — explicit TODO already filed
  (Technical Debt → Blocked or Deferred); this cycle only *re-fits* Ghost-GK, it does not change its
  carrier-param handling.
- Adding ruthless HPO to Ghost-GK (its re-fit keeps the existing fixed recipe — see §7 rationale).

---

## 2. Corpus & compute

**Source — pining-first, bronze-backup** (per owner decision). `scripts/_loader_pining.load_matches`
yields `(provider, match_id, actions, frames, home_team_id)` per match, already
`smooth_frames`→`derive_velocities`'d (so `vx/vy` are present). Owner token (full API access)
enables all three providers: `skillcorner` (public), `idsse` (public Sportec/DFL), `gradientsports`
(owner). `scripts/_loader_databricks` (bronze) is the fallback for operator-scale volume; today its
`_convert` only wires `idsse`/`sportec`, so a bronze pull for skillcorner/gradientsports would need
two more bronze→converter mappings (deferred unless §3 Phase-0 shows pining is too thin).

**Compute — DGX Spark** (`ssh karsten@192.168.68.73`, ARM aarch64, 119 GiB unified RAM; idle, so it
won't contend with interactive sessions on the Windows box). Prereqs on the box: `pip install -e
".[train,xgboost,kloppy]"` (xgboost has aarch64 wheels; ruthless/optuna are pure-Python;
`accessible-space` is **not** needed — xS `faithful` uses no DAS), `PINING_FOR_THE_DATA_TOKEN`
(owner) exported, and an `HF_TOKEN` for the publish step.

**Memory discipline:** the loader is a generator and the IDSSE tracking artifact is ~419 MB XML per
match. The trainer streams **one match at a time**, extracts the small feature matrix, and drops the
frames before the next match — only the accumulated `(X, y, groups, provider)` (small) stays
resident. This is the Ghost-GK `train_ghost_gk.py` streaming pattern.

---

## 3. xS trainer architecture (`scripts/train_xshot_occurrence.py` — rewrite)

Today's script buckets every game under one `"synthetic"` provider key and the objective does plain
`GroupKFold` on `game_id` — fine for a smoke, thin for a maintainer run. The rewrite keeps the
**ruthless HPO core unchanged** (it already *is* the standard tool — the point is to use it, not
hand-roll) and adds the reporting/curation around it.

**Phase 0 — corpus probe (list-only, no download).** Call the pining manifest endpoint
(`_list_matches` per provider) to count available matches per provider *before* committing. Decide:
(a) whether pining volume suffices for xS (it almost always does — shots are rare per frame but a
~1 s horizon at 25 fps makes ~25 positive frames per shot, ≈ hundreds of positives/match), and
(b) whether the Ghost-GK re-fit (which needs *scale* to non-regress vs its 36 k/537 k bundled
variants) should use pining or escalate to bronze (§7). Log the counts; never silently cap.

**Phase 1 — stream + extract + cache.** For each `(provider, match_id, actions, frames, home)`:
- `shots = actions[actions["type_name"].isin({"shot","shot_penalty","shot_freekick"})]`
  (the trainer passes the full `actions`; `prepare_xshot_training_data` filters internally).
- `X, y, groups = prepare_xshot_training_data(frames, shots_or_actions, home_team_id=home,
  horizon_seconds=1.0, attacking_third_only=True, carrier_params=DEFAULT_CARRIER_PARAMS)` (the shared
  constant from §4 — i.e. the 4.7.0 values `tolerance_m=3.0, beta=0.0, gamma=0.25`).
- Accumulate `X`, `y`, `groups (game_id)`, and a parallel `provider` array; drop `frames`.
- **Disk-cache** the concatenated `features.parquet` / `labels.parquet` / `groups.npy` /
  `providers.npy` under `--output-dir/xshot_occurrence_v1/_feature_cache/` so HPO and re-runs never
  re-pull/re-extract (Ghost-GK pattern; feature extraction + the 419 MB IDSSE pull dominate cost).

**Phase 2 — ruthless HPO, run once per candidate (review L4).** `XShotOccurrenceObjective(fold=...)`
+ `OptunaStrategy(cfg, seed=42).run(obj, backend=InProcessBackend())`, TPE, fresh SQLite store,
`assert_cache_equivalence` (1e-9) intact. Two correctness changes to the shipped objective:

- **CV → `StratifiedGroupKFold` (stratify on label, group on `game_id`), not plain `GroupKFold`
  (review M1).** At ~0.02 positives the current `GroupKFold` (`_xshot_occurrence_objective.py:57`)
  silently drops single-class *train* folds (`:60`) and only scores PR-AUC when the *test* fold has
  both classes (`:66`), so per-fold positive counts swing and `cross_fold_*_std` measures
  fold-assignment noise, not model stability. `StratifiedGroupKFold` (already used by Ghost-GK)
  stabilises per-fold positives. **Fix the misleading "match-stratified GroupKFold" wording** in the
  objective docstring + this spec when switching — it is *grouped*, now also *label-stratified*.
- **Objective/metric coherence — keep log-loss, DROP `scale_pos_weight` (review M2).** xS is consumed
  as a **calibrated `P(shot)`** (it feeds VAEP-style features), so the HPO objective stays the proper
  scoring rule **log-loss** and the search space drops `scale_pos_weight` (it deliberately
  miscalibrates for recall — log-loss penalises it, the gate didn't reward it; three conflicting
  opinions). The natural base rate stands (XGBoost's `base_score` starts calibrated at ~0.02). The
  searched space becomes `{n_estimators, max_depth, learning_rate, min_child_weight, reg_lambda}`.
  If discrimination is weak, the remedy is feature work (`extended`/TF-19) or post-hoc isotonic
  calibration — **not** reweighting. (This supersedes the code-spec §6.3 `scale_pos_weight`-primary
  choice, with the stronger calibration rationale.)

Curation: key the `fold` dict by **provider** (not the single `"synthetic"` bucket) so provider
identity survives into per-provider diagnostics.

**Phase 3 — final fit (two candidates) + diagnostics + artifact.**
- Build two row-masks over the cached matrix: `public` (skillcorner+idsse rows) and `full` (all rows)
  — no re-pull, no re-extract.
- **HPO runs ONCE per candidate (NOT per outer fold; review N1/L4).** Each candidate's hyperparameters
  are selected once via the objective's inner CV, then **frozen**. The §5 paired comparison evaluates
  fold deltas at those frozen hyperparameters (so Δ_k isolates the *data* effect, not per-fold retune
  noise). Total HPO studies = 2 (one per candidate), not 2·K.
- **Set `base_score` explicitly to the train positive rate (review N4).** The M2 "log-loss keeps
  `P(shot)` calibrated without `scale_pos_weight`" argument leans on XGBoost ≥ 2.0 auto-estimating the
  intercept; the `>=2.0,<3.0` pin guarantees that today, but setting `base_score` explicitly (and/or
  asserting the xgboost major version) makes the calibration claim immune to a future pin move.
- Shipped fit per surviving candidate: `XShotOccurrenceModel(params=best).fit(X_v, y_v,
  carrier_params=DEFAULT_CARRIER_PARAMS, horizon_seconds=1.0)` on **all** that candidate's games.
- **Per-provider** held-out PR-AUC / Brier / log-loss (the meaningful signal under ~0.02 positives —
  plain AUC>0.5 is nearly free), plus permutation importance (sanity: the paper warns `r` dominates;
  if `openGoal`/GK features are ~0 importance, that is a recorded TF-19 risk, not a blocker).
- `model.save(out/xshot_occurrence_v1)`; **round-trip verify** (`load()` → `predict_proba` identical).
- `metrics.json` with `n_games`, `n_samples`, `n_providers`, `providers`, positive-rate, best
  hyperparams, log-loss/PR-AUC/Brier (mean±std), per-provider PR-AUC, top features, artifact size,
  and an `acceptance` block (below). **All quality numbers are explicitly labelled CV/protocol
  estimates, NOT a held-out score of the exact shipped booster** (the shipped artifact is re-fit on
  *all* games with no held-out — review N7). Also record the **resolved concrete variant + its data
  provenance** (`shipped_variant: "public"|"full"`, `provider_list`) so a `from_variant("default")`
  pin's provenance is auditable (review N5).

**Acceptance gates — pre-registered, no tuning to observed (review M5).** Discrimination **and**
calibration are both gated (since xS is a calibrated probability, gating PR-AUC alone is insufficient
— review M2):
| Gate | Threshold (fixed before the run) |
|---|---|
| `pr_auc > positive_rate` (beats the trivial base-rate classifier) | per-fold mean, strict |
| `brier < base_rate_brier` (beats predicting the constant base rate) | per-fold mean, strict |
| `log_loss < uniform_baseline` (`-ln 0.5`) | per-fold mean, strict |
| `cross_provider_no_degradation` (no single provider `pr_auc < its base rate`) | all providers |
| `artifact_size_lt_5mb` (bundling sanity) | booster JSON |

`cross_fold_*_std` is **reported as a diagnostic, not gated** — a "< 0.05" cutoff tuned to the run
that produced it is non-falsifiable (review M5). If we ever gate stability, it is against a
pre-registered absolute or a prior run, labelled as such.

**The trainer is FAIL-CLOSED on these gates (review N3, shift-left).** The acceptance block is
asserted *inside* `train_xshot_occurrence.py`: any failing gate → the script **exits non-zero and
refuses to write the bundled artifact / `model.json`**, so a sub-bar model physically cannot reach
`publish_xshot_occurrence.py` or `_xshot_weights/`. The §9 e2e tests re-assert the same gates, but
they don't run in normal CI and they run *after* a human has already chosen to publish — the
fail-closed trainer makes "shipped a model that missed the bar" structurally impossible, not
discipline-dependent. (Cheap: the block is already computed; the change is `assert`/`sys.exit(1)` and
gating the `save()` call.)

---

## 4. Carrier-default sync (latent code bug)

`_xshot_occurrence.py:310` still hardcodes `_DEFAULT_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta":
0.5, "gamma": 1.0}` and `train_xshot_occurrence.py` CLI defaults to `--beta 0.5 --gamma 1.0` — both
the **pre-4.7.0** carrier values. The library default (`infer_ball_carrier`) is now `beta=0.0,
gamma=0.25`. Because a *trained* model records+consumes its own params (R3), this stale default only
bites the **untrained** path and any caller that constructs a model without metadata — but it is
exactly the drift R3 was meant to prevent, now sitting in the constant.

**Fix (best-practice, anti-drift; review L1) — single source of truth, not reflection.** Introduce a
module-level `DEFAULT_CARRIER_PARAMS` constant in `_ball_carrier.py`; make `infer_ball_carrier`'s
signature defaults *reference* it, and have xS import the same constant for `_DEFAULT_CARRIER_PARAMS`.
Update the train CLI defaults to read it too. This is preferred over
`inspect.signature(infer_ball_carrier)` reflection because reflection silently breaks the day the
carrier API moves to a `carrier_params: CarrierParams | None = None` dataclass (the plausible R3
direction). A unit test asserts xS's default *is* the shared constant (intent, not a version-baked
literal — review L6).

---

## 5. Variant evidence + bundling + Hub publish

**Decision deferred to evidence (not asserted up front).** We train **two candidates** and decide
ship-one-vs-two from measured held-out quality. The variant axis for a booster is **NOT size** (both
are KB-to-low-MB, both bundle) — Ghost-GK's `default`/`full` split exists only because its KDE
artifact grows with data (9 MB ↔ 91 MB), which does not apply here. The meaningful axis is **data
provenance / reproducibility**:

- **`public`** — `skillcorner` + `idsse` only. Reproducible/auditable by anyone with the public
  pining token, from public data alone.
- **`full`** — adds `gradientsports` (owner-tier). Potentially better generalization, not publicly
  reproducible.

Both train from the **same cached feature matrix** (§3) — one corpus pull, one extraction, two fits —
so the marginal cost of "do both" is a second HPO/fit run, not a second data pull.

**Evaluate on a COMMON public held-out set, at FIXED hyperparameters (review H3 + N1) — the
comparison must be apples-to-apples *and* must not re-tune per fold.** Scoring `full` on folds that
include GS and `public` on folds that don't confounds "GS matches are easier to predict" with "GS
*training* data helps." And re-running HPO inside each fold (the prior draft's "its own HPO" per fold)
would be 2·K studies *and* inject per-fold retune noise into every Δ_k — exactly the L5 confound. So:
- **HPO once per candidate, frozen** (Phase 3) → `params_public`, `params_full` (used for the *shipped*
  deployment fits). The paired protocol does **not** re-run HPO per outer fold.
- **The paired comparison holds hyperparameters SHARED across both arms (review P1).** Both `public`
  and `full` are fit at the **same** `params_public` inside the loop — so the *only* thing varying is
  the training data. Fitting `public` at `params_public` and `full` at `params_full` would reintroduce
  the L5 data+tune confound *inside* the test; using one shared param set makes `Δ_k` a clean
  **data-effect** signal *and* renders the HPO-selection optimism common-mode (identical for both arms
  ⇒ cancels exactly).
- Outer `StratifiedGroupKFold` over **public games only** (skillcorner + idsse). For each public fold
  `k` (held-out = public test games in `k`):
  - `public`: fit at `params_public` on public-train (folds ≠ `k`).
  - `full`: fit at `params_public` on public-train (folds ≠ `k`) **+ all GS games** (leakage-free —
    `full` never sees public-test games).
  - Score **both** on the **same** public held-out fold `k`.
- `Δ_k = PR-AUC(full) − PR-AUC(public)` is now a clean **data-effect** estimate — the claim §5 + the
  CHANGELOG want. `metrics.json` records `paired_delta_is_data_effect_shared_params: true` and
  `paired_hpo_nested: false` so the comparison's nature is explicit (review P1). Any residual held-out
  optimism is **common-mode** and cancels in the paired `Δ_k`;
  where cheaply arrangeable, draw the inner HPO CV to exclude the outer test games, but the paired
  delta is robust regardless.

**Pre-registered decision rule (paired, set before looking at the numbers).** Let `Δ_k = pr_auc_full,k
− pr_auc_public,k` (and the matched Brier delta) on the shared held-out fold `k`. **Ship two** iff
`Δ_k > 0` in **at least `K−1` of `K` folds** (consistent sign — a proper *paired* test, not the
invalid "2× within-model std" of the prior draft) **and** the mean `Δ` clears a pre-registered
absolute margin **and** `full` shows no cross-provider degradation. Then bundle **`full`** as the
shipped default (small ⇒ no size reason to bundle the weaker model — *inverts* Ghost-GK's
bundle-the-smaller logic) and publish **`public`** to Hub as the reproducible/auditable variant.
**Ship one** otherwise: the bundled **`public`** model (simplest, fully reproducible).

The shipped artifacts are then re-fit on *all* their available games (public-only, or public+GS) at
the chosen HPO; the protocol above is the *evidence*, not the final fit. **CHANGELOG honesty (review
L5):** each candidate is independently HPO'd, so the reported `Δ` is a *training-data + retune* joint
effect — word it "full corpus + retune improves common-held-out PR-AUC by X," never "GS data alone
adds X."

Final `from_variant` names are fixed at decision time (`"default"` always aliases the recommended
shipped model). The bundling/publish **infra is identical** regardless of how many ship.

**`default`-alias provenance stability (review N5, Hyrum).** Because `"default"` resolves to `public`
*or* `full` depending on evidence, and shipped models are re-fit across retrains, a downstream
`from_variant("default")` pin could silently change data provenance (→ predictions move with no
visible cause). Mitigations: (a) record the **resolved concrete variant + provider list** in both
`metadata.json` and `metrics.json` (so any consumer can see what "default" actually is); (b) treat
the `default`→variant mapping as **stable across versions** — any flip (`public`⇄`full`) is a
dedicated **CHANGELOG** entry, never a silent change.

**Bundling — mirror the Ghost-GK package layout:**
- Bundled artifact dir `silly_kicks/tracking/_xshot_weights/<variant>/{model.json, metadata.json,
  SHA256SUMS}` (parallel to `_ghost_gk_weights/`). At minimum the shipped default is bundled; if we
  ship two, the bundled default is `full` and `public` is Hub-hosted (it stays small enough to bundle
  too, but Hub is its canonical home — like Ghost-GK's `full`). The build already ships non-`.py`
  files inside `silly_kicks` (pyproject `include = ["silly_kicks"]`), so the dir bundles
  automatically; confirm with a `python -m build` wheel-content check (boosters are small, no
  wheel-size concern).
- `from_variant(name)` → `cls.load(_XSHOT_WEIGHTS_ROOT / name)` if bundled (SHA-256 verified); else,
  for a Hub-hosted variant, fall through to `from_hub` (Ghost-GK's exact fall-through). `"default"`
  aliases the recommended shipped model. An unknown variant → `FileNotFoundError`.
- `from_hub(repo_id="silly-kicks/xshot-occurrence-v1")` → `snapshot_download` + `load` (needs
  `huggingface_hub`, lazily imported as Ghost-GK's does; `load`/predict themselves need `xgboost`).
  When two ship, the public variant lives at a sub-path / revision of the same repo.

**`scripts/publish_xshot_occurrence.py`** (mirrors `publish_ghost_gk.py`): `--artifact-dir`,
`--repo-id` (default `silly-kicks/xshot-occurrence-v1`), `--verify-only`. Loads + SHA-verifies,
predicts on a tiny synthetic sample, `HfApi().upload_folder(...)`, then `from_hub` re-download and
assert identical predictions. The **same** verified artifact is copied into
`_xshot_weights/default/` for the wheel (one provenance, two delivery channels).

---

## 6. Metadata template (TF-38 insurance + R3 generalization)

Extend the xS `metadata.json` written by `save()` with:
- `pitch_length: 105.0`, `pitch_width: 68.0` — the physical pitch the goal-relative features assume.
- `geometry_version` — a string tag for `silly_kicks/tracking/_geometry.py` (e.g. `"goal-relative-1"`).
- `xgboost_version` + `training_platform` (e.g. `"linux-aarch64"`) — reproducibility/debuggability for
  a model trained on the ARM DGX box and served on x86 (review L2).
- (already present) `carrier_params`, `horizon_seconds`, `shot_types`, `feature_set`, `feature_names`,
  `params`, `version`.

**Fail-closed on a non-translation mismatch (review M4).** `load()` records these and checks the live
constants. A **pitch-dimension or unit mismatch** (`pitch_length`/`pitch_width` differ from recorded)
genuinely skews every goal-relative feature → **raise** (`IntegrityError`/`ValueError`), never warn —
a `warnings.warn` is invisible in a swallowed-stdout Spark/batch serve. A `geometry_version`
difference at *identical* pitch dimensions is the translation-invariant case (e.g. the TF-38
origin shift) → **warn** only. This turns silent skew into a loud failure for the one case that
matters, and is the template Ghost-GK's R3 follow-up copies. **Caveat (review L2):** no non-e2e test
asserts exact frozen predictions — XGBoost inference can differ at the ULP across arch/BLAS, so CI
asserts stay directional/in-bounds (see §9 H2 test).

---

## 7. Ghost-GK opportunistic re-fit (acceptance-gated)

Ghost-GK calls `infer_ball_carrier(frames)` with **library defaults** at both train and serve, so
re-running `train_ghost_gk.py` on the current (4.7.0) library automatically fits against the new
carrier defaults — **no Ghost-GK code change needed**, just compute on the already-staged corpus.

**Rationale for keeping it minimal:** the goal is to *erase the carrier skew*, not to re-optimize the
model — so the re-fit keeps Ghost-GK's existing fixed recipe (`n_estimators=500, max_depth=8,
subsample_fps=1.0`). Adding ruthless HPO simultaneously would conflate "new carrier regime" with "new
hyperparameters" and muddy the before/after read. (Ruthless-HPO-for-Ghost-GK is a separate future
improvement, not this cycle.)

**Acceptance-gated / non-regression (best-practice guard).** The skew impact is negligible
(`team_in_poss` is a long-tail feature, ≪ `defensive_line_x`=15.2), so we must NOT ship a *worse*
Ghost-GK merely to erase it. The re-fit is **accepted only if** it (a) passes Ghost-GK's existing
acceptance gates (overall MAE < 2 m, per-provider < 3 m, cross-fold std < 0.5 m) **and** (b) does not
regress euclidean MAE vs the currently-bundled `default` beyond a small tolerance on a common
held-out comparison. **Incumbent-bias caveat (review L3):** if that held-out set overlaps the bundled
model's *own* training games, the comparison favours the incumbent (it has seen those frames). Use a
held-out set demonstrably disjoint from the bundled model's training corpus where we can establish it;
where we can't (the bundled model's exact game list may be unavailable), treat the comparison as
**conservative** — biased toward *keeping* the incumbent, which is the safe direction for a re-fit
whose only purpose is erasing a negligible skew. **Corpus caveat:** the bundled Ghost-GK `default` is 36 k samples (full: 537 k);
if the pining corpus is materially thinner (Phase-0 probe), a re-fit on it could *under-fit* and
regress. In that case we **keep the bundled Ghost-GK, record the decision** (skew is negligible), and
note that a true Ghost-GK refresh wants a bronze-scale pull — folding it into the R3 follow-up PR.
This honours "in scope" without shipping a regression.

**PR-S81 mechanics (review N6).** (a) The asset reused across PR-S80→PR-S81 is the **raw match pull**
staged on the Spark box, **not** the xS `_feature_cache/` — Ghost-GK extracts its own KDE inputs via
`prepare_ghost_gk_training_data`, so the xS feature cache does not feed the GK re-fit (don't assume
it does). (b) PR-S81's re-fit must run against a **defined library commit** (carrier defaults +
`_geometry`), recorded in the GK `metadata.json` (the same L2 `xgboost_version`/`training_platform` +
a `library_commit`/version field), so the re-fit's *train* library matches the wheel's *serve*
library — necessary because PR-S80 (and possibly part-deux's fft-cic) may land between PR-S80 and
PR-S81 and move those constants.

---

## 8. xfn default-list wiring

`xshot_occurrence_xfns` was deliberately kept out of every default list until weights shipped (code
spec D1). Now wire it in — **into `pre_shot_gk_full_default_xfns` ONLY, not the general
`tracking_default_xfns` (review P3, owner-confirmed).** Rationale: `tracking_default_xfns` is the
general tracking-aware set (geometric/positional features with **no** trained-model dependency);
adding a bundled-weights, `[xgboost]`-at-frame-time, booster-inference feature to it would escalate
the runtime contract for *every* consumer of the broadest, most-depended-on list (a Hyrum break —
`import silly_kicks.tracking` already pulls xgboost at *import* time, but this would add a *frame-time*
weights load + inference to the general path). xS is a GKDV Layer-2 shot-context surface, so its home
is the shot/GK-context union, where it pairs naturally with the pre-shot-GK position/angle features.
This still satisfies scope item 6 ("turn xS on") — it is live in the shot-context default, which is
where a consumer wanting shot-imminence signal looks.

Covered by: a membership test (in `pre_shot_gk_full_default_xfns`) + a **negative** membership test
(NOT in `tracking_default_xfns`) + an introspection (`frames=None` → 3 NaN columns) test. The bundled
model load is **memoized** (`_VARIANT_CACHE`, review P3) so the default-list path doesn't reload +
SHA-verify per call. CHANGELOG records: xS wired into `pre_shot_gk_full_default_xfns` only;
`tracking_default_xfns` stays model-free.

---

## 9. Testing

**Flip the e2e placeholders (code spec §10.3) from skip → real (gates per §3 table):**
- `test_xshot_gradientsports_e2e` — full pipeline + acceptance: `log_loss < uniform_baseline`,
  **PR-AUC > base rate**, **Brier < base-rate Brier** (calibration, review M2); cross-fold std
  reported as a diagnostic, not asserted (review M5).
- `test_xshot_cross_provider` — trains on ≥2 providers; no single-provider degradation; the
  `public`-vs-`full` comparison uses the common public held-out protocol (§5 / review H3).
These stay `@pytest.mark.e2e` (need real provider data).

**New non-e2e (run in the regular suite — no network, uses the bundled artifact):**
- **`test_bundled_model_directional_quality` (review H2 + N2 — the real CI quality tripwire).** Since
  this is the *only* network-free signal that the bundled model isn't dead, its own robustness is
  load-bearing — so freeze at the **feature-vector layer, not raw frames**: commit a tiny set of
  **real extracted feature rows** (run `extract_xshot_features` on the slim real-provider fixtures
  once, label via `build_xshot_labels`, pick a handful of known true-positive and true-negative rows,
  freeze the 27-col **feature vectors** as a committed artifact). Assert **mean `P(shot | pos)` > mean
  `P(shot | neg)`** (or AUC > 0.5 + margin) over the mini-set. Freezing post-extraction sidesteps
  hand-built-frame schema fragility (a malformed frame would pass/fail for the wrong reason); multiple
  rows kill single-pair flakiness; it's a **ranking**, so arch-robust (no ULP). Pick rows that differ
  in `r` (ball-in-box vs centre-circle) so the tripwire stays valid even under the paper's
  `r`-dominance / weak-GK-feature risk. Catches a degenerate constant booster or a broken re-train
  that `from_variant(...).predict in-bounds` cannot. (Frame→feature correctness is separately covered
  by the real-provider extraction tests, code-spec §10.2a.)
- `test_from_variant_default_loads` — `from_variant("default")` loads + predicts in-bounds (covers the
  bundled-weights load path; the quality assertion lives in the directional test above).
- `test_bundled_metadata_matches_training_intent` (review L6 — intent-named, not version-baked) — the
  *shipped* model's `carrier_params` equal the shared `DEFAULT_CARRIER_PARAMS` constant (guards
  against shipping a stale-carrier-default-trained model), and `pitch_length/pitch_width/
  geometry_version/xgboost_version/training_platform` are present.
- `test_default_carrier_params_are_shared_constant` — xS `_DEFAULT_CARRIER_PARAMS is
  DEFAULT_CARRIER_PARAMS` (§4 anti-drift; review L1).
- `test_xshot_xfns_in_default_lists` — membership in the wired lists; introspection NaN behaviour.

**`negative_subsample` is now TRAIN-fold-only (review M3 — fixed this cycle, owner-directed).** The
prior implementation applied it pre-split in `prepare_xshot_training_data`, contaminating every
downstream CV eval fold's class balance *and* the `positive_rate`/`base_rate_brier` baselines. Fixed:
`prepare_xshot_training_data` no longer subsamples (it always returns the faithful distribution — the
train/serve-parity contract); a standalone **`subsample_negatives`** helper (train-only contract)
thins negatives in **train folds only** inside the objective's `_cv_logloss`, the trainer's
`_cv_metrics` + `_paired_data_effect`, and the final all-data fit — the held-out fold always keeps the
true balance. Deterministic per fold (`seed + fold_index`) so the cache-equivalence gate still holds.
The maintainer run used `None` (full distribution) regardless, so the shipped model is unaffected.

**Train/publish script smokes (regular suite, synthetic fixture):** the existing
`test_train_script_smoke` stays; add `test_publish_script_verify_only` (verify path, no network).

**Wheel-content check:** a build test (or a documented manual `python -m build` step) asserting
`_xshot_weights/default/*` is present in the wheel.

---

## 10. Execution runbook (DGX Spark)

1. `ssh karsten@192.168.68.73`; sync the repo to the cycle branch; `pip install -e
   ".[train,xgboost,kloppy]"`; export `PINING_FOR_THE_DATA_TOKEN`, `HF_TOKEN`.
2. **Phase 0 probe** (manifest counts) → decide pining-only vs bronze-escalation for Ghost-GK.
3. Run `train_xshot_occurrence.py --providers skillcorner,idsse,gradientsports --output-dir models/
   --n-trials <N>` as a **background job** (the IDSSE pulls + HPO are long). Poll per the
   background-task policy; the owner monitors and signals completion (no `gh run watch`-style loops).
4. The trainer is **fail-closed (N3)** — it exits non-zero and writes no artifact if any acceptance
   gate fails, so reaching this step at all means the gates passed. Review `metrics.json` + the §5
   `public`-vs-`full` paired comparison for the ship-one-vs-two decision, then
   `publish_xshot_occurrence.py --verify-only`, publish, and copy the verified artifact(s) into
   `_xshot_weights/<variant>/`.
5. **PR-S80 (this PR):** pull artifacts + `metrics.json` back to the repo; the single PR-S80 commit =
   xS weights + the trainer/objective/metadata/test/xfn-wiring code.
6. **PR-S81 (separate follow-up, same staged corpus):** run `train_ghost_gk.py` on the cached corpus;
   apply the §7 non-regression gate; ship (its own single commit + version bump) or keep-bundled +
   record. Do **not** fold into PR-S80 (L7 decision).

---

## 11. Version / release / ADR

- **Version — DECIDED: keep provisional until merge order is known (review H1).** The part-deux
  (fft-cic) session also branched off 4.7.0 and targets 4.8.0; neither change depends on the other.
  This PR's plan stays **version-agnostic** — do NOT hard-code a number — and the version-bump hard
  gate (`pyproject.toml`/`silly_kicks/__init__.py`/`TODO.md`/`CHANGELOG.md` all in sync) is applied at
  commit time with whatever number is correct then (4.8.0 if we merge first, else the next free
  minor). Whoever merges second re-bumps.
- **Release granularity — DECIDED: two PRs (review L7, Option B).** This spec/PR is **PR-S80 — xS
  weights only**. The opportunistic Ghost-GK re-fit (§7) is a **separate follow-up PR-S81** off the
  **same staged corpus** (the corpus stays on the Spark box, pulled/extracted once, reused) for
  independent revertability — mirroring how the Ghost-GK R3 structural fix was already split out.
  PR-S81 collapses to nothing if §7's non-regression gate keeps the incumbent (no artifact change to
  release). Each PR is its own single commit + version bump.
- **CHANGELOG `### Added`:** trained xS weights (bundled default + Hub; one or two variants per the
  §5 evidence — state the outcome and the `public`-vs-`full` PR-AUC delta that drove it; record the
  resolved variant + provider provenance — the `default` alias is stable across versions and any
  future `public`⇄`full` flip is its own CHANGELOG entry, review N5),
  `from_variant`/`from_hub` live, `xshot_occurrence_xfns` wired into default lists,
  `publish_xshot_occurrence.py`, xS metadata pitch-dims/geometry template. **`### Changed`:** xS `_DEFAULT_CARRIER_PARAMS` synced to
  4.7.0 (sourced from the library); Ghost-GK weights re-fit against 4.7.0 carrier defaults *(if the
  non-regression gate passes)*.
- **ADR-011** (trained-model lifecycle) already covers the staging; add a one-line note that xS
  collapsed to a single bundled variant (no size axis) and that model metadata now records pitch
  dims + geometry version as the coordinate-change insurance template.
- **TODO.md:** update the TF-16 row (weights shipped); GKDV program note (TF-16 Layer-2 fully closed,
  TF-19 xS-arm unblocked); the Ghost-GK R3 follow-up TODO stays.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| pining corpus too thin for a credible xS / non-regressing Ghost-GK | Phase-0 probe gates the decision; xS tolerates modest volume (positives are dense per shot); Ghost-GK re-fit is non-regression-gated with bronze-escalation noted |
| `r`-dominates → weak GK sensitivity (paper §4.2) | Recorded as a TF-19 risk, not a blocker; permutation importance surfaces it; motivates the deferred `extended` variant + TF-19 DAS/cover-shadow arms |
| Extreme class imbalance (~0.02) — a near-constant predictor wins on log-loss alone | We do NOT reweight (M2): log-loss keeps xS calibrated, and the *gates* are **PR-AUC > base rate** (discrimination) **+ Brier < base-rate Brier** (calibration), so a constant predictor fails the PR-AUC gate. Weak discrimination ⇒ feature work / calibration, not `scale_pos_weight` |
| ARM/aarch64 wheel gaps on DGX Spark | xgboost ships aarch64 wheels; ruthless/optuna pure-Python; `accessible-space` not required for `faithful` |
| Shipping a stale-carrier-default-trained model | Trainer passes the shared `DEFAULT_CARRIER_PARAMS` explicitly + records them; `test_bundled_metadata_matches_training_intent` guards the shipped artifact against the shared constant |
| Bundled booster bloats the wheel | Single small booster (<5 MB gate); wheel-content + size check |
