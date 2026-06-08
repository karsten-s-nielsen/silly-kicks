# TF-17 xCrossAttempt — PR-B (weights + shipped-surface GK validation + TF-19 wiring)

**Date:** 2026-06-06
**Status:** Draft — for review
**Layer:** GKDV Layer 2 (TF-17)
**Predecessor:** PR-A (code, untrained) — shipped as release **4.11.0** (#99); that code is now on **main @ 4.14.0** (B5: not a contradiction — released at 4.11.0, main has since advanced). Design: `docs/superpowers/specs/2026-06-03-tf17-xcross-attempt-design.md` (the settled architecture; this doc elaborates only the PR-B deliverables enumerated in its §10).
**Template:** PR-S80 (TF-16 xS weights cycle) — `project_pr_s80_tf16_weights_cycle`. Mirror its decisions almost verbatim.
**Successor:** PR-C (causal harness + ADR-015) — out of scope here; a null causal finding never gates PR-B.

---

## 0. Fact-check ledger (verified against the backend 2026-06-06)

| Claim | Status | Evidence |
|---|---|---|
| PR-A code is complete + on main @ 4.14.0 | ✅ | `_xcross_attempt.py` (extractor/model/surfaces), `_xcross_attempt_objective.py`, `_occurrence_labels.py`, `scripts/train_xcross_attempt.py`, atomic mirror all present |
| The trainer already has the two-candidate `public`/`full` paired test, fail-closed gates, feature cache | ✅ | `train_xcross_attempt.py` `_paired_data_effect`, `_gates`, `_cv_metrics`, Phase-1 cache |
| The trainer has **no** GK ablation, **no** substitution probe, **no** permutation importance | ✅ | grep: only `train_ghost_gk.py` computes importance; xS trainer computes none |
| `from_variant("default")` loads bundled-or-raises; **only `"public"` cascades to Hub** (`default` does NOT self-heal from Hub); `from_hub` **raises** | ✅ B4 | `_xcross_attempt.py:534-543` (`elif variant == "public"`), `:548-552` inert stub |
| No `_xcross_weights/` dir bundled; no `publish_xcross_attempt.py` | ✅ | `ls` — absent |
| HF: only `silly-kicks/ghost-gk-v1` exists; `xcross-attempt-v1` **and** `xshot-occurrence-v1` are **absent** | ✅ live | `HfApi().list_models(author='silly-kicks')`; `repo_info` 404 on both |
| **xS shipped public-only, bundled, NO Hub repo** — so a Hub repo for xCross is **contingent** on the paired test | ✅ | `_xshot_weights/` has only `default/` (no `full/`); `xshot-occurrence-v1` 404 |
| HF auth present as org admin `karstenskyt` | ✅ live | `HfApi().whoami()` |
| xfn-union wiring points | ✅ | `tracking/features.py:742` + `atomic/tracking/features.py:489` (both append `xshot_occurrence_xfns()`) |
| Hatch weights-exclude must be set on **both** wheel + sdist | ✅ | `pyproject.toml:114-123` (ghost-gk `full` excluded twice; the 4.10.0 sdist failure is the cited reason) |
| Pining corpus = 81 matches (10 skillcorner + 7 idsse public + 64 gradientsports owner) | ✅ (PR-A) | `_list_matches` with owner token; re-verify live at run time |
| Clean 4.13.0 GS frames/actions exist on the box at `~/Development/ghost_gk_refit/clean_cache` | ⚠️ to verify | PR-S83 note; **layout vs `--data-dir` contract verified on the box before use** (Fork B) |

**Carrier defaults (R3):** train against the **4.7.0** ball-carrier defaults — `DEFAULT_CARRIER_PARAMS = {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}` (`_ball_carrier.py`). The trainer already passes this shared constant; metadata records + inference consumes it. A future TF-24 carrier-default change is an xCross retrain trigger.

---

## 1. Purpose & scope

PR-B turns the untrained xCross code path into a **shipped, weighted feature** and produces the **headline TF-19-viability evidence**. Concretely (from PR-A design §10 PR-B):

1. **Maintainer training run** on the gated multi-provider pining corpus (DGX Spark), against the **clean 4.13.0 GS events** (the critical prereq — §6) and the 4.7.0 carrier defaults.
2. **Pre-registered two-candidate `public` vs `full` paired test** — run it, let the data decide (xS shipped public-only; xCross may differ). Fail-closed PR-AUC/Brier/log-loss acceptance gates.
3. **The headline deliverables — three shipped-surface GK validations** (§3): the **GK-block ablation**, the **GK-substitution-sensitivity probe** (TF-19's literal operation, with a placebo comparison), and **permutation importance** (which finally **measures `score_differential`'s weight** — the brief's UNMEASURED item).
4. **TF-19 wiring:** `xcross_attempt_xfns` into `pre_shot_gk_full_default_xfns` + atomic mirror **only** (not the general default).
5. **Ship:** bundle `default` in the wheel (hatch-exclude `full/` on **both** wheel + sdist; verify **both artifacts <100 MB**); publish `full` to Hub `silly-kicks/xcross-attempt-v1` **only if the paired test ships two**; model card; NOTICE/CHANGELOG/version/ADR-011/`uv.lock`.

**Out of scope (PR-C):** the `silly_kicks/_causal/` matching port, `scripts/validate_xcross_causal.py`, ADR-015. A null causal finding does not gate this PR.

**Non-negotiables carried from PR-S80:** drop `scale_pos_weight` (xCross is a calibrated P(cross)); `negative_subsample` is train-only and **off by default** (crosses are ~3× more common than shots — healthy base rate); `StratifiedGroupKFold` grouped on `game_id` (str-normalized); the paired test uses **shared** hyperparameters (data-effect, not data+tune); metadata is fail-closed on pitch-dim mismatch.

---

## 2. Decisions resolved in brainstorming (2026-06-06)

- **Fork A — eval-code home: private package module `silly_kicks/tracking/_xcross_eval.py`** (revised from "trainer-local" after review B1, owner-confirmed). The three eval functions (`gk_block_ablation`, `gk_substitution_probe`, `permutation_importance_report`) live in a **private** module; `train_xcross_attempt.py` calls into it. Rationale: `scripts/` is not a package (no `__init__.py`, verified), so trainer-local functions are unit-testable only via `importlib.util.spec_from_file_location` (executing the script's argparse/heavy-import surface) — the repo pattern at `tests/tracking/test_loader_pining_to_cache.py:13`, but a tax on every unit + e2e test. A private module gives clean hexagonal imports with **no premature general abstraction** — it stays private + single-repo, NOT promoted to ruthless-efficiency (whose 0.2.1 charter is *"a general optimisation/search substrate"* with zero model-evaluation surface — a category mismatch). Permutation importance delegates to `sklearn.inspection.permutation_importance` inside the module. This realizes the deferred-extraction's "model-evaluation home (NOT ruthless-efficiency)" **now, as private**, rather than later as public; promote to public only if/when a 2nd consumer (TF-19 / retro-xS ablation) lands.
- **Fork B — clean GS corpus source:** reuse the box `~/Development/ghost_gk_refit/clean_cache` **iff** its on-disk layout matches the trainer's `--data-dir DIR/*/{frames,actions}.parquet` contract; otherwise do a fresh `--providers skillcorner,idsse,gradientsports` pull with the owner token. xCross extracts its own features either way. Verified on the box before the run.
- **Fork C — single PR.** One branch `pr-s84-tf17-xcross-weights`, one commit. (PR-C separate, later.)

---

## 3. The three shipped-surface GK validations (headline; private `_xcross_eval.py`)

All three run on the **shipped candidate** (the paired-test winner — GK/carrier resolution is provider-variable, so the metrics must describe the actually-bundled model, not an arbitrary candidate), and write into the artifact's `metrics.json`. All assert *production*, never *GK wins* — a null is a reported finding (PR-A design H-2/H-3, R2-H1).

**They answer three genuinely different questions and are NOT collapsed into one AND-gate (review A2):**

| Validation | Question | Role |
|---|---|---|
| §3.1 ablation (retrain w/o block) | Does the GK block add **marginal predictive value** (held-out)? | **reported context** |
| §3.2 substitution probe (perturb GK, re-predict) | Does the surface **move when the GK moves**? | **the TF-19 viability GATE** |
| §3.3 permutation importance | How does the model **weight each feature** (incl. `score_differential`)? | **reported context** |

The probe and ablation can legitimately diverge: GK geometry is plausibly collinear with `dist_endline` / `box_off_def_ratio` / `gk_carrier_side`, so the surface can be highly probe-responsive (trees split on GK features) while a retrain-without-block barely changes held-out CV (the block's *marginal* value is absorbed by correlates). TF-19 only needs the surface to **move** — so **TF-19 viability gates on the probe alone** (§3.2), and ablation/importance are reported as context, never as the gate. Collapsing all three to one AND-gate (the PR-A draft's framing) could mask a real TF-19 signal behind CV redundancy.

### 3.1 GK-block ablation — `gk_block_ablation` (reported context: marginal predictive value)
Trained held-out **PR-AUC + log-loss** WITH vs WITHOUT the `XCROSS_GK_BLOCK` columns (the block is already a contiguous, droppable tail — `test_gk_block_isolatable` in PR-A). Same CV protocol + shared params as `_cv_metrics`; report `delta_pr_auc`, `delta_log_loss`. Answers *"does the GK block improve prediction?"* — informative, but **not** the TF-19 gate.

### 3.2 GK-substitution-sensitivity probe — `gk_substitution_probe` (THE TF-19 viability gate)
The product question: *does the per-frame surface actually move when GK position changes?* Deterministic (seed + fixed N + fixed panel — review B7), on a fixed sample of shipped-domain frames:
1. Predict P(cross | actual GK).
2. Apply a **panel of realistic GK perturbations** to the GK row — **synthetic displacements, NOT the ghost-GK model** (the ghost-GK mode would couple to ADR-014's `kde_backend` train/serve guard, pinned to **TF-19, not TF-17**; keep the PR-B probe self-contained). Re-extract (`extract_xcross_features`) and re-predict per perturbation. Summarize `|P(actual) − P(shifted)|` (mean / median / p90) over panel × frames.
3. **Corrected placebo controls (review A1) — the GK panel's deterministic 6-feature move would beat a random outfielder trivially (feature-set asymmetry), so a single random-player placebo is structurally biased toward "GK wins." Use TWO controls:**
   - **(a) Nearest-defender placebo (conservative):** apply a **geometrically-equivalent** displacement panel to the nearest defender (a feature-bearing, tactically-relevant outfielder). The GK→post-direction panel members have no outfielder analogue, so define the equivalent panel as the **rotation/translation-matched** subset (lateral ±2/±4 m, toward/away from goal by the same metres) applied identically to both — so "same-magnitude" is truly comparable. Tests whether GK geometry adds sensitivity **beyond a comparable defender**.
   - **(b) Random-outfielder band floor (averaged):** average `|Δ|` over **several** random outfielders (not one — a single draw can land on a zero-leverage player and floor the band), same seed/panel.
   Report the GK `|Δ|`, the nearest-defender `|Δ|`, and the random-band floor.

**What the probe does and does NOT establish (review C3):** it demonstrates the surface is **GK-responsive** (necessary for TF-19 to produce a non-degenerate `Δ_cross`) and bounds **generic positional sensitivity** — it does **not** establish that GK position is *causally* special (that is PR-C's matched causal harness). `tf19_ready:true` means "the surface moves enough when the GK moves," not "GK is causally important." Honest scoping of the nearest-defender control: it is **partially self-limiting** — re-extraction recomputes `dist_nearest_def` against a *floating* identity, so displacing the current nearest defender just promotes the 2nd-nearest and caps that control's `|Δ|`, which re-inflates the GK/control ratio. The averaged random band partially compensates, and the bias direction (cutting toward "GK looks better") is the **safe** side for a gate, but the probe's claim is responsiveness, not causal primacy. (Feature extraction is deterministic — there is no stochastic "jitter"; the placebo measures generic positional sensitivity, not noise.)

**TF-19 viability flag — PRE-REGISTERED NUMERIC RULE (review C1):** the threshold is pinned in code **before the run** (constants in `_xcross_eval.py`, asserted by a unit test), not chosen after seeing the distributions:
```
TF19_PROBE_RATIO     = 2.0     # GK median |Δ| must be >= 2x the STRONGER positional control
TF19_PROBE_ABS_FLOOR = 0.01    # AND GK median |Δ| >= 0.01 (1 percentage point of P(cross)) absolute
tf19_ready := (GK_median_absΔ >= TF19_PROBE_RATIO * max(nearest_def_median_absΔ, random_band_median_absΔ))
              AND (GK_median_absΔ >= TF19_PROBE_ABS_FLOOR)
```
The absolute floor is essential — a 2× ratio over a negligible band is still negligible, so a surface that shifts <1 pp when the GK moves is not TF-19-useful regardless of the ratio. **These two constants are the pre-registration; owner-confirmable before the run, frozen during it.** If `tf19_ready` is **false**, the surface **still ships** (a weak signal is not a build break) but is **never** shipped silently as "GK novelty" — the flag is loud and TF-19 consumption is gated on GK feature-engineering first. Ablation/importance being flat does **not** set `tf19_ready:false` (and vice-versa); only the probe gates.

### 3.3 Permutation importance — `permutation_importance_report` (reported context, incl. the score_differential measurement)
`sklearn.inspection.permutation_importance`, **computed on genuinely held-out CV models, NOT on the shipped weights' own training data (review C2).** The shipped candidate is the final fit on **all** data (§7), so it has no honest holdout by construction — permuting any in-sample fold would be leaky/optimistic and would bias the very `score_differential` number CHANGELOG quotes. So: for each `StratifiedGroupKFold` fold, fit on K−1 folds and run `permutation_importance` on the held-out fold K; **average across folds**. Label the result **"architecture-representative, NOT measured on the production weights' own training data"** in `metrics.json` and the CHANGELOG (option (a) — matches how `_cv_metrics` already works; lighter than carving a dedicated final-fit holdout). **Reproducibility spec (review B3):** `scoring="average_precision"` (matches the gates/CV — `_cv_metrics` uses `average_precision_score`; accuracy is wrong under imbalance), `n_repeats=10` (≥10), pinned `random_state`. Reported per feature into `metrics.json`.

**`score_differential` measurement (the brief's UNMEASURED item) + coverage guard (review B2):** `score_differential` is NaN when the shipped candidate's actions lack `result_id`/`result_name` (`_has_results`, `_xcross_attempt.py:257,291`), and xgboost treats NaN as missing → a trivially-near-zero importance that would make the CHANGELOG number meaningless. So **also report `score_differential_coverage` = the non-NaN fraction in the shipped training matrix** (same discipline as §6's range-probe), and the CHANGELOG number is **qualified by that coverage**. **Pre-run verification:** confirm whether the shipped providers' pining actions carry `result_id`/`result_name` (SPADL emits `result_id`, but verify it survives the loader) — if coverage is low, say so explicitly rather than quoting a misleading near-zero.

---

## 4. Model & serialization wiring (small, code-only)

- **`from_hub` real body:** replace the inert stub (`_xcross_attempt.py:548`) with the working ghost-GK pattern — `try: from huggingface_hub import snapshot_download except ImportError: raise ImportError("...[xcross]")`; `snapshot_download(repo_id)` → `cls.load(Path(local_dir))`. **Cascade wording corrected (B4):** `from_variant("default")` loads the bundled dir **or raises — it does NOT self-heal from Hub**; only `from_variant("public")` falls through to `from_hub`. Bundling `_xcross_weights/default/` makes the **production** path (`_resolve_model(None) → from_variant("default")`) live with no code change.
- **Who consumes the Hub `full` (B4)?** Production **always** loads the bundled `default` (the booster is tiny — approx. 1–2 MB — so unlike ghost-GK there is no size reason to off-load it). The Hub `full` is therefore **opt-in reproducibility/distribution only**, reachable via `from_hub()` / `from_variant("public")` — nothing loads it by default. **Decision:** publish `full` to Hub **iff the paired test ships two** (matches the brief + `project_tf17_xcross_next`); if the paired test ships public-only (the xS outcome), **mirror xS exactly — bundle only, create no Hub repo**. Either way the bundled `default` IS the shipped model; the Hub repo never becomes a hidden production dependency.
- **`[xcross]` optional extra — DECIDED: add it.** A tiny `[xcross]` extra declaring `huggingface_hub>=0.20.0` (mirrors `[ghost-gk]`), and the `from_hub` ImportError message names `pip install silly-kicks[xcross]`. Rationale: reusing `[ghost-gk]` for a cross feature is semantically wrong; xCross's `from_hub` only matters if `full` ships to Hub, but the extra + message ship regardless so the inert-path error is correct. One-line `pyproject.toml` addition.
- **Bundle:** write the trained `default/` (model.json + metadata.json + metrics.json + SHA256SUMS, ~1–2 MB like xS) under `silly_kicks/tracking/_xcross_weights/default/`. Hatch `exclude = [... "silly_kicks/tracking/_xcross_weights/full"]` on **both** `[tool.hatch.build.targets.wheel]` and `[tool.hatch.build.targets.sdist]` (the 4.10.0 sdist-overflow lesson). **Build + size-check BOTH `dist/*.whl` and `dist/*.tar.gz` < 100 MB.**

---

## 5. TF-19 wiring + tests

### 5.1 xfn-union wiring
Append `xcross_attempt_xfns()` to `pre_shot_gk_full_default_xfns` (`tracking/features.py:742`) and `atomic_pre_shot_gk_full_default_xfns` (`atomic/tracking/features.py:489`). **Not** the general `tracking_default_xfns` (Hyrum: don't add a frame-time bundled-weights/xgboost dep to the broad default — PR-S80 P3). `import silly_kicks` must still not import xgboost (lazy).

**Inert-but-wired check (review B8):** the xfn enters the `_full` bundle **regardless** of `tf19_ready` — i.e. even a weak-GK-signal column is exposed (intentional, mirrors xS wiring into the same bundle). Confirm no current consumer of `pre_shot_gk_full_default_xfns` treats every column as validated-useful (it is a VAEP feature factory — features are inputs to a learner, not asserted-informative), and record this one sentence so a future maintainer reading `tf19_ready:false` doesn't think the column was wired in error.

### 5.2 Directional CI tripwire (mirror `xshot_directional`)
Commit `tests/datasets/tracking/xcross_directional/frozen_rows.parquet` — cherry-picked maximally-separable rows (a wide-area near-byline cross-imminent state vs a quiet central state), feature columns + a `label` column (≥3 of each class). Liveness tripwire (mirror `test_bundled_model_is_live_not_degenerate`): `from_variant("default").predict_proba(rows)` → `roc_auc_score(label, p) >= 0.9` (scale-free, arch-robust). Plus `from_variant("default")` in-bounds + `metadata-matches-training-intent` (carrier params == shared constant; platform/geometry/provenance present) tests, mirroring xS's three integration tests.

### 5.3 e2e tests (`@pytest.mark.e2e`, flipped on in PR-B)
- `test_xcross_e2e` — full extract→fit→gate on real pining data; asserts the acceptance gates are computed.
- `test_xcross_cross_provider` — extractor runs per provider without crash; surface ∈ [0,1]/NaN.
- `test_surface_gk_block_ablation_runs` — `gk_block_ablation` runs and **emits both numbers** (not that GK wins).
- `test_gk_substitution_sensitivity_runs` — `gk_substitution_probe` runs, **emits GK `|Δ|` + nearest-defender control + averaged random-band** (not that it exceeds a threshold; inert → a reported `tf19_ready:false`, not a CI failure).

e2e import the eval functions directly from `silly_kicks.tracking._xcross_eval` (the B1 payoff — no script-load), calling them on a fixture-fit model; the trainer-CLI smoke (regular suite) still runs the script as a subprocess. Skip only when `PINING_FOR_THE_DATA_TOKEN` is unset.

---

## 6. ⚠️ CRITICAL PREREQ — clean 4.13.0 GS feature cache (before any training)

xCross consumes GS `score_diff`/game-state. 4.12.2 (`"O"`→fail) + 4.13.0 (own/cross-goal capture + nonEvent exclusion) corrected the GS goal stream; the **old** GS feature cache had `score_diff` in an impossible ±18 range (~640 phantom owngoals on the GS 89%-majority corpus). **Rebuild xCross's feature cache against the 4.13.0-clean GS events before training.**

- The box repo `~/Development/silly-kicks` must be at **≥4.13.0** (ff to current main / the new branch) so the GS converter is the clean one.
- **Delete any stale `xcross_attempt_v1/_feature_cache/`** on the box so Phase-1 re-extracts (the trainer auto-loads a cache if present — a stale one would silently poison the run).
- Source frames/actions per Fork B (reuse `clean_cache` iff layout matches, else fresh pining pull). The clean GS events come from the 4.13.0 converter regardless.
- After extraction, **sanity-check `score_differential`** in the rebuilt cache (review B6 — two thresholds, not one): **hard-fail** on the *impossible* range that is the phantom-owngoal signature (e.g. `|score_differential| >= 12`, which the old ±18 corruption hit) — abort the run; **soft-warn** on merely *unusual* values (`|score_differential| > 6`) — a legitimate rout (7-0, 8-1) is real, so warn-and-continue, never abort a clean run on a blowout. A probe in the run log, not a silent assumption.

---

## 7. Training run (DGX Spark — `project_pr_s80_tf16_weights_cycle` deploy)

`ssh karsten@192.168.68.73`; reuse a py3.12 venv (e.g. `~/sk-s81-venv` / `~/sk-phaseb-venv`) with `pip install -e ".[train,xgboost,kloppy]"` (add pyarrow if missing — the xS pilot caught it). Owner token passed inline through ssh (value stays out of logs; box env has none). `nohup … >log 2>&1 &` + poll `tail` (don't block).

```
python scripts/train_xcross_attempt.py \
  --providers skillcorner,idsse,gradientsports \   # OR --data-dir <clean_cache> if layout matches
  --output-dir ~/Development/xcross_refit \
  --n-trials <N> --horizon-seconds 1.0             # negative-subsample OFF (default)
```

Streams per match → caches features → HPO once per candidate → paired test → fail-closed gates → (on pass) final fit + save `xcross_attempt_v1/`. The three §3 validations run on the shipped candidate into `metrics.json`. Then publish: if `ship_two`, `full` → Hub via `publish_xcross_attempt.py` (mirror `publish_xshot_occurrence.py`) from `~/sk-s81-venv` (hf token at `~/.cache/huggingface/token`, `upload_folder(allow_patterns=...)`); the bundled `default` is fetched back to the workstation and committed.

**Horizon (PR-A design Q3):** pick by a **descriptive** wide-area-entry→cross lead-time quantile (e.g. median), NOT by maximizing held-out PR-AUC. Default `1.0 s`; record in metadata; report the lead-time profile.

---

## 8. Ship hygiene

- **Model card:** `docs/huggingface/model-cards/xcross-attempt-v1-model-card.md` (+ link from org card if a Hub repo is created). HF markdown: no `~` (strikethrough) — use "approx.". YAML frontmatter (license/tags/pipeline_tag/library_name). Only created/published **if** `full` ships to Hub; if public-only (like xS), the card documents the bundled model and **no Hub repo is created** (mirror xS exactly).
- **NOTICE:** PR-A already added the Cao et al. entry with the H2 state-vs-sender caveat — verify present; extend only if the GK-confounder finding warrants a sentence.
- **Version:** 4.14.0 → **4.15.0** (minor — new bundled weights + xfn-list behavior change). Four-file hard-gate sync: `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md` — **plus `uv.lock`**. CHANGELOG records the shipped variant, the GK-ablation/probe headline numbers, the measured `score_differential` importance, and the carrier-param retrain-trigger note.
- **ADR-011 note:** append a short "Update — TF-17 PR-B (xCross weights)" paragraph (2nd trained-model weights cycle after xS; same staged code→weights pattern; public-vs-full paired-test outcome recorded).
- **TODO.md:** delete the satisfied PR-B residual from the TF-17 row + the "xS / xCross re-fit on clean GS events" row's xCross half (xS half remains if unaddressed); leave PR-C residual.

---

## 9. Testing strategy (TDD)

- **Unit (regular suite) — clean imports from the private `_xcross_eval.py` module (no `importlib` script-load tax, the B1 payoff):** `test_gk_block_ablation_emits_two_metrics`, `test_gk_substitution_probe_emits_gk_and_two_controls` (asserts GK + nearest-defender + averaged-random-band all emitted, and the panel is geometrically-matched), `test_gk_substitution_probe_is_deterministic` (same seed → identical numbers, B7), `test_tf19_ready_reads_pinned_constants` (the gate uses `TF19_PROBE_RATIO`/`TF19_PROBE_ABS_FLOOR` from the module, not an inline literal — C1; assert both the ratio and the absolute-floor conditions fire), `test_permutation_importance_cv_held_out_and_reports_coverage` (CV-held-out fit; scoring=`average_precision`, `n_repeats=10`, fixed `random_state`; emits `score_differential` importance + `score_differential_coverage`; never permutes in-sample data — C2/B2/B3). Directional-fixture schema test (both classes, ≥3 each).
- **Integration:** bundled-model liveness tripwire (roc_auc ≥ 0.9), `from_variant("default")` in-bounds, metadata-intent, `from_variant("default")` does-NOT-cascade-to-Hub (B4 — assert it raises cleanly when bundled dir absent, mock to confirm no `snapshot_download` call), `from_hub` real-body shape (mock `snapshot_download`), xfn-union membership (`xcross_attempt_xfns()` element present in `pre_shot_gk_full_default_xfns` + atomic), `import silly_kicks` does-not-import-xgboost guard.
- **e2e (token-gated):** §5.3.
- **Lint trio (CI parity):** `uv run ruff check` + `uv run ruff format --check` + `uv run pyright silly_kicks/` (pyright pinned 1.1.409). Add a per-file ruff ignore for `train_xcross_attempt.py` if it uses `X`/`Y` (mirror the xS script entry).
- **No silent skips:** all e2e targets run with the token set (it is available) — surface any gap before `/final-review`.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Stale box feature cache silently poisons the run | §6: delete `_feature_cache/` + assert `score_differential` range in the log |
| Box repo < 4.13.0 → dirty GS events again | §6: ff box repo to the branch/main before extract; verify `__version__` |
| `full` artifact balloons sdist > 100 MB | §4: hatch-exclude `full/` on wheel **and** sdist; size-check both `dist/*` |
| Inert GK block (ablation + probe both flat) | §3.2 pre-registered contingency: ship + `tf19_ready:false` loud flag, not a build break |
| Paired test under-powered (few public games) | 17 public matches carry ≈1,150 open-play crosses (positive-rich); fail-closed `K≥2` |
| xgboost `base_score` string-format drift across versions | PR-S80 lesson: strip `[]` before `float()` if any test parses `save_config()` |
| Probe perturbation couples to ghost-GK/kde guard | §3.2: synthetic displacements only — TF-17 never consumes the ghost-GK mode |
| `score_differential` all-/mostly-NaN → meaningless importance (B2) | §3.3: report `score_differential_coverage`; qualify the CHANGELOG number; pre-run-verify provider result columns |
| Placebo structurally favors "GK wins" (feature-set asymmetry, A1) | §3.2: nearest-defender (feature-bearing) control + averaged random band + geometrically-matched panel; GK must beat BOTH |
| Probe gate masked by CV-ablation redundancy (A2) | §3: per-validation rule — `tf19_ready` gates on the probe alone; ablation/importance are context |
| `tf19_ready` "materially" chosen post-hoc = not pre-registered (C1) | §3.2: `TF19_PROBE_RATIO=2.0` + `TF19_PROBE_ABS_FLOOR=0.01` pinned in code, unit-asserted, frozen during the run |
| Importance leaks (in-sample on the all-data shipped fit) (C2) | §3.3: CV-held-out importance, labeled architecture-representative; never permuted on production training data |

---

## 11. Definition of done

1. `_xcross_eval.py` emits, into `metrics.json`: ablation (Δ PR-AUC/log-loss), substitution-probe (GK `|Δ|` + nearest-defender control + averaged random-band, seeded/fixed-N), permutation importance (**CV-held-out, architecture-representative**; pinned `scoring="average_precision"`/`n_repeats=10`/`random_state`) incl. `score_differential` importance + `score_differential_coverage`; `tf19_ready` set **from the pre-registered numeric rule** (`TF19_PROBE_RATIO`/`TF19_PROBE_ABS_FLOOR` constants — the probe vs both controls, NOT an AND over all three validations).
2. Training run complete on clean 4.13.0 GS corpus + 4.7.0 carrier defaults; acceptance gates pass; paired test decides public-vs-full.
3. `default` bundled under `_xcross_weights/default/`; `from_variant("default")` + (if shipped) `from_hub` live; both `dist/*` artifacts < 100 MB.
4. `xcross_attempt_xfns` wired into `pre_shot_gk_full_default_xfns` + atomic; `import silly_kicks` xgboost-free.
5. Directional tripwire + 3 integration + 4 e2e tests green; full non-e2e suite green; lint trio clean.
6. Model card (+ Hub publish iff full); NOTICE/CHANGELOG/version(4.15.0)/ADR-011/`uv.lock` synced; TODO groomed.
7. `/final-review` green → single commit on `pr-s84-tf17-xcross-weights` (explicit approval) → merge `--admin --squash --delete-branch` → CI green → tag.
