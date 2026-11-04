# TF-46 stoppage re-stage + xShot/xCross refit — design

Date: 2026-10-30. Author session: TF-46 Arm-1 (`tf-46-arm1-clean`). For two independent reviewers (A-O, B-O) and the owner.

This slice GOES FIRST: this session completes TF-46 Arm-1 and refits xShot/xCross before the HF-jobs compute cycle releases (owner decision 2026-10-30). Cost and wall-clock are not a constraint; correctness and best practice are. No quick hacks.

## Revision log

- **r1 → r2 (2026-10-30), after A-O + B-O spec review (both REQUEST CHANGES; approach endorsed).** Addressed the trainer-provenance gap, the `xcross_probe` miss, the false "silent" root cause, the phantom `_EMITTED_SHARD_COLUMNS`, the `_NOT_A_PIPELINE` misclassification, the preprocessed-frames attach point, and the partly-green red set.
- **r2 → r3 (2026-10-30), after A-O APPROVE + B-O REQUEST CHANGES.** A-O approved r2; B-O found two narrow blocking remainders that A-O approved past (verified correct by the author). Addressed: the §4.6 provenance transport does not survive the prep→select→fit DAG (B-01) → §4.6 now extends the shared `StudyInputs` contract; `xcross_probe` has NO producer in the repo (B-02) → §4.7 names the concrete build route; the raw-frame attach seam and its parity consequence (B-04/SPEC-13) → §4.1 + §8 name `scripts/_loader_pining.py` and state the parity behaviour; the §6(b) count needs an enforcing check (B SPEC-03); the red IDs come from a CI run in the plan (B SPEC-06); the golden has no fixture file (B SPEC-14); "superset" wording (B SPEC-16); the §3 two-seam separating clause (A cosmetic).
- **r3 → r4 (2026-10-30), after A-O APPROVE + B-O REQUEST CHANGES.** A-O approved r3's §4.6 rewrite as sound + identity-preserving. Closed B-O's one narrow remainder and the considers: the §4.1 parity paragraph was self-contradictory (SPEC-13) → rewritten to one rule (preflight runs flag-on against a regenerated flag-on reference; the generation rationale dropped); assert the probe `actions.parquet` is present (SPEC-17) → §4.3; name `adopt.py` and where the per-provider counts are recorded (SPEC-18) → §6(b) + §8; give `stoppage_sources` an `arms`-style back-compat default so a resume does not hit a missing file (A C-SPEC-01) → §4.6 + §7.

## 1. Motivation

The TF-46 Arm-1 slice adds the stoppage gate: `STOPPAGE_RULE_VERSION = 3` and `verify_stoppage_provenance` in `load()`. The shipped bundled xShot/xCross models predate the stoppage column, so they refuse to load (`IntegrityError`).

Current test state (exact IDs to be captured from a CI run in the plan, B-06): some tests are RED (bundled-model load/predict, metadata-golden), while others are GREEN BY DESIGN — the `REFUSED_UNTIL_RETRAIN` / `assert_bundle_refused` fixtures (`tests/_stoppage_fixtures.py:70,87`, consumed by `test_feature_contract.py:445` and `test_weights_bundle_golden.py:28`) assert the pre-stoppage bundles are refused. The retrain flips the second group: the refusal seam is removed and the bundles must then load. "Done" includes that flip.

A correct retrain is corpus-staging-coupled. The corpora are `xshot17` (10 skillcorner + 7 sportec — the bundled wheel default), `xshot115` (108 skillcorner + 7 sportec), and the xCross probe `xcross_probe` (gradientsports, pergame). SkillCorner dominates the first two. A correct fit needs real `stoppage_state` on the frames. Nothing attaches it today.

## 2. Scope and non-goals

In scope:
- Attach `stoppage_state` / `stoppage_source` on raw full-rate frames at staging (the `scripts/_loader_pining.py` seam that `scripts/materialize_tc3_frames.py` and the `xcross_probe` build consume).
- Re-materialize, re-stage and re-record `xshot17`, `xshot115` and `xcross_probe` with the slice converters.
- Edit both trainers and the shared `StudyInputs` contract to carry and set `model.stoppage_provenance` before `save()` (§4.6).
- Refit xShot and xCross (tiers public + sc_extended, variants default + position_only).
- Rebundle the models, flip the `REFUSED_UNTIL_RETRAIN` seam, and package the release.

Non-goals (surfaced, not silently deferred):
- Re-stage of `tc3` (ghost) and `sb360` (receiver). The schema-token bump makes their current pins stale-schema. See open decision O1.

## 3. Root cause (verified 2026-10-30; corrected after review)

- The `flat_tc3` shards (`shards/<gen>/*.parquet` + `_actions/`) are produced by `scripts/materialize_tc3_frames.py`; `scripts/hf_jobs/stage.py` only uploads them. `xcross_probe` is a separate `pergame` dataset with no producer in the repo (§4.7).
- The SkillCorner adapter sets `ball_state = "alive"` on every frame. SkillCorner, Gradient Sports and Metrica are NOT in `_DEAD_BALL_OBSERVED_PROVIDERS` (`= {"sportec", "idsse"}`).
- `stoppage_state` is written only by `derive_stoppage_state`; no producer or trainer calls it. The pins (recorded 2026-10-18) predate the slice.
- **Behaviour on the current pins is a LOUD fail-closed raise, not a silent pass** (r1 correction): `has_stoppage_state` raises for a dead-ball-unobserved provider whose frames lack the column (`_stoppages.py:647-652`); `stoppage_provenance` runs that gate first (`:688`). The extract ItemStage has `max_failures=0`, so a column-less SkillCorner match ABORTS the stage; no model is produced.
- **Two distinct silent-shrink paths exist — name them separately** (A cosmetic): (1) in the materializer, `for_each` records a raising item as a failure and carries on (`materialize_tc3_frames.py:265-267`); (2) in the trainer, `_extract_work` returns `None` (not a failure) for a match whose dead/unknown drop empties it. Path (1) is a staging concern; path (2) is a fit concern; the trainer does NOT carry on past a raise (`max_failures=0`). §6(b) guards both by counting contributing units per provider.
- Derive-at-fit-time is rejected: `event_stoppage_intervals` needs `gi_interruption_after` + marker-keyed restarts on the SkillCorner actions (`_stoppages.py:327-331`, `:246-251`), which only the slice converter emits; and ADR-123 Amendment B forbids a pining loader inside an HF-launched module. Re-stage with the slice converters is mandatory.

## 4. Design

### 4.1 Attach stoppage on raw full-rate frames (loader seam) — B-04 / SPEC-13

`derive_stoppage_state` requires full-rate frames built BEFORE smoothing/filtering; its ball-out start rule reads ball `x`/`y`. The raw full-rate frames exist only inside the pining loader builders (`scripts/_loader_pining.py`, the `_preprocess` call site at :785 = `smooth_frames` + `derive_velocities`; raw frames precede :981/:1099/:1361), before the dense-grid preprocess that the `tc3-frames-3` schema already carries. Attach `derive_stoppage_state(frames, actions=actions)` there, on the raw frames, behind an explicit STAGING flag. The flag keeps the EXTERNAL consumers unchanged (the `_loader_pining_to_cache` warmer, ad-hoc callers); it does NOT turn the attach off for the staging run itself — the preflight parity path is part of the write path and must see the stoppage columns.

Parity (SPEC-13, corrected): `assert_frames_parity` is a direct full-content checksum (`materialize_tc3_frames.py:91,:106`), generation-blind. The preflight parity path calls the module's `load_match` directly (`:149-153`), so it must load with the SAME staging flag as the write path — otherwise its flag-off frames lack `stoppage_state`/`stoppage_source` and parity raises "missing column(s)" (`:98-100`) on every staging run. ONE rule: preflight runs flag-on AND the `--reference-parquet` is regenerated flag-on, so parity keeps covering the stoppage-bearing write path (the module docstring's stated job, `:17-21`). The r3 cross-generation rationale was wrong and is dropped. §7 names the preflight parity test.

### 4.2 Schema token

Bump `_SHARD_SCHEMA_VERSION` `"tc3-frames-3"` → `"tc3-frames-4"` (`token_inputs["schema"]` already references it, so the bump lands a NEW generation — verified). `_EMITTED_SHARD_COLUMNS` does NOT exist here; `_work` returns the full frames DataFrame as the shard, so the schema is the frame column set (`TRACKING_FRAMES_COLUMNS`). Add a first-shard SUPERSET presence check that `stoppage_state` and `stoppage_source` are present on the returned frame (not a whitelist select). Fix the module docstring, which says the frames are written "unchanged".

### 4.3 Re-materialize, re-stage, re-record

- Re-materialize `xshot17` (17), `xshot115` (115) and the `xcross_probe` GS matches via pining with the slice converters. SkillCorner / Sportec use the public token; GS needs the owner token (`PINING_FOR_THE_DATA_TOKEN`). Full corpus; do not shrink.
- Assert the materialized shard count equals the expected unit count (17 / 115 / probe) before staging (`for_each` carries on past a failure, §3 path 1). For `xcross_probe`, also assert each unit's `actions.parquet` is present — `_work` writes `_actions` only when non-empty (`materialize_tc3_frames.py:309-311`) and the xCross probe consumer reads `files[1]` (SPEC-17).
- Upload with `stage.py` to `silly-kicks/hf-refit-xshot17` (public), `silly-kicks/hf-refit-xshot115` (restricted), `silly-kicks/hf-refit-xcross-probe` (restricted). Each push needs explicit owner approval.
- Re-record: `python -m scripts.hf_jobs datasets record {xshot17,xshot115,xcross_probe}` → new generation, unit list, corpus identity into `datasets.lock.json` (written only by `record`).

### 4.4 Fit

Run `scripts/hf_jobs/campaigns/cycle1.toml`: `xshot_occurrence` + `xcross_attempt`, tiers public (`xshot17`) + sc_extended (`xshot115`), variants default + position_only, xCross probe `xcross_probe`. Once frames carry `stoppage_state`, `prepare_*` takes the `has_stoppage_state` path and drops dead/unknown frames.

### 4.5 Rebundle and package

- Publish each model through `publish_model_with_card` (ADR-088); each publish needs explicit owner approval. sc_extended artifacts are Hub (non-wheel) but still required.
- Remove the `REFUSED_UNTIL_RETRAIN` / `assert_bundle_refused` seam and flip those tests to assert the bundles LOAD. `test_weights_bundle_golden.py` has no committed fixture file (it is a load + chirality check, `:27-38`, B-14); regenerate exactly what it verifies against the new weights. The metadata-golden is REPLACED by the refit, not reproduced from the old.
- Version bump (owner's call). `CHANGELOG.md`, including the GS `z_source` `"measured"` → `"estimated"` Hyrum note. `NOTICE`.

### 4.6 Trainer stoppage provenance via the StudyInputs contract (B-01 route) — blocking

Neither trainer sets `model.stoppage_provenance`; `save()` raises `RuntimeError` when it is None. The `_SIDE_COLS` route (r2) does NOT reach the fit stage: `_prep_work` persists through `persist_study_inputs`, whose contract is fixed `(X, y, groups, providers, match_ids, is_public, config, arms)` (`scripts/_study_shared.py`), and `_fit_work` reads `load_study_inputs` → `StudyInputs` (a dataclass, fields `providers`/`match_ids`/…). An extra side column dies at that boundary. The correct route:

- Capture per-match `stoppage_source` in `_extract_work` (frames in scope), carried into `_prep_work`.
- Extend the shared contract: add a `stoppage_sources` field to `StudyInputs`, persisted by `persist_study_inputs` as a per-match-aligned `stoppage_sources.npy` parallel to `match_ids.npy`, and read back by `load_study_inputs`. Additive and written/read symmetrically, so the fan-out byte-identical rebuild is preserved. Give it the `arms`-style back-compat default (tolerate an absent `stoppage_sources.npy` in `load_study_inputs`) or bump the prep stage version, so a resume over a pre-change prep shard does not hit a missing file — resume identity does not hash the inputs dir (A C-SPEC-01).
- At `_fit_work`, select the shipped arm's rows (the `arms` mask + `match_ids`, the registered fixed sequence in `scripts/_paired.py`), take their distinct `stoppage_sources`, `combine_stoppage_provenance`, and set `model.stoppage_provenance` before `save()`.
- Both `xShotOccurrenceModel` and `xCrossAttempt`, both position_only variants. `scripts/_study_shared.py` is listed in §8; §7 tests a real prep→fit pass.

### 4.7 `xcross_probe` staging (B-02 route) — blocking

No `xcross_probe` producer exists in the repo (only `datasets.toml:36-42`, the `train_xcross_attempt.py` consumer, and the `_toy_corpora.py:641` toy; the current probe's provenance is off-repo). Concrete build route, named: materialize the Gradient Sports probe matches as a flat corpus via `scripts/materialize_tc3_frames.py` (with the §4.1 stoppage attach) → convert to the `pergame` layout with `scripts/hf_jobs/repackage.py::flat_to_pergame` (`{provider}-{game}/{frames.parquet,shots.parquet}`) → `stage.py` upload to `hf-refit-xcross-probe` → `datasets record`. GS resolves via `events_v3` (needs GS actions with restart + goal rows; confirm the slice GS converter emits them — O6). Restricted dataset; owner-gated push. Listed in §8.

## 5. Blast radius / Hyrum

- The schema-token bump invalidates the current `flat_tc3` generation for ALL datasets (`tc3`, `xshot17`, `xshot115`, `sb360`). `tc3` / `sb360` not re-staged here → stale-schema pins (O1).
- `datasets.lock.json` identity changes; an artifact's recorded corpus identity must match at `adopt` (INV-9). Ghost/receiver artifacts on the OLD generation stay valid against their own recorded identity until re-staged.
- Extending `StudyInputs` changes a contract shared by both trainers and the study fan-out; it is additive and identity-preserving (§4.6).
- The GS `z_source` demotion is a separate value-changing provenance note, already owed in the slice.

## 6. Validation — the "done" proof (pre-registered, not deferred to the owner)

DONE requires ALL:
- (a) The RED tests clear (exact IDs captured from a CI run in the plan — the bundled-model load/predict and the metadata-golden), AND the `REFUSED_UNTIL_RETRAIN` seam is removed and its tests flipped to assert load, AND the regenerated `test_weights_bundle_golden` check passes.
- (b) The SkillCorner (and GS probe) units' recorded `stoppage_sources` include `events_v3` — proof the dead-ball filter applied — AND a per-provider CONTRIBUTING-UNIT count equals the expected corpus (17 / 115 / probe). The per-provider counts are recorded in the artifact `metrics.json`; `adopt` reads them and refuses a short corpus (not only the §4.3 materialize shard count, B SPEC-03 / SPEC-18). Report the per-provider `stoppage_source` counts and the SkillCorner dead/unknown drop fractions.

## 7. Testing (TDD — tests first)

- `materialize`/loader attach `stoppage_state`/`stoppage_source` on raw full-rate frames; the first-shard superset presence assertion fires when a column is missing; the token bump forces a new generation; the shard-count assertion fires on a short corpus.
- The preflight parity test (SPEC-13): a flag-on load against a regenerated flag-on reference passes; a flag-off load against a flag-on reference raises "missing column(s)" — proving preflight tracks the write path.
- Trainer provenance through a REAL prep→fit pass: `StudyInputs` carries `stoppage_sources`; `_fit_work` combines the shipped arm and stamps `stoppage_rule_version=3` + `events_v3`; `save`→`load` passes `verify_stoppage_provenance`; `save()` still raises when provenance is unset (regression guard).
- `derive_stoppage_state` on a slice-converter SkillCorner toy yields `events_v3` + non-degenerate dead/unknown spans; on a stale-actions toy (no `gi_interruption_after`) raises.
- A call-site guard that the actions carry no `t_kick` (no "load seam" exists today — B-07).
- `xcross_probe` GS build produces stoppage and the probe stage no longer raises.
- A per-provider contributing-unit-count assertion (§6(b)) on a toy corpus.
- The existing stoppage/xShot/xCross unit tests stay green (count stated by exact family at impl, not "140").
- `datasets record` round-trip (new generation + identity); preflight live-vs-lock.
- CI-faithful pytest, the ruff trio, `pyright` before any push.

## 8. File inventory

- `scripts/_loader_pining.py` — raw-frame stoppage attach behind a staging flag (§4.1); the other consumers unchanged.
- `scripts/materialize_tc3_frames.py` — `_SHARD_SCHEMA_VERSION` → `tc3-frames-4`; first-shard superset presence assertion; `assert_frames_parity` cross-generation note; docstring fix.
- `scripts/hf_jobs/repackage.py` (`flat_to_pergame`) + the `xcross_probe` staging route (§4.7).
- `scripts/_study_shared.py` — `StudyInputs` + `persist_study_inputs` + `load_study_inputs` gain `stoppage_sources` (§4.6).
- `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` — capture per-match source in `_extract_work`; combine + set `model.stoppage_provenance` in `_fit_work` before `save()`.
- `tests/_stoppage_fixtures.py`, `test_weights_bundle_golden.py`, `test_feature_contract.py` — remove `REFUSED_UNTIL_RETRAIN`; flip to assert-load; regenerate the golden check.
- `scripts/hf_jobs/datasets.lock.json` — re-recorded by `datasets record`.
- `scripts/hf_jobs/adopt.py` — reads the per-provider contributing-unit counts from the artifact `metrics.json` (§6(b)) and refuses a short corpus (SPEC-18).
- `CHANGELOG.md`, `NOTICE`, `silly_kicks/_version.py` — packaging.
- new / extended tests under `tests/scripts/` and the trainer provenance test (§7).

## 9. Workflow and commit gate

Feature branch `tf-46-arm1-clean`. One commit per cycle, fully tested. `require_clean_tree(git_provenance())` before any artifact write. Every branch commit and push, every HF dataset push, every HF campaign launch and every model publish needs explicit owner approval. Process: this spec → independent A-O and B-O review → owner approval → TDD plan → implementation → independent A/B implementation review → owner commit gate.

## 10. Open decisions (owner / reviewers)

- O1: Re-stage `tc3` (ghost) + `sb360` (receiver) now for one coherent generation, or defer to their next refit. Recommended: defer, recorded as an owed coherence item.
- O3: Confirm full-rate frames are available from pining for every SkillCorner / Sportec / GS match in scope (`derive_stoppage_state` requires full-rate).
- O4 (carried, A4 / B-03): record the cut scope as a short ADR (one ADR with the stoppage ADR, or two — B-12).
- O5: §4.1 attaches on raw frames via the `_loader_pining.py` seam behind a flag (recommended). Confirm at impl, or measure that smoothed `x`/`y` do not move the spans and document the delta.
- O6: Confirm the slice Gradient Sports converter emits the restart + goal actions `event_stoppage_intervals` needs for `xcross_probe` (§4.7).
- Minor (SPEC-10): the `xshot17`/`sb360` lock `generation` is a short string `"g"`, not a token hash; explain the mapping in the plan. Non-blocking.

## 11. ADR

Record the stoppage-staging decision (raw-frame loader-seam attach; schema token `tc3-frames-4`; `StudyInputs` provenance transport) as an ADR under `docs/superpowers/adrs/` (next number ADR-130; one ADR or two with O4), cross-referencing `STOPPAGE_RULE_VERSION = 3`, ADR-052, ADR-123 Amendment B and ADR-128.
