# Corpus widening and the visibility-flag split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task
> (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

Status: r3 APPROVED by reviewers A and B (r3 re-reviews, no open finding) and by the owner on 2026-11-12
("Approve both"). It carries spec r8 (the provider source: the row-aligned `_provider` / `inp.providers`, not
`groups`; A CW-SPEC-16 / B CW-SB-19) and answers the plan r2 re-reviews (A CWP-PLAN-07; B CW-PB-05). r2 answered plan
reviews A (CWP-PLAN-01, -02, CONSIDERs) and B (CW-PB-01 to -04) and carried spec r7 (owner ruling r7).

**Goal:** Split the visibility flag into raw-redistributable and publicly-obtainable registries, widen the xShot/xCross
public arm (89) and extended arm (187) with the probe held out, add the two-arm ship rule, the per-provider gate,
per-job arms, the dataset fields and checks, the adoption contract for both rule outcomes, and the size gates.

**Architecture:** `scripts/_corpus.py` owns three registries and the label; `scripts/hf_jobs/arm_policy.py` builds the
arms off-HF and `campaign.py` restricts them per job; the per-provider skill math sits beside `forecast_skill` in
`silly_kicks/tracking/_artifact_integrity.py`; the trainers gain a two-arm paired path (`scripts/_paired.py`);
`datasets.py` / `datasets.toml` carry `raw` + `provenance`; `adopt.py` / `manifest.py` read the new `shipped_arm`. No
committed bundle, pin, lock entry or manifest entry changes; `xshot187` is not declared (restage).

**Tech Stack:** Python (requires-python `>=3.10`; CI matrix 3.10-3.12), numpy, the ADR-128 stage pipeline,
`huggingface_hub` (`HfApi.dataset_info`), GitHub Actions (`publish.yml`).

**Spec:** `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (r7). "Spec N" below refers to its sections.

## Global Constraints

- Starts only after Part Deux's PR #302 has merged; branch `feat/corpus-widening` rebased on that `main`. Line
  numbers below are on `e5573a40`; re-locate each after the rebase before editing.
- One feature branch, no worktree, no subagents, ONE commit for the cycle after `/final-review`, the A/B
  implementation review and the owner's yes (the gate question names commit + push + draft PR; CI runs the full
  suites on the draft PR, owner ruling 2026-11-12).
- Registries (spec 3): `RAW_REDISTRIBUTABLE` = today's `PUBLIC_CORPUS` (SkillCorner 20 + IDSSE 7);
  `PUBLICLY_OBTAINABLE` = `RAW_REDISTRIBUTABLE` + gradientsports `3812`-`3859` and `10502`-`10517` (64);
  `HELD_OUT_PROBE` = gradientsports `10502`, `10503`.
- Arms (spec 5.1): `public` = obtainable minus probe; `sc_extended` = `public` + licensed SkillCorner; never `full`.
- Constants: `artifact_size_lt_5mb` limit `5_000_000` bytes; release limit `80_000_000` bytes per `dist/` file;
  per-provider skill uses `SKILL_LOG_CLIP = 1e-15` and strict `> 0` (`_gt0`), no bootstrap (spec 5.7).
- Version: the wheel changes (`_artifact_integrity.py`, spec 5.7), so a minor bump of `silly_kicks/_version.py`; the
  number is set at commit prep, not before. CHANGELOG entry keyed `PR-Snnn` at commit prep. STE prose; CI-scope lint.
- Commands. Sync the tree to the DGX checkout with `bash /d/Temp/hfu-topup/sync.sh` (never while another run reads
  `~/sk-fsg`). Tests: `ssh dgx 'cd ~/sk-fsg && .venv/bin/python -m pytest <files> -q -p no:cacheprovider'`.
  Lint: `ssh dgx 'cd ~/sk-fsg && .venv-lint/bin/ruff check silly_kicks/ tests/ scripts/ && .venv-lint/bin/ruff format
  --check silly_kicks/ tests/ scripts/ && .venv-lint/bin/pyright'`. "Expected: FAIL" means the new tests fail on an
  `ImportError`/`AttributeError` or the named assertion, and no other test in the files fails.

## Plan rulings (for the A/B review)

- **R1.** `fixed_sequence_ship` (three arms) is removed: `compute_arms` never emits `full` (spec 5.1). A hand-written
  `--arms-json` that contains `full` is refused in each trainer's prep (fail loud), NOT in `arms.parse_arms`, which
  stays a generic parser (`test_hf_arms.py:32` unchanged). Cost if wrong: one function to restore.
- **R2.** The release-size logic lives in `scripts/check_dist_sizes.py` (testable on a stub `dist/`, spec test 13),
  called by the new `publish.yml` step; "80 MB" = `80_000_000` bytes (decimal; below PyPI's 100 MiB either way).
- **R3.** The size gate is checked in `_fit_work` after the artifact is built, from the same `fit.files` byte total
  that already feeds `artifact_size_bytes` (xshot `:978`, xcross `:1232`); on failure it writes the FAILED payload
  with `artifact_size_lt_5mb: false` and raises, like the skill gates.
- **R4.** `validate_cover_shadow_rq1.py:178` and `validate_pass_risk_calibration.py:95` carry no per-row match ids;
  they label a GS-only pass corpus by hand (today `full`). They pass `all_obtainable=True`: the registry's GS part is
  the whole GS WC2022 corpus (64 games, the `tc3` lock). Their NEXT record reads `public` (Q-D); committed records stay
  `full`. Cost if wrong: a GS game outside the 64 mislabelled; Task 1 pins the 64 to the lock.
- **R5.** Spec r7 7.2 holds the complete list of edited gates, grouped by the decision that causes each edit. Each task
  below names the items it owns; Task 10 Step 1 greps for leftovers and every hit is either edited or in the spec's
  "unchanged" list.
- **R6.** `artifact_label` drops its `providers` parameter: with `full` retired nothing reads it. Every caller changes
  its keyword anyway (`all_public` → `all_obtainable`); callers are `scripts/` only, and `_corpus` has no entry in
  `docs/PRIVATE_CONSUMERS.md`. Cost if wrong: one parameter to restore.
- **R7.** xCross `--ship-variant` (`train_xcross_attempt.py:486`) drops `full` from its choices (no `full` arm exists).
- **R8.** The provider's training-fold rate (spec 5.7) is computed in `_study_shared.cv_unit` from the fold's training
  positions `tr` (the same rows behind today's `train_rate = y[tr].mean()`) and the arm's row-aligned provider
  identities, so the fold functions and `silly_kicks/tracking/_xcross_eval.py` keep their signatures. A test asserts
  the pooled rate equals `train_rate`.
- **R9.** `validate_trained_artifact_integrity.py` folds the two provider gates into `calibration_ok` (they are Brier
  and log-loss skill, like the ADR-131 members of that bucket). The committed verdict files are not regenerated.

## Review Focus

1. A legacy `--arms-json` with a `full` arm must refuse loudly in prep, never train a third arm (Task 5,
   `test_a_full_arm_is_refused`).
2. A one-arm job on a strict subset of the units must refuse at preflight (spec 5.5; Task 2
   `test_single_arm_strict_subset_is_refused`, Task 3 `test_preflight_refuses_a_strict_subset_job`).
3. `HfApi.dataset_info` failing (network) during the private-repo check must refuse, not pass (Task 6,
   `test_private_check_fails_closed_on_api_error`).
4. A committed manifest entry without `shipped_arm` must behave exactly as today (Task 7,
   `test_entry_without_shipped_arm_reads_as_tier`).
5. The release check must see BOTH files and fail when `dist/` lacks one (Task 8, `test_empty_dist_fails`).

---

### Task 1: the registries and the label (spec 3, 4)

**Files:** Modify `scripts/_corpus.py` (incl. its comments at `:15-18`, `:102-103`, `:150-165`),
`scripts/_coordination_corpus.py:255-268`, `scripts/validate_cover_shadow_rq1.py:178`,
`scripts/validate_pass_risk_calibration.py:95` (R4), `scripts/hf_jobs/arm_policy.py:24-46` (the import),
`scripts/hf_jobs/registry.py:60` (comment), and the spec 7.2 "rename" and "label" items:
`tests/scripts/test_corpus_guard.py`, `test_corpus_taxonomy.py`, `test_corpus_identity_split.py:6`,
`test_xshot_visibility_blind.py:21-22`, `test_xcross_visibility_blind.py:21-22`,
`tests/scripts/hf_jobs/test_visibility_blind_gate.py:13`, `test_coordination_corpus_visibility.py:13, :37`,
`test_derive_coordination_params.py:499`, the comments `tests/_skillcorner_sample.py:18` and
`tests/invariants/test_input_convention_detector.py:331`, `tests/scripts/hf_jobs/test_datasets_lock.py:65` (docstring),
`docs/context/conventions-core.md:38`. Do NOT edit `tests/fixtures/claude_md_at_77286f4.md`. Test
`tests/scripts/test_corpus_registries.py` (new).

**Interfaces — Produces:** `RAW_REDISTRIBUTABLE`, `PUBLICLY_OBTAINABLE`, `HELD_OUT_PROBE` (`dict[str, frozenset[str]]`,
identity-provider keys); `is_obtainable_row(*, providers: np.ndarray, match_ids: np.ndarray) -> np.ndarray` (fail-closed
bool mask); `artifact_label(*, all_obtainable: bool) -> str` (`"public"` / `"sc_extended"`, R6);
`unit_identity(unit: Mapping[str, str]) -> tuple[str, str]` (identity provider via `_CONVERTER_TO_IDENTITY`, game with a
pergame `<provider>-` prefix stripped).

- [ ] **Step 1: Write the failing tests** in `test_corpus_registries.py`: `test_gs_obtainable_equals_the_tc3_lock`
  (`PUBLICLY_OBTAINABLE["gradientsports"]` == the GS games of the committed lock `tc3`, via `unit_identity`);
  `test_probe_equals_the_xcross_probe_lock` (`HELD_OUT_PROBE` == `{unit_identity(u) for u in xcross_probe units}` ==
  `{("gradientsports","10502"),("gradientsports","10503")}`); `test_raw_is_inside_obtainable`;
  `test_unregistered_is_neither` and `test_unregistered_public_still_refuses` (spec test 2: `assert_public_corpus`
  with a pining-`public` GS match → `SystemExit`); `test_labels` (spec test 3: GS-only → `public`; GS + public
  SkillCorner → `public`; any licensed SkillCorner → `sc_extended`; no input yields `full`);
  `test_committed_full_records_untouched` (the six committed `docs/research/**` files carrying `"full"` still carry it;
  Q-D) and `test_full_label_still_renders` (spec test 3 "loads and verifies": no code validates the label vocabulary;
  the one reader, the card-field renderer at `scripts/hf_jobs/manifest.py:300-305`, renders a `metrics.json` with
  `corpus_visibility: full` as the line `` `corpus_visibility: full` ``).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** in `_corpus.py`: rename `PUBLIC_CORPUS` → `RAW_REDISTRIBUTABLE` everywhere (no alias);
  add the registries and functions above; `artifact_label` callers pass `all_obtainable` (coordination: every used
  pair obtainable; cover-shadow / pass-risk per R4). `assert_public_corpus` keeps reading pining and
  `RAW_REDISTRIBUTABLE`. The edited tests assert the new names and the new label values for their pairs.
  `BUNDLED_PUBLIC_ARM` and `rehydrate_public_corpus_ids` unchanged (spec 7).
- [ ] **Step 4: Run** the new test file and every test file listed under Files. Expected: PASS.

### Task 2: `compute_arms` and the arm restriction (spec 5.1, 5.5)

**Files:** Modify `scripts/hf_jobs/arm_policy.py:30-69`, `tests/scripts/hf_jobs/test_arm_policy.py:41-53` (F29
restated, spec 7.2); Test `tests/scripts/hf_jobs/test_arm_policy.py`.

**Interfaces — Consumes:** Task 1 registries, `unit_identity`. **Produces:** `compute_arms(units) -> {"arms": {...},
"ship_order": [...]}` with only `public` / `sc_extended`; `restrict_arms(payload: dict, arms: Sequence[str], units) -> dict`
(raises `ValueError` for an arm the payload lacks, or for ONE arm that is a strict subset of the non-probe `units`).

- [ ] **Step 1: Write the failing tests:** `test_public_is_obtainable_minus_probe`, `test_sc_extended_adds_licensed_skillcorner`,
  `test_no_full_arm` (a GS-containing fixture), `test_probe_unit_in_no_arm`, `test_arms_follow_staged_units` (dropping
  one staged unit changes the counts, not the code; spec 6.4), `test_restrict_both_arms_runs_paired`,
  `test_restrict_unknown_arm_refused`, `test_single_arm_strict_subset_is_refused` (Review Focus 2),
  `test_committed_shapes_pass` (`xshot17` → `["public"]`, `xshot115` → `["sc_extended"]`, from the committed lock);
  F29 becomes: for `xshot17`, `xshot115`, `tc3`, the union of the arms == the trainer keys of the NON-probe units,
  and no probe unit's key is in any arm.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** both functions; `ship_order` restricted to the job's arms in `public`, `sc_extended` order.
- [ ] **Step 4: Run** `test_arm_policy.py`, `test_hf_arms.py`. Expected: PASS.

### Task 3: campaign per-job `arms` (spec 5.5, 7.2)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`CampaignJob` gains `arms: tuple[str, ...] | None = None`; parsing
`:110-120`; `:202-203` passes `restrict_arms(...)`; preflight refusal), `scripts/hf_jobs/campaigns/tf46_refit.toml`,
`scripts/hf_jobs/campaigns/cycle1.toml`; Test `tests/scripts/hf_jobs/test_campaign_arms.py` (new) and the existing
campaign tests that build `--arms-json`.

**Interfaces — Consumes:** Task 2 `restrict_arms`.

- [ ] **Step 1: Write the failing tests:** `test_gated_job_without_arms_refused`, `test_job_gets_only_its_arms`
  (the `--arms-json` argument carries exactly the declared arms), `test_preflight_refuses_a_strict_subset_job`
  (a one-arm `public` job on a fixture dataset with licensed units is refused at preflight),
  `test_committed_campaigns_keep_todays_arms` (`tf46_refit` / `cycle1`: wheel jobs on `xshot17` → `["public"]`, Hub
  jobs on `xshot115` → `["sc_extended"]`; their `--arms-json` equals today's payload for those datasets).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**; add `arms` to every visibility-gated job of the two committed campaign files.
- [ ] **Step 4: Run** the new file + every `tests/scripts/hf_jobs/test_campaign*.py` and `test_launch*.py`. Expected: PASS.

### Task 4: the per-provider skill core (spec 5.7)

**Files:** Modify `silly_kicks/tracking/_artifact_integrity.py` (beside `forecast_skill`, `skill_gates`),
`scripts/_study_shared.py` (`cv_unit`, `cv_unit_fileset`, `read_cv_unit`, `cv_skill`), `scripts/_corpus_identity.py`
(beside `canonical_providers`), the cv stages of both trainers (xshot `:808-819`, xcross `:970-984`: pass `tr` and
`inp.providers[rows]`), `scripts/validate_trained_artifact_integrity.py:87-117` (R9), and the spec 7.2 pins
`tests/tracking/test_forecast_skill.py:26-34, :240-243`, `tests/scripts/test_nested_gate_folds.py:203-211`,
`tests/scripts/test_validate_trained_artifact_integrity.py:48-58, :87-94`, and the fold-unit round-trip tests; Test
`tests/tracking/test_provider_skill.py` (new), `tests/scripts/test_cv_unit_provider_rate.py` (new).

**Interfaces — Produces:** `provider_skill(y: np.ndarray, p: np.ndarray, r: np.ndarray, providers: np.ndarray, groups:
np.ndarray, fold: np.ndarray) -> dict[str, dict[str, Any]]` (`r` = each row's provider-own reference; `providers` =
row-aligned identities; fields `n_matches` (distinct `(provider, group)`), `n_oof_rows`, `n_positive`, `row_share`,
`brier_skill`, `log_loss_skill`, `pr_auc_lift`); `skill_gates` gains `provider_brier_skill_gt_0`,
`provider_log_loss_skill_gt_0` (from `metrics["skill"]["by_provider"]`; missing or empty → False);
`provider_identity(provider: str) -> str` (`_CONVERTER_TO_IDENTITY.get(provider, provider)`; raises `ValueError` on an
empty string; `canonical_providers` reuses it); `cv_unit(out, positive_rate, tr, te, y, groups, providers)` persists
`providers.npy` (held-out identities) and `train_rate_by_provider: dict[str, float]` (R8); `cv_skill` adds
`skill["by_provider"]`.

- [ ] **Step 1: Write the failing tests** (spec test 15 core; every fixture uses bare game-id `groups` and a separate
  provider column with `sportec` rows): `test_every_provider_skilled_passes`; `test_one_provider_unskilled_fails`
  (pooled `brier_skill` > 0, one provider's ≤ 0 → `provider_brier_skill_gt_0` False); `test_provider_rate_forecast_has_zero_skill`
  (p = each row's provider training-fold rate: pooled-reference skill > 0, provider skill exactly 0, gate False);
  `test_nan_provider_skill_fails`; `test_missing_by_provider_fails`; `test_by_provider_keys_are_identities`
  (keys `{"idsse", "skillcorner", ...}`: never `sportec`, never a game id); `test_empty_provider_refuses_the_unit`;
  `test_pooled_rate_equals_train_rate`; `test_unit_round_trips_providers_and_rates` (FileSet write/read);
  `test_validator_calibration_fails_on_provider_gate` (validator `calibration_ok` False and GO False when a provider
  gate fails).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The edited pins assert the new key sets; the validator fixtures gain `by_provider`.
- [ ] **Step 4: Run** the two new files + the pinned files above and `tests/scripts/test_integrity_counters.py`.
  Expected: PASS.

### Task 5: the two-arm paired path, provider reporting and the size gate (spec 5.2, 5.4, 5.6, 5.7)

**Files:** Modify `scripts/_paired.py` (remove `fixed_sequence_ship`; add `two_arm_ship`), `scripts/train_xshot_occurrence.py`
(`:619` predicate, `_paired_data_effect` per-provider deltas, `_select_work`, `_fit_work`, `_assemble_work`, prep
refusal of `full`), `scripts/train_xcross_attempt.py` (`:774`, `:486` per R7, the same stages), `scripts/_toy_corpora.py:524-534`
(two arms; also feeds `:1118`, `:1128` and the parity tests); the spec 7.2 "two-arm rule" items
`tests/scripts/test_paired_rules.py`, `tests/scripts/test_nested_gate_folds.py:38-79, 317-349`,
`tests/scripts/test_expect_variant_wiring.py:105-124`; Test `tests/scripts/test_two_arm_ship.py` (new).

**Interfaces — Consumes:** Task 4 `skill_gates`, `cv_skill`. **Produces:** `two_arm_ship(sc_extended_deltas:
Sequence[float]) -> tuple[str, str]` (`("sc_extended", why)` iff `clears_rule(deltas)`, else `("public", why)`);
`ARTIFACT_SIZE_LIMIT_BYTES = 5_000_000` in `scripts/_study_shared.py`; `paired["by_provider"][P]` = per-fold deltas
on P's public held-out rows, P from `provider_identity(inp.providers)` (Task 4; NaN allowed).

- [ ] **Step 1: Write the failing tests** (both trainers where it applies): `test_two_arm_ship_clears` /
  `test_two_arm_ship_does_not_clear` (`[0.1, 0.2, -0.01, 0.1, 0.05]` → `sc_extended`; `[0.1, -0.2, -0.01, 0.1, 0.05]`
  → `public`); `test_run_paired_needs_public_and_sc_extended`; `test_paired_select_ships_the_rule_choice` (stub
  `_paired_data_effect`; the shipped params are that arm's full-rows study); `test_a_full_arm_is_refused` (Review
  Focus 1); `test_probe_held_out` (spec test 6); `test_provider_deltas_reported_not_deciding` (spec test 16: the
  decision is the same with and without `paired["by_provider"]`); `test_provider_gate_fails_the_run` (a shipped arm
  failing `provider_brier_skill_gt_0` writes the FAILED payload and raises; no fallback to the other arm);
  `test_size_gate_fails_closed` (5_000_000 bytes → FAILED payload with `artifact_size_lt_5mb: false`, raises;
  4_999_999 passes and `acceptance["artifact_size_lt_5mb"]` is True). The three edited tests assert two arms
  (`cv_arms == ["public", "sc_extended"]`, study tags and `inp.arms` without `full`).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `--ship-variant` (xCross) otherwise unchanged. Folds, K and deltas are today's
  `_public_folds` / `_paired_data_effect` (spec 5.2), restricted to the two arms.
- [ ] **Step 4: Run** the new file + the three edited files, `test_parity_xshot_occurrence.py`,
  `test_parity_xcross_attempt.py`, `test_objective_fold_split.py`, `tests/tracking/test_xshot_occurrence_integration.py`,
  `tests/tracking/test_xcross_attempt_integration.py`. Expected: PASS.

### Task 6: dataset fields and checks (spec 6.1, 6.2)

**Files:** Modify `scripts/hf_jobs/datasets.py` (`Dataset.visibility` → `raw: str`, `provenance: str`; loader `:60-70`;
`record`: provenance check, `public_ids` rule `:281`, private-repo check), `scripts/hf_jobs/datasets.toml` (the six
entries, values per spec 6.2), `scripts/hf_jobs/campaign.py:204-208` (`--corpus-visibility` = `public` iff
`provenance == "public"`, else `restricted`) and the campaign mount preflight (private-repo check),
`tests/scripts/hf_jobs/test_datasets_lock.py:14`, `tests/scripts/test_train_receiver_model.py:409`; Test
`tests/scripts/hf_jobs/test_datasets_fields.py` (new).

**Interfaces — Produces:** `require_private(repo: str, *, api) -> None` (raises unless `dataset_info(repo).private`;
an API error raises too).

- [ ] **Step 1: Write the failing tests:** `test_fields_required` (missing `raw` or `provenance` refused; the old
  `visibility` key refused); `test_receiver_mapping` (sb360 → `restricted`, sb_open → `public`);
  `test_provenance_checked_at_record` (spec test 12: a fixture tracking dataset declared `public` with a licensed
  unit is refused; one with only obtainable units, probe units included, passes; the StatsBomb datasets keep their
  declared values); `test_public_ids_recorded_for_public_arm_units` (fixture record writes `public_ids` =
  `compute_arms`' public arm; the committed lock is not touched); `test_private_repo_required` (stub api
  `private=False` refused; `sb_open`'s upstream `statsbomb/open-data` exempt; its `snapshot_repo` checked);
  `test_private_check_fails_closed_on_api_error` (Review Focus 3); `test_campaign_preflight_refuses_public_staged_repo`
  (spec test 7, mount half: campaign preflight over a fixture campaign whose staged repo reports `private=False`
  refuses before any launch).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The committed lock is not re-recorded.
- [ ] **Step 4: Run** the new file + `test_datasets_lock.py`, `test_adopt.py`, `test_train_receiver_model.py`.
  Expected: PASS.

### Task 7: the adoption contract (spec 7.1)

**Files:** Modify `scripts/hf_jobs/manifest.py` (`Artifact.shipped_arm: str | None = None`; load; `effective_shipped_arm(a)
-> str` = `a.shipped_arm or a.tier`), `scripts/hf_jobs/adopt.py` (`:351-353`, `:130`, `:648`, `:684`, `:697` read
`effective_shipped_arm`; `:976` prints it beside `tier`); `floor.py` unchanged (it receives `shipped_arm` as its
argument); Test `tests/scripts/hf_jobs/test_adopt_shipped_arm.py` (new).

**Interfaces — Consumes:** Task 3 campaign instances.

- [ ] **Step 1: Write the failing tests:** `test_entry_without_shipped_arm_reads_as_tier` (Review Focus 4);
  `test_wheel_and_hub_instances_never_cross` and `test_adopt_each_outcome` (spec test 14: from a campaign fixture on a
  fixture `xshot187`, the wheel instance `tier=public` and the Hub instance `tier=sc_extended` of one `(job, variant)`;
  distinct store keys and ids; outcome `public`: wheel entry `public_ids`, rehydrate with the fixture's locked
  `public_ids`, certification basis `public`; outcome `sc_extended`: `digest`, no rehydrate,
  `floor.certify`'s `detail["basis"] == "cv_parity"`; each output only in its own destination);
  `test_shipped_variant_must_match_shipped_arm` (incl. an `sc_extended` run on an entry with no `shipped_arm`).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file + `test_adopt.py`, `tests/test_bundled_weights_corpus_policy.py`. Expected: PASS.

### Task 8: the release size check (spec 5.6)

**Files:** Create `scripts/check_dist_sizes.py`; Modify `.github/workflows/publish.yml` (a step after `python -m build`,
before the upload), `tests/test_ci_publish_guard_wired.py`; Test `tests/scripts/test_check_dist_sizes.py` (new).

**Interfaces — Produces:** `check(dist: Path, limit: int = 80_000_000) -> list[str]` (problems; prints each file's
size); `main(argv) -> int` (exit 1 on any problem).

- [ ] **Step 1: Write the failing tests:** `test_both_files_under_limit_pass`, `test_sdist_over_limit_fails`,
  `test_wheel_over_limit_fails`, `test_limit_boundary` (a file of exactly 80_000_000 bytes fails, 79_999_999 passes;
  sparse files via `truncate`), `test_empty_dist_fails` (Review Focus 5: no wheel or no sdist fails); in
  `test_ci_publish_guard_wired.py`: the build job runs `python scripts/check_dist_sizes.py dist` after
  `python -m build` and before `actions/upload-artifact`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.

### Task 9: records

**Files:** Create `docs/superpowers/adrs/ADR-132-corpus-provenance-split.md`; Modify `docs/context/corpus-drivers.md`
(the corpus paragraph), `docs/context/trained-models.md` (the per-provider gate beside ADR-131), ADR-038 and ADR-131
(status notes: amended by ADR-132), `AGENTS.md` (the bullet citing ADR-038 for `artifact_label` and the ADR-131
skill-gate bullet gain ADR-132; `tests/test_agents_md_budget.py` must pass), `CHANGELOG.md`, `silly_kicks/_version.py`
(minor bump at commit prep).

- [ ] **Step 1:** ADR-132: the two registries and why, the label as provenance, `full` retired, the probe hold-out,
  the two-arm rule, the per-provider gate (reference, constants, why), per-job arms, the dataset fields and
  private-repo rule, the adoption contract (`tier` = channel, `shipped_arm`), the size gates; supersedes D4; owner
  rulings Q-A to Q-E, r2 to r4, r7.
- [ ] **Step 2:** CHANGELOG entry and version at commit prep; run `tests/test_agents_md_budget.py`. Expected: PASS.

### Task 10: verification, reviews and the commit gate

- [ ] **Step 1:** Leftover grep (R5): `rg -n "PUBLIC_CORPUS|fixed_sequence_ship|artifact_label\(providers|\]\.visibility|\"full\": keys"
  scripts tests docs/context AGENTS.md -g '!tests/fixtures/**'`; every hit is edited or in spec 7.2's "unchanged"
  list (which names the `corpus_identity` `all_public=` callers and the SB360 `entry.visibility`). Lint trio (Global Constraints); the
  tests of Tasks 1-9; spec test 10 (committed bundles unchanged): `test_corpus_guard.py`, `test_datasets_lock.py`,
  `test_bundled_weights_corpus_policy.py`, `test_bundled_probe_ids_are_committed_ids.py`. Expected: 0 lint errors, PASS.
- [ ] **Step 2:** `/final-review` (before the A/B review, owner ruling).
- [ ] **Step 3:** A/B implementation review (`/review-impl`); fix or answer every finding.
- [ ] **Step 4:** Owner's yes to "commit, push and open a draft PR so CI runs?"; one commit; CI green before
  ready/merge, tag and release (each its own yes; wheel and sdist sizes checked by the new step).
- [ ] **Step 5:** Message Part Deux: the `xshot187` definition (SkillCorner 118 + IDSSE 7 + GS 62; private repo;
  `raw = restricted`, `provenance = extended`), declared with its first `datasets record`; the refit campaign with
  per-job `arms` and `mode = "refit"`.
