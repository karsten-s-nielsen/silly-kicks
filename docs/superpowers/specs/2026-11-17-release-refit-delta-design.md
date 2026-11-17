# The 4.144.0 Release Refit: Delta on Plan Rev 3 (design, r4)

**Status:** draft for A/B review. Owner: Karsten. Cycle PR-S209. Replaces the withdrawn
`2026-11-17-release-refit-design.md`. **Basis:** plan `2026-11-13-xshot187-refit-plan.md` (rev 3) and the directive
`D:\Development\_reviews\2026-11-17-release-refit-cycle-supervisor-directive-C.md` (scope, 2a/2b/4a, bars B1-B9; B2
resolved by R20 and design X; B2 and B8 cover the receiver). This spec states the changes against plan rev 3 (steps
that B9 removes: 4.2); everything else stands. Step 2 is done (directive 4a). Limit: 130 lines (supervisor).

## 1. Decisions in force

R5, R7, R8 (as written), R11, R16, R19, the directive, and (owner, 2026-11-17): R20, in a retrain run the public CV
parity is one-sided with the existing band (5 % relative), any improvement passes; R21, a spec with `cv_tolerance`
stays two-sided in every mode; R22, `receiver_model` joins the retrains (it pools the GS units of `tc3`,
`train_receiver_model.py:415-416`, `:423-431`); R23, the census is re-made; R24/R25/R26, the decision rule of 3.3.
Option H (owner's proposal, supervisor-confirmed): `win_probability` and `match_outcome_dependence` run in the
campaign to prove that they do not move. Open with the owner: the token step (4.4).

## 2. Branch and commits

New branch `feat/release-refit-144` from `6eee84ce`; `feat/xshot187-refit` (`2a8e5ef3`, cited by ADR-133) and PR #305
stay as the record. C1 = `2a8e5ef3`'s content (the ADR-132 item 10 expectation edit, `manifest.loads`, `verify`
`manifest_sha256`, tests, howto, ADR text) + sections 3 and 4 + the new lock entries. One C2: the adoption (Task 7).

## 3. Code (bar B5: at most 65 added code lines, 135 added test lines in total; the counts at C1 are reported)

### 3.1 Option A: the per-provider bound (R10)

`provider_skill` (`silly_kicks/tracking/_artifact_integrity.py:205-243`) adds per provider a `bootstrap` block with
eight keys: `unit`, `draws`, `seed`, `level`, `brier_gain_lb`, `log_loss_gain_lb`, `frac_brier_gain_le_0`,
`frac_log_loss_gain_le_0`; `n_matches` stays beside it. It uses the pooled constants (`SKILL_BOOTSTRAP_SEED`,
`SKILL_BOOTSTRAP_DRAWS`, `SKILL_LOWER_PERCENTILE`, `SKILL_LEVEL`), the pooled draw call `rng.integers(0, n, n)`, a
new `default_rng(SKILL_BOOTSTRAP_SEED)` per provider in sorted order, and per-match sums. No new constant. The
docstring "No bootstrap ..." becomes a report-only statement; ADR-132 item 5 (`:47`), its rejected alternative (`:93`)
and corpus-widening spec 5.7 get an amendment note (rejected as a gate, kept as a report). Tests: a two-provider unit
case against an inline computation; `skill_gates` gives the same result with and without the block. Bar B4: on
`~/xshot187/gatecheck/store` the values equal `~/dl-review-xdef/gatecheck_oracle.json` to 9 decimals.

### 3.2 The mode on the job (design X)

- `scripts/hf_jobs/campaign.py`: an optional per-job `mode` (validated like `arms`), default = the campaign mode;
  values `rematerialize`, `refit`, `retrain`, valid at both levels; `Instance.mode`. Only `verify` reads it.
- `scripts/hf_jobs/adopt.py`: `verify` passes `inst.mode` (today `c.mode`, `:789`). `retrain` = the standing
  certification (`_certify`, `floor.certify`) with the public CV parity one-sided; no acceptance gating.
  `adopt.py:860` does not change: a `retrain` job of an integrity family runs the integrity check and fails closed.
- `scripts/hf_jobs/floor.py`: the one-sided form calls `floor.regression_check(committed, fresh,
  direction=spec.metric_direction, delta=PUBLIC_CV_REL_TOL * abs(committed))`; no second comparison. A spec with
  `cv_tolerance` stays two-sided (R21; `registry.py:75-77`). The detail records `form: "one_sided" | "two_sided"`.
- Tests: the per-job mode; the one-sided band (better passes, worse beyond it fails, NaN fails); `cv_tolerance` stays
  two-sided. `test_campaign.py:160-162` refuses another value than `"retrain"`.

### 3.3 The census (R23): rebaseline path and decision rule

- Path, no new logic: the census runs at C1; `verify` reports its difference (`ok` false) with the diffs; Karsten
  rules; the adoption copy (4.1) leaves it out; C2 copies the fresh output into `docs/research/territory_ranking_census`
  by hand and sets `REBASELINE_FROM` (`adopt.py:54`) to the C1 commit (one line); the parity run at C2 finds it
  identical or inside the tolerance (`registry.py:343-345`): both `ok`. C1 stays reachable from `main`: `--merge`.
- R25 (the form of R24): `_rebaseline` (`adopt.py:526-561`) passes a difference only inside a declared tolerance
  (`ok = within`); every other difference is `ok` false and stops for an owner ruling. The classification (`delta`,
  `changed_modules`) stays in the report as information. So no input list decides: code, data, dependency and deleted
  modules are all covered; `input_contract()` does not change. R26: `changed` is computed only when `spec.name` has a
  `REBASELINE_FROM` entry (today `REBASELINE_FROM[spec.name]` raises `KeyError` for `cross_definition`,
  `registry.py:353-355`); else `delta: None`. Tests: a difference outside the tolerance with no changed module is not
  `ok`; a driver without an entry gets `ok` false, no exception. Docs: the `_rebaseline` docstring, the write-phase
  comment (`adopt.py:1251`), ADR-128 P15 (`:227`), `docs/howto/hf-jobs-training.md:176-180` (item 17 stays history).

## 4. Configuration, data and runs

### 4.1 The campaign (`scripts/hf_jobs/campaigns/tf46_refit.toml`, edited in place, R2)

- Campaign `mode = "refit"`; the header comment states the new content. 21 instances, datasets as in `cycle1.toml`
  except xShot/xCross: `xshot_occurrence` and `xcross_attempt`, tiers `public` (`arms = ["public", "sc_extended"]`)
  and `sc_extended`, on `xshot187` (+ `xcross_probe`), mode `refit`; mode `retrain`: `xsuccess` (first job, the long
  pole, `cycle1.toml:45-46`), `pass_completion`, `ghost_gk` (5), `ghost_outfield` (2), `receiver_model`; mode
  `rematerialize`: `win_probability`, `match_outcome_dependence`, `build_territory_ranking_census`.
- The bucket, `storage_limit_bytes = 1_000_000_000_000` (4.2) and `cycle1.toml`'s sizing tables (`:28-43`):
  `mount_cap_by_dataset` `tc3 = 6`; `peak_rss_by_stage` 8 GiB for `ghost_gk/extract`, `ghost_gk/cache`,
  `xsuccess/fit`; `units_per_job_by_stage` `"xsuccess/fit" = 3`. `test_campaign.py:165-175` pins all 21 modes.
- Expected basis: `refit_fresh_acceptance` (8 xShot/xCross); `cv_parity` (ghost x7, receiver); `public` one-sided
  (PassCompletion, XSuccess); `public` two-sided (win_probability; match_outcome by `cv_tolerance`); `rebaseline`.
- Reading, declared now: `compare` of the complete retrieved folder against launch 7, before `adopt`. win_probability
  and match_outcome IDENTICAL = not owed, not adopted; any difference is a STOP before `adopt` and an owner ruling.
- B8 needs no code: a `fit` stop leaves no output, so `verify` skips it (`adopt.py:818-824`); a `verify` fail blocks
  `adopt` (`:1210-1212`). The complete retrieved folder stays unchanged (R8); after the owner's decision (R19),
  `verify` and `adopt` run on a copy that holds only the instances to adopt.

### 4.2 Data and storage (directive 2a, 2b, B6, B9)

- New private repositories in the PRO user namespace: `karstenskyt/hf-refit-xshot187`, `karstenskyt/hf-refit-tc3`,
  `karstenskyt/hf-refit-xcross-probe`; bucket `karstenskyt/hf-refit-scratch`. Every create passes `private=True`;
  the plan reads the visibility back before the first upload. Dataset names stay; only `repo` changes.
- Upload sets: `xshot187` = `~/cross-def/xshot187-f5fc2e2b/out` as it is (the plan records `run_commit f5fc2e2b` and
  every file hash); `tc3` = `~/refit144/tc3/out` (generation `0f43213bb5bccd5c`, `run_commit 6eee84ce`, schema 30 ->
  33 columns); the probe = `~/refit144/probe/pergame`. Remote hashes equal the local ones; then `datasets record`.
- No Hub delete, prune or squash (B9; directive 2b.3): the delete of R3 (Task 7 Step 7) and Task 8 Step 1 fall away;
  R3's move of every `xshot17`/`xshot115` reference stays. Task 3 Step 1 checks `check_storage(org="karstenskyt",
  1e12, 20e9)`; its budget table becomes one line: `karstenskyt/` usage before the first launch and after the last.

### 4.3 Corners (directive 3.6, B3)

The committed `build_corner_routines` runs on the DGX at C1 (clean tree, pining data), then `validate_corners`. B3: its
gates read as in the 4.139.0 memo, corner counts per provider against 4.139.0. C2 adopts the bundles (plan rev 4).

### 4.4 Owner gates and order

In the order of plan rev 3, each with Karsten's yes in this session: each create (bucket, three repositories) and
upload; each `datasets record` (the lock goes into C1); C1 commit; push and PR; the launch; the census ruling; the
corner adoption; C2 commit and push; the parity launch (Task 8, at C2; `cycle1.toml` takes the bucket and the
`storage_limit_bytes` of 4.2, directive 2b.2); merge (`--merge`); tag; each publish. Token: `hf auth switch`, no code.

## 5. Bar B6: every campaign instance and the corners

| Instance(s) | Last fit (data) | Changes on the extract path to v4.144.0, or why it does not move |
|---|---|---|
| xShot x4 | `a1a3b52` (xshot17/xshot115) | `f8a7c47e` ADR-130, `96ebe960` ADR-131, `c381e7da` ADR-132, `f5fc2e2b` ADR-133 |
| xCross x4 | `a1a3b52` | the same four, `53e529b7` (7 trainer lines) |
| ghost-GK x5 | `a1a3b52` (tc3 at `251ab741`) | `abcfda86` ball-carrier pre-index, `f8a7c47e`, `f5fc2e2b`, the tc3 restage |
| ghost-outfield x2 | `a1a3b52` (tc3) | `abcfda86` (`infer_ball_carrier`), `f8a7c47e`, `f5fc2e2b`, the tc3 restage |
| receiver | `a1a3b52` (sb360 + tc3 GS pool) | `f5fc2e2b`, the tc3 restage |
| PassCompletion | `a1a3b52` (sb_open) | `f5fc2e2b` |
| XSuccess | `a1a3b52` (sb_open) | `abcfda86` (fit units), `f5fc2e2b` |
| census | 4.120.0 (`dd7ed4f`) | fits `PassCompletionModel` on pass rows (`build_...census.py:503-504`): moves with `f5fc2e2b` |
| win_probability | `a1a3b52` | not expected to move: shot types only (`train_win_probability.py:43`, `:97`, `:124`); measured in the run, before C2 |
| match_outcome | `a1a3b52` | not expected to move: shot types only (`train_match_outcome_dependence.py:58-71`); measured in the run, before C2 |
| corners (DGX) | `6ee7e627` (pining) | `b2865a72` ADR-128, `f8a7c47e`, `f5fc2e2b` (`6a99f5b8` changes only the bundles) |

`tc3` was staged at `251ab741` (pre-squash TF-65 C1; main `a34e992e`); the restage takes `5c544305`, `6f6508a2`,
`30a6e439`, `e39a2c5b`, `b2865a72`, `f8a7c47e`, `f5fc2e2b`. Launch 8 (`abcfda86`) rematerialized every campaign family
(not corners), IDENTICAL to launch 7 (fit-units spec 2026-11-10, 17.3). `z_source` (static check at `6ee7e627`): the
pining converters of the 4.139.0 fit wrote it (`silly_kicks/tracking/skillcorner.py:357` via `_loader_pining.py:1007`,
`silly_kicks/tracking/sportec.py:154` via `:1119`; SB360 `_snapshot.py:179`; unchanged at v4.144.0): no second reason.
