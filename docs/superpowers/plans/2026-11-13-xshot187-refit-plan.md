# The 4.144.0 Release Refit Implementation Plan (rev 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task
> (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** retrain every model that 4.144.0 (ADR-133) makes owed, on data staged from the release, and ship one
release with them.

**Architecture:** the ADR-128 campaign tooling, the ADR-131/132 gates and the standing certification (`floor.certify`).
This cycle adds three small code changes (the per-provider bound, the mode on the job, the rebaseline rule) and no new
gate, dataset kind or pipeline.

**Tech Stack:** Python 3.12 (`.venv-hfjobs`), `huggingface_hub`, HF Jobs `cpu-upgrade`, the DGX (`ssh
karsten@192.168.68.73`), pining.

**Spec:** `docs/superpowers/specs/2026-11-17-release-refit-delta-design.md` (r3, the delta on rev 3), the directive
`D:\Development\_reviews\2026-11-17-release-refit-cycle-supervisor-directive-C.md` (scope, 2a/2b/4a, bars B1-B9), and
for the carried Task 1: `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (r8) section 7.1, ADR-132.
Runbook: `docs/howto/hf-jobs-training.md`.

**Owner rulings.** R1 (the DGX for heavy work; parallel workers), R2 (edit `tf46_refit.toml` in place), R4, R5
(the parity proof: approved), R7, R8 (as written), R11 (retention: a separate cycle), R16 (keep the supervisor
informed), R19 (a failed gate: STOP and report; the owner decides) stand. R3: only its move of every `xshot17` /
`xshot115` reference stays; its Hub delete leaves the cycle (B9). R6 is replaced by the new branch. R20 to R24 and
option H are in spec section 1. R10 (option A) stays in the form of the directive. R9 (resume), R13/R14 and
R15/R17/R18 are superseded by the directive.

**Revision log.** rev 4 implements delta spec r3 (A/B reviewed; reports
`D:\Development\_reviews\2026-11-17-release-refit-delta-spec-*.md`) and directive C. rev 3 was owner-approved
(2026-11-13; A/B r3 APPROVE); its first run is on `feat/xshot187-refit` at `2a8e5ef3` (draft PR #305, the record).
As run (R6, owner 2026-11-13, pause): all 4 xCross arms of the `2a8e5ef` run failed `provider_brier_skill_gt_0` and
`provider_log_loss_skill_gt_0` while every pooled gate passed (the provider-dependent `cross` label); the driver exited
1 at 2026-11-13T21:45Z with 4/8 instances done (xShot) and 4 STUCK (xCross); `retrieve` copied the 4 xShot arms to
`~/xshot187/staging` as a record; nothing was adopted. 4.144.0 (ADR-133) fixed the label.

## Global Constraints

- Owner yes, per action, in this session: every commit, push, PR action, merge, tag, Hub repository or bucket create,
  Hub upload, `datasets record`, HF job launch, the census ruling, the corner adoption (after the B3 reading), Hub
  publish. No Hub delete, prune or squash in this cycle (B9).
- One branch `feat/release-refit-144` from `6eee84ce`. C1 = the campaign commit that the jobs run; C2 = the adoption,
  a NEW commit on top of C1 (never an amend). Merge with `--merge` (the census `REBASELINE_FROM` names C1).
- `/final-review` before each A/B implementation review. A/B review this plan, C1 and C2; reports in
  `D:\Development\_reviews\`. Send the supervisor each result named in "Evidence" below.
- Never the full test suite and never full `pyright` on the AI-PC; the DGX runs every test file that the change
  reaches (the Task 7 Step 1 selection), the lint trio and `pyright`; CI is the full-suite gate.
- Temp files under `D:\Temp\xshot187\` (AI-PC), `~/refit144/` and `~/xshot187/` (DGX). Nothing changes in
  `~/cross-def`, `~/xshot187/out`, `~/xshot187/gatecheck`, `~/xshot187/staging` or `~/dl-review-xdef`.
- Do not touch the HF write token, the historical wheels or the off-repo audit bundle. Eyestone NDA Rule #1.
  Databricks is not a source.
- Every new Hub repository and the bucket are PRIVATE, created with `private=True`, read back before the first upload.
- Size (bar B5): at most 65 added code lines and 135 added test lines in total for sections 3.1-3.3 of the spec; the
  carried C1 content does not count. Above a limit: stop and ask the supervisor.
- Write in STE. Version at C2 commit prep; one CHANGELOG entry keyed `PR-S209`, in C2.

## Evidence (the supervisor checks these from the artifacts)

E1 the B4 oracle comparison; E2 the visibility read-back of the bucket and the three repositories; E3 the remote
hashes against the local ones after each upload; E4 the measured line counts at C1; E5 `verify.json` of the complete
retrieved folder (21 rows, each with its `basis`; `form` is only in a public `cv_parity` detail:
`certification.cv_parity.form`, or `certification.comparison.cv_parity.form` in a public-tier refit row; an
owner-certified row and the rebaseline row have none); E6 `compare` against launch 7 before `adopt`; E7 the census
differences for the P15 ruling; E8 the B3 corner reading; E9 `karstenskyt/` usage before the first launch and after
the last; E10 the parity `compare` (B7); E11 the sha256 of every hand-copied file (corner bundles, census output)
against the run output.

## Review Focus

1. A `retrain` job of an integrity family: the integrity check runs and fails closed (`adopt.py:860` unchanged).
2. A public CV value that improves by any amount passes; one that worsens beyond the band fails; NaN fails.
3. A spec with `cv_tolerance` (match_outcome) stays two-sided in every mode.
4. A rebaseline difference outside the tolerance with no changed module: `ok` false (R25); no baseline entry: no crash.
5. One stopped instance: the others adopt from a copy of the retrieved folder; the complete folder stays unchanged.

## Design: the expectation edit (Task 1)

**The gap.** Spec §7.1 orders the adoption: (1) read each run's `shipped_variant`; (2) write `shipped_arm` and the
outcome fields into the manifest; (3) `verify`, then `adopt`. `verify` reads the working-tree manifest (`mf.load()`,
`adopt.py:779`) and needs no clean tree. `adopt` refuses a dirty tree and a HEAD other than the campaign commit
(`adopt.py:1086-1094`), and its rehydrate reads the public ids from the entry's `dataset` (`adopt.py:983`). The edit
cannot be in C1 (the run chooses `shipped_arm`; `dataset = "xshot187"` on the committed 17-match bundles fails
`test_the_staged_public_ids_are_the_bundled_public_arm`). A commit after C1 moves HEAD and the run keys. So step 3
can never run today. Both reviewers confirmed this.

**The rule.** `adopt` accepts ONE uncommitted change, `docs/artifacts/manifest.toml`, under rules 1 to 10. Rule 11
binds `verify` to `adopt` on every path.

1. A clean root (empty `git status --porcelain --untracked-files=all`) returns `None` FIRST, before any campaign or
   HEAD read. Today's clean path is unchanged after that.
2. A root whose status set is not exactly `{"docs/artifacts/manifest.toml"}` (staged or unstaged) also returns
   `None`, before any campaign or HEAD read. Today's checks then refuse it with today's messages (`adopt needs a
   clean tree`, `the root tree is not clean`), so every existing refusal test and message stays unchanged. Only a
   root that changed the manifest alone enters rules 3 to 10.
3. HEAD is the campaign commit. Else refuse (`is not the campaign commit`).
4. Both manifests load through `mf.loads` (HEAD's text from `git show HEAD:docs/artifacts/manifest.toml` through
   `run`; the working text from disk). A load error refuses.
5. Every top-level table other than `artifacts` is equal (parsed TOML). A comment-only change passes and is
   reviewed in the C2 diff.
6. The `artifacts` list has the same ids in the same order.
7. The campaign entries are `{a.id for inst in instances(c, commit) for a in artifacts_for(inst, head_man)}`, from
   HEAD's manifest (an edit of `job`, `variant` or `tier` cannot move an entry into the set). An entry outside the
   set is equal.
8. A campaign entry is equal after both sides drop `_EXPECTATION_FIELDS = frozenset({"dataset", "shipped_arm",
   "certification", "corpus", "expect"})` (spec §7.1's table). `tier` is the channel; `training_commit` is the one
   tool-written field (INV-9).
9. No removal: every `_EXPECTATION_FIELDS` key and every `expect` key present at HEAD stays present. A value may
   change; an `expect` key may be added (a stricter check).
10. Value domains, checked on every expectation value PRESENT in each campaign entry of the edited manifest
    (changed or not; entries outside the campaign are covered by rule 7, so their `blocked` and `legacy`
    certifications are never checked here): `dataset` is declared in `datasets.toml` and locked
    (`datasets.locked` returns its record); `shipped_arm` ∈ {`public`, `sc_extended`}; `corpus` ∈ {`public_ids`,
    `digest`}; `certification` ∈ {`public`, `owner`} (spec §7.1's values).
11. `verify` reads the manifest bytes once, parses them with `mf.loads`, and writes `manifest_sha256` into
    `verify.json`: the sha256 of the canonical JSON (`sort_keys=True`, compact separators) of the parsed TOML with
    every artifact's `training_commit` removed. `training_commit` is the one field `adopt` writes (INV-9), so the
    digest of a committed adoption equals the digest `verify` recorded, and the fixed point of
    `test_adopt_writes_only_allowlisted_paths_and_is_idempotent` (`:325-329`) holds. A comment or line-ending
    change does not move the digest; every value change does. `adopt` computes the same digest of the working
    manifest at entry and refuses when `verify.json` has no `manifest_sha256` or a different one (`the manifest
    changed after verify; run verify again`). This holds on the clean path too, so a `verify.json` written before
    this release is refused (fail closed; `verify` again).

On a write-phase failure, `_rollback` restores every path to HEAD and then writes the pre-adopt manifest bytes back.
Idempotence holds: a second `adopt` over an uncommitted adoption sees the bundles and `observed.json` beside the
manifest and refuses at rule 2.

**Rejected:** a third commit for the edit plus a `--campaign-commit` flag on `retrieve`, `verify` and `adopt` (more
code, three commits, the same result).

---

### Task 1: the expectation edit and the verify binding (carried from `2a8e5ef3`; built, A/B r2 APPROVE)

**Files:**
- Modify: `scripts/hf_jobs/manifest.py` (`loads(text: str, *, base: Path) -> Manifest`; `load` delegates to it)
- Modify: `scripts/hf_jobs/adopt.py` (`verify` at `:764-799`; `adopt` at `:1064-1113`; `_rollback` at `:910`; new
  `_EXPECTATION_FIELDS`, `_expectation_edit`, `_manifest_sha256`)
- Test: `tests/scripts/hf_jobs/test_adopt.py`
- Modify: `docs/howto/hf-jobs-training.md` (step 3: the one exception to "keep the tree clean until `verify`";
  step 7: `verify.json` holds `manifest_sha256`; step 8: the expectation edit and its rules),
  `docs/superpowers/adrs/ADR-132-corpus-provenance-split.md` (item 10; Status "Accepted; amended 2026-11-13")

**Interfaces:**
- Produces: `manifest.loads(text: str, *, base: Path) -> Manifest`; `adopt._manifest_sha256(text: str) -> str` (rule
  11's digest of a manifest's text);
  `adopt._expectation_edit(c: Campaign, commit: str, *, run: Callable[..., Any], root: pathlib.Path) -> bytes |
  None`; `adopt._rollback(run, root, *, keep: Mapping[str, bytes] | None = None) -> list[str]`; the
  `verify.json` key `manifest_sha256`.

**Fixtures** (all in `test_adopt.py`; `_TOY_MANIFEST` and every existing assertion stay unchanged):
- `_toy_repo` writes `"manifest_sha256": A._manifest_sha256(<the committed toy manifest text>)` into
  `verify.json`. This is the only change to an existing fixture. Every existing test and assertion stays unchanged:
  the fixed point (`:325-329`) holds by rule 11's `training_commit` exclusion; `:322-323` and
  `test_an_entry_refusal_keeps_uncommitted_work` (`:402-412`) keep their messages by rule 2 (both roots have more
  than the manifest changed, so today's checks answer them).
- `_toy_campaign(tmp_path) -> Campaign`: `C.load_campaign` on a tmp TOML with `[campaign] name = "toy"`, a bucket,
  `concurrency_cap = 1`, `attempts = 1`, and one job `ghost_gk` / `owner` / `tc3` / `["default"]` (as in
  `cycle1.toml`). Its `instances(c, C)` produce `(ghost_gk, default, owner)`, so the campaign entries are `toy_wheel`
  and `toy_hub`.
- `_add_entry(root, text)`: appends an `[[artifacts]]` block to the toy repo's manifest and commits it, before the
  test's own edit. Used for `toy_other` (`ghost_gk` / `sweeper` / `owner`, `training_commit = "b" * 40`), which no
  toy instance produces, and for a `toy_wheel` variant with `corpus` and an `expect` key at HEAD.
- `_dirty()`: REUSE the existing helper (`test_adopt.py:398`). Its `dirty_files` value is not read on the edit
  path (rule 2 decides from the root's own status), so no second definition is added.
- `_reverify(root, staging)`: rewrites `verify.json`'s `manifest_sha256` from the current working manifest (the
  operator runs `verify` again after the edit).

- [ ] **Step 1: Write the failing tests.** Each new test name starts `test_expectation_edit_`, and each asserts its
  own message, never only the exception type.

```python
def test_expectation_edit_loads_equals_load(): ...
    # mf.loads(MANIFEST.read_text(), base=REPO_ROOT) == mf.load() on the committed manifest
def test_expectation_edit_is_adopted(tmp_path):
    # toy_wheel: certification "owner" -> "public", add shipped_arm = "public", dataset = "tc3"; _reverify; adopt
    out = A.adopt(_toy_campaign(tmp_path), C, staging=staging, git=_dirty, root=root, gate=False, now=NOW)
    text = (root / "docs/artifacts/manifest.toml").read_text(encoding="utf-8")
    assert 'shipped_arm = "public"' in text and f'training_commit = "{C}"' in text and out["adopted"]
@pytest.mark.parametrize("field", ["training_commit", "destination", "tier", "refit_event", "commit_field", "card"])
def test_expectation_edit_refuses_a_non_expectation_field(tmp_path, field): ...  # names the entry id and the field
def test_expectation_edit_refuses_an_edit_of_another_entry(tmp_path): ...        # toy_other's dataset; names toy_other
def test_expectation_edit_refuses_a_change_outside_the_artifacts(tmp_path): ...  # a new [events.x]; names "events"
def test_expectation_edit_refuses_a_changed_artifact_list(tmp_path, change): ... # parametrize: add, remove, reorder
def test_expectation_edit_refuses_a_removal(tmp_path, removed): ...              # parametrize: corpus, an expect key
def test_expectation_edit_refuses_an_undeclared_dataset(tmp_path): ...           # dataset = "nowhere"
def test_expectation_edit_refuses_a_value_outside_its_domain(tmp_path, field, value): ...
    # (shipped_arm, "full"), (corpus, "none"), (certification, "legacy")
def test_expectation_edit_refuses_a_second_change(tmp_path, other): ...          # parametrize: tracked, untracked
    # a PRESERVATION test: today's message ("clean tree" with git=_dirty) before and after the change; it is green
    # in Step 2 by design, and mutant (j) (rule 2 accepts a second path) turns it red
def test_expectation_edit_refuses_a_manifest_changed_after_verify(tmp_path): ... # edit without _reverify
def test_expectation_edit_refuses_a_verify_report_without_a_digest(tmp_path): ...# clean root, digest removed
def test_expectation_edit_refuses_a_foreign_head(tmp_path): ...                  # "is not the campaign commit"
def test_expectation_edit_rollback_keeps_the_edit(tmp_path, monkeypatch):
    # the write phase fails (the pattern of test_adopt_fails_on_a_path_outside_the_allowlist)
    assert (root / "docs/artifacts/manifest.toml").read_bytes() == edited
    assert _porcelain(root) == " M docs/artifacts/manifest.toml"
def test_expectation_edit_second_adopt_refuses_at_the_status(tmp_path): ...      # names a bundle path
def test_expectation_edit_verify_records_the_manifest_digest(tmp_path): ...      # verify.json carries manifest_sha256
```

- [ ] **Step 2: Run them (AI-PC, one file):** `.venv-hfjobs/Scripts/python.exe -m pytest
  tests/scripts/hf_jobs/test_adopt.py -q -k expectation_edit`. Expected: every new test FAILS: `manifest.loads` and
  `A._manifest_sha256` do not exist (`AttributeError` in the fixtures), so this first RED is a collection-level
  failure. Then implement `manifest.loads` and `A._manifest_sha256` for real (the fixtures call them; they are
  pinned by `test_expectation_edit_loads_equals_load` and rule 11's text), add `_expectation_edit` as a stub that
  returns `None`, and run again. Expected: each test fails on its own assertion or message, never on a fixture
  error, except
  `test_expectation_edit_refuses_a_second_change` (green by design, see its comment).
- [ ] **Step 3: Implement** the design's rules 1 to 11. `adopt` calls `_expectation_edit` first: on `None`, today's
  checks run unchanged; on bytes, it skips `require_clean_tree` and keeps the HEAD check. Then rule 11 runs on both
  paths, before `_write_phase`. `_rollback` gets `keep={"docs/artifacts/manifest.toml": pre}`. Update the docstrings
  of `verify`, `adopt` and `_rollback`.
- [ ] **Step 4: Run the whole file:** the Step 2 command without `-k`. Expected: all PASS, the existing tests
  unchanged.
- [ ] **Step 5: Mutants** (scratch `D:\Temp\xshot187\mut\`), each fails at least one new test: (a) no field check;
  (b) no campaign-entry restriction; (c) no top-level comparison; (d) no id-list check; (e) no removal check; (f) no
  `dataset` declaration check; (g) no value-domain check; (h) no digest check; (i) no `keep` in `_rollback`; (j) a
  second path accepted; (k) the campaign read moved before the clean-root return (the existing `_Campaign()` tests
  fail).
- [ ] **Step 6: Docs:** the howto lines and ADR-132 item 10 as listed under Files.


### Task 0: the branch and the carried C1 content

**Files:** the 14 files of `2a8e5ef3` (`git show --stat 2a8e5ef3`).

- [ ] **Step 1:** save this rev 4 and the delta spec outside the tree (`D:\Temp\xshot187\`); restore the tracked plan
  (`git checkout -- docs/superpowers/plans/2026-11-13-xshot187-refit-plan.md`; the plan file is absent at `6eee84ce`,
  so a modified copy blocks the switch); `git switch -c feat/release-refit-144 6eee84ce`; `git cherry-pick --no-commit
  2a8e5ef3` (no file of `2a8e5ef3` is in `0cc07e6c..6eee84ce`: no overlap).
- [ ] **Step 2:** keep the expectation edit (`adopt.py`, `manifest.py`), its tests, the howto and ADR-132 item 10.
  Drop the old `xshot187` lock entry and the old `tf46_refit.toml` body (Tasks 5 and 6 rewrite them). Put back this
  rev 4 as the plan file and add the delta spec (`2026-11-17-release-refit-delta-design.md`); the withdrawn spec r1 is
  not committed. Expected: `git diff --cached --stat` lists only these files.
- [ ] **Step 3:** run `tests/scripts/hf_jobs/test_adopt.py` (AI-PC, one file). Expected: PASS on the new base.

### Task 2: the per-provider bound (spec 3.1, R10)

**Files:** Modify `silly_kicks/tracking/_artifact_integrity.py` (`provider_skill`); Test
`tests/tracking/test_provider_skill.py`; Modify `docs/superpowers/adrs/ADR-132-corpus-provenance-split.md` (`:47`,
`:93`), `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (5.7 note).

- [ ] **Step 1: failing tests.** `test_provider_bootstrap_block_has_the_eight_keys`;
  `test_provider_bootstrap_equals_an_inline_cluster_bootstrap` (two providers, values against an inline loop with
  `default_rng(SKILL_BOOTSTRAP_SEED)` per provider); `test_skill_gates_ignore_the_provider_bootstrap` (same gates with
  and without the block). Update `_FIELDS` (`test_provider_skill.py:56`). Run: FAIL (no `bootstrap` key).
- [ ] **Step 2: implement** per spec 3.1 (pooled constants, `rng.integers(0, n, n)`, per-match sums). Run: PASS.
- [ ] **Step 3: B4 (E1)** on the DGX: the new `provider_skill` on the OOF rows of the xCross `sc_extended` candidate,
  variants `default` and `position_only` (units `sc_extended__default`, `sc_extended__position_only`, stage `single`,
  read with `rows(units(...))` of `~/dl-review-xdef/gatecheck_probe.py`, imported read-only with `python -B` /
  `PYTHONDONTWRITEBYTECODE=1` so no `__pycache__` lands there; providers
  `gradientsports`, `idsse`, `skillcorner`), against `~/dl-review-xdef/gatecheck_oracle.json`. Expected: equal to 9
  decimals.
- [ ] **Step 4:** the docstring, the ADR-132 and spec 5.7 notes (rejected as a gate, kept as a report).

### Task 3: the mode on the job (spec 3.2, design X, R20, R21)

**Files:** Modify `scripts/hf_jobs/campaign.py`, `scripts/hf_jobs/adopt.py` (`:789`), `scripts/hf_jobs/floor.py`,
`docs/superpowers/adrs/ADR-130-stoppage-gate.md` (item 5, `:56-59`: the third mode, the mode on the job, R20, R21; its
"8 instances"), `docs/howto/hf-jobs-training.md` (`:63`); Test `tests/scripts/hf_jobs/test_campaign.py`,
`tests/scripts/hf_jobs/test_floor.py`.

- [ ] **Step 1: failing tests.** `test_job_mode_defaults_to_the_campaign_mode`;
  `test_job_mode_accepts_the_three_values`; `test_campaign_mode_accepts_retrain`;
  `test_job_mode_refuses_another_value`; `test_retrain_public_cv_parity_is_one_sided` (better by 50 % passes; worse by
  more than 5 % fails; NaN fails; `form == "one_sided"`); `test_cv_tolerance_stays_two_sided_in_retrain`;
  `test_verify_reads_the_instance_mode`. `test_campaign.py:160-162`: refuse `"nonsense"` instead of `"retrain"`.
  Run: FAIL.
- [ ] **Step 2: implement** per spec 3.2 (`floor.regression_check` with `delta=PUBLIC_CV_REL_TOL * abs(committed)`).
  Run: PASS.

### Task 4: the rebaseline decision rule (spec 3.3, R24/R25/R26)

**Files:** Modify `scripts/hf_jobs/adopt.py` (`_rebaseline` `:526-561`, its docstring, the write-phase comment
`:1251`), `docs/superpowers/adrs/ADR-128-compute-contract-hf-campaigns.md` (P15 `:227`; item 17 stays history),
`docs/howto/hf-jobs-training.md` (`:176-180`); Test `tests/scripts/hf_jobs/test_rebaseline_tolerance.py`.

- [ ] **Step 1: failing tests.** `test_an_environment_difference_outside_the_tolerance_is_not_ok` (no changed module,
  a difference beyond the tolerance: `ok` false, `delta == "environment"` kept as information);
  `test_a_driver_without_a_baseline_entry_stops_without_an_exception` (a `DriverSpec` not in `REBASELINE_FROM`, a
  difference: `ok` false, `delta is None`, no `KeyError`). The existing `:190-193` (a code delta inside the tolerance
  passes) stays green. Run: FAIL.
- [ ] **Step 2: implement:** `ok = within` (no tolerance: false); compute `changed` only when `spec.name in
  REBASELINE_FROM`. Update the docstring, the comment, ADR-128 P15 and the runbook. Run: PASS.
- [ ] **Step 3: E4.** Count the added code and test lines of Tasks 2-4 (`git diff --numstat`). Expected: at most 65
  and 135; else STOP and ask the supervisor. Report also the whole C1 delta outside the carried content by category
  (code, tests, configuration, lock, documents).

### Task 5: the Hub data (owner go per create, upload and record)

**Files:** Modify `scripts/hf_jobs/datasets.toml` (`repo` of `tc3`, `xcross_probe`; `[datasets.xshot187]` with
`repo = "karstenskyt/hf-refit-xshot187"`), `scripts/hf_jobs/datasets.lock.json`; Test
`tests/scripts/hf_jobs/test_datasets_fields.py` (`:161` the `tc3` repo, the `xshot187` values).

- [ ] **Step 1: storage (E9 before).** `check_storage(HfApi(), org="karstenskyt", limit_bytes=10**12,
  headroom_bytes=20 * 10**9)`. Expected: ok; record the usage.
- [ ] **Step 2: creates (owner go)** on the DGX: `create_bucket("karstenskyt/hf-refit-scratch", private=True)`;
  `create_repo(r, repo_type="dataset", private=True, exist_ok=False)` for the three repositories. **E2:** read back
  `dataset_info(r).private` and the bucket visibility. Expected: all private; else STOP.
- [ ] **Step 3: uploads (owner go)**, `upload_large_folder` with the shape of `hf-refit-xshot115`: `xshot187` from
  `~/cross-def/xshot187-f5fc2e2b/out` (shards, `_actions`, `home_teams.json`; no `.sha256` files); `tc3` from
  `~/refit144/tc3/out`; the probe from `~/refit144/probe/pergame`. **E3:** every remote LFS sha256 equals the local
  file's; a non-LFS file (`home_teams.json`) is downloaded and compared byte for byte.
- [ ] **Step 3b: B6, committed.** Write `docs/research/hf_campaigns/tf46_refit/stage_provenance.json` (in C1; the
  `ADOPT_PATHS` home for campaign records, `adopt.py:58`), from a script outside the clone
  (`~/refit144/stage_provenance.py`, its sha256 recorded in the file) run with the clean release clone `~/refit144/sk`.
  Top level: `run_commit` (that clone's commit, `6eee84ce`: the library at measurement time) and
  `run_tree_dirty: false`, as `tests/scripts/test_artifact_provenance_output.py:91-117` requires. Under `datasets`,
  per dataset: the Hub revision, the generation, the stage `manifest_all.json` `run_commit` and `run_tree_dirty`
  (`xshot187`: `f5fc2e2b`; `tc3` and the probe: `6eee84ce`), the `tc3` schema change (30 -> 33 columns), and the
  sha256 of every uploaded file.
- [ ] **Step 4:** the `datasets.toml` values; failing test values first; then `datasets record` for each (owner go).
  Expected: `xshot187` 187 units, `public_ids` 89; `tc3` 179 (generation `0f43213bb5bccd5c`); probe 2. Tests PASS.

### Task 6: the campaign file (spec 4.1)

**Files:** Modify `scripts/hf_jobs/campaigns/tf46_refit.toml`; Test `tests/scripts/hf_jobs/test_campaign.py:165-175`,
`tests/scripts/hf_jobs/test_campaign_arms.py`.

- [ ] **Step 1: failing test.** `test_tf46_refit_holds_the_21_release_instances`: the 21 `(job, tier, variant)`
  instances with their modes (spec 4.1), the bucket `karstenskyt/hf-refit-scratch`, `storage_limit_bytes == 10**12`,
  the three sizing tables, `xsuccess` first. Run: FAIL.
- [ ] **Step 2:** edit the file (the header comment states its new content). Run both files: PASS.

### Task 7: C1 gate, review, commit and launch

- [ ] **Step 1: DGX gate** of the staged C1 tree (binary patch, `git write-tree` equal to the AI-PC's): `ruff check
  silly_kicks/ tests/ scripts/`, `ruff format --check` (same scope), bare `pyright`, and every test file that the
  change reaches (not the full suite; 54 files at v4.144.0, e.g. `test_adopt_shipped_arm.py`, `test_campaign_cycle1.py`,
  `test_rebaseline_tolerance.py`, `test_parity_territory_census.py`, `test_compute_contract.py`,
  `tests/tracking/test_forecast_skill.py`), found by:

  ```bash
  git grep -l -E '_artifact_integrity|provider_skill|skill_gates|hf_jobs\.(campaign|adopt|floor|manifest)|_rebaseline|build_territory_ranking_census|input_contract|datasets\.(lock|toml)|tf46_refit|cycle1\.toml' -- tests/
  ```

  plus `tests/scripts/test_artifact_provenance_output.py` (the new `stage_provenance.json`) and
  `tests/test_artifact_manifest.py`. Expected: all green.
- [ ] **Step 2:** `/final-review`; A/B implementation review of C1; fix or answer every finding.
- [ ] **Step 3:** owner yes: commit C1; owner yes: push and open a draft PR. Wait for CI green.
- [ ] **Step 4:** remind the owner of the open token decision (directive 2b point 5). If he made a fine-grained
  token: `hf auth switch` on the DGX. No code.
- [ ] **Step 5:** on a clean DGX clone at C1: `campaign scripts/hf_jobs/campaigns/tf46_refit.toml preflight`
  (7 checks ok), `launch --dry-run` (no FATAL), owner go: `launch`. Report a STUCK or FATAL line at once.

### Task 8: retrieve, verify, rulings and adopt (DGX clone at C1)

- [ ] **Step 0: baselines** (reads only): a second clean clone at `a1a3b52`; `campaign
  scripts/hf_jobs/campaigns/cycle1.toml retrieve --staging ~/refit144/launch7`. Expected: 21 instances.
- [ ] **Step 1:** `retrieve --staging ~/refit144/staging` (the complete folder: the record; its artifact folders are
  never changed; `verify` adds only `verify.json`, `_open_data/` and the head-to-head files).
- [ ] **Step 2:** the §7.1 expectation edit (plan rev 3 Task 6 Step 2, `git show 2a8e5ef3:<this plan>`) on the 8
  xShot/xCross entries, from each run's
  recorded `shipped_variant`, BEFORE the first `verify` (`_certify` and the expectations read the working manifest,
  `adopt.py:708`, `:864`).
- [ ] **Step 3: E5.** `verify --staging ~/refit144/staging`. Expected `basis` per spec 4.1. **E6:** `compare
  --staging ~/refit144/staging --baseline ~/refit144/launch7`. A non-zero exit is expected (18 instances differ by
  design); the reading is by row: win_probability and match_outcome IDENTICAL; a difference in either is a STOP and an
  owner ruling. **E7:** the census row (`rebaseline`) and its diffs go to the owner for the P15 ruling.
- [ ] **Step 4:** a failed gate or certification: STOP; report to the owner with the values (R19, B8).
- [ ] **Step 5:** copy the instances to adopt (not the census; not win_probability or match_outcome when IDENTICAL; not
  a stopped instance) to `~/refit144/adopt`; `verify --staging ~/refit144/adopt`; `adopt` with the venv first on
  `PATH`. Expected: the write phase succeeds; `adopt` stops at the commit gate. Record the local gate's failing set in
  `~/refit144/gate_after_adopt.txt`; every failure must be a test of Task 9's Test list (the old pins); any other: STOP.
- [ ] **Step 6: corners (E8).** In a separate clean clone at C1 (`~/refit144/corners-sk`; the adopt clone is dirty):
  `build_corner_routines` with pining data, then `validate_corners`. Report its gates against the 4.139.0 memo and the
  corner counts per provider. Owner yes on this reading before Step 7's corner copy (B8).
- [ ] **Step 7:** after `adopt`: copy the two corner bundles into their wheel destinations and set the two manifest
  `training_commit` values to C1 by hand (legacy certification); after the owner's P15 ruling, copy the census output
  into `docs/research/territory_ranking_census` and set `REBASELINE_FROM["build_territory_ranking_census"]` to C1.
  **E11:** every copied file's sha256 equals the run output's. Run the local gate again. Carry the diff to the AI-PC
  as a binary patch; compare `git write-tree`.

### Task 9: the adoption commit C2

**Files:** plan rev 3 Task 7's list (`git show 2a8e5ef3:<this plan>`), plus `scripts/hf_jobs/campaigns/cycle1.toml`
(xShot/xCross on `xshot187`, the bucket, `storage_limit_bytes = 10**12`), `scripts/hf_jobs/adopt.py:54`, the corner
bundles, the census output, and `docs/artifacts/manifest.toml`: a new `[events.release144]` (version, ADR-133) as
`refit_event` of every adopted entry that carries `refit_event = "launch7"` today (ghost x7, outfield x4, receiver,
xShot x4, xCross x4, `hub_xsuccess`: 21) and of the two corner entries; the cards of these entries (11 card files).
Test: plan rev 3 Task 7's list, plus `test_campaign.py:58` (the cycle1 limit), `test_datasets_fields.py`,
`tests/tracking/test_position_only_bundled.py:107-117` (pins `ghost_default` to `launch7` / `a1a3b52`),
`tests/test_artifact_manifest.py:41-43` (every declared event is used).

- [ ] **Step 1: RED** on the DGX clone (C1 + the adoption): run the Test files. Expected: the failing set equals
  `~/refit144/gate_after_adopt.txt` (Task 8 Step 5).
- [ ] **Step 2:** retarget the tests; move `cycle1.toml`; drop the `xshot17`/`xshot115` declarations and lock entries
  (no Hub delete). `git grep -n -E "xshot17|xshot115"` outside history files: no hit.
- [ ] **Step 3:** the re-fit event and the cards (plan rev 3 Task 7 Step 3: `mf.render_card` per card). A
  win_probability or match_outcome that stayed IDENTICAL keeps its entry. `[events.launch7]` is dropped when no entry
  uses it after the change (`test_artifact_manifest.py:41-43`); it stays only if a B8-dropped instance keeps it.
- [ ] **Step 4: GREEN:** the Task 7 Step 1 gate with `launch7|a1a3b52|refit_event` added to its grep, plus the Test
  files and `tests/test_card_regions.py`.
- [ ] **Step 5:** `/final-review` (version, CHANGELOG `PR-S209`, TODO); A/B review of C2; owner yes: commit C2;
  owner yes: push; CI green.

### Task 10: the parity proof (R5) and the release

- [ ] **Step 1:** at C2: `campaign scripts/hf_jobs/campaigns/cycle1.toml preflight`, `launch --dry-run`, owner go
  `launch`, `retrieve --staging ~/refit144/full`.
- [ ] **Step 2:** `verify --staging ~/refit144/full` (`compare` reads the census verdict from its `verify.json`,
  `adopt.py:1518-1530`).
- [ ] **Step 3: E10.** `compare --staging ~/refit144/full --baseline ~/refit144/launch7 --baseline ~/refit144/adopt`
  (the last baseline is the folder where `adopt` ran: `compare_staging` takes its `_final` copy as the shipped form,
  `adopt.py:1500-1504`, `:1538-1540`). Expected: exit 0. A model that moves and is not owed: STOP (B7). An instance
  that the owner dropped under B8 differs from launch 7 by design: it is listed with his decision, not a STOP.
- [ ] **Step 4: E9 after.** `karstenskyt/` usage.
- [ ] **Step 5:** mark the PR ready; owner yes: merge (`--merge`); CI green on the merge commit; owner yes: tag; owner
  go: `publish --sandbox`, then the public publish of the moved Hub artifacts (none for an IDENTICAL one).
