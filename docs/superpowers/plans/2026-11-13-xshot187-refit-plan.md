# xshot187 Restage, One Refit and the Adoption Release Implementation Plan (rev 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task
> (owner rule: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** stage the 187-match `xshot187` dataset, run ONE xShot/xCross refit on it, and adopt the result as a release.

**Architecture:** the corpus-widening release (4.143.0, ADR-132) built the registries, the two-arm rule, the per-job
arms, the per-provider gates and the size gates. This cycle uses them. It adds one tooling fix: `adopt` accepts the
spec §7.1 manifest edit as the only uncommitted change, and `verify` binds its report to the manifest it checked.

**Tech Stack:** Python 3.12 (`.venv-hfjobs`), `huggingface_hub`, HF Jobs `cpu-upgrade`, the DGX (`ssh
karsten@192.168.68.73`), pining.

**Spec:** `docs/superpowers/specs/2026-11-12-corpus-widening-design.md` (r8), sections 5.5, 6.1 to 6.4, 7 and 7.1,
and `docs/superpowers/adrs/ADR-132-corpus-provenance-split.md`. Runbook: `docs/howto/hf-jobs-training.md`.

**Owner rulings (2026-11-13):**

- R1. Every heavy or long local run goes to the DGX, data jobs included. The AI-PC runs only edits, `git`, `ruff`,
  light Hub reads and ONE small test file at a time.
- R2. Edit `scripts/hf_jobs/campaigns/tf46_refit.toml` in place for the refit. Do not add a second refit file.
- R3. Move every `xshot17` and `xshot115` reference in the adoption commit C2. Delete the two Hub repos only after
  the merge, with the owner's go.
- R4. Fold the `adopt` fix (the design below) into this cycle "as long as it is gold standard".
- R5. The runbook's parity proof (Task 8): APPROVED as Task 8 states it (owner 2026-11-13): after C2 is pushed and
  green, before the merge and the tag; launch 7's baseline is retrieved again from the kept `a1a3b52` run keys.
  The plan rev 3 is owner-approved (2026-11-13).

**Revision log.** rev 3 answers round 2 (`...-r2-A.md`; `...-r2-B.md` approved). rev 2 answered review round 1 (`D:\Development\_reviews\2026-11-13-xshot187-refit-plan-A.md`,
`...-B.md`). A and B do not disagree.

| Finding | Answer |
|---|---|
| A-01 (the rule checks which fields change, not how) | Design rules 8 to 10: no removal of a HEAD field or `expect` key; `dataset` declared and locked; fixed value domains. Tests and mutants in Task 1 |
| A-02, B-01 (`verify.json` not bound to the manifest) | Design rule 11: `verify` records `manifest_sha256`; `adopt` refuses on a missing or different digest, on both paths. Runbook text in Task 1 Step 6 |
| A-03, B-02 (the tests cannot run on the named fixtures) | Task 1 names `_toy_campaign`, `_add_entry`, `_dirty` and the `verify.json` digest; rule 1 returns `None` on a clean root before any read; `manifest.loads` is in Files with its test |
| B-03 (the id-list clause has no test) | Test `test_expectation_edit_refuses_a_changed_artifact_list` (add, remove, reorder) and mutant (d) |
| B-04 (two refusal tests have no RED) | Every new test name starts `test_expectation_edit_`; Step 2 selects them all; each asserts its own message |
| A-04 (the gate after `adopt`; Task 7's RED) | Task 6 Step 4 names the expected red set; Task 7 Step 1 is RED on exactly that set, Step 4 GREEN |
| A-05 (the parity proof is in no task) | Task 8, held by R5 until the owner rules |
| A-06 (no card command) | Task 7 Step 3: `mf.render_card` through a scratch script, checked by `tests/test_card_regions.py` |
| A-07 (storage) | Task 3 Step 1 counts storage first; the budget table below; Task 8 frees C1's run keys first |
| A-08 (DGX clone, venv, credentials) | Task 2 Step 2 builds ONE clone with the runbook's `.venv-hfjobs`; Task 2 Step 3 checks the Hub and `gh` credentials |
| A imprecise, B imprecise | Task 7 lists `test_datasets_fields.py:230`, `test_campaign_arms.py:118-121`; `test_datasets_lock.py:79` asserts `== len(units)` |
| r2: A-09 (existing assertions; `_dirty`) | Rule 2 returns `None` for any other dirty root, so today's messages answer it; rule 11's digest drops `training_commit`, so the fixed point holds; the existing `_dirty` is reused |
| r2: A-10 (the local gate's `python`) | Task 6 Step 4 puts the venv first on `PATH` and checks `command -v python uvx` |
| r2: B note (rule 10's reading) | Rule 10 checks every present expectation value of each campaign entry |
| r3: A note (the stub breaks the fixture) | Task 1 Step 2 implements `manifest.loads` and `_manifest_sha256` first; only `_expectation_edit` is a stub |

## Global Constraints

- Owner yes, per action: every commit, push, PR action, merge, tag, Hub upload, `datasets record`, HF job launch,
  `prune --delete`, Hub publish and Hub repo delete. Nothing else deletes in `silly-kicks/hf-refit-scratch`.
- Two commits on one branch, `feat/xshot187-refit`: C1 (the campaign commit that the jobs run) and C2 (the
  adoption, a NEW commit on top of C1, never an amend: every adopted `training_commit` names C1).
- `/final-review` runs before each A/B implementation review. A/B review this plan, C1 and C2. Reports go to
  `D:\Development\_reviews\`.
- No full test suite and no full `pyright` on the AI-PC. Every wider run uses the DGX clone.
- All temp files under `D:\Temp\xshot187\` (AI-PC) and `~/xshot187/` (DGX).
- Do not touch the HF write token, the historical wheels or the off-repo audit bundle. Eyestone NDA Rule #1.
  Databricks is not a source.
- Every staged repo is a PRIVATE Hub repo (ADR-132 item 7).
- `xshot187` holds SkillCorner 118 (20 public + 98 licensed) + IDSSE 7 + Gradient Sports 62 (GS `3812`-`3859`,
  `10504`-`10517`); GS `10502`/`10503` stay out (spec 6.3). 187 and 89 are expected counts, not pins (spec 6.4).
- `datasets.toml` entry: `kind = "flat_tc3"`, `raw = "restricted"`, `provenance = "extended"`, declared in the SAME
  commit as its first `datasets record` (spec §7 table, `test_datasets_lock.py:59-61`).
- Refit campaign: `mode = "refit"`; the wheel jobs (`tier = "public"`) take `arms = ["public", "sc_extended"]`; the
  Hub jobs (`tier = "sc_extended"`) take `arms = ["sc_extended"]` (ADR-132 item 6).
- A provider gate that fails fails the run: no fallback, no reweighting. Report `skill["by_provider"]` to the owner,
  who decides (ADR-132 item 5).
- Write in STE. Version number: set at C2 commit prep (the wheel bundles change); one CHANGELOG entry for the
  release, keyed `PR-S209`, in C2.

**Storage budget** (org private storage; preflight check 7 passes while used + 20 GB headroom ≤ 100 GB, so used ≤
80 GB). Measured 2026-11-13: 52.2 GB (scratch bucket 23.7, `tc3` 15.7, `xshot115` 9.5, `xshot17` 2.5, probe 0.7).
One campaign is about 7.5 GB (the launch-7 run keys of `a1a3b52`: 7.28 GB).

| After | Used (about) |
|---|---|
| the `xshot187` upload (about 16 GB, Task 3) | 68 GB |
| the refit run (Task 5) | 76 GB |
| `prune --delete` of C1's refit run keys (Task 8 Step 1, owner go) | 68 GB |
| the `cycle1.toml` parity run at C2 (Task 8) | 76 GB |
| the delete of `hf-refit-xshot17` and `hf-refit-xshot115` (Task 7 Step 7, after the merge) | 64 GB |

## Review Focus

1. An `adopt` run over the manifest edit AND any other change (tracked or untracked): refused, nothing written
   (`test_expectation_edit_refuses_a_second_change`).
2. A manifest edit that weakens a check: a removed field or `expect` key, an undeclared `dataset`, a value outside
   its domain (`test_expectation_edit_refuses_a_removal`, `..._an_undeclared_dataset`, `..._a_value_outside_its_domain`).
3. A manifest edit after `verify`: refused until `verify` runs again (`test_expectation_edit_refuses_a_manifest_changed_after_verify`).
4. A write-phase failure during an adoption over an edit: the tree goes back to the edited manifest, not to HEAD
   (`test_expectation_edit_rollback_keeps_the_edit`).
5. A match that S1 excludes at staging: the record, the arms and every check follow the staged units (Task 2 Step 6
   reports it; `test_arms_follow_staged_units` pins the arm rule).

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

### Task 1: the expectation edit and the verify binding (C1)

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

### Task 2: materialize `xshot187` on the DGX (owner go: the run)

**Files:** none in the repo. Scratch: `D:\Temp\xshot187\roster.py`, `~/xshot187/` on the DGX.

- [ ] **Step 1: Roster.** `roster.py` builds `{"skillcorner": [...], "idsse": [...], "gradientsports": [...]}`:
  SkillCorner = the `tc3` lock's SkillCorner games ∪ `RAW_REDISTRIBUTABLE["skillcorner"]`; IDSSE =
  `RAW_REDISTRIBUTABLE["idsse"]`; GS = `PUBLICLY_OBTAINABLE["gradientsports"]` − `HELD_OUT_PROBE["gradientsports"]`.
  It asserts 118 / 7 / 62 and writes `roster.json`. Expected: `187`.
- [ ] **Step 2: One DGX clone** for the whole cycle: `git clone https://github.com/karsten-s-nielsen/silly-kicks
  ~/xshot187/sk` at `main` (`0cc07e6c`); build the runbook's venv there: `UV_PROJECT_ENVIRONMENT=.venv-hfjobs
  ~/.local/bin/uv sync --frozen --python 3.12.14 --extra train --extra xgboost --extra test --group hfjobs`; copy
  `roster.json` byte-identically (scp, LF). Expected: `git status` clean; `.venv-hfjobs/bin/python -c "import
  scripts.materialize_tc3_frames"` exits 0.
- [ ] **Step 3: Credentials** (read-only checks): `HfApi().whoami()` on the DGX shows the `silly-kicks` org with
  write and Jobs access; `gh` works per invocation with `GH_TOKEN` over stdin (memory
  `feedback_dgx_default_compute_box`). Expected: both pass; else STOP and tell the owner.
- [ ] **Step 4: Smoke.** `nohup` runner: `source ~/.pining_owner.env; .venv-hfjobs/bin/python -m
  scripts.materialize_tc3_frames --cache-dir ~/pining-cache --out ~/xshot187/out --providers idsse gradientsports
  skillcorner --match-ids-json ~/xshot187/roster.json --max-per-provider 1` (the cap is not in the token, so the
  full run resumes). Expected: 3 shards, exit 0.
- [ ] **Step 5: Full run**, same command without the cap. Poll the log. Expected: exit 0; shards + `.excluded.json`
  = 187 units.
- [ ] **Step 6: Report** to the owner: the staged unit count per provider, every S1 exclusion with its reason, the
  size on disk, and the wall time.

**As run (2026-11-13; the durable record, review A C1-IMPL-04).** A serial run (Step 5 as written) reached 30 of
187 matches; the owner stopped it ("this needs to run in parallel"). The staging then ran as 8 workers into ONE
`--out`, then ONE plain final pass:

- The clone `~/xshot187/sk` at `0cc07e6c` (clean). The materializer has no partition flag, so each worker ran an
  out-of-repo wrapper (`worker.py`, sha256 `36d6e3b8e429be4e2fe45dc3cdc68534a8a1e0bdf77254d64d98bfd4f2255954`;
  the launcher `stage_parallel.sh`, sha256 `5e94177e5d7dbe46794fe875c68e3a9b5c02551657755fc5aed20123b6892733`),
  which keeps the refs of one partition and then calls the committed `main()`. Its code, ABRIDGED (the file also
  has a docstring, an `assert 0 <= i < n` and a per-worker `print` of its ref count):

  ```python
  import hashlib, sys
  import scripts._loader_pining as lp
  i, n = (int(x) for x in sys.argv[1].split("/"))
  _orig = lp.pining_source
  def _partition(*args, **kwargs):
      refs, load = _orig(*args, **kwargs)
      return [r for r in refs if int(hashlib.sha256(f"{r.provider}\0{r.match_id}".encode()).hexdigest(), 16) % n == i], load
  lp.pining_source = _partition
  sys.argv = ["materialize_tc3_frames", *sys.argv[2:]]
  from scripts import materialize_tc3_frames
  materialize_tc3_frames.main()
  ```

  The partition rule is `int(sha256(f"{provider}\0{match_id}").hexdigest(), 16) % 8 == I`: a stable sha256
  partition like `in_worker_partition` of commit `67417107`, but NOT that function (it hashes `"p|m"` and takes the
  first 8 bytes). Every worker passed the same arguments
  (`--cache-dir ~/pining-cache --out ~/xshot187/out --providers idsse gradientsports skillcorner --match-ids-json
  roster.json`), so the token (providers, schema `tc3-frames-4`, roster hash) and the generation are the same for
  all: `3836f5993b9c5a9f`. Slices 30/31/20/17/26/26/24/13 = 187.
- Workers 15:05-15:32Z, 0 failed. The final pass (the committed command, no wrapper) skipped all 187 shards and
  wrote `home_teams.json` and `manifest_all.json`: `n_excluded = 0`, `n_failed = 0`, `run_commit = 0cc07e6c`,
  `run_tree_dirty = false`.
- Staged: SkillCorner 118, IDSSE 7, GS 62; S1 excluded none. Upload set 15.73 GB (187 shards, 187 `_actions`,
  `home_teams.json`); the `.sha256` resume files stay local, as for `xshot115`. Hub revision `36978240`.

### Task 3: upload and record `xshot187` (owner go: upload; owner go: record) (C1)

**Files:**
- Modify: `scripts/hf_jobs/datasets.toml` (new `[datasets.xshot187]` after `xshot115`),
  `scripts/hf_jobs/datasets.lock.json` (written by `datasets record`)
- Test: `tests/scripts/hf_jobs/test_datasets_fields.py`, `tests/scripts/hf_jobs/test_arm_policy.py`

- [ ] **Step 1: Storage.** `preflight.check_storage(HfApi(), org="silly-kicks", limit_bytes=100_000_000_000,
  headroom_bytes=20_000_000_000)` and `du` of the upload set in `~/xshot187/out`. Expected: used + upload + 7.5 GB
  (the refit run) ≤ 80 GB. Else STOP and tell the owner.
- [ ] **Step 2: Shape check.** List `silly-kicks/hf-refit-xshot115` (`HfApi().list_repo_tree`, recursive) and
  `~/xshot187/out`. Expected: the upload set has the same top-level shape (`shards/<gen>/`, `_actions/`,
  `home_teams.json`); write the allow patterns from it.
- [ ] **Step 3: Create and upload** on the DGX (owner go): `create_repo("silly-kicks/hf-refit-xshot187",
  repo_type="dataset", private=True, exist_ok=False)`, then `upload_large_folder(...)` with those patterns. Expected:
  `dataset_info(...).private is True`; the remote unit stems equal the local shard stems.
- [ ] **Step 4: Write the failing tests:** add `"xshot187": ("restricted", "extended")` to
  `test_committed_declarations_carry_the_spec_values`; add `"xshot187"` to the `dataset` parametrize of
  `test_emitted_keys_equal_the_trainer_keys_on_every_locked_unit`; new
  `test_xshot187_arms_are_the_registry_arms_of_its_staged_units`: `public` == the locked `public_ids` as in-job keys,
  `sc_extended` == every staged unit, no probe unit. Run each file. Expected: FAIL (`xshot187` not declared).
- [ ] **Step 5: Declare** the entry (exact text):

```toml
[datasets.xshot187]
repo = "silly-kicks/hf-refit-xshot187"
kind = "flat_tc3"
raw = "restricted"
provenance = "extended"
access = "mount"
read_concurrency = 4  # the tc3 measurement (same mount type)
description = "187-match idsse + SkillCorner + Gradient Sports corpus (ADR-132): both xShot/xCross arms"
```

- [ ] **Step 6: Record** (owner go): `.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record xshot187`.
  Expected: exit 0; `n_units` = the staged count; `public_ids` = the public-arm pairs (89 with no exclusion).
- [ ] **Step 7: Run** the two test files and `test_datasets_lock.py`. Expected: PASS.

### Task 4: the refit campaign (C1)

**Files:**
- Modify: `scripts/hf_jobs/campaigns/tf46_refit.toml`
- Test: `tests/scripts/hf_jobs/test_campaign_arms.py` (`test_committed_campaigns_keep_todays_arms`)

- [ ] **Step 1: Write the failing test.** In `test_committed_campaigns_keep_todays_arms`, add the `xshot187` branch:
  a wheel job (`tier == "public"`) has `arms == ("public", "sc_extended")` and `_arms_payload(inst) == today`; a Hub
  job has `arms == ("sc_extended",)` and the `sc_extended`-only payload. Keep the `xshot17`/`xshot115` branches for
  `cycle1.toml` (C2 removes them). Assert that every gated job of `tf46_refit.toml` reads `xshot187`. Run it.
  Expected: FAIL.
- [ ] **Step 2: Edit** `tf46_refit.toml`: header comment (ADR-132: the refit on `xshot187`, both arms on the wheel
  jobs); the mount comment (`xshot187` and `xcross_probe`); all four jobs `dataset = "xshot187"` with the arms of
  Global Constraints and their comments. `mode`, the bucket and the knobs stay.
- [ ] **Step 3: Run** `test_campaign_arms.py`. Expected: PASS.

### Task 5: C1 gate, review and launch

- [ ] **Step 1: DGX gate** of the staged C1 tree (`git diff --binary --cached main > D:\Temp\xshot187\c1.patch`;
  apply in `~/xshot187/sk` on a scratch branch; compare `git write-tree` with the AI-PC's): `ruff check silly_kicks/
  tests/ scripts/`, `ruff format --check silly_kicks/ tests/ scripts/`, bare `pyright`, `tests/scripts/hf_jobs`,
  `tests/test_bundled_weights_corpus_policy.py`, `tests/test_agents_md_budget.py`. Expected: all green.
- [ ] **Step 2:** `/final-review`, then the A/B implementation review of C1; fix or answer every finding.
- [ ] **Step 3:** owner yes: commit C1; owner yes: push and open a draft PR. Wait for CI green.
- [ ] **Step 4:** in `~/xshot187/sk`, check out C1 (clean). `campaign scripts/hf_jobs/campaigns/tf46_refit.toml
  preflight`. Expected: seven checks ok.
- [ ] **Step 5:** `launch --dry-run`. Expected: no FATAL; the first-stage launches of the 8 instances.
- [ ] **Step 6:** owner go: `launch` (background, log). Watch `status`; a STUCK or FATAL line is reported at once.

### Task 6: retrieve, verify and adopt (on the DGX clone at C1)

- [ ] **Step 1:** `retrieve --staging ~/xshot187/staging` on the clean C1 tree. Expected: 8 instances.
- [ ] **Step 2: The §7.1 edit** (manifest only, uncommitted): read each wheel run's recorded `shipped_variant`;
  per spec §7.1's table set, on the 4 wheel entries, `dataset = "xshot187"`, `shipped_arm`, `certification`,
  `corpus` and the `expect` values (`shipped_variant`, `reproducibility`, `provider_list` = the admitted providers of
  the shipped arm); on the 4 Hub entries, `dataset = "xshot187"`.
- [ ] **Step 3:** `verify --staging ...`. Expected: 8 `ok`; `verify.json` carries `manifest_sha256`. A failed
  provider gate or size gate: STOP; report `skill["by_provider"]` and the failing gate to the owner. No fallback.
- [ ] **Step 4:** `adopt --staging ...`, in a shell where the venv is first on `PATH`: `_local_gate` starts bare
  `python` and `uvx` (`adopt.py:1050-1054`), and the DGX also has `~/anaconda3`. Run `source
  .venv-hfjobs/bin/activate; export PATH="$VIRTUAL_ENV/bin:$HOME/.local/bin:$PATH"; command -v python uvx` first.
  Expected: `~/xshot187/sk/.venv-hfjobs/bin/python` and `~/.local/bin/uvx`; else STOP. Then `adopt`. Expected: the
  write phase succeeds and `adopt` stops at the commit gate. The
  diff touches only the allowlisted paths plus the §7.1 edit. The local gate is RED by design in the tests that pin
  the 17-match bundles or the `xshot17`/`xshot115` locks, for example
  `test_corpus_guard.py::test_committed_xshot_bundles_carry_the_canonical_public_corpus_ids` and
  `test_bundled_weights_corpus_policy.py::test_the_staged_public_ids_are_the_bundled_public_arm`. Record the failing
  set in `~/xshot187/gate_after_adopt.txt`. Expected: every failure is a test in Task 7's Test list, and each fails on
  the 17-match pin or on the outcome's corpus form. Any other failure: STOP, a defect.
- [ ] **Step 5:** carry the DGX diff to the AI-PC clone as a binary patch; compare `git write-tree`.

### Task 7: the adoption commit C2

**Files:**
- Modify: `scripts/hf_jobs/campaigns/cycle1.toml` (the 4 xShot/xCross jobs: `dataset = "xshot187"`, arms as in
  `tf46_refit.toml`), `scripts/hf_jobs/datasets.toml` and `datasets.lock.json` (drop `xshot17`, `xshot115`),
  `docs/artifacts/manifest.toml` (a new `[events.<name>]` with version and `ADR-132`; `refit_event` on the 8
  entries), the cards of the 8 entries and their wheel siblings, `scripts/_corpus.py` (the `BUNDLED_PUBLIC_ARM`
  comment; the pin stays for gk_completion), `scripts/_toy_corpora.py:35` (the comment), `CHANGELOG.md`,
  `TODO.md` (Current line), `silly_kicks/_version.py`
- Test (retarget each to `xshot187`, keeping its intent): `test_datasets_lock.py` (`:68`: the locked `public_ids`
  equal the registry public arm of the staged units, two sources (INV-9); `:72-79`: `n == len(units)`),
  `test_datasets_fields.py` (`:87-88`, `:174-176`, `:207`, `:226-230`), `test_campaign_cycle1.py:50`,
  `test_campaign_arms.py` (`:41`, `:49`, `:61`, `:72`, `:118-121`; one rule for both campaign files),
  `test_arm_policy.py` (`:44`, `:64-67`: the public arm == the locked `public_ids`, `:125-158`; a test that needs a
  public-only unit set uses a synthetic unit list, as `:80-100` do), `test_adopt.py` (`:40-115`),
  `test_dgx_e2e.py:83`, `test_bundled_weights_corpus_policy.py`, `tests/scripts/test_corpus_guard.py`,
  `test_bundled_probe_ids_are_committed_ids.py` (spec §7 table). `tests/scripts/hf_jobs/fixtures/real_jobinfo.json`
  stays: it is a recorded HF answer.

- [ ] **Step 1: RED.** On the DGX clone (C1 + the adoption), run the Test files. Expected: the failing set equals
  `gate_after_adopt.txt` (Task 6 Step 4).
- [ ] **Step 2:** retarget the tests; move `cycle1.toml`; drop the two lock and `datasets.toml` entries. Then
  `git grep -n -E "xshot17|xshot115"` outside CHANGELOG, specs, plans, ADRs and `real_jobinfo.json`. Expected: no hit.
- [ ] **Step 3: The re-fit event and the cards.** Declare `[events.<name>]` and set `refit_event` on the 8 entries.
  Re-render each affected card with `D:\Temp\xshot187\render_cards.py`: for every `card` of the 8 entries and of
  their wheel siblings, write `mf.render_card(root / card, man, card_id=card)` (the call of `adopt.py:1172`).
  Expected: `tests/test_card_regions.py` passes.
- [ ] **Step 4: GREEN.** DGX gate as in Task 5 Step 1, plus the Test files, `tests/test_card_regions.py`,
  `tests/test_metric_contracts.py` and the bundled-weights tests. Expected: all green.
- [ ] **Step 5:** `/final-review` (version bump and TODO), then A/B review of C2. Owner yes: commit C2; owner yes:
  push; CI green.
- [ ] **Step 6:** Task 8 if R5 rules it in; then mark the PR ready; owner yes: merge (`--merge`); owner yes: tag (the
  tag publishes to PyPI); owner go: `publish --sandbox`, post-verification; owner go: the public Hub publish.
- [ ] **Step 7:** owner go: delete `silly-kicks/hf-refit-xshot17` and `silly-kicks/hf-refit-xshot115`. Tell the
  TF-46 Arm 2 session the result.

### Task 8: the parity proof (R5: approved)

The runbook's proof for a refit of some arms (`docs/howto/hf-jobs-training.md:271-275`): the full `cycle1.toml` at
the adoption commit, compared with the shipped launch 7 and this refit. Recommended: run it after C2 is pushed and
green, before the merge and the tag, because it is the only proof that `cycle1.toml` at C2 reproduces every shipped
artifact.

- [ ] **Step 0: Baselines.** Launch 7's staging folder is gone; its run keys `a1a3b52` are kept in the bucket. In a
  second DGX clone checked out at `a1a3b52` (clean), run `campaign scripts/hf_jobs/campaigns/cycle1.toml retrieve
  --staging ~/xshot187/launch7` (reads only). Keep `~/xshot187/staging` (the refit). Expected: 21 instances
  retrieved for launch 7, 8 in the refit folder; else STOP and tell the owner.
- [ ] **Step 1: Storage** (owner go): `campaign scripts/hf_jobs/campaigns/tf46_refit.toml prune --delete
  --i-have-owner-approval <C1 commit7>` (HEAD is C2, so C1's run keys are prunable; the staging folder is the
  record). Expected: about 7.5 GB freed; `check_storage` ≤ 72 GB.
- [ ] **Step 2:** at C2: `campaign scripts/hf_jobs/campaigns/cycle1.toml preflight`, `launch --dry-run`, owner go
  `launch`, `retrieve --staging ~/xshot187/full`.
- [ ] **Step 3:** `compare --staging ~/xshot187/full --baseline <launch 7 staging> --baseline ~/xshot187/staging`.
  Expected: exit 0. A difference is a defect of the change: STOP.
