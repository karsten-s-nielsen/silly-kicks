# HFU fit units, ball-carrier pre-index, campaign hygiene — implementation plan

> **For the implementing session:** use `superpowers:executing-plans` and implement the tasks yourself, in this
> session. No subagents (owner rule). Steps use checkbox (`- [ ]`) syntax for tracking. This plan has NO commit
> step in any task: the cycle has one code commit, at the owner gate of Task 15.

Revision 3 (plan round 1: A APPROVE, B REQUEST CHANGES; round 2: A APPROVE, B REQUEST CHANGES with no blocking
finding; round 3: B APPROVE. The last two sections list the answers).

**Goal:** Run the 36 xgboost fits of the xsuccess `fit` stage as stage units, rewrite the slow part of
`_pre_index_frames` with no output change, and close six campaign hygiene items.

**Architecture:** Part A splits `XSuccessModel.fit` into steps that the library and the trainer both compose; the
trainer's `fit` stage gets a fixed grid of 36 units and a reduce that assembles the bundle. Part B replaces the
per-frame placement loop of `_pre_index_frames` and keeps the old function in the tests as an oracle. Part C adds
a storage check, a `prune` and a `compare` command, a lost-result rule in `plan()`, a read bound in `retrieve`
and a second `post_verify` path. One HF launch then proves that no artifact changed.

**Tech stack:** Python 3.12, pandas, numpy, xgboost, scikit-learn, `huggingface_hub` 2.2.0, pytest. Interpreter
for every local command: `.venv-hfjobs/Scripts/python.exe`.

**Spec:** `docs/superpowers/specs/2026-11-10-hfu-fit-units-carrier-index-hygiene-design.md` (revision 3b: revision 3,
owner-approved 2026-11-10, plus two changes from the plan review: the §3.2 "Removed"
list and the `lost_output` field of C4, both owner-confirmed 2026-11-10). Read it with this plan. Section numbers below are the spec's.

**Cycle ledger:** `.superpowers/sdd/2026-11-10-hfu-fit-units-plan/progress.md` (git-ignored). Every measured
figure and every decision of the build goes there first.

## Global constraints

- Branch `feat/hfu-reuse-fit-carrier`. No worktree. No subagents.
- **No commit, push, force-push, PR, merge, tag, HF job or bucket deletion without the owner's explicit yes for
  that action.** One code commit (Task 15) and one evidence commit (Task 17). A gate question for a commit names
  BOTH actions it covers, "commit and push"; every other action has its own question.
- **No output changes** (§2): Parts A and B must not change one byte of any artifact payload.
- Tests first: each test is seen to fail before its change. A test that compares new code with an oracle is
  also seen to FAIL on a deliberate fault (a permanent detection test, named in the task).
- Local runs: the named test files and `tests/scripts/hf_jobs` only. Do not run the full suite or full pyright
  locally; CI runs them.
- Local test command (the base for every "Run" line):
  `.venv-hfjobs/Scripts/python.exe -m pytest <paths> -q -p no:cacheprovider --basetemp D:\Temp\hfu-next\pytest`
- Every temp file goes under `D:\Temp\hfu-next\`. Never the C: drive. Set `TEMP` and `TMP` to
  `D:\Temp\hfu-next\tmp` for a command that makes temp files.
- Write scripts with the Write tool (a Bash heredoc breaks on an apostrophe). Git Bash has no `jq`.
- Prose in ASD-STE100. Every `warnings.warn(..., stacklevel=2)`. No `apply(axis=1)`. Id comparisons through
  `silly_kicks.id_compat`.
- The version number is set at commit preparation (Task 15), never before.
- A step that cannot meet its stated result stops for the owner. It is never weakened.

## Review focus

Inputs the spec implies but does not name. Each has a test in the task that owns the code.

1. A `category` frame key with an unused category: `groupby` then yields an empty group. The rewrite must place
   nothing for it, as the oracle does (Task 1).
2. A `FileSet` marker that is present but not valid JSON (a part upload): `plan()` must count the result as lost,
   not crash the `--plan` (Task 7).
3. `compare`: an artifact in only one staging folder, a missing `verify.json`, a `verify.json` of another commit
   and a failed census verdict must each be a reported difference with exit 1, never a skip (Task 2).
4. A `usedStorage` call that fails (a 429 or a missing field): the storage check must return a failed check with
   the reason, not raise (Task 5).
5. A calibrated fit whose kept rows hold one class only, and an empty corpus: the first gives absent `cal` units
   and a model without isotonic thresholds; the second stops with today's message (Task 11).
6. A run key with zero files (four exist in the bucket today): `prune` lists it and deletes it without an error
   (Task 6).
7. A stage that turns from reduced to not reduced inside one driver run: `_attribute` counts its finished jobs
   again (Task 7).

## File map

| File | Change |
|---|---|
| `silly_kicks/tracking/_ball_carrier.py` | Task 1: the placement part of `_pre_index_frames` |
| `tests/tracking/_pre_index_oracle.py` (new), `tests/tracking/test_pre_index_oracle_equality.py` (new), `tests/conftest.py` | Task 1 |
| `scripts/hf_jobs/identity.py` | Task 2: the cross-commit rule |
| `scripts/hf_jobs/adopt.py` | Task 2 (`compare_staging`), Task 3 (read bound), Task 4 (`post_verify`) |
| `scripts/hf_jobs/driver.py` | Tasks 2, 6 (commands), 7 (reports), 8 (STUCK line), 12 (knob) |
| `scripts/hf_jobs/preflight.py`, `campaign.py` | Task 5 (check 7), Task 12 (knob) |
| `scripts/hf_jobs/prune.py` (new) | Task 6 |
| `scripts/_stages.py` | Task 7: `plan()` |
| `silly_kicks/xsuccess/_model.py` | Task 10: the fit steps |
| `scripts/train_xsuccess.py`, `scripts/_toy_corpora.py` | Task 11: the unit grid, the reduce, the toy corpus |
| `scripts/hf_jobs/campaigns/cycle1.toml` | Tasks 5, 12 |
| `tests/test_bundled_card_citations.py` (new), `.gitattributes`, `TODO.md` | Task 9 |
| docs (Task 13) | `ADR-128`, two howtos, `.claude/skills/hf-campaign/SKILL.md`, `docs/context/tracking-features.md`, `docs/c4/architecture.dsl`, `CHANGELOG.md`, `TODO.md`, the old HFU spec §14.10a, this cycle's spec §6 |

---

### Task 1: Part B — the pre-index placement (spec §4)

**Files:**
- Create: `tests/tracking/_pre_index_oracle.py`, `tests/tracking/test_pre_index_oracle_equality.py`
- Modify: `silly_kicks/tracking/_ball_carrier.py:108-138`, `tests/conftest.py` (the root conftest: the fixture
  must be visible in `tests/tracking/` AND in `tests/scripts/`), and one line in each of
  `tests/tracking/test_ball_carrier.py`, `tests/tracking/test_ball_carrier_numba_parity.py`,
  `tests/scripts/test_coordination_driver_modules.py`

**Interfaces:**
- `pre_index_frames_oracle(frames) -> dict` in `_pre_index_oracle.py`: the current `_pre_index_frames`, the
  whole function (`_ball_carrier.py:38` to its `return`), copied verbatim; only the name changes.
- `assert_pre_index_equal(old: dict, new: dict) -> None`, the §4.3 equality rule: the same key set; arrays equal
  in dtype, shape and elements with NaN equal to NaN (an object array element by element); lists and dict keys
  and values equal in order with the same NaN rule; dtypes by `==`.
- `shadow_pre_index_oracle` (a pytest fixture in `tests/conftest.py`, with the oracle imported inside the
  fixture body): it replaces
  `silly_kicks.tracking._ball_carrier._pre_index_frames` with a wrapper that calls the oracle on a copy of the
  frames and the real function on the frames, applies `assert_pre_index_equal` (or, when the oracle raises,
  asserts that the real function raises the same type), counts its calls, and returns the real result.
- `_pre_index_frames(frames) -> dict` keeps its name, signature and every key.

- [ ] **Step 1: Copy the oracle and write the rule's own tests.** `test_assert_pre_index_equal_detects`: the
  rule raises `AssertionError` on two dicts that differ in one array element, in one dtype, in one list entry
  and in the key set (four cases), and passes on two dicts that each hold a NaN in `slot_to_pid` and an object
  array in `frame_meta_gid`.

- [ ] **Step 2: Shadow the existing tests.** Add `pytestmark = pytest.mark.usefixtures("shadow_pre_index_oracle")`
  to the three test modules above (their frames are built inline in the tests, so the shadow is how each of them
  becomes an oracle input). Add `test_the_shadow_ran` in BOTH directories (one in
  `tests/tracking/test_pre_index_oracle_equality.py`, one in `tests/scripts/test_coordination_driver_modules.py`):
  with the fixture on, one `infer_ball_carrier` call raises the wrapper's call count above 0.

- [ ] **Step 3: Write the equality test.** In `test_pre_index_oracle_equality.py`, two case lists of named frame
  builders, `CASES_OK` and `CASES_RAISE`:

```python
@pytest.mark.parametrize("build", CASES_OK)
def test_new_equals_oracle(build):
    frames = build(); before = frames.copy(deep=True)
    assert_pre_index_equal(pre_index_frames_oracle(frames.copy(deep=True)), _pre_index_frames(frames))
    pd.testing.assert_frame_equal(frames, before, check_exact=True)        # the input is not changed

@pytest.mark.parametrize("build", CASES_RAISE)
def test_new_raises_what_the_oracle_raises(build):
    with pytest.raises(Exception) as want: pre_index_frames_oracle(build())
    with pytest.raises(type(want.value)): _pre_index_frames(build())

@pytest.mark.parametrize("build", CASES_OK)
def test_infer_ball_carrier_is_unchanged(build, monkeypatch): ...
    # infer_ball_carrier on the same frames, once with _pre_index_frames patched to the oracle and once
    # unpatched: pd.testing.assert_frame_equal(check_exact=True, check_dtype=True)

def test_the_equality_test_detects_a_fault(monkeypatch): ...
    # patch _pre_index_frames to a wrapper that swaps slots 0 and 1 of the first frame with two players;
    # assert_pre_index_equal(oracle(frames), faulty(frames)) raises AssertionError  (the detection test)
```

  `CASES_OK`: each provider's toy frames with its real id dtypes (`scripts._toy_corpora.staged_frames` for every
  kind in `PROVIDER_ID_KIND`); a synthetic 20,000-frame match (22 players, 25 Hz); a null frame key; a
  `player_id` that occurs only on a null-frame-key row; a float NaN `player_id` on a player row; an object id
  column with `np.nan` beside numbers (`dict.get` finds it by identity); an object id column with two DISTINCT
  NaN objects on two player rows (`dict.get` finds only the one that is the key object); a duplicate player id
  in one frame; two
  ball rows in one frame; a frame with no player; a null `is_ball` in an object column; a null `is_ball` in a
  nullable `boolean` column; no `vx`/`vy`; `category` `team_id`; `category` frame keys; `category` frame keys
  with an unused category (review focus 1); float32 coordinates; unsorted input rows; an empty frame table.
  `CASES_RAISE`: an `Int64` `<NA>` `player_id` on a player row; an object `None` `player_id` beside strings.
  If a case put in `CASES_OK` makes the oracle raise, move it to `CASES_RAISE` and record it in the ledger.

  Add `test_the_cases_cover_the_semantics`: from the oracle's output on the named cases it asserts that at least
  one case has `player_slots == -1` inside a valid slot (B-S5), one has two equal slot ids in one frame (B-S6),
  one has a frame with `n_valid == 0`, one has `has_velocity` False, and `CASES_RAISE` is not empty.

- [ ] **Step 4: Run the tests.** Run: `<base> tests/tracking/test_pre_index_oracle_equality.py
  tests/tracking/test_ball_carrier.py tests/tracking/test_ball_carrier_numba_parity.py
  tests/scripts/test_coordination_driver_modules.py`. Expected: PASS (the two functions are the same code; the
  detection test and the rule's own tests prove that an inequality fails). This is the baseline.

- [ ] **Step 5: Rewrite lines 108-138 of `_pre_index_frames`.** Replace the `player_groups` dict, the group loop
  and the inner slot loop. Do not touch the code before line 108 or after line 138. Approach: map each player
  row to its frame index through the unique-frame table (a row whose frame key is null or absent is dropped,
  B-S3); take a stable order by frame index; the position of a row inside its frame is its rank in that order
  (B-S2); `n_valid` is the count per frame; `max_players` is its maximum, 1 when no row is placed (B-S4); scatter
  `x`, `y`, `vx`, `vy` into the float64 arrays (B-S7, B-S8); for the slot ids, look up the `player_id` object of
  each placed ROW with `pid_to_slot.get(pid, -1)` in ONE flat pass over the placed rows (never a loop per
  frame, and never a lookup per unique value: two distinct NaN objects must each get their own `dict.get`
  result, B-S5), then scatter the result. No float arithmetic.

- [ ] **Step 6: Run the Step 4 command again.** Expected: PASS.

- [ ] **Step 7: Scale guard and timing.** Register an ADR-073 guard for the one per-row pass (`SCALE_GUARDED`,
  `tests/test_scale_guards.py`): it must grow linearly in the rows. No per-frame Python loop may remain. Time the oracle and the new function on
  the 20,000-frame match (best of 3) with a script in `D:\Temp\hfu-next\`; record both figures in the ledger.

---

### Task 2: the cross-commit rule and `compare` (spec §6 step 3)

**Files:**
- Modify: `scripts/hf_jobs/identity.py`, `scripts/hf_jobs/adopt.py`, `scripts/hf_jobs/driver.py` (`main`)
- Test: `tests/scripts/hf_jobs/test_identity.py`, `tests/scripts/hf_jobs/test_adopt.py`,
  `tests/scripts/hf_jobs/test_hf_driver.py`

**Interfaces:**
- `identity.PROVENANCE_KEYS` gains `"artifact_size_bytes"`.
- `identity.cross_commit_differences(a: Path, b: Path, *, commit_file: str | None = None) -> list[str]`. Empty
  means identical. Rule: a file named `metadata.json`, `metrics.json` or `commit_file` is compared as canonical
  JSON without `PROVENANCE_KEYS` (recursively); `SHA256SUMS` on the lines whose file is not one of those;
  `MODEL_CARD.md` after `re.sub(rb"\b[0-9a-f]{40}\b", b"<commit>", ...)` and CRLF to LF; every other file byte
  for byte; a file in only one folder is a difference.
- `adopt.compare_staging(c: Campaign, *, staging: Path, baseline: Path) -> dict`: one entry per
  `"<job>-<tier>/<variant>"` of the campaign.
  - A trainer instance: `{"identical": bool, "differences": [...]}` from `cross_commit_differences` over
    `<folder>/<run_key>/<variant>/` (never `_final/`). An instance whose folder is absent on a side is
    `{"identical": False, "differences": ["present in only one staging folder"]}`.
  - A `DriverSpec` instance (the census): `{"census_ok": bool, "certification": {...}}` from
    `<staging>/verify.json`: `census_ok` is True only when the file exists, its `commit` starts with the commit7
    of the staging run keys, and the instance's row has `ok: true`.
  - Each side's run-key folder is found by the suffix `-<job>-<tier>` under the campaign prefix. Two matches on
    one side raise `AdoptionRefusedError` (ambiguous); zero is the absent case above.
- CLI: `campaign <file> compare --staging DIR --baseline DIR` prints one line per instance and the JSON; exit 0
  when every trainer instance is identical and `census_ok`, else 1. `compare` builds no `HfApi`.

- [ ] **Step 1: Write the failing tests.** `test_identity.py`, on toy folders in `tmp_path`:

```python
def test_identical_folders_have_no_difference(...)
def test_a_payload_byte_is_a_difference(...)                    # model.npz differs
def test_provenance_only_json_is_identical(...)                 # training_commit, training_platform, run_commit differ
def test_artifact_size_bytes_is_a_provenance_key(...)           # only that leaf differs -> []
def test_a_non_provenance_leaf_is_a_difference(...)             # "auc" differs -> one difference naming metrics.json
def test_sums_lines_of_normalized_files_are_dropped(...)        # SHA256SUMS lists metadata.json -> []
def test_match_outcome_shape(...)                               # commit inside model.json, one SUMS line; commit_file="model.json" -> []
def test_match_outcome_shape_without_commit_file_differs(...)   # same folders, commit_file=None -> a difference
def test_card_commit_is_normalized(...)                         # a 40-hex commit differs -> []; another word differs -> a difference
def test_a_file_in_one_folder_is_a_difference(...)
```

  `test_adopt.py`, `compare_staging` over toy staging folders: one identical and one different trainer instance;
  an instance folder absent on one side; two run-key folders on one side (refuses); the census with a good
  `verify.json` (`census_ok` True); no `verify.json`; a `verify.json` of another commit; a census row with
  `ok: false` (each of the last three: `census_ok` False). `test_hf_driver.py`: `main([... "compare",
  "--staging", a, "--baseline", b])` returns 1 on a trainer difference, 1 on `census_ok` False, 0 otherwise;
  `compare` without `--baseline` exits with a message.

- [ ] **Step 2: Run them.** Run: `<base> tests/scripts/hf_jobs/test_identity.py tests/scripts/hf_jobs/test_adopt.py
  tests/scripts/hf_jobs/test_hf_driver.py`. Expected: the new tests FAIL (`cross_commit_differences` not defined).

- [ ] **Step 3: Implement.** `cross_commit_differences` in `identity.py`, reusing `_strip`, `normalized_json` and
  the `normalized_sums` logic with the wider file set; `identity.compare` and `artifact_identity` keep their
  signatures. `compare_staging`: `commit_file` is the file part of the artifact's `commit_field` when that file
  is not `metadata.json` or `metrics.json` (`artifacts_for`). The command and `--baseline` in `driver.main`.

- [ ] **Step 4: Run the tests.** Expected: PASS. Also `tests/scripts/hf_jobs/test_rebaseline_tolerance.py` and
  `test_rebaseline_ast.py` (they read `PROVENANCE_KEYS` through `normalized_json`): PASS.

- [ ] **Step 5: Record the launch-6 against launch-7 evidence.** Run:
  `.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/cycle1.toml compare
  --staging D:\Development\_staging\cycle1-a1a3b52 --baseline D:\Development\_staging\cycle1-0552f58`, output to
  `D:\Temp\hfu-next\compare_6v7.txt`. Expected: 16 trainer instances identical; 4 different, each on
  `metrics.json` only (the xShot and xCross `public` instances, both variants); `census_ok` True (launch 7's
  `verify.json`); exit 1. Any other result stops for the owner. Task 13 copies the output into spec §6 step 4.

---

### Task 3: C1 — one read bound in `retrieve`

**Files:** Modify `scripts/hf_jobs/adopt.py:131-268`. Test: `tests/scripts/hf_jobs/test_adopt.py`.

**Interfaces:** `retrieve(..., workers: int = 4, read_workers: int = 8)`: one
`threading.BoundedSemaphore(read_workers)` for the whole call. Every store that `retrieve` opens is wrapped so
that `read_bytes` and `names` hold the semaphore (a small private class in `adopt.py`).

- [ ] **Step 1: Write the failing test.** `test_retrieve_never_reads_more_than_eight_at_once`: a fake store whose
  `read_bytes` sleeps 5 ms and records the number of calls in flight; 4 instances, 40 counter files each; assert
  the recorded maximum is at most 8 and at least 2 (non-vacuity), and that each record's counter totals equal
  those of a run with `read_workers=1`.
- [ ] **Step 2: Run it.** Expected: FAIL (a maximum above 8).
- [ ] **Step 3: Implement.** **Step 4: Run** `tests/scripts/hf_jobs/test_adopt.py`. PASS.

---

### Task 4: C2 — `post_verify` without a Hub class

**Files:** Modify `scripts/hf_jobs/adopt.py` (`publish`, `post_verify`). Test:
`tests/scripts/hf_jobs/test_publish_verify.py`, `tests/test_artifact_manifest.py`.

**Interfaces:** `post_verify(api, repo, a, local, *, run, allowlist, card, staging: pathlib.Path)`. With
`a.hub_class` non-empty: the `from_hub` round trip, as today. With an empty `hub_class`:
`api.snapshot_download(repo_id=repo, repo_type="model", local_dir=staging / "_post_verify" / repo.partition("/")[2])`,
then `load_check(a.job, that dir, run=run)`; a non-None return is the problem `"load failed: ..."`. `publish`
passes `staging`.

- [ ] **Step 1: Write the failing tests.** (a) `test_post_verify_uses_the_job_loader_without_a_hub_class`: an
  artifact with `hub_class=""`; a fake API records `snapshot_download`; a fake `run` records its argv; assert no
  argv holds `from_hub` and one holds `.load(`; a failing `run` gives `ok: False`. (b)
  `test_post_verify_uses_from_hub_with_a_hub_class`. (c) the manifest gate in `tests/test_artifact_manifest.py`:
  for every Hub artifact, `hub_class` non-empty implies that `LOADERS[a.job]` ends with `:<hub_class>` and that
  the class has `from_hub`; `hub_class` empty implies that the `LOADERS` class has `load`; both branches have at
  least one artifact (non-vacuity).
- [ ] **Step 2: Run them.** Expected: (a) FAILS. If (c) fails for an artifact, stop for the owner.
- [ ] **Step 3: Implement.** **Step 4: Run** the two files. PASS.

---

### Task 5: C3 — preflight check 7, `storage`

**Files:** Modify `scripts/hf_jobs/preflight.py`, `scripts/hf_jobs/campaign.py` (`Campaign`, `load_campaign`,
`cmd_preflight`), `scripts/hf_jobs/campaigns/cycle1.toml`. Test: `tests/scripts/hf_jobs/test_preflight.py`,
`test_campaign.py`, `test_campaign_cycle1.py`.

**Interfaces:**
- `Campaign.storage_limit_bytes: int = 100_000_000_000`, `Campaign.storage_headroom_bytes: int = 20_000_000_000`
  (read from `[campaign]`; `cycle1.toml` states both).
- `preflight.check_storage(api, *, org: str, limit_bytes: int, headroom_bytes: int) -> Check`. Used = the sum of
  `size` over `api.list_buckets(org)` plus `used_storage` of `api.dataset_info(id, expand=["usedStorage"])` and
  `api.model_info(id, expand=["usedStorage"])` for each PRIVATE dataset and model of `org`. `ok` is
  `used + headroom_bytes <= limit_bytes`. The detail names the used bytes by repo, largest first. An API error or
  a missing size gives `Check(ok=False, ...)` with the reason (review focus 4).
- `cmd_preflight` runs it as the seventh check, `org = c.bucket.split("/")[0]`.

- [ ] **Step 1: Write the failing tests** on a fake API (two buckets, two private datasets, one public dataset,
  one private model): under the bound; at the bound (`used + headroom == limit` passes); over the bound (the
  detail names the largest repo); a public repo is not counted; an API error fails the check with its reason; a
  missing size fails the check. In `test_campaign.py`: the two values load from the toml and default when
  absent; `cmd_preflight` runs seven checks.
- [ ] **Step 2: Run them.** FAIL (`check_storage` not defined). **Step 3: Implement** (and the "six" wording in
  `cmd_preflight`'s docstring and in any test that pins the count). **Step 4: Run** the three files. PASS.

---

### Task 6: C3 — `prune`

**Files:** Create `scripts/hf_jobs/prune.py`, `tests/scripts/hf_jobs/test_prune.py`. Modify
`scripts/hf_jobs/driver.py` (`main`). Test also: `tests/scripts/hf_jobs/test_hf_driver.py`.

**Interfaces:**
- `prune.run_keys_by_commit(c: Campaign, api) -> dict[str, list[dict]]`: `{commit7: [{"run_key", "files",
  "bytes"}]}` for every top-level bucket folder that matches `^<campaign>-(?P<c7>[0-9a-f]{7})-<job>-<tier>(-r\d+)?$`
  for a `(job, tier)` of `c.jobs` (the campaign name is `re.escape`d). Nothing else is ever listed.
- `prune.prune(c, *, head: str, delete: bool, approved: Sequence[str], api, manifest_commits: Mapping[str, int])
  -> dict`. Without `delete`: the listing for every commit other than `head[:7]`. With `delete`: deletes the run
  keys of each commit in `approved`; an empty `approved` refuses. Refusals (`PruneRefusedError`): a commit equal
  to `head[:7]`; a commit with a non-terminal job (`api.list_jobs(labels={"campaign": c.name})`, the `commit`
  label); a commit that is not in the listing. Deletion: per run key, list recursively,
  `api.batch_bucket_files(c.bucket, delete=...)` in chunks of 1,000, list again, record `files_left`. A `403`
  stops the run with `the org is at its storage limit: HF refuses the bucket delete; delete a private dataset or
  model first, then run prune again`. The result has `"ok": False` when any `files_left` is not 0.
- A commit that `manifest_commits` names prints `commit <c7> is the training_commit of <n> shipped artifact(s);
  its run keys are run records, the artifacts are in the wheel and on the Hub` before the delete.
- CLI: `campaign <file> prune` (list); `campaign <file> prune --delete --i-have-owner-approval <commit7> [...]`.
  Exit 0 when `ok`, else 1. `--delete` is a new `store_true` option. `--i-have-owner-approval` keeps its one
  definition; its `metavar` becomes `REPO_OR_COMMIT7` and its help says `publish: a repo; prune: a commit7`.

- [ ] **Step 1: Write the failing tests** on a fake API whose bucket holds two commits of campaign `cycle1` (two
  run keys each, one of them with zero files), one run key of a campaign named `cycle1b`, `_campaign/`,
  `_staged/`, `_canary/`:

```python
def test_listing_holds_only_this_campaigns_run_keys(...)      # cycle1b, _campaign, _staged, _canary absent
def test_listing_excludes_head(...)
def test_delete_removes_exactly_the_approved_commit(...)      # every other folder untouched
def test_delete_without_an_approval_refuses(...)
def test_delete_of_head_refuses(...)
def test_delete_with_a_live_job_refuses(...)
def test_delete_of_an_unknown_commit_refuses(...)
def test_a_replica_run_key_is_listed(...)                     # <campaign>-<c7>-<job>-<tier>-r1
def test_a_run_key_with_zero_files_is_listed_and_deleted_without_an_error(...)   # review focus 6
def test_a_partial_delete_reports_files_left_and_is_not_ok(...)   # a second run deletes the rest
def test_a_403_stops_with_the_storage_message(...)
def test_a_manifest_commit_prints_the_notice(...)
```

- [ ] **Step 2: Run them.** FAIL (module not found). **Step 3: Implement** `prune.py` and the command.
  `manifest_commits`: the `training_commit` of every artifact in `docs/artifacts/manifest.toml`, counted by its
  first 7 characters. **Step 4: Run** `test_prune.py` and `test_hf_driver.py`. PASS.

---

### Task 7: C4 — a `FileSet` result is present only with its members

**Files:** Modify `scripts/_stages.py` (`_ready`, `plan`), `scripts/hf_jobs/driver.py` (`decide`,
`_reduce_or_report`). Test: `tests/scripts/test_stage_contract.py`, `tests/scripts/hf_jobs/test_hf_driver.py`,
`tests/scripts/hf_jobs/test_bucket_store.py`.

**Interfaces:**
- `_stages._fileset_members_present(store, gen: str, stem: str) -> bool`. It reads `<gen>/<stem>.fileset.json`,
  parses `{"files": {...}}`, and returns whether every member name is in the listing of its directory under
  `<gen>/<stem>.files/`. A marker that is absent, unreadable or not valid JSON gives False (review focus 2).
  `unit_states` and `output_present` do NOT change (in-job code is unchanged).
- `plan()`, per stage record: a `done` unit whose result suffix is `.fileset.json` and whose members are not all
  present is counted `pending` and listed in the new field `"lost_units": [...]` (sorted; `[]` when none). The
  new field `"lost_output": bool` is True when `_output` is a `FileSet` whose members are not all present.
  `reduced` is `output_present(...) and not lost_output and not lost_units`. `_ready` uses the same `reduced`
  rule for each needed stage.
- Driver: `decide` reads `lost_units` and `lost_output` from the stage record. A unit in `lost_units` whose
  coverage is spent is reported `<stage>/<key>: result lost (members missing); free storage, stop the driver,
  then run resume`. A stage with `lost_output` whose reduce triggers are spent is reported `<stage>: output lost
  (members missing); free storage, stop the driver, then run resume`.

- [ ] **Step 1: Write the failing `plan()` tests** on a `LocalStore` with a two-stage toy pipeline (`prep` with
  one `FileSet` unit; a consumer stage):

```python
def test_a_marker_without_members_is_a_lost_pending_unit(...)   # prep: pending 1, done 0, lost_units ["only"], reduced False; consumer ready False
def test_some_members_missing_is_lost(...)
def test_a_memberless_output_beside_complete_units(...)         # lost_units [], lost_output True, reduced False
def test_a_garbled_marker_is_lost_not_a_crash(...)              # review focus 2
def test_a_complete_fileset_is_unchanged(...)                   # the record equals today's plus lost_units [] and lost_output False
def test_a_parquet_result_is_never_read_by_the_check(...)       # a counting store sees no read of the unit result
```

- [ ] **Step 2: Write the failing request-count test.** In `test_bucket_store.py`, with the counting fake API of
  the HFU spec §16.4 tests: a `--plan` over a store with `k` done `FileSet` results makes exactly ONE listing
  request and exactly `k` marker reads more than the same plan over the same store with `k = 0` `FileSet`
  results (the units stored as parquet). This is the count rule of spec C4 as a test.
- [ ] **Step 3: Write the failing driver tests.**
  (a) Through `_attribute` (review focus 7): a coverage cache that holds one COMPLETED run job for the unit and
  one COMPLETED reduce trigger for the stage; a plan whose stage record is `reduced: False`, `lost_units:
  ["only"]`. Assert that the unit's coverage is 1 and the stage's reduce-error count is 1 (the finished jobs
  count again once the stage is not reduced).
  (b) `decide` on that state with `attempts = 3`: one keyed launch for the unit.
  (c) `decide` with coverage 3: no launch; the report is exactly the lost-result text and does not contain
  `crashed with no record`.
  (d) `decide` with `lost_output: True`, no lost unit and reduce triggers spent: the lost-output text.
- [ ] **Step 4: Run them.** FAIL. **Step 5: Implement.**
- [ ] **Step 6: Check `scripts/hf_jobs/datasets.py:359-361`** (the download plan with `verify=False`). If a
  `uses_dataset` plain stage can have a `FileSet` `_output` whose loss now leaves it un-reduced without a data
  fetch for the rerun, write the case into the ledger and stop for the owner; if none exists, write the stage
  names checked into the ledger.
- [ ] **Step 7: Run** the three test files and `tests/scripts/test_compute_contract.py -m "not slow"`. PASS.

The real request counts of the four instances of HFU spec §16.4 are measured after launch 8 (Task 17 Step 5),
because only then does the bucket hold run keys whose tokens match HEAD.

---

### Task 8: C5 — the STUCK operator path

**Files:** Modify `scripts/hf_jobs/driver.py:153-154`. Test: `tests/scripts/hf_jobs/test_hf_driver.py`.

- [ ] **Step 1: Write the failing test.** `test_a_stuck_line_names_the_operator_path`: a step with one report
  prints exactly `STUCK <instance>: <reason>; fix the cause, stop the driver, then run resume` on stderr. A
  report that already ends with `then run resume` (Task 7's texts) is printed without a second suffix.
- [ ] **Step 2: Run it.** FAIL. **Step 3: Implement.** **Step 4: Run** the file; update every existing test that
  pins the old line. PASS.

---

### Task 9: C6 — the card citation gate, `.gitattributes`, the `TODO.md` row

**Files:** Create `tests/test_bundled_card_citations.py`. Modify `.gitattributes`, `TODO.md`.

**Interfaces:**
- `_EXPECTED_CARDS: tuple[str, ...]`: the 8 paths that `git ls-files 'silly_kicks/**/MODEL_CARD.md'` gives today,
  written out.
- `_cards() -> list[Path]`: every `MODEL_CARD.md` under `silly_kicks/` (`Path.rglob`), sorted.
- `missing_citations(card: Path) -> set[str]`: the back-ticked tokens of the card that match
  `^(SHA256SUMS|[A-Za-z0-9_.-]+\.(json|npz|md))$` and do not exist beside the card.
- `_CITED_ELSEWHERE: dict[str, dict[str, str]]`: `{card posix path: {file name: reason}}`.

- [ ] **Step 1: Write the tests.**

```python
def test_the_card_population_is_exact():        # [p.as_posix() for p in _cards()] == sorted(_EXPECTED_CARDS)
def test_every_cited_file_exists(card):         # parametrized over _cards(): missing_citations(card) <= set(_CITED_ELSEWHERE.get(card, {}))
def test_no_stale_exemption():                  # every _CITED_ELSEWHERE entry is in missing_citations(its card)
def test_a_card_that_cites_a_missing_file_fails(tmp_path):   # a toy card citing `metrics.json` in an empty dir -> {"metrics.json"}
```

- [ ] **Step 2: Run it.** Run: `<base> tests/test_bundled_card_citations.py`. `test_every_cited_file_exists` shows
  the cited names that are absent. For each: if the card cites a file of another place, add the exemption with
  its reason; if the card cites a file the bundle should hold, stop for the owner (a shipped defect).
- [ ] **Step 3: `.gitattributes`.** Add after line 20: `silly_kicks/xsuccess/weights/metrics.json diff`, with a
  one-line comment as lines 14-16 have. Run `git check-attr diff silly_kicks/xsuccess/weights/metrics.json`:
  expected `diff: set`.
- [ ] **Step 4: Remove the model-card gate row of `TODO.md`** (line 27 today). Run every test that reads
  `TODO.md` (`grep -l "TODO.md" tests/*.py`). PASS.

---

### Task 10: Part A — the fit steps in `XSuccessModel` (spec §3.2 "Library seam", A-R1)

**Files:** Modify `silly_kicks/xsuccess/_model.py:141-316`. Create `tests/xsuccess/_fit_oracle.py`,
`tests/xsuccess/test_fit_steps_equal_oracle.py`.

**Interfaces** (module-level private functions in `_model.py`; the trainer imports them in Task 11):
- `_training_arrays(actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `(X, y, tid)` of the kept rows
  (`type_id != _NON_ACTION`, every feature finite), exactly lines 167-172.
- `_xgb_params(params: dict | None, seed: int) -> dict`: `dict(params or _DEFAULT_PARAMS)` with `random_state`.
- `_fit_booster(X, y, p: dict)`: `xgb.XGBClassifier(**p).fit(X, y).get_booster()`.
- `_calibration_splits(X, y, seed: int) -> list[tuple[np.ndarray, np.ndarray]]`: `[]` unless
  `len(np.unique(y)) == 2 and len(y) >= 50` and `n_splits = min(5, int(np.bincount(y).min())) >= 2`; else the
  `StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)` pairs, in order.
- `_calibration_fold_proba(X, y, p: dict, train, test) -> np.ndarray`: fit `xgb.XGBClassifier(**p)` on the train
  rows, return `predict_proba(X[test])[:, 1]` in the dtype xgboost returns.
- `_isotonic_thresholds(oof, y) -> tuple[np.ndarray, np.ndarray]`: lines 214-216.
- `_raw_predict(booster, X) -> np.ndarray`: float64, NaN where a row has a non-finite feature, else
  `booster.predict(xgb.DMatrix(X[finite]))` (lines 263-266 without the isotonic step).
- `_calibrate(raw, iso_x, iso_y) -> np.ndarray`: `np.interp` on the finite entries when both thresholds exist;
  else `raw` unchanged.
- `_metadata_payload(*, feature_names, feature_set, params, iso_x, iso_y, probe_prediction: list) -> dict`: the
  dict `to_dict` returns, with the same key order.
- `fit`, `_fit_xgboost`, `predict_success`, `to_dict` and `_chirality_block` compose these. No public signature
  changes.

- [ ] **Step 1: Copy the oracle.** `tests/xsuccess/_fit_oracle.py` holds `OracleXSuccessModel(XSuccessModel)` with
  today's `fit`, `_fit_xgboost`, `_fit_per_type`, `predict_success`, `_predict_per_type`, `to_dict` and
  `_chirality_block`, copied verbatim (`_model.py:141-316`), so no oracle path runs new code.
- [ ] **Step 2: Write the failing test.** `test_fit_steps_equal_oracle.py`. Inputs built from
  `tests.scripts.test_progress_lines._xsuccess_pooled` (import it; extend the rows): (a) both classes, at least
  5 rows of each, at least 50 rows; (b) the rarer class has 4 rows; (c) 3 rows; (d) 2 rows; (e) 40 rows; (f) one
  class; (g) input (a) with `family="per_type_logistic"`. EVERY input also holds one `non_action` row and one
  row with a NaN `start_x`. For each:

```python
new = XSuccessModel().fit(actions, family=family)
old = OracleXSuccessModel().fit(actions, family=family)
assert _saved_files(new) == _saved_files(old)            # the raw bytes of model.json, metadata.json, SHA256SUMS
assert np.array_equal(new.predict_success(actions), old.predict_success(actions), equal_nan=True)
```

  Non-vacuity, asserted: every input has a `non_action` row and a non-finite feature row; the split counts are
  5, 4, 3, 2 for (a) to (d) (`len(_calibration_splits(X, y, 42))`) and `new._iso_x is not None` there; (e) and
  (f) have `new._iso_x is None`. Add `test_the_out_of_fold_vector_keeps_its_dtype` (a spy on
  `_isotonic_thresholds` and on the oracle's `IsotonicRegression.fit`: the same dtype). Add
  `test_the_oracle_test_detects_a_fault`: with `_calibration_fold_proba` patched to return its result plus 1e-7,
  `_saved_files(new) != _saved_files(old)` for input (a) (the detection test).
- [ ] **Step 3: Run it.** Run: `<base> tests/xsuccess/test_fit_steps_equal_oracle.py`. Expected: the tests that
  import the new functions FAIL.
- [ ] **Step 4: Implement** the functions and recompose `fit`, `_fit_xgboost`, `predict_success`, `to_dict`.
- [ ] **Step 5: Run** `<base> tests/xsuccess`. Expected: PASS. **If the saved bytes of any input differ from the
  oracle and the cause is the explicit loop against `cross_val_predict`, stop for the owner (A-R1).**

---

### Task 11: Part A — the unit grid and the reduce (spec §3.2-§3.5)

**Files:** Modify `scripts/train_xsuccess.py` (`cross_val_metrics`, `_FIT_ROWS`, `_fit_task`, `_fit_map`,
`_fit_work`, the `fit` stage in the `PIPELINE` block), `scripts/_toy_corpora.py` (`build_open_data`,
`TOY_BUILDERS`). Create `tests/scripts/_xsuccess_fit_oracle.py`, `tests/scripts/test_xsuccess_fit_units.py`.
Modify `tests/scripts/test_xsuccess_drivers_smoke.py:158-222`, `tests/scripts/test_parity_xsuccess.py`,
`tests/scripts/test_progress_lines.py:180-190`.

**Interfaces:**
- `FIT_UNITS: tuple[str, ...]`: the 36 keys `f"{fit}-{part}"`, `fit` in `("final", "fold0", ..., "fold4")`, `part`
  in `("main", "cal0", ..., "cal4")`, sorted.
- `ABSENT: dict = {"absent": True}`.
- `fit_unit(pooled: pd.DataFrame, key: str, *, family: str, params: dict | None) -> pd.DataFrame | FileSet | dict`.
  It calls `_cv_splits(pooled)` (the pooled row space). Rows of the fit: all pooled rows for `final`; fold `i`'s
  train rows for `fold<i>`. Then by key:
  - `fold<i>` with `i >= len(splits)`, or `cal<j>` with `j >= len(_calibration_splits(X, y, 42))`, or any `cal`
    unit of `per_type_logistic`: `ABSENT`.
  - `<fit>-cal<j>`: `pd.DataFrame({"p": _calibration_fold_proba(...)})`.
  - `fold<i>-main`, xgboost: `pd.DataFrame({"raw": _raw_predict(booster, xsuccess_features(pooled.iloc[test]))})`;
    `per_type_logistic`: the same frame with `XSuccessModel().fit(...).predict_success(pooled.iloc[test])`.
  - `final-main`: `FileSet({"model.json": <the bytes `save` writes>, "probe_raw.npy": <np.save bytes of the raw
    probe prediction>})`; `per_type_logistic`: `probe_raw.npy` holds `predict_success(probe)`. With no folds it
    also holds `in_sample_raw.npy` (the raw prediction on every pooled row; `predict_success(pooled)` for
    `per_type_logistic`).
- `fit_reduce(pooled, results: Mapping[str, Any], *, family: str, params: dict | None) -> tuple[dict,
  dict[str, bytes], str]`: `(metrics, files, feature_set)`, the triple `_fit_map` returns today (`files` =
  `model.json`, `metadata.json`, `SHA256SUMS`). It never imports xgboost.
- `_pooled(ctx) -> pd.DataFrame`: `ctx.upstream("extract")`; an empty frame raises today's
  `SystemExit("no on-ball rows collected from the corpus; nothing to fit")`.
- Stage: `Stage(name="fit", units=lambda ctx: list(FIT_UNITS), work=_fit_unit_work, needs=("refs", "extract",
  "hpo"), reduce=_fit_reduce, params=("family",))`. No `inner_parallel`, `inner_width`, `timeout_s`.
  `_fit_unit_work(key, ctx)` calls `fit_unit(_pooled(ctx), key, family=ctx.params["family"],
  params=ctx.upstream("hpo")["best_params"])`. `_fit_reduce(results, ctx)` calls `fit_reduce` and then runs the
  rest of today's `_fit_work` unchanged (the provenance in `metadata.json`, `metrics.json`, `MODEL_CARD.md`).
- Removed from the trainer: `_FIT_ROWS`, `_fit_task`, `_fit_map`, `_fit_work`, `cross_val_metrics` (with its
  `[progress] xsuccess fold k/n` line; `_fit_map` takes the `[progress] xsuccess fit k/n` line with it; each
  unit prints the `[unit] fit/<key> done in <s> s` line). The spec §3.2 "Removed" list names them
  (owner-confirmed 2026-11-10).
- `build_open_data(..., nonfinite_in: str | None = None)`: when set to a match id, the builder removes the
  `location` of the first `Pass` event of that match. `_xsuccess_toy(root)` calls it with
  `nonfinite_in="7298"`; `TOY_BUILDERS["scripts.train_xsuccess"] = _xsuccess_toy`.

- [ ] **Step 1: Copy the oracle.** `tests/scripts/_xsuccess_fit_oracle.py` holds today's `cross_val_metrics` and
  `_fit_work` verbatim, with `XSuccessModel` replaced by `OracleXSuccessModel` (Task 10), so no oracle path runs
  new code. Its serial branch is the one that runs (the test context has no pool).
- [ ] **Step 2: Write the failing shape tests.** `test_xsuccess_fit_units.py`. `_ctx(tmp_path, pooled,
  best_params, family)` builds a `RunContext` over a `LocalStore` (`tests.scripts._stage_harness.PROV`,
  `params_from_args`) and seeds the `refs`, `extract` (with its `_summary.json`) and `hpo` outputs with
  `write_result`. `_run_fit(ctx)` writes each `T._fit_unit_work(key, ctx)` result with `write_result`, reads all
  36 back with `done_results(ctx, "fit")`, and returns `T._fit_reduce(results, ctx)`. So every unit result
  crosses the store encoding (`write_result` and the verified read); this is the store path of `--run-all`
  without the open-data server, which the slow parity test of Step 7 adds. Then, for each of five corpora:

```python
new = _run_fit(ctx)                          # FileSet
old = oracle_fit_work("only", ctx)           # FileSet, the same ctx
assert new.files == old.files                # the RAW bytes of every file (the two runs share one ctx, so no
                                             # provenance differs; a key order or an indent change must fail)
```

  The corpora and their asserted non-vacuity:
  1. calibrated xgboost: 6 matches of unequal row counts and one row with a NaN `start_x`, built so that the
     fold of at least one match under `_cv_splits(pooled)` differs from its fold under `_cv_splits` of the
     finite rows only (A-R2: assert that difference); no unit result is `ABSENT`; `metadata.json` holds
     isotonic thresholds;
  2. fewer than 5 calibration splits: some `cal4` is `ABSENT` and some `cal0` is not;
  3. one class in a fit's kept rows (review focus 5): every `cal` of that fit is `ABSENT`;
  4. `per_type_logistic`: all 30 `cal` units are `ABSENT`;
  5. one match: every `fold` unit is `ABSENT` and `final-main` holds `in_sample_raw.npy`.

  Three more tests: `test_fit_reduce_does_not_import_xgboost` (corpus 1's results; `sys.modules["xgboost"] =
  None` during `fit_reduce`; it passes); `test_an_empty_corpus_stops_with_todays_message` (`_fit_unit_work` and
  `_fit_reduce` on an empty `extract` output raise `SystemExit` with that text; review focus 5);
  `test_the_shape_test_detects_a_fault` (with `_calibrate` patched to add 1e-9, `new.files != old.files` for
  corpus 1).
- [ ] **Step 3: Run them.** Run: `<base> tests/scripts/test_xsuccess_fit_units.py`. FAIL (`FIT_UNITS` not defined).
- [ ] **Step 4: Implement** `FIT_UNITS`, `fit_unit`, `fit_reduce`, `_pooled`, the two wrappers and the stage.
  Delete the removed names.
- [ ] **Step 5: The toy corpus.** Add `nonfinite_in` to `build_open_data` and `_xsuccess_toy`. In
  `test_parity_xsuccess.py` build the corpus with `nonfinite_in="7298"` and assert, on the serial run's `extract`
  output, that at least one row has a non-finite feature and that the matches have unequal row counts. **If
  removing a `location` does not give a non-finite feature row through the converter, try the first `Carry`
  event; if no raw edit gives one, stop for the owner.**
- [ ] **Step 6: Rewrite the dependent tests.** `test_xsuccess_drivers_smoke.py:158-222`: replace the three
  `_fit_map` / plan-record / width-one tests with `test_fit_plans_36_units_without_an_inner_pool` (the plan
  record has `n_units == 36`, `inner_parallel` False, `timeout_s` None and `units == list(FIT_UNITS)`).
  `test_parity_xsuccess.py`: `completion_out_of_order(store, gens["fit"])` in place of `inner_out_of_order` for
  `fit`; the `hpo` asserts stay. `test_progress_lines.py:180-190`: the `xsuccess` parameter of
  `test_native_fit_stages_print_fold_lines` is removed (the `pass_completion` one stays) and a new test asserts
  that a `fit` unit prints its `[unit] fit/<key> done` line. Fix any other user of the removed names
  (`grep -rn "_fit_map\|_fit_task\|_FIT_ROWS\|cross_val_metrics" tests scripts`).
- [ ] **Step 7: Run** `<base> tests/scripts/test_xsuccess_fit_units.py tests/scripts/test_xsuccess_drivers_smoke.py
  tests/scripts/test_progress_lines.py tests/scripts/test_compute_contract.py tests/scripts/test_toy_corpora.py
  tests/scripts/hf_jobs/test_scaffold.py -m "not slow"` (the last two read `TOY_BUILDERS`, which now maps
  xsuccess to `_xsuccess_toy`). PASS. Then the slow
  parity test once: `<base> tests/scripts/test_parity_xsuccess.py -m slow`. PASS, with the `fit` out-of-order
  assert and the Step 5 asserts true.
- [ ] **Step 8: Check the registry and the other open-data pipelines.** Run `<base>
  tests/scripts/hf_jobs/test_registry.py tests/scripts/hf_jobs/test_campaign_recipes.py`. PASS with no change to
  `registry.py`; if a change is needed, record why in the ledger. The other three users of `_open_data_toy` are
  not changed (`nonfinite_in` defaults to None).

---

### Task 12: the `units_per_job_by_stage` knob (spec §3.4)

**Files:** Modify `scripts/hf_jobs/campaign.py` (`Campaign`, `load_campaign`), `scripts/hf_jobs/driver.py`
(`_decide`, `decide`), `scripts/hf_jobs/campaigns/cycle1.toml`. Test: `tests/scripts/hf_jobs/test_hf_driver.py`,
`test_campaign.py`, `test_campaign_cycle1.py`, `test_executor.py`.

**Interfaces:**
- `Campaign.units_per_job_by_stage: Mapping[str, int]` (default empty), keyed `"<job>/<stage>"`, from
  `[campaign.units_per_job_by_stage]`.
- `decide(..., units_per_job_by_stage: Mapping[str, int] | None = None)`: the units per job of a stage is
  `units_per_job_by_stage.get(f"{inst.job}/{name}", units_per_job)`, for `_keyed_launches` and
  `_round_robin_launches`. An escalate launch is unchanged (one unit worker).
- `cycle1.toml`: `[campaign.units_per_job_by_stage]` `"xsuccess/fit" = 3`; `[campaign.peak_rss_by_stage]`
  `"xsuccess/fit" = 8589934592`; the comment at lines 32-33 says `fit` is 36 units in 12 jobs.

- [ ] **Step 1: Write the failing tests.** `test_a_stage_override_gives_twelve_jobs_of_three` (36 pending units,
  `units_per_job=20`, `max_shards=12`, the override 3: 12 launches of 3 keys each, contiguous in sorted order);
  `test_a_stage_without_an_override_uses_the_global_value` (the same plan under another job name: 2 launches);
  `test_the_override_reaches_the_round_robin_path`; `test_an_escalate_launch_keeps_one_worker`. In
  `test_campaign_cycle1.py`: `units_per_job_by_stage == {"xsuccess/fit": 3}` and
  `peak_rss_by_stage["xsuccess/fit"] == 8589934592`. In `test_executor.py`: the worker sizing with
  `peak_rss_bytes=8589934592` on a 30 GiB, 8-CPU box gives 3 workers.
- [ ] **Step 2: Run them.** FAIL. **Step 3: Implement.** **Step 4: Run** the four files. PASS.

---

### Task 13: docs that travel with the commit (spec §8, C7)

- [ ] **Step 1: ADR-128.** One amendment dated at commit preparation: the scope table of spec §1.3 with M1 to M8;
  Part A (36 units, no InnerPool, the knob); Part C (check 7, `prune`, `compare`, the lost-result rule, the
  `post_verify` rule); the launch-8 rule (a parity proof; nothing adopted when identical); `artifact_size_bytes`
  as a provenance key. Change "Known limits" only where a sentence is now false.
- [ ] **Step 2: The howtos.** `docs/howto/hf-jobs-training.md`: seven checks; `prune` and when to run it (stop
  the driver first); `compare`; a section "Recover a STUCK unit" (the baseline rule and the path); the two
  lost-result reports; `post_verify` for an artifact without a Hub class; `units_per_job_by_stage`; the `fit`
  stage without a 14,400 s limit. `docs/howto/compute-pipelines.md`: the xsuccess `fit` as the example of units
  in place of an InnerPool; remove `fit` from the `timeout_s` and `inner_parallel` examples.
- [ ] **Step 3: The campaign skill.** `.claude/skills/hf-campaign/SKILL.md`: the description line and the command
  list name `prune` and `compare`; the check count is seven. Result: `grep -c "prune" SKILL.md` and
  `grep -c "compare" SKILL.md` are each at least 1, and no line says six checks.
- [ ] **Step 4: `docs/context/tracking-features.md` and `CHANGELOG.md`.** One `CHANGELOG.md` entry per part:
  Part A (the `fit` stage as 36 units; `XSuccessModel` output unchanged); Part B (the pre-index: no output
  change, the removed `FutureWarning` on `category` frame keys, the two timings of Task 1 Step 7); Part C (check
  7, `prune`, `compare`, the lost-result rule, `post_verify`, the read bound, the STUCK line, the card gate).
- [ ] **Step 5: `TODO.md`.** The "Current" line (line 5) names this cycle's release; the model-card row is gone
  (Task 9).
- [ ] **Step 6: The C4 model.** Read `docs/c4/architecture.dsl:50-51` (`stageContract`, `campaignCtl`). No
  container is added or removed by this cycle. If a description names the check count or the command list,
  update it and regenerate `docs/c4/architecture.html` by the repo's C4 procedure; else write "no C4 change:
  <the two descriptions checked>" into the ledger. Run `<base> tests/test_c4_aggregator_count.py
  tests/test_c4_dsl_description_cap.py tests/test_c4_feature_column_count.py`. PASS.
- [ ] **Step 7: C7.** Add to §14.10a of `docs/superpowers/specs/2026-11-04-hfu-reduce-reads-and-job-exit-design.md`:
  `check_params_ulp.py` over `D:/Development/_staging/cycle1-a1a3b52`: 7 of 7 models within 3 ULP, `max_ulp=0`,
  exit 0; I3: `pool_coverage` 51,440 / 50,719, `keep_pool` false (source: the 4.140.0 cycle ledger, line 459).
- [ ] **Step 8: This cycle's spec §6 step 4.** Paste the Task 2 Step 5 output.
- [ ] **Step 9: Run the doc gates.** `<base> tests/test_agents_md_budget.py tests/test_card_regions.py` and any
  test that pins a howto or skill line (`grep -rln "hf-jobs-training\|compute-pipelines\|hf-campaign" tests`).
  PASS.

---

### Task 14: the one-time pre-index job script (spec §4.4) — written here, run in Task 16

- [ ] **Step 1: Write the job script** in `D:\Temp\hfu-next\pre_index_check\` (two files, after the pattern of
  `D:\Temp\hfu\audit\launch_audit_probe.py`): a launcher (one `cpu-upgrade` job at a commit given on the command
  line, the campaign image and install, `tc3` mounted read-only at its locked revision, no store write) and the
  in-job script. The in-job script picks the first unit in sorted order of each of the three providers in the
  `tc3` lock record, loads its frames through `scripts._driver_sources`, runs `pre_index_frames_oracle` and
  `_pre_index_frames` (three timed runs each) and calls `assert_pre_index_equal`. It asserts: exactly three
  matches, one per provider; the two functions are distinct objects; a match that cannot be loaded raises. It
  prints one `PRE_INDEX <provider> <match> frames=<n> oracle_s=<s> new_s=<s> equal=True` line per match and
  `PRE_INDEX_OK 3`. Record the sha256 of both files in the ledger.

---

### Task 15: the gates and the cycle commit

- [ ] **Step 1: Lint and types.** Run: `.venv-hfjobs/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/`,
  `.venv-hfjobs/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/`, and
  `.venv-hfjobs/Scripts/python.exe -m pyright <the changed and new .py files only>`. Expected: exit 0 for each.
  Show the exit codes.
- [ ] **Step 2: Targeted tests.** Run every test file this plan names plus `tests/scripts/hf_jobs`. Expected:
  PASS. Show the exit code and the counts.
- [ ] **Step 3: Implementation review.** Send the working-tree diff to reviewers A and B (`2#(Ragna): Review
  A-Deux`, `2#(D32): Review B-Deux`) with the spec, this plan and the two Task 14 script files. Answer every
  finding. Bring any A/B disagreement to the owner.
- [ ] **Step 4: `/final-review`.**
- [ ] **Step 5: Version.** Ask the owner for the version (the wheel changes). Bump `silly_kicks/_version.py`
  only, and the `CHANGELOG.md` heading.
- [ ] **Step 6: The commit gate.** Show the owner the file list and the diff summary. Ask one question that
  names both actions: "commit and push?". Wait for an explicit yes. Then one commit and the push.
- [ ] **Step 7: CI.** Watch CI (`gh run watch`). Green: go on. Red: read the failing job, fix, run the failing
  test locally, then ask the owner for an amend and for a force-push (two questions), and watch CI again.

---

### Task 16: run the one-time pre-index job

- [ ] **Step 1: Owner go.** Ask for the go, naming the pushed commit and the cost (about 5 min).
- [ ] **Step 2: Run** the Task 14 launcher in the background; save the full log to
  `D:\Temp\hfu-next\pre_index_check\job.log`. Expected: `PRE_INDEX_OK 3`. Anything else stops for the owner; a
  fix follows Task 15 Step 7's amend path.

---

### Task 17: launch 8, the parity proof, the evidence commit (spec §6, §9)

- [ ] **Step 1: Preflight.** `campaign scripts/hf_jobs/campaigns/cycle1.toml preflight`. Expected: seven checks ok.
- [ ] **Step 2: Owner go for the launch.** Then `launch` in the background, with a watcher. No tracked edit until
  Step 5 ends.
- [ ] **Step 3: `retrieve` and `verify`** into `D:\Development\_staging\cycle1-<commit7>`. Expected: 21 of 21 ok.
- [ ] **Step 4: `compare`** with `--baseline D:\Development\_staging\cycle1-a1a3b52`. Expected: the 20 trainer
  instances identical, `census_ok` True, exit 0. **A difference stops for the owner; nothing is adopted.**
- [ ] **Step 5: Record.** From the job records and logs: the job count, the wall time, the xsuccess `fit` stage
  time, the peak RSS of the `fit` units (the unit ledgers). The plan request counts of win_probability,
  xsuccess, the census and ghost_gk/default, each from one `--plan` of the finished instance through a counting
  API wrapper (read-only). The reference counts are 16, 16, 17 and 4 (ADR-128 amendment 2026-11-09, item 26;
  the 4.140.0 ledger, `progress.md:326`). The bound of spec C4: a count may exceed its reference by the
  instance's number of `FileSet` results; for xsuccess also by the added listing pages, computed as
  `ceil(N8 / P) - ceil(N7 / P)`, where `N7` and `N8` are the file counts of the xsuccess run key at launch 7
  (11,973) and at launch 8, and `P` is the page size that the listing requests of the count show. **A count above its bound is a defect: it stops for the owner.** Write the counts,
  the bounds, the `compare` output, and the Task 14 sha256 values with the Task 16 output into spec §6 step 5.
- [ ] **Step 6: The evidence commit gate.** Show the diff (the spec only). Ask "commit and push?"; wait for the
  owner's yes.
- [ ] **Step 7: Run keys.** Show the owner the `prune` listing. Delete only what the owner approves.
- [ ] **Step 8: PR, merge, tag.** Each on its own owner yes; the tag only after CI is green on `main`.

---

## Self-review record

- Spec coverage: §3 → Tasks 10, 11, 12; §3.5 toy corpus → Task 11 Step 5; §4 → Tasks 1, 14, 16; C1 → 3; C2 → 4;
  C3 → 5, 6; C4 → 7; C5 → 8; C6 → 9; C7 → 13; §6 → 2, 17; §7 → the tests of each task; §8 → 13 (every item has a
  step: ADR, two howtos, the skill, the context file, `CHANGELOG.md`, `TODO.md`, the C4 model); §9 → 15 to 17.
- No task has a commit step; the two commit gates are Task 15 Step 6 and Task 17 Step 6.
- Names used across tasks: `pre_index_frames_oracle`, `assert_pre_index_equal` (1, 14);
  `cross_commit_differences`, `compare_staging` (2, 17); `OracleXSuccessModel` (10, 11); `_calibration_splits`,
  `_raw_predict`, `_calibration_fold_proba`, `_calibrate`, `_metadata_payload` (10, 11); `FIT_UNITS`, `ABSENT`
  (11); `units_per_job_by_stage` (12, 13); `lost_units`, `lost_output` (7, 13).

## Round 1 answers

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-plan-A.md` (APPROVE), `…-plan-B.md` (REQUEST CHANGES).
A and B did not disagree on substance.

| Finding | Answer |
|---|---|
| B-01 (blocking), A-05 spec items with no task | Fixed: Task 11 Step 5 (`_toy_corpora.py`), Task 13 Steps 3 to 6 (the skill, `CHANGELOG.md` for all three parts, the `TODO.md` "Current" line, the C4 model). |
| A-01, B-02 the oracle inherits the new `fit` | Fixed: `OracleXSuccessModel` holds today's `fit` and both family paths; the trainer oracle uses it; every Task 10 input holds a `non_action` row and a non-finite row. |
| B-03 never seen red on an inequality | Fixed: a permanent detection test in Tasks 1, 10 and 11, and the rule's own tests in Task 1 Step 1. |
| A-02, B-04 canonical JSON hides a format change | Fixed: Task 11 compares the raw bytes of every file. |
| B-05 A-R2 non-vacuity | Fixed: corpus 1 asserts that a match changes its fold between the two row spaces. |
| A-03, B-06 `compare` and the census | Fixed: the absent-folder rule, `census_ok`, four census tests, the Step 5 expected output in the interface's terms. |
| B-07 the fixtures are inline | Fixed: the `shadow_pre_index_oracle` fixture makes every frame of the three test modules an oracle input. |
| A-04, B-08 the request count; `_attribute` | Fixed: the count rule is a test (Task 7 Step 2); the real counts are recorded after launch 8 (Task 17 Step 5); the `_attribute` test (Step 3a) and the lost-output report. |
| B-09 one yes, two actions | Fixed: the gate question names "commit and push" (global constraints, Tasks 15 and 17). |
| B-10 `cross_val_metrics` | Fixed: the spec §3.2 "Removed" list names it (revision 3a, shown to the owner); Task 11 Step 6 moves the progress test. |
| A-06, B consider: the script before the review; red CI | Fixed: Task 14 writes the script; Task 15 Step 7 names the red-CI path. |
| A-07 to A-13, A-16; B considers | Applied: the two case lists; the input-unchanged assert; the object id with `np.nan`; 4 splits; `_calibration_splits(X, y, seed)`; the store path stated; the empty-corpus guard and the no-xgboost test; the ledger path; the targeted type check; the literal card list; no stale line range; review focus 3, 5, 6, 7. |
| A-14 the `_attribute` step-level test | Fixed: Task 7 Step 3a. |

## Round 2 answers (revision 3)

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-plan-r2-A.md` (APPROVE), `…-plan-r2-B.md` (REQUEST
CHANGES, no blocking finding).

| Finding | Answer |
|---|---|
| A-17, B PLAN-07 the fixture is not visible in `tests/scripts/` | Fixed: the fixture is in `tests/conftest.py`; `test_the_shadow_ran` runs in both directories. |
| A-18, B PLAN-08 (a) the real counts have no rule | Fixed: Task 17 Step 5 states the reference, the bound and the stop. |
| B PLAN-10 the spec change waits for the owner | CONFIRMED by the owner on 2026-11-10 ("Option 1 confirmed"): remove `cross_val_metrics`, the pool path (`_fit_map`, `_fit_task`, `_FIT_ROWS`, `_fit_work`) and both progress lines (`[progress] xsuccess fold k/n` and `[progress] xsuccess fit k/n`). |
| A-19, B-20, B-22 `lost_output` and the spec revision | Fixed: spec 3b names `lost_output` and its report; the header cites 3b. |
| B-21 a lookup per unique id | Fixed: the lookup is per row object in one flat pass; a case with two distinct NaN objects is added; the pass has an ADR-073 guard. |
| B-23 the users of `TOY_BUILDERS` | Fixed: `test_toy_corpora.py` and `hf_jobs/test_scaffold.py` are in Task 11 Step 7. |
