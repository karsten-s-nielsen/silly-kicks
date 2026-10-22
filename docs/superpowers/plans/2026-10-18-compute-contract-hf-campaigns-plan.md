# Stage/unit compute contract + HF campaign orchestrator (Cycle 1) Implementation Plan

> **ADR number:** ADR-128 (renumbered from ADR-127 on 2026-10-18, after TF-46 took ADR-127 on `main` @ `cd2ebcb`;
> the branch was fast-forwarded onto that commit, owner go 2026-10-18).
>
> **For the implementing session:** use `superpowers:executing-plans` INLINE, in this session's checkout,
> on branch `feat/compute-contract-hf-campaigns`. **No subagents. No worktrees. No commit, push, merge, tag
> or Hub publish without the owner's explicit approval of that specific operation.** Ignore any skill text
> that recommends subagent-driven development, worktrees, or per-step commits. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Ship one stage/unit compute contract for every heavy trainer and corpus driver, a committed HF
campaign orchestrator on the typed `huggingface_hub` API, and a representative migrated job set that
exercises every framework part end to end on real HF `cpu-upgrade` jobs.

**Architecture:** `scripts/_stages.py` defines `Pipeline`, `Stage` and `ItemStage`, token-versioned
stores, checksum-verified unit results and the uniform CLI. `scripts/hf_jobs/executor.py` places the
units of one stage as W memory-capped worker subprocesses that claim keys from a job-local queue, and
gives an `inner_parallel` unit a spawn-process pool. HF jobs run the same executor inside a
digest-pinned image built from `uv.lock`. A stateless campaign controller
(`python -m scripts.hf_jobs campaign <file> <command>`) drives preflight, profile, plan, launch, status,
retrieve, verify, adopt and publish from committed TOML files, with the ledger in the bucket.

**Tech stack:** Python 3.10–3.12 (jobs and local verification on 3.12.14), pandas, numpy, pyarrow,
scikit-learn 1.9, xgboost, `ruthless-efficiency[optuna]` 0.7, `huggingface_hub` ≥ 2.2 (typed Jobs and
bucket API), `uv` 0.9.28, pytest, ruff 0.15.7, pyright 1.1.409.

**Spec:** `docs/superpowers/specs/2026-10-18-compute-contract-hf-campaigns-design.md` (r2, approved by
reviewers A and B). Read the spec before every task. Where this plan and the spec disagree, the spec
wins, except for the two spec-text corrections in "Engineering resolutions" below (P1, P2), which
ADR-128 records with their reasons.

---

## Global constraints

Every task implicitly includes these constraints. Values are copied from the spec and the house rules.

- **Flavor:** every HF job runs on `cpu-upgrade` (8 vCPU, 32 GB RAM, 50 GB disk, $0.03/h). No code path
  can select another flavor (INV-4). cpu-upgrade spend for profiling and campaigns is pre-approved.
- **Outward operations are owner-gated, every time:** `git commit`, `git push`, PR creation or update,
  merge, tag, any Hub model or dataset upload, and any non-sandbox publish. Stop and show the exact diff
  or file list, then wait for an explicit "yes" for that operation.
- **One commit = one fully tested, coherent state.** No micro-commits. Cycle 1 has one code commit and,
  if the campaign changes weights or adds the ledger summary, one artifact commit (ADR-037).
- **Docs, specs and plans travel with the approval-gated commit.** Never commit them separately.
- **Independent review:** reviewers A and B review the implementation before any commit. The author never
  runs its own review. Freeze the tree during each review.
- **Local gate before any commit proposal:** the FULL `tests/` tree,
  `python -m pytest tests/ -m "not e2e and not slow" -p no:randomly --benchmark-skip --continue-on-collection-errors`,
  then `python -m pytest tests/ -m "slow and not e2e" -p no:randomly --benchmark-skip` for every slow test
  this cycle adds or touches, plus `uvx ruff@0.15.7 check silly_kicks/ tests/ scripts/`,
  `uvx ruff@0.15.7 format --check silly_kicks/ tests/ scripts/`, and pyright scoped to changed files.
  Never a path subset for the full-suite run. Capture every `FAILED` line, never only the dots.
- **Environment rules:** never `pip install` into `.venv`. `uv run` needs `--no-project`. The pinned
  verification environment is `.venv-hfjobs` (Task 1). Never write the ORIGINAL silly-kicks folder.
- **Shell rules:** a Bash command expected to run longer than 30 s uses `run_in_background`. Never use
  `gh run watch`; poll with `gh run view <id> --json status,conclusion`.
- **Test file basenames are unique repo-wide** (`tests/scripts/` has no `__init__.py`). Before creating a
  test file, run `git ls-files "tests/**/<name>.py"` and confirm it prints nothing.
- **Prose** in docs, comments, commit messages and PR text follows ASD-STE100 (house rule).
- **Commit trailer:** the `Co-Authored-By:` line that the implementing session's attribution reminder
  names (the implementing model); no `Claude-Session:` trailer. PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **ADR number** ADR-128 is provisional. Re-confirm the next free number on `origin/main` and every remote
  branch at the commit gate (`git fetch --all && git ls-tree -r --name-only origin/main docs/superpowers/adrs | tail`).

---

## Evidence verified while planning (2026-10-18)

Each fact below was checked by reading code or by a read-only probe. Line numbers refer to `main` @
`7acf2e6`.

| # | Fact | Source |
|---|---|---|
| E1 | `huggingface_hub` 2.2.0 `HfApi.run_job(*, image, command: list[str], env, secrets, flavor, timeout, attempts, name, labels, volumes: list[Volume], …) -> JobInfo`; `list_jobs(*, status, labels, …)`; `inspect_job(*, job_id)`; `cancel_job(*, job_id)`; `fetch_job_logs(*, job_id, tail)`; `list_jobs_hardware() -> list[JobHardwareInfo]` (fields `name, cpu, ram, unit_cost_usd, …`); `list_bucket_tree(bucket_id, prefix, recursive)`; `download_bucket_files(bucket_id, files)`; `batch_bucket_files(bucket_id, add=…)`; `JobInfo` fields include `id, created_at, started_at, finished_at, labels, status, durations`; `JobStage` values `COMPLETED, CANCELED, ERROR, DELETED, SCHEDULING, RUNNING`. | `inspect.signature` in the scratchpad `hfcli` venv |
| E2 | `Volume(type="dataset", source=…, mount_path=…, revision=…, read_only=…, path=…)`; `read_only` is forced `True` for repos and defaults `False` for buckets. A staged dataset CAN be requested as a read-only, revision-pinned mount. This answers the API half of spec R3; the profile job verifies the runtime half. | `huggingface_hub/_jobs_api.py` source |
| E3 | `uv.lock` pins `huggingface-hub 1.16.1` and has no `statsbombpy`; `pyproject.toml` has no `[dependency-groups]`; `tomli 2.4.1` is in the lock but not in the `[test]` extra; CI installs with `uv pip install -e ".[kloppy,xgboost,test]"` (no lock) on py3.10/3.11/3.12. | `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml:110` |
| E4 | Docker Hub `python:3.12.14-bookworm` index digest `sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0` (pushed 2026-09-19); `python:3.12.14-slim-bookworm` `sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e`. Only the non-slim image contains `git`. | Docker Hub tags API |
| E5 | `uv 0.9.28` manylinux x86_64 wheel sha256 `7b8460a2b624d8ab27cb293a2c9f2393f9efc4e36e0fb886a6c2360e23fb48be` (2026-01-29); the local `uv` is 0.9.28. | PyPI JSON API |
| E6 | `statsbomb/open-data` `master` = `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb` (2026-09-07). The 4.138.0 campaign (2026-10-15) therefore read this revision. `statsbombpy.public` does `from statsbombpy.config import OPEN_DATA_PATHS` and formats those `.../open-data/master/...` URLs, so an in-place edit of that dict pins every call. | GitHub API; statsbombpy source |
| E7 | AGENTS.md is 27 725 EOL-normalized bytes against a 27 750-byte ceiling (`tests/test_agents_md_budget.py:32`): 25 bytes of headroom (spec R6). The invariant inventory checks tokens in `docs/context/*.md`, not in AGENTS.md. | measured |
| E8 | The I1 static closure gate (`tests/scripts/hf_jobs/test_visibility_blind_gate.py`) covers only xshot and xcross. Probing the same `_closure` for the other job modules finds: `train_ghost_gk.py` names `is_public_row` in a comment (`:1149`); `train_receiver_model.py:630` imports `match_visibility`/`select_match_ids` from `_loader_pining` and reaches `_corpus`; `train_win_probability`, `build_territory_ranking_census` and the three adapter trainers reach `_loader_pining` through `scripts/_sb_open_data.py:29` (`from scripts._loader_pining import LoadedMatch, resolve_cache_dir`). | `_closure` probe |
| E9 | In the 4.138.0 xshot/xcross runs `run_paired` was False (the `--arms` payload had `public` + `sc_extended`, or only `public`), so each variant ran ONE 50-trial study (`tag="single"`), whose objective evaluates 5 inner folds serially (`silly_kicks/tracking/_xshot_occurrence_objective.py:77-103`). | retrieved `metrics.json` + code |
| E10 | Retrieved 4.138.0 metrics: xshot/xcross sc_extended `corpus_n_matches=115`, public `17`; ghost `n_games=179`. Every staged match yielded rows, so a staging-time digest over all staged matches equals the in-job digest for these datasets. | scratchpad `retrieved/` |
| E11 | `ghost_gk` extraction token (`build_extraction_inputs`) keys on `feature_set`, `grid_x_max`, `subsample_fps`, carrier params, `with_actions`: `default` and `full` share it; `position_only`, `sweeper`, `sweeper_position_only` each differ. `ghost_outfield`'s token keys on `feature_set`, so its two variants do not share extraction. | `train_ghost_gk.py:182-200`, `train_ghost_outfield.py:43-59` |
| E12 | The registry `ghost_gk` `default` build omits `--variant default` (`registry.py:156`), but the committed default bundle's `metrics.json` records `"variant": "default"`; the trainer default is `full` (`train_ghost_gk.py:241`). A registry re-fit of `default` writes a different `metrics.json`. | code + committed metrics |
| E13 | `win_probability` and the census write `CorpusPassResult.manifest()` fields (`n_attempted`, …) into their outputs; the committed census reads `"n_attempted": 0` because its final pass resumed everything. These fields depend on resume history. | `train_win_probability.py:361`, `docs/research/territory_ranking_census/census.json` |
| E14 | `bootstrap_icc_and_power` draws each replicate with `rng.integers(0, k, size=k)` from ONE stream and consumes no RNG inside the fit (`scripts/_crossed_icc.py`), so pre-drawing all replicates in order and fitting them in parallel is byte-identical. | code |
| E15 | Ruff selects `N` (`pyproject.toml:238`), so a capitalised factory FUNCTION would fail N802; `ItemStage` is a class. | config |
| E16 | `write_table_atomically` uses a fixed temp name `{stem}.{tag}.tmp{suffix}` (`scripts/_partition.py:70`); two concurrent writers with one tag collide. | code |

---

## Engineering resolutions (applied; no owner ruling needed)

Planning found the points below. Each is resolved by engineering judgment inside the approved spec's
intent (INV-1..9, §5); none changes scope, drops a feature's capability, or defers work. Two of them
correct approved spec TEXT and are recorded in ADR-128 with their reason: **P1** (the spec's `-slim`
image has no `git`, so the job could neither check out the exact commit nor record provenance) and
**P2** (the manifest and the dataset record each become a hand-written TOML file plus a tool-written
JSON file, because `tomllib` cannot write and the split makes INV-9's two field classes physical). The
owner sees every consequence with an outward effect at the normal gates: a census number that moves
(P15) at G3, any card text change (P7, Task 19) at G5.

| # | Point | Resolution | Rejected alternative |
|---|---|---|---|
| P1 | In-job image. The spec says `python:3.12.<patch>-slim@sha256:…`. The slim image has no `git`, so the job cannot fetch the exact commit with git, and `git_provenance()` returns `unknown` (which `require_clean_tree` refuses). | Use the digest-pinned NON-slim `python:3.12.14-bookworm@sha256:dbbe4ceb…` (E4). Nothing resolves at job time; provenance code is unchanged. | Slim: Task 6 fetches the GitHub tarball for the SHA and Task 1 adds a new provenance path that trusts a verified archive marker. |
| P2 | File split for the two INV-9 field classes. `tomllib` cannot write TOML. | Expectation fields in `docs/artifacts/manifest.toml` (human-written, never tool-written); observed fields in `docs/artifacts/observed.json` (written only by `adopt`). Declared datasets in `scripts/hf_jobs/datasets.toml`; staging records (revision, units, identity) in `scripts/hf_jobs/datasets.lock.json` (written only by the staging tool), like `pyproject.toml` / `uv.lock`. Add `tomli>=2.0; python_version < '3.11'` to `[test]`. | One TOML file each, plus a `tomli-w` dependency and a TOML writer. |
| P3 | Meaning of "byte-identical artifact" across commits (criterion 3). `metadata.json` and `metrics.json` carry the commit, so they change on every commit even when the weights do not. | Same-commit re-run (INV-1): EVERY file byte-identical. Rebaseline vs 4.138.0: every non-JSON payload file byte-identical; `metadata.json` / `metrics.json` byte-identical after removing the provenance keys in `scripts/hf_jobs/identity.py:PROVENANCE_KEYS` (`training_commit`, `training_platform`, `run_commit`, `run_tree_dirty`, `run_tree_state`, `reproducibility_note`, `generation`, `sklearn_version` is NOT removed). | State another rule; Task 20 changes. |
| P4 | Pass-scoped counters (E13) break INV-1 under resume. | Migrated outputs write the CORPUS-scoped values a single fresh serial pass reports (`n_attempted` = units with a result or exclusion, `n_failed` = failed units, `n_counters_unrecorded` = 0, `n_excluded` = excluded units), and `generation` = the extract stage token. | Keep pass-scoped values and exclude them from identity checks. |
| P5 | INV-5 violations found by E8. | Fix all in Cycle 1 (Task 9): receiver takes `--corpus-visibility` and `--corpus-n-matches` as launcher parameters from `datasets.toml`; neutral `LoadedMatch`/`resolve_cache_dir` move to `scripts/_driver_sources.py` (re-exported by `_loader_pining`); the ghost comment is reworded; the I1 gate covers every module the campaign launches. | Leave as found and list in `_LEGACY_UNMIGRATED` reasons. |
| P6 | Open-data drift. statsbombpy reads `master`. | Pin `statsbomb/open-data` to `4b73468f…` (E6) through `SILLY_KICKS_OPEN_DATA_REVISION`: `scripts/_sb_open_data.assert_statsbomb_open_data_mode()` applies it (Task 9) ONLY when the variable is set (the launcher sets it in every open-data job; `datasets record sb_open` passes the revision explicitly); unset, it is a no-op, so non-campaign readers are unchanged. datasets.toml declares it. | Accept a floating public corpus (breaks INV-1 for public jobs). |
| P7 | Generated card markers change every card's bytes, which would make every Hub README differ from the repo card. | `_hub_publish.card_bytes` strips the `<!-- generated:… -->` marker lines, so the rendered card bytes (and the Hub README) are unchanged. | Re-publish all 10 cards on an owner go after merge. |
| P8 | AGENTS.md has 25 bytes of headroom (E7). | Shorten two Key-conventions bullets whose rationale already lives in `docs/context/tracking-features.md` (the "Vectorized spearman batch" and "Tracking-frame coords are float32" bullets), then add the ADR-128 bullet. Task 22 gives the exact text. | Raise `_CEILING_BYTES` (an anti-bloat brake) by an owner ruling. |
| P9 | Stage token "code version" (spec §5.1). | Default = the run commit (fail-safe: a code change never reuses a stale result). An extraction stage may opt into cross-commit resume with `cross_commit_version="<declared version>"`, which is today's ADR-052 declared-inputs rule. A dirty tree uses `dirty-<git_tree_hash>` (deterministic across processes) instead of a random nonce. The D21 `objective_id` rule for study stores is unchanged. | Another rule; Task 3 changes `compute_tokens`. |
| P10 | E12 registry defect. | The `ghost_gk` `default` build passes `--variant default`; a new `full` build passes `--variant full`. | Leave the defect; the rebaseline then reports a `metrics.json` delta for `default`. |
| P11 | Non-vacuity for `ghost_gk` `importance` (spec §5.2). It runs inside scikit-learn's joblib, so our code cannot observe completion order. | Assert that at W=2 the scorer ran in ≥ 2 distinct worker PIDs, and that importances are byte-identical at W=1 and W=2. | Replace sklearn's parallelism with our own pool (changes scikit-learn's seed stream; spec forbids). |
| P12 | The ADR-052 gate (`tests/scripts/test_corpus_driver_resilience.py`) recognises adoption by a call to `for_each`. Migrated modules declare `ItemStage(...)` instead; the executor calls `for_each`. | Extend the gate: `ItemStage` joins `_DRIVER_API` and `_SHARD_PRIMITIVES`, counts as `for_each` for the two escape-hatch tests, and its `key=` is checked by the key-pin test. Add a planted `ItemStage` driver test. | Make migrated modules call `for_each` directly (bypasses the executor). |
| P13 | Spec §5.3 item 4 (checksum sidecar on resume) for `for_each`. | `_driver.write_shard` writes `<key>.sha256` before the shard; `already_done` verifies it. A NEW generation carries a `.checksummed` marker, so a missing sidecar there means "not done"; a legacy generation without the marker keeps today's behaviour. All 49 `for_each` callers gain verified resume; signatures unchanged. | Checksums only for stage-contract results. |
| P14 | Profile history key (spec §5.5 "a stage token has no history"). With P9, most tokens change every commit. | Key history by `(pipeline, stage, declared-params digest)`, which excludes the code identity. | Profile on every commit. |
| P15 | Census baseline. The committed `census.json` came from 4.120.0 (`dd7ed4f`), not 4.138.0. | Certification `rebaseline` against the committed census: a numeric difference is classified as an environment delta or a code delta (`git log dd7ed4f..HEAD` over the `input_contract()` symbols). A code delta is reported to the owner and is not adopted without a ruling. | Adopt whatever the pinned run produces. |
| P16 | Legacy-equivalence check per migrated module. Committed goldens cannot be platform-portable (Windows dev box, Linux CI, Linux HF). | Per migration, a REQUIRED one-off verification step: run the pre-migration CLI and the new `--run-all` on the same toy corpus in `.venv-hfjobs`, and require byte-identical payload SHA-256 values. The committed W1N1-vs-W2N2 parity tests and the real-HF criterion 3 are the standing gates. | Commit goldens and skip them off the recording platform. |
| P17 | Legacy CLI modes that the stage contract supersedes, and in-job pining reads. | Remove: the per-trainer phase flags (spec §6.1); receiver `--owner-rows` (the store's resume-before-load already avoids the re-parse it existed for); census `--list-matches` (`--plan` lists every stage's `units`). Receiver's live-pining inputs (`--cache-dir`, `--pool-provider` over pining, the `--public-bundle` deployment walk over pining) become staged-dataset parameters (`--dataset`, `--pool-dataset`, `--deployment-dataset`), so the module is I1-clean (INV-5); the DGX runs them on locally materialised staged data (ruling 11). Same numbers when the staged data equals the pining data. | Keep each mode; the receiver then stays off the I1 population and in `_LEGACY_UNMIGRATED`. |
| P18 | The INV-1 campaign gate needs a FRESH store: a same-commit re-run under the derived run key `{campaign}-{commit7}-{job}-{tier}` resumes every unit and proves nothing. | Optional replica suffix: `--replica N` gives `{campaign}-{commit7}-{job}-{tier}-rN` (Task 8); the Task 25 re-run uses `--replica 2`. | Another isolation rule (e.g. a second bucket). |

---

## File structure (module boundaries)

New modules (one responsibility each):

| File | Responsibility |
|---|---|
| `scripts/_toml.py` | `load_toml(path)`: `tomllib` on 3.11+, `tomli` on 3.10. |
| `scripts/_stages.py` | The contract: `Param`, `Stage`, `ItemStage`, `Pipeline`, `FileSet`, `RunContext`, `Store`/`LocalStore`, token computation, checksum-verified unit/output IO, `plan()`, and `main(PIPELINE)` (the uniform CLI). Stdlib at import. |
| `scripts/_driver_sources.py` | Neutral loader port (ADR-052 D14): `LoadedMatch`, `resolve_cache_dir`, `staged_flat_source`, `staged_pergame_source`. The pining adapter stays `_loader_pining.pining_source`; the open-data adapter stays `_sb_open_data.open_data_source`. |
| `scripts/hf_jobs/executor.py` | `LocalExecutor` (W capped worker subprocesses, job-local claim queue, relaunch, failure rule, profile mode), the worker loop (`python -m scripts.hf_jobs.executor worker …`), `InnerPool`, per-unit ledger records. |
| `scripts/hf_jobs/store.py` | `BucketStore`: read-only `Store` over the HF bucket API, for the controller. |
| `scripts/hf_jobs/identity.py` | Artifact identity: payload SHA-256 values, provenance-stripped JSON digests, `compare_artifacts`. |
| `scripts/hf_jobs/ledger.py` | Collect unit/job records, summarise per stage, write the committed campaign summary. |
| `scripts/hf_jobs/planner.py` | Pure planning: N, W, inner W, partition, predicted wall/cost, derived timeout, irreducible floor. |
| `scripts/hf_jobs/datasets.py` | `datasets.toml` + `datasets.lock.json` model; `record` (staging-time identity); dataset contract check. |
| `scripts/hf_jobs/preflight.py` | The six no-spend checks and the report. |
| `scripts/hf_jobs/campaign.py` | Campaign-file model, run keys, the stateless controller and the `campaign` sub-commands. |
| `scripts/hf_jobs/manifest.py` | Artifact manifest + observed fields model, card-region renderer, `WEIGHTS_CLASSIFICATION`/`CARD_SOURCE`/`HUB_REGISTRY` derivation. |
| `scripts/hf_jobs/adopt.py` | The 10 `adopt` steps and the `publish` post-verification. |
| `scripts/hf_jobs/scaffold.py` | `scaffold trainer|driver <name>`. |
| `scripts/hf_jobs/__main__.py` | `python -m scripts.hf_jobs {campaign,scaffold,datasets}` dispatch. |
| `scripts/hf_jobs/campaigns/cycle1.toml` | The Cycle 1 campaign file. |
| `scripts/hf_jobs/datasets.toml`, `datasets.lock.json` | Declared datasets; staging records. |
| `docs/artifacts/manifest.toml`, `docs/artifacts/observed.json` | Artifact expectations; observed fields. |

Rewritten: `scripts/hf_jobs/launch.py` (typed API), `scripts/hf_jobs/registry.py` (`certification`,
`pipeline`, `DriverSpec`, `full` build, `--variant default`), `scripts/hf_jobs/dgx_e2e.py` (stage CLI),
`scripts/hf_jobs/floor.py`/`headtohead.py` (certification policy).

Changed library seams (numerics unchanged, byte-identical by construction):
`silly_kicks/tracking/_xshot_occurrence_objective.py` and `_xcross_attempt_objective.py` (fold plan /
fold eval / combine split + optional `fold_map`), `scripts/_crossed_icc.py` (`replicate_map`).

---

## Commit and review gates

Owner rule (2026-10-18): "minimal commits as needed for provenance, no micro-commits, execution will be
inline." Provenance needs exactly two commits (ADR-037): the CODE commit (the campaign must run on a
pushed, clean commit) and the ARTIFACT commit (adopted outputs produced on that commit's clean tree). A
third, fix-forward code commit exists only if real HF exposes a defect that the campaign commit must
contain; it is proposed like any other commit.

| Gate | When | Content |
|---|---|---|
| G1 | Task 23 | Tree frozen; reviewers A and B review the implementation (owner relays). |
| G2 | Task 23 | Owner approves the code commit; then, separately, the push and the PR. |
| G3 | Task 26 | Owner reviews the campaign report (criteria 2, 3, 4). |
| G5 | Task 27 | Owner approves each `publish --sandbox` run (private repos); any public or card-only publish is a separate go. |
| G4 | Task 28 | Reviewers A and B review the adoption diff; owner approves the artifact commit and, separately, its push. |
| G6 | After CI green | Owner decides version, merge, tag (not part of this plan's execution). |

---

## Task overview (dependency order)

| Task | Deliverable | Depends on |
|---|---|---|
| 1 | Deterministic env, `env_provenance`, `code_identity`, `_toml`, `hfjobs` group, `.venv-hfjobs` | — |
| 2 | Checksum-verified `for_each` resume | — |
| 3 | `_stages.py` contract + CLI + LocalStore | 1, 2 |
| 4 | Executor, worker loop, InnerPool, framework parity/non-vacuity/thread-pin gates | 3 |
| 5 | Ledger, planner, profile model | 4 |
| 6 | Typed launcher, job command, BucketStore, labels, flavor constant | 3, 5 |
| 7 | Datasets model + staging record + `datasets.lock.json` + preflight | 6 |
| 8 | Campaign file + stateless controller + CLI | 5, 6, 7 |
| 9 | `_driver_sources`, open-data pin, receiver INV-5 fix, I1 gate extension, ADR-052 gate `ItemStage` | 3 |
| 10 | Migrate `win_probability` | 4, 9 |
| 11 | Migrate `receiver_model` | 4, 9 |
| 12 | Migrate `ghost_outfield` | 4, 9 |
| 13 | Migrate `ghost_gk` (+ `full`) | 4, 9 |
| 14 | Objective fold split + R1 + migrate `xshot_occurrence` | 4, 9 |
| 15 | Migrate `xcross_attempt` (+ `probe`, dotdir) | 14 |
| 16 | `replicate_map` + migrate the census driver | 4, 9 |
| 17 | One-unit adapter stage for the three public trainers | 4 |
| 18 | Delete the legacy launch path; registry/floor/headtohead/dgx_e2e on the contract | 6, 10–17 |
| 19 | Artifact manifest, observed bootstrap, constant-test rewrites, card regions | 7 |
| 20 | `adopt` + `publish` post-verification + sandbox | 8, 19 |
| 21 | Born-ready gate + scaffolder | 10–18 |
| 22 | Skills, howtos, AGENTS.md, ADR-128, supersede markers, C4, CHANGELOG, TODO | all |
| 23 | Full local gate → freeze → review G1 → code commit G2 | all |
| 24 | Campaign run 1 (real HF): preflight → profile → launch → retrieve → verify → summary | 23 |
| 25 | Replica run (same commit, `--replica 2`): INV-1 campaign gate | 24 |
| 26 | Rebaseline report vs 4.138.0 → owner review G3 | 25 |
| 27 | Adopt (no commit) → `publish --sandbox` end to end G5 | 26 |
| 28 | Review of the adoption diff → artifact commit G4 | 27 |

Fix-forward rule for Tasks 24–28: a defect found on real HF is fixed in code with a failing test first,
the full local gate re-runs, the owner decides whether reviewers re-review, and the fix lands only through
a new owner-approved commit. The campaign then resumes; finished units with unchanged tokens are not
redone.

---
## Task 1: Deterministic environment, environment provenance, TOML shim, `hfjobs` group

Spec: §5.4 (Environment), INV-6, decisions P1, P2, P9.

**Files:**
- Modify: `scripts/_thread_pin.py` (add `DETERMINISTIC_VARS`, `deterministic_env`; leave `thread_pin_env` and `_PIN_VARS` unchanged)
- Modify: `scripts/_provenance.py` (add `env_provenance`, `code_identity`, module-level `import os`)
- Create: `scripts/_toml.py`
- Modify: `pyproject.toml` (new `[dependency-groups] hfjobs`; `tomli` marker-gated in `[project.optional-dependencies] test`)
- Modify: `uv.lock` (regenerated by `uv lock`, never by hand)
- Modify: `.gitignore` (add `.venv-hfjobs/`)
- Test: `tests/scripts/test_det_env.py`, `tests/scripts/test_env_provenance.py`, `tests/scripts/test_toml_shim.py`

**Interfaces:**
- Produces: `scripts._thread_pin.deterministic_env(base: Mapping[str, str] | None = None) -> dict[str, str]`;
  `scripts._thread_pin.DETERMINISTIC_VARS: dict[str, str]`;
  `scripts._provenance.env_provenance(*, root: pathlib.Path | None = None) -> dict`;
  `scripts._provenance.code_identity(prov: Mapping[str, object]) -> str`;
  `scripts._provenance.IMAGE_ENV = "SILLY_KICKS_IMAGE"`, `RUN_KEY_ENV = "SILLY_KICKS_RUN_KEY"`;
  `scripts._toml.load_toml(path) -> dict`, `scripts._toml.loads_toml(text: str) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_det_env.py`:

```python
"""ADR-128 worker environment: every thread pin + numba + the fixed interpreter/locale variables."""

from __future__ import annotations

import os

from scripts._thread_pin import _PIN_VARS, DETERMINISTIC_VARS, deterministic_env, thread_pin_env


def test_deterministic_env_pins_threads_numba_and_fixed_vars():
    env = deterministic_env({"PATH": "/bin", "OMP_NUM_THREADS": "8"})
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        assert env[var] == "1", var
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONUTF8"] == "1"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["TZ"] == "UTC"
    assert env["PATH"] == "/bin"


def test_deterministic_env_does_not_mutate_os_environ():
    before = dict(os.environ)
    deterministic_env()
    assert dict(os.environ) == before


def test_thread_pin_env_variable_set_is_unchanged():
    """Chesterton: `_parallel_launch` (DGX DAS runs) keeps numba threads free. The stage contract uses
    `deterministic_env`; `thread_pin_env` must not grow NUMBA_NUM_THREADS."""
    assert set(_PIN_VARS) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
    assert "NUMBA_NUM_THREADS" not in thread_pin_env({})
    assert set(DETERMINISTIC_VARS) >= set(_PIN_VARS)
```

`tests/scripts/test_env_provenance.py`:

```python
"""ADR-128 environment provenance (INV-6) and the stage-token code identity (decision P9)."""

from __future__ import annotations

import hashlib
import pathlib

import scripts._provenance as prov_mod

_KEYS = {"image", "python", "uv_lock_sha256", "packages", "platform", "machine", "commit", "tree_state", "run_key"}


def test_env_provenance_has_exactly_the_declared_keys(monkeypatch):
    monkeypatch.setenv(prov_mod.IMAGE_ENV, "python:3.12.14-bookworm@sha256:abc")
    monkeypatch.setenv(prov_mod.RUN_KEY_ENV, "cycle1-abc1234-ghost_gk-owner")
    out = prov_mod.env_provenance()
    assert set(out) == _KEYS
    assert out["image"] == "python:3.12.14-bookworm@sha256:abc"
    assert out["run_key"] == "cycle1-abc1234-ghost_gk-owner"
    assert out["packages"]["numpy"]  # installed in every test environment
    assert set(out["packages"]) == set(prov_mod._ENV_PACKAGES)


def test_env_provenance_hashes_the_committed_lock():
    root = pathlib.Path(prov_mod.__file__).resolve().parents[1]
    want = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    assert prov_mod.env_provenance(root=root)["uv_lock_sha256"] == want


def test_env_provenance_unset_image_is_none(monkeypatch):
    monkeypatch.delenv(prov_mod.IMAGE_ENV, raising=False)
    assert prov_mod.env_provenance()["image"] is None


def test_code_identity_clean_is_the_commit():
    assert prov_mod.code_identity({"commit": "a" * 40, "tree_state": "clean"}) == "a" * 40


def test_code_identity_dirty_is_the_deterministic_tree_hash(monkeypatch):
    monkeypatch.setattr(prov_mod, "git_tree_hash", lambda: "deadbeef")
    p = {"commit": "a" * 40, "tree_state": "dirty"}
    assert prov_mod.code_identity(p) == "dirty-deadbeef"
    assert prov_mod.code_identity(p) == prov_mod.code_identity(p)  # two processes agree


def test_code_identity_unknown_never_repeats():
    p = {"commit": "unknown", "tree_state": "unknown"}
    a, b = prov_mod.code_identity(p), prov_mod.code_identity(p)
    assert a.startswith("unknown-") and b.startswith("unknown-") and a != b
```

`tests/scripts/test_toml_shim.py`:

```python
from __future__ import annotations

from scripts._toml import load_toml, loads_toml


def test_load_toml_reads_tables_and_arrays(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('[campaign]\nname = "cycle1"\n[[jobs]]\njob = "ghost_gk"\nvariants = ["default"]\n', encoding="utf-8")
    d = load_toml(p)
    assert d["campaign"]["name"] == "cycle1"
    assert d["jobs"][0]["variants"] == ["default"]


def test_loads_toml_round_trips_text():
    assert loads_toml('a = 1\n') == {"a": 1}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/scripts/test_det_env.py tests/scripts/test_env_provenance.py tests/scripts/test_toml_shim.py -v -p no:randomly`
Expected: FAIL with `ImportError` (`deterministic_env`, `env_provenance`, `scripts._toml` do not exist).

- [ ] **Step 3: Implement**

Append to `scripts/_thread_pin.py` (below `thread_pin_env`):

```python
#: The ADR-128 stage-contract worker environment. Every unit is single-threaded (INV-2), so numba is
#: pinned here too, and the interpreter/locale variables are fixed so string hashing, text encoding,
#: collation and wall-clock rendering cannot differ between a laptop, the DGX and an HF job (INV-1).
#: Kept SEPARATE from `thread_pin_env` on purpose: `_parallel_launch` (the DGX DAS launcher) leaves
#: numba threads free, and widening its variable set would change those runs.
DETERMINISTIC_VARS: dict[str, str] = {
    **dict.fromkeys(_PIN_VARS, "1"),
    "NUMBA_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}


def deterministic_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) copied with every ``DETERMINISTIC_VARS`` entry set."""
    env = dict(os.environ if base is None else base)
    env.update(DETERMINISTIC_VARS)
    return env
```

In `scripts/_provenance.py`, add `import os` and `import sys` to the module imports (keep the existing ones), then append:

```python
#: Set by the HF launcher inside every job (ADR-128). Absent on a laptop or the DGX -> recorded as None.
IMAGE_ENV = "SILLY_KICKS_IMAGE"
RUN_KEY_ENV = "SILLY_KICKS_RUN_KEY"

#: The packages whose versions decide trained numbers (INV-6). Missing ones record None.
_ENV_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "xgboost", "numba", "ruthless-efficiency")


def env_provenance(*, root: pathlib.Path | None = None) -> dict:
    """The environment identity every stage-contract artifact stamps (ADR-128, INV-6).

    ``git_provenance()`` keeps its signature and keys (57 consumers); this is ADDITIVE. ``root`` is the
    repo root holding ``uv.lock`` (default: this checkout).
    """
    from importlib import metadata

    root = root if root is not None else pathlib.Path(__file__).resolve().parents[1]
    lock = root / "uv.lock"
    packages: dict[str, str | None] = {}
    for name in _ENV_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    git = git_provenance()
    return {
        "image": os.environ.get(IMAGE_ENV),
        "python": _platform.python_version(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "packages": packages,
        "platform": git["platform"],
        "machine": git["machine"],
        "commit": git["commit"],
        "tree_state": git["tree_state"],
        "run_key": os.environ.get(RUN_KEY_ENV),
    }


def code_identity(prov: Mapping[str, object]) -> str:
    """The code identity a stage token digests (ADR-128, decision P9).

    Clean tree -> the commit. Dirty tree -> ``dirty-<git tree hash>``: a CONTENT hash, so every process
    that computes it on the same working tree agrees (N simulated jobs on one dev box share stores), and
    any edit changes it (a stale result is never reused). Unknown provenance -> a fresh random nonce, so
    nothing ever resumes on code that cannot be described (the D21 fail-closed reading).
    """
    state = prov.get("tree_state")
    if state == "clean":
        return str(prov["commit"])
    if state == "dirty":
        return f"dirty-{git_tree_hash()}"
    return f"unknown-{uuid.uuid4().hex}"
```

Add `import hashlib` to the imports of `scripts/_provenance.py`. `sys` is not needed; do not add it.

Create `scripts/_toml.py`:

```python
"""Read TOML on every supported interpreter (ADR-128).

``tomllib`` is stdlib from Python 3.11; the CI matrix still runs 3.10, where the API-identical
``tomli`` backport is used (marker-gated in the ``[test]`` extra and resolved in ``uv.lock``). Reading
only: every TOML file in this repo is human-written (the campaign file, ``datasets.toml``, the artifact
manifest). Tool-written records are JSON (decision P2).
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover - exercised on the py3.10 CI leg only
    import tomli as _toml


def load_toml(path) -> dict:
    """Parse the TOML file at ``path``."""
    with open(path, "rb") as fh:
        return _toml.load(fh)


def loads_toml(text: str) -> dict:
    """Parse TOML ``text``."""
    return _toml.loads(text)
```

In `pyproject.toml`, add to the `test = [` list (keep the existing entries and their comments):

```toml
    # ADR-128: scripts/_toml.py reads the campaign/dataset/manifest TOML files; tomllib is 3.11+.
    "tomli>=2.0; python_version < '3.11'",
```

and add a new top-level table after `[project.optional-dependencies]` ends (before `[project.urls]`):

```toml
[dependency-groups]
# ADR-128: the HF campaign launcher and the in-job fetch/run environment. Locked in uv.lock and
# installed with `uv sync --frozen --group hfjobs` -- in every HF job and in the local `.venv-hfjobs`
# verification environment (never in `.venv`). statsbombpy is here so public-data jobs never resolve it
# at job time (the 4.138.0 jobs `pip install`ed it floating).
hfjobs = [
    "huggingface_hub>=2.2,<3",
    "statsbombpy>=1.13",
]
```

Append to `.gitignore`: `.venv-hfjobs/`.

- [ ] **Step 4: Regenerate the lock and build `.venv-hfjobs`**

Run (background; can take minutes):
`uv lock`
Expected: exit 0; `git diff --stat uv.lock` shows `huggingface-hub` moved to a 2.x release, `statsbombpy` added, `tomli` gains the marker. If `uv lock` reports a conflict, STOP and report it; do not loosen another dependency.

Run: `uv lock --check` → expected exit 0.

Run (background):
```
uv python install 3.12.14
UV_PROJECT_ENVIRONMENT=.venv-hfjobs uv sync --frozen --python 3.12.14 --extra train --extra xgboost --extra test --group hfjobs
.venv-hfjobs/Scripts/python.exe -c "import sys, huggingface_hub, sklearn, xgboost; print(sys.version, huggingface_hub.__version__, sklearn.__version__, xgboost.__version__)"
```
(POSIX path: `.venv-hfjobs/bin/python`.) Expected: `3.12.14`, hub `2.x`, sklearn `1.9.x`, xgboost as locked.
`git status --short` must NOT list `.venv-hfjobs/`.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/scripts/test_det_env.py tests/scripts/test_env_provenance.py tests/scripts/test_toml_shim.py -v -p no:randomly`
Expected: PASS (in `.venv`, py3.10, `tomli` from the lock-installed env — if `.venv` lacks `tomli`, run this file in `.venv-hfjobs` and record that `.venv` is a py3.10 legacy env; CI installs `[test]` and gets it).

Also run the existing consumers that must stay green:
`python -m pytest tests/scripts/test_parallel_launch.py tests/scripts/test_provenance_wiring.py -q -p no:randomly`
(`git ls-files "tests/**/test_parallel_launch*.py"` names the launcher test file; use the real name.)

---

## Task 2: Checksum-verified `for_each` resume (spec §5.3 item 4, decision P13)

**Files:**
- Modify: `scripts/_driver.py` (`generation_dir`, `write_shard`, `already_done`; new `checksum_path`, `_write_bytes_atomically`, `_CHECKSUMMED`)
- Test: `tests/scripts/test_driver_checksum.py`

**Interfaces:**
- Produces: `scripts._driver.checksum_path(generation, key) -> pathlib.Path`;
  `scripts._driver._write_bytes_atomically(path: pathlib.Path, data: bytes) -> None` (uuid temp name,
  `os.replace`); `scripts._driver._CHECKSUMMED = ".checksummed"`.
- Unchanged signatures: `for_each`, `write_shard(path, frame, *, tag)`, `already_done(generation, key)`,
  `generation_dir(shard_root, *, token_inputs, token_reason=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_driver_checksum.py`:

```python
"""ADR-128 / spec 5.3 item 4: a resumed for_each never trusts a shard whose checksum fails."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts import _driver as d


def _gen(tmp_path):
    return d.generation_dir(tmp_path / "root", token_inputs={"t": 1})


def test_a_new_generation_is_marked_checksummed(tmp_path):
    gen = _gen(tmp_path)
    assert (gen / d._CHECKSUMMED).is_file()


def test_write_shard_writes_a_matching_sidecar(tmp_path):
    gen = _gen(tmp_path)
    p = d.shard_path(gen, "m1")
    d.write_shard(p, pd.DataFrame({"a": [1, 2]}), tag="w0")
    side = d.checksum_path(gen, "m1")
    assert side.read_text(encoding="utf-8").strip() == hashlib.sha256(p.read_bytes()).hexdigest()
    assert d.already_done(gen, "m1")


def test_a_corrupted_shard_is_not_done_and_is_removed(tmp_path):
    gen = _gen(tmp_path)
    p = d.shard_path(gen, "m2")
    d.write_shard(p, pd.DataFrame({"a": [1]}), tag="w0")
    p.write_bytes(p.read_bytes()[:-7])  # truncated / poisoned on the bucket
    with pytest.warns(UserWarning, match="checksum mismatch"):
        assert not d.already_done(gen, "m2")
    assert not p.exists() and not d.checksum_path(gen, "m2").exists()


def test_a_marked_generation_without_a_sidecar_is_not_done(tmp_path):
    gen = _gen(tmp_path)
    pd.DataFrame({"a": [1]}).to_parquet(d.shard_path(gen, "m3"), index=False)  # written by nothing we trust
    with pytest.warns(UserWarning, match="untrusted shard"):
        assert not d.already_done(gen, "m3")


def test_a_legacy_generation_keeps_file_exists_means_done(tmp_path):
    legacy = tmp_path / "legacy" / "0123456789abcdef"
    legacy.mkdir(parents=True)  # pre-ADR-128: no .checksummed marker
    pd.DataFrame({"a": [1]}).to_parquet(d.shard_path(legacy, "m4"), index=False)
    assert d.already_done(legacy, "m4")


def test_for_each_redoes_a_poisoned_shard_on_resume(tmp_path):
    calls = []

    def work(item):
        calls.append(item)
        return pd.DataFrame({"v": [item]})

    kw = {"key": str, "work": work, "shard_root": tmp_path / "r", "token_inputs": {"x": 1}}
    res = d.for_each(["a", "b"], **kw)
    d.shard_path(res.shard_dir, "a").write_bytes(b"not parquet")
    calls.clear()
    with pytest.warns(UserWarning, match="checksum mismatch"):
        d.for_each(["a", "b"], **kw)
    assert calls == ["a"]


def test_no_temp_files_survive(tmp_path):
    gen = _gen(tmp_path)
    d.write_shard(d.shard_path(gen, "m5"), pd.DataFrame({"a": [1]}), tag="w0")
    assert not [p for p in gen.iterdir() if p.name.endswith(".part")]
```

- [ ] **Step 2: Run and confirm the failures**

Run: `python -m pytest tests/scripts/test_driver_checksum.py -v -p no:randomly`
Expected: FAIL (`_CHECKSUMMED`/`checksum_path` missing; the poisoned shard is reported done).

- [ ] **Step 3: Implement in `scripts/_driver.py`**

Add below `KEY_SEPARATOR`:

```python
#: Written into every generation directory this module CREATES (ADR-128, spec 5.3 item 4). In a marked
#: generation every shard carries a ``<key>.sha256`` sidecar and a shard whose sidecar is missing or does
#: not match is NOT done. A generation without the marker predates checksums and keeps the old rule
#: (file exists == done), so resuming a pre-ADR-128 generation behaves exactly as before.
_CHECKSUMMED = ".checksummed"
```

Replace the body of `generation_dir` after `gen = root / _token(...)`:

```python
    _warn_on_flat_shards(root, gen)
    is_new = not gen.exists()
    gen.mkdir(parents=True, exist_ok=True)
    if is_new:  # idempotent: two processes creating one new generation both write the same marker
        (gen / _CHECKSUMMED).write_text("sha256\n", encoding="utf-8")
    return gen
```

Add after `shard_path`:

```python
def checksum_path(generation, key) -> pathlib.Path:
    """The ``<key>.sha256`` sidecar beside a shard -- invisible to every ``*.parquet`` glob."""
    return pathlib.Path(generation) / f"{join_key(key)}.sha256"


def _write_bytes_atomically(path: pathlib.Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a UNIQUE temp name + ``os.replace``.

    Unique (uuid) rather than ``write_table_atomically``'s fixed ``{stem}.{tag}.tmp`` name, because two
    jobs may write the same deterministic file concurrently (an idempotent reduce, ADR-128); a fixed temp
    name lets one writer replace the other's half-written temp.
    """
    import os
    import uuid

    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
```

Replace `write_shard`'s body (keep its docstring and add one paragraph to it, stated below):

```python
    import hashlib
    import io

    import pandas as pd

    path = pathlib.Path(path)
    buf = io.BytesIO()
    (pd.DataFrame() if frame is None else frame).to_parquet(buf, index=False)
    data = buf.getvalue()
    # Sidecar FIRST: a kill between the two writes leaves a sidecar with no shard, which reads as "not
    # run" and is overwritten by the re-run. The reverse order would leave a marked-generation shard
    # with no sidecar, which reads as untrusted and is redone -- also safe, but wasteful.
    side = path.with_name(f"{path.name[: -len('.parquet')]}.sha256")
    _write_bytes_atomically(side, f"{hashlib.sha256(data).hexdigest()}\n".encode("ascii"))
    _write_bytes_atomically(path, data)
```

Docstring paragraph to append to `write_shard`: "``tag`` is accepted for signature compatibility and no longer names the temp file (ADR-128: temp names are unique). Every shard gets a ``<key>.sha256`` sidecar; `already_done` verifies it."

Replace `already_done`:

```python
def already_done(generation, key) -> bool:
    """True when this item's shard exists in this generation and can be TRUSTED -- empty shards included.

    In a ``.checksummed`` generation (ADR-128) the shard's sidecar must exist and match its bytes; a
    mismatching or missing sidecar means the shard was truncated or poisoned (e.g. on a bucket mount),
    so the shard and sidecar are deleted with a warning and the item is redone (INV-3: never trust a unit
    whose checksum fails). In a legacy generation a present sidecar is still verified; an absent one
    keeps the old "file exists == done" rule.
    """
    import hashlib

    gen = pathlib.Path(generation)
    shard = shard_path(gen, key)
    if not shard.is_file():
        return False
    side = checksum_path(gen, key)
    if not side.is_file():
        if (gen / _CHECKSUMMED).is_file():
            warnings.warn(f"untrusted shard {shard} (no checksum sidecar in a checksummed generation); redoing it", stacklevel=2)
            shard.unlink()
            return False
        return True
    want = side.read_text(encoding="ascii").strip()
    if hashlib.sha256(shard.read_bytes()).hexdigest() == want:
        return True
    warnings.warn(f"checksum mismatch for {shard}; deleting it and redoing the item", stacklevel=2)
    shard.unlink()
    side.unlink()
    return False
```

- [ ] **Step 4: Run the new tests and every existing driver test**

Run: `python -m pytest tests/scripts/test_driver_checksum.py tests/scripts/test_driver.py tests/scripts/test_driver_load_hook.py tests/scripts/test_driver_resume_oracle.py tests/scripts/test_partition.py tests/scripts/test_corpus_driver_resilience.py -q -p no:randomly`
Expected: PASS. If a test asserted that a generation directory holds ONLY `*.parquet` files, it now sees
`.checksummed` and `*.sha256`: read why it asserted that (Chesterton). If the assertion protects a glob
consumer, the consumer already uses `*.parquet` and the test is updated to the same glob; record each such
edit in the task notes for the reviewers.

Then run the 49 `for_each` callers' own tests in one batch (background):
`python -m pytest tests/ -m "not e2e and not slow" -p no:randomly --benchmark-skip -q -k "driver or shard or resume or build_ or validate_ or measure_ or calibrate_ or train_"`
Expected: PASS (this is a smoke batch, not the gate; the full gate is Task 23).

---
## Task 3: The stage/unit contract — `scripts/_stages.py`

Spec: §5.1, §5.2 mechanisms 2–4, §5.3 items 1, 2, 4, INV-1, INV-3, decision P9.

**Files:**
- Create: `scripts/_stages.py`
- Test: `tests/scripts/test_stage_contract.py`

**Interfaces:**
- Consumes: `scripts._driver._token`, `_write_bytes_atomically`, `join_key`, `generation_dir`,
  `write_exclusion`, `for_each`; `scripts._provenance.git_provenance`, `require_clean_tree`,
  `env_provenance`, `code_identity`; `scripts._item_outcome.ItemExcluded`.
- Produces (later tasks rely on these exact names):
  - `KEY_WIDTH = 4`, `int_key(i: int) -> str`
  - `FileSet(files: Mapping[str, bytes])`
  - `Param(name, type=str, default=None, required=False, choices=None, flag=False, help="")`
  - `Stage(name, units, work, needs=(), reduce=None, params=(), declare=None, cross_commit_version=None, inner_parallel=False, reduce_order=None, max_failures=0, abort_first_k=3, uses_dataset=False)`
    — `units(ctx) -> Sequence[str]`, `work(key: str, ctx) -> result`, `reduce(results: dict[str, Any], ctx) -> result`
  - `ItemStage(name, refs, key, work, load=None, counters=None, needs=(), reduce=None, params=(), declare=None, cross_commit_version=None, reduce_order=None, max_failures=0, abort_first_k=3, uses_dataset=False)`
    — `refs(ctx) -> Sequence[ref]`, `key(ref) -> key`, `load(ref, ctx) -> item`, `work(item, ctx) -> DataFrame | None`
  - `Pipeline(name, module, stages, params=(), output=None)` with `.stage(name)`, `.validate()`
  - `RunContext(pipeline, params, prov, store, data_root, token_inputs, tokens, pool=None)` with
    `.stage_dir(stage)`, `.generation(stage)`, `.upstream(stage)`, `.summary(stage)`
  - `Store` protocol; `LocalStore(root)`; `open_store(uri: str) -> Store`
  - `compute_tokens(pipeline, params, code_id, dataset_identity) -> tuple[dict[str, dict], dict[str, str]]`
  - `make_context(pipeline, params, prov, *, store, data_root) -> RunContext`
  - `write_result(store, base, result) -> str`, `read_result(store, base) -> Any`, `MISSING`
  - `unit_states(ctx, stage, keys, *, verify: bool) -> dict[str, str]` (values `done|excluded|failed|pending`)
  - `sorted_units(ctx, stage) -> list[str]`
  - `run_unit(ctx, stage, key, *, worker: str, pool=None, ref=None) -> str`
  - `reduce_stage(ctx, name) -> None` (raises `StageIncompleteError`); `output_present(ctx, name, *, verify) -> bool`
  - `plan(ctx) -> dict`; `export_output(ctx, dest: pathlib.Path) -> None`
  - `context_to_json(ctx) -> str`, `context_from_json(text: str) -> RunContext`
  - `ordered_values(results: Mapping[str, Any]) -> list`
  - `materialize_fileset(ctx, stage, key, dest) -> pathlib.Path`; `item_counters(ctx, stage) -> tuple[dict, int]`
  - `build_parser(pipeline) -> argparse.ArgumentParser`; `main(pipeline, argv=None) -> int`
  - Exceptions: `StageIncompleteError`, `UpstreamNotReducedError`

**Store layout (one meaning per path; every path is POSIX-relative to the store root):**

| Path | Meaning |
|---|---|
| `<pipeline>/<stage>/<token>/` | the stage generation (token = `_driver._token(token_inputs)`, 16 hex) |
| `…/<token>/.checksummed` | marker (Task 2) |
| `…/<token>/<key>.parquet` \| `.json` \| `.fileset.json` | one unit result (exactly one of the three) |
| `…/<token>/<key>.files/<name>` | the files of a `FileSet` unit result |
| `…/<token>/<key>.sha256` | checksum of the primary result bytes (written FIRST) |
| `…/<token>/<key>.excluded.json` | ADR-052 D13 exclusion marker |
| `…/<token>/<key>.counters.json` | ADR-052 counters sidecar (ItemStage) |
| `…/<token>/<key>.failed.json` | recorded failure (retried on the next run) |
| `…/<token>/<key>.ledger.json` | per-unit ledger record (NOT a stage output; excluded from identity) |
| `…/<token>/_output.*`, `_output.sha256`, `_output.files/` | the reduced stage output |
| `…/<token>/_summary.json` | corpus-scoped unit counts written at reduce time (decision P4) |
| `…/<token>/_profile.json` | profile measurement (Task 5) |

Keys never start with `_` (reserved for `_output`, `_summary`, `_profile`); `sorted_units` refuses one.

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_stage_contract.py` (pure, in-process; the executor gates are Task 4):

```python
"""ADR-128 stage contract: DAG validation, tokens, checksum-verified results, plan JSON, reduce order."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import _stages as s


def _p(**kw):
    def units(ctx):
        return [s.int_key(i) for i in range(3)]

    stages = kw.pop(
        "stages",
        (
            s.Stage(name="a", units=units, work=lambda k, c: {"k": k}, params=("x",)),
            s.Stage(name="b", units=lambda c: ["only"], work=lambda k, c: {"n": 1}, needs=("a",)),
        ),
    )
    return s.Pipeline(name="t", module="tests.scripts._toy_pipeline", stages=stages, params=(s.Param("x", int, 1),), **kw)


def _ctx(tmp_path, pipeline=None, params=None, code="c" * 40):
    pipeline = pipeline or _p()
    prov = {"commit": code, "tree_state": "clean", "code_identity": code}
    return s.make_context(pipeline, params or {"x": 1}, prov, store=s.LocalStore(tmp_path), data_root=None)


def test_int_key_orders_numerically():
    assert sorted([s.int_key(i) for i in (10, 2, 1)]) == ["0001", "0002", "0010"]


def test_validate_rejects_a_forward_need():
    bad = (s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1, needs=("a",)),
           s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1))
    with pytest.raises(ValueError, match="declared BEFORE"):
        _p(stages=bad).validate()


def test_validate_rejects_an_undeclared_param():
    bad = (s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, params=("nope",)),)
    with pytest.raises(ValueError, match="undeclared param"):
        _p(stages=bad).validate()


def test_tokens_change_with_params_upstream_and_code_but_not_with_unrelated_params(tmp_path):
    p = _p()
    _, t1 = s.compute_tokens(p, {"x": 1}, "c1", None)
    _, t2 = s.compute_tokens(p, {"x": 2}, "c1", None)
    _, t3 = s.compute_tokens(p, {"x": 1}, "c2", None)
    assert t1["a"] != t2["a"] and t1["b"] != t2["b"]  # b changes through its upstream token
    assert t1["a"] != t3["a"]


def test_cross_commit_version_replaces_the_code_identity():
    st = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, cross_commit_version="extract-v1")
    p = _p(stages=(st,))
    assert s.compute_tokens(p, {"x": 1}, "c1", None)[1] == s.compute_tokens(p, {"x": 1}, "c2", None)[1]


def test_dataset_identity_enters_only_dataset_stages():
    a = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True)
    b = s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1)
    p = _p(stages=(a, b))
    _, t1 = s.compute_tokens(p, {"x": 1}, "c", {"name": "d", "revision": "r1"})
    _, t2 = s.compute_tokens(p, {"x": 1}, "c", {"name": "d", "revision": "r2"})
    assert t1["a"] != t2["a"] and t1["b"] == t2["b"]


@pytest.mark.parametrize(
    "result",
    [pd.DataFrame({"a": [1.5, 2.0]}), {"z": [1, 2.5, None]}, s.FileSet({"model.json": b"{}", "SHA256SUMS": b"x\n"})],
)
def test_results_round_trip_with_a_verified_checksum(tmp_path, result):
    store = s.LocalStore(tmp_path)
    s.write_result(store, "p/st/tok/0001", result)
    back = s.read_result(store, "p/st/tok/0001")
    if isinstance(result, pd.DataFrame):
        pd.testing.assert_frame_equal(back, result)
    elif isinstance(result, s.FileSet):
        assert dict(back.files) == dict(result.files)
    else:
        assert back == result


def test_write_result_is_byte_deterministic(tmp_path):
    a, b = s.LocalStore(tmp_path / "a"), s.LocalStore(tmp_path / "b")
    for st in (a, b):
        s.write_result(st, "g/k", pd.DataFrame({"x": [0.1, 0.2]}))
    assert (tmp_path / "a/g/k.parquet").read_bytes() == (tmp_path / "b/g/k.parquet").read_bytes()


def test_unit_states_detect_a_poisoned_result(tmp_path):
    ctx = _ctx(tmp_path)
    gen = ctx.generation("a")
    s.write_result(ctx.store, f"{gen}/0000", {"k": "0000"})
    (tmp_path / gen / "0000.json").write_bytes(b'{"k":"tampered"}\n')
    assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000"], verify=False) == {"0000": "done"}
    with pytest.warns(UserWarning, match="checksum"):
        assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000"], verify=True) == {"0000": "pending"}


def test_reduce_receives_results_in_reduce_order_not_write_order(tmp_path):
    seen = []

    def red(results, ctx):
        seen.extend(results)
        return {"order": list(results)}

    st = s.Stage(name="a", units=lambda c: ["0002", "0000", "0001"], work=lambda k, c: {"k": k}, reduce=red)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    for k in ("0002", "0000", "0001"):  # completion order != key order
        s.run_unit(ctx, st, k, worker="w0")
    s.reduce_stage(ctx, "a")
    assert seen == ["0000", "0001", "0002"]
    assert ctx.upstream("a") == {"order": ["0000", "0001", "0002"]}
    assert json.loads((tmp_path / ctx.generation("a") / "_summary.json").read_text())["n_done"] == 3


def test_reduce_refuses_an_incomplete_stage(tmp_path):
    ctx = _ctx(tmp_path)
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0000", worker="w0")
    with pytest.raises(s.StageIncompleteError):
        s.reduce_stage(ctx, "a")


def test_a_failed_unit_is_recorded_not_raised_and_blocks_a_zero_tolerance_reduce(tmp_path):
    def work(k, c):
        if k == "0001":
            raise ValueError("bad item")
        return {"k": k}

    st = s.Stage(name="a", units=lambda c: ["0000", "0001"], work=work)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    assert s.run_unit(ctx, st, "0000", worker="w0") == "done"
    assert s.run_unit(ctx, st, "0001", worker="w0") == "failed"
    assert "bad item" in (tmp_path / ctx.generation("a") / "0001.failed.json").read_text()
    with pytest.raises(s.StageIncompleteError, match="failed"):
        s.reduce_stage(ctx, "a")


def test_plan_reports_counts_and_unknown_downstream_units(tmp_path):
    ctx = _ctx(tmp_path)
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0000", worker="w0")
    out = s.plan(ctx)
    a, b = out["stages"]
    assert (a["n_units"], a["done"], a["pending"], a["reduced"]) == (3, 1, 2, False)
    assert b["n_units"] == 1  # static units are known even before the upstream reduces
    assert b["ready"] is False


def test_context_json_round_trips(tmp_path):
    ctx = _ctx(tmp_path)
    back = s.context_from_json(s.context_to_json(ctx))
    assert back.tokens == ctx.tokens and back.params == ctx.params and back.pipeline.name == "toy"


def test_keys_may_not_start_with_an_underscore(tmp_path):
    st = s.Stage(name="a", units=lambda c: ["_output"], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    with pytest.raises(ValueError, match="reserved"):
        s.sorted_units(ctx, st)


def test_keys_may_not_contain_a_dot(tmp_path):
    st = s.Stage(name="a", units=lambda c: ["a.counters"], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    with pytest.raises(ValueError, match="outside"):
        s.sorted_units(ctx, st)
```

`test_context_json_round_trips` imports the toy module by name, so it needs Task 4's
`tests/scripts/_toy_pipeline.py`; write that file in this task (Task 4 Step 1 gives its content) and keep it
minimal until Task 4 extends it.

- [ ] **Step 2: Run and confirm the failures**

Run: `python -m pytest tests/scripts/test_stage_contract.py -v -p no:randomly`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._stages'`.

- [ ] **Step 3: Implement `scripts/_stages.py`**

```python
"""The stage/unit compute contract (ADR-128).

WHY THIS EXISTS. The 4.138.0 re-fit campaign ran 5-9 hour jobs on one of eight vCPUs, from uncommitted
scratchpad orchestration, because the per-trainer phase flags (ADR-123) had untested real-HF paths. This
module is the ONE contract every heavy trainer and corpus driver declares instead: a static DAG of
stages, each a set of independent, keyed, single-threaded units plus an optional reducer. The executor
(`scripts/hf_jobs/executor.py`) places units as W worker processes per job x N jobs, and every output is
byte-identical for every (W, N) (INV-1) because:

1. a unit is single-threaded (the executor starts workers with `_thread_pin.deterministic_env`);
2. a unit writes ONE result file per key, atomically, with a checksum sidecar written first;
3. every reduction receives results in a declared total order of keys, never completion order
   (floating-point addition is not associative, so this rule is load-bearing);
4. a token digests the code identity, the stage's declared parameters, the dataset identity and the
   upstream tokens, so a stale or foreign result is never reused.

Resume (INV-3): a result is trusted only when its checksum matches; a kill loses at most the units in
flight. ADR-052's `for_each` is the implementation of an `ItemStage`, so its shards, exclusion markers,
counters sidecars, conservation and injectivity apply unchanged.

Vocabulary (spec section 3, one meaning each): unit, stage, pipeline, worker, job, store.

Stdlib only at import; pandas is imported inside the functions that need it (the `_driver` rule).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import io
import json
import os
import pathlib
import re
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from scripts._driver import _token, _write_bytes_atomically, join_key
from scripts._item_outcome import ItemExcluded

#: Numeric key components are zero-padded to this width, so lexicographic key order == numeric order.
KEY_WIDTH = 4
_RESULT_SUFFIXES = (".parquet", ".json", ".fileset.json")
_OUTPUT = "_output"
MISSING = object()


class StageIncompleteError(RuntimeError):
    """A reduce was requested while units are pending, or failures exceed the stage's tolerance."""


class UpstreamNotReducedError(RuntimeError):
    """A stage read an upstream stage whose output does not exist yet."""


def int_key(i: int) -> str:
    """A zero-padded key component for an integer (folds, replicates): ``int_key(3) == "0003"``."""
    if i < 0:
        raise ValueError(f"key components are non-negative, got {i}")
    return f"{i:0{KEY_WIDTH}d}"


@dataclasses.dataclass(frozen=True)
class FileSet:
    """A result made of named files (e.g. a model artifact, or a persisted study-input directory).

    Names are RELATIVE POSIX paths (``"model.json"``, ``"corpus/X.npy"``): no absolute path, no ``..``,
    no backslash, no component starting with ``.``. A model artifact uses flat names only (ADR-088's
    top-level-files rule is enforced again at publish)."""

    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        for name in self.files:
            parts = pathlib.PurePosixPath(name).parts
            if not name or "\\" in name or name.startswith("/") or not parts or any(p in ("..", "") or p.startswith(".") for p in parts):
                raise ValueError(f"FileSet names must be relative POSIX paths without dot-components, got {name!r}")


@dataclasses.dataclass(frozen=True)
class Param:
    """One pipeline parameter; its CLI flag is ``--`` + ``name`` with ``_`` -> ``-``."""

    name: str
    type: Callable[[str], Any] = str
    default: Any = None
    required: bool = False
    choices: tuple[Any, ...] | None = None
    flag: bool = False
    help: str = ""

    @property
    def cli(self) -> str:
        return "--" + self.name.replace("_", "-")


@dataclasses.dataclass(frozen=True)
class Stage:
    """A set of keyed units plus an optional reducer (spec 5.1).

    ``units(ctx)`` enumerates the keys deterministically (it may read upstream outputs).
    ``work(key, ctx)`` is pure, deterministic and single-threaded and returns ONE result: a DataFrame,
    a JSON-native value or a ``FileSet``. ``reduce(results, ctx)`` receives ``{key: result}`` in
    ``reduce_order`` order (default: sorted keys) and returns the stage output. An ``inner_parallel``
    unit runs alone at the top level of a job and receives ``ctx.pool`` (an ``InnerPool``); every
    combination of inner results MUST iterate ``ordered_values`` (sorted inner keys).

    ``params`` names the pipeline parameters the stage's numbers depend on; ``declare(params)`` adds
    any further declared inputs (the ADR-052 rule: a token closes silent omission, not mis-declaration).
    ``cross_commit_version`` opts the stage into resume across commits (decision P9); default None
    means the run commit is part of the token. ``max_failures`` (None = unlimited) is the failed-unit
    tolerance: the executor aborts when it is exceeded and the reduce refuses. ``abort_first_k``: the
    executor aborts when the first k units that complete in this job have all failed (systematic bug).
    """

    name: str
    units: Callable[[RunContext], Sequence[str]]
    work: Callable[[str, RunContext], Any]
    needs: tuple[str, ...] = ()
    reduce: Callable[[dict[str, Any], RunContext], Any] | None = None
    params: tuple[str, ...] = ()
    declare: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None
    cross_commit_version: str | None = None
    inner_parallel: bool = False
    reduce_order: Callable[[str], Any] | None = None
    max_failures: int | None = 0
    abort_first_k: int = 3
    uses_dataset: bool = False
    #: An inner_parallel stage's inner key count (e.g. 5 CV folds): the planner never asks for more
    #: inner workers than there are inner keys. Ignored when ``inner_parallel`` is False.
    inner_width: int = 1


@dataclasses.dataclass(frozen=True)
class ItemStage:
    """A stage whose units are corpus ITEMS, executed through `_driver.for_each` (ADR-052).

    ``refs(ctx)`` returns cheap references; ``key(ref)`` the ADR-052 shard key (keys stay the
    pre-migration keys, `_KEY_EXCEPTIONS`); ``load(ref, ctx)`` the item (resume-before-load);
    ``work(item, ctx)`` one tidy DataFrame or None ("ran, produced nothing"). Exclusions
    (`ItemExcluded`), counters sidecars and conservation are `for_each`'s.
    """

    name: str
    refs: Callable[[RunContext], Sequence[Any]]
    key: Callable[[Any], Any]
    work: Callable[[Any, RunContext], Any]
    load: Callable[[Any, RunContext], Any] | None = None
    counters: Callable[[Any, Any], Mapping[str, object]] | None = None
    needs: tuple[str, ...] = ()
    reduce: Callable[[dict[str, Any], RunContext], Any] | None = None
    params: tuple[str, ...] = ()
    declare: Callable[[Mapping[str, Any]], Mapping[str, object]] | None = None
    cross_commit_version: str | None = None
    reduce_order: Callable[[str], Any] | None = None
    max_failures: int | None = 0
    abort_first_k: int = 3
    uses_dataset: bool = False
    inner_parallel: bool = dataclasses.field(default=False, init=False)

    def units(self, ctx: RunContext) -> list[str]:
        return [join_key(self.key(r)) for r in self.refs(ctx)]

    def ref_map(self, ctx: RunContext) -> dict[str, Any]:
        return {join_key(self.key(r)): r for r in self.refs(ctx)}


AnyStage = Stage | ItemStage


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """A static DAG of stages. Declaration order IS the topological order (validated)."""

    name: str
    module: str
    stages: tuple[AnyStage, ...]
    params: tuple[Param, ...] = ()
    output: str | None = None

    def stage(self, name: str) -> AnyStage:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(f"{self.name}: no stage {name!r} (stages: {[s.name for s in self.stages]})")

    def validate(self) -> None:
        names = [st.name for st in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate stage names {names}")
        declared = {p.name for p in self.params}
        seen: set[str] = set()
        for st in self.stages:
            if not st.name or st.name.startswith("_") or "/" in st.name or "__" in st.name:
                raise ValueError(f"{self.name}: invalid stage name {st.name!r}")
            for need in st.needs:
                if need not in seen:
                    raise ValueError(f"{self.name}: {st.name} needs {need!r}, which is not declared BEFORE it")
            for p in st.params:
                if p not in declared:
                    raise ValueError(f"{self.name}: stage {st.name} names undeclared param {p!r}")
            seen.add(st.name)
        if self.output is not None and self.output not in seen:
            raise ValueError(f"{self.name}: output stage {self.output!r} is not declared")


class Store(Protocol):
    """POSIX-relative byte store. `LocalStore` executes; `hf_jobs.store.BucketStore` only reads."""

    def read_bytes(self, rel: str) -> bytes | None: ...
    def write_bytes(self, rel: str, data: bytes) -> None: ...
    def names(self, rel_dir: str) -> set[str]: ...
    def delete(self, rel: str) -> None: ...
    def local_path(self, rel: str) -> pathlib.Path | None: ...


@dataclasses.dataclass(frozen=True)
class LocalStore:
    root: pathlib.Path

    def _p(self, rel: str) -> pathlib.Path:
        return pathlib.Path(self.root) / pathlib.PurePosixPath(rel)

    def read_bytes(self, rel: str) -> bytes | None:
        p = self._p(rel)
        return p.read_bytes() if p.is_file() else None

    def write_bytes(self, rel: str, data: bytes) -> None:
        p = self._p(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomically(p, data)

    def names(self, rel_dir: str) -> set[str]:
        p = self._p(rel_dir)
        return {c.name for c in p.iterdir() if c.is_file()} if p.is_dir() else set()

    def delete(self, rel: str) -> None:
        self._p(rel).unlink(missing_ok=True)

    def local_path(self, rel: str) -> pathlib.Path | None:
        return self._p(rel)


def open_store(uri: str | os.PathLike) -> Store:
    """``hf://buckets/<ns>/<bucket>/<prefix>`` -> read-only BucketStore; anything else -> LocalStore."""
    text = str(uri)
    if text.startswith("hf://buckets/"):
        from scripts.hf_jobs.store import BucketStore

        return BucketStore.from_uri(text)
    return LocalStore(pathlib.Path(text))


@dataclasses.dataclass
class RunContext:
    pipeline: Pipeline
    params: Mapping[str, Any]
    prov: Mapping[str, Any]
    store: Store
    data_root: pathlib.Path | None
    token_inputs: Mapping[str, Mapping[str, object]]
    tokens: Mapping[str, str]
    pool: Any = None

    def stage_dir(self, stage: str) -> str:
        return f"{self.pipeline.name}/{stage}"

    def generation(self, stage: str) -> str:
        return f"{self.stage_dir(stage)}/{self.tokens[stage]}"

    def upstream(self, stage: str) -> Any:
        out = read_result(self.store, f"{self.generation(stage)}/{_OUTPUT}")
        if out is MISSING:
            raise UpstreamNotReducedError(f"{self.pipeline.name}: upstream stage {stage!r} has no output yet")
        return out

    def summary(self, stage: str) -> dict:
        raw = self.store.read_bytes(f"{self.generation(stage)}/_summary.json")
        if raw is None:
            raise UpstreamNotReducedError(f"{self.pipeline.name}: stage {stage!r} has no summary yet")
        return json.loads(raw)


def _canon_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_tokens(
    pipeline: Pipeline, params: Mapping[str, Any], code_id: str, dataset_identity: Mapping[str, object] | None
) -> tuple[dict[str, dict], dict[str, str]]:
    """The declared token inputs and the token (generation name) of every stage, in DAG order."""
    inputs: dict[str, dict] = {}
    tokens: dict[str, str] = {}
    for st in pipeline.stages:
        d: dict[str, object] = {
            "pipeline": pipeline.name,
            "stage": st.name,
            "code": st.cross_commit_version or code_id,
            "params": {p: (params[p].as_posix() if isinstance(params[p], pathlib.PurePath) else params[p]) for p in st.params},
            "upstream": {n: tokens[n] for n in sorted(st.needs)},
        }
        if st.declare is not None:
            d["declared"] = dict(st.declare(params))
        if st.uses_dataset:
            d["dataset"] = dict(dataset_identity) if dataset_identity is not None else None
        # JSON-canonical BEFORE digesting: `ruthless.fingerprint` is type-tagged (a tuple and a list
        # digest differently), and the inputs reach workers through JSON (`context_to_json`). Without
        # this, a declared tuple would give the parent and the worker two different generations.
        d = json.loads(json.dumps(d, sort_keys=True))
        inputs[st.name] = d
        tokens[st.name] = _token(d, None)
    return inputs, tokens


def _dataset_identity(params: Mapping[str, Any], data_root: pathlib.Path | None) -> dict | None:
    """``{"name", "repo", "revision"}`` from ``datasets.lock.json`` for a declared dataset; for a local
    dev/test corpus (no ``dataset`` param), a digest of the corpus file listing (names + sizes)."""
    name = params.get("dataset")
    if name:
        from scripts.hf_jobs.datasets import locked

        try:
            rec = locked(str(name))
            return {"name": str(name), "repo": rec.get("repo"), "revision": rec["revision"]}
        except KeyError:
            if os.environ.get("SILLY_KICKS_IN_HF_JOB"):  # in a job every dataset is recorded; never guess
                raise
    if data_root is None:
        return None
    root = pathlib.Path(data_root) / str(name) if name else pathlib.Path(data_root)
    rows = sorted((p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file())
    return {"name": str(name or "local"), "listing_sha256": _sha(_canon_json(rows))}


def make_context(
    pipeline: Pipeline, params: Mapping[str, Any], prov: Mapping[str, Any], *, store: Store, data_root
) -> RunContext:
    pipeline.validate()
    code_id = str(prov.get("code_identity") or prov["commit"])
    inputs, tokens = compute_tokens(pipeline, params, code_id, _dataset_identity(params, data_root))
    return RunContext(pipeline, dict(params), dict(prov), store, data_root, inputs, tokens)


def _encode(result: Any) -> tuple[str, bytes, dict[str, bytes]]:
    if isinstance(result, FileSet):
        manifest = {"files": {n: _sha(b) for n, b in sorted(result.files.items())}}
        return ".fileset.json", _canon_json(manifest), dict(result.files)
    if hasattr(result, "to_parquet"):
        buf = io.BytesIO()
        result.to_parquet(buf, index=False)
        return ".parquet", buf.getvalue(), {}
    if result is None:
        import pandas as pd

        buf = io.BytesIO()
        pd.DataFrame().to_parquet(buf, index=False)
        return ".parquet", buf.getvalue(), {}
    return ".json", _canon_json(result), {}


def write_result(store: Store, base: str, result: Any) -> str:
    """Persist one result at ``base`` (+ suffix). Checksum sidecar FIRST, primary bytes LAST."""
    suffix, data, extra = _encode(result)
    for other in _RESULT_SUFFIXES:
        if other != suffix:
            store.delete(base + other)
    for name, payload in sorted(extra.items()):
        store.write_bytes(f"{base}.files/{name}", payload)
    digest = _sha(data)
    store.write_bytes(base + ".sha256", f"{digest}\n".encode("ascii"))
    store.write_bytes(base + suffix, data)
    return digest


def _present_suffix(names: set[str], stem: str) -> str | None:
    hits = [sfx for sfx in _RESULT_SUFFIXES if f"{stem}{sfx}" in names]
    return hits[0] if hits else None


def _verified(store: Store, base: str, suffix: str) -> bool:
    data = store.read_bytes(base + suffix)
    side = store.read_bytes(base + ".sha256")
    if data is None or side is None or _sha(data) != side.decode("ascii").strip():
        return False
    if suffix == ".fileset.json":
        for name, want in json.loads(data)["files"].items():
            payload = store.read_bytes(f"{base}.files/{name}")
            if payload is None or _sha(payload) != want:
                return False
    return True


def read_result(store: Store, base: str) -> Any:
    """The verified result at ``base``, or ``MISSING`` (absent or failing its checksum)."""
    rel_dir, _, stem = base.rpartition("/")
    suffix = _present_suffix(store.names(rel_dir), stem)
    if suffix is None or not _verified(store, base, suffix):
        return MISSING
    data = store.read_bytes(base + suffix)
    assert data is not None  # noqa: S101 -- _verified read it a moment ago
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(io.BytesIO(data))
    if suffix == ".json":
        return json.loads(data)
    names = json.loads(data)["files"]
    return FileSet({n: store.read_bytes(f"{base}.files/{n}") or b"" for n in names})


#: A joined key is a file stem: letters, digits, '_' and '-' only. No '.', so `<key>.json` (a result) can
#: never collide with another key's `<key>.counters.json` / `.ledger.json` / `.failed.json` sidecar.
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


def sorted_units(ctx: RunContext, st: AnyStage) -> list[str]:
    keys = [join_key(k) for k in st.units(ctx)]
    bad = [k for k in keys if k.startswith("_")]
    if bad:
        raise ValueError(f"{st.name}: keys {bad[:3]} start with '_', which is reserved for stage files")
    odd = [k for k in keys if not _KEY_RE.match(k)]
    if odd:
        raise ValueError(f"{st.name}: keys {odd[:3]} contain characters outside [A-Za-z0-9_-]")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{st.name}: units() is not injective")
    return sorted(keys)


def unit_states(ctx: RunContext, st: AnyStage, keys: Sequence[str], *, verify: bool) -> dict[str, str]:
    """``done`` | ``excluded`` | ``failed`` | ``pending`` per key. ``verify`` re-hashes every present
    result (the in-job resume check, INV-3); a failing one is deleted with a warning and is pending.
    Without ``verify`` (the controller's plan) presence of result + sidecar counts as done."""
    gen = ctx.generation(st.name)
    names = ctx.store.names(gen)
    out: dict[str, str] = {}
    for k in keys:
        suffix = _present_suffix(names, k)
        if suffix is not None and f"{k}.sha256" in names:
            if not verify or _verified(ctx.store, f"{gen}/{k}", suffix):
                out[k] = "done"
                continue
            warnings.warn(f"checksum failure for {gen}/{k}{suffix}; deleting it and redoing the unit", stacklevel=2)
            for sfx in (suffix, ".sha256"):
                ctx.store.delete(f"{gen}/{k}{sfx}")
        if f"{k}.excluded.json" in names:
            out[k] = "excluded"
        elif f"{k}.failed.json" in names:
            out[k] = "failed"
        else:
            out[k] = "pending"
    return out


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS high-water mark for this process (Linux ``clear_refs``)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_rss(reset_ok: bool) -> tuple[int | None, str]:
    """(bytes, scope): ``unit`` after a successful reset, else ``process`` (a monotone upper bound)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024, ("unit" if reset_ok else "process")
    except OSError:
        pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]
        return int(ru * (1 if sys.platform == "darwin" else 1024)), "process"
    except (ImportError, AttributeError):
        return None, "unavailable"


def _previous_attempts(store: Store, base: str) -> int:
    raw = store.read_bytes(base + ".ledger.json")
    return int(json.loads(raw).get("attempt", 0)) if raw else 0


def _run_item(ctx: RunContext, st: ItemStage, key: str, ref: Any) -> tuple[str, str | None]:
    from scripts._driver import for_each

    root = ctx.store.local_path(ctx.stage_dir(st.name))
    if root is None:
        raise RuntimeError("an ItemStage executes only on a local (or mounted) store")
    load = (lambda r: st.load(r, ctx)) if st.load is not None else None
    res = for_each(
        [ref],
        key=st.key,
        load=load,
        work=lambda item: st.work(item, ctx),
        counters=st.counters,
        shard_root=root,
        token_inputs=ctx.token_inputs[st.name],
        tag=st.name,
        label=st.name,
        max_consecutive_failures=2,  # one item per call: a failure is recorded, never an abort
    )
    if res.shard_dir.name != ctx.tokens[st.name]:
        raise RuntimeError(f"{st.name}: for_each generation {res.shard_dir.name} != stage token {ctx.tokens[st.name]}")
    if res.failures:
        return "failed", str(next(iter(res.failures.values())))
    if res.exclusions:
        return "excluded", None
    return "done", None


def run_unit(ctx: RunContext, st: AnyStage, key: str, *, worker: str, pool: Any = None, ref: Any = None) -> str:
    """Run ONE unit, persist its result (or exclusion / failure) and its ledger record. Returns the outcome."""
    gen = ctx.generation(st.name)
    base = f"{gen}/{key}"
    attempt = _previous_attempts(ctx.store, base) + 1
    reset_ok = _reset_peak_rss()
    started = time.time_ns()
    outcome, error = "done", None
    unit_ctx = dataclasses.replace(ctx, pool=pool)
    try:
        if isinstance(st, ItemStage):
            outcome, error = _run_item(unit_ctx, st, key, ref)
        else:
            write_result(ctx.store, base, st.work(key, unit_ctx))
    except ItemExcluded as exc:
        from scripts._driver import write_exclusion

        local = ctx.store.local_path(gen)
        if local is None:
            raise
        local.mkdir(parents=True, exist_ok=True)
        write_exclusion(local, key, exc)
        outcome = "excluded"
    except MemoryError:
        outcome, error = "failed", "MemoryError: unit exceeded its memory cap"
    except Exception as exc:  # broad BY DESIGN: recorded per unit, never swallowed (the for_each rule)
        outcome, error = "failed", f"{type(exc).__name__}: {exc}"
    finished = time.time_ns()
    if outcome == "failed":
        ctx.store.write_bytes(base + ".failed.json", _canon_json({"error": error, "attempt": attempt}))
    else:
        ctx.store.delete(base + ".failed.json")
    peak, scope = _peak_rss(reset_ok)
    record = {
        "stage": st.name,
        "key": key,
        "outcome": outcome,
        "error": error,
        "worker": worker,
        "attempt": attempt,
        "job_label": os.environ.get("SILLY_KICKS_JOB_LABEL"),
        "started_ns": started,
        "finished_ns": finished,
        "wall_s": (finished - started) / 1e9,
        "peak_rss_bytes": peak,
        "rss_scope": scope,
        "inner_workers": getattr(pool, "workers", None),
        "inner_completion": list(getattr(pool, "drain_log", lambda: [])()),
    }
    ctx.store.write_bytes(base + ".ledger.json", _canon_json(record))
    return outcome


def output_present(ctx: RunContext, name: str, *, verify: bool) -> bool:
    gen = ctx.generation(name)
    suffix = _present_suffix(ctx.store.names(gen), _OUTPUT)
    if suffix is None:
        return False
    return (not verify) or _verified(ctx.store, f"{gen}/{_OUTPUT}", suffix)


def reduce_stage(ctx: RunContext, name: str) -> None:
    """Write the stage output from every unit result, in ``reduce_order``. Idempotent and atomic, so
    two jobs that both see a complete stage may both reduce: they write identical bytes."""
    st = ctx.pipeline.stage(name)
    keys = sorted_units(ctx, st)
    states = unit_states(ctx, st, keys, verify=True)
    pending = [k for k, v in states.items() if v == "pending"]
    failed = [k for k, v in states.items() if v == "failed"]
    if pending:
        raise StageIncompleteError(f"{ctx.pipeline.name}/{name}: {len(pending)} unit(s) pending, e.g. {pending[:3]}")
    if st.max_failures is not None and len(failed) > st.max_failures:
        raise StageIncompleteError(
            f"{ctx.pipeline.name}/{name}: {len(failed)} unit(s) failed (tolerance {st.max_failures}): {failed[:5]}"
        )
    gen = ctx.generation(name)
    excluded = {}
    for k in keys:
        if states[k] == "excluded":
            raw = ctx.store.read_bytes(f"{gen}/{k}.excluded.json")
            excluded[k] = json.loads(raw)["reason"] if raw else "unknown"
    done = sorted((k for k, v in states.items() if v == "done"), key=st.reduce_order or (lambda k: k))
    summary = {
        "stage": name,
        "token": ctx.tokens[name],
        "n_units": len(keys),
        "n_done": len(done),
        "n_excluded": len(excluded),
        "n_failed": len(failed),
        "failed_keys": failed,
        "excluded": excluded,
    }
    ctx.store.write_bytes(f"{gen}/_summary.json", _canon_json(summary))
    results = {k: read_result(ctx.store, f"{gen}/{k}") for k in done}
    output = st.reduce(results, ctx) if st.reduce is not None else summary
    write_result(ctx.store, f"{gen}/{_OUTPUT}", output)


def ordered_values(results: Mapping[str, Any]) -> list:
    """Values in sorted-key order -- the ONLY way an inner combination may iterate pool results."""
    return [results[k] for k in sorted(results)]


def materialize_fileset(ctx: RunContext, stage: str, key: str, dest: pathlib.Path) -> pathlib.Path:
    """Copy a verified FileSet (unit ``key``, or the stage output when ``key == "_output"``) to local
    ``dest`` -- e.g. so a worker can memory-map a persisted design matrix. Returns ``dest``."""
    res = read_result(ctx.store, f"{ctx.generation(stage)}/{key}")
    if not isinstance(res, FileSet):
        raise TypeError(f"{stage}/{key} is not a FileSet result")
    for name, data in res.files.items():
        p = pathlib.Path(dest) / pathlib.PurePosixPath(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return pathlib.Path(dest)


def item_counters(ctx: RunContext, stage: str) -> tuple[dict, int]:
    """An ItemStage's ADR-052 counters merged in sorted key order (the `_driver._merge_counters` rule),
    and the number of done units whose sidecar is missing (`n_counters_unrecorded`)."""
    from scripts._driver import _merge_counters

    st = ctx.pipeline.stage(stage)
    gen = ctx.generation(stage)
    totals: dict = {}
    missing = 0
    for key, state in unit_states(ctx, st, sorted_units(ctx, st), verify=False).items():
        if state != "done":
            continue
        raw = ctx.store.read_bytes(f"{gen}/{key}.counters.json")
        if raw is None:
            missing += 1
            continue
        _merge_counters(totals, json.loads(raw), key)
    return totals, missing


def _history_key(token_inputs: Mapping) -> str:
    """Decision P14 (single-sourced with `hf_jobs.ledger.history_key`, which delegates here): costs are
    keyed by WHAT a stage computes, never by the commit or the upstream tokens."""
    keep = {k: token_inputs.get(k) for k in ("pipeline", "stage", "params", "declared", "dataset")}
    return _sha(json.dumps(keep, sort_keys=True).encode("utf-8"))[:16]


def _ready(ctx: RunContext, st: AnyStage) -> bool:
    return all(output_present(ctx, n, verify=False) for n in st.needs)


def plan(ctx: RunContext) -> dict:
    """The JSON the launcher reads (spec 5.1): per stage token, unit counts, done/pending, readiness."""
    stages = []
    for st in ctx.pipeline.stages:
        gen = ctx.generation(st.name)
        rec: dict[str, Any] = {
            "name": st.name,
            "needs": list(st.needs),
            "token": ctx.tokens[st.name],
            "generation": gen,
            "inner_parallel": st.inner_parallel,
            "inner_width": getattr(st, "inner_width", 1),
            "history_key": _history_key(ctx.token_inputs[st.name]),
            "item_stage": isinstance(st, ItemStage),
            "ready": _ready(ctx, st),
            "reduced": output_present(ctx, st.name, verify=False),
            "n_units": None,
            "done": 0,
            "excluded": 0,
            "failed": 0,
            "pending": None,
            "units": None,
        }
        try:
            keys = sorted_units(ctx, st)
        except UpstreamNotReducedError:
            keys = None
        if keys is not None:
            states = unit_states(ctx, st, keys, verify=False)
            rec.update(
                n_units=len(keys),
                done=sum(v == "done" for v in states.values()),
                excluded=sum(v == "excluded" for v in states.values()),
                failed=sum(v == "failed" for v in states.values()),
                pending=sum(v == "pending" for v in states.values()),
                units=keys,
            )
        stages.append(rec)
    return {
        "pipeline": ctx.pipeline.name,
        "module": ctx.pipeline.module,
        "output": ctx.pipeline.output,
        "params": {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in ctx.params.items()},
        "code_identity": ctx.prov.get("code_identity"),
        "stages": stages,
    }


def export_output(ctx: RunContext, dest: pathlib.Path) -> None:
    """Write the pipeline output (a FileSet) into ``dest`` (``--run-all --out``)."""
    if ctx.pipeline.output is None:
        raise ValueError(f"{ctx.pipeline.name} declares no output stage")
    out = ctx.upstream(ctx.pipeline.output)
    if not isinstance(out, FileSet):
        raise TypeError(f"{ctx.pipeline.name}: output stage must return a FileSet, got {type(out).__name__}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in out.files.items():
        _write_bytes_atomically(dest / name, data)


def _jsonable(params: Mapping[str, Any]) -> dict:
    return {k: (str(v) if isinstance(v, pathlib.PurePath) else v) for k, v in params.items()}


def context_to_json(ctx: RunContext) -> str:
    root = ctx.store.local_path("")
    if root is None:
        raise ValueError("only a local store can be handed to workers")
    return json.dumps(
        {
            "module": ctx.pipeline.module,
            "params": _jsonable(ctx.params),
            "prov": ctx.prov,
            "store_root": str(root),
            "data_root": None if ctx.data_root is None else str(ctx.data_root),
            "token_inputs": ctx.token_inputs,
            "tokens": ctx.tokens,
        },
        sort_keys=True,
    )


def context_from_json(text: str) -> RunContext:
    d = json.loads(text)
    pipeline = importlib.import_module(d["module"]).PIPELINE
    params = dict(d["params"])
    for p in pipeline.params:  # restore declared types (argparse already converted them once)
        if params.get(p.name) is not None and p.type is pathlib.Path:
            params[p.name] = pathlib.Path(params[p.name])
    data_root = None if d["data_root"] is None else pathlib.Path(d["data_root"])
    return RunContext(pipeline, params, d["prov"], LocalStore(pathlib.Path(d["store_root"])), data_root, d["token_inputs"], d["tokens"])


def build_parser(pipeline: Pipeline) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"python -m {pipeline.module}",
        description=f"{pipeline.name}: ADR-128 stage-contract pipeline ({', '.join(s.name for s in pipeline.stages)}).",
    )
    for p in pipeline.params:
        if p.flag:
            ap.add_argument(p.cli, dest=p.name, action="store_true", help=p.help)
        else:
            ap.add_argument(p.cli, dest=p.name, type=p.type, default=p.default, required=p.required, choices=p.choices, help=p.help)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the plan JSON (stages, tokens, unit counts) and exit")
    mode.add_argument("--stages", help="comma-separated stages to run in order; only the first may be sharded")
    mode.add_argument("--run-all", action="store_true", help="run the whole DAG locally (the serial parity reference)")
    ap.add_argument("--store", required=True, help="store root: a directory, or hf://buckets/<ns>/<b>/<prefix> (--plan only)")
    ap.add_argument("--data-root", type=pathlib.Path, default=None, help="where the dataset is mounted or unpacked")
    ap.add_argument("--shard", default="0/1", help="I/N: this job's disjoint slice of the first stage's units")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inner-workers", type=int, default=1, help="pool size for an inner_parallel stage")
    ap.add_argument("--cap-bytes", type=int, default=None, help="per-worker memory cap (planner)")
    ap.add_argument("--peak-rss-bytes", type=int, default=None, help="per-unit peak RSS hint for worker sizing")
    ap.add_argument("--profile", type=int, default=None, help="profile mode: run K evenly spaced units at W=1 then W=max")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="--run-all only: write the output FileSet here")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; recorded, never laundered)")
    return ap


def _parse_shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    shard = (int(i), int(n))
    if shard[1] < 1 or not (0 <= shard[0] < shard[1]):
        raise ValueError(f"--shard must be I/N with 0 <= I < N, got {text!r}")
    return shard


def main(pipeline: Pipeline, argv: Sequence[str] | None = None) -> int:
    """The uniform CLI every migrated trainer and driver exposes (spec 5.1)."""
    args = build_parser(pipeline).parse_args(argv)
    params = {p.name: getattr(args, p.name) for p in pipeline.params}
    from scripts._provenance import code_identity, env_provenance, git_provenance, require_clean_tree

    git = git_provenance()
    if not args.plan:
        require_clean_tree(git, allow_dirty=args.allow_dirty)
    prov = {**git, "env": env_provenance(), "code_identity": code_identity(git)}
    ctx = make_context(pipeline, params, prov, store=open_store(args.store), data_root=args.data_root)
    if args.plan:
        print(json.dumps(plan(ctx), sort_keys=True))
        return 0
    from scripts._parallel_launch import RefusalError
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor, StageFailedError

    ex = LocalExecutor(
        ExecutorConfig(
            workers=args.workers,
            inner_workers=args.inner_workers,
            cap_bytes=args.cap_bytes,
            peak_rss_bytes=args.peak_rss_bytes,
            profile_units=args.profile,
        )
    )
    try:
        if args.run_all:
            for st in pipeline.stages:
                ex.run_stage(ctx, st.name, shard=(0, 1))
                ex.require_reduced(ctx, st.name)
            if args.out is not None:
                export_output(ctx, args.out)
            return 0
        names = [n for n in args.stages.split(",") if n]
        shard = _parse_shard(args.shard)
        if len(names) > 1 and shard[1] != 1:
            raise SystemExit("--stages with more than one stage requires --shard 0/1 (only a single stage fans out)")
        for i, name in enumerate(names):
            ex.run_stage(ctx, name, shard=shard if i == 0 else (0, 1))
            ex.try_reduce(ctx, name)
        return 0
    except RefusalError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (StageFailedError, StageIncompleteError) as exc:
        print(f"STAGE FAILED: {exc}", file=sys.stderr)
        return 3
```

Note on `--plan`: it never requires a clean tree (it writes nothing). `ensure upstream reduced` lives in
the executor (Task 4) because a reduce runs in a deterministic-env subprocess.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/scripts/test_stage_contract.py -v -p no:randomly`
Expected: PASS. `test_reduce_receives_results_in_reduce_order_not_write_order` calls `reduce_stage`
in-process on purpose (a pure ordering check); production reduces run through the executor.

- [ ] **Step 5: Lint and type-check the new module**

Run: `uvx ruff@0.15.7 check scripts/_stages.py tests/scripts/test_stage_contract.py && uvx ruff@0.15.7 format --check scripts/_stages.py tests/scripts/test_stage_contract.py && pyright scripts/_stages.py`
Expected: no findings. A forward reference to `RunContext` inside the `Stage` annotations is legal under
`from __future__ import annotations`.

---
## Task 4: Executor, worker loop, `InnerPool`, and the framework CI gates

Spec: §5.2 (executor refinement, mechanisms 1–3, memory bound, failure rule, CI determinism gates),
§5.3 (resume), INV-1, INV-2, INV-3.

**Files:**
- Create: `scripts/hf_jobs/executor.py`
- Create: `tests/scripts/_toy_pipeline.py` (the framework's toy pipeline; started in Task 3)
- Test: `tests/scripts/hf_jobs/test_executor.py`, `tests/scripts/hf_jobs/test_inner_pool.py`

**Interfaces:**
- Consumes: Task 3 (`_stages`), `scripts._mem_cap.detect_backend/preflight/wrap`,
  `scripts._parallel_launch.size_workers/should_launch/available_ram_bytes/RefusalError`,
  `scripts._thread_pin.deterministic_env`, `scripts._driver.shard_select`.
- Produces: `ExecutorConfig(workers=1, inner_workers=1, cap_bytes=None, peak_rss_bytes=None, headroom_bytes=2 GiB, max_relaunch=3, poll_interval_s=0.2, profile_units=None, mem_backend=None)`;
  `LocalExecutor(cfg).run_stage(ctx, name, *, shard=(0, 1)) -> StageReport`, `.try_reduce(ctx, name) -> bool`,
  `.require_reduced(ctx, name) -> None`; `StageReport(stage, n_mine, n_done, n_excluded, n_failed, workers, relaunched, completion_order)`;
  `StageFailedError`; `InnerPool(workers, *, cap_bytes=None, backend="none")` with `.map_keyed(fn, items) -> dict`,
  `.drain_log() -> list[str]`, `.shutdown()`; CLI `python -m scripts.hf_jobs.executor {worker,reduce} …`.

- [ ] **Step 1: Write the toy pipeline**

`tests/scripts/_toy_pipeline.py`:

```python
"""A toy ADR-128 pipeline that exercises every executor path (framework tests only).

`gen` is an ItemStage whose low keys are SLOW, so at W >= 2 completion order differs from key order.
`folds` is inner_parallel: each unit maps 4 inner keys over the InnerPool, inner key 0000 is slow.
`final` combines fold scores by a SEQUENTIAL float sum whose value depends on order (1e16 + 1 loses the
1), so a reduce that iterated completion order instead of key order would change the bytes.
`env` returns the worker's environment pins. `strict` fails on `--fail-keys` with zero tolerance.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, int_key, main, ordered_values
from scripts._thread_pin import DETERMINISTIC_VARS

_VALUES = (1e16, 1.0, -1e16, 1.0)


def _refs(ctx):
    return [("toy", int_key(i)) for i in range(int(ctx.params["n_items"]))]


def _crash_once(ctx, key: str) -> None:
    marker = ctx.params.get("crash_once")
    if marker and key == "0003" and not pathlib.Path(marker).exists():
        pathlib.Path(marker).write_text("crashed", encoding="utf-8")
        os._exit(137)  # an OOM-killed worker: no cleanup, no ledger record


def _gen_work(item, ctx):
    _, key = item
    _crash_once(ctx, key)
    if f"toy__{key}" in str(ctx.params.get("fail_keys") or "").split(","):  # gen keys are JOINED keys
        raise ValueError(f"injected failure for {key}")
    time.sleep(0.4 if int(key) % 4 == 0 else 0.0)
    return pd.DataFrame({"i": [int(key)], "v": [_VALUES[int(key) % len(_VALUES)]]})


def _gen_reduce(results, ctx):
    return pd.concat(list(results.values()), ignore_index=True)


def _inner_eval(fold: int, j: int, values: tuple) -> float:
    time.sleep(0.3 if j == 0 else 0.0)
    return float(values[(fold + j) % len(values)])


def _fold_work(key, ctx):
    df = ctx.upstream("gen")
    items = {int_key(j): (int(key), j, tuple(df["v"].tolist())) for j in range(4)}
    res = ctx.pool.map_keyed(_inner_eval, items)
    total = 0.0
    for v in ordered_values(res):  # sequential, order-sensitive: the canonical inner reduce
        total += v
    return {"fold": key, "score": total}


def _final_work(key, ctx):
    folds = ctx.upstream("folds")
    total = 0.0
    for fold in sorted(folds):
        total += folds[fold]["score"]
    return FileSet({"result.json": json.dumps({"total": total}, sort_keys=True).encode()})


def _env_work(key, ctx):
    return {v: os.environ.get(v) for v in sorted(DETERMINISTIC_VARS)}


def _strict_work(key, ctx):
    fail = str(ctx.params.get("fail_keys") or "")
    if fail == "*" or key in fail.split(","):
        raise ValueError(f"strict failure {key}")
    return {"k": key}


PIPELINE = Pipeline(
    name="toy",
    module="tests.scripts._toy_pipeline",
    params=(
        Param("n_items", int, 12),
        Param("fail_keys", str, ""),
        Param("crash_once", str, ""),
    ),
    stages=(
        ItemStage(name="gen", refs=_refs, key=lambda r: r, work=_gen_work, reduce=_gen_reduce, params=("n_items",), max_failures=None),
        Stage(
            name="folds",
            units=lambda ctx: [int_key(f) for f in range(3)],
            work=_fold_work,
            needs=("gen",),
            inner_parallel=True,
            reduce=lambda results, ctx: dict(results),
        ),
        Stage(name="final", units=lambda ctx: ["only"], work=_final_work, needs=("folds",), reduce=lambda r, ctx: r["only"]),
        Stage(name="env", units=lambda ctx: ["only"], work=_env_work),
        Stage(name="strict", units=lambda ctx: [int_key(i) for i in range(6)], work=_strict_work, params=("fail_keys",)),
        Stage(
            name="sloppy",
            units=lambda ctx: [int_key(i) for i in range(6)],
            work=_strict_work,
            params=("fail_keys",),
            max_failures=None,  # tolerant: only the first-k rule can abort it
        ),
    ),
    output="final",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
```

Note: `crash_once` and `fail_keys` are pipeline params that `gen` does NOT declare except `n_items`, so a
crash/fail run and a clean run share `gen` tokens on purpose (the resume test relies on it).

- [ ] **Step 2: Write the failing framework tests**

`tests/scripts/hf_jobs/test_inner_pool.py`:

```python
"""InnerPool: results keyed, completion order recorded, combination order is the caller's (sorted)."""

from __future__ import annotations

from scripts._stages import ordered_values
from scripts.hf_jobs.executor import InnerPool
from tests.scripts._toy_pipeline import _inner_eval


def test_w1_runs_in_sorted_key_order():
    pool = InnerPool(1)
    out = pool.map_keyed(_inner_eval, {"0001": (0, 1, (1.0, 2.0)), "0000": (0, 0, (1.0, 2.0))})
    assert pool.drain_log() == ["0000", "0001"]
    assert ordered_values(out) == [1.0, 2.0]


def test_w2_completes_out_of_order_but_keys_the_results():
    pool = InnerPool(2)
    try:
        items = {f"000{j}": (0, j, (1.0, 2.0, 3.0, 4.0)) for j in range(4)}
        out = pool.map_keyed(_inner_eval, items)
        log = pool.drain_log()
    finally:
        pool.shutdown()
    assert sorted(log) == sorted(items)
    assert log != sorted(items)  # non-vacuity: inner key 0000 sleeps, so it finishes last
    assert ordered_values(out) == [1.0, 2.0, 3.0, 4.0]
```

`tests/scripts/hf_jobs/test_executor.py`:

```python
"""ADR-128 framework gates (spec 5.2): W1N1 vs W2N2 byte identity, non-vacuity, thread pins, resume,
failure rule, checksum resume. Every run goes through the real CLI and real worker subprocesses."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

_MOD = "tests.scripts._toy_pipeline"
_VOLATILE = (".ledger.json", ".failed.json", "_profile.json")


def _cli(store, *args, check=True):
    cmd = [sys.executable, "-m", _MOD, "--store", str(store), "--allow-dirty", *args]
    p = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 -- fixed argv
    if check and p.returncode != 0:
        raise AssertionError(f"{cmd} rc={p.returncode}\n{p.stdout}\n{p.stderr}")
    return p


def _plan(store):
    return json.loads(_cli(store, "--plan").stdout)


def _outputs(store) -> dict[str, str]:
    root = pathlib.Path(store)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(_VOLATILE)
    }


def _fanned(store, stages=("gen", "folds", "final")):
    for st in stages:
        flag = "--inner-workers" if st == "folds" else "--workers"
        procs = [
            subprocess.Popen(  # noqa: S603 -- two concurrent "jobs"
                [sys.executable, "-m", _MOD, "--store", str(store), "--allow-dirty", "--stages", st, "--shard", f"{i}/2", flag, "2"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for i in range(2)
        ]
        for p in procs:
            out, err = p.communicate()
            assert p.returncode == 0, err
    # the reduce of the last stage ran in whichever job saw it complete; make sure it exists
    _cli(store, "--stages", stages[-1])


def _ledger(store, stage):
    gen = next(s["generation"] for s in _plan(store)["stages"] if s["name"] == stage)
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / gen).glob("*.ledger.json")]
    return sorted(recs, key=lambda r: r["finished_ns"])


def _seq(values: dict, order) -> float:
    """Sequential float sum in ``order`` (the toy reduce's arithmetic, order-sensitive by design)."""
    total = 0.0
    for k in order:
        total += values[k]
    return total


def test_w1n1_and_w2n2_are_byte_identical_and_non_vacuous(tmp_path):
    serial, fanned = tmp_path / "serial", tmp_path / "fanned"
    _cli(serial, "--run-all", "--workers", "1", "--out", str(tmp_path / "out1"))
    _fanned(fanned)
    a, b = _outputs(serial), _outputs(fanned)
    keep = {k: v for k, v in a.items() if "/env/" not in k and "/strict/" not in k}
    assert keep == {k: v for k, v in b.items() if k in keep}
    # non-vacuity 1: in each fanned job the gen units completed out of key order
    gen = _ledger(fanned, "gen")
    by_job: dict[str, list[str]] = {}
    for r in gen:
        by_job.setdefault(r["job_label"] or r["worker"], []).append(r["key"])
    assert any(keys != sorted(keys) for keys in by_job.values()) or [r["key"] for r in gen] != sorted(r["key"] for r in gen)
    # non-vacuity 2: the inner pool completed out of inner-key order on at least one fold
    assert any(r["inner_completion"] != sorted(r["inner_completion"]) for r in _ledger(fanned, "folds"))
    # non-vacuity 3 (on THIS run's data): for every fold the stored score equals the sorted-key
    # sequential sum of its inner values, and for at least one fold whose pool completed out of order,
    # re-summing in COMPLETION order gives a different float (1e16 + 1 loses the 1). A completion-order
    # inner reduce would therefore have changed the bytes the identity assertion above compares.
    from scripts._stages import LocalStore, read_result
    from tests.scripts._toy_pipeline import _inner_eval

    gens = {s["name"]: s["generation"] for s in _plan(fanned)["stages"]}
    store = LocalStore(fanned)
    v = tuple(read_result(store, f"{gens['gen']}/_output")["v"].tolist())
    differs = False
    for r in _ledger(fanned, "folds"):
        inner = {j: _inner_eval(int(r["key"]), int(j), v) for j in r["inner_completion"]}
        stored = read_result(store, f"{gens['folds']}/{r['key']}")["score"]
        assert stored == _seq(inner, sorted(inner)), r["key"]
        differs |= _seq(inner, r["inner_completion"]) != _seq(inner, sorted(inner))
    assert differs


def test_workers_run_with_the_deterministic_environment(tmp_path):
    _cli(tmp_path, "--stages", "env", "--workers", "1")
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "env")
    env = json.loads((tmp_path / gen / "only.json").read_text())
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1" and env["PYTHONHASHSEED"] == "0" and env["TZ"] == "UTC"


def test_a_killed_worker_is_relaunched_and_no_finished_unit_is_redone(tmp_path):
    marker = tmp_path / "crashed.marker"
    p = _cli(tmp_path / "s", "--stages", "gen", "--workers", "2", "--crash-once", str(marker))
    assert marker.exists() and "relaunching worker" in p.stderr
    recs = _ledger(tmp_path / "s", "gen")
    assert sorted(r["key"] for r in recs) == [f"{i:04d}" for i in range(12)]
    assert all(r["attempt"] == 1 for r in recs)  # a finished unit is never redone
    clean = tmp_path / "clean"
    _cli(clean, "--stages", "gen", "--workers", "1")
    _cli(tmp_path / "s", "--stages", "gen")  # reduce
    _cli(clean, "--stages", "gen")
    pick = lambda d: {k: v for k, v in _outputs(d).items() if "/gen/" in k}  # noqa: E731
    assert pick(tmp_path / "s") == pick(clean)


def test_resume_runs_only_pending_units(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--shard", "0/2", "--workers", "2")
    before = {r["key"]: r["finished_ns"] for r in _ledger(tmp_path, "gen")}
    _cli(tmp_path, "--stages", "gen", "--workers", "2")
    after = {r["key"]: r["finished_ns"] for r in _ledger(tmp_path, "gen")}
    assert all(after[k] == v for k, v in before.items())
    assert len(after) == 12


def test_a_poisoned_unit_is_redone_on_resume(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--workers", "1")
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "gen")
    shard = tmp_path / gen / "toy__0005.parquet"
    shard.write_bytes(shard.read_bytes()[:-5])
    _cli(tmp_path, "--stages", "gen", "--workers", "1")
    assert {r["key"]: r["attempt"] for r in _ledger(tmp_path, "gen")}["toy__0005"] == 2


def test_zero_tolerance_stage_aborts_on_a_failure(tmp_path):
    p = _cli(tmp_path, "--stages", "strict", "--workers", "1", "--fail-keys", "0002", check=False)
    assert p.returncode == 3 and "STAGE FAILED" in p.stderr
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "strict")
    assert (tmp_path / gen / "0002.failed.json").is_file()


def test_first_k_all_failed_aborts_early(tmp_path):
    p = _cli(tmp_path, "--stages", "sloppy", "--workers", "1", "--fail-keys", "*", check=False)
    assert p.returncode == 3 and "first 3" in p.stderr
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "sloppy")
    assert len(list((tmp_path / gen).glob("*.failed.json"))) < 6  # it stopped early


def test_plan_json_reports_progress(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--shard", "0/2", "--workers", "1")
    gen = next(s for s in _plan(tmp_path)["stages"] if s["name"] == "gen")
    assert (gen["n_units"], gen["done"], gen["pending"]) == (12, 6, 6)


@pytest.mark.parametrize("n_items", [12])
def test_tolerant_stage_reduces_over_successes(tmp_path, n_items):
    _cli(tmp_path, "--stages", "gen", "--workers", "1", "--fail-keys", "toy__0001")
    s = json.loads((tmp_path / next(x["generation"] for x in _plan(tmp_path)["stages"] if x["name"] == "gen") / "_summary.json").read_text())
    assert (s["n_done"], s["n_failed"]) == (n_items - 1, 1)
```

(The `fail_keys` value for `gen` is the JOINED key `toy__0001`; for `strict`/`sloppy` it is the bare key,
because their units are `int_key` strings.)

- [ ] **Step 3: Run and confirm the failures**

Run: `python -m pytest tests/scripts/hf_jobs/test_inner_pool.py tests/scripts/hf_jobs/test_executor.py -v -p no:randomly`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.hf_jobs.executor'`.

- [ ] **Step 4: Implement `scripts/hf_jobs/executor.py`**

```python
"""Place one stage's units as W memory-capped worker processes (ADR-128, spec 5.2).

Reuses the DGX launcher model (2026-09-29 spec) rather than a `ProcessPoolExecutor`: a pool worker killed
by the OOM killer breaks the whole pool (`BrokenProcessPool`), while a capped SUBPROCESS dies alone and
its relaunch resumes from the persisted units (INV-3). Workers are capped by `_mem_cap.wrap` (rlimit
inside an HF container), started with `_thread_pin.deterministic_env` (INV-2), sized by
`_parallel_launch.size_workers` and gated by `should_launch` backpressure.

Workers take keys from a JOB-LOCAL claim queue (O_EXCL claim files on local disk, never on the bucket
mount), so one slow unit never stalls a static partition. Across jobs the partition is static
(`_driver.shard_select`), so two jobs never run one unit.

A laptop, the DGX and an HF job run this same code (ruling 11): an HF job is only
`python -m <module> --stages S --shard I/N --workers W` inside the pinned image.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

GiB = 1024**3
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
#: Fallback RAM budget where neither /proc/meminfo nor psutil exists (a Windows dev box): the cpu-upgrade size.
_FALLBACK_RAM = 32 * GiB


class StageFailedError(RuntimeError):
    """A stage exceeded its failure tolerance, or a worker kept dying."""


@dataclasses.dataclass(frozen=True)
class ExecutorConfig:
    workers: int = 1
    inner_workers: int = 1
    cap_bytes: int | None = None
    peak_rss_bytes: int | None = None
    headroom_bytes: int = 2 * GiB
    max_relaunch: int = 3
    poll_interval_s: float = 0.2
    profile_units: int | None = None
    mem_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class StageReport:
    stage: str
    n_mine: int
    n_done: int
    n_excluded: int
    n_failed: int
    workers: int
    relaunched: int
    completion_order: tuple[str, ...]


def _inner_init(cap_bytes: int | None, backend: str) -> None:  # pragma: no cover - runs in pool children
    if backend == "rlimit" and cap_bytes:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (int(cap_bytes), int(cap_bytes)))  # type: ignore[attr-defined]


class InnerPool:
    """The pool an ``inner_parallel`` unit maps its inner keys over (spec 5.2: no pool inside a pool).

    Spawn context; children inherit the worker's deterministic environment, so each is single-threaded.
    ``map_keyed(fn, {inner_key: args})`` returns ``{inner_key: value}`` and records completion order;
    callers MUST combine with `_stages.ordered_values` (sorted inner keys), never completion order.
    ``fn`` must be a module-level function (pickled by reference).
    """

    def __init__(self, workers: int, *, cap_bytes: int | None = None, backend: str = "none") -> None:
        self.workers = max(1, int(workers))
        self._cap, self._backend = cap_bytes, backend
        self._ex: concurrent.futures.ProcessPoolExecutor | None = None
        self._log: list[str] = []

    def _executor(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._ex is None:
            self._ex = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_inner_init,
                initargs=(self._cap, self._backend),
            )
        return self._ex

    def map_keyed(self, fn: Callable[..., Any], items: Mapping[str, tuple]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.workers == 1:
            for k in sorted(items):
                out[k] = fn(*items[k])
                self._log.append(k)
            return out
        ex = self._executor()
        futures = {ex.submit(fn, *args): k for k, args in items.items()}
        for fut in concurrent.futures.as_completed(futures):
            k = futures[fut]
            out[k] = fut.result()
            self._log.append(k)
        return out

    def drain_log(self) -> list[str]:
        log, self._log = self._log, []
        return log

    def shutdown(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None


def _claim(claims: pathlib.Path, key: str, wid: str) -> bool:
    try:
        fd = os.open(claims / f"{key}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        fh.write(wid)
    return True


def _claims_of(claims: pathlib.Path, wid: str) -> list[str]:
    out = []
    for p in claims.glob("*.claim"):
        try:
            if p.read_text(encoding="ascii") == wid:
                out.append(p.name[: -len(".claim")])
        except OSError:
            continue
    return out


def _read_outcomes(outdir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for p in outdir.glob("*.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["finished_ns"])


def _terminate(procs: dict[str, subprocess.Popen]) -> None:
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    for p in procs.values():
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


class LocalExecutor:
    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg

    def _ram(self) -> int:
        from scripts._parallel_launch import available_ram_bytes

        try:
            return available_ram_bytes()
        except RuntimeError:
            return _FALLBACK_RAM

    # ---- reduce ---------------------------------------------------------------------------------
    def try_reduce(self, ctx, name: str) -> bool:
        """Reduce ``name`` if every unit is done/excluded (failures within tolerance). Idempotent: two
        jobs that both see a complete stage write identical bytes atomically. Returns presence."""
        from scripts._stages import output_present, sorted_units, unit_states

        if output_present(ctx, name, verify=True):
            return True
        st = ctx.pipeline.stage(name)
        states = unit_states(ctx, st, sorted_units(ctx, st), verify=False)
        if any(v == "pending" for v in states.values()):
            return False
        self._subprocess(ctx, ["reduce", "--stage", name])
        return output_present(ctx, name, verify=True)

    def require_reduced(self, ctx, name: str) -> None:
        from scripts._stages import StageIncompleteError

        if not self.try_reduce(ctx, name):
            raise StageIncompleteError(f"{ctx.pipeline.name}: upstream stage {name!r} is incomplete")

    def _subprocess(self, ctx, args: list[str]) -> None:
        from scripts._stages import context_to_json
        from scripts._thread_pin import deterministic_env

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="sk-reduce-"))
        try:
            (tmp / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
            cmd = [sys.executable, "-m", "scripts.hf_jobs.executor", *args, "--ctx", str(tmp / "ctx.json")]
            rc = subprocess.run(cmd, env=deterministic_env(), cwd=REPO_ROOT, check=False).returncode  # noqa: S603
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if rc != 0:
            raise StageFailedError(f"{ctx.pipeline.name}: '{' '.join(args)}' failed (rc={rc})")

    # ---- units ----------------------------------------------------------------------------------
    def run_stage(self, ctx, name: str, *, shard: tuple[int, int] = (0, 1)) -> StageReport:
        from scripts._driver import shard_select
        from scripts._stages import sorted_units

        st = ctx.pipeline.stage(name)
        for need in st.needs:
            self.require_reduced(ctx, need)
        mine = shard_select(sorted_units(ctx, st), shard[0], shard[1])
        if self.cfg.profile_units:
            return self._profile(ctx, st, mine)
        return self._run(ctx, st, mine, workers=self.cfg.workers, inner=self.cfg.inner_workers)

    def _run(self, ctx, st, keys: list[str], *, workers: int, inner: int) -> StageReport:
        from scripts._parallel_launch import size_workers
        from scripts._stages import unit_states

        states = unit_states(ctx, st, keys, verify=True)  # INV-3: a resume re-hashes before it skips
        pending = [k for k in keys if states[k] in ("pending", "failed")]
        relaunched, order, w = 0, (), 0
        if pending:
            w = 1 if st.inner_parallel else max(1, min(workers, len(pending)))
            if self.cfg.peak_rss_bytes and not st.inner_parallel:
                w = min(w, size_workers(self.cfg.peak_rss_bytes, self._ram(), os.cpu_count() or 1, self.cfg.headroom_bytes))
            relaunched, order = self._dispatch(ctx, st, pending, w, inner if st.inner_parallel else 1)
        final = unit_states(ctx, st, keys, verify=False)
        n_failed = sum(v == "failed" for v in final.values())
        report = StageReport(
            st.name,
            len(keys),
            sum(v == "done" for v in final.values()),
            sum(v == "excluded" for v in final.values()),
            n_failed,
            w,
            relaunched,
            order,
        )
        if st.max_failures is not None and n_failed > st.max_failures:
            raise StageFailedError(
                f"{ctx.pipeline.name}/{st.name}: {n_failed} failed unit(s) > tolerance {st.max_failures}; "
                f"see *.failed.json in {ctx.generation(st.name)}"
            )
        return report

    def _check_failure_rule(self, st, outdir: pathlib.Path, procs) -> None:
        rows = _read_outcomes(outdir)
        failed = [r for r in rows if r["outcome"] == "failed"]
        if st.max_failures is not None and len(failed) > st.max_failures:
            _terminate(procs)
            raise StageFailedError(f"{st.name}: {len(failed)} failed unit(s) > tolerance {st.max_failures} (e.g. {failed[0]['key']})")
        first = rows[: st.abort_first_k]
        if len(first) == st.abort_first_k and all(r["outcome"] == "failed" for r in first):
            _terminate(procs)
            raise StageFailedError(f"{st.name}: the first {st.abort_first_k} completed units all failed (systematic)")

    def _dispatch(self, ctx, st, pending: list[str], workers: int, inner: int) -> tuple[int, tuple[str, ...]]:
        from scripts._mem_cap import detect_backend, preflight, wrap
        from scripts._parallel_launch import should_launch
        from scripts._stages import context_to_json, unit_states
        from scripts._thread_pin import deterministic_env

        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"sk-{st.name}-"))
        procs: dict[str, subprocess.Popen] = {}
        try:
            (tmp / "ctx.json").write_text(context_to_json(ctx), encoding="utf-8")
            (tmp / "pending.json").write_text(json.dumps(pending), encoding="utf-8")
            claims, outdir = tmp / "claims", tmp / "outcomes"
            claims.mkdir()
            outdir.mkdir()
            backend = detect_backend(self.cfg.mem_backend)  # type: ignore[arg-type]
            cap = self.cfg.cap_bytes
            if cap:
                preflight(backend, cap_bytes=cap)
            env = deterministic_env()

            def launch(wid: str) -> None:
                argv = [
                    sys.executable, "-m", "scripts.hf_jobs.executor", "worker",
                    "--ctx", str(tmp / "ctx.json"), "--stage", st.name, "--pending", str(tmp / "pending.json"),
                    "--claims", str(claims), "--outcomes", str(outdir), "--worker-id", wid,
                    "--inner-workers", str(inner), "--mem-backend", backend,
                ]
                if cap:
                    argv += ["--cap-bytes", str(cap)]
                pre = None
                if cap:
                    argv, pre = wrap(backend, argv, cap_bytes=cap)
                procs[wid] = subprocess.Popen(argv, env=env, cwd=REPO_ROOT, preexec_fn=pre)  # noqa: S603 -- own argv

            slots = [f"w{i}" for i in range(workers)]
            launch(slots.pop(0))  # the first worker always starts: sizing already proved one fits
            attempts: dict[str, int] = {}
            relaunched = 0
            while procs or slots:
                time.sleep(self.cfg.poll_interval_s)
                while slots and (not self.cfg.peak_rss_bytes or should_launch(self._ram(), self.cfg.peak_rss_bytes, self.cfg.peak_rss_bytes)):
                    launch(slots.pop(0))
                self._check_failure_rule(st, outdir, procs)
                for wid, p in list(procs.items()):
                    rc = p.poll()
                    if rc is None:
                        continue
                    del procs[wid]
                    if rc == 0:
                        continue
                    held = _claims_of(claims, wid)
                    for k, state in unit_states(ctx, st, held, verify=False).items():
                        if state == "pending":
                            (claims / f"{k}.claim").unlink(missing_ok=True)  # the dead worker's unit is free again
                    left = [k for k, v in unit_states(ctx, st, pending, verify=False).items() if v == "pending" and not (claims / f"{k}.claim").exists()]
                    if not left:
                        continue
                    attempts[wid] = attempts.get(wid, 0) + 1
                    if attempts[wid] > self.cfg.max_relaunch:
                        _terminate(procs)
                        raise StageFailedError(f"{st.name}: worker {wid} died {attempts[wid]} times (last rc={rc})")
                    print(f"relaunching worker {wid} (rc={rc}); {len(left)} unit(s) left", file=sys.stderr, flush=True)
                    relaunched += 1
                    launch(wid)
                if not procs and slots:
                    # Backpressure must never deadlock: with nothing running, waiting frees no RAM, and
                    # sizing already proved ONE worker fits. Launch one if unclaimed work is left.
                    states = unit_states(ctx, st, pending, verify=False)
                    if any(v == "pending" and not (claims / f"{k}.claim").exists() for k, v in states.items()):
                        launch(slots.pop(0))
                    else:
                        slots.clear()
            self._check_failure_rule(st, outdir, {})
            order = tuple(r["key"] for r in _read_outcomes(outdir))
            return relaunched, order
        finally:
            _terminate(procs)
            shutil.rmtree(tmp, ignore_errors=True)

    # ---- profile (spec 5.5) -----------------------------------------------------------------------
    def _peak_of(self, ctx, st, keys: list[str]) -> int | None:
        peaks = []
        for k in keys:
            raw = ctx.store.read_bytes(f"{ctx.generation(st.name)}/{k}.ledger.json")
            if raw:
                v = json.loads(raw).get("peak_rss_bytes")
                if v:
                    peaks.append(int(v))
        return max(peaks) if peaks else None

    def _profile(self, ctx, st, mine: list[str]) -> StageReport:
        """Run K evenly spaced units: two at W=1 (inner W=1), the rest at the RAM-bounded maximum W
        (inner W for an inner_parallel stage). Writes ``_profile.json``. The units are real: their
        results persist and the campaign never redoes them."""
        from scripts._parallel_launch import size_workers
        from scripts._stages import _canon_json

        k = max(1, min(int(self.cfg.profile_units or 1), len(mine)))
        sample = sorted({mine[(i * len(mine)) // k] for i in range(k)})
        first, rest = sample[:2], sample[2:]
        t0 = time.monotonic()
        self._run(ctx, st, first, workers=1, inner=1)
        wall_1 = time.monotonic() - t0
        peak = self._peak_of(ctx, st, first) or self.cfg.peak_rss_bytes
        if st.inner_parallel:
            wmax, inner = 1, max(1, self.cfg.inner_workers)
        else:
            inner = 1
            wmax = self.cfg.workers
            if peak:
                wmax = min(wmax, size_workers(int(peak * 1.3), self._ram(), os.cpu_count() or 1, self.cfg.headroom_bytes))
        t1 = time.monotonic()
        report = self._run(ctx, st, rest, workers=wmax, inner=inner) if rest else self._run(ctx, st, [], workers=1, inner=1)
        wall_max = time.monotonic() - t1
        par = inner if st.inner_parallel else wmax
        rate_1 = len(first) / wall_1 if wall_1 > 0 else None
        rate_max = len(rest) / wall_max if (rest and wall_max > 0) else None
        record = {
            "stage": st.name,
            "sample": sample,
            "w1": {"units": first, "wall_s": wall_1},
            "wmax": {"workers": wmax, "inner_workers": inner, "units": rest, "wall_s": wall_max},
            "peak_rss_bytes": peak,
            "efficiency": (rate_max / (par * rate_1)) if (rate_max and rate_1) else None,
            "nproc": os.cpu_count(),
        }
        ctx.store.write_bytes(f"{ctx.generation(st.name)}/_profile.json", _canon_json(record))
        return report


def worker_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="executor worker")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--pending", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--inner-workers", type=int, default=1)
    ap.add_argument("--mem-backend", default="none")
    ap.add_argument("--cap-bytes", type=int, default=None)
    a = ap.parse_args(argv)
    from scripts._stages import ItemStage, context_from_json, run_unit

    ctx = context_from_json(pathlib.Path(a.ctx).read_text(encoding="utf-8"))
    st = ctx.pipeline.stage(a.stage)
    pending = json.loads(pathlib.Path(a.pending).read_text(encoding="utf-8"))
    refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
    pool = InnerPool(a.inner_workers, cap_bytes=a.cap_bytes, backend=a.mem_backend) if st.inner_parallel else None
    log = pathlib.Path(a.outcomes) / f"{a.worker_id}.jsonl"
    claims = pathlib.Path(a.claims)
    try:
        for key in pending:
            if not _claim(claims, key, a.worker_id):
                continue
            outcome = run_unit(ctx, st, key, worker=a.worker_id, pool=pool, ref=refs.get(key))
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "outcome": outcome, "finished_ns": time.time_ns()}) + "\n")
    finally:
        if pool is not None:
            pool.shutdown()
    return 0


def reduce_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="executor reduce")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    from scripts._stages import StageIncompleteError, context_from_json, reduce_stage

    ctx = context_from_json(pathlib.Path(a.ctx).read_text(encoding="utf-8"))
    try:
        reduce_stage(ctx, a.stage)
    except StageIncompleteError as exc:
        print(f"STAGE FAILED: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    _CMDS = {"worker": worker_main, "reduce": reduce_main}
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        raise SystemExit(f"usage: python -m scripts.hf_jobs.executor {{{','.join(_CMDS)}}} ...")
    raise SystemExit(_CMDS[sys.argv[1]](sys.argv[2:]))
```

Implementation notes the code relies on:
- `SILLY_KICKS_JOB_LABEL` is set by the launcher (Task 6). The test harness leaves it unset, so
  `test_w1n1_and_w2n2…` groups ledger records by `worker` when it is absent; the two fanned "jobs" share
  worker ids, which is why the assertion falls back to the global order.
- `_run` re-hashes every key of the slice once (`verify=True`), before dispatch. Workers trust that list
  and the claim queue; they never list the store per unit (a 3961-unit stage on a bucket mount would pay
  O(n²) listings otherwise).

- [ ] **Step 5: Run the gates and measure them**

Run: `python -m pytest tests/scripts/hf_jobs/test_inner_pool.py tests/scripts/hf_jobs/test_executor.py tests/scripts/test_stage_contract.py -v -p no:randomly --durations=0`
Expected: PASS. Record each test's duration from `--durations=0`. Mark every test that takes more than 10 s
with `@pytest.mark.slow` (house rule ADR-023) and note its measured cost in a one-line comment
(`# slow: measured 23 s on <machine> (ADR-023)`). Re-run the file to confirm.

- [ ] **Step 6: Run on the pinned interpreter too**

Run: `.venv-hfjobs/Scripts/python.exe -m pytest tests/scripts/hf_jobs/test_executor.py tests/scripts/hf_jobs/test_inner_pool.py -q -p no:randomly`
Expected: PASS.

- [ ] **Step 7: Lint and type-check**

Run: `uvx ruff@0.15.7 check scripts/hf_jobs/executor.py tests/scripts/_toy_pipeline.py tests/scripts/hf_jobs/test_executor.py tests/scripts/hf_jobs/test_inner_pool.py && pyright scripts/hf_jobs/executor.py scripts/_stages.py`
Expected: clean.

---
## Task 5: Ledger and planner (pure)

Spec: §5.5 (ledger, planner, profile), decision P14, R4 (measured scaling), R5 (concurrency cap).

**Files:**
- Create: `scripts/hf_jobs/planner.py`, `scripts/hf_jobs/ledger.py`
- Test: `tests/scripts/hf_jobs/test_planner.py`, `tests/scripts/hf_jobs/test_ledger.py`

**Interfaces:**
- Produces (planner): `Constraints(vcpu=8, usable_bytes=30 GiB, headroom_bytes=2 GiB, target_s=3600.0, startup_s=300.0, concurrency_cap=10, usd_per_hour=0.03, timeout_safety=1.5, timeout_floor_s=1800.0, cap_factor=1.3)`;
  `UnitCost(p50_s, p95_s, p99_s, peak_rss_bytes, n_samples)`;
  `StagePlan(stage, n_units, n_jobs, workers, inner_workers, partition, predicted_job_wall_s, predicted_cost_usd, timeout_s, cap_bytes, peak_rss_bytes, irreducible_floor_s, basis)`;
  `plan_stage(stage, keys, cost, *, inner_parallel, n_inner=1, efficiency=None, per_unit=None, c=Constraints()) -> StagePlan`;
  `lpt_partition(costs: Mapping[str, float], n_jobs: int) -> list[list[str]]`.
- Produces (ledger): `percentile(values, q) -> float`; `unit_records(store, generation) -> list[dict]`;
  `unit_cost(records) -> UnitCost | None`; `history_key(token_inputs: Mapping) -> str`;
  `job_record(info) -> dict`; `summarize(*, campaign, campaign_sha256, prov, stages, jobs, artifacts) -> dict`;
  `write_summary(path, summary) -> None`; `load_history(paths) -> dict[str, dict]`.

- [ ] **Step 1: Write the failing tests**

`tests/scripts/hf_jobs/test_planner.py`:

```python
from __future__ import annotations

import pytest

from scripts._parallel_launch import RefusalError
from scripts.hf_jobs.planner import Constraints, UnitCost, lpt_partition, plan_stage

GiB = 1024**3
KEYS = [f"m{i:03d}" for i in range(179)]


def test_workers_are_ram_bounded_not_vcpu_bounded():
    p = plan_stage("extract", KEYS, UnitCost(30.0, 40.0, 50.0, 6 * GiB, 20), inner_parallel=False)
    assert p.workers == 3  # (30 - 2) // (6 * 1.3) == 3, below 8 vCPU
    assert p.cap_bytes == int(6 * GiB * 1.3)


def test_n_jobs_meets_the_target_with_startup():
    c = Constraints(target_s=3600, startup_s=300)
    p = plan_stage("extract", KEYS, UnitCost(60.0, 70.0, 80.0, 1 * GiB, 20), inner_parallel=False, c=c)
    # 179 units * 60 s / (8 workers * 3300 s) -> 1 job
    assert p.n_jobs == 1 and p.workers == 8
    slow = plan_stage("extract", KEYS, UnitCost(600.0, 700.0, 800.0, 1 * GiB, 20), inner_parallel=False, c=c)
    assert slow.n_jobs == 5  # ceil(179*600 / (8*3300)) = ceil(4.07)
    assert slow.predicted_job_wall_s <= 3600 + 800


def test_measured_efficiency_reduces_effective_parallelism():
    full = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False)
    poor = plan_stage("fit", KEYS, UnitCost(600, 600, 600, 1 * GiB, 5), inner_parallel=False, efficiency=0.5)
    assert poor.n_jobs > full.n_jobs


def test_concurrency_cap_clips_n_and_the_plan_says_so():
    p = plan_stage("x", KEYS, UnitCost(3000, 3000, 3000, 1 * GiB, 5), inner_parallel=False, c=Constraints(concurrency_cap=10))
    assert p.n_jobs == 10 and p.predicted_job_wall_s > 3600


def test_a_unit_longer_than_the_target_is_the_irreducible_floor():
    p = plan_stage("fit", ["only"], UnitCost(5000, 5400, 5600, 9 * GiB, 1), inner_parallel=False)
    assert p.irreducible_floor_s == 5400 and p.n_jobs == 1


def test_inner_parallel_uses_one_top_level_worker_and_bounded_inner_workers():
    p = plan_stage("study", ["single"], UnitCost(2400, 2600, 2700, 3 * GiB, 1), inner_parallel=True, n_inner=5)
    assert (p.workers, p.inner_workers, p.n_jobs) == (1, 5, 1)


def test_inner_parallel_fans_units_across_jobs():
    tags = [f"a_f{i}" for i in range(15)]
    p = plan_stage("study", tags, UnitCost(1200, 1300, 1400, 2 * GiB, 3), inner_parallel=True, n_inner=5)
    assert p.n_jobs == 6  # ceil(15*1200 / 3300)


def test_timeout_is_derived_with_a_floor():
    p = plan_stage("x", KEYS[:4], UnitCost(10, 10, 10, 1 * GiB, 4), inner_parallel=False)
    assert p.timeout_s == 1800
    big = plan_stage("x", KEYS, UnitCost(1000, 1200, 1500, 1 * GiB, 9), inner_parallel=False)
    assert big.timeout_s >= int(1.5 * big.predicted_job_wall_s)


def test_a_unit_that_cannot_fit_refuses():
    with pytest.raises(RefusalError):
        plan_stage("x", KEYS, UnitCost(1, 1, 1, 40 * GiB, 1), inner_parallel=False)


def test_unmeasured_stage_plans_nothing():
    p = plan_stage("x", KEYS, None, inner_parallel=False)
    assert p.basis == "unmeasured" and p.n_jobs == 0


def test_lpt_partition_is_balanced_disjoint_and_deterministic():
    costs = {f"k{i}": float(c) for i, c in enumerate([9, 7, 6, 5, 5, 4, 3, 2])}
    a = lpt_partition(costs, 3)
    assert a == lpt_partition(dict(reversed(costs.items())), 3)
    assert sorted(k for part in a for k in part) == sorted(costs)
    loads = sorted(sum(costs[k] for k in part) for part in a)
    assert loads[-1] - loads[0] <= 4
```

`tests/scripts/hf_jobs/test_ledger.py`:

```python
from __future__ import annotations

import json

from scripts._stages import LocalStore
from scripts.hf_jobs.ledger import history_key, percentile, summarize, unit_cost, unit_records


def _rec(store, gen, key, wall, peak, outcome="done"):
    store.write_bytes(f"{gen}/{key}.ledger.json", json.dumps({"key": key, "outcome": outcome, "wall_s": wall, "peak_rss_bytes": peak, "attempt": 1}).encode())


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.5) == 2 and percentile([1, 2, 3, 4], 0.95) == 4


def test_unit_cost_from_done_records_only(tmp_path):
    store = LocalStore(tmp_path)
    for i, w in enumerate([10, 20, 30, 40]):
        _rec(store, "p/s/t", f"k{i}", w, 100 + i)
    _rec(store, "p/s/t", "bad", 999, 10**12, outcome="failed")
    c = unit_cost(unit_records(store, "p/s/t"))
    assert (c.p50_s, c.p95_s, c.peak_rss_bytes, c.n_samples) == (20, 40, 103, 4)


def test_history_key_ignores_code_and_upstream():
    base = {"pipeline": "g", "stage": "cv", "params": {"x": 1}, "dataset": {"name": "tc3", "revision": "r"}}
    a = history_key({**base, "code": "c1", "upstream": {"cache": "t1"}})
    b = history_key({**base, "code": "c2", "upstream": {"cache": "t2"}})
    assert a == b != history_key({**base, "params": {"x": 2}, "code": "c1", "upstream": {}})


def test_summary_carries_top_level_provenance():
    s = summarize(campaign="cycle1", campaign_sha256="ab", prov={"commit": "c" * 40, "dirty": False}, stages=[], jobs=[], artifacts=[])
    assert s["run_commit"] == "c" * 40 and s["run_tree_dirty"] is False
```

- [ ] **Step 2: Run and confirm the failures**

Run: `python -m pytest tests/scripts/hf_jobs/test_planner.py tests/scripts/hf_jobs/test_ledger.py -v -p no:randomly`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `scripts/hf_jobs/planner.py`**

```python
"""Pure planning (spec 5.5): N jobs x W workers per stage from measured unit costs, a 60-minute target,
the cpu-upgrade box (8 vCPU, ~30 GB usable) and the account concurrency cap. Outputs never depend on
the plan (INV-1), so the planner may partition freely (LPT when per-unit costs are known)."""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections.abc import Mapping, Sequence

GiB = 1024**3


@dataclasses.dataclass(frozen=True)
class Constraints:
    vcpu: int = 8
    usable_bytes: int = 30 * GiB
    headroom_bytes: int = 2 * GiB
    target_s: float = 3600.0
    startup_s: float = 300.0  # replaced by the ledger's measured median job startup
    concurrency_cap: int = 10
    usd_per_hour: float = 0.03  # replaced by list_jobs_hardware() at preflight
    timeout_safety: float = 1.5
    timeout_floor_s: float = 1800.0
    cap_factor: float = 1.3


@dataclasses.dataclass(frozen=True)
class UnitCost:
    p50_s: float
    p95_s: float
    p99_s: float
    peak_rss_bytes: int
    n_samples: int


@dataclasses.dataclass(frozen=True)
class StagePlan:
    stage: str
    n_units: int
    n_jobs: int
    workers: int
    inner_workers: int
    partition: tuple[tuple[str, ...], ...] | None  # None: round-robin `--shard I/N`
    predicted_job_wall_s: float
    predicted_cost_usd: float
    timeout_s: int
    cap_bytes: int | None
    peak_rss_bytes: int | None
    irreducible_floor_s: float | None
    basis: str  # "measured" | "unmeasured"


def lpt_partition(costs: Mapping[str, float], n_jobs: int) -> list[list[str]]:
    """Longest-processing-time greedy: biggest unit first onto the least-loaded job. Ties break by key, so
    the partition is a pure function of the costs (input order is irrelevant)."""
    n_jobs = max(1, n_jobs)
    heap = [(0.0, j) for j in range(n_jobs)]
    parts: list[list[str]] = [[] for _ in range(n_jobs)]
    for key in sorted(costs, key=lambda k: (-costs[k], k)):
        load, j = heapq.heappop(heap)
        parts[j].append(key)
        heapq.heappush(heap, (load + costs[key], j))
    return [sorted(p) for p in parts if p]


def _workers(peak: int, c: Constraints) -> int:
    from scripts._parallel_launch import size_workers  # RefusalError when one unit cannot fit

    return size_workers(int(peak * c.cap_factor), c.usable_bytes + c.headroom_bytes, c.vcpu, c.headroom_bytes)


def plan_stage(
    stage: str,
    keys: Sequence[str],
    cost: UnitCost | None,
    *,
    inner_parallel: bool,
    n_inner: int = 1,
    efficiency: float | None = None,
    per_unit: Mapping[str, float] | None = None,
    c: Constraints = Constraints(),
) -> StagePlan:
    n = len(keys)
    if cost is None or n == 0:
        return StagePlan(stage, n, 0, 0, 0, None, 0.0, 0.0, 0, None, None, None, "unmeasured")
    eff = efficiency if efficiency is not None else 1.0
    budget = max(1.0, c.target_s - c.startup_s)
    cap = int(cost.peak_rss_bytes * c.cap_factor)
    if inner_parallel:
        inner = max(1, min(c.vcpu, n_inner, _workers(cost.peak_rss_bytes, c)))
        n_jobs = max(1, min(n, c.concurrency_cap, math.ceil(n * cost.p50_s / budget)))
        per_job = math.ceil(n / n_jobs)
        wall = c.startup_s + per_job * cost.p50_s
        wall_p99 = c.startup_s + per_job * cost.p99_s
        workers, partition = 1, None
    else:
        workers = max(1, _workers(cost.peak_rss_bytes, c))
        inner = 1
        par = workers * eff
        total = sum(per_unit.values()) if per_unit else cost.p50_s * n
        n_jobs = max(1, min(n, c.concurrency_cap, math.ceil(total / (par * budget))))
        if per_unit:
            parts = lpt_partition(per_unit, n_jobs)
            n_jobs = len(parts)
            partition = tuple(tuple(p) for p in parts)
            busiest = max(sum(per_unit[k] for k in p) for p in parts)
        else:
            partition = None
            busiest = math.ceil(n / n_jobs) * cost.p50_s
        wall = c.startup_s + max(busiest / par, cost.p95_s)
        wall_p99 = c.startup_s + max(math.ceil(n / n_jobs) * cost.p99_s / par, cost.p99_s)
    floor = cost.p95_s if cost.p95_s > budget else None
    timeout = int(max(c.timeout_floor_s, c.timeout_safety * wall_p99))
    usd = n_jobs * wall / 3600.0 * c.usd_per_hour
    return StagePlan(stage, n, n_jobs, workers, inner, partition, wall, usd, timeout, cap, cost.peak_rss_bytes, floor, "measured")
```

`plan_stage` returns a `partition` only when per-unit costs are known (a previous run of the same keys,
e.g. the Task 25 replica run). Task 8 uploads each part to the bucket as
`<run_key>/_partitions/<pipeline>/<stage>/<token>/<i>of<n>.json` and launches job i with
`--shard-keys /scratch/<run_key>/_partitions/…/<i>of<n>.json`. Add that option in THIS task:
`_stages.build_parser` gains `--shard-keys PATH` (a JSON list of keys; the job runs exactly these keys;
refused together with a non-default `--shard` and with more than one stage), `main` passes it to
`LocalExecutor.run_stage(ctx, name, keys=…)`, and `run_stage` gains `keys: list[str] | None = None`
(when given, `mine = sorted(set(keys) & set(sorted_units(ctx, st)))`, and a listed key that is not a
unit raises `ValueError`). Add two tests to `tests/scripts/test_stage_contract.py` /
`tests/scripts/hf_jobs/test_executor.py`: `--shard-keys` runs exactly the listed units; an unknown key is
refused. Outputs never depend on the partition (INV-1), so this is pure placement.

- [ ] **Step 4: Implement `scripts/hf_jobs/ledger.py`**

```python
"""The run ledger (spec 5.5). The full per-unit record lives in the store (`<key>.ledger.json`, written by
`_stages.run_unit`); job records come from the HF API (`JobInfo` via labels); a compact per-campaign
summary is committed as `docs/research/hf_campaigns/<campaign>.json` with top-level `run_commit` and
`run_tree_dirty` (it passes `test_artifact_provenance_output`). The planner learns from those summaries."""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Iterable, Mapping, Sequence

from scripts.hf_jobs.planner import UnitCost


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    s = sorted(values)
    return float(s[max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))])


def unit_records(store, generation: str) -> list[dict]:
    out = []
    for name in sorted(store.names(generation)):
        if name.endswith(".ledger.json"):
            raw = store.read_bytes(f"{generation}/{name}")
            if raw:
                out.append(json.loads(raw))
    return out


def unit_cost(records: Iterable[dict]) -> UnitCost | None:
    done = [r for r in records if r.get("outcome") == "done" and r.get("wall_s") is not None]
    if not done:
        return None
    walls = [float(r["wall_s"]) for r in done]
    peaks = [int(r["peak_rss_bytes"]) for r in done if r.get("peak_rss_bytes")]
    return UnitCost(percentile(walls, 0.5), percentile(walls, 0.95), percentile(walls, 0.99), max(peaks) if peaks else 0, len(done))


def history_key(token_inputs: Mapping) -> str:
    """Decision P14: costs are keyed by WHAT is computed, not by which commit computed it. Single-sourced
    in `_stages._history_key` (the plan JSON carries it per stage)."""
    from scripts._stages import _history_key

    return _history_key(token_inputs)


def _iso(dt) -> str | None:
    return None if dt is None else dt.isoformat()


def job_record(info) -> dict:
    started, finished = getattr(info, "started_at", None), getattr(info, "finished_at", None)
    wall = (finished - started).total_seconds() if (started and finished) else None
    status = getattr(info, "status", None)
    return {
        "id": info.id,
        "labels": dict(info.labels or {}),
        "stage": getattr(status, "stage", None),
        "message": getattr(status, "message", None),
        "created_at": _iso(getattr(info, "created_at", None)),
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "wall_s": wall,
    }


def summarize(*, campaign: str, campaign_sha256: str, prov: Mapping, stages: list[dict], jobs: list[dict], artifacts: list[dict]) -> dict:
    return {
        "run_commit": prov["commit"],
        "run_tree_dirty": bool(prov.get("dirty", True)),
        "campaign": campaign,
        "campaign_sha256": campaign_sha256,
        "env": prov.get("env"),
        "stages": stages,
        "jobs": jobs,
        "artifacts": artifacts,
    }


def write_summary(path, summary: Mapping) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_history(paths: Iterable[pathlib.Path]) -> dict[str, dict]:
    """``{history_key: stage summary}`` from committed campaign summaries; later files win."""
    hist: dict[str, dict] = {}
    for p in sorted(paths):
        for st in json.loads(pathlib.Path(p).read_text(encoding="utf-8")).get("stages", []):
            if st.get("history_key"):
                hist[st["history_key"]] = st
    return hist
```

- [ ] **Step 5: Run the tests, lint, type-check**

Run: `python -m pytest tests/scripts/hf_jobs/test_planner.py tests/scripts/hf_jobs/test_ledger.py tests/scripts/test_stage_contract.py -v -p no:randomly`
Expected: PASS. If an expected number in a planner test disagrees with the code, re-derive the arithmetic in
the test comment first; change the code only if the arithmetic shows the code is wrong.

Run: `uvx ruff@0.15.7 check scripts/hf_jobs/planner.py scripts/hf_jobs/ledger.py && pyright scripts/hf_jobs/planner.py scripts/hf_jobs/ledger.py`

---

## Task 6: Typed launcher, job command, `BucketStore`, labels, flavor constant

Spec: §5.4 (launcher on the typed API, environment), INV-4, INV-6, decisions P1, P6; spec §6.2 (the
launcher symbols this task replaces). The legacy functions stay in place until Task 18 deletes them
(Task 18 also rewrites their tests); this task ADDS the typed surface beside them.

**Files:**
- Modify: `scripts/hf_jobs/launch.py` (add the typed surface at the top; legacy code below it untouched until Task 18)
- Create: `scripts/hf_jobs/store.py`
- Test: `tests/scripts/hf_jobs/test_launch_typed.py`, `tests/scripts/hf_jobs/test_bucket_store.py`

**Interfaces:**
- Produces: `FLAVOR = "cpu-upgrade"`; `IMAGE`; `PYTHON_BIN = "/usr/local/bin/python3.12"`; `UV_REQUIREMENT`;
  `REPO_URL`; `BUCKET_MOUNT = "/scratch"`; `DATA_MOUNT = "/data"`;
  `DatasetMount(name, repo, revision, access)` (`access` ∈ `mount|download|open`) with `.path`;
  `JobSpec(campaign, run_key, job, variant, stages, shard, module, args, commit, bucket, workers, inner_workers, cap_bytes, peak_rss_bytes, timeout_s, attempts, datasets, extras, profile=None, shard_keys_path=None, extra_env=None)`;
  `labels_for(spec) -> dict[str, str]`; `job_env(spec) -> dict[str, str]`; `job_command(spec) -> list[str]`;
  `volumes_for(spec) -> list[Volume]`; `launch(spec, *, api, token) -> str`;
  `jobs_for(api, *, campaign, commit) -> list[JobInfo]`; `TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}`.
- Produces (store): `BucketStore(bucket, prefix, api=None)` with `from_uri(uri)`, `names`, `read_bytes`,
  `write_bytes` (raises `PermissionError`), `delete` (raises), `local_path` (returns None), and
  `upload_bytes(rel, data)` (controller records only: `_campaign/…`, `_partitions/…`).

- [ ] **Step 1: Write the failing tests**

`tests/scripts/hf_jobs/test_launch_typed.py`:

```python
"""Typed HF Jobs launcher (ADR-128): one image, one flavor, labels, pinned env, exact-commit checkout."""

from __future__ import annotations

import ast
import pathlib
import shlex

from scripts.hf_jobs import launch as L

SHA = "4" * 40


def _spec(**kw):
    base = dict(
        campaign="cycle1", run_key="cycle1-4444444-ghost_gk-owner", job="ghost_gk", variant="default",
        stages=("extract",), shard=(1, 4), module="scripts.train_ghost_gk",
        args=("--feature-set", "faithful", "--dataset", "tc3"), commit=SHA, bucket="silly-kicks/hf-refit-scratch",
        workers=3, inner_workers=1, cap_bytes=8 * 2**30, peak_rss_bytes=6 * 2**30, timeout_s=3600, attempts=3,
        datasets=(L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "mount"),), extras=(),
    )
    base.update(kw)
    return L.JobSpec(**base)


class _FakeApi:
    def __init__(self):
        self.calls = []

    def run_job(self, **kw):
        self.calls.append(kw)

        class _Info:
            id = "job123"

        return _Info()


def test_launch_uses_the_typed_api_with_the_one_flavor_and_image():
    api = _FakeApi()
    assert L.launch(_spec(), api=api, token="hf_x") == "job123"
    kw = api.calls[0]
    assert kw["flavor"] == "cpu-upgrade" == L.FLAVOR
    assert kw["image"] == L.IMAGE and "@sha256:" in L.IMAGE
    assert kw["secrets"] == {"HF_TOKEN": "hf_x"}
    assert kw["timeout"] == 3600 and kw["attempts"] == 3
    assert kw["labels"] == {
        "campaign": "cycle1", "run_key": "cycle1-4444444-ghost_gk-owner", "job": "ghost_gk",
        "variant": "default", "stage": "extract", "shard": "1of4", "commit": SHA,
    }


def test_no_other_flavor_is_reachable():
    """INV-4: the only flavor literal in the launcher package is the constant."""
    for path in pathlib.Path("scripts/hf_jobs").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        flavors = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("cpu-")
        }
        assert flavors <= {"cpu-upgrade"}, (path, flavors)


def test_job_env_is_deterministic_and_carries_identity():
    env = L.job_env(_spec())
    assert env["PYTHONHASHSEED"] == "0" and env["NUMBA_NUM_THREADS"] == "1"
    assert env["SILLY_KICKS_IMAGE"] == L.IMAGE
    assert env["SILLY_KICKS_RUN_KEY"] == "cycle1-4444444-ghost_gk-owner"
    assert env["SILLY_KICKS_JOB_LABEL"] == "ghost_gk.default.extract.1of4"
    assert env["SILLY_KICKS_IN_HF_JOB"] == "1"


def test_job_command_checks_out_the_exact_commit_and_syncs_frozen():
    cmd = L.job_command(_spec())
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert f"git fetch -q --depth 1 origin {SHA}" in script
    assert f'test "$(git rev-parse HEAD)" = "{SHA}"' in script
    assert "--require-hashes" in script and L.UV_REQUIREMENT in script
    assert "uv sync --frozen" in script and "--group hfjobs" in script and "--extra train" in script
    run = script.splitlines()[-1]
    argv = shlex.split(run)
    assert argv[:3] == [".venv/bin/python", "-m", "scripts.train_ghost_gk"]
    assert argv[argv.index("--stages") + 1] == "extract" and argv[argv.index("--shard") + 1] == "1/4"
    assert argv[argv.index("--store") + 1] == "/scratch/cycle1-4444444-ghost_gk-owner"
    assert argv[argv.index("--data-root") + 1] == "/data"


def test_volumes_mount_the_bucket_rw_and_the_dataset_ro_at_a_pinned_revision():
    v = L.volumes_for(_spec())
    bucket, ds = v
    assert (bucket.type, bucket.source, bucket.mount_path, bucket.read_only) == ("bucket", "silly-kicks/hf-refit-scratch", "/scratch", False)
    assert (ds.type, ds.source, ds.mount_path, ds.revision, ds.read_only) == ("dataset", "silly-kicks/hf-refit-tc3", "/data/tc3", "rev1", True)


def test_download_access_fetches_only_this_jobs_units():
    s = _spec(datasets=(L.DatasetMount("tc3", "silly-kicks/hf-refit-tc3", "rev1", "download"),))
    script = L.job_command(s)[2]
    assert "-m scripts.hf_jobs.datasets fetch tc3" in script and "--shard 1/4" in script
    assert all(vol.type != "dataset" for vol in L.volumes_for(s))


def test_open_data_jobs_pin_the_revision():
    s = _spec(datasets=(L.DatasetMount("sb_open", "statsbomb/open-data", "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb", "open"),))
    assert L.job_env(s)["SILLY_KICKS_OPEN_DATA_REVISION"] == "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
```

`tests/scripts/hf_jobs/test_bucket_store.py`:

```python
from __future__ import annotations

import pathlib

import pytest

from scripts.hf_jobs.store import BucketStore


class _File:
    def __init__(self, path):
        self.type, self.path = "file", path


class _Api:
    def __init__(self, files):
        self.files = files

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        return [_File(p) for p in self.files if p.startswith(prefix or "") and "/" not in p[len(prefix or "") :]]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False):
        for src, dest in files:
            if src in self.files:
                pathlib.Path(dest).write_bytes(self.files[src])


def test_from_uri_splits_bucket_and_prefix():
    s = BucketStore.from_uri("hf://buckets/silly-kicks/hf-refit-scratch/cycle1-abc-ghost_gk-owner")
    assert (s.bucket, s.prefix) == ("silly-kicks/hf-refit-scratch", "cycle1-abc-ghost_gk-owner")


def test_names_and_read_bytes_through_the_api():
    api = _Api({"rk/p/s/t/k.json": b"{}\n", "rk/p/s/t/k.sha256": b"x\n", "rk/p/s/t/sub/deep.json": b""})
    s = BucketStore("b/x", "rk", api=api)
    assert s.names("p/s/t") == {"k.json", "k.sha256"}
    assert s.read_bytes("p/s/t/k.json") == b"{}\n"
    assert s.read_bytes("p/s/t/missing.json") is None


def test_bucket_store_is_read_only():
    s = BucketStore("b/x", "rk", api=_Api({}))
    with pytest.raises(PermissionError):
        s.write_bytes("a", b"")
    assert s.local_path("a") is None


def test_controller_uploads_only_its_own_records():
    calls = []

    class _UpApi(_Api):
        def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
            calls.append((bucket_id, add))

    s = BucketStore("b/x", "rk", api=_UpApi({}))
    s.upload_bytes("_campaign/preflight.json", b"{}")
    assert calls == [("b/x", [(b"{}", "rk/_campaign/preflight.json")])]
    with pytest.raises(PermissionError):
        s.upload_bytes("ghost_gk/extract/tok/k.json", b"{}")
```

- [ ] **Step 2: Run and confirm the failures**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch_typed.py tests/scripts/hf_jobs/test_bucket_store.py -v -p no:randomly`
Expected: FAIL (`AttributeError: module 'scripts.hf_jobs.launch' has no attribute 'JobSpec'`, missing `store`).

- [ ] **Step 3: Implement `scripts/hf_jobs/store.py`**

```python
"""Read-only `Store` over the HF bucket API (ADR-128): the controller's view of a campaign store.

Jobs write through the bucket MOUNT (a `LocalStore` at /scratch/<run_key>); the stateless controller only
lists and reads small files (plan state, summaries, ledger records) through the API, so it can stop and
restart at any time."""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from typing import Any


@dataclasses.dataclass
class BucketStore:
    bucket: str  # "<namespace>/<bucket>"
    prefix: str  # the run key
    api: Any = None

    @classmethod
    def from_uri(cls, uri: str) -> BucketStore:
        rest = uri.removeprefix("hf://buckets/")
        ns, bucket, *prefix = rest.split("/")
        return cls(f"{ns}/{bucket}", "/".join(prefix))

    def _api(self):
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi()
        return self.api

    def _full(self, rel: str) -> str:
        return "/".join(p for p in (self.prefix, rel.strip("/")) if p)

    def names(self, rel_dir: str) -> set[str]:
        base = self._full(rel_dir) + "/"
        out = set()
        for entry in self._api().list_bucket_tree(self.bucket, prefix=base, recursive=False):
            if getattr(entry, "type", None) == "file":
                tail = entry.path[len(base) :]
                if "/" not in tail:
                    out.add(tail)
        return out

    def read_bytes(self, rel: str) -> bytes | None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = pathlib.Path(tmp) / "f"
            self._api().download_bucket_files(self.bucket, [(self._full(rel), str(dest))], raise_on_missing_files=False)
            return dest.read_bytes() if dest.is_file() else None

    def write_bytes(self, rel: str, data: bytes) -> None:
        raise PermissionError("BucketStore is read-only: jobs write through the bucket mount")

    def upload_bytes(self, rel: str, data: bytes) -> None:
        """The controller's ONLY write: its own records under `_campaign/` and `_partitions/` (preflight
        report, planned partitions). Never a unit result -- those are written in-job through the mount."""
        if not (rel.startswith("_campaign/") or rel.startswith("_partitions/")):
            raise PermissionError(f"controller uploads are limited to _campaign/ and _partitions/, got {rel!r}")
        self._api().batch_bucket_files(self.bucket, add=[(data, self._full(rel))])

    def delete(self, rel: str) -> None:
        raise PermissionError("BucketStore is read-only: jobs write through the bucket mount")

    def local_path(self, rel: str) -> pathlib.Path | None:
        return None
```

Before trusting `entry.type == "file"`, run once in `.venv-hfjobs`:
`python -c "import dataclasses, huggingface_hub as h; print([f.name for f in dataclasses.fields(h.BucketFile)])"`
(planning saw `['type', 'path', 'size', 'xet_hash', 'mtime', 'uploaded_at']` in hub 2.2.0). If the locked
hub version names them differently, follow the locked version and update the fake in the test.

- [ ] **Step 4: Add the typed surface to `scripts/hf_jobs/launch.py`**

Insert after the module imports (replace the module docstring with the one below; the legacy functions
keep working until Task 18):

```python
"""Launch ADR-128 stage-contract jobs on HF Jobs through the typed `huggingface_hub` API.

One image (digest-pinned, decision P1), one flavor (`cpu-upgrade`, INV-4), one in-job environment built
with `uv sync --frozen` from the committed `uv.lock` (INV-6), and an exact-commit checkout verified with
`git rev-parse` (the clone commit is the campaign commit; preflight proved it is pushed and green). Each
job carries labels (`campaign`, `run_key`, `job`, `variant`, `stage`, `shard`, `commit`) so the stateless
controller rebuilds state from `list_jobs(labels=...)`. Storage: the run's bucket prefix is mounted
read-write at /scratch (the store; persists across `attempts`, so a retry resumes); each staged dataset is
mounted read-only at /data/<name> at its pinned revision (or fetched per shard, `access = "download"`).
The `hf` CLI stays a human tool; nothing here shells out to it.
"""
```

```python
import dataclasses
import shlex
from collections.abc import Mapping
from typing import Any, Final

from scripts._thread_pin import DETERMINISTIC_VARS

FLAVOR: Final = "cpu-upgrade"
IMAGE: Final = "python:3.12.14-bookworm@sha256:dbbe4ceb97851e2e5fa83798b239811f871cb743b259ba3563737349f6bcfaa0"
PYTHON_BIN: Final = "/usr/local/bin/python3.12"
UV_REQUIREMENT: Final = "uv==0.9.28 --hash=sha256:7b8460a2b624d8ab27cb293a2c9f2393f9efc4e36e0fb886a6c2360e23fb48be"
DATA_MOUNT: Final = "/data"
TERMINAL = frozenset({"COMPLETED", "CANCELED", "ERROR", "DELETED"})


@dataclasses.dataclass(frozen=True)
class DatasetMount:
    name: str
    repo: str
    revision: str
    access: str  # "mount" | "download" | "open" (StatsBomb open data, fetched per unit, revision-pinned)

    @property
    def path(self) -> str:
        return f"{DATA_MOUNT}/{self.name}"


@dataclasses.dataclass(frozen=True)
class JobSpec:
    campaign: str
    run_key: str
    job: str
    variant: str
    stages: tuple[str, ...]
    shard: tuple[int, int]
    module: str
    args: tuple[str, ...]
    commit: str
    bucket: str
    workers: int
    inner_workers: int
    cap_bytes: int | None
    peak_rss_bytes: int | None
    timeout_s: int
    attempts: int
    datasets: tuple[DatasetMount, ...]
    extras: tuple[str, ...]
    profile: int | None = None
    shard_keys_path: str | None = None
    extra_env: Mapping[str, str] | None = None


def _label(spec: JobSpec) -> str:
    return f"{spec.job}.{spec.variant}.{'-'.join(spec.stages)}.{spec.shard[0]}of{spec.shard[1]}"


def labels_for(spec: JobSpec) -> dict[str, str]:
    return {
        "campaign": spec.campaign,
        "run_key": spec.run_key,
        "job": spec.job,
        "variant": spec.variant,
        "stage": "-".join(spec.stages),
        "shard": f"{spec.shard[0]}of{spec.shard[1]}",
        "commit": spec.commit,
    }


def job_env(spec: JobSpec) -> dict[str, str]:
    from scripts._provenance import IMAGE_ENV, RUN_KEY_ENV

    env = dict(DETERMINISTIC_VARS)
    env.update(
        {
            IMAGE_ENV: IMAGE,
            RUN_KEY_ENV: spec.run_key,
            "SILLY_KICKS_JOB_LABEL": _label(spec),
            "SILLY_KICKS_IN_HF_JOB": "1",  # in-job: an unrecorded dataset is an error, never a local guess
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    for d in spec.datasets:
        if d.access == "open":
            env["SILLY_KICKS_OPEN_DATA_REVISION"] = d.revision
    env.update(spec.extra_env or {})
    return env


def _run_argv(spec: JobSpec) -> list[str]:
    argv = [".venv/bin/python", "-m", spec.module, *spec.args, "--store", f"{BUCKET_MOUNT}/{spec.run_key}", "--data-root", DATA_MOUNT]
    argv += ["--stages", ",".join(spec.stages), "--shard", f"{spec.shard[0]}/{spec.shard[1]}"]
    argv += ["--workers", str(spec.workers), "--inner-workers", str(spec.inner_workers)]
    if spec.cap_bytes:
        argv += ["--cap-bytes", str(spec.cap_bytes)]
    if spec.peak_rss_bytes:
        argv += ["--peak-rss-bytes", str(spec.peak_rss_bytes)]
    if spec.profile:
        argv += ["--profile", str(spec.profile)]
    if spec.shard_keys_path:
        argv += ["--shard-keys", spec.shard_keys_path]
    return argv


def job_command(spec: JobSpec) -> list[str]:
    q = shlex.quote
    extras = " ".join(f"--extra {q(e)}" for e in spec.extras)
    lines = [
        "set -euo pipefail",
        "git init -q /work",
        "cd /work",
        f"git remote add origin {q(REPO_URL)}",
        f"git fetch -q --depth 1 origin {spec.commit}",
        "git checkout -q --detach FETCH_HEAD",
        f'test "$(git rev-parse HEAD)" = "{spec.commit}"',
        f"printf '%s\\n' {q(UV_REQUIREMENT)} > /tmp/uv-req.txt",
        f"{PYTHON_BIN} -m pip install -q --disable-pip-version-check --require-hashes -r /tmp/uv-req.txt",
        f"uv sync --frozen --python {PYTHON_BIN} --extra train {extras} --group hfjobs".replace("  ", " "),
    ]
    run = _run_argv(spec)
    for d in spec.datasets:
        if d.access == "download":
            fetch = [".venv/bin/python", "-m", "scripts.hf_jobs.datasets", "fetch", d.name, "--dest", d.path,
                     "--module", spec.module, "--stage", spec.stages[0], "--shard", f"{spec.shard[0]}/{spec.shard[1]}",
                     "--store", f"{BUCKET_MOUNT}/{spec.run_key}", "--data-root", DATA_MOUNT, "--", *spec.args]
            lines.append(" ".join(q(a) for a in fetch))
    lines.append(" ".join(q(a) for a in run))
    return ["bash", "-c", "\n".join(lines)]


def volumes_for(spec: JobSpec) -> list[Any]:
    from huggingface_hub import Volume

    vols = [Volume(type="bucket", source=spec.bucket, mount_path=BUCKET_MOUNT, read_only=False)]
    for d in spec.datasets:
        if d.access == "mount":
            vols.append(Volume(type="dataset", source=d.repo, mount_path=d.path, revision=d.revision, read_only=True))
    return vols


def launch(spec: JobSpec, *, api: Any, token: str) -> str:
    """Start ONE job; returns its id. The token rides `secrets` (never the command line)."""
    info = api.run_job(
        image=IMAGE,
        command=job_command(spec),
        env=job_env(spec),
        secrets={"HF_TOKEN": token},
        flavor=FLAVOR,
        timeout=spec.timeout_s,
        attempts=spec.attempts,
        labels=labels_for(spec),
        volumes=volumes_for(spec),
    )
    return info.id


def jobs_for(api: Any, *, campaign: str, commit: str) -> list[Any]:
    return list(api.list_jobs(labels={"campaign": campaign, "commit": commit}))
```

Keep `REPO_URL` and `BUCKET_MOUNT` (already defined in the legacy module). The legacy `build_argv`
contains the literal `"python:3.12"` and `spec.flavor`; `test_no_other_flavor_is_reachable` still passes
because the registry flavors are all `cpu-upgrade` (the `gk_retention` BLOCKED entry says `cpu-xl`, a
literal in `registry.py`: change it to `FLAVOR` imported from `launch` and keep `databricks_blocked=True` —
it is never launched, and INV-4 says no code path selects another flavor). The `datasets fetch`
subcommand is implemented in Task 7.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/scripts/hf_jobs/test_launch_typed.py tests/scripts/hf_jobs/test_bucket_store.py tests/scripts/hf_jobs/test_registry.py tests/scripts/hf_jobs/test_registry_flavor.py -v -p no:randomly`
Expected: PASS. (`Volume` comes from `huggingface_hub`: in `.venv` with hub 1.16 it does not exist; run
these two files in `.venv-hfjobs`, and add `pytest.importorskip("huggingface_hub", minversion="2.2")` at
the top of `test_launch_typed.py` for the matrix legs that resolve an older hub. CI installs `[test]`
floating, which resolves hub 2.x; record in the task notes which version CI resolved.)

---
## Task 7: Datasets (declared + staging lock), toy corpora, preflight

Spec: §5.4 (Preflight), §5.5 (datasets declared once), §5.7 (corpus expectation recorded at staging),
§12 R3, decisions P2, P6.

**Files:**
- Create: `scripts/hf_jobs/datasets.py`, `scripts/hf_jobs/datasets.toml`, `scripts/hf_jobs/datasets.lock.json`
- Create: `scripts/hf_jobs/preflight.py`
- Create: `scripts/_toy_corpora.py` (toy staged-layout corpora shared by tests and preflight check 5)
- Modify: `scripts/hf_jobs/stage.py` (docstring: staging now ends with `datasets record`)
- Test: `tests/scripts/hf_jobs/test_datasets_lock.py`, `tests/scripts/hf_jobs/test_preflight.py`

**Interfaces:**
- Produces (datasets): `Declared(name, repo, kind, visibility, access, revision, description)`;
  `declared() -> dict[str, Declared]`; `locked(name) -> dict` (KeyError when unrecorded);
  `record(name, *, api=None, fs=None, now=None) -> dict` (writes the lock entry);
  `pergame_name(provider: str, game: str) -> str` (`f"{provider}-{game}"`, single-sourced with
  `repackage.flat_to_pergame`); `expected_identity(name, naming: str) -> dict`;
  `staged_units(name) -> list[dict]` (`{"stem", "source_provider", "game"}`);
  CLI `python -m scripts.hf_jobs datasets {record,show,fetch} …`.
- Produces (toy corpora): `build_flat_corpus(root, *, n_games, providers, seed=0) -> pathlib.Path`
  (writes `shards/<gen>/<provider>__<game>.parquet`, `_actions/…`, `home_teams.json`) and
  `build_pergame_corpus(root, …)`; `build_open_data(root, …)` (Task 9); `TOY_PARAMS: dict[str, tuple[str, ...]]`
  (module → toy CLI args); `TOY_BUILDERS: dict[str, Callable[[Path], Path]]` (module → the builder that
  returns its `--data-root`); `build_toy_for(module, root) -> Path` (dispatches on `TOY_BUILDERS`).
- Produces (preflight): `Check(name, ok, detail, flags=())`; `run_preflight(campaign, *, api, run=subprocess.run, now=None) -> dict`;
  `require_preflight(campaign, commit, *, store) -> dict`; `REPORT_REL = "_campaign/{campaign}/{commit7}/preflight.json"`.

- [ ] **Step 1: Declare the datasets**

Create `scripts/hf_jobs/datasets.toml`:

```toml
# ADR-128: every dataset a campaign may read, declared ONCE. Human-written; no tool writes this file.
# `python -m scripts.hf_jobs datasets record <name>` MEASURES the staged dataset and writes the pinned
# revision, the unit list and the expected corpus identity into datasets.lock.json (decision P2), like
# pyproject.toml / uv.lock. Preflight check 4 compares the live dataset with the lock; `adopt` compares
# every artifact's recorded corpus with the lock (INV-9: an expectation recorded before the run).

[datasets.tc3]
repo = "silly-kicks/hf-refit-tc3"
kind = "flat_tc3"
visibility = "restricted"
access = "mount"
description = "179-match TC3 flat corpus (gradientsports + skillcorner + sportec): ghost_gk, ghost_outfield"

[datasets.xshot17]
repo = "silly-kicks/hf-refit-xshot17"
kind = "flat_tc3"
visibility = "public"
access = "mount"
description = "the 17-match public arm (7 idsse + 10 skillcorner): the wheel xShot and xCross bundles"

[datasets.xshot115]
repo = "silly-kicks/hf-refit-xshot115"
kind = "flat_tc3"
visibility = "restricted"
access = "mount"
description = "115-match idsse + SkillCorner corpus: the Hub sc_extended xShot and xCross artifacts"

[datasets.xcross_probe]
repo = "silly-kicks/hf-refit-xcross-probe"
kind = "pergame"
visibility = "restricted"
access = "mount"
description = "the held-out Gradient Sports TF-19 probe matches for xcross_attempt"

[datasets.sb360]
repo = "silly-kicks/hf-refit-sb360"
kind = "flat_tc3"
visibility = "restricted"
access = "mount"
description = "licensed StatsBomb 360 freeze-frame corpus for receiver_model"

[datasets.sb_open]
repo = "statsbomb/open-data"
kind = "statsbomb_open"
visibility = "public"
access = "open"
revision = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
description = "StatsBomb open data pinned to a commit (decision P6): the public trainers and the census"
```

- [ ] **Step 2: Write the failing tests**

`tests/scripts/hf_jobs/test_datasets_lock.py`:

```python
"""Datasets: declared once, measured at staging into the lock, identity computed like the in-job trainer."""

from __future__ import annotations

import json

import pytest

from scripts import _corpus, _corpus_identity
from scripts.hf_jobs import datasets as D
from scripts.hf_jobs.repackage import flat_to_pergame


def test_declared_kinds_access_and_visibility_are_valid():
    for name, d in D.declared().items():
        assert d.kind in D.KINDS and d.access in D.ACCESS and d.visibility in ("public", "restricted"), name
        assert (d.revision is not None) == (d.kind == "statsbomb_open"), name


def test_pergame_name_matches_the_repackage_tool(tmp_path):
    from scripts._toy_corpora import build_flat_corpus

    root = build_flat_corpus(tmp_path / "flat", n_games=2, providers=("skillcorner",))
    flat_to_pergame(root, tmp_path / "pg")
    stems = sorted(p.stem for p in (root / "shards").glob("*/*.parquet"))
    want = sorted(D.pergame_name(*s.split("__", 1)) for s in stems)
    assert sorted(p.name for p in (tmp_path / "pg").iterdir()) == want


def test_every_campaign_dataset_is_locked():
    lock = json.loads(D.LOCK.read_text(encoding="utf-8"))
    assert set(lock) == set(D.declared())
    for name, rec in lock.items():
        assert rec["revision"] and rec["kind"] == D.declared()[name].kind


def test_public_dataset_ids_agree_with_the_in_repo_registry():
    """Two independent sources (INV-9): the staging measurement and the PUBLIC_CORPUS registry."""
    assert D.locked("xshot17")["public_ids"] == _corpus.bundled_public_arm_pairs()


def test_expected_identity_uses_the_trainer_function():
    rec = D.locked("xshot115")
    units = rec["units"]
    providers = [u["source_provider"] for u in units]
    names = [D.pergame_name(u["stem"].split("__", 1)[0], u["game"]) for u in units]
    want = _corpus_identity.corpus_identity(providers, names, all_public=False)
    assert D.expected_identity("xshot115", "pergame") == want
    assert want["corpus_n_matches"] == 115


def test_record_refuses_a_kind_mismatch(tmp_path):
    class _Api:
        def dataset_info(self, repo, revision=None):
            return type("I", (), {"sha": "f" * 40})()

        def list_repo_files(self, repo, repo_type=None, revision=None):
            return ["foo/frames.parquet"]  # pergame layout, declared flat_tc3

    with pytest.raises(ValueError, match="declared kind"):
        D.record("tc3", api=_Api(), fs=None, lock_path=tmp_path / "lock.json")
```

`tests/scripts/hf_jobs/test_preflight.py`:

```python
"""Preflight (spec 5.4): six no-spend checks; launch refuses without a passing report for this commit."""

from __future__ import annotations

import json

import pytest

from scripts.hf_jobs import preflight as P

SHA = "a" * 40


def _ok(name):
    return P.Check(name, True, "ok")


def test_report_ok_only_when_every_check_passes():
    r = P.assemble_report("cycle1", SHA, "sha", [_ok("ref"), _ok("lock"), P.Check("api", False, "no labels")], flags=[])
    assert r["ok"] is False and r["commit"] == SHA


def test_estimate_flags_do_not_fail_the_report():
    r = P.assemble_report("cycle1", SHA, "sha", [_ok("ref")], flags=["ghost_gk/fit predicted 74 min > 60"])
    assert r["ok"] is True and r["flags"]


def test_require_preflight_refuses_another_commit_or_campaign_hash():
    class _Store:
        def __init__(self, data):
            self.data = data

        def read_bytes(self, rel):
            return self.data

    rep = json.dumps({"ok": True, "commit": SHA, "campaign_sha256": "h1"}).encode()
    assert P.require_preflight("cycle1", SHA, "h1", store=_Store(rep))["ok"]
    with pytest.raises(SystemExit, match="commit"):
        P.require_preflight("cycle1", "b" * 40, "h1", store=_Store(rep))
    with pytest.raises(SystemExit, match="campaign file changed"):
        P.require_preflight("cycle1", SHA, "h2", store=_Store(rep))
    with pytest.raises(SystemExit, match="no preflight"):
        P.require_preflight("cycle1", SHA, "h1", store=_Store(None))


def test_api_contract_check_reads_the_signature_and_the_flavor_catalog():
    class _Hw:
        name, cpu, ram, unit_cost_usd = "cpu-upgrade", "8 vCPU", "32 GB", 0.03

    class _Api:
        def run_job(self, *, image, command, env, secrets, flavor, timeout, attempts, labels, volumes):
            raise AssertionError

        def list_jobs(self, *, labels=None):
            return []

        def list_jobs_hardware(self):
            return [_Hw()]

    c = P.check_api(_Api())
    assert c.ok, c.detail


def test_api_contract_check_fails_when_volumes_disappear():
    class _Api:
        def run_job(self, *, image, command, env, secrets, flavor, timeout, attempts, labels):
            raise AssertionError

        def list_jobs(self, *, labels=None):
            return []

        def list_jobs_hardware(self):
            return []

    assert not P.check_api(_Api()).ok


def test_ref_check_requires_head_pushed_and_green():
    def run(argv, **kw):
        out = {
            ("git", "rev-parse", "HEAD"): SHA + "\n",
            ("git", "branch", "-r", "--contains", SHA): "  origin/feat/x\n",
            ("gh",): json.dumps([{"status": "completed", "conclusion": "success", "workflowName": "CI"}]),
        }
        key = tuple(argv) if argv[0] == "git" else ("gh",)
        return type("R", (), {"returncode": 0, "stdout": out.get(key, ""), "stderr": ""})()

    assert P.check_ref(SHA, run=run).ok
```

- [ ] **Step 3: Run and confirm the failures**

Run: `python -m pytest tests/scripts/hf_jobs/test_datasets_lock.py tests/scripts/hf_jobs/test_preflight.py -v -p no:randomly`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `scripts/_toy_corpora.py`**

Build the toy corpora from the builders the existing tests already use, so every migrated pipeline reads
the SAME staged layout an HF job reads (`shards/<gen>/<provider>__<game>.parquet`, `_actions/<same>.parquet`,
`home_teams.json`):
- move the frame/action synthesis of `tests/tracking/test_train_ghost_gk_cli.py::_tiny_cache` and
  `tests/scripts/test_train_ghost_outfield.py::_write_tiny_corpus` into `build_flat_corpus` (keep the
  test helpers as thin wrappers that call it, so their existing tests keep passing unchanged);
- move `tests/tracking/test_xshot_occurrence_integration.py::_write_synthetic_train_dir` and
  `tests/tracking/test_xcross_attempt_integration.py::_write_synthetic_train_dir` frame/shot synthesis
  into `build_flat_corpus(..., with_shots=True)` and `build_pergame_corpus(...)` (same wrapper rule);
- `TOY_PARAMS` maps each migrated module to the CLI arguments its toy `--run-all` needs (small `n_trials`,
  `--cv-folds 2`, `--n-estimators 10`, …); Tasks 10–16 fill one entry each, and the compute gate
  (Task 21) asserts the mapping covers every compliant module.

Each builder takes `seed` and is byte-deterministic (assert in a test that two builds hash equal).
`build_flat_corpus(dir)` builds ONE dataset IN `dir`; `build_toy_for(module, root)` builds
`root/<toy dataset name>` and returns `root` — the `--data-root` — where the toy dataset name is the
`--dataset` value in `TOY_PARAMS[module]` (e.g. `toygk`). Toy dataset names are never in
`datasets.lock.json`, so their identity is the local listing digest (`_stages._dataset_identity`).

- [ ] **Step 5: Implement `scripts/hf_jobs/datasets.py`**

```python
"""Declared datasets and staging records (ADR-128, spec 5.5 / 5.7, decision P2).

`datasets.toml` is human-written: repo, kind, visibility, access, and (for open data) the pinned commit.
`datasets.lock.json` is written ONLY by `record`, which measures the staged dataset: its revision, its
unit list, and the expected corpus identity computed with the SAME `_corpus_identity.corpus_identity` the
in-job trainer uses, over the SAME raw names (decision INV-9: the expectation exists before any job runs).
This module runs off-HF (operator, preflight, adopt); in-job code reads only `locked()` through
`_stages._dataset_identity` and `_driver_sources`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
DECLARED = HERE / "datasets.toml"
LOCK = HERE / "datasets.lock.json"
KINDS = ("flat_tc3", "pergame", "statsbomb_open")
ACCESS = ("mount", "download", "open")


@dataclasses.dataclass(frozen=True)
class Declared:
    name: str
    repo: str
    kind: str
    visibility: str
    access: str
    revision: str | None
    description: str


def declared() -> dict[str, Declared]:
    from scripts._toml import load_toml

    raw = load_toml(DECLARED)["datasets"]
    return {
        n: Declared(n, d["repo"], d["kind"], d["visibility"], d["access"], d.get("revision"), d.get("description", ""))
        for n, d in raw.items()
    }


def _read_lock(path: pathlib.Path = LOCK) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def locked(name: str, *, path: pathlib.Path = LOCK) -> dict:
    lock = _read_lock(path)
    if name not in lock:
        raise KeyError(f"dataset {name!r} has no staging record; run `python -m scripts.hf_jobs datasets record {name}`")
    return lock[name]


def pergame_name(provider: str, game: str) -> str:
    """The per-game directory name the xshot/xcross extraction keys on (`repackage.flat_to_pergame`)."""
    return f"{provider}-{game}" if game else provider


def staged_units(name: str) -> list[dict]:
    return list(locked(name)["units"])


def expected_identity(name: str, naming: str) -> dict:
    """The corpus identity an artifact trained on dataset ``name`` must record (spec 5.7)."""
    rec = locked(name)
    ident = rec["identity"].get(naming)
    if ident is None:
        raise KeyError(f"{name}: no {naming!r} identity recorded (namings: {sorted(rec['identity'])})")
    return ident


def _source_provider(fs: Any, path: str) -> str:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        if "source_provider" not in pf.schema_arrow.names or pf.num_row_groups == 0:
            return "unknown"
        col = pf.read_row_group(0, columns=["source_provider"])["source_provider"]
        return str(col[0].as_py()) if len(col) else "unknown"


def _identities(units: list[dict]) -> dict:
    from scripts._corpus_identity import corpus_identity

    providers = [u["source_provider"] for u in units]
    flat = [u["game"] for u in units]
    pergame = [pergame_name(u["stem"].split("__", 1)[0], u["game"]) for u in units]
    return {
        "flat": corpus_identity(providers, flat, all_public=False),
        "pergame": corpus_identity(providers, pergame, all_public=False),
    }


def _public_ids(units: list[dict]) -> list[list[str]]:
    from scripts._corpus_identity import _CONVERTER_TO_IDENTITY

    return sorted([[_CONVERTER_TO_IDENTITY.get(u["stem"].split("__", 1)[0], u["stem"].split("__", 1)[0]), u["game"]] for u in units])


def record(name: str, *, api: Any = None, fs: Any = None, now: str | None = None, lock_path: pathlib.Path = LOCK) -> dict:
    """Measure the staged dataset ``name`` and write its lock entry. Off-HF, read-only on the Hub."""
    d = declared()[name]
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if d.kind == "statsbomb_open":
        from scripts._sb_open_data import all_open_competitions, list_open_data_refs, pin_open_data_revision

        pin_open_data_revision(d.revision)
        refs = list_open_data_refs(all_open_competitions())
        keys = [r.match_id for r in refs]
        import hashlib

        entry = {
            "repo": d.repo,
            "revision": d.revision,
            "kind": d.kind,
            "n_units": len(keys),
            "units_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            "recorded_at": stamp,
        }
    else:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        rev = api.dataset_info(d.repo, revision=None).sha
        files = api.list_repo_files(d.repo, repo_type="dataset", revision=rev)
        if d.kind == "flat_tc3":
            shards = sorted(f for f in files if f.startswith("shards/") and f.endswith(".parquet"))
            gens = sorted({f.split("/")[1] for f in shards})
            if len(gens) != 1 or "home_teams.json" not in files:
                raise ValueError(f"{name}: declared kind flat_tc3 needs ONE shards/<gen>/ and home_teams.json; saw gens={gens}")
            stems = [pathlib.PurePosixPath(f).stem for f in shards]
            paths = {s: f"datasets/{d.repo}@{rev}/shards/{gens[0]}/{s}.parquet" for s in stems}
            generation = gens[0]
        elif d.kind == "pergame":
            dirs = sorted({f.split("/")[0] for f in files if f.endswith("/frames.parquet") and not f.startswith(".")})
            if not dirs:
                raise ValueError(f"{name}: declared kind pergame needs <game>/frames.parquet files")
            stems = dirs
            paths = {s: f"datasets/{d.repo}@{rev}/{s}/frames.parquet" for s in stems}
            generation = None
        else:  # pragma: no cover - guarded by KINDS
            raise ValueError(d.kind)
        if d.kind == "flat_tc3" and any(f.endswith("/frames.parquet") for f in files) and not shards:
            raise ValueError(f"{name}: declared kind flat_tc3 but the repo has a pergame layout")
        if fs is None:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
        units = []
        for s in stems:
            game = s.split("__", 1)[1] if "__" in s else s
            units.append({"stem": s, "source_provider": _source_provider(fs, paths[s]), "game": game})
        entry = {
            "repo": d.repo,
            "revision": rev,
            "kind": d.kind,
            "generation": generation,
            "units": units,
            "n_units": len(units),
            "identity": _identities(units),
            "public_ids": _public_ids(units) if d.visibility == "public" else None,
            "recorded_at": stamp,
        }
    lock = _read_lock(lock_path)
    lock[name] = entry
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry
```

The kind-mismatch guard in the test hits the `flat_tc3` branch with no `shards/` → `ValueError("…declared kind flat_tc3 needs ONE shards/<gen>/…")`; make that message contain the words `declared kind` (it does).

Add `fetch` (download access, spec 5.3 item 3: resume-before-download):

```python
def fetch(name: str, *, dest: pathlib.Path, module: str, stage: str, shard: str, store: str, data_root: str, args: list[str]) -> list[str]:
    """Download ONLY the files of this job's pending units (resume-before-download)."""
    import importlib

    from huggingface_hub import snapshot_download

    from scripts._driver import shard_select
    from scripts._provenance import code_identity, git_provenance
    from scripts._stages import ItemStage, build_parser, make_context, open_store, sorted_units, unit_states

    pipeline = importlib.import_module(module).PIPELINE
    ns = build_parser(pipeline).parse_args([*args, "--plan", "--store", store, "--data-root", data_root])
    params = {p.name: getattr(ns, p.name) for p in pipeline.params}
    git = git_provenance()
    ctx = make_context(pipeline, params, {**git, "code_identity": code_identity(git)}, store=open_store(store), data_root=pathlib.Path(data_root))
    st = pipeline.stage(stage)
    if not isinstance(st, ItemStage):
        return []
    i, n = (int(x) for x in shard.split("/"))
    keys = shard_select(sorted_units(ctx, st), i, n)
    states = unit_states(ctx, st, keys, verify=False)
    refs = st.ref_map(ctx)
    patterns = sorted({f for k in keys if states[k] not in ("done", "excluded") for f in getattr(refs[k], "files", ())})
    if patterns:
        rec = locked(name)
        patterns.append("home_teams.json")
        snapshot_download(rec["repo"], repo_type="dataset", revision=rec["revision"], local_dir=str(dest), allow_patterns=patterns)
    return patterns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.hf_jobs datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("name")
    s = sub.add_parser("show")
    s.add_argument("name")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    for flag in ("--dest", "--module", "--stage", "--shard", "--store", "--data-root"):
        f.add_argument(flag, required=True)
    f.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps({k: v for k, v in record(a.name).items() if k != "units"}, indent=2))
    elif a.cmd == "show":
        print(json.dumps(locked(a.name), indent=2))
    else:
        rest = [x for x in a.rest if x != "--"]
        got = fetch(a.name, dest=pathlib.Path(a.dest), module=a.module, stage=a.stage, shard=a.shard, store=a.store, data_root=a.data_root, args=rest)
        print(f"fetched {len(got)} pattern(s) for {a.name}", file=sys.stderr)
    return 0
```

The `--shard-keys` route (Task 5) is not in `fetch`: a job launched with `--shard-keys` uses
`access = "mount"` datasets only; `campaign` refuses `download` access together with an LPT partition.

- [ ] **Step 6: Record the six datasets (read-only on the Hub; owner token)**

Run (background; reads one row group per shard over the network):
```
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record tc3
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record xshot17
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record xshot115
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record xcross_probe
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record sb360
.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs datasets record sb_open
```
(`python -m scripts.hf_jobs datasets …` is dispatched by Task 8's `__main__`; until Task 8 exists, run
`python -m scripts.hf_jobs.datasets record <name>` with `if __name__ == "__main__": raise SystemExit(main())`.)
Expected: `n_units` 179, 17, 115, (probe count), (sb360 count), 3961 (open data at the pinned revision).
Compare with E10: 179 / 17 / 115. A different count is a STOP: report it; never edit the lock by hand.
The lock is restricted-dataset METADATA (stems, providers, digests), like the committed `metrics.json`
digests; it holds no frames. Confirm `public_ids` is set only for `xshot17` and `sb_open` has none.

- [ ] **Step 7: Implement `scripts/hf_jobs/preflight.py`**

```python
"""Preflight (spec 5.4): six no-spend checks. `launch` refuses without a passing report for the same
commit and the same campaign-file hash. The report lives in the bucket (the controller is stateless)."""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import json
import re
import subprocess
import sys
import tempfile
from typing import Any

REPORT_REL = "_campaign/{campaign}/{commit7}/preflight.json"
_RUN_JOB_PARAMS = {"image", "command", "env", "secrets", "flavor", "timeout", "attempts", "labels", "volumes"}


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    flags: tuple[str, ...] = ()


def _sh(run, argv: list[str]) -> tuple[int, str]:
    p = run(argv, capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout or "")


def check_ref(commit: str, *, run=subprocess.run) -> Check:
    rc, head = _sh(run, ["git", "rev-parse", "HEAD"])
    if rc or head.strip() != commit:
        return Check("ref", False, f"HEAD {head.strip()[:12]} != campaign commit {commit[:12]}")
    _sh(run, ["git", "fetch", "-q", "origin"])
    rc, remote = _sh(run, ["git", "branch", "-r", "--contains", commit])
    if rc or not remote.strip():
        return Check("ref", False, f"{commit[:12]} is not on any remote branch (push it first)")
    rc, out = _sh(run, ["gh", "run", "list", "--commit", commit, "--json", "status,conclusion,workflowName", "--limit", "50"])
    runs = json.loads(out or "[]") if rc == 0 else []
    green = [r for r in runs if r.get("status") == "completed" and r.get("conclusion") == "success" and r.get("workflowName") == "CI"]
    if not green:
        return Check("ref", False, f"no green CI run for {commit[:12]} (saw {[(r.get('workflowName'), r.get('conclusion')) for r in runs]})")
    return Check("ref", True, f"{commit[:12]} pushed and CI green")


def _image_resolves(image: str) -> bool:
    import urllib.request

    repo, _, digest = image.partition("@")
    name = repo.split(":")[0]
    token = json.load(urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/{name}:pull", timeout=30))["token"]  # noqa: S310 -- fixed https URL
    req = urllib.request.Request(  # noqa: S310 -- fixed https URL
        f"https://registry-1.docker.io/v2/library/{name}/manifests/{digest}",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.status == 200


def check_lock(*, run=subprocess.run) -> Check:
    from scripts.hf_jobs.launch import IMAGE

    rc, _ = _sh(run, ["uv", "lock", "--check"])
    if rc:
        return Check("lock", False, "uv lock --check failed: uv.lock is stale")
    try:
        ok = _image_resolves(IMAGE)
    except OSError as exc:
        return Check("lock", False, f"image digest did not resolve: {exc}")
    return Check("lock", ok, f"uv.lock current; {IMAGE} resolves" if ok else f"{IMAGE} does not resolve")


def check_api(api: Any) -> Check:
    params = set(inspect.signature(api.run_job).parameters)
    missing = _RUN_JOB_PARAMS - params
    if missing:
        return Check("api", False, f"run_job lacks {sorted(missing)}")
    if "labels" not in inspect.signature(api.list_jobs).parameters:
        return Check("api", False, "list_jobs lacks labels=")
    hw = [h for h in api.list_jobs_hardware() if getattr(h, "name", None) == "cpu-upgrade"]
    if not hw:
        return Check("api", False, "cpu-upgrade is not in the flavor catalog")
    cpu = int(re.findall(r"\d+", str(hw[0].cpu))[0])
    ram = int(re.findall(r"\d+", str(hw[0].ram))[0])
    if (cpu, ram) != (8, 32):
        return Check("api", False, f"cpu-upgrade is now {cpu} vCPU / {ram} GB, not 8 / 32: re-plan before spending")
    return Check("api", True, f"run_job/list_jobs typed surface present; cpu-upgrade 8 vCPU / 32 GB at ${getattr(hw[0], 'unit_cost_usd', '?')}/h")


def check_datasets(names: list[str], *, api: Any) -> Check:
    from scripts.hf_jobs.datasets import declared, locked

    bad = []
    for name in sorted(set(names)):
        d, rec = declared()[name], None
        try:
            rec = locked(name)
        except KeyError as exc:
            bad.append(str(exc))
            continue
        if d.kind == "statsbomb_open":
            continue  # a commit SHA is immutable; `record` already enumerated it
        try:
            files = set(api.list_repo_files(d.repo, repo_type="dataset", revision=rec["revision"]))
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            bad.append(f"{name}: revision {rec['revision'][:12]} unreadable ({exc})")
            continue
        for u in rec["units"]:
            want = f"shards/{rec['generation']}/{u['stem']}.parquet" if d.kind == "flat_tc3" else f"{u['stem']}/frames.parquet"
            if want not in files:
                bad.append(f"{name}: {want} missing at the pinned revision")
                break
        if d.kind == "flat_tc3" and "home_teams.json" not in files:
            bad.append(f"{name}: home_teams.json missing")
    return Check("datasets", not bad, "; ".join(bad) or "every dataset matches its lock")


def check_plans(instances: list[Any], *, run=subprocess.run) -> Check:
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for

    bad = []
    for inst in instances:
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = _sh(run, [sys.executable, "-m", inst.module, *inst.args, "--plan", "--store", f"{tmp}/s", "--data-root", f"{tmp}/d"])
            if rc or not json.loads(out or "{}").get("stages"):
                bad.append(f"{inst.job}/{inst.variant}: --plan failed")
                continue
        if inst.module in TOY_PARAMS:
            with tempfile.TemporaryDirectory() as tmp:
                data = build_toy_for(inst.module, f"{tmp}/d")
                rc, _ = _sh(run, [sys.executable, "-m", inst.module, *TOY_PARAMS[inst.module], "--run-all", "--store", f"{tmp}/s", "--data-root", str(data), "--allow-dirty"])
                if rc:
                    bad.append(f"{inst.module}: toy --run-all failed")
    return Check("plan", not bad, "; ".join(sorted(set(bad))) or "every --plan and toy --run-all passed")


def assemble_report(campaign: str, commit: str, campaign_sha256: str, checks: list[Check], *, flags: list[str], now: str | None = None) -> dict:
    return {
        "campaign": campaign,
        "commit": commit,
        "campaign_sha256": campaign_sha256,
        "created_at": now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "checks": [dataclasses.asdict(c) for c in checks],
        "flags": list(flags),
        "ok": all(c.ok for c in checks),
    }


def require_preflight(campaign: str, commit: str, campaign_sha256: str, *, store: Any) -> dict:
    raw = store.read_bytes(REPORT_REL.format(campaign=campaign, commit7=commit[:7]))
    if raw is None:
        raise SystemExit(f"no preflight report for {campaign}@{commit[:7]}: run `campaign <file> preflight` first")
    rep = json.loads(raw)
    if rep.get("commit") != commit:
        raise SystemExit(f"preflight report is for commit {rep.get('commit', '?')[:12]}, not {commit[:12]}")
    if rep.get("campaign_sha256") != campaign_sha256:
        raise SystemExit("the campaign file changed since preflight: re-run preflight")
    if not rep.get("ok"):
        raise SystemExit(f"preflight failed: {[c for c in rep['checks'] if not c['ok']]}")
    return rep
```

`check_estimate` lives in Task 8 (it needs the controller's planning loop) and contributes `flags`.

- [ ] **Step 8: Run the tests**

Run (in `.venv-hfjobs`, since the lock test reads the recorded lock and the identity test imports pandas/pyarrow):
`.venv-hfjobs/Scripts/python.exe -m pytest tests/scripts/hf_jobs/test_datasets_lock.py tests/scripts/hf_jobs/test_preflight.py -v -p no:randomly`
Expected: PASS. Then run in `.venv` too (py3.10 + `tomli`).

---

## Task 8: Campaign file, run keys, stateless controller, `python -m scripts.hf_jobs`

Spec: §5.5 (campaign file, run keys, CLI, stateless controller, ledger summary, profile, planner),
§5.4 (launch refuses without preflight), INV-7, §12 R5. Registry changes needed by the controller land
here; Task 18 removes the legacy fields.

**Files:**
- Create: `scripts/hf_jobs/campaign.py`, `scripts/hf_jobs/__main__.py`, `scripts/hf_jobs/campaigns/cycle1.toml`
- Modify: `scripts/hf_jobs/registry.py` (add `certification`, `pipeline`, `tiers`; `DriverSpec` + `DRIVERS`; the `full` ghost build; `--variant default`)
- Test: `tests/scripts/hf_jobs/test_campaign.py`, `tests/scripts/hf_jobs/test_campaign_cycle1.py`

**Interfaces:**
- Consumes: Tasks 3–7.
- Produces: `Campaign`, `CampaignJob`, `Instance(job, tier, variant, module, args, datasets, run_key)`;
  `load_campaign(path) -> Campaign`; `run_key(campaign, commit, job, tier, *, replica=None) -> str`;
  `instances(campaign, commit, *, replica=None) -> list[Instance]`;
  `next_actions(plan_json, running, history, constraints) -> list[Action]` (pure);
  `Action(kind: "profile"|"run"|"reduce", stages, n_jobs, workers, inner_workers, cap_bytes, peak_rss_bytes, timeout_s, partition, profile)`;
  `step(campaign, commit, *, api, token, replica=None, dry_run=False) -> dict` (one stateless advance);
  CLI `python -m scripts.hf_jobs campaign <file> {preflight,profile,plan,launch,resume,status,retrieve,verify,adopt,publish,cancel,summary} [--replica N] [--watch] [--dry-run]`.
- Registry: `TrainerSpec.certification: str` (`public` | `owner` | `rebaseline`), `TrainerSpec.pipeline: str`
  (dotted module path; `== module` for migrated trainers), `TrainerSpec.adapter: bool = False`;
  `DriverSpec(name, module, certification, output_rel)`; `DRIVERS: dict[str, DriverSpec]`.

- [ ] **Step 1: Write the campaign file**

`scripts/hf_jobs/campaigns/cycle1.toml`:

```toml
# ADR-128 Cycle 1 campaign (spec 7.2): every migrated job, both xShot/xCross tiers, the census driver,
# and the three adapter-launched public trainers. Run with:
#   .venv-hfjobs/Scripts/python -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/cycle1.toml <command>
# The commit is HEAD at preflight (pushed, CI green). Run keys are derived; never hand-picked.

[campaign]
name = "cycle1"
bucket = "silly-kicks/hf-refit-scratch"
target_minutes = 60
concurrency_cap = 10
attempts = 3

[[jobs]]
job = "ghost_gk"
tier = "owner"
dataset = "tc3"
variants = ["default", "position_only", "sweeper", "sweeper_position_only", "full"]

[[jobs]]
job = "ghost_outfield"
tier = "owner"
dataset = "tc3"
variants = ["default", "position_only"]

[[jobs]]
job = "xshot_occurrence"
tier = "public"
dataset = "xshot17"
variants = ["default", "position_only"]

[[jobs]]
job = "xshot_occurrence"
tier = "sc_extended"
dataset = "xshot115"
variants = ["default", "position_only"]

[[jobs]]
job = "xcross_attempt"
tier = "public"
dataset = "xshot17"
probe_dataset = "xcross_probe"
variants = ["default", "position_only"]

[[jobs]]
job = "xcross_attempt"
tier = "sc_extended"
dataset = "xshot115"
probe_dataset = "xcross_probe"
variants = ["default", "position_only"]

[[jobs]]
job = "receiver_model"
tier = "owner"
dataset = "sb360"
variants = ["default"]

[[jobs]]
job = "win_probability"
tier = "public"
dataset = "sb_open"
variants = ["default"]

[[jobs]]
job = "build_territory_ranking_census"
tier = "public"
dataset = "sb_open"
variants = ["default"]

[[jobs]]
job = "match_outcome_dependence"
tier = "public"
dataset = "sb_open"
variants = ["default"]

[[jobs]]
job = "pass_completion"
tier = "public"
dataset = "sb_open"
variants = ["default"]

[[jobs]]
job = "xsuccess"
tier = "public"
dataset = "sb_open"
variants = ["default"]
```

- [ ] **Step 2: Extend the job registry**

In `scripts/hf_jobs/registry.py`:
- add to `TrainerSpec`: `certification: str = "owner"`, `pipeline: str = ""` (empty for an adapter),
  `adapter: bool = False`;
- `win_probability`: `certification="public"`; the three adapter trainers (`match_outcome_dependence`,
  `pass_completion`, `xsuccess`): `certification="public"`, `adapter=True`, `pipeline=""`; `ghost_gk`, `ghost_outfield`,
  `receiver_model`: `certification="owner"`; `xshot_occurrence`, `xcross_attempt`: certification is per
  TIER (public tier = `public`, sc_extended tier = `owner`): add `tier_certification: dict[str, str] | None = None`
  and use `{"public": "public", "sc_extended": "owner"}` for both;
- `ghost_gk` builds: `default` run_args become `("--feature-set", "faithful", "--variant", "default")`
  (decision P10) and add `VariantBuild("full", ("--feature-set", "faithful", "--variant", "full"), "", "ghost_gk_v1")`
  (Hub-only: empty `committed_path`; the artifact manifest names its Hub destination);
- add `DriverSpec` and `DRIVERS = {"build_territory_ranking_census": DriverSpec("build_territory_ranking_census", "scripts.build_territory_ranking_census", "rebaseline", "docs/research/territory_ranking_census")}`;
- add `def spec_for(name) -> TrainerSpec | DriverSpec` that searches `REGISTRY`, `DRIVERS`, `BLOCKED`;
- `pipeline` stays `""` on every non-adapter entry until its migration task (10–16) sets it to the
  module path; `win_probability` is set in Task 10, not here.

Update `tests/scripts/hf_jobs/test_registry.py` for the new fields (read it first; keep every existing
assertion that still holds, and add: every `REGISTRY` entry has a `certification` in
`{"public","owner","rebaseline"}`; a migrated entry's `pipeline` imports and exposes `PIPELINE`; an adapter
entry has `pipeline == ""`).

- [ ] **Step 3: Write the failing controller tests**

`tests/scripts/hf_jobs/test_campaign.py`:

```python
"""Stateless campaign controller (spec 5.5): pure next-action logic over --plan JSON + job labels."""

from __future__ import annotations

from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.planner import Constraints, UnitCost

SHA = "abcdef1" + "0" * 33


def _plan(stages):
    return {"pipeline": "ghost_gk", "stages": stages}


def _st(name, *, n=4, done=0, pending=None, ready=True, reduced=False, inner=False, hk=None, units=None):
    return {
        "name": name, "n_units": n, "done": done, "pending": (n - done) if pending is None else pending,
        "failed": 0, "excluded": 0, "ready": ready, "reduced": reduced, "inner_parallel": inner,
        "inner_width": 5 if inner else 1, "history_key": hk or name, "units": units or [f"u{i}" for i in range(n)],
        "token": f"t_{name}",
    }


def test_run_key_is_derived():
    c = C.Campaign(name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h")
    assert C.run_key(c, SHA, "ghost_gk", "owner") == "cycle1-abcdef1-ghost_gk-owner"
    assert C.run_key(c, SHA, "ghost_gk", "owner", replica=2) == "cycle1-abcdef1-ghost_gk-owner-r2"


def test_a_stage_without_history_is_profiled_first():
    acts = C.next_actions(_plan([_st("extract", n=179)]), running=set(), history={}, c=Constraints())
    assert [a.kind for a in acts] == ["profile"] and acts[0].stages == ("extract",)


def test_a_stage_with_history_is_fanned_by_the_planner():
    hist = {"extract": {"cost": UnitCost(600, 700, 800, 6 * 2**30, 20), "efficiency": 0.9}}
    acts = C.next_actions(_plan([_st("extract", n=179)]), running=set(), history=hist, c=Constraints())
    assert acts[0].kind == "run" and acts[0].n_jobs > 1 and acts[0].workers == 3


def test_a_running_stage_is_not_relaunched():
    acts = C.next_actions(_plan([_st("extract", n=179)]), running={"extract"}, history={}, c=Constraints())
    assert acts == []


def test_the_barrier_waits_for_upstream():
    p = _plan([_st("extract", done=4, reduced=True), _st("cv", ready=True), _st("fit", n=1, ready=False)])
    acts = C.next_actions(p, running=set(), history={"cv": {"cost": UnitCost(60, 60, 60, 2**30, 5), "efficiency": 1.0}}, c=Constraints())
    assert [a.stages for a in acts] == [("cv",)]


def test_consecutive_single_unit_stages_run_in_one_job():
    p = _plan([_st("extract", done=4, reduced=True), _st("cache", n=1), _st("fit", n=1, ready=False)])
    acts = C.next_actions(p, running=set(), history={}, c=Constraints())
    assert acts[0].stages == ("cache",) and acts[0].kind == "run"  # fit is not ready: only cache now


def test_a_finished_but_unreduced_output_stage_gets_a_reduce_job():
    p = _plan([_st("assemble", n=1, done=1, reduced=False)])
    assert C.next_actions(p, running=set(), history={}, c=Constraints())[0].kind == "reduce"


def test_everything_reduced_means_done():
    assert C.next_actions(_plan([_st("a", done=4, reduced=True)]), running=set(), history={}, c=Constraints()) == []
```

`tests/scripts/hf_jobs/test_campaign_cycle1.py`:

```python
"""The committed Cycle 1 campaign file resolves against the registries and the dataset lock."""

from __future__ import annotations

import pathlib

from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.datasets import declared
from scripts.hf_jobs.registry import spec_for

CYCLE1 = pathlib.Path("scripts/hf_jobs/campaigns/cycle1.toml")


def test_cycle1_names_known_jobs_variants_and_datasets():
    c = C.load_campaign(CYCLE1)
    for j in c.jobs:
        spec = spec_for(j.job)
        assert j.dataset in declared()
        if hasattr(spec, "builds"):
            assert set(j.variants) <= {b.variant for b in spec.builds}, j.job


def test_cycle1_covers_the_spec_job_set():
    jobs = {(j.job, j.tier) for j in C.load_campaign(CYCLE1).jobs}
    assert {
        ("ghost_gk", "owner"), ("ghost_outfield", "owner"), ("xshot_occurrence", "public"),
        ("xshot_occurrence", "sc_extended"), ("xcross_attempt", "public"), ("xcross_attempt", "sc_extended"),
        ("receiver_model", "owner"), ("win_probability", "public"), ("build_territory_ranking_census", "public"),
        ("match_outcome_dependence", "public"), ("pass_completion", "public"), ("xsuccess", "public"),
    } == jobs


def test_instances_carry_dataset_args_and_derived_run_keys():
    c = C.load_campaign(CYCLE1)
    insts = C.instances(c, "abcdef1" + "0" * 33)
    xc = next(i for i in insts if i.job == "xcross_attempt" and i.tier == "sc_extended" and i.variant == "default")
    assert ["--dataset", "xshot115"] == list(xc.args[xc.args.index("--dataset") : xc.args.index("--dataset") + 2])
    assert "--probe-dataset" in xc.args and "--arms-json" in xc.args
    assert xc.run_key == "cycle1-abcdef1-xcross_attempt-sc_extended"
```

- [ ] **Step 4: Run and confirm the failures**

Run: `python -m pytest tests/scripts/hf_jobs/test_campaign.py tests/scripts/hf_jobs/test_campaign_cycle1.py -v -p no:randomly`
Expected: FAIL (`ModuleNotFoundError: scripts.hf_jobs.campaign`).

- [ ] **Step 5: Implement `scripts/hf_jobs/campaign.py`**

Write the module with these parts, in this order:

1. **Model.** `CampaignJob(job, tier, dataset, variants, probe_dataset=None)`; `Campaign(name, bucket,
   target_minutes, concurrency_cap, attempts, jobs, path, sha256)`; `load_campaign(path)` reads TOML via
   `_toml.load_toml` and sets `sha256 = hashlib.sha256(path.read_bytes()).hexdigest()`.
2. **Run keys.** `run_key(c, commit, job, tier, *, replica=None)` → `f"{c.name}-{commit[:7]}-{job}-{tier}"`
   plus `f"-r{replica}"` when given. The replica suffix is the INV-1 campaign gate's fresh store
   (Task 25): a same-commit re-run without it would resume every unit and prove nothing.
3. **Instances.** `instances(c, commit, *, replica=None)`: for each job × variant:
   `spec = registry.spec_for(job)`; `module = spec.pipeline or spec.module`; `args = base_args + build.run_args
   + ("--dataset", job.dataset)`; for a probe job add `("--probe-dataset", job.probe_dataset)`; for a
   visibility-gated job add `("--arms-json", json.dumps(arm_policy.compute_arms([u["stem"] for u in datasets.staged_units(job.dataset)]), sort_keys=True))`
   (computed OFF-HF from the lock, ADR-123 Amendment B); for `receiver_model` add
   `("--corpus-visibility", declared()[dataset].visibility, "--corpus-n-matches", str(locked(dataset)["n_units"]))`
   (decision P5); an adapter instance's `module` is `scripts.hf_jobs.adapter` and its args are
   `("--target", spec.module, "--", *spec.base_args, *build.run_args)` (Task 17). `datasets` =
   `DatasetMount(name, repo, revision, access)` for `dataset` and `probe_dataset` from `declared()` + `locked()`.
4. **Plan read.** `read_plan(inst, *, bucket) -> dict`: `subprocess.run([sys.executable, "-m", inst.module,
   *inst.args, "--plan", "--store", f"hf://buckets/{bucket}/{inst.run_key}"], capture_output=True, text=True,
   check=True)` → `json.loads(stdout)`.
5. **History.** `load_campaign_history()` = `ledger.load_history(Path("docs/research/hf_campaigns").glob("*.json"))`
   merged with live measurements: for each stage of a plan, `ledger.unit_cost(ledger.unit_records(store,
   generation))` and the stage's `_profile.json` efficiency read through `BucketStore`. `history[history_key]
   = {"cost": UnitCost, "efficiency": float | None, "per_unit": {key: wall_s}}`.
6. **Pure next-action logic** (`next_actions(plan, running, history, c)`), the function the tests pin:
   - walk stages in order; skip a stage with `reduced` True;
   - if not `ready` → stop (stage barrier);
   - if `n_units` is None → stop (units unknown until upstream output exists);
   - if the stage name is in `running` → stop (one action per stage at a time);
   - if `pending == 0` and not `reduced` → `Action(kind="reduce", stages=(name,), n_jobs=1, workers=1, …)`;
   - if `n_units <= 2` → `Action(kind="run", stages=group, n_jobs=1, workers=1, inner_workers=inner_width if inner_parallel else 1, timeout=2×target)` where `group` extends with following stages while each next stage has `n_units == 1`, is `ready` only through stages in the group (`needs ⊆ group ∪ reduced`), and is not `inner_parallel`;
   - else if `history_key` not in history → `Action(kind="profile", stages=(name,), n_jobs=1, workers=c.vcpu, inner_workers=min(c.vcpu, inner_width), profile=min(n_units, 2 * c.vcpu), timeout=2×target)`;
   - else → `planner.plan_stage(...)` from history → `Action(kind="run", stages=(name,), n_jobs=plan.n_jobs, workers=plan.workers, inner_workers=plan.inner_workers, cap_bytes=plan.cap_bytes, peak_rss_bytes=plan.peak_rss_bytes, timeout_s=plan.timeout_s, partition=plan.partition)`;
   - return after the first actionable stage (one stage per instance per step: the barrier).
7. **Step.** `step(c, commit, *, api, token, replica=None, dry_run=False)`:
   - `preflight.require_preflight(c.name, commit, c.sha256, store=BucketStore(c.bucket, ""))`;
   - `jobs = launch.jobs_for(api, campaign=c.name, commit=commit)`; `running` = labels of jobs whose
     `status.stage` ∉ `launch.TERMINAL`, keyed `(run_key, variant, stage)`;
   - `budget = c.concurrency_cap - len(running)`;
   - a FAILED instance = its latest job for the stage it is on ended `ERROR` after HF `attempts`, and
     the stage still has pending units → report it, skip it (the others continue, spec 5.5);
   - for each instance: `plan = read_plan(...)`; for each action from `next_actions`, build `JobSpec`s
     (`n_jobs` shards, or the uploaded LPT parts via `BucketStore.upload_bytes("_partitions/...")`),
     launch while `budget > 0`;
   - return a report dict `{instance: {"stage", "action", "jobs": [ids], "failed": reason | None}}`.
   `dry_run` prints the JobSpecs and the planned cost and launches nothing.
8. **Commands.** `preflight` (all six checks; `check_estimate` = `next_actions` over every instance with the
   loaded history; a `run` action whose `predicted_job_wall_s > target` becomes a flag naming the stage;
   uploads the report to `_campaign/<name>/<commit7>/preflight.json`); `profile` (= `step` but only
   `profile` actions); `plan` (prints every instance's next action + predicted wall/cost);
   `launch`/`resume` (= `step`; with `--watch`, loop `step` every 60 s until every instance's output stage
   is reduced or every remaining instance is failed; prints one status line per loop); `status` (table:
   instance × stage done/pending, running jobs, failures); `retrieve` (Task 20); `verify`/`adopt`/`publish`
   (Task 20); `cancel` (`api.cancel_job` for every running job of this campaign+commit, after printing them);
   `summary` (writes `docs/research/hf_campaigns/<name>.json` via `ledger.summarize`: per stage
   `history_key`, `n_units`, `n_jobs`, `workers`, `inner_workers`, p50/p95/p99 unit wall, peak RSS,
   efficiency, startup (median of first-unit start minus `JobInfo.started_at`), max job wall, attempts,
   resumed units, irreducible floor; per job `ledger.job_record`; per artifact `identity.artifact_identity`).
   The summary's `run_commit` is the campaign commit and `run_tree_dirty` is False (the jobs ran a clean
   checkout); the `summary` command itself refuses a dirty local tree (`require_clean_tree`), because the
   file is a registered artifact (ADR-037).
9. **CLI.** `main(argv)`: `campaign <file> <command> [--replica N] [--watch] [--dry-run] [--staging DIR]`.
   Resolve `commit = git rev-parse HEAD`; `token = huggingface_hub.get_token()` (refuse if None);
   `api = HfApi()`.

Create `scripts/hf_jobs/__main__.py`:

```python
"""`python -m scripts.hf_jobs {campaign,datasets,scaffold} ...` (ADR-128)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("campaign", "datasets", "scaffold"):
        print("usage: python -m scripts.hf_jobs {campaign,datasets,scaffold} ...", file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "campaign":
        from scripts.hf_jobs.campaign import main as run
    elif cmd == "datasets":
        from scripts.hf_jobs.datasets import main as run
    else:
        from scripts.hf_jobs.scaffold import main as run
    return int(run(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
```

(`scaffold` arrives in Task 21; until then the import fails only when that subcommand is used.)

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/scripts/hf_jobs/test_campaign.py tests/scripts/hf_jobs/test_campaign_cycle1.py tests/scripts/hf_jobs/test_registry.py -v -p no:randomly`
Expected: PASS for the controller tests; `test_campaign_cycle1` needs the migrated modules' `PIPELINE`
params (`--dataset`, `--probe-dataset`, `--arms-json`) for its `--plan` round-trip only in Task 18's
end-to-end check; this task's three tests use the registry and the lock only.

- [ ] **Step 7: Dry-run against the real registry (no spend)**

Run: `.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/cycle1.toml plan`
Expected at this point: every not-yet-migrated module reports that it has no `PIPELINE`; this is the
signal Tasks 10–17 remove. Re-run after Task 18; then every instance must print an action.

---
## Task 9: Neutral loader port, open-data pin, I1 gate on every HF job, ADR-052 gate recognises `ItemStage`

Spec: §5.8 item 4 (dataset-agnostic driver loader, I1 extension), INV-5, decisions P5, P6, P12, P17.

**Files:**
- Create: `scripts/_driver_sources.py`
- Modify: `scripts/_loader_pining.py` (re-export `LoadedMatch`, `resolve_cache_dir`, `CORPUS_CACHE_ENV` from `_driver_sources`; keep every other symbol)
- Modify: `scripts/_sb_open_data.py` (import from `_driver_sources`; `pin_open_data_revision`; `SILLY_KICKS_OPEN_DATA_BASE` test override)
- Modify: `scripts/train_ghost_gk.py:1149` (comment reword: no banned substring)
- Modify: `tests/scripts/hf_jobs/test_visibility_blind_gate.py` (derived population)
- Modify: `tests/scripts/test_corpus_driver_resilience.py` (`ItemStage` recognition + planted test)
- Modify: `docs/PRIVATE_CONSUMERS.md` (record the move; read it first)
- Test: `tests/scripts/test_driver_sources.py`, `tests/scripts/test_open_data_pin.py`

**Interfaces:**
- Produces: `scripts._driver_sources`: `CORPUS_CACHE_ENV`, `resolve_cache_dir(cache_dir)`, `LoadedMatch`
  (moved verbatim), `StagedRef(provider, game, stem, source_provider, files)` with `.key -> (provider, game)`,
  `dataset_dir(data_root, dataset) -> Path`, `staged_flat_refs(data_root, dataset) -> list[StagedRef]`,
  `staged_pergame_refs(data_root, dataset) -> list[StagedRef]`, `home_teams(data_root, dataset) -> dict`.
- Produces: `scripts._sb_open_data.OPEN_DATA_REVISION_ENV = "SILLY_KICKS_OPEN_DATA_REVISION"`,
  `OPEN_DATA_BASE_ENV = "SILLY_KICKS_OPEN_DATA_BASE"`, `pin_open_data_revision(revision=None) -> str | None`.
- Produces (tests): `tests/scripts/_open_data_server.py` — `serve_open_data(root) -> ContextManager[str]`
  (a local `http.server` over a toy open-data tree; yields the base URL).

- [ ] **Step 1: Write the failing tests**

`tests/scripts/test_driver_sources.py`:

```python
"""Neutral loader port (ADR-052 D14 / ADR-128): refs read the lock (HF) or the local tree (dev/toy)."""

from __future__ import annotations

import ast
import pathlib

from scripts import _driver_sources as S
from scripts._toy_corpora import build_flat_corpus, build_pergame_corpus


def test_flat_refs_list_units_and_their_files(tmp_path):
    root = build_flat_corpus(tmp_path / "tc3", n_games=3, providers=("skillcorner",))
    refs = S.staged_flat_refs(tmp_path, "tc3")
    assert [r.stem for r in refs] == sorted(r.stem for r in refs)
    gen = next((root / "shards").iterdir()).name
    assert refs[0].files == (f"shards/{gen}/{refs[0].stem}.parquet", f"_actions/{refs[0].stem}.parquet")
    assert refs[0].source_provider == "skillcorner" and refs[0].key == (refs[0].provider, refs[0].game)


def test_pergame_refs_skip_dot_directories(tmp_path):
    build_pergame_corpus(tmp_path / "probe", n_games=2, provider="gradientsports")
    (tmp_path / "probe" / ".cache").mkdir()  # what `hf download --local-dir` leaves behind
    (tmp_path / "probe" / ".cache" / "frames.parquet").write_bytes(b"")
    assert all(not r.stem.startswith(".") for r in S.staged_pergame_refs(tmp_path, "probe"))


def test_driver_sources_carries_no_visibility_concept():
    src = pathlib.Path(S.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "scripts._loader_pining" not in mods and "scripts._corpus" not in mods


def test_loader_pining_still_exports_the_moved_names():
    from scripts import _loader_pining as P

    assert P.LoadedMatch is S.LoadedMatch and P.resolve_cache_dir is S.resolve_cache_dir
```

`tests/scripts/test_open_data_pin.py`:

```python
"""Decision P6: StatsBomb open data is read at ONE pinned commit."""

from __future__ import annotations

import pytest

statsbombpy = pytest.importorskip("statsbombpy")

from scripts import _sb_open_data as O  # noqa: E402

REV = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"


def test_pin_rewrites_every_open_data_url(monkeypatch):
    from statsbombpy import config

    saved = dict(config.OPEN_DATA_PATHS)
    try:
        assert O.pin_open_data_revision(REV) == REV
        assert all(f"/open-data/{REV}/data/" in u for u in config.OPEN_DATA_PATHS.values())
        assert O.pin_open_data_revision(REV) == REV  # idempotent
    finally:
        config.OPEN_DATA_PATHS.clear()
        config.OPEN_DATA_PATHS.update(saved)


def test_pin_reads_the_environment(monkeypatch):
    from statsbombpy import config

    saved = dict(config.OPEN_DATA_PATHS)
    monkeypatch.setenv(O.OPEN_DATA_REVISION_ENV, REV)
    try:
        O.assert_statsbomb_open_data_mode()
        assert all(REV in u for u in config.OPEN_DATA_PATHS.values())
    finally:
        config.OPEN_DATA_PATHS.clear()
        config.OPEN_DATA_PATHS.update(saved)


def test_a_non_sha_revision_is_refused():
    with pytest.raises(ValueError, match="40-hex"):
        O.pin_open_data_revision("master")
```

Extend `tests/scripts/hf_jobs/test_visibility_blind_gate.py`: keep `_resolve`, `_imported_modules`,
`_closure`, `_banned_hits` and the anti-rot tests; replace the two per-trainer tests with a DERIVED
population (ADR-056):

```python
def _hf_job_entries() -> list[str]:
    """Every module an HF campaign launches: each registry trainer's pipeline module (or its module, for
    an adapter-launched trainer: the adapter runs its CLI in-job) and every driver (spec 5.8 item 4)."""
    from scripts.hf_jobs.registry import DRIVERS, REGISTRY

    mods = {s.pipeline or s.module for s in REGISTRY.values()} | {d.module for d in DRIVERS.values()}
    return sorted(m.removeprefix("scripts.") for m in mods)


@pytest.mark.parametrize("entry", _hf_job_entries())
def test_every_hf_job_closure_is_visibility_blind(entry):
    hits = {p.name: _banned_hits(p) for p in _closure(entry) if _banned_hits(p)}
    assert not hits, f"visibility concept reachable from the HF job {entry}: {hits}"


def test_the_population_is_derived_and_non_empty():
    assert {"train_xshot_occurrence", "train_xcross_attempt", "train_ghost_gk", "build_territory_ranking_census"} <= set(_hf_job_entries())
```

Add `import pytest` to the file. Until Task 11 migrates the receiver, its entry fails this gate (E8);
in THIS task add `_I1_PENDING = {"train_receiver_model": "Task 11 removes the in-job match_visibility read (decision P5)"}`
and skip those entries with that reason; Task 11 deletes the dict and the skip, so the committed state
has no pending entry.

Extend `tests/scripts/test_corpus_driver_resilience.py` (decision P12):
- `_DRIVER_API`: add `"ItemStage"`; `_SHARD_PRIMITIVES`: add `"ItemStage"`;
- `_uses_for_each(tree)`: `return bool({"for_each", "ItemStage"} & _called_names(tree))` (an `ItemStage` is
  executed by `for_each`, which asserts conservation and injectivity internally);
- `_for_each_load_key_sources()`: match `clr._call_name(node) in ("for_each", "ItemStage")`;
- add:

```python
def test_detection_catches_a_planted_ITEMSTAGE_driver():
    """ADR-128: a migrated driver declares ItemStage(load=..., key=...) and never calls for_each itself.
    It must still be in the population (it pulls a corpus) and count as adopted."""
    tree = _parse(
        "from scripts._sb_open_data import open_data_source\n"
        "from scripts._stages import ItemStage\n"
        "def _refs(ctx):\n    return open_data_source([(43, 106)])[0]\n"
        "S = ItemStage(name='x', refs=_refs, key=lambda ref: ref.key, load=lambda r, c: r, work=lambda i, c: None)\n"
    )
    assert _is_corpus_driver(tree) and _adopts(tree) and _uses_for_each(tree)
```

- [ ] **Step 2: Run and confirm the failures**

Run: `python -m pytest tests/scripts/test_driver_sources.py tests/scripts/test_open_data_pin.py tests/scripts/hf_jobs/test_visibility_blind_gate.py tests/scripts/test_corpus_driver_resilience.py -v -p no:randomly`
Expected: FAIL (missing module, missing `pin_open_data_revision`, I1 hits for ghost_gk/win_probability/census/adapters, the planted `ItemStage` test).

- [ ] **Step 3: Implement**

Create `scripts/_driver_sources.py`. Move `CORPUS_CACHE_ENV`, `resolve_cache_dir` and `LoadedMatch`
VERBATIM from `scripts/_loader_pining.py` (keep their docstrings); in `_loader_pining.py` replace the
definitions with `from scripts._driver_sources import CORPUS_CACHE_ENV, LoadedMatch, resolve_cache_dir  # noqa: F401 -- re-exported (PRIVATE_CONSUMERS)`.
Then add:

```python
@dataclasses.dataclass(frozen=True)
class StagedRef:
    """A cheap reference to ONE unit of a staged neutral dataset (ADR-123 Amendment B, ADR-128).

    ``files`` are the dataset-relative paths the unit reads, so a job fetches only its own units
    (resume-before-download, spec 5.3 item 3). ``source_provider`` comes from the staging lock (HF) or
    from the shard's first row group (a local dev/toy tree)."""

    provider: str
    game: str
    stem: str
    source_provider: str
    files: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.game)


def dataset_dir(data_root, dataset: str | None) -> pathlib.Path:
    root = pathlib.Path(data_root)
    return root / dataset if dataset else root


def _lock_units(dataset: str | None):
    if not dataset:
        return None
    try:
        from scripts.hf_jobs.datasets import locked

        return locked(dataset)
    except KeyError:
        if os.environ.get("SILLY_KICKS_IN_HF_JOB"):  # in-job: an unrecorded dataset is a hard error
            raise
        return None


def _first_provider(path: pathlib.Path) -> str:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if "source_provider" not in pf.schema_arrow.names or pf.num_row_groups == 0:
        return "unknown"
    col = pf.read_row_group(0, columns=["source_provider"])["source_provider"]
    return str(col[0].as_py()) if len(col) else "unknown"


def staged_flat_refs(data_root, dataset: str | None) -> list[StagedRef]:
    """Units of a ``flat_tc3`` dataset: ``shards/<gen>/<provider>__<game>.parquet`` + ``_actions/``."""
    rec = _lock_units(dataset)
    if rec is not None:
        gen = rec["generation"]
        units = [(u["stem"], u["source_provider"]) for u in rec["units"]]
    else:
        root = dataset_dir(data_root, dataset)
        gens = sorted(p.name for p in (root / "shards").iterdir() if p.is_dir() and not p.name.startswith("."))
        if len(gens) != 1:
            raise ValueError(f"{root}: expected ONE shards/<generation>/ directory, found {gens}")
        gen = gens[0]
        units = [(p.stem, _first_provider(p)) for p in sorted((root / "shards" / gen).glob("*.parquet"))]
    refs = []
    for stem, src in sorted(units):
        provider, _, game = stem.partition("__")
        refs.append(StagedRef(provider, game, stem, src, (f"shards/{gen}/{stem}.parquet", f"_actions/{stem}.parquet")))
    return refs


def staged_pergame_refs(data_root, dataset: str | None) -> list[StagedRef]:
    """Units of a ``pergame`` dataset: ``<game>/frames.parquet`` (+ ``actions.parquet``/``shots.parquet``).
    Dot-directories are never units (the ``hf download`` ``.cache/`` crash of 4.138.0)."""
    rec = _lock_units(dataset)
    if rec is not None:
        units = [(u["stem"], u["source_provider"]) for u in rec["units"]]
    else:
        root = dataset_dir(data_root, dataset)
        units = [
            (p.name, _first_provider(p / "frames.parquet"))
            for p in sorted(root.iterdir())
            if p.is_dir() and not p.name.startswith(".") and (p / "frames.parquet").is_file()
        ]
    return [
        StagedRef(src, stem, stem, src, (f"{stem}/frames.parquet", f"{stem}/actions.parquet", f"{stem}/shots.parquet"))
        for stem, src in sorted(units)
        if not stem.startswith(".")
    ]


def home_teams(data_root, dataset: str | None) -> dict:
    p = dataset_dir(data_root, dataset) / "home_teams.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
```

(Imports at module top: `dataclasses`, `json`, `os`, `pathlib`, `typing.NamedTuple` for `LoadedMatch`,
`pandas` only under `TYPE_CHECKING` if `LoadedMatch`'s annotations need it — read the moved code.)

In `scripts/_sb_open_data.py`:
- replace `from scripts._loader_pining import LoadedMatch, resolve_cache_dir` with
  `from scripts._driver_sources import LoadedMatch, resolve_cache_dir`;
- fix the two docstring mentions of `_loader_pining` so the I1 substring gate stays meaningful (it scans
  for banned SYMBOLS, not module names; the module name is not banned, so the docstrings may stay — verify
  with the gate);
- add:

```python
OPEN_DATA_REVISION_ENV = "SILLY_KICKS_OPEN_DATA_REVISION"
#: TEST-ONLY: point statsbombpy at a local http server mirroring the open-data tree (never set in a job).
OPEN_DATA_BASE_ENV = "SILLY_KICKS_OPEN_DATA_BASE"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def pin_open_data_revision(revision: str | None = None) -> str | None:
    """Pin every statsbombpy open-data URL to ONE commit of statsbomb/open-data (ADR-128, decision P6).

    statsbombpy formats ``config.OPEN_DATA_PATHS`` (``.../open-data/master/data/...``) on every call and
    ``statsbombpy.public`` binds that same dict, so an in-place edit pins every later call in this process.
    ``revision`` defaults to ``$SILLY_KICKS_OPEN_DATA_REVISION``; ``None`` there leaves the URLs alone.
    ``$SILLY_KICKS_OPEN_DATA_BASE`` (tests) replaces the whole host+revision prefix instead.
    """
    from statsbombpy import config  # type: ignore[import-not-found]  # optional network dep

    base = os.environ.get(OPEN_DATA_BASE_ENV)
    rev = revision or os.environ.get(OPEN_DATA_REVISION_ENV)
    if base:
        for k, url in list(config.OPEN_DATA_PATHS.items()):
            config.OPEN_DATA_PATHS[k] = re.sub(r"^https?://.+?/data/", base.rstrip("/") + "/data/", url)
        return rev
    if not rev:
        return None
    if not _SHA.match(rev):
        raise ValueError(f"open-data revision must be a 40-hex commit SHA, got {rev!r}")
    for k, url in list(config.OPEN_DATA_PATHS.items()):
        config.OPEN_DATA_PATHS[k] = re.sub(r"/open-data/[^/]+/data/", f"/open-data/{rev}/data/", url)
    return rev
```

and call `pin_open_data_revision()` as the LAST statement of `assert_statsbomb_open_data_mode()` (every
open-data consumer calls that first: `list_open_data_refs`, `train_win_probability.main`, the census and
the three adapter trainers). Add `import os` and `import re` at the top.

Create `tests/scripts/_open_data_server.py`: a `contextlib.contextmanager` that starts
`http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=root))` in a
daemon thread, yields `f"http://127.0.0.1:{port}"`, and shuts down. Add to `scripts/_toy_corpora.py` a
`build_open_data(root, *, match_ids=("3754058", "7298", "7584"))` that writes `data/competitions.json`,
`data/matches/43/106.json`, `data/events/<id>.json` (copied from `tests/datasets/statsbomb/raw/events/`),
and `data/lineups/<id>.json` built from each match's "Starting XI" events (`[{team_id, team_name, lineup:[{player_id, player_name}]}]`).

In `scripts/train_ghost_gk.py:1149` reword the comment `fail-closed, like is_public_row's default` to
`fail-closed, like the public-row rule's default` (the substring gate scans the source).

- [ ] **Step 4: Run the tests and the gates**

Run: `python -m pytest tests/scripts/test_driver_sources.py tests/scripts/test_open_data_pin.py tests/scripts/hf_jobs/test_visibility_blind_gate.py tests/scripts/test_corpus_driver_resilience.py tests/scripts/test_source_factories.py -v -p no:randomly`
Expected: PASS (receiver skipped with the Task 11 reason). If a Rule A–D test in
`test_corpus_driver_resilience.py` names an `ItemStage` site, read the rule's docstring in
`tests/scripts/_corpus_load_rules.py`: an `ItemStage(load=…)` is the D14 resume-before-load seam, so the
rule must treat it exactly like `for_each(load=…)` — extend the rule's call-name check by `"ItemStage"` and
add one planted case beside the rule's existing plants.

Run the PRIVATE_CONSUMERS check: `git grep -n "_loader_pining" docs/PRIVATE_CONSUMERS.md` and record the
re-export there in the file's own format.

---

## Migration procedure (applies to Tasks 10–16)

Spec §5.6 rule: **only orchestration moves.** Feature extraction, objectives, fits and seeds stay
unchanged; each module keeps `--run-all` as its serial parity reference. Every migration task follows
these steps in order. The task text gives the module-specific stage table, code and test names.

- **M1 — Capture the legacy reference (decision P16).** Before editing the module, in `.venv-hfjobs`,
  build the module's toy corpus with `scripts/_toy_corpora.py`, run the CURRENT CLI on it into
  `<scratchpad>/legacy/<job>/out`, and record `sha256` of every artifact file plus the full
  `metrics.json` into `<scratchpad>/legacy/<job>/reference.json`. The task gives the exact command.
- **M2 — Write the failing parity test** `tests/scripts/test_parity_<job>.py` with the shared helpers in
  `tests/scripts/_parity.py` (created in Task 10): a serial `--run-all --workers 1` and a fanned run (two
  concurrent `--shard i/2` jobs per multi-unit stage, `--workers 2`, `--inner-workers 2`), then: every
  stage output and unit result byte-identical (volatile ledger/failed/profile files excluded), the
  exported artifact byte-identical, and for each fanned multi-unit stage the completion order differed
  from key order (non-vacuity, spec §5.2). Mark `@pytest.mark.slow` with its measured cost (ADR-023).
- **M3 — Refactor.** Delete the module's `argparse` and phase flags. Move each block of the legacy
  `main()` into the named module-level stage function (the task maps line ranges to functions). Declare
  `PIPELINE = Pipeline(...)` and end the module with
  `if __name__ == "__main__": raise SystemExit(main(PIPELINE))` (import `main` from `scripts._stages`).
  Keep every helper the tests import (`extract_match`, `_cv_splits`, …) at module level with its name.
- **M4 — Green.** Run the parity test and the module's existing unit tests; update a test only where it
  called a deleted CLI flag, and say so in the task notes.
- **M5 — Legacy equivalence.** Run the new `--run-all --out <scratchpad>/legacy/<job>/new` on the SAME toy
  corpus; the payload `sha256` values and `metrics.json` must equal M1's, except the keys decision P4
  names (`generation`; and pass-scoped counters only when M1 ran a resumed pass — it does not). Any other
  difference is a STOP: report it with both values.
- **M6 — Wire.** Set `registry.REGISTRY[<job>].pipeline = "scripts.<module>"`; add the module's
  `TOY_PARAMS` entry; keep its `_KEY_EXCEPTIONS` entry (keys are unchanged) or add one with the
  pre-migration key; delete the module's superseded tests (listed per task) — their behaviour is covered
  by the parity test.
- **M7 — Lint/type.** `uvx ruff@0.15.7 check` + `format --check` on the touched files; `pyright` on the module.

`tests/scripts/_parity.py` (created in Task 10, shared):

```python
"""Shared W1N1-vs-W2N2 parity harness for migrated pipelines (ADR-128, spec 5.2)."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

VOLATILE = (".ledger.json", ".failed.json", "_profile.json")


def cli(module, store, data_root, args, *extra, check=True, env=None):
    cmd = [sys.executable, "-m", module, *args, "--store", str(store), "--data-root", str(data_root), "--allow-dirty", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603 -- fixed argv
    if check and p.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nrc={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}")
    return p


def plan(module, store, data_root, args, env=None):
    return json.loads(cli(module, store, data_root, args, "--plan", env=env).stdout)


def run_serial(module, store, data_root, args, out, env=None):
    cli(module, store, data_root, args, "--run-all", "--workers", "1", "--out", str(out), env=env)


def run_fanned(module, store, data_root, args, out, *, jobs=2, workers=2, env=None):
    """Advance stage by stage like the controller: a multi-unit stage runs as `jobs` concurrent shard
    processes (each labelled like an HF job, so the ledger groups units per job); an inner_parallel stage
    gets --inner-workers; then export the output."""
    base = dict(os.environ if env is None else env)
    for st in plan(module, store, data_root, args, env)["stages"]:
        p = next(s for s in plan(module, store, data_root, args, env)["stages"] if s["name"] == st["name"])
        n = p["n_units"] or 0
        flags = ["--inner-workers", str(workers)] if p["inner_parallel"] else ["--workers", str(workers)]
        k = min(jobs, n) if n > 1 else 1
        procs = [
            subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", module, *args, "--store", str(store), "--data-root", str(data_root), "--allow-dirty",
                 "--stages", st["name"], "--shard", f"{i}/{k}", *flags],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**base, "SILLY_KICKS_JOB_LABEL": f"{st['name']}.{i}of{k}"},
            )
            for i in range(k)
        ]
        for proc in procs:
            _o, e = proc.communicate()
            assert proc.returncode == 0, e[-4000:]
        cli(module, store, data_root, args, "--stages", st["name"], env=env)  # idempotent reduce
    cli(module, store, data_root, args, "--run-all", "--out", str(out), env=env)  # nothing pending: exports


def outputs(store) -> dict[str, str]:
    root = pathlib.Path(store)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(VOLATILE)
    }


def artifact(out) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path(out).iterdir()) if p.is_file()}


def completion_out_of_order(store, generation: str) -> bool:
    """True when, INSIDE at least one job, units finished out of key order -- the W >= 2 worker path,
    not merely two jobs interleaving."""
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    jobs: dict[str, list[dict]] = {}
    for r in recs:
        jobs.setdefault(r.get("job_label") or "", []).append(r)
    return any(
        [r["key"] for r in sorted(rs, key=lambda r: r["finished_ns"])] != sorted(r["key"] for r in rs)
        for rs in jobs.values()
    )


def inner_out_of_order(store, generation: str) -> bool:
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    return any(r.get("inner_completion") and r["inner_completion"] != sorted(r["inner_completion"]) for r in recs)
```

Every parity test has this shape (the task gives `MODULE`, `ARGS`, the toy builder, and the stages that
must show out-of-order completion):

```python
@pytest.mark.slow  # measured: <N> s on <machine> (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    data = build_toy(tmp_path / "data")
    run_serial(MODULE, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, ARGS)["stages"]}
    for stage in OUT_OF_ORDER_STAGES:
        assert completion_out_of_order(tmp_path / "s2", gens[stage]), stage
    for stage in INNER_STAGES:
        assert inner_out_of_order(tmp_path / "s2", gens[stage]), stage
```

A toy unit can be so fast that completion order happens to equal key order. If a non-vacuity assertion
fails for that reason, do NOT weaken it: make the toy corpus heterogeneous (e.g. one large first match)
so the first key is genuinely slowest, and record that in the builder's docstring.

---
## Task 10: Migrate `win_probability` (public source, large-N fan-out, resume-before-download)

Spec: §5.6 row `win_probability`; decisions P4, P6. Follow the migration procedure M1–M7.

**Files:**
- Modify: `scripts/train_win_probability.py` (lines 246–377 become stage functions; lines 1–243 keep their pure helpers unchanged)
- Create: `tests/scripts/_parity.py` (the shared harness above), `tests/scripts/test_parity_win_probability.py`
- Modify: `scripts/_toy_corpora.py` (`TOY_PARAMS["scripts.train_win_probability"]`), `scripts/hf_jobs/registry.py` (`pipeline=`)
- Modify: the module's existing tests that call `main([...])` (find them: `git grep -ln "train_win_probability" tests/`)

**Pipeline:**

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `refs` | Stage | `["only"]` | `_refs_work`: the competition resolution (`:259-260`, `:277-286`) + `list_open_data_refs(comps, max_matches=…)` | the ordered ref list + `competitions` (JSON) |
| `extract` | ItemStage | one per open-data match | `_extract_work`: `match_shard(actions, home, match_date)` (`:295-297`) | concat in sorted key order (`:309-314`) |
| `fit` | Stage | `["only"]` | `_fit_work`: `:315-370` | the weights FileSet (output) |

- [ ] **M1 — legacy reference.** In `.venv-hfjobs`, build the toy open-data tree
(`scripts/_toy_corpora.build_open_data(<scratchpad>/legacy/wp/od)`), serve it
(`tests/scripts/_open_data_server.serve_open_data`), and run with `SILLY_KICKS_OPEN_DATA_BASE=<url>`:
`python -m scripts.train_win_probability --out <scratchpad>/legacy/wp/out --competition-id 43 --season-id 106 --allow-dirty`.
Record `sha256` of `out/weights/{model.json,SHA256SUMS,metrics.json,MODEL_CARD.md}` and the metrics JSON.
(Write the three-line driver for this as a scratchpad script; it is a one-off verification, not orchestration.)

- [ ] **M2 — failing parity test** `tests/scripts/test_parity_win_probability.py`:

```python
from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")

from scripts._toy_corpora import build_open_data  # noqa: E402
from tests.scripts._open_data_server import serve_open_data  # noqa: E402
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial  # noqa: E402

MODULE = "scripts.train_win_probability"
ARGS = ["--competition-id", "43", "--season-id", "106"]


@pytest.mark.slow  # measured: <fill in from --durations> (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    root = build_open_data(tmp_path / "od")
    with serve_open_data(root) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        run_serial(MODULE, tmp_path / "s1", tmp_path / "data", ARGS, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, tmp_path / "o2", env=env)
        gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, env=env)["stages"]}
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"model.json", "SHA256SUMS", "metrics.json", "MODEL_CARD.md"}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
```

(`--data-root` is required by the CLI but unused by an open-data pipeline; `tmp_path / "data"` may be empty.)

Run: `.venv-hfjobs/Scripts/python.exe -m pytest tests/scripts/test_parity_win_probability.py -v -p no:randomly` → FAIL (`--plan` unknown: the module still has its legacy argparse).

- [ ] **M3 — refactor.** Replace `main()` (`:246-370`) and the `__main__` block (`:373-377`) with:

```python
import dataclasses

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, main

_PARAMS = (
    Param("dataset", str, "sb_open", help="declared dataset name (scripts/hf_jobs/datasets.toml)"),
    Param("all_competitions", flag=True, help="full public open-data corpus (every release)"),
    Param("competition_id", int, None, help="open-data competition (default 43 = World Cup)"),
    Param("season_id", int, None, help="open-data season (default 106 = 2022)"),
    Param("max_matches", int, None, help="cap the number of matches"),
    Param("cache_dir", str, None, help="raw open-data events cache root (else $SILLY_KICKS_CORPUS_CACHE_DIR)"),
)


def _competitions(params) -> list[tuple[int, int]]:
    import scripts._sb_open_data as sbod

    if params["all_competitions"] and (params["competition_id"] is not None or params["season_id"] is not None):
        raise ValueError("--all-competitions is mutually exclusive with --competition-id/--season-id")
    sbod.assert_statsbomb_open_data_mode()
    if params["all_competitions"]:
        return [(int(c[0]), int(c[1])) for c in sbod.all_open_competitions()]
    return [
        (
            _DEFAULT_COMPETITION_ID if params["competition_id"] is None else params["competition_id"],
            _DEFAULT_SEASON_ID if params["season_id"] is None else params["season_id"],
        )
    ]


def _refs_work(key, ctx) -> dict:
    import scripts._sb_open_data as sbod

    comps = _competitions(ctx.params)
    refs = sbod.list_open_data_refs(comps, max_matches=ctx.params["max_matches"])
    return {"competitions": [list(c) for c in sorted(comps)], "refs": [dataclasses.asdict(r) for r in refs]}


def _extract_refs(ctx):
    from scripts._sb_open_data import OpenDataRef

    return [OpenDataRef(**d) for d in ctx.upstream("refs")["refs"]]


def _load_match(ref, ctx):
    """Resume-before-download: the executor calls this only for a pending unit (ADR-052 D14)."""
    from scripts._driver_sources import resolve_cache_dir
    from scripts._sb_open_data import assert_statsbomb_open_data_mode, load_open_data_match

    assert_statsbomb_open_data_mode()  # pins the revision in THIS worker process (decision P6)
    return load_open_data_match(ref, cache_dir=resolve_cache_dir(ctx.params["cache_dir"])), ref.match_date


def _extract_work(item, ctx):
    lm, match_date = item
    # the legacy date_map came from the same sb.matches manifest the ref was built from (:288-291)
    return match_shard(lm.actions, lm.home_team_id, match_date)


def _extract_reduce(results, ctx):
    frames = list(results.values())  # sorted key order == the legacy sorted shard glob (:309)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_SHARD_COLUMNS)


def _fit_work(key, ctx) -> FileSet:
    import tempfile

    from silly_kicks.win_probability import WinProbabilityModel, WinProbabilityParams

    pooled = ctx.upstream("extract")
    if not len(pooled):
        raise RuntimeError("no action rows collected from the corpus; nothing to fit")
    comps = [tuple(c) for c in ctx.upstream("refs")["competitions"]]
    params = WinProbabilityParams.default()
    # ---- lines :318-345 verbatim (strength, games, calibration_metrics, fit, certify_coherence,
    #      corpus expected goals, model_expected_goals) ----
    ...
    prov = ctx.prov
    summary = ctx.summary("extract")
    # Decision P4: CORPUS-scoped counters (the values a single fresh serial pass reports), never
    # pass-scoped ones, so resume history cannot change metrics.json (INV-1).
    pass_fields = {
        "generation": ctx.tokens["extract"],
        "n_attempted": summary["n_units"],
        "n_failed": summary["n_failed"],
        "n_counters_unrecorded": 0,
        "n_excluded": summary["n_excluded"],
    }
    out = {
        **metrics,
        "corpus_goals_per_match": corpus_goals_per_match,
        "model_expected_goals": float(model_expected_goals),
        "expected_goals_tol": 0.5,
        "n_matches": n_matches,
        "n_action_rows": len(pooled),
        "n_competitions": len(comps),
        "competitions": [list(c) for c in sorted(comps)],
        "training_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        **pass_fields,
    }
    payload = json.dumps(out, indent=2, default=str)
    with tempfile.TemporaryDirectory() as tmp:
        wdir = Path(tmp) / "weights"
        model.save(wdir)
        files = {p.name: p.read_bytes() for p in sorted(wdir.iterdir()) if p.is_file()}
    files["metrics.json"] = payload.encode("utf-8")
    files["MODEL_CARD.md"] = render_model_card(out).encode("utf-8")
    print(payload)
    return FileSet(files)


PIPELINE = Pipeline(
    name="win_probability",
    module="scripts.train_win_probability",
    params=_PARAMS,
    stages=(
        Stage(
            name="refs",
            units=lambda ctx: ["only"],
            work=_refs_work,
            reduce=lambda r, ctx: r["only"],
            params=("all_competitions", "competition_id", "season_id", "max_matches"),
            uses_dataset=True,
        ),
        ItemStage(
            name="extract",
            refs=_extract_refs,
            key=lambda ref: ref.key,
            load=_load_match,
            work=_extract_work,
            needs=("refs",),
            reduce=_extract_reduce,
            declare=lambda p: {"model": "WinProbabilityModel", "cols": list(_SHARD_COLUMNS)},
            cross_commit_version="win-probability-shard-1",
            uses_dataset=True,
            max_failures=None,  # legacy for_each recorded failures and fit over the rest
        ),
        Stage(name="fit", units=lambda ctx: ["only"], work=_fit_work, needs=("refs", "extract"), reduce=lambda r, ctx: r["only"]),
    ),
    output="fit",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
```

The `...` marks where lines `:318-345` move VERBATIM (they reference `pooled`, `params`, `model`; no
other change). Delete the `sys.path.insert` hack (the module runs as `python -m`). The legacy `--out`
directory layout (`<out>/weights/…`) becomes `--run-all --out <dir>` writing the four files directly; a
reader that copied `<out>/weights` into the wheel now copies `<out>` (Task 20's `adopt` does this).

- [ ] **M4–M7** as in the procedure. `TOY_PARAMS["scripts.train_win_probability"] = ("--competition-id", "43", "--season-id", "106")`
(the toy builder for open data is `build_open_data`; `build_toy_for` dispatches on the module). Registry:
`REGISTRY["win_probability"].pipeline = "scripts.train_win_probability"`, `builds[0].run_args = ("--all-competitions",)`
(unchanged), `fresh_subdir = ""` (the export writes the bundle files directly). `_KEY_EXCEPTIONS`: none
(`ref.key`). M5: `metrics.json` may differ from M1 ONLY in `generation`.

---

## Task 11: Migrate `receiver_model` (owner_sb360 staged source; INV-5 fix)

Spec: §5.6 row `receiver_model`; decisions P5, P17. Procedure M1–M7.

**Files:**
- Modify: `scripts/train_receiver_model.py` (`:365-404` sources, `:407-427` deployment, `:440-481` extraction, `:492-677` main)
- Create: `tests/scripts/test_parity_receiver_model.py`
- Modify: `tests/scripts/hf_jobs/test_visibility_blind_gate.py` (delete `_I1_PENDING` and its skip)
- Delete: `tests/scripts/test_train_receiver_mapreduce.py` (the `--shard-index` paths it tests no longer exist)
- Modify: other receiver tests that call `main()` or `_corpus_source` (`git grep -ln "train_receiver_model" tests/`)

**Pipeline:**

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `extract` | ItemStage | one per staged unit of `--dataset` | `extract_candidate_rows(actions, frames, feature_set, labeling_strategy)` (`:463`), counters `_coverage_counters` | concat in sorted key order (legacy `reconcile`), raw ids |
| `pool` | ItemStage | one per unit of `--pool-dataset` (none when unset) | as `extract`, with the pool provider's strategy | concat, raw ids |
| `fit` | Stage | `["only"]` | `:595-660` without the pining block: namespace ids (`_namespace_game_ids`), vacuity check, pooling gate, fit, `cv_top1`, manifest | `{"model": FileSet, "manifest": …}` as a FileSet (`model/…` files + `manifest.json`) |
| `deploy` | ItemStage | one per unit of `--deployment-dataset` (none when unset or feature_set != owner) | `_deployment_counts_for_match(public_model, owner_model, actions, frames, match_id=…)` | `_pooled_deployment(concat)` |
| `assemble` | Stage | `["only"]` | `:661-672`: velocity ablation, deployment result, write `metrics.json` | the bundle FileSet (output) |

Parameters (the legacy flags that stay; P17 for the rest): `dataset`, `feature_set` (`public|owner`),
`provider`, `pool_dataset`, `pool_provider`, `deployment_dataset`, `public_bundle` (a repo path),
`match_ids_json` (a JSON list filtering the PRIMARY units by `stem`), `min_rows`, `min_passes`,
`corpus_visibility` (`public|restricted`, REQUIRED), `corpus_n_matches` (int, REQUIRED).

- [ ] **M1** — legacy reference on a toy flat corpus of SB360-like frames (`build_flat_corpus(..., providers=("statsbomb",), freeze_frames=True)`;
reuse the frame synthesis of `tests/scripts/test_train_receiver*.py`): legacy
`python -m scripts.train_receiver_model --out <o> --shard-root <s> --data-dir <toy> --provider statsbomb --feature-set public --min-passes 1 --allow-dirty`.
The legacy run imports `_loader_pining.select_match_ids` after the fit (`:630-638`), which needs a pining
manifest: run it with `SILLY_KICKS_PINING_MANIFEST_DIR` pointing at a toy offline manifest
(`<dir>/statsbomb/matches.json` listing the toy match ids with visibility `restricted`) — the
offline-manifest path `_list_matches` already supports. Record `corpus_visibility` and `corpus_n_matches`
it computes: they are the values the new run receives as parameters.

- [ ] **M2** — parity test (`MODULE = "scripts.train_receiver_model"`, ARGS
`["--dataset", "toyrecv", "--provider", "statsbomb", "--feature-set", "public", "--min-passes", "1", "--corpus-visibility", "restricted", "--corpus-n-matches", "<toy n>"]`,
data root containing `toyrecv/`; out-of-order stage `extract`).

- [ ] **M3** — refactor:
  - delete `_corpus_source` (`:365-379`), the pining branch of `_extract_provider_rows`, `--cache-dir`,
    `--owner-rows`, `--shard-index`, and the `scripts._loader_pining` / `scripts._corpus` imports
    (`:630`, `:487`): `corpus_visibility` and `corpus_n_matches` come from the parameters (decision P5) and
    `_corpus_visibility` (`:484-489`) is deleted;
  - `_data_dir_source` (`:382-404`) becomes `_refs(ctx, dataset_param)` over
    `_driver_sources.staged_flat_refs(ctx.data_root, ctx.params[dataset_param])` filtered by
    `match_ids_json` for the primary, and `_load(ref, ctx)` reading `frames`/`_actions` by `ref.files`;
  - the unit key stays the legacy `ref.match_id` VALUE (`stem`): `key=lambda ref: ref.stem` (update
    `_KEY_EXCEPTIONS["train_receiver_model"] = "ref.stem"`);
  - `declare` for `extract`/`pool` = the legacy `token_inputs` (`:465-471`) with `provider` = the stage's
    provider param; `cross_commit_version="receiver-rows-2"`; `counters=_coverage_counters`;
    `max_failures=None` (faithful: the legacy `for_each` recorded a failed match and `reconcile` fitted
    the rest, `:459-481`; the stage summary now RECORDS `n_failed`, and `assemble` writes it into
    `label_coverage` so a thinned corpus is visible in `metrics.json` rather than silent);
  - `fit` reads `ctx.upstream("extract")` and `item_counters(ctx, "extract")` for `label_coverage`;
    the model is saved to a temp dir and returned as FileSet entries `model/<file>`;
  - `deploy` loads the public bundle (`ReceiverModel.load(public_bundle)`) and the owner model from
    `materialize_fileset(ctx, "fit", "only", tmp)`; its token declares `"commit"` as the legacy one did
    (`:422`) — the code identity is already in the token (P9), so `declare` keeps `schema`, `provider`, `columns`;
  - `assemble` writes `metrics.json` with `json.dumps(manifest, indent=2) + "\n"` (legacy `:672` bytes) and
    copies the `model/` files to the bundle root (`model.json`, `SHA256SUMS`; read the committed
    `silly_kicks/tracking/_receiver_weights/default/` file set and match it, including `MODEL_CARD.md` if
    the legacy run wrote one there — it does not; the wheel card is committed separately).

- [ ] **M4–M7.** In `test_visibility_blind_gate.py` delete `_I1_PENDING` and its skip: the receiver
closure must now be clean. Registry: `pipeline="scripts.train_receiver_model"`, `base_args=("--provider",
"statsbomb", "--feature-set", "public")` (unchanged); the controller adds `--dataset`, `--corpus-visibility`,
`--corpus-n-matches` (Task 8). M5: equality with M1 including `corpus_visibility`/`corpus_n_matches`.

---

## Task 12: Migrate `ghost_outfield`

Spec: §5.6 row `ghost_outfield` (no whole-corpus cache stage; extraction NOT shared across the two
variants — its token keys on `feature_set`, evidence E11). Procedure M1–M7.

**Files:**
- Modify: `scripts/train_ghost_outfield.py` (`:143-201` argparse and `:353-536` main become stage functions; `:38-140`, `:204-350` helpers unchanged)
- Create: `tests/scripts/test_parity_ghost_outfield.py`
- Delete: `tests/scripts/test_train_ghost_outfield_mapreduce.py`
- Modify: `tests/scripts/test_train_ghost_outfield.py` (its CLI smoke uses the stage CLI; `_write_tiny_corpus` delegates to `_toy_corpora.build_flat_corpus`, Task 7)

**Pipeline:**

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `extract` | ItemStage | staged flat units | `_work` (`:410-419`): `extract_match(frames, _actions_for(…), fs, n_rearguard, home_team_id=_home_team_for(…), subsample_fps)` | concat of non-empty shards in sorted key order (`:440-444`) |
| `cv` | Stage | `int_key(k)` for `k < len(_cv_splits(data, cv_folds)[0])` (static: `min(cv_folds, n_games)` — read `n_games` from the `extract` summary) | `_run_one_fold_of(data, providers, tr, te, …)` (`:309`) | `_aggregate_folds(results in fold order)` (`:334`) or the `None` tuple when CV is infeasible |
| `fit` | Stage | `["only"]` | `:488-517`: final fit, provenance, save, round-trip, coherence | FileSet `model.npz`, `metadata.json`, `SHA256SUMS`, `coherence.json` |
| `assemble` | Stage | `["only"]` | `:519-536`: `metrics.json` | the bundle FileSet (output): `model.npz`, `metadata.json`, `SHA256SUMS`, `metrics.json` |

Parameters: `dataset`, `variant` (`default|position_only`), `n_estimators`, `max_depth`, `cv_folds`,
`n_rearguard`, `subsample_fps`, `training_platform`. The legacy `--actions-dir`/`--home-teams`/`--data-dir`
become the dataset's `_actions/`, `home_teams.json` and `shards/<gen>/` (Task 9 `staged_flat_refs`).

- [ ] **M1** — legacy reference: `python -m scripts.train_ghost_outfield --data-dir <toy>/shards/<gen> --actions-dir <toy>/_actions --home-teams <toy>/home_teams.json --output-dir <o> --variant default --n-estimators 10 --max-depth 3 --cv-folds 4 --allow-dirty`
(toy corpus with ≥ 6 games so 4-fold CV is feasible).
- [ ] **M2** — parity test, ARGS `["--dataset", "toyof", "--variant", "default", "--n-estimators", "10", "--max-depth", "3", "--cv-folds", "4"]`
(4 folds, so each of the 2 fanned jobs runs 2 folds on 2 workers: the W=2 path is exercised on `cv`);
out-of-order stages `extract`, `cv`. M1 uses `--cv-folds 4` too.
- [ ] **M3** — key: the legacy `_key` (`:404-408`) gives the stem for the flat layout: `key=lambda ref: ref.stem`,
add `_KEY_EXCEPTIONS["train_ghost_outfield"] = "ref.stem"`. `declare=lambda p: extraction_inputs(p["variant"], p["subsample_fps"])`
(`:43-59`, unchanged), `cross_commit_version="ghost-outfield-extract-1"`, `max_failures=None` (legacy WARNs
and continues, `:430-431`). The `cv` units read the `extract` output once per worker; cache it in a
module-level `functools.lru_cache(maxsize=1)` keyed by the `extract` token so a worker running several
folds loads the corpus once. `require_training_sklearn()` runs at the start of `cv`, `fit` (`:359`).
`metrics.json` bytes: `json.dump(metrics, f, indent=2)` with `newline="\n"` (`:534-535`) — reproduce exactly
(`json.dumps(metrics, indent=2).encode()`; `json.dump` writes no trailing newline).
- [ ] **M4–M7.** Registry: `pipeline="scripts.train_ghost_outfield"`; `fresh_subdir` → `""` (export writes
the bundle files directly).

---

## Task 13: Migrate `ghost_gk` (five variants incl. Hub-only `full`; shared extraction; importance at W > 1)

Spec: §5.6 row `ghost_gk`; decisions P10, P11; evidence E11, E12. Procedure M1–M7.

**Files:**
- Modify: `scripts/train_ghost_gk.py` (`:217-340` argparse and `:373-1232` main become stage functions; `:39-214`, `:343-370` helpers unchanged)
- Create: `tests/scripts/test_parity_ghost_gk.py`, `tests/scripts/test_ghost_importance_parallel.py`
- Delete: `tests/tracking/test_train_ghost_gk_mapreduce.py`
- Modify: `tests/tracking/test_train_ghost_gk_cli.py` (`_tiny_cache` delegates to `_toy_corpora`; CLI calls use the stage CLI)

**Pipeline:**

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `validate` | Stage | `["only"]` | `:472-519`: provider fail-fast (`validate_corpus_providers`, `validate_corpus_visibility`) + schema check | `{stem: provider}` (JSON) |
| `extract` | ItemStage | staged flat units | `_work` (`:698-763`) per game in the unit's file, packed by `_pack` | default summary |
| `cache` | Stage | `["only"]` | concat of `extract` shards in sorted key order (`:807-812`) | the combined DataFrame (parquet) — downstream calls `_unpack` (`:106-127`) |
| `cv` | Stage | `int_key(k)`, `k < cv_folds` | `run_one_fold` (`:904-954`) on the (sub-sampled) corpus and the seeded split (`:865-897`) | fold dicts in fold order |
| `fit` | Stage | `["only"]` | `:993-1027` final fit + BLOCKING parity gate; `:1082-1110` save + round-trip | FileSet `rfcde_weights.npz`, `metadata.json`, `SHA256SUMS` |
| `importance` | Stage, `inner_parallel=True`, `inner_width=8` | `["only"]` | `:1029-1079` with `n_jobs=ctx.pool.workers` (decision P11) | `{"skipped": bool, "top": [[name, value], …]}` |
| `assemble` | Stage | `["only"]` | `:1113-1226` metrics (fold aggregation, bias block, acceptance) | the bundle FileSet (output): `rfcde_weights.npz`, `metadata.json`, `SHA256SUMS`, `metrics.json` |

Variant sharing (E11): `extract`/`cache` declare only the extraction inputs (`build_extraction_inputs`,
`:182-200`), so `default` and `full` share them; `cv`/`fit`/`importance` declare `feature_set`,
`grid_x_max`, `n_estimators`, `max_depth`, `cv_folds`, carrier params, `subsample_cap`, `keeper_grouped`
— NOT `variant` — so `default` and `full` also share CV and the final fit; only `assemble` (the `variant`
label in `metrics.json`) differs. Registry builds (decision P10): `default` passes `--variant default`;
new `full` build passes `--variant full`.

Parameters: `dataset`, `variant`, `feature_set`, `grid_x_max`, `subsample_fps`, `n_estimators`,
`max_depth`, `cv_folds`, `carrier_beta`, `carrier_gamma`, `carrier_tolerance`, `subsample_cap`,
`training_platform`, `skip_permutation_importance` (flag), `perm_importance_sample`, `keeper_grouped`
(flag), `expansion_keepers` (path, read from the dataset root when relative), `with_actions` (flag,
default True: every staged flat dataset carries `_actions/`; it keys the extraction token exactly as
`bool(actions_by_game)` did, `:607`).

- [ ] **M1** — legacy reference: `python -m scripts.train_ghost_gk --data-dir <toy>/shards/<gen> --actions-dir <toy>/_actions --home-teams <toy>/home_teams.json --output-dir <o> --variant default --n-estimators 10 --max-depth 3 --cv-folds 4 --perm-importance-sample 200 --allow-dirty`
(sklearn ≥ 1.9: `.venv-hfjobs` only; the trainer refuses otherwise; toy corpus ≥ 6 games across 2 providers).
- [ ] **M2** — `tests/scripts/test_parity_ghost_gk.py`: ARGS as M1 (stage-CLI spelling, `--dataset toygk`);
out-of-order stages `extract`, `cv` (4 folds: each fanned job runs 2 folds at W=2).
`tests/scripts/test_ghost_importance_parallel.py` (decision P11):

```python
"""P11: importance at W=2 is byte-identical to W=1 and really ran on >= 2 worker processes."""

import json
import os

import pytest

from scripts._train_guard import sklearn_supports_training
from scripts._toy_corpora import build_toy_for
from tests.scripts._parity import cli, plan

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not sklearn_supports_training(), reason="needs sklearn>=1.9")]
MODULE = "scripts.train_ghost_gk"
ARGS = ["--dataset", "toygk", "--variant", "default", "--n-estimators", "10", "--max-depth", "3", "--cv-folds", "2", "--perm-importance-sample", "200"]


def test_importance_w1_equals_w2_and_uses_two_processes(tmp_path):
    data = build_toy_for(MODULE, tmp_path / "data")
    log = tmp_path / "pids.log"
    env = {**os.environ, "SILLY_KICKS_IMPORTANCE_PID_LOG": str(log)}
    for name, inner in (("a", "1"), ("b", "2")):
        cli(MODULE, tmp_path / name, data, ARGS, "--run-all", "--inner-workers", inner, env=env if inner == "2" else None)
    gen = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "a", data, ARGS)["stages"]}["importance"]
    a = (tmp_path / "a" / gen / "only.json").read_bytes()
    b = (tmp_path / "b" / gen / "only.json").read_bytes()
    assert a == b
    assert len(set(log.read_text().split())) >= 2
```

The PID log is a test hook: in `_SklearnWrapper.predict` (`:1044-1046`) append `os.getpid()` to the file
named by `$SILLY_KICKS_IMPORTANCE_PID_LOG` when that variable is set (one line, `open(path, "a")`); the
variable is never set outside this test.
- [ ] **M3** — key `ref.stem` (`_KEY_EXCEPTIONS["train_ghost_gk"] = "ref.stem"`; the legacy key
`(provider, game_id, home)` was per game inside a file, and every staged flat file holds ONE game — the
`home` component's purpose, invalidating a corrected home mapping, is now served by the dataset identity
in the token). `extract` work loads the unit's frames file and its `_actions/<stem>.parquet`, groups the
actions by `game_id`, and runs the legacy per-game `_work` body for each `game_id` in the file (sorted),
concatenating the packs; the bias counters (`:715-752`) are summed over the file's games and returned by
`counters`. `cross_commit_version="ghost-gk-extract-1"`, `declare=build_extraction_inputs(...)`,
`max_failures=0` (legacy raised, `:792-796`). `cv`, `fit`, `importance` call `require_training_sklearn()`
first (`:388-390`) and re-derive `features, labels, groups, provider_labels, keepers` via `_unpack` +
the subsample cap (`:846-855`) from the `cache` output (one function, `_corpus(ctx)`, `lru_cache`d by the
`cache` token). The seeded split (`:860-897`) is one function `_folds(ctx)` used by `cv` only.
`assemble` reads the bias totals with `item_counters(ctx, "extract")`, so a resumed run still reports the
whole corpus (`n_games_counters_unrecorded` = the missing-sidecar count). `metrics.json` bytes:
`json.dump(metrics, f, indent=2)` (`:1224-1225`).
- [ ] **M4–M7.** Registry: `pipeline="scripts.train_ghost_gk"`; builds per decision P10; `fresh_subdir` → `""`.

---
## Task 14: Fold-parallel objective, R1 decision, migrate `xshot_occurrence`

Spec: §5.6 row `xshot_occurrence`; §5.2 (`inner_parallel`, mechanism 3 inside a unit); §5.3 R1 rule;
ruling 4 (fold-parallel inside each seeded-TPE trial; numerics unchanged); evidence E9.

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence_objective.py` (split `_cv_logloss`, `fold_map` kwarg)
- Modify: `scripts/train_xshot_occurrence.py` (`_hpo_once` `fold_map`; `_cv_metrics` split; `_paired_data_effect` `params_for`; `:474-680` study functions and `:682-913` main become stages)
- Modify: `scripts/hf_jobs/arms.py` (add `parse_arms(text) -> ArmSpec`; `load_arms` delegates to it)
- Create: `tests/tracking/test_objective_fold_split.py`, `tests/scripts/test_study_resume_r1.py`, `tests/scripts/test_parity_xshot_occurrence.py`
- Delete: `tests/tracking/test_xshot_xcross_shard_index.py`; the xshot halves of `tests/scripts/test_expect_variant_wiring.py` that drive `enumerate_studies`/`run_one_study`/`assemble_studies` move to the new stage functions (read the file; keep every assertion about shipped-variant behaviour, re-pointed at `_select_work`/`_assemble_work`)

**Interfaces:**
- Library (numerics unchanged): `cv_fold_plan(y, groups) -> list[tuple[int, np.ndarray, np.ndarray]]`;
  `eval_fold(X, y, fold_i, tr, te, params, *, negative_subsample=None, subsample_seed=42) -> tuple[float, float | None, float] | None`;
  `combine_folds(results) -> tuple[float, float, float]`; `XShotOccurrenceObjective(..., fold_map=None)` where
  `fold_map(params: dict) -> dict[str, tuple | None]` (keys `int_key(fold_i)`).
- Trainer: `_study_fold_task(corpus_dir, rows_path, plan_path, fold_i, params, ns, seed)` (module-level,
  picklable); `STUDY_RESUME: bool` (set by the R1 step).

**Pipeline:**

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `extract` | ItemStage | staged flat units (`--dataset`) | legacy `_extract._work` (`:99-118`) on `(prov, pergame_name, shots, frames, home)` built from the flat unit exactly as `repackage` + `_iter_matches_from_dir` built it | concat in legacy order: `reduce_order` = the per-game dir name (`:151`, sorted game dirs) |
| `prep` | Stage | `["only"]` | `:844-906`: X/y/groups/providers/match_ids, arms from `--arms-json`, masks, `run_paired`, D21 `objective_inputs`, `persist_study_inputs` | FileSet of the study-input dir; reduce output = the config JSON + `study_tags` + `cv_arms` |
| `study` | Stage, `inner_parallel=True`, `inner_width=5` | `study_tags` (`"single"`, or `{arm}_f{k}` + every arm name when paired) | `_hpo_once(..., fold_map=…)` on the tag's rows (`:489-519` row logic) | `{tag: {"params", "n_trials"}}` |
| `cv` | Stage | `(arm, int_key(k))` for each cv arm and its folds | one fold of `_cv_metrics` at the arm's final params | `{arm: _cv_metrics dict}` |
| `select` | Stage | `["only"]` | `:551-629` ship decision (paired deltas with `params_for`, or `ship_order[-1]`) | `{"shipped", "why", "candidates"}` |
| `fit` | Stage | `["only"]` | `:631-658` gates (fail-closed), subsample, fit, save, reload check | FileSet `model.json`, `metadata.json`, `SHA256SUMS` |
| `assemble` | Stage | `["only"]` | `:659-678` metrics | the bundle FileSet (output) |

- [ ] **Step 1 (library, test first): fold split is byte-identical.** `tests/tracking/test_objective_fold_split.py`:

```python
"""Ruling 4: the fold-parallel path computes the SAME numbers as the serial objective."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from silly_kicks.tracking import _xshot_occurrence_objective as O  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    y = (X["a"] + rng.normal(scale=0.5, size=400) > 1.2).astype(int).to_numpy()
    groups = np.repeat(np.arange(20), 20).astype(str)
    return X, y, groups


P = {"n_estimators": 20.0, "max_depth": 3.0, "learning_rate": 0.1, "min_child_weight": 1.0, "reg_lambda": 1.0}


def test_split_equals_the_monolith():
    X, y, g = _data()
    want = O._cv_logloss(X, y, g, P)
    plan = O.cv_fold_plan(y, g)
    got = O.combine_folds([O.eval_fold(X, y, i, tr, te, P) for i, tr, te in plan])
    assert got == want  # exact, not approx


def test_fold_map_objective_equals_the_serial_objective():
    from ruthless.result import Candidate

    X, y, g = _data(1)
    plan = O.cv_fold_plan(y, g)

    def fold_map(params):
        # deliberately REVERSED completion order; the objective must combine by sorted key
        return {f"{i:04d}": O.eval_fold(X, y, i, tr, te, params) for i, tr, te in reversed(plan)}

    serial = O.XShotOccurrenceObjective(fold={"t": [(X, pd.Series(y), g)]})
    mapped = O.XShotOccurrenceObjective(fold={"t": [(X, pd.Series(y), g)]}, fold_map=fold_map)
    c = Candidate(params=P)
    assert mapped.evaluate_patch(mapped.prepare(), c) == serial.evaluate_patch(serial.prepare(), c)
```

(Construct `Candidate` with the signature the installed `ruthless` exposes; read
`ruthless.result.Candidate` first.) Run → FAIL. Then implement in `_xshot_occurrence_objective.py`:

```python
def cv_fold_plan(y, groups) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The seeded StratifiedGroupKFold split `_cv_logloss` uses -- single-sourced so a fold-parallel
    evaluation and the serial one walk the SAME folds."""
    groups = np.asarray(groups).astype(str)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        n_splits = 2
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = np.asarray(y)
    return [(i, tr, te) for i, (tr, te) in enumerate(gkf.split(np.zeros((len(y), 1)), y, groups))]


def eval_fold(X, y, fold_i, tr, te, params, *, negative_subsample=None, subsample_seed=42):
    """One fold of `_cv_logloss`: ``(log_loss, pr_auc | None, brier)`` or None for a skipped fold."""
    import xgboost as xgb

    if len(np.unique(y[tr])) < 2:
        return None
    Xtr, ytr = X.iloc[tr], y[tr]
    if negative_subsample:
        Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=subsample_seed + fold_i)
        if len(np.unique(ytr)) < 2:
            return None
    clf = xgb.XGBClassifier(**_pinned_params(params))
    clf.fit(Xtr.to_numpy(dtype=float), ytr)
    p = clf.predict_proba(X.iloc[te].to_numpy(dtype=float))[:, 1]
    pr = average_precision_score(y[te], p) if len(np.unique(y[te])) == 2 else None
    return (log_loss(y[te], p, labels=[0, 1]), pr, brier_score_loss(y[te], p))


def combine_folds(results) -> tuple[float, float, float]:
    """Means over the evaluated folds, in FOLD order (the caller passes them in fold order)."""
    kept = [r for r in results if r is not None]
    if not kept:
        return float("inf"), float("nan"), float("nan")
    prs = [r[1] for r in kept if r[1] is not None]
    return (
        float(np.mean([r[0] for r in kept])),
        float(np.mean(prs)) if prs else float("nan"),
        float(np.mean([r[2] for r in kept])),
    )
```

`_cv_logloss` becomes `return combine_folds([eval_fold(X, y, i, tr, te, params, negative_subsample=…, subsample_seed=…) for i, tr, te in cv_fold_plan(y, groups)])`
(keep its docstring). `StratifiedGroupKFold.split` reads `X` only for its length, so the zeros array is
equivalent (the test proves it). `XShotOccurrenceObjective.__init__` gains `fold_map=None`;
`evaluate_patch` uses it when set:
`res = self._fold_map(self._params(candidate)); ll, pr, br = combine_folds([res[k] for k in sorted(res)])`.
`evaluate` (the independent recompute, H1) stays serial. Run → PASS. Run the existing objective tests
(`git grep -ln "_xshot_occurrence_objective" tests/`) → PASS.

- [ ] **Step 2 (R1, spec §5.3): decide mid-study resume by test.** `tests/scripts/test_study_resume_r1.py`
runs `_hpo_once` on a 300-row toy in three SUBPROCESSES (fresh sampler state each time): A = 6 trials
uninterrupted in store `a/`; B1 = 3 trials in store `b/`; B2 = 6 trials in the SAME store `b/`. It
records `params_A`, `params_B`. Run it once and read the result:
  - `params_A == params_B` → resume is faithful: set `STUDY_RESUME = True` in
    `scripts/train_xshot_occurrence.py` (the study store lives in the stage generation on the store mount,
    as the 4.138.0 jobs kept it in `/scratch`), and the test asserts equality;
  - otherwise → set `STUDY_RESUME = False` (the store lives in a unit-local temp dir; an interrupted study
    unit restarts from trial 0 and its ledger record shows `attempt > 1`), and the test asserts
    INEQUALITY, so a future `ruthless` that makes resume faithful turns the test red and prompts the switch.
  Record the observed outcome and both param dicts in the task notes and in ADR-128 (Task 22).

- [ ] **Step 3: M1 legacy reference.** Toy flat corpus with shots (`build_flat_corpus(..., with_shots=True)`,
≥ 8 games, 2 providers); arms JSON from `arm_policy.compute_arms(stems)`; legacy run on the repackaged
per-game tree: `python -m scripts.hf_jobs.repackage <toy> <pg>` then
`python -m scripts.train_xshot_occurrence --data-dir <pg> --arms <arms.json> --output-dir <o> --n-trials 4 --allow-dirty`.
Record `xshot_occurrence_v1/{model.json,metadata.json,SHA256SUMS,metrics.json}`. Do it twice: once with an
all-public arms file (non-paired, `single`) and once with a three-arm file (`public`/`sc_extended`/`full`,
paired) so both select paths have a reference.

- [ ] **Step 4: M2 parity tests** (`test_parity_xshot_occurrence.py`, both arm shapes, `--n-trials 4`):
out-of-order stages `extract` (and `study` when paired: 15 study units across 2 jobs × 1 top-level
worker do not run concurrently inside a job — the inner pool is the parallel path there); inner
out-of-order stage `study` (5 folds at `--inner-workers 2`).

- [ ] **Step 5: M3 refactor** (the stage functions):
  - `extract` refs: `staged_flat_refs(ctx.data_root, dataset)`; ref tuple for the legacy key:
    `_PergameRef = NamedTuple("_PergameRef", [("prov", str), ("mid", str), ("stem", str), ("files", tuple)])`
    with `prov = ref.source_provider`, `mid = pergame_name(ref.provider, ref.game)`; `key=_source_key`
    (unchanged; `_KEY_EXCEPTIONS` unchanged); `load` reads the frames shard and `_actions/<stem>.parquet`,
    builds `shots = repackage._shots_frame(actions, _shot_type_ids(), _id_to_name())`, and
    `home = frames["team_id"].dropna().iloc[0]` (legacy `:64`); `work` = the legacy `_work` body moved to a
    module-level `_extract_work(item, ctx)`. `declare` = the legacy token inputs (`:134-142`);
    `cross_commit_version="xshot-extract-1"`; `max_failures=0` (legacy raised, `:146-147`).
  - `prep`: `arms.parse_arms(ctx.params["arms_json"])`; `objective_inputs = {"driver": "train_xshot_occurrence", "args": {k: v for k, v in ctx.params.items() if k not in {"dataset", "n_trials"}}, "match_ids": sorted(map(str, match_ids.tolist()))}`;
    `config` keeps every legacy key (`:880-893`) with `study_db_dir` removed (the study stage owns its
    store) and adds `"study_tags"` and `"cv_arms": {arm: n_splits}`.
  - `study`: rows mask per tag as `run_one_study` (`:500-505`) for `{arm}_f{k}`, `arms[arm]` for an arm
    name, all rows for `single`; write `rows.npy` and the `cv_fold_plan` (`npz`, `tr_<i>`/`te_<i>`) to a
    unit-local temp dir; `fold_map = None if ctx.pool is None or ctx.pool.workers == 1 else (lambda params: ctx.pool.map_keyed(_study_fold_task, {int_key(i): (corpus_dir, rows_path, plan_path, i, params, ns, seed) for i, _, _ in plan}))`;
    `_hpo_once(..., study_shard_dir=None, fold_map=fold_map)` with the store dir per `STUDY_RESUME`.
    `_study_fold_task` caches `(X.iloc[rows].reset_index(drop=True), y[rows].astype(int))` per
    `(corpus_dir, rows_path)` and the plan per `plan_path` in module-level dicts (one entry each; a
    spawned child serves one study).
  - `cv`: `_cv_metrics` splits into `_cv_metrics_plan(y, groups)`, `_cv_metrics_fold(X, y, i, tr, te, params, ns, seed)`,
    `_cv_metrics_combine(fold_results, y)`; `_cv_metrics` becomes their composition (same test-first rule
    as Step 1: add an exact-equality test to `test_objective_fold_split.py`).
  - `select`: `_paired_data_effect` gains `params_for: Callable[[str], dict] | None = None`; when given it
    replaces the `_hpo_once` call (`:416-431`) with `params_for(f"{name}_f{fold}")`; the stage passes
    `lambda tag: ctx.upstream("study")[tag]["params"]`.
  - `fit`: on a failed gate, write the legacy `metrics_FAILED.json` payload to
    `<generation>/only.metrics_FAILED.json` through the store, then raise `AcceptanceGatesFailedError`
    (the unit fails; the stage reports it).
  - `assemble`: `json.dumps(metrics, indent=2)` (legacy `:677` writes `json.dump(..., indent=2)`), and
    `artifact_size_bytes` = the summed sizes of the three `fit` files (`:675` counted exactly those).
  Delete `enumerate_studies`, `run_one_study`, `assemble_studies`, the argparse block and every phase
  flag. Keep `_fit_study_for_test` (a test seam). Find every test that calls a deleted name
  (`git grep -n "enumerate_studies\|run_one_study\|assemble_studies\|--prep-only\|--study-list" tests/`):
  re-point an assertion that still describes a live property (shipped-variant selection, study
  determinism on the mmap corpus) at the stage functions; delete a test whose only subject was the
  removed fan-out path (its property is now the parity test's), and list each deletion with its reason
  in the task notes for the reviewers.

- [ ] **Step 6: M4–M7.** Registry: `pipeline="scripts.train_xshot_occurrence"`, `fresh_subdir=""`,
`base_args=("--n-trials", "50")`; the controller adds `--dataset` and `--arms-json` (Task 8). M5 for both
arm shapes.

---

## Task 15: Migrate `xcross_attempt` (+ held-out `probe` stage; dotdir fix)

Spec: §5.6 row `xcross_attempt`. Same structure and steps as Task 14 (library twin
`silly_kicks/tracking/_xcross_attempt_objective.py` gets the same `cv_fold_plan`/`eval_fold`/`combine_folds`
split and `fold_map`, test first in `tests/tracking/test_objective_fold_split.py`). Differences:

**Files:** `scripts/train_xcross_attempt.py` (`:54-71` iterator dotdir skip; `:79-245` probe helpers;
`:519-813` study functions; `:816-1148` main); `tests/scripts/test_parity_xcross_attempt.py`;
`tests/tracking/test_xcross_attempt_integration.py` (CLI calls → stage CLI).

**Extra stage** `probe` (ItemStage over `staged_pergame_refs(ctx.data_root, ctx.params["probe_dataset"])`,
key `_source_key` on `(source_provider, stem)`):
- unit work = the probe-capture part of the legacy `_extract._work` for ONE match (`:120-160`): when the
  match's provider is in `--probe-providers` (default `gradientsports`), return a FileSet
  `frames.parquet`, `actions.parquet`, `meta.json` (the per-match provenance the cohort records: provider,
  match_id, home_team_id, match groups); otherwise return an empty FileSet;
- reduce = build the cohort from the FIRST `probe_keep` (2) captured units in sorted key order (legacy
  iteration order: sorted game dirs), then the legacy `_write_probe_sample` writes into a temp
  `_probe_sample/` dir whose files become the stage output FileSet; the legacy `res.skipped` bundle element
  (`:190-194`, the "cohort cannot be rebuilt from shards" guard) is 0 by construction — the captures are
  persisted per unit, so a resumed probe stage rebuilds the identical cohort;
- `assemble` needs `probe` and runs the legacy TF-19 block (`:741-800`) against
  `materialize_fileset(ctx, "probe", "_output", tmp / "_probe_sample")`.

**Dotdir fix** (the 4.138.0 `.cache` crash, spec §2.2 item 2): `_iter_matches_from_dir` (`:55`) skips
directories whose name starts with `.` — a one-line change with a test in
`tests/tracking/test_xcross_attempt_integration.py` (a `.cache/` beside the game dirs is ignored).
`staged_pergame_refs` already skips them (Task 9).

Parameters add `probe_dataset`, `probe_providers`, `probe_comparison_providers`, `home_teams` is the
dataset's `home_teams.json` (xcross USES home for `score_differential`, `:59-69`).

M1: legacy `--probe-data-dir <toy pergame probe>` run. M5 for both arm shapes. Registry
`pipeline="scripts.train_xcross_attempt"`, `probe=True`; the controller adds `--probe-dataset xcross_probe`.

---

## Task 16: `replicate_map` + migrate the census driver

Spec: §5.6 row `build_territory_ranking_census` (`bootstrap` stays ONE `inner_parallel` unit; single RNG
stream, evidence E14); decisions P4, P15, P17.

**Files:**
- Modify: `scripts/_crossed_icc.py` (`bootstrap_icc_and_power(..., replicate_map=None)`)
- Modify: `scripts/build_territory_ranking_census.py` (`:508-713` main → stages; helpers unchanged)
- Create: `tests/scripts/test_bootstrap_replicate_map.py`, `tests/scripts/test_parity_territory_census.py`
- Modify: `tests/scripts/test_build_territory_ranking_census.py` (CLI calls → stage CLI; `--list-matches` tests → `--plan` `units`)

- [ ] **Step 1 (test first): the parallel bootstrap is byte-identical.**

```python
"""E14: pre-drawing every replicate from the ONE stream, then fitting in any order, is byte-identical."""

import numpy as np

from scripts._crossed_icc import bootstrap_icc_and_power


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    d = rng.integers(0, 40, n)
    t = (d // 4 + rng.integers(0, 2, n)) % 12
    y = rng.normal(size=n) + 0.3 * (d % 5)
    return y, d.astype(str), t.astype(str)


def test_reversed_replicate_map_equals_serial():
    y, d, t = _toy()
    kw = dict(n_boot=60, alpha=0.05, rng_seed=7)
    serial = bootstrap_icc_and_power(y, d, t, **kw)

    def rmap(task):  # deliberately reversed completion; the function must combine by replicate index
        fn, items = task
        return {k: fn(*items[k]) for k in sorted(items, reverse=True)}

    assert bootstrap_icc_and_power(y, d, t, replicate_map=rmap, **kw) == serial
```

(Match the real keyword names of `bootstrap_icc_and_power` — read its signature, including any
`effect_size` argument the census passes, `build_territory_ranking_census.py:196-204`.)

Then restructure `bootstrap_icc_and_power`: draw `picks = [rng.integers(0, k, size=k) for _ in range(n_boot)]`
FIRST, in replicate order (the same draw sequence the loop made); move the loop body that turns one
`picked` into one ICC value (including its exception handling for a degenerate replicate) into a
module-level `_replicate_icc(y, defender_codes, team_codes, members, picked) -> float | None`; with
`replicate_map=None` the loop calls it in order; otherwise
`res = replicate_map((_replicate_icc, {int_key(i): (y, defender_codes, team_codes, members, picks[i]) for i in range(n_boot)}))`
and the combination iterates `res[int_key(i)]` for `i` in order. Everything after the loop is unchanged.
For the census, `replicate_map` is built from `ctx.pool.map_keyed`; to avoid pickling the arrays once
per replicate, the census passes chunks of 50 replicates per task and a task function that loads the
arrays from a unit-local `npz` by path (cached per child, the Task 14 pattern).

- [ ] **Step 2: M1** — legacy reference on the toy open-data tree (3 matches; `--fit-fraction 0.34`
leaves 2 scored matches): `SILLY_KICKS_OPEN_DATA_BASE=<url> python -m scripts.build_territory_ranking_census --out <o> --competition-id 43 --season-id 106 --fit-fraction 0.34 --allow-dirty`.
The toy census may not license a ranking; record `census.json` as is.

- [ ] **Step 3: M2** — parity test; out-of-order stage `score`; inner out-of-order stage `bootstrap`.

- [ ] **Step 4: M3** — stages:

| Stage | Kind | Units | Work (source lines) | Reduce / output |
|---|---|---|---|---|
| `refs` | Stage | `["only"]` | competition resolution (`:546-583`) + `open_data_source` refs (`:585-590`), sorted by match id, split by `fit_fraction` (`:591-595`) | `{"competitions", "fit_refs", "score_refs"}` |
| `fit_prepass` | ItemStage | `fit_refs` | `_prune_actions_slice(item.actions)` (`:620`); key `ref.key`; legacy token (`:622-627`) | default summary |
| `prefit` | Stage | `["only"]` | `_fit_disjoint_models` over the shards in sorted key order (`:630-631`) | FileSet `xt.json` (`ExpectedThreat.to_dict`) + the completion model's `save` files |
| `score` | ItemStage | `score_refs` | `_score_match(item, xt=…, completion_model=…)` with the models loaded once per worker from `prefit`; key `str(ref.match_id)`; legacy token (`:639-646`) | default summary |
| `bootstrap` | Stage, `inner_parallel=True`, `inner_width=8` | `["only"]` | concat score shards in sorted key order (`:666-669`), `tier1_counts`, `tier2_icc` with `replicate_map` | `{"tier1", "tier2", "defender_team_metric": …}` as FileSet (`census_core.json`, `defender_team_metric.parquet`) |
| `report` | Stage | `["only"]` | `:677-713`: verdict, ranking, `census.json` | the output FileSet `census.json` (+ `ranking.parquet`) |

`report` replaces `**res.manifest()` (`:703`) with the corpus-scoped `score` summary (decision P4) and
keeps `run_commit`/`run_tree_dirty`/`run_tree_state`/`input_contract`. `--list-matches` is removed
(decision P17): `--plan` lists every stage's `units`. Register the module in `DRIVERS` (Task 8) with
`certification="rebaseline"` and `output_rel="docs/research/territory_ranking_census"`.

- [ ] **Step 5: M4–M7.** The driver stays in `ARTIFACT_DRIVERS` (`tests/scripts/test_provenance_wiring.py`);
the stage CLI offers `--allow-dirty` and the module imports `scripts._provenance` (the two parametrized
checks there read the source: keep an explicit `from scripts._provenance import git_provenance` use in
`report` for the run stamp).

---

## Task 17: One-unit adapter stage for the three public trainers

Spec: §5.6 ("One-unit adapter stage (Cycle 1 to Cycle 2)"), §12 R7.

**Files:**
- Create: `scripts/hf_jobs/adapter.py`
- Test: `tests/scripts/hf_jobs/test_adapter.py`

**Interfaces:** `scripts.hf_jobs.adapter.PIPELINE` with params `target` (module path), `target_args`
(JSON list), `fresh_subdir` (where the target writes its artifact inside `--out`); one stage `run` with
unit `only`; CLI shape `python -m scripts.hf_jobs.adapter --target scripts.train_xsuccess --target-args '["--competition-id","43"]' --fresh-subdir weights --dataset sb_open --store … --stages run`.

- [ ] **Step 1: failing test** — run the adapter against a toy target module
(`tests/scripts/_toy_legacy_trainer.py`: an argparse CLI with `--out` and `--shard-root` that writes
`<out>/weights/model.json` deterministically from its args, and refuses a second run if a marker says it
was killed mid-way — use it to prove the target's own resume through the persistent `--out`): assert the
`run` output is a FileSet of `<out>/weights/*`, the target ran with the deterministic env, a re-run is a
no-op (the unit is done), and a killed first attempt resumes through the target's own shards.

- [ ] **Step 2: implement.** The unit's work:

```python
def _run_work(key, ctx) -> FileSet:
    """Run an unmigrated trainer's EXISTING CLI unchanged as ONE unit (spec 5.6). Its `--out` (and
    `--shard-root`) live on the persistent store, so the target's own for_each shards keep its resume."""
    import json
    import subprocess
    import sys

    from scripts._thread_pin import deterministic_env

    root = ctx.store.local_path(f"{ctx.generation('run')}/target_out")
    root.mkdir(parents=True, exist_ok=True)
    args = json.loads(ctx.params["target_args"])
    argv = [sys.executable, "-m", ctx.params["target"], *args, "--out", str(root)]
    if ctx.params["target"] == "scripts.train_match_outcome_dependence":
        argv += ["--shard-root", str(root / "shards")]  # its for_each defaults to ephemeral /tmp (launch.py:158-161)
    subprocess.run(argv, check=True, env=deterministic_env())  # noqa: S603 -- registry-pinned module + args
    art = root / ctx.params["fresh_subdir"] if ctx.params["fresh_subdir"] else root
    return FileSet({p.name: p.read_bytes() for p in sorted(art.iterdir()) if p.is_file()})
```

`run` declares `target`, `target_args`, `fresh_subdir` and `uses_dataset=True` (the open-data revision is
pinned through the job env, decision P6). The registry entries of the three trainers keep their legacy
`module`, `base_args` and builds; Task 8's controller turns them into adapter instances. The three modules
stay in `_LEGACY_UNMIGRATED` (Task 21) with reason "Cycle 2: real pipeline; runs via the one-unit adapter
stage (ADR-128 §5.6)".

---
## Task 18: Delete the legacy launch path; registry, floor, head-to-head and DGX e2e on the contract

Spec: §5.5 (Replaced), §5.6, §6.1, §6.2, §6.6, §12 R7, ruling 11.

**Files:**
- Modify: `scripts/hf_jobs/launch.py` — delete every legacy symbol (spec §6.2): `_out_dir`, `_fresh_dir`,
  `_corpus_steps`, `_trainer_cmd`, `_phase_flags`, `build_phase_command`, `build_inner_command`,
  `corpus_flat_stems`, `arms_payload`, `build_argv`, `subprocess_runner`, `launch_variant_job`,
  `launch_all_variants`, `_TERMINAL`, `_STAGE_RE`, `_terminal_stage`, `poll_until_terminal`,
  `retrieve_variant`, `_PHASE_SEQUENCES`, `StagePhaseError`, `phase_sequence`, `_parse_job_id`,
  `run_stage`, `wait_all`, `job_output`, `_studies_from_prep`, `run_pipeline`, `make_local_runner`; drop
  the now-unused imports (`base64`, `os`, `re`, `subprocess`, `time`, `TrainerSpec`, `VariantBuild`).
- Modify: `scripts/hf_jobs/registry.py` — remove `flavor`, `timeout` (planner-derived now), `output_flag`
  (the stage CLI has one `--out`), `source` and `SOURCES` (datasets.toml declares the shape); keep `name`,
  `module`, `extras`, `base_args`, `builds`, `primary_metric`, `metric_direction`, `delta`, `probe`,
  `visibility_gated`, `databricks_blocked` and the Task 8 fields. Update its module docstring: "how each
  trainer runs as a stage-contract pipeline and how it is certified".
- Modify: `scripts/hf_jobs/floor.py` — `certify(spec, tier, *, fresh_metrics, committed_metrics, headtohead=None) -> tuple[bool, dict]`
  (public: `regression_check` on the head-to-head pair; owner: CV-parity, `primary_metric` read from both
  `metrics.json` files with the registry `delta`; rebaseline: P3 identity comparison + P15 rule);
  `evaluate_spec` stays as the public-tier primitive.
- Modify: `scripts/hf_jobs/dgx_e2e.py` — rewrite `run_local_e2e(instance, *, data_root, store, rss_ceiling, measure)`
  over the stage CLI: plan → for each stage, run `python -m <module> … --stages S --workers W` under
  `measure_rss`, record per-stage peak RSS and wall, `blocked` when a stage exits non-zero or reaches the
  ceiling; keep `check_shipped_variant`, `measure_rss`, `CPU_UPGRADE_RAM_BYTES` unchanged.
- Modify: `tests/scripts/hf_jobs/test_registry.py`, `test_registry_flavor.py`, `test_floor.py`,
  `test_headtohead.py` (only where they read removed fields), `test_dgx_e2e.py` (rewritten against the new
  `run_local_e2e` with an injected `measure`), `test_dgx_e2e_shipped_variant.py` (unchanged if it only
  tests `check_shipped_variant`).
- Delete: `tests/scripts/hf_jobs/test_launch.py`, `tests/scripts/hf_jobs/test_launch_orchestrator.py`
  (their subject is deleted; the typed launcher is covered by `test_launch_typed.py`, the controller by
  `test_campaign.py`).

- [ ] **Step 1: Prove nothing else uses the legacy symbols.** Run
`git grep -n -E "run_pipeline|wait_all|StagePhaseError|phase_sequence|_phase_flags|build_phase_command|launch_variant_job|launch_all_variants|retrieve_variant|poll_until_terminal|build_argv|build_inner_command|arms_payload|corpus_flat_stems|make_local_runner|_studies_from_prep" -- scripts tests`.
Expected: hits only in the files this task modifies or deletes, plus historic docs (spec §6.2: specs,
plans and ADRs are records and are not edited). Any other hit is a STOP: report it.

- [ ] **Step 2: Rewrite the tests first** (`test_dgx_e2e.py`, `test_floor.py` additions for `certify`):

```python
def test_owner_certification_is_cv_parity():
    spec = reg.REGISTRY["ghost_gk"]
    ok, d = certify(spec, "owner", fresh_metrics={"cv_mae_euclidean_mean": 1.15}, committed_metrics={"cv_mae_euclidean_mean": 1.13})
    assert ok and d["basis"] == "cv_parity"
    bad, _ = certify(spec, "owner", fresh_metrics={"cv_mae_euclidean_mean": 1.30}, committed_metrics={"cv_mae_euclidean_mean": 1.13})
    assert not bad


def test_public_certification_needs_the_headtohead_pair():
    spec = reg.REGISTRY["win_probability"]
    with pytest.raises(KeyError):
        certify(spec, "public", fresh_metrics={}, committed_metrics={}, headtohead={})
```

The owner metric names come from each trainer's own `metrics.json` (ghost: `cv_mae_euclidean_mean`;
outfield: `cv_mae`; receiver: `top1_cv`; xshot/xcross sc_extended: `candidates[<shipped>].metrics.pr_auc`).
Add `cv_metric: str` to `TrainerSpec` (a dotted path into `metrics.json`) and set it per entry.

- [ ] **Step 3: Delete and rewrite; run.** Run:
`python -m pytest tests/scripts/hf_jobs/ -q -p no:randomly` and
`.venv-hfjobs/Scripts/python.exe -m pytest tests/scripts/hf_jobs/ -q -p no:randomly`. Expected: PASS.
Run the I1 gate (`test_visibility_blind_gate.py`): its population now includes every migrated pipeline
and the census; PASS with no pending entry.

- [ ] **Step 4: End-to-end plan over the real campaign (no spend).**
`.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/cycle1.toml plan`
Expected: every one of the 21 instances (ghost_gk 5, ghost_outfield 2, xshot 2 + 2, xcross 2 + 2,
receiver 1, win_probability 1, census 1, three adapters 1 each) prints a first action (`profile` for an
un-profiled multi-unit stage, `run` for a single-unit stage). Record the output in the task notes.

---

## Task 19: Artifact manifest, observed fields, constant-test rewrites, generated card regions

Spec: §5.7 (artifact manifest, corpus expectation, generated card regions), §6.3, INV-9, decisions P2, P7.

**Files:**
- Create: `docs/artifacts/manifest.toml`, `docs/artifacts/observed.json` (empty object `{}` until Task 20's bootstrap)
- Create: `scripts/hf_jobs/manifest.py`
- Create: `tests/test_artifact_manifest.py`, `tests/test_card_regions.py`
- Modify (rewrite to read the manifest; assertions unchanged): `tests/test_bundled_weights_corpus_policy.py`,
  `tests/test_bundled_weights_classification.py`, `tests/tracking/test_position_only_bundled.py`,
  `tests/test_model_cards_match_bundles.py`
- Modify: `scripts/_hub_publish.py` (`CARD_SOURCE` derived; `card_bytes` strips region markers, P7),
  `scripts/validate_hub_variants.py` (`HUB_REGISTRY` derived), `scripts/publish_model_card.py` (reads
  `CARD_SOURCE` as before — no change needed if it imports the name)
- Modify: the 9 cards in `docs/huggingface/model-cards/` and `silly_kicks/tracking/_receiver_weights/default/MODEL_CARD.md`
  (insert inline region markers around the generated text; NO other text change)

**Interfaces:**
- Produces: `manifest.RefitEvent(id, version, adr, title, basis, cite)`; `manifest.Artifact(id, destination, job, variant, tier, certification, dataset, naming, corpus, training_commit, commit_field, refit_event, card, wheel_sibling, mirror_of, hub_class, hub_role, expect)`
  with `.kind` (`wheel`|`hub`), `.path` (wheel dir), `.repo_id` (hub); `load() -> Manifest` with
  `.events`, `.artifacts`, `.roots`; `weights_classification() -> dict[str, tuple[str, str]]`;
  `card_sources() -> dict[str, str]`; `hub_registry() -> dict[str, tuple[str, str]]`;
  `violations(artifact, read=None) -> list[str]`; `render_card(path, manifest) -> str`;
  `stale_regions(path, manifest) -> list[str]`; `REGION`; `strip_markers(text) -> str`.

- [ ] **Step 1: Write the manifest.** `docs/artifacts/manifest.toml` — every value copied from the
CURRENT test constants and committed metadata (they were reviewed expectations; a value no artifact
carries is a defect, the corpus-policy test's own rule). Header comment: the two field classes (INV-9),
"human-written; no tool writes this file", "observed fields live in observed.json (written only by
`adopt`)". Content outline:

```toml
# Re-fit events (cards render these; the version/ADR a card states comes from HERE).
[events.f1b]
version = "4.128.0"
adr = "ADR-106"
title = "F1b float32-frame re-fit"
basis = "on float32-stored frames"
cite = ""

[events.tf65]
version = "4.134.0"
adr = "ADR-120"
title = "TF-65 velocity-gap re-fit"
basis = "on the corrected floor velocities"
cite = "TF-65 / ADR-120, "

[events.adr126]
version = "4.138.0"
adr = "ADR-126"
title = "ADR-126 xShot type_id label re-fit"
basis = "on the corrected type_id shot label"
cite = "ADR-126, "

# Weights roots (replaces WEIGHTS_CLASSIFICATION; reasons copied verbatim).
[roots."silly_kicks/tracking/_xshot_weights"]
classification = "frame_geometry"
reason = "xShot features from frame positions"
# … the other nine roots from tests/test_bundled_weights_classification.py:21-35 …

[[artifacts]]
id = "xshot_default"
destination = "wheel:silly_kicks/tracking/_xshot_weights/default"
job = "xshot_occurrence"
variant = "default"
tier = "public"
certification = "public"
dataset = "xshot17"
naming = "pergame"
corpus = "public_ids"  # metrics.json corpus_match_ids == datasets.lock.json[xshot17].public_ids
training_commit = "4095251730ce216decf99ec8e47d8e1162a04a1d"
commit_field = "metadata.json:training_commit"
refit_event = "adr126"
[artifacts.expect]
"metadata.json:shipped_variant" = "public"
"metadata.json:provider_list" = ["idsse", "skillcorner"]
"metrics.json:shipped_variant" = "public"
"metrics.json:reproducibility" = "public"
```

Write one `[[artifacts]]` table per artifact, 27 in all (17 wheel + 10 Hub):
- 11 tracking wheel dirs: `xshot_{default,position_only}` (commit `_ADR126`, event `adr126`),
  `xcross_default` (`_TF65`, `tf65`), `xcross_position_only` (`_M`, `f1b`), `ghost_default` (`_TF65`),
  `ghost_position_only` (`_M`, plus `"metrics.json:reproducibility" = "restricted"`), `ghost_sweeper`
  (`_TF65`), `ghost_sweeper_position_only` (`_REUSED`), `outfield_default` (`_TF65`),
  `outfield_position_only` (`_REUSED`), `receiver_default` (`_M`, `commit_field = "metrics.json:run_commit"`,
  expect `providers_trained` / `corpus_visibility`) — every commit, field and expect value from
  `tests/test_bundled_weights_corpus_policy.py:15-87` (`_x_policy`, `_ghost_policy`, `_gof_policy`);
- 6 event-only/other wheel dirs (classification + training commit where the file carries one;
  `certification` from the registry): `win_probability`, `match_outcome`, `expected_passing`, `xsuccess`,
  `retention_default`, `retention_skillcorner` (the last two `job = "gk_retention"`, `certification = "blocked"`);
- 10 Hub repos from `HUB_REGISTRY` (`validate_hub_variants.py:17-28`): `hub_class` and `hub_role`
  copied; `card` from `CARD_SOURCE`; the 5 `hf_only` cards' `wheel_sibling` from `_HF_ONLY`
  (`test_model_cards_match_bundles.py:50-56`); the 4 mirrors' `mirror_of` from `_MIRRORS`
  (`:43-48`); the 4 xshot/xcross Hub repos are tier `sc_extended`, `dataset = "xshot115"`,
  `corpus = "digest"`, `naming = "pergame"`. Do NOT infer any Hub artifact's commit or event: for every
  Hub repo, read `training_commit` from the published file (`HfApi().hf_hub_download(repo, "metadata.json")`,
  read-only) and set `refit_event` to the event declared for that commit. A Hub commit that matches no
  declared event (planning expects `4095251…` for the 4.138.0 xshot/xcross sc_extended and ghost `full`
  re-fits, which is `adr126`'s commit, but xcross's re-fit was the velocity fix, not the label fix) gets
  its OWN `[events.*]` entry whose `title`/`basis`/`version`/`adr` come from `CHANGELOG.md` [4.138.0] —
  list each such entry in the task notes for the reviewers;
- the Hub repo `silly-kicks/ghost-gk-v1` is artifact `ghost_full` (job `ghost_gk`, variant `full`, tier
  `owner`, dataset `tc3`, `hub_role = "hf_only"`, `wheel_sibling = "ghost_default"`).

- [ ] **Step 2: Write the failing tests.** `tests/test_artifact_manifest.py`:

```python
"""INV-9: artifacts are checked against EXPECTATIONS declared before a run (manifest + dataset lock),
never against values derived from the artifact itself."""

import json
from pathlib import Path

import pytest

from scripts.hf_jobs import manifest as M
from scripts.hf_jobs.datasets import declared, locked

MAN = M.load()


def test_wheel_population_equals_the_discovered_dirs():
    found = {p.as_posix() for p in Path("silly_kicks/tracking").glob("_*_weights/*") if p.is_dir() and not p.name.startswith((".", "_")) and p.name != "full"}
    found |= {p.as_posix() for p in Path("silly_kicks").glob("*/weights") if p.is_dir()}
    found |= {p.as_posix() for p in Path("silly_kicks/xtgk/_retention_weights").glob("*") if p.is_dir()}
    assert found == {a.path for a in MAN.artifacts.values() if a.kind == "wheel"}


def test_hub_population_equals_the_card_set():
    cards = {p.name.removesuffix("-model-card.md") for p in Path("docs/huggingface/model-cards").glob("*-model-card.md")}
    hub_cards = {Path(a.card).name.removesuffix("-model-card.md") for a in MAN.artifacts.values() if a.kind == "hub" and a.card.startswith("docs/")}
    assert cards == hub_cards


@pytest.mark.parametrize("aid", sorted(a.id for a in MAN.artifacts.values() if a.kind == "wheel" and a.commit_field))
def test_wheel_artifact_matches_its_expectations(aid):
    assert not M.violations(MAN.artifacts[aid])


def test_every_event_is_used_and_every_reference_resolves():
    used = {a.refit_event for a in MAN.artifacts.values() if a.refit_event}
    assert used == set(MAN.events)
    for a in MAN.artifacts.values():
        assert a.dataset is None or a.dataset in declared(), a.id
        assert a.wheel_sibling is None or a.wheel_sibling in MAN.artifacts, a.id
        assert a.mirror_of is None or a.mirror_of in MAN.artifacts, a.id


def test_the_checker_catches_a_wrong_corpus():
    """Anti-rot: an artifact whose corpus ids differ from the staging record is reported."""
    a = MAN.artifacts["xshot_default"]
    fake = {"metrics.json": {"shipped_variant": "public", "reproducibility": "public", "corpus_match_ids": [["idsse", "X"]]},
            "metadata.json": {"shipped_variant": "public", "provider_list": ["idsse", "skillcorner"], "training_commit": a.training_commit}}
    assert any("corpus" in v for v in M.violations(a, read=fake.__getitem__))


def test_observed_fields_never_feed_an_expectation():
    """INV-9: no expectation key may name an observed field."""
    observed_keys = {"sha256", "metrics_summary", "env", "campaign", "adopted_at"}
    for a in MAN.artifacts.values():
        assert not (set(a.expect) & observed_keys)
```

`tests/test_card_regions.py`:

```python
"""Generated card regions equal their re-render (spec 5.7); markers never reach the Hub (P7)."""

from pathlib import Path

import pytest

from scripts._hub_publish import card_bytes
from scripts.hf_jobs import manifest as M

MAN = M.load()
CARDS = sorted({a.card for a in MAN.artifacts.values() if a.card} | {"silly_kicks/tracking/_receiver_weights/default/MODEL_CARD.md"})


@pytest.mark.parametrize("card", CARDS)
def test_regions_are_fresh(card):
    assert not M.stale_regions(Path(card), MAN)


@pytest.mark.parametrize("card", CARDS)
def test_every_card_has_its_required_regions(card):
    names = {m.group(1) for m in M.REGION.finditer(Path(card).read_text(encoding="utf-8"))}
    assert names >= M.required_regions(card, MAN), card


def test_card_bytes_strip_markers(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("A <!-- generated:hub_note -->B<!-- /generated:hub_note --> C\n", encoding="utf-8")
    assert card_bytes(p) == b"A B C\n"


def test_a_stale_region_is_detected(tmp_path):
    card = Path(CARDS[0])
    text = card.read_text(encoding="utf-8")
    m = next(M.REGION.finditer(text))
    poisoned = text[: m.start(2)] + "STALE" + text[m.end(2) :]
    p = tmp_path / card.name
    p.write_text(poisoned, encoding="utf-8")
    assert M.stale_regions(p, MAN, card_id=str(card))
```

- [ ] **Step 3: Implement `scripts/hf_jobs/manifest.py`.** Parts:
  1. Loader: `load()` reads `docs/artifacts/manifest.toml` via `_toml.load_toml`; `Artifact.kind` from the
     `destination` prefix; `path` = the text after `wheel:`; `repo_id` = after `hub:`.
  2. `weights_classification()` = `{root: (r["classification"], r["reason"]) for root, r in roots}`.
  3. `card_sources()` = `{a.repo_id: a.card for a in hub artifacts}`; `hub_registry()` =
     `{a.repo_id: (a.hub_class, a.hub_role) for a in hub artifacts}`.
  4. `violations(a, read=None)`: `read(fname) -> dict` defaults to the wheel file; checks every
     `expect` entry (dotted key after `:`), `commit_field` == `training_commit`, and the corpus:
     `corpus == "public_ids"` → `metrics.json:corpus_match_ids == locked(dataset)["public_ids"]`;
     `corpus == "digest"` → `corpus_match_ids_sha256`/`corpus_n_matches` == `expected_identity(dataset, naming)`;
     messages contain the word `corpus` for corpus mismatches.
  5. Renderers, keyed by region name: `refit_head` (mirror cards: `f"**{e.title} (silly-kicks {e.version} / {e.adr}; `training_commit={short}`).**"`
     with `short` = the MIRRORED wheel artifact's training commit[:7] — reproduces `_refit_head`,
     `test_model_cards_match_bundles.py:76-83`), `hub_note` (Hub-only cards: `f"In silly-kicks {e.version} the wheel's bundled `{name}` was re-fit {e.basis} ({e.cite}`training_commit={short}`)."`
     with the SIBLING's event — reproduces `_hub_note`, `:86-111`), `hub_refit` (Hub-only cards: one
     sentence stating the Hub artifact's OWN event, `f"This Hub artifact was last re-fit in silly-kicks {e.version} ({e.adr}, `training_commit={short}`)."`),
     `provenance` (the receiver wheel card: `_provenance_line`, `:126-131`, with `version` from the
     artifact's event), and `row:<label>` (mirror-card table value cells: the formats in `_GK_ROWS` /
     `_OUTFIELD_ROWS`, `:25-41`, applied to the mirrored wheel `metrics.json`).
  6. `REGION = re.compile(r"<!-- generated:([a-z0-9_:-]+) -->(.*?)<!-- /generated:\1 -->", re.S)`;
     `render_card`, `stale_regions(path, man, card_id=None) -> list[str]` (names whose content differs),
     `required_regions(card, man) -> set[str]` (mirror: `refit_head` + one `row:` per label; Hub-only:
     `hub_note`, `hub_refit`; the receiver wheel card: `provenance`), `strip_markers(text)` (removes the
     inline marker comments only).

- [ ] **Step 4: Insert the markers.** For each card, wrap the EXISTING text of each region in inline
markers (`<!-- generated:hub_note -->…<!-- /generated:hub_note -->`), changing no character inside. For a
`row:<label>` region, wrap the table row's VALUE cell content. If a Hub-only card has no sentence that
states its own re-fit event, add the `hub_refit` sentence (region-wrapped) after the `hub_note` sentence:
this changes the card's rendered text, so record each such card in the task notes — its Hub README is then
re-published with `publish_card_only` under an owner go in Task 27. Run `tests/test_card_regions.py` until
`stale_regions` is empty for every card.

- [ ] **Step 5: Rewrite the constant-pinning tests on the manifest.**
  - `tests/test_bundled_weights_corpus_policy.py`: `POLICY`/`_ANCHOR`/`_x_policy`/`_ghost_policy`/
    `_gof_policy` and the commit constants are replaced by `M.load()`; keep every test function and
    assertion (`test_every_variant_dir_is_registered_exactly` compares the discovered dirs with the
    manifest's tracking wheel artifacts; `test_dir_matches_its_corpus_policy` calls `M.violations`;
    `test_dir_traces_to_its_adr_anchor` reads `commit_field`/`training_commit`; the anti-rot
    `test_the_checker_catches_a_wrong_variant`, `test_underivable_is_empty`,
    `test_recorded_runs_were_clean`, `test_ghost_position_only_reproducibility_note_names_its_own_commit`
    unchanged in substance).
  - `tests/test_bundled_weights_classification.py`: `WEIGHTS_CLASSIFICATION = M.weights_classification()`
    (the name stays: `test_bundled_weights_corpus_policy` imports it); assertions unchanged.
  - `tests/tracking/test_position_only_bundled.py`: `_C1…_C7` are replaced by manifest lookups
    (`M.load().artifacts["ghost_default"].training_commit`, …); the per-test comments keep the history
    (`_C2`/`_C3` were superseded commits named only in prose — keep them in the comments, not as constants).
  - `tests/test_model_cards_match_bundles.py`: `_PROVENANCE_VERSION`, `_TF65_*`, `_ADR126_*`,
    `_MIRRORS`, `_HF_ONLY`, `_refit_head`, `_hub_note`, `_provenance_line` are replaced by the manifest and
    `M.render_*`; keep `test_card_population_is_exact`, the row-level checks
    (`test_mirror_card_table_row_equals_the_bundle` with `_row_value`) and
    `test_a_stale_table_cell_fails_even_when_the_right_number_is_elsewhere` (anti-vacuity).
  - `scripts/_hub_publish.py`: `CARD_SOURCE = manifest.card_sources()` (computed at import; the
    `xsuccess-v1` entry comes from the manifest's `card = "silly_kicks/xsuccess/weights/MODEL_CARD.md"`);
    `card_bytes` returns `strip_markers(text).encode` with LF endings (decision P7).
  - `scripts/validate_hub_variants.py`: `HUB_REGISTRY = manifest.hub_registry()`.

- [ ] **Step 6: Run the FULL tree** (this task touches cards and pinned-constant tests;
memory: a path subset missed 12 contract tests in 4.138.0):
`python -m pytest tests/ -m "not e2e and not slow" -p no:randomly --benchmark-skip --continue-on-collection-errors -q`
Expected: PASS. Capture every `FAILED` line if not.

---
## Task 20: Artifact identity, `retrieve`/`verify`/`adopt`, `publish` post-verification, sandbox

Spec: §5.7 (`adopt` 10 steps, ADR-037 guard, `publish`), §5.5 (`TrainerSpec.certification`), §7 criteria
3 and 5, INV-9, decisions P3, P15.

**Files:**
- Create: `scripts/hf_jobs/identity.py`, `scripts/hf_jobs/adopt.py`
- Modify: `scripts/hf_jobs/campaign.py` (`retrieve`, `verify`, `adopt`, `publish` commands delegate here)
- Modify: `scripts/train_xshot_occurrence.py`, `scripts/train_xcross_attempt.py` (extract `counters` for the ADR-126 integrity inputs)
- Test: `tests/scripts/hf_jobs/test_identity.py`, `tests/scripts/hf_jobs/test_adopt.py`, `tests/scripts/hf_jobs/test_publish_verify.py`

**Interfaces:**
- `identity.PROVENANCE_KEYS = ("training_commit", "training_platform", "run_commit", "run_tree_dirty", "run_tree_state", "reproducibility_note", "generation")`;
  `artifact_identity(dir) -> {"files": {name: sha256}, "normalized": {name: sha256}}` (`normalized` for
  `metadata.json`/`metrics.json` only: canonical JSON without `PROVENANCE_KEYS`, recursively);
  `compare(a, b, *, same_commit: bool) -> list[str]` (same commit: every file sha must match; across
  commits: every non-JSON payload sha + the normalized JSON shas must match).
- `adopt.retrieve(campaign, commit, *, staging, api) -> dict[instance_id, Path]`;
  `adopt.verify(campaign, commit, *, staging, run=subprocess.run) -> dict` (certification per
  `TrainerSpec.certification`/tier; ADR-126 integrity for xshot/xcross);
  `adopt.adopt(campaign, commit, *, staging, now=None) -> dict` (steps 2–10; never commits);
  `adopt.bootstrap_observed(*, api) -> dict` (observed fields for the artifacts that exist today);
  `adopt.publish(artifact_ids, *, sandbox: bool, api, staging) -> dict` (owner go every time).

- [ ] **Step 1: Integrity inputs travel with the run.** In Tasks 14/15's `extract` stages add
`counters=_integrity_counters` returning, per match, the quantities the ADR-126 verdicts were computed
from (read `docs/research/trained_artifact_integrity/verdicts/*.json` first: their `n_events`/`fps`
inputs define the quantity — events = shots (xshot) or crosses (xcross) in the match's labelled domain,
frames and duration for `fps`). `adopt.verify` reads them with `item_counters(ctx, "extract")`; they never
enter a token or an artifact.

- [ ] **Step 2: Failing tests.** `tests/scripts/hf_jobs/test_identity.py`:

```python
import json

from scripts.hf_jobs.identity import PROVENANCE_KEYS, artifact_identity, compare


def _write(d, meta_commit, weights=b"W"):
    d.mkdir(parents=True)
    (d / "model.json").write_bytes(weights)
    (d / "metadata.json").write_text(json.dumps({"training_commit": meta_commit, "feature_names": ["a"]}))
    (d / "metrics.json").write_text(json.dumps({"run_commit": meta_commit, "pr_auc": 0.08}))


def test_cross_commit_compare_ignores_only_provenance(tmp_path):
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c2")
    assert compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False) == []
    assert compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=True)


def test_a_weight_or_metric_change_is_always_reported(tmp_path):
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c2", weights=b"X")
    assert any("model.json" in d for d in compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False))


def test_provenance_keys_are_exactly_the_declared_set():
    assert "sklearn_version" not in PROVENANCE_KEYS and "training_commit" in PROVENANCE_KEYS
```

`tests/scripts/hf_jobs/test_adopt.py` (fakes only; every side effect injected):
- refuses a dirty tree (`require_clean_tree`) and a tree whose HEAD differs from the campaign commit;
- refuses an artifact whose `training_commit` differs from the campaign commit (INV-9 step 4);
- refuses an artifact whose recorded dataset revision differs from `datasets.lock.json`;
- refuses a corpus digest/n that differs from `expected_identity` (and a public id list that differs
  from `public_ids`);
- refuses a shipped tier (`shipped_variant`) that differs from the manifest `tier`;
- on success: writes ONLY files under the wheel destinations, `docs/artifacts/observed.json`, the cards,
  `docs/research/trained_artifact_integrity/verdicts/*`, `docs/research/hf_campaigns/*`, and a CHANGELOG
  draft beside the repo (`<staging>/CHANGELOG-draft.md`), proven by comparing `git status --porcelain`
  before/after against an allowlist (the ADR-037 diff-scope guard);
- a public wheel bundle goes through `rehydrate_public_corpus_ids` after verification (its
  `corpus_match_ids` equal the lock's `public_ids`); an owner bundle keeps the digest;
- is idempotent (a second run changes nothing).

`tests/scripts/hf_jobs/test_publish_verify.py`:
- `publish(..., sandbox=True)` pre-creates `silly-kicks/sandbox-<name>` with `private=True` BEFORE calling
  `publish_model_with_card` (whose own `create_repo(exist_ok=True)` then finds the private repo);
- post-verification fails on a nested path, on a remote `SHA256SUMS` different from the local one, on a
  README different from `card_bytes(card)`, and on a failed `from_hub` load;
- `sandbox=False` refuses unless `--i-have-owner-approval <repo>` names every target repo (the CLI flag
  the owner's go maps to; it never defaults).

- [ ] **Step 3: Implement.**
  - `identity.py` as the interface says.
  - `retrieve`: for each instance whose output stage is reduced (read through `BucketStore`), write the
    output FileSet into `<staging>/<run_key>/<variant>/`; never into the repo.
  - `verify`: load each artifact (`SHA256SUMS`, chirality, feature contract) by running the model class's
    `load` in a subprocess under `.venv-hfjobs`; then certification: `public` → run
    `python -m scripts.hf_jobs.headtohead <trainer> --committed-dir <wheel dir> --fresh-dir <staging dir> --feature-set <fs> --metrics-json <staging>/headtohead.json`
    and `floor.certify(spec, "public", …)`; `owner` → `floor.certify(spec, "owner", fresh_metrics=…, committed_metrics=<wheel metrics.json or observed.json metrics_summary for a Hub artifact>)`;
    `rebaseline` (census) → `identity.compare(..., same_commit=False)` + decision P15 (a numeric difference
    is classified by `git log dd7ed4f..<commit> -- <input_contract() modules>`: empty log → environment
    delta, else code delta, which STOPS adoption of that artifact for an owner ruling); xshot/xcross →
    additionally `python -m scripts.validate_trained_artifact_integrity --family … --shipped … --fresh … --n-events … --fps … --horizon-s 1.0 --out <staging>/verdicts/<id>.json`
    with the Step 1 counters. Output: `<staging>/verify.json`.
  - `adopt`: spec §5.7 steps 1–10 in order; step 4 checks EXACTLY the five expectations (training
    commit, dataset revision, corpus digest + n or public ids, shipped tier); step 5 copies only the
    `MODEL_ONLY_ALLOWLIST` files (+ wheel `MODEL_CARD.md` where the wheel dir has one) into each wheel
    destination; step 6 writes `observed.json` entries `{sha256, metrics_summary, env, campaign, adopted_at}`;
    step 7 re-renders card regions (`manifest.render_card`); step 8 writes verdict JSONs with top-level
    `run_commit`/`run_tree_dirty`; step 9 writes `<staging>/CHANGELOG-draft.md`; step 10 runs the full
    `tests/` tree + scoped ruff + pyright and prints the report, then STOPS ("commit gate: owner approval
    required").
  - `bootstrap_observed`: hashes the wheel artifacts in the repo and downloads (read-only) each Hub
    repo's `metadata.json`/`metrics.json`/`SHA256SUMS`, verifies each against the manifest expectations
    (the same five checks), and writes the observed fields; run it once in this task so `observed.json`
    describes the artifacts that ship today (Task 19's tests do not read it; `publish` and `verify` do).
  - `publish`: for each Hub artifact id: target repo = `artifact.repo_id`, or `f"silly-kicks/sandbox-{name}"`
    with `--sandbox` (`name` = the repo name after `silly-kicks/`); sandbox → `api.create_repo(repo, repo_type="model", private=True, exist_ok=True)`
    first; then the UNCHANGED `_hub_publish.publish_model_with_card(api, <staging dir>, repo, model_card=artifact.card)`;
    then post-verify (model-only file set, remote `SHA256SUMS` == local, README == `card_bytes(card)`,
    `<Class>.from_hub(repo)` round-trip in a subprocess); results to `<staging>/publish.json` and the bucket
    `_campaign/<name>/<commit7>/publish.json`.

- [ ] **Step 4: Run.** `python -m pytest tests/scripts/hf_jobs/test_identity.py tests/scripts/hf_jobs/test_adopt.py tests/scripts/hf_jobs/test_publish_verify.py -v -p no:randomly`
→ PASS. Then run `bootstrap_observed` for real (read-only on the Hub) and commit nothing; the resulting
`docs/artifacts/observed.json` is part of the code commit's tree (it records the shipped state the
Cycle 1 campaign compares against).

---

## Task 21: Born-ready compute gate and scaffolder

Spec: §5.8 items 1–3, INV-8, §12 R8.

**Files:**
- Create: `tests/scripts/test_compute_contract.py`, `scripts/hf_jobs/scaffold.py`, `tests/scripts/hf_jobs/test_scaffold.py`
- Modify: `scripts/hf_jobs/registry.py`, `scripts/_toy_corpora.py` (insertion markers for the scaffolder)

- [ ] **Step 1: Write the gate** `tests/scripts/test_compute_contract.py`:

```python
"""Born-ready compute gate (ADR-128, ADR-056 derive-and-assert). Every train_*.py and every corpus walker
(a module that calls for_each or declares an ItemStage) is a compute-contract pipeline, or is on the legacy
list with an explicit approval, or is named in _NOT_A_PIPELINE with a reason."""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

import tests.scripts._corpus_load_rules as clr

_SEED = "ADR-128 landing seed"
_RULING = re.compile(r"^owner ruling \d{4}-\d{2}-\d{2}: .+")

#: Members that are legitimately not compute pipelines (the ADR-056 _NOT_A_DRIVER pattern). Adding one is
#: a reviewed decision.
_NOT_A_PIPELINE: dict[str, str] = {
    "_loader_pining_to_cache": "the pining CACHE WARMER: it downloads raw artifacts into a local cache and computes nothing a stage could own; it runs only where the pining token lives (DGX), never on HF.",
    "materialize_tc3_frames": "the DGX STAGING tool: it materialises the neutral flat corpus that `datasets record` then measures; it is the input side of every staged dataset, not a compute pipeline.",
    "_xt_corpus": "a LIBRARY helper (the xT corpus pass) called by drivers; its pipeline is its caller's.",
}

#: Non-compliant members on the day the gate lands. Every entry carries a reason and an approval; a later
#: addition must cite a dated owner ruling ("owner ruling YYYY-MM-DD: ..."). The PR diff is the review
#: point for growth (no in-repo test can see history).
_LEGACY_UNMIGRATED: dict[str, dict[str, str]] = {
    **{
        n: {"reason": "Cycle 3+ driver burn-down (ADR-128 section 11)", "approval": _SEED}
        for n in (
            "build_gkdv_arm_values", "build_layer2_spells", "build_rq_pass_scores", "build_sb360_coverage",
            "build_skillcorner_s1_event_validity", "build_tf19_instrument_responsiveness",
            "build_tf56_positioning_validity", "build_tf60_layer3_arm_values", "calibrate_coordination",
            "calibrate_tracking_defaults", "calibrate_xt_bandwidth", "check_stage1_argmax",
            "derive_coordination_params", "derive_opengoal_range", "evolve_xsuccess_features",
            "measure_ball_contact_validation", "measure_box_constant_delta",
            "measure_cover_shadow_argmax_agreement", "measure_f1b_feature_delta", "measure_gs_shot_distribution",
            "measure_rc4_orientation", "run_signoff_power", "tune_structural_pass_sigma",
            "validate_coordination_numerics", "validate_das_native_parity", "validate_gk_decision",
            "validate_match_outcome_calibration", "validate_sb360_licensed_corpus", "validate_shot_goalmouth_sb",
            "validate_team_coordination", "validate_team_kpi_reliability", "validate_territorial_defense",
            "validate_territory_counterfactual", "validate_xcross_causal", "validate_xs_probe", "validate_xshot_causal",
        )
    },
    **{
        n: {"reason": "Cycle 2: real pipeline; runs via the one-unit adapter stage (ADR-128 section 5.6)", "approval": _SEED}
        for n in ("train_match_outcome_dependence", "train_pass_completion", "train_xsuccess")
    },
    "train_gk_retention": {"reason": "blocked: no non-Databricks data path (registry BLOCKED)", "approval": _SEED},
}


def _calls(tree) -> set[str]:
    return {clr._call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}


def _population() -> dict[str, ast.AST]:
    return {
        name: tree
        for name, tree in clr.all_script_trees().items()
        if name.startswith("train_") or ({"for_each", "ItemStage"} & _calls(tree))
    }


def _declares_pipeline(tree) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PIPELINE" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and clr._call_name(n.value) == "Pipeline"
        for n in tree.body
    )


def _compliant() -> list[str]:
    return sorted(set(_population()) - set(_LEGACY_UNMIGRATED) - set(_NOT_A_PIPELINE))


def test_lists_are_subsets_of_the_derived_population_and_disjoint():
    pop = set(_population())
    assert set(_LEGACY_UNMIGRATED) <= pop, sorted(set(_LEGACY_UNMIGRATED) - pop)
    assert set(_NOT_A_PIPELINE) <= pop, sorted(set(_NOT_A_PIPELINE) - pop)
    assert not set(_LEGACY_UNMIGRATED) & set(_NOT_A_PIPELINE)


@pytest.mark.parametrize("name", sorted(_LEGACY_UNMIGRATED))
def test_every_legacy_entry_carries_reason_and_approval(name):
    e = _LEGACY_UNMIGRATED[name]
    assert set(e) == {"reason", "approval"} and e["reason"].strip()
    assert e["approval"] == _SEED or _RULING.match(e["approval"]), e


@pytest.mark.parametrize("name", sorted(_NOT_A_PIPELINE))
def test_not_a_pipeline_names_an_existing_file_with_a_reason(name):
    assert pathlib.Path(f"scripts/{name}.py").is_file() and len(_NOT_A_PIPELINE[name]) > 40


@pytest.mark.parametrize("name", _compliant())
def test_member_declares_a_valid_pipeline(name):
    assert _declares_pipeline(clr.all_script_trees()[name]), f"{name}.py must declare PIPELINE = Pipeline(...) (ADR-128)"
    mod = importlib.import_module(f"scripts.{name}")
    mod.PIPELINE.validate()
    assert mod.PIPELINE.module == f"scripts.{name}"


@pytest.mark.parametrize("name", _compliant())
def test_member_plans_on_its_toy_input(name, tmp_path):
    from scripts._stages import LocalStore, make_context, plan
    from scripts._toy_corpora import TOY_PARAMS, build_toy_for

    mod = importlib.import_module(f"scripts.{name}")
    assert mod.PIPELINE.module in TOY_PARAMS, f"{name}: register a toy input in scripts/_toy_corpora.TOY_PARAMS"
    data = build_toy_for(mod.PIPELINE.module, tmp_path / "data")
    from scripts._stages import build_parser

    ns = build_parser(mod.PIPELINE).parse_args([*TOY_PARAMS[mod.PIPELINE.module], "--plan", "--store", str(tmp_path / "s")])
    params = {p.name: getattr(ns, p.name) for p in mod.PIPELINE.params}
    ctx = make_context(mod.PIPELINE, params, {"commit": "t", "tree_state": "clean", "code_identity": "t"}, store=LocalStore(tmp_path / "s"), data_root=data)
    out = plan(ctx)
    assert [s["name"] for s in out["stages"]] == [s.name for s in mod.PIPELINE.stages]


@pytest.mark.parametrize("name", _compliant())
def test_member_registers_a_parity_fixture_and_a_job_entry(name):
    from scripts.hf_jobs.registry import DRIVERS, REGISTRY

    mod = importlib.import_module(f"scripts.{name}")
    tests = list(pathlib.Path("tests/scripts").glob("test_parity_*.py"))
    assert any(f'"{mod.PIPELINE.module}"' in t.read_text(encoding="utf-8") for t in tests), f"{name}: no W1N1-vs-W2N2 parity test"
    assert any(s.pipeline == mod.PIPELINE.module for s in REGISTRY.values()) or any(d.module == mod.PIPELINE.module for d in DRIVERS.values())


def test_the_gate_FLAGS_a_member_without_PIPELINE():
    """Red fixture (spec 5.8): a toy corpus walker without PIPELINE is a member and is NOT compliant."""
    tree = ast.parse("from scripts._driver import for_each\n\ndef main():\n    for_each([], key=str, work=print, shard_root='x', token_inputs={'a': 1})\n")
    assert {"for_each"} & _calls(tree)
    assert not _declares_pipeline(tree)


def test_the_compliant_set_is_the_cycle1_migration():
    assert set(_compliant()) == {
        "train_ghost_gk", "train_ghost_outfield", "train_xshot_occurrence", "train_xcross_attempt",
        "train_receiver_model", "train_win_probability", "build_territory_ranking_census",
    }
```

(Use `tests/scripts/_corpus_load_rules.py`'s real helper names — `all_script_trees`, `_call_name` — as
they exist; read the module first.) If TF-46 merged into `main` before this lands (spec §12 R8), its
drivers join `_LEGACY_UNMIGRATED` with `_SEED` and `test_the_compliant_set_is_the_cycle1_migration` is
unchanged. If the derived population at implementation time differs from this list in any other way,
STOP and report the difference (it is a fact this plan did not see).

- [ ] **Step 2: Scaffolder (test first).** `tests/scripts/hf_jobs/test_scaffold.py` runs
`scaffold.main(["trainer", "toy_born_ready", "--root", <tmp copy of scripts/ + tests/scripts/_parity.py + docs/artifacts/manifest.toml>])`
and asserts: `scripts/train_toy_born_ready.py` declares a valid `PIPELINE` (gate predicates
`_declares_pipeline` + `validate`), `tests/scripts/test_parity_toy_born_ready.py` exists and names the
module, the registry file gained a `TrainerSpec` entry between the scaffold markers, `TOY_PARAMS` gained
the module, `manifest.toml` gained a commented `[[artifacts]]` stub (trainer only), and running the
generated parity test passes. Same for `driver` (`scripts/build_toy_born_ready.py`, a `DriverSpec`, no
manifest stub).

`scaffold.py` writes from string templates (one per kind): the module template is a two-stage pipeline —
an `ItemStage` `extract` over `staged_flat_refs` that returns one row per match and a `fit` (trainer) or
`report` (driver) single-unit stage that returns a `FileSet` — with a module docstring that points at
`docs/howto/compute-pipelines.md`. Registry and `TOY_PARAMS` insertion is text insertion before the
markers `# --- scaffold: trainers above ---`, `# --- scaffold: drivers above ---`,
`# --- scaffold: toys above ---` (add the markers in this task). It refuses an existing name.

- [ ] **Step 3: Run.** `python -m pytest tests/scripts/test_compute_contract.py tests/scripts/hf_jobs/test_scaffold.py -v -p no:randomly`
in `.venv-hfjobs` (the toy plans import the trainers) → PASS.

---
## Task 22: Skills, howtos, AGENTS.md, ADR-128, supersede markers, C4, CHANGELOG, TODO

Spec: §5.8 items 5–8, ruling 7 (no MCP), §12 R6, decision P8.

**Files:**
- Create: `.claude/skills/compute-pipelines/SKILL.md`, `.claude/skills/hf-campaign/SKILL.md`,
  `docs/howto/compute-pipelines.md`, `docs/superpowers/adrs/ADR-128-compute-contract-hf-campaigns.md`
- Modify: `.claude/skills/authoring-a-metric/SKILL.md`, `.claude/skills/corpus-drivers/SKILL.md`,
  `.claude/commands/new-metric.md`, `docs/howto/hf-jobs-training.md` (rewrite),
  `docs/howto/corpus-drivers-runbook.md`, `AGENTS.md`, `docs/context/tracking-features.md` (only if a
  moved WHY is missing there), `docs/superpowers/adrs/ADR-122-hf-jobs-training-harness.md`,
  `docs/superpowers/adrs/ADR-123-hf-jobs-parallel-harness.md`, `docs/c4/architecture.dsl`,
  `docs/c4/architecture.html`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: Skills (short shims; substance lives in the howtos).**
`.claude/skills/compute-pipelines/SKILL.md`:

```markdown
---
name: compute-pipelines
description: Use when writing or migrating a silly-kicks trainer (train_*.py) or a corpus driver (build_/validate_/measure_* that walks matches) — the ADR-128 stage/unit contract (PIPELINE, Stage, ItemStage), single-threaded units, sorted-key reduces, ordered_values for inner pools, checksum resume, the W1N1-vs-W2N2 parity test, the profile stage, the scaffolder, or the born-ready compute gate.
---

# Compute pipelines

Read `docs/howto/compute-pipelines.md` and follow it. It is the canonical runbook; this shim owns no
substance. Start a new module with `python -m scripts.hf_jobs scaffold trainer|driver <name>`; it passes
`tests/scripts/test_compute_contract.py` on day one.
```

`.claude/skills/hf-campaign/SKILL.md`: same shape; description "Use when running, resuming, inspecting
or adopting an HF Jobs campaign — `python -m scripts.hf_jobs campaign <file> …`, preflight, profile,
launch/resume, status, retrieve, verify, adopt, publish (--sandbox), the run ledger, datasets record —
always on cpu-upgrade, with owner gates on every commit, push and Hub publish."; body points at
`docs/howto/hf-jobs-training.md` and repeats the three owner gates (commit, push, publish) as one line
each. `authoring-a-metric/SKILL.md`, `corpus-drivers/SKILL.md` and `.claude/commands/new-metric.md` gain
ONE sentence each: "A trained model or a corpus pass is a compute pipeline: follow the `compute-pipelines`
skill (ADR-128)."

- [ ] **Step 2: Howtos.**
  - `docs/howto/compute-pipelines.md` (new): sections "When a module is a pipeline" (the gate's population
    rule), "Units" (one key, one result, single-threaded; key components zero-padded with `int_key`; keys
    never start with `_`), "Stages" (`Stage` vs `ItemStage`; `units`, `work`, `reduce`, `reduce_order`,
    `params`/`declare`, `cross_commit_version`, `max_failures`/`abort_first_k`, `inner_parallel`/`inner_width`,
    `uses_dataset`), "Determinism rules" (the four mechanisms of `_stages.py`'s docstring; `ordered_values`
    is the only way to combine pool results; no wall-clock, no unseeded RNG, no hash-order iteration),
    "Resilience" (what resumes, checksum sidecars, the failure rule), "Data" (`_driver_sources`, datasets
    declared in `datasets.toml` + recorded in the lock; never read pining in a module the campaign launches —
    the I1 gate), "Parity test" (the `_parity.py` shape; non-vacuity), "Sizing" (`profile`, the planner,
    RAM-bounded W, the irreducible floor), "Scaffold and register" (scaffolder, `TOY_PARAMS`, registry,
    manifest stub), "Gate" (`test_compute_contract.py`, `_LEGACY_UNMIGRATED` approvals). Link ADR-128, the spec.
  - `docs/howto/hf-jobs-training.md` (rewrite; keep the title): prerequisites (`.venv-hfjobs` build
    command, HF token, `gh` auth), the staged-dataset contract (kinds, `datasets record`), the campaign
    file, the command sequence (`preflight` → `profile` → `launch`/`resume --watch` (run in the background) →
    `status` → `retrieve` → `verify` → `adopt` → owner gate → `publish --sandbox` → owner gate), the ledger
    and its committed summary, the `--replica` re-run for the INV-1 gate, cost (cpu-upgrade only, per-second
    billing, the planner's estimate), and "what the `hf` CLI is still for" (humans only). Remove every
    `--shard-index`/`--fold`/`--study`/`--assemble`/`--prep-only` instruction (spec §6.1).
  - `docs/howto/corpus-drivers-runbook.md`: a new first paragraph — a NEW driver starts from the
    scaffolder on the stage contract; the `for_each` instructions below remain the reference for the
    `_LEGACY_UNMIGRATED` drivers and for `ItemStage` semantics.
  Run `python -m pytest tests/test_howto_links_wired.py -q` → PASS.

- [ ] **Step 3: AGENTS.md (decision P8; spec R6).** Measure first:
`python -c "import pathlib; print(len(pathlib.Path('AGENTS.md').read_text(encoding='utf-8').encode()))"`
(expected 27 725 before edits). Then:
  1. Replace the "Vectorized spearman batch is byte-identical via concat-and-slice" bullet with:
     `- **Vectorized spearman batch is byte-identical via concat-and-slice** — single-sourced `_spearman_combine`; `compute_threat_pc_batch` scores EXPLICIT slices, never the frame-keyed cache (ADR-043 landmine). (ADR-105; `docs/context/tracking-features.md`)`
  2. Replace the "Tracking-frame coords are float32 STORAGE / float64 COMPUTE" bullet with:
     `- **Tracking-frame coords are float32 STORAGE / float64 COMPUTE; `team_id` is `category`, `player_id` is not** — kernels upcast at the boundary (`test_frame_coord_upcast_gate`); `id_compat._decat` unwraps (ADR-019). Frames only; SPADL stays float64. (ADR-106; `docs/context/tracking-features.md`)`
  3. Before step 1/2 edits, grep `docs/context/tracking-features.md` for every token the long versions
     carried (`compute_pitch_control_batch`, `per-frame row order`, `batch_size`, `estimate_das_cost`,
     `DasCostWarning`, `numba/ADR-076 bit-identity`, `ADR-103 rule`). Copy any missing sentence from the
     old bullet into that file's matching section, so no WHY is lost (the file is the class-1.5 home).
  4. Add after the "Every corpus driver adopts `scripts/_driver.py`" bullet:
     `- **Heavy compute is a stage-contract pipeline** — every trainer and corpus walker declares `PIPELINE` (`scripts/_stages.py`): keyed single-threaded units, sorted-key reduces, checksum resume, W workers × N `cpu-upgrade` jobs byte-identical for any (W, N); campaigns only from committed `scripts/hf_jobs/campaigns/*.toml`; gate `test_compute_contract`. (ADR-128; `docs/howto/compute-pipelines.md`)`
  Run `python -m pytest tests/test_agents_md_budget.py -q` → PASS (ceiling, 600-char bullets, pointers,
  inventory). If still over the ceiling, STOP and report the byte count (do not cut other bullets).

- [ ] **Step 4: ADR-128** from `docs/superpowers/adrs/ADR-TEMPLATE.md`: Context (spec §1–§2 evidence),
Decision (the contract; the executor refinement over a process pool; pinned environment incl. P1; the
typed launcher; preflight; the stateless controller, ledger, planner, profile; the artifact manifest and
INV-9 incl. P2; adopt/publish; born-ready governance; DGX as an executor), Consequences (the legacy
launch path is gone; three public trainers ride the adapter until Cycle 2; the census runs on the
contract; resume across commits only for declared extraction stages, P9), the R1 outcome from Task 14
with both param dicts, the engineering resolutions P1–P18 (P1 and P2 named as spec-text corrections), Alternatives (B incremental, C external engine — spec
§3 ruling 2). Re-confirm the number: `git fetch --all` then list `docs/superpowers/adrs/` on
`origin/main` and every remote branch; renumber everywhere (`git grep -n "ADR-128"`) if 128 is taken (it was renumbered from ADR-127 on 2026-10-18: TF-46 took ADR-127 on `main` @ `cd2ebcb`).
ADR-122 and ADR-123: change the Status row to `Accepted — superseded in part by ADR-128 (launch path,
per-trainer phase flags)` and add one blockquote line under the table naming what ADR-128 replaces.

- [ ] **Step 5: C4.** In `docs/c4/architecture.dsl` add `hfJobs = softwareSystem "HuggingFace Jobs" "cpu-upgrade compute for stage-contract campaigns: typed run_job API, bucket store mount, revision-pinned dataset mounts" "External"`,
and inside `sillyKicks` two containers (descriptions ≤ 200 characters, `tests/test_c4_dsl_description_cap.py`):
`stageContract = container "scripts/_stages + hf_jobs.executor" "ADR-128 stage/unit contract: PIPELINE DAGs of keyed single-threaded units, checksum-verified resume, W capped workers per job, byte-identical for any (W, N)." "Python" "Library"`
and `campaignCtl = container "scripts/hf_jobs campaign" "Stateless campaign controller: preflight, profile, planner, typed launcher, ledger, adopt, publish; drives HF Jobs from committed campaign files; artifact manifest." "Python" "Library"`,
with relationships `campaignCtl -> hfJobs "launches jobs, reads labels"`, `campaignCtl -> hfHub "reads datasets, publishes models (ADR-088)"`,
`hfJobs -> stageContract "runs --stages S --shard I/N"`, `stageContract -> calibration "runs trainers and drivers"`
(use the DSL's existing view and style conventions; read the file first). Regenerate
`docs/c4/architecture.html` with the `mad-scientist-skills:c4` skill's rendering workflow and the pinned
Graphviz `dot`. Run the C4 tests (`git ls-files "tests/test_c4_*.py"`) → PASS.

- [ ] **Step 6: CHANGELOG and TODO.** `CHANGELOG.md`: a new top section
`## [Unreleased] — compute contract + HF campaigns (ADR-128)` with Added / Changed / Removed / Fixed
lists (Fixed: the xcross probe `.cache` dotdir crash; the registry `ghost_gk` `default` `--variant`
(P10); the receiver's in-job pining read (P5)); the version replaces `[Unreleased]` at the commit gate
(spec §10: the version is decided there). `TODO.md`: add the two approved later cycles (spec §11, ruling
1) under On Deck — "ADR-128 Cycle 2: real pipelines for match_outcome_dependence, pass_completion,
xsuccess (delete the adapter stage when the trainer half of `_LEGACY_UNMIGRATED` is empty)" and "ADR-128
Cycle 3+: corpus-driver burn-down, TF-46 first, then by measured wall time"; keep the file's shape
(`tests/test_todo_md_format.py`).

---

## Task 23: Full local gate → freeze → independent review (G1) → the code commit (G2)

Spec: §8 (local gate), §10. Owner rule (2026-10-18): "minimal commits as needed for provenance, no
micro-commits, execution inline". This is the FIRST of the two commits ADR-037 provenance needs: the
campaign must run on a pushed, clean commit.

- [ ] **Step 1: Full local gate** (background; capture every `FAILED`):
```
python -m pytest tests/ -m "not e2e and not slow" -p no:randomly --benchmark-skip --continue-on-collection-errors -q
.venv-hfjobs/Scripts/python.exe -m pytest tests/ -m "slow and not e2e" -p no:randomly --benchmark-skip -q -k "parity or executor or stage_contract or study_resume or importance or bootstrap_replicate or scaffold or compute_contract"
uvx ruff@0.15.7 check silly_kicks/ tests/ scripts/
uvx ruff@0.15.7 format --check silly_kicks/ tests/ scripts/
pyright <every changed .py file>
python -m pytest tests/ -p no:randomly --co -q   # collection: no basename clash, no import error
```
Expected: all green. The known local-only failure `test_das_golden_fixture` (local scipy 1.15.3, proven
pre-existing on a clean stash in the 4.138.0 cycle) is reported, not fixed. Any other failure is fixed
before the gate is declared green, and the full run is repeated.

- [ ] **Step 2: Durations.** New test files land in existing directories, so
`tests/test_ci_durations_coverage_wired.py` stays green. List every new `@pytest.mark.slow` test with its
measured cost and the new total for the `slow` job (ADR-023); report it to the owner with the commit
proposal. A CI-measured `.test_durations` regeneration (ADR-074) follows the first green CI run if a
shard drifts; it is a separate owner decision.

- [ ] **Step 3: Freeze and hand off for review (G1).** No edits while reviewers A and B review (the owner
relays). Fix or answer every finding; take any A/B disagreement to the owner; re-run Step 1 after fixes.

- [ ] **Step 4: Commit proposal (G2).** Show the owner `git status --short`, `git diff --stat`, the
release decisions to make in ONE list (version number or "no release"; the ADR number re-confirmed),
and the commit message (Conventional Commits, STE prose, the implementing session's `Co-Authored-By`
line, no `Claude-Session:`). Commit only after an explicit "yes" for this commit. Then ask separately
for the push (the branch has no upstream: `git push -u origin feat/compute-contract-hf-campaigns`) and,
separately, for the PR. Poll CI with `gh run view <id> --json status,conclusion` (never `gh run watch`);
read EVERY failing shard's log if CI is red.

---

> **Status of Tasks 24–28 (owner ruling, 2026-10-21): superseded.** Task 24 ran (20 of 21 instances completed;
> win_probability's extract reduce did not finish). Tasks 25–28 did not run: the replica, the rebaseline, adopt, the
> sandbox publish and the artifact commit move to the campaign-throughput cycle (ADR-129), which fixes the defects
> Task 24 found and re-fits on the fixed code. The code commit merges without a version bump. Record: ADR-128
> "Cycle 1 outcome".

## Task 24: Campaign run 1 on real HF (criteria 2 and 4)

cpu-upgrade spend is pre-approved; nothing in this task commits, pushes or publishes.

- [ ] **Step 1:** `.venv-hfjobs/Scripts/python.exe -m scripts.hf_jobs campaign scripts/hf_jobs/campaigns/cycle1.toml preflight`
→ every check `ok`; read the flags (stages predicted above 60 minutes).
- [ ] **Step 2:** `… campaign cycle1.toml profile` then `… resume --watch` (background). The first
profile job of a staged dataset verifies the read-only dataset mount (spec R3): if a job fails reading
`/data/<name>`, set that dataset's `access = "download"` in `datasets.toml` (a reviewed one-line change
that the next code commit carries), re-run preflight, and continue.
- [ ] **Step 3:** `… status` until every instance's output stage is reduced; for a failed instance read
its job logs (`api.fetch_job_logs`) and the unit `*.failed.json`; a code defect follows the fix-forward
rule (header) — a new owner-approved code commit, then `resume` (finished units are not redone).
- [ ] **Step 4:** `… retrieve --staging <scratchpad>/cycle1` then `… verify --staging <scratchpad>/cycle1`.
- [ ] **Step 5:** `… summary` writes `docs/research/hf_campaigns/cycle1.json`. Criterion 4: every job's
wall from `JobInfo` ≤ 60 minutes, or the stage is listed with `irreducible_floor_s` and its measured unit
time (expected for the single xShot/xCross `study` units, evidence E9). Report the table.

---

## Task 25: Same-commit replica run (INV-1 campaign gate, criterion 3 second half)

- [ ] `… campaign cycle1.toml resume --replica 2 --watch`, then `retrieve --replica 2` into
`<scratchpad>/cycle1-r2`. For every artifact: `identity.compare(run1, run2, same_commit=True) == []`
(EVERY file byte-identical). Any difference is a STOP: report the artifact, the file and both hashes —
it means a determinism defect, which the fix-forward rule handles.

---

## Task 26: Rebaseline report (criterion 3) → owner review (G3)

- [ ] For every artifact the manifest declares and the campaign produced: `identity.compare(4.138.0
artifact, run 1 artifact, same_commit=False)` (wheel: the committed dir; Hub: the observed bootstrap
files). Write `<scratchpad>/cycle1/rebaseline.md`: per artifact `identical` or the differing files, and for
each difference the recorded environment delta (`env_provenance` of the 4.138.0 artifact where it
exists, e.g. `sklearn_version` in ghost metadata, vs the pinned run's) — spec §5.4 rebaseline rule; the
census uses decision P15. A difference with no environment explanation is a STOP. Present the report and
the Task 24 wall table to the owner (G3).

---

## Task 27: Adopt, then publish to private sandbox repos (criterion 5; G5)

- [ ] **Step 1:** `… campaign cycle1.toml adopt --staging <scratchpad>/cycle1` on a clean tree at the
campaign commit. It ends with the full `tests/` tree green and STOPS at the commit gate. Inspect
`git status --short`: only allowlisted paths changed.
- [ ] **Step 2 (owner go, every time):** `… publish --sandbox --staging <scratchpad>/cycle1` → every Hub
artifact goes to `silly-kicks/sandbox-<name>` (private) with all post-verification checks green
(model-only file set, remote `SHA256SUMS` == local, README == rendered card, `from_hub` round-trip).
`publish.json` lands in the bucket ledger; re-run `… summary` so the committed summary carries it.
Public Hub repos change only if weights changed, and only on a separate owner go after the artifact
commit (spec §7.5); a card-only re-publish (`publish_card_only`) is needed for every card whose
`hub_refit` sentence Task 19 added — list them for that owner go.

---

## Task 28: The artifact commit (G4)

The second and last commit provenance needs (ADR-037): the adopted artifacts, `observed.json`, cards,
verdicts and `docs/research/hf_campaigns/cycle1.json`, produced on the code commit's clean tree.

- [ ] Freeze; reviewers A and B review the adoption diff (owner relays); fix or answer every finding.
- [ ] Re-run the FULL `tests/` tree (bundled-weights/card/provenance change: never a subset).
- [ ] Commit proposal → explicit "yes" → commit; push on a separate "yes"; poll CI; read every failing
shard's log. Version, merge and tag stay owner decisions (G6).

---

## Self-review (author check against the spec; NOT a substitute for reviews A and B)

| Spec item | Task |
|---|---|
| §5.1 stage/unit contract, uniform CLI, `for_each` compatibility | 3, 9 (ItemStage on `for_each`, gate P12) |
| §5.2 executor refinement, mechanisms 1–4, memory bound, failure rule, partition | 4, 5 |
| §5.2 CI gates: W1N1-vs-W2N2 per job, non-vacuity, thread pin, ≥ 2 inner folds at W ≥ 2, importance/bootstrap at W > 1 | 4, 10–16 (parity tests), 13 (P11), 14, 16 |
| §5.2 campaign gate (same-commit SHA) | 25 |
| §5.3 resilience preserved + extended (every stage resumes, campaign resume, resume-before-download, checksums, ledger) | 2, 3, 4, 7 (`fetch`), 8 |
| §5.3 R1 mid-study restart rule | 14 Step 2 |
| §5.4 pinned image + `uv sync --frozen` + `hfjobs` group + `.venv-hfjobs` + fixed vars + `env_provenance` | 1, 6 (P1) |
| §5.4 rebaseline rule (criterion 3) | 20 (`identity`, P3), 26 |
| §5.4 typed launcher, labels, flavor constant + test | 6 |
| §5.4 preflight 1–6 | 7, 8 (`check_estimate`) |
| §5.5 campaign file, datasets declared once, derived run keys, CLI, stateless controller, ledger + committed summary, profile, planner, certification single-sourced | 7, 8, 5, 20 |
| §5.5 replaced: scratchpad drivers, uniform timeout, legacy DAG | 8, 18 |
| §5.6 migration table (7 jobs) + adapter | 10–17 |
| §5.7 manifest (expectation vs observed, INV-9), staging-time corpus expectation, generated card regions, adopt 1–10, ADR-037 guard, publish + sandbox | 7, 19, 20, 27 |
| §5.8 compute gate (`{reason, approval}`, red fixture, `_NOT_A_PIPELINE`), shared CLI, scaffolder, driver loader + I1 extension, skills, howtos, AGENTS.md, ADR-128 + markers, C4, no MCP, TF-46 rule | 9, 21, 22 |
| §6 caller sweep (phase flags, launch symbols, constants, `for_each`, `git_provenance`, registry/floor) | 18 Step 1, 19, 2 (`for_each` signatures unchanged), 1 (`git_provenance` unchanged), 8/18 |
| §7 acceptance 1–5 | 23 (1), 24 (2, 4), 25–26 (3), 27 (5) |
| §8 testing + CI budget statement | every task (test first), 23 Step 2 |
| §10 commit/review/release | 23, 28 (owner rule: provenance-minimal commits) |
| §12 R3 dataset volumes, R4 RFCDE scaling, R5 concurrency cap, R6 AGENTS budget, R8 TF-46 | 6/24 (R3), 4/5 profile + planner efficiency (R4), 5/8 (R5), 22 (R6), 21 (R8) |

Placeholder scan: every `<…>` in commands is an operator-supplied path or value (scratchpad dirs, toy
counts, measured durations), never missing design. Type consistency: `Stage`/`ItemStage` field names,
`RunContext` methods, `ExecutorConfig`, `JobSpec`, `StagePlan`, `Action`, `Artifact` are used with the
names Task 3–8/19 define.

---

## Execution handoff

Execute this plan INLINE with `superpowers:executing-plans` on the owner's implementation model, in this
checkout, on `feat/compute-contract-hf-campaigns`. No subagents. No worktrees. Stop at every gate
(G1–G6) and wait for the owner. Report and continue at checkpoints; never stall silently. A NEW fact
found during implementation that would change scope or an approved behaviour is surfaced to the owner;
engineering choices inside the spec's intent are made and recorded in the task notes and ADR-128.

---

## Revision log

**r2 (2026-10-18)** — responses to plan review r1 (A APPROVE, B APPROVE; reports
`D:\Development\_reviews\2026-10-18-compute-contract-hf-campaigns-plan-{A,B}.md`).

| Finding | Change |
|---|---|
| B CCHC-PLAN-01 (non-vacuity 3 was a tautology over a detached tuple) | Task 4 `test_w1n1_and_w2n2_are_byte_identical_and_non_vacuous`: non-vacuity 3 now reads THIS run's data — every fold's stored score equals the sorted-key sequential sum of its inner values, and at least one fold's completion-order re-sum differs. Helper `_seq` added. |
| A CCHF-PLAN-02 (is the open-data pin process-global?) | No: the pin is applied by `assert_statsbomb_open_data_mode()` only when `SILLY_KICKS_OPEN_DATA_REVISION` is set (Task 9). The P6 row said "at import", which contradicted Task 9; it now states the gate. |
| A CCHF-PLAN-01 (deleted phase-flag tests keep their live properties) | No plan change: Tasks 14/15 already require each deletion to be listed with its reason; it is a G1 implementation-review check. |

**r3 (2026-10-18)** — owner feedback: P1–P18 are engineering resolutions inside the spec's intent, not
owner decisions. The section is renamed "Engineering resolutions (applied)", its columns state the
resolution and the rejected alternative, gate G0 is removed, and the handoff no longer waits for
rulings. Substance of every row is unchanged; P1 and P2 are named as spec-text corrections for ADR-128.
