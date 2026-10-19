# How-to: train silly-kicks models on HF Jobs

The operator runbook for HF Jobs campaigns (ADR-128; it supersedes the ADR-122/ADR-123 launch path). A
campaign re-fits trainers and runs corpus drivers as stage-contract pipelines (`scripts/_stages.py`) on
`cpu-upgrade` HF Jobs: W workers per job x N jobs per stage, byte-identical for every (W, N), resumable at
any point. One committed CLI drives it: `python -m scripts.hf_jobs campaign <file> <command>`
(`scripts/hf_jobs/campaign.py`). Writing a new pipeline is `docs/howto/compute-pipelines.md`.

Three owner gates, every time:
- **Commit** -- `adopt` stops at the commit gate; nothing commits without an explicit owner "yes".
- **Push** -- a separate "yes".
- **Hub publish** -- `publish` needs an owner go per run (`--sandbox` first); a public repo also needs
  `--i-have-owner-approval <repo>` for every target.

`cpu-upgrade` only (8 vCPU / 32 GB / $0.03 per hour, INV-4): the flavor is one constant
(`scripts/hf_jobs/launch.py` `FLAVOR`); speed comes from W and N, never a bigger flavor.

## Prerequisites

- `.venv-hfjobs` (Python 3.12, the pinned in-job environment; the image is the digest-pinned
  `python:3.12.14-bookworm`, decision P1). Build it from the committed lock:

  ```bash
  UV_PROJECT_ENVIRONMENT=.venv-hfjobs uv sync --frozen --python 3.12.14 --extra train --extra xgboost --extra test --group hfjobs
  ```

  Run every campaign command and every local verification with `.venv-hfjobs/Scripts/python.exe` (Windows)
  or `.venv-hfjobs/bin/python`.
- An HF token: `hf auth login` (the `hf` CLI is for humans only; the controller uses the typed
  `huggingface_hub` API). The token rides `secrets`, never the command line.
- `gh auth login` (preflight checks that the campaign commit is pushed and its CI is green).
- The campaign commit is a CLEAN, PUSHED commit (ADR-037): jobs clone exactly that commit.

## Staged datasets

Every dataset a campaign reads is declared ONCE in `scripts/hf_jobs/datasets.toml` (name, repo, kind,
visibility, access). Kinds: `flat_tc3` (`shards/<gen>/<provider>__<game>.parquet` frames +
`_actions/<stem>.parquet` + `home_teams.json`), `pergame` (`<game>/frames.parquet` + `actions.parquet`), and
StatsBomb open data (fetched per unit at the pinned revision, decision P6). In-job code reads only these
neutral shapes (INV-5); how a dataset was built stays off HF.

After a dataset is staged, record it: `python -m scripts.hf_jobs datasets record <name>`. It MEASURES the
staged dataset and writes the pinned revision, the unit list and the expected corpus identity (digest + n;
for a public dataset the canonical id list) into `scripts/hf_jobs/datasets.lock.json`. `datasets show
<name>` prints the record. `adopt` checks every artifact against it (INV-9).

## The campaign file

`scripts/hf_jobs/campaigns/<name>.toml` (committed, INV-7): the bucket, `target_minutes` (60),
`concurrency_cap`, `attempts`, and per job the trainer or driver, its tier, its dataset (plus
`probe_dataset` for xcross) and its variants. Run keys are derived: `{campaign}-{commit7}-{job}-{tier}`;
`--replica N` adds `-rN`, a FRESH store for the same-commit re-run (INV-1 gate, decision P18).

## The command sequence

Run long commands in the background; `--watch` steps every 60 s until done or failed.

1. `preflight` -- six no-spend checks (commit pushed and green, lockfile and image, API contract, datasets,
   every instance's `--plan`, the planner's estimate). It uploads the report the launch gate requires and
   flags any stage predicted above the target.
2. `plan` -- each instance's next action (`profile` / `run` / `reduce`) and predicted wall; launches nothing.
3. `profile` -- measures a sample of every unprofiled multi-unit stage (unit wall, peak RSS).
4. `launch` / `resume --watch` -- the stateless controller: each step rebuilds state from the store
   (`--plan` per instance) and the job labels, then launches each instance's next stage, sized by the
   planner (N jobs x W workers to meet the target; LPT partition by measured cost). A killed job loses at
   most its in-flight units; `resume` never redoes a finished unit.
5. `status` -- per instance, stage progress and the running jobs. A failed instance: read its job logs and
   the unit `*.failed.json`; a code defect is a new owner-approved commit, then `resume`.
6. `retrieve --staging DIR` -- copies each reduced output (and `<instance>.record.json`: what the job read,
   the extract counters) into DIR, OUTSIDE the repo.
7. `verify --staging DIR` -- loads every artifact in `.venv-hfjobs`, checks the five expectations (training
   commit = campaign commit; recorded dataset revision = the pin; corpus digest + n = the record; shipped
   tier = the expected tier), certifies per the registry (`public`: head-to-head floor; `owner`: CV parity on
   `cv_metric`; `rebaseline`: identity comparison, decision P15) and runs the ADR-126 integrity gate for
   xShot/xCross. Writes `DIR/verify.json`.
8. `adopt --staging DIR` -- on a clean tree at the campaign commit: bundles the verified files into the
   wheel dirs (a public bundle is rehydrated with the canonical public ids), writes
   `docs/artifacts/observed.json`, copies each adopted artifact's `training_commit` into
   `docs/artifacts/manifest.toml` (the one field a tool writes there), re-renders the generated card
   regions, writes the verdict JSONs and a CHANGELOG draft, runs the full local gate, and STOPS at the
   commit gate. Its diff touches only the allowlisted paths.
9. Owner gate (commit), then declare the re-fit event in the manifest (version, ADR) and re-render the
   cards (`tests/test_card_regions.py` flags a stale region).
10. `publish --sandbox --staging DIR` (owner go) -- private `silly-kicks/sandbox-<name>` repos, then the
    post-verification (model-only file set, remote `SHA256SUMS` = local, README = the rendered card, a
    `from_hub` round-trip). A public publish is a separate owner go after the artifact commit.
11. `summary` -- writes the committed ledger summary `docs/research/hf_campaigns/<campaign>.json` (unit
    costs the planner learns from).

## The DGX as an executor

The DGX runs the SAME contract (ruling 11): `scripts/hf_jobs/dgx_e2e.py` `run_local_e2e` plans an instance
and runs each stage through the stage CLI with peak-RSS measurement against the 32 GB `cpu-upgrade` ceiling
(a stage over the ceiling, or a non-zero exit, is `blocked`).

## Cost

Per-second billing on `cpu-upgrade` ($0.03 per hour): N short jobs cost the same as one long job; the
planner prints the predicted cost per action. The scratch bucket adds a small per-GB-month cost.
