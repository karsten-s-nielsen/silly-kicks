# How-to: train silly-kicks models on HF Jobs

The operator runbook for the HF Jobs training harness (`scripts/hf_jobs/`, ADR-122 + ADR-123). Re-fits
**9 trainers** (4 public + 5 owner-tier) on ephemeral `cpu-upgrade` HF Jobs, parallelized as a map/reduce
DAG per trainer, head-to-head-scores each fresh fit against the committed model (no-regression floor), and
PRs the cleared weights back. `gk_retention` is registered but blocked (no non-Databricks data path);
`gk_completion` was retired upstream (ADR-121).

Two invariants (ADR-123):
- **The in-job code is dataset-agnostic.** It reads a staged **neutral parquet dataset** via a data path;
  it never receives pining information (no token, no manifest, no `--providers`). How a dataset was built
  is off-HF — anyone can stage their own in the documented shape (below) and run the harness.
- **`cpu-upgrade` only** (8 vCPU / 32 GB / $0.03-hr). Storage is a read-write **HF Storage Bucket** mounted
  per job (decoupled from the ≤ 50 GB ephemeral disk); `--attempts N` auto-retries and **resumes** from the
  bucket shards (`for_each`, ADR-052). Outputs live in the bucket — no in-job upload.

## Prerequisites

- **`hf` CLI ≥ v2.2** — the launch needs `-v/--volume` (bucket mount); v1.7.1 lacks it. If the system `hf`
  is older, `uv pip install "huggingface_hub[cli]"` in an isolated venv and point the launcher at it.
- `hf` logged in; the token rides `--secrets HF_TOKEN` (value from the launcher env, never the cmdline).
- The GitHub repo must be **public** (the in-job `git clone` passes no auth).
- The harness commit **pushed** to `feat/hf-jobs-training-harness` (the jobs clone that ref).
- A private scratch bucket: `hf buckets create silly-kicks/hf-refit-scratch --private`.

## The staged-dataset contract (how anyone stages their own)

The in-job trainers read these NEUTRAL parquet shapes. Build them however you like (the owner materializes
with pining on the DGX; you could use statsbombpy, kloppy, or your own loader) — the repo is agnostic.

- **Flat** (`owner_flat`: ghost_gk/ghost_outfield, and every owner head-to-head eval): a dataset dir with
  `shards/<gen>/<key>.parquet` (tracking frames, one match per file, carrying `source_provider`) +
  `_actions/<key>.parquet` (SPADL actions) + `home_teams.json` (`{game_id: home_team_id}`).
- **Per-game** (`owner_pergame`: xshot/xcross training): the harness `repackage`s the flat dataset in-job
  to `{game}/frames.parquet`+`shots.parquet`+`actions.parquet`, so stage the flat shape.
- **Flat, read via `--data-dir`** (`owner_sb360`: receiver): the same flat shards+`_actions` shape.
- **xcross probe slice**: a per-game dataset of the held-out TF-19 probe matches (e.g. gradientsports
  `10502`/`10503`), each `{game}/{frames,actions}.parquet` with `source_provider=gradientsports`. xcross
  reproduces the probe from this via `--probe-data-dir` — no pining in-job.
- Public trainers fetch StatsBomb **open** data in-job (no staging).

Materialize + upload (owner's DGX path; pining is used HERE, off-HF, and only neutral parquet is pushed):

```python
from scripts.hf_jobs.stage import stage_corpus   # neutral-parquet uploader; build the dirs however you stage
stage_corpus(tc3_dir,   "silly-kicks/hf-refit-tc3",         uploader=my_uploader)  # flat (ghost + evals)
stage_corpus(xshot17_dir, "silly-kicks/hf-refit-xshot17",   uploader=my_uploader)  # flat, committed 17 (xshot/xcross)
stage_corpus(gs_probe_dir, "silly-kicks/hf-refit-xcross-probe", uploader=my_uploader)  # per-game probe slice
stage_corpus(sb360_flat_dir, "silly-kicks/hf-refit-sb360",  uploader=my_uploader)  # flat (receiver)
```

## DGX e2e gate (before any HF spend)

Run every trainer's full map/reduce pipeline LOCALLY against the staged neutral corpus and measure
**peak RSS per phase vs the 32 GB cpu-upgrade ceiling** (`scripts/hf_jobs/dgx_e2e.py`):

```python
from scripts.hf_jobs.dgx_e2e import run_local_e2e
summary = run_local_e2e(spec, build, data_args="--data-dir /data/tc3/shards/<gen> --actions-dir … --home-teams …",
                        out_dir="/tmp/e2e", corpus_dir="/data/tc3", n_extract=4, n_folds=5)
assert not summary["blocked"], summary   # a phase >= 32 GB (OOM risk) or non-zero exit BLOCKS the wave
```

The serial reduce is the phase fan-out does not shrink; a 32 GB OOM is not `--attempts`-fixable and the
DGX's ample RAM hides it, so it is measured here. If `blocked` (a reduce over 32 GB), **stop** and get the
owner's ruling on the reduce executor/flavor — do not launch the HF wave.

## Launch the map/reduce waves

```python
from huggingface_hub import get_token
from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import run_pipeline, retrieve_variant

BUCKET, RUN = "silly-kicks/hf-refit-scratch", "refit-2026-10-14"
spec = reg.REGISTRY["ghost_gk"]
for build in spec.builds:                       # one pipeline per committed variant
    run_pipeline(
        spec, build, tag="feat/hf-jobs-training-harness", run_key=RUN,
        corpus_repo="silly-kicks/hf-refit-tc3",  # ghost/xshot/xcross/receiver: the staged flat/pergame dataset
        probe_repo="silly-kicks/hf-refit-xcross-probe",  # xcross only (spec.probe)
        bucket=BUCKET, attempts=3, n_extract=8, n_folds=spec_cv_folds,
    )
    retrieve_variant(BUCKET, RUN, spec.name, build.variant, f"runs/{spec.name}/{build.variant}")
```

`run_pipeline` sequences the trainer's phases (`phase_sequence(spec)`): `extract` (`--shard-index I/N`
fan-out) → `build_cache` (ghost_gk) → `fold`/`study` (fan-out) → `reduce` (serial final fit + the
head-to-head floor). Each stage's jobs launch in parallel and the orchestrator waits for all before the
next; a terminal-ERROR stage raises `StagePhaseError`. Receiver's reduce is a plain no-`--shard-index` run
(reconcile all shards + fit). Public trainers are a single `full` job. No in-job upload; `retrieve_variant`
copies the bucket output out with `hf cp`.

Per source mode (all via `corpus_repo`, no `sb360_repo`): public → none; `owner_flat` (ghost) → the flat
dataset; `owner_pergame` (xshot/xcross) → the flat dataset (repackaged in-job), **xcross also**
`probe_repo`; `owner_sb360` (receiver) → the flat dataset read via `--data-dir`.

**Visibility-gated trainers (`xshot_occurrence`, `xcross_attempt`; ADR-123 Amendment B).** These ship a
`public`/`sc_extended`/`full` candidate chosen by per-match public/owner visibility. The in-job code is
visibility-blind: `run_pipeline` computes the arm membership OFF-HF (`arm_policy.compute_arms` over the
in-repo `PUBLIC_CORPUS`, emitting the dataset's own match keys) and passes it as the neutral `--arms`
parameter (base64-embedded `/tmp/arms.json`); the trainer applies it by membership and never computes the
policy. Anyone staging their own corpus supplies their own `--arms` (or lets `run_pipeline` compute it from
their registry). The DGX e2e asserts the fresh shipped candidate matches the committed one.

## Apply the no-regression floor (per variant)

```python
from scripts.hf_jobs.floor import evaluate_spec
ok, detail = evaluate_spec(spec, run_metrics)   # False => regression beyond δ => BLOCK, hand to owner
```

The reduce job wrote the head-to-head pair + stamp into the variant's `metrics.json`, extracted at the
variant's feature vintage (`position_only` scored on a 26-feature frame). Build the delta table (metric,
committed, fresh, δ, verdict; one row per variant); any `ok is False` BLOCKS PR-back — investigate, never
auto-accept.

## Chirality re-confirm + PR weights back

Load a retrieved artifact in-repo and confirm `verify_chirality` + `SHA256SUMS` + `_feature_contract` pass
(first HF artifact, P1). Then copy each cleared variant's weights into `silly_kicks/<weights_path>/`,
matching the committed bundle shape (no new `metrics.json` where none existed), run the full suite green,
stamp the lakehouse re-mat owed, and open the single owner-gated weights commit. Merge #284 **non-squash**
so the harness-tune commit + the weights commit both stay reachable (the weights' `run_commit` provenance).

## Cost

Everything on `cpu-upgrade` ($0.03/hr). Map/reduce fans extraction + folds/studies across many short jobs;
the serial reduce is the long pole. A full re-fit is well under $50. The scratch bucket adds a small
$/GB-month cost — delete it after the campaign (`hf buckets delete silly-kicks/hf-refit-scratch`).
