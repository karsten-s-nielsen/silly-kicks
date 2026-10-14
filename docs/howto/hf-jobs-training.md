# How-to: train silly-kicks models on HF Jobs

The operator runbook for the HF Jobs training harness (`scripts/hf_jobs/`, ADR-122 + its 2026-10-14
amendment). Re-fits **9 trainers** (4 public + 5 owner-tier) on ephemeral HF Jobs, one job per committed
variant (**15 jobs**), head-to-head-scores each fresh fit against the committed model, and PRs the cleared
weights back. `gk_retention` is registered but blocked (no non-Databricks data path); `gk_completion` was
retired upstream (ADR-121).

Storage is a read-write **HF Storage Bucket** mounted into each job (decoupled from the flavor's fixed
≤ 50 GB ephemeral disk); with `--attempts N` a crashed/timed-out job auto-retries and **resumes** from the
shards already written to the bucket (corpus-driver `for_each`, ADR-052). Outputs live in the bucket — no
in-job upload.

## Prerequisites

- **`hf` CLI ≥ v2.2** — the launch needs `-v/--volume` (bucket mount), which CLI v1.7.1 LACKS. If the
  system `hf` is older, install a current one in an isolated venv (`uv pip install "huggingface_hub[cli]"`)
  and point the launcher at that binary. Confirm with `hf jobs run --help | grep -- --volume`.
- `hf` logged in; the token is read via `huggingface_hub.get_token()` and forwarded to the job as
  `--secrets HF_TOKEN` (value from the launcher env, never on the cmdline).
- The GitHub repo must be **public** (the in-job `git clone` passes no GitHub auth).
- **Commit 1 (the harness) committed AND pushed** to `feat/hf-jobs-training-harness` before P1 — the jobs
  clone that pushed ref for `scripts/hf_jobs/**` + the committed weights. Pushed, not merged.
- A private **scratch bucket**: `hf buckets create silly-kicks/hf-refit-scratch --private`.

## 1. Stage the corpora (DGX/local → private HF datasets)

Three staged datasets (all FULL corpora — never a subsample), each faithful to the committed weights'
recorded provenance, materialized where the pining data lives (ZERO Databricks):

**(a) Flat TC3** `silly-kicks/hf-refit-tc3` — 179 matches (GS 64 + idsse 7 + SC 108),
`shards/<gen>/<provider>__<game>.parquet` + `_actions/` + `home_teams.json`. Serves `ghost_gk`/
`ghost_outfield` and every owner head-to-head eval (`build_eval._owner_shards`). The committed ghost
corpus recorded `providers=[gradientsports, skillcorner, sportec]`; **sportec ≡ idsse** (the DFL matches
are built by `sportec_spadl`, relabeled `idsse` in the current materialization), so this IS the faithful
ghost corpus.

**(b) xshot/xcross** `silly-kicks/hf-refit-xshot` — the **exact 17** `corpus_match_ids` recorded in the
committed xshot/xcross `metrics.json` (7 idsse + 10 skillcorner), generation at the weights' own
`run_commit a34e992`. A SEPARATE dataset because `owner_pergame` repackages the WHOLE staged corpus in-job
(no match-id filter), so the 17 cannot share the full-179 dataset.

**(c) SB360 bundle (receiver)** `silly-kicks/hf-refit-sb360` — `cache/statsbomb/<id>/{events,metadata,
roster,freeze_frames}` (populated once by a token-bearing `load_statsbomb_matches(match_ids, token,
cache_dir=…)` over the 30 ids) + an OFFLINE `manifest/statsbomb/matches.json` + `match_ids.json`. The job
sets `SILLY_KICKS_PINING_MANIFEST_DIR` so enumeration is offline (zero token/network in-job).

```python
from scripts.hf_jobs.stage import stage_corpus, write_offline_manifest
stage_corpus(tc3_dir, "silly-kicks/hf-refit-tc3", uploader=my_uploader)
stage_corpus(xshot17_dir, "silly-kicks/hf-refit-xshot", uploader=my_uploader)
# sb360: populate cache/ (load_statsbomb_matches, token), write match_ids.json, then:
write_offline_manifest(["statsbomb"], "<bundle>/manifest")   # live API, needs the owner token
stage_corpus("<bundle>", "silly-kicks/hf-refit-sb360", uploader=my_uploader)
```

Public trainers (`win_probability` [full `--all-competitions`, 3961 matches], `pass_completion`,
`match_outcome_dependence`, `xsuccess`) fetch StatsBomb open data in-job — no staging.

## 2. Launch (one ephemeral Job per committed variant)

```python
from huggingface_hub import get_token
from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import launch_all_variants, poll_until_terminal, retrieve_variant

BUCKET = "silly-kicks/hf-refit-scratch"
spec = reg.REGISTRY["ghost_gk"]
ids = launch_all_variants(                      # one hf jobs run per VariantBuild -> {variant: job_id}
    spec, tag="feat/hf-jobs-training-harness",  # the pushed Commit-1 ref, NOT v4.134.0
    run_key="refit-2026-10-14",
    corpus_repo="silly-kicks/hf-refit-tc3",     # owner trainers: tc3 (ghost) / hf-refit-xshot (xshot,xcross)
    sb360_repo=None,                            # receiver ALSO needs "silly-kicks/hf-refit-sb360"
    token=get_token(), bucket=BUCKET, attempts=3)
for variant, jid in ids.items():
    poll_until_terminal(jid)                    # reads `hf jobs inspect --format json`
    retrieve_variant(BUCKET, "refit-2026-10-14", spec.name, variant, f"runs/{spec.name}/{variant}")
```

Per source mode: public → no `corpus_repo`; `owner_flat` (ghost_gk, ghost_outfield) → `corpus_repo=tc3`;
`owner_pergame` (xshot, xcross) → `corpus_repo=hf-refit-xshot` (repackaged in-job); `owner_sb360`
(receiver) → `corpus_repo=tc3` (for the eval) + `sb360_repo=hf-refit-sb360` (its training corpus).

`build_argv` emits `hf jobs run --flavor … --secrets HF_TOKEN --timeout … --attempts N -v
hf://buckets/<bucket>:/scratch -d python:3.12 -- bash -lc "<cmd>"`. The `--` before `bash` is mandatory
(else `-lc` is eaten — `--dry-run` confirms); `--name` is never passed. `<cmd>` clones the ref,
`uv pip install -e '.[train]' … huggingface_hub` (the last for the in-job `hf download`), feeds the staged
corpus per the source mode, runs THIS variant via `python -m scripts.train_<name>` writing to
`/scratch/{run_key}/{name}/{variant}`, then runs the head-to-head scorer on THAT variant. Nothing is
uploaded — the output persists in the mounted bucket; `retrieve_variant` copies it out with `hf cp`.

**Resume / retry.** `--attempts N` auto-retries a crashed/timed-out job; the shard-root persists in the
bucket, so `for_each` skips already-written shards (`[k/N] … 0.0s skip (shard exists)`) and continues. To
resume manually, relaunch the SAME `run_key` + bucket. **Concurrency** is bounded by the account Jobs quota
(≥ 2 confirmed); spend is per-job-second and concurrency-independent — fan out with a bounded pool.

## 3. Apply the no-regression floor (per variant)

Each retrieved variant's `metrics.json` carries the in-job head-to-head pair + the platform stamp:

```python
from scripts.hf_jobs.floor import evaluate_spec
ok, detail = evaluate_spec(spec, run_metrics)   # False => regression beyond δ => BLOCK, hand to owner
```

Build the head-to-head delta table (metric, committed, fresh, δ, verdict; one row per variant). Any
`ok is False` BLOCKS that trainer/variant from PR-back — investigate, never auto-accept.

## 4. Chirality re-confirm (first HF artifact, P1)

Load a retrieved artifact in-repo and confirm `verify_chirality` + `SHA256SUMS` + `_feature_contract`
pass. The measured cross-platform delta is 0.0 (`docs/research/pr5_platform_atol/`) and HF is save-x86 /
load-x86 (same-arch), so this is expected to pass; a failure is a genuine P0 surprise — stop and report.

## 5. PR weights back

Copy each cleared trainer/variant's `weights/` artifacts (retrieved from the bucket) into
`silly_kicks/<weights_path>/`, matching the bundle's existing committed shape (do NOT introduce a
`metrics.json` into a bundle that never had one). Run the full suite green, stamp the downstream lakehouse
re-materialize owed (handed to the owner), and open the single PR. Weights land in one owner-gated commit,
not split per model. Merge #284 **non-squash** (merge-commit/rebase) so both commits stay in `main` and the
weights' `run_commit` stays reachable.

## Cost

CPU-only flavors: `cpu-upgrade` ($0.03/hr) public, `cpu-xl` ($1.00/hr) xshot/xcross/receiver,
`cpu-performance` ($1.90/hr) the RAM-bound ghost RFCDE fits. A full 15-job re-fit is well under $50. The
scratch bucket adds a small $/GB-month storage cost — delete it after the campaign (`hf buckets delete`).
```
