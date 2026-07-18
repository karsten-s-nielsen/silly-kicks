# HF Hub upload instructions — TF-19 PR-2 HF-only variants

**Not executed in this PR.** Publishing to Hugging Face Hub needs an authenticated `hf`
CLI / `huggingface_hub` token belonging to the `silly-kicks` HF org; no such credential is
available in this code path. This document is the owner follow-up runbook. Everything
described here happens **after** this PR merges.

## What ships to the Hub

Three weight artifacts are HF-only (never bundled in the PyPI wheel):

1. **xS `sc_extended`** — the Stage-B xShotOccurrence weights, trained with the 98 owner
   SkillCorner matches admitted.
2. **xCross `sc_extended`** — the Stage-B xCrossAttempt weights, same corpus. This is the
   variant whose frozen GK-substitution probe is recorded in `decision_table.md`
   (ratio ~2.21x, `tf19_ready=false` on the absolute-floor prong).
3. **Ghost-GK `full`** — the expanded 179-match ghost-GK retrain (up from 81 matches;
   `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` already documents the new
   179-match / ~1.04M-frame `full` variant in this PR).

All three repos are **model repos that do not exist yet** and must be created before the
first upload:

- `silly-kicks/xshot-occurrence-v1`
- `silly-kicks/xcross-attempt-v1`
- `silly-kicks/ghost-gk-v1`

(The `ghost-gk-v1` repo may already carry a `default`/older `full` revision from a prior
release — check before assuming a from-scratch create; the other two are net-new per
`_HF_REPO_ID` in `_xshot_occurrence.py` / `_xcross_attempt.py`, which point at repo ids that
have never been populated.)

## Local staging paths (this PR's artifacts)

xS and xCross `sc_extended` are staged locally, **already base_score-normalized** (the
xgboost-3.x bracketed `base_score` fix from this PR was applied to these Hub-bound
artifacts too, not just the bundled defaults — confirmed by file mtimes: `model.json` and
`SHA256SUMS` were rewritten after the original `metadata.json`/`metrics.json`):

```
C:\Users\Karsten\AppData\Local\Temp\claude\D--Development-karstenskyt--silly-kicks-part-deux\e14c809d-84c7-4487-992d-d7b587dcaed0\scratchpad\weights\xs_sc_extended\
    SHA256SUMS
    metadata.json
    metrics.json
    model.json

C:\Users\Karsten\AppData\Local\Temp\claude\D--Development-karstenskyt--silly-kicks-part-deux\e14c809d-84c7-4487-992d-d7b587dcaed0\scratchpad\weights\xcross_sc_extended\
    SHA256SUMS
    metadata.json
    metrics.json
    model.json
```

Upload the whole directory contents (all four files) — `SHA256SUMS` is what `load()`
verifies against before the chirality check ever runs, and `metrics.json` is the probe /
CV record referenced by `decision_table.md`.

The full ghost-GK retrain lives on the DGX training box (not copied locally — it is ~200 MB):

```
karsten@192.168.68.73:~/Development/sk_stageB_448/ghost_full/ghost_gk_v1/
    SHA256SUMS
    metadata.json
    metrics.json
    rfcde_weights.npz      (~208 MB)
    _feature_cache/         <- EXCLUDE from the upload (training-time cache, not a served artifact)
```

`_feature_cache/` must **not** be uploaded — it is the on-disk feature-extraction cache
`scripts/train_ghost_gk.py` writes during training, not part of the served model contract
(`GhostGkModel.load()` reads only `rfcde_weights.npz` + `metadata.json`, verified against
`SHA256SUMS`). Pull the four served files down first, e.g.:

```
scp karsten@192.168.68.73:~/Development/sk_stageB_448/ghost_full/ghost_gk_v1/{SHA256SUMS,metadata.json,metrics.json,rfcde_weights.npz} <local_staging_dir>/
```

## Repo creation + upload call pattern

```python
from huggingface_hub import HfApi

api = HfApi()  # picks up the token from `hf auth login` / HF_TOKEN env var

for repo_id in (
    "silly-kicks/xshot-occurrence-v1",
    "silly-kicks/xcross-attempt-v1",
    "silly-kicks/ghost-gk-v1",
):
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

# NOTE: upload the served files to the repo ROOT (no path_in_repo). `from_hub` does
# `snapshot_download(repo_id)` then `load()` on the download ROOT, which expects SHA256SUMS at
# the top level — so each repo serves exactly ONE variant, at its root (see the note below).
api.upload_folder(
    repo_id="silly-kicks/xshot-occurrence-v1",
    folder_path=r"C:\...\scratchpad\weights\xs_sc_extended",
    repo_type="model",
    commit_message="TF-19 PR-2: sc_extended xS weights (98 owner SkillCorner matches admitted)",
)

api.upload_folder(
    repo_id="silly-kicks/xcross-attempt-v1",
    folder_path=r"C:\...\scratchpad\weights\xcross_sc_extended",
    repo_type="model",
    commit_message="TF-19 PR-2: sc_extended xCross weights (98 owner SkillCorner matches admitted)",
)

api.upload_folder(
    repo_id="silly-kicks/ghost-gk-v1",
    folder_path="<local_staging_dir_for_ghost_full>",  # the 4 served files, _feature_cache excluded
    repo_type="model",
    commit_message="TF-19 PR-2: full ghost-GK retrain, 179 matches / ~1.04M frames",
)
```

Notes on the pattern:

- **Upload to the repo ROOT, not a `path_in_repo` subfolder.** `from_hub` (in
  `_xshot_occurrence.py` / `_xcross_attempt.py` / `_ghost_gk.py`) does
  `local_dir = snapshot_download(repo_id)` then `load(Path(local_dir))`, and `load()` requires
  `local_dir/SHA256SUMS` at the TOP level. A `path_in_repo="sc_extended"` upload would put the
  files under `local_dir/sc_extended/`, and `load()` would raise `IntegrityError: SHA256SUMS not
  found`. So each `*-v1` repo serves exactly ONE Hub variant, at its root (xS/xCross → `sc_extended`;
  ghost → `full`; the `public`/`default` arm is bundled in the wheel, never fetched from the Hub).
  If a repo ever needs to host multiple Hub variants, `from_hub` must first be made
  subfolder-aware — do not work around it by nesting the upload.
- `create_repo(..., exist_ok=True)` is safe to call even after the repo exists (idempotent;
  matters if `ghost-gk-v1` already has content from an earlier release).
- Use a real HF token with write access to the `silly-kicks` org (`hf auth login` or
  `HF_TOKEN` env var); do not hardcode a token in any script that gets committed.

## Routing: how `from_variant("sc_extended")` finds these

Both `XShotOccurrenceModel.from_variant` (`silly_kicks/tracking/_xshot_occurrence.py`) and
`XCrossAttemptModel.from_variant` (`silly_kicks/tracking/_xcross_attempt.py`) route any
variant not found bundled in the wheel — currently `"public"` and, as of this PR,
`"sc_extended"` — to `from_hub(_HF_REPO_ID)`:

- xS: `_HF_REPO_ID = "silly-kicks/xshot-occurrence-v1"`
- xCross: `_HF_REPO_ID = "silly-kicks/xcross-attempt-v1"`

`from_hub` calls `huggingface_hub.snapshot_download(repo_id=repo_id)` and then `cls.load()`
on the downloaded directory — the same `load()` that enforces the chirality fingerprint
(this PR's fail-closed enforcement applies identically whether the artifact came from the
bundled wheel or the Hub). Until the repos above are created and populated, calling
`from_variant("sc_extended")` will raise (Hub 404 / repo-not-found) — that is the expected,
correct behavior pre-upload; it is not a bug to "fix" before this PR merges.

Ghost-GK's Hub routing (`GhostGkModel.from_variant`, `_ghost_gk.py`) works the same way for
the `"full"` variant: bundled `default` ships in the wheel, `full` is Hub-only via
`_HF_REPO_ID = "silly-kicks/ghost-gk-v1"`.

## Post-upload verification (run after the owner uploads)

```python
from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel
from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
from silly_kicks.tracking._ghost_gk import GhostGkModel

xs = XShotOccurrenceModel.from_variant("sc_extended")   # resolves via from_hub, no FileNotFoundError
xc = XCrossAttemptModel.from_variant("sc_extended")     # resolves via from_hub, no FileNotFoundError
gk = GhostGkModel.from_variant("full")                  # resolves via from_hub, no FileNotFoundError
```

A successful `from_variant` call here means two things happened correctly: (1) the Hub
download resolved the uploaded files, and (2) `load()`'s chirality re-verification passed
on the downloaded artifact — i.e. the `metadata.json` chirality block that travelled with
the upload reproduces on whatever machine runs this verification, within the cross-platform
tolerance (`atol=1e-3`, `rtol=1e-2`; see ADR-040). A chirality `IntegrityError` on this
verification step means the upload is corrupt or mismatched, not that the tolerance needs
loosening — re-upload from the staging paths above rather than patching the check.

## Licensing note

The `sc_extended` weights are trained in part on the 98 owner-tier SkillCorner matches
(restricted visibility, per ADR-038's visibility-keyed corpus taxonomy). The owner has
approved distributing the **trained model** (learned parameters only — tree structure /
leaf values / calibration) on the Hub, not the underlying raw tracking data. This mirrors
the existing Gradient Sports precedent already stated in the ghost-GK model card ("only the
trained model weights are distributed here; the underlying raw tracking data is not
redistributed") — the same boundary applies to `sc_extended`'s owner-tier SkillCorner
matches. Do not attach any raw match/tracking files to these repos; only the four served
artifact files (`model.json` / `rfcde_weights.npz`, `metadata.json`, `metrics.json`,
`SHA256SUMS`) per variant.
