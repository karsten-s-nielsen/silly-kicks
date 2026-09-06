"""Shared HuggingFace publish helper: upload MODEL-ONLY, never the trainer's co-located caches.

The trainers write their internal `_feature_cache/`, `shards/` (RAW per-match tracking frames from
restricted owner-tier providers) and `_probe_sample/` INTO the same artifact directory as the model
files. A bare ``upload_folder(dir)`` therefore ships raw restricted data to a public repo (this
happened once -- see ADR-072 + the incident memory). This helper uploads a fixed allowlist and then
fail-closes if the repo carries ANY nested path afterwards (a `foo/bar` filename == a leaked
subdirectory), so the mistake cannot recur silently.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

#: The COMPLETE set of files a distributed model artifact may contain. Everything else the trainer
#: leaves in the artifact dir (feature caches, raw-frame shards, probe samples) is training-internal
#: and MUST NOT be published. `allow_patterns` are matched on the repo-relative path, so these bare
#: names match only root-level files -- a `_feature_cache/metadata.json` is NOT matched.
MODEL_ONLY_ALLOWLIST: tuple[str, ...] = (
    "model.json",  # xShot / xCross XGBoost booster
    "rfcde_weights.npz",  # ghost-GK weights
    "model.npz",  # ghost-outfield weights (boosted-mean x/y ensembles)
    "metadata.json",
    "metrics.json",
    "SHA256SUMS",
    "README.md",
)


def upload_model_only(api: Any, artifact_dir: str, repo_id: str) -> None:
    """Upload ONLY the allowlisted model files, then fail-closed on any nested repo path.

    ``allow_patterns`` bounds what is UPLOADED; the post-upload check bounds what the repo ENDS UP
    with (it also catches a nested path left by an earlier bad publish, which the allowlist cannot
    remove). A nested path raises ``SystemExit`` -- the operator must delete it from the repo.
    """
    api.upload_folder(
        folder_path=artifact_dir,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=list(MODEL_ONLY_ALLOWLIST),
    )
    info = api.model_info(repo_id)
    nested = sorted(s.rfilename for s in (info.siblings or []) if "/" in s.rfilename)
    if nested:
        raise SystemExit(
            f"PUBLISH LEAK GUARD: {repo_id} carries non-model paths after upload -- "
            f"{nested[:8]}{' ...' if len(nested) > 8 else ''}. A model artifact repo must be "
            "model-only (no _feature_cache/ shards/ _probe_sample/). Delete these paths from the "
            "repo before it is used."
        )


def publish_model_with_card(api: Any, artifact_dir: str, repo_id: str, *, model_card: str) -> None:
    """The ONE publish seam every ``publish_*`` script uses: create the repo (idempotent), stage the
    allowlisted model files + the model card (as ``README.md``) into a temp dir, and upload model-only.

    The model card is REQUIRED. A card staged by hand -- copying it to ``README.md`` before an upload
    that pulls from the raw artifact dir -- is exactly how model cards get dropped from a release
    (measured: the TF-60 PR5 cards were dropped, twice across the ghost cycles). Threading the card
    through the single upload seam makes a card-less repo UNREPRESENTABLE for every publisher, and the
    temp-dir staging means the committed weights dir is never mutated. ``create_repo`` is idempotent
    (``exist_ok=True``) so this is safe for both a brand-new repo and a re-publish.
    """
    card = Path(model_card)
    if not card.is_file():
        raise SystemExit(f"model card {card} does not exist (required for a real publish, uploaded as README.md).")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        for f in Path(artifact_dir).iterdir():
            if f.is_file():
                shutil.copy2(f, stage / f.name)
        shutil.copy2(card, stage / "README.md")
        upload_model_only(api, str(stage), repo_id)
