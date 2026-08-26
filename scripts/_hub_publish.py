"""Shared HuggingFace publish helper: upload MODEL-ONLY, never the trainer's co-located caches.

The trainers write their internal `_feature_cache/`, `shards/` (RAW per-match tracking frames from
restricted owner-tier providers) and `_probe_sample/` INTO the same artifact directory as the model
files. A bare ``upload_folder(dir)`` therefore ships raw restricted data to a public repo (this
happened once -- see ADR-072 + the incident memory). This helper uploads a fixed allowlist and then
fail-closes if the repo carries ANY nested path afterwards (a `foo/bar` filename == a leaked
subdirectory), so the mistake cannot recur silently.
"""

from __future__ import annotations

from typing import Any

#: The COMPLETE set of files a distributed model artifact may contain. Everything else the trainer
#: leaves in the artifact dir (feature caches, raw-frame shards, probe samples) is training-internal
#: and MUST NOT be published. `allow_patterns` are matched on the repo-relative path, so these bare
#: names match only root-level files -- a `_feature_cache/metadata.json` is NOT matched.
MODEL_ONLY_ALLOWLIST: tuple[str, ...] = (
    "model.json",  # xShot / xCross XGBoost booster
    "rfcde_weights.npz",  # ghost-GK weights
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
