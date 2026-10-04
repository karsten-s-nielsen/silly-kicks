"""Shared HuggingFace publish helper: upload MODEL-ONLY, never the trainer's co-located caches.

The trainers write their internal `_feature_cache/`, `shards/` (RAW per-match tracking frames from
restricted owner-tier providers) and `_probe_sample/` INTO the same artifact directory as the model
files. A bare ``upload_folder(dir)`` therefore ships raw restricted data to a public repo (this
happened once -- see ADR-072 + the incident memory). This helper uploads a fixed allowlist and then
fail-closes if the repo carries ANY nested path afterwards (a `foo/bar` filename == a leaked
subdirectory), so the mistake cannot recur silently.
"""

from __future__ import annotations

import hashlib
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

_CARD_DIR = "docs/huggingface/model-cards"

#: Every Hub repo of the org -> the in-repo card its README must equal (combined-cycle spec 9, D9).
#: The single source for the card-only seam and for validate_hub_variants (whose test asserts these
#: keys equal its HUB_REGISTRY). A card is never taken from a free path, so it cannot reach the wrong repo.
CARD_SOURCE: dict[str, str] = {
    **{
        f"silly-kicks/{name}": f"{_CARD_DIR}/{name}-model-card.md"
        for name in (
            "xshot-occurrence-v1",
            "xshot-occurrence-position-only-v1",
            "xcross-attempt-v1",
            "xcross-attempt-position-only-v1",
            "ghost-gk-v1",
            "ghost-gk-sweeper-v1",
            "ghost-gk-sweeper-position-only-v1",
            "ghost-outfield-v1",
            "ghost-outfield-position-only-v1",
        )
    },
    "silly-kicks/xsuccess-v1": "silly_kicks/xsuccess/weights/MODEL_CARD.md",
}


def card_bytes(model_card: str | Path) -> bytes:
    """The card exactly as the Hub serves it: LF line endings. A Windows checkout with
    core.autocrlf=true holds CRLF; every Hub README is LF (measured 2026-10-02)."""
    card = Path(model_card)
    if not card.is_file():
        raise SystemExit(f"model card {card} does not exist (required for a real publish, uploaded as README.md).")
    return card.read_bytes().replace(b"\r\n", b"\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hub_readme(api: Any, repo_id: str) -> bytes | None:
    """The repo's current README.md bytes (fresh download), or None when it has none."""
    files = {s.rfilename for s in (api.model_info(repo_id).siblings or [])}
    if "README.md" not in files:
        return None
    path = api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", force_download=True)
    return Path(path).read_bytes()


def publish_card_only(api: Any, repo_id: str, *, root: str | Path = ".", verify_only: bool = False) -> dict:
    """The card-only seam (ADR-088 amendment): push the registered card as README.md of an EXISTING
    model repo, then read it back.

    Card-only pushes used to be ad-hoc README uploads (1b56ad8, aae6fdb) with no guard and no read-back.
    Refuses an unregistered repo before any network. ``api.model_info`` raises on a missing repo (a
    card-only push never creates one). An unchanged card is not re-uploaded. After an upload the README is
    re-downloaded and must be byte-identical, else ``SystemExit``. ``verify_only`` reports and stops.
    """
    if repo_id not in CARD_SOURCE:
        raise SystemExit(f"{repo_id} is not a registered Hub repo (CARD_SOURCE) -- register it first.")
    data = card_bytes(Path(root) / CARD_SOURCE[repo_id])
    before = _hub_readme(api, repo_id)
    out = {
        "repo_id": repo_id,
        "card": CARD_SOURCE[repo_id],
        "card_sha256": _sha256(data),
        "hub_sha256_before": None if before is None else _sha256(before),
        "changed": before != data,
        "uploaded": False,
    }
    if verify_only or not out["changed"]:
        return out
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"docs: model card from {CARD_SOURCE[repo_id]}",
    )
    after = _hub_readme(api, repo_id)
    if after is None or after != data:  # None: the README vanished (narrows for the type checker too)
        raise SystemExit(f"CARD READ-BACK MISMATCH: {repo_id} README.md != {CARD_SOURCE[repo_id]} after upload.")
    out.update(uploaded=True, hub_sha256_after=_sha256(after))
    return out


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
    (``exist_ok=True``) so this is safe for both a brand-new repo and a re-publish. The card is staged
    through :func:`card_bytes` (LF line endings), so this seam and :func:`publish_card_only` publish
    identical bytes for the same card.
    """
    data = card_bytes(model_card)  # before any network: a missing card refuses with no repo created
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        for f in Path(artifact_dir).iterdir():
            if f.is_file():
                shutil.copy2(f, stage / f.name)
        (stage / "README.md").write_bytes(data)
        upload_model_only(api, str(stage), repo_id)
