"""The model-only publish guard (`scripts/_hub_publish.upload_model_only`).

Born from the 4.94.0 incident: `upload_folder(artifact_dir)` shipped the trainer's co-located
`_feature_cache/`, `shards/` (raw restricted frames) and `_probe_sample/` to public repos. The guard
uploads a fixed allowlist and fail-closes on any nested repo path afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._hub_publish import MODEL_ONLY_ALLOWLIST, publish_model_with_card, upload_model_only


class _Sibling:
    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


class _Info:
    def __init__(self, files: list[str]) -> None:
        self.siblings = [_Sibling(f) for f in files]


class _FakeApi:
    """Records the upload/create calls and returns a controllable file listing from ``model_info``."""

    def __init__(self, repo_files: list[str]) -> None:
        self._files = repo_files
        self.upload_kwargs: dict = {}
        self.created: dict | None = None
        self.staged_files: set[str] = set()

    def create_repo(self, **kwargs) -> None:
        self.created = kwargs

    def upload_folder(self, **kwargs) -> None:
        self.upload_kwargs = kwargs
        # Capture what was actually staged, when the folder is real (the publish_model_with_card path);
        # the upload_model_only unit tests pass a non-existent "artifact-dir", so guard on is_dir().
        folder = Path(kwargs["folder_path"])
        self.staged_files = {f.name for f in folder.iterdir() if f.is_file()} if folder.is_dir() else set()

    def model_info(self, repo_id: str):
        return _Info(self._files)


def test_allowlist_is_model_only_and_excludes_training_internals():
    # The allowlist must cover every model file shape (model.json = xShot/xCross booster;
    # rfcde_weights.npz = ghost-GK; model.npz = ghost-outfield) and the metadata trio -- and nothing
    # that names a training-internal directory.
    assert set(MODEL_ONLY_ALLOWLIST) == {
        "model.json",
        "rfcde_weights.npz",
        "model.npz",
        "metadata.json",
        "metrics.json",
        "SHA256SUMS",
        "README.md",
    }
    assert not any(bad in "".join(MODEL_ONLY_ALLOWLIST) for bad in ("_feature_cache", "shards", "_probe_sample"))


def test_upload_passes_the_allowlist_as_allow_patterns():
    api = _FakeApi(["model.json", "metadata.json", "metrics.json", "SHA256SUMS"])
    upload_model_only(api, "artifact-dir", "silly-kicks/xshot-occurrence-v1")
    assert api.upload_kwargs["allow_patterns"] == list(MODEL_ONLY_ALLOWLIST)
    assert api.upload_kwargs["repo_type"] == "model"
    assert api.upload_kwargs["repo_id"] == "silly-kicks/xshot-occurrence-v1"


def test_clean_model_only_repo_passes():
    api = _FakeApi(["model.json", "metadata.json", "metrics.json", "SHA256SUMS", "README.md", ".gitattributes"])
    upload_model_only(api, "artifact-dir", "silly-kicks/xcross-attempt-v1")  # must not raise


def test_publish_model_with_card_creates_repo_and_stages_card(tmp_path):
    # The shared publish seam every publish_* script uses: it create_repo's (idempotent), and it
    # stages the artifact's model files + the model card AS README.md into the uploaded folder -- so
    # the card can never be silently dropped and a brand-new repo is created rather than erroring.
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "model.npz").write_bytes(b"weights")
    (art / "metadata.json").write_text("{}")
    (art / "SHA256SUMS").write_text("deadbeef  model.npz\n")
    card = tmp_path / "the-card.md"
    card.write_text("# model card\n")

    api = _FakeApi(["model.npz", "metadata.json", "SHA256SUMS", "README.md"])
    publish_model_with_card(api, str(art), "silly-kicks/ghost-outfield-v1", model_card=str(card))

    assert api.created == {"repo_id": "silly-kicks/ghost-outfield-v1", "repo_type": "model", "exist_ok": True}
    # The card was staged as README.md alongside the model files (not left behind).
    assert "README.md" in api.staged_files
    assert {"model.npz", "metadata.json", "SHA256SUMS"} <= api.staged_files


def test_publish_model_with_card_refuses_missing_card(tmp_path):
    api = _FakeApi([])
    with pytest.raises(SystemExit, match="does not exist"):
        publish_model_with_card(api, str(tmp_path), "silly-kicks/x", model_card=str(tmp_path / "nope.md"))
    assert api.created is None  # refused BEFORE create_repo / any network


@pytest.mark.parametrize(
    "leaked",
    [
        "shards/gradientsports__10502.parquet",
        "_probe_sample/frames.parquet",
        "_feature_cache/features.parquet",
    ],
)
def test_nested_path_after_upload_fails_closed(leaked):
    api = _FakeApi(["model.json", "metadata.json", "metrics.json", "SHA256SUMS", leaked])
    with pytest.raises(SystemExit) as exc:
        upload_model_only(api, "artifact-dir", "silly-kicks/ghost-gk-v1")
    assert "PUBLISH LEAK GUARD" in str(exc.value)
    assert leaked in str(exc.value)
