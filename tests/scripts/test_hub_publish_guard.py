"""The model-only publish guard (`scripts/_hub_publish.upload_model_only`).

Born from the 4.94.0 incident: `upload_folder(artifact_dir)` shipped the trainer's co-located
`_feature_cache/`, `shards/` (raw restricted frames) and `_probe_sample/` to public repos. The guard
uploads a fixed allowlist and fail-closes on any nested repo path afterwards.
"""

from __future__ import annotations

import pytest

from scripts._hub_publish import MODEL_ONLY_ALLOWLIST, upload_model_only


class _Sibling:
    def __init__(self, rfilename: str) -> None:
        self.rfilename = rfilename


class _Info:
    def __init__(self, files: list[str]) -> None:
        self.siblings = [_Sibling(f) for f in files]


class _FakeApi:
    """Records the upload call and returns a controllable file listing from ``model_info``."""

    def __init__(self, repo_files: list[str]) -> None:
        self._files = repo_files
        self.upload_kwargs: dict = {}

    def upload_folder(self, **kwargs) -> None:
        self.upload_kwargs = kwargs

    def model_info(self, repo_id: str):
        return _Info(self._files)


def test_allowlist_is_model_only_and_excludes_training_internals():
    # The allowlist must cover both model file shapes and the metadata trio -- and nothing that
    # names a training-internal directory.
    assert set(MODEL_ONLY_ALLOWLIST) == {
        "model.json",
        "rfcde_weights.npz",
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
