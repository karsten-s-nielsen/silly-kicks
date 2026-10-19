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
