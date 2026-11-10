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
        p = prefix or ""
        return [_File(f) for f in self.files if f.startswith(p) and (recursive or "/" not in f[len(p) :])]

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


class _CountingApi(_Api):
    """``list_bucket_tree`` honours ``recursive`` and every call is counted (HFU spec 16.4)."""

    def __init__(self, files):
        super().__init__(files)
        self.lists: list[tuple[str, bool]] = []
        self.downloads = 0
        self.added: list = []

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        self.lists.append((prefix or "", bool(recursive)))
        return super().list_bucket_tree(bucket_id, prefix, recursive=recursive)

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False):
        self.downloads += 1
        super().download_bucket_files(bucket_id, files, raise_on_missing_files=raise_on_missing_files)

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
        self.added.extend(add or [])
        for data, path in add or []:
            self.files[path] = data


_TREE = {"rk/p/s/t/k.json": b"{}\n", "rk/p/s/t/k.sha256": b"x\n", "rk/p/s/t/sub/deep.json": b"", "rk/p/u/a.json": b"1"}


def test_one_recursive_listing_serves_every_directory():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "rk", api=api)
    assert s.names("p/s/t") == {"k.json", "k.sha256"}
    assert s.names("p/u") == {"a.json"}
    assert s.names("p/s/t/sub") == {"deep.json"}
    assert s.names("nowhere") == set()
    assert api.lists == [("rk/", True)]


def test_reads_are_memoized_including_a_missing_file():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "rk", api=api)
    assert s.read_bytes("p/u/a.json") == s.read_bytes("p/u/a.json") == b"1"
    assert s.read_bytes("p/u/missing.json") is None and s.read_bytes("p/u/missing.json") is None
    assert api.downloads == 2


def test_an_upload_is_visible_to_the_next_names_and_read():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "rk", api=api)
    assert s.names("_campaign") == set() and s.read_bytes("_campaign/r.json") is None
    s.upload_bytes("_campaign/r.json", b"{}")
    assert s.names("_campaign") == {"r.json"} and s.read_bytes("_campaign/r.json") == b"{}"
    assert len(api.lists) == 1 and api.downloads == 1


def test_an_empty_prefix_store_never_lists_the_whole_bucket():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "", api=api)
    assert s.names("rk/p/u") == {"a.json"}
    assert api.lists == [("rk/p/u/", False)]
