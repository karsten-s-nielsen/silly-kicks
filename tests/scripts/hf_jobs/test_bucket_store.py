from __future__ import annotations

import pathlib
import types

import pytest

from scripts.hf_jobs.store import BucketStore


class _File:
    def __init__(self, path):
        # xet_hash drives the server-side copy; the fake uses the source path as the opaque hash token.
        self.type, self.path, self.xet_hash = "file", path, path


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


def test_the_controller_handle_refuses_unit_results_and_accepts_its_prefixes():
    calls = []
    api = types.SimpleNamespace(batch_bucket_files=lambda b, add=(), delete=(): calls.append((add, delete)))
    st = BucketStore("ns/b", "rk", api=api)
    for rel in ("ghost_gk/extract/tok/0001.parquet", "ghost_gk/extract/tok/_output.parquet", "_ledger/x.json"):
        with pytest.raises(PermissionError):
            st.upload_bytes(rel, b"x")
    for rel in ("_actions/p/s/t/a.json", "_operator/p/s/t/u.json", "_attest/s/t/c.json"):
        st.upload_bytes(rel, b"x")
    assert len(calls) == 3


class _MutApi(_Api):
    """A dict-backed bucket with recursive listing and `batch_bucket_files(add=, copy=, delete=)`."""

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        pre = prefix or ""
        hits = [p for p in self.files if p.startswith(pre)]
        return [_File(p) for p in hits if recursive or "/" not in p[len(pre) :]]

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
        for data, dest in add or ():
            self.files[dest] = data
        for _repo_type, _repo_id, xet_hash, dest in copy or ():  # the real hf 2.2 copy-by-xet-hash 4-tuple
            self.files[dest] = self.files[xet_hash]  # the fake's xet_hash token IS the source path
        for path in delete or ():
            self.files.pop(path, None)


def test_quarantine_generation_moves_the_generation_and_its_attestations():
    files = {
        "rk/p/s/t/0001.parquet": b"a",
        "rk/p/s/t/0001.sha256": b"h",
        "rk/p/s/t/_output.json": b"o",
        "rk/_attest/s/t/c1.json": b"{}",
        "rk/p/s/other/0001.parquet": b"keep",
    }
    api = _MutApi(dict(files))
    moved = BucketStore("b/x", "rk", api=api).quarantine_generation("p", "s", "t", {"reason": "canary mismatch"})
    assert sorted(moved) == sorted(k for k in files if "/t/" in k)
    assert not any(k.startswith(("rk/p/s/t/", "rk/_attest/s/t/")) for k in api.files)
    q = [k for k in api.files if k.startswith("rk/_quarantine/s/t/")]
    assert len(q) == 5 and any(k.endswith("record.json") for k in q)  # 4 moved files + the record
    assert api.files["rk/p/s/other/0001.parquet"] == b"keep"


def test_store_size_sums_every_file_under_the_run_key():
    class _Sized(_MutApi):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
            return [
                types.SimpleNamespace(type="file", path=p, size=len(v))
                for p, v in self.files.items()
                if p.startswith(prefix or "")
            ]

    api = _Sized({"rk/a": b"12345", "rk/b/c": b"123", "other/x": b"9" * 50})
    assert BucketStore("b/x", "rk", api=api).size_bytes() == 8


def test_the_fake_copy_shape_matches_the_real_batch_bucket_files_signature():
    """CCTP-IMPL-18: the quarantine copy uses hf `batch_bucket_files(copy=[(repo_type, repo_id, xet_hash,
    dest)])`. Real HF never runs locally (hard rule), so this binds the fake (`_MutApi.batch_bucket_files`) to
    the real client's signature: `copy` is a list of 4-tuples, and add/copy/delete stay keyword-only."""
    import inspect

    hub = pytest.importorskip("huggingface_hub")
    method = getattr(hub.HfApi, "batch_bucket_files", None)
    assert method is not None, "huggingface_hub.HfApi.batch_bucket_files is gone (the quarantine copy relies on it)"
    params = inspect.signature(method).parameters
    assert {"add", "copy", "delete"} <= set(params)
    assert all(params[p].kind is inspect.Parameter.KEYWORD_ONLY for p in ("add", "copy", "delete"))
    assert "tuple[str,str,str,str]" in str(params["copy"].annotation).replace(" ", "")  # the copy-by-xet-hash 4-tuple
