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


def test_bucket_store_read_many_makes_one_call_for_a_batch():
    """HFU launch top-up spec §22.2a (F3): one `download_bucket_files` call for every file of a batch."""
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "rk", api=api)
    got = s.read_many(["p/u/a.json", "p/s/t/k.json", "p/u/missing.json"])
    assert got == {"p/u/a.json": b"1", "p/s/t/k.json": b"{}\n", "p/u/missing.json": None}
    assert api.downloads == 1


def test_bucket_store_read_many_memoizes():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "rk", api=api)
    s.read_many(["p/u/a.json", "p/u/missing.json"])
    assert s.read_many(["p/u/a.json", "p/u/missing.json"]) == {"p/u/a.json": b"1", "p/u/missing.json": None}
    assert s.read_bytes("p/u/a.json") == b"1" and s.read_bytes("p/u/missing.json") is None
    assert s.read_many([]) == {}
    assert api.downloads == 1  # the second batch, the two reads and the empty batch made no call
    assert s.read_many(["p/u/a.json", "p/s/t/k.json"]) == {"p/u/a.json": b"1", "p/s/t/k.json": b"{}\n"}
    assert api.downloads == 2  # one call, for the one rel that is not memoized


def test_an_empty_prefix_store_never_lists_the_whole_bucket():
    api = _CountingApi(dict(_TREE))
    s = BucketStore("b/x", "", api=api)
    assert s.names("rk/p/u") == {"a.json"}
    assert api.lists == [("rk/p/u/", False)]


# ---- the plan's request count with the FileSet member check (HFU fit-units spec C4) -------------------


def _plan_requests(tmp_path, *, fileset: bool) -> tuple[list, int, dict]:
    """Run a one-stage toy pipeline into a local store, serve its files from a counting fake bucket, and plan it.
    ``fileset``: the unit result and the stage output are FileSets; else both are frames."""
    import pandas as pd

    from scripts import _stages as s

    def work(key, ctx):
        return s.FileSet({"X.npy": b"x", "sub/ids.npy": b"i"}) if fileset else pd.DataFrame({"a": [1]})

    def reduce(results, ctx):
        return s.FileSet({"out.bin": b"o"}) if fileset else pd.DataFrame({"b": [2]})

    pipeline = s.Pipeline(
        name="t",
        module="tests.scripts._toy_pipeline",
        stages=(s.Stage(name="prep", units=lambda c: ["u1", "u2"], work=work, reduce=reduce),),
        params=(),
    )
    prov = {"commit": "c" * 40, "tree_state": "clean", "code_identity": "c" * 40}
    local = s.make_context(pipeline, {}, prov, store=s.LocalStore(tmp_path), data_root=None)
    for key in ("u1", "u2"):
        s.run_unit(local, pipeline.stage("prep"), key, worker="w0")
    s.reduce_stage(local, "prep")
    files = {f"rk/{p.relative_to(tmp_path).as_posix()}": p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    api = _CountingApi(files)
    ctx = s.make_context(pipeline, {}, prov, store=BucketStore("b/x", "rk", api=api), data_root=None)
    record = s.plan(ctx)["stages"][0]
    return api.lists, api.downloads, record


def test_the_member_check_adds_one_marker_read_per_fileset_result_and_no_listing(tmp_path):
    lists_f, reads_f, rec_f = _plan_requests(tmp_path / "f", fileset=True)
    lists_p, reads_p, rec_p = _plan_requests(tmp_path / "p", fileset=False)
    assert lists_f == lists_p == [("rk/", True)]  # ONE recursive listing, with and without FileSets
    assert reads_f - reads_p == 3  # two unit markers and the output marker; no member is read
    assert (rec_f["done"], rec_f["reduced"], rec_f["lost_units"], rec_f["lost_output"]) == (2, True, [], False)
    assert (rec_p["done"], rec_p["reduced"]) == (2, True)
