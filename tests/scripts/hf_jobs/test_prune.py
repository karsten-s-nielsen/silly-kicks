"""`campaign prune` (HFU fit-units spec C3): list and delete the scratch run keys of superseded commits.

Deletion is irreversible, so every refusal has a test, and the fake bucket proves that nothing outside the
approved commits' run keys is touched. Fakes only (no network).
"""

from __future__ import annotations

import types

import pytest

from scripts.hf_jobs import prune as P
from scripts.hf_jobs.campaign import CampaignJob

HEAD = "aaaaaaa" + "0" * 33
OLD = "bbbbbbb"
OLDER = "ccccccc"


def _campaign(name: str = "cycle1"):
    return types.SimpleNamespace(
        name=name,
        bucket="org/scratch",
        jobs=(
            CampaignJob("xsuccess", "public", "sb_open", ("default",)),
            CampaignJob("ghost_gk", "owner", "tc3", ("default", "full")),
        ),
    )


class _Bucket:
    """A fake bucket: ``{path: size}``; top-level folders exist while they hold a file or are in ``empty``."""

    def __init__(self) -> None:
        self.files: dict[str, int] = {}
        self.empty: set[str] = set()
        self.deleted: list[str] = []
        self.jobs: list = []
        self.keep: set[str] = set()  # paths a delete call leaves behind (a partial delete)
        self.fail_403 = False
        self.fail_with: BaseException | None = None

    def add(self, run_key: str, n: int, size: int = 10) -> None:
        for i in range(n):
            self.files[f"{run_key}/stage/tok/u{i:03d}.parquet"] = size
        if n == 0:
            self.empty.add(run_key)

    def list_bucket_tree(self, bucket, prefix="", recursive=False):
        assert bucket == "org/scratch"
        if not recursive:
            assert prefix == ""  # prune lists the top level once, never the whole bucket recursively
            tops = sorted({p.split("/")[0] for p in self.files} | self.empty)
            return [types.SimpleNamespace(type="directory", path=t) for t in tops]
        assert prefix.endswith("/") and prefix.count("/") == 1  # one run key at a time
        return [
            types.SimpleNamespace(type="file", path=p, size=s)
            for p, s in sorted(self.files.items())
            if p.startswith(prefix)
        ]

    def batch_bucket_files(self, bucket, *, delete):
        if self.fail_with is not None:
            raise self.fail_with
        if self.fail_403:
            raise RuntimeError("403 Forbidden: Private storage limit reached for org")
        assert len(delete) <= 1000
        for p in delete:
            if p not in self.keep:
                self.files.pop(p)
                self.deleted.append(p)
        self.empty -= {p.split("/")[0] for p in delete}

    def list_jobs(self, labels=None):
        assert labels == {"campaign": "cycle1"}
        return self.jobs


def _world() -> _Bucket:
    b = _Bucket()
    b.add(f"cycle1-{HEAD[:7]}-xsuccess-public", 3)
    b.add(f"cycle1-{OLD}-xsuccess-public", 4)
    b.add(f"cycle1-{OLD}-ghost_gk-owner", 0)  # a run key with zero files
    b.add(f"cycle1-{OLDER}-xsuccess-public", 2)
    b.add(f"cycle1-{OLDER}-ghost_gk-owner-r1", 2)  # a replica store
    b.add(f"cycle1b-{OLD}-xsuccess-public", 5)  # another campaign whose name starts with this one's
    b.add(f"cycle1-{OLD}-unknown_job-public", 5)  # not a job of this campaign
    b.add("_campaign", 2)
    b.add("_staged", 2)
    b.add("_canary", 2)
    return b


def _run(b: _Bucket, *, delete: bool = False, approved=(), manifest=None, lines=None) -> dict:
    return P.prune(
        _campaign(),
        head=HEAD,
        delete=delete,
        approved=approved,
        api=b,
        manifest_commits=manifest or {},
        say=(lines.append if lines is not None else lambda _s: None),
    )


def test_listing_holds_only_this_campaigns_run_keys():
    got = P.run_keys_by_commit(_campaign(), _world())
    keys = sorted(r["run_key"] for rows in got.values() for r in rows)
    assert keys == [
        f"cycle1-{HEAD[:7]}-xsuccess-public",
        f"cycle1-{OLD}-ghost_gk-owner",
        f"cycle1-{OLD}-xsuccess-public",
        f"cycle1-{OLDER}-ghost_gk-owner-r1",
        f"cycle1-{OLDER}-xsuccess-public",
    ]
    assert got[OLD] == [
        {"run_key": f"cycle1-{OLD}-ghost_gk-owner", "files": 0, "bytes": 0},
        {"run_key": f"cycle1-{OLD}-xsuccess-public", "files": 4, "bytes": 40},
    ]


def test_listing_excludes_head():
    out = _run(_world())
    assert sorted(out["listing"]) == [OLD, OLDER] and out["ok"] is True and out["deleted"] == {}


def test_delete_removes_exactly_the_approved_commit():
    b = _world()
    before = dict(b.files)
    out = _run(b, delete=True, approved=[OLD])
    gone = {p for p in before if p not in b.files}
    assert gone == {p for p in before if p.startswith(f"cycle1-{OLD}-xsuccess-public/")}
    assert out["ok"] is True
    assert out["deleted"][OLD] == [
        {"run_key": f"cycle1-{OLD}-ghost_gk-owner", "files": 0, "bytes": 0, "files_left": 0},
        {"run_key": f"cycle1-{OLD}-xsuccess-public", "files": 4, "bytes": 40, "files_left": 0},
    ]


def test_a_run_key_with_zero_files_is_listed_and_deleted_without_an_error():
    out = _run(_world(), delete=True, approved=[OLD])
    assert {"run_key": f"cycle1-{OLD}-ghost_gk-owner", "files": 0, "bytes": 0, "files_left": 0} in out["deleted"][OLD]


def test_a_replica_run_key_is_listed():
    assert f"cycle1-{OLDER}-ghost_gk-owner-r1" in [r["run_key"] for r in _run(_world())["listing"][OLDER]]


def test_delete_without_an_approval_refuses():
    b = _world()
    with pytest.raises(P.PruneRefusedError, match="approval"):
        _run(b, delete=True, approved=[])
    assert b.deleted == []


def test_delete_of_head_refuses():
    b = _world()
    with pytest.raises(P.PruneRefusedError, match="HEAD"):
        _run(b, delete=True, approved=[OLD, HEAD[:7]])
    assert b.deleted == []  # the refusal comes before any delete


def test_delete_with_a_live_job_refuses():
    b = _world()
    b.jobs = [
        types.SimpleNamespace(labels={"commit": OLD + "1" * 33}, status=types.SimpleNamespace(stage="RUNNING")),
        types.SimpleNamespace(labels={"commit": OLDER + "1" * 33}, status=types.SimpleNamespace(stage="COMPLETED")),
    ]
    with pytest.raises(P.PruneRefusedError, match="non-terminal"):
        _run(b, delete=True, approved=[OLD])
    assert b.deleted == []
    assert _run(b, delete=True, approved=[OLDER])["ok"] is True  # a terminal job does not block


def test_delete_of_an_unknown_commit_refuses():
    b = _world()
    with pytest.raises(P.PruneRefusedError, match="no run key"):
        _run(b, delete=True, approved=["ddddddd"])
    assert b.deleted == []


def test_a_partial_delete_reports_files_left_and_is_not_ok():
    b = _world()
    stuck = f"cycle1-{OLD}-xsuccess-public/stage/tok/u001.parquet"
    b.keep = {stuck}
    out = _run(b, delete=True, approved=[OLD])
    assert out["ok"] is False
    assert [r["files_left"] for r in out["deleted"][OLD]] == [0, 1]
    b.keep = set()
    again = _run(b, delete=True, approved=[OLD])  # a second run deletes what is left
    assert again["ok"] is True and stuck not in b.files


def test_a_403_stops_with_the_storage_message():
    b = _world()
    b.fail_403 = True
    lines: list[str] = []
    out = _run(b, delete=True, approved=[OLD, OLDER], lines=lines)
    assert out["ok"] is False and "storage limit" in out["stopped"]
    assert b.deleted == [] and OLDER not in out["deleted"]  # the run stops at the first refusal
    assert any("storage limit" in ln for ln in lines)


class _HttpError(RuntimeError):
    """An error with the ``response.status_code`` that `huggingface_hub` HTTP errors carry."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = types.SimpleNamespace(status_code=status_code)


def test_a_403_status_code_stops_with_the_storage_message():
    b = _world()
    b.fail_with = _HttpError("Forbidden", 403)
    out = _run(b, delete=True, approved=[OLD])
    assert out["ok"] is False and "storage limit" in out["stopped"]


def test_another_error_with_the_digits_403_is_raised():
    """The digits 403 in an error text (a path, a byte count) are not the storage limit."""
    b = _world()
    b.fail_with = RuntimeError("upload of 14031 bytes to u403.parquet timed out")
    with pytest.raises(RuntimeError, match="timed out"):
        _run(b, delete=True, approved=[OLD])


def test_a_manifest_commit_prints_the_notice():
    lines: list[str] = []
    _run(_world(), delete=True, approved=[OLD], manifest={OLD: 25}, lines=lines)
    assert f"commit {OLD} is the training_commit of 25 shipped artifact(s)" in "\n".join(lines)
    quiet: list[str] = []
    _run(_world(), delete=True, approved=[OLDER], manifest={OLD: 25}, lines=quiet)
    assert not any("training_commit" in ln for ln in quiet)


def test_manifest_commits_counts_by_commit7():
    got = P.manifest_commits()
    assert got and all(len(k) == 7 for k in got) and sum(got.values()) >= 25
