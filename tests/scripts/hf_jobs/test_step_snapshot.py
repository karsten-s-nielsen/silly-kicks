"""`StoreSnapshot` reads a campaign store in bulk, not one file per API call (TF-85).

Run 1's controller step took 804 s: 42 serial ``--plan`` subprocesses and, for every done stage of every
instance, one ``list_bucket_tree`` plus one ``download_bucket_files`` per ``.ledger.json`` record. A step now
lists each distinct store once and downloads the small record files of a directory in one batch, with the
bytes cached by xet hash across ``--watch`` steps. These tests pin that read behaviour against a dict-backed
bucket that logs every API call, so "one call, not N" is an assertion, not a measurement."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import threading
from typing import Any

from scripts.hf_jobs.store import BucketStore, StoreSnapshot


class Entry:
    """A `list_bucket_tree` entry: a file carries a xet hash (its content address)."""

    def __init__(self, path: str, data: bytes) -> None:
        self.type = "file"
        self.path = path
        self.xet_hash = hashlib.sha256(data).hexdigest()


class Bucket:
    """A dict-backed HF bucket API. ``calls`` logs every call in order as ``(name, detail)``."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.calls: list[tuple[str, Any]] = []
        self._lock = threading.Lock()

    def _log(self, name: str, detail: Any) -> None:
        with self._lock:
            self.calls.append((name, detail))

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        pre = prefix or ""
        self._log("list_bucket_tree", (pre, bool(recursive)))
        hits = [p for p in self.files if p.startswith(pre)]
        return [Entry(p, self.files[p]) for p in hits if recursive or "/" not in p[len(pre) :]]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False):
        self._log("download_bucket_files", [src for src, _ in files])
        for src, dest in files:
            if src in self.files:
                pathlib.Path(dest).write_bytes(self.files[src])

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None):
        self._log("batch_bucket_files", {"add": len(add or []), "copy": len(copy or []), "delete": len(delete or [])})
        for data, dest in add or []:
            self.files[dest] = data


def _json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True) + "\n").encode()


RK = "campaign-abc1234-ghost_gk-owner"
GEN = "ghost_gk/extract/t_extract"


def _world() -> dict[str, bytes]:
    """A store with a generation: three unit ledgers, a profile, and a large result file beside them."""
    files = {
        f"{RK}/{GEN}/u0.ledger.json": _json({"key": "u0", "outcome": "done", "wall_s": 30.0}),
        f"{RK}/{GEN}/u1.ledger.json": _json({"key": "u1", "outcome": "done", "wall_s": 31.0}),
        f"{RK}/{GEN}/u2.ledger.json": _json({"key": "u2", "outcome": "done", "wall_s": 32.0}),
        f"{RK}/{GEN}/_profile.json": _json({"efficiency": 0.8}),
        f"{RK}/{GEN}/u0.parquet": b"\x00" * 1024,  # a large result file beside the records
    }
    return files


def _snap(files: dict[str, bytes], content: dict | None = None) -> tuple[StoreSnapshot, Bucket]:
    bucket = Bucket(files)
    return StoreSnapshot(BucketStore("ns/b", RK, api=bucket), content), bucket


def test_the_store_is_listed_once_then_names_are_served_from_memory() -> None:
    snap, bucket = _snap(_world())
    assert snap.names(GEN) == {"u0.ledger.json", "u1.ledger.json", "u2.ledger.json", "_profile.json", "u0.parquet"}
    snap.names(GEN)
    snap.list_tree("")
    assert bucket.count("list_bucket_tree") == 1  # one recursive listing covers every later names/list_tree


def test_the_one_listing_is_recursive() -> None:
    snap, bucket = _snap(_world())
    snap.names(GEN)
    assert bucket.calls[0] == ("list_bucket_tree", (f"{RK}/", True))


def test_a_missing_file_is_known_from_the_listing_without_a_download() -> None:
    snap, bucket = _snap(_world())
    assert snap.read_bytes(f"{GEN}/nope.ledger.json") is None
    assert bucket.count("download_bucket_files") == 0


def test_the_record_files_of_a_directory_are_downloaded_in_one_batch() -> None:
    snap, bucket = _snap(_world())
    first = snap.read_bytes(f"{GEN}/u0.ledger.json")
    assert first is not None
    assert json.loads(first)["key"] == "u0"
    # reading the first record pulled every record sibling (ledgers + profile) in ONE call, not one per file
    assert bucket.count("download_bucket_files") == 1
    pulled = bucket.calls[-1][1]
    assert all(p.endswith(".ledger.json") or p.endswith("_profile.json") for p in pulled)
    assert len(pulled) == 4  # u0, u1, u2 ledgers + _profile.json, never the .parquet
    # the siblings are now cached: no further download
    snap.read_bytes(f"{GEN}/u1.ledger.json")
    snap.read_bytes(f"{GEN}/_profile.json")
    assert bucket.count("download_bucket_files") == 1


def test_a_large_result_file_is_read_alone_never_bulk_pulled() -> None:
    snap, bucket = _snap(_world())
    snap.read_bytes(f"{GEN}/u0.parquet")
    assert bucket.count("download_bucket_files") == 1
    assert bucket.calls[-1][1] == [f"{RK}/{GEN}/u0.parquet"]  # only the result file, not the record siblings


def test_the_content_cache_carries_across_steps_so_an_unchanged_store_downloads_nothing() -> None:
    content: dict[str, bytes] = {}
    snap1, _ = _snap(_world(), content)
    snap1.read_bytes(f"{GEN}/u0.ledger.json")
    snap2, bucket2 = _snap(_world(), content)  # a later --watch step over the same (unchanged) store
    got = snap2.read_bytes(f"{GEN}/u1.ledger.json")
    assert got is not None and json.loads(got)["key"] == "u1"
    assert bucket2.count("download_bucket_files") == 0  # every byte came from the shared content cache


def test_read_after_write_serves_the_uploaded_bytes_and_moves_the_digest() -> None:
    snap, _ = _snap(_world())
    before = snap.digest()
    snap.upload_bytes("_campaign/preflight/report.json", b"hi")
    assert snap.read_bytes("_campaign/preflight/report.json") == b"hi"
    assert snap.digest() != before


def test_a_snapshot_can_be_copied() -> None:
    snap, _ = _snap(_world())
    copy.copy(snap)  # __getattr__ must not recurse before _store is set


def test_xet_hashes_are_every_content_address_in_the_listing() -> None:
    snap, _ = _snap(_world())
    assert snap.xet_hashes() == {Entry(p, d).xet_hash for p, d in _world().items()}


# ---- the controller step reads through snapshots (TF-85) -----------------------------------------

from scripts.hf_jobs import campaign as C  # noqa: E402
from tests.scripts.hf_jobs import _step_world  # noqa: E402

#: The step's report over the world: the read-path change must not move a single decision. The planner
#: decisions are the intended post-HFU ones. The ghost_gk/default run stage keeps its four pending units in ONE
#: 8-worker job: budget_floor is now wired (10 x startup_s, reviewer B D1-IMPL-10), so the planner does not fan
#: four tiny units into four startup-dominated jobs -- the eight workers split them within one job (R1
#: saturated, R2 split by worker). A first run with no recorded reduce_wall takes the generous profile ceiling
#: for its inline reduce (R3), and every planned action carries the nullable T8 shard_id/n_shards fields plus
#: the T11 inner_cap_bytes field (None for a non-inner stage). The read path is still decision-neutral: only
#: these planner outputs changed, nothing in done/failed/profile routing.
BASELINE = {
    "tf85-abcdef1-ghost_gk-owner/default": {
        "action": "run",
        "jobs": [],
        "n_jobs": 1,
        "planned": {
            "cap_bytes": 4026531840,
            "escalate": False,
            "inner_cap_bytes": None,
            "inner_workers": 1,
            "kind": "run",
            "n_jobs": 1,
            "n_shards": None,
            "partition": [["u6", "u7", "u8", "u9"]],
            "peak_rss_bytes": 1000000000,
            "predicted_cost_usd": 0.0027916666666666667,
            "predicted_job_wall_s": 335.0,
            "profile": None,
            "reason": None,
            "shard_id": None,
            "stages": ["extract"],
            "timeout_s": 7200,
            "workers": 8,
        },
        "stages": ["extract"],
    },
    "tf85-abcdef1-ghost_gk-owner/position_only": {
        "action": "profile",
        "jobs": [],
        "n_jobs": 1,
        "planned": {
            "cap_bytes": None,
            "escalate": False,
            "inner_cap_bytes": None,
            "inner_workers": 1,
            "kind": "profile",
            "n_jobs": 1,
            "n_shards": None,
            "partition": None,
            "peak_rss_bytes": None,
            "predicted_cost_usd": None,
            "predicted_job_wall_s": None,
            "profile": 4,
            "reason": None,
            "shard_id": None,
            "stages": ["extract"],
            "timeout_s": 7200,
            "workers": 8,
        },
        "stages": ["extract"],
    },
    "tf85-abcdef1-win_probability-public/default": {
        "action": None,
        "failed": "tf85-abcdef1-win_probability-public/default: --plan failed (rc=1): boom",
        "jobs": [],
    },
    "tf85-abcdef1-xsuccess-public/default": {"action": "done", "failed": None, "jobs": []},
}


def test_the_step_report_is_unchanged_by_the_read_path_rewrite(monkeypatch) -> None:
    report, _ = _step_world.run_world(C, patch=monkeypatch.setattr)
    assert report == BASELINE  # no decision moved


def test_each_distinct_store_is_listed_once_recursively_per_step(monkeypatch) -> None:
    _, bucket = _step_world.run_world(C, patch=monkeypatch.setattr)
    lists = [d for n, d in bucket.calls if n == "list_bucket_tree"]
    assert len(lists) == 3  # one per distinct run_key (A shared by two variants, B, C), not one per stage
    assert all(recursive for _, recursive in lists)  # one recursive listing, never a per-directory listing
    assert {prefix.rstrip("/") for prefix, _ in lists} == {_step_world.RK_A, _step_world.RK_B, _step_world.RK_C}


def test_a_done_stages_unit_ledgers_are_downloaded_in_bulk_not_one_per_file(monkeypatch) -> None:
    _, bucket = _step_world.run_world(C, patch=monkeypatch.setattr)
    downloads = [d for n, d in bucket.calls if n == "download_bucket_files"]
    assert len(downloads) <= 2  # at most one batch per store with reads (was 9 per-file downloads before TF-85)
    big = max(downloads, key=len)
    assert len(big) == 7  # A's six extract ledgers + its profile, pulled in one call
    assert all(p.endswith(".ledger.json") or p.endswith("_profile.json") for p in big)


def test_an_unchanged_store_re_plans_only_a_failing_instance_and_downloads_nothing_next_step(monkeypatch) -> None:
    cache: dict = {}
    calls: list[str] = []
    _step_world.run_world(C, patch=monkeypatch.setattr, cache=cache, tree="t1", plan_calls=calls)
    before = len(calls)
    _, bucket2 = _step_world.run_world(C, patch=monkeypatch.setattr, cache=cache, tree="t1", plan_calls=calls)
    second = calls[before:]
    assert second == ["tf85-abcdef1-win_probability-public/default"]  # only the failing plan re-runs; rest cached
    assert bucket2.count("download_bucket_files") == 0  # every byte came from the content cache carried in `cache`


def test_a_tree_change_re_plans_every_instance(monkeypatch) -> None:
    cache: dict = {}
    calls: list[str] = []
    counts = []
    for tree in ("t1", "t1", "t2"):
        before = len(calls)
        _step_world.run_world(C, patch=monkeypatch.setattr, cache=cache, tree=tree, plan_calls=calls)
        counts.append(len(calls) - before)
    assert counts == [4, 1, 4]  # all four, then only the failing plan, then all four again on the tree change


def test_a_step_refuses_a_dirty_tree_and_launches_nothing(monkeypatch) -> None:
    # D1: on a dirty tree every `--plan` token becomes `dirty-<hash>`, so the computed generations no longer
    # match the stores written under the committed tokens -- a tree edit mid-run must refuse, not mis-plan.
    import pytest

    with pytest.raises(RuntimeError, match="dirty tree"):
        _step_world.run_world(C, patch=monkeypatch.setattr, tree="dirty-0f0f0f")


def test_a_replica_plans_on_its_own_costs_not_the_main_runs(monkeypatch) -> None:
    # a main run and its fresh replica share a generation TOKEN but live in different stores (NF7): the cost
    # history is measured per store, so the replica must never read the main run's measured costs.
    gen = "ghost_gk/extract/t"

    def store(rk: str, wall: float) -> StoreSnapshot:
        files = {
            f"{rk}/{gen}/u0.ledger.json": _json({"key": "u0", "outcome": "done", "wall_s": wall, "peak_rss_bytes": 1})
        }
        return StoreSnapshot(BucketStore("ns/b", rk, api=Bucket(files)))

    main = C.Instance("ghost_gk", "owner", "default", "m", (), (), "rk-main")
    repl = C.Instance("ghost_gk", "owner", "default", "m", (), (), "rk-r1")
    stage = {"name": "extract", "generation": gen, "done": 1, "history_key": "extract"}
    plans = {
        main.id: {"pipeline": "ghost_gk", "stages": [dict(stage)]},
        repl.id: {"pipeline": "ghost_gk", "stages": [dict(stage)]},
    }
    snaps = {"rk-main": store("rk-main", 30.0), "rk-r1": store("rk-r1", 90.0)}
    views = C.instance_views([main, repl], plans, snaps, cache={})
    assert views[main.id]["extract"]["per_unit"] == {"u0": 30.0}
    assert views[repl.id]["extract"]["per_unit"] == {"u0": 90.0}  # its own store's cost, not the main run's 30.0
