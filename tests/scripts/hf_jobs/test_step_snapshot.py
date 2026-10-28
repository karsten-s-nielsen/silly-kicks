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
import time
from typing import Any

from scripts.hf_jobs.store import BucketStore, StoreSnapshot


class Entry:
    """A `list_bucket_tree` entry: a file carries a xet hash (its content address) and, like the HF API's
    ``BucketFile``, an ``uploaded_at`` (None when the test does not set one, as a `StoreSnapshot` carries none)."""

    def __init__(self, path: str, data: bytes, uploaded_at: float | None = None) -> None:
        self.type = "file"
        self.path = path
        self.xet_hash = hashlib.sha256(data).hexdigest()
        self.uploaded_at = uploaded_at


class Dir:
    """A `list_bucket_tree` directory entry (a non-recursive listing returns one per directory below the prefix)."""

    def __init__(self, path: str) -> None:
        self.type = "directory"
        self.path = path


class Bucket:
    """A dict-backed HF bucket API. ``calls`` logs every call in order as ``(name, detail)``."""

    def __init__(self, files: dict[str, bytes], times: dict[str, float] | None = None) -> None:
        self.files = dict(files)
        self.times = dict(times or {})  # path -> uploaded_at (the HF API's BucketFile.uploaded_at)
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
        if recursive:
            return [Entry(p, self.files[p], self.times.get(p)) for p in hits]
        # non-recursive, as the HF API answers it: the files at this level and one entry per directory below it
        files = [Entry(p, self.files[p], self.times.get(p)) for p in hits if "/" not in p[len(pre) :]]
        dirs = sorted({pre + p[len(pre) :].split("/", 1)[0] for p in hits if "/" in p[len(pre) :]})
        return files + [Dir(d) for d in dirs]

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
#: the T11 inner_cap_bytes field (None for a non-inner stage), the HFU S3 item 5 `wio` mark (False: these
#: records carry no unit CPU, so no stage is WIO) and the S3 item 6 `gated` mark (False: no earlier generation).
#: The read path is still decision-neutral: only
#: these planner outputs changed, nothing in done/failed/profile routing. Instance ids name the tier (spec r4
#: D18 (c)).
BASELINE = {
    "tf85-abcdef1-ghost_gk-owner/owner/default": {
        "action": "run",
        "jobs": [],
        "n_jobs": 1,
        "planned": {
            "cap_bytes": 4026531840,
            "escalate": False,
            "gated": False,
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
            "wio": False,
            "workers": 8,
        },
        "stages": ["extract"],
    },
    "tf85-abcdef1-ghost_gk-owner/owner/position_only": {
        "action": "profile",
        "jobs": [],
        "n_jobs": 1,
        "planned": {
            "cap_bytes": None,
            "escalate": False,
            "gated": False,
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
            "wio": False,
            "workers": 8,
        },
        "stages": ["extract"],
    },
    "tf85-abcdef1-win_probability-public/public/default": {
        "action": None,
        "failed": "tf85-abcdef1-win_probability-public/public/default: --plan failed (rc=1): boom",
        "jobs": [],
    },
    "tf85-abcdef1-xsuccess-public/public/default": {"action": "done", "failed": None, "jobs": []},
}


def test_the_step_report_is_unchanged_by_the_read_path_rewrite(monkeypatch) -> None:
    report, _ = _step_world.run_world(C, patch=monkeypatch.setattr)
    assert report == BASELINE  # no decision moved


def test_each_distinct_store_is_listed_once_recursively_per_step(monkeypatch) -> None:
    _, bucket = _step_world.run_world(C, patch=monkeypatch.setattr)
    lists = [d for n, d in bucket.calls if n == "list_bucket_tree"]
    stores = [d for d in lists if d[0]]  # the bucket root ("") is the MEM history source's run-key listing
    assert len(stores) == 3  # one per distinct run_key (A shared by two variants, B, C), not one per stage
    assert all(recursive for _, recursive in stores)  # one recursive listing, never a per-directory listing
    assert {prefix.rstrip("/") for prefix, _ in stores} == {_step_world.RK_A, _step_world.RK_B, _step_world.RK_C}
    # S3 item 6 (owner ruling 2026-10-31, "Bucket search"): at most ONE non-recursive root listing per controller,
    # for the run keys of earlier campaigns; a snapshotted store's stage files come from its snapshot
    assert [d for d in lists if not d[0]] in ([], [("", False)])


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
    assert second == ["tf85-abcdef1-win_probability-public/public/default"]  # only the failing plan re-runs
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


def _peaks_bucket(gens: dict[str, dict]) -> Bucket:
    """A bucket with stage generations: ``{"<rk>/<pipeline>/<stage>/<token>": {"inputs": dict | None, "summary": bool,
    "units": {key: (peak, finished_ns)}}}``."""
    files: dict[str, bytes] = {}
    times: dict[str, float] = {}
    for gen, g in gens.items():
        if g.get("inputs") is not None:
            files[f"{gen}/_inputs.json"] = _json(g["inputs"])
        if g.get("summary"):
            files[f"{gen}/_summary.json"] = _json({"n_done": len(g["units"])})
        for key, (peak, fin) in g["units"].items():
            rel = f"{gen}/{key}.ledger.json"
            files[rel] = _json({"key": key, "outcome": "done", "peak_rss_bytes": peak, "finished_ns": fin})
            times[rel] = float(fin)  # the ledger's upload time mirrors its finish order (never concurrent)
    return Bucket(files, times)


_EXTRACT = {"pipeline": "ghost_gk", "stage": "extract", "params": {}, "declared": {"v": 1}, "dataset": "tc3"}


def _hk(inputs: dict) -> str:
    from scripts._stages import _history_key

    return _history_key(inputs)


def test_a_fresh_store_is_gated_by_an_earlier_campaign_with_the_same_computation() -> None:
    """S3 item 6, owner ruling 2026-10-31 ("Bucket search"; review S3B-02): a new commit's campaign starts in a fresh
    store; its first `extract` generation is gated by the REDUCED generation of the same computation (`history_key`)
    in another run key of the campaign bucket, the newest one by its ledgers."""
    cur = "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/cur"
    bucket = _peaks_bucket(
        {
            cur: {"inputs": None, "summary": False, "units": {}},
            "canary-aaaaaaa-ghost_gk-tc3/ghost_gk/extract/t1": {
                "inputs": {**_EXTRACT, "code": "aaaaaaa"},
                "summary": True,
                "units": {"m1": (2_000, 300), "m2": (5_510, 310)},
            },
            "cycle1-0000000-ghost_gk-tc3/ghost_gk/extract/t0": {
                "inputs": {**_EXTRACT, "code": "0000000"},
                "summary": True,
                "units": {"m1": (9, 100)},
            },
        }
    )
    got = C.bucket_unit_peaks(bucket, "ns/b", "ghost_gk", "ghost_gk", "extract", _hk(_EXTRACT), exclude=cur)
    assert got == {"m1": 2_000, "m2": 5_510}  # the newest matching generation, from another run key


def test_a_sibling_tier_or_a_partial_generation_never_gates() -> None:
    """Review S3B-03: a generation of another computation (a sibling tier's params) and an unreduced one (no
    `_summary.json` / `_inputs.json` yet) are never the history: no first-run relaxation from either."""
    cur = "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/cur"
    sibling = {**_EXTRACT, "params": {"arms_json": "tier-b"}}
    bucket = _peaks_bucket(
        {
            cur: {"inputs": None, "summary": False, "units": {}},
            "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/sib": {
                "inputs": sibling,
                "summary": True,
                "units": {"m1": (7, 900)},
            },
            "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/part": {
                "inputs": None,
                "summary": False,
                "units": {"m1": (8, 950)},
            },
            "cycle1-ccccccc-ghost_outfield-tc3/ghost_gk/extract/other": {
                "inputs": _EXTRACT,
                "summary": True,
                "units": {"m1": (6, 990)},
            },
        }
    )
    assert C.bucket_unit_peaks(bucket, "ns/b", "ghost_gk", "ghost_gk", "extract", _hk(_EXTRACT), exclude=cur) is None
    assert _hk(sibling) != _hk(_EXTRACT)  # non-vacuity: the sibling really is another computation


def test_the_peak_search_downloads_only_the_newest_generations_ledgers() -> None:
    """HFU canary run 2: the search chose the newest reduced generation of a stage but downloaded EVERY matching
    generation's unit ledgers to find it (win_probability: 13 generations x 3,961 ledgers, 591 s). Newest-first by
    the ledgers' upload time downloads only the winner's ledgers; an older matching generation's are never read."""
    cur = "cycle1-ddddddd-ghost_gk-tc3/ghost_gk/extract/cur"
    old = {"m1": (10, 100), "m2": (11, 110)}  # the OLDER generation: its ledgers must NOT be downloaded
    new = {"m1": (20, 200), "m2": (21, 210)}  # the newer generation (later finish/upload): the only one kept
    bucket = _peaks_bucket(
        {
            cur: {"inputs": None, "summary": False, "units": {}},
            "cycle1-aaaaaaa-ghost_gk-tc3/ghost_gk/extract/told": {"inputs": _EXTRACT, "summary": True, "units": old},
            "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/tnew": {"inputs": _EXTRACT, "summary": True, "units": new},
        }
    )
    got = C.bucket_unit_peaks(bucket, "ns/b", "ghost_gk", "ghost_gk", "extract", _hk(_EXTRACT), exclude=cur)
    assert got == {"m1": 20, "m2": 21}  # the newest generation's peaks
    ledger_dl = [
        paths
        for n, paths in bucket.calls
        if n == "download_bucket_files" and any(p.endswith(".ledger.json") for p in paths)
    ]
    assert len(ledger_dl) == 1  # one ledger download, not one per matching generation
    assert all("/told/" not in p for p in ledger_dl[0])  # the older generation's ledgers are never read


def test_the_peak_search_falls_back_to_the_exact_finish_scan_without_upload_times() -> None:
    """A run key read through the step snapshot carries no upload time; the search then keeps the exact
    `finished_ns` scan (download each candidate, newest finish wins), so the result is unchanged there."""
    cur = "cycle1-ddddddd-ghost_gk-tc3/ghost_gk/extract/cur"
    bucket = _peaks_bucket(
        {
            cur: {"inputs": None, "summary": False, "units": {}},
            "cycle1-aaaaaaa-ghost_gk-tc3/ghost_gk/extract/told": {
                "inputs": _EXTRACT,
                "summary": True,
                "units": {"m1": (10, 100)},
            },
            "cycle1-bbbbbbb-ghost_gk-tc3/ghost_gk/extract/tnew": {
                "inputs": _EXTRACT,
                "summary": True,
                "units": {"m1": (20, 200)},
            },
        }
    )
    bucket.times.clear()  # no upload time anywhere -> the fallback finished_ns scan
    got = C.bucket_unit_peaks(bucket, "ns/b", "ghost_gk", "ghost_gk", "extract", _hk(_EXTRACT), exclude=cur)
    assert got == {"m1": 20}  # newest by finished_ns, as before


def test_concurrent_peak_searches_list_each_store_once() -> None:
    """HFU canary run 2 (parallel): `unit_peak_views` searches the misses in a thread pool; a single-flight lock on
    the shared ``_reads`` must still list and download each store ONCE (review S3B-12). A controlled race: the first
    search is held inside `list_bucket_tree`; a second, concurrent search of the same store must wait on the
    in-flight entry, not issue its own listing."""
    gen = "cycle1-aaaaaaa-ghost_gk-tc3/ghost_gk/extract/t1"
    base_bucket = _peaks_bucket({gen: {"inputs": _EXTRACT, "summary": True, "units": {"m1": (7, 70)}}})
    entered, release = threading.Event(), threading.Event()

    class Held(Bucket):
        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
            if prefix and "ghost_gk/extract/" in prefix:
                entered.set()
                assert release.wait(5)
            return super().list_bucket_tree(bucket_id, prefix, recursive=recursive)

    bucket = Held(base_bucket.files, base_bucket.times)
    reads: dict = {C._INFLIGHT: {}}
    lock = threading.Lock()
    rk = gen.split("/", 1)[0]
    results: dict[int, Any] = {}

    def call(i: int) -> None:
        results[i] = C.bucket_unit_peaks(
            bucket,
            "ns/b",
            "ghost_gk",
            "ghost_gk",
            "extract",
            _hk(_EXTRACT),
            exclude="none",
            run_keys=[rk],
            snaps={},
            reads=reads,
            lock=lock,
        )

    t1 = threading.Thread(target=call, args=(0,))
    t1.start()
    assert entered.wait(5)  # t1 is inside the (held) listing of the store
    t2 = threading.Thread(target=call, args=(1,))
    t2.start()
    time.sleep(0.3)  # give t2 its chance to (wrongly) issue a second listing of the same store
    release.set()
    t1.join(5)
    t2.join(5)
    assert results[0] == results[1] == {"m1": 7}
    assert bucket.count("list_bucket_tree") == 1  # single-flight: one listing despite two concurrent searches


def test_a_failed_search_releases_its_waiters_and_leaves_no_in_flight_entry() -> None:
    """Reviews R5-OPT-01 / R5B-12 (BLOCKING): a `compute()` that raises -- the network I/O of a search CAN fail --
    must release the single-flight waiters and clear the in-flight entry, never leave them blocked in `ev.wait()`
    (a silent controller hang that would also poison `_reads` on later `--watch` steps). A woken waiter recomputes:
    a transient error recovers, so one worker's failure does not fail the rest."""
    gen = "cycle1-aaaaaaa-ghost_gk-tc3/ghost_gk/extract/t1"
    base_bucket = _peaks_bucket({gen: {"inputs": _EXTRACT, "summary": True, "units": {"m1": (7, 70)}}})
    entered, release = threading.Event(), threading.Event()

    class Flaky(Bucket):
        """The first listing of the store blocks, then raises (a transient HF error); later listings succeed."""

        def __init__(self, files: dict[str, bytes], times: dict[str, float]) -> None:
            super().__init__(files, times)
            self.attempts = 0

        def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
            if prefix and "ghost_gk/extract/" in prefix:
                self.attempts += 1
                if self.attempts == 1:
                    entered.set()
                    assert release.wait(5)
                    raise ConnectionError("transient HF API error")
            return super().list_bucket_tree(bucket_id, prefix, recursive=recursive)

    bucket = Flaky(base_bucket.files, base_bucket.times)
    reads: dict = {C._INFLIGHT: {}}
    lock = threading.Lock()
    rk = gen.split("/", 1)[0]
    outcomes: dict[int, Any] = {}

    def call(i: int) -> None:
        try:
            outcomes[i] = C.bucket_unit_peaks(
                bucket,
                "ns/b",
                "ghost_gk",
                "ghost_gk",
                "extract",
                _hk(_EXTRACT),
                exclude="none",
                run_keys=[rk],
                snaps={},
                reads=reads,
                lock=lock,
            )
        except Exception as exc:  # the owner's transient error surfaces to its own caller
            outcomes[i] = exc

    t1 = threading.Thread(target=call, args=(0,), daemon=True)  # the owner: its compute raises
    t1.start()
    assert entered.wait(5)
    t2 = threading.Thread(
        target=call, args=(1,), daemon=True
    )  # the waiter: must not hang (daemon: RED can't wedge exit)
    t2.start()
    time.sleep(0.3)
    release.set()
    t1.join(5)
    t2.join(5)
    assert not t1.is_alive() and not t2.is_alive()  # no hang
    assert isinstance(outcomes[0], ConnectionError)  # the error reached the owner's caller, not swallowed
    assert outcomes[1] == {"m1": 7}  # the woken waiter recomputed and recovered from the transient error
    assert not reads[C._INFLIGHT]  # the failed entry is cleared, so later --watch steps are not poisoned
    assert C.bucket_unit_peaks(
        bucket,
        "ns/b",
        "ghost_gk",
        "ghost_gk",
        "extract",
        _hk(_EXTRACT),
        exclude="none",
        run_keys=[rk],
        snaps={},
        reads=reads,
        lock=lock,
    ) == {"m1": 7}  # a later call still works


_CENSUS = {"pipeline": "territory_ranking_census", "stage": "rank", "params": {}, "declared": {}, "dataset": "x"}


def _peak_plan(pipeline: str, stage: str, generation: str, inputs: dict) -> dict:
    return {
        "pipeline": pipeline,
        "stages": [{"name": stage, "generation": generation, "history_key": _hk(inputs), "reduced": False}],
    }


def test_a_pipeline_named_unlike_its_job_finds_its_history_by_the_job_in_the_run_key() -> None:
    """Review S3B-11: a run key names its JOB (`{campaign}-{commit7}-{job}-{dataset}`); the job
    `build_territory_ranking_census` runs the pipeline `territory_ranking_census`, so a filter on the pipeline name
    never matched and the stage was never gated."""
    job, pipeline = "build_territory_ranking_census", "territory_ranking_census"
    cur_rk = f"cycle1-bbbbbbb-{job}-x"
    bucket = _peaks_bucket(
        {
            f"cycle1-aaaaaaa-{job}-x/{pipeline}/rank/t1": {
                "inputs": {**_CENSUS, "code": "aaaaaaa"},
                "summary": True,
                "units": {"u1": (4_000, 10)},
            },
        }
    )
    inst = C.Instance(job, "x", "default", "scripts.build_territory_ranking_census", (), (), cur_rk)
    views = C.unit_peak_views(
        [inst],
        {inst.id: _peak_plan(pipeline, "rank", f"{pipeline}/rank/cur", _CENSUS)},
        {},
        api=bucket,
        bucket="ns/b",
        cache={},
    )
    assert views[inst.id] == {"rank": {"u1": 4_000}}


def test_the_history_search_lists_each_earlier_store_of_the_job_once_per_stage() -> None:
    """Review S3B-12: the search's API cost per controller is bounded -- ONE root listing, then for each earlier run
    key of the SAME job one listing and one `_inputs.json` download per stage (shared by every instance and generation
    that asks), and one ledger download per matching generation (each also shared). Another job's run keys are never
    listed."""
    old = [f"cycle1-{c * 7}-ghost_gk-tc3" for c in "abc"]
    gens: dict[str, dict] = {
        f"{rk}/ghost_gk/extract/t{i}": {
            "inputs": {**_EXTRACT, "code": str(i)},
            "summary": True,
            "units": {"m1": (i, i)},
        }
        for i, rk in enumerate(old)
    }
    gens["cycle1-ddddddd-ghost_outfield-tc3/ghost_outfield/extract/o"] = {
        "inputs": {**_EXTRACT, "pipeline": "ghost_outfield"},
        "summary": True,
        "units": {"m1": (9, 9)},
    }
    bucket = _peaks_bucket(gens)
    cur_rk = "cycle1-eeeeeee-ghost_gk-tc3"
    a = C.Instance("ghost_gk", "tc3", "default", "scripts.train_ghost_gk", (), (), cur_rk)
    b = C.Instance("ghost_gk", "tc3", "position_only", "scripts.train_ghost_gk", ("--x",), (), cur_rk)
    plans = {
        a.id: _peak_plan("ghost_gk", "extract", "ghost_gk/extract/ga", _EXTRACT),
        b.id: _peak_plan("ghost_gk", "extract", "ghost_gk/extract/gb", _EXTRACT),
    }
    views = C.unit_peak_views([a, b], plans, {}, api=bucket, bucket="ns/b", cache={})
    assert views[a.id] == views[b.id] == {"extract": {"m1": 2}}  # the newest by its ledgers: run key "ccc..."
    lists = [d for n, d in bucket.calls if n == "list_bucket_tree"]
    assert lists.count(("", False)) == 1  # the root, once per controller
    stage_lists = sorted(prefix for prefix, _rec in lists if prefix)
    assert stage_lists == sorted(f"{rk}/ghost_gk/extract/" for rk in old)  # once per earlier store, not per instance
    assert not any("ghost_outfield" in p for p in stage_lists)  # another job's store is never listed
    # HFU canary run 2 (newest-first): per earlier store one `_inputs.json` download, then the ledgers of the
    # SINGLE newest generation only -- never the losers' (before: a ledger download per matching generation).
    downloads = [paths for n, paths in bucket.calls if n == "download_bucket_files"]
    ledger_dl = [paths for paths in downloads if any(p.endswith(".ledger.json") for p in paths)]
    assert len(ledger_dl) == 1 and all("-ccccccc-" in p for p in ledger_dl[0])  # only the winner's ledgers
