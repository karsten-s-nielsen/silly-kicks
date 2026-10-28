"""A dry-run controller step over a dict-backed bucket with real work to decide (TF-85).

The world has an instance with a done stage (its unit ledgers must be read), a second variant that shares
that store, a failing ``--plan`` and a finished instance. `run_world` steps it once through the campaign
module's real `step` over the real store reads and returns the report plus the bucket's call log, so the same
world runs through the step before TF-85 and after it and the two reports compare.

This module uses only seams that exist on ``main``: it drives `step` through its public signature and patches
the same collaborators the old and new step share."""

from __future__ import annotations

import hashlib
import json
import pathlib
import threading
from collections.abc import Callable
from typing import Any

SHA = "abcdef1" + "0" * 33
RK_A = "tf85-abcdef1-ghost_gk-owner"
RK_B = "tf85-abcdef1-win_probability-public"
RK_C = "tf85-abcdef1-xsuccess-public"


class Entry:
    def __init__(self, path: str, data: bytes) -> None:
        self.type, self.path, self.xet_hash = "file", path, hashlib.sha256(data).hexdigest()


class Bucket:
    """A dict-backed HF bucket + job list. ``calls`` logs every API call in order as ``(name, detail)``."""

    def __init__(self, files: dict[str, bytes], jobs: list[Any]) -> None:
        self.files, self.jobs = dict(files), list(jobs)
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

    def list_jobs(self, *, labels=None):
        self._log("list_jobs", dict(labels or {}))
        want = dict(labels or {})
        return [j for j in self.jobs if all(j.labels.get(k) == v for k, v in want.items())]


def _json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True) + "\n").encode()


def _stage(pipeline: str, name: str, *, n: int, done: int, needs=(), reduced=False, ready=True, token=None) -> dict:
    units = [f"u{i}" for i in range(n)]
    tok = token or f"t_{name}"
    return {
        "name": name,
        "needs": list(needs),
        "token": tok,
        "generation": f"{pipeline}/{name}/{tok}",
        "inner_parallel": False,
        "inner_width": 1,
        "history_key": name,
        "item_stage": False,
        "ready": ready,
        "reduced": reduced,
        "n_units": n,
        "done": done,
        "excluded": 0,
        "failed": 0,
        "pending": n - done,
        "units": units,
        "pending_units": units[done:],
        "failed_units": {},
        "memory_units": [],
        "max_failures": 0,
    }


def _plan(pipeline: str, stages: list[dict]) -> dict:
    return {"pipeline": pipeline, "module": f"scripts.train_{pipeline}", "stages": stages}


def build(campaign: Any) -> tuple[Any, list[Any], dict[str, Any], Bucket]:
    """``(campaign, instances, plans by instance id, bucket)`` for the ``campaign`` module."""
    C = campaign
    c = C.Campaign(
        name="tf85",
        bucket="b/x",
        target_minutes=60,
        dispatch_cap=10,
        shard_cap=10,
        attempts=3,
        jobs=(),
        path=None,
        sha256="h",
    )
    a1 = C.Instance("ghost_gk", "owner", "default", "scripts.train_ghost_gk", (), (), RK_A)
    a2 = C.Instance("ghost_gk", "owner", "position_only", "scripts.train_ghost_gk", (), (), RK_A)
    b = C.Instance("win_probability", "public", "default", "scripts.train_win_probability", (), (), RK_B)
    d = C.Instance("xsuccess", "public", "default", "scripts.train_xsuccess", (), (), RK_C)
    plans: dict[str, Any] = {
        a1.id: _plan("ghost_gk", [_stage("ghost_gk", "extract", n=10, done=6)]),
        a2.id: _plan("ghost_gk", [_stage("ghost_gk", "extract", n=4, done=0, token="t_extract_po")]),
        b.id: RuntimeError(f"{b.id}: --plan failed (rc=1): boom"),
        d.id: _plan("xsuccess", [_stage("xsuccess", "run", n=1, done=1, reduced=True)]),
    }
    done = {"outcome": "done", "peak_rss_bytes": 10**9}
    files: dict[str, bytes] = {f"{RK_A}/ghost_gk/extract/t_extract/_profile.json": _json({"efficiency": 0.8})}
    for i in range(6):  # six measured units of A's extract: six ledgers that must be read to cost the stage
        files[f"{RK_A}/ghost_gk/extract/t_extract/u{i}.ledger.json"] = _json(
            {"key": f"u{i}", "wall_s": 30.0 + i, **done}
        )
    files[f"{RK_A}/ghost_gk/extract/t_extract/u0.parquet"] = b"\x00" * 2048  # a large result beside the ledgers
    files[f"{RK_C}/xsuccess/run/t_run/u0.ledger.json"] = _json({"key": "u0", "wall_s": 12.0, **done})
    return c, [a1, a2, b, d], plans, Bucket(files, [])


def run_world(
    campaign: Any,
    *,
    patch: Callable[[Any, str, Any], None],
    cache: dict | None = None,
    tree: str = "tree",
    plan_calls: list[str] | None = None,
) -> tuple[dict, Bucket]:
    """Step the world once (dry run) through the ``campaign`` module. ``patch(obj, name, value)`` stubs a seam;
    ``cache`` carries the step cache from an earlier step (as ``--watch`` does); ``tree`` is the tree identity
    the step sees; ``plan_calls`` records the instance id of every ``--plan`` call."""
    C = campaign
    from scripts.hf_jobs import launch as launcher
    from scripts.hf_jobs import preflight

    c, insts, plans, bucket = build(C)

    def read_plan(inst, *, bucket: str, run=None):
        if plan_calls is not None:
            plan_calls.append(inst.id)
        got = plans[inst.id]
        if isinstance(got, Exception):
            raise got
        return json.loads(json.dumps(got))

    patch(preflight, "require_preflight", lambda *a, **k: {})
    patch(C, "instances", lambda c, commit, replica=None: list(insts))
    patch(C, "read_plan", read_plan)
    patch(C, "committed_history", dict)
    patch(C, "planning_constraints", lambda c, **k: c.constraints())
    patch(launcher, "jobs_for", lambda api, *, campaign, commit: list(bucket.jobs))
    if hasattr(C, "_tree_identity"):  # TF-85: the plan cache key's tree identity (a git call)
        patch(C, "_tree_identity", lambda: tree)
    report = C.step(c, SHA, api=bucket, token="hf_x", dry_run=True, cache=cache)
    return json.loads(json.dumps(report, default=str)), bucket
