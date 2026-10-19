"""A toy UNMIGRATED trainer for tests/scripts/hf_jobs/test_adapter.py (ADR-128 spec 5.6).

An argparse CLI with ``--out`` (and ``--shard-root``) that shards two "matches" resume-by-file, then writes
``<out>/weights/model.json`` deterministically from its arguments. ``$TOY_DIE_AFTER_FIRST`` simulates a kill
mid-way: the first shard is written, then the process exits non-zero -- so a later attempt with the SAME
arguments must resume through the target's own shards. Every run and every shard write is logged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard-root", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    shards = Path(a.shard_root) if a.shard_root else out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.log", "a", encoding="utf-8") as fh:
        fh.write("run\n")
    for i in range(2):
        f = shards / f"m{i}.json"
        if f.exists():
            continue  # the target's OWN resume (its for_each)
        f.write_text(json.dumps({"i": i, "v": a.seed * (i + 1)}), encoding="utf-8")
        with open(out / "writes.log", "a", encoding="utf-8") as fh:
            fh.write(f"m{i}\n")
        if os.environ.get("TOY_DIE_AFTER_FIRST"):
            raise SystemExit(3)
    vals = [json.loads((shards / f"m{i}.json").read_text(encoding="utf-8"))["v"] for i in range(2)]
    (out / "weights").mkdir(exist_ok=True)
    model = {"sum": sum(vals), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    (out / "weights" / "model.json").write_text(json.dumps(model, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
