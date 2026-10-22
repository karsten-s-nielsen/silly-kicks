"""Shared W1N1-vs-W2N2 parity harness for migrated pipelines (ADR-128, spec 5.2).

``run_serial`` is the serial parity reference (``--run-all --workers 1``); ``run_fanned`` advances stage by
stage like the controller, as ``jobs`` concurrent shard processes per multi-unit stage at ``workers`` (or
``--inner-workers`` for an inner_parallel stage). The parity tests then require every stage output, every
unit result and the exported artifact to be byte-identical, and the completion order to differ from key
order on each fanned path (non-vacuity).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

VOLATILE = (".ledger.json", ".failed.json", "_profile.json")


def cli(module, store, data_root, args, *extra, check=True, env=None):
    cmd = [
        sys.executable,
        "-m",
        module,
        *args,
        "--store",
        str(store),
        "--data-root",
        str(data_root),
        "--allow-dirty",
        *extra,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603 -- fixed argv
    if check and p.returncode != 0:
        raise AssertionError(f"{' '.join(cmd)}\nrc={p.returncode}\n{p.stdout[-4000:]}\n{p.stderr[-4000:]}")
    return p


def plan(module, store, data_root, args, env=None):
    return json.loads(cli(module, store, data_root, args, "--plan", env=env).stdout)


def run_serial(module, store, data_root, args, out, env=None):
    cli(module, store, data_root, args, "--run-all", "--workers", "1", "--out", str(out), env=env)


def run_fanned(module, store, data_root, args, out, *, jobs=2, workers=2, env=None):
    """Advance stage by stage like the controller: a multi-unit stage runs as `jobs` concurrent shard
    processes (each labelled like an HF job, so the ledger groups units per job); an inner_parallel stage
    gets --inner-workers; then export the output."""
    base = dict(os.environ if env is None else env)
    for st in plan(module, store, data_root, args, env)["stages"]:
        p = next(s for s in plan(module, store, data_root, args, env)["stages"] if s["name"] == st["name"])
        n = p["n_units"] or 0
        flags = ["--inner-workers", str(workers)] if p["inner_parallel"] else ["--workers", str(workers)]
        k = min(jobs, n) if n > 1 else 1
        procs = [
            subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    module,
                    *args,
                    "--store",
                    str(store),
                    "--data-root",
                    str(data_root),
                    "--allow-dirty",
                    "--stages",
                    st["name"],
                    "--shard",
                    f"{i}/{k}",
                    *flags,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**base, "SILLY_KICKS_JOB_LABEL": f"{st['name']}.{i}of{k}"},
            )
            for i in range(k)
        ]
        for proc in procs:
            _o, e = proc.communicate()
            assert proc.returncode == 0, e[-4000:]
        cli(module, store, data_root, args, "--stages", st["name"], env=env)  # idempotent reduce
    cli(module, store, data_root, args, "--run-all", "--out", str(out), env=env)  # nothing pending: exports


def outputs(store) -> dict[str, str]:
    root = pathlib.Path(store)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(VOLATILE)
    }


def artifact(out) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path(out).iterdir()) if p.is_file()
    }


def completion_out_of_order(store, generation: str) -> bool:
    """True when, INSIDE at least one job, units finished out of key order -- the W >= 2 worker path,
    not merely two jobs interleaving."""
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    jobs: dict[str, list[dict]] = {}
    for r in recs:
        jobs.setdefault(r.get("job_label") or "", []).append(r)
    return any(
        [r["key"] for r in sorted(rs, key=lambda r: r["finished_ns"])] != sorted(r["key"] for r in rs)
        for rs in jobs.values()
    )


def fanned_jobs(store, generation: str) -> int:
    """How many distinct jobs ran the units of ``generation`` (the N > 1 fan-out really happened)."""
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    return len({r.get("job_label") for r in recs if r.get("job_label")})


def inner_out_of_order(store, generation: str) -> bool:
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    return any(r.get("inner_completion") and r["inner_completion"] != sorted(r["inner_completion"]) for r in recs)


def ran_concurrently(store, generation: str) -> bool:
    """True when, INSIDE at least one job, two units ran on DIFFERENT workers with OVERLAPPING wall
    intervals -- proof that the W >= 2 path really executed in parallel.

    The non-vacuity for a stage whose completion order follows key order BY CONSTRUCTION. Example:
    ghost_outfield's `cv`. StratifiedGroupKFold puts the largest game in fold 0's test set, so fold 0
    trains on the least data and finishes first in every job; no toy heterogeneity can invert that. The
    worker path's ORDER non-vacuity is carried by the same module's `extract` stage
    (`completion_out_of_order`), and the order-sensitive reduce by the framework gate
    (tests/scripts/hf_jobs/test_executor.py)."""
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    jobs: dict[str, list[dict]] = {}
    for r in recs:
        jobs.setdefault(r.get("job_label") or "", []).append(r)
    for rs in jobs.values():
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                if (
                    a["worker"] != b["worker"]
                    and a["started_ns"] < b["finished_ns"]
                    and b["started_ns"] < a["finished_ns"]
                ):
                    return True
    return False
