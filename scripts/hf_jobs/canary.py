"""The rung-3 canary, committed (spec r3.4 item 3; HFU plan R3-8 S3 item 13; owner ruling 2026-10-30).

It proves the production store on the real bucket mount, on the canary bucket ``silly-kicks/hf-refit-canary`` only:

``writer RUN`` (an HF measurement job in the canary mode, `measure --canary`, so the ONLY writable mount is the
    canary bucket at ``/canary``): the toy pipeline runs ``gen`` (``--n-items`` units, W=8, the production executor
    and store on the mount) and ``combined`` (reads every unit back); every unit is verified in this container; 64
    probe files go through ``LocalStore.write_bytes`` with their write window in ``mtime/_manifest.json``; and one
    real-scale stage output (``--real-output``, a file under the read-only campaign mount, e.g. a ``ghost_gk``
    ``extract`` reduce ``_output``) is copied through the same writer with its sha256 in ``real/_manifest.json``.
``reader-mount RUN`` (a FRESH measurement job, canary mode): the cross-container view through the mount, every unit
    and probe file again, and the mtime each probe file shows against the writer's recorded window (the r3 D6(b)
    orphan sweep trusts that mtime).
``reader-api RUN`` (on this machine, with the existing HF login; jobs get no secrets): every unit, probe file and
    the real-scale output read through the bucket API (`StoreSnapshot` over `BucketStore`, `HfApi`), not the mount.

The writer and both readers run ONE tree (the same commit and patch). A reader never recomputes generation tokens
from its own tree; it takes the writer's generations by listing the store. Prints ``RUNG3_*`` lines.

    python -m scripts.hf_jobs.canary reader-api RUN
    python -m scripts.hf_jobs measure --run-id rung3-RUN-w --probe scripts/hf_jobs/canary.py --canary --patch \\
        --bucket CAMPAIGN_BUCKET -- writer RUN --real-output RELATIVE_PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

if __package__ in (None, "") and Path("/work/scripts").is_dir():  # run as the measurement probe (/tmp/probe.py)
    sys.path.insert(0, "/work")

#: Inside the measurement job: the checked-out tree, its environment, the canary mount, the campaign mount.
JOB_WORK: Final = Path("/work")
JOB_PYTHON: Final = "/work/.venv/bin/python"
CANARY_ROOT: Final = Path("/canary")
CAMPAIGN_ROOT: Final = Path("/scratch")
N_ITEMS: Final = 2000
N_MTIME: Final = 64
#: The gap between probe writes, so the mtimes a reader sees are distinct values, not one placeholder (16 s in all).
PROBE_GAP_S: Final = 0.25
SUFFIXES: Final = (".parquet", ".json", ".fileset.json")
_STAGES: Final = ("gen", "combined")


def log(*parts: object) -> None:
    print(*parts, flush=True)


def _toy(python: str, store: Path, n_items: int, *args: str) -> list[str]:
    return [python, "-m", "tests.scripts._toy_pipeline", "--n-items", str(n_items), "--store", str(store), *args]


def _sh(argv: list[str], *, cwd: Path, **env_extra: str) -> subprocess.CompletedProcess:
    t0 = time.time()
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env={**os.environ, **env_extra})  # noqa: S603
    log("RUNG3_CMD", " ".join(argv[1:])[:200], "rc", p.returncode, f"wall_s={time.time() - t0:.1f}")
    if p.returncode:
        log((p.stdout or "")[-2000:])
        log((p.stderr or "")[-3000:])
        raise SystemExit(f"RUNG3_FAILED {argv[1:4]} rc={p.returncode}")
    return p


def _keys(stage: str, n_items: int) -> list[str]:
    return [f"toy__{i:04d}" for i in range(n_items)] if stage == "gen" else ["only"]


def verify_units(view: Any, generation: str, keys: Sequence[str]) -> dict:
    """Every unit of ``generation`` in ``view`` (any store: `LocalStore` on the mount, or `StoreSnapshot` over the
    bucket API): a result file, a sha256 sidecar equal to the result's digest, a ledger record; no orphan temp file;
    the reduce's ``_output``."""
    names = view.names(generation)
    out: dict = {"units": 0, "ok": 0, "missing_result": 0, "bad_sha": 0, "missing_ledger": 0, "bad": []}
    out["tmp_files"] = sum(1 for n in names if n.startswith(".tmp-"))
    for key in keys:
        out["units"] += 1
        result = next((f"{key}{s}" for s in SUFFIXES if f"{key}{s}" in names), None)
        if result is None:
            out["missing_result"] += 1
            out["bad"].append(f"{key}: no result")
            continue
        data = view.read_bytes(f"{generation}/{result}")
        side = view.read_bytes(f"{generation}/{key}.sha256")
        if data is None or side is None or side.decode("ascii").strip() != hashlib.sha256(data).hexdigest():
            out["bad_sha"] += 1
            out["bad"].append(f"{key}: sha256")
            continue
        if f"{key}.ledger.json" not in names:
            out["missing_ledger"] += 1
            out["bad"].append(f"{key}: no ledger")
            continue
        out["ok"] += 1
    out["bad"] = out["bad"][:10]
    out["output_present"] = any(f"_output{s}" in names for s in SUFFIXES)
    return out


def writer_generations(tree: set[str]) -> dict[str, str]:
    """The WRITER's generation of each stage, from a listing of the run's ``store/`` (paths relative to it): one
    token directory per stage in this fresh store. Never recomputed: a reader's tokens depend on its own tree."""
    out = {}
    for name in _STAGES:
        gens = sorted({"/".join(p.split("/")[:3]) for p in tree if len(p.split("/")) > 3 and p.split("/")[1] == name})
        log("RUNG3_GENERATIONS", name, gens)
        if len(gens) != 1:
            raise SystemExit(f"RUNG3_FAILED expected ONE {name} generation, found {len(gens)}")
        out[name] = gens[0]
    return out


def _local_tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def writer(
    run: str,
    *,
    root: Path,
    python: str,
    work: Path,
    n_items: int,
    real_output: Path | None,
    probe_gap_s: float = PROBE_GAP_S,
) -> dict:
    """The writer (see the module docstring). Returns the verification rows it logged."""
    from scripts._stages import LocalStore

    base = root / f"rung3-{run}"
    store = base / "store"
    base.mkdir(parents=True, exist_ok=True)
    _sh(_toy(python, store, n_items, "--stages", "gen", "--workers", "8", "--allow-dirty"), cwd=work)
    _sh(_toy(python, store, n_items, "--stages", "combined", "--workers", "1", "--allow-dirty"), cwd=work)
    view = LocalStore(store)
    gens = writer_generations(_local_tree(store))
    rows = {}
    for name in _STAGES:
        rows[name] = verify_units(view, gens[name], _keys(name, n_items))
        log("RUNG3_WRITER_VERIFY", name, json.dumps(rows[name]))
    out = LocalStore(base)
    manifest = {}
    for i in range(N_MTIME):
        payload = os.urandom(4096 + 97 * i)
        before = time.time()
        out.write_bytes(f"mtime/{i:03d}.bin", payload)
        after = time.time()
        manifest[f"{i:03d}"] = {"before": before, "after": after, "sha256": hashlib.sha256(payload).hexdigest()}
        time.sleep(probe_gap_s)
    out.write_bytes("mtime/_manifest.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
    if real_output is not None:  # C05: one real-scale stage output written through the production writer
        data = real_output.read_bytes()
        t0 = time.time()
        out.write_bytes(f"real/{real_output.name}", data)
        wall = time.time() - t0
        real = {
            "name": real_output.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "wall_s": wall,
        }
        out.write_bytes("real/_manifest.json", json.dumps(real, sort_keys=True).encode("utf-8"))
        log("RUNG3_WRITER_REAL", json.dumps(real))
    log("RUNG3_WRITER_DONE", run)
    return rows


def _check_probes(view: Any, manifest: Mapping[str, Mapping], *, mtime_of=None) -> dict:
    """Every probe file present with its sha256; with ``mtime_of`` (the mount view), its mtime against the window."""
    bad, deltas = [], []
    for name, m in sorted(manifest.items()):
        data = view.read_bytes(f"mtime/{name}.bin")
        if data is None:
            bad.append(f"{name}: missing")
            continue
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            bad.append(f"{name}: sha256")
        if mtime_of is not None:
            mt = mtime_of(f"mtime/{name}.bin")
            deltas.append(mt - m["after"])
            if mt <= 0 or mt < m["before"] - 5.0 or mt > m["after"] + 300.0:
                bad.append(f"{name}: mtime {mt:.2f} outside [{m['before']:.2f}-5, {m['after']:.2f}+300]")
    row: dict = {"files": len(manifest), "bad": bad[:10]}
    if deltas:
        row.update(
            delta_min=round(min(deltas), 2),
            delta_median=round(statistics.median(deltas), 2),
            delta_max=round(max(deltas), 2),
            distinct=len({round(d, 3) for d in deltas}),
        )
    return row


def _check_real(view: Any) -> dict | None:
    raw = view.read_bytes("real/_manifest.json")
    if raw is None:
        return None
    real = json.loads(raw)
    data = view.read_bytes(f"real/{real['name']}")
    return {**real, "ok": data is not None and hashlib.sha256(data).hexdigest() == real["sha256"]}


def read_back(view: Any, store_view: Any, tree: set[str], *, n_items: int, mtime_of=None) -> dict:
    """A reader's checks over ``view`` (the run's directory) and ``store_view`` (its ``store/``)."""
    gens = writer_generations(tree)
    rows: dict = {name: verify_units(store_view, gens[name], _keys(name, n_items)) for name in _STAGES}
    raw = view.read_bytes("mtime/_manifest.json")
    if raw is None:
        raise SystemExit("RUNG3_FAILED the writer's mtime manifest is missing")
    rows["probes"] = _check_probes(view, json.loads(raw), mtime_of=mtime_of)
    rows["real"] = _check_real(view)
    return rows


def reader_mount(run: str, *, root: Path, n_items: int) -> dict:
    from scripts._stages import LocalStore

    base = root / f"rung3-{run}"
    rows = read_back(
        LocalStore(base),
        LocalStore(base / "store"),
        _local_tree(base / "store"),
        n_items=n_items,
        mtime_of=lambda rel: (base / rel).stat().st_mtime,
    )
    log("RUNG3_READER_MOUNT", json.dumps(rows))
    return rows


def reader_api(run: str, *, bucket: str, n_items: int, api: Any = None) -> dict:
    from scripts.hf_jobs.store import BucketStore, StoreSnapshot

    run_view = StoreSnapshot(BucketStore(bucket, f"rung3-{run}", api=api))
    store_view = StoreSnapshot(BucketStore(bucket, f"rung3-{run}/store", api=api))
    rows = read_back(run_view, store_view, store_view.list_tree(""), n_items=n_items)
    log("RUNG3_READER_API", json.dumps(rows))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.hf_jobs.measure import CANARY_BUCKET

    ap = argparse.ArgumentParser(prog="python -m scripts.hf_jobs.canary", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("mode", choices=("writer", "reader-mount", "reader-api"))
    ap.add_argument("run")
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--real-output", default=None, help="writer: a file under the campaign mount, relative to it")
    a = ap.parse_args(argv)
    if a.mode == "writer":
        real = CAMPAIGN_ROOT / a.real_output if a.real_output else None
        writer(a.run, root=CANARY_ROOT, python=JOB_PYTHON, work=JOB_WORK, n_items=a.n_items, real_output=real)
    elif a.mode == "reader-mount":
        reader_mount(a.run, root=CANARY_ROOT, n_items=a.n_items)
    else:
        reader_api(a.run, bucket=CANARY_BUCKET, n_items=a.n_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
