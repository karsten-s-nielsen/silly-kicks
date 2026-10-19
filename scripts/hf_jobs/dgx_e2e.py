"""DGX end-to-end gate (ADR-123, ADR-128): run every stage of a campaign instance LOCALLY against the staged
neutral corpus and measure **peak RSS per stage vs the 32 GB cpu-upgrade ceiling** + wall-time, BEFORE any
HF spend. A serial stage (a final fit, an inner_parallel study) is the one fan-out does not shrink, and
cpu-upgrade is an 8x RAM cut from cpu-performance; a 32 GB OOM is NOT ``--attempts``-fixable and the DGX's
ample RAM would otherwise hide it, so the gate measures it here. Any stage over the ceiling sets ``blocked``
-> the owner rules before the HF wave runs.

This runs on the DGX (Linux, real corpus); the sequencing + ceiling logic is unit-tested offline with an
injected ``measure`` and ``plan_fn``. It runs the stage CLI of the instance (``--plan``, then ``--stages S``
per stage, then the ``--run-all --out`` export) against a local data root and store -- no clone, no download.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

CPU_UPGRADE_RAM_BYTES = 32 * 2**30  # the cpu-upgrade ceiling the reduce phase must fit under


def check_shipped_variant(*, fresh: str | None, committed: str | None) -> str | None:
    """ADR-123 Amendment B / spec §5: the fresh visibility-gated fit must ship the SAME candidate the
    committed model shipped (``public``), not a worse ``sc_extended`` -- the GATE-1 gap that let an
    exit-0 run pass while shipping the wrong candidate. Returns an error string on a mismatch, else None.
    """
    if fresh != committed:
        return f"shipped-candidate mismatch: fresh shipped {fresh!r}, committed shipped {committed!r}"
    return None


def measure_rss(cmd: str, *, bash: str = "bash") -> tuple[str, int, int]:
    """Run ``bash -c <cmd>`` and return ``(output, exit_code, peak_rss_bytes)``.

    Uses ``/usr/bin/time -v`` (GNU time) for the process tree's "Maximum resident set size"; raises if it
    is unavailable (the gate is a DGX/Linux step -- ``resource`` is Unix-only and ``ru_maxrss`` is a
    cumulative high-water mark, so GNU time's per-invocation figure is the honest one)."""
    gnu_time = shutil.which("time") or ("/usr/bin/time" if shutil.which("/usr/bin/time") else None)
    if gnu_time is None:
        raise RuntimeError("measure_rss needs GNU /usr/bin/time -v (a DGX/Linux gate step); not found")
    # `bash -c` (NOT a login shell): inherit the caller's env/PATH verbatim so the trainer's bare `python`
    # resolves to the active venv deterministically, rather than a profile-reset login PATH.
    p = subprocess.run(  # noqa: S603
        [gnu_time, "-v", bash, "-c", cmd], capture_output=True, text=True
    )
    out = p.stdout + p.stderr
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", out)
    peak = int(m.group(1)) * 1024 if m else -1
    return out, p.returncode, peak


def _cli(instance, *, data_root, store, extra: Sequence[str]) -> list[str]:
    """One stage-CLI invocation of ``instance`` against a LOCAL data root and store (no clone, no download)."""
    return [
        "python",
        "-m",
        instance.module,
        *instance.args,
        "--store",
        str(store),
        "--data-root",
        str(data_root),
        "--allow-dirty",  # the e2e runs from a dirty scratch checkout; the artifact still records it
        *extra,
    ]


def _plan(instance, *, data_root, store) -> dict:
    argv = _cli(instance, data_root=data_root, store=store, extra=["--plan"])
    p = subprocess.run(argv, capture_output=True, text=True, check=True)  # noqa: S603 -- campaign-derived argv
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_local_e2e(
    instance,
    *,
    data_root,
    store,
    out,
    rss_ceiling: int = CPU_UPGRADE_RAM_BYTES,
    workers: int = 1,
    inner_workers: int = 1,
    measure=measure_rss,
    plan_fn=None,
) -> dict:
    """Run every stage of a campaign ``instance`` LOCALLY in DAG order (ADR-128), measuring peak RSS + wall per
    stage under ``measure``, then export the output to ``out``. Returns ``{"stages": [{stage, peak_rss, wall,
    ok, code}...], "blocked": bool, "max_rss": int, "rss_ceiling": int}``. ``blocked`` when a stage exits
    non-zero or reaches ``rss_ceiling`` -- the gate for the HF wave. ``plan_fn(instance)`` (default: the
    stage CLI's ``--plan``) supplies the stage list, so the sequencing is testable offline."""
    plan = plan_fn(instance) if plan_fn is not None else _plan(instance, data_root=data_root, store=store)
    stages: list[dict] = []
    for st in plan["stages"]:
        flags = ["--inner-workers", str(inner_workers)] if st.get("inner_parallel") else ["--workers", str(workers)]
        cmd = shlex.join(_cli(instance, data_root=data_root, store=store, extra=["--stages", st["name"], *flags]))
        t0 = time.time()
        _out, code, peak = measure(cmd)
        rec = {"stage": st["name"], "peak_rss": peak, "wall": time.time() - t0, "code": code}
        rec["ok"] = code == 0 and 0 <= peak < rss_ceiling
        stages.append(rec)
        if not rec["ok"]:
            break  # a later stage cannot run on a failed upstream
    blocked = any(not s["ok"] for s in stages)
    result: dict = {
        "stages": stages,
        "blocked": blocked,
        "max_rss": max((s["peak_rss"] for s in stages), default=0),
        "rss_ceiling": rss_ceiling,
    }
    if blocked:
        return result
    export = shlex.join(_cli(instance, data_root=data_root, store=store, extra=["--run-all", "--out", str(out)]))
    _o, code, _peak = measure(export)
    if code != 0:
        result["blocked"] = True
        result["export_error"] = f"export exited {code}"
        return result
    # ADR-123 Amendment B / spec section 5: a visibility-gated trainer must ship the SAME candidate as committed
    # (``public``), not a worse ``sc_extended`` -- the GATE-1 exit-0-only gap. Read both shipped_variants.
    from scripts.hf_jobs.registry import TrainerSpec, spec_for

    spec = spec_for(instance.job)
    if isinstance(spec, TrainerSpec) and spec.visibility_gated:
        build = next(b for b in spec.builds if b.variant == instance.variant)
        try:
            fresh = json.loads((Path(out) / "metrics.json").read_text(encoding="utf-8"))["shipped_variant"]
            committed = json.loads(
                (Path("silly_kicks") / build.committed_path / "metrics.json").read_text(encoding="utf-8")
            )["shipped_variant"]
        except (OSError, KeyError, ValueError) as e:
            result["blocked"] = True
            result["shipped_variant_error"] = f"could not read shipped_variant: {e}"
            return result
        mismatch = check_shipped_variant(fresh=fresh, committed=committed)
        result["shipped_variant"] = {"fresh": fresh, "committed": committed}
        if mismatch:
            result["blocked"] = True
            result["shipped_variant_error"] = mismatch
    return result
