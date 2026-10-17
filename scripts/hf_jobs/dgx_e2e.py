"""DGX end-to-end gate (ADR-123): run a trainer's full map/reduce pipeline LOCALLY against the staged
neutral corpus and measure **peak RSS per phase vs the 32 GB cpu-upgrade ceiling** + wall-time, BEFORE any
HF spend. The serial reduce (ghost/receiver final fit) is the phase fan-out does not shrink and cpu-upgrade
is an 8x RAM cut from cpu-performance; a 32 GB OOM is NOT ``--attempts``-fixable and the DGX's ample RAM
would otherwise hide it, so the gate measures it here (spec Amendment / §8 / §11). Any phase over the ceiling
sets ``blocked`` -> the owner rules on the reduce executor/flavor before the HF wave runs.

This runs on the DGX (Linux, real corpus); the sequencing + ceiling logic is unit-tested offline with an
injected ``measure``. It reuses the launch orchestrator's phase plan + flag builder, but runs LOCAL trainer
commands (no git clone, no ``hf download``) against a local ``--data-dir`` corpus.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

from scripts.hf_jobs.launch import _fresh_dir, _phase_flags, _trainer_cmd, phase_sequence
from scripts.hf_jobs.registry import TrainerSpec, VariantBuild

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


def _local_cmd(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    data_args: str,
    out_dir: str,
    corpus_dir: str,
    probe_data_dir: str | None,
    phase: str,
    shard=None,
    fold=None,
    study=None,
    cv_folds=None,
    allow_dirty: bool = True,
    arms_path: str | None = None,
) -> str:
    """A LOCAL phase command (no clone, no download): the trainer module + phase flags against a local
    ``--data-dir`` corpus. ``reduce``/``full`` append the head-to-head floor over the same local corpus.

    ``allow_dirty`` (default True): the e2e runs from a dirty scratch checkout, so every trainer gets
    ``--allow-dirty`` (it still records ``run_tree_dirty`` -- the hatch permits a dev run, never launders it).
    ``arms_path`` (ADR-123 Amendment B): a visibility-gated trainer gets the local --arms policy file on the
    extraction phases (the DGX runbook computes it off the local corpus, exactly as the launcher does on HF)."""
    extra = f"--probe-data-dir {probe_data_dir}" if (spec.probe and probe_data_dir) else ""
    if spec.visibility_gated and phase in ("full", "prep"):
        if arms_path is None:
            raise ValueError(f"{spec.name}: visibility-gated DGX e2e needs arms_path (the local --arms policy)")
        extra = f"{extra} --arms {arms_path}".strip()
    extra = f"{extra} {_phase_flags(spec, build, phase, out_dir, shard=shard, fold=fold, study=study)}".strip()
    if cv_folds is not None and spec.name in ("ghost_gk", "ghost_outfield"):
        extra = f"{extra} --cv-folds {cv_folds}".strip()
    if allow_dirty:
        extra = f"{extra} --allow-dirty".strip()
    cmd = _trainer_cmd(spec, build, data_args, out_dir, extra)
    if phase in ("full", "reduce"):
        committed = f"silly_kicks/{build.committed_path}"
        cmd += (
            f" && python -m scripts.hf_jobs.headtohead {spec.name} --committed-dir {committed} "
            f"--fresh-dir {_fresh_dir(out_dir, build)} --feature-set {build.feature_set} "
            f"--metrics-json {out_dir}/metrics.json --corpus-dir {corpus_dir}"
        )
    return cmd


def _studies(prep_out: str) -> list[str]:
    for line in reversed(prep_out.splitlines()):
        line = line.strip()
        if line.startswith("{") and "studies" in line:
            try:
                import json

                return list(json.loads(line)["studies"])
            except (ValueError, KeyError, TypeError):
                continue
    raise RuntimeError("could not parse the study list from the prep output")


def run_local_e2e(
    spec: TrainerSpec,
    build: VariantBuild,
    *,
    data_args: str,
    out_dir: str,
    corpus_dir: str,
    probe_data_dir: str | None = None,
    n_extract: int = 2,
    n_folds: int = 2,
    rss_ceiling: int = CPU_UPGRADE_RAM_BYTES,
    allow_dirty: bool = True,
    arms_path: str | None = None,
    measure=measure_rss,
) -> dict:
    """Run the trainer's phase DAG locally, measuring peak RSS + wall per phase. Returns
    ``{"phases": [{phase, peak_rss, wall, ok, code}...], "blocked": bool, "max_rss": int}``. ``blocked`` is
    True if any phase OOM-risks (peak RSS >= ``rss_ceiling``) or a phase exits non-zero -- the gate for the
    HF wave. ``data_args`` is the trainer's local corpus wiring (``--data-dir <corpus_dir> ...``), supplied
    by the DGX runbook because it is trainer+corpus-shape specific."""

    def cmd(phase, **kw):
        return _local_cmd(
            spec,
            build,
            data_args=data_args,
            out_dir=out_dir,
            corpus_dir=corpus_dir,
            probe_data_dir=probe_data_dir,
            phase=phase,
            cv_folds=n_folds,
            allow_dirty=allow_dirty,
            arms_path=arms_path,
            **kw,
        )

    phases: list[dict] = []
    prep_out = ""
    for phase in phase_sequence(spec):
        if phase == "extract":
            jobs = [("extract", cmd("extract", shard=f"{i}/{n_extract}")) for i in range(n_extract)]
        elif phase == "fold":
            jobs = [("fold", cmd("fold", fold=k)) for k in range(n_folds)]
        elif phase == "study":
            jobs = [("study", cmd("study", study=s)) for s in _studies(prep_out)]
        else:
            jobs = [(phase, cmd(phase))]
        for label, c in jobs:
            t0 = time.time()
            out, code, peak = measure(c)
            rec = {"phase": label, "peak_rss": peak, "wall": time.time() - t0, "code": code}
            rec["ok"] = code == 0 and 0 <= peak < rss_ceiling
            phases.append(rec)
            if phase == "prep":
                prep_out = out
    max_rss = max((p["peak_rss"] for p in phases), default=0)
    blocked = any(not p["ok"] for p in phases)
    result = {"phases": phases, "blocked": blocked, "max_rss": max_rss, "rss_ceiling": rss_ceiling}
    # ADR-123 Amendment B / spec §5: a visibility-gated trainer must ship the SAME candidate as committed
    # (``public``), not a worse ``sc_extended`` -- the GATE-1 exit-0-only gap. Read both shipped_variants.
    if spec.visibility_gated and not blocked:
        import json
        from pathlib import Path

        try:
            fresh = json.loads((Path(_fresh_dir(out_dir, build)) / "metrics.json").read_text())["shipped_variant"]
            committed = json.loads((Path("silly_kicks") / build.committed_path / "metrics.json").read_text())[
                "shipped_variant"
            ]
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
