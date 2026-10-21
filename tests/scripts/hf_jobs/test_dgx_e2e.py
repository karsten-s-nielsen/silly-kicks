"""DGX e2e gate (ADR-123, ADR-128): every stage of a campaign instance runs LOCALLY in DAG order with per-stage
peak RSS vs the 32 GB cpu-upgrade ceiling. Sequencing + ceiling logic tested offline with an injected
``measure`` and ``plan_fn``; ``measure_rss`` itself is a DGX/Linux step (GNU time), so its test skips where GNU
time is unavailable."""

from __future__ import annotations

import json
import shutil

import pytest

from scripts.hf_jobs.campaign import Instance
from scripts.hf_jobs.dgx_e2e import CPU_UPGRADE_RAM_BYTES, measure_rss, run_local_e2e

_GHOST_STAGES = ["validate", "extract", "cache", "cv", "fit", "importance", "assemble"]


def _instance(job="ghost_gk", variant="default", module="scripts.train_ghost_gk", args=("--dataset", "tc3")):
    return Instance(job, "owner", variant, module, tuple(args), (), "rk")


def _plan(names, inner=("importance",)):
    return lambda _instance: {"stages": [{"name": n, "inner_parallel": n in inner} for n in names]}


def test_every_stage_runs_in_dag_order_and_passes_under_the_ceiling(tmp_path):
    seen: list[str] = []

    def fake(cmd):
        seen.append(cmd)
        return "", 0, 1_000_000  # 1 MB peak -- well under the 32 GB ceiling

    summary = run_local_e2e(
        _instance(),
        data_root=tmp_path / "d",
        store=tmp_path / "s",
        out=tmp_path / "o",
        measure=fake,
        plan_fn=_plan(_GHOST_STAGES),
    )
    assert summary["blocked"] is False
    assert [s["stage"] for s in summary["stages"]] == _GHOST_STAGES
    assert all(f"--stages {n}" in c for n, c in zip(_GHOST_STAGES, seen, strict=False))
    assert "--inner-workers" in seen[_GHOST_STAGES.index("importance")]  # the inner_parallel stage owns the pool
    assert "--run-all" in seen[-1] and "--out" in seen[-1]  # the export after every stage


def test_a_stage_over_the_rss_ceiling_blocks_and_stops_the_run(tmp_path):
    def fake(cmd):
        rss = 40 * 2**30 if "--stages fit" in cmd else 1_000_000  # the serial final fit OOM-risks (> 32 GB)
        return "", 0, rss

    summary = run_local_e2e(
        _instance(),
        data_root=tmp_path / "d",
        store=tmp_path / "s",
        out=tmp_path / "o",
        rss_ceiling=CPU_UPGRADE_RAM_BYTES,
        measure=fake,
        plan_fn=_plan(_GHOST_STAGES),
    )
    assert summary["blocked"] is True
    fit = next(s for s in summary["stages"] if s["stage"] == "fit")
    assert fit["ok"] is False and fit["peak_rss"] == 40 * 2**30
    assert [s["stage"] for s in summary["stages"]][-1] == "fit", "a later stage ran on a failed upstream"


def test_a_failing_stage_blocks(tmp_path):
    summary = run_local_e2e(
        _instance(),
        data_root=tmp_path / "d",
        store=tmp_path / "s",
        out=tmp_path / "o",
        measure=lambda cmd: ("", 3 if "--stages cv" in cmd else 0, 1000),
        plan_fn=_plan(_GHOST_STAGES),
    )
    assert summary["blocked"] is True and summary["stages"][-1]["stage"] == "cv"


def test_a_visibility_gated_instance_must_ship_the_committed_candidate(tmp_path):
    """ADR-123 Amendment B: an exit-0 run that ships a different candidate than the committed bundle BLOCKS."""
    inst = _instance("xshot_occurrence", "default", "scripts.train_xshot_occurrence", ("--dataset", "xshot17"))
    out = tmp_path / "o"

    def fake(cmd):
        if "--run-all" in cmd:  # the export writes the fresh bundle
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps({"shipped_variant": "sc_extended"}), encoding="utf-8")
        return "", 0, 1000

    summary = run_local_e2e(
        inst,
        data_root=tmp_path / "d",
        store=tmp_path / "s",
        out=out,
        measure=fake,
        plan_fn=_plan(["extract"], inner=()),
    )
    committed = json.loads(open("silly_kicks/tracking/_xshot_weights/default/metrics.json", encoding="utf-8").read())
    assert summary["shipped_variant"] == {"fresh": "sc_extended", "committed": committed["shipped_variant"]}
    assert summary["blocked"] is (committed["shipped_variant"] != "sc_extended")


def test_measure_rss_reports_peak_or_skips_off_linux():
    if shutil.which("time") is None and shutil.which("/usr/bin/time") is None:
        pytest.skip("GNU /usr/bin/time -v unavailable (this is a DGX/Linux gate step)")
    out, code, peak = measure_rss("python -c 'x=bytearray(60000000); print(len(x))'")
    assert code == 0 and peak > 20_000_000, out
