"""ADR-123 DGX e2e gate: the phase DAG runs locally with per-phase peak-RSS vs the 32 GB cpu-upgrade
ceiling. Sequencing + ceiling logic tested offline with an injected ``measure``; ``measure_rss`` itself is
a DGX/Linux step (GNU time), so its test skips where GNU time is unavailable."""

from __future__ import annotations

import shutil

import pytest

from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.dgx_e2e import CPU_UPGRADE_RAM_BYTES, measure_rss, run_local_e2e


def _default(name):
    spec = reg.REGISTRY[name]
    return spec, spec.builds[0]


def test_run_local_e2e_ghost_sequences_and_passes_under_ceiling():
    spec, b = _default("ghost_gk")
    seen = []

    def fake(cmd):
        seen.append(cmd)
        return "", 0, 1_000_000  # 1 MB peak -- well under the 32 GB ceiling

    summary = run_local_e2e(
        spec,
        b,
        data_args="--data-dir /c --actions-dir /c/_actions --home-teams /c/home.json",
        out_dir="/o",
        corpus_dir="/c",
        n_extract=2,
        n_folds=2,
        measure=fake,
    )
    assert summary["blocked"] is False
    assert [p["phase"] for p in summary["phases"]] == ["extract", "extract", "build_cache", "fold", "fold", "reduce"]
    assert any("headtohead ghost_gk" in c for c in seen)  # reduce runs the floor over the local corpus


def test_run_local_e2e_blocks_when_reduce_exceeds_the_rss_ceiling():
    spec, b = _default("ghost_gk")

    def fake(cmd):
        rss = 40 * 2**30 if "--assemble" in cmd else 1_000_000  # the serial reduce OOM-risks (> 32 GB)
        return "", 0, rss

    summary = run_local_e2e(
        spec,
        b,
        data_args="--data-dir /c",
        out_dir="/o",
        corpus_dir="/c",
        n_extract=2,
        n_folds=2,
        rss_ceiling=CPU_UPGRADE_RAM_BYTES,
        measure=fake,
    )
    assert summary["blocked"] is True
    reduce = next(p for p in summary["phases"] if p["phase"] == "reduce")
    assert reduce["ok"] is False and reduce["peak_rss"] == 40 * 2**30


def test_run_local_e2e_xshot_enumerates_studies_from_prep():
    spec, b = _default("xshot_occurrence")

    def fake(cmd):
        if "--prep-only" in cmd:
            return '{"study_root": "/s", "studies": ["a_f0", "a_f1"]}', 0, 1000
        return "", 0, 1000

    summary = run_local_e2e(
        spec,
        b,
        data_args="--data-dir /pg",
        out_dir="/o",
        corpus_dir="/c",
        n_extract=2,
        arms_path="/tmp/arms.json",  # noqa: S108 -- a fixed in-job path string, not a real local tempfile
        measure=fake,
    )
    assert [p["phase"] for p in summary["phases"]] == ["extract", "extract", "prep", "study", "study", "reduce"]


def test_measure_rss_reports_peak_or_skips_off_linux():
    if shutil.which("time") is None and shutil.which("/usr/bin/time") is None:
        pytest.skip("GNU /usr/bin/time -v unavailable (this is a DGX/Linux gate step)")
    out, code, peak = measure_rss("python -c 'x=bytearray(60000000); print(len(x))'")
    assert code == 0 and peak > 20_000_000, out
