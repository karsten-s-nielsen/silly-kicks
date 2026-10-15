"""ADR-123 staged map/reduce orchestrator: per-trainer phase DAG, fail-loud waits, dynamic study
enumeration. Offline (injected runner) -- the real HF run + the DGX local-runner path are exercised by
the DGX e2e (Task 11)."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.launch import (
    StagePhaseError,
    build_phase_command,
    make_local_runner,
    phase_sequence,
    run_pipeline,
    wait_all,
)


def _bash_runs_posix() -> bool:
    """True when the bare ``bash`` make_local_runner invokes runs a POSIX command. This must probe
    ``bash`` EXACTLY as make_local_runner does -- bare, not an absolute ``shutil.which`` path -- because
    on Windows CreateProcess searches System32 before PATH, so bare ``bash`` hits the WSL stub (no distro
    -> exit 1) even when Git Bash is on PATH. make_local_runner's target is the DGX / HF Linux container,
    never Windows; the ubuntu shards keep this gate non-vacuous (inspect/logs routing reads the closure
    store only the run leg populates, so the whole routing test needs the shell)."""
    try:
        p = subprocess.run(["bash", "-c", "echo ok"], capture_output=True, text=True, timeout=15)  # noqa: S607
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0 and "ok" in p.stdout


BUCKET = "ns/scratch"


def _default(name):
    spec = reg.REGISTRY[name]
    return spec, spec.builds[0]


def _fake_runner(studies=None):
    """A runner handling run/inspect/logs: records each launched inner, reports COMPLETED, serves studies."""
    seen: list[str] = []
    seq = {"n": 0}

    def runner(argv):
        head = argv[:3]
        if head == ["hf", "jobs", "run"]:
            seen.append(argv[-1])  # ... -- bash -lc <inner>
            seq["n"] += 1
            h = f"6abcde{seq['n']}"
            return f"id={h} url=https://huggingface.co/jobs/ns/{h}", 0
        if head == ["hf", "jobs", "inspect"]:
            return json.dumps([{"id": argv[3], "status": {"stage": "COMPLETED"}}]), 0
        if head == ["hf", "jobs", "logs"]:
            return json.dumps({"study_root": "/s", "studies": studies or []}), 0
        return "", 0

    return runner, seen


def _noop(_):
    return None


# --- phase plans -----------------------------------------------------------------------------------
def test_phase_sequences_per_trainer():
    assert phase_sequence(reg.REGISTRY["ghost_gk"]) == ["extract", "build_cache", "fold", "reduce"]
    assert phase_sequence(reg.REGISTRY["ghost_outfield"]) == ["extract", "fold", "reduce"]  # no build_cache
    assert phase_sequence(reg.REGISTRY["xcross_attempt"]) == ["extract", "prep", "study", "reduce"]
    assert phase_sequence(reg.REGISTRY["receiver_model"]) == ["extract", "reduce"]
    assert phase_sequence(reg.REGISTRY["win_probability"]) == ["full"]  # public single-job


# --- per-phase command flags -----------------------------------------------------------------------
def test_ghost_phase_flags():
    spec, b = _default("ghost_gk")
    kw = {"tag": "r", "run_key": "r1", "corpus_repo": "ns/tc3"}
    assert "--shard-index 0/4" in build_phase_command(spec, b, phase="extract", shard="0/4", **kw)
    assert "--build-cache" in build_phase_command(spec, b, phase="build_cache", **kw)
    assert "--fold 2" in build_phase_command(spec, b, phase="fold", fold=2, **kw)
    reduce = build_phase_command(spec, b, phase="reduce", **kw)
    assert "--assemble" in reduce and "scripts.hf_jobs.headtohead ghost_gk" in reduce  # reduce runs the floor
    # cv_folds MUST be threaded so the trainer's split has exactly n_folds folds (else assemble loads missing
    # fold sidecars -- the DGX e2e caught this). On ghost phases; NOT on a non-ghost trainer.
    assert "--cv-folds 2" in build_phase_command(spec, b, phase="fold", fold=0, cv_folds=2, **kw)
    assert "--cv-folds 2" in build_phase_command(spec, b, phase="reduce", cv_folds=2, **kw)
    rcv = build_phase_command(
        reg.REGISTRY["receiver_model"],
        reg.REGISTRY["receiver_model"].builds[0],
        phase="reduce",
        corpus_repo="ns/sb360",
        tag="r",
        run_key="r1",
        cv_folds=2,
    )
    assert "--cv-folds" not in rcv  # receiver has no CV-fold seam


def test_xcross_study_phase_uses_study_root_and_probe():
    spec, b = _default("xcross_attempt")
    inner = build_phase_command(
        spec, b, tag="r", run_key="r1", corpus_repo="ns/xshot", probe_repo="ns/probe", phase="study", study="cand_f0"
    )
    assert "--study cand_f0 --shard-root /scratch/r1/xcross_attempt/default/xcross_attempt_v1/studies" in inner
    assert "--probe-data-dir /tmp/probe" in inner  # probe threaded on every xcross phase


def test_receiver_reduce_is_plain_run_plus_floor_no_assemble():
    spec, b = _default("receiver_model")
    inner = build_phase_command(spec, b, tag="r", run_key="r1", corpus_repo="ns/sb360", phase="reduce")
    assert "--assemble" not in inner  # receiver has no --assemble; reduce reconciles all shards + fits
    assert "--data-dir /tmp/corpus" in inner and "scripts.hf_jobs.headtohead receiver_model" in inner


# --- wait / fail-loud ------------------------------------------------------------------------------
def test_wait_all_raises_on_terminal_error():
    def runner(argv):
        return json.dumps([{"id": argv[3], "status": {"stage": "ERROR"}}]), 0

    with pytest.raises(StagePhaseError):
        wait_all(["ns/j1"], runner=runner, sleep=_noop)


# --- full DAG sequencing ---------------------------------------------------------------------------
def test_run_pipeline_ghost_sequences_extract_buildcache_fold_reduce():
    spec, b = _default("ghost_gk")
    runner, seen = _fake_runner()
    run_pipeline(
        spec,
        b,
        tag="r",
        run_key="r1",
        corpus_repo="ns/tc3",
        bucket=BUCKET,
        attempts=1,
        n_extract=2,
        n_folds=2,
        runner=runner,
        sleep=_noop,
    )
    # 2 extract + 1 build-cache + 2 fold + 1 reduce, in order
    flags = ["--shard-index 0/2", "--shard-index 1/2", "--build-cache", "--fold 0", "--fold 1", "--assemble"]
    assert len(seen) == 6
    for inner, expected in zip(seen, flags, strict=True):
        assert expected in inner, (expected, inner)
    # fold + reduce carry --cv-folds 2 (== n_folds), so the trainer split matches the fanned fold count
    assert "--cv-folds 2" in seen[3] and "--cv-folds 2" in seen[5]


def test_run_pipeline_xshot_enumerates_studies_from_prep():
    spec, b = _default("xshot_occurrence")
    runner, seen = _fake_runner(studies=["a_f0", "a_f1"])
    run_pipeline(
        spec,
        b,
        tag="r",
        run_key="r1",
        corpus_repo="ns/xshot",
        bucket=BUCKET,
        attempts=1,
        n_extract=2,
        runner=runner,
        sleep=_noop,
    )
    # 2 extract + 1 prep + 2 study (enumerated) + 1 reduce
    assert len(seen) == 6
    assert "--prep-only" in seen[2]
    assert "--study a_f0" in seen[3] and "--study a_f1" in seen[4]
    assert "--assemble" in seen[5]


@pytest.mark.skipif(not _bash_runs_posix(), reason="needs a working POSIX bash; make_local_runner targets DGX/Linux")
def test_make_local_runner_routes_run_inspect_logs(tmp_path):
    runner = make_local_runner()
    out, code = runner(
        ["hf", "jobs", "run", "--flavor", "cpu-upgrade", "-d", "python:3.12", "--", "bash", "-lc", "echo hi"]
    )
    assert code == 0 and "id=" in out
    jid = out.split("id=")[1].split()[0]
    insp, _ = runner(["hf", "jobs", "inspect", f"local/{jid}", "--format", "json"])
    assert '"COMPLETED"' in insp
    logs, _ = runner(["hf", "jobs", "logs", f"local/{jid}"])
    assert "hi" in logs
