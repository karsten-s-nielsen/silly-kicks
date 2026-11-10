"""Job liveness (HFU spec §11, Change 4): every covered in-job Python process prints an ``[alive]`` start line at
once and a line every 300 s (its unit, the unit's age, its main thread's innermost Python frame and its child
processes), and arms a `faulthandler` dump 600 s before the job's deadline.

E12 (spec §1.1): a cv job ran 2 h and printed nothing after ``Installed 55 packages``, so a stall and a slow fold
looked the same. Every entry point is tested through its REAL command line with the deadline variable set; the
in-process tests patch the thread and the dump so nothing leaks between tests.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEADLINE_ENV = "SILLY_KICKS_JOB_DEADLINE_EPOCH"
MARGIN_S = 600


def _env(dump_in_s: int = 3, **extra: str) -> dict[str, str]:
    """A child environment whose deadline dump fires ``dump_in_s`` seconds from now."""
    return {
        **os.environ,
        DEADLINE_ENV: str(int(time.time()) + MARGIN_S + dump_in_s),
        "SILLY_KICKS_JOB_LABEL": "t",
        **extra,
    }


def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, *argv],
        env=env or _env(dump_in_s=120),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _start_lines(stderr: str, role: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if re.match(rf"\[alive\] t {re.escape(role)} pid=\d+ start; ", ln)]


# ---- the real entry points ----------------------------------------------------------------------------


def test_alive_lines_name_the_stuck_function():
    code = textwrap.dedent(
        """
        import time
        from scripts.hf_jobs.liveness import start_job_watch
        start_job_watch("main", interval_s=0.5)
        def _stall():
            time.sleep(6)
        _stall()
        """
    )
    p = _run(["-c", code], env=_env(dump_in_s=3))
    assert p.returncode == 0, p.stderr
    assert len(_start_lines(p.stderr, "main")) == 1
    beats = [ln for ln in p.stderr.splitlines() if ln.startswith("[alive] t main ") and " up=" in ln]
    assert len(beats) >= 2, p.stderr
    assert all(re.search(r" at=\S+:\d+ _stall ", ln) for ln in beats), beats
    assert "Timeout (" in p.stderr and "_stall" in p.stderr.split("Timeout (", 1)[1]


@pytest.mark.parametrize(
    ("argv", "role", "fails"),
    [
        (["-m", "tests.scripts._toy_pipeline", "--plan", "--store", "{tmp}"], "main", False),
        (["-m", "scripts.hf_jobs.executor", "worker", "--worker-id", "w9", "--ctx", "{tmp}/missing"], "w9", True),
        (["-m", "scripts.hf_jobs.executor", "reduce", "--ctx", "{tmp}/missing"], "reduce", True),
        (["-m", "scripts.hf_jobs.datasets", "fetch"], "fetch", True),  # the usage error comes AFTER the start line
    ],
)
def test_entry_point_starts_the_watch(tmp_path, argv, role, fails):
    p = _run([a.replace("{tmp}", str(tmp_path)) for a in argv])
    if fails:
        assert p.returncode != 0
    assert len(_start_lines(p.stderr, role)) == 1, p.stderr


def test_pool_children_start_the_watch():
    code = textwrap.dedent(
        """
        from scripts.hf_jobs.executor import InnerPool
        pool = InnerPool(2, role="w0.pool")
        try:
            assert pool.map_keyed(abs, {"a": (-1,), "b": (-2,)}) == {"a": 1, "b": 2}
        finally:
            pool.shutdown()
        """
    )
    p = _run(["-c", code])
    assert p.returncode == 0, p.stderr
    # Each child gets its own index, so its own role and its own dump time (owner decision 2026-11-05, 3b).
    assert len(_start_lines(p.stderr, "w0.pool0")) == 1, p.stderr
    assert len(_start_lines(p.stderr, "w0.pool1")) == 1, p.stderr


# ---- in process ---------------------------------------------------------------------------------------


@pytest.fixture
def liveness(monkeypatch):
    """The module with its one-start flag reset, and the thread start and the dump recorded, not run."""
    import faulthandler
    import threading

    import scripts.hf_jobs.liveness as mod

    calls: dict[str, list] = {"dump": [], "thread": []}
    monkeypatch.setattr(mod, "_started", False)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls["dump"].append((a, k)))
    monkeypatch.setattr(threading.Thread, "start", lambda self: calls["thread"].append(self.name))
    mod.calls = calls  # type: ignore[attr-defined]
    yield mod
    faulthandler.cancel_dump_traceback_later()


def test_run_unit_exposes_the_current_unit(tmp_path):
    from scripts._stages import LocalStore, Pipeline, Stage, current_unit, make_context, read_result, run_unit
    from tests.scripts._stage_harness import PROV

    def _seen(key, ctx):
        unit = current_unit()
        assert unit is not None
        return list(unit[:2])

    def _boom(key, ctx):
        raise ValueError("boom")

    pipeline = Pipeline(
        name="slot",
        module=__name__,
        stages=(
            Stage(name="ok", units=lambda c: ["only"], work=_seen),
            Stage(name="bad", units=lambda c: ["only"], work=_boom),
        ),
    )
    ctx = make_context(pipeline, {}, PROV, store=LocalStore(tmp_path), data_root=None)

    assert run_unit(ctx, pipeline.stage("ok"), "only", worker="t") == "done"
    assert read_result(ctx.store, f"{ctx.generation('ok')}/only") == ["ok", "only"]
    assert current_unit() is None
    assert run_unit(ctx, pipeline.stage("bad"), "only", worker="t") == "failed"
    assert current_unit() is None


def test_parse_proc_stat_handles_odd_comm(liveness):
    line = "4242 ((a) b) (c) S 17 1 1 0 -1 0 0 0 0 0 250 50 0 0 20 0 1 0 1 1 1"
    assert liveness.parse_proc_stat(line) == (17, "S", 300)


def _fake_proc(root: pathlib.Path, pid: int, ppid: int, state: str, ticks: int, wchan: str | None) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(
        f"{pid} (python x) {state} {ppid} 0 0 0 0 0 0 0 0 0 {ticks} 0 0 0 20 0 1 0", encoding="ascii"
    )
    if wchan is not None:
        (d / "wchan").write_text(wchan, encoding="ascii")


def test_direct_children_skips_vanished_and_others(liveness, tmp_path):
    proc = tmp_path / "proc"
    _fake_proc(proc, 11, 10, "R", 500, "0")
    _fake_proc(proc, 12, 10, "D", 70, "request_wait_answer")
    _fake_proc(proc, 13, 99, "S", 5, "do_wait")  # another parent's child
    (proc / "14").mkdir()  # a process that exited mid-scan: its directory has no stat
    (proc / "self").mkdir()  # not a pid

    assert liveness.direct_children(10, proc=proc) == {11: ("R", 500, "-"), 12: ("D", 70, "request_wait_answer")}


def test_children_field_reports_cpu_delta(liveness):
    before = {11: ("R", 350, "-")}
    now = {11: ("R", 500, "-"), 12: ("S", 40, "do_wait")}
    assert liveness.children_field(now, before, 100) == "11:R:cpu+1.5:-,12:S:cpu+0.0:do_wait"


def test_no_proc_gives_dash(liveness, tmp_path):
    assert liveness.direct_children(10, proc=tmp_path / "absent") == {}
    assert liveness.children_field({}, {}, 100) == "-"


def test_watch_is_off_without_the_deadline(liveness, capsys):
    assert liveness.start_job_watch("main", env={}) is False
    assert liveness.calls == {"dump": [], "thread": []}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("bad", ["x", "", "1.5"])
def test_malformed_deadline_turns_liveness_off(liveness, capsys, bad):
    assert liveness.start_job_watch("main", env={DEADLINE_ENV: bad}) is False
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and "liveness off" in err[0]
    assert liveness.calls == {"dump": [], "thread": []}


def test_dump_delay_is_the_deadline_minus_the_margin(liveness, monkeypatch):
    assert liveness.start_job_watch("main", env={DEADLINE_ENV: str(int(time.time()) + 1000)}) is True
    (args, kwargs) = liveness.calls["dump"][0]
    timeout = kwargs.get("timeout", args[0] if args else None)
    assert 398 <= timeout <= 401 and kwargs.get("repeat") is False
    assert liveness.calls["thread"] == ["sk-alive"]

    monkeypatch.setattr(liveness, "_started", False)
    assert liveness.start_job_watch("main", env={DEADLINE_ENV: str(int(time.time()) - 5)}) is True
    (args, kwargs) = liveness.calls["dump"][1]
    assert kwargs.get("timeout", args[0] if args else None) == 1


def test_dump_times_of_one_job_never_share_a_second(liveness):
    # Task 8 D: main, w0 and w1 dumped in the same second and their tracebacks interleaved character by
    # character. A full cpu-upgrade job: main, the reduce, 8 workers and 8 pool children each.
    roles = ["main", "reduce"]
    roles += [f"w{i}" for i in range(8)] + [f"w{i}.pool{j}" for i in range(8) for j in range(8)]
    margins = [liveness.dump_margin_s(r) for r in roles]
    assert len(set(margins)) == len(roles)
    assert min(margins) == liveness.DUMP_MARGIN_S  # no process dumps later than the §11.4 margin


def test_start_line_names_numba(liveness, capsys):
    """HFU spec 13.2: every job states whether numba is active (the runtime proof of the core dependency)."""
    import numba

    assert liveness.start_job_watch("main", env={DEADLINE_ENV: str(int(time.time()) + 1000)}) is True
    out = capsys.readouterr()
    starts = [ln for ln in (out.out + out.err).splitlines() if " start; " in ln]
    assert len(starts) == 1 and f"numba={numba.__version__}" in starts[0], starts
