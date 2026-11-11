"""Job liveness (HFU spec §11, Change 4): every covered in-job Python process prints an ``[alive]`` start line at
once and a line every 300 s (its unit, the unit's age, its main thread's innermost Python frame and its child
processes), and prints every thread's stack 600 s before the job's deadline (the deadline dump).

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
    """The beats. The deadline is far away (the `_run` default), so no dump runs in this child."""
    code = textwrap.dedent(
        """
        import time
        from scripts.hf_jobs.liveness import start_job_watch
        start_job_watch("main", interval_s=0.5)
        def _stall():
            time.sleep(3)
        _stall()
        """
    )
    p = _run(["-c", code])
    assert p.returncode == 0, p.stderr
    assert len(_start_lines(p.stderr, "main")) == 1
    beats = [ln for ln in p.stderr.splitlines() if ln.startswith("[alive] t main ") and " up=" in ln]
    assert len(beats) >= 2, p.stderr
    assert all(re.search(r" at=\S+:\d+ _stall ", ln) for ln in beats), beats
    assert "Timeout (" not in p.stderr


def test_the_deadline_dump_names_the_stuck_function():
    """The dump. Two rules keep this test independent of the PC's load (it failed once under load, 2026-11-10):

    - The child sets its OWN deadline after the numba import is warm. `start_job_watch` arms the dump BEFORE its
      numba probe, so a deadline 3 s from the PARENT's clock fired inside that import when the import was slow:
      the dump then named the import, not `_stall`.
    - No beat runs (a long interval), and the assert reads the DUMP's own line format. A beat line also holds
      the word `_stall`.

    The dump comes from a Python thread, never from the `faulthandler` watchdog: that watchdog reads the frames of
    a running thread without the GIL and ended working processes with SIGSEGV (HF jobs `6af366809737d9a9e0914f3a`
    and `6af3700bab34849891acb4a8`: 18 of 40 processes in an xgboost fit, 38 of 40 in pandas work, 40 of 40 in
    `infer_ball_carrier`).
    """
    code = textwrap.dedent(
        f"""
        import os
        import time
        from silly_kicks._numba_dispatch import numba_version
        numba_version()
        from scripts.hf_jobs.liveness import start_job_watch
        os.environ[{DEADLINE_ENV!r}] = str(int(time.time()) + {MARGIN_S} + 3)
        start_job_watch("main", interval_s=3600)
        def _stall():
            time.sleep(5)
        _stall()
        """
    )
    p = _run(["-c", code])
    assert p.returncode == 0, p.stderr
    assert len(_start_lines(p.stderr, "main")) == 1
    assert "Timeout (" not in p.stderr, p.stderr  # the header of a `faulthandler` dump
    assert len(re.findall(r"^\[alive\] t main pid=\d+ deadline dump: ", p.stderr, flags=re.M)) == 1, p.stderr
    assert re.search(r'File "<string>", line \d+ in _stall', p.stderr), p.stderr


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
    """The module with its one-start flag reset. No thread starts: each start is recorded, and the dump
    thread's target runs at once as a recorder, so ``calls["delay"]`` holds the ``(role, delay)`` that
    `start_job_watch` gave it. ``calls["dump"]`` records a call of the `faulthandler` watchdog (never made)."""
    import faulthandler
    import threading

    import scripts.hf_jobs.liveness as mod

    calls: dict[str, list] = {"dump": [], "thread": [], "delay": []}

    def start(self):
        calls["thread"].append(self.name)
        if self.name == "sk-dump":
            self.run()  # the recorder below, with the arguments the watch gave the thread

    monkeypatch.setattr(mod, "_started", False)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls["dump"].append((a, k)))
    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(mod, "_dump_later", lambda role, delay_s, stop: calls["delay"].append((role, delay_s)))
    mod.calls = calls  # type: ignore[attr-defined]
    return mod


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
    assert liveness.calls == {"dump": [], "thread": [], "delay": []}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("bad", ["x", "", "1.5"])
def test_malformed_deadline_turns_liveness_off(liveness, capsys, bad):
    assert liveness.start_job_watch("main", env={DEADLINE_ENV: bad}) is False
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 1 and "liveness off" in err[0]
    assert liveness.calls == {"dump": [], "thread": [], "delay": []}


def test_dump_delay_is_the_deadline_minus_the_margin(liveness):
    now = 1_000_000.0
    assert liveness.dump_delay_s(1_001_000, "main", now) == 400
    assert liveness.dump_delay_s(1_001_000, "w0", now) == 390
    assert liveness.dump_delay_s(999_995, "main", now) == 1  # a process that starts inside the margin: the floor


@pytest.mark.parametrize(("role", "margin"), [("main", 600), ("reduce", 601), ("w0", 610), ("w1.pool2", 623)])
def test_the_dump_thread_gets_the_delay_of_its_role(liveness, monkeypatch, role, margin):
    """`start_job_watch` gives the dump thread `dump_delay_s(deadline, role, now)`: the deadline minus the
    role's margin for a far deadline, and the 1 s floor for a process that starts inside its margin."""
    before = time.time()
    assert liveness.start_job_watch(role, env={DEADLINE_ENV: str(int(before) + 1000)}) is True
    after = time.time()
    ((got_role, delay),) = liveness.calls["delay"]
    assert got_role == role
    assert int(before) + 1000 - margin - after <= delay <= int(before) + 1000 - margin - before
    assert delay > 300  # far from the floor

    monkeypatch.setattr(liveness, "_started", False)
    assert liveness.start_job_watch(role, env={DEADLINE_ENV: str(int(time.time()) + margin - 5)}) is True
    assert liveness.calls["delay"][1] == (role, 1)


def test_start_dump_thread_starts_one_named_daemon_thread(liveness):
    thread = liveness.start_dump_thread("plan", 5.0)
    assert thread.name == "sk-dump" and thread.daemon is True
    assert liveness.calls["thread"] == ["sk-dump"] and liveness.calls["delay"] == [("plan", 5.0)]


@pytest.mark.parametrize(("value", "delay"), [("600", 570.0), ("31.5", 1.5), ("20", 1.0)])
def test_the_plan_deadline_starts_the_dump_thread_30_s_before_it(liveness, value, delay):
    """HFU spec 13.4: a `--plan` process prints its stacks `PLAN_DUMP_LEAD_S` before the driver ends it (1 s at
    the least), with the dump of the job watch -- never with the `faulthandler` watchdog."""
    from scripts._stages import PLAN_DUMP_LEAD_S, arm_plan_deadline

    assert PLAN_DUMP_LEAD_S == 30
    assert arm_plan_deadline({"SILLY_KICKS_PLAN_DEADLINE_S": value}) is True
    assert liveness.calls["delay"] == [("plan", delay)] and liveness.calls["thread"] == ["sk-dump"]
    assert liveness.calls["dump"] == []  # `faulthandler.dump_traceback_later` is never called


@pytest.mark.parametrize("env", [{}, {"SILLY_KICKS_PLAN_DEADLINE_S": "soon"}])
def test_no_plan_deadline_starts_nothing(liveness, env):
    from scripts._stages import arm_plan_deadline

    assert arm_plan_deadline(env) is False
    assert liveness.calls == {"dump": [], "thread": [], "delay": []}


def test_the_watch_starts_a_dump_thread_and_never_the_faulthandler_watchdog(liveness):
    assert liveness.start_job_watch("main", env={DEADLINE_ENV: str(int(time.time()) + 1000)}) is True
    assert liveness.calls["thread"] == ["sk-dump", "sk-alive"]  # the dump is armed before the numba probe
    assert liveness.calls["dump"] == []  # `faulthandler.dump_traceback_later` is never called


def test_stack_dump_lists_every_thread_main_first():
    import threading

    import scripts.hf_jobs.liveness as mod

    gate, ready = threading.Event(), threading.Event()

    def _parked_function():
        ready.set()
        gate.wait(30)

    t = threading.Thread(target=_parked_function, name="parked-thread", daemon=True)
    t.start()
    assert ready.wait(10)
    try:
        text = mod.stack_dump("w3")
    finally:
        gate.set()
        t.join(10)
    lines = text.splitlines()
    assert re.fullmatch(r"\[alive\] \S+ w3 pid=\d+ deadline dump: \d+ thread\(s\), most recent call first", lines[0])
    assert lines[1].startswith("Thread MainThread ") and lines[1].endswith(" [main]:")
    assert re.fullmatch(r'  File ".*liveness\.py", line \d+ in stack_dump', lines[2])  # the most recent call first
    assert re.search(
        r'  File ".*test_job_liveness\.py", line \d+ in test_stack_dump_lists_every_thread_main_first', text
    )
    parked = text.split("Thread parked-thread ", 1)[1]
    assert " in _parked_function" in parked and " [main]" not in parked


def test_stack_dump_caps_a_deep_stack():
    import scripts.hf_jobs.liveness as mod

    def deep(k: int) -> str:
        return mod.stack_dump("main") if k == 0 else deep(k - 1)

    lines = deep(mod.MAX_DUMP_FRAMES + 50).splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("Thread ")]
    block = lines[heads[0] + 1 : heads[1] if len(heads) > 1 else None]  # the main thread's frames
    assert sum(ln.startswith("  File ") for ln in block) == mod.MAX_DUMP_FRAMES and block[-1] == "  ..."


def test_say_writes_one_record_in_one_write(monkeypatch):
    """Two threads of a process write (the beats and the dump), and so does the main thread. A record whose
    text and newline are two writes lets another record land between them (seen in the HF proof job: a dump
    header on the end of another line)."""
    import scripts.hf_jobs.liveness as mod

    writes: list[str] = []

    class _Stderr:
        def write(self, text: str) -> int:
            writes.append(text)
            return len(text)

        def flush(self) -> None:
            writes.append("<flush>")

    monkeypatch.setattr(sys, "stderr", _Stderr())
    mod._say("a record")
    assert writes == ["a record\n", "<flush>"]


def test_dump_later_prints_once_never_raises_and_stops(capsys, monkeypatch):
    import threading

    import scripts.hf_jobs.liveness as mod

    mod._dump_later("main", 0.0, threading.Event())
    assert capsys.readouterr().err.count("deadline dump: ") == 1

    stop = threading.Event()
    stop.set()
    mod._dump_later("main", 0.0, stop)  # the watch was stopped: no dump
    assert capsys.readouterr().err == ""

    def boom(role):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "stack_dump", boom)
    mod._dump_later("main", 0.0, threading.Event())  # liveness never fails the job it reports on
    assert "deadline dump failed: RuntimeError: boom" in capsys.readouterr().err


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
