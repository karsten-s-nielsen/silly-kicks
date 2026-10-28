"""Task 2 guard: the pluggable memory-cap backend (cgroup / rlimit / none)."""

import subprocess
import sys

import pytest

from scripts import _mem_cap

GiB = 1024**3


def test_cgroup_wrap_uses_the_user_manager_when_not_root(monkeypatch):
    # A non-root `systemd-run --scope` asks the SYSTEM manager and is refused ("Interactive authentication
    # required"): every launcher worker failed on the DGX (combined-cycle Task 17). Non-root needs --user.
    monkeypatch.setattr(_mem_cap.os, "geteuid", lambda: 1000, raising=False)
    argv, preexec = _mem_cap.wrap("cgroup", ["python", "x.py"], cap_bytes=14 * GiB)
    assert argv[:4] == ["systemd-run", "--user", "--scope", "-p"]
    assert "MemoryMax=15032385536" in " ".join(argv)  # 14 GiB
    assert argv[-3:] == ["--", "python", "x.py"]
    assert preexec is None


def test_cgroup_wrap_uses_the_system_manager_as_root(monkeypatch):
    monkeypatch.setattr(_mem_cap.os, "geteuid", lambda: 0, raising=False)
    argv, _ = _mem_cap.wrap("cgroup", ["python", "x.py"], cap_bytes=14 * GiB)
    assert argv[:3] == ["systemd-run", "--scope", "-p"]


def test_preflight_refuses_a_cgroup_scope_that_cannot_start(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="Failed to start transient scope unit: Interactive authentication required."
        )

    monkeypatch.setattr(_mem_cap.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(_mem_cap.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Interactive authentication required"):
        _mem_cap.preflight("cgroup", cap_bytes=GiB)
    assert calls and calls[0][:2] == ["systemd-run", "--user"] and calls[0][-1] == "true"  # the wrapped no-op


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("systemd-run"), subprocess.TimeoutExpired(["systemd-run"], 30)],
    ids=["no-systemd-run", "probe-hangs"],
)
def test_preflight_turns_a_missing_or_hanging_probe_into_a_refusal(monkeypatch, error):
    # B r3 m-3: a forced cgroup backend without systemd-run, or a probe that hangs, must surface as the same
    # clean refusal (RuntimeError -> the launcher's REFUSED), never a traceback or an endless wait.
    seen: dict = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        raise error

    monkeypatch.setattr(_mem_cap.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="cannot start a scope"):
        _mem_cap.preflight("cgroup", cap_bytes=GiB)
    assert seen.get("timeout") is not None


def test_preflight_passes_a_working_scope_and_never_probes_other_backends(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(_mem_cap.subprocess, "run", fake_run)
    _mem_cap.preflight("cgroup", cap_bytes=GiB)
    assert len(calls) == 1
    _mem_cap.preflight("rlimit", cap_bytes=GiB)
    _mem_cap.preflight("none", cap_bytes=GiB)
    assert len(calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS preexec is POSIX-only")
def test_rlimit_wrap_returns_preexec_not_argv_change():
    argv, preexec = _mem_cap.wrap("rlimit", ["python", "x.py"], cap_bytes=14 * GiB)
    assert argv == ["python", "x.py"]  # command unchanged
    assert callable(preexec)  # RLIMIT_AS set in the child pre-exec


def test_none_wrap_is_a_passthrough():
    argv, preexec = _mem_cap.wrap("none", ["python", "x.py"], cap_bytes=1)
    assert argv == ["python", "x.py"] and preexec is None


def test_explicit_backend_overrides_detection():
    assert _mem_cap.detect_backend(explicit="none") == "none"


def test_detect_returns_a_valid_backend_on_this_os():
    assert _mem_cap.detect_backend() in {"cgroup", "rlimit", "none"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows: no cgroup, no fork/RLIMIT_AS preexec")
def test_windows_selects_none():
    assert _mem_cap.detect_backend() == "none"


# ---- the rss backend (ADR-128: real HF, 2026-10-20) ------------------------------------------------
# RLIMIT_AS caps ADDRESS SPACE. The planner sizes caps from measured peak RSS, and on real HF a numpy +
# pyarrow worker reserved about 1 GB more address space than its RSS: every planned job hit
# `MemoryError` at 1.4-1.7 GB RSS under a 2.51 GB RLIMIT_AS. A WATCHED caller (the executor) polls each
# child's process-tree RSS instead -- the quantity it sizes from, like a cgroup MemoryMax.


def _fake_proc(root, procs: dict[int, tuple[int, int]]):
    """A /proc tree: ``{pid: (ppid, resident_pages)}``; comm holds spaces and parens, like a real one."""
    root.mkdir(parents=True, exist_ok=True)
    for pid, (ppid, pages) in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} (py (worker) x) S {ppid} 1 1 0 -1", encoding="ascii")
        (d / "statm").write_text(f"999999 {pages} 10 1 0 5 0", encoding="ascii")
    (root / "self").mkdir(exist_ok=True)
    (root / "self" / "statm").write_text("1 1 1 1 0 1 0", encoding="ascii")
    return root


def test_largest_rss_is_the_biggest_single_process_of_a_worker_tree(tmp_path):
    """The cap is PER PROCESS: the planner sizes it from a unit's own peak RSS (VmHWM, children excluded),
    and an inner-parallel worker's pool children are each sized like one unit. So the watched quantity
    is the largest member of the worker's tree, never their sum (Reviewer B CCHC-IMPL-05)."""
    proc = _fake_proc(tmp_path / "proc", {10: (1, 100), 11: (10, 90), 12: (10, 95), 13: (11, 30), 20: (1, 1000)})
    page = _mem_cap._page_size()
    assert _mem_cap.largest_rss(10, proc=proc) == (10, 100 * page)  # not the 315-page sum
    assert _mem_cap.largest_rss(11, proc=proc) == (11, 90 * page)  # an unrelated bigger process is ignored
    assert _mem_cap.largest_rss(99, proc=proc) is None  # gone: nothing to measure
    assert sorted(_mem_cap.process_tree(10, proc=proc)) == [10, 11, 12, 13]


def test_an_inner_pool_child_over_the_cap_is_the_one_reported(tmp_path):
    proc = _fake_proc(tmp_path / "proc", {10: (1, 100), 11: (10, 500), 12: (10, 95)})
    assert _mem_cap.largest_rss(10, proc=proc) == (11, 500 * _mem_cap._page_size())


def test_total_rss_sums_the_tree_where_largest_takes_the_max(tmp_path):
    """Reviewer B D1-IMPL-18: an inner_parallel worker's pool shares the box, so its watched quantity is the
    TREE TOTAL (the unit process PLUS its pool children) against the tree budget -- not the largest single
    process. total_rss SUMS the tree; largest_rss takes the MAX. The root pid is reported for the kill message."""
    proc = _fake_proc(tmp_path / "proc", {10: (1, 100), 11: (10, 90), 12: (10, 95), 13: (11, 30), 20: (1, 1000)})
    page = _mem_cap._page_size()
    assert _mem_cap.total_rss(10, proc=proc) == (10, (100 + 90 + 95 + 30) * page)  # the 315-page tree sum
    assert _mem_cap.total_rss(11, proc=proc) == (11, (90 + 30) * page)  # subtree of 11 is {11, 13}
    assert _mem_cap.total_rss(99, proc=proc) is None  # gone: nothing to measure


def test_a_watched_caller_on_linux_without_systemd_gets_rss_not_rlimit(tmp_path, monkeypatch):
    proc = _fake_proc(tmp_path / "proc", {})
    monkeypatch.setattr(_mem_cap, "_systemd_run_available", lambda: False)
    monkeypatch.setattr(_mem_cap, "_rlimit_available", lambda: True)
    assert _mem_cap.detect_backend(watched=True, proc=proc) == "rss"
    assert _mem_cap.detect_backend(proc=proc) == "rlimit"  # an unwatched caller (the DGX launcher) is unchanged
    assert _mem_cap.detect_backend(watched=True, proc=tmp_path / "no-proc") == "rlimit"
    monkeypatch.setattr(_mem_cap, "_systemd_run_available", lambda: True)
    assert _mem_cap.detect_backend(watched=True, proc=proc) == "cgroup"  # the DGX keeps its cgroup


def test_rss_wrap_is_a_passthrough_the_caller_watches():
    argv, preexec = _mem_cap.wrap("rss", ["python", "x.py"], cap_bytes=GiB)
    assert argv == ["python", "x.py"] and preexec is None
    _mem_cap.preflight("rss", cap_bytes=GiB)  # no probe needed
