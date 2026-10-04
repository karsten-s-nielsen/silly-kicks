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
