"""Task 2 guard: the pluggable memory-cap backend (cgroup / rlimit / none)."""

import sys

import pytest

from scripts import _mem_cap

GiB = 1024**3


def test_cgroup_wrap_builds_systemd_run_scope_argv():
    argv, preexec = _mem_cap.wrap("cgroup", ["python", "x.py"], cap_bytes=14 * GiB)
    assert argv[:3] == ["systemd-run", "--scope", "-p"]
    assert "MemoryMax=15032385536" in " ".join(argv)  # 14 GiB
    assert argv[-3:] == ["--", "python", "x.py"]
    assert preexec is None


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
