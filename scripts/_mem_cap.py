"""Pluggable per-worker memory cap for the parallel launcher.

The launcher runs several trainer/driver processes at once on one host. Without a
per-process ceiling, one worker's peak RSS can drive the box into the OOM killer,
which on the DGX has silently killed sibling workers before. This module wraps a
child command with the strongest cap the host actually supports:

- ``cgroup`` (Linux + systemd): ``systemd-run [--user] --scope -p MemoryMax=<bytes>`` puts a
  hard ceiling on the child *and its descendants* -- the correct bound because the
  trainers themselves fork/spawn. This is the DGX path. A non-root user must go through
  the user manager (``--user``; detached runs need ``loginctl enable-linger``): the system
  manager refuses it ("Interactive authentication required"). :func:`preflight` proves a
  scope can start before any worker launches.
- ``rss`` (Linux without systemd, e.g. an HF Jobs container; only for a WATCHED caller): no wrap. The
  caller polls the largest single RSS in each child's process tree (`largest_rss`) and kills a child
  whose tree holds a process over its cap (`kill_tree`). This bounds the quantity the planner sizes from
  (a unit's own peak RSS), per process like ``RLIMIT_AS``.
- ``rlimit`` (POSIX without systemd, e.g. macOS): a best-effort ``RLIMIT_AS`` set in
  a pre-exec hook. Bounds address space of the direct child only, not descendants. It is NOT an RSS
  bound: on real HF (2026-10-20) a numpy + pyarrow worker reserved about 1 GB more address space than
  its RSS, so an RSS-sized ``RLIMIT_AS`` failed every unit with ``MemoryError``.
- ``none`` (Windows, or when nothing is available): no cap; the launcher relies on
  admission backpressure alone.

Detection never hard-imports a POSIX-only module at load time, so the module imports
cleanly on Windows (where the launcher's unit tests run).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

Backend = Literal["cgroup", "rss", "rlimit", "none"]
_PROC = Path("/proc")

# argv, preexec_fn -- preexec_fn is None on every platform except the rlimit backend,
# where subprocess runs it in the child between fork and exec.
WrapResult = tuple[list[str], Callable[[], None] | None]


def _systemd_run_available() -> bool:
    """True when this is Linux and ``systemd-run`` resolves on PATH."""
    if not sys.platform.startswith("linux"):
        return False
    from shutil import which

    return which("systemd-run") is not None


def _rlimit_available() -> bool:
    """True when ``resource.RLIMIT_AS`` and ``os.fork`` both exist (POSIX)."""
    if not hasattr(os, "fork"):
        return False
    try:
        import resource
    except ImportError:
        return False
    return hasattr(resource, "RLIMIT_AS")


def _proc_available(proc: Path = _PROC) -> bool:
    """True when a Linux ``/proc`` with per-process ``statm`` exists."""
    return (proc / "self" / "statm").is_file()


def detect_backend(explicit: Backend | None = None, *, watched: bool = False, proc: Path = _PROC) -> Backend:
    """Pick the strongest supported cap backend for this host.

    ``explicit`` (from an operator flag) short-circuits detection so a run can be
    forced onto a weaker backend or ``none`` for debugging. ``watched``: the caller enforces the
    ``rss`` backend itself (`largest_rss` + `kill_tree`), so it replaces ``rlimit`` where ``/proc``
    exists. An unwatched caller (the DGX launcher) is unchanged.
    """
    if explicit is not None:
        return explicit
    if _systemd_run_available():
        return "cgroup"
    if watched and _proc_available(proc):
        return "rss"
    if _rlimit_available():
        return "rlimit"
    return "none"


def wrap(backend: str, argv: Sequence[str], *, cap_bytes: int) -> WrapResult:
    """Return the argv (and optional pre-exec hook) that runs ``argv`` under ``cap_bytes``.

    The caller passes the result straight to ``subprocess.Popen(argv, preexec_fn=...)``.
    """
    cmd = list(argv)
    if backend == "cgroup":
        # geteuid is POSIX-only; the cgroup backend is only ever detected on Linux.
        manager = [] if getattr(os, "geteuid", lambda: 0)() == 0 else ["--user"]
        prefix = ["systemd-run", *manager, "--scope", "-p", f"MemoryMax={int(cap_bytes)}", "--"]
        return prefix + cmd, None
    if backend == "rlimit":
        import resource

        def _set_rlimit() -> None:  # pragma: no cover - runs in the forked child
            # `resource` is POSIX-only; a cross-platform type checker lacks its stubs.
            resource.setrlimit(resource.RLIMIT_AS, (int(cap_bytes), int(cap_bytes)))  # type: ignore[attr-defined]

        return cmd, _set_rlimit
    if backend in ("rss", "none"):  # rss: the caller watches the child; nothing to wrap
        return cmd, None
    raise ValueError(f"unknown memory-cap backend: {backend!r}")


# ---- the rss backend: process-tree RSS from /proc ---------------------------------------------------


def _page_size() -> int:
    sysconf = getattr(os, "sysconf", None)
    try:
        return int(sysconf("SC_PAGE_SIZE")) if sysconf is not None else 4096
    except (ValueError, OSError):
        return 4096


def _children(proc: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text(encoding="ascii", errors="replace")
            # "pid (comm) state ppid ...": comm may hold spaces and parens, so split after the LAST ')'
            ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue  # the process exited mid-scan
        out.setdefault(ppid, []).append(int(d.name))
    return out


def process_tree(pid: int, *, proc: Path = _PROC) -> list[int]:
    """``pid`` and every descendant (a worker and its inner-pool children), parents first."""
    kids = _children(proc)
    out, todo = [], [pid]
    while todo:
        p = todo.pop(0)
        out.append(p)
        todo.extend(sorted(kids.get(p, ())))
    return out


def largest_rss(pid: int, *, proc: Path = _PROC) -> tuple[int, int] | None:
    """``(pid, bytes)`` of the biggest single resident set (``statm`` resident pages) among ``pid`` and
    its descendants, or None when ``pid`` is gone or there is no ``/proc``.

    The cap is PER PROCESS, like ``RLIMIT_AS`` (which each inner-pool child also received): the planner
    sizes it from a unit's own peak RSS (``VmHWM``, children excluded), and an inner-parallel worker's
    pool children are each sized like one unit. A tree sum would kill a healthy inner-parallel worker."""
    if not (proc / str(pid)).is_dir():
        return None
    page, best = _page_size(), None
    for p in process_tree(pid, proc=proc):
        try:
            rss = int((proc / str(p) / "statm").read_text(encoding="ascii").split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
        if best is None or rss > best[1]:
            best = (p, rss)
    return best


def kill_tree(pid: int, *, proc: Path = _PROC) -> bool:
    """SIGKILL ``pid`` and its descendants, leaves first. False when there is no ``/proc`` (the caller
    kills the direct child itself)."""
    if not _proc_available(proc):
        return False
    import signal

    for p in reversed(process_tree(pid, proc=proc)):
        try:
            os.kill(p, signal.SIGKILL)  # type: ignore[attr-defined]  # POSIX-only; guarded by /proc
        except OSError:
            continue
    return True


#: How long the one-off scope probe may take before it counts as a failure to start (a hung user manager).
_PREFLIGHT_TIMEOUT_S = 30


def preflight(backend: str, *, cap_bytes: int) -> None:
    """Prove the backend can start a capped child, ONCE, before any worker launches.

    A cgroup scope that cannot start would otherwise fail every worker at launch and burn every
    relaunch. Runs the wrapped no-op ``true`` and raises ``RuntimeError`` with its stderr on failure.
    The other backends need no probe.
    """
    if backend != "cgroup":
        return
    argv, _ = wrap(backend, ["true"], cap_bytes=cap_bytes)
    hint = "-- enable the user manager (loginctl enable-linger) or pass --mem-backend rlimit|none"
    try:
        # argv is our own fixed probe (systemd-run + `true`), not untrusted input.
        res = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=_PREFLIGHT_TIMEOUT_S)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:  # no systemd-run on PATH, or a probe that hangs
        raise RuntimeError(f"memory-cap backend 'cgroup' cannot start a scope ({argv[0]}): {exc} {hint}") from exc
    if res.returncode != 0:
        raise RuntimeError(
            f"memory-cap backend 'cgroup' cannot start a scope ({' '.join(argv[:-2])}): {res.stderr.strip()} {hint}"
        )
