"""Pluggable per-worker memory cap for the parallel launcher.

The launcher runs several trainer/driver processes at once on one host. Without a
per-process ceiling, one worker's peak RSS can drive the box into the OOM killer,
which on the DGX has silently killed sibling workers before. This module wraps a
child command with the strongest cap the host actually supports:

- ``cgroup`` (Linux + systemd): ``systemd-run --scope -p MemoryMax=<bytes>`` puts a
  hard ceiling on the child *and its descendants* -- the correct bound because the
  trainers themselves fork/spawn. This is the DGX path.
- ``rlimit`` (POSIX without systemd, e.g. macOS): a best-effort ``RLIMIT_AS`` set in
  a pre-exec hook. Bounds address space of the direct child only, not descendants.
- ``none`` (Windows, or when nothing is available): no cap; the launcher relies on
  admission backpressure alone.

Detection never hard-imports a POSIX-only module at load time, so the module imports
cleanly on Windows (where the launcher's unit tests run).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from typing import Literal

Backend = Literal["cgroup", "rlimit", "none"]

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


def detect_backend(explicit: Backend | None = None) -> Backend:
    """Pick the strongest supported cap backend for this host.

    ``explicit`` (from an operator flag) short-circuits detection so a run can be
    forced onto a weaker backend or ``none`` for debugging.
    """
    if explicit is not None:
        return explicit
    if _systemd_run_available():
        return "cgroup"
    if _rlimit_available():
        return "rlimit"
    return "none"


def wrap(backend: str, argv: Sequence[str], *, cap_bytes: int) -> WrapResult:
    """Return the argv (and optional pre-exec hook) that runs ``argv`` under ``cap_bytes``.

    The caller passes the result straight to ``subprocess.Popen(argv, preexec_fn=...)``.
    """
    cmd = list(argv)
    if backend == "cgroup":
        prefix = ["systemd-run", "--scope", "-p", f"MemoryMax={int(cap_bytes)}", "--"]
        return prefix + cmd, None
    if backend == "rlimit":
        import resource

        def _set_rlimit() -> None:  # pragma: no cover - runs in the forked child
            # `resource` is POSIX-only; a cross-platform type checker lacks its stubs.
            resource.setrlimit(resource.RLIMIT_AS, (int(cap_bytes), int(cap_bytes)))  # type: ignore[attr-defined]

        return cmd, _set_rlimit
    if backend == "none":
        return cmd, None
    raise ValueError(f"unknown memory-cap backend: {backend!r}")
