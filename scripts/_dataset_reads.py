"""Bounded reads from read-only dataset mounts (ADR-128, the dataset read gate).

WHY THIS EXISTS. On real HF (2026-10-20 probes: the tc3 dataset mount, 360 files / 15.7 GB, one
`cpu-upgrade` job alone) whole-file sequential reads were clean at 1, 2 and 4 concurrent readers
(184 s, 90 s, 65 s), stalled about 183 s per file at 6, and failed every large file at 8 with
`OSError: [Errno 5] Input/output error` after 180.6 s, with 0 bytes read. Pyarrow reading parquet
DIRECTLY on the mount (many small range reads per file) took 126-462 s per file at only 4 readers. The
fix measured clean: 8 workers, at most 4 whole-file reads at a time, each file parsed from memory
(360 of 360 files, slowest 10.5 s, 69 s in all). This module is that rule's ONE implementation: a
job-wide counting semaphore over every worker process, and a whole-file read under it.

The gate is active only when ``SILLY_KICKS_DATASET_READ_CAP`` is set: `launch.job_env` sets it in every HF
job that mounts a dataset (the smallest ``read_concurrency`` that `datasets.toml` declares). Unset (a
laptop, the DGX, the test suite) the reads are the same whole-file reads with no limit.

The semaphore is ``cap`` lock files with an exclusive non-blocking lock each (``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows). The kernel drops a lock when its holder dies, so an OOM-killed worker
never leaks a slot. Stdlib only.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import IO

#: In-job: the most concurrent whole-file reads from dataset mounts, across every worker of the job.
READ_CAP_ENV = "SILLY_KICKS_DATASET_READ_CAP"
#: Where the slot lock files live (default: a fixed directory under the system temp directory).
READ_GATE_DIR_ENV = "SILLY_KICKS_DATASET_READ_GATE_DIR"
_POLL_S = 0.05

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fh: IO[bytes]) -> bool:
        fh.seek(0)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fh: IO[bytes]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fh: IO[bytes]) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(fh: IO[bytes]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_cap() -> int | None:
    """The job's read cap, or None (no gate). A value that is not a positive integer is an error."""
    raw = os.environ.get(READ_CAP_ENV, "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError(f"{READ_CAP_ENV} must be a positive integer, got {raw!r}") from None
    if cap < 1:
        raise ValueError(f"{READ_CAP_ENV} must be a positive integer, got {raw!r}")
    return cap


def _gate_dir() -> pathlib.Path:
    base = os.environ.get(READ_GATE_DIR_ENV, "").strip()
    return pathlib.Path(base) if base else pathlib.Path(tempfile.gettempdir()) / "silly-kicks-dataset-read-gate"


def _acquire(gate: pathlib.Path, cap: int) -> IO[bytes]:
    while True:
        for i in range(cap):
            fh = open(gate / f"slot-{i}.lock", "a+b")  # held open until `read_slot` releases the lock
            if _try_lock(fh):
                return fh
            fh.close()
        time.sleep(_POLL_S)


_HELD = threading.local()


@contextlib.contextmanager
def read_slot() -> Iterator[None]:
    """Hold one of the job's read slots for the body (a no-op when no cap is set). Every read of a
    dataset-mount file runs inside one: a whole-file read (`read_bytes`) or a parquet footer read.
    Re-entrant per thread: a read inside a held slot reuses it (a second slot would deadlock at cap 1)."""
    cap = read_cap()
    if cap is None or getattr(_HELD, "depth", 0):
        _HELD.depth = getattr(_HELD, "depth", 0) + 1
        try:
            yield
        finally:
            _HELD.depth -= 1
        return
    gate = _gate_dir()
    gate.mkdir(parents=True, exist_ok=True)
    held = _acquire(gate, cap)
    _HELD.depth = 1
    try:
        yield
    finally:
        _HELD.depth = 0
        try:
            _unlock(held)
        finally:
            held.close()


def read_bytes(path: str | os.PathLike) -> bytes:
    """The whole file, read sequentially under one read slot."""
    with read_slot(), open(path, "rb") as fh:
        return fh.read()
