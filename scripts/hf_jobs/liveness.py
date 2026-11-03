"""Job liveness: an in-job process shows it is alive, and a timed-out one leaves its stacks (HFU spec §11).

E12 (cycle 1): a cv job ran its 2 h and printed nothing after ``Installed 55 packages``; a plain `Stage` printed
nothing per unit, so a stall and a slow fold looked the same. Every covered in-job Python process (the stage
main, each executor worker, the reduce subprocess, the ``datasets fetch`` pre-step and each `InnerPool` child)
calls `start_job_watch` at its entry point. It then:

- prints ONE ``[alive]`` start line at once, naming the UTC time its deadline dump fires;
- prints an ``[alive]`` line every `ALIVE_INTERVAL_S` from a daemon thread: the unit the process runs (from
  `scripts._stages.current_unit`) and its age, the main thread's innermost PYTHON frame (a C call such as a
  FUSE ``read`` shows the line that made it) and its direct child processes from ``/proc`` -- so a third-party
  pool child (joblib/loky under scikit-learn) is reported by its parent (owner decision 2026-11-05);
- arms `faulthandler.dump_traceback_later` `dump_margin_s(role)` before the job deadline: at least
  `DUMP_MARGIN_S`, and a different second for each process of one job, so two dumps do not interleave in the log
  (Task 8 D: three same-second dumps mixed character by character; owner decision 2026-11-05). That watchdog is a
  C thread that needs no GIL, so the dump fires even when a C extension holds the GIL and stops the alive thread.

Everything starts only when `DEADLINE_ENV` is set: `launch.job_command` exports it at bash start. Local runs,
tests and the controller print nothing new. The watch reads the unit slot, one frame and ``/proc``; it touches
no store, result, token or generation, so no artifact byte can change. It never raises.

"""

from __future__ import annotations

import datetime as dt
import faulthandler
import os
import pathlib
import sys
import threading
import time
from collections.abc import Mapping
from typing import Final

DEADLINE_ENV: Final = "SILLY_KICKS_JOB_DEADLINE_EPOCH"
ALIVE_INTERVAL_S: Final = 300
DUMP_MARGIN_S: Final = 600
DUMP_STAGGER_S: Final = 10  # one band per executor worker: the worker, then its pool children 1 s apart
_POOL_SLOTS: Final = DUMP_STAGGER_S - 1
_FIXED_OFFSETS: Final = {"reduce": 1}  # it runs beside `main`; `fetch` runs before it

_started = False  # one watch per process: a second call is a no-op


def dump_margin_s(role: str) -> int:
    """Seconds before the deadline at which ``role``'s dump fires. ``main`` (and every other role) dumps
    `DUMP_MARGIN_S` before it, ``reduce`` 1 s earlier. Worker ``w<i>`` owns the band that
    starts ``DUMP_STAGGER_S * (i + 1)`` s earlier; its pool child ``w<i>.pool<j>`` dumps ``1 + j % 9`` s into the
    band. So the processes of one job dump in different seconds (only a worker with more than 9 pool children
    shares a second between two of them)."""
    worker, _, pool = role.partition(".pool")
    offset = _FIXED_OFFSETS.get(role, 0)
    if worker[:1] == "w" and worker[1:].isdigit():
        offset = DUMP_STAGGER_S * (int(worker[1:]) + 1)
        if role != worker:
            offset += 1 + (int(pool) if pool.isdigit() else 0) % _POOL_SLOTS
    return DUMP_MARGIN_S + offset


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _label() -> str:
    return os.environ.get("SILLY_KICKS_JOB_LABEL") or "-"


def parse_proc_stat(text: str) -> tuple[int, str, int]:
    """(ppid, state, utime + stime clock ticks) from one ``/proc/<pid>/stat`` line. The ``comm`` field may hold
    spaces and parentheses, so the fields are taken after its LAST ``)``."""
    rest = text[text.rindex(")") + 1 :].split()
    return int(rest[1]), rest[0], int(rest[11]) + int(rest[12])


def direct_children(pid: int, *, proc: pathlib.Path = pathlib.Path("/proc")) -> dict[int, tuple[str, int, str]]:
    """{child pid: (state, cpu ticks, wchan or ``-``)} for every DIRECT child of ``pid``. A process that exits
    during the scan is skipped; without ``proc`` (not Linux) the result is empty."""
    out: dict[int, tuple[str, int, str]] = {}
    try:
        entries = list(proc.iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            ppid, state, ticks = parse_proc_stat((d / "stat").read_text(encoding="ascii", errors="replace"))
        except (OSError, ValueError, IndexError):
            continue
        if ppid != pid:
            continue
        try:
            wchan = (d / "wchan").read_text(encoding="ascii", errors="replace").strip()
        except OSError:
            wchan = ""
        out[int(d.name)] = (state, ticks, wchan if wchan and wchan != "0" else "-")
    return out


def children_field(now: Mapping[int, tuple[str, int, str]], before: Mapping, clk_tck: int) -> str:
    """``pid:state:cpu+<seconds since the previous line>:wchan`` per child, in pid order; ``-`` for none."""
    parts = []
    for pid in sorted(now):
        state, ticks, wchan = now[pid]
        prev = before[pid][1] if pid in before else ticks
        parts.append(f"{pid}:{state}:cpu+{(ticks - prev) / clk_tck:.1f}:{wchan}")
    return ",".join(parts) or "-"


def _main_frame() -> str:
    frame = sys._current_frames().get(threading.main_thread().ident or -1)
    if frame is None:
        return "-"
    code = frame.f_code
    return f"{code.co_filename}:{frame.f_lineno} {code.co_name}"


def alive_line(role: str, *, started: float, now: float, children: str) -> str:
    """One ``[alive]`` line: who, how long up, which unit for how long, where the main thread is, its children."""
    from scripts._stages import current_unit

    unit = current_unit()
    what = f"unit={unit[0]}/{unit[1]} age={now - unit[2]:.0f}s" if unit else "unit=- age=-"
    return (
        f"[alive] {_label()} {role} pid={os.getpid()} up={now - started:.0f}s {what} at={_main_frame()} "
        f"children={children}"
    )


def _beat(role: str, interval_s: float, stop: threading.Event) -> None:
    started = time.monotonic()
    sysconf = getattr(os, "sysconf", None)  # POSIX only; `/proc` exists only where it does
    clk_tck = int(sysconf("SC_CLK_TCK")) if sysconf is not None else 100
    before: dict = {}
    while not stop.wait(interval_s):
        try:
            now = direct_children(os.getpid())
            _say(alive_line(role, started=started, now=time.monotonic(), children=children_field(now, before, clk_tck)))
            before = now
        except Exception as exc:  # broad BY DESIGN: liveness must never fail the job it reports on
            _say(f"[alive] {_label()} {role}: alive line failed: {type(exc).__name__}: {exc}")


def start_job_watch(role: str, *, env: Mapping[str, str] = os.environ, interval_s: float = ALIVE_INTERVAL_S) -> bool:
    """Start this process's watch (module docstring). Returns True when started; False when the deadline
    variable is unset, malformed (one warning line) or the watch already runs. Never raises."""
    global _started
    value = env.get(DEADLINE_ENV)
    if value is None or _started:
        return False
    try:
        deadline = int(value)
    except ValueError:
        _say(f"[alive] {role}: liveness off: bad {DEADLINE_ENV}={value!r}")
        return False
    _started = True
    try:
        dump_at = deadline - dump_margin_s(role)
        when = dt.datetime.fromtimestamp(dump_at, dt.timezone.utc).isoformat(timespec="seconds")
        faulthandler.dump_traceback_later(timeout=max(1, dump_at - time.time()), repeat=False)
        # HFU spec 13.2: the job states numba is active. Probed after the dump is armed (the first numba import takes
        # seconds and must not delay the deadline dump) and before the beats start (a beat must name the caller's
        # frame, never the numba import).
        from silly_kicks._numba_dispatch import numba_version

        _say(f"[alive] {_label()} {role} pid={os.getpid()} start; deadline dump at {when}; numba={numba_version()}")
        threading.Thread(target=_beat, args=(role, interval_s, threading.Event()), name="sk-alive", daemon=True).start()
    except Exception as exc:  # broad BY DESIGN: liveness must never fail the job it reports on
        _say(f"[alive] {role}: liveness failed to start: {type(exc).__name__}: {exc}")
    return True
