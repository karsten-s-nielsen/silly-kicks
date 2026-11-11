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
- prints the DEADLINE DUMP (`stack_dump`: every thread's Python stack) `dump_margin_s(role)` before the job
  deadline: at least `DUMP_MARGIN_S`, and a different second for each process of one job, so two dumps do not
  interleave in the log (Task 8 D: three same-second dumps mixed character by character; owner decision
  2026-11-05). A daemon thread prints it, so it holds the GIL while it reads the frames.

The dump is NOT `faulthandler.dump_traceback_later` (owner decision 2026-11-10). That watchdog is a C thread that
reads the frames of a running thread without the GIL. Measured on HF (jobs `6af366809737d9a9e0914f3a` and
`6af3700bab34849891acb4a8`, Python 3.12.14; 40 processes per workload, each with a dump every 0.05 s for 3 s,
where production prints ONE dump per process): its dump ended the process with a signal (SIGSEGV; one SIGBUS) in
18 of 40 processes during an xgboost fit, 38 of 40 during pandas work, 40 of 40 during `infer_ball_carrier`, 40
of 40 during a numba compile and 36 of 40 during the numba import; 0 of 40 while the main thread slept. So a slow
job that would still finish could be ended by its own diagnostic. The cost of the Python dump: while a C
extension holds the GIL in one long call, the dump waits until that call returns (the alive lines have the same
limit).

Everything starts only when `DEADLINE_ENV` is set: `launch.job_command` exports it at bash start. Local runs,
tests and the controller print nothing new. The watch reads the unit slot, one frame and ``/proc``; it touches
no store, result, token or generation, so no artifact byte can change. It never raises.

"""

from __future__ import annotations

import datetime as dt
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
MAX_DUMP_FRAMES: Final = 100  # per thread, as `faulthandler` caps it
DUMP_MARK: Final = "deadline dump: "  # in the header line of every dump; `campaign._plan_timeout_detail` reads it

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


def dump_delay_s(deadline: int, role: str, now: float) -> float:
    """Seconds from ``now`` until ``role``'s deadline dump; 1 s for a process that starts inside its margin."""
    return max(1, deadline - dump_margin_s(role) - now)


def _say(text: str) -> None:
    """One record as ONE write (`print` writes the text and the newline apart, so a record of another thread
    could land between them)."""
    sys.stderr.write(f"{text}\n")
    sys.stderr.flush()


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


def stack_dump(role: str) -> str:
    """The deadline dump as ONE text block: a header line, then every thread's Python stack, the main thread
    first and the most recent call first (the line format of `faulthandler`, at most `MAX_DUMP_FRAMES` frames
    per thread). The caller is a Python thread, so it holds the GIL while it reads the frames."""
    names = {t.ident: t.name for t in threading.enumerate()}
    main = threading.main_thread().ident
    frames = sys._current_frames()
    lines = [f"[alive] {_label()} {role} pid={os.getpid()} {DUMP_MARK}{len(frames)} thread(s), most recent call first"]
    for ident in sorted(frames, key=lambda i: (i != main, i)):
        lines.append(f"Thread {names.get(ident, '?')} (0x{ident:x}){' [main]' if ident == main else ''}:")
        frame, depth = frames[ident], 0
        while frame is not None and depth < MAX_DUMP_FRAMES:
            code = frame.f_code
            lines.append(f'  File "{code.co_filename}", line {frame.f_lineno} in {code.co_name}')
            frame, depth = frame.f_back, depth + 1
        if frame is not None:
            lines.append("  ...")
    return "\n".join(lines)


def _dump_later(role: str, delay_s: float, stop: threading.Event) -> None:
    """Print the deadline dump once, ``delay_s`` from now (nothing when the watch stops first)."""
    if stop.wait(delay_s):
        return
    try:
        _say(stack_dump(role))
    except Exception as exc:  # broad BY DESIGN: liveness must never fail the job it reports on
        _say(f"[alive] {_label()} {role}: deadline dump failed: {type(exc).__name__}: {exc}")


def start_dump_thread(role: str, delay_s: float, stop: threading.Event | None = None) -> threading.Thread:
    """Start the daemon thread that prints `stack_dump(role)` once, ``delay_s`` from now (the job watch's deadline
    dump, and the dump of a `--plan` process before the driver ends it: `scripts._stages.arm_plan_deadline`)."""
    thread = threading.Thread(
        target=_dump_later, args=(role, delay_s, stop or threading.Event()), name="sk-dump", daemon=True
    )
    thread.start()
    return thread


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
        stop = threading.Event()  # never set today: both threads end with the process
        start_dump_thread(role, dump_delay_s(deadline, role, time.time()), stop)
        # HFU spec 13.2: the job states numba is active. Probed after the dump is armed (the first numba import takes
        # seconds and must not delay the deadline dump) and before the beats start (a beat must name the caller's
        # frame, never the numba import).
        from silly_kicks._numba_dispatch import numba_version

        _say(f"[alive] {_label()} {role} pid={os.getpid()} start; deadline dump at {when}; numba={numba_version()}")
        threading.Thread(target=_beat, args=(role, interval_s, stop), name="sk-alive", daemon=True).start()
    except Exception as exc:  # broad BY DESIGN: liveness must never fail the job it reports on
        _say(f"[alive] {role}: liveness failed to start: {type(exc).__name__}: {exc}")
    return True
