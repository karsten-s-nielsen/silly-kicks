"""Repair a rename the HF bucket mount lost (HFU spec §12, Change 5; owner decision 2026-11-05, 2c).

Task 8 B and probes r2/r3 (12 concurrent cpu-upgrade jobs) measured it: the mount loses ~0.3-0.4% of `os.replace`
renames, in bursts. The writer's own ``stat`` shows the final name at once, and an ``fsync`` before the rename does
not help. The container's listing shows the loss within 5 s, and the ``.tmp-<name>.<uuid>.part`` still holds the
full bytes, so one more ``os.replace`` repairs it. A listing can also miss a final for one pass only, with no temp
part (a flicker): that is re-checked, never repaired and never an error.

`_driver._write_bytes_atomically` records every write in this process's journal (final path -> temp path); every
delete of a written file forgets it. `confirm_job_writes` runs once at the end of each process that writes to the
store (executor worker, reduce subprocess, stage main). It moves the same bytes the writer wrote, so no artifact
byte can change.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from collections.abc import Callable, Mapping
from typing import Final

#: Settle time in an HF run job: 6x the observed <= 5 s (probe r3). 0 elsewhere, so local runs never wait.
JOB_SETTLE_S: Final = 30
ROUNDS: Final = 3
_DEADLINE_ENV: Final = "SILLY_KICKS_JOB_DEADLINE_EPOCH"  # set by `launch.job_command`: this is an HF run job

_journal: dict[pathlib.Path, pathlib.Path] = {}


class LostWriteError(RuntimeError):
    """A written file is still missing after every repair pass."""


def record(final: pathlib.Path, tmp: pathlib.Path) -> None:
    """Journal one atomic write; a later write to the same final replaces the entry."""
    _journal[pathlib.Path(final)] = pathlib.Path(tmp)


def forget(final: pathlib.Path) -> None:
    """A deliberate delete: the file is no longer expected."""
    _journal.pop(pathlib.Path(final), None)


def journal() -> dict[pathlib.Path, pathlib.Path]:
    return dict(_journal)


def clear() -> None:
    _journal.clear()


def _list_names(d: pathlib.Path) -> set[str]:
    try:
        return set(os.listdir(d))
    except (FileNotFoundError, NotADirectoryError):
        return set()


def confirm_writes(
    *,
    settle_s: float,
    rounds: int = ROUNDS,
    sleep: Callable[[float], None] = time.sleep,
    list_names: Callable[[pathlib.Path], set[str]] = _list_names,
) -> int:
    """Up to ``rounds + 1`` passes: settle, list every journaled directory once, and run ``os.replace`` again for
    each missing final whose temp part is listed. Returns the number of repairs and clears the journal; raises
    `LostWriteError` naming every final still missing after the last pass."""
    repaired, repaired_dirs = 0, set()
    for attempt in range(rounds + 1):
        sleep(settle_s)
        listed = {d: list_names(d) for d in sorted({f.parent for f in _journal})}
        missing = {f: t for f, t in _journal.items() if f.name not in listed[f.parent]}
        if not missing:
            break
        if attempt == rounds:
            for final in sorted(missing):  # one greppable line each: a lost ledger.json alone relaunches nothing
                print(f"[confirm] LOST {final}", file=sys.stderr, flush=True)
            raise LostWriteError(
                f"{len(missing)} written file(s) still missing after {rounds} repair pass(es): "
                f"{sorted(str(f) for f in missing)}"
            )
        for final, tmp in missing.items():
            if tmp.name not in listed[final.parent]:  # a flicker, or gone: re-check it next pass
                continue
            try:
                os.replace(tmp, final)
            except OSError as exc:  # the listing was stale: the next pass decides
                print(f"[confirm] repair of {final} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue
            repaired += 1
            repaired_dirs.add(str(final.parent))
    if repaired:
        print(f"[confirm] repaired {repaired} lost rename(s) in {sorted(repaired_dirs)}", file=sys.stderr, flush=True)
    clear()
    return repaired


def settle_s_for(env: Mapping[str, str]) -> float:
    return JOB_SETTLE_S if env.get(_DEADLINE_ENV) is not None else 0


def confirm_job_writes(*, env: Mapping[str, str] = os.environ) -> int:
    """`confirm_writes` with the job's settle time; returns at once when this process wrote nothing."""
    if not _journal:
        return 0
    return confirm_writes(settle_s=settle_s_for(env))
