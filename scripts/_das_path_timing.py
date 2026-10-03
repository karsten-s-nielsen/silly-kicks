"""add_das / das_xfns per-match timing, for whichever silly-kicks the RUNNING interpreter imports.

The parity driver's --benchmark runs this file as a subprocess in TWO pandas-2 interpreters (spec 12
D1): the OLD path (released silly-kicks 4.127.0 + accessible-space 2.0.15) and the NEW path (this
cycle's C1 installed with pandas<3), so the ratios measure the engine, not a pandas major. The driver
strips PYTHONPATH and checks the recorded `native` marker. Top-level imports are stdlib + pandas only.

A memory ceiling guards each run (combined-cycle Phase B): the OLD path's das_xfns needed 113.4 GiB on an
IDSSE match and was OOM-killed inside a 110G cap on a GS match, while the new path needed 2.6 GB. A
watchdog polls this process's resident size; above the ceiling it writes what finished plus an
``over_memory`` record and exits cleanly, so "did not fit" is a recorded result, not a crashed box.

    <python> _das_path_timing.py <in_dir> <out_dir> <repeat> [<memory_limit_gib>]
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import mmap
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

_T = TypeVar("_T")

#: The default ceiling (GiB) when the driver passes none: below the DGX's 119 GiB, with room for the OS.
_DEFAULT_LIMIT_GIB = 100.0


def _rss_bytes() -> int | None:
    """This process's resident set size (Linux ``/proc/self/statm``); ``None`` where it cannot be measured."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * mmap.PAGESIZE  # cross-platform stdlib constant (os.sysconf is POSIX-only)
    except (OSError, ValueError, IndexError):
        return None


def _hard_exit(code: int) -> None:
    """End the process NOW (the ceiling's stop; a seam tests replace)."""
    os._exit(code)


class _Ceiling:
    """A hard memory ceiling for one timing run.

    ``check()`` polls the resident size once; above ``limit_gib`` it writes ``state`` (everything that
    finished, plus the provenance fields) with ``over_memory=True``, the phase in progress, the limit and the
    peak, then calls ``exit_fn(0)``. A hard process exit is the only reliable stop: a numpy call in C cannot
    be interrupted from a thread, and the next allocation could take the whole box down.
    """

    def __init__(
        self,
        limit_gib: float,
        out_path: Path,
        state: dict,
        *,
        interval: float = 0.2,
        rss: Callable[[], int | None] | None = None,
        exit_fn: Callable[[int], Any] | None = None,
    ) -> None:
        self.limit_gib = float(limit_gib)
        self.out_path = Path(out_path)
        self.state = state
        self.interval = interval
        # resolved at construction (not as defaults) so the module-level seams stay replaceable
        self.rss = rss if rss is not None else _rss_bytes
        self.exit_fn = exit_fn if exit_fn is not None else _hard_exit
        self.peak = 0
        self.measured = False
        self._stop = threading.Event()

    def sample(self) -> int | None:
        """One reading into ``peak``; ``None`` (and ``measured`` stays False) where it cannot be measured."""
        now = self.rss()
        if now is not None:
            self.measured = True
            self.peak = max(self.peak, now)
        return now

    def check(self) -> bool:
        now = self.sample()
        if now is None or now <= self.limit_gib * 2**30:
            return False
        record = {
            **self.state,
            "over_memory": True,
            "limit_gib": self.limit_gib,
            "peak_gib": round(self.peak / 2**30, 1),
        }
        self.out_path.write_text(json.dumps(record), encoding="utf-8")
        self.exit_fn(0)
        return True

    def start(self) -> None:
        threading.Thread(target=self._watch, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            if self.check():
                return


def best_of(fn: Callable[[], _T], repeat: int) -> tuple[_T, float]:
    """``(result, best_seconds)`` over ``max(1, repeat)`` calls of ``fn`` -- the minimum wall time;
    the result is the last call's. (A stdlib-only copy: this file runs under the OLD-path interpreter,
    which cannot import the repo's scripts.)"""
    t0 = time.perf_counter()
    result = fn()
    best = time.perf_counter() - t0
    for _ in range(max(1, repeat) - 1):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def time_add_das_and_xfns(
    frames: pd.DataFrame, actions: pd.DataFrame, *, repeat: int, warmup: bool, state: dict | None = None
) -> dict:
    """Best-of-``repeat`` seconds for ``add_das`` (links precomputed, untimed) and ``das_xfns`` on one match.

    ``state`` (shared with the memory ceiling) carries the phase in progress and every time already
    measured, so a ceiling trip in ``das_xfns`` still records the ``add_das`` time.
    """
    import silly_kicks.tracking.features as feats
    import silly_kicks.tracking.utils as tu
    import silly_kicks.vaep.feature_framework as ff

    state = {} if state is None else state
    links, _report = tu.link_actions_to_frames(actions, frames)
    states = ff.gamestates(actions, nb_prev_actions=3)

    def _add():
        return feats.add_das(actions, frames, links=links)

    def _xfn():
        return feats.das_xfns[0](states, frames)

    # Each call warms up (JIT and first-call caches stay out of the timed region on both paths) and is
    # timed before the next starts, so what finished is recorded before anything heavier runs.
    for name, fn in (("add_das", _add), ("das_xfns", _xfn)):
        state["phase"] = name
        if warmup:
            fn()
        _, state[f"{name}_s"] = best_of(fn, repeat)
    state.pop("phase", None)
    return {"add_das_s": state["add_das_s"], "das_xfns_s": state["das_xfns_s"]}


def _main(in_dir: str, out_dir: str, repeat: int, limit_gib: float = _DEFAULT_LIMIT_GIB) -> None:
    out = Path(out_dir) / "timing.json"
    state: dict = {
        "repeat": repeat,
        "silly_kicks": importlib.metadata.version("silly-kicks"),
        # the native engine exists only after ADR-107: the discriminator between the two paths
        "native": importlib.util.find_spec("silly_kicks.tracking._das_engine") is not None,
        "pandas": pd.__version__,
        "phase": "load",
    }
    ceiling = _Ceiling(limit_gib, out, state)
    ceiling.start()
    frames = pd.read_parquet(Path(in_dir) / "frames.parquet")
    actions = pd.read_parquet(Path(in_dir) / "actions.parquet")
    timing = time_add_das_and_xfns(frames, actions, repeat=repeat, warmup=True, state=state)
    ceiling.stop()
    ceiling.sample()  # one final reading (never a trip): a run shorter than one poll interval still has a size
    record = {k: v for k, v in state.items() if k != "phase"}
    record.update(timing)
    record.update(
        {
            "over_memory": False,
            "limit_gib": float(limit_gib),
            "memory_measured": ceiling.measured,
            "peak_gib": round(ceiling.peak / 2**30, 1) if ceiling.measured else None,
        }
    )
    out.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    _main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        float(sys.argv[4]) if len(sys.argv) > 4 else _DEFAULT_LIMIT_GIB,
    )
