"""The dataset read gate (ADR-128): a job-wide counting semaphore over worker processes, and whole-file
reads under it. The real-HF evidence is in `scripts/_dataset_reads.py`; these tests pin the mechanism."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time

import pytest

from scripts import _dataset_reads as dr

REPO = pathlib.Path(__file__).resolve().parents[2]

_HOLDER = """
import pathlib, sys, time
from scripts import _dataset_reads as dr
out, go, hold = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
while not go.exists():
    time.sleep(0.01)
with dr.read_slot():
    t0 = time.monotonic_ns()
    (out.parent / (out.name + ".held")).write_text("1")
    time.sleep(hold)
    t1 = time.monotonic_ns()
out.write_text(f"{t0} {t1}")
"""


@pytest.fixture
def gate(tmp_path, monkeypatch):
    d = tmp_path / "gate"
    monkeypatch.setenv(dr.READ_GATE_DIR_ENV, str(d))
    return d


def _env(gate: pathlib.Path, cap: int) -> dict[str, str]:
    import os

    return {**os.environ, dr.READ_CAP_ENV: str(cap), dr.READ_GATE_DIR_ENV: str(gate)}


def _max_overlap(intervals: list[tuple[int, int]]) -> int:
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda e: (e[0], e[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def test_without_a_cap_there_is_no_gate_and_the_read_is_whole(tmp_path, gate, monkeypatch):
    monkeypatch.delenv(dr.READ_CAP_ENV, raising=False)
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert dr.read_cap() is None
    assert dr.read_bytes(p) == b"abc" * 1000
    assert not gate.exists()  # no lock directory: nothing was gated


@pytest.mark.parametrize("raw", ["0", "-1", "four", "1.5"])
def test_a_cap_that_is_not_a_positive_integer_refuses(raw, monkeypatch):
    monkeypatch.setenv(dr.READ_CAP_ENV, raw)
    with pytest.raises(ValueError, match=dr.READ_CAP_ENV):
        dr.read_cap()


def test_the_cap_bounds_concurrent_holders_across_processes_and_is_reached(tmp_path, gate):
    """Five worker PROCESSES, cap 2: never more than 2 hold a slot at once (the bound), and 2 do (the
    gate does not serialize below its cap -- non-vacuity)."""
    go = tmp_path / "go"
    outs = [tmp_path / f"w{i}.txt" for i in range(5)]
    procs = [
        subprocess.Popen(  # noqa: S603 -- own interpreter, own script
            [sys.executable, "-c", _HOLDER, str(o), str(go), "0.5"], cwd=REPO, env=_env(gate, 2)
        )
        for o in outs
    ]
    time.sleep(1.0)  # let every process import and reach the barrier
    go.write_text("1")
    for p in procs:
        assert p.wait(timeout=60) == 0
    intervals = [tuple(int(x) for x in o.read_text().split()) for o in outs]
    assert _max_overlap(intervals) == 2  # type: ignore[arg-type]


def test_threads_in_one_process_share_the_same_cap(tmp_path, gate, monkeypatch):
    monkeypatch.setenv(dr.READ_CAP_ENV, "1")
    intervals: list[tuple[int, int]] = []
    lock = threading.Lock()

    def hold():
        with dr.read_slot():
            t0 = time.monotonic_ns()
            time.sleep(0.2)
            t1 = time.monotonic_ns()
        with lock:
            intervals.append((t0, t1))

    threads = [threading.Thread(target=hold) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(intervals) == 3 and _max_overlap(intervals) == 1


def test_a_killed_holder_releases_its_slot(tmp_path, gate, monkeypatch):
    """An OOM-killed worker must never leak a slot: the kernel drops the lock with the process."""
    go, out = tmp_path / "go", tmp_path / "holder.txt"
    go.write_text("1")
    holder = subprocess.Popen(  # noqa: S603 -- own interpreter, own script
        [sys.executable, "-c", _HOLDER, str(out), str(go), "600"], cwd=REPO, env=_env(gate, 1)
    )
    try:
        held = out.parent / (out.name + ".held")
        deadline = time.monotonic() + 60
        while not held.exists():
            assert time.monotonic() < deadline, "the holder never took its slot"
            time.sleep(0.02)
        monkeypatch.setenv(dr.READ_CAP_ENV, "1")
        got = threading.Event()

        def take():
            with dr.read_slot():
                got.set()

        t = threading.Thread(target=take, daemon=True)
        t.start()
        assert not got.wait(0.5)  # the slot is held: the reader waits
        holder.kill()
        holder.wait(timeout=30)
        assert got.wait(10)  # the dead holder's slot is free again
    finally:
        if holder.poll() is None:
            holder.kill()


def test_a_nested_slot_in_one_thread_reuses_the_held_slot(tmp_path, gate, monkeypatch):
    """Re-entrant per thread: a read inside a held slot (a fallback whole-file read during a footer check)
    must not wait for a second slot -- with cap 1 it would deadlock."""
    monkeypatch.setenv(dr.READ_CAP_ENV, "1")
    p = tmp_path / "f.bin"
    p.write_bytes(b"x")
    done = threading.Event()

    def nested():
        with dr.read_slot():
            assert dr.read_bytes(p) == b"x"
        done.set()

    threading.Thread(target=nested, daemon=True).start()
    assert done.wait(10), "a nested read waited for a second slot (deadlock)"


def test_the_store_gate_is_separate_from_the_dataset_gate(tmp_path, monkeypatch):
    monkeypatch.setenv(dr.READ_CAP_ENV, "1")
    monkeypatch.setenv(dr.STORE_READ_CAP_ENV, "1")
    monkeypatch.setenv(dr.READ_GATE_DIR_ENV, str(tmp_path))
    with dr.read_slot():  # a held dataset slot never blocks a store read
        with dr.store_read_slot():
            pass
    assert dr.store_read_cap() == 1


def test_store_read_cap_refuses_a_non_positive_value(monkeypatch):
    monkeypatch.setenv(dr.STORE_READ_CAP_ENV, "0")
    with pytest.raises(ValueError, match="positive"):
        dr.store_read_cap()
