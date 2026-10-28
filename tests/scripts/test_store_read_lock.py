"""HFU S3 review (owner ruling 2026-10-31, "Per-file read lock"): `LocalStore.read_bytes` never lets two processes of
a container read ONE file at the same time.

On HF the bucket mount nearly hangs when several processes read the same cold file at once: five readers of one
156-215 MB `cache` generation took 546 s and 4,161 s, where one reader takes 1.4-1.6 s and two take 2.4 s (jobs
`6ae62805c78863f9c62f3a2c`, `6ae634a00e0338d1bb7d3a15`); the five ghost_gk `cv` folds of a THREADS job spent 455 of
their 527 s in `read_bytes` (job `6ae634ccc78863f9c62f3e3e`). A warm reread is fast (1,177 MB/s), so readers of one
file that take turns cost one cold read; with the lock the same five folds took 70-75 s and their reads 5 s or less
(job `6ae63a7cc78863f9c62f4051`). Readers of DIFFERENT files run at once (four at once: 1.1 s each), and the lock
must not serialize them.

The lock is POSIX `fcntl` (HF and CI run Linux); without `fcntl` the read takes no lock.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

pytest.importorskip("fcntl")  # POSIX only: the lock is a no-op without fcntl (see the module docstring)

REPO = pathlib.Path(__file__).resolve().parents[2]

# A child process: slow every Path.read_bytes down by HOLD_S, record when its read of the store file started and
# ended, and read through LocalStore so the store's lock (if any) wraps the read.
_CHILD = r"""
import json, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from scripts._stages import LocalStore
root, rel, out, go = pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5])
real = pathlib.Path.read_bytes
span = {}
def slow(self):
    span["start"] = time.time()
    time.sleep(0.4)
    data = real(self)
    span["end"] = time.time()
    return data
pathlib.Path.read_bytes = slow
while not go.exists():  # every child starts its read at the same moment
    time.sleep(0.005)
data = LocalStore(root).read_bytes(rel)
out.write_text(json.dumps({**span, "n": len(data)}))
"""


def _read_spans(tmp_path: pathlib.Path, rels: list[str]) -> list[dict]:
    root = tmp_path / "store"
    for rel in set(rels):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(rel.encode() * 1000)
    go = tmp_path / "go"
    outs = [tmp_path / f"span{i}.json" for i in range(len(rels))]
    env = {**os.environ, "TMPDIR": str(tmp_path / "locks")}  # the lock directory, isolated per test
    (tmp_path / "locks").mkdir()
    procs = [
        subprocess.Popen(  # noqa: S603 -- sys.executable + a fixed literal, no untrusted input
            [sys.executable, "-c", _CHILD, str(REPO), str(root), rel, str(out), str(go)], env=env
        )
        for rel, out in zip(rels, outs, strict=True)
    ]
    time.sleep(1.5)  # every child imported and waits for `go`
    go.write_text("go")
    for p in procs:
        assert p.wait(timeout=60) == 0
    return [json.loads(o.read_text()) for o in outs]


def _overlaps(spans: list[dict]) -> int:
    """The number of pairs of reads whose intervals overlap."""
    s = sorted(spans, key=lambda r: r["start"])
    return sum(1 for i, a in enumerate(s) for b in s[i + 1 :] if b["start"] < a["end"])


def test_readers_of_one_file_take_turns(tmp_path):
    spans = _read_spans(tmp_path, ["gen/_output.parquet"] * 4)
    assert all(r["n"] == len(b"gen/_output.parquet") * 1000 for r in spans)  # every reader got the whole file
    assert _overlaps(spans) == 0


def test_readers_of_different_files_still_read_at_once(tmp_path):
    """Non-vacuity (ADR-051): the lock is per file, never global -- four readers of four files overlap."""
    spans = _read_spans(tmp_path, [f"gen/part-{i}.parquet" for i in range(4)])
    assert _overlaps(spans) >= 3


def test_a_missing_file_reads_as_none_without_a_lock_file(tmp_path, monkeypatch):
    """`read_result` probes absent candidate suffixes on every read: a probe of an absent file creates no lock file."""
    import tempfile

    from scripts._stages import LocalStore

    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    assert LocalStore(tmp_path / "store").read_bytes("gen/absent.json") is None
    assert not list((tmp_path / "tmp").rglob("*.lock"))


def test_a_present_file_reads_its_bytes_under_the_lock(tmp_path, monkeypatch):
    """The lock file lives in the container's temp directory, never in the store (the bucket mount)."""
    import tempfile

    from scripts._stages import LocalStore

    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    root = tmp_path / "store"
    (root / "gen").mkdir(parents=True)
    (root / "gen" / "a.json").write_bytes(b'{"a": 1}')
    assert LocalStore(root).read_bytes("gen/a.json") == b'{"a": 1}'
    assert len(list((tmp_path / "tmp").rglob("*.lock"))) == 1
    assert not list(root.rglob("*.lock"))


def test_a_file_deleted_while_its_reader_waits_reads_as_none(tmp_path, monkeypatch):
    """Review S3B-14: a reader that passed `is_file()` and waits for the lock while the holder reopens the stage (spec
    r4 D19 deletes the unit and the reduced output) reads None -- MISSING, which takes the reopen path -- never
    `FileNotFoundError`, which `run_unit` would record as a failed attempt with the wrong cause."""
    import tempfile
    import threading

    from scripts._stages import LocalStore, _file_read_lock

    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    root = tmp_path / "store"
    (root / "gen").mkdir(parents=True)
    path = root / "gen" / "_output.parquet"
    path.write_bytes(b"x" * 100)
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with _file_read_lock(path):
            held.set()
            release.wait(10)
            path.unlink()  # the reopen deletes the file while the other reader waits on the lock

    result: dict = {}

    def reader() -> None:
        result["got"] = LocalStore(root).read_bytes("gen/_output.parquet")

    t = threading.Thread(target=holder)
    t.start()
    assert held.wait(10)
    r = threading.Thread(target=reader)
    r.start()
    time.sleep(0.5)
    assert r.is_alive()  # non-vacuity: the reader passed `is_file()` and waits on the lock
    release.set()
    t.join(10)
    r.join(10)
    assert result == {"got": None}
