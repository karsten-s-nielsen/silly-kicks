"""A FileSet is read from the store once per job (HFU spec §13.3, F2): five cv workers each read the same 388 MB prep
FileSet through the HF bucket mount, and five concurrent readers took over 6,600 s where one took under 300 s. With
the executor's cache variable set, the first process reads and the others copy the local cache. Store reads of
8 MB or more log their size and rate."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import types
from typing import Any

import pytest

from scripts import _stages as S

REPO = pathlib.Path(__file__).resolve().parents[2]
FILES = {"X.npy": b"x" * 1000, "meta/rows.json": b'{"n": 3}'}


def _ctx(root: pathlib.Path, token: str = "tok1") -> Any:  # a RunContext stand-in: generation + store only
    store = S.LocalStore(root)
    S.write_result(store, f"p/prep/{token}/only", S.FileSet(dict(FILES)))
    return types.SimpleNamespace(generation=lambda stage: f"p/{stage}/{token}", store=store)


def _read_tree(root: pathlib.Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


_CHILD = textwrap.dedent(
    """
    import dataclasses, os, pathlib, sys, types
    sys.path.insert(0, {repo!r})
    from scripts import _stages as S

    counter = pathlib.Path({counter!r})

    @dataclasses.dataclass(frozen=True)
    class Counting(S.LocalStore):
        def read_bytes(self, rel):
            if "/prep/" in rel and not rel.endswith((".json", ".sha256")) or rel.endswith(".fileset.json"):
                with open(counter, "a") as f:
                    f.write(rel + "\\n")
            return super().read_bytes(rel)

    ctx = types.SimpleNamespace(generation=lambda st: "p/prep/tok1", store=Counting(pathlib.Path({root!r})))
    S.materialize_fileset(ctx, "prep", "only", pathlib.Path(sys.argv[1]))
    """
)


def test_fileset_read_once_across_processes(tmp_path, monkeypatch):
    _ctx(tmp_path / "store")
    counter = tmp_path / "reads.txt"
    cache = tmp_path / "cache"
    cache.mkdir()
    code = _CHILD.format(repo=str(REPO), counter=str(counter), root=str(tmp_path / "store"))
    env = {**os.environ, S.FILESET_CACHE_ENV: str(cache)}
    procs = [
        subprocess.Popen([sys.executable, "-c", code, str(tmp_path / f"dest{i}")], env=env, cwd=REPO)  # noqa: S603
        for i in range(3)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    reads = counter.read_text().splitlines()
    assert len([r for r in reads if r.endswith(".fileset.json")]) == 1, reads  # ONE store read of the FileSet
    trees = [_read_tree(tmp_path / f"dest{i}") for i in range(3)]
    assert trees[0] == trees[1] == trees[2] == FILES


def test_killed_holder_leaves_no_cache(tmp_path, monkeypatch):
    """A holder killed mid-write leaves only its `.part-<pid>` dir: the next process reads again, and the cache
    directory it publishes is complete."""
    ctx = _ctx(tmp_path / "store")
    cache = tmp_path / "cache"
    entry = cache / "p" / "prep" / "tok1" / "only"
    stale = entry.parent / "only.part-99999"
    stale.mkdir(parents=True)
    (stale / "X.npy").write_bytes(b"partial")
    monkeypatch.setenv(S.FILESET_CACHE_ENV, str(cache))
    S.materialize_fileset(ctx, "prep", "only", tmp_path / "dest")
    assert _read_tree(entry) == FILES
    assert _read_tree(tmp_path / "dest") == FILES
    assert not stale.exists()


def test_new_generation_gets_new_cache_entry(tmp_path, monkeypatch):
    """Review Focus 2: a new token is a new cache entry, never the old bytes."""
    cache = tmp_path / "cache"
    monkeypatch.setenv(S.FILESET_CACHE_ENV, str(cache))
    S.materialize_fileset(_ctx(tmp_path / "store", "tok1"), "prep", "only", tmp_path / "a")
    store2 = tmp_path / "store2"
    S.write_result(S.LocalStore(store2), "p/prep/tok2/only", S.FileSet({"X.npy": b"new"}))
    ctx2: Any = types.SimpleNamespace(generation=lambda stage: "p/prep/tok2", store=S.LocalStore(store2))
    S.materialize_fileset(ctx2, "prep", "only", tmp_path / "b")
    assert _read_tree(tmp_path / "b") == {"X.npy": b"new"}


def test_no_env_means_direct_read(tmp_path, monkeypatch):
    monkeypatch.delenv(S.FILESET_CACHE_ENV, raising=False)
    ctx = _ctx(tmp_path / "store")
    S.materialize_fileset(ctx, "prep", "only", tmp_path / "dest")
    assert _read_tree(tmp_path / "dest") == FILES


def test_large_read_prints_rate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "READ_LOG_BYTES", 1000)
    store = S.LocalStore(tmp_path)
    (tmp_path / "big.bin").write_bytes(b"z" * 5000)
    assert store.read_bytes("big.bin") == b"z" * 5000
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("[read] big.bin ")]
    assert len(lines) == 1 and " MB in " in lines[0] and "MB/s)" in lines[0], err


def test_slow_read_prints_progress(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "READ_LOG_BYTES", 10)
    monkeypatch.setattr(S, "READ_CHUNK_BYTES", 10)
    monkeypatch.setattr(S, "READ_PROGRESS_S", 0.0)  # every chunk is "over the interval"
    (tmp_path / "big.bin").write_bytes(b"z" * 35)
    assert S.LocalStore(tmp_path).read_bytes("big.bin") == b"z" * 35
    err = capsys.readouterr().err
    assert any(" MB after " in ln for ln in err.splitlines() if ln.startswith("[read] big.bin ")), err


def test_small_and_missing_reads_print_nothing(tmp_path, capsys):
    """Review Focus 5."""
    store = S.LocalStore(tmp_path)
    (tmp_path / "small.bin").write_bytes(b"s" * 10)
    (tmp_path / "adir").mkdir()
    assert store.read_bytes("small.bin") == b"s" * 10
    assert store.read_bytes("missing.bin") is None
    assert store.read_bytes("adir") is None
    assert "[read]" not in capsys.readouterr().err


@pytest.mark.parametrize("n", [0, 1])
def test_executor_owns_the_cache_dir(tmp_path, monkeypatch, n):
    """`LocalExecutor.run_stage` sets the cache variable for its workers and removes the directory after."""
    from scripts.hf_jobs import executor as E

    seen = {}

    def fake_run(self, ctx, st, keys, *, workers, inner):
        seen["env"] = os.environ.get(S.FILESET_CACHE_ENV)
        if n:
            raise RuntimeError("stage failed")
        return "report"

    monkeypatch.delenv(S.FILESET_CACHE_ENV, raising=False)
    monkeypatch.setattr(E.LocalExecutor, "_run", fake_run)
    st = types.SimpleNamespace(needs=(), name="s", inner_parallel=False)
    ctx = types.SimpleNamespace(pipeline=types.SimpleNamespace(stage=lambda name: st, name="p"))
    monkeypatch.setattr(S, "sorted_units", lambda ctx, st: ["0"])
    monkeypatch.setattr(S, "deferred_memory_units", lambda ctx, st, keys: set())
    ex = E.LocalExecutor(E.ExecutorConfig(workers=1))
    if n:
        with pytest.raises(RuntimeError):
            ex.run_stage(ctx, "s")
    else:
        assert ex.run_stage(ctx, "s") == "report"
    assert seen["env"] and not pathlib.Path(seen["env"]).exists()
    assert os.environ.get(S.FILESET_CACHE_ENV) is None
