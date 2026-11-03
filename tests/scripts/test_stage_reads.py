"""Listed reads: a reduce site lists its generation once and reads each file once (HFU spec §3, ADR-128).

The HF bucket mount serves ~24 small files/s, and the old reduce re-listed the whole generation directory per
unit (``read_result`` -> ``store.names``): ~47 M stats for 3,961 units (spec E3-E5). These tests pin the
listing and read budget of the run-job reduce path and the INV-3 behaviour the rework must keep.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, NamedTuple

import pandas as pd
import pytest

import scripts._stages as S
from scripts._item_outcome import ItemExcluded
from scripts._stages import (
    FileSet,
    ItemStage,
    LocalStore,
    Param,
    Pipeline,
    Stage,
    StageIncompleteError,
    combine_frames,
    int_key,
    make_context,
    read_result,
    reduce_stage,
    run_unit,
    sorted_units,
    unit_states,
)
from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor
from tests.scripts._stage_harness import PROV


class _CountingStore:
    """A `LocalStore` that counts directory listings and file reads (the mount's two cost drivers)."""

    def __init__(self, root: pathlib.Path) -> None:
        self.inner = LocalStore(root)
        self.names_calls = 0
        self.reads = 0

    def reset(self) -> None:
        self.names_calls = 0
        self.reads = 0

    def read_bytes(self, rel: str) -> bytes | None:
        self.reads += 1
        return self.inner.read_bytes(rel)

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.inner.write_bytes(rel, data)

    def names(self, rel_dir: str) -> set[str]:
        self.names_calls += 1
        return self.inner.names(rel_dir)

    def delete(self, rel: str) -> None:
        self.inner.delete(rel)

    def local_path(self, rel: str) -> pathlib.Path | None:
        return self.inner.local_path(rel)


def _frame_work(key: str, ctx) -> pd.DataFrame:
    return pd.DataFrame({"k": [key], "v": [int(key) * 1.5]})


def _fileset_work(key: str, ctx) -> FileSet:
    return FileSet({"m.bin": f"member {key}".encode()})


def _excluded_work(key: str, ctx) -> Any:
    raise ItemExcluded("no_data", details={"key": key})


def _frames(results: dict, ctx) -> pd.DataFrame:
    return combine_frames(list(results.values()), ids=())


def _pipeline(work=_frame_work, reduce=_frames) -> Pipeline:
    return Pipeline(
        name="reads",
        module=__name__,
        params=(Param("n", int, 8),),
        stages=(
            Stage(
                name="s",
                units=lambda ctx: [int_key(i) for i in range(int(ctx.params["n"]))],
                work=work,
                reduce=reduce,
                params=("n",),
            ),
        ),
        output="s",
    )


def _built(store, n: int, pipeline: Pipeline | None = None):
    """A generation of ``n`` units of stage ``s``, every unit run, nothing reduced."""
    pipeline = pipeline or _pipeline()
    ctx = make_context(pipeline, {"n": n}, PROV, store=store, data_root=None)
    st = pipeline.stage("s")
    for key in sorted_units(ctx, st):
        run_unit(ctx, st, key, worker="test")
    return ctx


def _gen_dir(ctx) -> pathlib.Path:
    path = ctx.store.local_path(ctx.generation("s"))
    assert path is not None
    return path


def _reduce_in_process(monkeypatch) -> None:
    """The single-process seam of the run-job reduce path (spec §3.3): the reduce subprocess runs in-process."""
    monkeypatch.setattr(
        LocalExecutor, "_subprocess", lambda self, ctx, args: reduce_stage(ctx, args[args.index("--stage") + 1])
    )


@pytest.mark.parametrize("n", [8, 32])
def test_run_job_reduce_path_budget_is_constant(tmp_path, monkeypatch, n):
    store = _CountingStore(tmp_path / "store")
    ctx = _built(store, n)
    gen = ctx.generation("s")
    reference = S._encode(
        _frames({k: read_result(store.inner, f"{gen}/{k}") for k in sorted_units(ctx, ctx.pipeline.stage("s"))}, ctx)
    )[1]
    _reduce_in_process(monkeypatch)
    store.reset()

    assert LocalExecutor(ExecutorConfig()).try_reduce(ctx, "s")

    # output_present, unit_states, deferred_memory_units -> unit_states, reduce_stage, output_present
    assert store.names_calls == 5
    # 2 per unit (payload + .sha256), then the final output_present reads `_output` and its `.sha256`
    assert store.reads == 2 * n + 2
    assert (_gen_dir(ctx) / "_output.parquet").read_bytes() == reference


def test_budget_guard_detects_the_rescan(tmp_path, monkeypatch):
    real = S.read_listed

    def rescanning(store, rel_dir, key, names):  # today's per-key rescan: ignores the caller's listing
        return real(store, rel_dir, key, store.names(rel_dir))

    monkeypatch.setattr(S, "read_listed", rescanning)
    _reduce_in_process(monkeypatch)
    calls = {}
    for n in (8, 32):
        store = _CountingStore(tmp_path / f"store{n}")
        ctx = _built(store, n)
        store.reset()
        LocalExecutor(ExecutorConfig()).try_reduce(ctx, "s")
        calls[n] = store.names_calls
    assert calls[32] > 5
    assert calls[32] > calls[8]


def test_reduce_ignores_crash_leftovers(tmp_path):
    ctx = _built(LocalStore(tmp_path / "store"), 4)
    gen = _gen_dir(ctx)
    (gen / ".tmp-0001.parquet.0123abcd.part").write_bytes(b"half a file")
    (gen / ".tmp-_output.parquet.4567cdef.part").write_bytes(b"half an output")
    (gen / "_summary.json").write_text('{"orphan": true}', encoding="utf-8")  # a summary without `_output`

    reduce_stage(ctx, "s")

    out = ctx.upstream("s")
    assert out["k"].tolist() == ["0000", "0001", "0002", "0003"]
    assert out["v"].tolist() == [0.0, 1.5, 3.0, 4.5]


def test_reduce_refuses_a_corrupt_fileset_member(tmp_path):
    pipeline = _pipeline(work=_fileset_work, reduce=lambda r, ctx: r["0001"])
    ctx = _built(LocalStore(tmp_path / "store"), 3, pipeline)
    gen = _gen_dir(ctx)
    (gen / "0001.files" / "m.bin").write_bytes(b"tampered")

    with pytest.warns(UserWarning, match="checksum failure"), pytest.raises(StageIncompleteError, match="pending"):
        reduce_stage(ctx, "s")

    assert not (gen / "0001.fileset.json").exists()
    assert not (gen / "0001.sha256").exists()
    assert not (gen / "_output.fileset.json").exists()


def test_read_bytes_matches_is_file_semantics(tmp_path, monkeypatch):
    store = LocalStore(tmp_path)
    (tmp_path / "a.files").mkdir()
    (tmp_path / "f.bin").write_bytes(b"x")

    assert store.read_bytes("a.files") is None  # a directory is not a file
    assert store.read_bytes("missing.bin") is None
    assert store.read_bytes("f.bin") == b"x"

    def eio(*args, **kwargs):
        raise OSError(5, "Input/output error")

    import scripts._stages as stages_mod

    # read_bytes opens the file itself (size via fstat for the [read] rate line, HFU spec 13.3): inject the EIO there.
    monkeypatch.setattr(stages_mod, "open", eio, raising=False)
    with pytest.raises(OSError, match="Input/output error"):  # a real I/O error on a real file still raises
        store.read_bytes("f.bin")


def test_reduce_with_no_done_units_writes_summary(tmp_path):
    ctx = _built(LocalStore(tmp_path / "store"), 3, _pipeline(work=_excluded_work))

    reduce_stage(ctx, "s")

    summary = ctx.summary("s")
    assert (summary["n_units"], summary["n_done"], summary["n_excluded"]) == (3, 0, 3)
    assert ctx.upstream("s").empty


def test_reduce_verifies_before_writing_summary(tmp_path):
    ctx = _built(LocalStore(tmp_path / "store"), 4)
    gen = _gen_dir(ctx)
    (gen / "0002.parquet").write_bytes(b"not the bytes the checksum names")

    with pytest.warns(UserWarning, match="checksum failure"), pytest.raises(StageIncompleteError, match="pending"):
        reduce_stage(ctx, "s")

    assert not (gen / "0002.parquet").exists()
    assert not (gen / "0002.sha256").exists()
    assert not (gen / "_summary.json").exists()
    assert not (gen / "_inputs.json").exists()


def test_names_uses_scandir_and_skips_dirs(tmp_path, monkeypatch):
    (tmp_path / "0001.files").mkdir()
    (tmp_path / "0001.fileset.json").write_text("{}", encoding="utf-8")
    calls = []
    real = os.scandir

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(os, "scandir", counting)

    assert LocalStore(tmp_path).names(".") == {"0001.fileset.json"}
    assert len(calls) == 1
    assert LocalStore(tmp_path).names("absent") == set()


def test_unit_states_with_names_does_not_list(tmp_path):
    store = _CountingStore(tmp_path / "store")
    ctx = _built(store, 4)
    st = ctx.pipeline.stage("s")
    names = store.inner.names(ctx.generation("s"))
    store.reset()

    states = unit_states(ctx, st, sorted_units(ctx, st), verify=False, names=names)

    assert set(states.values()) == {"done"}
    assert store.names_calls == 0


def test_done_results_lists_once(tmp_path):
    store = _CountingStore(tmp_path / "store")
    ctx = _built(store, 6)
    store.reset()

    results = S.done_results(ctx, "s")

    assert list(results) == [int_key(i) for i in range(6)]
    assert [r["v"].iloc[0] for r in results.values()] == [i * 1.5 for i in range(6)]
    assert store.names_calls == 1
    assert store.reads == 2 * 6


class _WriteFencedStore(_CountingStore):
    """Refuses any read once the first file has been written: a reduce must read and verify everything first."""

    def __init__(self, root: pathlib.Path) -> None:
        super().__init__(root)
        self.fenced = False

    def read_bytes(self, rel: str) -> bytes | None:
        assert not self.fenced, f"read of {rel} after the reduce started writing"
        return super().read_bytes(rel)

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.fenced = True
        super().write_bytes(rel, data)


def test_reduce_reads_nothing_after_its_first_write(tmp_path):
    store = _WriteFencedStore(tmp_path / "store")
    ctx = _built(store.inner, 4)  # build through the plain store; the fence guards the reduce only
    ctx.store = store

    reduce_stage(ctx, "s")

    assert store.fenced
    assert (_gen_dir(ctx) / "_output.parquet").exists()


# ---- Task 2: the trainer reduce sites (spec §3.2) ------------------------------------------------------


class _Ref(NamedTuple):
    prov: str
    mid: str


def _extract_work(item, ctx) -> pd.DataFrame:
    _prov, mid = item
    return pd.DataFrame({"_group": [1], "_game_id": [mid], "v": [float(mid[1:])]})


def _site_pipeline() -> Pipeline:
    return Pipeline(
        name="sites",
        module=__name__,
        params=(Param("n", int, 6), Param("probe_comparison_providers", str, "p")),
        stages=(
            ItemStage(
                name="extract",
                refs=lambda ctx: [_Ref("p", f"m{i:02d}") for i in range(int(ctx.params["n"]))],
                key=lambda r: (r.prov, r.mid),
                work=_extract_work,
                params=("n",),
            ),
        ),
    )


def _site_ctx(tmp_path: pathlib.Path, n: int) -> tuple[Any, _CountingStore]:
    store = _CountingStore(tmp_path / "store")
    pipeline = _site_pipeline()
    ctx = make_context(pipeline, {"n": n, "probe_comparison_providers": "p"}, PROV, store=store, data_root=None)
    st = pipeline.stage("extract")
    assert isinstance(st, ItemStage)
    refs = st.ref_map(ctx)
    for key in sorted_units(ctx, st):
        run_unit(ctx, st, key, worker="test", ref=refs[key])
    store.reset()
    return ctx, store


def _census_ids(ctx):
    from scripts.build_territory_ranking_census import _done_results

    return [p["_game_id"].iloc[0] for p in _done_results(ctx, "extract")]


def _ghost_gk_ids(ctx):
    from scripts.train_ghost_gk import _cache_work

    return _cache_work("only", ctx)["_game_id"].tolist()


@pytest.mark.parametrize("site", [_census_ids, _ghost_gk_ids])
@pytest.mark.parametrize("n", [4, 12])
def test_reduce_site_lists_once(tmp_path, site, n):
    ctx, store = _site_ctx(tmp_path, n)
    assert site(ctx) == [f"m{i:02d}" for i in range(n)]
    assert store.names_calls == 1
    assert store.reads == 2 * n


def test_xcross_probe_loop_lists_once(tmp_path, monkeypatch):
    import scripts.train_xcross_attempt as X

    seen: list = []
    monkeypatch.setattr(X, "_cohort_files", lambda ctx, records, refs, providers: seen.extend(records) or FileSet({}))
    ctx, store = _site_ctx(tmp_path, 12)

    X._probe_comparison_work("only", ctx)

    assert len(seen) == X.PROBE_KEEP  # the lazy early stop: only the first PROBE_KEEP matches are read
    assert store.names_calls == 1
    assert store.reads == 2 * X.PROBE_KEEP


# --- done_results reads in parallel and reports a long read (spec §15.5) -------------------------------------


def test_done_results_reads_in_parallel_in_sorted_order(tmp_path, monkeypatch):
    import threading
    import time

    ctx, store = _site_ctx(tmp_path, 120)
    st, gen = ctx.pipeline.stage("extract"), ctx.generation("extract")
    names = store.names(gen)
    serial = {k: S.read_listed(store, gen, k, names) for k in S.sorted_units(ctx, st)}
    store.reset()
    live = {"now": 0, "max": 0}
    lock, inner = threading.Lock(), store.read_bytes

    def slow_read(rel):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.002)
        try:
            return inner(rel)
        finally:
            with lock:
                live["now"] -= 1

    monkeypatch.setattr(store, "read_bytes", slow_read)
    got = S.done_results(ctx, "extract")
    assert list(got) == sorted(got) and list(got) == list(serial)
    assert all(got[k].equals(serial[k]) for k in got)
    assert store.names_calls == 1 and live["max"] > 1


def test_done_results_reports_a_long_read(tmp_path, monkeypatch, capsys):
    ctx, _store = _site_ctx(tmp_path, 120)
    monkeypatch.setattr(S, "_READ_PROGRESS_S", 0.0)
    S.done_results(ctx, "extract")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[read] extract results ")]
    assert lines and lines[-1].startswith("[read] extract results 120/120 in ")


def test_done_results_is_quiet_on_a_short_read(tmp_path, capsys):
    ctx, _store = _site_ctx(tmp_path, 12)
    S.done_results(ctx, "extract")
    assert not [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[read] extract results ")]


def test_reduce_reads_in_parallel_in_reduce_order_and_reports(tmp_path, monkeypatch, capsys):
    """HFU spec 16.3 (F27): the reduce reads its results over the pool, in the stage's ``reduce_order``, with the
    ``[read]`` progress line; its output is the serial reduce's."""
    import threading
    import time

    desc = Pipeline(
        name="reads",
        module=__name__,
        params=(Param("n", int, 8),),
        stages=(
            Stage(
                name="s",
                units=lambda ctx: [int_key(i) for i in range(int(ctx.params["n"]))],
                work=_frame_work,
                reduce=_frames,
                params=("n",),
                reduce_order=lambda k: -int(k),
            ),
        ),
        output="s",
    )
    store = _CountingStore(tmp_path)
    ctx = _built(store, 250, desc)
    gen = ctx.generation("s")
    names = store.names(gen)
    order = sorted(S.sorted_units(ctx, desc.stage("s")), key=lambda k: -int(k))
    expected = _frames({k: S.read_listed(store, gen, k, names) for k in order}, ctx)
    live = {"now": 0, "max": 0}
    lock, inner = threading.Lock(), store.read_bytes

    def slow_read(rel):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.002)
        try:
            return inner(rel)
        finally:
            with lock:
                live["now"] -= 1

    monkeypatch.setattr(store, "read_bytes", slow_read)
    monkeypatch.setattr(S, "_READ_PROGRESS_S", 0.0)
    reduce_stage(ctx, "s")
    pd.testing.assert_frame_equal(S.read_result(store, f"{gen}/_output"), expected)
    assert live["max"] > 1
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[read] s results ")]
    assert lines and lines[-1].startswith("[read] s results 250/250 in ")
