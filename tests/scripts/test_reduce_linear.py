"""HFU r3 D1: a stage reduce is LINEAR in its unit count.

It lists the generation a constant number of times and reads each unit result once. The quadratic form --
`read_result` calling `store.names` for every key, and every shard read up to three times -- ran the 3,961-unit
`win_probability` `extract` reduce for hours through the HF bucket mount (spec r3 D1), where one listing of the
generation costs a full directory scan with a `stat` per entry. On local disk the same code costs milliseconds,
so only a COUNT of listings, entries and reads catches it.
"""

from __future__ import annotations

import collections
import dataclasses
import json

import pandas as pd
import pytest

from scripts._stages import (
    ItemStage,
    LocalStore,
    Pipeline,
    Stage,
    StageIncompleteError,
    combine_frames,
    make_context,
    read_result,
    reduce_from_partials,
    reduce_stage,
    reduce_subrange,
    run_unit,
    sorted_units,
)
from tests.scripts._stage_harness import params_from_args


class _CountingStore:
    """A `LocalStore` that counts directory listings, the entries those listings return, and reads per path."""

    def __init__(self, root) -> None:
        self._s = LocalStore(root)
        self.names_calls = 0
        self.entries = 0
        self.reads: collections.Counter[str] = collections.Counter()

    def write_bytes(self, rel: str, data: bytes) -> None:
        self._s.write_bytes(rel, data)

    def write_if_absent(self, rel: str, data: bytes) -> bool:
        return self._s.write_if_absent(rel, data)

    def delete(self, rel: str) -> None:
        self._s.delete(rel)

    def local_path(self, rel: str):
        return self._s.local_path(rel)

    def names(self, rel_dir: str) -> set[str]:
        out = self._s.names(rel_dir)
        self.names_calls += 1
        self.entries += len(out)
        return out

    def read_bytes(self, rel: str) -> bytes | None:
        self.reads[rel] += 1
        return self._s.read_bytes(rel)

    def reset(self) -> None:
        self.names_calls = 0
        self.entries = 0
        self.reads.clear()


def _concat(results, ctx):
    return combine_frames([f for f in results.values() if len(f)], ids=("game_id",))


def _plain(n: int) -> Pipeline:
    return Pipeline(
        name="lin",
        module="tests.scripts.test_reduce_linear",
        params=(),
        stages=(
            Stage(
                name="gen",
                units=lambda ctx: [f"u{i:05d}" for i in range(n)],
                work=lambda key, ctx: pd.DataFrame({"game_id": [key], "v": [int(key[1:])]}),
                reduce=_concat,
                shardable_reduce=("game_id",),
            ),
        ),
        output="gen",
    )


def _item(n: int) -> Pipeline:
    """The production shape: an `ItemStage` (one unit per match) whose reduce is a plain concat."""
    return Pipeline(
        name="linitem",
        module="tests.scripts.test_reduce_linear",
        params=(),
        stages=(
            ItemStage(
                name="ex",
                refs=lambda ctx: [f"m{i:05d}" for i in range(n)],
                key=lambda ref: ref,
                work=lambda item, ctx: pd.DataFrame({"game_id": [item], "v": [1]}),
                reduce=_concat,
                shardable_reduce=("game_id",),
            ),
        ),
        output="ex",
    )


def _ctx(pipeline: Pipeline, store):
    prov = {"commit": "t", "tree_state": "clean", "code_identity": "t"}
    return make_context(pipeline, params_from_args(pipeline, []), prov, store=store, data_root=None)


def _run_all(ctx) -> str:
    name = ctx.pipeline.output
    st = ctx.pipeline.stage(name)
    refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
    for k in sorted_units(ctx, st):
        run_unit(ctx, st, k, worker="w0", ref=refs.get(k))
    return name


def _unit_result_reads(store: _CountingStore, gen: str) -> dict[str, int]:
    """Reads of unit RESULT files (not the `_`-prefixed stage files, not sidecars or ledgers)."""
    out = {}
    for path, n in store.reads.items():
        if not path.startswith(f"{gen}/"):
            continue
        name = path[len(gen) + 1 :]
        if "/" not in name and not name.startswith("_") and name.endswith(".parquet"):
            out[name] = n
    return out


@pytest.mark.parametrize("make", [_plain, _item], ids=["stage", "item_stage"])
def test_reduce_lists_the_generation_a_constant_number_of_times_and_reads_each_result_once(tmp_path, make):
    n = 40
    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(make(n), store)
    name = _run_all(ctx)
    store.reset()
    reduce_stage(ctx, name)
    assert store.names_calls <= 4, f"{store.names_calls} listings for {n} units: one per unit is the D1 defect"
    reads = _unit_result_reads(store, ctx.generation(name))
    assert len(reads) == n  # non-vacuous: every unit result was read
    assert set(reads.values()) == {1}, f"unit results read more than once: {reads}"


@pytest.mark.parametrize("path", ["single", "sharded"])
def test_reduce_directory_entries_visited_grow_subquadratically(tmp_path, path):
    from tests._perf_structural import assert_subquadratic_growth

    def measure(n: int) -> int:
        store = _CountingStore(tmp_path / f"{path}{n}")
        ctx = _ctx(_plain(n), store)
        name = _run_all(ctx)
        store.reset()
        if path == "single":
            reduce_stage(ctx, name)
        else:
            for i in range(4):
                reduce_subrange(ctx, name, i, 4)
            reduce_from_partials(ctx, name, 4)
        return store.entries

    assert_subquadratic_growth(measure, sizes=(16, 64, 256), label=f"{path} reduce: directory entries visited")


def test_read_result_finds_a_result_without_listing_its_directory(tmp_path):
    """The structural gate behind every per-key `read_result` loop (the reduces and the module-level loops in
    `ghost_gk` `_cache_work`, `xcross` `_probe_comparison_work` and the census `_done_results`, B review R3-02):
    `read_result` without a `names` listing PROBES the candidate files and never lists the directory."""
    from scripts._stages import MISSING, FileSet, write_result

    store = _CountingStore(tmp_path / "s")
    for i in range(20):
        write_result(store, f"g/u{i:04d}", pd.DataFrame({"v": [i]}))
    write_result(store, "g/j0000", {"a": 1})
    write_result(store, "g/f0000", FileSet({"x.bin": b"1", "y.bin": b"22"}))
    store.reset()
    for i in range(20):
        assert read_result(store, f"g/u{i:04d}")["v"].tolist() == [i]
    assert read_result(store, "g/j0000") == {"a": 1}
    fs = read_result(store, "g/f0000")
    assert isinstance(fs, FileSet) and fs.files == {"x.bin": b"1", "y.bin": b"22"}
    assert read_result(store, "g/absent") is MISSING
    assert store.names_calls == 0, f"read_result listed the directory {store.names_calls} time(s)"
    assert store.reads["g/u0007.parquet"] == 1  # each result file is read once (verified on the same bytes)


def _named(stage_name: str, n: int, make_frame) -> Pipeline:
    """A one-stage pipeline whose stage carries the name a module-level loop hard-codes."""
    return Pipeline(
        name="site",
        module="tests.scripts.test_reduce_linear",
        params=(),
        stages=(
            Stage(
                name=stage_name, units=lambda ctx: [f"u{i:05d}" for i in range(n)], work=lambda k, ctx: make_frame(k)
            ),
        ),
        output=stage_name,
    )


def test_census_done_results_lists_a_constant_number_of_times(tmp_path):
    """Spec r3 D1 (B review R3-02): the census `_done_results` reads every done unit of a stage."""
    from scripts.build_territory_ranking_census import _done_results

    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_named("score", 40, lambda k: pd.DataFrame({"game_id": [k]})), store)
    _run_all(ctx)
    store.reset()
    assert len(_done_results(ctx, "score")) == 40
    assert store.names_calls <= 2, f"{store.names_calls} listings for 40 units"


def test_cache_work_lists_a_constant_number_of_times(tmp_path):
    """Spec r3 D1 (B review R3-02): `ghost_gk` `_cache_work` reads every `select` unit (the 28.5-min `cache`; spec
    r4 D19 moved it from the `extract` units to the `select` shards)."""
    from scripts.train_ghost_gk import _cache_work

    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_named("select", 40, lambda k: pd.DataFrame({"_game_id": [k], "v": [1.0]})), store)
    _run_all(ctx)
    reduce_stage(ctx, "select")  # spec r4 D19: the cache reads the units the REDUCED `select` combined
    store.reset()
    assert len(_cache_work("only", ctx)) == 40
    assert store.names_calls <= 2, f"{store.names_calls} listings for 40 units"


@dataclasses.dataclass(frozen=True)
class _ProbeRef:
    prov: str
    mid: str


def test_probe_comparison_lists_a_constant_number_of_times(tmp_path, monkeypatch):
    """Spec r3 D1 (B review R3-02): `xcross` `_probe_comparison_work` reads `extract` units until it holds
    `PROBE_KEEP` non-empty ones; here 38 of 40 are empty, so it reads them all."""
    import scripts.train_xcross_attempt as X
    from scripts._stages import FileSet, Param

    monkeypatch.setattr(X, "_cohort_files", lambda ctx, records, refs, flt: FileSet({"n": str(len(records)).encode()}))
    refs = [_ProbeRef("gs", f"{i:04d}") for i in range(40)]
    pipe = Pipeline(
        name="probe",
        module="tests.scripts.test_reduce_linear",
        params=(
            Param("probe_comparison_providers", str, "gs"),
            # HFU R4-10: the cohort reads only this tier's arms, and every arm key must name a unit (fail closed).
            Param("arms_json", str, json.dumps({"arms": {"public": [f"gs__{i:04d}" for i in range(40)]}})),
        ),
        stages=(
            ItemStage(
                name="extract",
                refs=lambda ctx: refs,
                key=lambda r: f"{r.prov}__{r.mid}",
                work=lambda r, ctx: pd.DataFrame({"_group": [r.mid]} if int(r.mid) >= 38 else {"_group": []}),
            ),
        ),
        output="extract",
    )
    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(pipe, store)
    _run_all(ctx)
    store.reset()
    assert X._probe_comparison_work("only", ctx).files == {"n": b"2"}  # non-vacuous: both kept units found
    assert store.names_calls <= 2, f"{store.names_calls} listings while reading 40 units"


def test_orphan_sweep_spares_an_in_flight_temp(tmp_path):
    """Spec r3 D6(b), B reviews SPEC-R3-10/12: a reduce sweeps an orphan `.tmp-*` older than `ORPHAN_TEMP_AGE_S`
    (2474b73 left 100), but never a concurrent writer's in-flight temp (two jobs may reduce at once), and never a
    temp whose mtime is an implausible placeholder (earlier than the generation marker)."""
    import os
    import time

    from scripts._driver import _CHECKSUMMED
    from scripts._stages import ORPHAN_TEMP_AGE_S

    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_item(4), store)
    name = _run_all(ctx)
    gen = tmp_path / "s" / ctx.generation(name)
    now = time.time()
    started = now - 2 * ORPHAN_TEMP_AGE_S
    os.utime(gen / _CHECKSUMMED, (started, started))  # the generation began long ago
    old = gen / ".tmp-m00000.parquet.aaaa.part"
    old.write_bytes(b"old")
    os.utime(old, (now - ORPHAN_TEMP_AGE_S - 60, now - ORPHAN_TEMP_AGE_S - 60))
    live = gen / ".tmp-m00001.parquet.bbbb.part"  # a concurrent writer, between its write and its os.replace
    live.write_bytes(b"live")
    placeholder = gen / ".tmp-m00002.parquet.cccc.part"
    placeholder.write_bytes(b"placeholder")
    os.utime(placeholder, (0, 0))
    reduce_stage(ctx, name)
    assert not old.exists() and live.exists() and placeholder.exists()
    os.replace(live, gen / "late.bin")  # the concurrent write still completes
    assert (gen / "late.bin").read_bytes() == b"live"


class _SlowConcurrencyStore(_CountingStore):
    """A counting store whose reads take a little time and record how many are in flight at once."""

    def __init__(self, root) -> None:
        import threading

        super().__init__(root)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def read_bytes(self, rel: str) -> bytes | None:
        import time

        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(0.02)
            return super().read_bytes(rel)
        finally:
            with self._lock:
                self.in_flight -= 1


def test_unit_states_verifies_through_a_bounded_pool(tmp_path):
    """Spec r3 D1(b): every run job starts with `unit_states(verify=True)` (INV-3: re-hash before skipping). A
    reduce-only job -- the D3 resume -- re-hashes ALL its units first; sequentially that is about 24 min for the
    3,961 extract shards at the measured 0.36 s per key. The checks run through a bounded pool."""
    from scripts._stages import unit_states

    store = _SlowConcurrencyStore(tmp_path / "s")
    ctx = _ctx(_plain(24), store)
    name = _run_all(ctx)
    st = ctx.pipeline.stage(name)
    store.reset()
    states = unit_states(ctx, st, sorted_units(ctx, st), verify=True)
    assert set(states.values()) == {"done"} and len(states) == 24
    assert store.max_in_flight > 1, "unit verification ran one read at a time"


def test_reduce_output_equals_the_concat_of_each_unit_read_directly(tmp_path):
    """Regression guard for the D1 rewrite: the output is the stage reduce over every unit in reduce order."""
    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_item(25), store)
    name = _run_all(ctx)
    reduce_stage(ctx, name)
    gen = ctx.generation(name)
    keys = sorted_units(ctx, ctx.pipeline.stage(name))
    direct = combine_frames([pd.read_parquet(tmp_path / "s" / gen / f"{k}.parquet") for k in keys], ids=("game_id",))
    assert len(direct) == 25
    assert read_result(store, f"{gen}/_output").equals(direct)


def test_a_corrupt_unit_result_still_fails_the_reduce_and_is_deleted(tmp_path):
    """INV-3 is unchanged by D1: a unit whose checksum fails is deleted (so a re-run redoes it) and the reduce
    refuses -- it never concatenates a poisoned unit."""
    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_plain(5), store)
    name = _run_all(ctx)
    bad = tmp_path / "s" / ctx.generation(name) / "u00002.parquet"
    bad.write_bytes(bad.read_bytes() + b"x")
    with pytest.raises(StageIncompleteError):
        reduce_stage(ctx, name)
    assert not bad.exists()


def test_reduce_reports_progress(tmp_path, capsys):
    """A long reduce is never silent (spec r3 D1): it announces its unit count and its end."""
    store = _CountingStore(tmp_path / "s")
    ctx = _ctx(_plain(12), store)
    name = _run_all(ctx)
    capsys.readouterr()
    reduce_stage(ctx, name)
    err = capsys.readouterr().err
    assert "reduce lin/gen: reading 12 unit(s)" in err
    assert "reduce lin/gen: 12 unit(s) read" in err
