"""INV-11: a reduce lists each generation once and reads each result file once, verified, before it writes."""

from __future__ import annotations

import collections

import pandas as pd
import pytest

from scripts import _stages as s


class CountingStore(s.LocalStore):
    calls: collections.Counter  # declared for pyright (set via object.__setattr__ on the frozen base)

    def __init__(self, root):
        super().__init__(root)
        object.__setattr__(self, "calls", collections.Counter())

    def names(self, rel_dir):
        self.calls[("names", rel_dir)] += 1
        return super().names(rel_dir)

    def read_bytes(self, rel):
        self.calls[("read", rel)] += 1
        return super().read_bytes(rel)


def _pipeline(n: int, kind: str) -> s.Pipeline:
    def work(k, ctx):
        if kind == "fileset":
            return s.FileSet({"a.bin": k.encode(), "b.bin": (k * 2).encode()})
        # real mixed id types: int ids from one provider, str ids from another (Cycle 1 defect class)
        gid = int(k) if int(k) % 2 else f"J{k}"
        return pd.DataFrame({"game_id": [gid], "x": [float(k)]})

    def reduce(results, ctx):
        if kind == "fileset":
            return {k: sorted(v.files) for k, v in results.items()}
        return s.combine_frames(list(results.values()), ids=("game_id",))

    st = s.Stage(name="a", units=lambda c: [s.int_key(i) for i in range(n)], work=work, reduce=reduce)
    return s.Pipeline(name="t", module="tests.scripts._toy_pipeline", stages=(st,))


def _ctx(tmp_path, n, kind):
    store = CountingStore(tmp_path)
    prov = {"commit": "c" * 40, "tree_state": "clean", "code_identity": "c" * 40}
    ctx = s.make_context(_pipeline(n, kind), {}, prov, store=store, data_root=None)
    for k in s.sorted_units(ctx, ctx.pipeline.stage("a")):
        s.run_unit(ctx, ctx.pipeline.stage("a"), k, worker="w0")
    store.calls.clear()
    return ctx, store


@pytest.mark.parametrize("kind", ["parquet", "fileset"])
@pytest.mark.parametrize("n", [50, 100, 200])
def test_reduce_lists_once_and_reads_each_file_once(tmp_path, n, kind):
    ctx, store = _ctx(tmp_path, n, kind)
    s.reduce_stage(ctx, "a")
    gen = ctx.generation("a")
    assert store.calls[("names", gen)] == 1
    reads = {rel: c for (op, rel), c in store.calls.items() if op == "read"}
    assert max(reads.values()) == 1, {r: c for r, c in reads.items() if c > 1}
    # every result file was read: data + sidecar (+ 2 members for a fileset)
    per_unit = 2 + (2 if kind == "fileset" else 0)
    assert sum(1 for r in reads if r.startswith(gen + "/0")) == n * per_unit


def test_a_corrupt_result_writes_nothing_and_returns_its_unit_to_pending(tmp_path):
    ctx, store = _ctx(tmp_path, 20, "parquet")
    gen = ctx.generation("a")
    victim = tmp_path / gen / "0007.parquet"
    victim.write_bytes(victim.read_bytes()[:-3] + b"xyz")
    with pytest.raises(s.StageIncompleteError, match="0007"):
        s.reduce_stage(ctx, "a")
    names = store.names(gen)
    assert "_summary.json" not in names and "_inputs.json" not in names
    assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0007"], verify=False)["0007"] == "pending"
    assert any(p.name.startswith("0007") for p in (tmp_path / gen / s.CORRUPT_DIR).rglob("*"))
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0007", worker="w0")
    s.reduce_stage(ctx, "a")
    assert "_output.parquet" in store.names(gen)


def test_unit_states_carries_its_listing(tmp_path):
    ctx, _store = _ctx(tmp_path, 3, "parquet")
    states = s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000", "0001", "0002"], verify=False)
    assert dict(states) == {"0000": "done", "0001": "done", "0002": "done"}
    assert "0000.parquet" in states.names and isinstance(states.names, frozenset)


def test_plan_and_live_history_are_linear_with_one_crashed_job(tmp_path):
    """S5.11: --plan lists each generation once and reads no result bytes; live history reads one file per job
    aggregate plus unit records only for the units of the job that wrote no aggregate."""
    import json as _json

    from scripts.hf_jobs import campaign as cp
    from scripts.hf_jobs import phases

    ctx, store = _ctx(tmp_path, 200, "parquet")
    gen = ctx.generation("a")
    keys = sorted(store.names(gen))
    units = [k[:-8] for k in keys if k.endswith(".parquet")]
    for j in range(4):  # 5 jobs x 40 units; job 4 crashed before writing its aggregate
        rows = {u: {"outcome": "done", "wall_s": 1.0, "finished_ns": 1} for u in units[j * 40 : (j + 1) * 40]}
        rel = phases.cost_rel("t", "a", ctx.tokens["a"], f"t.d.a.{j}of5.X", "A")
        store.write_bytes(rel, _json.dumps(rows).encode())
    store.calls.clear()
    p = s.plan(ctx)
    assert store.calls[("names", gen)] == 1
    assert not any(op == "read" and rel.endswith(".parquet") for (op, rel) in store.calls)
    store.calls.clear()
    cp.live_history(p, store, {})
    unit_reads = [rel for (op, rel) in store.calls if op == "read" and rel.endswith(".ledger.json")]
    assert len(unit_reads) == 40  # only job 4's units


def test_the_reduce_record_carries_bytes_read(tmp_path, monkeypatch):
    """S5.2 / CCTP-IMPL-05: read_results reports total bytes read; the reduce ledger record carries it."""
    import json as _json

    monkeypatch.setenv("SILLY_KICKS_JOB_LABEL", "g.v.a.0of1")
    monkeypatch.setenv("SILLY_KICKS_ATTEMPT_ID", "20261021T000000Z_0001")
    ctx, store = _ctx(tmp_path, 5, "parquet")
    gen = ctx.generation("a")
    keys = ["0000", "0001", "0002", "0003", "0004"]
    results, corrupt, nbytes = s.read_results(store, gen, keys, names=store.names(gen), cap=4)
    assert not corrupt and len(results) == 5 and nbytes > 0
    s.reduce_stage(ctx, "a")
    recs = [_json.loads(p.read_text()) for p in (tmp_path / "_ledger").rglob("*.json") if "_reduce" in str(p)]
    assert recs and recs[0]["bytes_read"] > 0
