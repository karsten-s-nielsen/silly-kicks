"""HFU T8: a sharded (range-partitioned) reduce is byte-identical to the single reduce, and its empty/id
policy matches the single reduce. The controller orchestration (sub-reduce actions) is tested in
tests/scripts/hf_jobs/test_campaign.py; this file pins the pure byte-identity + policy core."""

from __future__ import annotations

import pandas as pd

from scripts._stages import Pipeline, Stage, combine_frames, reduce_key_ranges


def _mid(key: str) -> str:
    """reduce_order analog of xshot/xcross `_mid_of`: order by the `<mid>` part of a `<prov>__<mid>` key, which
    differs from a plain full-key sort whenever the provider prefix reorders the keys."""
    return key.partition("__")[2]


def _shardy_work(key, ctx):
    if key == "sp__0003":  # one EXCLUDED unit (reviewer B D1-IMPL-09: the sharded path must skip it, not read it)
        from scripts._item_outcome import ItemExcluded

        raise ItemExcluded("no usable rows for this unit")
    m = _mid(key)
    return pd.DataFrame({"g": [int(m)], "mid": [m]})


def _shardy_reduce(results, ctx):
    parts = [f for f in results.values() if len(f)]
    if not parts:
        raise RuntimeError("no data")
    return combine_frames(parts, ids=("g",))


#: A shardable stage whose reduce_order (`_mid`) differs from a plain full-key sort, with one excluded unit.
#: `_mid` order of the DONE units: 0001, 0005, 0007, 0009 (sp__0003 excluded). Plain full-key sort would be
#: gs__0007, gs__0009, kl__0001, kl__0005 -> mids 0007, 0009, 0001, 0005 -- a DIFFERENT order.
_SHARDY = Pipeline(
    name="shardytest",
    module="tests.scripts.test_sharded_reduce",
    params=(),
    stages=(
        Stage(
            name="ex",
            units=lambda ctx: ["kl__0001", "gs__0009", "kl__0005", "sp__0003", "gs__0007"],
            work=_shardy_work,
            reduce=_shardy_reduce,
            reduce_order=_mid,
            shardable_reduce=("g",),
            max_failures=0,
        ),
    ),
    output="ex",
)


def _frames():
    return {
        "k0": pd.DataFrame({"game_id": [1, 2], "v": [10, 20]}),
        "k1": pd.DataFrame({"game_id": ["a"], "v": [30]}),  # mixed int/str ids across parts
        "k2": pd.DataFrame({"game_id": [3], "v": [40]}),
        "k3": pd.DataFrame({"game_id": ["b", "c"], "v": [50, 60]}),
    }


def test_sharded_concat_is_byte_identical_to_single_concat():
    frames = _frames()
    keys = ["k0", "k1", "k2", "k3"]
    full = combine_frames([frames[k] for k in keys], ids=("game_id",))
    ranges = reduce_key_ranges(keys, 2)  # [[k0,k1],[k2,k3]] -- contiguous, sorted
    assert len(ranges) >= 2  # non-vacuous: really sharded
    partials = [combine_frames([frames[k] for k in r], ids=("game_id",)) for r in ranges]
    sharded = combine_frames(partials, ids=("game_id",))
    assert sharded.equals(full)  # ordered range-concat == single concat (associativity + idempotent storable ids)


def test_an_empty_range_contributes_an_empty_partial_no_spurious_raise():
    frames = _frames()
    # one empty range among non-empty ones: its partial is empty, the final concat equals the single reduce
    empty_partial = combine_frames([], ids=("game_id",))
    assert empty_partial.empty
    full = combine_frames([frames[k] for k in ["k0", "k1"]], ids=("game_id",))
    non_empty = combine_frames([frames["k0"], frames["k1"]], ids=("game_id",))
    sharded = combine_frames([empty_partial, non_empty], ids=("game_id",))
    assert sharded.equals(full)  # an empty range did not raise and did not change the result


def test_sharded_reduce_mixed_id_types():
    """HFU T8 S3 (B-03): a range mixing int and str ids canonicalizes to the SAME ids as the single reduce.
    `storable_id_column` is idempotent, so the per-range sub-concats and the final concat each canonicalize
    and the sharded id column equals the single-reduce id column value-for-value and dtype-for-dtype."""
    frames = _frames()  # game_id mixes int (1,2,3) and str ('a','b','c') across parts
    keys = ["k0", "k1", "k2", "k3"]
    single = combine_frames([frames[k] for k in keys], ids=("game_id",))
    ranges = reduce_key_ranges(keys, 2)
    assert len(ranges) >= 2  # non-vacuous
    partials = [combine_frames([frames[k] for k in r], ids=("game_id",)) for r in ranges]
    sharded = combine_frames(partials, ids=("game_id",))
    assert sharded["game_id"].equals(single["game_id"])  # same canonical ids, sharded or single
    assert sharded["game_id"].map(type).eq(str).all()  # mixed int/str were canonicalized to strings, both paths


def test_sharded_concat_multi_column_ids_byte_identical():
    """CONSIDER (reviewer A D-IMPL-08): the declared shardable reduces use MULTI-column ids (ghost_outfield's
    ``game_id``/``player_id``/``team_id``). A production-shape arm over a 3-column id -- including mixed int/str
    across parts -- rather than relying on the associativity argument alone: the sharded range-concats equal the
    single concat value-for-value and dtype-for-dtype."""
    frames = {
        "k0": pd.DataFrame({"game_id": [1, 2], "player_id": ["a", "b"], "team_id": [10, 10], "v": [1, 2]}),
        "k1": pd.DataFrame({"game_id": ["g"], "player_id": [3], "team_id": ["t"], "v": [3]}),  # mixed int/str ids
        "k2": pd.DataFrame({"game_id": [4], "player_id": ["c"], "team_id": [20], "v": [4]}),
    }
    ids = ("game_id", "player_id", "team_id")
    keys = ["k0", "k1", "k2"]
    single = combine_frames([frames[k] for k in keys], ids=ids)
    ranges = reduce_key_ranges(keys, 2)
    assert len(ranges) >= 2  # non-vacuity: really sharded
    partials = [combine_frames([frames[k] for k in r], ids=ids) for r in ranges]
    sharded = combine_frames(partials, ids=ids)
    assert sharded.equals(single)  # multi-column id: byte-identical, sharded or single


def test_reduce_key_ranges_partitions_the_given_order_contiguously():
    # It PRESERVES the given order (the caller supplies reduce_order) -- it does NOT sort (reviewer B D1-IMPL-08):
    # plain-sort boundaries would not align with a custom reduce_order and would reorder rows across shards.
    assert reduce_key_ranges(["c", "a", "b", "d", "e"], 2) == [["c", "a", "b"], ["d", "e"]]
    assert reduce_key_ranges(["a", "b"], 5) == [["a"], ["b"]]  # never more ranges than keys
    assert reduce_key_ranges([], 3) == [[]]  # an all-empty stage: one empty range


def test_parity_sharded_reduce_equals_single_reduce_via_cli(tmp_path):
    """HFU T8 S8: the `_parity` sharded-reduce arm on the subprocess seam the controller actually dispatches
    -- gen units with the inline reduce disabled, >= 2 `--subreduce` partials, a final `--from-partials`
    concat. The gen reduce output (and every unit result) is byte-identical to the single inline reduce."""
    from tests.scripts import _parity

    module = "tests.scripts._toy_pipeline"
    data = tmp_path / "d"
    data.mkdir()
    single_store = tmp_path / "single"
    sharded_store = tmp_path / "sharded"
    args: list[str] = []  # toy defaults: 12 gen units
    _parity.cli(module, single_store, data, args, "--stages", "gen")  # units + the single inline reduce
    n = _parity.run_sharded_reduce(module, sharded_store, data, args, stage="gen", n_ranges=3)
    assert n >= 2  # non-vacuity: really sharded
    single = _parity.outputs(single_store)
    sharded = _parity.outputs(sharded_store)
    assert single  # non-empty
    shared = {k: v for k, v in sharded.items() if k in single}
    assert shared == single  # every gen unit result + the gen _output byte-identical; sharded adds only partials


def test_sharded_reduce_equals_single_reduce_end_to_end(tmp_path):
    """HFU T8 parity: run a shardable stage's units, sub-reduce its sorted-key ranges into partials, then the
    final concat; the output is byte-identical to the single reduce (what the controller emits across jobs)."""
    from scripts._stages import (
        _OUTPUT,
        LocalStore,
        make_context,
        read_result,
        reduce_from_partials,
        reduce_stage,
        reduce_subrange,
    )
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor
    from tests.scripts import _toy_pipeline
    from tests.scripts._stage_harness import params_from_args

    pipeline = _toy_pipeline.PIPELINE
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, []),
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=None,
    )
    LocalExecutor(ExecutorConfig(workers=1, poll_interval_s=0.05)).run_stage(ctx, "gen")  # all units done
    gen = ctx.generation("gen")
    n = 3
    for i in range(n):
        reduce_subrange(ctx, "gen", i, n)  # the worker owns the range; the controller passes only the count
    reduce_from_partials(ctx, "gen", n)
    sharded = read_result(ctx.store, f"{gen}/{_OUTPUT}")
    reduce_stage(ctx, "gen")  # overwrites _output with the single (whole-stage) reduce
    single = read_result(ctx.store, f"{gen}/{_OUTPUT}")
    assert len(single) > 0 and sharded.equals(single)  # byte-identical across jobs


def test_sharded_reduce_honors_reduce_order_and_skips_excluded(tmp_path):
    """Reviewer B D1-IMPL-08/09: a shardable stage with a custom reduce_order (differs from plain key sort) and
    an EXCLUDED unit. The sharded reduce must be byte-identical to the single reduce -- same selection (the
    excluded unit skipped, never read) AND same concat order (reduce_order, not the controller's plain key sort).
    With the plain-sort sharded path this either raised on the excluded unit or concatenated in the wrong order."""
    from scripts._stages import (
        _OUTPUT,
        LocalStore,
        make_context,
        read_result,
        reduce_from_partials,
        reduce_stage,
        reduce_subrange,
        run_unit,
        sorted_units,
    )
    from tests.scripts._stage_harness import params_from_args

    pipeline = _SHARDY
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, []),
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=None,
    )
    st = pipeline.stage("ex")
    for k in sorted_units(ctx, st):
        run_unit(ctx, st, k, worker="w0")
    gen = ctx.generation("ex")
    reduce_stage(ctx, "ex")  # the single reduce
    single = read_result(ctx.store, f"{gen}/{_OUTPUT}")
    assert single["mid"].tolist() == ["0001", "0005", "0007", "0009"]  # reduce_order, excluded sp__0003 skipped
    n = 2
    for i in range(n):
        reduce_subrange(ctx, "ex", i, n)  # worker owns done-filter + reduce_order; controller passes only the count
    reduce_from_partials(ctx, "ex", n)
    sharded = read_result(ctx.store, f"{gen}/{_OUTPUT}")
    assert sharded.equals(single)  # byte-identical: same selection, same concat order


def test_cli_main_runs_atexit_and_flushes_before_a_hard_os_exit(monkeypatch):
    """HFU: a migrated module's CLI entry must survive a native C++ teardown abort at interpreter shutdown
    (observed on Linux CI: Arrow "terminate called without an active exception", SIGABRT, AFTER a correct
    `--subreduce` already wrote its output). `cli_main` runs the registered atexit handlers (so the worker-
    input temp-dir rmtree fence is preserved and any pool is joined), flushes the std streams (so the
    `--plan` JSON on stdout is not lost), THEN `os._exit(rc)` -- a hard exit that skips the C++ static
    destructors. The HF controller marks ANY nonzero rc as a failed job, so a correct run must never be
    failed by a teardown abort, and `main`'s rc must reach the exit unchanged."""
    import atexit as _atexit

    import pytest

    from scripts import _stages

    order: list = []
    monkeypatch.setattr(_stages, "main", lambda pipeline, argv=None: 7)  # rc must propagate unchanged
    monkeypatch.setattr(_atexit, "_run_exitfuncs", lambda: order.append("atexit"))
    monkeypatch.setattr(_stages.sys.stdout, "flush", lambda: order.append("flush_out"))
    monkeypatch.setattr(_stages.sys.stderr, "flush", lambda: order.append("flush_err"))

    class _HardExit(BaseException):
        pass

    def _fake_os_exit(code):
        order.append(("exit", code))
        raise _HardExit  # stand in for the real _exit so the test can observe it

    monkeypatch.setattr(_stages.os, "_exit", _fake_os_exit)
    with pytest.raises(_HardExit):
        _stages.cli_main(_SHARDY, [])
    assert order == ["atexit", "flush_out", "flush_err", ("exit", 7)]  # cleanups + flush BEFORE the hard exit


def test_hard_exit_runs_atexit_and_flushes_before_os_exit(monkeypatch):
    """HFU D1-IMPL-26: the shared hardened exit, used by every migrated module's `cli_main` AND the executor
    reduce/worker child `__main__`. It runs the Python atexit handlers, flushes the std streams, THEN
    `os._exit(rc)` -- so a native C++ teardown abort at interpreter shutdown cannot fail a correct child, and
    `rc` is carried through unchanged."""
    import atexit as _atexit

    import pytest

    from scripts import _stages

    order: list = []
    monkeypatch.setattr(_atexit, "_run_exitfuncs", lambda: order.append("atexit"))
    monkeypatch.setattr(_stages.sys.stdout, "flush", lambda: order.append("flush_out"))
    monkeypatch.setattr(_stages.sys.stderr, "flush", lambda: order.append("flush_err"))

    class _HardExit(BaseException):
        pass

    def _fake_os_exit(code):
        order.append(("exit", code))
        raise _HardExit

    monkeypatch.setattr(_stages.os, "_exit", _fake_os_exit)
    with pytest.raises(_HardExit):
        _stages.hard_exit(5)
    assert order == ["atexit", "flush_out", "flush_err", ("exit", 5)]  # cleanups + flush BEFORE the hard exit


def test_reduce_from_partials_refuses_a_shard_count_mismatch(tmp_path):
    """Reviewer B D1-IMPL-19: each sub-reduce stamps the n_shards it was computed for; the final reduce refuses
    (loud) if a partial carries a DIFFERENT count, so a crash plus a changed reduce_wall (which re-partitions the
    keys) cannot silently concatenate a stale subset of the partials."""
    import pytest

    from scripts._stages import (
        LocalStore,
        StageIncompleteError,
        make_context,
        reduce_from_partials,
        reduce_subrange,
        run_unit,
        sorted_units,
    )
    from tests.scripts._stage_harness import params_from_args

    pipeline = _SHARDY
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, []),
        {"commit": "t", "tree_state": "clean", "code_identity": "t"},
        store=LocalStore(tmp_path / "s"),
        data_root=None,
    )
    st = pipeline.stage("ex")
    for k in sorted_units(ctx, st):
        run_unit(ctx, st, k, worker="w0")
    reduce_subrange(ctx, "ex", 0, 2)  # partials stamped n_shards=2
    reduce_subrange(ctx, "ex", 1, 2)
    with pytest.raises(StageIncompleteError, match="different shard count"):
        reduce_from_partials(ctx, "ex", 3)  # asked for 3 -> stale 2-shard partials -> refuse, never concat a subset
