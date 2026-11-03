"""A unit failure records WHERE it happened, for plain `Stage` and `ItemStage` units (HFU spec §4.3, G3).

Cycle 1's failure record held only ``"OSError: [Errno 5] Input/output error"`` (spec E9): no path, no frame, so
the EIO site could not be attributed. The record gains a bounded, additive ``traceback``; the ``error`` string
keeps its exact format (Hyrum: the driver and ``campaign status`` read it).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

import scripts._driver as drv
from scripts._stages import ItemStage, LocalStore, Pipeline, Stage, make_context, run_unit, sorted_units
from tests.scripts._stage_harness import PROV


def _raise_eio(*_args) -> None:
    raise OSError(5, "Input/output error")


def _raise_big(*_args) -> None:
    raise OSError(5, "x" * 1_000_000)


def _recurse(depth: int) -> None:
    if depth == 0:
        _raise_deep()
    _recurse(depth - 1)


def _raise_deep() -> None:
    raise OSError(5, "deep")


def _failure_record(ctx, stage: str, key: str) -> dict:
    raw = ctx.store.read_bytes(f"{ctx.generation(stage)}/{key}.failed.json")
    assert raw is not None, f"no failure record for {stage}/{key}"
    return json.loads(raw)


def _run_all(pipeline: Pipeline, store_root: pathlib.Path):
    ctx = make_context(pipeline, {}, PROV, store=LocalStore(store_root), data_root=None)
    for st in pipeline.stages:
        refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
        for key in sorted_units(ctx, st):
            run_unit(ctx, st, key, worker="test", ref=refs.get(key))
    return ctx


def test_plain_stage_failure_records_traceback(tmp_path):
    pipeline = Pipeline(
        name="locus",
        module=__name__,
        stages=(Stage(name="s", units=lambda ctx: ["only"], work=lambda key, ctx: _raise_eio()),),
    )
    ctx = _run_all(pipeline, tmp_path)

    record = _failure_record(ctx, "s", "only")

    assert record["error"] == "OSError: [Errno 5] Input/output error"
    assert "_raise_eio" in record["traceback"]
    assert 'File "' in record["traceback"]


def test_item_stage_failure_records_traceback(tmp_path):
    pipeline = Pipeline(
        name="locus",
        module=__name__,
        stages=(
            ItemStage(
                name="items",
                refs=lambda ctx: ["m1"],
                key=lambda r: r,
                load=lambda ref, ctx: _raise_eio(),
                work=lambda item, ctx: pd.DataFrame({"v": [1]}),
            ),
        ),
    )
    ctx = _run_all(pipeline, tmp_path)

    record = _failure_record(ctx, "items", "m1")

    assert record["error"] == "OSError: [Errno 5] Input/output error"
    assert "_raise_eio" in record["traceback"]


def test_traceback_is_capped():
    try:
        _raise_big()
    except OSError as exc:
        big = drv.format_failure_traceback(exc)
    assert len(big) <= 7001
    assert "_raise_big" in big  # the frames survive a 1 MB message
    assert 'File "' in big

    try:
        _recurse(200)
    except OSError as exc:
        deep = drv.format_failure_traceback(exc)
    assert len(deep) <= 7001
    assert "_raise_deep" in deep  # the frames NEAREST the raise are the ones kept
