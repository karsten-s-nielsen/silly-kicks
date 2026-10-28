"""A toy ADR-128 pipeline that exercises every executor path (framework tests only).

`gen` is an ItemStage whose low keys are SLOW, so at W >= 2 completion order differs from key order.
`folds` is inner_parallel: each unit maps 4 inner keys over the InnerPool, inner key 0000 is slow.
`final` combines fold scores by a SEQUENTIAL float sum whose value depends on order (1e16 + 1 loses the
1), so a reduce that iterated completion order instead of key order would change the bytes.
`combined` reads every unit `gen` combined (spec r4 D19: a broken one reopens `gen`).
`env` returns the worker's environment pins. `strict` fails on `--fail-keys` with zero tolerance.
`hold` sleeps `--hold-s` seconds (0 by default): in the rung-5 canary its job is the sibling a halt must cancel.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import pandas as pd

from scripts._stages import FileSet, ItemStage, Param, Pipeline, Stage, cli_main, int_key, ordered_values
from scripts._thread_pin import DETERMINISTIC_VARS, RUNTIME_THREADS_VAR

_VALUES = (1e16, 1.0, -1e16, 1.0)


def _refs(ctx):
    return [("toy", int_key(i)) for i in range(int(ctx.params["n_items"]))]


def _crash_once(ctx, key: str) -> None:
    marker = ctx.params.get("crash_once")
    if marker and key == "0003" and not pathlib.Path(marker).exists():
        pathlib.Path(marker).write_text("crashed", encoding="utf-8")
        os._exit(137)  # an OOM-killed worker: no cleanup, no ledger record


def _gen_work(item, ctx):
    _, key = item
    _crash_once(ctx, key)
    if f"toy__{key}" in str(ctx.params.get("fail_keys") or "").split(","):  # gen keys are JOINED keys
        raise ValueError(f"injected failure for {key}")
    time.sleep(0.4 if int(key) % 4 == 0 else 0.0)
    return pd.DataFrame({"i": [int(key)], "v": [_VALUES[int(key) % len(_VALUES)]]})


def _gen_reduce(results, ctx):
    return pd.concat(list(results.values()), ignore_index=True)


def _inner_eval(fold: int, j: int, values: tuple) -> float:
    time.sleep(0.3 if j == 0 else 0.0)
    return float(values[(fold + j) % len(values)])


def _fold_work(key, ctx):
    df = ctx.upstream("gen")
    items = {int_key(j): (int(key), j, tuple(df["v"].tolist())) for j in range(4)}
    res = ctx.pool.map_keyed(_inner_eval, items)
    total = 0.0
    for v in ordered_values(res):  # sequential, order-sensitive: the canonical inner reduce
        total += v
    return {"fold": key, "score": total}


def _final_work(key, ctx):
    folds = ctx.upstream("folds")
    total = 0.0
    for fold in sorted(folds):
        total += folds[fold]["score"]
    return FileSet({"result.json": json.dumps({"total": total}, sort_keys=True).encode()})


def _combined_work(key, ctx):
    # spec r4 D19: every unit `gen` combined, read in a pool; a missing or corrupt one reopens `gen`
    from scripts._stages import read_combined_units

    res = read_combined_units(ctx, "gen")
    return {"keys": list(res), "i": [int(df["i"].iloc[0]) for df in res.values()]}


def _env_work(key, ctx):
    return {v: os.environ.get(v) for v in sorted(DETERMINISTIC_VARS)}


def _thread_env_work(key, ctx):
    # the worker's thread env + the n_jobs-lever transport, for the thread_safe injection tests (T2)
    keys = [*sorted(DETERMINISTIC_VARS), RUNTIME_THREADS_VAR]
    return {v: os.environ.get(v) for v in keys}


def _nfit_work(key, ctx):
    # A deterministic OpenMP (HistGB) fit whose stored output must be byte-identical whatever per-worker
    # thread budget the executor injects -- the EXECUTOR-level (W,N) thread-axis parity subject (B46).
    import io

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(int(key))
    X = rng.standard_normal((400, 12))
    y = rng.standard_normal(400)
    reg = HistGradientBoostingRegressor(max_iter=15, random_state=42).fit(X, y)
    grid = np.random.default_rng(999).standard_normal((64, 12))
    buf = io.BytesIO()
    np.save(buf, reg.predict(grid).astype(np.float64))
    return FileSet({"preds.npy": buf.getvalue()})


def _hold_work(key, ctx):
    """HFU rung 5 (C3): stay alive for ``hold_s`` seconds, independent of every other stage."""
    time.sleep(float(ctx.params["hold_s"]))
    return {"held_s": float(ctx.params["hold_s"])}


def _strict_work(key, ctx):
    fail = str(ctx.params.get("fail_keys") or "")
    if fail == "*" or key in fail.split(","):
        raise ValueError(f"strict failure {key}")
    return {"k": key}


PIPELINE = Pipeline(
    name="toy",
    module="tests.scripts._toy_pipeline",
    params=(
        Param("n_items", int, 12),
        Param("fail_keys", str, ""),
        Param("crash_once", str, ""),
        Param("nfit_units", int, 4),  # the THREADS executor tests run 5 (HFU S3 item 2, review S3B-06)
        Param("hold_s", float, 0.0),  # HFU rung 5: the canary halt arm's sibling job (C3)
    ),
    stages=(
        ItemStage(
            name="gen",
            refs=_refs,
            key=lambda r: r,
            work=_gen_work,
            reduce=_gen_reduce,
            params=("n_items",),
            max_failures=None,
            shardable_reduce=(),  # HFU T8: a plain concat reduce (no id columns) -> shardable
        ),
        Stage(
            name="folds",
            units=lambda ctx: [int_key(f) for f in range(3)],
            work=_fold_work,
            needs=("gen",),
            inner_parallel=True,
            reduce=lambda results, ctx: dict(results),
        ),
        Stage(
            name="final",
            units=lambda ctx: ["only"],
            work=_final_work,
            needs=("folds",),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(
            name="combined",
            units=lambda ctx: ["only"],
            work=_combined_work,
            needs=("gen",),
            reduce=lambda r, ctx: r["only"],
        ),
        Stage(name="env", units=lambda ctx: ["only"], work=_env_work),
        Stage(name="env_omp", units=lambda ctx: ["only"], work=_thread_env_work, thread_safe=True, thread_lever="omp"),
        Stage(
            name="nfit",
            units=lambda ctx: [int_key(i) for i in range(int(ctx.params["nfit_units"]))],
            params=("nfit_units",),
            work=_nfit_work,
            thread_safe=True,
            thread_lever="omp",
        ),
        Stage(
            name="env_njobs",
            units=lambda ctx: ["only"],
            work=_thread_env_work,
            thread_safe=True,
            thread_lever="n_jobs",
        ),
        Stage(
            name="strict", units=lambda ctx: [int_key(i) for i in range(6)], work=_strict_work, params=("fail_keys",)
        ),
        Stage(
            name="sloppy",
            units=lambda ctx: [int_key(i) for i in range(6)],
            work=_strict_work,
            params=("fail_keys",),
            max_failures=None,  # tolerant: only the first-k rule can abort it
        ),
        Stage(name="hold", units=lambda ctx: ["only"], work=_hold_work, params=("hold_s",)),
    ),
    output="final",
)

if __name__ == "__main__":
    cli_main(PIPELINE)
