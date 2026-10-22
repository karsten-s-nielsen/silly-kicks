"""In-process stage runner for unit tests of migrated pipelines (ADR-128).

``run_in_process`` walks a pipeline's stages in DAG order and runs every unit through the REAL
``_stages.run_unit`` / ``reduce_stage`` (checksummed results, ItemStage via ``for_each``), without the
executor's worker subprocesses -- so a unit test exercises the module's stage functions in milliseconds.
The executor itself, and W1N1-vs-W2N2 parity, are covered by ``tests/scripts/hf_jobs/test_executor.py``
and the ``test_parity_*`` files.
"""

from __future__ import annotations

import pathlib

from scripts._stages import (
    FileSet,
    ItemStage,
    LocalStore,
    RunContext,
    StageIncompleteError,
    build_parser,
    make_context,
    reduce_stage,
    run_unit,
    sorted_units,
)

PROV = {"commit": "t" * 40, "tree_state": "clean", "dirty": False, "code_identity": "t" * 40}


def params_from_args(pipeline, args: list[str]) -> dict:
    """The pipeline parameters exactly as the uniform CLI would parse ``args``."""
    ns = build_parser(pipeline).parse_args([*args, "--plan", "--store", "unused"])
    return {p.name: getattr(ns, p.name) for p in pipeline.params}


def run_in_process(pipeline, args: list[str], *, store, data_root, prov=None) -> RunContext:
    """Run every stage; raises ``StageIncompleteError`` at the first stage whose reduce refuses."""
    ctx = make_context(
        pipeline,
        params_from_args(pipeline, args),
        prov or PROV,
        store=LocalStore(pathlib.Path(store)),
        data_root=data_root,
    )
    for st in pipeline.stages:
        refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
        for key in sorted_units(ctx, st):
            run_unit(ctx, st, key, worker="test", ref=refs.get(key))
        reduce_stage(ctx, st.name)
    return ctx


def failure_of(ctx: RunContext, stage: str, key: str = "only") -> str:
    """The recorded error of a failed unit (its ``.failed.json``)."""
    import json

    raw = ctx.store.read_bytes(f"{ctx.generation(stage)}/{key}.failed.json")
    return json.loads(raw)["error"] if raw else ""


def export(ctx: RunContext, dest: pathlib.Path) -> pathlib.Path:
    """Write the pipeline output FileSet into ``dest`` and return it."""
    assert ctx.pipeline.output is not None, f"{ctx.pipeline.name} declares no output stage"
    out = ctx.upstream(ctx.pipeline.output)
    assert isinstance(out, FileSet)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in out.files.items():
        (dest / name).write_bytes(data)
    return dest


def run_toy(module: str, tmp: pathlib.Path) -> dict[str, bytes]:
    """Run ``module``'s toy pipeline end to end (``toy_session``) and return the output stage's FileSet as
    ``{name: bytes}`` -- the commit-free output gate (S5.10) reads it."""
    import importlib
    import os

    from scripts._toy_corpora import TOY_PARAMS, toy_session

    pipe = importlib.import_module(module).PIPELINE
    tmp = pathlib.Path(tmp)
    with toy_session(module, str(tmp / "d")) as (data, extra_env):
        os.environ.update(extra_env)
        ctx = run_in_process(pipe, list(TOY_PARAMS[module]), store=tmp / "s", data_root=data)
    assert ctx.pipeline.output is not None, f"{module} declares no output stage"
    out = ctx.upstream(ctx.pipeline.output)
    assert isinstance(out, FileSet), f"{module} output is not a FileSet"
    return dict(out.files)


__all__ = ["PROV", "StageIncompleteError", "export", "failure_of", "params_from_args", "run_in_process", "run_toy"]
