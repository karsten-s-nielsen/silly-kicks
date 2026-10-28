"""Spec r3 D6: the pipeline SURVIVES a mount that silently loses writes.

On 2474b73 one job's `hf-mount` lost about 0.85 % of its renames: the run job COMPLETED, yet some unit results,
sidecars and ledgers never reached their names (an orphan `.tmp-*` was left for each). Detection already exists
(absent = pending; a sidecar mismatch = untrusted, redone). This file proves the whole recovery path end to end:
a local plan-act-replan loop with the REAL `plan`, `live_history`, `next_actions`, executor and reduces, where
after every run action each unit file the action wrote is lost with probability 0.1. It must finish, produce
the same bytes as a loss-free run, and report the loss.
"""

from __future__ import annotations

import pathlib
import random
import types

from scripts._stages import LocalStore, make_context
from scripts._stages import plan as stage_plan
from scripts.hf_jobs import campaign as C
from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor
from scripts.hf_jobs.planner import Constraints
from tests.scripts import _toy_pipeline
from tests.scripts._parity import outputs
from tests.scripts._stage_harness import params_from_args

_UNIT_FILE_SUFFIXES = (".parquet", ".json", ".sha256", ".ledger.json")


def _files(root: pathlib.Path) -> set[pathlib.Path]:
    return {p for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()


def _ctx(root: pathlib.Path):
    pipeline = _toy_pipeline.PIPELINE
    prov = {"commit": "t", "tree_state": "clean", "code_identity": "t"}
    return make_context(pipeline, params_from_args(pipeline, []), prov, store=LocalStore(root), data_root=None)


def _drive(root: pathlib.Path, *, loss: float, seed: int = 7, max_steps: int = 40) -> tuple[int, list[str]]:
    """Run the toy pipeline to completion through the controller's decisions; return (files lost, loss reports)."""
    ctx = _ctx(root)
    rng = random.Random(seed)  # noqa: S311 -- deterministic test fault injection, not cryptography
    cache: dict = {}
    jobs: list = []
    lost = 0
    reports: list[str] = []
    for _step in range(max_steps):
        p = stage_plan(ctx)
        for st in p["stages"]:
            history = [j for j in reversed(jobs) if st["name"] in j.labels["stage"].split("-")]
            why = C._lost_after_completed(st["name"], history, int(st.get("pending") or 0))
            if why:
                reports.append(why)
        acts = C.next_actions(p, running=set(), history=C.live_history(p, ctx.store, cache), c=Constraints())
        if not acts:
            assert all(st.get("reduced") for st in p["stages"]), "no action left but a stage is not reduced"
            return lost, reports
        for a in acts:
            assert a.kind != "halt", a.reason
            ex = LocalExecutor(
                ExecutorConfig(
                    workers=max(1, a.workers),
                    inner_workers=max(1, a.inner_workers),
                    profile_units=a.profile,
                    poll_interval_s=0.02,
                )
            )
            for name in a.stages:
                before = _files(root)  # per stage: each stage's own writes face the fault once, before its reduce
                if a.kind in ("run", "profile"):
                    for part in a.partition or (None,):
                        ex.run_stage(ctx, name, keys=None if part is None else list(part))
                    if a.kind == "run":
                        # The mount fault, made visible BEFORE the reduce (the 2474b73 case: the lost files of one
                        # job's container are missing for whichever job completes the stage). Some of what the
                        # job wrote never reaches its name; an orphan temp is left for each.
                        for f in sorted(_files(root) - before):
                            unit_file = not f.name.startswith(("_", ".")) and f.name.endswith(_UNIT_FILE_SUFFIXES)
                            if unit_file and rng.random() < loss:
                                tmp = f.with_name(f".tmp-{f.name}.{rng.getrandbits(32):08x}.part")
                                tmp.write_bytes(f.read_bytes())
                                f.unlink()
                                lost += 1
                    ex.try_reduce(ctx, name)
                elif a.kind == "reduce":
                    ex.try_reduce(ctx, name)
            jobs.append(
                types.SimpleNamespace(
                    id=f"j{len(jobs)}",
                    labels={"stage": "-".join(a.stages), "kind": a.kind},
                    status=types.SimpleNamespace(stage="COMPLETED"),
                )
            )
    raise AssertionError(f"the pipeline did not finish within {max_steps} controller steps")


def test_a_lossy_store_completes_every_unit_through_plan_launch_resume(tmp_path):
    lost, reports = _drive(tmp_path / "lossy", loss=0.1)
    clean_lost, clean_reports = _drive(tmp_path / "clean", loss=0.0)
    assert lost > 0 and clean_lost == 0  # non-vacuous: the fault really fired
    assert reports and not clean_reports  # the loss is named, never silently absorbed
    clean = outputs(tmp_path / "clean")
    lossy = outputs(tmp_path / "lossy")
    assert clean and {k: lossy.get(k) for k in clean} == clean  # every output byte-identical to the loss-free run
