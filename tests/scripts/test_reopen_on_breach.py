"""Spec r4 D19: a stage that reads the units of a REDUCED upstream reads every unit that upstream's reduce combined,
in a pool, and a missing or corrupt one REOPENS the upstream stage -- it never becomes a smaller corpus.

Readiness and ``reduced`` are both "``_output`` present", so deleting only the broken unit would leave the upstream
reduced and nothing would ever redo it. `_read_units` therefore also deletes the upstream ``_output`` (and
``_reduce.json``) and raises `UnitsInvalidatedError`; `run_unit` re-raises it instead of recording a failure of the
reading unit, and a worker child ends its job the way an invalidated reduce does (spec r3 D6(d)). The toy pipeline's
`gen` is the upstream and `combined` the reader (`tests/scripts/_toy_pipeline.py`).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts import _stages as S
from scripts.hf_jobs.campaign import next_actions
from scripts.hf_jobs.planner import Constraints
from tests.scripts._stage_harness import PROV, params_from_args
from tests.scripts._toy_pipeline import PIPELINE

_VICTIM = "toy__0005"


def _ctx(root, *args):
    return S.make_context(
        PIPELINE, params_from_args(PIPELINE, list(args)), PROV, store=S.LocalStore(root), data_root=None
    )


def _run_gen(ctx, keys=None) -> None:
    """Run `gen` in-process (every unit, or ``keys``) and reduce it, as a job would; the executor's inline reduce
    also writes `_reduce.json`, so the store holds it here too."""
    st = ctx.pipeline.stage("gen")
    refs = st.ref_map(ctx)
    for k in keys or S.sorted_units(ctx, st):
        S.run_unit(ctx, st, k, worker="w0", ref=refs[k])
    S.reduce_stage(ctx, "gen")
    ctx.store.write_bytes(f"{ctx.generation('gen')}/_reduce.json", b'{"stage": "gen", "wall_s": 1.0}\n')


def _output_bytes(ctx, name: str) -> bytes | None:
    gen = ctx.generation(name)
    sfx = S._present_suffix(ctx.store.names(gen), S._OUTPUT)
    return None if sfx is None else ctx.store.read_bytes(f"{gen}/{S._OUTPUT}{sfx}")


def _combined(ctx) -> dict:
    reader = ctx.pipeline.stage("combined")
    assert S.run_unit(ctx, reader, "only", worker="w0") == "done"
    return S.read_result(ctx.store, f"{ctx.generation('combined')}/only")


def _break(store, gen: str, how: str) -> None:
    sfx = S._present_suffix(store.names(gen), _VICTIM)
    assert sfx is not None
    if how == "corrupt":  # the bytes change under their checksum (a truncated write, bit rot)
        store.write_bytes(f"{gen}/{_VICTIM}{sfx}", b"not the unit")
    else:  # the unit is gone after the reduce (a lost write on the mount)
        store.delete(f"{gen}/{_VICTIM}{sfx}")
        store.delete(f"{gen}/{_VICTIM}.sha256")


@pytest.mark.parametrize("how", ["corrupt", "lost"])
def test_a_broken_upstream_unit_reopens_the_reduced_stage(tmp_path, how):
    clean = _ctx(tmp_path / "clean")
    _run_gen(clean)
    want_output, want = _output_bytes(clean, "gen"), _combined(clean)

    ctx = _ctx(tmp_path / "s")
    _run_gen(ctx)
    _break(ctx.store, ctx.generation("gen"), how)
    with pytest.raises(S.UnitsInvalidatedError):
        S.run_unit(ctx, ctx.pipeline.stage("combined"), "only", worker="w0")
    gen, reader = ctx.generation("gen"), ctx.generation("combined")
    # The upstream reopened: its broken unit, its `_output` and its `_reduce.json` are deleted ...
    assert S._present_suffix(ctx.store.names(gen), _VICTIM) is None
    assert _output_bytes(ctx, "gen") is None and ctx.store.read_bytes(f"{gen}/_reduce.json") is None
    # ... and the reading unit spent none of its attempts: no failure record, no ledger.
    assert ctx.store.read_bytes(f"{reader}/only.failed.json") is None
    assert ctx.store.read_bytes(f"{reader}/only.ledger.json") is None
    # The real readiness logic: `gen` is unreduced with exactly the broken unit pending; `combined` waits on it.
    rows = {s["name"]: s for s in S.plan(ctx)["stages"]}
    assert not rows["gen"]["reduced"] and rows["gen"]["pending_units"] == [_VICTIM]
    assert not rows["combined"]["ready"]
    acts = next_actions(S.plan(ctx), set(), {}, Constraints())
    assert ("gen",) in [a.stages for a in acts] and all("combined" not in a.stages for a in acts)
    # The redo: one unit, the same reduce bytes, and then the reader's result equals the clean run's.
    _run_gen(ctx, keys=[_VICTIM])
    assert _output_bytes(ctx, "gen") == want_output
    assert _combined(ctx) == want


def test_a_failure_the_upstream_tolerated_reopens_nothing(tmp_path):
    """The other side of the band (B r3 R4S-r3-01): `gen` tolerates failures (`max_failures=None`), so it reduces
    without a failed unit. The reader expects the units the reduce COMBINED (its `_summary.json`), not every unit
    minus the exclusions, so the tolerated failure is not missing: nothing reopens."""
    ctx = _ctx(tmp_path / "s", "--fail-keys", _VICTIM)
    _run_gen(ctx)
    raw = ctx.store.read_bytes(f"{ctx.generation('gen')}/_summary.json")
    assert raw is not None
    summary = json.loads(raw)
    assert summary["failed_keys"] == [_VICTIM]  # non-vacuity: the reduce really dropped a unit
    got = _combined(ctx)
    units = S.sorted_units(ctx, ctx.pipeline.stage("gen"))
    assert got["keys"] == [k for k in units if k != _VICTIM] == S.combined_units(ctx, "gen")
    assert _output_bytes(ctx, "gen") is not None  # still reduced


def _cli(store, *args):
    cmd = [sys.executable, "-m", "tests.scripts._toy_pipeline", "--store", str(store), "--allow-dirty", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 -- fixed argv


@pytest.mark.slow  # slow: five stage CLI jobs with worker children, measured 18 s on the Windows dev box (ADR-023)
def test_a_worker_that_finds_its_upstream_broken_ends_the_job_cleanly(tmp_path):
    """Spec r4 D19 + r3 D6(d): the reading unit runs in an executor worker child. The child exits with the
    invalidated code instead of dying, the executor relaunches nothing, and the controller-launched stage job ends
    with rc 0 and names the reopen -- an ERROR job would halt the instance, and a relaunch would spend the reader's
    attempts on a stage that cannot be ready."""
    root = tmp_path / "s"
    for stage in ("gen", "combined"):
        assert _cli(root, "--stages", stage, "--workers", "2").returncode == 0
    plan = _cli(root, "--plan")  # the CLI's own generations (its token carries the real code identity)
    gens = {s["name"]: s["generation"] for s in json.loads(plan.stdout)["stages"]}
    store, gen, reader = S.LocalStore(root), gens["gen"], gens["combined"]
    want = S.read_result(store, f"{reader}/only")
    for rel in ("only.json", "only.sha256", f"{S._OUTPUT}.json", f"{S._OUTPUT}.sha256"):
        store.delete(f"{reader}/{rel}")  # the reader runs again, now over the broken upstream
    _break(store, gen, "corrupt")
    p = _cli(root, "--stages", "combined", "--workers", "2")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "UNITS INVALIDATED" in p.stderr and "relaunching worker" not in p.stderr
    assert S._present_suffix(store.names(gen), S._OUTPUT) is None  # `gen` reopened
    assert store.read_bytes(f"{reader}/only.failed.json") is None  # the reader spent no attempt
    # The next jobs redo the unit and the reader; the result equals the first run's.
    for stage in ("gen", "combined"):
        assert _cli(root, "--stages", stage, "--workers", "2").returncode == 0
    assert S.read_result(store, f"{reader}/only") == want
