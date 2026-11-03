"""The job log trail (HFU spec §13.6, F5): every unit prints a start line and an end line with its duration, and every
run exit prints one job summary line. In retrain launch 2 a unit under 300 s left no trace and no unit had a
measured duration; the 2-timeout rule (§13.5) reads these lines to know which units started."""

from __future__ import annotations

import re

import pandas as pd
import pytest

import scripts._stages as S
from scripts._item_outcome import ItemExcluded
from scripts._parallel_launch import RefusalError
from scripts._stages import LocalStore, Param, Pipeline, Stage, int_key, make_context, run_unit
from scripts.hf_jobs import executor as E
from tests.scripts._stage_harness import PROV


def _work(key: str, ctx) -> pd.DataFrame:
    if key == int_key(1):
        raise ValueError("boom")
    if key == int_key(2):
        raise ItemExcluded("no_data", details={})
    return pd.DataFrame({"k": [key]})


def _pipeline() -> Pipeline:
    return Pipeline(
        name="unitlines",
        module=__name__,
        params=(Param("n", int, 3),),
        stages=(
            Stage(
                name="s",
                units=lambda ctx: [int_key(i) for i in range(int(ctx.params["n"]))],
                work=_work,
                reduce=lambda results, ctx: S.combine_frames(list(results.values()), ids=()),
                params=("n",),
            ),
        ),
        output="s",
    )


def test_unit_lines_with_duration(tmp_path, capsys):
    pipeline = _pipeline()
    ctx = make_context(pipeline, {"n": 3}, PROV, store=LocalStore(tmp_path), data_root=None)
    st = pipeline.stage("s")
    for key in (int_key(0), int_key(1), int_key(2)):
        run_unit(ctx, st, key, worker="t")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[unit] ")]
    keys = [int_key(i) for i in range(3)]
    assert lines[0::2] == [f"[unit] s/{k} start" for k in keys], lines
    ends = lines[1::2]
    for k, outcome, ln in zip(keys, ("done", "failed", "excluded"), ends, strict=True):
        assert re.fullmatch(rf"\[unit\] s/{k} {outcome} in \d+\.\d s", ln), ln


@pytest.mark.parametrize(
    ("outcome", "rc"),
    [("ok", 0), ("refused", 2), ("failed", 3)],
)
def test_job_summary_line(tmp_path, monkeypatch, capsys, outcome, rc):
    report = E.StageReport("s", 3, 2, 0, 1, 1, 0, ())

    def fake_run_stage(self, ctx, name, *, shard=(0, 1), keys=None):
        if outcome == "refused":
            raise RefusalError("not enough memory")
        if outcome == "failed":
            exc = E.StageFailedError("the first 3 completed units all failed")
            exc.report = report  # set by the real run_stage (test_systematic_stop_attaches_the_unit_counts)
            raise exc
        return report

    monkeypatch.setattr(E.LocalExecutor, "run_stage", fake_run_stage)
    monkeypatch.setattr(E.LocalExecutor, "try_reduce", lambda self, ctx, name: False)
    got = S.main(_pipeline(), ["--store", str(tmp_path), "--stages", "s", "--shard", "0/1", "--allow-dirty"])
    assert got == rc
    jobs = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[job] ")]
    want = "0/0 done, 0 failed" if outcome == "refused" else "2/3 done, 1 failed"  # exit 3 counts (B IMPL-13)
    assert len(jobs) == 1 and re.fullmatch(rf"\[job\] {want}, exit {rc}, \d+ s", jobs[0]), jobs


def test_systematic_stop_attaches_the_unit_counts(tmp_path, monkeypatch):
    """HFURR-IMPL-13: a StageFailedError out of `run_stage` carries the stage's unit counts from the store."""
    import types

    st = types.SimpleNamespace(needs=(), name="s", inner_parallel=False)
    ctx = types.SimpleNamespace(pipeline=types.SimpleNamespace(stage=lambda name: st, name="p"))
    monkeypatch.setattr(S, "sorted_units", lambda ctx, st: ["0", "1", "2"])
    monkeypatch.setattr(S, "deferred_memory_units", lambda ctx, st, keys: set())
    monkeypatch.setattr(S, "unit_states", lambda ctx, st, keys, verify: {"0": "done", "1": "failed", "2": "pending"})

    def boom(self, ctx, st, keys, *, workers, inner):
        raise E.StageFailedError("the first 1 completed units all failed (systematic)")

    monkeypatch.setattr(E.LocalExecutor, "_run", boom)
    with pytest.raises(E.StageFailedError) as info:
        E.LocalExecutor(E.ExecutorConfig(workers=1)).run_stage(ctx, "s")
    rep = info.value.report
    assert rep is not None and (rep.n_mine, rep.n_done, rep.n_failed) == (3, 1, 1)


def test_plan_prints_no_job_line(tmp_path, capsys):
    assert S.main(_pipeline(), ["--store", str(tmp_path), "--plan"]) == 0
    assert "[job] " not in capsys.readouterr().err
