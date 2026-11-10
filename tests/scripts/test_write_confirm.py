"""A lost rename on the HF bucket mount is repaired in the job (HFU spec §12, Change 5, owner decision 2026-11-05 2c).

Task 8 B and probes r2/r3 (12 cpu-upgrade jobs): the mount loses ~0.3-0.4% of `os.replace` renames. The writer's own
`stat` shows the final at once, so no immediate check sees the loss; the container's listing shows it after <= 5 s,
and the `.tmp` part still holds the bytes, so one more `os.replace` repairs it. Some listings also miss a final for
one pass only, with no `.tmp` part (a flicker): that is re-checked, never repaired and never an error.

The tests simulate the mount: a "lost" rename is the final moved back to its journaled temp name.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from scripts import _write_confirm as wc
from scripts._driver import _write_bytes_atomically

DEADLINE_ENV = "SILLY_KICKS_JOB_DEADLINE_EPOCH"


@pytest.fixture(autouse=True)
def _empty_journal():
    wc.clear()
    yield
    wc.clear()


def _lose(final: pathlib.Path) -> pathlib.Path:
    """What the mount does: the final vanishes, its temp part keeps the bytes."""
    tmp = wc.journal()[final]
    os.replace(final, tmp)
    return tmp


class _Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, s: float) -> None:
        self.calls.append(s)


def test_nothing_lost_settles_once_and_clears_the_journal(tmp_path):
    _write_bytes_atomically(tmp_path / "a.parquet", b"aaa")
    assert set(wc.journal()) == {tmp_path / "a.parquet"}
    sleeps = _Sleeps()
    assert wc.confirm_writes(settle_s=7, sleep=sleeps) == 0
    assert sleeps.calls == [7]
    assert wc.journal() == {}


def test_a_lost_rename_is_repaired_from_its_temp_part(tmp_path, capsys):
    final = tmp_path / "k.sha256"
    _write_bytes_atomically(final, b"digest\n")
    _write_bytes_atomically(tmp_path / "other.parquet", b"x")
    tmp = _lose(final)
    assert not final.exists() and tmp.exists()

    assert wc.confirm_writes(settle_s=0, sleep=_Sleeps()) == 1
    assert final.read_bytes() == b"digest\n"
    assert not tmp.exists()
    assert wc.journal() == {}
    assert "[confirm] repaired 1 lost rename(s)" in capsys.readouterr().err


def test_a_loss_that_does_not_repair_raises_naming_the_path(tmp_path, monkeypatch, capsys):
    final = tmp_path / "k.parquet"
    _write_bytes_atomically(final, b"p")
    _lose(final)
    monkeypatch.setattr(wc.os, "replace", lambda src, dst: None)  # the mount loses every repair too
    sleeps = _Sleeps()
    with pytest.raises(wc.LostWriteError, match=r"k\.parquet"):
        wc.confirm_writes(settle_s=1, rounds=3, sleep=sleeps)
    assert len(sleeps.calls) == 4  # rounds + 1 settled listings
    # One greppable log line per lost path (impl review B IMPL-05: a lost ledger.json alone relaunches nothing).
    assert f"[confirm] LOST {final}" in capsys.readouterr().err


def test_the_deadline_variable_is_the_one_liveness_uses():
    from scripts.hf_jobs import liveness

    assert wc._DEADLINE_ENV == liveness.DEADLINE_ENV == DEADLINE_ENV  # one name, two modules


def test_a_lost_exclusion_marker_rename_is_repaired(tmp_path):
    # `write_exclusion` had its own temp + os.replace, outside the journal.
    from scripts._driver import exclusion_path, read_exclusion, write_exclusion
    from scripts._item_outcome import ItemExcluded

    gen = tmp_path / "gen"
    gen.mkdir()
    write_exclusion(gen, "k", ItemExcluded("no events"))
    marker = exclusion_path(gen, "k")
    assert set(wc.journal()) == {marker}
    _lose(marker)
    assert read_exclusion(gen, "k") is None
    assert wc.confirm_writes(settle_s=0, sleep=_Sleeps()) == 1
    marker_payload = read_exclusion(gen, "k")
    assert marker_payload is not None and marker_payload["reason"] == "no events"


def test_a_listing_flicker_is_rechecked_not_repaired(tmp_path):
    final = tmp_path / "k.ledger.json"
    _write_bytes_atomically(final, b"{}")
    seen: list[int] = []

    def flicker(d: pathlib.Path) -> set[str]:
        seen.append(1)
        names = set(os.listdir(d))
        return names - {final.name} if len(seen) == 1 else names  # missing from the FIRST listing only

    assert wc.confirm_writes(settle_s=0, sleep=_Sleeps(), list_names=flicker) == 0
    assert len(seen) == 2
    assert final.read_bytes() == b"{}"


def test_a_final_missing_without_its_temp_for_every_pass_raises(tmp_path):
    final = tmp_path / "k.parquet"
    _write_bytes_atomically(final, b"p")
    final.unlink()  # gone, and no temp part to repair from (and nobody forgot it)
    with pytest.raises(wc.LostWriteError, match=r"k\.parquet"):
        wc.confirm_writes(settle_s=0, sleep=_Sleeps())


def test_a_deleted_file_is_forgotten(tmp_path):
    from scripts._stages import LocalStore

    store = LocalStore(tmp_path)
    store.write_bytes("g/k.failed.json", b"{}")
    store.delete("g/k.failed.json")
    assert wc.journal() == {}
    assert wc.confirm_writes(settle_s=0, sleep=_Sleeps()) == 0


def test_settle_time_is_30_s_in_an_hf_job_and_0_elsewhere():
    assert wc.settle_s_for({DEADLINE_ENV: "1900000000"}) == wc.JOB_SETTLE_S == 30
    assert wc.settle_s_for({}) == 0


def test_confirm_job_writes_with_an_empty_journal_returns_at_once(monkeypatch):
    sleeps = _Sleeps()
    monkeypatch.setattr(wc.time, "sleep", sleeps)
    assert wc.confirm_job_writes(env={DEADLINE_ENV: "1900000000"}) == 0
    assert sleeps.calls == []


# ---- the entry points ---------------------------------------------------------------------------------


@pytest.fixture
def toy_ctx(tmp_path):
    from scripts._stages import LocalStore, context_to_json, make_context
    from tests.scripts._stage_harness import PROV
    from tests.scripts._toy_pipeline import PIPELINE

    ctx = make_context(
        PIPELINE,
        {"n_items": 12, "fail_keys": "", "crash_once": ""},
        PROV,
        store=LocalStore(tmp_path / "store"),
        data_root=None,
    )
    path = tmp_path / "ctx.json"
    path.write_text(context_to_json(ctx), encoding="utf-8")
    return ctx, path


def _worker_argv(tmp_path, ctx_path, stage="env", pending=("only",)):
    (tmp_path / "pending.json").write_text(json.dumps(list(pending)), encoding="utf-8")
    for d in ("claims", "outcomes"):
        (tmp_path / d).mkdir(exist_ok=True)
    return [
        "--ctx",
        str(ctx_path),
        "--stage",
        stage,
        "--pending",
        str(tmp_path / "pending.json"),
        "--claims",
        str(tmp_path / "claims"),
        "--outcomes",
        str(tmp_path / "outcomes"),
        "--worker-id",
        "w0",
    ]


def test_worker_confirms_its_writes_after_its_units(tmp_path, toy_ctx, monkeypatch):
    from scripts.hf_jobs import executor

    calls: list[int] = []
    monkeypatch.setattr(wc, "confirm_job_writes", lambda **k: calls.append(len(wc.journal())) or 0)
    assert executor.worker_main(_worker_argv(tmp_path, toy_ctx[1])) == 0
    assert calls and calls[0] > 0  # called once, AFTER the unit wrote its result


def test_a_worker_whose_confirm_raises_fails(tmp_path, toy_ctx, monkeypatch):
    from scripts.hf_jobs import executor

    def boom(**k):
        raise wc.LostWriteError("lost: x")

    monkeypatch.setattr(wc, "confirm_job_writes", boom)
    with pytest.raises(wc.LostWriteError):
        executor.worker_main(_worker_argv(tmp_path, toy_ctx[1]))


def test_reduce_confirms_its_writes(tmp_path, toy_ctx, monkeypatch):
    from scripts._stages import run_unit
    from scripts.hf_jobs import executor

    ctx, ctx_path = toy_ctx
    assert run_unit(ctx, ctx.pipeline.stage("env"), "only", worker="t") == "done"
    wc.clear()  # only the reduce's own writes count below
    calls: list[int] = []
    monkeypatch.setattr(wc, "confirm_job_writes", lambda **k: calls.append(len(wc.journal())) or 0)
    assert executor.reduce_main(["--stage", "env", "--ctx", str(ctx_path)]) == 0
    assert calls and calls[0] > 0


def test_stage_main_confirms_its_writes(tmp_path, monkeypatch):
    from scripts._stages import main
    from tests.scripts._toy_pipeline import PIPELINE

    calls: list[int] = []
    monkeypatch.setattr(wc, "confirm_job_writes", lambda **k: calls.append(1) or 0)
    assert main(PIPELINE, ["--store", str(tmp_path), "--stages", "env", "--workers", "1", "--allow-dirty"]) == 0
    assert calls == [1]
