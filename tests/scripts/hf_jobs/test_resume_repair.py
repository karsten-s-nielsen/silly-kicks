"""Resume housekeeping and run-history (TF-85): a killed unit's lost ledger is repaired at the reduce (D7),
and every unit records the fit host's platform in its ledger (D3).

Run 1 left 31 done units with a result but no `.ledger.json` (a job killed between the two writes), so their
cost went missing from the history. The repair runs in `reduce_stage`, which sees EVERY unit of the stage -- a
lost-ledger unit is `done` (its result verifies) and so absent from any later job's pending shard (reviewer B
B02). The platform of the fit host lives in the unit ledger (run-history), never in the artifact metadata (D3,
reviewer B B03)."""

from __future__ import annotations

import json

from scripts._stages import (
    LocalStore,
    Pipeline,
    Stage,
    make_context,
    reduce_stage,
    repair_missing_ledgers,
    run_unit,
)

GEN = "win_probability/extract/t"


def test_a_done_unit_with_no_ledger_is_repaired(tmp_path) -> None:
    s = LocalStore(tmp_path)
    s.write_bytes(f"{GEN}/u0.parquet", b"data")  # the result is present and verified...
    s.write_bytes(f"{GEN}/u0.sha256", b"deadbeef\n")  # ...but the kill lost the ledger write that follows it
    assert repair_missing_ledgers(s, GEN, ["u0"], stage="extract") == ["u0"]
    raw = s.read_bytes(f"{GEN}/u0.ledger.json")
    assert raw is not None
    rec = json.loads(raw)
    assert rec["outcome"] == "done" and rec["wall_s"] is None and rec["repaired"] is True and rec["platform"] is None


def test_a_unit_that_already_has_a_ledger_is_left_alone(tmp_path) -> None:
    s = LocalStore(tmp_path)
    s.write_bytes(f"{GEN}/u0.parquet", b"data")
    s.write_bytes(f"{GEN}/u0.ledger.json", json.dumps({"key": "u0", "outcome": "done", "wall_s": 5.0}).encode())
    assert repair_missing_ledgers(s, GEN, ["u0"], stage="extract") == []
    raw = s.read_bytes(f"{GEN}/u0.ledger.json")
    assert raw is not None and json.loads(raw)["wall_s"] == 5.0  # the real record is not overwritten


# ---- run through the real stage runner (the wiring, not just the helper) -------------------------

PROV = {
    "commit": "t" * 40,
    "code_identity": "t" * 40,
    "dirty": False,
    "tree_state": "clean",
    "platform": "Linux-X-6.12",
}


def _ctx(tmp_path):
    pl = Pipeline(
        name="mini",
        module="m",
        stages=(
            Stage(
                name="s",
                units=lambda ctx: ["u0", "u1"],
                work=lambda key, ctx: {"v": key},
                reduce=lambda r, ctx: dict(r),
            ),
        ),
        output="s",
    )
    return make_context(pl, {}, PROV, store=LocalStore(tmp_path), data_root=None)


def test_run_unit_records_the_fit_host_platform_in_the_ledger(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    run_unit(ctx, ctx.pipeline.stage("s"), "u0", worker="w")
    raw = ctx.store.read_bytes(f"{ctx.generation('s')}/u0.ledger.json")
    assert raw is not None
    assert json.loads(raw)["platform"] == "Linux-X-6.12"  # D3: the fit host is run-history, not the artifact


def test_reduce_repairs_a_done_unit_whose_ledger_was_lost(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    st = ctx.pipeline.stage("s")
    for k in ("u0", "u1"):
        run_unit(ctx, st, k, worker="w")
    gen = ctx.generation("s")
    ctx.store.delete(f"{gen}/u1.ledger.json")  # a kill lost u1's ledger between its result and its ledger write
    assert ctx.store.read_bytes(f"{gen}/u1.ledger.json") is None
    reduce_stage(ctx, "s")  # the reduce sees EVERY unit (not just a job's shard) and repairs the lost ledger
    raw = ctx.store.read_bytes(f"{gen}/u1.ledger.json")
    assert raw is not None and json.loads(raw)["repaired"] is True
    other = ctx.store.read_bytes(f"{gen}/u0.ledger.json")
    assert other is not None and json.loads(other).get("repaired") is not True  # the real ledger is untouched


def test_stage_workers_ignores_a_repair_record() -> None:
    """B07/M6: a D7 repair record ran in no job (worker None). Counting its null worker inside a real job's
    label would report 2 outer workers for a 1-worker stage, so resume over-sizes the next launch. Deleting
    the repaired-skip makes this stage report 2 workers (reviewer B mutant M6)."""
    from scripts.hf_jobs.ledger import stage_workers

    real = {"job_label": "L", "worker": 0, "inner_workers": 4}
    repair = {"job_label": "L", "worker": None, "inner_workers": None, "repaired": True}
    outer, inner = stage_workers([real, repair])
    assert outer == 1 and inner == 4


# ---- B14: the repair must never clobber a real ledger that raced in after the snapshot ------------


def test_local_store_write_if_absent_refuses_to_overwrite(tmp_path) -> None:
    """The create-only primitive behind the B14 fix: write when absent, REFUSE (return False, keep the bytes)
    when present -- atomic via O_EXCL, so two writers never clobber each other's record."""
    s = LocalStore(tmp_path)
    assert s.write_if_absent("d/f.json", b"first") is True
    assert s.write_if_absent("d/f.json", b"second") is False
    assert s.read_bytes("d/f.json") == b"first"


class _StaleNamesStore:
    """A store whose directory listing is STALE: it reports no ledgers though one is on disk. It models the B14
    window -- a straggler wrote the real ledger after the reduce took its done-keys snapshot."""

    def __init__(self, inner: LocalStore) -> None:
        self._inner = inner

    def names(self, rel_dir: str) -> set:
        return set()  # stale: the snapshot missed the straggler's just-written ledger

    def read_bytes(self, rel: str):
        return self._inner.read_bytes(rel)

    def write_bytes(self, rel: str, data: bytes) -> None:
        self._inner.write_bytes(rel, data)

    def write_if_absent(self, rel: str, data: bytes) -> bool:
        return self._inner.write_if_absent(rel, data)

    def delete(self, rel: str) -> None:
        self._inner.delete(rel)

    def local_path(self, rel: str):
        return self._inner.local_path(rel)


def test_repair_never_clobbers_a_ledger_that_appeared_after_the_snapshot(tmp_path) -> None:
    """B14: the done-keys snapshot is stale; a straggler writes the REAL ledger between the snapshot and the
    repair write. Repair must create-only, so the real wall survives. An unconditional write lets the null-wall
    repair record clobber it (one unit's measured cost lost)."""
    inner = LocalStore(tmp_path)
    inner.write_bytes(f"{GEN}/u1.parquet", b"data")
    inner.write_bytes(f"{GEN}/u1.ledger.json", json.dumps({"key": "u1", "outcome": "done", "wall_s": 12.5}).encode())
    store = _StaleNamesStore(inner)
    repair_missing_ledgers(store, GEN, ["u1"], stage="extract")
    raw = inner.read_bytes(f"{GEN}/u1.ledger.json")
    assert raw is not None
    kept = json.loads(raw)
    assert kept["wall_s"] == 12.5  # the real ledger (and its measured cost) is preserved, not a null-wall record


def test_write_if_absent_cleanup_keeps_a_file_a_straggler_replaced(tmp_path, monkeypatch) -> None:
    """B16: if the write fails AFTER a straggler os.replace'd a real ledger onto the path, the cleanup must
    unlink only THIS call's own file, never the straggler's. Unlinking by path alone deletes the real ledger
    (reviewer B reproduced ENOSPC on Linux)."""
    import os

    import pytest

    s = LocalStore(tmp_path)
    target = tmp_path / "d" / "f.json"
    real = b'{"wall_s": 12.5}'
    orig_fdopen = os.fdopen

    def racing_fdopen(fd, *a, **k):
        os.close(fd)  # release our handle first so the replace is clean on every OS
        tmp = tmp_path / "d" / ".straggler"
        tmp.write_bytes(real)
        os.replace(tmp, target)  # the straggler installs the real ledger at the path...
        raise OSError(28, "No space left on device")  # ...then our write fails

    monkeypatch.setattr(os, "fdopen", racing_fdopen)
    with pytest.raises(OSError):
        s.write_if_absent("d/f.json", b"null-wall")
    monkeypatch.setattr(os, "fdopen", orig_fdopen)
    assert target.read_bytes() == real  # the real ledger survived; cleanup did not delete it


def test_write_if_absent_removes_its_own_partial_on_write_failure(tmp_path, monkeypatch) -> None:
    """B17: a write that fails with NO straggler leaves no partial -- cleanup removes this call's own file, so a
    later repair can still create it. Skipping the unlink (M12) leaves the empty partial behind and blocks it."""
    import os

    import pytest

    s = LocalStore(tmp_path)
    target = tmp_path / "d" / "f.json"

    def failing_fdopen(fd, *a, **k):
        os.close(fd)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    with pytest.raises(OSError):
        s.write_if_absent("d/f.json", b"null-wall")
    assert not target.exists()  # the own partial was removed, not left to block a later repair
