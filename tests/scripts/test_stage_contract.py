"""ADR-128 stage contract: DAG validation, tokens, checksum-verified results, plan JSON, reduce order."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import _stages as s


def _p(**kw):
    def units(ctx):
        return [s.int_key(i) for i in range(3)]

    stages = kw.pop(
        "stages",
        (
            s.Stage(name="a", units=units, work=lambda k, c: {"k": k}, params=("x",)),
            s.Stage(name="b", units=lambda c: ["only"], work=lambda k, c: {"n": 1}, needs=("a",)),
        ),
    )
    return s.Pipeline(
        name="t", module="tests.scripts._toy_pipeline", stages=stages, params=(s.Param("x", int, 1),), **kw
    )


def _ctx(tmp_path, pipeline=None, params=None, code="c" * 40):
    pipeline = pipeline or _p()
    prov = {"commit": code, "tree_state": "clean", "code_identity": code}
    return s.make_context(pipeline, params or {"x": 1}, prov, store=s.LocalStore(tmp_path), data_root=None)


def test_int_key_orders_numerically():
    assert sorted([s.int_key(i) for i in (10, 2, 1)]) == ["0001", "0002", "0010"]


def test_validate_rejects_a_forward_need():
    bad = (
        s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1, needs=("a",)),
        s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1),
    )
    with pytest.raises(ValueError, match="declared BEFORE"):
        _p(stages=bad).validate()


def test_validate_rejects_an_undeclared_param():
    bad = (s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, params=("nope",)),)
    with pytest.raises(ValueError, match="undeclared param"):
        _p(stages=bad).validate()


def test_tokens_change_with_params_upstream_and_code_but_not_with_unrelated_params(tmp_path):
    p = _p()
    _, t1 = s.compute_tokens(p, {"x": 1}, "c1", None)
    _, t2 = s.compute_tokens(p, {"x": 2}, "c1", None)
    _, t3 = s.compute_tokens(p, {"x": 1}, "c2", None)
    assert t1["a"] != t2["a"] and t1["b"] != t2["b"]  # b changes through its upstream token
    assert t1["a"] != t3["a"]


def test_cross_commit_version_replaces_the_code_identity():
    st = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, cross_commit_version="extract-v1")
    p = _p(stages=(st,))
    assert s.compute_tokens(p, {"x": 1}, "c1", None)[1] == s.compute_tokens(p, {"x": 1}, "c2", None)[1]


def test_dataset_identity_enters_only_dataset_stages():
    a = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True)
    b = s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1)
    p = _p(stages=(a, b))
    _, t1 = s.compute_tokens(p, {"x": 1}, "c", {"name": "d", "revision": "r1"})
    _, t2 = s.compute_tokens(p, {"x": 1}, "c", {"name": "d", "revision": "r2"})
    assert t1["a"] != t2["a"] and t1["b"] == t2["b"]


def test_a_second_dataset_keys_only_the_stage_that_reads_it():
    """A stage reading another dataset parameter (xcross's held-out probe) is keyed by THAT identity --
    and only it: the ``--dataset`` stages do not move when the probe dataset changes, and vice versa."""
    a = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True)
    b = s.Stage(name="b", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True, dataset_param="probe")
    p = _p(stages=(a, b))
    base = {"name": "d", "revision": "r1"}
    _, t1 = s.compute_tokens(p, {"x": 1}, "c", base, dataset_identities={"probe": {"name": "p", "revision": "q1"}})
    _, t2 = s.compute_tokens(p, {"x": 1}, "c", base, dataset_identities={"probe": {"name": "p", "revision": "q2"}})
    _, t3 = s.compute_tokens(
        p, {"x": 1}, "c", {"name": "d", "revision": "r2"}, dataset_identities={"probe": {"name": "p", "revision": "q1"}}
    )
    assert t1["a"] == t2["a"] and t1["b"] != t2["b"]
    assert t1["a"] != t3["a"] and t1["b"] == t3["b"]


def test_a_local_dataset_identity_sees_a_same_size_edit_of_a_small_file(tmp_path):
    """A local (unlocked) dataset is identified by its file listing. Names and sizes alone miss a corrected
    ``home_teams.json`` of the same byte length (``{"100": 1}`` -> ``{"100": 2}``), and that file drives the
    goal-relative flip of every ghost feature -- so small files enter the identity by CONTENT."""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "home_teams.json").write_text('{"100": 1}', encoding="utf-8")
    before = s._dataset_identity({"dataset": "ds"}, tmp_path)
    (root / "home_teams.json").write_text('{"100": 2}', encoding="utf-8")
    after = s._dataset_identity({"dataset": "ds"}, tmp_path)
    assert before is not None and after is not None
    assert before != after
    # Non-vacuity: an identical rewrite keeps the identity (the digest is of content, not of mtime).
    (root / "home_teams.json").write_text('{"100": 2}', encoding="utf-8")
    assert s._dataset_identity({"dataset": "ds"}, tmp_path) == after


def test_a_local_dataset_identity_keys_a_large_file_by_name_and_size_only(tmp_path, monkeypatch):
    """The other side of the band: a file above the content-digest ceiling is never read (a real local
    corpus holds multi-GB parquet), so only its name and size enter the identity."""
    monkeypatch.setattr(s, "_CONTENT_DIGEST_MAX_BYTES", 4)
    root = tmp_path / "ds"
    root.mkdir()
    (root / "big.parquet").write_bytes(b"aaaaaaaa")
    before = s._dataset_identity({"dataset": "ds"}, tmp_path)
    (root / "big.parquet").write_bytes(b"bbbbbbbb")
    assert s._dataset_identity({"dataset": "ds"}, tmp_path) == before
    (root / "big.parquet").write_bytes(b"bbbbbbbbb")
    assert s._dataset_identity({"dataset": "ds"}, tmp_path) != before


@pytest.mark.parametrize(
    "result",
    [pd.DataFrame({"a": [1.5, 2.0]}), {"z": [1, 2.5, None]}, s.FileSet({"model.json": b"{}", "SHA256SUMS": b"x\n"})],
)
def test_results_round_trip_with_a_verified_checksum(tmp_path, result):
    store = s.LocalStore(tmp_path)
    s.write_result(store, "p/st/tok/0001", result)
    back = s.read_result(store, "p/st/tok/0001")
    if isinstance(result, pd.DataFrame):
        pd.testing.assert_frame_equal(back, result)
    elif isinstance(result, s.FileSet):
        assert dict(back.files) == dict(result.files)
    else:
        assert back == result


def test_write_result_is_byte_deterministic(tmp_path):
    a, b = s.LocalStore(tmp_path / "a"), s.LocalStore(tmp_path / "b")
    for st in (a, b):
        s.write_result(st, "g/k", pd.DataFrame({"x": [0.1, 0.2]}))
    assert (tmp_path / "a/g/k.parquet").read_bytes() == (tmp_path / "b/g/k.parquet").read_bytes()


def test_unit_states_detect_a_poisoned_result(tmp_path):
    ctx = _ctx(tmp_path)
    gen = ctx.generation("a")
    s.write_result(ctx.store, f"{gen}/0000", {"k": "0000"})
    (tmp_path / gen / "0000.json").write_bytes(b'{"k":"tampered"}\n')
    assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000"], verify=False) == {"0000": "done"}
    with pytest.warns(UserWarning, match="checksum"):
        assert s.unit_states(ctx, ctx.pipeline.stage("a"), ["0000"], verify=True) == {"0000": "pending"}


def test_reduce_receives_results_in_reduce_order_not_write_order(tmp_path):
    seen = []

    def red(results, ctx):
        seen.extend(results)
        return {"order": list(results)}

    st = s.Stage(name="a", units=lambda c: ["0002", "0000", "0001"], work=lambda k, c: {"k": k}, reduce=red)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    for k in ("0002", "0000", "0001"):  # completion order != key order
        s.run_unit(ctx, st, k, worker="w0")
    s.reduce_stage(ctx, "a")
    assert seen == ["0000", "0001", "0002"]
    assert ctx.upstream("a") == {"order": ["0000", "0001", "0002"]}
    assert json.loads((tmp_path / ctx.generation("a") / "_summary.json").read_text())["n_done"] == 3


def test_reduce_refuses_an_incomplete_stage(tmp_path):
    ctx = _ctx(tmp_path)
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0000", worker="w0")
    with pytest.raises(s.StageIncompleteError):
        s.reduce_stage(ctx, "a")


def test_a_failed_unit_is_recorded_not_raised_and_blocks_a_zero_tolerance_reduce(tmp_path):
    def work(k, c):
        if k == "0001":
            raise ValueError("bad item")
        return {"k": k}

    st = s.Stage(name="a", units=lambda c: ["0000", "0001"], work=work)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    assert s.run_unit(ctx, st, "0000", worker="w0") == "done"
    assert s.run_unit(ctx, st, "0001", worker="w0") == "failed"
    assert "bad item" in (tmp_path / ctx.generation("a") / "0001.failed.json").read_text()
    with pytest.raises(s.StageIncompleteError, match="failed"):
        s.reduce_stage(ctx, "a")


def test_plan_reports_counts_and_unknown_downstream_units(tmp_path):
    ctx = _ctx(tmp_path)
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0000", worker="w0")
    out = s.plan(ctx)
    a, b = out["stages"]
    assert (a["n_units"], a["done"], a["pending"], a["reduced"]) == (3, 1, 2, False)
    assert b["n_units"] == 1  # static units are known even before the upstream reduces
    assert b["ready"] is False


def test_plan_names_the_pending_units_not_just_their_count(tmp_path):
    """Real HF Cycle 1: the controller took the FIRST `pending` keys of `units` (all keys, any state) and so
    handed a job units that were already done; the pending unit was never scheduled and every relaunch was
    a no-op. The plan names the pending keys themselves, in canonical order."""
    ctx = _ctx(tmp_path)
    s.run_unit(ctx, ctx.pipeline.stage("a"), "0000", worker="w0")
    a = s.plan(ctx)["stages"][0]
    assert a["pending_units"] == ["0001", "0002"] and len(a["pending_units"]) == a["pending"]
    assert "0000" in a["units"] and "0000" not in a["pending_units"]


def test_context_json_round_trips(tmp_path):
    ctx = _ctx(tmp_path)
    back = s.context_from_json(s.context_to_json(ctx))
    assert back.tokens == ctx.tokens and back.params == ctx.params and back.pipeline.name == "toy"


def test_keys_may_not_start_with_an_underscore(tmp_path):
    st = s.Stage(name="a", units=lambda c: ["_output"], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    with pytest.raises(ValueError, match="reserved"):
        s.sorted_units(ctx, st)


def test_keys_may_not_contain_a_dot(tmp_path):
    st = s.Stage(name="a", units=lambda c: ["a.counters"], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    with pytest.raises(ValueError, match="outside"):
        s.sorted_units(ctx, st)


def test_validate_refuses_a_dataset_stage_that_declares_no_files():
    """A `download` job fetches exactly what a stage declares; a plain dataset Stage without a declaration
    would read files no job fetched (spec R3). `()` is a declaration: the stage reads no dataset file."""
    bad = s.Stage(name="a", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True)
    with pytest.raises(ValueError, match="dataset_files"):
        _p(stages=(bad,)).validate()
    _p(
        stages=(
            s.Stage(
                name="a", units=lambda c: ["x"], work=lambda k, c: 1, uses_dataset=True, dataset_files=lambda c: ()
            ),
        )
    ).validate()


def test_profile_sample_is_k_evenly_spaced_units_and_empty_for_none():
    keys = [s.int_key(i) for i in range(10)]
    assert s.profile_sample(keys, 4) == ["0000", "0002", "0005", "0007"]
    assert s.profile_sample(keys, 50) == keys
    assert s.profile_sample(keys, None) == ["0000"]
    assert s.profile_sample([], 4) == []


def test_an_unset_second_dataset_has_no_identity(tmp_path):
    """An optional second dataset parameter left unset reads nothing: its identity is None, never a
    digest of the whole data root (which moved the token with every unrelated file under it)."""
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "f.txt").write_text("x", encoding="utf-8")
    assert s._dataset_identity({"probe": None}, tmp_path, param="probe") is None
    assert s._dataset_identity({"dataset": None}, tmp_path) is not None  # the --dataset default is unchanged


def test_plan_names_each_failed_unit_with_its_attempt_count(tmp_path):
    """The controller retries a failed unit while its attempts stay below the campaign limit, so the plan
    carries each failed unit's recorded attempt (`<key>.failed.json`)."""
    st = s.Stage(name="a", units=lambda c: [s.int_key(i) for i in range(3)], work=lambda k, c: 1 / 0)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    s.run_unit(ctx, st, "0001", worker="t")
    s.run_unit(ctx, st, "0001", worker="t")
    rec = s.plan(ctx)["stages"][0]
    assert rec["failed_units"] == {"0001": 2} and rec["pending_units"] == ["0000", "0002"]
    assert rec["max_failures"] == 0  # the stage's declared tolerance travels with its failures


def test_a_unit_killed_for_memory_keeps_its_attempt_count(tmp_path):
    """A unit killed over its memory cap has no ledger record (the kill came before it); its attempt
    lives in `<key>.failed.json`, so the next run counts on from it -- or the retry limit never ends."""
    st = s.Stage(name="a", units=lambda c: [s.int_key(i) for i in range(2)], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    s.record_memory_failure(ctx.store, ctx.generation("a"), "0000", "over its memory cap")
    rec = s.plan(ctx)["stages"][0]
    assert rec["failed_units"] == {"0000": 1} and rec["memory_units"] == ["0000"]
    s.record_memory_failure(ctx.store, ctx.generation("a"), "0000", "over its memory cap")
    assert s.plan(ctx)["stages"][0]["failed_units"] == {"0000": 2}
    s.run_unit(ctx, st, "0000", worker="t")  # now it fits: done, attempt 3 on the ledger
    raw = ctx.store.read_bytes(f"{ctx.generation('a')}/0000.ledger.json")
    assert raw is not None
    led = json.loads(raw)
    assert led["attempt"] == 3 and s.plan(ctx)["stages"][0]["memory_units"] == []


def test_a_mixed_id_column_becomes_the_legacy_strings_and_a_uniform_one_is_untouched():
    """Parquet cannot store a column mixing int and str ids (gradientsports/skillcorner ints with idsse
    `J03WMX`): real HF, ghost_outfield's extract reduce. The legacy combined in memory and grouped on
    `np.array(ids.tolist())` (ghost_gk) or `astype(str)` (ghost_outfield), which give `'10502'`."""
    import numpy as np
    import pandas as pd

    mixed = pd.Series([10502, "J03WMX", 3829], dtype=object)
    out = s.storable_id_column(mixed)
    assert out.tolist() == ["10502", "J03WMX", "3829"] == np.array(mixed.tolist()).tolist()
    ints = pd.Series([10502, 3829])
    assert s.storable_id_column(ints) is ints  # one type: unchanged (dtype and values)
    strs = pd.Series(["J03WMX", "J03WN1"], dtype=object)
    assert s.storable_id_column(strs) is strs


def test_a_memory_failure_below_the_attempt_limit_is_deferred_not_failed(tmp_path):
    """A unit recorded over its memory cap waits for its escalation job (W=1, the whole box): it is
    DEFERRED until it has failed `UNIT_ATTEMPTS` times (Reviewer B CCHC-IMPL-12)."""
    st = s.Stage(name="a", units=lambda c: [s.int_key(i) for i in range(3)], work=lambda k, c: 1)
    ctx = _ctx(tmp_path, _p(stages=(st,)))
    gen = ctx.generation("a")
    s.record_memory_failure(ctx.store, gen, "0001", "over its memory cap")
    assert s.deferred_memory_units(ctx, st, ["0000", "0001", "0002"]) == {"0001"}
    for _ in range(s.UNIT_ATTEMPTS - 1):
        s.record_memory_failure(ctx.store, gen, "0001", "over its memory cap")
    assert s.deferred_memory_units(ctx, st, ["0001"]) == set()  # spent: a real failure now
