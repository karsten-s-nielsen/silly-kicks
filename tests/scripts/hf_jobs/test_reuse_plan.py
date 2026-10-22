"""S5.10 on toys: an attested token is reused (no replica, one canary unit); an unattested token runs main +
replica; a mismatch quarantines and re-plans. The census read-through case lands with CanaryReadThroughStore."""

from __future__ import annotations

import pytest


def test_canary_key_is_deterministic_per_head_and_moves_with_head():
    from scripts.hf_jobs import campaign as C

    units = [f"u{i}" for i in range(50)]
    a = C.canary_key(units, "tok", "a" * 40)
    assert a == C.canary_key(units, "tok", "a" * 40) and a in units
    assert any(C.canary_key(units, "tok", h * 40) != a for h in "bcdef")


HEAD = "f" * 40


def _two_stage(reduced: bool):
    from tests.scripts.hf_jobs.test_campaign import _plan, _st

    return _plan(
        [
            _st("a", n=50, reduced=reduced, done=50 if reduced else 0),
            _st("b", n=1, needs=("a",), reduced=reduced, done=1 if reduced else 0, ready=reduced),
        ]
    )


def test_an_attested_stage_skips_the_replica_and_runs_one_canary_unit():
    from scripts.hf_jobs import campaign as C

    att = {("a", "t_a"), ("b", "t_b")}
    replica, _ = C.plan_for_role(_two_stage(False), "replica", att, HEAD)
    assert all(s["reduced"] for s in replica["stages"])  # nothing to replicate
    _, canary = C.plan_for_role(_two_stage(False), "canary", att, HEAD)
    units_a = [f"u{i}" for i in range(50)]
    assert canary == {"a": [C.canary_key(units_a, "t_a", HEAD)], "b": ["u0"]}  # a single-unit stage: the whole stage


def test_an_unattested_stage_runs_in_main_and_replica_and_gets_no_canary():
    from scripts.hf_jobs import campaign as C

    replica, _ = C.plan_for_role(_two_stage(False), "replica", set(), HEAD)
    assert not any(s["reduced"] for s in replica["stages"])
    _, canary = C.plan_for_role(_two_stage(False), "canary", set(), HEAD)
    assert canary == {}


def test_a_canary_mismatch_quarantines_and_the_next_plan_recomputes():
    from scripts.hf_jobs import campaign as C

    calls = []
    quarantined = C.canary_verdict(main_bytes=b"same", canary_bytes=b"diff", quarantine=lambda: calls.append(1))
    assert quarantined is False and calls == [1]
    replica, _ = C.plan_for_role(_two_stage(False), "replica", {("b", "t_b")}, HEAD)  # "a" lost its attestation
    assert [s["reduced"] for s in replica["stages"]] == [False, True]


def test_the_read_through_store_serves_main_only_when_allowed_and_writes_local(tmp_path):
    from scripts._stages import LocalStore
    from scripts.hf_jobs.campaign import CanaryReadThroughStore

    main = LocalStore(tmp_path / "main")
    local = LocalStore(tmp_path / "canary")
    # A realistic multi-segment generation (pipeline/stage/token), so the first-segment bug (CCTP-IMPL-29) is
    # caught: both gens share the pipeline segment "p", so a first-segment match would wrongly serve genB too.
    main.write_bytes("p/a/t1/u0.json", b"main-A")
    main.write_bytes("p/b/t2/u0.json", b"main-B")
    rt = CanaryReadThroughStore(local, main, allow={"p/a/t1"})
    assert rt.read_bytes("p/a/t1/u0.json") == b"main-A"  # allowed generation -> read-through from main
    assert rt.read_bytes("p/b/t2/u0.json") is None  # same pipeline segment, NOT allowed -> no read-through
    rt.write_bytes("p/c/t3/u0.json", b"canary-C")
    assert local.read_bytes("p/c/t3/u0.json") == b"canary-C" and main.read_bytes("p/c/t3/u0.json") is None
    local.write_bytes("p/a/t1/u0.json", b"local-A")
    assert rt.read_bytes("p/a/t1/u0.json") == b"local-A"  # local shadows main
    assert rt.names("p/a/t1") == {"u0.json"}  # allowed -> union(local, main)
    assert rt.names("p/b/t2") == set()  # not allowed -> local only


class _MemStore:
    """An in-memory stand-in for ``BucketStore`` (flat path -> bytes) for the attestation/canary passes."""

    def __init__(self, files=None):
        self.files = dict(files or {})

    def read_bytes(self, rel):
        return self.files.get(rel)

    def upload_bytes(self, rel, data):
        self.files[rel] = data

    def write_bytes(self, rel, data):
        self.files[rel] = data

    def list_tree(self, rel_dir):
        base = rel_dir.rstrip("/") + "/"
        return {k[len(base) :] for k in self.files if k.startswith(base)}

    def names(self, rel_dir):
        base = rel_dir.rstrip("/") + "/"
        return {k[len(base) :] for k in self.files if k.startswith(base) and "/" not in k[len(base) :]}

    def quarantine_generation(self, pipeline, stage, token, record):
        moved = [k for k in list(self.files) if k.startswith(f"{pipeline}/{stage}/{token}/")]
        for k in moved:
            del self.files[k]
        return moved


def test_generation_diff_compares_results_and_excludes_run_history():
    from scripts.hf_jobs import adopt as A

    main = _MemStore({"g/u0.json": b"r", "g/_inputs.json": b"i", "g/u0.ledger.json": b"m", "g/.corrupt/x": b"z"})
    rep = _MemStore({"g/u0.json": b"r", "g/_inputs.json": b"i", "g/u0.ledger.json": b"DIFF"})
    assert A.generation_diff(main, rep, "g") == []  # ledger + .corrupt excluded; results identical
    rep.files["g/u0.json"] = b"CHANGED"
    assert A.generation_diff(main, rep, "g") == ["g/u0.json: bytes differ"]
    del rep.files["g/_inputs.json"]
    assert "g/_inputs.json: in main only" in A.generation_diff(main, rep, "g")


def _one_job_campaign():
    from scripts.hf_jobs import campaign as C

    return C.Campaign(
        name="cycle1", bucket="b/x", target_minutes=60, concurrency_cap=10, attempts=3, jobs=(), path=None, sha256="h"
    )


def _wire_stores(monkeypatch, stores: dict):
    import types

    from scripts.hf_jobs import campaign as C
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import store as S

    def _inst(job, tier, variant, rk):
        return C.Instance(job, tier, variant, "scripts.train_ghost_gk", (), (), rk)

    def instances(c, commit, *, replica=None, canary=False):
        rk = C.run_key(c, commit, "ghost_gk", "owner", replica=replica, canary=canary)
        return [_inst("ghost_gk", "owner", "default", rk)]

    monkeypatch.setattr(C, "instances", instances)
    monkeypatch.setattr(
        C,
        "read_plan",
        lambda inst, bucket: {
            "pipeline": "ghost_gk",
            "stages": [{"name": "extract", "generation": "gen_e", "token": "t_e"}],
        },
    )
    monkeypatch.setattr(S, "BucketStore", lambda bucket, rk, api=None: stores[rk])

    # This campaign's jobs (B#45-B): `_attestation_pass` scopes the attestation env to these run keys' labels.
    def _fake_job(rk):
        return types.SimpleNamespace(
            id=f"j-{rk}",
            labels={
                "campaign": "cycle1",
                "commit": "f" * 40,
                "run_key": rk,
                "job": "ghost_gk",
                "variant": "default",
                "stage": "extract",
                "shard": "0of1",
            },
        )

    monkeypatch.setattr(
        L,
        "jobs_for",
        lambda api, *, campaign, commit: [_fake_job(rk) for rk in stores],  # every wired run key
    )


def test_the_attestation_pass_writes_an_attestation_only_for_a_byte_identical_generation(monkeypatch):
    from scripts.hf_jobs import adopt as A

    c = _one_job_campaign()
    main = _MemStore({"gen_e/u0.json": b"r", "gen_e/u0.ledger.json": b"m"})
    rep = _MemStore({"gen_e/u0.json": b"r", "gen_e/u0.ledger.json": b"DIFF"})
    # The phase record's label is the in-job label (`label_of`): job.variant.stage.shard. The attestation env is
    # scoped to THIS campaign's jobs (B#45-B), whose label_of must match this record for its env to be used.
    main.files["_ledger/_jobs/ghost_gk.default.extract.0of1.a1.json"] = (
        b'{"label": "ghost_gk.default.extract.0of1", "attempt": "a1", "phases": {},'
        b' "env": {"platform": "Linux-x", "tree_state": "clean"}}'
    )
    # The replica job records its own provenance too; both sides must carry a tree_state to attest (B#32).
    rep.files["_ledger/_jobs/ghost_gk.default.extract.0of1.a1.json"] = (
        b'{"label": "ghost_gk.default.extract.0of1", "attempt": "a1", "phases": {},'
        b' "env": {"platform": "Linux-x", "tree_state": "clean"}}'
    )
    stores = {C_runkey(c, replica=None): main, C_runkey(c, replica=1): rep}
    _wire_stores(monkeypatch, stores)
    out = A._attestation_pass(c, "f" * 40, api=None, with_replica=1, now="2026-10-22T00:00:00+00:00")
    assert out["ghost_gk-owner/default"]["extract"]["identical"] is True
    import json

    rec = json.loads(main.files["_attest/extract/t_e/" + "f" * 40 + ".json"])
    assert rec["main"]["env"]["platform"] == "Linux-x" and rec["main"]["tree_state"] == "clean"
    # a byte difference in a RESULT blocks the attestation
    rep.files["gen_e/u0.json"] = b"CHANGED"
    stores2 = {C_runkey(c, replica=None): _MemStore(main.files), C_runkey(c, replica=1): rep}
    _wire_stores(monkeypatch, stores2)
    out2 = A._attestation_pass(c, "f" * 40, api=None, with_replica=1, now="n")
    assert out2["ghost_gk-owner/default"]["extract"]["identical"] is False


def test_the_attestation_pass_never_attests_an_empty_generation(monkeypatch):
    # B#38 regression: an empty-vs-empty compare is vacuously identical; never attest a generation that never ran.
    from scripts.hf_jobs import adopt as A

    c = _one_job_campaign()
    main, rep = _MemStore(), _MemStore()
    stores = {C_runkey(c, replica=None): main, C_runkey(c, replica=1): rep}
    _wire_stores(monkeypatch, stores)
    out = A._attestation_pass(c, "f" * 40, api=None, with_replica=1, now="n")
    assert out["ghost_gk-owner/default"]["extract"]["materialized"] is False  # never ran -> not a comparison
    assert not any(k.startswith("_attest/") for k in main.files)


def test_the_attestation_pass_refuses_unknown_provenance(monkeypatch):
    # B#32: a missing tree_state means `phases._env` returned {} (env_provenance unreadable) -> UNKNOWN
    # provenance. `_provenance.py` treats unknown as untrustworthy, never clean. The attestation must be
    # refused (no reuse token); else `stamp_provenance` later stamps run_tree_dirty=False on a committed
    # artifact -- a false clean claim. The determinism compare still counts (the bytes matched).
    from scripts.hf_jobs import adopt as A

    c = _one_job_campaign()
    main = _MemStore({"gen_e/u0.json": b"r"})
    rep = _MemStore({"gen_e/u0.json": b"r"})  # byte-identical: determinism holds
    # a phase record with an EMPTY env -- the swallowed-exception case `phases._env` returns {}
    main.files["_ledger/_jobs/ghost_gk.default.extract.0of1.a1.json"] = (
        b'{"label": "ghost_gk.default.extract.0of1", "attempt": "a1", "phases": {}, "env": {}}'
    )
    stores = {C_runkey(c, replica=None): main, C_runkey(c, replica=1): rep}
    _wire_stores(monkeypatch, stores)
    out = A._attestation_pass(c, "f" * 40, api=None, with_replica=1, now="n")
    row = out["ghost_gk-owner/default"]["extract"]
    assert row["materialized"] is True and row["identical"] is True  # the bytes matched
    assert row["attested"] is False and "unknown provenance" in row["reason"]  # but provenance unknown
    assert not any(k.startswith("_attest/") for k in main.files)  # no false-clean attestation written


def test_an_attested_reuse_skip_is_not_a_comparison(monkeypatch):
    # B#47: main holds the generation, the replica skipped it (attested reuse) -> NOT a determinism mismatch.
    # Removing this branch would fail bar 6 on every reuse campaign (the r3 blocker).
    from scripts.hf_jobs import adopt as A

    c = _one_job_campaign()
    main, rep = _MemStore({"gen_e/u0.json": b"r"}), _MemStore()
    stores = {C_runkey(c, replica=None): main, C_runkey(c, replica=1): rep}
    _wire_stores(monkeypatch, stores)
    out = A._attestation_pass(c, "f" * 40, api=None, with_replica=1, now="n")
    row = out["ghost_gk-owner/default"]["extract"]
    assert row["materialized"] is False and "attested reuse" in row["reason"]
    assert not any(k.startswith("_attest/") for k in main.files)
    assert A._comparison_map({"attestations": out}, 1) == {}  # no replica|... entry


def test_comparison_map_excludes_unmaterialized_and_keys_the_replica_number():
    # B#47: a stage that materialized only one side (an attested-reuse skip) is NOT a compare and is excluded.
    # B#48: a replica compare carries its replica number, so merging a `--with-replica 2` match never clobbers
    # a `--with-replica 1` mismatch.
    from scripts.hf_jobs import adopt as A

    report = {
        "attestations": {
            "i1": {
                "extract": {"materialized": True, "identical": True},  # a real compare
                "assemble": {"materialized": False, "reason": "replica skipped (attested reuse)"},  # B#47
            }
        },
        "canary": {"i1": {"score": {"ok": True}}},
    }
    assert A._comparison_map(report, 1) == {"replica|1|i1|extract": True, "canary|i1|score": True}
    mismatch1 = A._comparison_map({"attestations": {"i1": {"extract": {"materialized": True, "identical": False}}}}, 1)
    match2 = A._comparison_map({"attestations": {"i1": {"extract": {"materialized": True, "identical": True}}}}, 2)
    merged = {**mismatch1, **match2}  # as `_merge_comparisons` merges successive verify calls
    assert merged["replica|1|i1|extract"] is False and merged["replica|2|i1|extract"] is True


def test_comparisons_accumulate_across_verify_calls(monkeypatch):
    # B#35b regression: a later `verify --canary` must NOT clobber an earlier `verify --with-replica` mismatch;
    # comparisons are keyed by kind|instance|stage and merged.
    from scripts.hf_jobs import adopt as A
    from scripts.hf_jobs import store as S

    c = _one_job_campaign()
    root = _MemStore()
    monkeypatch.setattr(S, "BucketStore", lambda bucket, rk, api=None: root)
    A._merge_comparisons(c, "f" * 40, {"replica|i/v|a": False}, api=None)  # a replica mismatch
    A._merge_comparisons(c, "f" * 40, {"canary|i/v|a": True}, api=None)  # a later canary-only verify
    assert sorted(A.read_comparisons(c, "f" * 40, api=None)) == [False, True]  # mismatch preserved


def test_the_canary_pass_quarantines_main_on_a_mismatch(monkeypatch):
    from scripts.hf_jobs import adopt as A

    c = _one_job_campaign()
    main = _MemStore({"ghost_gk/extract/t_e/x": b"keep", "gen_e/u0.json": b"main"})
    canary = _MemStore({"gen_e/u0.json": b"DIFF"})
    stores = {C_runkey(c, replica=None): main, C_runkey(c, canary=True): canary}
    _wire_stores(monkeypatch, stores)
    _out, findings = A._canary_pass(c, "f" * 40, api=None)
    assert any("unsound_reuse" in f for f in findings)
    assert "ghost_gk/extract/t_e/x" not in main.files  # the main generation was quarantined


def C_runkey(c, *, replica=None, canary=False):
    from scripts.hf_jobs import campaign as C

    return C.run_key(c, "f" * 40, "ghost_gk", "owner", replica=replica, canary=canary)


def test_schedule_canary_launches_one_unit_per_attested_stage(monkeypatch):
    import datetime

    from scripts.hf_jobs import adopt as A
    from scripts.hf_jobs import campaign as C
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import store as S
    from scripts.hf_jobs.planner import Constraints

    c = _one_job_campaign()
    main = _MemStore({"_attest/extract/t_e/anc.json": b'{"commit": "anc"}'})
    canary = _MemStore()
    stores = {C_runkey(c, replica=None): main, C_runkey(c, canary=True): canary}
    main_plan = {
        "pipeline": "ghost_gk",
        "stages": [
            {"name": "extract", "generation": "gen_e", "token": "t_e", "units": ["u0", "u1", "u2"], "n_units": 3}
        ],
    }
    monkeypatch.setattr(
        C,
        "instances",
        lambda c, commit, *, replica=None, canary=False: [
            C.Instance(
                "ghost_gk",
                "owner",
                "default",
                "scripts.train_ghost_gk",
                (),
                (),
                C.run_key(c, commit, "ghost_gk", "owner", canary=canary),
                role="canary" if canary else "main",
            )
        ],
    )
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: main_plan)
    monkeypatch.setattr(S, "BucketStore", lambda bucket, rk, api=None: stores[rk])
    monkeypatch.setattr(A, "is_ancestor", lambda commit, head, **k: True)
    monkeypatch.setattr(C, "planning_constraints", lambda c: Constraints(target_s=3600.0, concurrency_cap=10))
    launched = []
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "canary-job")
    now = datetime.datetime(2026, 10, 22, tzinfo=datetime.timezone.utc)
    left = C._schedule_canary(c, "f" * 40, api=None, token="hf_x", budget=5, report={}, now=now)
    assert left == 4 and len(launched) == 1
    spec = launched[0]
    assert spec.read_through == "/scratch/ghost_gk-owner"
    assert spec.stages == ("extract",) and spec.shard_keys_path and spec.read_through_allow
    wanted = C.canary_key(["u0", "u1", "u2"], "t_e", "f" * 40)
    import json as _json

    rk = C_runkey(c, canary=True)
    sk_rel = spec.shard_keys_path[len(f"/scratch/{rk}/") :]
    assert _json.loads(canary.files[sk_rel]) == [wanted]


def test_schedule_canary_allows_a_canaried_stages_attested_upstream(monkeypatch):
    # B#29/#45-A regression: in an attested CHAIN a->b (both canaried), stage b must still read a's reused
    # generation from main. The per-job allow = attested gens minus the stage's OWN; a whole-instance
    # "stages not in canary_units" set leaves b's allow empty and quarantines a sound generation.
    import datetime
    import json as _json

    from scripts.hf_jobs import adopt as A
    from scripts.hf_jobs import campaign as C
    from scripts.hf_jobs import launch as L
    from scripts.hf_jobs import store as S
    from scripts.hf_jobs.planner import Constraints

    c = _one_job_campaign()
    main = _MemStore({"_attest/a/t_a/anc.json": b'{"commit": "anc"}', "_attest/b/t_b/anc.json": b'{"commit": "anc"}'})
    canary = _MemStore()
    stores = {C_runkey(c, replica=None): main, C_runkey(c, canary=True): canary}
    main_plan = {
        "pipeline": "ghost_gk",
        "stages": [
            {"name": "a", "generation": "p/a/t_a", "token": "t_a", "units": ["u0", "u1", "u2"], "n_units": 3},
            {"name": "b", "generation": "p/b/t_b", "token": "t_b", "units": ["u0"], "n_units": 1, "needs": ["a"]},
        ],
    }
    monkeypatch.setattr(
        C,
        "instances",
        lambda c, commit, *, replica=None, canary=False: [
            C.Instance(
                "ghost_gk",
                "owner",
                "default",
                "scripts.train_ghost_gk",
                (),
                (),
                C.run_key(c, commit, "ghost_gk", "owner", canary=canary),
                role="canary" if canary else "main",
            )
        ],
    )
    monkeypatch.setattr(C, "read_plan", lambda inst, bucket: main_plan)
    monkeypatch.setattr(S, "BucketStore", lambda bucket, rk, api=None: stores[rk])
    monkeypatch.setattr(A, "is_ancestor", lambda commit, head, **k: True)
    monkeypatch.setattr(C, "planning_constraints", lambda c: Constraints(target_s=3600.0, concurrency_cap=10))
    launched = []
    monkeypatch.setattr(L, "launch", lambda spec, *, api, token: launched.append(spec) or "canary-job")
    now = datetime.datetime(2026, 10, 22, tzinfo=datetime.timezone.utc)
    C._schedule_canary(c, "f" * 40, api=None, token="hf_x", budget=5, report={}, now=now)
    rk = C_runkey(c, canary=True)
    allow_by_stage = {
        spec.stages[0]: _json.loads(canary.files[spec.read_through_allow[len(f"/scratch/{rk}/") :]])
        for spec in launched
    }
    assert set(allow_by_stage) == {"a", "b"}
    assert "p/a/t_a" in allow_by_stage["b"] and "p/b/t_b" not in allow_by_stage["b"]  # b reads a; never its own
    assert "p/a/t_a" not in allow_by_stage["a"]  # a never reads its own generation


def test_jobspec_read_through_adds_the_canary_argv():
    import dataclasses

    from scripts.hf_jobs.launch import _run_argv
    from tests.scripts.hf_jobs.test_launch_typed import _spec

    spec = dataclasses.replace(
        _spec(),
        shard_keys_path="/scratch/rk/sk.json",
        read_through="/scratch/main",
        read_through_allow="/scratch/rk/a.json",
    )
    argv = _run_argv(spec)
    assert "--read-through" in argv and argv[argv.index("--read-through") + 1] == "/scratch/main"
    assert argv[argv.index("--read-through-allow") + 1] == "/scratch/rk/a.json"


@pytest.mark.slow
def test_the_canary_runs_over_the_real_census_graph(tmp_path):
    import importlib
    import os

    from scripts._stages import (
        ItemStage,
        LocalStore,
        StageIncompleteError,
        make_context,
        read_result,
        run_unit,
        sorted_units,
    )
    from scripts._toy_corpora import TOY_PARAMS, toy_session
    from scripts.hf_jobs.campaign import CanaryReadThroughStore, canary_key
    from tests.scripts._stage_harness import PROV, params_from_args, run_in_process

    mod = "scripts.build_territory_ranking_census"
    pipe = importlib.import_module(mod).PIPELINE
    args = list(TOY_PARAMS[mod])
    with toy_session(mod, str(tmp_path / "d")) as (data, extra_env):
        os.environ.update(extra_env)
        try:
            main = run_in_process(pipe, args, store=tmp_path / "main", data_root=data)
        except StageIncompleteError as e:
            pytest.skip(f"census toy fit unavailable in this env (CI runs it): {e}")
        canary_store = CanaryReadThroughStore(LocalStore(tmp_path / "canary"), main.store, allow=set())
        ctx = make_context(pipe, params_from_args(pipe, args), PROV, store=canary_store, data_root=data)
        recompute = {"fit_prepass", "prefit", "score"}  # the canary re-runs these; everything else is reused
        canary_store.allow = {ctx.generation(s.name) for s in pipe.stages if s.name not in recompute}
        for name in recompute:
            st = pipe.stage(name)
            units = sorted_units(ctx, st)  # reads `refs` through the main store
            key = canary_key(units, ctx.tokens[name], HEAD)
            refs = st.ref_map(ctx) if isinstance(st, ItemStage) else {}
            run_unit(ctx, st, key, worker="canary", ref=refs.get(key))
            gen = ctx.generation(name)
            assert read_result(LocalStore(tmp_path / "canary"), f"{gen}/{key}") is not None
            sfx = next(
                x for x in (".parquet", ".json", ".fileset.json") if (tmp_path / "main" / gen / f"{key}{x}").is_file()
            )
            assert (tmp_path / "canary" / gen / f"{key}{sfx}").read_bytes() == (
                tmp_path / "main" / gen / f"{key}{sfx}"
            ).read_bytes()
