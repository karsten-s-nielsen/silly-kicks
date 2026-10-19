"""ADR-128 framework gates (spec 5.2): W1N1 vs W2N2 byte identity, non-vacuity, thread pins, resume,
failure rule, checksum resume. Every run goes through the real CLI and real worker subprocesses."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

_MOD = "tests.scripts._toy_pipeline"
_VOLATILE = (".ledger.json", ".failed.json", "_profile.json")


def _cli(store, *args, check=True):
    cmd = [sys.executable, "-m", _MOD, "--store", str(store), "--allow-dirty", *args]
    p = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 -- fixed argv
    if check and p.returncode != 0:
        raise AssertionError(f"{cmd} rc={p.returncode}\n{p.stdout}\n{p.stderr}")
    return p


def _plan(store, *params):
    """The plan JSON. ``params`` must repeat every pipeline parameter the queried stage DECLARES: a
    declared parameter enters the stage token, so a plan without it resolves another generation."""
    return json.loads(_cli(store, "--plan", *params).stdout)


def _outputs(store) -> dict[str, str]:
    root = pathlib.Path(store)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(_VOLATILE)
    }


def _fanned(store, stages=("gen", "folds", "final")):
    for st in stages:
        flag = "--inner-workers" if st == "folds" else "--workers"
        procs = [
            subprocess.Popen(  # noqa: S603 -- two concurrent "jobs"
                [
                    sys.executable,
                    "-m",
                    _MOD,
                    "--store",
                    str(store),
                    "--allow-dirty",
                    "--stages",
                    st,
                    "--shard",
                    f"{i}/2",
                    flag,
                    "2",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for i in range(2)
        ]
        for p in procs:
            _out, err = p.communicate()
            assert p.returncode == 0, err
    # the reduce of the last stage ran in whichever job saw it complete; make sure it exists
    _cli(store, "--stages", stages[-1])


def _ledger(store, stage):
    gen = next(s["generation"] for s in _plan(store)["stages"] if s["name"] == stage)
    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / gen).glob("*.ledger.json")]
    return sorted(recs, key=lambda r: r["finished_ns"])


def _seq(values: dict, order) -> float:
    """Sequential float sum in ``order`` (the toy reduce's arithmetic, order-sensitive by design)."""
    total = 0.0
    for k in order:
        total += values[k]
    return total


@pytest.mark.slow  # slow: measured 46 s on the Windows dev box (ADR-023)
def test_w1n1_and_w2n2_are_byte_identical_and_non_vacuous(tmp_path):
    serial, fanned = tmp_path / "serial", tmp_path / "fanned"
    _cli(serial, "--run-all", "--workers", "1", "--out", str(tmp_path / "out1"))
    _fanned(fanned)
    a, b = _outputs(serial), _outputs(fanned)
    ran = ("gen", "folds", "final")  # the fanned run's stages; --run-all also runs env/strict/sloppy
    keep = {k: v for k, v in a.items() if k.split("/")[1] in ran}
    assert keep and keep == {k: v for k, v in b.items() if k.split("/")[1] in ran}
    # non-vacuity 1: in each fanned job the gen units completed out of key order
    gen = _ledger(fanned, "gen")
    by_job: dict[str, list[str]] = {}
    for r in gen:
        by_job.setdefault(r["job_label"] or r["worker"], []).append(r["key"])
    assert any(keys != sorted(keys) for keys in by_job.values()) or [r["key"] for r in gen] != sorted(
        r["key"] for r in gen
    )
    # non-vacuity 2: the inner pool completed out of inner-key order on at least one fold
    assert any(r["inner_completion"] != sorted(r["inner_completion"]) for r in _ledger(fanned, "folds"))
    # non-vacuity 3 (on THIS run's data): for every fold the stored score equals the sorted-key
    # sequential sum of its inner values, and for at least one fold whose pool completed out of order,
    # re-summing in COMPLETION order gives a different float (1e16 + 1 loses the 1). A completion-order
    # inner reduce would therefore have changed the bytes the identity assertion above compares.
    from scripts._stages import LocalStore, read_result
    from tests.scripts._toy_pipeline import _inner_eval

    gens = {s["name"]: s["generation"] for s in _plan(fanned)["stages"]}
    store = LocalStore(fanned)
    v = tuple(read_result(store, f"{gens['gen']}/_output")["v"].tolist())
    differs = False
    for r in _ledger(fanned, "folds"):
        inner = {j: _inner_eval(int(r["key"]), int(j), v) for j in r["inner_completion"]}
        stored = read_result(store, f"{gens['folds']}/{r['key']}")["score"]
        assert stored == _seq(inner, sorted(inner)), r["key"]
        differs |= _seq(inner, r["inner_completion"]) != _seq(inner, sorted(inner))
    assert differs


def test_workers_run_with_the_deterministic_environment(tmp_path):
    _cli(tmp_path, "--stages", "env", "--workers", "1")
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "env")
    env = json.loads((tmp_path / gen / "only.json").read_text())
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1" and env["PYTHONHASHSEED"] == "0" and env["TZ"] == "UTC"


@pytest.mark.slow  # slow: measured 18 s on the Windows dev box (ADR-023)
def test_a_killed_worker_is_relaunched_and_no_finished_unit_is_redone(tmp_path):
    marker = tmp_path / "crashed.marker"
    p = _cli(tmp_path / "s", "--stages", "gen", "--workers", "2", "--crash-once", str(marker))
    assert marker.exists() and "relaunching worker" in p.stderr
    recs = _ledger(tmp_path / "s", "gen")
    assert sorted(r["key"] for r in recs) == [f"toy__{i:04d}" for i in range(12)]  # gen keys are JOINED keys
    assert all(r["attempt"] == 1 for r in recs)  # a finished unit is never redone
    clean = tmp_path / "clean"
    _cli(clean, "--stages", "gen", "--workers", "1")
    _cli(tmp_path / "s", "--stages", "gen")  # reduce
    _cli(clean, "--stages", "gen")
    pick = lambda d: {k: v for k, v in _outputs(d).items() if "/gen/" in k}  # noqa: E731
    assert pick(tmp_path / "s") == pick(clean)


@pytest.mark.slow  # slow: measured 14 s on the Windows dev box (ADR-023)
def test_resume_runs_only_pending_units(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--shard", "0/2", "--workers", "2")
    before = {r["key"]: r["finished_ns"] for r in _ledger(tmp_path, "gen")}
    _cli(tmp_path, "--stages", "gen", "--workers", "2")
    after = {r["key"]: r["finished_ns"] for r in _ledger(tmp_path, "gen")}
    assert all(after[k] == v for k, v in before.items())
    assert len(after) == 12


@pytest.mark.slow  # slow: measured 14 s on the Windows dev box (ADR-023)
def test_a_poisoned_unit_is_redone_on_resume(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--workers", "1")
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == "gen")
    shard = tmp_path / gen / "toy__0005.parquet"
    shard.write_bytes(shard.read_bytes()[:-5])
    _cli(tmp_path, "--stages", "gen", "--workers", "1")
    assert {r["key"]: r["attempt"] for r in _ledger(tmp_path, "gen")}["toy__0005"] == 2


def test_zero_tolerance_stage_aborts_on_a_failure(tmp_path):
    p = _cli(tmp_path, "--stages", "strict", "--workers", "1", "--fail-keys", "0002", check=False)
    assert p.returncode == 3 and "STAGE FAILED" in p.stderr
    gen = next(s["generation"] for s in _plan(tmp_path, "--fail-keys", "0002")["stages"] if s["name"] == "strict")
    assert (tmp_path / gen / "0002.failed.json").is_file()


def test_first_k_all_failed_aborts_early(tmp_path):
    p = _cli(tmp_path, "--stages", "sloppy", "--workers", "1", "--fail-keys", "*", check=False)
    assert p.returncode == 3 and "first 3" in p.stderr
    gen = next(s["generation"] for s in _plan(tmp_path, "--fail-keys", "*")["stages"] if s["name"] == "sloppy")
    n_failed = len(list((tmp_path / gen).glob("*.failed.json")))
    assert 3 <= n_failed < 6  # it ran the first k=3 and stopped early (>= 3 proves the right generation)


def test_plan_json_reports_progress(tmp_path):
    _cli(tmp_path, "--stages", "gen", "--shard", "0/2", "--workers", "1")
    gen = next(s for s in _plan(tmp_path)["stages"] if s["name"] == "gen")
    assert (gen["n_units"], gen["done"], gen["pending"]) == (12, 6, 6)


@pytest.mark.parametrize("n_items", [12])
def test_tolerant_stage_reduces_over_successes(tmp_path, n_items):
    _cli(tmp_path, "--stages", "gen", "--workers", "1", "--fail-keys", "toy__0001")
    gen = next(x["generation"] for x in _plan(tmp_path)["stages"] if x["name"] == "gen")
    s = json.loads((tmp_path / gen / "_summary.json").read_text())
    assert (s["n_done"], s["n_failed"]) == (n_items - 1, 1)


def test_shard_keys_runs_exactly_the_listed_units(tmp_path):
    """Planner LPT placement (Task 5): a job runs exactly the keys of its part, nothing else."""
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(["toy__0005", "toy__0001"]), encoding="utf-8")
    _cli(tmp_path / "s", "--stages", "gen", "--workers", "1", "--shard-keys", str(keys))
    assert sorted(r["key"] for r in _ledger(tmp_path / "s", "gen")) == ["toy__0001", "toy__0005"]


def test_shard_keys_refuse_an_unknown_key_and_a_shard(tmp_path):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps(["toy__0001", "toy__9999"]), encoding="utf-8")
    p = _cli(tmp_path / "s", "--stages", "gen", "--shard-keys", str(keys), check=False)
    assert p.returncode != 0 and "not units" in p.stderr
    p = _cli(tmp_path / "s", "--stages", "gen", "--shard", "1/2", "--shard-keys", str(keys), check=False)
    assert p.returncode != 0 and "--shard-keys" in p.stderr


# ---- the rss backend (ADR-128: RLIMIT_AS failed every capped unit on real HF, 2026-10-20) -------------


def _toy_ctx(tmp_path):
    from scripts._stages import LocalStore, build_parser, make_context
    from tests.scripts._toy_pipeline import PIPELINE

    ns = build_parser(PIPELINE).parse_args(["--plan", "--store", str(tmp_path / "s")])
    params = {p.name: getattr(ns, p.name) for p in PIPELINE.params}
    prov = {"commit": "t", "tree_state": "clean", "code_identity": "t"}
    return make_context(PIPELINE, params, prov, store=LocalStore(tmp_path / "s"), data_root=None)


def test_a_worker_over_its_rss_cap_is_killed_and_the_relaunch_limit_names_the_cap(tmp_path):
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor, StageFailedError

    cfg = ExecutorConfig(workers=2, cap_bytes=10**9, mem_backend="rss", max_relaunch=1, poll_interval_s=0.05)
    ex = LocalExecutor(cfg, rss_of=lambda pid: (pid, 2 * 10**9))  # every worker is over its cap
    with pytest.raises(StageFailedError, match=r"over its memory cap \(RSS 2000000000 B > cap 1000000000 B\)"):
        ex.run_stage(_toy_ctx(tmp_path), "gen")


def test_a_worker_within_its_rss_cap_runs_to_completion(tmp_path):
    from scripts._stages import sorted_units, unit_states
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    seen: list[int] = []
    cfg = ExecutorConfig(workers=2, cap_bytes=10**9, mem_backend="rss", poll_interval_s=0.05)
    ex = LocalExecutor(cfg, rss_of=lambda pid: seen.append(pid) or (pid, 10**8))
    ctx = _toy_ctx(tmp_path)
    report = ex.run_stage(ctx, "gen")
    st = ctx.pipeline.stage("gen")
    assert report.n_done == 12 and set(unit_states(ctx, st, sorted_units(ctx, st), verify=False).values()) == {"done"}
    assert seen  # non-vacuity: the watchdog measured the live workers


class _FakeProc:
    def __init__(self, pid):
        self.pid, self.killed = pid, False

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9


def test_an_over_cap_unit_is_recorded_as_a_memory_failure_and_named(tmp_path):
    """The in-progress unit of a worker over its cap is a FAILED unit with a memory reason: the controller
    escalates it to a job of its own (W=1, the whole box). A finished unit of the same worker is not named."""
    from scripts._stages import plan, run_unit
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    from scripts._stages import ItemStage

    st = ctx.pipeline.stage("gen")
    assert isinstance(st, ItemStage)
    refs = st.ref_map(ctx)
    keys = sorted(refs)
    run_unit(ctx, st, keys[0], worker="w0", ref=refs[keys[0]])  # finished earlier by the same worker
    claims = tmp_path / "claims"
    claims.mkdir()
    for k in keys[:2]:
        (claims / f"{k}.claim").write_text("w0", encoding="ascii")
    ex = LocalExecutor(ExecutorConfig(cap_bytes=10**9, mem_backend="rss"), rss_of=lambda pid: (pid, 2 * 10**9))
    procs, over = {"w0": _FakeProc(4242)}, {}
    recorded = ex._enforce_rss_cap(ctx, st, procs, claims, 10**9, over, lambda pid: False)
    assert recorded == {"w0"} and procs["w0"].killed
    assert keys[1] in over["w0"] and keys[0] not in over["w0"]
    rec = next(r for r in plan(ctx)["stages"] if r["name"] == "gen")
    assert rec["failed_units"] == {keys[1]: 1} and rec["memory_units"] == [keys[1]]


def test_a_worker_over_its_cap_with_no_unit_in_progress_records_nothing(tmp_path):
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    claims = tmp_path / "claims"
    claims.mkdir()
    ex = LocalExecutor(ExecutorConfig(cap_bytes=10**9, mem_backend="rss"), rss_of=lambda pid: (pid, 2 * 10**9))
    procs, over = {"w0": _FakeProc(4242)}, {}
    assert ex._enforce_rss_cap(ctx, ctx.pipeline.stage("gen"), procs, claims, 10**9, over, lambda pid: False) == set()
    assert procs["w0"].killed and "no unit in progress" in over["w0"]


def test_only_a_kill_with_no_unit_recorded_counts_toward_the_relaunch_limit():
    """Reviewer B CCHC-IMPL-11: a worker killed over its cap WITH its unit recorded made progress (the unit
    escalates); only a death with nothing recorded is charged against `max_relaunch`."""
    from scripts.hf_jobs.executor import _charge_death

    attempts: dict[str, int] = {}
    kills = {"w0"}
    assert _charge_death("w0", kills, attempts) == 0 and kills == set()  # recorded: not charged, consumed
    assert _charge_death("w0", kills, attempts) == 1  # the next death of w0 with nothing recorded is
    assert _charge_death("w1", kills, attempts) == 1 and attempts == {"w0": 1, "w1": 1}


def test_a_deferred_memory_unit_neither_fails_a_strict_stage_nor_lets_it_reduce(tmp_path):
    """Reviewer B CCHC-IMPL-12: a job that records a memory unit must COMPLETE (an ERROR would count toward
    the controller's ERRORs-in-a-row bound before the escalation runs), skip the deferred unit, and not
    reduce (a tolerant stage would drop the unit). Only an --escalate job runs it."""
    from scripts._stages import record_memory_failure, unit_states
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    st = ctx.pipeline.stage("strict")  # max_failures=0
    assert st.max_failures == 0
    record_memory_failure(ctx.store, ctx.generation("strict"), "0002", "over its memory cap")
    ex = LocalExecutor(ExecutorConfig(workers=2, poll_interval_s=0.05))
    report = ex.run_stage(ctx, "strict")  # no StageFailedError
    states = unit_states(ctx, st, [f"{i:04d}" for i in range(6)], verify=False)
    assert states["0002"] == "failed" and report.n_done == 5  # the deferred unit was not run here
    assert not ex.try_reduce(ctx, "strict")
    esc = LocalExecutor(ExecutorConfig(workers=1, escalate=True, poll_interval_s=0.05))
    esc.run_stage(ctx, "strict", keys=["0002"])
    assert unit_states(ctx, st, ["0002"], verify=False)["0002"] == "done"
    assert esc.try_reduce(ctx, "strict")


def test_a_memory_unit_recorded_during_a_strict_run_lets_the_job_complete(tmp_path, monkeypatch):
    """Reviewer B CCHC-IMPL-12: the unit is recorded over its cap WHILE the job runs (the watchdog path),
    so the end-of-run tolerance check -- not the slice filter -- must exclude it: the job completes."""
    from scripts._stages import record_memory_failure, run_unit
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)

    def dispatch(ctx_, st_, pending, workers, inner):
        for k in pending:
            if k == "0002":  # the watchdog killed this worker over its cap: the unit is recorded, not run
                record_memory_failure(ctx_.store, ctx_.generation(st_.name), k, "over its memory cap")
            else:
                run_unit(ctx_, st_, k, worker="w0")
        return 0, tuple(pending)

    ex = LocalExecutor(ExecutorConfig(workers=1))
    monkeypatch.setattr(ex, "_dispatch", dispatch)
    report = ex.run_stage(ctx, "strict")  # a StageFailedError here would end the HF job in ERROR
    assert report.n_done == 5 and report.n_failed == 0
    assert not ex.try_reduce(ctx, "strict")


def test_the_stage_cli_escalate_flag_runs_a_deferred_unit(tmp_path):
    """Reviewer B CCHC-IMPL-13: through the real CLI. Without --escalate the deferred unit is skipped (a
    job that dropped the flag would COMPLETE without it and re-escalate every step); with it, it runs."""
    from scripts._stages import LocalStore, record_memory_failure

    store = tmp_path / "s"
    _cli(store, "--stages", "strict", "--workers", "1")  # every unit done
    gen = next(x["generation"] for x in _plan(store)["stages"] if x["name"] == "strict")
    for suffix in (".json", ".sha256", ".ledger.json"):  # unit 0002 never finished: its worker was killed
        (store / gen / f"0002{suffix}").unlink()
    assert record_memory_failure(LocalStore(store), gen, "0002", "over its memory cap") == 1  # deferred
    _cli(store, "--stages", "strict", "--workers", "1")
    assert (store / gen / "0002.failed.json").exists() and not (store / gen / "0002.json").exists()
    _cli(store, "--stages", "strict", "--workers", "1", "--escalate")
    assert (store / gen / "0002.json").exists() and not (store / gen / "0002.failed.json").exists()
