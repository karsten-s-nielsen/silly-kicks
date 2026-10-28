"""ADR-128 framework gates (spec 5.2): W1N1 vs W2N2 byte identity, non-vacuity, thread pins, resume,
failure rule, checksum resume. Every run goes through the real CLI and real worker subprocesses."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

_MOD = "tests.scripts._toy_pipeline"
# _reduce.json carries the timing-dependent inline-reduce WALL (T4/B-19); parity ignores it, as _parity.VOLATILE does
_VOLATILE = (".ledger.json", ".failed.json", "_profile.json", "_reduce.json")


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
    assert _ledger(tmp_path, "env")[0]["thread_budget"] == 1  # a pinned stage records a budget of 1


def _env_of(tmp_path, stage):
    _cli(tmp_path, "--stages", stage, "--workers", "1")
    gen = next(s["generation"] for s in _plan(tmp_path)["stages"] if s["name"] == stage)
    return json.loads((tmp_path / gen / "only.json").read_text())


def test_thread_safe_omp_worker_raises_only_omp_and_keeps_blas_serial(tmp_path):
    # T2/B38: the omp lever gives HistGB its cores via OMP but keeps BLAS (cho_factor/einsum) serial for P3
    from scripts._parallel_launch import available_cpus
    from scripts._thread_pin import RUNTIME_THREADS_VAR, thread_budget

    budget = thread_budget(available_cpus(), 1, 1)
    env = _env_of(tmp_path, "env_omp")
    assert env["OMP_NUM_THREADS"] == str(budget)
    assert env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert env[RUNTIME_THREADS_VAR] is None  # omp lever never exports the runtime var
    assert _ledger(tmp_path, "env_omp")[0]["thread_budget"] == budget


def test_thread_safe_njobs_worker_raises_omp_keeps_blas_serial_and_exports_runtime_threads(tmp_path):
    # T2 + HFU plan R3-8 S3 item 1: the xgb lever carries the budget in SILLY_KICKS_RUNTIME_THREADS AND in
    # OMP_NUM_THREADS (XGBoost caps nthread at omp_get_max_threads()); BLAS and numba stay "1" (B38)
    from scripts._parallel_launch import available_cpus
    from scripts._thread_pin import RUNTIME_THREADS_VAR, thread_budget

    budget = thread_budget(available_cpus(), 1, 1)
    env = _env_of(tmp_path, "env_njobs")
    assert env["OMP_NUM_THREADS"] == str(budget)
    assert env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "1"
    assert env[RUNTIME_THREADS_VAR] == str(budget)
    assert _ledger(tmp_path, "env_njobs")[0]["thread_budget"] == budget


def test_a_job_planned_at_four_workers_runs_each_thread_safe_worker_at_the_threads_budget(tmp_path):
    """S3 item 2 (S2B-04), at the EXECUTOR: a thread_safe stage's job with `--workers 4` runs 4 workers, and each
    unit records the THREADS budget of 4 workers (`thread_budget(vcpu, 4, 1)`), not the W=1 budget."""
    from scripts._parallel_launch import available_cpus
    from scripts._thread_pin import thread_budget

    _cli(tmp_path, "--stages", "nfit", "--workers", "4")
    recs = _ledger(tmp_path, "nfit")
    assert len(recs) == 4
    assert {r["thread_budget"] for r in recs} == {thread_budget(available_cpus(), 4, 1)}
    assert len({r["worker"] for r in recs}) >= 2  # the units really ran on more than one worker


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


def test_the_rss_watch_uses_the_injected_quantity_and_budget(tmp_path):
    """Reviewer B D1-IMPL-18: the rss watch kills on whatever `rss_fn` returns vs the budget it is GIVEN, so
    run_stage passes the TREE TOTAL (`total_rss`) + the tree budget for an inner_parallel stage. A tree UNDER
    the budget survives even when one process is huge (a per-process worker-cap check would have killed it); a
    tree OVER the budget is killed."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx, st = _toy_ctx(tmp_path), None
    st = ctx.pipeline.stage("gen")
    claims = tmp_path / "claims"
    claims.mkdir()
    ex = LocalExecutor(ExecutorConfig(cap_bytes=10**9, inner_cap_bytes=2 * 10**8, mem_backend="rss"))
    budget = 30 * 10**9  # the tree budget an inner stage would pass (worker cap + inner * child cap)
    procs = {"w0": _FakeProc(4242)}
    ex._enforce_rss_cap(ctx, st, procs, claims, budget, {}, lambda pid: False, rss_fn=lambda pid: (pid, 24 * 10**9))
    assert not procs["w0"].killed  # tree total 24G under the 30G budget -> survives (per-process would not apply here)
    ex._enforce_rss_cap(ctx, st, procs, claims, budget, {}, lambda pid: False, rss_fn=lambda pid: (pid, 31 * 10**9))
    assert procs["w0"].killed  # tree total 31G over the 30G budget -> killed


def _gen_reduced(ctx):
    """Run and reduce the toy `gen` stage, so an inner stage (`folds`) can read it as upstream."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=1, poll_interval_s=0.05))
    ex.run_stage(ctx, "gen")
    ex.try_reduce(ctx, "gen")


def test_an_inner_stage_run_stage_watches_the_tree_total_not_the_largest(tmp_path):
    """Reviewer B D1-IMPL-22: run_stage SELECTS total_rss + the tree budget for an inner_parallel stage (`folds`),
    end to end. The watch calls total_rss and NEVER largest_rss; a tree over the budget kills with a message
    naming that budget (worker cap + inner * inner_cap) -- a per-process worker-cap check would not have fired."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor, StageFailedError

    ctx = _toy_ctx(tmp_path)
    _gen_reduced(ctx)
    seen_total: list[int] = []
    seen_largest: list[int] = []

    def total_rss(pid):
        seen_total.append(pid)
        return (pid, 99 * 10**9)  # over the tree budget

    def rss_of(pid):
        seen_largest.append(pid)
        return (pid, 10**8)  # tiny: a per-process worker-cap check would NOT kill

    cfg = ExecutorConfig(
        workers=1,
        inner_workers=3,
        cap_bytes=10**9,
        inner_cap_bytes=2 * 10**8,
        mem_backend="rss",
        max_relaunch=1,
        poll_interval_s=0.05,
    )
    ex = LocalExecutor(cfg, rss_of=rss_of, total_rss=total_rss)
    budget = 10**9 + 3 * 2 * 10**8  # cap + inner * inner_cap = 1_600_000_000
    with pytest.raises(StageFailedError, match=rf"RSS 99000000000 B > cap {budget} B"):
        ex.run_stage(ctx, "folds")
    assert seen_total and not seen_largest  # the inner stage watched the TREE TOTAL, never the largest process


def test_a_non_inner_stage_run_stage_watches_the_largest_not_the_tree_total(tmp_path):
    """Reviewer B D1-IMPL-22 (mirror): a non-inner stage (`gen`) uses largest_rss, never total_rss -- is_inner is
    False (no inner_cap_bytes), so the round-2 per-process behaviour is unchanged."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    seen_total: list[int] = []
    seen_largest: list[int] = []

    def total_rss(pid):
        seen_total.append(pid)
        return (pid, 10**8)

    def rss_of(pid):
        seen_largest.append(pid)
        return (pid, 10**8)

    ex = LocalExecutor(
        ExecutorConfig(workers=2, cap_bytes=10**9, mem_backend="rss", poll_interval_s=0.05),
        rss_of=rss_of,
        total_rss=total_rss,
    )
    ex.run_stage(ctx, "gen")
    assert seen_largest and not seen_total  # the non-inner stage watched the largest process, never the tree total


def test_cgroup_scope_memorymax_is_the_tree_budget_for_an_inner_stage(tmp_path, monkeypatch):
    """Reviewer B D1-IMPL-22: for an inner_parallel stage the cgroup scope's MemoryMax is the TREE budget (worker
    cap + inner * inner_cap), so the healthy tree is not OOM-killed under the worker cap; for a non-inner stage it
    is the worker cap. `wrap` is captured (and strips the systemd-run prefix so the worker runs on this box)."""
    from scripts import _mem_cap
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    caps: list[tuple[str, int]] = []

    def rec_wrap(backend, argv, *, cap_bytes):
        caps.append((backend, cap_bytes))
        return list(argv), None  # strip the systemd-run scope so the worker launches locally

    monkeypatch.setattr(_mem_cap, "wrap", rec_wrap)
    monkeypatch.setattr(_mem_cap, "preflight", lambda *a, **k: None)
    gex = LocalExecutor(ExecutorConfig(workers=1, cap_bytes=10**9, mem_backend="cgroup", poll_interval_s=0.05))
    gex.run_stage(ctx, "gen")  # non-inner -> MemoryMax = the worker cap
    gex.try_reduce(ctx, "gen")  # reduce gen as folds' upstream
    assert caps and all(b == "cgroup" for b, _ in caps) and {cb for _, cb in caps} == {10**9}
    caps.clear()
    LocalExecutor(
        ExecutorConfig(
            workers=1,
            inner_workers=3,
            cap_bytes=10**9,
            inner_cap_bytes=2 * 10**8,
            mem_backend="cgroup",
            poll_interval_s=0.05,
        )
    ).run_stage(ctx, "folds")  # inner -> MemoryMax = the tree budget
    assert caps and {cb for _, cb in caps} == {10**9 + 3 * 2 * 10**8}


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


def test_profile_of_a_thread_safe_stage_runs_at_the_production_thread_budget(tmp_path):
    # §E / T8: the profile of a thread_safe stage measures cost at the thread budget it uses in production
    # (not pinned 1), so the recorded per-unit cost is not thread-confounded. For a solo fit (workers=1) the
    # production budget is the full box, applied by _dispatch on the profile path (composes with T2).
    from scripts._parallel_launch import available_cpus
    from scripts._thread_pin import thread_budget
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    cfg = ExecutorConfig(workers=1, profile_units=1, poll_interval_s=0.05)
    ex = LocalExecutor(cfg)
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "env_omp")
    gen = ctx.generation("env_omp")  # the in-process generation (same commit as the run)
    recs = [json.loads(p.read_text()) for p in (tmp_path / "s" / gen).glob("*.ledger.json")]
    assert recs and recs[0]["thread_budget"] == thread_budget(available_cpus(), 1, 1) > 0


@pytest.mark.slow  # 8 HistGB fits (4 units x 2 worker counts); invariant
def test_executor_thread_safe_stage_byte_identical_across_worker_counts(tmp_path):
    # B46: the ADR-128 (W,N) parity gate's THREAD axis at the EXECUTOR level -- a thread_safe stage's stored
    # artifacts are byte-identical whether run at W=1 (budget=cpus) or W=2 (budget=cpus//2), proving the
    # per-worker thread-budget injection preserves determinism end-to-end.
    _cli(tmp_path / "w1", "--stages", "nfit", "--workers", "1")
    _cli(tmp_path / "w2", "--stages", "nfit", "--workers", "2")
    a = {k: v for k, v in _outputs(tmp_path / "w1").items() if k.split("/")[1] == "nfit"}
    b = {k: v for k, v in _outputs(tmp_path / "w2").items() if k.split("/")[1] == "nfit"}
    assert a and a == b


def test_thread_safe_non_inner_profile_is_bounded_and_skips_wmax(tmp_path):
    """HFU T3/B-24: a thread_safe non-inner stage is profiled in ONE bounded job -- only the first sample
    (k<=2) at W=1, NO wmax phase (W=1 has no multi-worker phase), so the profile job stays short and the
    per-unit wall is a clean 8-thread measurement. nfit has 4 units; the old path ran 2 at W=1 + 2 at wmax."""
    import json

    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    cfg = ExecutorConfig(workers=4, profile_units=4, poll_interval_s=0.05)
    ex = LocalExecutor(cfg)
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "nfit")  # thread_safe, non-inner
    gen = ctx.generation("nfit")
    ledgers = list((tmp_path / "s" / gen).glob("*.ledger.json"))
    assert len(ledgers) == 2  # only the bounded `first` sample ran, not all 4
    rec = json.loads((tmp_path / "s" / gen / "_profile.json").read_text())
    assert rec["wmax"]["units"] == []  # the wmax phase was skipped for a W=1 stage


def _done_units(tmp_path, ctx, stage: str) -> int:
    return len(list((tmp_path / "s" / ctx.generation(stage)).glob("*.ledger.json")))


def test_a_repeat_profile_runs_pending_units_only(tmp_path):
    """Spec r3 D5 (B DIAG-04): a repeat profile must make progress. It sampled from ALL shard units, done ones
    included, so the 2474b73 `ghost_outfield` cv re-profile re-picked its two done units and did nothing."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=2, profile_units=4, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "gen")
    first = _done_units(tmp_path, ctx, "gen")
    assert 0 < first < 12  # non-vacuous: a partial profile
    ex.run_stage(ctx, "gen")
    assert _done_units(tmp_path, ctx, "gen") > first


def test_profile_never_overwrites_an_existing_record(tmp_path):
    """Spec r3 D5: a later profile must not overwrite the stage's measured `_profile.json` (2474b73: a no-op
    re-profile replaced the real record with a 0.33 s measurement of two already-done units)."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=2, profile_units=4, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "gen")
    rec = tmp_path / "s" / ctx.generation("gen") / "_profile.json"
    before = rec.read_bytes()
    ex.run_stage(ctx, "gen")
    assert rec.read_bytes() == before


def test_the_profile_measures_the_reduce_cost_of_its_sample(tmp_path):
    """Spec r3 D2 (b): before any reduce has run, the controller needs the reduce's per-unit cost. The profile
    reads its sample back through the reduce's pooled reader and, for a shardable reduce, combines and encodes it
    exactly as a sub-reduce does, timed; `_profile.json` records both per-unit costs."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=2, profile_units=4, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "gen")  # the toy `gen` reduce is a shardable concat
    rec = json.loads((tmp_path / "s" / ctx.generation("gen") / "_profile.json").read_text())
    cost = rec["reduce_cost"]
    assert cost["n"] == 4
    assert cost["read_s_per_unit"] > 0 and cost["combine_s_per_unit"] > 0
    assert cost["bytes_per_unit"] > 0  # B r4 R4-03: the size the cold-read floor scales with


def test_the_profile_of_a_non_shardable_reduce_measures_only_the_read(tmp_path):
    """Spec r3 D2 (b): a non-shardable reduce may not accept a subset of its units, so the profile never runs it
    on the sample; it measures the read and leaves the combine unmeasured (None)."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=2, profile_units=2, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "strict")  # a plain stage with no reduce: nothing to combine
    rec = json.loads((tmp_path / "s" / ctx.generation("strict") / "_profile.json").read_text())
    assert rec["reduce_cost"]["read_s_per_unit"] > 0
    assert rec["reduce_cost"]["combine_s_per_unit"] is None


def test_a_reduce_that_finds_a_corrupt_unit_leaves_the_stage_incomplete_not_failed(tmp_path):
    """Spec r3 D6(d): a corrupt (for example truncated) unit result is a recoverable loss. The reduce deletes it
    (INV-3) and the stage is INCOMPLETE -- not a job failure, which the controller would halt on -- so the next
    run re-runs the unit and the reduce then completes."""
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=2, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "gen")
    gen_dir = tmp_path / "s" / ctx.generation("gen")
    victim = sorted(p for p in gen_dir.glob("*.parquet") if not p.name.startswith("_"))[0]
    victim.write_bytes(victim.read_bytes() + b"x")  # the bytes no longer match the sidecar
    assert ex.try_reduce(ctx, "gen") is False
    assert not victim.exists()  # deleted for re-run
    ex.run_stage(ctx, "gen")
    assert ex.try_reduce(ctx, "gen") is True


def test_units_invalidated_exit_non_zero_for_run_all_and_zero_for_a_controller_job(tmp_path, monkeypatch):
    """B r4 R4-07: spec D6(d)'s exit 0 is for a controller-launched stage job (the controller re-runs the units; an
    ERROR job would halt it). A `--run-all` caller has no controller behind it: an incomplete pipeline must exit
    non-zero (3, as any incomplete stage), never 0 with no output written."""
    from scripts._stages import UnitsInvalidatedError, main
    from scripts.hf_jobs.executor import LocalExecutor
    from tests.scripts._toy_pipeline import PIPELINE

    def invalidated(self, ctx, name, *args, **kwargs):
        raise UnitsInvalidatedError(f"{name}: 1 corrupt unit deleted (INV-3)")

    monkeypatch.setattr(LocalExecutor, "require_reduced", invalidated)
    monkeypatch.setattr(LocalExecutor, "run_stage", invalidated)
    store = str(tmp_path / "s")
    assert main(PIPELINE, ["--store", store, "--allow-dirty", "--run-all"]) == 3
    assert main(PIPELINE, ["--store", store, "--allow-dirty", "--stages", "gen"]) == 0


def test_job_record_is_volatile_for_parity():
    """Spec r3 D8 (B plan review PLAN-R3-02): the job record holds timing, so parity must ignore it."""
    from tests.scripts._parity import VOLATILE

    assert ".job.json" in VOLATILE


def test_job_record_is_read_in_bulk_by_the_summary():
    """Spec r3 D8 (B plan review PLAN-R3-02): the job record is a controller record (batched in one listing)."""
    from scripts.hf_jobs.store import _is_record

    assert _is_record("toy.default.gen.0of1.host.123.job.json")


def test_stage_cli_writes_a_job_record_with_total_cpu_and_wall(tmp_path, monkeypatch):
    """Spec r3 D8: R1 is a JOB property, measured. The stage CLI of an HF job (its label set) writes ONE record with
    the job's total CPU (itself plus every reaped child) and its wall from the process start."""
    import os

    monkeypatch.setenv("SILLY_KICKS_JOB_LABEL", "toy.default.gen.0of1")
    store = tmp_path / "s"
    _cli(store, "--stages", "gen", "--workers", "2")
    recs = [json.loads(p.read_text()) for p in (store / "_jobs").glob("*.job.json")]
    assert len(recs) == 1
    r = recs[0]
    assert r["job_label"] == "toy.default.gen.0of1" and r["stages"] == ["gen"]
    assert r["cpu_s"] > 0 and r["wall_s"] > 0 and r["cpu_s"] <= r["wall_s"] * (os.cpu_count() or 1) + 1


def test_stage_cli_writes_no_job_record_outside_a_job(tmp_path, monkeypatch):
    monkeypatch.delenv("SILLY_KICKS_JOB_LABEL", raising=False)
    store = tmp_path / "s"
    _cli(store, "--stages", "gen", "--workers", "1")
    assert not (store / "_jobs").exists()


def test_run_unit_records_cpu_s(tmp_path):
    """Spec r3 D8: each unit's ledger carries its CPU seconds (stage diagnostics; provenance only)."""
    store = tmp_path / "s"
    _cli(store, "--stages", "gen", "--workers", "1")
    recs = _ledger(store, "gen")
    assert recs and all(isinstance(r.get("cpu_s"), float) and r["cpu_s"] >= 0 for r in recs)


def test_unit_cpu_counts_an_inner_child_that_exits_mid_unit_once():
    """B r4 R4-11: a pool child alive at the unit's start and reaped during it lands its WHOLE-life CPU in the
    reaped-children total; only its CPU after the unit's start belongs to the unit. A live child counts its delta, a
    child born during the unit counts in full, and nothing goes negative."""
    from scripts._stages import _unit_cpu_s

    start = {101: 10.0, 102: 20.0}  # 101 lives on; 102 is reaped during the unit
    end = {101: 15.0, 102: None, 103: 4.0}  # 103 is born during the unit
    reaped_delta = 3.0 + 26.0  # the unit process's own 3 s + child 102's whole life (26 s, 20 of them earlier)
    assert _unit_cpu_s(reaped_delta, start, end) == pytest.approx(3.0 + 5.0 + 6.0 + 4.0)
    assert _unit_cpu_s(0.0, {7: 50.0}, {7: 1.0}) == 0.0  # a reused pid reads lower: never negative


_LOAD_FIELDS = ("upstream_fetch_s", "upstream_verify_s", "upstream_decode_s")


def _gen_ledgers(store_root: pathlib.Path, generation: str) -> dict[str, dict]:
    recs = (json.loads(p.read_text()) for p in (store_root / generation).glob("*.ledger.json"))
    return {r["key"]: r for r in recs}


def _reduced_toy_gen(tmp_path):
    """A toy context whose `gen` stage is run and reduced in-process (so `upstream("gen")` resolves)."""
    from scripts._stages import reduce_stage, run_unit

    ctx = _toy_ctx(tmp_path)
    gen = ctx.pipeline.stage("gen")
    refs = gen.ref_map(ctx)  # type: ignore[union-attr]  # gen is an ItemStage
    for k in sorted(refs):
        run_unit(ctx, gen, k, worker="w0", ref=refs[k])
    reduce_stage(ctx, "gen")
    return ctx


def test_a_loading_unit_records_its_upstream_load_times(tmp_path):
    """Spec r3 D11 (b): the upstream load runs INSIDE a unit (a worker's first), so `RunContext.upstream` times
    its fetch, verify and decode and `run_unit` records them in that unit's ledger record."""
    import dataclasses

    from scripts._stages import run_unit

    ctx = _reduced_toy_gen(tmp_path)
    st = dataclasses.replace(ctx.pipeline.stage("final"), work=lambda key, c: {"rows": len(c.upstream("gen"))})
    run_unit(ctx, st, "u0", worker="w0")
    rec = _gen_ledgers(tmp_path / "s", ctx.generation("final"))["u0"]
    assert rec["upstream_loads"] == 1
    load = [rec[f] for f in _LOAD_FIELDS]
    assert all(isinstance(v, float) and v >= 0 for v in load)
    # The unit wall comes from `time.time_ns()` stamps, the load from `time.perf_counter()`. The `time` clock can tick
    # coarsely (Windows Python 3.12: 15.6 ms), so a fast unit's wall can read one tick short of the load inside it.
    tick = time.get_clock_info("time").resolution
    assert 0 < sum(load) <= rec["wall_s"] + tick


def test_a_unit_reusing_the_loaded_upstream_records_no_load(tmp_path):
    """Spec r3 D11 (b): a unit whose worker already holds the upstream (the trainers' per-process cache, e.g.
    `train_ghost_outfield._data`) records no load, so only the unit that loaded carries the load time."""
    import dataclasses

    from scripts._stages import run_unit

    ctx = _reduced_toy_gen(tmp_path)
    held: dict = {}

    def work(key, c):
        if "gen" not in held:
            held["gen"] = c.upstream("gen")
        return {"rows": len(held["gen"])}

    st = dataclasses.replace(ctx.pipeline.stage("final"), work=work)
    for key in ("u0", "u1"):
        run_unit(ctx, st, key, worker="w0")
    recs = _gen_ledgers(tmp_path / "s", ctx.generation("final"))
    assert recs["u0"]["upstream_loads"] == 1 and sum(recs["u0"][f] for f in _LOAD_FIELDS) > 0
    assert recs["u1"]["upstream_loads"] == 0
    assert [recs["u1"][f] for f in _LOAD_FIELDS] == [0.0, 0.0, 0.0]


def test_reduce_json_is_volatile_and_a_controller_record():
    """HFU T4/B-18/B-27: the wall-time {gen}/_reduce.json must be VOLATILE (parity ignores it) and a
    controller record (batched in the one store listing), exactly like _profile.json."""
    from scripts.hf_jobs.store import _is_record
    from tests.scripts._parity import VOLATILE

    assert "_reduce.json" in VOLATILE
    assert _is_record("_reduce.json")  # _is_record takes a BASENAME (B-09)


def test_try_reduce_records_the_reduce_wall(tmp_path):
    """HFU T4/B-19/B-04: try_reduce times the reduce and writes {gen}/_reduce.json so the planner can budget
    the inline reduce; the wall is the job that actually ran the reduce (idempotent: later jobs see the
    output present and do not overwrite)."""
    import json

    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ex = LocalExecutor(ExecutorConfig(workers=1, poll_interval_s=0.05))
    ctx = _toy_ctx(tmp_path)
    ex.run_stage(ctx, "gen")  # an ItemStage with a reduce
    assert ex.try_reduce(ctx, "gen")
    gen = ctx.generation("gen")
    rec = json.loads((tmp_path / "s" / gen / "_reduce.json").read_text())
    assert rec["stage"] == "gen" and rec["wall_s"] >= 0


def _double_for_child_peak(x):  # module-level so the pool can pickle it by reference
    return x * 2


def test_inner_call_returns_result_and_child_peak():
    """HFU T11 S5b: the pool-child wrapper runs the work and reports the child's own peak RSS, so the unit
    ledger can record the per-child peak. Where /proc and `resource` are both absent the peak is None, never
    a crash."""
    from scripts.hf_jobs.executor import _inner_call

    result, peak = _inner_call(_double_for_child_peak, (21,))
    assert result == 42 and (peak is None or isinstance(peak, int))


def test_inner_pool_exposes_child_peak_attribute():
    """HFU T11 S5b: after a real >1-worker map the pool exposes child_peak_rss_bytes (the max over children,
    or None where the platform cannot measure it); the W=1 in-process path records no separate child peak."""
    from scripts.hf_jobs.executor import InnerPool

    pool = InnerPool(2)
    try:
        out = pool.map_keyed(_double_for_child_peak, {"a": (1,), "b": (2,)})
    finally:
        pool.shutdown()
    assert out == {"a": 2, "b": 4}
    assert pool.child_peak_rss_bytes is None or isinstance(pool.child_peak_rss_bytes, int)

    solo = InnerPool(1)
    solo.map_keyed(_double_for_child_peak, {"a": (1,)})
    assert solo.child_peak_rss_bytes is None  # the W=1 path runs in-process: no separate child


def test_cross_stage_overlap_asserted_and_red_when_serial():
    """HFU T12 S2: the cross-stage overlap assertion is True when two independent stages' jobs overlap in wall
    time (the DAG schedule dispatched them concurrently) and goes RED when they run serially (non-vacuity).
    Two jobs of the SAME stage overlapping is not a cross-stage overlap."""
    from tests.scripts._parity import cross_stage_overlap

    overlapping = [
        {"stage": "a", "started_ns": 0, "finished_ns": 100},
        {"stage": "b", "started_ns": 50, "finished_ns": 150},  # b starts before a ends -> concurrent
    ]
    serial = [
        {"stage": "a", "started_ns": 0, "finished_ns": 100},
        {"stage": "b", "started_ns": 100, "finished_ns": 200},  # b starts when a ends -> serial
    ]
    same_stage = [
        {"stage": "a", "started_ns": 0, "finished_ns": 100},
        {"stage": "a", "started_ns": 50, "finished_ns": 150},  # same stage: not a CROSS-stage overlap
    ]
    assert cross_stage_overlap(overlapping) is True  # the DAG ran a and b concurrently
    assert cross_stage_overlap(serial) is False  # non-vacuity: RED when serialized
    assert cross_stage_overlap(same_stage) is False


@pytest.mark.slow  # slow: 2 real nfit jobs x their fits; invariant, runs in the dedicated slow job
def test_thread_axis_parity_at_w1(tmp_path):
    """HFU T12 S1: at the production W=1 shape (one worker per job at the full thread budget), the thread_safe
    `nfit` stage fanned across >=2 jobs produces output byte-identical to the serial run. fanned_jobs>=2 proves
    the fan really happened; the explicit 1-vs-8-thread byte-identity is carried by the per-model identity
    arms (tests/tracking/test_thread_identity.py), the only cross-platform-deterministic thread control."""
    import os

    from tests.scripts._parity import fanned_jobs

    serial, fanned = tmp_path / "serial", tmp_path / "fanned"
    _cli(serial, "--stages", "nfit", "--workers", "1")  # the W=1 serial reference
    procs = [
        subprocess.Popen(  # noqa: S603 -- W=1 per job, nfit's units sharded across 2 jobs at the full thread budget
            [
                sys.executable,
                "-m",
                _MOD,
                "--store",
                str(fanned),
                "--allow-dirty",
                "--stages",
                "nfit",
                "--shard",
                f"{i}/2",
                "--workers",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "SILLY_KICKS_JOB_LABEL": f"nfit.{i}of2"},
        )
        for i in range(2)
    ]
    for p in procs:
        _o, err = p.communicate()
        assert p.returncode == 0, err[-2000:]
    _cli(fanned, "--stages", "nfit")  # idempotent reduce in whichever job saw the stage complete
    gen = next(s["generation"] for s in _plan(fanned)["stages"] if s["name"] == "nfit")
    a = {k: v for k, v in _outputs(serial).items() if "/nfit/" in k}
    b = {k: v for k, v in _outputs(fanned).items() if "/nfit/" in k}
    assert a and a == b  # byte-identical at the production W=1 shape
    assert fanned_jobs(fanned, gen) >= 2  # the fan across >=2 jobs really happened


@pytest.mark.slow  # slow: 2 real nfit fits at 1 vs N threads; invariant, runs in the dedicated slow job
@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU affinity pinning is Linux-only")
def test_thread_axis_parity_1_vs_N_via_affinity(tmp_path):
    """HFU T12 / reviewer B D1-IMPL-11 (owner ruling 2026-10-29): a GENUINE 1-vs-N thread arm on the executor's
    worker env-injection path. `available_cpus()` reads CPU AFFINITY, so pinning the worker subprocess to ONE
    CPU drives the injected thread budget to 1 and pinning it to ALL drives it to N; the thread_safe `nfit`
    stage's stored output must be byte-identical across the two. This runs the real `--stages nfit` subprocess
    (the env-injection path the in-process T11 arms skip). Linux-only (sched_setaffinity); CI runs it."""
    allcpus = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]  # Linux-only; the test is skipped elsewhere
    if len(allcpus) < 2:
        pytest.skip("needs >= 2 available CPUs for a real 1-vs-N contrast")

    def run(store, cpus, stage):
        p = subprocess.Popen(  # noqa: S603 -- fixed argv; preexec_fn pins affinity in the child before exec
            [sys.executable, "-m", _MOD, "--store", str(store), "--allow-dirty", "--stages", stage, "--workers", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "SILLY_KICKS_JOB_LABEL": f"{stage}.0of1"},
            preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)),  # type: ignore[attr-defined]  # Linux-only
        )
        _o, err = p.communicate()
        assert p.returncode == 0, err[-2000:]

    one, full = tmp_path / "one", tmp_path / "full"
    # NON-VACUITY (reviewer B D1-IMPL-21): env_omp records the injected OMP budget; pinning 1 CPU vs N must
    # change it, else the affinity never took and the byte-identity below would pass vacuously.
    run(one, allcpus[:1], "env_omp")
    run(full, allcpus, "env_omp")
    e1 = {k: v for k, v in _outputs(one).items() if "/env_omp/" in k}
    eN = {k: v for k, v in _outputs(full).items() if "/env_omp/" in k}
    assert e1 and eN and e1 != eN  # the injected thread budget really differed 1 vs N
    # BYTE-IDENTITY: nfit's stored output is identical at 1 thread vs N on the real worker injection path.
    run(one, allcpus[:1], "nfit")
    run(full, allcpus, "nfit")
    a = {k: v for k, v in _outputs(one).items() if "/nfit/" in k}
    b = {k: v for k, v in _outputs(full).items() if "/nfit/" in k}
    assert a and a == b


@pytest.mark.parametrize(
    ("wio", "peak", "launched"),
    [(True, 360 * 10**6, 12), (False, 360 * 10**6, 8), (True, 3 * 2**30, 10)],
)
def test_a_job_planned_at_twelve_workers_launches_twelve_only_for_a_wio_stage(
    tmp_path, monkeypatch, wio, peak, launched
):
    """S3 item 5 (S2B-04), at the EXECUTOR: on an 8-vCPU box a job planned at W=12 launches 12 workers for a WIO stage
    and 8 for a stage that is not WIO; the RAM part of `size_workers` still binds (30 GiB // 3 GiB = 10)."""
    from scripts._stages import sorted_units
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    st = ctx.pipeline.stage("gen")
    keys = sorted_units(ctx, st)
    assert len(keys) >= 12  # non-vacuity: enough pending units for 12 workers
    ex = LocalExecutor(ExecutorConfig(workers=12, peak_rss_bytes=peak, wio=wio))
    monkeypatch.setattr(ex, "_cpus", lambda: 8)
    monkeypatch.setattr(ex, "_ram", lambda: 32 * 2**30)
    seen: list[int] = []
    monkeypatch.setattr(ex, "_dispatch", lambda ctx, st, pending, w, inner: seen.append(w) or (0, ()))
    report = ex._run(ctx, st, keys, workers=12, inner=1)
    assert seen == [launched] and report.workers == launched


def test_the_stage_cli_passes_wio_to_the_executor(tmp_path, monkeypatch):
    """S3 item 5: `--wio` on the stage CLI reaches `ExecutorConfig.wio` (the job command carries it, see
    test_campaign)."""
    import scripts.hf_jobs.executor as E
    from scripts._stages import main
    from tests.scripts._toy_pipeline import PIPELINE

    seen = []

    class _StopError(Exception):
        pass

    class _Capture:
        def __init__(self, cfg, *a, **k):
            seen.append(cfg)
            raise _StopError

    monkeypatch.setattr(E, "LocalExecutor", _Capture)
    for flags, wio in ((["--wio"], True), ([], False)):
        with pytest.raises(_StopError):
            main(PIPELINE, ["--store", str(tmp_path / "s"), "--allow-dirty", "--stages", "gen", *flags])
        assert seen[-1].wio is wio


def test_a_gated_job_planned_at_eight_workers_launches_eight(tmp_path, monkeypatch):
    """S3 item 6 (MEM), at the EXECUTOR: today `size_workers` clamps a job of 7.2 GB units to 4 workers on 32 GiB;
    with per-unit peaks the gate admits each claim, so the job launches the planned 8."""
    from scripts._stages import sorted_units
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    st = ctx.pipeline.stage("gen")
    keys = sorted_units(ctx, st)
    peaks = {k: 5_510_000_000 for k in keys}
    seen: list[int] = []
    for unit_peaks in (None, peaks):
        ex = LocalExecutor(ExecutorConfig(workers=8, peak_rss_bytes=7_200_000_000, unit_peaks=unit_peaks))
        monkeypatch.setattr(ex, "_cpus", lambda: 8)
        monkeypatch.setattr(ex, "_ram", lambda: 32 * 2**30)
        monkeypatch.setattr(ex, "_dispatch", lambda ctx, st, pending, w, inner: seen.append(w) or (0, ()))
        ex._run(ctx, st, keys, workers=8, inner=1)
    assert seen == [4, 8]


def test_retained_heap_never_stalls_a_gated_stage(tmp_path, monkeypatch, capsys):
    """S3 item 6 progress rule (S2B-r2-01), with real workers: seven workers retain 4 GB each once started, and one
    unit reserves 7.2 GB (1.3 x 5.51 GB) of a 30 GB budget. The idle workers that hold the heap are stopped, fresh
    workers take their slots once the gate admits them, and every unit completes."""
    from scripts._stages import sorted_units
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx(tmp_path)
    st = ctx.pipeline.stage("gen")
    keys = sorted_units(ctx, st)
    peaks = {k: 1_000_000_000 for k in keys}
    peaks[keys[7]] = 5_510_000_000
    first: list[int] = []

    def rss(pid: int) -> tuple[int, int]:
        if pid not in first and len(first) < 7:
            first.append(pid)
        return pid, (4 * 10**9 if pid in first else 0)

    ex = LocalExecutor(ExecutorConfig(workers=8, unit_peaks=peaks, poll_interval_s=0.05), rss_of=rss, total_rss=rss)
    monkeypatch.setattr(ex, "_cpus", lambda: 8)
    monkeypatch.setattr(ex, "_ram", lambda: 30 * 10**9 + ex.cfg.headroom_bytes)
    report = ex._run(ctx, st, keys, workers=8, inner=1)
    assert report.n_done == len(keys) and report.n_failed == 0
    assert "gate: stopping idle worker" in capsys.readouterr().err  # non-vacuity: rule (i) really ran


def test_the_stage_cli_reads_the_unit_peaks(tmp_path, monkeypatch):
    """S3 item 6: `--unit-peaks FILE` on the stage CLI reaches `ExecutorConfig.unit_peaks`."""
    import scripts.hf_jobs.executor as E
    from scripts._stages import main
    from tests.scripts._toy_pipeline import PIPELINE

    seen = []

    class _StopError(Exception):
        pass

    class _Capture:
        def __init__(self, cfg, *a, **k):
            seen.append(cfg)
            raise _StopError

    monkeypatch.setattr(E, "LocalExecutor", _Capture)
    f = tmp_path / "peaks.json"
    f.write_text(json.dumps({"toy__0000": 123}), encoding="utf-8")
    base = ["--store", str(tmp_path / "s"), "--allow-dirty", "--stages", "gen"]
    for flags, want in ((["--unit-peaks", str(f)], {"toy__0000": 123}), ([], None)):
        with pytest.raises(_StopError):
            main(PIPELINE, [*base, *flags])
        assert seen[-1].unit_peaks == want


def _toy_ctx_with(tmp_path, *args: str):
    from scripts._stages import LocalStore, build_parser, make_context
    from tests.scripts._toy_pipeline import PIPELINE

    ns = build_parser(PIPELINE).parse_args(["--plan", "--store", str(tmp_path / "s"), *args])
    params = {p.name: getattr(ns, p.name) for p in PIPELINE.params}
    prov = {"commit": "t", "tree_state": "clean", "code_identity": "t"}
    return make_context(PIPELINE, params, prov, store=LocalStore(tmp_path / "s"), data_root=None)


@pytest.mark.parametrize(
    ("peak", "launched", "budget"),
    [(None, 5, 2), (10 * 2**30, 3, 3)],
    ids=["planned-w5-gets-ceil-budget-2", "rss-clamped-w3-gets-its-own-budget-3"],
)
def test_a_threads_job_runs_each_worker_at_the_budget_of_the_w_it_launches(
    tmp_path, monkeypatch, peak, launched, budget
):
    """HFU S3 item 2 at the EXECUTOR (review S3B-06), on an 8-vCPU box: a thread_safe job planned at W=5 launches 5
    workers, each at ceil(8 / 5) = 2 threads (the floor rule gives 1); a peak-RSS hint that fits 3 of them launches
    3, each at ceil(8 / 3) = 3 threads (the planned W=5 would give 2). The applied budget is read from the ledgers."""
    import json as _json

    from scripts._stages import sorted_units
    from scripts.hf_jobs.executor import ExecutorConfig, LocalExecutor

    ctx = _toy_ctx_with(tmp_path, "--nfit-units", "5")
    st = ctx.pipeline.stage("nfit")
    keys = sorted_units(ctx, st)
    assert len(keys) == 5
    ex = LocalExecutor(ExecutorConfig(workers=5, peak_rss_bytes=peak, poll_interval_s=0.05))
    monkeypatch.setattr(ex, "_cpus", lambda: 8)
    monkeypatch.setattr(ex, "_ram", lambda: 32 * 2**30)
    report = ex._run(ctx, st, keys, workers=5, inner=1)
    assert report.workers == launched and report.n_done == 5
    gen = tmp_path / "s" / ctx.generation("nfit")
    assert {_json.loads(p.read_text())["thread_budget"] for p in gen.glob("*.ledger.json")} == {budget}
