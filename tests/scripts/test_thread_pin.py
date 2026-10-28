"""Task 1 guard: the shared thread-pin env sets every relevant thread var to 1."""

import pytest

from scripts._thread_pin import (
    RUNTIME_THREADS_VAR,
    runtime_n_jobs_from_env,
    thread_budget,
    thread_pin_env,
    thread_safe_env,
)


def test_sets_all_four_thread_vars_to_one():
    env = thread_pin_env(base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"  # aarch64 DGX numpy -> OpenBLAS
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"  # macOS Accelerate
    assert env["PATH"] == "/x"  # base preserved


def test_does_not_mutate_the_passed_base():
    base = {"PATH": "/x"}
    thread_pin_env(base=base)
    assert "OMP_NUM_THREADS" not in base


# ---- Cycle 1: opt-in per-job thread budget (ADR-128 INV-2 amendment) ----


@pytest.mark.parametrize(
    ("vcpu", "workers", "inner", "expect"),
    [
        (8, 1, 1, 8),
        (8, 2, 1, 4),
        (8, 3, 1, 3),  # ceil(8 / 3): 9 threads on 8 vCPU (ADR-128 Amendment 2026-10-31 (a))
        (8, 4, 1, 2),
        (8, 5, 1, 2),  # ceil(8 / 5): the measured THREADS shape, 10 threads on 8 vCPU
        (8, 1, 5, 2),  # an inner pool of 5 children (the study's folds), Amendment (c)
        (8, 1, 4, 2),
        (8, 6, 1, 1),  # 6 or 7 processes keep the floor: no HF measurement covers 12 or 14 threads
        (8, 7, 1, 1),
        (8, 8, 1, 1),
        (8, 16, 1, 1),
        (1, 1, 1, 1),
    ],
)
def test_thread_budget_shares_vcpu_with_a_bounded_oversubscription(vcpu, workers, inner, expect):
    b = thread_budget(vcpu, workers, inner)
    assert b == expect
    assert 1 <= b <= vcpu  # never 0, never more than the box per process


@pytest.mark.parametrize("procs", range(1, 17))
def test_thread_budget_oversubscribes_at_most_two_threads_and_only_for_five_processes_or_fewer(procs):
    b = thread_budget(8, procs, 1)
    if procs > 8:
        assert b == 1  # more processes than vCPU: each keeps its one thread (the WIO shape), never more
    else:
        assert b * procs - 8 <= (2 if procs <= 5 else 0)


def test_omp_lever_raises_only_omp_and_keeps_blas_serial():
    # B38: HistGB is OpenMP; BLAS (cho_factor/einsum) is an unproven axis -> pin it
    env = thread_safe_env("omp", 8, base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert RUNTIME_THREADS_VAR not in env
    assert env["PATH"] == "/x"


def test_n_jobs_lever_raises_omp_keeps_blas_serial_and_exports_runtime_threads():
    # xgb: the budget reaches the fit as the runtime n_jobs read trainer-side, AND OpenMP carries it, because
    # XGBoost caps nthread at omp_get_max_threads() (HFU plan R3-8 S3 item 1; S1 measured the lever dead: one core
    # at an 8-thread budget). BLAS and numba stay serial: the xgb determinism evidence varies OpenMP only (B38).
    env = thread_safe_env("n_jobs", 8, base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert env[RUNTIME_THREADS_VAR] == "8"


_THREADPOOL_PROBE = (
    "import json, threadpoolctl, xgboost, sklearn.ensemble\n"
    "pools = threadpoolctl.threadpool_info()\n"
    "print(json.dumps([[i['internal_api'], i['user_api'], i['num_threads']] for i in pools]))\n"
)


@pytest.mark.parametrize("budget", [2, 4])
def test_n_jobs_lever_process_runs_every_openmp_library_at_the_budget_and_blas_at_one(budget):
    """S3 item 1 (a): in a process started with the production n_jobs-lever env, every OpenMP library that
    `threadpoolctl` lists (xgboost's and sklearn's) runs at the budget, and every BLAS library at 1."""
    import json
    import subprocess
    import sys

    env = thread_safe_env("n_jobs", budget)
    out = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a constant probe
        [sys.executable, "-c", _THREADPOOL_PROBE], env=env, capture_output=True, text=True, check=True
    )
    pools = json.loads(out.stdout)
    openmp = [n for _internal, user, n in pools if user == "openmp"]
    blas = [n for _internal, user, n in pools if user == "blas"]
    assert openmp, f"no OpenMP library loaded: {pools}"  # non-vacuity: xgboost brings one
    assert openmp == [budget] * len(openmp), pools
    assert blas == [1] * len(blas), pools


def test_thread_safe_env_rejects_an_unknown_lever():
    with pytest.raises(ValueError, match="thread_lever"):
        thread_safe_env("bogus", 8)  # type: ignore[arg-type] -- intentionally invalid lever at runtime


def test_runtime_n_jobs_from_env_reads_the_transport_var():
    # the trainer-side reader for the n_jobs lever (no env read inside the library fit)
    assert runtime_n_jobs_from_env(base={}) == 1  # unset -> single-threaded
    assert runtime_n_jobs_from_env(base={RUNTIME_THREADS_VAR: "8"}) == 8
    assert runtime_n_jobs_from_env(base={RUNTIME_THREADS_VAR: ""}) == 1  # empty -> 1


def test_deterministic_env_strips_a_stray_runtime_threads_var():
    # B50: a pinned worker must NOT inherit an n_jobs-lever budget left in the parent env
    from scripts._thread_pin import deterministic_env

    env = deterministic_env(base={"PATH": "/x", RUNTIME_THREADS_VAR: "8"})
    assert RUNTIME_THREADS_VAR not in env
    assert env["OMP_NUM_THREADS"] == "1"
