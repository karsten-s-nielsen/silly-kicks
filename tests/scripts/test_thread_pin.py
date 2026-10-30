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
    [(8, 1, 1, 8), (8, 8, 1, 1), (8, 3, 1, 2), (8, 1, 4, 2), (8, 16, 1, 1), (1, 1, 1, 1)],
)
def test_thread_budget_shares_vcpu_and_never_oversubscribes(vcpu, workers, inner, expect):
    b = thread_budget(vcpu, workers, inner)
    assert b == expect
    assert b >= 1 and b <= vcpu  # never 0, never oversubscribes the box


def test_omp_lever_raises_only_omp_and_keeps_blas_serial():
    # B38: HistGB is OpenMP; BLAS (cho_factor/einsum) is an unproven axis -> pin it
    env = thread_safe_env("omp", 8, base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["NUMBA_NUM_THREADS"] == "1"
    assert RUNTIME_THREADS_VAR not in env
    assert env["PATH"] == "/x"


def test_n_jobs_lever_keeps_all_env_serial_and_exports_runtime_threads():
    # xgb: budget reaches the fit via a runtime arg read trainer-side, never the thread env
    env = thread_safe_env("n_jobs", 8, base={"PATH": "/x"})
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["VECLIB_MAXIMUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "1"
    assert env[RUNTIME_THREADS_VAR] == "8"


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
