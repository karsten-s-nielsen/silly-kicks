"""Task 1 guard: the shared thread-pin env sets every relevant thread var to 1."""

from scripts._thread_pin import thread_pin_env


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
