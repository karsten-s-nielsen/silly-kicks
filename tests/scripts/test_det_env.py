"""ADR-128 worker environment: every thread pin + numba + the fixed interpreter/locale variables."""

from __future__ import annotations

import os

from scripts._thread_pin import _PIN_VARS, DETERMINISTIC_VARS, deterministic_env, thread_pin_env


def test_deterministic_env_pins_threads_numba_and_fixed_vars():
    env = deterministic_env({"PATH": "/bin", "OMP_NUM_THREADS": "8"})
    pinned = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    )
    for var in pinned:
        assert env[var] == "1", var
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONUTF8"] == "1"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["TZ"] == "UTC"
    assert env["PATH"] == "/bin"


def test_deterministic_env_does_not_mutate_os_environ():
    before = dict(os.environ)
    deterministic_env()
    assert dict(os.environ) == before


def test_thread_pin_env_variable_set_is_unchanged():
    """Chesterton: `_parallel_launch` (DGX DAS runs) keeps numba threads free. The stage contract uses
    `deterministic_env`; `thread_pin_env` must not grow NUMBA_NUM_THREADS."""
    assert set(_PIN_VARS) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
    assert "NUMBA_NUM_THREADS" not in thread_pin_env({})
    assert set(DETERMINISTIC_VARS) >= set(_PIN_VARS)
