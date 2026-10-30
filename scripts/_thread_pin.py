"""Single-source the thread-pin env so parallel workers don't oversubscribe cores.

aarch64 DGX numpy links OpenBLAS, so ``OPENBLAS_NUM_THREADS`` is load-bearing; macOS
Accelerate reads ``VECLIB_MAXIMUM_THREADS``. ADR-128: every stage worker starts under
``deterministic_env`` (INV-2); the ghost ``importance`` stage's loky pool inherits that env. A ``thread_safe``
stage is the one opt-in (ADR-128 amendment): ``thread_budget`` + ``thread_safe_env`` raise its declared lever
only (`OMP_NUM_THREADS`, or `SILLY_KICKS_RUNTIME_THREADS` read trainer-side into the runtime ``n_jobs``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

#: The executor->trainer transport for the n_jobs lever (ADR-128 Cycle-1 amendment, B37): the executor sets
#: it at an ``n_jobs``-lever ``thread_safe`` worker launch; the trainer reads it TRAINER-SIDE (never inside a
#: library ``fit``) and passes it as the runtime ``n_jobs``. Absent/"1" => single-threaded.
RUNTIME_THREADS_VAR = "SILLY_KICKS_RUNTIME_THREADS"

_PIN_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def thread_pin_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) copied with every thread var pinned to ``"1"``."""
    env = dict(os.environ if base is None else base)
    for var in _PIN_VARS:
        env[var] = "1"
    return env


#: The ADR-128 stage-contract worker environment. Every unit is single-threaded (INV-2; a `thread_safe` stage
#: is the one opt-in, which overrides its lever's var only -- see `thread_safe_env`), so numba is
#: pinned here too, and the interpreter/locale variables are fixed so string hashing, text encoding,
#: collation and wall-clock rendering cannot differ between a laptop, the DGX and an HF job (INV-1).
#: Kept SEPARATE from `thread_pin_env` on purpose: `_parallel_launch` (the DGX DAS launcher) leaves
#: numba threads free, and widening its variable set would change those runs.
DETERMINISTIC_VARS: dict[str, str] = {
    **dict.fromkeys(_PIN_VARS, "1"),
    "NUMBA_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}


def deterministic_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) copied with every ``DETERMINISTIC_VARS`` entry set. Also STRIPS
    a stray ``RUNTIME_THREADS_VAR`` so a pinned (non-``thread_safe``) worker never inherits an n_jobs-lever
    budget left in the parent env -- that would mis-thread a pinned fit and mis-record its ledger (B50)."""
    env = dict(os.environ if base is None else base)
    env.update(DETERMINISTIC_VARS)
    env.pop(RUNTIME_THREADS_VAR, None)
    return env


def runtime_n_jobs_from_env(base: Mapping[str, str] | None = None) -> int:
    """Read the n_jobs-lever budget TRAINER-SIDE from ``RUNTIME_THREADS_VAR`` (default ``os.environ``). A
    trainer passes the result as the model's runtime ``n_jobs`` -- the budget never reaches the library ``fit``
    through the environment (no Hyrum hidden dependency). Unset or empty => ``1`` (single-threaded)."""
    env = os.environ if base is None else base
    return max(1, int(env.get(RUNTIME_THREADS_VAR) or 1))


def thread_budget(vcpu: int, workers: int, inner_workers: int) -> int:
    """The per-worker thread budget so ``threads x units_in_flight <= vcpu`` (ADR-128 INV-2 opt-in): a solo
    unit (``workers=inner_workers=1``) gets all cores; concurrent workers split them. Never 0, never > vcpu."""
    return max(1, vcpu // max(1, workers * inner_workers))


def thread_safe_env(
    lever: Literal["omp", "n_jobs"], budget: int, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The worker environment for a ``thread_safe`` stage: start from ``deterministic_env`` (every thread var
    ``"1"``) and raise ONLY the stage's declared lever to ``budget``.

    - ``"omp"`` (HistGB): set ``OMP_NUM_THREADS`` to the budget. ``OPENBLAS/MKL/VECLIB`` and ``NUMBA`` stay
      ``"1"`` -- HistGB is OpenMP ``prange``, and the ghost assemble's BLAS (``cho_factor``/``einsum``) is an
      axis the determinism evidence (OMP varied only) never covered, so it stays serial for byte-identity (B38).
    - ``"n_jobs"`` (xgb): leave every thread var ``"1"`` and export the budget as ``RUNTIME_THREADS_VAR`` for the
      trainer to read trainer-side into the runtime ``n_jobs`` (no env read inside the library ``fit``).
    """
    env = deterministic_env(base)
    if lever == "omp":
        env["OMP_NUM_THREADS"] = str(budget)
    elif lever == "n_jobs":
        env[RUNTIME_THREADS_VAR] = str(budget)
    else:
        raise ValueError(f"unknown thread_lever {lever!r} (expected 'omp' or 'n_jobs')")
    return env
