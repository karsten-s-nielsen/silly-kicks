"""Single-source the thread-pin env so parallel workers don't oversubscribe cores.

aarch64 DGX numpy links OpenBLAS, so ``OPENBLAS_NUM_THREADS`` is load-bearing; macOS
Accelerate reads ``VECLIB_MAXIMUM_THREADS``. ADR-128: every stage worker starts under
``deterministic_env`` (INV-2); the ghost ``importance`` stage's loky pool inherits that env.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

_PIN_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def thread_pin_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``base`` (default ``os.environ``) copied with every thread var pinned to ``"1"``."""
    env = dict(os.environ if base is None else base)
    for var in _PIN_VARS:
        env[var] = "1"
    return env


#: The ADR-128 stage-contract worker environment. Every unit is single-threaded (INV-2), so numba is
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
    """Return ``base`` (default ``os.environ``) copied with every ``DETERMINISTIC_VARS`` entry set."""
    env = dict(os.environ if base is None else base)
    env.update(DETERMINISTIC_VARS)
    return env
