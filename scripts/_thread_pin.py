"""Single-source the thread-pin env so parallel workers don't oversubscribe cores.

aarch64 DGX numpy links OpenBLAS, so ``OPENBLAS_NUM_THREADS`` is load-bearing; macOS
Accelerate reads ``VECLIB_MAXIMUM_THREADS``. ``train_ghost_gk.py``'s loky launch (OMP-only)
should adopt this helper in a later cleanup.
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
