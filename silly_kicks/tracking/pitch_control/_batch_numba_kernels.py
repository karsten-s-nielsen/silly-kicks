"""numba kernels of the batched spearman core (spec 2026-11-08 tracking-batch section 4.2, W2).

``_spearman_batch._batch_kernels`` loads this module through ``silly_kicks._numba_dispatch.import_kernels`` in the
``PITCH_CONTROL`` family, unless ``SILLY_KICKS_PITCH_CONTROL_FORCE_NUMPY=1`` selects the numpy reference; a missing
numba raises ``NumbaUnavailableError``, never a silent fallback.

A separate module from ``_numba_kernels``: the frozen test oracles reached ``_numba_kernels`` through its dynamic
kernel import, so a new kernel there would have entered their closure. The oracles ended with ruling C5 (spec
section 7).
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "numba is required for _batch_numba_kernels (numba is a core dependency of silly-kicks; "
        "repair the install: pip install --force-reinstall silly-kicks)"
    ) from e

# The on-disk cache policy of every kernel module, evaluated at this module's own import (see the comment in
# ``_numba_kernels``): off by default, on with SILLY_KICKS_NUMBA_CACHE=1 or a NUMBA_CACHE_DIR.
_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))


@njit(cache=_NUMBA_CACHE)
def influence_rows_numba(
    tti_rows: np.ndarray,
    opp_min_rows: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Numba logistic influence, elementwise over rows that each carry their own opponent minimum TTI.

    ``tti_rows`` and ``opp_min_rows`` are (n_rows, n_targets); the scalar expression equals ``influence_numba``'s.
    """
    k = np.pi / (np.sqrt(3.0) * sigma)
    n_rows = tti_rows.shape[0]
    n_targets = tti_rows.shape[1]
    result = np.empty((n_rows, n_targets))

    for i in range(n_rows):
        for j in range(n_targets):
            exponent = -k * (opp_min_rows[i, j] - tti_rows[i, j])
            result[i, j] = 1.0 / (1.0 + np.exp(exponent))
    return result
