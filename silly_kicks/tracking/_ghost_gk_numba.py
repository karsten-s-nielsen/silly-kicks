"""Optional numba-accelerated ghost-GK KDE hot loop.

Serial ``@njit`` fully-fused closed-form weighted-Gaussian KDE over the fixed grid -- no
``(kb, m)`` temporaries (the numpy closed-form kernel is memory-bound; this keeps it in
registers for a ~10x single-thread speedup, kill-gate-validated). Lazily imported only on the
``cpu-numba`` path:

    try:
        from ._ghost_gk_numba import _kde_numba_loop
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False

Setup (weighted covariance + Cholesky PD-branch + det/norm) stays in numpy (``_kde_setup``);
numba does ONLY the exp+reduction loop, so the singular->uniform boundary stays cho_factor's
(== 4.2.0). Serial (NOT ``parallel=True``): in Databricks serverless ``applyInPandas``, Spark
already saturates cores across (period, frame_batch) groups; an in-group ``prange`` would
oversubscribe.

See docs/superpowers/specs/2026-06-01-ghost-gk-kde-numba-acceleration-design.md.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover - exercised only without the [numba] extra
    msg = "numba is required for the cpu-numba ghost-GK backend. Install with: pip install silly-kicks[numba]"
    raise ImportError(msg) from e

# On-disk cache OFF by default (serverless read-only paths) -- same rationale as
# pitch_control/_numba_kernels.py. Full native JIT speed retained; only cross-process
# persistence is dropped (one-time per-process recompile).
_NUMBA_CACHE = os.environ.get("SILLY_KICKS_NUMBA_CACHE", "0") == "1" or bool(os.environ.get("NUMBA_CACHE_DIR"))


@njit(cache=_NUMBA_CACHE)
def _kde_numba_loop(gx, gy, xs, ys, w, h11, h12, h22, inv_det, norm):
    """Fused weighted-Gaussian KDE over the grid. gx,gy: (m,); xs,ys,w: (k,) -> (m,).

    energy = 0.5/det * (h22*dx^2 - 2*h12*dx*dy + h11*dy^2); density = norm * sum_j w_j exp(-energy).
    No (k, m) temporaries -- scalar accumulate in registers.
    """
    m = gx.shape[0]
    k = w.shape[0]
    out = np.zeros(m)
    half = 0.5 * inv_det
    for j in range(k):
        wj = w[j]
        xj = xs[j]
        yj = ys[j]
        for i in range(m):
            ddx = gx[i] - xj
            ddy = gy[i] - yj
            e = half * (h22 * ddx * ddx - 2.0 * h12 * ddx * ddy + h11 * ddy * ddy)
            out[i] += wj * np.exp(-e)
    return out * norm
