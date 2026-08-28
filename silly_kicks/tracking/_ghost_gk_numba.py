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


@njit(cache=_NUMBA_CACHE)
def _leaf_values_numba(left, right, feat, thr, miss, val, offsets, x):
    """Summed reached-leaf ``value`` per sample -- fused per-sample boosted-tree walk.

    Adapter for :func:`silly_kicks.tracking._ghost_gk._vectorized_leaf_values`; BIT-identical
    (same ops, same tree order). ``left``/``right`` are per-tree LOCAL child indices; the walk
    accesses the flat arrays at ``base + cur`` where ``base = offsets[t]``. ``x`` is the feature
    matrix (n_samples, n_features). Carries the SAME convergence guard as the numpy sibling (a
    >depth-cap tree would read an internal node's ``value`` = garbage), so both RAISE on
    non-convergence -- the asymmetric contract vs :func:`_leaf_indices_numba`, which never reads
    ``value`` and never raises.
    """
    n = x.shape[0]
    n_trees = offsets.shape[0] - 1
    out = np.zeros(n)
    for s in range(n):
        acc = 0.0
        for t in range(n_trees):
            base = offsets[t]
            cur = 0  # local index within tree; root=0, leaf iff left==0
            for _ in range(100):  # depth bound, matches the numpy path
                gi = base + cur
                if left[gi] == 0:
                    break
                fv = x[s, feat[gi]]
                go_left = (miss[gi] != 0) if np.isnan(fv) else (fv <= thr[gi])
                cur = left[gi] if go_left else right[gi]
            # Convergence guard -- matches numpy _vectorized_leaf_values' RuntimeError.
            if left[base + cur] != 0:
                raise RuntimeError("leaf traversal did not converge within depth cap")
            acc += val[base + cur]
        out[s] = acc
    return out


@njit(cache=_NUMBA_CACHE)
def _leaf_indices_numba(left, right, feat, thr, miss, val, offsets, x):
    """Reached-leaf LOCAL index per (sample, tree) -- fused per-sample boosted-tree walk.

    Adapter for :func:`silly_kicks.tracking._ghost_gk._vectorized_leaf_indices`; BIT-identical.
    Returns the per-tree LOCAL ``cur`` (matching numpy's ``current``), NOT global ``base + cur``.
    Carries NO convergence guard -- the numpy sibling never reads ``value`` and returns the
    non-converged index silently, so adding a raise here would ITSELF break bit-identity in the
    >depth-cap case. ``val`` is unused (kept for signature symmetry with :func:`_leaf_values_numba`
    so the dispatch passes an identical 8-arg tuple to both -- do NOT drop it).
    """
    n = x.shape[0]
    n_trees = offsets.shape[0] - 1
    out = np.zeros((n, n_trees), dtype=np.int64)
    for s in range(n):
        for t in range(n_trees):
            base = offsets[t]
            cur = 0
            for _ in range(100):
                gi = base + cur
                if left[gi] == 0:
                    break
                fv = x[s, feat[gi]]
                go_left = (miss[gi] != 0) if np.isnan(fv) else (fv <= thr[gi])
                cur = left[gi] if go_left else right[gi]
            out[s, t] = cur  # LOCAL index, matching numpy's `current`. NO guard (numpy doesn't raise here).
    return out
