"""Cached pitch grid + target array (ADR-068).

``grid_x`` / ``grid_y`` / ``targets`` depend only on the cell counts, yet all three pitch-control
backends (``_spearman`` / ``_fernandez_bornn`` / ``_voronoi``) rebuilt them (two ``np.linspace`` + a
``np.meshgrid`` + a ``column_stack``) on EVERY per-frame surface computation. They are hoisted here
and memoized per ``(grid_cells_x, grid_cells_y)`` -- a tiny config domain, so the bounded cache is
the ``@functools.cache``-on-config idiom, not an unbounded growth risk.
"""

from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=16)
def pitch_grid(grid_cells_x: int, grid_cells_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return READ-ONLY ``(grid_x, grid_y, targets)`` for the given cell counts.

    ``grid_x`` (nx,) in [0, 105]; ``grid_y`` (ny,) in [0, 68]; ``targets`` (nx*ny, 2) is the raveled
    ``meshgrid(grid_x, grid_y)`` -- byte-identical to the per-backend construction it replaces. The
    arrays are read-only so callers cannot mutate the shared cached objects (``PitchControlSurface``
    treats its grid arrays as immutable regardless).
    """
    grid_x = np.linspace(0.0, 105.0, grid_cells_x)
    grid_y = np.linspace(0.0, 68.0, grid_cells_y)
    gx, gy = np.meshgrid(grid_x, grid_y)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    for arr in (grid_x, grid_y, targets):
        arr.setflags(write=False)
    return grid_x, grid_y, targets
