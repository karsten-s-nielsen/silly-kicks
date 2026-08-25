"""Structural perf guard (ADR-068): the pitch grid + targets are memoized per (grid_cells_x,
grid_cells_y), so computing multiple same-dimension surfaces builds np.meshgrid ONCE."""

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control._grids import pitch_grid
from silly_kicks.tracking.pitch_control._params import SpearmanParams
from silly_kicks.tracking.pitch_control._spearman import compute_spearman
from tests._perf_structural import call_counter


def _frame():
    return pd.DataFrame(
        {
            "is_ball": [False, False, False, False, True],
            "is_goalkeeper": [True, False, True, False, False],
            "team_id": [1, 1, 2, 2, np.nan],
            "player_id": [10, 11, 20, 21, np.nan],
            "x": [30.0, 60.0, 45.0, 70.0, 50.0],
            "y": [34.0, 20.0, 40.0, 34.0, 34.0],
            "vx": [0.0, 0.0, 0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_meshgrid_built_once_across_same_dim_surfaces(monkeypatch):
    pitch_grid.cache_clear()  # deterministic: start cold
    calls = call_counter(monkeypatch, np, "meshgrid")
    params = SpearmanParams()
    s1 = compute_spearman(_frame(), attacking_team_id=1, params=params)
    s2 = compute_spearman(_frame(), attacking_team_id=1, params=params)
    assert calls["n"] == 1  # cached; pre-ADR-068 meshgrid was rebuilt on every surface
    assert s1.surface.shape == s2.surface.shape
    pitch_grid.cache_clear()  # leave the cache clean for other tests


def test_cached_grid_is_read_only_and_correct():
    pitch_grid.cache_clear()
    gx, gy, targets = pitch_grid(10, 8)
    assert not gx.flags.writeable and not gy.flags.writeable and not targets.flags.writeable
    # byte-identical to the direct construction it replaced
    egx, egy = np.meshgrid(np.linspace(0.0, 105.0, 10), np.linspace(0.0, 68.0, 8))
    expected = np.column_stack([egx.ravel(), egy.ravel()])
    assert np.array_equal(targets, expected)
    pitch_grid.cache_clear()
