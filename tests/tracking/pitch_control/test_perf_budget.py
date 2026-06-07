"""Structural performance guard for compute_pitch_control (TF-7).

Replaces flaky wall-clock budgets (50ms/75ms single-frame ceilings) with a deterministic
invariant: the per-team influence kernel runs O(teams) times — ONE vectorised pass over the
whole grid per team — NOT once per grid cell. A regression to a per-cell Python loop is the
real blow-up the ms-budget proxied; it would call the kernel ~grid-cells (>1000) times.

Vectorisation *correctness* (numba == numpy, per-method values) is covered by test_numba_parity.py
+ test_spearman.py / test_fernandez_bornn.py / test_voronoi.py; this file guards the cost contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking.pitch_control._fernandez_bornn as _fb
import silly_kicks.tracking.pitch_control._spearman as _sp
from silly_kicks.tracking.pitch_control import compute_pitch_control
from tests._perf_structural import call_counter


@pytest.fixture(scope="module")
def full_frame_22():
    """Realistic 22-player frame (11v11 + ball)."""
    rng = np.random.default_rng(777)
    rows = []
    for i in range(11):
        rows.append(
            {
                "player_id": 100 + i,
                "team_id": 1,
                "x": rng.uniform(5, 100),
                "y": rng.uniform(5, 63),
                "vx": rng.uniform(-3, 3),
                "vy": rng.uniform(-3, 3),
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    for i in range(11):
        rows.append(
            {
                "player_id": 200 + i,
                "team_id": 2,
                "x": rng.uniform(5, 100),
                "y": rng.uniform(5, 63),
                "vx": rng.uniform(-3, 3),
                "vy": rng.uniform(-3, 3),
                "is_ball": False,
                "is_goalkeeper": i == 0,
            }
        )
    rows.append(
        {
            "player_id": np.nan,
            "team_id": np.nan,
            "x": 52.5,
            "y": 34.0,
            "vx": 0,
            "vy": 0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


def test_spearman_influence_kernel_is_grid_vectorised(full_frame_22, monkeypatch) -> None:
    """Spearman: ``_compute_influence`` runs once per team (2), not once per grid cell."""
    calls = call_counter(monkeypatch, _sp, "_compute_influence")

    result = compute_pitch_control(full_frame_22, 1, method="spearman")

    n_cells = result.surface.size
    assert result.surface.shape[0] > 0
    assert calls["n"] == 2, (
        f"Spearman ran the influence kernel {calls['n']}x for a {n_cells}-cell grid (expected 2, "
        "one vectorised pass per team). A per-cell regression makes this scale with grid cells."
    )


def test_fernandez_bornn_influence_kernel_is_grid_vectorised(full_frame_22, monkeypatch) -> None:
    """Fernandez-Bornn: ``_compute_gaussian_influence`` runs once for the whole grid, not per cell."""
    calls = call_counter(monkeypatch, _fb, "_compute_gaussian_influence")

    result = compute_pitch_control(full_frame_22, 1, method="fernandez_bornn")

    n_cells = result.surface.size
    assert result.surface.shape[0] > 0
    assert calls["n"] == 1, (
        f"Fernandez-Bornn ran the influence kernel {calls['n']}x for a {n_cells}-cell grid "
        "(expected 1 vectorised pass). A per-cell regression makes this scale with grid cells."
    )


def test_voronoi_returns_full_grid_surface(full_frame_22) -> None:
    """Voronoi: scipy-vectorised tessellation yields the full 2-D control grid (no per-cell loop to
    regress; this locks the surface shape so a degenerate scalar/empty return is caught)."""
    result = compute_pitch_control(full_frame_22, 1, method="voronoi")

    assert result.surface.ndim == 2
    assert result.surface.shape[0] > 0 and result.surface.shape[1] > 0
    assert np.all(np.isfinite(result.surface))
