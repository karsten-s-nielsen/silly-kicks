"""Structural perf guard (ADR-068): PitchControlSurface caches its interpolator, so repeated
at_point/at_points queries on one (cached) surface build RegularGridInterpolator ONCE."""

import numpy as np
import scipy.interpolate as _si

from silly_kicks.tracking.pitch_control._surface import PitchControlSurface
from tests._perf_structural import call_counter


def _surface():
    gx = np.linspace(0.0, 105.0, 10)
    gy = np.linspace(0.0, 68.0, 8)
    surf = np.linspace(0.0, 1.0, gx.size * gy.size).reshape(gy.size, gx.size)
    return PitchControlSurface(grid_x=gx, grid_y=gy, surface=surf, method="spearman", attacking_team_id=1)


def test_interpolator_built_once_across_queries(monkeypatch):
    s = _surface()
    calls = call_counter(monkeypatch, _si, "RegularGridInterpolator")
    v1 = s.at_point(50.0, 34.0)
    v2 = s.at_point(80.0, 20.0)
    _ = s.at_points(np.array([[10.0, 10.0], [90.0, 60.0]]))
    assert calls["n"] == 1  # cached; pre-ADR-068 this was one build PER call
    assert 0.0 <= v1 <= 1.0 and 0.0 <= v2 <= 1.0


def test_cached_interpolation_is_unchanged(monkeypatch):
    # Value parity: the cached interpolator returns the same values a fresh one would.
    s = _surface()
    fresh = _si.RegularGridInterpolator(
        (s.grid_y, s.grid_x),
        s.surface,
        method="linear",
        bounds_error=False,
        fill_value=None,  # type: ignore[arg-type]
    )
    for x, y in [(50.0, 34.0), (0.0, 0.0), (105.0, 68.0), (12.3, 45.6)]:
        expected = float(np.clip(fresh(np.array([[y, x]]))[0], 0.0, 1.0))
        assert s.at_point(x, y) == expected
