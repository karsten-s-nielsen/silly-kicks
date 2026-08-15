"""One-off: the migrated xCross predicate is VALUE-IDENTICAL to the expression it replaces.

Characterization only. It compares against a literal copy of the pre-migration expression, which
ceases to exist in the source after this migration -- so it is deliberately temporary. The DURABLE
guarantee lives in ``test_geometry_box_predicate_parity.py``.

If this shows any delta, ADR-050 §6's premise is wrong and the migration STOPS: the whole reason
xCross needs no re-stamp is that ``feature_contract()`` hashes VALUES and declared CONSTANTS, never
source, so a value-identical migration cannot move the fingerprint.
"""

from __future__ import annotations

import numpy as np

import silly_kicks.spadl.config as _spc
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._geometry import in_penalty_area_goal_relative_array

_BOX_DEPTH_M = _spc.penalty_area_depth
_BOX_HALF_WIDTH_M = _spc.penalty_area_half_width


def _legacy(gr_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Verbatim copy of the pre-migration ``_xcross_attempt.py`` predicate, minus ``& ~is_ball``."""
    return (gr_x <= _BOX_DEPTH_M) & (np.abs(y - _geo.GOAL_Y) <= _BOX_HALF_WIDTH_M)


def _dense_grid() -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-10.0, 30.0, 401)
    ys = np.linspace(0.0, 68.0, 681)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def test_migration_is_value_identical_over_a_dense_grid():
    gx, gy = _dense_grid()
    assert np.array_equal(_legacy(gx, gy), in_penalty_area_goal_relative_array(gx, gy))


def test_grid_covers_both_outcomes():
    """A grid that is all-True or all-False would make the identity above vacuous."""
    gx, gy = _dense_grid()
    out = _legacy(gx, gy)
    assert out.any() and not out.all()
