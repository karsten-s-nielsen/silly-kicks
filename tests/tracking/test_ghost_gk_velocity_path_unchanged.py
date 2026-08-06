"""The cycle's premise: ghost positions on velocity-bearing frames do not move.

Captured on the unmodified tree BEFORE the refusal lands. If this test ever fails, the
degradation cycle has become a retrain trigger and the scope decision must be revisited
rather than absorbed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_b

_BASELINE = pathlib.Path(__file__).parent / "data" / "ghost_velocity_path_baseline.npz"


def _serve():
    actions, frames, _links = build_leg_b()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    return out[["ghost_gk_x", "ghost_gk_y"]].to_numpy(dtype=float)


@pytest.mark.skipif(not _BASELINE.is_file(), reason="baseline not captured yet")
def test_velocity_path_positions_are_unchanged():
    ref = np.load(_BASELINE)["positions"]
    got = _serve()
    assert got.shape == ref.shape, f"row count changed: {got.shape} vs {ref.shape}"
    np.testing.assert_array_equal(got, ref)


def test_the_baseline_is_not_vacuous():
    """A baseline of all-NaN would make the assertion above pass while proving nothing."""
    got = _serve()
    assert len(got) > 0, "fixture produced no ghost rows"
    assert np.isfinite(got).all(), "velocity-bearing leg must produce finite ghosts"
