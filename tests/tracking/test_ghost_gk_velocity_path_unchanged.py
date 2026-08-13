"""The cycle's premise: ghost positions on velocity-bearing frames do not move.

Captured on the unmodified tree BEFORE the refusal lands. If this test ever fails, the
degradation cycle has become a retrain trigger and the scope decision must be revisited
rather than absorbed.

**RE-CAPTURED at the ghost box-constant re-fit -- revisited, not absorbed.** The baseline above
guards the DEGRADATION cycle: it asserts that adding the velocity refusal did not move served
positions. What moved them here is a different cause entirely -- a DECLARED re-fit of the bundled
weights onto the canonical penalty-area constant -- which is the one condition under which this
baseline is expected to move, and absorbing it silently would have been the failure the docstring
warns about.

Measured effect of that re-fit on this fixture, recorded so the next reader does not have to
re-derive it -- pre-re-fit versus the SHIPPED weights, 6 rows, all finite: **max |dx| 0.2835 m,
max |dy| 1.5804 m, mean 0.4831 m, median 0.2511 m**. The baseline now pins the POST-re-fit
positions, so the tripwire still does its original job: any future degradation-class change that
moves these numbers fails again.

(The re-fit was performed twice -- once under scikit-learn 1.7.2, then again under 1.9.0 after the
training environment was pinned to Python 3.12 -- and these numbers are against the FINAL 1.9.0
weights. The intermediate capture is not recorded here on purpose: a baseline that pins weights
which were never shipped documents nothing.)

Corollary worth keeping: this is the ONLY committed golden that pins bundled-model OUTPUT.
``ghost_gk_kde_golden.npz`` stores input FEATURES (outputs are computed fresh) and
``ghost_gk_refactor_golden.npz`` uses locally-fit models, so neither moves on a re-fit -- verified
at this one, where exactly one golden failed and the other two were correctly unaffected.
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
