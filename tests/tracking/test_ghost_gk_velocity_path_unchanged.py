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

**RE-CAPTURED AGAIN at the ADR-067 native-SkillCorner re-fit (position-only cycle) -- revisited, not
absorbed.** The bundled ghost ``default`` was re-fit onto the NATIVE SkillCorner corpus (removing the
kloppy y-inversion contamination; ``training_commit=a0fc9f9``), which is again a DECLARED re-fit of
the bundled weights -- the one condition under which this baseline is expected to move. Measured
effect on this fixture (``sb360-fixture-2``), the prior box-constant kloppy ``default`` (de8ca604)
versus the SHIPPED native weights, 6 rows, all finite: **max |dx| 0.2029 m, max |dy| 0.8237 m,
mean 0.2218 m, median 0.1608 m**. The baseline now pins the POST-native-re-fit positions, so the
tripwire still fails on any future degradation-class change that moves them.

**RE-CAPTURED AGAIN at the ADR-089 both-axes convention re-fit (TF-60 Layer-3) -- revisited, not
absorbed.** All five bundled ghost-GK variants were re-fit after the goal-relative feature transform
was unified from x-only to the correct both-axes 180-degree point reflection (signed-y features + the
target ``gk_y`` now flip for a defended goal at high x; ADR-089/ADR-051 8b). Because the model's
served ``gr_y`` became goal-relative rather than absolute-frame y, ``add_ghost_gk``'s action-LTR
reprojection changed from a flip-GATED ``y -> 68 - gr_y`` (away only) to a UNIFORM ``y -> 68 - gr_y``:
the keeper's goal-relative flip is the complement of the acting team's action flip, so the per-action
reflection cancels against the model's own and both axes reproject uniformly. (The flip-gated form
double-flipped the flip=False rows once the model's y became goal-relative -- caught by
``test_ghost_gk_mirror_invariant``, which was transient-red on the chirality mismatch during Phase A
and so had not yet exercised its assertion.) This is a DECLARED re-fit + reprojection correction -- the
condition under which this baseline is expected to move. Measured effect on this fixture
(``sb360-fixture-2``), the prior ADR-067 native ``default`` (a0fc9f9) versus the SHIPPED both-axes
weights (``training_commit=22678fd``) with the corrected reprojection, 6 rows, all finite:
**max |dx| 0.5631 m, max |dy| 2.3927 m, mean 0.7452 m, median 0.2735 m** -- the small x delta and the
larger y delta are the both-axes signature. The baseline pins the POST-both-axes positions; the model
``gr_y`` is bit-identical across numpy 2.2.6 (py3.10) and 2.4.2 (py3.12+) (verified) and the uniform
``68 - gr_y`` is a deterministic transform of it, so ``assert_array_equal`` holds on every CI leg.

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
