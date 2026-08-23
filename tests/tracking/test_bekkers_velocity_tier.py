"""Task 4 (ADR-063 amendment): bekkers_pi honest-NaN tier on velocity-unavailable frames.

bekkers_pi is velocity-derived (probabilistic TTI + a velocity-GATED active-pressing ``speed_threshold``
filter). Its zero-velocity form is artifact-dependent, NOT a smooth limit of the same model -> SUPPRESS
to honest-NaN on declared-velocity-unavailable (SB360) frames (Tier-3); keep the loud raise on a
forgotten ``derive_velocities()``. ``andrienko_oval`` / ``link_zones`` are positional and unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking.features import pressure_on_actor
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

# Reuse the real single-action/single-defender fixture (handles action<->frame linkage) so the
# assertions cannot fail for a linkage-inadequacy reason.
from tests.tracking.test_pressure_methods_invariants import _make_one_action_frame


def _scene():
    """A close, fast presser (speed = |v| = 3.0 m/s > speed_threshold 2.0) so bekkers_pi scores > 0."""
    return _make_one_action_frame((50.0, 34.0), (51.0, 34.0, 3.0, 0.0))


def _declared_unavailable(frames):
    """SB360 shape: drop vx/vy, mark EVERY row speed_source=unavailable."""
    out = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()
    out["speed"] = np.nan
    out["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    return out


def _undeclared_missing_velocity(frames):
    """Forgot-derive_velocities(): no vx/vy and NO speed_source marker."""
    return frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns]).copy()


def test_declared_unavailable_bekkers_is_nan_not_raise():
    actions, frames = _scene()
    out = pressure_on_actor(actions, _declared_unavailable(frames), method="bekkers_pi")
    assert out.isna().all()


def test_undeclared_missing_velocity_still_raises():
    actions, frames = _scene()
    with pytest.raises(ValueError, match="derive_velocities"):
        pressure_on_actor(actions, _undeclared_missing_velocity(frames), method="bekkers_pi")


def test_velocity_bearing_bekkers_is_scored():
    # NON-VACUITY for the declared-NaN test: the same scene WITH velocity scores > 0, so the NaN
    # above is suppression, not "nothing scores here".
    actions, frames = _scene()
    out = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert out.notna().any() and (out.abs() > 0).any()


def test_andrienko_unaffected_on_declared_frames():
    # The change is bekkers-only: the positional method still computes on velocity-less frames.
    actions, frames = _scene()
    out = pressure_on_actor(actions, _declared_unavailable(frames), method="andrienko_oval")
    assert out.notna().any()


def test_artifact_dependence_on_the_real_surface():
    """Drives the REAL _pressure_bekkers gate end-to-end (not a hand-rolled np.where): same positions
    + present vx/vy, defender speed BELOW vs ABOVE the active-pressing threshold -> the discrete gate
    materially moves the real output, so the zero-velocity form is gate-dependent (Tier-3), not a
    smooth limit. If the two regimes ever coincide, the Tier-3 decision must be revisited."""
    actions, frames = _scene()
    below = frames.copy()
    below["speed"] = 0.0  # presser below threshold -> filtered out
    above = frames.copy()
    above["speed"] = 10.0  # presser above threshold -> counted
    s_below = pressure_on_actor(actions, below, method="bekkers_pi")
    s_above = pressure_on_actor(actions, above, method="bekkers_pi")
    assert not np.allclose(s_below.fillna(0.0).to_numpy(), s_above.fillna(0.0).to_numpy())
    assert (s_above.fillna(0.0) >= s_below.fillna(0.0)).all()


def test_atomic_declared_unavailable_bekkers_is_nan_not_raise():
    # ADR-063 amendment: atomic.tracking.features.pressure_on_actor is a SEPARATE re-implementation
    # (not a delegation), so it carries its own copy of the Tier-3 seam. Without it, atomic bekkers on
    # a declared-velocity-less SB360 frame would raise the impossible-on-a-freeze-frame ValueError.
    from silly_kicks.atomic.tracking.features import pressure_on_actor as atomic_pressure_on_actor

    actions, frames = _scene()
    actions = actions.rename(columns={"start_x": "x", "start_y": "y"})  # atomic uses (x, y) anchors
    out = atomic_pressure_on_actor(actions, _declared_unavailable(frames), method="bekkers_pi")
    assert out.isna().all()
