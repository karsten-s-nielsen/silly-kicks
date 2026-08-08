"""compute_threat_pc: the public facade over the xT-weighted Voronoi threat integral.

FIXTURE PROVENANCE (plan defect, corrected here). The PR-3 plan told this module to import
``_frame`` and ``_fitted_xt`` from ``tests.tracking.test_cover_shadows``. **Neither exists.**
That module builds frames through ``tests.tracking._gk_test_helpers._make_two_team_frame``
and takes its xT from the session-scoped ``fitted_xt`` *pytest fixture* in
``tests/conftest.py`` -- a fixture cannot be called from another test module, and gkdv's
arm tests need a plain callable. So the two helpers are DEFINED here, on the same verified
anchor, and ``tests/gkdv/test_arms.py`` imports them from this module.

GEOMETRY IS LOAD-BEARING -- do not "simplify" the layout. Measured on this tree: with a
full 10-a-side layout, or with the ball close to the attacked goal, moving the defending
keeper changes the threat integral by EXACTLY 0.0. The keeper only registers when it is the
nearest defender to cells inside a dangerous receiver's Voronoi region: Spearman influence
is a logistic of (opponent-min-TTI - own-TTI), so a keeper screened by a nearer defender
underflows to exactly zero and stays there however far it moves. This layout therefore
leaves the deep zone to the keeper alone. A test written on a naive layout would pass
vacuously while measuring nothing about the keeper -- which is precisely the failure mode
the ghost-substitution arm exists to detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking import compute_threat_pc
from tests.tracking._gk_test_helpers import _make_two_team_frame
from tests.tracking._goal_map_helpers import goal_map_for

#: ADR-055 replaced ``home_team_id=1`` at this file's re-keyed call sites. Its frames carry
#: game 1 / period 1 with teams {1, 2} and each keeper at its own end, so this states exactly
#: what ``home_team_id=1`` meant and matches what ``resolve_defended_goals`` derives there.
HOME_GOAL_MAP = goal_map_for({1: 0.0, 2: 105.0})

#: Home (team 1) defends x=0 and is the DEFENDING team; away (team 2) attacks toward x=0.
#: No home outfielder sits deep, so the keeper is the only defender covering the space
#: between the dangerous receivers and the goal it defends.
_HOME_OUTFIELD = [(28.0, 28.0), (30.0, 40.0), (38.0, 34.0), (46.0, 30.0)]
_AWAY_OUTFIELD = [(14.0, 30.0), (14.0, 38.0), (22.0, 34.0), (34.0, 44.0)]
_BALL_XY = (30.0, 34.0)

#: The defending keeper covering its goalmouth.
GK_ON_LINE = (3.0, 34.0)
#: The same keeper abandoned 22 m upfield. Measured on this tree: the threat integral is
#: flat for any ghost x >= 13, so the planted contrast does not sit on a knife edge.
GK_OUT_OF_POSITION = (25.0, 34.0)


def _fitted_xt():
    """A fitted ExpectedThreat, identical to the ``fitted_xt`` conftest fixture.

    Duplicated as a plain callable because a pytest fixture cannot be invoked from another
    test module, and the gkdv arm tests need to build two frames' worth of xT themselves.
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _frame(gk_xy: tuple[float, float] = GK_ON_LINE):
    """A single keeper-sensitive frame with the home (defending) keeper at ``gk_xy``."""
    frame = _make_two_team_frame(
        home_positions=_HOME_OUTFIELD,
        away_positions=_AWAY_OUTFIELD,
        home_gk_pos=gk_xy,
        away_gk_pos=(100.0, 34.0),
    )
    ball = frame["is_ball"].astype(bool)
    frame.loc[ball, "x"] = _BALL_XY[0]
    frame.loc[ball, "y"] = _BALL_XY[1]
    return frame


def test_returns_a_finite_scalar():
    value = compute_threat_pc(_frame(), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    assert isinstance(value, float)
    assert np.isfinite(value)
    # A zero integral would make every downstream delta trivially zero.
    assert value > 0.0


def test_moving_the_keeper_changes_the_value():
    """NON-VACUITY: if this cannot move, the whole arm is dead."""
    on_line = compute_threat_pc(_frame(GK_ON_LINE), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    upfield = compute_threat_pc(
        _frame(GK_OUT_OF_POSITION), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP
    )
    assert on_line != upfield, "threat_pc is insensitive to keeper position -- arm would be vacuous"


def test_keeper_covering_its_goalmouth_suppresses_more_threat():
    """Directional anchor: the facade must respond with the physically right SIGN.

    ``test_moving_the_keeper_changes_the_value`` only proves the number moves; it would
    still pass if the response were inverted. This pins the direction the arm's polarity
    contract is built on.
    """
    on_line = compute_threat_pc(_frame(GK_ON_LINE), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    upfield = compute_threat_pc(
        _frame(GK_OUT_OF_POSITION), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP
    )
    assert on_line < upfield, (
        "a keeper covering its own goalmouth must leave the attackers LESS threat than the "
        "same keeper stranded 22 m upfield"
    )


def test_identical_frames_give_an_identical_value():
    frame = _frame()
    first = compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    second = compute_threat_pc(frame.copy(), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    assert first == second


def test_does_not_mutate_the_caller_frame():
    frame = _frame()
    before = frame.copy(deep=True)
    compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
    import pandas as pd

    pd.testing.assert_frame_equal(frame, before)


def test_rejects_frames_that_are_not_period_normalized():
    frame = _frame()
    frame["team_attacking_direction"] = "rtl"
    with pytest.raises(ValueError, match="compute_threat_pc"):
        compute_threat_pc(frame, attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)


def test_facade_forwards_the_method_to_pitch_control():
    """The facade must PASS ITS ``method`` THROUGH, not quietly pick one of its own.

    Written after a mutation test escaped: replacing the forwarded ``method`` with a
    hard-coded GK-blind ``"voronoi"`` left every other test in this module green. That is
    not a fixture weakness -- it is the documented behaviour of ``lambda_gk``, which is a
    GAIN applied AFTER the influence field is built, so keeper POSITION still enters
    through TTI under every method. Position-sensitivity tests therefore cannot see a
    method swap, and only a call-level assertion can.
    """
    import silly_kicks.tracking._cover_shadows as cover_shadows

    seen: list[str] = []
    real = cover_shadows.compute_pitch_control

    def _spy(frame, attacking_team_id, *, method, **kwargs):
        seen.append(method)
        return real(frame, attacking_team_id, method=method, **kwargs)

    original = cover_shadows.compute_pitch_control
    cover_shadows.compute_pitch_control = _spy
    try:
        compute_threat_pc(_frame(), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP)
        compute_threat_pc(_frame(), attacking_team_id=2, xt=_fitted_xt(), goal_map=HOME_GOAL_MAP, method="voronoi")
    finally:
        cover_shadows.compute_pitch_control = original

    assert seen == ["spearman", "voronoi"], (
        f"the facade did not forward its method verbatim (saw {seen}). lambda_gk exists ONLY "
        "on SpearmanParams, so a swapped method silently drops the keeper's control-rate "
        "multiplier while still looking position-sensitive."
    )


def test_accepts_no_pitch_control_cache():
    """The cache key excludes player positions, so caching would silently zero every delta."""
    import inspect

    assert "pitch_control_cache" not in inspect.signature(compute_threat_pc).parameters
