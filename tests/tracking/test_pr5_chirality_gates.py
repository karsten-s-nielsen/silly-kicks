"""PR 5 gates -- the goal-relative transform must be a 180-degree POINT REFLECTION.

``_geometry.py`` had ``to_goal_relative_x``/``_vx`` and no ``to_goal_relative_y``, so ``goal_x=105``
was an x-only mirror (determinant -1) while ``goal_x=0`` was the identity (+1): the two goal ends
used frames of OPPOSITE HANDEDNESS. Every RADIAL feature stayed byte-identical and every BEARING
negated, which is why it survived -- distances and radii all agree. Measured pre-fix on this
fixture: xS 12 of 27 features flip sign, xCross 3 of 16.

The gates here are deliberately landed RED before the fix (ADR-051's detection-first rule): a gate
written after its own repair arrives green and is never observed failing.

FIXTURE NOTE: these use a PR-5-local ``pr5_scene()``, NOT the shared ``canonical_scene()``. That
fixture is referenced across 8 of the 10 ``_mirror_entries`` modules and several pin MEASURED
tolerances to it (``_DAS_MIRROR_TOL = 15.0`` justified by a measured 12.0349; pitch-control's
7.45e-20; ``defensive_line_x moves 23.75 m``). Adding players to it would change nearest-defender
distance, pitch control, DAS, team shape and line detection for every entry, and the xfail ledger
this cycle is calibrated on is measured on the current scene.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._xcross_attempt import extract_xcross_features
from silly_kicks.tracking._xshot_occurrence import extract_xshot_features
from tests.tracking._mirror_registry import AWAY, HOME, canonical_scene, mirror_frames

#: Measured on ``canonical_scene()``: the away/defending team is 2 and the carrier is player 11,
#: 2.83 m from the ball at (38, 23). Hard-coded WITH assertions rather than re-derived by a
#: heuristic -- a heuristic that silently picks the wrong team produces a plausible number, which
#: is this cycle's signature failure. A renumbered upstream fixture fails loudly instead.
CARRIER = 11


def pr5_scene() -> pd.DataFrame:
    """One y-asymmetric frame snapshot with a shadowing defender and an attacker in the box.

    Derived from ``canonical_scene()``'s first frame so the 20-column schema is inherited rather
    than re-invented, then augmented on a COPY -- ``canonical_scene`` is ``functools.cache``d and
    must never be mutated.

    The two added players exist to close a measured vacuity: on the bare fixture xS ``openGoal`` is
    saturated at 1.0 and xCross ``box_off_def_ratio`` is 0.0 in BOTH legs, so the identity assertion
    proves nothing about them.
    """
    _actions, frames = canonical_scene()
    fid = frames["frame_id"].min()
    f = frames[frames["frame_id"] == fid].reset_index(drop=True).copy()

    assert (f["team_id"] == AWAY).any(), "AWAY id absent -- upstream fixture renumbered"
    assert (f["player_id"] == CARRIER).any(), "carrier absent -- upstream fixture renumbered"

    template = f[~f["is_ball"]].iloc[0]

    def _row(pid, team, x, y, direction):
        r = template.copy()
        r["player_id"], r["team_id"] = pid, team
        r["x"], r["y"] = float(x), float(y)
        r["is_goalkeeper"] = False
        r["team_attacking_direction"] = direction
        return r

    extra = pd.DataFrame(
        [
            # Defender on the ball->goal-centre segment: ball (38, 23) -> goal (105, 34), so the
            # midpoint is (71.5, 28.5). Drops openGoal below its saturated 1.0.
            _row(901, AWAY, 71.5, 28.5, "rtl"),
            # Attacker inside the attacked penalty area (gr_x <= 16.5 => x >= 88.5;
            # |y - 34| <= 20.16). Lifts box_off_def_ratio above 0.
            _row(902, HOME, 96.0, 30.0, "ltr"),
        ]
    )
    return pd.concat([f, extra], ignore_index=True)


def _xs(frame: pd.DataFrame, goal_x: float) -> pd.Series:
    return extract_xshot_features(frame, gk_team_id=AWAY, goal_x=goal_x).iloc[0]


def _xc(frame: pd.DataFrame, goal_x: float) -> pd.Series:
    return extract_xcross_features(
        frame, gk_team_id=AWAY, goal_x=goal_x, carrier_player_id=CARRIER, score_differential=0.0
    ).iloc[0]


def _lr_mirror(frame: pd.DataFrame) -> pd.DataFrame:
    """``y -> 68 - y`` at a FIXED goal end. NOT the point reflection: x is untouched."""
    out = frame.copy()
    out["y"] = _geo.PITCH_WIDTH - out["y"].to_numpy(dtype=float)
    if "vy" in out.columns:
        out["vy"] = -out["vy"].to_numpy(dtype=float)
    return out


def _flip_report(extract_fn) -> tuple[int, list[str], float]:
    """Same physical scene, both goal ends. Post-fix every delta must be exactly 0.

    Returns ``(n_compared, sign_flipped_names, worst_abs_delta)``. ``n_compared`` exists to be
    ASSERTED: if a feature goes NaN in either leg it silently leaves the comparison, and without
    that count the two assertions below pass vacuously on a shrunken set.
    """
    base = extract_fn(pr5_scene(), 105.0)
    mirrored = extract_fn(mirror_frames(pr5_scene()), 0.0)
    both_finite = [k for k in base.index if np.isfinite(float(base[k])) and np.isfinite(float(mirrored[k]))]
    flips = [
        k
        for k in both_finite
        if float(base[k]) != 0.0 and float(base[k]) == pytest.approx(-float(mirrored[k]), rel=1e-9)
    ]
    worst = max(abs(float(base[k]) - float(mirrored[k])) for k in both_finite)
    return len(both_finite), flips, worst


def test_pr5_scene_exercises_the_clamped_and_box_features():
    """The two features the local fixture exists for must be non-degenerate in the base leg."""
    scene = pr5_scene()
    assert 0.0 < float(_xs(scene, 105.0)["openGoal"]) < 1.0
    assert float(_xc(scene, 105.0)["box_off_def_ratio"]) > 0.0


@pytest.mark.parametrize(("extract_fn", "n_features"), [(_xs, 27), (_xc, 16)])
def test_features_identical_under_point_reflection(extract_fn, n_features):
    """Gate 2. The reflection is a ROTATION (x->105-x AND y->68-y), not an x mirror.

    The word is load-bearing: an x-only mirror yields IDENTICAL features under the chiral transform
    and DIFFERENT ones under the fixed one, so a gate built from "physically mirrored" read
    literally would pass today and fail after the fix.
    """
    n_compared, flips, worst = _flip_report(extract_fn)
    assert n_compared == n_features, f"compared {n_compared} of {n_features} -- fixture regressed"
    assert flips == [], flips
    assert worst == pytest.approx(0.0, abs=1e-9)


def test_gate2_would_fail_under_the_chiral_transform(monkeypatch):
    """Gate 3, the permanent non-vacuity partner.

    Observing red once proves the gate today; this keeps it proven AFTER the fix, when the natural
    red signal is gone. Plants the pre-fix behaviour and requires gate 2 to notice.
    """
    monkeypatch.setattr(_geo, "to_goal_relative_y", lambda y, *, goal_x: y)
    n_compared, _flips, worst = _flip_report(_xc)
    assert n_compared == 16, f"compared {n_compared} of 16 -- the plant shrank the comparison"
    assert worst > 1e-6, "planting the chiral transform moved nothing -- gate 2 is vacuous"


@pytest.mark.parametrize(("length", "axis"), [(_geo.PITCH_LENGTH, "x"), (_geo.PITCH_WIDTH, "y")])
def test_grid_centres_are_mirror_symmetric(length, axis):
    """Gate 4a. NOTE: the x half is ALREADY green today -- only y lands RED.

    Recorded so "landed red" is not read as covering both axes: ADR-051's detection-first rule is
    about OBSERVING failure, and half of this gate structurally cannot fail. It is kept as a
    regression guard, since a future `res` change can break x instead (at res=2.0 it is x that
    becomes asymmetric).
    """
    from silly_kicks.tracking import _xcross_attempt as _xc_mod

    centres = _xc_mod._grid_centres(length, 3.0)
    assert set(np.round(centres, 9)) == set(np.round(length - centres, 9))


def test_dominant_region_is_left_right_mirror_invariant():
    """Gate 4b. Same goal end, scene mirrored left-right.

    The spec's 5.4% / 17.74 m^2 was measured on canonical_scene(), NOT this fixture -- do not quote
    it here. This scene's own pre-fix gap is recorded in the PR-5 plan once measured.
    """
    a = float(_xc(pr5_scene(), 105.0)["space_controlled"])
    b = float(_xc(_lr_mirror(pr5_scene()), 105.0)["space_controlled"])
    assert a == pytest.approx(b, abs=1e-9), f"{a} vs {b}"


def test_geometry_version_was_bumped_for_the_point_reflection():
    """Gate 5. Needs its own assertion because a forgotten bump FAILS NOTHING -- the
    geometry_version mismatch path only warns; the fail-closed prong is on pitch dims."""
    assert _geo.GEOMETRY_VERSION == "goal-relative-2"


def test_geometry_sentinel_still_differs_from_the_library_constant():
    """The sentinel in test_xshot_occurrence.py must create a REAL mismatch.

    Behavioural, not a substring search over source: CLAUDE.md is explicit that keyword tests over
    source are not evidence of behaviour, and a grep would pass on a comment. A gate that exists to
    prevent green-by-construction decay must not itself be green by construction.
    """
    from tests.tracking.test_xshot_occurrence import _GEOMETRY_SENTINEL

    assert _GEOMETRY_SENTINEL != _geo.GEOMETRY_VERSION
