"""build_ghost_frames(goal_map=): inject a whole-unit GoalMap for windowed Stage-2 scoring.

The lakehouse windows Stage-2 scoring to stay under the serverless UDF-group memory cap, but the
defended-goal end is a WHOLE-UNIT estimator (mean GK x per game/period/team). Resolved per window it
is a smaller-sample estimator that disagrees with the whole-unit value on ~7.1% of skillcorner
team-frames (``tracking/_gk_resolve.py``), which would change the ghost delta. ``goal_map=`` lets the
caller pin the whole-unit map once and inject the SAME value into every window.

``goal_map=None`` recomputes internally -> byte-identical to the pre-parameter behaviour. These tests
pin both halves of the contract: the default path is unchanged, and the parameter is load-bearing
(not silently ignored) -- the gkdv engine recomputes the goal map in TWO places (the domain filter
and the ghost-serving flip), so a param that only reached the first would still drift the positions.
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd

from silly_kicks.gkdv import build_ghost_frames
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, resolve_defended_goals
from tests.gkdv._fixtures import in_domain_frames
from tests.tracking.test_ghost_gk import _fitted_model


def _build(frames: pd.DataFrame, **kwargs):
    return build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1, **kwargs)


def _invert(gm: GoalMap) -> GoalMap:
    """A GoalMap with every defended end reflected (0 <-> field_length), same keys/provenance.

    Reflecting both ends keeps the two-team structure intact (so ``attacked_goal`` still resolves),
    which is exactly the toggle that flips the ghost-serving mirror -- a non-vacuous stand-in for a
    window whose mean-GK-x map disagrees with the whole-unit one.
    """
    fl = float(spadlconfig.field_length)

    def _swap(m) -> MappingProxyType:
        return MappingProxyType({k: fl - v for k, v in m.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def test_goal_map_none_is_byte_identical_to_the_whole_unit_recompute():
    frames = in_domain_frames()
    whole_unit = resolve_defended_goals(frames)

    cf_default, prov_default, report_default = _build(frames)
    cf_none, prov_none, _ = _build(frames, goal_map=None)
    cf_inject, prov_inject, _ = _build(frames, goal_map=whole_unit)

    # Non-vacuity: the default path actually scores the frame these assertions are about.
    assert report_default.n_frames_scored == 1, report_default.drop_reasons

    # None -> internal recompute == omitting the argument.
    assert cf_none.equals(cf_default)
    assert prov_none.equals(prov_default)
    # Injecting the SAME whole-unit map the engine would have recomputed is a no-op (both the
    # domain filter and the serving flip must consume it, or this would diverge).
    assert cf_inject.equals(cf_default)
    assert prov_inject.equals(prov_default)


def test_goal_map_is_load_bearing_not_silently_ignored():
    frames = in_domain_frames()
    inverted = _invert(resolve_defended_goals(frames))

    _cf_default, prov_default, report_default = _build(frames)
    _cf_inverted, prov_inverted, _ = _build(frames, goal_map=inverted)

    assert report_default.n_frames_scored == 1, report_default.drop_reasons
    # A goal map pinning the opposite ends MUST change the output; if the parameter were ignored
    # the provenance (served ghost positions + domain membership) would be identical.
    assert not prov_inverted.equals(prov_default)
