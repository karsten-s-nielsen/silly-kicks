"""build_restdefense_ghost_frames(goal_map=): inject a whole-unit GoalMap into BOTH ghost serves.

Mirrors the gkdv seam. The function recomputes the defended-goal map for the domain filter AND each of
its ghost serves (``serve_ghost_outfield_positions`` for the rearguard, ``serve_ghost_gk_positions`` for
the keeper) re-resolves it again for the frame-coordinate flip. A windowed caller injects the whole-unit
map once so all three stay consistent; ``goal_map=None`` recomputes internally (byte-identical).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

from silly_kicks.restdefense._counterfactual import build_restdefense_ghost_frames
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, resolve_defended_goals
from tests.restdefense.test_counterfactual import _carrier, _committed_frame
from tests.tracking.test_ghost_gk import _fitted_model
from tests.tracking.test_ghost_outfield_model import _fit_toy


def _invert(gm: GoalMap) -> GoalMap:
    fl = float(spadlconfig.field_length)

    def _swap(m) -> MappingProxyType:
        return MappingProxyType({k: fl - v for k, v in m.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def _build(which: Literal["keeper", "rearguard"], model, **kwargs):
    frames = _committed_frame()
    return build_restdefense_ghost_frames(
        frames, which=which, model=model, home_team_id=1, carrier=_carrier(), **kwargs
    )


def test_rearguard_goal_map_none_is_byte_identical():
    model = _fit_toy()[0]
    whole = resolve_defended_goals(_committed_frame())
    cf_d, prov_d, rep_d = _build("rearguard", model)
    cf_n, prov_n, _ = _build("rearguard", model, goal_map=None)
    cf_i, prov_i, _ = _build("rearguard", model, goal_map=whole)
    assert rep_d.n_frames_scored >= 1, rep_d.drop_reasons
    assert cf_n.equals(cf_d) and prov_n.equals(prov_d)
    assert cf_i.equals(cf_d) and prov_i.equals(prov_d)


def test_rearguard_goal_map_is_load_bearing():
    model = _fit_toy()[0]
    inverted = _invert(resolve_defended_goals(_committed_frame()))
    _cf_d, prov_d, rep_d = _build("rearguard", model)
    _cf_i, prov_i, _ = _build("rearguard", model, goal_map=inverted)
    assert rep_d.n_frames_scored >= 1, rep_d.drop_reasons
    assert not prov_i.equals(prov_d)


def test_keeper_goal_map_none_is_byte_identical():
    model = _fitted_model()[0]
    whole = resolve_defended_goals(_committed_frame())
    cf_d, prov_d, rep_d = _build("keeper", model)
    cf_n, prov_n, _ = _build("keeper", model, goal_map=None)
    cf_i, prov_i, _ = _build("keeper", model, goal_map=whole)
    assert rep_d.n_frames_scored >= 1, rep_d.drop_reasons
    assert cf_n.equals(cf_d) and prov_n.equals(prov_d)
    assert cf_i.equals(cf_d) and prov_i.equals(prov_d)


def test_keeper_goal_map_is_load_bearing():
    model = _fitted_model()[0]
    inverted = _invert(resolve_defended_goals(_committed_frame()))
    _cf_d, prov_d, rep_d = _build("keeper", model)
    _cf_i, prov_i, _ = _build("keeper", model, goal_map=inverted)
    assert rep_d.n_frames_scored >= 1, rep_d.drop_reasons
    assert not prov_i.equals(prov_d)
