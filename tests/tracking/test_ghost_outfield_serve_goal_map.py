"""serve_ghost_outfield_positions(goal_map=): inject a whole-unit GoalMap for the frame-coord flip.

Sibling of the ghost-GK serve seam. The served goal-relative rearguard positions are projected to
frame coordinates by flipping iff the defending end is at high x; that end is resolved from the frames,
so a windowed caller must inject the whole-unit map to keep the flip stable. ``goal_map=None`` recomputes
internally -> byte-identical to the pre-parameter behaviour.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, resolve_defended_goals, serve_ghost_outfield_positions
from tests.tracking.test_ghost_outfield_model import _fit_toy, _toy_actions


def _one():
    model, frames = _fit_toy()
    one = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].copy()
    return model, one


def _serve(model, one: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = serve_ghost_outfield_positions(one, model=model, home_team_id=1, actions=_toy_actions(), **kwargs)
    return out.sort_values("slot_index").reset_index(drop=True)


def _invert(gm: GoalMap) -> GoalMap:
    fl = float(spadlconfig.field_length)

    def _swap(m) -> MappingProxyType:
        return MappingProxyType({k: fl - v for k, v in m.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def test_goal_map_none_is_byte_identical_to_the_whole_frame_recompute():
    model, one = _one()
    whole = resolve_defended_goals(one)
    default = _serve(model, one)
    assert len(default) == 4, "four rearguard slots expected (non-vacuity)"
    assert _serve(model, one, goal_map=None).equals(default)
    assert _serve(model, one, goal_map=whole).equals(default)


def test_goal_map_injection_flips_the_frame_coordinates():
    model, one = _one()
    inverted = _invert(resolve_defended_goals(one))
    default = _serve(model, one)
    flipped = _serve(model, one, goal_map=inverted)
    assert len(default) == 4
    # goal-relative = model output, independent of the goal map.
    assert np.allclose(default["ghost_gr_x"].to_numpy(), flipped["ghost_gr_x"].to_numpy())
    # frame-coordinate x is the flipped projection -> inverting the defended ends must move it.
    assert not np.allclose(default["ghost_x"].to_numpy(), flipped["ghost_x"].to_numpy())
