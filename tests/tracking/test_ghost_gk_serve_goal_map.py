"""serve_ghost_gk_positions(goal_map=): inject a whole-unit GoalMap for the frame-coordinate flip.

``serve_ghost_gk_positions`` derives the defended-goal end from the frames to decide whether the
model's goal-relative ghost position is mirrored into frame coordinates (``ghost_x`` = flip iff the
defended goal is at high x). Under the lakehouse's windowed Stage-2 scoring that derivation would run
per window; ``goal_map=`` lets the caller inject the whole-unit map so the flip is stable across
windows. ``goal_map=None`` recomputes internally -> byte-identical to the pre-parameter behaviour.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, resolve_defended_goals, serve_ghost_gk_positions
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def _serve(frames: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = serve_ghost_gk_positions(frames, model=_fitted_model()[0], home_team_id=1, **kwargs)
    return out.sort_values("gk_team_id").reset_index(drop=True)


def _invert(gm: GoalMap) -> GoalMap:
    fl = float(spadlconfig.field_length)

    def _swap(m) -> MappingProxyType:
        return MappingProxyType({k: fl - v for k, v in m.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def test_goal_map_none_is_byte_identical_to_the_whole_frame_recompute():
    frames = _make_ghost_gk_frames()
    whole = resolve_defended_goals(frames)

    default = _serve(frames)
    none = _serve(frames, goal_map=None)
    inject = _serve(frames, goal_map=whole)

    assert len(default) == 2, "one GK per team expected (non-vacuity)"
    assert none.equals(default)
    assert inject.equals(default)


def test_goal_map_injection_flips_the_frame_coordinates():
    frames = _make_ghost_gk_frames()
    inverted = _invert(resolve_defended_goals(frames))

    default = _serve(frames)
    flipped = _serve(frames, goal_map=inverted)

    assert len(default) == 2, "one GK per team expected (non-vacuity)"
    # The goal-RELATIVE position is the model output and does not depend on the goal map.
    assert np.allclose(default["ghost_gr_x"].to_numpy(), flipped["ghost_gr_x"].to_numpy())
    # The frame-coordinate x is the flipped projection -> inverting the defended ends must move it.
    assert not np.allclose(default["ghost_x"].to_numpy(), flipped["ghost_x"].to_numpy())
