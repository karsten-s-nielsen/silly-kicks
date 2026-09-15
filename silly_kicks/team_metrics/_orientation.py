"""Bring the opponent's action rows into a team's action-LTR frame (opponent-relative KPIs).

SPADL is per-acting-team-LTR, so combining *our* rows and *the opponent's* rows in one pitch zone
(PPDA, field tilt) requires a 180-degree point reflection of the opponent's coordinates into the
acting team's frame -- the ADR-028 mixed-frame class. Done through the one public reflection seam
(``silly_kicks.reflection``, ADR-045), never a hand-rolled mirror.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.reflection import reflect_columns


def reflect_into_team_frame(opp_actions: pd.DataFrame) -> pd.DataFrame:
    """Point-reflect every opponent row's ``start``/``end`` coordinates into the acting team's frame.

    Pure: returns a NEW frame (``reflect_columns`` never mutates its input). Non-position columns are
    carried through unchanged.
    """
    mask = np.ones(len(opp_actions), dtype=bool)
    return reflect_columns(
        opp_actions,
        mask,
        point_x=["start_x", "end_x"],
        point_y=["start_y", "end_y"],
    )
