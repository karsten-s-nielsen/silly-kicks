"""The behind-the-ball / goal-side counting primitive (TF-60, ADR-080).

No PUBLIC counting surface existed before TF-60; the only prior art is ``GhostGkModel``'s private
``defenders_behind_ball`` (``tracking/_ghost_gk.py``, ``to_gr_x(x) < ball_x``). This primitive reuses
that SEMANTICS (count players goal-side of the ball) but takes its orientation from the caller (a
``GoalMap`` end resolved once per match, ADR-055) instead of a per-frame ``to_gr_x`` -- the same
1-D-along-the-attacking-axis count ``compute_packing_metrics`` uses, and it inherits the documented
far-touchline caveat (a wide player inside the x-band is counted).

Ids are compared via ``id_compat.ids_match`` (ADR-019), never raw ``==``; a ball row carries NA
``team_id`` and is excluded for free. The batched form builds the frame grouping ONCE with
``group_rows`` (ADR-068) and is scale-guarded (ADR-073).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import RowGroups, group_rows
from silly_kicks.id_compat import ids_match

from ._columns import RD_FRAME_KEYS


def bool_flag(series: pd.Series) -> np.ndarray:
    """Coerce a boolean flag column (bool / object-of-bools / nullable ``boolean``) to a numpy bool
    array, NA -> False, WITHOUT the object-``fillna`` downcast FutureWarning and WITHOUT the
    ``astype(bool)``-on-``"false"`` string trap: the explicit ``astype("boolean")`` refuses a genuine
    string rather than truthily accepting it. Real provider frames carry ``is_goalkeeper`` as OBJECT
    (Python bools), so a plain ``.fillna(False)`` warns on every call."""
    return pd.Series(series).astype("boolean").fillna(False).to_numpy(dtype=bool)


def count_goalside(
    frame_rows: pd.DataFrame,
    *,
    team_id,
    ball_x: float,
    goal_x: float,
    include_gk: bool = True,
) -> int:
    """Players of ``team_id`` between the ball and the REFERENCE goal ``goal_x`` (goal-side).

    ``goal_x`` is the goal to count TOWARD, NOT necessarily the team's own goal -- rest-defense's
    numerical-superiority counts BOTH teams toward team A's defended goal G_A (spec §7.1), so the
    caller passes the SAME ``goal_x`` for A's rearguard and B's already-advanced players. Counting
    is inclusive of both band endpoints. An empty frame (unobserved) yields 0; the FOV companion
    (ADR-077) is what tells a consumer whether a 0 is "none there" or "none visible".
    """
    own = frame_rows[ids_match(frame_rows["team_id"], team_id)]  # NA-team ball rows excluded (ADR-058)
    if not include_gk and "is_goalkeeper" in own.columns:
        own = own.loc[~bool_flag(own["is_goalkeeper"])]
    xs = own["x"].to_numpy(dtype=float)
    xs = xs[np.isfinite(xs)]
    lo, hi = (goal_x, ball_x) if goal_x <= ball_x else (ball_x, goal_x)
    return int(np.count_nonzero((xs >= lo) & (xs <= hi)))


def count_goalside_by_sample(
    samples: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    team_col: str = "team_id",
    ball_x_col: str = "ball_x",
    goal_x_col: str = "own_goal_x",
    include_gk: bool = True,
    groups: RowGroups | None = None,
) -> pd.Series:
    """Per-sample :func:`count_goalside`, one row per ``samples`` row (``Int64`` Series).

    Builds the per-frame grouping ONCE (``group_rows`` on ``RD_FRAME_KEYS``) then serves each
    sample's frame slice in O(1) -- no rescan-in-loop (ADR-068). The orchestrator threads its own
    pre-built ``groups`` in (one grouping per match, shared across every per-sample metric); when
    called standalone (e.g. the scale guard) it builds its own. Registered in
    ``tests/_scale_guarded.SCALE_GUARDED`` and growth-guarded (ADR-073).
    """
    if groups is None:
        groups = group_rows(frames, tuple(RD_FRAME_KEYS))
    counts: list[int] = []
    for row in samples.itertuples(index=False):
        key = tuple(getattr(row, k) for k in RD_FRAME_KEYS)
        frame_rows = groups.get(*key)
        counts.append(
            count_goalside(
                frame_rows,
                team_id=getattr(row, team_col),
                ball_x=float(getattr(row, ball_x_col)),
                goal_x=float(getattr(row, goal_x_col)),
                include_gk=include_gk,
            )
        )
    return pd.Series(counts, index=samples.index, dtype="Int64")
