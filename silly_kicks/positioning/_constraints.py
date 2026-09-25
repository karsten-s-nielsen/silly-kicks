"""Feasibility constraints for repositioning candidates (spec section 5).

A ``Constraint`` answers "may this player be placed at this candidate?" -- an infeasible
candidate is rejected at the SA proposal step (never scored). The built-in
:class:`ReachabilityConstraint` bounds a candidate by time-to-intercept from the player's REAL
position + velocity (never a mid-search intermediate).

See NOTICE for full bibliographic citations (Pleuler TTI).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

# Public seam: the kinematic TTI shared with pitch control.
from silly_kicks.tracking import compute_tti

from ._config import ReachabilityParams


@runtime_checkable
class Constraint(Protocol):
    """Feasibility of placing ``player_id`` at ``candidate_xy`` in ``frame``.

    Examples
    --------
    Any object with a boolean ``is_feasible(player_id, candidate_xy, frame)`` is a ``Constraint``::

        class AlwaysFeasible:
            def is_feasible(self, player_id, candidate_xy, frame):
                return True

        constraints = [AlwaysFeasible()]  # accepted by optimise_positions
    """

    def is_feasible(self, player_id, candidate_xy: tuple[float, float], frame: pd.DataFrame) -> bool:
        """True iff ``player_id`` may be repositioned to ``candidate_xy`` in ``frame``.

        Examples
        --------
        Reject a candidate a player cannot reach in time::

            feasible = constraint.is_feasible(player_id=10, candidate_xy=(9.0, 30.0), frame=frame)
        """
        ...


class ReachabilityConstraint:
    """A candidate is feasible iff the player can time-to-intercept it within the horizon.

    ``compute_tti(real_pos, real_vel, candidate, reaction_time, max_acceleration) <=
    max_reach_seconds``, evaluated against the player's REAL position + velocity in ``frame`` -- so
    the reachable set is anchored to where the player actually is, never to a mid-search proposal.
    An unknown player, or one with non-finite position/velocity, is fail-closed (infeasible).

    Examples
    --------
    >>> ReachabilityConstraint(ReachabilityParams.default())  # doctest: +ELLIPSIS
    <silly_kicks.positioning._constraints.ReachabilityConstraint object at ...>
    """

    def __init__(self, params: ReachabilityParams) -> None:
        self._params = params

    def is_feasible(self, player_id, candidate_xy: tuple[float, float], frame: pd.DataFrame) -> bool:
        """True iff the player can time-to-intercept ``candidate_xy`` within the horizon.

        Examples
        --------
        Feasibility is judged against the player's REAL position + velocity in ``frame``::

            c = ReachabilityConstraint(ReachabilityParams.default())
            reachable = c.is_feasible(player_id=10, candidate_xy=(9.0, 30.0), frame=frame)
        """
        rows = frame[ids_match(frame["player_id"], player_id) & ~frame["is_ball"].astype(bool)]
        if len(rows) == 0:
            return False  # unknown player -> fail-closed
        r = rows.iloc[0]
        pos = np.array([[float(r["x"]), float(r["y"])]], dtype=float)
        vel = np.array([[float(r["vx"]), float(r["vy"])]], dtype=float)
        target = np.array([[float(candidate_xy[0]), float(candidate_xy[1])]], dtype=float)
        if not np.isfinite(pos).all() or not np.isfinite(vel).all() or not np.isfinite(target).all():
            return False
        tti = float(
            compute_tti(
                pos,
                vel,
                target,
                reaction_time=self._params.reaction_time,
                max_acceleration=self._params.max_acceleration,
            ).ravel()[0]
        )
        return bool(tti <= self._params.max_reach_seconds)
