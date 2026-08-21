"""Order-sensitive ``add_*`` consumers are robust to non-chronological ``action_id`` (spec §3d).

The public enrichers ``add_possessions`` / ``add_gk_role`` / ``add_pre_shot_gk_context`` /
``add_gk_distribution_metrics`` (+ their atomic mirrors) do neighbour / ``.shift`` / window lookups
over the actions ordered by ``action_id``. A persisted mart may carry a non-chronological
``action_id``, so they must establish order by ``time_seconds`` (with ``action_id`` as the tiebreak)
via the shared ``_sort_actions_chronological_or_action_id`` helper -- not by ``action_id`` alone.
``add_possessions`` is the RED-witness (its possession boundaries flip when the team sequence is
reordered); the other consumers share the same helper, so this pins the helper + its wiring.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions

_PASS = spadlconfig.actiontype_id["pass"]


def _base() -> pd.DataFrame:
    # Two possessions by time: team 100 (t=10,11) then team 200 (t=12,13). Gaps are < max_gap_seconds
    # (7.0), so the ONLY possession boundary is the team-change at t=12 -> correct possession_id by
    # time is [0, 0, 1, 1]. (Larger gaps would make each action its own possession and mask the test.)
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 1, 1],
            "time_seconds": [10.0, 11.0, 12.0, 13.0],
            "team_id": [100, 100, 200, 200],
            "player_id": [1, 2, 12, 13],
            "type_id": [_PASS, _PASS, _PASS, _PASS],
            "start_x": [10.0, 20.0, 80.0, 90.0],
            "start_y": [34.0, 34.0, 34.0, 34.0],
            "end_x": [20.0, 30.0, 90.0, 95.0],
            "end_y": [34.0, 34.0, 34.0, 34.0],
        }
    )


def _possessions_by_time(actions: pd.DataFrame) -> list[int]:
    out = add_possessions(actions)
    return out.sort_values("time_seconds")["possession_id"].tolist()


def test_add_possessions_is_action_id_order_insensitive():
    chrono = _base().assign(action_id=[0, 1, 2, 3])
    # action_id order disagrees with time order: the two team-200 rows carry the LOW action_ids.
    shuffled = _base().assign(action_id=[2, 3, 0, 1])

    assert _possessions_by_time(chrono) == _possessions_by_time(shuffled)
    # Pin the chronologically-correct assignment (not just internal consistency).
    assert _possessions_by_time(chrono) == [0, 0, 1, 1]
