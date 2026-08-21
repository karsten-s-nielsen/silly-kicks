"""``add_restart_coordinates`` is robust to non-chronological ``action_id`` (spec §3d).

A persisted mart may carry a non-chronological ``action_id`` (the very defect the converter fix
removes for fresh conversions). ``resolve_restart_geometry`` imputes a restart's destination from
the *chronological* next action via a positional ``.shift(-1)``, so ``add_restart_coordinates`` must
establish order by ``time_seconds`` (with ``action_id`` as a tiebreak), NOT by ``action_id`` alone
-- otherwise a mart's scrambled ``action_id`` yields the wrong neighbour. Defense-in-depth for the
one path the converter-boundary guard cannot reach (it reads marts, not fresh conversions).
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_restart_coordinates

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_PASS = spadlconfig.actiontype_id["pass"]


def _base() -> pd.DataFrame:
    # A goal-kick (native end NaN -> destination imputed from the chronological next action's start)
    # followed by two passes. Chronologically the goal-kick's next action is the pass at t=12.
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 12.0, 14.0],
            "type_id": [_GOALKICK, _PASS, _PASS],
            "start_x": [5.0, 30.0, 60.0],
            "start_y": [34.0, 40.0, 20.0],
            "end_x": [float("nan"), 35.0, 65.0],
            "end_y": [float("nan"), 40.0, 20.0],
        }
    )


_GEOM_COLS = [
    "enriched_end_x",
    "enriched_end_y",
    "end_coord_source",
    "enriched_start_x",
    "enriched_start_y",
    "start_coord_source",
]


def _run_aligned(actions: pd.DataFrame) -> pd.DataFrame:
    out = add_restart_coordinates(actions)
    return out.sort_values("time_seconds").reset_index(drop=True)[["time_seconds", *_GEOM_COLS]]


def test_restart_geometry_is_action_id_order_insensitive():
    chrono = _base().assign(action_id=[0, 1, 2])
    # Same rows, but action_id order DISAGREES with time order (mart-shaped).
    shuffled = _base().assign(action_id=[2, 0, 1])

    a = _run_aligned(chrono)
    b = _run_aligned(shuffled)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_goalkick_destination_is_the_chronological_next_action():
    # Pin the correct value: the goal-kick's imputed destination is the t=12 pass's start (30, 40),
    # regardless of action_id numbering.
    for action_id in ([0, 1, 2], [2, 0, 1]):
        out = add_restart_coordinates(_base().assign(action_id=action_id))
        gk = out[out["type_id"] == _GOALKICK].iloc[0]
        assert gk["enriched_end_x"] == 30.0, f"action_id={action_id}: end_x={gk['enriched_end_x']}"
        assert gk["enriched_end_y"] == 40.0, f"action_id={action_id}: end_y={gk['enriched_end_y']}"
