"""Extended move-set + xtgk-local transition/action-prob builders (ADR-036 §G2).

Classic xT's move-set excludes goal-kicks/throw-ins; this GK surface needs them in the
transition law and the p_shot/p_move split. These builders REUSE xthreat's shared low-level
seams (grid binning, the KDE kernel) and never modify xthreat's public functions. On a
pass-only cohort they are byte-identical to the stock builders (proven in test_xtgk_builder_parity).

Singh consumes ALL extended moves (success computed internally); KDE consumes SUCCESSFUL
extended moves only (mirrors _get_successful_move_actions) — the two paths need different
populations, which is why a single injected DataFrame on the public builders would be wrong.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _count, _get_flat_indexes, _safe_divide
from silly_kicks.xthreat._params import GridSpec, KDEParams
from silly_kicks.xthreat._transitions import _kde_transition_from_grouped, _zone_centres

Method = Literal["singh_counts", "kde_smoothed"]

MOVE_TYPE_IDS: tuple[int, ...] = (
    spadlconfig.actiontype_id["pass"],
    spadlconfig.actiontype_id["dribble"],
    spadlconfig.actiontype_id["cross"],
    spadlconfig.actiontype_id["goalkick"],
    spadlconfig.actiontype_id["throw_in"],
)
_SHOT = spadlconfig.actiontype_id["shot"]
_SUCCESS = spadlconfig.result_id["success"]


def extended_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """All ball-progressing actions incl. goal-kicks/throw-ins (any result)."""
    return actions[actions["type_id"].isin(MOVE_TYPE_IDS)]


def _successful(move_actions: pd.DataFrame) -> pd.DataFrame:
    return move_actions[move_actions["result_id"] == _SUCCESS]


def xtgk_action_prob(actions: pd.DataFrame, l: int, w: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(p_shot, p_move) per cell over the EXTENDED move-set. Mirrors xthreat _action_prob."""
    move = extended_move_actions(actions)
    shots = actions[actions["type_id"] == _SHOT]
    movematrix = _count(move.start_x, move.start_y, l, w)
    shotmatrix = _count(shots.start_x, shots.start_y, l, w)
    total = movematrix + shotmatrix
    return _safe_divide(shotmatrix, total), _safe_divide(movematrix, total)


def _singh_transition(actions: pd.DataFrame, grid: GridSpec) -> npt.NDArray[np.float64]:
    """Byte-identical to xthreat.singh_transition_matrix but over the extended move-set."""
    l, w = grid.n_zones_x, grid.n_zones_y
    n = w * l
    move = extended_move_actions(actions).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_cell = _get_flat_indexes(move.end_x, move.end_y, l, w).to_numpy()
    is_success = (move.result_id == _SUCCESS).to_numpy()
    start_counts = np.zeros(n)
    np.add.at(start_counts, start_cell, 1.0)
    counts = np.zeros((n, n))
    np.add.at(counts, (start_cell[is_success], end_cell[is_success]), 1.0)
    transition = np.zeros((n, n))
    nz = start_counts > 0
    transition[nz] = counts[nz] / start_counts[nz, None]
    return transition


def _bin_extended_successful(actions: pd.DataFrame, grid: GridSpec):
    """Group SUCCESSFUL extended-move destinations by source zone (mirrors
    _bin_destinations_by_source, defaults: keep every row)."""
    l, w = grid.n_zones_x, grid.n_zones_y
    centres = _zone_centres(grid)
    move = _successful(extended_move_actions(actions)).dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if len(move) == 0:
        return {}, centres
    start_cell = _get_flat_indexes(move.start_x, move.start_y, l, w).to_numpy()
    end_xy = move[["end_x", "end_y"]].to_numpy(dtype=np.float64)
    order = np.argsort(start_cell, kind="stable")
    sc_sorted = start_cell[order]
    end_sorted = end_xy[order]
    boundaries = np.flatnonzero(np.diff(sc_sorted)) + 1
    zone_per_group = sc_sorted[np.concatenate(([0], boundaries))]
    groups = np.split(end_sorted, boundaries)
    grouped = {int(s): pts for s, pts in zip(zone_per_group, groups, strict=True)}
    return grouped, centres


def _kde_transition(actions: pd.DataFrame, grid: GridSpec, params: KDEParams) -> npt.NDArray[np.float64]:
    grouped, centres = _bin_extended_successful(actions, grid)
    return _kde_transition_from_grouped(grouped, centres, grid, params)


def xtgk_transition_matrix(
    actions: pd.DataFrame,
    grid: GridSpec,
    *,
    method: Method = "singh_counts",
    params: KDEParams | None = None,
) -> npt.NDArray[np.float64]:
    if method == "kde_smoothed":
        return _kde_transition(actions, grid, params or KDEParams())
    return _singh_transition(actions, grid)
