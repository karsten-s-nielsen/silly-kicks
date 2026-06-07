"""Grid binning + per-cell probability helpers (relocated verbatim from the legacy module).

Keyed on (l, w) ints; GridSpec callers unpack via grid.n_zones_x / grid.n_zones_y.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

M: int = 12
N: int = 16


def _get_cell_indexes(x: pd.Series, y: pd.Series, l: int = N, w: int = M) -> tuple[pd.Series, pd.Series]:
    xi = x.divide(spadlconfig.field_length).multiply(l)
    yj = y.divide(spadlconfig.field_width).multiply(w)
    xi = xi.astype("int64").clip(0, l - 1)
    yj = yj.astype("int64").clip(0, w - 1)
    return xi, yj


def _get_flat_indexes(x: pd.Series, y: pd.Series, l: int = N, w: int = M) -> pd.Series:
    xi, yj = _get_cell_indexes(x, y, l, w)
    return yj.rsub(w - 1).mul(l).add(xi)


def _count(x: pd.Series, y: pd.Series, l: int = N, w: int = M) -> npt.NDArray[np.int_]:
    """Count the number of actions occurring in each cell of the grid.

    Parameters
    ----------
    x : pd.Series
        The x-coordinates of the actions.
    y : pd.Series
        The y-coordinates of the actions.
    l : int
        Amount of grid cells in the x-dimension of the grid.
    w : int
        Amount of grid cells in the y-dimension of the grid.

    Returns
    -------
    np.ndarray
        A matrix, denoting the amount of actions occurring in each cell. The
        top-left corner is the origin.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]  # type: ignore[reportAssignmentType]

    flat_indexes = _get_flat_indexes(x, y, l, w)
    vc = flat_indexes.value_counts(sort=False)
    vector = np.zeros(w * l, dtype=int)
    vector[vc.index] = vc
    return vector.reshape((w, l))


def _safe_divide(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.divide(a, b, out=np.zeros_like(a, dtype="float64"), where=b != 0, casting="unsafe")


def _scoring_prob(actions: pd.DataFrame, l: int = N, w: int = M) -> npt.NDArray[np.float64]:
    """Compute the probability of scoring when taking a shot for each cell.

    Parameters
    ----------
    actions : pd.DataFrame
        Actions, in SPADL format.
    l : int
        Amount of grid cells in the x-dimension of the grid.
    w : int
        Amount of grid cells in the y-dimension of the grid.

    Returns
    -------
    np.ndarray
        A matrix, denoting the probability of scoring for each cell.
    """
    shot_actions = actions[(actions.type_id == spadlconfig.actiontype_id["shot"])]
    goals = shot_actions[(shot_actions.result_id == spadlconfig.result_id["success"])]

    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, l, w)
    goalmatrix = _count(goals.start_x, goals.start_y, l, w)  # type: ignore[reportAttributeAccessIssue]
    return _safe_divide(goalmatrix, shotmatrix)


def _get_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Get all ball-progressing actions.

    These include passes, dribbles and crosses. Take-ons are ignored because
    they typically coincide with dribbles and do not move the ball to
    a different cell.

    Parameters
    ----------
    actions : pd.DataFrame
        Actions, in SPADL format.

    Returns
    -------
    pd.DataFrame
        All ball-progressing actions in the input dataframe.
    """
    return actions[  # type: ignore[reportReturnType]
        (actions.type_id == spadlconfig.actiontype_id["pass"])
        | (actions.type_id == spadlconfig.actiontype_id["dribble"])
        | (actions.type_id == spadlconfig.actiontype_id["cross"])
    ]


def _get_successful_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Get all successful ball-progressing actions.

    These include successful passes, dribbles and crosses.

    Parameters
    ----------
    actions : pd.DataFrame
        Actions, in SPADL format.

    Returns
    -------
    pd.DataFrame
        All ball-progressing actions in the input dataframe.
    """
    move_actions = _get_move_actions(actions)
    return move_actions[(move_actions.result_id == spadlconfig.result_id["success"])]  # type: ignore[reportReturnType]


def _action_prob(
    actions: pd.DataFrame, l: int = N, w: int = M
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the probability of taking an action in each cell of the grid.

    The options are: shooting or moving.

    Parameters
    ----------
    actions : pd.DataFrame
        Actions, in SPADL format.
    l : int
        Amount of grid cells in the x-dimension of the grid.
    w : int
        Amount of grid cells in the y-dimension of the grid.

    Returns
    -------
    shotmatrix : np.ndarray
        For each cell the probability of choosing to shoot.
    movematrix : np.ndarray
        For each cell the probability of choosing to move.
    """
    move_actions = _get_move_actions(actions)
    shot_actions = actions[(actions.type_id == spadlconfig.actiontype_id["shot"])]

    movematrix = _count(move_actions.start_x, move_actions.start_y, l, w)
    shotmatrix = _count(shot_actions.start_x, shot_actions.start_y, l, w)
    totalmatrix = movematrix + shotmatrix

    return _safe_divide(shotmatrix, totalmatrix), _safe_divide(movematrix, totalmatrix)
