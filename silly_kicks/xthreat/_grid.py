"""Grid binning + per-cell probability helpers (relocated verbatim from the legacy module).

Keyed on (l, w) ints; GridSpec callers unpack via grid.n_zones_x / grid.n_zones_y.
"""

import dataclasses
from typing import TypedDict

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


def _scoring_prob_from_counts(goal_counts: npt.ArrayLike, shot_counts: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Per-cell P(goal | shot) from raw zone counts -- the count-based core of :func:`_scoring_prob`.

    No prior/smoothing: exactly ``_safe_divide(goal, shot)`` (0 where no shots). Shared by
    ``fit(actions)`` (via ``_scoring_prob``) and ``ExpectedThreat.fit_from_counts`` so the two agree.
    """
    return _safe_divide(goal_counts, shot_counts)


def _shot_zone_counts(actions: pd.DataFrame, l: int = N, w: int = M) -> npt.NDArray[np.int_]:
    """Per-zone shot counts (valid START), ``(w, l)``.

    The ONE extractor `fit()` (via `_scoring_prob` / `_action_prob`) and `ExpectedThreat.zone_counts`
    both call, so the fit path and the counts path cannot diverge on which rows they count (spec §4.4).
    """
    shot_actions = actions[(actions.type_id == spadlconfig.actiontype_id["shot"])]
    return _count(shot_actions.start_x, shot_actions.start_y, l, w)


def _goal_zone_counts(actions: pd.DataFrame, l: int = N, w: int = M) -> npt.NDArray[np.int_]:
    """Per-zone goal counts (shots with a successful result), valid START, ``(w, l)``. Shared extractor."""
    shot_actions = actions[(actions.type_id == spadlconfig.actiontype_id["shot"])]
    goals = shot_actions[(shot_actions.result_id == spadlconfig.result_id["success"])]
    return _count(goals.start_x, goals.start_y, l, w)  # type: ignore[reportAttributeAccessIssue]


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
    return _scoring_prob_from_counts(_goal_zone_counts(actions, l, w), _shot_zone_counts(actions, l, w))


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


def _action_prob_from_counts(
    shot_counts: npt.ArrayLike, move_counts: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-cell (P(shoot), P(move)) from raw zone counts -- the count-based core of :func:`_action_prob`.

    ``total = move + shot``; returns ``(_safe_divide(shot, total), _safe_divide(move, total))``. No
    smoothing. ``move_counts`` here is the VALID-START move population (matching ``_action_prob``), NOT
    the Singh transition denominator (valid start AND end). Shared by ``fit`` + ``fit_from_counts``.
    """
    total = np.asarray(move_counts) + np.asarray(shot_counts)
    return _safe_divide(shot_counts, total), _safe_divide(move_counts, total)


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
    return _action_prob_from_counts(_shot_zone_counts(actions, l, w), _move_start_zone_counts(actions, l, w))


def _move_start_zone_counts(actions: pd.DataFrame, l: int = N, w: int = M) -> npt.NDArray[np.int_]:
    """Per-zone counts of ALL move actions (pass/dribble/cross, any result) by valid START, ``(w, l)``.

    This is the ``_action_prob`` move population -- valid start only -- which DIFFERS from the Singh
    transition denominator (valid start AND end). Shared by `fit()` and `zone_counts`."""
    move_actions = _get_move_actions(actions)
    return _count(move_actions.start_x, move_actions.start_y, l, w)


class _ZoneCountKwargs(TypedDict):
    """The keyword arguments :meth:`ExpectedThreat.fit_from_counts` accepts (single serialization source)."""

    shot_counts: npt.NDArray[np.integer]
    goal_counts: npt.NDArray[np.integer]
    move_counts: npt.NDArray[np.integer]
    transition_start_counts: npt.NDArray[np.integer]
    transition_counts: npt.NDArray[np.integer]


@dataclasses.dataclass(frozen=True, eq=False)
class XtZoneCounts:
    """The five per-zone integer count aggregates an xT fit reduces (SK-XT-COUNTS / spec §4.4).

    Additive across partitions (``__add__`` sums element-wise), so a producer fits from one distributed
    ``groupBy`` (per-competition counts summed) via :meth:`ExpectedThreat.fit_from_counts`. Built by
    :meth:`ExpectedThreat.zone_counts` from the SAME extractors ``fit()`` uses, so a counts-based fit is
    byte-identical to ``fit(actions)``. ``eq=False`` because the fields are ndarrays (no default ``==``).

    Examples
    --------
    Sum per-match counts and fit one grid (Singh-only)::

        total = XtZoneCounts.zeros(16, 12)
        for actions in per_match_actions:
            total = total + ExpectedThreat(l=16, w=12).zone_counts(actions)
        xt = ExpectedThreat(l=16, w=12).fit_from_counts(**total.as_fit_kwargs())
    """

    l: int
    w: int
    shot_counts: npt.NDArray[np.int64]
    goal_counts: npt.NDArray[np.int64]
    move_counts: npt.NDArray[np.int64]
    transition_start_counts: npt.NDArray[np.int64]
    transition_counts: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        n = self.w * self.l
        for name, shape in (
            ("shot_counts", (self.w, self.l)),
            ("goal_counts", (self.w, self.l)),
            ("move_counts", (self.w, self.l)),
            ("transition_start_counts", (self.w, self.l)),
            ("transition_counts", (n, n)),
        ):
            arr = np.asarray(getattr(self, name), dtype=np.int64)
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape} for (l={self.l}, w={self.w})")
            object.__setattr__(self, name, arr)  # frozen: coerce to int64 in place

    @classmethod
    def zeros(cls, l: int, w: int) -> "XtZoneCounts":
        """An all-zero counts object on an ``l`` x ``w`` grid -- the identity for :meth:`__add__`.

        Examples
        --------
        Start a corpus reduction from the additive identity::

            total = XtZoneCounts.zeros(16, 12)  # then: total = total + per_match_counts
        """
        n = w * l
        return cls(
            l,
            w,
            np.zeros((w, l), dtype=np.int64),
            np.zeros((w, l), dtype=np.int64),
            np.zeros((w, l), dtype=np.int64),
            np.zeros((w, l), dtype=np.int64),
            np.zeros((n, n), dtype=np.int64),
        )

    def __add__(self, other: "XtZoneCounts") -> "XtZoneCounts":
        if (self.l, self.w) != (other.l, other.w):
            raise ValueError(f"grid mismatch: (l={self.l}, w={self.w}) + (l={other.l}, w={other.w})")
        return XtZoneCounts(
            self.l,
            self.w,
            self.shot_counts + other.shot_counts,
            self.goal_counts + other.goal_counts,
            self.move_counts + other.move_counts,
            self.transition_start_counts + other.transition_start_counts,
            self.transition_counts + other.transition_counts,
        )

    def as_fit_kwargs(self) -> _ZoneCountKwargs:
        """The kwargs :meth:`ExpectedThreat.fit_from_counts` accepts.

        Examples
        --------
        Round-trip a counts object into a fitted grid::

            xt = ExpectedThreat(l=16, w=12).fit_from_counts(**counts.as_fit_kwargs())
        """
        return _ZoneCountKwargs(
            shot_counts=self.shot_counts,
            goal_counts=self.goal_counts,
            move_counts=self.move_counts,
            transition_start_counts=self.transition_start_counts,
            transition_counts=self.transition_counts,
        )
